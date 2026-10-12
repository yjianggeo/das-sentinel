import json
from datetime import UTC, datetime, timedelta

import pytest
from das_sentinel.persistence.db import create_local_engine, session_factory
from das_sentinel.persistence.model_scan_models import (
    ModelScan,
    ModelScanReceipt,
    ModelScanUnit,
)
from das_sentinel.persistence.models import Base, ProcessingProfile, Site
from das_sentinel.services import model_scans

NOW = datetime(2026, 10, 12, 12, tzinfo=UTC)
PREVIOUS = NOW - timedelta(minutes=1)


@pytest.fixture
def refresh_context(tmp_path, monkeypatch):
    monkeypatch.setattr(model_scans, "utc_now", lambda: NOW)
    engine = create_local_engine(f"sqlite+pysqlite:///{tmp_path / 'refresh.sqlite3'}")
    Base.metadata.create_all(engine)
    try:
        with session_factory(engine)() as session:
            site = Site(name="refresh", data_path=str(tmp_path), timezone="Asia/Shanghai", cable_type="downhole")
            session.add(site)
            session.flush()
            profile = ProcessingProfile(site_id=site.id, version=1, payload_json="{}", is_current=True)
            session.add(profile)
            session.flush()
            scan = ModelScan(site_id=site.id, profile_id=profile.id, request_id="refresh",
                             config_json="{}", config_sha256="a" * 64, status="running", updated_at=PREVIOUS)
            session.add(scan)
            session.flush()
            yield session, scan, model_scans.ModelScanService(session, tmp_path)
    finally:
        engine.dispose()


def add_unit(session, scan, key, *, status, kind="infer", pages_json="{}"):
    unit = ModelScanUnit(scan_id=scan.id, kind=kind, unit_key=key, status=status,
                         payload_json="{}", pages_json=pages_json)
    session.add(unit)
    session.flush()
    return unit


def refresh_vm_steps(session, api, scan_id):
    connection = session.connection().connection.driver_connection
    steps = 0

    def count_steps():
        nonlocal steps
        steps += 100
        return 0

    connection.set_progress_handler(count_steps, 100)
    try:
        api._refresh(scan_id)
    finally:
        connection.set_progress_handler(None, 0)
    return steps


@pytest.mark.parametrize("active_status", ["queued", "running"])
@pytest.mark.parametrize("has_failure", [False, True])
@pytest.mark.parametrize("history_count", [0, 4096])
def test_active_refresh_has_bounded_sqlite_work_with_completed_history(
        refresh_context, active_status, has_failure, history_count):
    session, scan, api = refresh_context
    session.add_all([ModelScanUnit(scan_id=scan.id, kind="infer", unit_key=f"history-{i}",
                                   status="succeeded", payload_json="{}") for i in range(history_count)])
    add_unit(session, scan, "active", status=active_status)
    if has_failure:
        add_unit(session, scan, "failed", status="failed")
    session.flush()

    steps = refresh_vm_steps(session, api, scan.id)

    assert scan.status == "running"
    assert scan.updated_at == PREVIOUS
    # A refresh with active work must not revisit every completed unit.
    assert steps <= 1000, f"active refresh consumed {steps} SQLite VM steps"


@pytest.mark.parametrize("status", ["paused", "cancelled"])
def test_controlled_scan_keeps_status_without_reading_units(refresh_context, status):
    session, scan, api = refresh_context
    add_unit(session, scan, "discover", status="succeeded", kind="discover", pages_json="invalid json")
    scan.status = status
    session.flush()

    steps = refresh_vm_steps(session, api, scan.id)

    assert scan.status == status
    assert scan.updated_at == PREVIOUS
    assert steps == 0


@pytest.mark.parametrize("unit_status,receipt_error,source_error,expected", [
    (None, False, False, "succeeded"),
    ("succeeded", False, False, "succeeded"),
    ("cancelled", False, False, "succeeded"),
    ("failed", True, True, "failed"),
    ("succeeded", True, False, "completed_with_errors"),
    ("succeeded", False, True, "completed_with_errors"),
    ("succeeded", True, True, "completed_with_errors"),
])
def test_no_active_units_preserve_terminal_status_and_errors(
        refresh_context, unit_status, receipt_error, source_error, expected):
    session, scan, api = refresh_context
    if unit_status:
        unit = add_unit(session, scan, "terminal", status=unit_status, kind="discover",
                        pages_json=json.dumps({"page": {"errors": [{"path": "missing.h5"}] if source_error else []}}))
        if receipt_error:
            session.add(ModelScanReceipt(scan_id=scan.id, infer_unit_id=unit.id, start_utc=PREVIOUS,
                                         inference_key="error", status="unjudgeable", result_json="{}",
                                         result_sha256="b" * 64, error_code="coverage_incomplete"))

    api._refresh(scan.id)
    session.flush()
    session.expire(scan)

    assert scan.status == expected
    assert scan.updated_at == NOW


@pytest.mark.parametrize("other_active_status", ["queued", "running"])
def test_other_batch_active_work_and_errors_do_not_prevent_completion(refresh_context, other_active_status):
    session, scan, api = refresh_context
    add_unit(session, scan, "terminal", status="succeeded")
    other = ModelScan(site_id=scan.site_id, profile_id=scan.profile_id, request_id="other",
                      config_json="{}", config_sha256="c" * 64, status="running", updated_at=PREVIOUS)
    session.add(other)
    session.flush()
    add_unit(session, other, "active", status=other_active_status)
    add_unit(session, other, "failed", status="failed")
    discover = add_unit(session, other, "discover", status="succeeded", kind="discover",
                        pages_json='{"page":{"errors":[{"path":"missing.h5"}]}}')
    session.add(ModelScanReceipt(scan_id=other.id, infer_unit_id=discover.id, start_utc=PREVIOUS,
                                 inference_key="other-error", status="unjudgeable", result_json="{}",
                                 result_sha256="d" * 64))

    api._refresh(scan.id)

    assert scan.status == "succeeded"
    assert other.status == "running"
    assert other.updated_at == PREVIOUS


@pytest.mark.parametrize("before,after,expected", [
    ("queued", "succeeded", "succeeded"),
    ("succeeded", "queued", "running"),
    ("running", "failed", "failed"),
])
def test_refresh_observes_pending_unit_change_in_writer_transaction(refresh_context, before, after, expected):
    session, scan, api = refresh_context
    unit = add_unit(session, scan, "changing", status=before)
    session.commit()
    unit.status = after

    api._refresh(scan.id)
    session.flush()

    assert scan.status == expected
    session.rollback()
    assert scan.status == "running"
    assert scan.updated_at == PREVIOUS
    assert unit.status == before


def test_refresh_observes_new_active_unit_before_explicit_flush(refresh_context):
    session, scan, api = refresh_context
    add_unit(session, scan, "terminal", status="succeeded")
    session.commit()
    session.add(ModelScanUnit(scan_id=scan.id, kind="infer", unit_key="new", status="queued", payload_json="{}"))

    api._refresh(scan.id)

    assert scan.status == "running"
    assert scan.updated_at == PREVIOUS
