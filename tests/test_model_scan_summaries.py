import json
from datetime import UTC, datetime, timedelta

import pytest
from das_sentinel.persistence.db import create_local_engine, session_factory
from das_sentinel.persistence.model_scan_models import (
    ModelScan,
    ModelScanReceipt,
    ModelScanSource,
    ModelScanUnit,
)
from das_sentinel.persistence.models import Base, ProcessingProfile, Site
from das_sentinel.services import model_scans
from sqlalchemy import event

NOW = datetime(2026, 10, 9, 12, tzinfo=UTC)
WINDOW_START = datetime(2023, 9, 27, tzinfo=UTC)
LARGE_JSON = json.dumps({"padding": "x" * 65536})
MODEL = {"id": "fixed-model", "threshold": .04}


@pytest.fixture
def summary_context(tmp_path, monkeypatch):
    monkeypatch.setattr(model_scans, "utc_now", lambda: NOW)
    engine = create_local_engine(f"sqlite+pysqlite:///{tmp_path / 'summaries.sqlite3'}")
    Base.metadata.create_all(engine)
    try:
        with session_factory(engine)() as session:
            site = Site(name="summary", data_path=str(tmp_path), timezone="Asia/Shanghai", cable_type="downhole")
            session.add(site)
            session.flush()
            profile = ProcessingProfile(site_id=site.id, version=1, payload_json="{}", is_current=True)
            session.add(profile)
            session.flush()
            config = json.dumps({"model": MODEL, "directories": [str(tmp_path)], "time_ranges": None})
            scan = ModelScan(site_id=site.id, profile_id=profile.id, request_id="summary",
                             config_json=config, config_sha256="a" * 64, status="running",
                             created_at=NOW, updated_at=NOW)
            session.add(scan)
            session.flush()
            yield session, scan, model_scans.ModelScanService(session, tmp_path)
    finally:
        engine.dispose()


def add_unit(session, scan, kind, key, **values):
    row = ModelScanUnit(scan_id=scan.id, kind=kind, unit_key=key, payload_json=LARGE_JSON, **values)
    session.add(row)
    session.flush()
    return row


def add_receipt(session, scan, unit, index, status, **values):
    row = ModelScanReceipt(scan_id=scan.id, infer_unit_id=unit.id,
                           start_utc=WINDOW_START + timedelta(seconds=index * 30),
                           inference_key=str(index), status=status, result_json=LARGE_JSON,
                           result_sha256="b" * 64, **values)
    session.add(row)
    return row


def captured_summary(session, api, scan):
    session.flush()
    session.expire_all()
    queries = []

    def capture(_connection, _cursor, statement, parameters, _context, _many):
        queries.append((" ".join(statement.split()), parameters))

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        summary = api.get(scan.id)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    return summary, queries


@pytest.mark.parametrize("query_table,query_status,projection", [
    ("model_scan_units", "failed",
     "SELECT model_scan_units.kind, model_scan_units.error_code, model_scan_units.error_message"),
    ("model_scan_receipts", "unjudgeable",
     "SELECT model_scan_receipts.start_utc, model_scan_receipts.error_code, model_scan_receipts.error_message"),
])
def test_summary_error_queries_omit_large_payloads_without_changing_errors(
        summary_context, tmp_path, query_table, query_status, projection):
    session, scan, api = summary_context
    source_errors = [{"path": f"source-{i}.h5", "error": f"missing-{i}"} for i in range(7)]
    discover = add_unit(session, scan, "discover", "inventory", status="succeeded",
                        pages_json=json.dumps({"page": {"errors": source_errors}}))
    add_unit(session, scan, "plan", "coverage", status="succeeded", pages_json='{"total":100}')
    for i in reversed(range(7)):
        add_unit(session, scan, "infer" if i % 2 == 0 else "materialize", f"failure-{i}",
                 status="failed", pages_json=LARGE_JSON, finished_at=NOW - timedelta(minutes=i),
                 error_code=None if i == 0 else f"unit-{i}",
                 error_message=None if i == 0 else f"failed-{i}")
    errors = [("coverage_incomplete", None), ("temporary_read", "read failed"),
              (None, None), (None, "unknown cause"), ("invalid_prediction", ""),
              ("late-5", "outside limit"), ("late-6", "outside limit")]
    for i in reversed(range(7)):
        code, message = errors[i]
        add_receipt(session, scan, discover, i, "unjudgeable", error_code=code, error_message=message)
    for i, status in enumerate(("negative", "negative", "ready", "selected", "pending_artifacts",
                                "protected", "excluded", "cancelled"), start=7):
        add_receipt(session, scan, discover, i, status, reused_from="previous" if i == 7 else None)
    for i in range(2):
        session.add(ModelScanSource(scan_id=scan.id, discovery_unit_id=discover.id, path=f"source-{i}.h5",
                                    fingerprint="c" * 64, metadata_json="{}",
                                    start_utc=WINDOW_START, end_utc=WINDOW_START + timedelta(minutes=1)))
    other = ModelScan(site_id=scan.site_id, profile_id=scan.profile_id, request_id="other",
                      config_json=scan.config_json, config_sha256="d" * 64)
    session.add(other)
    session.flush()
    unrelated = add_unit(session, other, "infer", "unrelated", status="failed", finished_at=NOW,
                         error_code="unrelated", error_message="must not leak")
    add_receipt(session, other, unrelated, -1, "unjudgeable", error_code="unrelated")

    summary, queries = captured_summary(session, api, scan)

    assert summary["counts"] == {"negative": 2, "unjudgeable": 7, "protected": 1, "excluded": 1,
                                 "cancelled": 1, "ready": 1, "selected": 3, "pending_artifacts": 2,
                                 "judged": 5, "sources": 2, "planned": 15, "source_errors": 7, "reused": 1}
    assert summary["errors"] == [
        {"phase": "infer", "code": None, "message": None},
        {"phase": "materialize", "code": "unit-1", "message": "failed-1"},
        {"phase": "infer", "code": "unit-2", "message": "failed-2"},
        {"phase": "materialize", "code": "unit-3", "message": "failed-3"},
        {"phase": "infer", "code": "unit-4", "message": "failed-4"},
        *[{"phase": "discover", "code": "source_unavailable", "message": f"source-{i}.h5: missing-{i}"}
          for i in range(5)],
        {"phase": "infer", "code": "coverage_incomplete", "message": "2023-09-27T00:00:00+00:00: coverage_incomplete"},
        {"phase": "infer", "code": "temporary_read", "message": "2023-09-27T00:00:30+00:00: read failed"},
        {"phase": "infer", "code": None, "message": "2023-09-27T00:01:00+00:00: None"},
        {"phase": "infer", "code": None, "message": "2023-09-27T00:01:30+00:00: unknown cause"},
        {"phase": "infer", "code": "invalid_prediction", "message": "2023-09-27T00:02:00+00:00: invalid_prediction"},
    ]
    assert summary | {"counts": None, "errors": None} == {
        "id": scan.id, "site_id": scan.site_id, "status": "running", "model": MODEL,
        "directories": [str(tmp_path)], "time_ranges": None, "counts": None, "errors": None,
        "discovery_complete": True, "total": 100, "remaining_seconds": None,
        "estimated_finish_at": None, "estimate_status": "waiting_worker", "retryable": True,
        "created_at": "2026-10-09T12:00:00+00:00", "updated_at": "2026-10-09T12:00:00+00:00"}
    failed_queries = [statement for statement, params in queries
                      if f"FROM {query_table}" in statement and query_status in params and "ORDER BY" in statement]
    assert len(failed_queries) == 1
    assert failed_queries[0].split(" FROM ", 1)[0] == projection


def add_estimate_data(session, scan, *, infer_samples=4, materialize_samples=4):
    add_unit(session, scan, "discover", "inventory", status="succeeded")
    add_unit(session, scan, "plan", "coverage", status="succeeded", pages_json='{"total":20}')
    active = add_unit(session, scan, "infer", "active", status="running", heartbeat_at=NOW)
    for i, status in enumerate(("negative", "negative", "ready", "selected")):
        add_receipt(session, scan, active, i, status)
    for kind, count, duration in (("infer", infer_samples, 20), ("materialize", materialize_samples, 5)):
        for i in range(count):
            finished = NOW - timedelta(minutes=i + 1)
            add_unit(session, scan, kind, f"sample-{i}", status="succeeded", pages_json=LARGE_JSON,
                     started_at=finished - timedelta(seconds=duration), finished_at=finished,
                     window_count=10 if kind == "infer" else 0)


def test_estimate_queries_omit_large_payloads_and_preserve_sample_filters(summary_context):
    session, scan, api = summary_context
    add_estimate_data(session, scan)
    invalid_samples = [
        ("missing_start", None, NOW - timedelta(seconds=10), 10, "succeeded"),
        ("missing_finish", NOW - timedelta(seconds=30), None, 10, "succeeded"),
        ("old", NOW - timedelta(hours=3), NOW - timedelta(hours=3) + timedelta(seconds=20), 1, "succeeded"),
        ("reversed", NOW - timedelta(seconds=10), NOW - timedelta(seconds=20), 10, "succeeded"),
        ("zero_duration", NOW - timedelta(seconds=20), NOW - timedelta(seconds=20), 10, "succeeded"),
        ("zero_windows", NOW - timedelta(seconds=30), NOW - timedelta(seconds=10), 0, "succeeded"),
        ("unfinished", NOW - timedelta(seconds=30), NOW - timedelta(seconds=10), 1, "queued"),
    ]
    for key, started, finished, windows, status in invalid_samples:
        add_unit(session, scan, "infer", key, status=status, started_at=started, finished_at=finished,
                 window_count=windows, pages_json=LARGE_JSON)
    other = ModelScan(site_id=scan.site_id, profile_id=scan.profile_id, request_id="other",
                      config_json=scan.config_json, config_sha256="d" * 64)
    session.add(other)
    session.flush()
    add_unit(session, other, "infer", "unrelated", status="succeeded", window_count=1,
             started_at=NOW - timedelta(minutes=10), finished_at=NOW)

    summary, queries = captured_summary(session, api, scan)

    assert summary["counts"] == {"negative": 2, "unjudgeable": 0, "protected": 0, "excluded": 0,
                                 "cancelled": 0, "ready": 1, "selected": 2, "pending_artifacts": 1,
                                 "judged": 4, "sources": 0, "planned": 4, "source_errors": 0, "reused": 0}
    assert summary["errors"] == []
    assert summary["retryable"] is False
    assert summary["discovery_complete"] is True
    assert summary["total"] == 20
    assert summary["estimate_status"] == "running"
    assert summary["remaining_seconds"] == 77
    assert summary["estimated_finish_at"] == "2026-10-09T12:01:17+00:00"
    sample_queries = [(statement, params) for statement, params in queries
                      if "FROM model_scan_units" in statement and "succeeded" in params
                      and "ORDER BY" in statement and ("infer" in params or "materialize" in params)]
    assert len(sample_queries) == 2
    for statement, params in sample_queries:
        assert statement.split(" FROM ", 1)[0] == (
            "SELECT model_scan_units.started_at, model_scan_units.finished_at, model_scan_units.window_count")
        assert "ORDER BY model_scan_units.finished_at DESC LIMIT ? OFFSET ?" in statement
        assert params[-2:] == (30, 0)


@pytest.mark.parametrize("infer_samples,materialize_samples", [(3, 4), (4, 3)])
def test_estimate_requires_four_valid_samples_per_needed_kind(summary_context, infer_samples, materialize_samples):
    session, scan, api = summary_context
    add_estimate_data(session, scan, infer_samples=infer_samples, materialize_samples=materialize_samples)
    add_unit(session, scan, "infer", "zero_windows", status="succeeded", window_count=0,
             started_at=NOW - timedelta(seconds=30), finished_at=NOW - timedelta(seconds=10))
    add_unit(session, scan, "materialize", "zero_duration", status="succeeded", window_count=1,
             started_at=NOW - timedelta(seconds=10), finished_at=NOW - timedelta(seconds=10))

    summary, _queries = captured_summary(session, api, scan)

    assert summary["total"] == 20
    assert summary["estimate_status"] == "estimating"
    assert summary["remaining_seconds"] is None
    assert summary["estimated_finish_at"] is None


def test_estimate_uses_only_thirty_latest_samples(summary_context):
    session, scan, api = summary_context
    add_estimate_data(session, scan, infer_samples=30)
    add_unit(session, scan, "infer", "older_outlier", status="succeeded", window_count=1,
             started_at=NOW - timedelta(hours=1), finished_at=NOW - timedelta(minutes=31))

    summary, _queries = captured_summary(session, api, scan)

    assert summary["estimate_status"] == "running"
    assert summary["remaining_seconds"] == 77
    assert summary["estimated_finish_at"] == "2026-10-09T12:01:17+00:00"


@pytest.mark.parametrize("error_code,retryable", [("coverage_incomplete", False), (None, False), ("temporary_read", True)])
def test_summary_preserves_retryability_for_window_errors(summary_context, error_code, retryable):
    session, scan, api = summary_context
    unit = add_unit(session, scan, "discover", "inventory", status="succeeded")
    add_receipt(session, scan, unit, 0, "unjudgeable", error_code=error_code)

    summary, _queries = captured_summary(session, api, scan)

    assert summary["retryable"] is retryable
    assert summary["counts"]["unjudgeable"] == 1
    assert summary["errors"] == [{"phase": "infer", "code": error_code,
                                  "message": f"2023-09-27T00:00:00+00:00: {error_code}"}]
