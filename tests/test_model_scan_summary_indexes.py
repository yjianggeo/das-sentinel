"""Keep task summary predicates and ETA samples off large unit payloads."""
from datetime import UTC, datetime, timedelta

import pytest
from das_sentinel.persistence.db import upgrade_database
from das_sentinel.persistence.model_scan_models import ModelScanUnit
from sqlalchemy import func, select

from tests.test_event_windows import context as context  # noqa: PLC0414
from tests.test_model_scans import create, service

INDEX = "ix_model_scan_unit_summary"


def populated(context, tmp_path):
    api = service(context, tmp_path)
    scan = create(api, context, tmp_path)
    now = datetime(2026, 10, 9, 12, tzinfo=UTC)
    for number in range(80):
        unit = ModelScanUnit(
            scan_id=scan["id"], kind="infer" if number % 2 else "materialize",
            unit_key=str(number), payload_json='{"padding":"' + "x" * 20000 + '"}',
            status="succeeded", started_at=now - timedelta(seconds=number + 20),
            finished_at=now - timedelta(seconds=number), window_count=10,
        )
        context[0].add(unit)
    active = ModelScanUnit(scan_id=scan["id"], kind="infer", unit_key="active",
                           payload_json="{}", status="running", heartbeat_at=now)
    context[0].add(active)
    context[0].flush()
    context[0].connection().exec_driver_sql("ANALYZE model_scan_units")
    return scan["id"], active, now


def test_upgrade_adds_summary_index_without_rewriting_units(context, tmp_path):
    session = context[0]
    populated(context, tmp_path)
    session.commit()
    engine = session.get_bind()
    with engine.begin() as con:
        before = con.exec_driver_sql("SELECT * FROM model_scan_units ORDER BY id").all()
        con.exec_driver_sql("DROP INDEX IF EXISTS " + INDEX)
        con.exec_driver_sql("UPDATE alembic_version SET version_num='0010_window_list_indexes'")
    upgrade_database(str(engine.url))
    upgrade_database(str(engine.url))
    with engine.connect() as con:
        names = {row[0] for row in con.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='index'")}
        after = con.exec_driver_sql("SELECT * FROM model_scan_units ORDER BY id").all()
        assert INDEX in names
        assert before == after
        assert con.exec_driver_sql("PRAGMA foreign_key_check").all() == []


@pytest.mark.parametrize("kind", ["heartbeat", "samples"])
def test_summary_queries_use_covering_index_without_payload_reads(context, tmp_path, kind):
    scan_id, _active, now = populated(context, tmp_path)
    if kind == "heartbeat":
        query = select(func.max(ModelScanUnit.heartbeat_at)).where(
            ModelScanUnit.scan_id == scan_id, ModelScanUnit.status == "running")
        assert context[0].scalar(query) == now
    else:
        query = select(ModelScanUnit.started_at, ModelScanUnit.finished_at, ModelScanUnit.window_count).where(
            ModelScanUnit.scan_id == scan_id, ModelScanUnit.kind == "infer",
            ModelScanUnit.status == "succeeded", ModelScanUnit.started_at.is_not(None),
            ModelScanUnit.finished_at >= now - timedelta(hours=2),
        ).order_by(ModelScanUnit.finished_at.desc()).limit(30)
        rows = context[0].execute(query).all()
        assert len(rows) == 30
        assert all(row.window_count == 10 for row in rows)
        assert [row.finished_at for row in rows] == sorted((row.finished_at for row in rows), reverse=True)
    compiled = query.compile(dialect=context[0].get_bind().dialect, compile_kwargs={"literal_binds": True})
    plan = context[0].connection().exec_driver_sql("EXPLAIN QUERY PLAN " + str(compiled)).all()
    assert any("COVERING INDEX " + INDEX in row[3] for row in plan), plan


def test_summary_index_tracks_heartbeat_and_completion_updates(context, tmp_path):
    scan_id, active, now = populated(context, tmp_path)
    session = context[0]
    active.heartbeat_at = now + timedelta(seconds=15)
    session.flush()
    assert session.scalar(select(func.max(ModelScanUnit.heartbeat_at)).where(
        ModelScanUnit.scan_id == scan_id, ModelScanUnit.status == "running")) == active.heartbeat_at
    active.status = "succeeded"
    active.started_at, active.finished_at, active.window_count = now, now + timedelta(seconds=20), 7
    session.flush()
    assert session.scalar(select(func.max(ModelScanUnit.heartbeat_at)).where(
        ModelScanUnit.scan_id == scan_id, ModelScanUnit.status == "running")) is None
    latest = session.execute(select(ModelScanUnit.started_at, ModelScanUnit.finished_at, ModelScanUnit.window_count).where(
        ModelScanUnit.scan_id == scan_id, ModelScanUnit.status == "succeeded", ModelScanUnit.kind == "infer",
    ).order_by(ModelScanUnit.finished_at.desc()).limit(1)).one()
    assert latest == (now, now + timedelta(seconds=20), 7)
