"""Keep calendar success-evidence queries off model receipt payload rows."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from das_sentinel.persistence import db
from das_sentinel.persistence.model_scan_models import ModelScanReceipt
from das_sentinel.services.scan_coverage import _available_years, _evidence_queries
from sqlalchemy import event

from tests.test_event_windows import context as context  # noqa: PLC0414
from tests.test_model_scans import create, service

INDEX = "ix_model_scan_receipt_coverage"
REVISION = "0012_model_scan_coverage_index"
BASE = datetime(2024, 1, 1, 16, tzinfo=UTC)


def populated(context, tmp_path):
    session, site, _profile, _windows = context
    api = service(context, tmp_path)
    scan = create(api, context, tmp_path)
    unit = api._add_unit(scan["id"], "infer", "coverage", {"windows": []})
    for number in range(160):
        start = BASE + timedelta(seconds=(number % 80) * 30)
        if number >= 80:
            start = start.replace(year=2023)
        session.add(ModelScanReceipt(
            scan_id=scan["id"], infer_unit_id=unit.id, start_utc=start,
            inference_key=str(number), status="negative", score=.01,
            model_input_sha256="a" * 64,
            result_json='{"padding":"' + "x" * 20000 + '"}', result_sha256="b" * 64,
        ))
    for number, (status, score, signature) in enumerate((
        ("pending", None, None), ("unjudgeable", -.1, "a" * 64),
        ("negative", .01, "short"), ("cancelled", .2, "a" * 64),
    )):
        session.add(ModelScanReceipt(
            scan_id=scan["id"], infer_unit_id=unit.id,
            start_utc=BASE + timedelta(hours=2, seconds=number * 30),
            inference_key="edge-" + str(number), status=status, score=score,
            model_input_sha256=signature, result_json="{}", result_sha256="",
        ))
    session.flush()
    session.connection().exec_driver_sql("ANALYZE model_scan_receipts")
    return session, site


def test_coverage_index_migration_round_trip_preserves_receipts(context, tmp_path):
    session, _site = populated(context, tmp_path)
    session.commit()
    engine = session.get_bind()
    with engine.begin() as connection:
        before = connection.exec_driver_sql("SELECT * FROM model_scan_receipts ORDER BY id").all()
        connection.exec_driver_sql("DROP INDEX IF EXISTS " + INDEX)
        connection.exec_driver_sql("UPDATE alembic_version SET version_num='0011_model_scan_summary_index'")
    db.upgrade_database(str(engine.url))
    db.upgrade_database(str(engine.url))
    with engine.connect() as connection:
        names = {row[0] for row in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='index'")}
        assert INDEX in names
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == REVISION
        assert connection.exec_driver_sql("SELECT * FROM model_scan_receipts ORDER BY id").all() == before
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []

    configuration = Config()
    configuration.set_main_option("script_location", str(Path(db.__file__).parent / "migrations"))
    configuration.set_main_option("sqlalchemy.url", str(engine.url))
    command.downgrade(configuration, "0011_model_scan_summary_index")
    with engine.connect() as connection:
        names = {row[0] for row in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='index'")}
        assert INDEX not in names
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == "0011_model_scan_summary_index"
        assert connection.exec_driver_sql("SELECT * FROM model_scan_receipts ORDER BY id").all() == before
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    db.upgrade_database(str(engine.url))
    with engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT * FROM model_scan_receipts ORDER BY id").all() == before
        assert connection.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"


@pytest.mark.parametrize("kind", ["year", "day"])
@pytest.mark.parametrize("filter_site", [False, True])
def test_calendar_receipt_queries_use_covering_index(context, tmp_path, kind, filter_site):
    session, site = populated(context, tmp_path)
    start = datetime(2023, 12, 31, 16, tzinfo=UTC) if kind == "year" else BASE
    end = datetime(2024, 12, 31, 16, tzinfo=UTC) if kind == "year" else BASE + timedelta(days=1)
    query = _evidence_queries(site.id if filter_site else None, start, end)[1]
    starts = session.scalars(query).all()
    assert len(starts) == 81
    assert min(starts) == datetime(2024, 1, 1, 16, tzinfo=UTC)
    assert max(starts) == datetime(2024, 1, 1, 18, 1, 30, tzinfo=UTC)
    compiled = query.compile(dialect=session.get_bind().dialect, compile_kwargs={"literal_binds": True})
    plan = session.connection().exec_driver_sql("EXPLAIN QUERY PLAN " + str(compiled)).all()
    assert any("COVERING INDEX " + INDEX in row[3] for row in plan), plan


@pytest.mark.parametrize("filter_site", [False, True])
def test_available_years_receipt_query_uses_covering_index(context, tmp_path, filter_site):
    session, site = populated(context, tmp_path)
    statements = []

    def capture(_connection, _cursor, statement, parameters, _execution_context, _many):
        if "FROM model_scan_receipts" in statement:
            statements.append((statement, parameters))

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", capture)
    try:
        assert _available_years(session, site.id if filter_site else None) == [2023, 2024]
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert len(statements) == 1
    statement, parameters = statements[0]
    plan = session.connection().exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters).all()
    assert any("COVERING INDEX " + INDEX in row[3] for row in plan), plan
