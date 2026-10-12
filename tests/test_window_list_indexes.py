"""Keep large analysis JSON off the window-list filtering and sorting path."""

import json

import pytest
from das_sentinel.persistence.db import upgrade_database
from das_sentinel.persistence.window_models import WindowAnalysis
from das_sentinel.services.event_windows import enqueue_windows
from sqlalchemy import event

from tests.test_event_windows import context as context  # noqa: PLC0414
from tests.test_event_windows import plan

LIST_INDEXES = {
    "ix_window_analysis_list",
    "ix_model_scan_receipt_ready_window",
    "ix_window_review_latest_bound",
}


def test_migration_adds_list_indexes_without_rewriting_analysis(context):
    session, site, profile, _service = context
    analysis = enqueue_windows(session, site.id, profile.id, [plan()])[0]
    analysis.status = "succeeded"
    analysis.input_sha256 = "a" * 64
    analysis.result_json = '{"candidate":true,"score":2.5}'
    session.commit()
    engine = session.get_bind()
    with engine.connect() as con:
        before = con.exec_driver_sql("SELECT * FROM window_analyses").fetchall()
    upgrade_database(str(engine.url))
    with engine.connect() as con:
        names = {row[0] for row in con.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )}
        after = con.exec_driver_sql("SELECT * FROM window_analyses").fetchall()
    assert LIST_INDEXES <= names
    assert before == after


@pytest.mark.parametrize("filters", [
    {"review_status": "all", "sort": "default"},
    {"review_status": "all", "sort": "borehole_coverage"},
    {"review_status": "unreviewed", "candidate_only": True, "sort": "borehole_coverage"},
])
def test_list_queries_cover_analysis_filter_and_score_without_json_table_reads(context, filters):
    session, site, profile, service = context
    first = enqueue_windows(session, site.id, profile.id, [plan()])[0]
    second = enqueue_windows(session, site.id, profile.id, [plan() | {
        "start_utc": "2023-09-27T00:01:00+00:00",
        "end_utc": "2023-09-27T00:02:00+00:00",
    }])[0]
    for analysis, score, candidate in [(first, 1, False), (second, 3, True)]:
        analysis.status = "succeeded"
        analysis.input_sha256 = "a" * 64
        analysis.result_json = json.dumps({"score": score, "candidate": candidate, "padding": "x" * 100000})
    session.flush()
    # The diagnostic/production database has populated statistics; so must this fixture.
    session.connection().exec_driver_sql("ANALYZE window_analyses")
    session.connection().exec_driver_sql("ANALYZE event_windows")
    captured = []

    def collect(_con, _cursor, sql, parameters, _context, _executemany):
        if sql.startswith("SELECT") and "FROM event_windows JOIN window_analyses" in sql:
            captured.append((sql, parameters))

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", collect)
    try:
        page = service.list(window_seconds=60, **filters)
    finally:
        event.remove(engine, "before_cursor_execute", collect)
    assert page["total"] == (1 if filters.get("candidate_only") else 2)
    assert page["items"][0]["id"] == second.window_id
    assert len(captured) == 2
    for sql, parameters in captured:
        if sql.startswith("SELECT count(") and not filters.get("candidate_only"):
            continue
        steps = session.connection().exec_driver_sql("EXPLAIN QUERY PLAN " + sql, parameters).fetchall()
        assert any("COVERING INDEX ix_window_analysis_list" in row[3] for row in steps), steps


def test_expression_index_updates_when_analysis_result_changes(context):
    session, site, profile, service = context
    analyses = []
    for minute, score in [(0, 1), (1, 2)]:
        start = f"2023-09-27T00:0{minute}:00+00:00"
        stop = f"2023-09-27T00:0{minute + 1}:00+00:00"
        analysis = enqueue_windows(session, site.id, profile.id, [plan() | {
            "start_utc": start, "end_utc": stop,
        }])[0]
        analysis.status = "succeeded"
        analysis.input_sha256 = "a" * 64
        analysis.result_json = json.dumps({"score": score, "candidate": True})
        analyses.append(analysis)
    session.flush()
    assert service.list(window_seconds=60, review_status="all")["items"][0]["id"] == analyses[1].window_id
    first = session.get(WindowAnalysis, analyses[0].id)
    first.result_json = '{"score":4,"candidate":false}'
    session.flush()
    assert service.list(window_seconds=60, review_status="all")["items"][0]["id"] == first.window_id
    candidates = service.list(window_seconds=60, review_status="all", candidate_only=True)
    assert candidates["total"] == 1
    assert candidates["items"][0]["id"] == analyses[1].window_id
