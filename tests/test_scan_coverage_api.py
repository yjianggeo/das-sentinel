from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from das_sentinel.persistence import Repositories
from das_sentinel.persistence.model_scan_models import (
    ModelScan,
    ModelScanReceipt,
    ModelScanUnit,
)
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowSkip,
)
from sqlalchemy import event

from tests import test_dashboard_api as original

dashboard_api = original.dashboard_api
BEIJING = timezone(timedelta(hours=8))


def add_window(session, identities, start, *, status="succeeded", sha="a" * 64, seconds=30):
    window = EventWindow(id=str(uuid4()), site_id=identities[0], start_utc=start,
        end_utc=start + timedelta(seconds=seconds), window_seconds=seconds)
    session.add(window)
    session.flush()
    analysis = WindowAnalysis(window_id=window.id, profile_id=identities[1], status=status,
        plan_json="{}", plan_sha256=str(uuid4()).replace("-", "") * 2,
        algorithm_version="whole-window-v2", input_sha256=sha,
        result_json='{"candidate":false,"score":0}')
    session.add(analysis)
    session.flush()
    window.latest_analysis_id = analysis.id
    return window, analysis


def add_receipt(session, identities, start, *, status="negative", score=.1, sha="b" * 64,
                finished_at=None):
    scan = ModelScan(site_id=identities[0], profile_id=identities[1], request_id=str(uuid4()),
        config_json="{}", config_sha256="c" * 64, status="cancelled")
    session.add(scan)
    session.flush()
    unit = ModelScanUnit(scan_id=scan.id, kind="infer", unit_key=str(uuid4()),
        payload_json="{}", status="succeeded")
    session.add(unit)
    session.flush()
    receipt = ModelScanReceipt(scan_id=scan.id, infer_unit_id=unit.id, start_utc=start,
        inference_key="d" * 64, status=status, score=score, model_input_sha256=sha,
        result_json="{}", result_sha256="e" * 64, finished_at=finished_at)
    session.add(receipt)
    return receipt


def add_skip(session, identities, start, *, seconds=30, policy="empty-regions-v1",
             reason="no_qualified_regions"):
    session.add(WindowSkip(analysis_id=str(uuid4()), window_id=str(uuid4()),
        site_id=identities[0], start_utc=start, window_seconds=seconds,
        plan_sha256="f" * 64, policy_version=policy, worker_id="rule-worker",
        attempt_count=1, reason=reason,
        finished_at=datetime(2026, 10, 1, tzinfo=UTC)))


@pytest.mark.parametrize("url", ["/api/v1/scan-coverage?year=2024",
    "/api/v1/scan-coverage/day?date=2024-01-01"])
def test_scan_coverage_requires_an_authenticated_admin(dashboard_api, url):
    client, _, _ = dashboard_api
    client.cookies.clear()
    assert client.get(url).status_code == 401


def test_empty_coverage_returns_complete_leap_year_and_current_year_default(dashboard_api):
    client, _, _ = dashboard_api
    response = client.get("/api/v1/scan-coverage", params={"year": 2024})
    assert response.status_code == 200, "scan coverage API missing"
    payload = response.json()
    assert payload["year"] == 2024
    assert payload["timezone"] == "Asia/Shanghai"
    assert payload["available_years"] == []
    assert len(payload["days"]) == 366
    assert payload["days"][0] == {"date": "2024-01-01", "scanned_seconds": 0}
    assert payload["days"][59] == {"date": "2024-02-29", "scanned_seconds": 0}
    assert payload["days"][-1] == {"date": "2024-12-31", "scanned_seconds": 0}
    assert all(day["scanned_seconds"] == 0 for day in payload["days"])
    assert payload["total_scanned_seconds"] == payload["scanned_days"] == 0
    before = datetime.now(BEIJING).year
    default = client.get("/api/v1/scan-coverage").json()
    after = datetime.now(BEIJING).year
    assert default["year"] in (before, after)
    assert default["available_years"] == []
    assert client.get("/api/v1/scan-coverage/day", params={"date": "2024-02-29"}).json() == {
        "date": "2024-02-29", "timezone": "Asia/Shanghai", "scanned_seconds": 0,
        "intervals": [],
    }


def test_three_success_evidence_sources_and_batches_are_unioned(dashboard_api):
    client, sessions, identities = dashboard_api
    start = datetime(2024, 2, 28, 16, tzinfo=UTC)
    with sessions.begin() as session:
        window, _ = add_window(session, identities, start)
        session.add(WindowAnalysis(window_id=window.id, profile_id=identities[1],
            status="succeeded", plan_json="{}", plan_sha256="9" * 64,
            algorithm_version="other-version", input_sha256="1" * 64, result_json="{}"))
        add_receipt(session, identities, start)
        add_receipt(session, identities, start)
        add_skip(session, identities, start)
        add_receipt(session, identities, start + timedelta(seconds=30), score=0)
        add_skip(session, identities, start + timedelta(seconds=60))
        add_window(session, identities, start + timedelta(days=1))
    response = client.get("/api/v1/scan-coverage", params={"year": 2024})
    assert response.status_code == 200
    payload = response.json()
    assert payload["available_years"] == [2024]
    assert payload["total_scanned_seconds"] == 120
    assert payload["scanned_days"] == 2
    assert payload["days"][59] == {"date": "2024-02-29", "scanned_seconds": 90}
    assert payload["days"][60] == {"date": "2024-03-01", "scanned_seconds": 30}
    detail = client.get("/api/v1/scan-coverage/day", params={"date": "2024-02-29"}).json()
    assert detail == {"date": "2024-02-29", "timezone": "Asia/Shanghai",
        "scanned_seconds": 90, "intervals": [{"start_at": "2024-02-29T00:00:00+08:00",
            "end_at": "2024-02-29T00:01:30+08:00"}]}


def test_all_sites_use_time_union_and_site_filter_selects_its_evidence(dashboard_api):
    client, sessions, identities = dashboard_api
    start = datetime(2023, 12, 31, 16, tzinfo=UTC)
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(name="other", data_path="/data/other",
            timezone="UTC", cable_type="urban")
        profile = repositories.profiles.add(site_id=site.id, version=1,
            payload={}, is_current=True)
        other = (site.id, profile.id, identities[2])
        add_window(session, identities, start)
        add_receipt(session, other, start)
        add_skip(session, other, start + timedelta(seconds=30))
        add_receipt(session, identities, datetime(2022, 12, 31, 16, tzinfo=UTC))
        other_id = site.id
    all_sites = client.get("/api/v1/scan-coverage").json()
    assert all_sites["year"] == 2024
    assert all_sites["available_years"] == [2023, 2024]
    assert all_sites["total_scanned_seconds"] == 60
    first = client.get("/api/v1/scan-coverage", params={"site_id": identities[0]}).json()
    assert first["total_scanned_seconds"] == 30
    assert first["available_years"] == [2023, 2024]
    other_site = client.get("/api/v1/scan-coverage", params={"site_id": other_id}).json()
    assert other_site["total_scanned_seconds"] == 60
    assert other_site["available_years"] == [2024]
    detail = client.get("/api/v1/scan-coverage/day",
        params={"date": "2024-01-01", "site_id": identities[0]}).json()
    assert detail["scanned_seconds"] == 30
    assert detail["intervals"] == [{"start_at": "2024-01-01T00:00:00+08:00",
        "end_at": "2024-01-01T00:00:30+08:00"}]


def test_beijing_year_boundary_and_midnight_interval_end_are_half_open(dashboard_api):
    client, sessions, identities = dashboard_api
    with sessions.begin() as session:
        add_window(session, identities, datetime(2023, 12, 31, 15, 59, tzinfo=UTC))
        add_skip(session, identities, datetime(2023, 12, 31, 15, 59, 30, tzinfo=UTC))
        add_receipt(session, identities, datetime(2023, 12, 31, 16, tzinfo=UTC))
        add_receipt(session, identities, datetime(2023, 12, 31, 16, 1, tzinfo=UTC))
    previous = client.get("/api/v1/scan-coverage", params={"year": 2023}).json()
    assert len(previous["days"]) == 365
    assert previous["days"][-1] == {"date": "2023-12-31", "scanned_seconds": 60}
    assert previous["total_scanned_seconds"] == 60
    current = client.get("/api/v1/scan-coverage").json()
    assert current["year"] == 2024
    assert current["available_years"] == [2023, 2024]
    assert current["days"][0] == {"date": "2024-01-01", "scanned_seconds": 60}
    previous_detail = client.get("/api/v1/scan-coverage/day", params={"date": "2023-12-31"}).json()
    assert previous_detail["intervals"] == [{"start_at": "2023-12-31T23:59:00+08:00",
        "end_at": "2024-01-01T00:00:00+08:00"}]
    current_detail = client.get("/api/v1/scan-coverage/day", params={"date": "2024-01-01"}).json()
    assert current_detail["intervals"] == [
        {"start_at": "2024-01-01T00:00:00+08:00", "end_at": "2024-01-01T00:00:30+08:00"},
        {"start_at": "2024-01-01T00:01:00+08:00", "end_at": "2024-01-01T00:01:30+08:00"},
    ]


def test_historical_success_counts_even_after_failed_latest_and_retired_profile(dashboard_api):
    client, sessions, identities = dashboard_api
    start = datetime(2024, 6, 1, tzinfo=UTC)
    with sessions.begin() as session:
        window, _ = add_window(session, identities, start)
        replacement = WindowAnalysis(window_id=window.id, profile_id=identities[1],
            status="failed", plan_json="{}", plan_sha256="2" * 64,
            algorithm_version="whole-window-v2", input_sha256=None, result_json=None)
        session.add(replacement)
        session.flush()
        window.latest_analysis_id = replacement.id
        profile = Repositories(session).profiles.get(identities[1])
        profile.is_current = False
    response = client.get("/api/v1/scan-coverage", params={"year": 2024})
    assert response.status_code == 200
    assert response.json()["total_scanned_seconds"] == 30


@pytest.mark.parametrize("status", ["cancelled", "protected", "excluded", "selected", "ready", "negative"])
def test_valid_prediction_survives_later_status_and_missing_finish_time(dashboard_api, status):
    client, sessions, identities = dashboard_api
    with sessions.begin() as session:
        add_receipt(session, identities, datetime(2024, 1, 1, tzinfo=UTC), status=status, score=1)
    response = client.get("/api/v1/scan-coverage", params={"year": 2024})
    assert response.status_code == 200
    assert response.json()["total_scanned_seconds"] == 30


def test_unsuccessful_and_non_30_second_evidence_does_not_count(dashboard_api):
    client, sessions, identities = dashboard_api
    start = datetime(2024, 1, 1, tzinfo=UTC)
    with sessions.begin() as session:
        for index, state in enumerate(("queued", "running", "failed", "cancelled")):
            add_window(session, identities, start + timedelta(minutes=index), status=state)
        add_window(session, identities, start + timedelta(minutes=5), sha="")
        add_window(session, identities, start + timedelta(minutes=6), sha=None)
        add_window(session, identities, start + timedelta(minutes=7), seconds=60)
        for index, state in enumerate(("excluded", "cancelled", "protected", "pending", "unjudgeable")):
            add_receipt(session, identities, start + timedelta(minutes=10 + index),
                status=state, score=None, sha=None, finished_at=start)
        for index, (score, sha) in enumerate(((None, "a" * 64), (.1, None), (.1, ""),
                (.1, "a" * 63), (-.1, "a" * 64), (1.1, "a" * 64))):
            add_receipt(session, identities, start + timedelta(minutes=20 + index), score=score, sha=sha)
        add_skip(session, identities, start + timedelta(minutes=30), seconds=60)
        add_skip(session, identities, start + timedelta(minutes=31), policy="other-policy")
        add_skip(session, identities, start + timedelta(minutes=32), reason="excluded_2023_two_metre")
    payload = client.get("/api/v1/scan-coverage", params={"year": 2024}).json()
    assert payload["available_years"] == []
    assert payload["total_scanned_seconds"] == payload["scanned_days"] == 0
    assert client.get("/api/v1/scan-coverage/day", params={"date": "2024-01-01"}).json()["intervals"] == []


@pytest.mark.parametrize("year", ["abc", "", "2024.5", 1899, 2101])
def test_year_must_be_an_integer_in_the_supported_range(dashboard_api, year):
    client, _, _ = dashboard_api
    assert client.get("/api/v1/scan-coverage", params={"year": year}).status_code == 422


@pytest.mark.parametrize("year", [1900, 2100])
def test_supported_year_limits_are_inclusive(dashboard_api, year):
    client, _, _ = dashboard_api
    response = client.get("/api/v1/scan-coverage", params={"year": year})
    assert response.status_code == 200
    assert response.json()["year"] == year
    assert len(response.json()["days"]) == 365


@pytest.mark.parametrize("params", [{}, {"date": "2024-02-30"}, {"date": "2024-2-01"},
    {"date": "20240101"}, {"date": "2024-01-01T00:00:00Z"},
    {"date": "1899-12-31"}, {"date": "2101-01-01"}])
def test_day_requires_a_real_iso_calendar_date(dashboard_api, params):
    client, _, _ = dashboard_api
    assert client.get("/api/v1/scan-coverage/day", params=params).status_code == 422


@pytest.mark.parametrize("path, params", [("/api/v1/scan-coverage", {"year": 2024}),
    ("/api/v1/scan-coverage/day", {"date": "2024-01-01"})])
def test_unknown_site_is_not_reported_as_empty_coverage(dashboard_api, path, params):
    client, _, identities = dashboard_api
    assert client.get(path, params=params | {"site_id": identities[0]}).status_code == 200
    assert client.get(path, params=params | {"site_id": "missing-site"}).status_code == 404


def test_coverage_queries_read_narrow_columns_without_large_result_payloads(dashboard_api):
    client, sessions, identities = dashboard_api
    with sessions.begin() as session:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        add_window(session, identities, start)
        add_receipt(session, identities, start + timedelta(seconds=30))
        add_skip(session, identities, start + timedelta(seconds=60))
    engine = client.app.state.session_factory.kw["bind"]
    queries = []

    def capture(connection, cursor, statement, parameters, context, executemany):
        if any(table in statement for table in ("window_analyses", "model_scan_receipts", "window_skips")):
            queries.append(statement.lower())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        response = client.get("/api/v1/scan-coverage", params={"year": 2024})
        assert response.status_code == 200
        response = client.get("/api/v1/scan-coverage/day", params={"date": "2024-01-01"})
        assert response.status_code == 200
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert queries
    assert all("result_json" not in query and "plan_json" not in query and "config_json" not in query
               for query in queries)


@pytest.mark.parametrize("filter_site", [False, True])
def test_history_evidence_uses_covering_indexes_instead_of_large_analysis_rows(dashboard_api, filter_site):
    client, sessions, identities = dashboard_api
    with sessions.begin() as session:
        _, analysis = add_window(session, identities, datetime(2024, 1, 1, tzinfo=UTC))
        analysis.result_json = '{"padding":"' + "x" * 100_000 + '"}'
    engine = client.app.state.session_factory.kw["bind"]
    queries = []

    def capture(connection, cursor, statement, parameters, context, executemany):
        if "window_analyses" in statement:
            queries.append((statement, parameters))

    params = {"site_id": identities[0]} if filter_site else {}
    event.listen(engine, "before_cursor_execute", capture)
    try:
        response = client.get("/api/v1/scan-coverage", params=params | {"year": 2024})
        assert response.status_code == 200
        assert response.json()["total_scanned_seconds"] == 30
        response = client.get("/api/v1/scan-coverage/day", params=params | {"date": "2024-01-01"})
        assert response.status_code == 200
        assert response.json()["scanned_seconds"] == 30
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert queries
    with engine.connect() as connection:
        for statement, parameters in queries:
            plan = connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters).all()
            analysis_reads = [row[3] for row in plan if "window_analyses" in row[3]]
            assert analysis_reads, plan
            assert all("COVERING INDEX" in step for step in analysis_reads), plan
            assert any("ix_window_analysis_list" in step for step in analysis_reads), plan


@pytest.mark.parametrize("filter_site", [False, True])
def test_receipt_projection_stays_on_covering_index_with_many_scan_batches(dashboard_api, filter_site):
    client, sessions, identities = dashboard_api
    start = datetime(2023, 12, 31, 16, tzinfo=UTC)
    with sessions.begin() as session:
        for batch in range(16):
            scan = ModelScan(site_id=identities[0], profile_id=identities[1],
                request_id=f"batch-{batch}", config_json="{}", config_sha256="a" * 64)
            session.add(scan)
            session.flush()
            unit = ModelScanUnit(scan_id=scan.id, kind="infer", unit_key="coverage",
                payload_json="{}", status="succeeded")
            session.add(unit)
            session.flush()
            for number in range(40):
                session.add(ModelScanReceipt(scan_id=scan.id, infer_unit_id=unit.id,
                    start_utc=start + timedelta(seconds=number * 30),
                    inference_key="b" * 64, status="negative", score=.1,
                    model_input_sha256="c" * 64, result_sha256="d" * 64,
                    result_json='{"padding":"' + "x" * 20_000 + '"}'))
    engine = client.app.state.session_factory.kw["bind"]
    queries = []

    def capture(connection, cursor, statement, parameters, context, executemany):
        if "FROM model_scan_receipts" in statement:
            queries.append((statement, parameters))

    params = {"site_id": identities[0]} if filter_site else {}
    event.listen(engine, "before_cursor_execute", capture)
    try:
        response = client.get("/api/v1/scan-coverage", params=params | {"year": 2024})
        assert response.status_code == 200
        assert response.json()["available_years"] == [2024]
        assert response.json()["total_scanned_seconds"] == 1200
        response = client.get("/api/v1/scan-coverage/day", params=params | {"date": "2024-01-01"})
        assert response.status_code == 200
        assert response.json()["scanned_seconds"] == 1200
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert queries
    with engine.connect() as connection:
        for statement, parameters in queries:
            plan = connection.exec_driver_sql("EXPLAIN QUERY PLAN " + statement, parameters).all()
            receipt_reads = [row for row in plan if "model_scan_receipts" in row[3]]
            assert receipt_reads, plan
            assert all("COVERING INDEX ix_model_scan_receipt_coverage" in row[3]
                       for row in receipt_reads), plan
            materialized = {row[0] for row in plan if row[3].startswith("MATERIALIZE ")}
            assert all(row[1] in materialized for row in receipt_reads), plan
