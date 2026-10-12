import json
from datetime import UTC, datetime, timedelta

import pytest

from tests.test_events_api import WORKER_TOKEN

pytest_plugins = ("tests.test_events_api",)
from das_sentinel.persistence.db import (
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import ProcessingProfile, Site
from das_sentinel.persistence.window_models import EventWindow
from das_sentinel.services.event_windows import (
    WindowError,
    WindowService,
    enqueue_windows,
)
from sqlalchemy import select


@pytest.fixture
def context(tmp_path, monkeypatch):
    import sys

    from das_sentinel.services.discovery import fingerprint

    source_path = tmp_path / "source.h5"
    source_path.write_bytes(b"synthetic source")
    source = {
        "path": str(source_path),
        "fingerprint": fingerprint(source_path),
        "size_bytes": source_path.stat().st_size,
        "mtime_ns": source_path.stat().st_mtime_ns,
        "start_utc": "2023-09-26T23:59:00+00:00",
        "end_utc": "2023-09-27T00:02:00+00:00",
        "sample_rate_hz": 100,
        "distance_m": [0.0, 1.0],
        "unit": "strain",
    }
    original_plan = plan

    def source_plan(coverage="complete"):
        return original_plan(coverage) | {"sources": [source.copy()]}

    monkeypatch.setattr(sys.modules[__name__], "plan", source_plan)
    url = f"sqlite+pysqlite:///{tmp_path / 'windows.sqlite3'}"
    upgrade_database(url)
    engine = create_local_engine(url)
    with session_factory(engine)() as session:
        site = Site(
            name="test",
            data_path=str(tmp_path),
            timezone="Asia/Shanghai",
            cable_type="downhole",
        )
        session.add(site)
        session.flush()
        profile = ProcessingProfile(
            site_id=site.id, version=1, payload_json="{}", is_current=True
        )
        session.add(profile)
        session.flush()
        yield (
            session,
            site,
            profile,
            WindowService(session, tmp_path / "window-artifacts"),
        )
    engine.dispose()


def plan(coverage="complete"):
    return {
        "start_utc": "2023-09-27T00:00:00+00:00",
        "end_utc": "2023-09-27T00:01:00+00:00",
        "sources": [],
        "coverage": coverage,
    }


def test_window_identity_idempotency_and_versions(context):
    s, site, p, _service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    assert enqueue_windows(s, site.id, p.id, [plan()])[0].id == a.id
    other = plan()
    other["algorithm_version"] = "future"
    b = enqueue_windows(s, site.id, p.id, [other])[0]
    assert a.window_id == b.window_id and a.id != b.id
    assert len(s.scalars(select(EventWindow)).all()) == 1


def test_coverage_sort_is_independent_and_ignores_stale_measurements(context):
    from das_sentinel.persistence.window_models import WindowSpatialCoverage

    s, site, p, service = context
    first = enqueue_windows(s, site.id, p.id, [plan()], priority=9000)[0]
    second = enqueue_windows(s, site.id, p.id, [plan() | {
        "start_utc": "2023-09-27T00:01:00+00:00", "end_utc": "2023-09-27T00:02:00+00:00",
    }])[0]
    for analysis, score in [(first, 10), (second, 90)]:
        analysis.status, analysis.input_sha256 = "succeeded", "a" * 64
        analysis.result_json = '{"score":0.2}'
        s.add(WindowSpatialCoverage(analysis_id=analysis.id, input_sha256="a" * 64,
            algorithm_version="borehole-coverage-v1", score=score,
            payload_json=json.dumps({"score": score, "algorithm_version": "borehole-coverage-v1"})))
    s.flush()
    before = [(a.result_json, a.input_sha256) for a in (first, second)]
    original = service.list(review_status="all", window_seconds=60)
    assert original["items"][0]["id"] == first.window_id
    ranked = service.list(review_status="all", window_seconds=60, sort="borehole_coverage", limit=1)
    assert ranked["total"] == 2 and ranked["items"][0]["id"] == second.window_id
    assert ranked["items"][0]["spatial_coverage"]["score"] == 90
    assert ranked["items"][0]["latest_review"] is None
    second.input_sha256 = "b" * 64
    s.flush()
    ranked = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
    assert ranked["items"][0]["id"] == first.window_id
    assert ranked["items"][1]["spatial_coverage"] is None
    assert first.result_json == before[0][0] and second.result_json == before[1][0]


def test_coverage_upgrade_keeps_legacy_scores_visible_and_sorts_adjusted_scores(context, monkeypatch):
    from das_sentinel.persistence.window_models import WindowSpatialCoverage

    monkeypatch.setattr("das_sentinel.borehole_coverage.BOREHOLE_COVERAGE_VERSION", "borehole-coverage-v3")
    monkeypatch.setattr("das_sentinel.borehole_coverage.BOREHOLE_COVERAGE_READ_VERSIONS",
        ("borehole-coverage-v1", "borehole-coverage-v2", "borehole-coverage-v3"), raising=False)
    s, site, p, service = context
    legacy = enqueue_windows(s, site.id, p.id, [plan()])[0]
    current = enqueue_windows(s, site.id, p.id, [plan() | {
        "start_utc": "2023-09-27T00:01:00+00:00", "end_utc": "2023-09-27T00:02:00+00:00",
    }])[0]
    for analysis, version, score, coverage_score in [
        (legacy, "borehole-coverage-v1", 75, 75),
        (current, "borehole-coverage-v2", 68, 80),
    ]:
        analysis.status, analysis.input_sha256, analysis.result_json = "succeeded", "a" * 64, "{}"
        payload = {"algorithm_version": version, "score": score}
        if version.endswith("v2"):
            payload.update(coverage_score=coverage_score, common_mode_penalty=12)
        s.add(WindowSpatialCoverage(analysis_id=analysis.id, input_sha256="a" * 64,
            algorithm_version=version, score=score, payload_json=json.dumps(payload)))
    s.flush()
    page = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
    assert page["total"] == 2
    assert [item["id"] for item in page["items"]] == [legacy.window_id, current.window_id]
    assert page["items"][0]["spatial_coverage"]["score"] == 75
    assert page["items"][1]["spatial_coverage"]["coverage_score"] == 80
    assert page["items"][1]["spatial_coverage"]["score"] == 68


def common_mode_window(context, minute, *, score=10, coverage_score=90,
                       version="borehole-coverage-v2", labels=None):
    from das_sentinel.persistence.models import Admin
    from das_sentinel.persistence.window_models import (
        WindowReview,
        WindowSpatialCoverage,
    )

    session, site, profile, _service = context
    start = datetime.fromisoformat(plan()["start_utc"]) + timedelta(minutes=minute)
    analysis = enqueue_windows(session, site.id, profile.id, [plan() | {
        "start_utc": start.isoformat(), "end_utc": (start + timedelta(minutes=1)).isoformat(),
    }])[0]
    analysis.status, analysis.input_sha256, analysis.result_json = "succeeded", "a" * 64, '{"score":0.2}'
    payload = {"algorithm_version": version, "score": score, "coverage_score": coverage_score,
        "common_mode_penalty": coverage_score - score, "common_mode_status": "component_supported",
        "common_mode_energy_fraction": .95, "common_mode_channel_fraction": .9,
        "residual_coverage_score": 12}
    if version == "borehole-coverage-v1":
        payload = {"algorithm_version": version, "score": score}
    session.add(WindowSpatialCoverage(analysis_id=analysis.id, input_sha256=analysis.input_sha256,
        algorithm_version=version, score=score, payload_json=json.dumps(payload)))
    if labels is not None:
        if session.get(Admin, 1) is None:
            session.add(Admin(id=1, password_hash="test"))
            session.flush()
        session.add(WindowReview(window_id=analysis.window_id, analysis_id=analysis.id,
            input_sha256=analysis.input_sha256, labels_json=json.dumps(labels),
            request_id=f"common-mode-{minute}", admin_id=1,
            created_at=datetime(2026, 10, 9, tzinfo=UTC)))
    session.flush()
    return analysis


@pytest.mark.parametrize("labels, protected, expected_score", [
    ({"earthquake": "present"}, True, 90),
    ({"earthquake": "absent", "blast": "present", "traffic": "unknown"}, True, 90),
    ({"earthquake": "absent"}, False, 10),
    ({"earthquake": "unknown"}, False, 10),
    (None, False, 10),
])
def test_common_mode_score_protects_only_current_positive_reviews(context, labels, protected, expected_score):
    from das_sentinel.persistence.window_models import (
        WindowReview,
        WindowSpatialCoverage,
    )

    session, _site, _profile, service = context
    analysis = common_mode_window(context, 0, labels=labels)
    stored = session.get(WindowSpatialCoverage, analysis.id)
    before = (stored.score, stored.payload_json, analysis.result_json, analysis.input_sha256,
        [row.labels_json for row in session.scalars(select(WindowReview))])
    for item in [service.detail(analysis.window_id),
                 service.list(review_status="all", window_seconds=60)["items"][0],
                 service.list(review_status="all", window_seconds=60, sort="borehole_coverage")["items"][0]]:
        payload = item["spatial_coverage"]
        assert payload["score"] == expected_score
        assert payload["review_protected"] is protected
        assert payload["common_mode_penalty"] == (0 if protected else 80)
        assert payload["coverage_score"] == 90
        assert payload["common_mode_status"] == "component_supported"
        assert payload["common_mode_energy_fraction"] == .95
        assert payload["common_mode_channel_fraction"] == .9
        assert payload["residual_coverage_score"] == 12
    assert (stored.score, stored.payload_json, analysis.result_json, analysis.input_sha256,
        [row.labels_json for row in session.scalars(select(WindowReview))]) == before


def test_common_mode_positive_legacy_coverage_falls_back_to_saved_score(context):
    _session, _site, _profile, service = context
    analysis = common_mode_window(context, 0, score=75, version="borehole-coverage-v1",
        labels={"earthquake": "present"})
    payload = service.detail(analysis.window_id)["spatial_coverage"]
    assert payload["review_protected"] is True
    assert payload["score"] == 75 and payload["common_mode_penalty"] == 0


@pytest.mark.parametrize("reason", ["analysis", "input", "latest_unknown", "latest_noise", "source"])
def test_common_mode_positive_review_must_match_current_analysis(context, reason):
    from das_sentinel.persistence.window_models import (
        WindowReview,
        WindowSpatialCoverage,
    )

    session, site, profile, service = context
    analysis = common_mode_window(context, 0, labels={"earthquake": "present"})
    if reason == "analysis":
        newer = enqueue_windows(session, site.id, profile.id, [plan() | {
            "start_utc": "2023-09-27T00:01:00+00:00", "end_utc": "2023-09-27T00:02:00+00:00",
        }])[0]
        session.add(WindowReview(window_id=analysis.window_id, analysis_id=newer.id,
            input_sha256=analysis.input_sha256, labels_json='{"earthquake":"present"}',
            request_id=reason, admin_id=1, created_at=datetime(2026, 10, 9, 0, 0, 1, tzinfo=UTC)))
    elif reason == "input":
        analysis.input_sha256 = "b" * 64
        session.get(WindowSpatialCoverage, analysis.id).input_sha256 = "b" * 64
    elif reason.startswith("latest_"):
        session.add(WindowReview(window_id=analysis.window_id, analysis_id=analysis.id,
            input_sha256=analysis.input_sha256,
            labels_json=json.dumps({"earthquake": "unknown" if reason == "latest_unknown" else "absent"}),
            request_id=reason, admin_id=1, created_at=datetime(2026, 10, 9, 0, 0, 1, tzinfo=UTC)))
    elif reason == "source":
        source = json.loads(analysis.plan_json)["sources"][0]
        from pathlib import Path
        Path(source["path"]).write_bytes(b"changed source")
    session.flush()
    payload = service.detail(analysis.window_id)["spatial_coverage"]
    assert payload["review_protected"] is False
    assert payload["score"] == 10 and payload["common_mode_penalty"] == 80


def test_common_mode_protected_score_orders_before_pagination(context):
    _session, _site, _profile, service = context
    protected = common_mode_window(context, 0, labels={"earthquake": "present"})
    unreviewed = common_mode_window(context, 1, score=80)
    noise = common_mode_window(context, 2, score=70, labels={"earthquake": "absent"})
    unknown = common_mode_window(context, 3, score=50, labels={"earthquake": "unknown"})
    page = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
    assert [item["id"] for item in page["items"]] == [
        protected.window_id, unreviewed.window_id, noise.window_id, unknown.window_id]
    assert [item["spatial_coverage"]["score"] for item in page["items"]] == [90, 80, 70, 50]
    page = service.list(review_status="all", window_seconds=60, sort="borehole_coverage", limit=1, offset=1)
    assert page["total"] == 4 and page["items"][0]["id"] == unreviewed.window_id
    assert service.list(review_status="reviewed_earthquake", window_seconds=60, sort="borehole_coverage")["total"] == 1
    assert service.list(review_status="reviewed_noise", window_seconds=60, sort="borehole_coverage")["total"] == 1
    assert service.list(review_status="unreviewed", window_seconds=60, sort="borehole_coverage")["total"] == 2


@pytest.mark.parametrize("initial_state", ["present", "absent"])
def test_common_mode_sort_and_display_use_same_review_snapshot(context, monkeypatch, initial_state):
    from das_sentinel.persistence.window_models import WindowReview

    session, _site, _profile, service = context
    first = common_mode_window(context, 0, labels={"earthquake": initial_state})
    second = common_mode_window(context, 1, score=50)
    session.commit()
    sessions = session_factory(session.get_bind())
    original = service._favorite_ids
    published = False

    def publish_after_ordering(*args):
        nonlocal published
        if not published:
            published = True
            with sessions.begin() as writer:
                writer.add(WindowReview(window_id=first.window_id, analysis_id=first.id,
                    input_sha256=first.input_sha256,
                    labels_json=json.dumps({"earthquake": "absent" if initial_state == "present" else "present"}),
                    request_id="concurrent-change", admin_id=1,
                    created_at=datetime(2026, 10, 9, 0, 0, 1, tzinfo=UTC)))
        return original(*args)

    monkeypatch.setattr(service, "_favorite_ids", publish_after_ordering)
    page = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
    expected_order = [first.window_id, second.window_id] if initial_state == "present" else [second.window_id, first.window_id]
    assert [item["id"] for item in page["items"]] == expected_order
    payload = next(item["spatial_coverage"] for item in page["items"] if item["id"] == first.window_id)
    assert payload["review_protected"] is (initial_state == "present")
    assert payload["score"] == (90 if initial_state == "present" else 10)
    fresh = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
    assert [item["id"] for item in fresh["items"]] == expected_order[::-1]


def test_common_mode_stale_inherited_positive_does_not_protect_score(context):
    from das_sentinel.persistence.models import Review
    from das_sentinel.persistence.window_models import WindowLabelImport, WindowReview
    from das_sentinel.services.window_label_imports import _snapshot

    from tests.test_window_label_imports import legacy

    session, _site, _profile, service = context
    event, _source_review = legacy(context)
    analysis = common_mode_window(context, 0, labels={"earthquake": "present"})
    review = session.scalar(select(WindowReview))
    session.add(WindowLabelImport(window_id=analysis.window_id, analysis_id=analysis.id,
        payload_json=json.dumps({"records": [_snapshot(session, event)]}),
        admin_id=1, applied_review_id=review.id))
    session.flush()
    assert service.detail(analysis.window_id)["spatial_coverage"]["review_protected"] is True
    session.add(Review(event_id=event.id, admin_id=1, label="noise", note="corrected source",
        created_at=_source_review.created_at + timedelta(seconds=1)))
    session.flush()
    for item in [service.detail(analysis.window_id),
                 service.list(review_status="all", window_seconds=60, sort="borehole_coverage")["items"][0]]:
        assert item["spatial_coverage"]["review_protected"] is False
        assert item["spatial_coverage"]["score"] == 10
        assert item["latest_review"]["stale"] is True
    assert service.list(review_status="reviewed_earthquake", window_seconds=60)["total"] == 0


def test_common_mode_noncurrent_profile_cannot_protect_coverage(context):
    session, _site, profile, service = context
    analysis = common_mode_window(context, 0, labels={"earthquake": "present"})
    profile.is_current = False
    session.flush()
    assert service.detail(analysis.window_id)["spatial_coverage"] is None
    page = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
    assert page["items"][0]["spatial_coverage"] is None and page["total"] == 1
    assert service.list(review_status="reviewed_earthquake", window_seconds=60)["total"] == 0


def test_common_mode_protected_unknown_coverage_keeps_null_sort(context):
    from das_sentinel.persistence.window_models import WindowSpatialCoverage

    session, _site, _profile, service = context
    unknown = common_mode_window(context, 0, labels={"earthquake": "present"})
    known = common_mode_window(context, 1, score=50)
    row = session.get(WindowSpatialCoverage, unknown.id)
    row.score = None
    row.payload_json = json.dumps(json.loads(row.payload_json) | {"coverage_score": None, "score": None})
    session.flush()
    page = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
    assert [item["id"] for item in page["items"]] == [known.window_id, unknown.window_id]
    assert page["items"][1]["spatial_coverage"]["score"] is None
    assert page["items"][1]["spatial_coverage"]["review_protected"] is True


def test_common_mode_saving_protected_score_keeps_measurement_candidate(context):
    from das_sentinel.borehole_coverage import BOREHOLE_COVERAGE_VERSION
    from das_sentinel.persistence.window_models import WindowSpatialCoverage

    session, _site, _profile, service = context
    analysis = common_mode_window(context, 0, labels={"earthquake": "present"})
    measured = {"algorithm_version": BOREHOLE_COVERAGE_VERSION, "score": 10,
        "coverage_score": 90, "common_mode_penalty": 80, "common_mode_status": "component_supported",
        "analysis_id": analysis.id, "input_sha256": analysis.input_sha256}
    saved = service.save_spatial_coverage(analysis.window_id, measured)
    assert saved["review_protected"] is True and saved["score"] == 90 and saved["common_mode_penalty"] == 0
    row = session.get(WindowSpatialCoverage, analysis.id)
    assert row.score == 10 and json.loads(row.payload_json)["common_mode_penalty"] == 80
    assert "review_protected" not in json.loads(row.payload_json)
    prepared = service.prepare_spatial_coverage(analysis.window_id, analysis.id, analysis.input_sha256)
    assert prepared["score"] == 10 and prepared["common_mode_penalty"] == 80
    service.save_spatial_coverage(analysis.window_id, prepared)
    assert row.score == 10 and json.loads(row.payload_json)["common_mode_penalty"] == 80


@pytest.mark.parametrize("state", ["absent", "unknown"])
def test_common_mode_saving_nonpositive_review_skips_source_validity(context, monkeypatch, state):
    from das_sentinel.borehole_coverage import BOREHOLE_COVERAGE_VERSION

    _session, _site, _profile, service = context
    analysis = common_mode_window(context, 0, labels={"earthquake": state})

    def reject_validity(*_args):
        pytest.fail("Non-positive coverage return must skip review/source validity checks")

    monkeypatch.setattr(service, "_review_dict", reject_validity)
    monkeypatch.setattr(service, "_current", reject_validity)
    measured = {"algorithm_version": BOREHOLE_COVERAGE_VERSION, "score": 10,
        "coverage_score": 90, "common_mode_penalty": 80, "common_mode_status": "component_supported",
        "analysis_id": analysis.id, "input_sha256": analysis.input_sha256}
    saved = service.save_spatial_coverage(analysis.window_id, measured)
    assert saved["review_protected"] is False
    assert saved["score"] == 10 and saved["common_mode_penalty"] == 80


@pytest.mark.parametrize("old_version", ["borehole-coverage-v1", "borehole-coverage-v2"])
def test_coverage_upgrade_remeasures_legacy_cache_without_changing_analysis(context, monkeypatch, old_version):
    import hashlib

    import numpy as np
    from das_sentinel.persistence.window_models import WindowSpatialCoverage

    monkeypatch.setattr("das_sentinel.borehole_coverage.BOREHOLE_COVERAGE_VERSION", "borehole-coverage-v3")
    monkeypatch.setattr("das_sentinel.borehole_coverage.BOREHOLE_COVERAGE_READ_VERSIONS",
        ("borehole-coverage-v1", "borehole-coverage-v2", "borehole-coverage-v3"), raising=False)
    s, site, p, service = context
    analysis = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("w")
    result = make_result(service, analysis)
    service.result(analysis.id, "w", 1)
    filtered = service.artifact_root / analysis.id / "attempt-1" / "filtered.npz"
    np.savez_compressed(filtered, waveform=np.zeros(result["shape"], dtype=np.float32))
    result["artifacts"].append({"kind": "filtered_waveform", "path": filtered.name,
        "media_type": "application/octet-stream", "sha256": hashlib.sha256(filtered.read_bytes()).hexdigest()})
    analysis.result_json = json.dumps(result)
    before = (analysis.result_json, analysis.input_sha256)
    legacy = WindowSpatialCoverage(analysis_id=analysis.id, input_sha256=analysis.input_sha256,
        algorithm_version=old_version, score=75,
        payload_json=json.dumps({"algorithm_version": old_version, "score": 75}))
    s.add(legacy)
    s.flush()
    measured = service.prepare_spatial_coverage(analysis.window_id, analysis.id, analysis.input_sha256)
    assert measured["algorithm_version"] == "borehole-coverage-v3"
    assert measured["score"] is None  # This fixture has only two surface channels.
    assert legacy.algorithm_version == old_version
    with pytest.raises(WindowError, match="coverage_version_changed"):
        service.save_spatial_coverage(analysis.window_id, measured | {"algorithm_version": old_version})
    service.save_spatial_coverage(analysis.window_id, measured)
    assert service.detail(analysis.window_id)["spatial_coverage"]["algorithm_version"] == "borehole-coverage-v3"
    assert (analysis.result_json, analysis.input_sha256) == before


def test_coverage_sorted_count_does_not_read_measurements_or_large_analysis_fields(context):
    import sqlite3

    from das_sentinel.persistence.window_models import WindowSpatialCoverage
    from sqlalchemy import event

    s, site, p, service = context
    analysis = enqueue_windows(s, site.id, p.id, [plan()])[0]
    analysis.status, analysis.input_sha256, analysis.result_json = "succeeded", "a" * 64, "{}"
    s.add(WindowSpatialCoverage(analysis_id=analysis.id, input_sha256="a" * 64,
        algorithm_version="borehole-coverage-v1", score=10,
        payload_json='{"score":10,"algorithm_version":"borehole-coverage-v1"}'))
    s.flush()
    engine = s.get_bind()
    connection = s.connection().connection.driver_connection

    def authorize(action, table, column, _database, _trigger):
        if action == sqlite3.SQLITE_READ and (table == "window_spatial_coverage"
                or (table == "window_analyses" and column in
                    {"status", "input_sha256", "plan_json", "result_json"})):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def restrict_count(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith("SELECT COUNT("):
            connection.set_authorizer(authorize)

    def restore_reads(*_args):
        connection.set_authorizer(None)

    event.listen(engine, "before_cursor_execute", restrict_count)
    event.listen(engine, "after_cursor_execute", restore_reads)
    try:
        page = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
        assert page["total"] == 1
        assert page["items"][0]["spatial_coverage"]["score"] == 10
    finally:
        connection.set_authorizer(None)
        event.remove(engine, "before_cursor_execute", restrict_count)
        event.remove(engine, "after_cursor_execute", restore_reads)


@pytest.mark.parametrize("initial_second_score", [None, 1])
def test_coverage_sort_and_display_use_same_measurement_snapshot(context, monkeypatch, initial_second_score):
    from das_sentinel.persistence.window_models import WindowSpatialCoverage

    s, site, p, service = context
    first = enqueue_windows(s, site.id, p.id, [plan()])[0]
    second = enqueue_windows(s, site.id, p.id, [plan() | {
        "start_utc": "2023-09-27T00:01:00+00:00", "end_utc": "2023-09-27T00:02:00+00:00",
    }])[0]
    for analysis, score in [(first, 10), (second, initial_second_score)]:
        analysis.status, analysis.input_sha256, analysis.result_json = "succeeded", "a" * 64, "{}"
        if score is not None:
            s.add(WindowSpatialCoverage(analysis_id=analysis.id, input_sha256="a" * 64,
                algorithm_version="borehole-coverage-v1", score=score,
                payload_json=json.dumps({"score": score, "algorithm_version": "borehole-coverage-v1"})))
    s.commit()
    sessions = session_factory(s.get_bind())
    original = service._favorite_ids
    published = False

    def publish_after_ordering(*args):
        nonlocal published
        if not published:
            published = True
            with sessions.begin() as writer:
                row = writer.get(WindowSpatialCoverage, second.id)
                if row is None:
                    row = WindowSpatialCoverage(analysis_id=second.id, input_sha256="a" * 64,
                        algorithm_version="borehole-coverage-v1")
                    writer.add(row)
                row.score = 90
                row.payload_json = json.dumps({"score": 90, "algorithm_version": "borehole-coverage-v1"})
        return original(*args)

    monkeypatch.setattr(service, "_favorite_ids", publish_after_ordering)
    page = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
    assert [item["id"] for item in page["items"]] == [first.window_id, second.window_id]
    assert page["items"][0]["spatial_coverage"]["score"] == 10
    if initial_second_score is None:
        assert page["items"][1]["spatial_coverage"] is None
    else:
        assert page["items"][1]["spatial_coverage"]["score"] == initial_second_score
    fresh = service.list(review_status="all", window_seconds=60, sort="borehole_coverage")
    assert fresh["items"][0]["id"] == second.window_id
    assert fresh["items"][0]["spatial_coverage"]["score"] == 90


def test_coverage_commit_rejects_changed_analysis_and_preserves_labels(context):
    from das_sentinel.borehole_coverage import BOREHOLE_COVERAGE_VERSION
    s, site, p, service = context
    analysis = enqueue_windows(s, site.id, p.id, [plan()])[0]
    analysis.status, analysis.input_sha256, analysis.result_json = "succeeded", "a" * 64, "{}"
    payload = {"analysis_id": analysis.id, "input_sha256": "a" * 64,
        "algorithm_version": BOREHOLE_COVERAGE_VERSION, "score": 75.0,
        "coverage_fraction": .75, "deep_coverage_fraction": .8, "continuous_span_m": 2000.,
        "median_correlation": .7, "valid_coverage_fraction": 1.}
    saved = service.save_spatial_coverage(analysis.window_id, payload)
    assert saved["score"] == 75 and service.detail(analysis.window_id)["spatial_coverage"] == saved
    assert analysis.result_json == "{}" and service.detail(analysis.window_id)["latest_review"] is None
    analysis.input_sha256 = "b" * 64
    with pytest.raises(WindowError, match="stale_analysis"):
        service.save_spatial_coverage(analysis.window_id, payload)
    assert service.detail(analysis.window_id)["spatial_coverage"] is None


def test_coverage_becomes_unknown_when_profile_is_superseded(context):
    from das_sentinel.borehole_coverage import BOREHOLE_COVERAGE_VERSION
    s, site, p, service = context
    analysis = enqueue_windows(s, site.id, p.id, [plan()])[0]
    analysis.status, analysis.input_sha256, analysis.result_json = "succeeded", "a" * 64, "{}"
    payload = {"analysis_id": analysis.id, "input_sha256": "a" * 64,
        "algorithm_version": BOREHOLE_COVERAGE_VERSION, "score": 95.0}
    service.save_spatial_coverage(analysis.window_id, payload)
    p.is_current = False
    s.flush()
    assert service.list(review_status="all", window_seconds=60,
        sort="borehole_coverage")["items"][0]["spatial_coverage"] is None
    with pytest.raises(WindowError, match="stale_analysis"):
        service.save_spatial_coverage(analysis.window_id, payload)


def test_coverage_measurement_rejects_corrupt_cached_waveform(context):
    s, site, p, service = context
    analysis = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("w")
    make_result(service, analysis)
    service.result(analysis.id, "w", 1)
    path = service.artifact_root / analysis.id / "attempt-1" / "waveform.npz"
    path.write_bytes(b"corrupt cache")
    with pytest.raises(WindowError, match="artifact_hash_mismatch"):
        service.prepare_spatial_coverage(analysis.window_id, analysis.id, analysis.input_sha256)


def test_coverage_api_rejects_invalid_sort_and_requires_csrf(event_api):
    client, _settings, _sessions, _csrf, _ids = event_api
    assert client.get("/api/v1/event-windows?sort=unknown").status_code == 422
    response = client.post("/api/v1/event-windows/missing/spatial-coverage", json={
        "analysis_id": "missing", "input_sha256": "a" * 64,
    })
    assert response.status_code == 403


def test_model_review_attention_is_separate_from_human_review(context):
    s, site, p, service = context
    analysis = enqueue_windows(s, site.id, p.id, [plan()])[0]
    attention = {"kind": "FN", "source_review_id": "original-review", "note": "Recheck held-out miss"}
    analysis.result_json = json.dumps({"model_review_attention": attention})
    s.flush()
    listed = service.list(review_status="all", window_seconds=60)["items"][0]
    detail = service.detail(analysis.window_id)
    assert listed["model_review_attention"] == detail["model_review_attention"] == attention
    assert listed["latest_review"] is None


def test_incomplete_never_claimed_and_cancel_late_result(context):
    s, site, p, service = context
    bad = enqueue_windows(s, site.id, p.id, [plan("incomplete")])[0]
    assert bad.status == "failed" and bad.error_code == "coverage_incomplete"
    assert service.detail(bad.window_id)['analysis_error_code'] == 'coverage_incomplete'
    assert service.claim("one") is None
    good = enqueue_windows(s, site.id, p.id, [plan()])[0]
    assert service.claim("one")["id"] == good.id
    with pytest.raises(WindowError):
        service.heartbeat(good.id, "two", 1)
    service.cancel(good.id)
    service.cancel(good.id)
    with pytest.raises(WindowError):
        service.result(good.id, "one", 1)
    with pytest.raises(WindowError):
        service.retry(good.id)


def test_review_multilabel_idempotency_and_stale(context):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    a.status = "succeeded"
    a.input_sha256 = "a" * 64
    a.result_json = "{}"
    s.flush()
    payload = {
        "analysis_id": a.id,
        "input_sha256": a.input_sha256,
        "labels": {"earthquake": "present", "blast": "unknown", "traffic": "present"},
        "note": "",
        "request_id": "r1",
        "admin_id": 1,
    }
    # Foreign keys require the singleton administrator.
    from das_sentinel.persistence.models import Admin

    s.add(Admin(id=1, password_hash="test"))
    s.flush()
    r = service.review(a.window_id, **payload)
    assert r["labels"] == {"earthquake": "present"} and not r["background_only"]
    assert service.review(a.window_id, **payload)["id"] == r["id"]
    with pytest.raises(WindowError):
        service.review(a.window_id, **(payload | {"note": "different"}))
    p.is_current = False
    s.flush()
    s.add(
        ProcessingProfile(
            site_id=site.id, version=2, payload_json='{"changed":true}', is_current=True
        )
    )
    s.flush()
    assert service.detail(a.window_id)["latest_review"]["stale"]
    with pytest.raises(WindowError):
        service.review(a.window_id, **(payload | {"request_id": "r2"}))


def test_expired_attempt_cannot_write_after_same_worker_reclaims(context):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    assert service.claim("same")["attempt_count"] == 1
    a.heartbeat_at = datetime.now(UTC) - timedelta(minutes=10)
    s.flush()
    assert service.claim("same")["attempt_count"] == 2
    with pytest.raises(WindowError, match="claim_lost"):
        service.heartbeat(a.id, "same", 1)
    assert service.heartbeat(a.id, "same", 2)["status"] == "running"


def test_result_rejects_forged_signature(context):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("one")
    r = make_result(service, a)
    r["input_sha256"] = "a" * 64
    write_result(service, a, r)
    with pytest.raises(WindowError, match="signature"):
        service.result(a.id, "one", 1)


def make_result(service, a):
    import hashlib

    from das_sentinel.whole_window import compute_input_sha256

    folder = service.artifact_root / a.id / "attempt-1"
    folder.mkdir(parents=True, exist_ok=True)
    r = plan() | {
        "schema_version": 1,
        "algorithm_version": "whole-window-v1",
        "sample_rate_hz": 100,
        "shape": [2, 6000],
        "unit": "strain",
        "grid_sha256": "0" * 64,
        "profile": {},
        "recognition_range_m": {"start": None, "stop": None},
        "valid_channel_mask": [True, True],
        "recognition_channel_mask": [True, True],
        "read_start_utc": "2023-09-26T23:59:19.990000+00:00",
        "read_end_utc": "2023-09-27T00:01:40.010000+00:00",
        "conditioning_start_utc": "2023-09-26T23:59:30+00:00",
        "conditioning_end_utc": "2023-09-27T00:01:30+00:00",
        "read_sources": [],
        "features": {},
        "candidate": True,
        "score": 1,
        "evidence_count": 0,
    }
    from das_sentinel.config import ProcessingProfile as ProfileConfig

    source = r["sources"][0]
    r["read_sources"] = [
        {
            k: source[k]
            for k in (
                "path",
                "fingerprint",
                "size_bytes",
                "mtime_ns",
                "sample_rate_hz",
                "unit",
            )
        }
        | {
            "sample_start": 1999,
            "sample_stop": 16001,
            "channel_start": 0,
            "channel_stop": 2,
            "source_start_utc": source["start_utc"],
            "source_stop_utc": source["end_utc"],
            "read_start_utc": r["read_start_utc"],
            "read_stop_utc": r["read_end_utc"],
            "source_verified_unchanged": True,
            "boundary_utc": source["start_utc"],
            "previous_source_end_utc": None,
            "previous_seam_seconds": None,
        }
    ]
    r["profile"] = ProfileConfig().model_dump(mode="json")
    r.pop("coverage")
    r["input_sha256"] = compute_input_sha256(r)
    import numpy as np

    distance = np.array([0.0, 1.0])
    r["grid_sha256"] = hashlib.sha256(distance.astype("<f8").tobytes()).hexdigest()
    r["input_sha256"] = compute_input_sha256(r)
    (folder / "metadata.json").write_text(json.dumps(r))
    (folder / "evidence.json").write_text("[]")
    np.savez_compressed(
        folder / "waveform.npz",
        waveform=np.tile(np.sin(np.arange(6000) / 100), (2, 1)),
        distance_m=distance,
        relative_seconds=np.arange(6000) / 100,
        valid_channel_mask=np.ones(2, dtype=bool),
        recognition_channel_mask=np.ones(2, dtype=bool),
        region_id=np.zeros(2, dtype=int),
    )
    for name in ["waterfall.png", "waveform.png"]:
        (folder / name).write_bytes(b"test")
    r["artifacts"] = [
        {
            "kind": kind,
            "path": name,
            "media_type": (
                "application/octet-stream"
                if name.endswith(".npz")
                else "image/png"
                if name.endswith(".png")
                else "application/json"
            ),
            "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
        }
        for kind, name in [
            ("waveform", "waveform.npz"),
            ("metadata", "metadata.json"),
            ("evidence", "evidence.json"),
            ("waterfall", "waterfall.png"),
            ("waveform_plot", "waveform.png"),
        ]
    ]
    write_result(service, a, r)
    return r


def write_result(service, a, r):
    (service.artifact_root / a.id / "attempt-1" / "result.json").write_text(
        json.dumps(r)
    )


def test_window_auth_csrf_worker_token_and_attempt(event_api):
    client, _settings, sessions, _csrf, ids = event_api
    with sessions.begin() as s:
        a = enqueue_windows(s, ids["site_a"], ids["profile_a"], [plan()])[0]
        aid = a.id
        wid = a.window_id
    assert client.post("/api/v1/window-jobs/" + aid + "/cancel").status_code == 403
    assert (
        client.post(
            "/api/v1/internal/window-jobs/claim", json={"worker_id": "w"}
        ).status_code
        == 401
    )
    claim = client.post(
        "/api/v1/internal/window-jobs/claim",
        json={"worker_id": "w"},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert claim.status_code == 200 and claim.json()["attempt_count"] == 1
    assert (
        client.post(
            "/api/v1/internal/window-jobs/" + aid + "/heartbeat",
            json={"worker_id": "x", "attempt_count": 1},
            headers={"X-Worker-Token": WORKER_TOKEN},
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/api/v1/internal/window-jobs/" + aid + "/heartbeat",
            json={"worker_id": "w"},
            headers={"X-Worker-Token": WORKER_TOKEN},
        ).status_code
        == 422
    )
    client.cookies.clear()
    for url in [
        "/api/v1/event-windows",
        "/api/v1/event-windows/" + wid,
        "/api/v1/event-windows/" + wid + "/artifacts/waterfall",
        "/api/v1/window-jobs",
    ]:
        assert client.get(url).status_code == 401


def test_priority_and_sql_pagination(context):
    s, site, p, service = context
    first = enqueue_windows(s, site.id, p.id, [plan()], priority=2)[0]
    later = plan()
    later.update(
        start_utc="2023-09-27T00:01:00+00:00", end_utc="2023-09-27T00:02:00+00:00"
    )
    second = enqueue_windows(
        s,
        site.id,
        p.id,
        [later],
        priority=10,
        catalog_matches=[{"catalog_id": "catalog"}],
    )[0]
    page = service.list(window_seconds=60, review_status="all", limit=1)
    assert page["total"] == 2 and page["items"][0]["analysis_id"] == second.id
    assert (
        service.list(window_seconds=60, review_status="all", limit=1, offset=1)["items"][0]["analysis_id"]
        == first.id
    )
    assert service.claim("w")["id"] == second.id
    assert service.list(window_seconds=60, recommended=True)["total"] == 1


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("hash", "artifact_hash"),
        ("path", "invalid_manifest"),
        ("nan", "invalid_json"),
        ("source", "identity"),
        ("window", "identity"),
        ("profile", "profile"),
        ("evidence", "evidence_count"),
    ],
)
def test_invalid_results(context, mutation, message):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("one")
    r = make_result(service, a)
    if mutation == "hash":
        r["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "path":
        r["artifacts"][0]["path"] = "../waveform.npz"
    elif mutation == "nan":
        r["score"] = float("nan")
    elif mutation == "source":
        r["sources"] = [{"path": "forged"}]
    elif mutation == "window":
        r["start_utc"] = "2023-09-27T01:00:00+00:00"
    elif mutation == "profile":
        from das_sentinel.whole_window import compute_input_sha256

        r["profile"]["scan_distance_start_m"] = 99
        r["input_sha256"] = compute_input_sha256(r)
    elif mutation == "evidence":
        r["evidence_count"] = 100001
    write_result(service, a, r)
    with pytest.raises(WindowError, match=message):
        service.result(a.id, "one", 1)


def test_success_and_version_bound_evidence(context):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("one")
    make_result(service, a)
    assert service.result(a.id, "one", 1)["status"] == "succeeded"
    assert service.evidence(a.window_id, analysis_id=a.id) == {
        "items": [],
        "total": 0,
        "limit": 50,
        "offset": 0,
    }
    assert "analysis_id=" + a.id in service.detail(a.window_id)["artifacts"][0]["url"]
    with pytest.raises(WindowError):
        service.evidence(a.window_id, limit=101)


def test_analysis_plan_immutable(context):
    from sqlalchemy.exc import IntegrityError

    s, site, p, _service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    with pytest.raises(IntegrityError), s.begin_nested():
        a.plan_json = "{}"
        s.flush()


def test_result_rejects_invalid_waveform_array(context):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("one")
    r = make_result(service, a)
    folder = service.artifact_root / a.id / "attempt-1"
    import hashlib

    (folder / "waveform.npz").write_bytes(b"not an array")
    r["artifacts"][0]["sha256"] = hashlib.sha256(b"not an array").hexdigest()
    write_result(service, a, r)
    with pytest.raises(WindowError, match="array"):
        service.result(a.id, "one", 1)


def test_manifest_media_type_and_numeric_overflow(context):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("one")
    r = make_result(service, a)
    r["artifacts"][0]["media_type"] = "text/html"
    write_result(service, a, r)
    with pytest.raises(WindowError, match="manifest"):
        service.result(a.id, "one", 1)
    r = make_result(service, a)
    raw = json.dumps(r).replace('"score": 1', '"score": 1e999')
    (service.artifact_root / a.id / "attempt-1" / "result.json").write_text(raw)
    with pytest.raises(WindowError, match="invalid_json"):
        service.result(a.id, "one", 1)


def test_concurrent_claim_single_owner(context):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    aid = a.id
    s.commit()
    sessions = session_factory(s.get_bind())
    ready = Barrier(2)

    def claim(worker):
        with sessions.begin() as separate:
            ready.wait()
            return WindowService(separate, service.artifact_root).claim(worker)

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(claim, ["one", "two"]))
    assert sum(v is not None for v in values) == 1
    assert next(v for v in values if v is not None)["id"] == aid


def test_legacy_clues_never_become_window_reviews(event_api):
    from das_sentinel.persistence.models import Review
    from das_sentinel.persistence.window_models import WindowReview

    client, _settings, sessions, _csrf, ids = event_api
    with sessions.begin() as s:
        old = [(r.id, r.label, r.note) for r in s.scalars(select(Review)).all()]
        a = enqueue_windows(
            s,
            ids["site_a"],
            ids["profile_a"],
            [plan()],
            legacy_event_ids=[ids["event_2"]],
        )[0]
        wid = a.window_id
    response = client.get("/api/v1/event-windows/" + wid).json()
    assert response["latest_review"] is None and response["legacy_events"][0]["label"]
    with sessions() as s:
        assert [(r.id, r.label, r.note) for r in s.scalars(select(Review)).all()] == old
        assert s.scalars(select(WindowReview)).all() == []


def test_result_rejects_claim_reuse_and_cross_analysis(context):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("one")
    make_result(service, a)
    other = plan()
    other["algorithm_version"] = "other"
    b = enqueue_windows(s, site.id, p.id, [other])[0]
    service.claim("two")
    with pytest.raises(WindowError, match="claim_lost"):
        service.result(a.id, "two", 1)
    with pytest.raises(WindowError, match="claim_lost"):
        service.fail(b.id, "one", "bad", "", 1)


def test_source_changes_make_saved_review_stale(context, tmp_path):
    from das_sentinel.persistence.models import Admin
    from das_sentinel.services.discovery import fingerprint

    s, site, p, service = context
    source = tmp_path / "source.h5"
    source.write_bytes(b"original")
    planned = plan()
    planned["sources"] = [
        {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "fingerprint": fingerprint(source),
        }
    ]
    a = enqueue_windows(s, site.id, p.id, [planned])[0]
    a.status = "succeeded"
    a.input_sha256 = "a" * 64
    a.result_json = "{}"
    s.add(Admin(id=1, password_hash="test"))
    s.flush()
    args = {
        "analysis_id": a.id,
        "input_sha256": a.input_sha256,
        "labels": {"earthquake": "unknown", "blast": "unknown", "traffic": "unknown"},
        "note": "",
        "request_id": "source-review",
        "admin_id": 1,
    }
    assert not service.review(a.window_id, **args)["background_only"]
    source.write_bytes(b"changed file")
    assert service.detail(a.window_id)["latest_review"]["stale"]
    with pytest.raises(WindowError, match="stale"):
        service.review(a.window_id, **(args | {"request_id": "new"}))


def test_claim_only_latest_analysis_and_identifiable_jobs(context):
    s, site, p, service = context
    enqueue_windows(s, site.id, p.id, [plan()])[0]
    changed = plan()
    changed["algorithm_version"] = "new"
    latest = enqueue_windows(s, site.id, p.id, [changed])[0]
    claim = service.claim("w")
    assert claim["id"] == latest.id
    assert service.claim("second") is None
    job = service.jobs(window_seconds=60)["items"][0]
    assert job["site_id"] == site.id and job["site_timezone"] == "Asia/Shanghai"
    assert job["profile_id"] == p.id and job["start_utc"].minute == 0


def test_migration_keeps_legacy_review_rows(tmp_path):
    from pathlib import Path

    from alembic import command
    from alembic.config import Config
    from das_sentinel.persistence import db
    from das_sentinel.persistence.models import Review
    from das_sentinel.persistence.repositories import Repositories

    from tests.test_persistence import _seed_reviewed_event

    url = f"sqlite+pysqlite:///{tmp_path / 'legacy.sqlite3'}"
    config = Config()
    config.set_main_option(
        "script_location", str(Path(db.__file__).parent / "migrations")
    )
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "0002_job_model_and_review_request")
    engine = create_local_engine(url)
    sessions = session_factory(engine)
    with sessions.begin() as s:
        _, _, review = _seed_reviewed_event(Repositories(s))
        before = (review.id, review.event_id, review.label, review.note)
    command.upgrade(config, "head")
    with sessions() as s:
        review = s.get(Review, before[0])
        assert (review.id, review.event_id, review.label, review.note) == before
        assert s.scalars(select(EventWindow)).all() == []
    engine.dispose()


def test_bounded_evidence_result(context, monkeypatch):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("w")
    make_result(service, a)
    monkeypatch.setattr("das_sentinel.services.event_windows.MAX_EVIDENCE", -1)
    with pytest.raises(WindowError, match="evidence_count"):
        service.result(a.id, "w", 1)
    monkeypatch.setattr("das_sentinel.services.event_windows.MAX_JSON_BYTES", 1)
    with pytest.raises(WindowError, match="json_too_large"):
        service.result(a.id, "w", 1)


def test_list_orders_priority_then_score_then_time(context):
    s, site, p, service = context
    first = enqueue_windows(s, site.id, p.id, [plan()])[0]
    later = plan()
    later.update(
        start_utc="2023-09-27T00:01:00+00:00", end_utc="2023-09-27T00:02:00+00:00"
    )
    second = enqueue_windows(s, site.id, p.id, [later])[0]
    first.result_json = '{"score":0.2}'
    second.result_json = '{"score":0.9}'
    s.flush()
    assert [v["analysis_id"] for v in service.list(window_seconds=60, review_status="all")["items"]] == [
        second.id,
        first.id,
    ]
    s.get(EventWindow, first.window_id).priority = 10
    s.flush()
    assert service.list(window_seconds=60, review_status="all")["items"][0]["analysis_id"] == first.id


@pytest.mark.parametrize(
    "mutation",
    ["empty", "samples", "channel", "read_time", "conditioning", "fingerprint"],
)
def test_provenance_cannot_be_forged_with_recomputed_signature(context, mutation):
    import hashlib

    from das_sentinel.whole_window import compute_input_sha256

    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("w")
    r = make_result(service, a)
    if mutation == "empty":
        r["read_sources"] = []
    elif mutation == "samples":
        r["read_sources"][0]["sample_start"] += 1
    elif mutation == "channel":
        r["read_sources"][0]["channel_start"] = 1
    elif mutation == "read_time":
        r["read_start_utc"] = r["conditioning_start_utc"]
    elif mutation == "conditioning":
        r["conditioning_start_utc"] = r["start_utc"]
    elif mutation == "fingerprint":
        r["read_sources"][0]["fingerprint"] = "0" * 64
    r["input_sha256"] = compute_input_sha256(r)
    folder = service.artifact_root / a.id / "attempt-1"
    (folder / "metadata.json").write_text(
        json.dumps({k: v for k, v in r.items() if k != "artifacts"})
    )
    r["artifacts"][1]["sha256"] = hashlib.sha256(
        (folder / "metadata.json").read_bytes()
    ).hexdigest()
    write_result(service, a, r)
    with pytest.raises(WindowError, match="provenance"):
        service.result(a.id, "w", 1)


@pytest.mark.parametrize(
    "profile_payload",
    [{}, {"scan_distance_start_m": 299, "scan_distance_stop_m": 499}],
)
def test_real_cross_file_core_result_native_provenance(context, tmp_path, profile_payload):
    from das_sentinel.config import ProcessingProfile as ProfileConfig
    from das_sentinel.whole_window import analyze_window, inspect_sources, plan_windows

    from tests.test_whole_window import BASE, write

    s, site, p, service = context
    if profile_payload:
        p.is_current = False
        s.flush()
        p = ProcessingProfile(
            site_id=site.id,
            version=2,
            payload_json=json.dumps(profile_payload),
            is_current=True,
        )
        s.add(p)
        s.flush()
    paths = [
        write(tmp_path / "before.h5", -60, 120),
        write(tmp_path / "after.h5", 60, 120),
    ]
    selected = next(
        v
        for v in plan_windows(inspect_sources(paths))
        if v["start_utc"] == BASE.isoformat()
    )
    a = enqueue_windows(s, site.id, p.id, [selected])[0]
    service.claim("worker")
    result = analyze_window(
        paths,
        BASE,
        ProfileConfig.model_validate(profile_payload),
        service.artifact_root / a.id / "attempt-1",
    )
    assert len(result["read_sources"]) == 2
    if profile_payload:
        assert result["recognition_channel_mask"] == [
            False, False, False, True, True, False, False, False
        ]
    assert service.result(a.id, "worker", 1)["status"] == "succeeded"


@pytest.mark.parametrize("mutation", ["recognition", "range", "region", "valid"])
def test_result_rejects_forged_derived_channel_contract(context, mutation):
    import hashlib

    import numpy as np
    from das_sentinel.whole_window import compute_input_sha256

    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("w")
    r = make_result(service, a)
    folder = service.artifact_root / a.id / "attempt-1"
    with np.load(folder / "waveform.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    if mutation == "recognition":
        arrays["recognition_channel_mask"][:] = False
        r["recognition_channel_mask"] = [False, False]
    elif mutation == "range":
        r["recognition_range_m"] = {"start": 999, "stop": 1000}
    elif mutation == "region":
        arrays["region_id"][:] = 99
    else:
        arrays["valid_channel_mask"][:] = False
        arrays["recognition_channel_mask"][:] = False
        r["valid_channel_mask"] = [False, False]
        r["recognition_channel_mask"] = [False, False]
    np.savez_compressed(folder / "waveform.npz", **arrays)
    r["input_sha256"] = compute_input_sha256(r)
    (folder / "metadata.json").write_text(
        json.dumps({k: v for k, v in r.items() if k != "artifacts"})
    )
    for artifact in r["artifacts"]:
        artifact["sha256"] = hashlib.sha256(
            (folder / artifact["path"]).read_bytes()
        ).hexdigest()
    write_result(service, a, r)
    with pytest.raises(WindowError, match="invalid_array"):
        service.result(a.id, "w", 1)
    assert a.status == "running"


def test_retry_rejects_superseded_failed_analysis(context):
    s, site, p, service = context
    a = enqueue_windows(s, site.id, p.id, [plan()])[0]
    service.claim("w")
    service.fail(a.id, "w", "error", "", 1)
    b = enqueue_windows(s, site.id, p.id, [plan() | {"algorithm_version": "new"}])[0]
    service.cancel(b.id)
    with pytest.raises(WindowError, match="retry_requires_latest_analysis"):
        service.retry(a.id)
    assert a.status == "failed"
    assert b.status == "cancelled"
    assert service.claim("w") is None


def test_review_api_accepts_blank_note_and_preserves_unknown(event_api):
    client, _settings, sessions, csrf, ids = event_api
    with sessions.begin() as s:
        a = enqueue_windows(s, ids["site_a"], ids["profile_a"], [plan()])[0]
        a.status = "succeeded"
        a.input_sha256 = "a" * 64
        a.result_json = "{}"
        wid, aid = a.window_id, a.id
    body = {
        "analysis_id": aid,
        "input_sha256": "a" * 64,
        "labels": {"earthquake": "present", "blast": "unknown", "traffic": "present"},
        "note": "",
        "request_id": "blank-note",
    }
    body["labels"] = {"earthquake": "unknown"}
    response = client.post(
        f"/api/v1/event-windows/{wid}/reviews",
        json=body,
        headers={"X-CSRF-Token": csrf},
    )
    assert response.status_code == 200, response.text
    assert response.json()["note"] == ""
    assert response.json()["labels"] == body["labels"]
    assert not response.json()["background_only"]
    repeated = client.post(
        f"/api/v1/event-windows/{wid}/reviews",
        json=body,
        headers={"X-CSRF-Token": csrf},
    )
    assert repeated.status_code == 200
    assert repeated.json()["id"] == response.json()["id"]


@pytest.mark.parametrize("endpoint", ["event-windows", "window-jobs"])
def test_duration_query_routes_half_minutes_and_preserves_history(event_api, endpoint):
    client, _settings, sessions, _csrf, ids = event_api
    from datetime import datetime, timedelta

    original = plan()
    short = original | {"end_utc": (datetime.fromisoformat(original["start_utc"]) + timedelta(seconds=30)).isoformat()}
    with sessions.begin() as session:
        old = enqueue_windows(session, ids["site_a"], ids["profile_a"], [original])[0]
        new = enqueue_windows(session, ids["site_a"], ids["profile_a"], [short])[0]
        expected = {30: new.window_id, 60: old.window_id}
    for seconds, wid in expected.items():
        response = client.get(f"/api/v1/{endpoint}?window_seconds={seconds}")
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 1
        item = response.json()["items"][0]
        assert item["window_id" if endpoint == "window-jobs" else "id"] == wid
        assert item["window_seconds"] == seconds
    assert client.get(f"/api/v1/{endpoint}").json()["items"][0]["window_seconds"] == 30
    assert client.get(f"/api/v1/{endpoint}?window_seconds=45").status_code == 422
