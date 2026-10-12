from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import Event, Job, ProcessingRun, Review
from fastapi.testclient import TestClient
from sqlalchemy import event as sqlalchemy_event

BASE = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.fixture
def review_env(tmp_path: Path):
    raw = tmp_path / "raw"
    app_data = tmp_path / "app-data"
    raw.mkdir()
    app_data.mkdir()
    settings = Settings(
        data_root=raw,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}",
        api_url="http://testserver",
        session_secret="s" * 32,
        worker_token="w" * 40,
        secure_cookies=False,
    )
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    with sessions.begin() as session:
        repos = Repositories(session)
        repos.admins.add(password_hash=PasswordHasher().hash("correct horse"))
        ids = {}
        for name in ("a", "b"):
            site = repos.sites.add(
                name=name, data_path=str(raw), timezone="UTC", cable_type="urban"
            )
            ids[f"site_{name}"] = site.id
            for version in (1, 2):
                profile = repos.profiles.add(
                    site_id=site.id,
                    version=version,
                    payload=ProcessingProfile(device="cpu").model_dump(mode="json"),
                    is_current=version == 2,
                )
                for directory in ("one", "two"):
                    source = repos.source_files.add(
                        site_id=site.id,
                        path=f"/data/{directory}/{name}-{version}.h5",
                        fingerprint=f"{len(ids):064x}",
                        size_bytes=1,
                        mtime_ns=1,
                    )
                    for pipeline in ("v1", "v2"):
                        run = repos.runs.add(
                            source_file_id=source.id,
                            processing_profile_id=profile.id,
                            pipeline_version=pipeline,
                            requested_backend="cpu",
                            actual_backend="cpu",
                            status="succeeded",
                        )
                        ids[f"{name}_{version}_{directory}_{pipeline}"] = run.id
    yield settings, sessions, ids
    engine.dispose()


def _event(
    session,
    ids,
    uid,
    *,
    seconds=0,
    run="a_1_one_v1",
    value=1.0,
    channels=(0, 10),
    features=None,
    label=None,
    machine=None,
):
    repos = Repositories(session)
    row = repos.events.add(
        run_id=ids[run],
        event_uid=uid,
        start_sample=0,
        end_sample=10,
        channel_start=channels[0],
        channel_stop=channels[1],
        occurred_at=BASE + timedelta(seconds=seconds),
        features=features
        if features is not None
        else {
            "duration_s": 1.0,
            "rms": value,
            "dominant_frequency_hz": value,
            "coherence": value,
            "spectral_entropy": value,
        },
    )
    if label:
        repos.reviews.add(row.id, label=label, admin_id=1)
    if machine:
        repos.events.add_classification(
            row.id, label=machine, decision_score=0.99, method="rules"
        )
    return row.id


def _batch(env, *, offset=0, limit=50, site="a"):
    # The missing service is asserted as a missing behavior before implementation.
    from das_sentinel.services import review_batches

    settings, sessions, ids = env
    with sessions() as session:
        return review_batches.get_review_batch(
            session,
            settings.app_data_dir,
            site_id=ids[f"site_{site}"],
            offset=offset,
            limit=limit,
        )


def _members(batch):
    return [
        member["event_uid"] for group in batch["groups"] for member in group["members"]
    ]


def test_groups_nearby_fragments_without_crossing_time_space_source_or_versions(
    review_env,
):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        _event(session, ids, "anchor")
        _event(session, ids, "near", seconds=90)
        _event(session, ids, "no-chain", seconds=180)
        _event(session, ids, "other-day", seconds=86400)
        _event(session, ids, "other-space", channels=(10, 20))
        _event(session, ids, "other-source", run="a_1_two_v1")
        _event(session, ids, "other-profile", run="a_2_one_v1")
        _event(session, ids, "other-pipeline", run="a_1_one_v2")
        _event(session, ids, "other-site", run="b_1_one_v1")
        _event(session, ids, "different-signal", value=1000)
    batch = _batch(review_env)
    grouped = {
        frozenset(member["event_uid"] for member in group["members"])
        for group in batch["groups"]
    }
    assert frozenset({"anchor", "near"}) in grouped
    assert batch["total_unreviewed"] == 9
    assert batch["total_groups"] == 8
    assert "other-site" not in _members(batch)
    assert all(group["representative"] in group["members"] for group in batch["groups"])
    assert all(group["catalog_window"] is None for group in batch["groups"])
    assert all(
        group["representatives"] == [group["representative"]]
        for group in batch["groups"]
    )


def test_latest_human_correction_and_compatible_references_control_priority(review_env):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        corrected = _event(
            session, ids, "corrected", seconds=-1000, value=1, label="earthquake"
        )
        # Equal timestamps must use review ID as the tie breaker.
        original = session.query(Review).filter_by(event_id=corrected).one()
        session.add(
            Review(event_id=corrected, label="noise", created_at=original.created_at)
        )
        eq = _event(
            session, ids, "confirmed-eq", seconds=-500, value=10, label="earthquake"
        )
        _event(
            session,
            ids,
            "incompatible-eq",
            run="a_2_one_v1",
            label="earthquake",
            value=1,
        )
        _event(
            session, ids, "foreign-eq", run="b_1_one_v1", label="earthquake", value=1
        )
        _event(session, ids, "eq-like", value=10, machine="noise")
        _event(session, ids, "noise-like", seconds=500, value=1, machine="earthquake")
        _event(session, ids, "no-reference", seconds=600, run="a_1_one_v2", value=10)
    batch = _batch(review_env)
    by_name = {group["representative"]["event_uid"]: group for group in batch["groups"]}
    assert set(by_name) == {"eq-like", "noise-like", "no-reference"}
    assert batch["reference_counts"] == {
        "earthquake": 2,
        "blast": 0,
        "traffic": 0,
        "noise": 1,
    }
    assert (
        by_name["eq-like"]["priority_score"] > by_name["noise-like"]["priority_score"]
    )
    assert by_name["eq-like"]["selection_kind"] == "earthquake"
    assert by_name["eq-like"]["similar_earthquake_event_id"] == eq
    assert by_name["no-reference"]["similar_earthquake_event_id"] is None
    assert by_name["eq-like"]["representative"]["machine_label"] == "noise"


def test_complete_deterministic_pagination_keeps_every_unreviewed_event_once(
    review_env,
):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        for index in range(87):
            _event(
                session,
                ids,
                f"event-{index}",
                seconds=index * 300,
                value=1 + index % 7,
                machine="noise",
            )
        _event(session, ids, "reviewed", label="noise")
    whole = _batch(review_env, limit=50)
    pages = [
        _batch(review_env, offset=offset, limit=20) for offset in (0, 20, 40, 60, 80)
    ]
    names = [name for page in pages for name in _members(page)]
    assert len(names) == len(set(names)) == 87
    assert set(names) == {f"event-{index}" for index in range(87)}
    assert names[:50] == _members(whole)
    assert _batch(review_env, offset=20, limit=20) == pages[1]
    assert sum(group["selection_kind"] == "audit" for group in pages[0]["groups"]) == 4
    assert _batch(review_env, offset=100)["groups"] == []


def test_group_size_is_bounded_at_twenty_without_losing_fragments(review_env):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        for index in range(43):
            _event(session, ids, f"fragment-{index}", seconds=index)
    batch = _batch(review_env)
    assert sorted(len(group["members"]) for group in batch["groups"]) == [3, 20, 20]
    assert len(set(_members(batch))) == 43


def test_bad_features_remain_visible_and_are_not_treated_as_zero_similarity(review_env):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        _event(session, ids, "good", label="earthquake")
        for index, features in enumerate(
            (
                {},
                {"rms": True, "duration_s": "1"},
                {"rms": float("nan"), "duration_s": float("inf")},
                {"time_of_day_hours": 1, "machine_label": 1, "source_path": 1},
            )
        ):
            bad = _event(session, ids, f"bad-{index}", features={})
            session.get(Event, bad).features_json = json.dumps(features)
        broken = _event(session, ids, "broken-json")
        session.get(Event, broken).features_json = "{"
    batch = _batch(review_env)
    assert len(batch["groups"]) == 5
    assert set(_members(batch)) == {"bad-0", "bad-1", "bad-2", "bad-3", "broken-json"}
    assert all(
        group["similar_earthquake_event_id"] is None for group in batch["groups"]
    )
    assert any("特征" in warning for warning in batch["warnings"])
    json.dumps(batch, allow_nan=False)


def test_catalog_is_optional_and_matches_only_aware_utc_intersections_for_this_site(
    review_env,
):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        _event(session, ids, "catalog-hit", seconds=10)
        _event(session, ids, "no-hit", seconds=1000)
    without = _batch(review_env)
    assert without["catalog_window_count"] == 0
    windows = [
        {
            "site_id": ids["site_a"],
            "catalog_id": "cat",
            "phase": "P",
            "start_utc": "2026-08-01T08:00:05+08:00",
            "end_utc": "2026-08-01T08:00:20+08:00",
            "source_url": "https://earthquake.usgs.gov/earthquakes/eventpage/cat",
        },
        {
            "site_id": ids["site_b"],
            "catalog_id": "foreign",
            "phase": "S",
            "start_utc": "2026-08-01T00:00:00Z",
            "end_utc": "2026-08-02T00:00:00Z",
            "source_url": "https://example.org",
        },
        {
            "site_id": ids["site_a"],
            "catalog_id": "bad",
            "phase": "P",
            "start_utc": "2026-08-01T00:00:00",
            "end_utc": "2026-08-02T00:00:00",
            "source_url": "https://example.org",
        },
    ]
    (settings.app_data_dir / "review-catalog-windows.json").write_text(
        json.dumps({"schema_version": 1, "windows": windows}), encoding="utf-8"
    )
    batch = _batch(review_env)
    assert batch["catalog_window_count"] == 1
    by_name = {group["representative"]["event_uid"]: group for group in batch["groups"]}
    assert by_name["catalog-hit"]["selection_kind"] == "earthquake"
    assert (
        by_name["catalog-hit"]["priority_score"] > by_name["no-hit"]["priority_score"]
    )
    assert any(
        "cat" in reason and "目录" in reason
        for reason in by_name["catalog-hit"]["reasons"]
    )
    assert all(
        "foreign" not in reason
        for group in batch["groups"]
        for reason in group["reasons"]
    )
    assert by_name["catalog-hit"]["similar_earthquake_event_id"] is None


def test_missing_site_and_empty_site(review_env):
    settings, sessions, _ = review_env
    assert _batch(review_env)["total_unreviewed"] == 0
    from das_sentinel.services.review_batches import get_review_batch

    with sessions() as session, pytest.raises(LookupError):
        get_review_batch(session, settings.app_data_dir, site_id="missing")


def test_api_auth_validation_and_read_only_response(review_env, monkeypatch):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        _event(session, ids, "visible")
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    app = create_app(settings)
    with TestClient(app) as client:
        assert (
            client.get(
                "/api/v1/review-batches", params={"site_id": ids["site_a"]}
            ).status_code
            == 401
        )
        client.post("/api/v1/session", json={"password": "correct horse"})
        mutations = []

        def detect_write(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
                mutations.append(statement)

        sqlalchemy_event.listen(app.state.engine, "before_cursor_execute", detect_write)
        try:
            response = client.get(
                "/api/v1/review-batches", params={"site_id": ids["site_a"]}
            )
            assert response.status_code == 200, response.text
            assert _members(response.json()) == ["visible"]
            group = response.json()["groups"][0]
            assert group["catalog_window"] is None
            assert group["representatives"] == [group["representative"]]
            assert response.json()["limit"] == 20
            assert (
                client.get(
                    "/api/v1/review-batches",
                    params={"site_id": ids["site_a"], "limit": 51},
                ).status_code
                == 422
            )
            assert (
                client.get(
                    "/api/v1/review-batches",
                    params={"site_id": ids["site_a"], "offset": -1},
                ).status_code
                == 422
            )
            assert (
                client.get(
                    "/api/v1/review-batches", params={"site_id": "missing"}
                ).status_code
                == 404
            )
            assert mutations == []
        finally:
            sqlalchemy_event.remove(
                app.state.engine, "before_cursor_execute", detect_write
            )


def test_each_full_block_reserves_ten_earthquake_six_diverse_and_four_audits(
    review_env,
):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        _event(session, ids, "reference", seconds=-1000, value=10, label="earthquake")
        _event(session, ids, "noise-reference", seconds=-500, value=1, label="noise")
        for index in range(65):
            _event(
                session,
                ids,
                f"candidate-{index}",
                seconds=index * 300,
                value=10 if index < 35 else 1,
            )
    batch = _batch(review_env, limit=40)
    for start in (0, 20):
        kinds = [
            group["selection_kind"] for group in batch["groups"][start : start + 20]
        ]
        assert kinds.count("earthquake") == 10
        assert kinds.count("diverse") == 6
        assert kinds.count("audit") == 4


def test_time_context_and_source_name_do_not_become_earthquake_evidence(review_env):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        _event(session, ids, "eq", label="earthquake", value=10, seconds=-1000)
        first = _event(session, ids, "candidate", value=10)
        _event(session, ids, "control", value=1, seconds=500)
    before = _batch(review_env)
    with sessions.begin() as session:
        row = session.get(Event, first)
        row.features_json = json.dumps(
            {
                **json.loads(row.features_json),
                "time_of_day_hours": 999999,
                "source_path": "earthquake/confirmed-eq",
                "machine_label": "earthquake",
                "site_id": "unrelated",
            }
        )
    assert _batch(review_env) == before


def test_saved_review_is_excluded_from_next_request(review_env):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        first = _event(session, ids, "first")
        _event(session, ids, "second", seconds=10)
    assert set(_members(_batch(review_env))) == {"first", "second"}
    with sessions.begin() as session:
        Repositories(session).reviews.add(first, label="noise", admin_id=1)
    after = _batch(review_env)
    assert _members(after) == ["second"]
    assert after["total_unreviewed"] == 1


def test_group_cannot_join_fragments_with_no_common_spatial_overlap(review_env):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        _event(session, ids, "wide", channels=(0, 20))
        _event(session, ids, "left", seconds=10, channels=(0, 9))
        _event(session, ids, "right", seconds=20, channels=(11, 20))
    batch = _batch(review_env)
    assert batch["total_groups"] == 2
    assert sorted(len(group["members"]) for group in batch["groups"]) == [1, 2]


def test_group_cannot_absorb_long_anchor_past_the_120_second_span(review_env):
    _, sessions, ids = review_env
    with sessions.begin() as session:
        _event(
            session,
            ids,
            "long-anchor",
            features={"duration_s": 180, "rms": 1, "coherence": 1},
        )
        _event(session, ids, "short", seconds=10, features={"rms": 1, "coherence": 1})
    batch = _batch(review_env)
    assert batch["total_groups"] == 2


def test_service_imports_in_a_fresh_process_without_initializing_api():
    source_root = Path(__file__).resolve().parents[1] / "backend" / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from das_sentinel.services.review_batches import get_review_batch; "
                "print(callable(get_review_batch), 'das_sentinel.api' in sys.modules)"
            ),
            str(source_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True False"


def _job_for_run(session, run_id):
    run = session.get(ProcessingRun, run_id)
    job = Job(
        source_file_id=run.source_file_id,
        processing_profile_id=run.processing_profile_id,
        pipeline_version=run.pipeline_version,
        status="succeeded",
    )
    session.add(job)
    session.flush()
    run.job_id = job.id
    return job.id


def _preferred_window(settings, site_id, job_id, **overrides):
    window = {
        "site_id": site_id,
        "catalog_id": "new-catalog",
        "phase": "P",
        "start_utc": BASE.isoformat(),
        "end_utc": (BASE + timedelta(days=1)).isoformat(),
        "source_url": "https://earthquake.usgs.gov/earthquakes/eventpage/new-catalog",
        "preferred_job_ids": [job_id],
        **overrides,
    }
    (settings.app_data_dir / "review-catalog-windows.json").write_text(
        json.dumps({"schema_version": 1, "windows": [window]}), encoding="utf-8"
    )


def test_preferred_catalog_jobs_precede_old_groups_across_pages(review_env):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        job_id = _job_for_run(session, ids["a_1_one_v2"])
        _event(session, ids, "reference", label="earthquake", value=10, seconds=-500)
        for index in range(30):
            _event(session, ids, f"old-{index}", seconds=index * 300, value=10)
        for index in range(24):
            _event(
                session, ids, f"new-{index}", run="a_1_one_v2",
                seconds=10000 + index * 300, value=1, machine="noise",
            )
    _write_windows(
        settings,
        [
            _window(
                ids["site_a"], catalog_id=f"new-{index}",
                start=10000 + index * 300, end=10010 + index * 300,
                preferred_job_ids=[job_id],
            )
            for index in range(24)
        ],
    )
    pages = [_batch(review_env, offset=offset, limit=20) for offset in (0, 20, 40)]
    groups = [group for page in pages for group in page["groups"]]
    names = [group["representative"]["event_uid"] for group in groups]
    assert names[:24] == [f"new-{index}" for index in range(24)]
    assert len(names) == len(set(names)) == 54
    assert all(group["selection_kind"] == "earthquake" for group in groups[:24])
    assert all(
        any("本批目录优先" in reason for reason in group["reasons"])
        for group in groups[:24]
    )
    assert all(
        not any("本批目录优先" in reason for reason in group["reasons"])
        for group in groups[24:]
    )
    assert min(g["priority_score"] for g in groups[:24]) > max(
        g["priority_score"] for g in groups[24:]
    )
    assert _batch(review_env, offset=20, limit=20) == pages[1]


def test_preferred_catalog_is_job_scoped_and_only_promotes_overlapping_events(review_env):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        job_id = _job_for_run(session, ids["a_1_one_v2"])
        first = _event(session, ids, "new-hit", seconds=20, run="a_1_one_v2")
        _event(session, ids, "new-still-hit", seconds=25, run="a_1_one_v2")
        _event(session, ids, "new-outside", seconds=80, run="a_1_one_v2")
        _event(session, ids, "old-same-time", seconds=20)
    _preferred_window(
        settings, ids["site_a"], job_id,
        start_utc=(BASE + timedelta(seconds=10)).isoformat(),
        end_utc=(BASE + timedelta(seconds=40)).isoformat(),
    )
    batch = _batch(review_env)
    assert set(_members({"groups": batch["groups"][:1]})) == {
        "new-hit", "new-still-hit", "old-same-time",
    }
    assert batch["total_groups"] == 2
    assert batch["groups"][0]["representative"]["event_uid"] == "new-hit"
    assert any("本批目录优先" in reason for reason in batch["groups"][0]["reasons"])
    for group in batch["groups"][1:]:
        assert not any("本批目录优先" in reason for reason in group["reasons"])
    assert _members({"groups": batch["groups"][1:]}) == ["new-outside"]
    with sessions.begin() as session:
        Repositories(session).reviews.add(first, label="noise", admin_id=1)
    after_review = _batch(review_env)
    assert after_review["groups"][0]["id"] == batch["groups"][0]["id"]
    assert after_review["groups"][0]["representative"]["event_uid"] == "new-still-hit"
    assert after_review["groups"][0]["selection_kind"] == "earthquake"
    assert any(
        "本批目录优先" in reason for reason in after_review["groups"][0]["reasons"]
    )
    assert set(_members({"groups": after_review["groups"][:1]})) == {
        "new-still-hit", "old-same-time",
    }


def test_preferred_catalog_automatically_includes_later_results_and_excludes_reviewed(review_env):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        job_id = _job_for_run(session, ids["a_1_one_v2"])
        _event(session, ids, "old", seconds=500)
    _preferred_window(settings, ids["site_a"], job_id)
    assert _members(_batch(review_env)) == ["old"]
    with sessions.begin() as session:
        event_id = _event(session, ids, "arrived-later", run="a_1_one_v2")
    assert _members(_batch(review_env))[0] == "arrived-later"
    with sessions.begin() as session:
        Repositories(session).reviews.add(event_id, label="noise", admin_id=1)
    assert _members(_batch(review_env)) == ["old"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"preferred_job_ids": None}, {"preferred_job_ids": "job"},
        {"preferred_job_ids": [""]}, {"preferred_job_ids": [1]},
    ],
)
def test_invalid_preferred_catalog_scope_cannot_promote_all_events(review_env, overrides):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        job_id = _job_for_run(session, ids["a_1_one_v2"])
        _event(session, ids, "candidate", run="a_1_one_v2")
    _preferred_window(settings, ids["site_a"], job_id, **overrides)
    batch = _batch(review_env)
    assert batch["catalog_window_count"] == 0
    assert _members(batch) == ["candidate"]
    assert any("无效目录窗口" in warning for warning in batch["warnings"])


def _window(site_id, *, catalog_id="catalog", phase="P", start=0, end=120, **extra):
    return {
        "site_id": site_id,
        "catalog_id": catalog_id,
        "phase": phase,
        "start_utc": (BASE + timedelta(seconds=start)).isoformat(),
        "end_utc": (BASE + timedelta(seconds=end)).isoformat(),
        "source_url": f"https://earthquake.usgs.gov/earthquakes/eventpage/{catalog_id}",
        **extra,
    }


def _write_windows(settings, windows):
    (settings.app_data_dir / "review-catalog-windows.json").write_text(
        json.dumps({"schema_version": 1, "windows": windows}), encoding="utf-8"
    )


def test_catalog_window_collects_all_fragments_across_sources_channels_and_versions(
    review_env,
):
    settings, sessions, ids = review_env
    runs = ("a_1_one_v1", "a_1_two_v1", "a_1_one_v2", "a_2_one_v1")
    with sessions.begin() as session:
        for index in range(43):
            _event(
                session, ids, f"fragment-{index}", seconds=10 + index,
                channels=(index * 10, (index + 1) * 10), run=runs[index % 4],
            )
        _event(session, ids, "reviewed", seconds=15, label="noise")
        _event(session, ids, "outside", seconds=150)
        _event(session, ids, "foreign", seconds=15, run="b_1_one_v1")
    _write_windows(settings, [_window(ids["site_a"], end=100)])

    batch = _batch(review_env)
    assert batch["total_groups"] == 2
    assert batch["total_unreviewed"] == 44
    group = batch["groups"][0]
    assert {member["event_uid"] for member in group["members"]} == {
        f"fragment-{index}" for index in range(43)
    }
    assert len(_members(batch)) == len(set(_members(batch))) == 44
    assert group["catalog_window"] == {
        "catalog_id": "catalog", "phase": "P",
        "start_utc": "2026-08-01T00:00:00Z", "end_utc": "2026-08-01T00:01:40Z",
        "source_url": "https://earthquake.usgs.gov/earthquakes/eventpage/catalog",
    }
    assert group["start_utc"] == "2026-08-01T00:00:10Z"
    assert group["end_utc"] == "2026-08-01T00:00:53Z"
    assert group["representatives"][0] == group["representative"]
    assert len(group["representatives"]) == 3
    assert all(member in group["members"] for member in group["representatives"])
    assert any("逐条复核" in reason for reason in group["reasons"])
    assert not any("空间重叠且信号相近" in reason for reason in group["reasons"])


def test_overlapping_windows_assign_once_by_job_then_overlap_then_stable_order(
    review_env,
):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        job_id = _job_for_run(session, ids["a_1_one_v2"])
        _event(
            session, ids, "own-job", seconds=25, run="a_1_one_v2",
            features={"duration_s": 30, "rms": 1},
        )
        _event(
            session, ids, "longest", seconds=25,
            features={"duration_s": 30, "rms": 1},
        )
        _event(session, ids, "tie", seconds=25)
    windows = [
        _window(ids["site_a"], catalog_id="early", start=0, end=40,
                preferred_job_ids=[job_id]),
        _window(ids["site_a"], catalog_id="late", start=20, end=100),
    ]
    _write_windows(settings, windows)
    batch = _batch(review_env)
    by_catalog = {
        group["catalog_window"]["catalog_id"]: {
            member["event_uid"] for member in group["members"]
        }
        for group in batch["groups"]
    }
    assert by_catalog == {"early": {"own-job", "tie"}, "late": {"longest"}}
    assert len(_members(batch)) == len(set(_members(batch))) == 3
    _write_windows(settings, list(reversed(windows)))
    assert _batch(review_env) == batch


def test_duplicate_normalized_windows_merge_job_scopes_and_keep_a_stable_id(review_env):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        job_id = _job_for_run(session, ids["a_1_one_v2"])
        _event(session, ids, "old", seconds=10)
    first = _window(ids["site_a"], start=5, end=100)
    duplicate = {
        **first,
        "start_utc": "2026-08-01T08:00:05+08:00",
        "end_utc": "2026-08-01T08:01:40+08:00",
        "preferred_job_ids": [job_id],
    }
    _write_windows(settings, [first, duplicate, first])
    before = _batch(review_env)
    assert before["catalog_window_count"] == 1
    group_id = before["groups"][0]["id"]
    with sessions.begin() as session:
        arrived = _event(session, ids, "arrived", seconds=20, run="a_1_one_v2")
    with_new = _batch(review_env)
    assert with_new["total_groups"] == 1
    assert with_new["groups"][0]["id"] == group_id
    assert with_new["groups"][0]["representative"]["event_uid"] == "arrived"
    assert any("本批目录优先" in reason for reason in with_new["groups"][0]["reasons"])
    _write_windows(settings, [duplicate, first])
    assert _batch(review_env) == with_new
    with sessions.begin() as session:
        Repositories(session).reviews.add(arrived, label="noise", admin_id=1)
    after = _batch(review_env)
    assert after["groups"][0]["id"] == group_id
    assert _members(after) == ["old"]


def test_same_catalog_p_and_s_windows_are_separate_and_site_scoped(review_env):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        p_job = _job_for_run(session, ids["a_1_one_v1"])
        s_job = _job_for_run(session, ids["a_1_one_v2"])
        _event(session, ids, "p", seconds=10)
        _event(session, ids, "s", seconds=10, run="a_1_one_v2")
        _event(session, ids, "other-site", seconds=10, run="b_1_one_v1")
    _write_windows(settings, [
        _window(ids["site_a"], phase="S", preferred_job_ids=[s_job]),
        _window(ids["site_a"], phase="P", preferred_job_ids=[p_job]),
        _window(ids["site_b"], phase="P"),
    ])
    batch = _batch(review_env)
    assert batch["total_groups"] == batch["catalog_window_count"] == 2
    assert {
        group["catalog_window"]["phase"]: _members({"groups": [group]})
        for group in batch["groups"]
    } == {"P": ["p"], "S": ["s"]}
    group_ids = {group["id"] for group in batch["groups"]}
    assert len(group_ids) == 2
    assert _batch(review_env, site="b")["groups"][0]["id"] not in group_ids


def test_window_keeps_compatible_reference_scores_with_mixed_processing_versions(
    review_env,
):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        _event(session, ids, "reference", value=10, label="earthquake", seconds=-10)
        _event(session, ids, "compatible", value=10, seconds=10)
        _event(session, ids, "incompatible", value=10, seconds=20, run="a_1_one_v2")
    _write_windows(settings, [_window(ids["site_a"])])
    both = _batch(review_env)
    assert both["total_groups"] == 1
    assert both["groups"][0]["representative"]["event_uid"] == "compatible"
    with sessions.begin() as session:
        compatible = session.query(Event).filter_by(event_uid="compatible").one()
        Repositories(session).reviews.add(compatible.id, label="noise", admin_id=1)
    only_incompatible = _batch(review_env)["groups"][0]
    assert only_incompatible["similar_earthquake_event_id"] is None
    assert only_incompatible["priority_score"] == 40
    assert not any("信号距离" in reason for reason in only_incompatible["reasons"])
    assert only_incompatible["id"] == both["groups"][0]["id"]


def test_window_representatives_cover_distant_channels_and_times_before_near_duplicates(
    review_env,
):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        _event(session, ids, "reference", value=10, label="earthquake", seconds=-10)
        _event(session, ids, "anchor", value=10)
        for index in range(1, 6):
            _event(session, ids, f"duplicate-{index}", value=10, seconds=index)
        _event(session, ids, "distant-channel", channels=(900, 910), seconds=1)
        _event(session, ids, "later-time", seconds=500)
    _write_windows(settings, [_window(ids["site_a"], end=600)])
    batch = _batch(review_env)
    assert batch["total_groups"] == 1
    group = batch["groups"][0]
    assert group["representative"]["event_uid"] == "anchor"
    assert group["representatives"][0] == group["representative"]
    assert {member["event_uid"] for member in group["representatives"]} == {
        "anchor", "distant-channel", "later-time",
    }
    assert all(member in group["members"] for member in group["representatives"])
    assert _batch(review_env) == batch


def test_catalog_api_preserves_all_members_and_exposes_window_and_representatives(
    review_env, monkeypatch,
):
    settings, sessions, ids = review_env
    with sessions.begin() as session:
        for index in range(24):
            _event(session, ids, f"api-{index}", seconds=index, channels=(index, index + 1))
    _write_windows(settings, [_window(ids["site_a"])])
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    with TestClient(create_app(settings)) as client:
        client.post("/api/v1/session", json={"password": "correct horse"})
        response = client.get("/api/v1/review-batches", params={"site_id": ids["site_a"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_groups"] == 1
    group = body["groups"][0]
    assert len(group["members"]) == 24
    assert group["representatives"][0] == group["representative"]
    assert len(group["representatives"]) == 3
    assert group["catalog_window"]["catalog_id"] == "catalog"
