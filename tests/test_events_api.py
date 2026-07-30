from __future__ import annotations

import hashlib
from datetime import UTC, datetime
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
from das_sentinel.persistence.models import (
    Artifact,
    Classification,
    DetectionEvidence,
    Event,
    Job,
    ProcessingRun,
    Review,
)
from das_sentinel.services.discovery import fingerprint
from fastapi.testclient import TestClient
from sqlalchemy import func, select

PASSWORD = "correct horse"
WORKER_TOKEN = "worker-token-" + "x" * 32


def _settings(tmp_path: Path) -> Settings:
    raw = tmp_path / "raw"
    app_data = tmp_path / "app-data"
    raw.mkdir()
    app_data.mkdir()
    return Settings(
        data_root=raw,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}",
        api_url="http://testserver",
        session_secret="s" * 32,
        worker_token=WORKER_TOKEN,
        secure_cookies=False,
        pipeline_version="pipeline-v1",
    )


@pytest.fixture
def event_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    source_path = settings.data_root / "input.h5"
    source_path.write_bytes(b"source")
    with sessions.begin() as session:
        repositories = Repositories(session)
        repositories.admins.add(password_hash=PasswordHasher().hash(PASSWORD))
        site_a = repositories.sites.add(
            name="site-a",
            data_path=str(settings.data_root),
            timezone="UTC",
            cable_type="urban",
        )
        site_b = repositories.sites.add(
            name="site-b",
            data_path=str(settings.data_root),
            timezone="UTC",
            cable_type="subsea",
        )
        profile_a = repositories.profiles.add(
            site_id=site_a.id,
            version=2,
            payload=ProcessingProfile(device="cpu").model_dump(mode="json"),
            is_current=True,
        )
        profile_b = repositories.profiles.add(
            site_id=site_b.id,
            version=1,
            payload=ProcessingProfile(device="cpu").model_dump(mode="json"),
            is_current=True,
        )
        source_a = repositories.source_files.add(
            site_id=site_a.id,
            path=str(source_path.resolve()),
            fingerprint=fingerprint(source_path),
            size_bytes=source_path.stat().st_size,
            mtime_ns=source_path.stat().st_mtime_ns,
        )
        source_b = repositories.source_files.add(
            site_id=site_b.id,
            path=str(source_path.resolve()),
            fingerprint="b" * 64,
            size_bytes=source_path.stat().st_size,
            mtime_ns=source_path.stat().st_mtime_ns,
        )
        run_a = repositories.runs.add(
            source_file_id=source_a.id,
            processing_profile_id=profile_a.id,
            pipeline_version="pipeline-v1",
            requested_backend="cuda",
            actual_backend="cpu",
            backend_fallback_reason="cuda_unavailable",
            status="succeeded",
            timings={"total": 5.0},
        )
        run_a.started_at = datetime(2026, 7, 30, tzinfo=UTC)
        run_a.finished_at = datetime(2026, 7, 30, 0, 1, tzinfo=UTC)
        run_b = repositories.runs.add(
            source_file_id=source_b.id,
            processing_profile_id=profile_b.id,
            pipeline_version="pipeline-v1",
            requested_backend="cpu",
            actual_backend="cpu",
            status="succeeded",
        )
        event_1 = repositories.events.add(
            run_id=run_a.id,
            event_uid="event-1",
            start_sample=10,
            end_sample=20,
            channel_start=1,
            channel_stop=3,
            occurred_at=datetime(2026, 7, 30, 0, 0, 10, tzinfo=UTC),
            features={"rms": 1.0},
        )
        repositories.events.add_evidence(
            event_1.id,
            detector="energy",
            score=0.8,
            payload={"ratio": 4.0},
        )
        repositories.events.add_classification(
            event_1.id,
            label="unknown",
            decision_score=0.55,
            method="rules",
            reasons=["low margin"],
        )
        event_2 = repositories.events.add(
            run_id=run_a.id,
            event_uid="event-2",
            start_sample=20,
            end_sample=30,
            channel_start=2,
            channel_stop=4,
            occurred_at=datetime(2026, 7, 30, 0, 0, 20, tzinfo=UTC),
        )
        repositories.events.add_classification(
            event_2.id,
            label="traffic",
            decision_score=0.9,
            method="rules",
        )
        repositories.reviews.add(
            event_2.id,
            label="traffic",
            note="vehicle",
            admin_id=1,
        )
        event_3 = repositories.events.add(
            run_id=run_b.id,
            event_uid="event-3",
            start_sample=30,
            end_sample=40,
            channel_start=3,
            channel_stop=5,
            occurred_at=datetime(2026, 7, 31, tzinfo=UTC),
        )
        repositories.events.add_classification(
            event_3.id,
            label="blast",
            decision_score=0.8,
            method="rules",
        )
        ids = {
            "site_a": site_a.id,
            "site_b": site_b.id,
            "profile_a": profile_a.id,
            "source_a": source_a.id,
            "run_a": run_a.id,
            "event_1": event_1.id,
            "event_2": event_2.id,
            "event_3": event_3.id,
        }
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    with TestClient(create_app(settings)) as client:
        login = client.post("/api/v1/session", json={"password": PASSWORD})
        yield client, settings, sessions, login.headers["x-csrf-token"], ids
    engine.dispose()


def test_event_routes_require_admin_and_review_requires_csrf(event_api):
    client, _, _, token, ids = event_api
    client.cookies.clear()
    assert client.get("/api/v1/events").status_code == 401
    assert client.get(f"/api/v1/events/{ids['event_1']}").status_code == 401
    assert (
        client.post(
            f"/api/v1/events/{ids['event_1']}/reviews",
            json={"label": "earthquake", "note": "visible P/S"},
            headers={"X-CSRF-Token": token},
        ).status_code
        == 401
    )
    login = client.post("/api/v1/session", json={"password": PASSWORD})
    assert (
        client.post(
            f"/api/v1/events/{ids['event_1']}/reviews",
            json={"label": "earthquake"},
            headers={"X-CSRF-Token": "wrong"},
        ).status_code
        == 403
    )
    assert login.status_code == 204


def test_event_list_filters_and_stable_bounded_pagination(event_api):
    client, _, _, _, ids = event_api
    all_events = client.get("/api/v1/events").json()
    assert [item["event_uid"] for item in all_events] == [
        "event-3",
        "event-2",
        "event-1",
    ]
    assert [item["event_uid"] for item in client.get(
        "/api/v1/events", params={"site_id": ids["site_a"]}
    ).json()] == ["event-2", "event-1"]
    assert [item["event_uid"] for item in client.get(
        "/api/v1/events", params={"machine_label": "traffic"}
    ).json()] == ["event-2"]
    assert [item["event_uid"] for item in client.get(
        "/api/v1/events", params={"reviewed": "true"}
    ).json()] == ["event-2"]
    assert [item["event_uid"] for item in client.get(
        "/api/v1/events", params={"reviewed": "false"}
    ).json()] == ["event-3", "event-1"]
    assert [item["event_uid"] for item in client.get(
        "/api/v1/events",
        params={
            "start_utc": "2026-07-30T00:00:15Z",
            "end_utc": "2026-07-30T23:59:59Z",
        },
    ).json()] == ["event-2"]
    assert [item["event_uid"] for item in client.get(
        "/api/v1/events", params={"limit": 1, "offset": 1}
    ).json()] == ["event-2"]
    assert client.get("/api/v1/events", params={"limit": 101}).status_code == 422
    assert client.get(
        "/api/v1/events", params={"start_utc": "2026-07-30"}
    ).status_code == 422


def test_event_detail_contains_audit_history_and_no_local_paths(event_api):
    client, settings, _, _, ids = event_api
    response = client.get(f"/api/v1/events/{ids['event_1']}")

    assert response.status_code == 200
    detail = response.json()
    assert detail["site_id"] == ids["site_a"]
    assert detail["run"]["requested_backend"] == "cuda"
    assert detail["run"]["actual_backend"] == "cpu"
    assert detail["run"]["source_file_id"] == ids["source_a"]
    assert detail["run"]["source_fingerprint"] == fingerprint(
        settings.data_root / "input.h5"
    )
    assert detail["run"]["processing_profile_id"] == ids["profile_a"]
    assert detail["run"]["profile_version"] == 2
    assert detail["run"]["pipeline_version"] == "pipeline-v1"
    assert detail["evidence"][0]["detector"] == "energy"
    assert detail["machine_history"][0]["label"] == "unknown"
    assert detail["review_history"] == []
    assert str(settings.data_root) not in response.text
    assert "source_path" not in response.text
    assert "artifact_path" not in response.text
    assert client.get("/api/v1/events/missing").status_code == 404


def test_event_review_appends_without_mutating_machine_or_prior_review(event_api):
    client, _, sessions, token, ids = event_api
    first = client.post(
        f"/api/v1/events/{ids['event_1']}/reviews",
        json={"label": "earthquake", "note": "visible P/S"},
        headers={"X-CSRF-Token": token},
    )
    second = client.post(
        f"/api/v1/events/{ids['event_1']}/reviews",
        json={"label": "noise", "note": "second opinion"},
        headers={"X-CSRF-Token": token},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    detail = client.get(f"/api/v1/events/{ids['event_1']}").json()
    assert detail["machine_label"] == "unknown"
    assert [item["label"] for item in detail["review_history"]] == [
        "earthquake",
        "noise",
    ]
    assert detail["latest_review"]["label"] == "noise"
    with sessions() as session:
        assert session.scalar(
            select(func.count()).select_from(Classification)
            .where(Classification.event_id == ids["event_1"])
        ) == 1
        assert session.scalar(
            select(func.count()).select_from(Review)
            .where(Review.event_id == ids["event_1"])
        ) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"label": "unknown"},
        {"label": "earthquake", "note": "x" * 2001},
        {"label": "earthquake", "extra": True},
    ],
)
def test_review_schema_is_strict_and_bounded(event_api, payload):
    client, _, _, token, ids = event_api
    assert client.post(
        f"/api/v1/events/{ids['event_1']}/reviews",
        json=payload,
        headers={"X-CSRF-Token": token},
    ).status_code == 422


def _result_payload(ids, source_path: Path) -> dict:
    event_uid = "00000000-0000-0000-0000-000000000099"
    artifact_dir = source_path.parent.parent / "app-data" / "artifacts" / event_uid
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    suffixes = {
        "waterfall_raw": ".png",
        "waterfall_filtered": ".png",
        "waveform": ".png",
        "psd": ".png",
        "stft": ".png",
        "evidence": ".json",
    }
    for kind, suffix in suffixes.items():
        path = artifact_dir / f"{kind}{suffix}"
        content = kind.encode()
        path.write_bytes(content)
        artifacts.append(
            {
                "event_uid": event_uid,
                "kind": kind,
                "path": str(path),
                "media_type": (
                    "application/json" if suffix == ".json" else "image/png"
                ),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "worker_id": "worker-a",
        "source_file_id": ids["source_a"],
        "processing_profile_id": ids["profile_a"],
        "profile_version": 2,
        "pipeline_version": "pipeline-result",
        "source_fingerprint": fingerprint(source_path),
        "source_start_time_utc": "2026-07-30T00:00:00Z",
        "sample_rate_hz": 1000.0,
        "requested_backend": "cuda",
        "actual_backend": "cpu",
        "backend_fallback_reason": "cuda_unavailable",
        "timings_ms": {"total": 12.5},
        "events": [
            {
                "event_uid": event_uid,
                "start_sample": 1000,
                "end_sample": 2000,
                "channel_start": 1,
                "channel_stop": 5,
                "features": {"rms": 1.2},
                "evidence": [
                    {
                        "detector": "energy",
                        "score": 0.8,
                        "payload": {"ratio": 4.0},
                    }
                ],
                "classification": {
                    "label": "unknown",
                    "decision_score": 0.55,
                    "method": "rules",
                    "reasons": ["low margin"],
                },
            }
        ],
        "artifacts": artifacts,
    }


def _claim_result_job(event_api):
    client, settings, _, token, ids = event_api
    source_path = settings.data_root / "input.h5"
    response = client.post(
        "/api/v1/jobs",
        json={
            "source_file_id": ids["source_a"],
            "processing_profile_id": ids["profile_a"],
            "pipeline_version": "pipeline-result",
        },
        headers={"X-CSRF-Token": token},
    )
    job_id = response.json()["id"]
    assert client.post(
        "/api/v1/internal/jobs/claim",
        json={"worker_id": "worker-a"},
        headers={"X-Worker-Token": WORKER_TOKEN},
    ).status_code == 200
    return job_id, _result_payload(ids, source_path)


def test_internal_result_persists_complete_audit_graph_atomically(event_api):
    client, _, sessions, _, ids = event_api
    job_id, payload = _claim_result_job(event_api)

    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    with sessions() as session:
        job = session.get(Job, job_id)
        run = session.scalar(
            select(ProcessingRun).where(ProcessingRun.job_id == job_id)
        )
        assert job is not None and job.status == "succeeded"
        assert run is not None
        assert run.source_file_id == ids["source_a"]
        assert run.requested_backend == "cuda"
        assert run.actual_backend == "cpu"
        event = session.scalar(
            select(Event).where(Event.processing_run_id == run.id)
        )
        assert event is not None
        assert event.occurred_at == datetime(2026, 7, 30, 0, 0, 1, tzinfo=UTC)
        assert session.scalar(
            select(func.count()).select_from(DetectionEvidence)
            .where(DetectionEvidence.event_id == event.id)
        ) == 1
        assert session.scalar(
            select(func.count()).select_from(Classification)
            .where(Classification.event_id == event.id)
        ) == 1
        assert session.scalar(
            select(func.count()).select_from(Artifact)
            .where(Artifact.event_id == event.id)
        ) == 6


def test_rejected_result_rolls_back_every_row_and_keeps_job_running(event_api):
    client, _, sessions, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    payload["artifacts"][0]["sha256"] = "0" * 64

    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 422
    with sessions() as session:
        assert session.get(Job, job_id).status == "running"
        assert session.scalar(
            select(func.count()).select_from(ProcessingRun)
            .where(ProcessingRun.job_id == job_id)
        ) == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(source_file_id="wrong"),
        lambda payload: payload.update(profile_version=999),
        lambda payload: payload.update(pipeline_version="wrong"),
        lambda payload: payload.update(source_fingerprint="0" * 64),
        lambda payload: payload.update(sample_rate_hz="Infinity"),
        lambda payload: payload["events"][0].update(start_sample=-1),
        lambda payload: payload["events"][0].update(
            start_sample=10**18,
            end_sample=10**18 + 1,
        ),
        lambda payload: payload["events"][0]["classification"].update(
            label="unsupported"
        ),
        lambda payload: payload.update(extra=True),
    ],
)
def test_internal_result_schema_and_identity_are_strict(event_api, mutate):
    client, _, sessions, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    mutate(payload)

    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code in {409, 422}
    with sessions() as session:
        assert session.get(Job, job_id).status == "running"


def _register_artifact(event_api, *, content=b"trusted", stored_path=None):
    _, settings, sessions, _, ids = event_api
    root = settings.app_data_dir / "artifacts"
    target = root / ids["event_1"] / "waterfall.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    with sessions.begin() as session:
        artifact = Repositories(session).artifacts.add(
            event_id=ids["event_1"],
            kind="waterfall_raw",
            path=stored_path or str(target.relative_to(root)),
            media_type="image/png",
            sha256=hashlib.sha256(content).hexdigest(),
        )
        artifact_id = artifact.id
    return artifact_id, target


def test_artifact_download_requires_auth_registration_and_matching_digest(event_api):
    client, _, _, _, _ = event_api
    artifact_id, target = _register_artifact(event_api)

    response = client.get(f"/api/v1/artifacts/{artifact_id}")
    assert response.status_code == 200
    assert response.content == b"trusted"
    assert response.headers["content-type"] == "image/png"
    assert str(target) not in response.text
    client.cookies.clear()
    assert client.get(f"/api/v1/artifacts/{artifact_id}").status_code == 401
    login = client.post("/api/v1/session", json={"password": PASSWORD})
    assert login.status_code == 204
    assert client.get("/api/v1/artifacts/missing").status_code == 404
    target.write_bytes(b"tampered")
    assert client.get(f"/api/v1/artifacts/{artifact_id}").status_code == 409


@pytest.mark.parametrize(
    "stored_path",
    [
        "../outside.png",
        "/absolute/outside.png",
        "event/missing.png",
    ],
)
def test_artifact_download_rejects_unsafe_or_missing_stored_path(
    event_api,
    stored_path,
):
    client, _, _, _, _ = event_api
    artifact_id, _ = _register_artifact(event_api, stored_path=stored_path)
    response = client.get(f"/api/v1/artifacts/{artifact_id}")
    assert response.status_code in {404, 409}
    assert stored_path not in response.text


def test_artifact_download_rejects_registered_symlink(
    event_api,
    tmp_path: Path,
):
    client, settings, sessions, _, ids = event_api
    root = settings.app_data_dir / "artifacts"
    event_dir = root / ids["event_1"]
    event_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    linked = event_dir / "waterfall.png"
    try:
        linked.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    with sessions.begin() as session:
        artifact_id = Repositories(session).artifacts.add(
            event_id=ids["event_1"],
            kind="waterfall_raw",
            path=str(linked.relative_to(root)),
            media_type="image/png",
            sha256=hashlib.sha256(b"outside").hexdigest(),
        ).id

    response = client.get(f"/api/v1/artifacts/{artifact_id}")

    assert response.status_code == 409
    assert str(outside) not in response.text
    assert response.content != b"outside"


def test_artifact_download_rejects_symlink_and_ancestor_replacement_race(
    event_api,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import das_sentinel.api.routes.artifacts as artifact_route

    client, settings, sessions, _, ids = event_api
    root = settings.app_data_dir / "artifacts"
    parent = root / ids["event_1"]
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / "waterfall.png"
    target.write_bytes(b"trusted")
    outside = tmp_path / "attacker"
    outside.mkdir()
    (outside / "waterfall.png").write_bytes(b"attacker")
    with sessions.begin() as session:
        artifact_id = Repositories(session).artifacts.add(
            event_id=ids["event_1"],
            kind="waterfall_raw",
            path=str(target.relative_to(root)),
            media_type="image/png",
            sha256=hashlib.sha256(b"trusted").hexdigest(),
        ).id
    moved = root / "original"
    original_open = artifact_route._open_artifact_component
    replaced = False

    def replace_before_open(path, parent_handle, *, expect_directory):
        nonlocal replaced
        if Path(path) == parent and not replaced:
            replaced = True
            parent.rename(moved)
            try:
                parent.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                pytest.skip(f"symlink unavailable: {error}")
        return original_open(
            path,
            parent_handle,
            expect_directory=expect_directory,
        )

    monkeypatch.setattr(
        artifact_route,
        "_open_artifact_component",
        replace_before_open,
    )
    response = client.get(f"/api/v1/artifacts/{artifact_id}")
    assert response.status_code == 409
    assert b"attacker" not in response.content
