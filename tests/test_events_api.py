from __future__ import annotations

import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.domain import EventRecord
from das_sentinel.features import extract_features
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
from das_sentinel.services.jobs import JobValidationError
from fastapi.testclient import TestClient
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy import func, select

from tests.synthetic import preprocessed_block

PASSWORD = "correct horse"
WORKER_TOKEN = "worker-token-" + "x" * 32


def _seed_events_with_extract_features_output(event_api, count: int) -> str:
    client, _, sessions, _, ids = event_api
    del client
    block = preprocessed_block(
        np.tile(np.linspace(-1.0, 1.0, 80, dtype=np.float32)[:, None], (1, 4))
    )
    record = EventRecord(
        event_id="feature-template",
        start_sample=1_010,
        end_sample=1_070,
        peak_sample=1_040,
        channel_start=0,
        channel_stop=4,
        evidence=(),
    )
    template = extract_features(
        block,
        record,
        context={
            "site_id": ids["site_a"],
            "cable_type": "urban",
            "device_config": "cpu",
        },
    )
    created = []
    with sessions.begin() as session:
        repositories = Repositories(session)
        for index in range(count):
            event = repositories.events.add(
                run_id=ids["run_a"],
                event_uid=f"real-feature-{index}",
                start_sample=100 + index * 10,
                end_sample=105 + index * 10,
                channel_start=0,
                channel_stop=4,
                occurred_at=datetime(2026, 8, 1, 0, index, tzinfo=UTC),
                features={**template, "rms": float(index + 1)},
            )
            repositories.events.add_classification(
                event.id, label="traffic", decision_score=0.9, method="rules"
            )
            created.append(event.id)
    return created[0]


def test_similar_events_accept_real_string_context_features(event_api):
    client, _, _, _, _ = event_api
    event_id = _seed_events_with_extract_features_output(event_api, count=6)

    response = client.get(f"/api/v1/events/{event_id}/similar?limit=5")

    assert response.status_code == 200
    assert len(response.json()) == 5


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


def test_event_list_select_count_is_constant_across_page_sizes(event_api):
    client, _, _, _, _ = event_api
    engine = client.app.state.engine
    select_counts = []

    for limit in (1, 3, 100):
        statements = []

        def count_selects(
            connection,
            cursor,
            statement,
            parameters,
            context,
            executemany,
            captured=statements,
        ):
            del connection, cursor, parameters, context, executemany
            if statement.lstrip().upper().startswith("SELECT"):
                captured.append(statement)

        sqlalchemy_event.listen(engine, "before_cursor_execute", count_selects)
        try:
            response = client.get("/api/v1/events", params={"limit": limit})
        finally:
            sqlalchemy_event.remove(
                engine,
                "before_cursor_execute",
                count_selects,
            )
        assert response.status_code == 200
        select_counts.append(len(statements))

    assert select_counts[0] == select_counts[1] == select_counts[2]
    assert select_counts[0] <= 3


def test_event_list_uses_created_at_then_id_for_latest_machine_label(event_api):
    client, _, sessions, _, ids = event_api
    with sessions.begin() as session:
        Repositories(session).events.add_classification(
            ids["event_1"],
            label="traffic",
            decision_score=0.99,
            method="backdated",
            created_at=datetime(2020, 1, 1, tzinfo=UTC),
        )

    traffic = client.get(
        "/api/v1/events",
        params={"machine_label": "traffic"},
    )
    unknown = client.get(
        "/api/v1/events",
        params={"machine_label": "unknown"},
    )

    assert traffic.status_code == 200
    assert unknown.status_code == 200
    assert "event-1" not in [item["event_uid"] for item in traffic.json()]
    event = next(item for item in unknown.json() if item["event_uid"] == "event-1")
    assert event["machine_label"] == "unknown"


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


def _seed_active_result_model(event_api) -> str:
    _, settings, sessions, _, ids = event_api
    model_path = (
        settings.app_data_dir
        / "artifacts"
        / "models"
        / ids["site_a"]
        / "7"
        / "model.bundle"
    )
    with sessions.begin() as session:
        model = Repositories(session).models.add(
            site_id=ids["site_a"],
            version="7",
            artifact_path=str(model_path),
            bundle_sha256="7" * 64,
            status="active",
            metrics={"feature_names": ["duration_s", "rms"]},
        )
        model_id = model.id
    return model_id


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


def test_bound_model_result_persists_one_identity_and_fusion_method(event_api):
    client, _, sessions, _, _ = event_api
    model_id = _seed_active_result_model(event_api)
    job_id, payload = _claim_result_job(event_api)
    payload["model_version_id"] = model_id
    payload["events"][0]["classification"]["method"] = "fusion"

    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 200
    with sessions() as session:
        run = session.scalar(
            select(ProcessingRun).where(ProcessingRun.job_id == job_id)
        )
        assert run is not None
        classification = session.scalar(
            select(Classification)
            .join(Event)
            .where(Event.processing_run_id == run.id)
        )
        assert classification is not None
        assert run.model_version_id == model_id
        assert classification.model_version_id == model_id
        assert classification.method == "fusion"


def test_result_rejects_payload_model_identity_mismatch(event_api):
    client, _, sessions, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    payload["model_version_id"] = "00000000-0000-0000-0000-000000000007"
    payload["events"][0]["classification"]["method"] = "fusion"

    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 409
    with sessions() as session:
        assert session.get(Job, job_id).status == "running"


def test_result_rejects_method_without_model_identity(event_api):
    client, _, sessions, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    payload["events"][0]["classification"]["method"] = "fusion"

    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 422
    with sessions() as session:
        assert session.get(Job, job_id).status == "running"


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


def test_result_hashes_artifacts_without_materializing_paths(
    event_api,
    monkeypatch: pytest.MonkeyPatch,
):
    client, settings, _, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    artifact_root = settings.app_data_dir / "artifacts"
    original_read_bytes = Path.read_bytes

    def reject_artifact_path_reads(path: Path):
        if path.is_relative_to(artifact_root):
            raise AssertionError("result validation must use a held file handle")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_artifact_path_reads)
    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"


def test_result_holds_same_artifact_handles_through_job_completion(
    event_api,
    monkeypatch: pytest.MonkeyPatch,
):
    import das_sentinel.services.results as results_module

    client, _, _, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    original_hash = results_module._stream_sha256
    original_complete = results_module.JobService.complete
    hashed_streams = []

    def record_hash(stream):
        hashed_streams.append(stream)
        return original_hash(stream)

    def assert_held_during_complete(service, *args, **kwargs):
        assert len(hashed_streams) == 6
        assert all(not stream.closed for stream in hashed_streams)
        return original_complete(service, *args, **kwargs)

    monkeypatch.setattr(results_module, "_stream_sha256", record_hash)
    monkeypatch.setattr(
        results_module.JobService,
        "complete",
        assert_held_during_complete,
    )
    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 200
    assert len(hashed_streams) == 12
    assert all(
        first is second
        for first, second in zip(
            hashed_streams[:6],
            hashed_streams[6:],
            strict=True,
        )
    )
    assert all(stream.closed for stream in hashed_streams)


def test_result_change_after_job_completion_rolls_back_atomically(
    event_api,
    monkeypatch: pytest.MonkeyPatch,
):
    import das_sentinel.services.results as results_module

    client, _, sessions, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    original_hash = results_module._stream_sha256
    original_complete = results_module.JobService.complete
    completion_reached = False
    streams = []

    def mark_completion(service, *args, **kwargs):
        nonlocal completion_reached
        completed = original_complete(service, *args, **kwargs)
        completion_reached = True
        return completed

    def change_one_digest_on_recheck(stream):
        streams.append(stream)
        digest = original_hash(stream)
        if completion_reached:
            return "0" * 64
        return digest

    monkeypatch.setattr(
        results_module.JobService,
        "complete",
        mark_completion,
    )
    monkeypatch.setattr(
        results_module,
        "_stream_sha256",
        change_one_digest_on_recheck,
    )
    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 422
    assert completion_reached
    assert streams
    assert all(stream.closed for stream in streams)
    with sessions() as session:
        assert session.get(Job, job_id).status == "running"
        assert session.scalar(
            select(func.count()).select_from(ProcessingRun)
            .where(ProcessingRun.job_id == job_id)
        ) == 0


def test_result_rejects_symlink_without_reading_external_file(
    event_api,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _, sessions, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    linked = Path(payload["artifacts"][0]["path"])
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"external")
    linked.unlink()
    try:
        linked.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    payload["artifacts"][0]["sha256"] = hashlib.sha256(b"external").hexdigest()
    original_read_bytes = Path.read_bytes
    path_reads = []

    def track_path_reads(path: Path):
        if path == linked or path == outside:
            path_reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", track_path_reads)
    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/result",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 422
    assert path_reads == []
    with sessions() as session:
        assert session.get(Job, job_id).status == "running"
        assert session.scalar(
            select(func.count()).select_from(ProcessingRun)
            .where(ProcessingRun.job_id == job_id)
        ) == 0


def test_result_rejects_oversized_artifact_before_reading_it(
    event_api,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _, sessions, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    oversized = Path(payload["artifacts"][0]["path"])
    with oversized.open("wb") as stream:
        stream.seek(64 * 1024 * 1024)
        stream.write(b"x")

    def reject_path_reads(path: Path):
        raise AssertionError(f"oversized artifact was materialized: {path}")

    monkeypatch.setattr(Path, "read_bytes", reject_path_reads)
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


def test_result_rejects_too_many_artifacts_before_opening_any(
    event_api,
    monkeypatch: pytest.MonkeyPatch,
):
    import das_sentinel.services.results as results_module

    client, _, sessions, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    base_event = payload["events"][0]
    base_artifacts = payload["artifacts"]
    events = []
    artifacts = []
    for index in range(101):
        event_uid = f"bounded-event-{index}"
        events.append({**base_event, "event_uid": event_uid})
        artifacts.extend(
            {**artifact, "event_uid": event_uid}
            for artifact in base_artifacts
        )
    payload["events"] = events
    payload["artifacts"] = artifacts

    def reject_any_open(*args, **kwargs):
        del args, kwargs
        raise AssertionError("artifact count must be rejected before opening")

    monkeypatch.setattr(
        results_module,
        "_open_anchored_source",
        reject_any_open,
    )
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


def test_result_stream_hash_enforces_limit_while_reading():
    import das_sentinel.services.results as results_module

    stream = io.BytesIO(b"123456789")

    with pytest.raises(JobValidationError, match="size limit"):
        results_module._stream_sha256(stream, maximum_bytes=8)

    assert stream.tell() == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO semantics")
def test_result_rejects_fifo_without_blocking(
    event_api,
):
    client, _, _, _, _ = event_api
    job_id, payload = _claim_result_job(event_api)
    fifo = Path(payload["artifacts"][0]["path"])
    fifo.unlink()
    os.mkfifo(fifo)
    completed_promptly = True

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.post,
            f"/api/v1/internal/jobs/{job_id}/result",
            json=payload,
            headers={"X-Worker-Token": WORKER_TOKEN},
        )
        try:
            response = future.result(timeout=1)
        except TimeoutError:
            completed_promptly = False
            writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            os.close(writer)
            response = future.result(timeout=2)

    assert completed_promptly
    assert response.status_code == 422


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


@pytest.mark.skipif(os.name != "nt", reason="Windows handle sharing semantics")
def test_windows_artifact_parent_handle_blocks_replacement(
    event_api,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import das_sentinel.api.routes.artifacts as artifact_route

    client, settings, _, _, ids = event_api
    artifact_id, target = _register_artifact(event_api)
    root = settings.app_data_dir / "artifacts"
    parent = target.parent
    moved = root / f"{ids['event_1']}-moved"
    outside = tmp_path / "attacker"
    outside.mkdir()
    (outside / target.name).write_bytes(b"attacker")
    original_open = artifact_route._open_artifact_component
    replacement_was_blocked = False

    def attempt_replacement_after_open(path, parent_handle, *, expect_directory):
        nonlocal replacement_was_blocked
        opened = original_open(
            path,
            parent_handle,
            expect_directory=expect_directory,
        )
        if Path(path) == parent:
            try:
                parent.rename(moved)
            except OSError:
                replacement_was_blocked = True
            else:
                parent.symlink_to(outside, target_is_directory=True)
        return opened

    monkeypatch.setattr(
        artifact_route,
        "_open_artifact_component",
        attempt_replacement_after_open,
    )
    response = client.get(f"/api/v1/artifacts/{artifact_id}")

    assert replacement_was_blocked
    assert response.status_code == 200
    assert response.content == b"trusted"
