from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Thread

import pytest
from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import Settings
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.services.jobs import JobService
from fastapi.testclient import TestClient

PASSWORD = "correct horse"
WORKER_TOKEN = "worker-secret-" + "x" * 32


def _settings(tmp_path: Path) -> Settings:
    data_root = tmp_path / "raw"
    app_data = tmp_path / "app-data"
    data_root.mkdir(parents=True)
    app_data.mkdir(parents=True)
    return Settings(
        data_root=data_root,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}",
        api_url="http://testserver",
        session_secret="s" * 32,
        worker_token=WORKER_TOKEN,
        pipeline_version="pipeline-v1",
        secure_cookies=False,
    )


def _seed(settings: Settings, *, stale: bool = False) -> tuple[str, str, str]:
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    try:
        sessions = session_factory(engine)
        with sessions.begin() as session:
            repositories = Repositories(session)
            repositories.admins.add(password_hash=PasswordHasher().hash(PASSWORD))
            site = repositories.sites.add(
                name="urban-a",
                data_path=str(settings.data_root),
                timezone="Asia/Shanghai",
                cable_type="urban",
            )
            profile = repositories.profiles.add(
                site_id=site.id,
                version=4,
                payload={"device": "auto"},
                is_current=True,
            )
            source = repositories.source_files.add(
                site_id=site.id,
                path=str(settings.data_root / "input.h5"),
                fingerprint="a" * 64,
                size_bytes=1,
                mtime_ns=2,
            )
            if stale:
                service = JobService(session, settings.app_data_dir / "artifacts")
                job = service.enqueue(source.id, profile.id, "pipeline-v1")
                service.claim(
                    "dead-worker",
                    now=datetime.now(UTC) - timedelta(minutes=3),
                )
                return source.id, profile.id, job.id
            return source.id, profile.id, ""
    finally:
        engine.dispose()


@pytest.fixture
def api_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, Settings, str, str, str]]:
    settings = _settings(tmp_path)
    source_id, profile_id, _ = _seed(settings)
    monkeypatch.setattr(
        "das_sentinel.api.app.check_database_startup",
        lambda engine: None,
    )
    with TestClient(create_app(settings)) as client:
        login = client.post("/api/v1/session", json={"password": PASSWORD})
        yield (
            client,
            settings,
            login.headers["x-csrf-token"],
            source_id,
            profile_id,
        )


def _enqueue(
    client: TestClient,
    csrf_token: str,
    source_id: str,
    profile_id: str,
):
    return client.post(
        "/api/v1/jobs",
        json={
            "source_file_id": source_id,
            "processing_profile_id": profile_id,
            "pipeline_version": "pipeline-v1",
        },
        headers={"X-CSRF-Token": csrf_token},
    )


def test_public_job_routes_require_admin_and_csrf(api_client):
    client, _, token, source_id, profile_id = api_client
    client.cookies.clear()
    assert client.get("/api/v1/jobs").status_code == 401
    assert _enqueue(client, token, source_id, profile_id).status_code == 401

    login = client.post("/api/v1/session", json={"password": PASSWORD})
    assert _enqueue(
        client,
        "wrong",
        source_id,
        profile_id,
    ).status_code == 403
    created = _enqueue(
        client,
        login.headers["x-csrf-token"],
        source_id,
        profile_id,
    )
    assert created.status_code == 201
    assert client.get("/api/v1/jobs").json()[0]["id"] == created.json()["id"]
    assert client.get(f"/api/v1/jobs/{created.json()['id']}").status_code == 200


@pytest.mark.parametrize("token", [None, "", "bad"])
def test_internal_route_rejects_missing_or_wrong_token(api_client, token):
    client, _, _, _, _ = api_client
    headers = {} if token is None else {"X-Worker-Token": token}

    response = client.post(
        "/api/v1/internal/jobs/claim",
        json={"worker_id": "worker-a"},
        headers=headers,
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_browser_session_cannot_replace_worker_token(api_client):
    client, _, _, _, _ = api_client
    assert client.get("/api/v1/session").status_code == 200

    response = client.post(
        "/api/v1/internal/jobs/claim",
        json={"worker_id": "worker-a"},
    )

    assert response.status_code == 401


def test_enqueue_claim_heartbeat_complete_protocol_and_no_token_leak(api_client):
    client, settings, token, source_id, profile_id = api_client
    created = _enqueue(client, token, source_id, profile_id)
    job_id = created.json()["id"]

    claim = client.post(
        "/api/v1/internal/jobs/claim",
        json={"worker_id": "worker-a"},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert claim.status_code == 200
    assert claim.json()["id"] == job_id
    assert claim.json()["source_size_bytes"] == 1
    assert claim.json()["source_mtime_ns"] == 2
    assert claim.json()["profile_version"] == 4
    assert claim.json()["profile_payload"] == {"device": "auto"}

    heartbeat = client.post(
        f"/api/v1/internal/jobs/{job_id}/heartbeat",
        json={"worker_id": "worker-a", "numerator": 4, "denominator": 10},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["progress_numerator"] == 4

    artifact = settings.app_data_dir / "artifacts" / job_id / "waterfall.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"png")
    completed = client.post(
        f"/api/v1/internal/jobs/{job_id}/complete",
        json={
            "worker_id": "worker-a",
            "profile_version": 4,
            "pipeline_version": "pipeline-v1",
            "artifact_paths": [str(artifact)],
        },
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "succeeded"

    exposed = created.text + claim.text + heartbeat.text + completed.text
    assert WORKER_TOKEN not in exposed
    assert "worker_token" not in exposed


def test_protocol_maps_not_found_conflict_and_validation(api_client):
    client, _, token, source_id, profile_id = api_client
    missing = client.post(
        "/api/v1/internal/jobs/missing/heartbeat",
        json={"worker_id": "worker-a", "numerator": 0, "denominator": 0},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert missing.status_code == 404

    created = _enqueue(client, token, source_id, profile_id).json()
    invalid = client.post(
        f"/api/v1/internal/jobs/{created['id']}/heartbeat",
        json={"worker_id": "worker-a", "numerator": 2, "denominator": 1},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert invalid.status_code == 422

    claim = client.post(
        "/api/v1/internal/jobs/claim",
        json={"worker_id": "worker-a"},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert claim.status_code == 200
    wrong_owner = client.post(
        f"/api/v1/internal/jobs/{created['id']}/fail",
        json={
            "worker_id": "worker-b",
            "error_code": "failed",
            "error_message": "failed",
            "retry_eligible": True,
        },
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert wrong_owner.status_code == 409

    missing_public = client.get("/api/v1/jobs/missing")
    assert missing_public.status_code == 404


@pytest.mark.parametrize(
    "progress",
    [
        {"numerator": "1", "denominator": 1},
        {"numerator": 1.0, "denominator": 1},
        {"numerator": True, "denominator": 1},
        {"numerator": 0, "denominator": 2**63},
    ],
)
def test_heartbeat_api_rejects_non_strict_or_sqlite_overflow_progress(
    api_client,
    progress,
):
    client, _, token, source_id, profile_id = api_client
    job_id = _enqueue(client, token, source_id, profile_id).json()["id"]
    client.post(
        "/api/v1/internal/jobs/claim",
        json={"worker_id": "worker-a"},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/heartbeat",
        json={"worker_id": "worker-a", **progress},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 422
    unchanged = client.get(f"/api/v1/jobs/{job_id}")
    assert unchanged.json()["status"] == "running"
    assert unchanged.json()["progress_numerator"] == 0
    assert unchanged.json()["progress_denominator"] == 0


def test_public_retry_cancel_state_rules(api_client):
    client, _, token, source_id, profile_id = api_client
    job_id = _enqueue(client, token, source_id, profile_id).json()["id"]
    client.post(
        "/api/v1/internal/jobs/claim",
        json={"worker_id": "worker-a"},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    failed = client.post(
        f"/api/v1/internal/jobs/{job_id}/fail",
        json={
            "worker_id": "worker-a",
            "error_code": "failed",
            "error_message": "failed",
            "retry_eligible": True,
        },
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert failed.status_code == 200
    assert client.post(
        f"/api/v1/jobs/{job_id}/retry",
        headers={"X-CSRF-Token": token},
    ).status_code == 200
    cancelled = client.post(
        f"/api/v1/jobs/{job_id}/cancel",
        headers={"X-CSRF-Token": token},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert client.post(
        f"/api/v1/jobs/{job_id}/retry",
        headers={"X-CSRF-Token": token},
    ).status_code == 409


def test_concurrent_internal_claims_return_job_once(api_client):
    client, _, token, source_id, profile_id = api_client
    job_id = _enqueue(client, token, source_id, profile_id).json()["id"]
    responses = []

    def claim(worker_id: str) -> None:
        responses.append(
            client.post(
                "/api/v1/internal/jobs/claim",
                json={"worker_id": worker_id},
                headers={"X-Worker-Token": WORKER_TOKEN},
            )
        )

    threads = [Thread(target=claim, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert sorted(response.status_code for response in responses) == [200, 204]
    assert [response.json()["id"] for response in responses if response.status_code == 200] == [
        job_id
    ]


def test_complete_rejects_unsafe_artifact_symlink_without_state_change(
    api_client,
    tmp_path: Path,
):
    client, settings, token, source_id, profile_id = api_client
    job_id = _enqueue(client, token, source_id, profile_id).json()["id"]
    client.post(
        "/api/v1/internal/jobs/claim",
        json={"worker_id": "worker-a"},
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    link = settings.app_data_dir / "artifacts" / "linked.png"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")

    response = client.post(
        f"/api/v1/internal/jobs/{job_id}/complete",
        json={
            "worker_id": "worker-a",
            "profile_version": 4,
            "pipeline_version": "pipeline-v1",
            "artifact_paths": [str(link)],
        },
        headers={"X-Worker-Token": WORKER_TOKEN},
    )

    assert response.status_code == 422
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"


def test_startup_recovers_stale_job_after_database_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    _, _, job_id = _seed(settings, stale=True)
    order = []

    def checked(engine):
        del engine
        order.append("database")

    original = JobService.recover_stale

    def recovered(service, *, now=None):
        order.append("recovery")
        return original(service, now=now)

    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", checked)
    monkeypatch.setattr(JobService, "recover_stale", recovered)
    with TestClient(create_app(settings)) as client:
        login = client.post("/api/v1/session", json={"password": PASSWORD})
        response = client.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-CSRF-Token": login.headers["x-csrf-token"]},
        )

    assert order == ["database", "recovery"]
    assert response.json()["status"] == "queued"


def test_blocking_claim_database_work_does_not_block_event_loop(
    api_client,
    monkeypatch: pytest.MonkeyPatch,
):
    client, _, token, source_id, profile_id = api_client
    _enqueue(client, token, source_id, profile_id)
    entered = Event()
    release = Event()
    ticked = Event()
    responses = []
    original = JobService.claim

    def blocking_claim(service, worker_id, *, now=None):
        entered.set()
        release.wait(5)
        return original(service, worker_id, now=now)

    monkeypatch.setattr(JobService, "claim", blocking_claim)

    @client.app.get("/api/v1/job-write-tick")
    async def tick():
        ticked.set()
        return {"ok": True}

    request = Thread(
        target=lambda: responses.append(
            client.post(
                "/api/v1/internal/jobs/claim",
                json={"worker_id": "worker-a"},
                headers={"X-Worker-Token": WORKER_TOKEN},
            )
        )
    )
    request.start()
    assert entered.wait(2)
    tick_request = Thread(target=lambda: client.get("/api/v1/job-write-tick"))
    tick_request.start()
    try:
        assert ticked.wait(0.5)
    finally:
        release.set()
        request.join(5)
        tick_request.join(5)
    assert responses[0].status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"worker_id": ""},
        {"worker_id": "x" * 201},
        {"worker_id": "ok", "extra": True},
    ],
)
def test_internal_schemas_are_strict_and_bounded(api_client, payload):
    client, _, _, _, _ = api_client
    response = client.post(
        "/api/v1/internal/jobs/claim",
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/api/v1/internal/jobs/missing/heartbeat",
            {"worker_id": "   ", "numerator": 0, "denominator": 0},
        ),
        (
            "/api/v1/internal/jobs/missing/complete",
            {
                "worker_id": "   ",
                "profile_version": 1,
                "pipeline_version": "pipeline-v1",
            },
        ),
        (
            "/api/v1/internal/jobs/missing/fail",
            {
                "worker_id": "   ",
                "error_code": "failed",
                "error_message": "failed",
            },
        ),
        (
            "/api/v1/internal/jobs/missing/complete",
            {
                "worker_id": "worker-a",
                "profile_version": 1,
                "pipeline_version": "   ",
            },
        ),
    ],
)
def test_every_internal_identity_and_version_is_nonblank(
    api_client,
    endpoint,
    payload,
):
    client, _, _, _, _ = api_client
    response = client.post(
        endpoint,
        json=payload,
        headers={"X-Worker-Token": WORKER_TOKEN},
    )
    assert response.status_code == 422
