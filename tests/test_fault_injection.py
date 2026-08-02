"""Fault-injection acceptance: a corrupt file fails without blocking the next file.

Runs the real stack (API served by Uvicorn on a local port + persistence +
worker) against one corrupt HDF5 and one valid synthetic ZD HDF5, then
asserts per-file outcomes through the public jobs API.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.worker.main import HttpxWorkerClient, Worker

from tests.fixtures import write_zd_h5

PASSWORD = "correct horse"
WORKER_TOKEN = "worker-token-" + "x" * 32
API_PORT = 8631
API_BASE = f"http://127.0.0.1:{API_PORT}"


def _settings(tmp_path: Path) -> Settings:
    raw = tmp_path / "raw"
    app_data = tmp_path / "app-data"
    raw.mkdir()
    app_data.mkdir()
    return Settings(
        data_root=raw,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}",
        api_url=API_BASE,
        session_secret="s" * 32,
        worker_token=WORKER_TOKEN,
        secure_cookies=False,
        pipeline_version="pipeline-v1",
    )


@pytest.fixture
def running_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)

    corrupt = settings.data_root / "corrupt.h5"
    write_zd_h5(corrupt, fs=1000.0)
    valid = settings.data_root / "valid.h5"
    write_zd_h5(valid, fs=1000.0)

    with sessions.begin() as session:
        repositories = Repositories(session)
        repositories.admins.add(password_hash=PasswordHasher().hash(PASSWORD))
        site = repositories.sites.add(
            name="fault-site",
            data_path=str(settings.data_root),
            timezone="UTC",
            cable_type="downhole",
        )
        repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=ProcessingProfile(device="cpu").model_dump(mode="json"),
            is_current=True,
        )
        site_id = site.id

    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    app = create_app(settings)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=API_PORT, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30.0
    while not server.started:
        if time.monotonic() > deadline:
            thread.join(timeout=1)
            raise RuntimeError("uvicorn test server did not start")
        time.sleep(0.05)

    client = httpx.Client(base_url=API_BASE, timeout=30.0)
    try:
        login = client.post("/api/v1/session", json={"password": PASSWORD})
        csrf = login.headers["x-csrf-token"]

        clock = iter([100.0, 115.0])
        monkeypatch.setattr("das_sentinel.api.routes.scans.monotonic", lambda: next(clock))
        client.post(f"/api/v1/sites/{site_id}/scan", json={}, headers={"X-CSRF-Token": csrf})
        enqueued = client.post(
            f"/api/v1/sites/{site_id}/scan", json={}, headers={"X-CSRF-Token": csrf}
        )
        assert len(enqueued.json()) == 2

        # Fault injection: the file changes after being fingerprinted and
        # enqueued, so the worker must refuse it and mark the job failed
        # without blocking the remaining queue.
        corrupt.write_bytes(b"this is no longer the fingerprinted HDF5 file" * 16)

        worker_client = HttpxWorkerClient(
            settings.api_url,
            settings.worker_token,
        )
        worker = Worker(
            settings=settings,
            client=worker_client,
            worker_id="fault-worker",
            heartbeat_interval_seconds=1.0,
        )
        # Bounded drain: each file takes one claim, plus one idle poll.
        for _ in range(4):
            outcome = worker.run_once()
            if outcome.status == "idle":
                break
        worker_client.close()

        yield client, sessions, {"corrupt": corrupt, "valid": valid}
    finally:
        client.close()
        server.should_exit = True
        thread.join(timeout=10)
        engine.dispose()


def test_corrupt_file_fails_without_blocking_next_file(running_stack) -> None:
    client, sessions, files = running_stack

    jobs = client.get("/api/v1/jobs").json()
    assert len(jobs) == 2

    with sessions.begin() as session:
        repositories = Repositories(session)
        status_by_path = {
            repositories.source_files.get(job["source_file_id"]).path: job["status"]
            for job in jobs
        }

    assert status_by_path[str(files["corrupt"])] == "failed"
    assert status_by_path[str(files["valid"])] == "succeeded"

    failed = next(job for job in jobs if job["status"] == "failed")
    assert failed["error_code"]
    assert failed["error_message"]
