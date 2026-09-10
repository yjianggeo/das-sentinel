from __future__ import annotations

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
from fastapi.testclient import TestClient

PASSWORD = "correct horse"


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
        worker_token="w" * 40,
        secure_cookies=False,
        pipeline_version="v2",
    )


@pytest.fixture
def similar_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        repositories.admins.add(password_hash=PasswordHasher().hash(PASSWORD))
        site = repositories.sites.add(
            name="site-a",
            data_path=str(settings.data_root),
            timezone="UTC",
            cable_type="downhole",
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=ProcessingProfile(device="cpu").model_dump(mode="json"),
            is_current=True,
        )
        source = repositories.source_files.add(
            site_id=site.id,
            path="/data/a.h5",
            fingerprint="a" * 64,
            size_bytes=1,
            mtime_ns=1,
        )
        run = repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="pipeline-v1",
            requested_backend="cpu",
            actual_backend="cpu",
            status="succeeded",
        )
        event_ids = []
        for index in range(6):
            event = repositories.events.add(
                run_id=run.id,
                event_uid=f"event-{index}",
                start_sample=0,
                end_sample=10,
                channel_start=0,
                channel_stop=2,
                occurred_at=datetime(2026, 7, 30, index, tzinfo=UTC),
                features={"rms": float(index), "duration_s": float(index * 2)},
            )
            event_ids.append(event.id)
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    with TestClient(create_app(settings)) as client:
        client.post("/api/v1/session", json={"password": PASSWORD})
        yield client, event_ids
    engine.dispose()


def test_similar_events_returns_neighbours_without_self(similar_api):
    client, event_ids = similar_api

    response = client.get(f"/api/v1/events/{event_ids[2]}/similar")

    assert response.status_code == 200, response.text
    neighbours = response.json()
    assert 1 <= len(neighbours) <= 5
    assert all(item["id"] != event_ids[2] for item in neighbours)
    assert {item["id"] for item in neighbours} <= set(event_ids)


def test_similar_events_unknown_event_returns_404(similar_api):
    client, _ = similar_api
    assert client.get("/api/v1/events/missing/similar").status_code == 404


def test_similar_events_requires_admin(similar_api):
    client, event_ids = similar_api
    client.cookies.clear()
    assert client.get(f"/api/v1/events/{event_ids[0]}/similar").status_code == 401
