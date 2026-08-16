from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

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
from das_sentinel.persistence.models import Event
from fastapi.testclient import TestClient


@pytest.fixture
def dashboard_api(tmp_path, monkeypatch):
    data = tmp_path / "data"
    app_data = tmp_path / "app-data"
    data.mkdir()
    app_data.mkdir()
    settings = Settings(
        data_root=data,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'dashboard.sqlite3'}",
        session_secret="s" * 32,
        worker_token="w" * 40,
        secure_cookies=False,
    )
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    with sessions.begin() as session:
        Repositories(session).admins.add(password_hash=PasswordHasher().hash("correct horse"))
    now_utc = datetime(2026, 8, 6, 0, 30, tzinfo=UTC)
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    monkeypatch.setattr("das_sentinel.api.routes.dashboard.utc_now", lambda: now_utc)
    with TestClient(create_app(settings)) as client:
        client.post("/api/v1/session", json={"password": "correct horse"})
        yield client, sessions, now_utc
    engine.dispose()


def _seed_dashboard_site(
    sessions,
    *,
    name: str,
    timezone: str,
    occurred_at: datetime,
    label: str,
    count: int = 1,
    enabled: bool = True,
    statuses: tuple[str, ...] = (),
) -> str:
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name=name,
            data_path=f"/data/{name}",
            timezone=timezone,
            cable_type="urban",
            enabled=enabled,
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=ProcessingProfile().model_dump(mode="json"),
            is_current=True,
        )
        source = repositories.source_files.add(
            site_id=site.id,
            path=f"/data/{name}/source.h5",
            fingerprint=hashlib.sha256(name.encode()).hexdigest(),
            size_bytes=1,
            mtime_ns=1,
        )
        for status in statuses:
            repositories.jobs.add(
                source_file_id=source.id,
                processing_profile_id=profile.id,
                pipeline_version=f"v-{status}-{len(statuses)}",
                status=status,
            )
        run = repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="v1",
            requested_backend="cpu",
            actual_backend="cpu",
            status="succeeded",
        )
        for index in range(count):
            event = repositories.events.add(
                run_id=run.id,
                event_uid=f"{name}-{index}",
                start_sample=index * 10,
                end_sample=index * 10 + 5,
                channel_start=0,
                channel_stop=2,
                occurred_at=occurred_at + timedelta(seconds=index),
            )
            repositories.events.add_classification(
                event.id,
                label=label,
                decision_score=0.9,
                method="rules",
            )
        return event.id


def test_dashboard_requires_an_authenticated_session(dashboard_api):
    client, _, _ = dashboard_api
    client.cookies.clear()

    response = client.get("/api/v1/dashboard")

    assert response.status_code == 401


def test_dashboard_uses_site_local_half_open_today_and_all_latest_labels(dashboard_api):
    client, sessions, now_utc = dashboard_api
    _seed_dashboard_site(
        sessions,
        name="shanghai",
        timezone="Asia/Shanghai",
        occurred_at=now_utc,
        label="earthquake",
    )
    _seed_dashboard_site(
        sessions,
        name="los-angeles",
        timezone="America/Los_Angeles",
        occurred_at=now_utc,
        label="noise",
    )
    history_event_id = _seed_dashboard_site(
        sessions,
        name="history",
        timezone="UTC",
        occurred_at=now_utc - timedelta(days=2),
        label="traffic",
        count=101,
    )
    _seed_dashboard_site(
        sessions,
        name="disabled",
        timezone="Asia/Shanghai",
        occurred_at=now_utc,
        label="blast",
        enabled=False,
    )
    _seed_dashboard_site(
        sessions,
        name="shanghai-start",
        timezone="Asia/Shanghai",
        occurred_at=datetime(2026, 8, 5, 16, tzinfo=UTC),
        label="blast",
    )
    _seed_dashboard_site(
        sessions,
        name="shanghai-end",
        timezone="Asia/Shanghai",
        occurred_at=datetime(2026, 8, 6, 16, tzinfo=UTC),
        label="blast",
    )
    with sessions.begin() as session:
        repositories = Repositories(session)
        assert session.get(Event, history_event_id) is not None
        repositories.events.add_classification(
            history_event_id,
            label="earthquake",
            decision_score=0.95,
            method="rules",
        )

    response = client.get("/api/v1/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["today_event_count"] == 3
    assert payload["machine_label_counts"] == {
        "earthquake": 2,
        "noise": 1,
        "traffic": 100,
        "blast": 3,
    }


def test_dashboard_groups_all_job_statuses(dashboard_api):
    client, sessions, now_utc = dashboard_api
    _seed_dashboard_site(
        sessions,
        name="jobs",
        timezone="UTC",
        occurred_at=now_utc,
        label="unknown",
        statuses=("queued", "queued", "running", "succeeded", "failed", "cancelled"),
    )

    response = client.get("/api/v1/dashboard")

    assert response.status_code == 200
    assert response.json()["job_counts"] == {
        "queued": 2,
        "running": 1,
        "succeeded": 1,
        "failed": 1,
        "cancelled": 1,
    }
