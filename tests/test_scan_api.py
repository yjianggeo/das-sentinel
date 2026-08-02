from __future__ import annotations

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

from tests.fixtures import write_zd_h5

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
def scan_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    write_zd_h5(settings.data_root / "a.h5")
    write_zd_h5(settings.data_root / "b.h5")
    with sessions.begin() as session:
        repositories = Repositories(session)
        repositories.admins.add(password_hash=PasswordHasher().hash(PASSWORD))
        site = repositories.sites.add(
            name="site-a",
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
        disabled = repositories.sites.add(
            name="site-disabled",
            data_path=str(settings.data_root),
            timezone="UTC",
            cable_type="downhole",
            enabled=False,
        )
        repositories.profiles.add(
            site_id=disabled.id,
            version=1,
            payload=ProcessingProfile(device="cpu").model_dump(mode="json"),
            is_current=True,
        )
        ids = {"site": site.id, "disabled": disabled.id}
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    with TestClient(create_app(settings)) as client:
        login = client.post("/api/v1/session", json={"password": PASSWORD})
        yield client, login.headers["x-csrf-token"], ids, settings
    engine.dispose()


def test_scan_enqueues_only_after_stability_window(scan_api, monkeypatch):
    client, token, ids, _ = scan_api
    clock = iter([100.0, 115.0, 130.0])
    monkeypatch.setattr(
        "das_sentinel.api.routes.scans.monotonic", lambda: next(clock)
    )

    first = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={},
        headers={"X-CSRF-Token": token},
    )
    assert first.status_code == 201, first.text
    assert first.json() == []

    second = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={},
        headers={"X-CSRF-Token": token},
    )
    assert second.status_code == 201, second.text
    assert len(second.json()) == 2
    assert {job["pipeline_version"] for job in second.json()} == {"pipeline-v1"}

    listed = client.get("/api/v1/jobs").json()
    assert len(listed) == 2

    again = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={},
        headers={"X-CSRF-Token": token},
    )
    assert again.json() == []


def test_scan_rejects_path_outside_site_root(scan_api):
    client, token, ids, _ = scan_api
    response = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": "/etc"},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 422


def test_scan_unknown_or_disabled_site(scan_api):
    client, token, ids, _ = scan_api
    missing = client.post(
        "/api/v1/sites/does-not-exist/scan",
        json={},
        headers={"X-CSRF-Token": token},
    )
    assert missing.status_code == 404
    disabled = client.post(
        f"/api/v1/sites/{ids['disabled']}/scan",
        json={},
        headers={"X-CSRF-Token": token},
    )
    assert disabled.status_code == 409


def test_scan_requires_admin_and_csrf(scan_api):
    client, _, ids, _ = scan_api
    client.cookies.clear()
    assert (
        client.post(f"/api/v1/sites/{ids['site']}/scan", json={}).status_code == 401
    )
    client.post("/api/v1/session", json={"password": PASSWORD})
    assert (
        client.post(f"/api/v1/sites/{ids['site']}/scan", json={}).status_code == 403
    )
