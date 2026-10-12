from __future__ import annotations

import os
import time
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
from das_sentinel.persistence.models import SourceFile
from fastapi.testclient import TestClient
from sqlalchemy import select

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
        pipeline_version="v2",
    )


@pytest.fixture
def scan_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    selected = settings.data_root / "jjk" / "day"
    selected.mkdir(parents=True)
    (settings.data_root / "empty").mkdir()
    write_zd_h5(selected / "a.h5")
    write_zd_h5(selected / "b.h5")
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
    app = create_app(settings)
    app.state.scan_coordinator.stable_seconds = 0.01
    with TestClient(app) as client:
        login = client.post("/api/v1/session", json={"password": PASSWORD})
        yield client, login.headers["x-csrf-token"], ids, settings
    engine.dispose()


def poll_scan(client: TestClient, scan_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/scans/{scan_id}")
        assert response.status_code == 200, response.text
        snapshot = response.json()
        if snapshot["phase"] in {"completed", "failed"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError(f"scan {scan_id} did not reach a terminal phase")


def test_scan_directory_listing_is_shallow_and_root_relative(scan_api):
    client, _, ids, _ = scan_api

    root = client.get(f"/api/v1/sites/{ids['site']}/scan-directories")
    assert root.status_code == 200, root.text
    assert root.json() == {
        "site_id": ids["site"],
        "site_name": "site-a",
        "path": "",
        "parent_path": None,
        "directories": [
            {"name": "empty", "path": "empty"},
            {"name": "jjk", "path": "jjk"},
        ],
    }

    nested = client.get(
        f"/api/v1/sites/{ids['site']}/scan-directories", params={"path": "jjk"}
    )
    assert nested.status_code == 200, nested.text
    assert nested.json() == {
        "site_id": ids["site"],
        "site_name": "site-a",
        "path": "jjk",
        "parent_path": "",
        "directories": [{"name": "day", "path": "jjk/day"}],
    }


@pytest.mark.parametrize("path", ["", "   ", "\t\n", "."])
def test_scan_rejects_blank_or_site_root(scan_api, path):
    client, token, ids, _ = scan_api
    response = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": path},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "path",
    ["../outside", "/absolute", "missing", "jjk/day/a.h5"],
)
def test_scan_rejects_unsafe_or_invalid_directory(scan_api, path):
    client, token, ids, settings = scan_api
    response = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": path},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 422
    assert str(settings.data_root) not in response.text


@pytest.mark.skipif(os.name != "nt", reason="requires Windows drive semantics")
@pytest.mark.parametrize(
    "path_template",
    ["{drive}", "{drive}/jjk/day", "//server/share"],
    ids=["drive-only", "drive-absolute", "unc"],
)
def test_scan_rejects_windows_anchored_directory(
    scan_api,
    path_template,
):
    client, token, ids, settings = scan_api
    submitted = path_template.format(drive=settings.data_root.drive)
    response = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": submitted},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 422
    assert str(settings.data_root) not in response.text


@pytest.mark.skipif(os.name != "nt", reason="requires Windows path normalization")
@pytest.mark.parametrize("path", ["...", "....", ".. .", ". /."])
def test_scan_rejects_windows_normalized_root_before_start(scan_api, path):
    client, token, ids, settings = scan_api
    response = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": path},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 422
    assert str(settings.data_root) not in response.text
    coordinator = client.app.state.scan_coordinator
    assert coordinator._snapshots == {}
    assert coordinator._tasks_by_site == {}
    assert client.get(f"/api/v1/sites/{ids['site']}/active-scan").json() is None
    assert client.get("/api/v1/jobs").json() == []
    with client.app.state.session_factory() as session:
        assert list(session.scalars(select(SourceFile))) == []


@pytest.mark.parametrize("selected_name", [" archive", "archive"])
def test_scan_preserves_browsed_leading_space_path(scan_api, selected_name):
    client, token, ids, settings = scan_api
    for name in (" archive", "archive"):
        directory = settings.data_root / name / "day"
        directory.mkdir(parents=True)
        write_zd_h5(directory / "selected.h5")
    listing_url = f"/api/v1/sites/{ids['site']}/scan-directories"
    root = client.get(listing_url)
    assert root.status_code == 200
    selected = next(
        item for item in root.json()["directories"] if item["name"] == selected_name
    )
    nested = client.get(listing_url, params={"path": selected["path"]})
    assert nested.status_code == 200
    browsed_path = nested.json()["directories"][0]["path"]
    assert browsed_path == f"{selected_name}/day"

    started = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": browsed_path},
        headers={"X-CSRF-Token": token},
    )
    assert started.status_code == 202
    terminal = poll_scan(client, started.json()["id"])
    assert terminal["phase"] == "completed"
    assert terminal["submitted_files"] == 1
    with client.app.state.session_factory() as session:
        assert list(session.scalars(select(SourceFile.path))) == [
            str(settings.data_root / selected_name / "day" / "selected.h5")
        ]
    assert started.json()["path"] == browsed_path
    assert terminal["path"] == browsed_path
    assert len(client.get("/api/v1/jobs").json()) == 0
    assert client.get("/api/v1/window-jobs").json()["total"] == 1


def test_scan_directory_error_does_not_expose_absolute_path(scan_api):
    client, _, ids, settings = scan_api
    response = client.get(
        f"/api/v1/sites/{ids['site']}/scan-directories",
        params={"path": "missing"},
    )
    assert response.status_code == 422
    assert str(settings.data_root) not in response.text


def test_scan_returns_immediately_then_completes_automatic_recheck(scan_api):
    client, token, ids, _ = scan_api
    started = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": "jjk/day"},
        headers={"X-CSRF-Token": token},
    )
    assert started.status_code == 202
    scan_id = started.json()["id"]
    terminal = poll_scan(client, scan_id)
    assert terminal["phase"] == "completed"
    assert terminal["initial_found_files"] == 2
    assert terminal["verification_found_files"] == 2
    assert terminal["submitted_files"] == 2
    assert len(client.get("/api/v1/jobs").json()) == 0
    # Both fixture files occupy the same UTC minute, so they share one window.
    assert client.get("/api/v1/window-jobs").json()["total"] == 1


def test_duplicate_start_returns_existing_active_scan(scan_api):
    client, token, ids, _ = scan_api
    client.app.state.scan_coordinator.stable_seconds = 1
    first = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": "jjk/day"},
        headers={"X-CSRF-Token": token},
    )
    second = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": "empty"},
        headers={"X-CSRF-Token": token},
    )
    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["path"] == "jjk/day"


def test_active_scan_lookup_returns_current_scan(scan_api):
    client, token, ids, _ = scan_api
    client.app.state.scan_coordinator.stable_seconds = 1
    assert client.get(f"/api/v1/sites/{ids['site']}/active-scan").json() is None
    started = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": "jjk/day"},
        headers={"X-CSRF-Token": token},
    )
    active = client.get(f"/api/v1/sites/{ids['site']}/active-scan")
    assert active.status_code == 200
    assert active.json()["id"] == started.json()["id"]


def test_unknown_scan_returns_404(scan_api):
    client, _, _, _ = scan_api
    response = client.get("/api/v1/scans/does-not-exist")
    assert response.status_code == 404


def test_scan_unknown_or_disabled_site(scan_api):
    client, token, ids, _ = scan_api
    missing = client.post(
        "/api/v1/sites/does-not-exist/scan",
        json={"path": "jjk/day"},
        headers={"X-CSRF-Token": token},
    )
    assert missing.status_code == 404
    disabled = client.post(
        f"/api/v1/sites/{ids['disabled']}/scan",
        json={"path": "jjk/day"},
        headers={"X-CSRF-Token": token},
    )
    assert disabled.status_code == 409


def test_scan_directory_unknown_or_disabled_site(scan_api):
    client, _, ids, _ = scan_api
    missing = client.get("/api/v1/sites/does-not-exist/scan-directories")
    assert missing.status_code == 404
    disabled = client.get(f"/api/v1/sites/{ids['disabled']}/scan-directories")
    assert disabled.status_code == 409


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/sites/site-id/scan-directories",
        "/api/v1/sites/site-id/active-scan",
        "/api/v1/scans/scan-id",
    ],
)
def test_scan_gets_require_admin(scan_api, path):
    client, _, _, _ = scan_api
    client.cookies.clear()
    assert client.get(path).status_code == 401


def test_scan_post_requires_admin_and_csrf(scan_api):
    client, _, ids, _ = scan_api
    client.cookies.clear()
    path = f"/api/v1/sites/{ids['site']}/scan"
    assert client.post(path, json={"path": "jjk/day"}).status_code == 401
    client.post("/api/v1/session", json={"password": PASSWORD})
    assert client.post(path, json={"path": "jjk/day"}).status_code == 403


def test_empty_directory_scan_completes_successfully(scan_api):
    client, token, ids, _ = scan_api
    started = client.post(
        f"/api/v1/sites/{ids['site']}/scan",
        json={"path": "empty"},
        headers={"X-CSRF-Token": token},
    )
    assert started.status_code == 202
    terminal = poll_scan(client, started.json()["id"])
    assert terminal["phase"] == "completed"
    assert terminal["initial_found_files"] == 0
    assert terminal["verification_found_files"] == 0
    assert terminal["submitted_files"] == 0
    assert terminal["error_message"] is None
