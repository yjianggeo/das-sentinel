from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from das_sentinel.api.app import create_app
from das_sentinel.config import Settings
from das_sentinel.persistence import upgrade_database


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = tmp_path / "data"
    app_data = tmp_path / "app-data"
    data.mkdir()
    app_data.mkdir()
    settings = Settings(
        data_root=data,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}",
        api_url="http://testserver",
        session_secret="s" * 32,
        worker_token="w" * 40,
        secure_cookies=False,
    )
    upgrade_database(settings.database_url)
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    monkeypatch.setattr("das_sentinel.api.routes.health.check_database_startup", lambda _: None)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_healthz_is_json_and_not_spa(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["content-type"].startswith("application/json")


def test_healthz_redacts_probe_failure(client, monkeypatch) -> None:
    def fail_probe(settings):
        raise OSError(f"cannot write {settings.session_secret}")

    monkeypatch.setattr("das_sentinel.api.routes.health.probe_storage", fail_probe)
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {"detail": "Service unavailable"}
    assert client.app.state.settings.session_secret not in response.text
