from __future__ import annotations

from pathlib import Path

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
def spa_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    with sessions.begin() as session:
        Repositories(session).admins.add(
            password_hash=PasswordHasher().hash(PASSWORD)
        )
    dist = tmp_path / "web"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    monkeypatch.setenv("DAS_WEB_DIST", str(dist))
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    with TestClient(create_app(settings)) as client:
        yield client
    engine.dispose()


def test_spa_fallback_serves_index_for_client_routes(spa_client):
    response = spa_client.get("/events")
    assert response.status_code == 200
    assert response.text == "<html>spa</html>"


def test_spa_serves_real_static_files(spa_client):
    response = spa_client.get("/assets/app.js")
    assert response.status_code == 200
    assert response.text == "console.log(1)"


def test_unknown_api_path_stays_json_404(spa_client):
    response = spa_client.get("/api/v1/definitely-not-a-route")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_path_traversal_does_not_escape_dist(spa_client):
    response = spa_client.get("/..%2F..%2Fsecret.txt")
    assert response.status_code == 200
    assert response.text == "<html>spa</html>"
