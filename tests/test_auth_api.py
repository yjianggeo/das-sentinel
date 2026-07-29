from __future__ import annotations

from collections.abc import Iterator
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
from das_sentinel.persistence.db import SQLiteRuntimeError
from fastapi.testclient import TestClient

PASSWORD = "correct horse"
PASSWORD_HASH = PasswordHasher().hash(PASSWORD)


def _settings(tmp_path: Path, *, secure_cookies: bool = False) -> Settings:
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
        worker_token="w" * 32,
        secure_cookies=secure_cookies,
    )


def _seed_admin(settings: Settings, *, disabled: bool = False) -> None:
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    try:
        sessions = session_factory(engine)
        with sessions.begin() as session:
            Repositories(session).admins.add(
                password_hash=PASSWORD_HASH,
                disabled=disabled,
            )
    finally:
        engine.dispose()


@pytest.fixture
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    settings = _settings(tmp_path)
    _seed_admin(settings)
    monkeypatch.setattr(
        "das_sentinel.api.app.check_database_startup",
        lambda engine: None,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_login_sets_strict_http_only_session_and_returns_csrf(client: TestClient):
    response = client.post("/api/v1/session", json={"password": PASSWORD})

    assert response.status_code == 204
    assert response.headers["x-csrf-token"]
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "secure" not in cookie


def test_secure_cookie_configuration_sets_secure_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path, secure_cookies=True)
    _seed_admin(settings)
    monkeypatch.setattr(
        "das_sentinel.api.app.check_database_startup",
        lambda engine: None,
    )

    with TestClient(create_app(settings)) as secure_client:
        response = secure_client.post(
            "/api/v1/session",
            json={"password": PASSWORD},
        )

    assert response.status_code == 204
    assert "secure" in response.headers["set-cookie"].lower()


def test_authenticated_session_recovers_same_csrf_token(client: TestClient):
    login = client.post("/api/v1/session", json={"password": PASSWORD})
    token = login.headers["x-csrf-token"]

    response = client.get("/api/v1/session")

    assert response.status_code == 200
    assert response.json() == {"admin_id": 1, "csrf_token": token}


@pytest.mark.parametrize("password", ["wrong password", ""])
def test_bad_password_returns_unauthorized_and_clears_old_session(
    client: TestClient,
    password: str,
):
    assert client.post(
        "/api/v1/session",
        json={"password": PASSWORD},
    ).status_code == 204

    response = client.post("/api/v1/session", json={"password": password})

    assert response.status_code == 401
    assert client.get("/api/v1/session").status_code == 401


def test_missing_and_disabled_admin_use_the_same_unauthorized_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "das_sentinel.api.app.check_database_startup",
        lambda engine: None,
    )
    settings_missing = _settings(tmp_path / "missing")
    upgrade_database(settings_missing.database_url)
    with TestClient(create_app(settings_missing)) as missing_client:
        missing = missing_client.post(
            "/api/v1/session",
            json={"password": PASSWORD},
        )

    settings_disabled = _settings(tmp_path / "disabled")
    _seed_admin(settings_disabled, disabled=True)
    with TestClient(create_app(settings_disabled)) as disabled_client:
        disabled = disabled_client.post(
            "/api/v1/session",
            json={"password": PASSWORD},
        )

    assert missing.status_code == disabled.status_code == 401
    assert missing.json() == disabled.json() == {"detail": "Unauthorized"}


def test_login_replaces_untrusted_session_cookie(client: TestClient):
    client.cookies.set(
        "session",
        "attacker-fixed-value",
        domain="testserver.local",
        path="/",
    )

    response = client.post("/api/v1/session", json={"password": PASSWORD})

    assert response.status_code == 204
    assert client.cookies.get("session") != "attacker-fixed-value"
    assert response.headers["x-csrf-token"]


def test_tampered_signed_cookie_is_unauthorized(client: TestClient):
    client.post("/api/v1/session", json={"password": PASSWORD})
    cookie = client.cookies.get("session")
    assert cookie is not None
    client.cookies.clear()
    client.cookies.set("session", f"{cookie[:-1]}x")

    response = client.get("/api/v1/session")

    assert response.status_code == 401


def test_disabled_admin_invalidates_existing_session(
    client: TestClient,
    tmp_path: Path,
):
    login = client.post("/api/v1/session", json={"password": PASSWORD})
    assert login.status_code == 204
    settings = _settings_for_existing_tree(tmp_path)
    engine = create_local_engine(settings.database_url)
    try:
        sessions = session_factory(engine)
        with sessions.begin() as session:
            admin = Repositories(session).admins.get()
            assert admin is not None
            admin.disabled = True
            session.flush()
    finally:
        engine.dispose()

    response = client.get("/api/v1/session")

    assert response.status_code == 401


@pytest.mark.parametrize(
    "csrf_header",
    [
        None,
        "wrong",
        "TOKEN-WITH-DIFFERENT-CASE",
    ],
)
def test_logout_requires_exact_csrf(
    client: TestClient,
    csrf_header: str | None,
):
    login = client.post("/api/v1/session", json={"password": PASSWORD})
    token = login.headers["x-csrf-token"]
    headers = (
        {}
        if csrf_header is None
        else {"X-CSRF-Token": csrf_header if csrf_header != "wrong" else token + "x"}
    )

    response = client.delete("/api/v1/session", headers=headers)

    assert response.status_code == 403
    assert client.get("/api/v1/session").status_code == 200


def test_logout_clears_session(client: TestClient):
    login = client.post("/api/v1/session", json={"password": PASSWORD})

    response = client.delete(
        "/api/v1/session",
        headers={"X-CSRF-Token": login.headers["x-csrf-token"]},
    )

    assert response.status_code == 204
    assert client.get("/api/v1/session").status_code == 401


def test_credentials_and_hash_never_appear_in_session_response_or_cookie(
    client: TestClient,
):
    response = client.post("/api/v1/session", json={"password": PASSWORD})

    exposed = (
        response.text
        + response.headers.get("set-cookie", "")
        + response.headers.get("x-csrf-token", "")
    )
    assert PASSWORD not in exposed
    assert PASSWORD_HASH not in exposed
    assert "password_hash" not in exposed


def test_lifespan_checks_database_and_disposes_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    _seed_admin(settings)
    checked = []

    def record_check(engine):
        checked.append(engine)

    monkeypatch.setattr(
        "das_sentinel.api.app.check_database_startup",
        record_check,
    )
    app = create_app(settings)
    engine = app.state.engine
    original_pool = engine.pool
    with TestClient(app):
        assert checked == [engine]

    assert engine.pool is not original_pool


def test_failed_database_startup_check_disposes_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    settings = _settings(tmp_path)
    _seed_admin(settings)

    def reject_database(engine):
        raise SQLiteRuntimeError("unsafe SQLite runtime")

    monkeypatch.setattr(
        "das_sentinel.api.app.check_database_startup",
        reject_database,
    )
    app = create_app(settings)
    engine = app.state.engine
    original_pool = engine.pool

    with (
        pytest.raises(SQLiteRuntimeError, match="unsafe SQLite runtime"),
        TestClient(app),
    ):
        pass

    assert engine.pool is not original_pool


def _settings_for_existing_tree(tmp_path: Path) -> Settings:
    app_data = tmp_path / "app-data"
    return Settings(
        data_root=tmp_path / "raw",
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}",
        api_url="http://testserver",
        session_secret="s" * 32,
        worker_token="w" * 32,
        secure_cookies=False,
    )
