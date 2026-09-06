from pathlib import Path
from types import SimpleNamespace

import pytest
from das_sentinel import runtime
from das_sentinel.api.app import create_runtime_app
from das_sentinel.config import Settings
from das_sentinel.persistence.db import SQLiteRuntimeError
from das_sentinel.runtime import RuntimeProbeError, probe_storage
from fastapi import FastAPI
from pydantic import ValidationError


def _runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = tmp_path / "data"
    app_data = tmp_path / "app-data"
    data.mkdir()
    app_data.mkdir()
    monkeypatch.setenv("DAS_DATA_ROOT", str(data))
    monkeypatch.setenv("DAS_APP_DATA_DIR", str(app_data))
    monkeypatch.setenv(
        "DAS_DATABASE_URL", f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}"
    )
    monkeypatch.setenv("DAS_SESSION_SECRET", "s" * 48)
    monkeypatch.setenv("DAS_WORKER_TOKEN", "w" * 48)
    monkeypatch.setenv("DAS_SECURE_COOKIES", "false")


def test_runtime_factory_builds_app_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime_env(monkeypatch, tmp_path)
    app = create_runtime_app()
    assert isinstance(app, FastAPI)
    assert app.state.settings.data_root == tmp_path / "data"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_secret", "change-me-session-secret-at-least-32-chars"),
        ("worker_token", "change-me-worker-token-at-least-32-chars"),
    ],
)
def test_settings_rejects_example_or_repeated_secrets(
    field: str, value: str, tmp_path: Path
) -> None:
    values = {
        "data_root": tmp_path,
        "app_data_dir": tmp_path,
        "database_url": f"sqlite+pysqlite:///{tmp_path / 'db.sqlite3'}",
        "session_secret": "s" * 48,
        "worker_token": "w" * 48,
        field: value,
    }
    with pytest.raises(ValidationError, match="insecure example value"):
        Settings(**values)


def test_storage_probe_normalizes_app_data_failures(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    app_data = tmp_path / "app-data"
    app_data.write_text("not a directory", encoding="utf-8")
    settings = Settings(
        data_root=data_root,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{tmp_path / 'db.sqlite3'}",
        session_secret="s" * 48,
        worker_token="w" * 48,
    )

    with pytest.raises(RuntimeProbeError) as error:
        probe_storage(settings)

    assert settings.session_secret not in str(error.value)


def test_entrypoint_preflights_before_migration() -> None:
    text = Path("docker/entrypoint-api.sh").read_text(encoding="utf-8")
    assert text.index("python -m das_sentinel.runtime") < text.index("alembic upgrade head")
    assert "--factory" in text
    assert "create_runtime_app" in text


def test_entrypoint_defaults_database_url_before_preflight_and_migration() -> None:
    text = Path("docker/entrypoint-api.sh").read_text(encoding="utf-8")
    export = (
        'export DAS_DATABASE_URL="${DAS_DATABASE_URL:-'
        'sqlite+pysqlite:////app-data/das-sentinel.sqlite3}"'
    )
    assert export in text
    assert text.index(export) < text.index("python -m das_sentinel.runtime")
    assert text.index(export) < text.index("alembic upgrade head")


def test_runtime_preflight_loads_settings_and_validates_selected_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "das_sentinel.persistence.db._sqlite_driver",
        lambda: SimpleNamespace(sqlite_version="3.53.3"),
    )

    settings = runtime.preflight_from_environment()

    assert settings.data_root == tmp_path / "data"


def test_runtime_preflight_rejects_unsafe_selected_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runtime_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "das_sentinel.persistence.db._sqlite_driver",
        lambda: SimpleNamespace(sqlite_version="3.51.1"),
    )

    with pytest.raises(SQLiteRuntimeError, match="3.51.3"):
        runtime.preflight_from_environment()
