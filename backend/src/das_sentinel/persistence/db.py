from __future__ import annotations

import asyncio
import importlib
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

MINIMUM_SAFE_SQLITE = (3, 51, 3)
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)\Z")


class SQLiteRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatabaseStartupReport:
    sqlite_version: tuple[int, int, int]
    journal_mode: str
    foreign_keys: bool
    busy_timeout_ms: int


class AsyncWriteCoordinator:
    """Application-scoped lock injected into the single API writer."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()


def validate_local_sqlite_url(database_url: str) -> Path:
    try:
        url = make_url(database_url)
    except Exception as error:
        raise ValueError("invalid database URL") from error
    if url.drivername not in {"sqlite", "sqlite+pysqlite"}:
        raise ValueError("database URL must use local SQLite")
    if url.host or url.username or url.password or url.port:
        raise ValueError("SQLite database URL must not contain an authority")
    if url.query:
        raise ValueError("SQLite database URL query parameters are not allowed")
    if not url.database or url.database == ":memory:":
        raise ValueError("SQLite database must be a local file")

    decoded = unquote(url.database)
    lowered = decoded.lower()
    if lowered.startswith(("file:", "smb:", "nfs:")):
        raise ValueError("remote or URI SQLite paths are not allowed")
    if decoded.startswith(("\\\\", "//")):
        raise ValueError("UNC/network SQLite paths are not allowed")

    windows_path = PureWindowsPath(decoded)
    native_path = Path(decoded)
    if windows_path.drive.startswith("\\\\"):
        raise ValueError("UNC/network SQLite paths are not allowed")
    if not (native_path.is_absolute() or windows_path.is_absolute()):
        raise ValueError("SQLite database path must be absolute")
    return native_path


def create_local_engine(database_url: str) -> Engine:
    validate_local_sqlite_url(database_url)
    engine = create_engine(database_url, module=_sqlite_driver())

    @event.listens_for(engine, "connect")
    def _configure_sqlite(dbapi_connection, connection_record) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.fetchone()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()

    return engine


def _sqlite_driver():
    try:
        return importlib.import_module("pysqlite3.dbapi2")
    except ModuleNotFoundError as error:
        if error.name not in {"pysqlite3", "pysqlite3.dbapi2"}:
            raise
        return sqlite3


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def validate_sqlite_runtime(
    version: str,
    *,
    patched_versions: Iterable[str] = (),
) -> tuple[int, int, int]:
    match = _VERSION.fullmatch(version)
    if match is None:
        raise SQLiteRuntimeError(f"unrecognized SQLite version: {version}")
    parsed = tuple(int(part) for part in match.groups())
    if parsed < MINIMUM_SAFE_SQLITE and version not in set(patched_versions):
        raise SQLiteRuntimeError(
            "SQLite 3.51.3+ or an explicitly approved WAL-reset patched "
            f"release is required; found {version}"
        )
    return parsed


def check_database_startup(
    engine: Engine,
    *,
    patched_versions: Iterable[str] = (),
) -> DatabaseStartupReport:
    with engine.connect() as connection:
        version_text = connection.exec_driver_sql(
            "SELECT sqlite_version()"
        ).scalar_one()
        parsed = validate_sqlite_runtime(
            version_text, patched_versions=patched_versions
        )
        journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
        foreign_keys = bool(
            connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()
        )
        busy_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()

    if journal_mode.lower() != "wal":
        raise SQLiteRuntimeError("SQLite journal_mode must be WAL")
    if not foreign_keys:
        raise SQLiteRuntimeError("SQLite foreign_keys must be ON")
    if busy_timeout != 5000:
        raise SQLiteRuntimeError("SQLite busy_timeout must be 5000 ms")
    return DatabaseStartupReport(
        sqlite_version=parsed,
        journal_mode=journal_mode.lower(),
        foreign_keys=foreign_keys,
        busy_timeout_ms=busy_timeout,
    )


def upgrade_database(database_url: str) -> None:
    validate_local_sqlite_url(database_url)
    repository_root = Path(__file__).resolve().parents[4]
    configuration = Config(str(repository_root / "alembic.ini"))
    configuration.set_main_option(
        "script_location",
        str(
            repository_root
            / "backend"
            / "src"
            / "das_sentinel"
            / "persistence"
            / "migrations"
        ),
    )
    configuration.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(configuration, "head")

    engine = create_local_engine(database_url)
    try:
        with engine.connect():
            pass
    finally:
        engine.dispose()
