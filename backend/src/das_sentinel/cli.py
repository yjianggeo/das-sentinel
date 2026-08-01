from __future__ import annotations

import csv
import os
import sqlite3
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

app = typer.Typer(no_args_is_help=True, help="DAS Sentinel administration CLI.")
admin_app = typer.Typer(no_args_is_help=True)
site_app = typer.Typer(no_args_is_help=True)
jobs_app = typer.Typer(no_args_is_help=True)
events_app = typer.Typer(no_args_is_help=True)
models_app = typer.Typer(no_args_is_help=True)
app.add_typer(admin_app, name="admin")
app.add_typer(site_app, name="site")
app.add_typer(jobs_app, name="jobs")
app.add_typer(events_app, name="events")
app.add_typer(models_app, name="models")

_MUTATING = {"POST", "PATCH", "PUT", "DELETE"}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _app_data_dir() -> Path:
    return Path(_env("DAS_APP_DATA_DIR", "/app-data"))


def _database_url() -> str:
    return _env(
        "DAS_DATABASE_URL",
        f"sqlite+pysqlite:///{_app_data_dir() / 'das-sentinel.sqlite3'}",
    )


class ApiClient:
    """Same-origin REST client with session login and CSRF handling."""

    def __init__(self, base_url: str, password: str) -> None:
        import httpx

        self._client = httpx.Client(base_url=base_url, timeout=60.0)
        login = self._client.post("/api/v1/session", json={"password": password})
        if login.status_code != 204:
            self._client.close()
            typer.echo(f"API login failed (HTTP {login.status_code})", err=True)
            raise typer.Exit(code=1)
        self._csrf = login.headers.get("x-csrf-token", "")

    def request(self, method: str, path: str, **kwargs: Any):
        method = method.upper()
        if method in _MUTATING:
            headers = dict(kwargs.pop("headers", {}) or {})
            headers["X-CSRF-Token"] = self._csrf
            kwargs["headers"] = headers
        response = self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except ValueError:
                detail = response.text[:200]
            typer.echo(
                f"{method} {path} failed (HTTP {response.status_code}): {detail}",
                err=True,
            )
            raise typer.Exit(code=1)
        return response

    def get(self, path: str, **kwargs: Any):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any):
        return self.request("POST", path, **kwargs)


def _connect() -> ApiClient:
    password = os.environ.get("DAS_ADMIN_PASSWORD")
    if not password:
        password = typer.prompt("管理员密码", hide_input=True)
    return ApiClient(_env("DAS_API_URL", "http://127.0.0.1:8000"), password)


@admin_app.command("init")
def admin_init(
    password: Annotated[
        str,
        typer.Option(prompt=True, hide_input=True, confirmation_prompt=True),
    ],
) -> None:
    """Create the single administrator account (bootstrap, local DB write)."""
    from argon2 import PasswordHasher

    from das_sentinel.persistence import (
        Repositories,
        create_local_engine,
        session_factory,
        upgrade_database,
    )

    if len(password) < 8:
        typer.echo("password must contain at least 8 characters", err=True)
        raise typer.Exit(code=1)
    _app_data_dir().mkdir(parents=True, exist_ok=True)
    database_url = _database_url()
    upgrade_database(database_url)
    engine = create_local_engine(database_url)
    try:
        sessions = session_factory(engine)
        with sessions.begin() as session:
            repositories = Repositories(session)
            if repositories.admins.get(1) is not None:
                typer.echo("admin already exists; use the web session to manage it",
                           err=True)
                raise typer.Exit(code=1)
            repositories.admins.add(password_hash=PasswordHasher().hash(password))
        typer.echo("Admin initialized.")
    finally:
        engine.dispose()


@site_app.command("add")
def site_add(
    name: str,
    data_path: str,
    timezone: str = "UTC",
    cable_type: str = "downhole",
) -> None:
    client = _connect()
    response = client.post(
        "/api/v1/sites",
        json={
            "name": name,
            "data_path": data_path,
            "timezone": timezone,
            "cable_type": cable_type,
        },
    )
    typer.echo(response.json())


@site_app.command("list")
def site_list() -> None:
    client = _connect()
    for site in client.get("/api/v1/sites").json():
        typer.echo(
            f"{site['id']}\t{site['name']}\t{site['data_path']}\t"
            f"enabled={site['enabled']}"
        )


@site_app.command("show")
def site_show(site_id: str) -> None:
    client = _connect()
    typer.echo(client.get(f"/api/v1/sites/{site_id}").json())


@app.command("scan")
def scan(
    site_id: str,
    path: Annotated[str | None, typer.Argument()] = None,
    settle_seconds: Annotated[float, typer.Option(min=0.0)] = 10.0,
) -> None:
    """Discover files under PATH (default: site root) and enqueue jobs."""
    client = _connect()
    payload = {"path": path} if path is not None else {}
    first = client.post(f"/api/v1/sites/{site_id}/scan", json=payload).json()
    if settle_seconds > 0:
        typer.echo(f"等待文件稳定 {settle_seconds:.0f} 秒…")
        time.sleep(settle_seconds)
    jobs = client.post(f"/api/v1/sites/{site_id}/scan", json=payload).json()
    typer.echo(f"本次入队 {len(jobs)} 个任务（首轮 {len(first)} 个）。")


@app.command("watch")
def watch(
    site_id: str,
    poll_seconds: Annotated[float, typer.Option(min=1.0)] = 30.0,
    once: Annotated[bool, typer.Option("--once")] = False,
) -> None:
    """Poll the site directory through the API scan endpoint."""
    client = _connect()
    try:
        while True:
            jobs = client.post(f"/api/v1/sites/{site_id}/scan", json={}).json()
            if jobs:
                typer.echo(f"入队 {len(jobs)} 个任务。")
            if once:
                return
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        typer.echo("watch 已停止。")


@jobs_app.command("list")
def jobs_list(status: Annotated[str | None, typer.Option()] = None) -> None:
    client = _connect()
    params = {"status": status} if status else {}
    for job in client.get("/api/v1/jobs", params=params).json():
        typer.echo(f"{job['id']}\t{job['status']}\tattempts={job['attempt_count']}")


@jobs_app.command("show")
def jobs_show(job_id: str) -> None:
    client = _connect()
    typer.echo(client.get(f"/api/v1/jobs/{job_id}").json())


@jobs_app.command("retry")
def jobs_retry(job_id: str) -> None:
    client = _connect()
    typer.echo(client.post(f"/api/v1/jobs/{job_id}/retry").json())


@jobs_app.command("cancel")
def jobs_cancel(job_id: str) -> None:
    client = _connect()
    typer.echo(client.post(f"/api/v1/jobs/{job_id}/cancel").json())


@events_app.command("export")
def events_export(
    out: Annotated[Path, typer.Option("--out")] = Path("events.csv"),
    site_id: Annotated[str | None, typer.Option()] = None,
) -> None:
    client = _connect()
    params = {"site_id": site_id, "limit": 100} if site_id else {"limit": 100}
    events = client.get("/api/v1/events", params=params).json()
    fields = [
        "id",
        "event_uid",
        "site_id",
        "occurred_at",
        "start_sample",
        "end_sample",
        "channel_start",
        "channel_stop",
        "machine_label",
        "latest_review",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow({field: event.get(field) for field in fields})
    typer.echo(f"已导出 {len(events)} 条事件到 {out}")


@app.command("train")
def train(site_id: str) -> None:
    client = _connect()
    model = client.post(f"/api/v1/sites/{site_id}/models").json()
    typer.echo(f"候选模型 {model['id']}（version {model['version']}）已创建。")


@models_app.command("list")
def models_list(site_id: str) -> None:
    client = _connect()
    for model in client.get(f"/api/v1/sites/{site_id}/models").json():
        typer.echo(
            f"{model['id']}\tv{model['version']}\t{model['status']}\t"
            f"macro_f1={model['metrics'].get('macro_f1', 0):.3f}"
        )


@models_app.command("publish")
def models_publish(model_id: str) -> None:
    client = _connect()
    model = client.post(f"/api/v1/models/{model_id}/publish").json()
    typer.echo(f"模型 {model['id']} 已发布（active）。")


@models_app.command("rollback")
def models_rollback(site_id: str) -> None:
    client = _connect()
    model = client.post(f"/api/v1/sites/{site_id}/models/rollback").json()
    typer.echo(f"已回滚到模型 {model['id']}。")


@app.command("benchmark")
def benchmark(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    device: Annotated[Literal["auto", "cpu", "cuda"], typer.Option()] = "cpu",
) -> None:
    """Run the full pipeline on one local HDF5 file and print timings."""
    from das_sentinel.config import ProcessingProfile
    from das_sentinel.pipeline import analyze_file
    from das_sentinel.readers.base import ReaderRegistry
    from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
    from das_sentinel.signal.backend import select_backend

    artifact_root = _app_data_dir() / "artifacts" / "benchmark"
    artifact_root.mkdir(parents=True, exist_ok=True)
    backend = select_backend(device)
    registry = ReaderRegistry([ZdHdf5Reader()])
    reader = registry.resolve(file)
    started = time.perf_counter()
    result = analyze_file(
        file,
        reader,
        ProcessingProfile(device=device),
        backend,
        artifact_root,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    typer.echo(f"backend: {result.actual_backend}")
    if result.backend_fallback_reason:
        typer.echo(f"fallback: {result.backend_fallback_reason}")
    typer.echo(f"events: {len(result.events)}")
    typer.echo(f"audit candidates: {result.audit_candidate_count}")
    typer.echo(f"wall time: {wall_ms:.0f} ms")
    for stage, elapsed in sorted(result.timings_ms.items()):
        typer.echo(f"  {stage}: {elapsed:.0f} ms")


@app.command("doctor")
def doctor() -> None:
    """Validate runtime, storage, backends, and configuration presence."""
    from sqlalchemy.exc import SQLAlchemyError

    from das_sentinel.persistence import (
        check_database_startup,
        create_local_engine,
    )
    from das_sentinel.persistence.db import SQLiteRuntimeError, validate_sqlite_runtime
    from das_sentinel.signal.backend import BackendUnavailable, select_backend

    failures = 0

    try:
        version = validate_sqlite_runtime(sqlite3.sqlite_version)
        typer.echo(f"SQLite runtime: {'.'.join(map(str, version))}")
    except SQLiteRuntimeError as error:
        typer.echo(f"SQLite runtime: UNSUPPORTED ({error})")
        failures += 1

    app_data = _app_data_dir()
    try:
        app_data.mkdir(parents=True, exist_ok=True)
        probe = app_data / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        typer.echo("Application storage: writable")
    except OSError as error:
        typer.echo(f"Application storage: NOT writable ({error})")
        failures += 1

    artifacts = app_data / "artifacts"
    try:
        artifacts.mkdir(parents=True, exist_ok=True)
        probe = artifacts / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        typer.echo("Artifact store: writable")
    except OSError as error:
        typer.echo(f"Artifact store: NOT writable ({error})")
        failures += 1

    data_root = Path(_env("DAS_DATA_ROOT", "/data"))
    if data_root.is_dir() and os.access(data_root, os.R_OK):
        typer.echo(f"Data root: readable ({data_root})")
    else:
        typer.echo(f"Data root: missing or unreadable ({data_root}) (warning)")

    backend = select_backend("cpu")
    typer.echo(f"CPU backend: available ({backend.name})")
    try:
        cuda = select_backend("cuda")
        typer.echo(f"CUDA backend: available ({cuda.name})")
    except BackendUnavailable as error:
        typer.echo(f"CUDA backend: unavailable ({error})")

    typer.echo(
        "Worker token: present"
        if os.environ.get("DAS_WORKER_TOKEN")
        else "Worker token: missing (warning)"
    )
    typer.echo(
        "Session secret: present"
        if os.environ.get("DAS_SESSION_SECRET")
        else "Session secret: missing (warning)"
    )

    database_url = _database_url()
    if "///" in database_url and Path(database_url.split("///", 1)[1]).exists():
        try:
            engine = create_local_engine(database_url)
            try:
                report = check_database_startup(engine)
                typer.echo(f"Database: ready (SQLite {report.sqlite_version})")
            finally:
                engine.dispose()
        except (SQLiteRuntimeError, OSError, SQLAlchemyError) as error:
            typer.echo(f"Database: FAILED startup check ({error})")
            failures += 1
    else:
        typer.echo("Database: not initialized (warning)")

    if failures:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
