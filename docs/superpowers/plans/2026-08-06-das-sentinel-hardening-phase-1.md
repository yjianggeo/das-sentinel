# DAS Sentinel Phase 1 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CPU/NAS deployment bootable and secure, make workers recoverable, and make published models, event identities, and human reviews produce consistent persisted results.

**Architecture:** Keep FastAPI as the single SQLite writer and workers as stateless analysis clients. Bind immutable profile and model identities when a job is first claimed, carry those identities through pipeline output, artifacts, and persistence, and reject corrupted snapshots rather than silently degrading. Use the existing REST, classifier, artifact, and Alembic boundaries; add only focused runtime probes and one schema migration.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy/Alembic, SQLite/pysqlite3, NumPy/CuPy, scikit-learn/skops, React 19, TypeScript, Vitest/MSW, Pytest, Docker Compose, GitHub Actions.

## Global Constraints

- Work only in `Z:/das-event-platform/.worktrees/das-sentinel-hardening` on branch `feature/das-sentinel-hardening`.
- Preserve Python 3.12 and the current REST + SQLite single-writer architecture.
- The runtime SQLite floor remains `3.51.3`; the Linux CPU image must statically bundle SQLite `3.53.3` from `https://sqlite.org/2026/sqlite-amalgamation-3530300.zip`.
- Verify the SQLite archive with SHA-256 `646421e12aac110282ef8cc68f1a62d4bb15fc7b8f09da0b53e29ee690500431` and the `pysqlite3==0.5.4` sdist with SHA-256 `fbc69bfdc0cb43a5badd5403b126d5151371b5037e0397ba9802bb440c5b0021`.
- Never whitelist an unsafe local SQLite build and never run a migration before the selected DB-API driver passes its version gate.
- Containers use `/data` for raw input and `/app-data` for database, models, and artifacts; raw input stays read-only.
- The GPU worker uses fixed CUDA `12.8.1-devel-ubuntu24.04` and must execute a real CuPy FFT smoke test before claiming work.
- A bound model failure is `invalid_model`, is not silently downgraded to rules, and is not automatically retryable.
- Existing CPU/CUDA numerical parity remains `rtol=1e-4, atol=1e-5`.
- Do not make the 64 pre-existing mypy findings a release gate; Ruff, Pytest, frontend tests/build, and available container checks are gates.
- Every production change starts from a failing focused test and ends with a focused test run plus a small commit.

---

## File Structure

- Create `backend/src/das_sentinel/runtime.py`: environment construction, selected SQLite preflight, storage probes, and CUDA startup smoke orchestration.
- Create `backend/src/das_sentinel/api/routes/health.py`: unauthenticated health endpoint with redacted failure responses.
- Create `backend/src/das_sentinel/persistence/migrations/versions/0002_job_model_and_review_request.py`: nullable job model binding and nullable review request id with a unique event/request constraint.
- Create `docker/build-pysqlite3-wheel.sh`: one hashed SQLite/pysqlite3 source build used by both Docker and Python CI.
- Create `.github/workflows/ci.yml`: Python, frontend, CPU image, and GPU build gates.
- Modify existing API, job, worker, pipeline, persistence, Compose, Docker, frontend, and operations files in place; do not restructure adjacent modules.
- Create focused test files only when no existing test file owns the behavior: `tests/test_runtime.py`, `tests/test_health.py`, and `tests/test_model_inference_flow.py`.

---

### Task 1: Runtime App Factory, Secret Validation, and Health Endpoint

**Files:**
- Create: `backend/src/das_sentinel/runtime.py`
- Create: `backend/src/das_sentinel/api/routes/health.py`
- Create: `tests/test_runtime.py`
- Create: `tests/test_health.py`
- Modify: `backend/src/das_sentinel/api/app.py`
- Modify: `backend/src/das_sentinel/api/routes/__init__.py`
- Modify: `backend/src/das_sentinel/config.py`
- Test: `tests/test_spa_static.py`

**Interfaces:**
- Produces: `create_runtime_app() -> FastAPI` in `das_sentinel.api.app` for Uvicorn factory startup.
- Produces: `probe_storage(settings: Settings) -> None` in `das_sentinel.runtime`; raises `RuntimeProbeError` without embedding secrets in messages.
- Produces: `GET /healthz -> {"status": "ok"}` or HTTP 503 `{"detail": "Service unavailable"}`.
- Consumes: existing `Settings`, `create_app(settings)`, `check_database_startup(engine)` and app state engine/session configuration.

- [ ] **Step 1: Write failing runtime factory and secret tests**

```python
# tests/test_runtime.py
from pathlib import Path

import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from das_sentinel.api.app import create_runtime_app
from das_sentinel.config import Settings


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
```

- [ ] **Step 2: Run the runtime tests and verify they fail**

Run: `pytest tests/test_runtime.py -q`

Expected: collection or assertions fail because `create_runtime_app` and insecure-secret validation do not exist.

- [ ] **Step 3: Implement the minimal factory and secret validator**

```python
# backend/src/das_sentinel/config.py, inside Settings
@model_validator(mode="after")
def reject_insecure_secrets(self) -> "Settings":
    for name, value in (
        ("session_secret", self.session_secret),
        ("worker_token", self.worker_token),
    ):
        lowered = value.lower()
        if lowered.startswith("change-me-"):
            raise ValueError(f"{name} uses an insecure example value")
    return self
```

```python
# backend/src/das_sentinel/api/app.py
def create_runtime_app() -> FastAPI:
    return create_app(Settings())  # type: ignore[call-arg]
```

- [ ] **Step 4: Write failing health behavior tests**

```python
# tests/test_health.py
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
```

- [ ] **Step 5: Run the health tests and verify they fail**

Run: `pytest tests/test_health.py tests/test_spa_static.py -q`

Expected: `/healthz` is handled by the SPA fallback or returns 404 because no health router exists.

- [ ] **Step 6: Implement storage probing and the health router**

```python
# backend/src/das_sentinel/runtime.py
from pathlib import Path
from tempfile import NamedTemporaryFile

from das_sentinel.config import Settings


class RuntimeProbeError(RuntimeError):
    pass


def _require_readable_directory(path: Path) -> None:
    if not path.is_dir():
        raise RuntimeProbeError("required directory is unavailable")
    next(path.iterdir(), None)


def probe_storage(settings: Settings) -> None:
    _require_readable_directory(settings.data_root)
    artifact_root = settings.app_data_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=artifact_root, prefix=".health-", delete=True) as probe:
        probe.write(b"ok")
        probe.flush()
```

```python
# backend/src/das_sentinel/api/routes/health.py
from fastapi import APIRouter, HTTPException, Request, status

from das_sentinel.persistence import check_database_startup
from das_sentinel.runtime import probe_storage

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz(request: Request) -> dict[str, str]:
    try:
        check_database_startup(request.app.state.engine)
        probe_storage(request.app.state.settings)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        ) from error
    return {"status": "ok"}
```

Include `health_router` before `_mount_spa(app)` and call `probe_storage(settings)` in lifespan after the database check.

- [ ] **Step 7: Run focused and API tests**

Run: `pytest tests/test_runtime.py tests/test_health.py tests/test_spa_static.py tests/test_config.py -q`

Expected: all selected tests pass and `/healthz` never returns HTML.

- [ ] **Step 8: Commit Task 1**

```bash
git add backend/src/das_sentinel/api/app.py backend/src/das_sentinel/api/routes/health.py backend/src/das_sentinel/api/routes/__init__.py backend/src/das_sentinel/config.py backend/src/das_sentinel/runtime.py tests/test_runtime.py tests/test_health.py tests/test_spa_static.py
git commit -m "fix: add validated runtime app startup"
```

---

### Task 2: Safe SQLite Build, Pre-Migration Gate, and NAS Compose Defaults

**Files:**
- Modify: `backend/src/das_sentinel/runtime.py`
- Modify: `backend/src/das_sentinel/persistence/db.py`
- Modify: `backend/src/das_sentinel/persistence/__init__.py`
- Create: `docker/build-pysqlite3-wheel.sh`
- Modify: `docker/Dockerfile.cpu`
- Modify: `docker/entrypoint-api.sh`
- Modify: `compose.yaml`
- Modify: `compose.cpu.yaml`
- Modify: `compose.gpu.yaml`
- Modify: `.env.example`
- Modify: `pyproject.toml`
- Modify: `docs/operations.md`
- Test: `tests/test_persistence.py`
- Test: `tests/test_compose.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Produces: `validate_selected_sqlite_runtime() -> tuple[int, int, int]` using the same module returned by `_sqlite_driver()`.
- Produces: `python -m das_sentinel.runtime` exit code 0 only after Settings and SQLite preflight pass; it performs no migration.
- Consumes: Task 1 `create_runtime_app()` and `probe_storage()`.

- [ ] **Step 1: Write failing selected-driver and entry-order tests**

```python
# tests/test_persistence.py
def test_selected_sqlite_driver_is_the_runtime_being_validated(monkeypatch):
    selected = SimpleNamespace(sqlite_version="3.53.3")
    monkeypatch.setattr("das_sentinel.persistence.db._sqlite_driver", lambda: selected)
    assert validate_selected_sqlite_runtime() == (3, 53, 3)


def test_selected_sqlite_driver_rejects_unsafe_version(monkeypatch):
    selected = SimpleNamespace(sqlite_version="3.51.1")
    monkeypatch.setattr("das_sentinel.persistence.db._sqlite_driver", lambda: selected)
    with pytest.raises(SQLiteRuntimeError, match="3.51.3"):
        validate_selected_sqlite_runtime()


def test_upgrade_rejects_unsafe_driver_before_alembic(monkeypatch, tmp_path):
    selected = SimpleNamespace(sqlite_version="3.51.1")
    migration_called = False

    def record_upgrade(*args, **kwargs):
        nonlocal migration_called
        migration_called = True

    monkeypatch.setattr("das_sentinel.persistence.db._sqlite_driver", lambda: selected)
    monkeypatch.setattr("das_sentinel.persistence.db.command.upgrade", record_upgrade)
    with pytest.raises(SQLiteRuntimeError):
        upgrade_database(f"sqlite+pysqlite:///{tmp_path / 'unsafe.sqlite3'}")
    assert migration_called is False
```

```python
# tests/test_runtime.py
def test_entrypoint_preflights_before_migration() -> None:
    text = Path("docker/entrypoint-api.sh").read_text(encoding="utf-8")
    assert text.index("python -m das_sentinel.runtime") < text.index("alembic upgrade head")
    assert "--factory" in text
    assert "create_runtime_app" in text
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest tests/test_persistence.py tests/test_runtime.py -q`

Expected: the selected-driver function is missing and the entrypoint still targets the nonexistent module-level `app`.

- [ ] **Step 3: Implement selected-driver preflight and runtime module entry**

```python
# backend/src/das_sentinel/persistence/db.py
def validate_selected_sqlite_runtime() -> tuple[int, int, int]:
    return validate_sqlite_runtime(_sqlite_driver().sqlite_version)
```

Call `validate_selected_sqlite_runtime()` as the first executable line of `upgrade_database`, before constructing Alembic configuration or calling `command.upgrade`. This protects direct CLI administrator initialization as well as the container entrypoint.

```python
# backend/src/das_sentinel/runtime.py
def preflight_from_environment() -> Settings:
    settings = Settings()  # type: ignore[call-arg]
    validate_selected_sqlite_runtime()
    return settings


def main() -> None:
    preflight_from_environment()


if __name__ == "__main__":
    main()
```

```sh
# docker/entrypoint-api.sh, api branch
python -m das_sentinel.runtime
alembic upgrade head
exec uvicorn das_sentinel.api.app:create_runtime_app --factory --host 0.0.0.0 --port 8000
```

- [ ] **Step 4: Write failing reproducible image and Compose tests**

```python
# tests/test_compose.py
def test_host_data_root_is_only_a_bind_source(compose_cpu: dict) -> None:
    for service in ("api", "worker"):
        raw = next(v for v in compose_cpu["services"][service]["volumes"] if v["target"] == "/data")
        assert raw["source"] == "${DAS_HOST_DATA_ROOT}"
        assert compose_cpu["services"][service]["environment"]["DAS_DATA_ROOT"] == "/data"


def test_services_restart_and_api_binds_loopback(compose_cpu: dict) -> None:
    assert compose_cpu["services"]["api"]["ports"] == ["127.0.0.1:${DAS_HTTP_PORT:-8000}:8000"]
    assert compose_cpu["services"]["api"]["restart"] == "unless-stopped"
    assert compose_cpu["services"]["worker"]["restart"] == "unless-stopped"


def test_cpu_image_pins_verified_static_sqlite() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.cpu").read_text("utf-8")
    builder = (ROOT / "docker" / "build-pysqlite3-wheel.sh").read_text("utf-8")
    assert "sqlite-amalgamation-3530300.zip" in builder
    assert "646421e12aac110282ef8cc68f1a62d4bb15fc7b8f09da0b53e29ee690500431" in builder
    assert "fbc69bfdc0cb43a5badd5403b126d5151371b5037e0397ba9802bb440c5b0021" in builder
    assert "build_static bdist_wheel" in builder
    assert "SELECT sqlite_version()" in dockerfile
```

- [ ] **Step 5: Run Compose tests and verify they fail**

Run: `pytest tests/test_compose.py -q`

Expected: old `DAS_DATA_ROOT` bind sources, missing restarts/loopback binding, and unpinned wheel installation fail assertions.

- [ ] **Step 6: Build the static wheel through one reusable hashed script**

Create `docker/build-pysqlite3-wheel.sh` with this complete body:

```sh
#!/bin/sh
set -eu

output_dir=$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${1:?output directory required}")
sqlite_archive=sqlite-amalgamation-3530300.zip
sqlite_sha256=646421e12aac110282ef8cc68f1a62d4bb15fc7b8f09da0b53e29ee690500431
pysqlite_sha256=fbc69bfdc0cb43a5badd5403b126d5151371b5037e0397ba9802bb440c5b0021
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
cd "$build_dir"
curl --fail --location --output "$sqlite_archive" "https://sqlite.org/2026/$sqlite_archive"
echo "$sqlite_sha256  $sqlite_archive" | sha256sum --check --strict
python -m pip download --no-deps --no-binary=:all: pysqlite3==0.5.4
echo "$pysqlite_sha256  pysqlite3-0.5.4.tar.gz" | sha256sum --check --strict
unzip -q "$sqlite_archive"
tar -xzf pysqlite3-0.5.4.tar.gz
cp sqlite-amalgamation-3530300/sqlite3.c sqlite-amalgamation-3530300/sqlite3.h pysqlite3-0.5.4/
python -m pip install wheel
cd pysqlite3-0.5.4
python setup.py build_static bdist_wheel
mkdir -p "$output_dir"
cp dist/*.whl "$output_dir/"
```

Use it in a dedicated `sqlite-wheel` stage in `docker/Dockerfile.cpu`; retain the existing frontend stage and non-root runtime user:

```dockerfile
FROM python:3.12-slim-bookworm AS sqlite-wheel
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl unzip \
    && rm -rf /var/lib/apt/lists/*
COPY docker/build-pysqlite3-wheel.sh /usr/local/bin/build-pysqlite3-wheel
RUN chmod +x /usr/local/bin/build-pysqlite3-wheel \
    && /usr/local/bin/build-pysqlite3-wheel /build/wheels
```

Copy the wheel directory into the runtime stage, install it before `pip install .`, remove `pysqlite3-binary` from `pyproject.toml`, and assert the selected driver inside the image:

```dockerfile
COPY --from=sqlite-wheel /build/wheels/ /tmp/wheels/
RUN pip install /tmp/wheels/*.whl \
    && pip install . \
    && python -c "import pysqlite3.dbapi2 as db; con=db.connect(':memory:'); assert con.execute('SELECT sqlite_version()').fetchone()[0] == '3.53.3'" \
    && rm -rf /tmp/wheels
```

- [ ] **Step 7: Correct Compose variables and runtime defaults**

Apply these exact service properties:

```yaml
# compose.yaml api service
restart: unless-stopped
ports: ["127.0.0.1:${DAS_HTTP_PORT:-8000}:8000"]
environment:
  DAS_DATA_ROOT: /data
volumes:
  - type: bind
    source: ${DAS_HOST_DATA_ROOT}
    target: /data
    read_only: true
```

Apply `restart: unless-stopped`, `DAS_DATA_ROOT: /data`, and `${DAS_HOST_DATA_ROOT}` to both worker overlays. Change `.env.example` to `DAS_HOST_DATA_ROOT=/path/to/raw-das-data` and add optional `DAS_HTTP_PORT=8000`. Make the health command parse JSON and require `status == "ok"`.

- [ ] **Step 8: Document first boot and NAS security**

In `docs/operations.md`, write the exact sequence:

```text
1. Copy .env.example to .env, generate both secrets, and set DAS_HOST_DATA_ROOT.
2. Give the container user read-only access to raw DAS data and read/write access only to the app-data volume.
3. Start API: docker compose -f compose.yaml -f compose.cpu.yaml up -d api.
4. Verify /healthz, then initialize admin with docker compose ... exec api das-sentinel admin init.
5. Start worker and configure the NAS HTTPS reverse proxy to 127.0.0.1:DAS_HTTP_PORT.
6. Rotate secrets by stopping the stack, editing .env, and restarting; existing sessions are invalidated.
```

Retain backup/restore instructions and explicitly state that `/data` is excluded from writes and database placement.

- [ ] **Step 9: Run focused verification**

Run: `pytest tests/test_persistence.py tests/test_runtime.py tests/test_compose.py tests/test_cli.py -q`

Run when Docker is available: `docker build -f docker/Dockerfile.cpu -t das-sentinel:phase1 .`

Run when Docker is available: `docker run --rm --entrypoint python das-sentinel:phase1 -c "import pysqlite3.dbapi2 as db; assert db.sqlite_version == '3.53.3'"`

Expected: Python tests pass; Docker commands report SQLite `3.53.3`. If Docker is unavailable, record the commands as not executed rather than passed.

- [ ] **Step 10: Commit Task 2**

```bash
git add .env.example backend/src/das_sentinel/runtime.py backend/src/das_sentinel/persistence/db.py backend/src/das_sentinel/persistence/__init__.py docker/build-pysqlite3-wheel.sh docker/Dockerfile.cpu docker/entrypoint-api.sh compose.yaml compose.cpu.yaml compose.gpu.yaml docs/operations.md pyproject.toml tests/test_persistence.py tests/test_runtime.py tests/test_compose.py
git commit -m "fix: make NAS runtime startup reproducible"
```

---

### Task 3: Worker Transport Recovery, Runtime Stale Recovery, and CUDA Smoke

**Files:**
- Modify: `backend/src/das_sentinel/worker/main.py`
- Modify: `backend/src/das_sentinel/worker/__main__.py`
- Modify: `backend/src/das_sentinel/api/routes/internal.py`
- Modify: `backend/src/das_sentinel/signal/backend.py`
- Modify: `backend/src/das_sentinel/runtime.py`
- Modify: `docker/Dockerfile.gpu`
- Test: `tests/test_worker.py`
- Test: `tests/test_worker_main.py`
- Test: `tests/test_worker_protocol.py`
- Test: `tests/test_backend.py`
- Test: `tests/test_compose.py`

**Interfaces:**
- Produces: `WorkerTransientError` for transport, timeout, and HTTP 5xx; `WorkerFatalProtocolError` for 401 and other non-retryable protocol responses.
- Produces: `run_forever(..., sleep: Callable[[float], None] = time.sleep)` with backoff `1, 2, 4, 8, 16, 30` seconds and reset after any successful poll.
- Produces: `smoke_test_cuda() -> None` that executes CuPy allocation, FFT, NumPy transfer, and device synchronization.
- Produces: `_failure_retry_eligible(error) -> bool`; `SourceChangedError`, invalid-claim `ValueError`/`TypeError`, and later `InvalidModelError` return false.
- Consumes: existing API write coordinator and `JobService.recover_stale()`.

- [ ] **Step 1: Write failing worker loop backoff tests**

```python
# tests/test_worker_main.py
class _SequenceWorker:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def run_once(self):
        value = next(self.outcomes)
        if isinstance(value, BaseException):
            raise value
        return WorkerOutcome(status=value)


def test_transient_failures_back_off_and_success_resets_delay() -> None:
    sleeps = []
    worker = _SequenceWorker([
        WorkerTransientError("offline"),
        WorkerTransientError("offline"),
        "idle",
        WorkerTransientError("offline"),
    ])
    run_forever(worker, idle_seconds=0.0, max_iterations=4, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0, 0.0, 1.0]


def test_fatal_protocol_error_exits_without_retry() -> None:
    worker = _SequenceWorker([WorkerFatalProtocolError("unauthorized")])
    with pytest.raises(WorkerFatalProtocolError, match="unauthorized"):
        run_forever(worker, max_iterations=1, sleep=lambda _: None)
```

- [ ] **Step 2: Write failing HTTP classification tests**

```python
# tests/test_worker.py
@pytest.mark.parametrize("status", [500, 502, 503])
def test_http_client_maps_server_errors_to_transient(status: int) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, request=request))
    client = HttpxWorkerClient("http://api", "w" * 40, transport=transport)
    with pytest.raises(WorkerTransientError):
        client.claim("worker-1")


def test_http_client_maps_unauthorized_to_fatal() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401, request=request))
    client = HttpxWorkerClient("http://api", "w" * 40, transport=transport)
    with pytest.raises(WorkerFatalProtocolError):
        client.claim("worker-1")


def test_invalid_claim_is_reported_nonretryable(tmp_path: Path) -> None:
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol([replace(_claim(source), source_size_bytes=999)])
    outcome = _worker(
        tmp_path,
        protocol,
        lambda *args: pytest.fail("invalid claim must not be analyzed"),
    ).run_once()
    assert outcome.error_code == "invalid_claim"
    assert protocol.failures[0][4] is False
```

- [ ] **Step 3: Run worker tests and verify they fail**

Run: `pytest tests/test_worker_main.py tests/test_worker.py -q`

Expected: exception classes and injectable sleep do not exist; 5xx currently escapes as raw `HTTPStatusError`.

- [ ] **Step 4: Implement request classification and bounded backoff**

In `HttpxWorkerClient`, route every request through `_request(method, path, json=None)`. Convert `httpx.TransportError` to `WorkerTransientError`, HTTP 5xx to `WorkerTransientError`, HTTP 401 to `WorkerFatalProtocolError`, and remaining HTTP 4xx to `WorkerFatalProtocolError` with status only, not response bodies containing sensitive payloads.

```python
BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)


def run_forever(worker, *, idle_seconds=DEFAULT_IDLE_SECONDS, max_iterations=None, sleep=time.sleep):
    iterations = 0
    transient_failures = 0
    while max_iterations is None or iterations < max_iterations:
        try:
            outcome = worker.run_once()
        except WorkerTransientError:
            delay = BACKOFF_SECONDS[min(transient_failures, len(BACKOFF_SECONDS) - 1)]
            transient_failures += 1
            sleep(delay)
        else:
            transient_failures = 0
            if outcome.status == "idle":
                sleep(idle_seconds)
        iterations += 1
    return iterations
```

In `Worker.run_once`, re-raise `WorkerTransientError` and `WorkerFatalProtocolError` before the generic per-file failure handler. Pass `_failure_retry_eligible(error)` into `_report_failure`; it returns false for `SourceChangedError`, `ValueError`, and `TypeError`, and true for ordinary analysis/persistence failures. Task 5 extends it for `InvalidModelError`. A transport loss after a claim leaves the server job running so heartbeat expiry and claim recovery can safely requeue it.

- [ ] **Step 5: Write and implement runtime stale recovery on claim**

```python
# tests/test_worker_protocol.py
def test_claim_recovers_stale_job_without_api_restart(api_client) -> None:
    client, token, ids = api_client
    stale_id = _enqueue(client, token, ids)
    first = client.post(
        "/api/v1/internal/jobs/claim",
        headers={"X-Worker-Token": token},
        json={"worker_id": "dead-worker"},
    ).json()["id"]
    assert first == stale_id
    with client.app.state.session_factory.begin() as session:
        row = session.get(Job, stale_id)
        row.heartbeat_at = utc_now() - STALE_AFTER - timedelta(seconds=1)
    recovered = client.post(
        "/api/v1/internal/jobs/claim",
        headers={"X-Worker-Token": token},
        json={"worker_id": "replacement-worker"},
    )
    assert recovered.status_code == 200
    assert recovered.json()["id"] == stale_id
    assert recovered.json()["worker_id"] == "replacement-worker"
```

Change `_claim()` to use one transaction and one service instance:

```python
def _claim(sessions, artifact_root, worker_id: str) -> JobClaim | None:
    with sessions.begin() as session:
        service = JobService(session, artifact_root)
        service.recover_stale()
        return service.claim(worker_id)
```

- [ ] **Step 6: Write failing real CUDA smoke tests**

```python
# tests/test_backend.py
def test_cuda_smoke_executes_fft_transfer_and_sync(monkeypatch) -> None:
    calls = []
    fake = FakeCupy(calls)
    monkeypatch.setattr("das_sentinel.signal.backend._load_cupy", lambda: fake)
    smoke_test_cuda()
    assert calls == ["asarray", "rfft", "asnumpy", "synchronize"]
```

`FakeCupy` in the same test file must expose `asarray`, `fft.rfft`, `asnumpy`, and `cuda.Stream.null.synchronize`, returning the NumPy FFT of `[0, 1, 0, -1]` so the production smoke can verify finite output.

- [ ] **Step 7: Implement CUDA smoke and use the devel image**

```python
# backend/src/das_sentinel/signal/backend.py
def smoke_test_cuda() -> None:
    cupy = _load_cupy()
    if cupy is None:
        raise BackendUnavailable(_CUPY_LOAD_ERROR or "CuPy/CUDA is unavailable")
    values = cupy.asarray([0.0, 1.0, 0.0, -1.0], dtype=cupy.float32)
    spectrum = cupy.fft.rfft(values)
    host = np.asarray(cupy.asnumpy(spectrum))
    cupy.cuda.Stream.null.synchronize()
    if host.shape != (3,) or not np.isfinite(host).all():
        raise BackendUnavailable("CUDA FFT smoke test returned invalid output")
```

Call `smoke_test_cuda()` in worker `main()` before constructing the HTTP client when `settings.worker_device == "cuda"`. Change the GPU base image to:

```dockerfile
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS runtime
```

- [ ] **Step 8: Run focused verification**

Run: `pytest tests/test_worker_main.py tests/test_worker.py tests/test_worker_protocol.py tests/test_backend.py tests/test_compose.py -q`

Expected: transport/5xx retry, 401 fail-fast, stale recovery, and fake CUDA smoke tests pass.

- [ ] **Step 9: Commit Task 3**

```bash
git add backend/src/das_sentinel/worker/main.py backend/src/das_sentinel/worker/__main__.py backend/src/das_sentinel/api/routes/internal.py backend/src/das_sentinel/signal/backend.py backend/src/das_sentinel/runtime.py docker/Dockerfile.gpu tests/test_worker.py tests/test_worker_main.py tests/test_worker_protocol.py tests/test_backend.py tests/test_compose.py
git commit -m "fix: recover workers across transient failures"
```

---

### Task 4: Persist Immutable Model Bindings on Jobs

**Files:**
- Create: `backend/src/das_sentinel/persistence/migrations/versions/0002_job_model_and_review_request.py`
- Modify: `backend/src/das_sentinel/persistence/models.py`
- Modify: `backend/src/das_sentinel/services/jobs.py`
- Modify: `backend/src/das_sentinel/api/schemas.py`
- Modify: `backend/src/das_sentinel/api/routes/internal.py`
- Modify: `backend/src/das_sentinel/worker/main.py`
- Test: `tests/test_persistence.py`
- Test: `tests/test_jobs.py`
- Test: `tests/test_worker_protocol.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Produces: nullable `Job.model_version_id` and nullable `Review.request_id` columns; unique index `uq_review_event_request` on `(event_id, request_id)`.
- Produces: all-or-none claim fields `model_version_id`, `model_version`, `model_artifact_path`, `model_bundle_sha256`, and `model_feature_names`.
- Consumes: `TrainingService.active_model(site_id)` selection semantics, implemented in the job service as the same ordered active-model query.
- Invariant: first claim binds the current active model or an explicit no-model snapshot; stale retry claims retain the first binding by checking pre-increment `attempt_count`.

- [ ] **Step 1: Write failing migration shape test**

```python
# tests/test_persistence.py
def test_hardening_migration_adds_model_binding_and_review_idempotency(migrated_engine):
    inspector = inspect(migrated_engine)
    job_columns = {item["name"] for item in inspector.get_columns("jobs")}
    review_columns = {item["name"] for item in inspector.get_columns("reviews")}
    review_indexes = {
        tuple(item["column_names"]): item["unique"]
        for item in inspector.get_indexes("reviews")
    }
    assert "model_version_id" in job_columns
    assert "request_id" in review_columns
    assert review_indexes[("event_id", "request_id")] is True
```

- [ ] **Step 2: Run the migration test and verify it fails**

Run: `pytest tests/test_persistence.py::test_hardening_migration_adds_model_binding_and_review_idempotency -q`

Expected: both new columns are absent.

- [ ] **Step 3: Add migration and ORM fields**

```python
# 0002_job_model_and_review_request.py
revision = "0002_job_model_and_review_request"
down_revision = "0001_initial"


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id"),
            nullable=True,
        ),
    )
    op.add_column("reviews", sa.Column("request_id", sa.String(36), nullable=True))
    op.create_index(
        "uq_review_event_request",
        "reviews",
        ["event_id", "request_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_review_event_request", table_name="reviews")
    op.drop_column("reviews", "request_id")
    op.drop_column("jobs", "model_version_id")
```

Mirror both columns and the unique review index in `persistence/models.py`. Direct `ALTER TABLE ... ADD COLUMN` preserves the existing append-only review triggers; extend the migration test to assert those triggers still exist.

- [ ] **Step 4: Write failing first-claim snapshot tests**

```python
# tests/test_jobs.py
def test_first_claim_binds_active_model_and_stale_reclaim_keeps_it(job_database):
    with seeded_service(job_database) as (service, repositories, site, profile, source):
        active = repositories.models.add(
            site_id=site.id,
            version="7",
            artifact_path=str(service.artifact_root / "models" / site.id / "7" / "model.bundle"),
            bundle_sha256="7" * 64,
            status="active",
            metrics={"feature_names": ["duration_s", "rms"]},
        )
        queued = service.enqueue(source.id, profile.id, "pipeline-v1")
        first = service.claim("worker-a")
        assert first.id == queued.id
        assert first.model_version_id == active.id
        assert first.model_version == "7"
        assert first.model_bundle_sha256 == active.bundle_sha256
        assert first.model_feature_names == ("duration_s", "rms")
        service.recover_stale(
            now=first.heartbeat_at + STALE_AFTER + timedelta(seconds=1)
        )
        repositories.models.set_status(active.id, status="retired")
        repositories.models.add(
            site_id=site.id,
            version="8",
            artifact_path=str(service.artifact_root / "models" / site.id / "8" / "model.bundle"),
            bundle_sha256="8" * 64,
            status="active",
            metrics={"feature_names": ["duration_s", "rms"]},
        )
        second = service.claim("worker-b")
        assert second.model_version_id == active.id


def test_first_claim_without_active_model_stays_rules_only_after_reclaim(job_database):
    with seeded_service(job_database) as (service, repositories, site, profile, source):
        service.enqueue(source.id, profile.id, "pipeline-v1")
        first = service.claim("worker-a")
        assert first.model_version_id is None
        service.recover_stale(
            now=first.heartbeat_at + STALE_AFTER + timedelta(seconds=1)
        )
        repositories.models.add(
            site_id=site.id,
            version="1",
            artifact_path=str(service.artifact_root / "models" / site.id / "1" / "model.bundle"),
            bundle_sha256="1" * 64,
            status="active",
            metrics={"feature_names": ["duration_s", "rms"]},
        )
        second = service.claim("worker-b")
        assert second.model_version_id is None
```

- [ ] **Step 5: Implement claim binding and schema transport**

Before incrementing `attempt_count` in `JobService.claim`, select and assign an active model only when `row.attempt_count == 0`. Extend `JobClaim`, `JobClaimResponse`, `WorkerClaim.from_dict()`, `WorkerClaim.to_dict()`, and `claim_response()` with:

```python
model_version_id: str | None
model_version: str | None
model_artifact_path: str | None
model_bundle_sha256: str | None
model_feature_names: tuple[str, ...] | None
```

Add a Pydantic `model_validator` to `JobClaimResponse` requiring all five values to be either present or absent. In `JobService._claim`, load the saved `row.model_version_id`; if present, require its site to equal the source site, decode `metrics_json["feature_names"]`, validate the tuple with `FeatureVectorizer`, and return its immutable version/path/digest/feature schema.

- [ ] **Step 6: Make failed non-retryable jobs explicitly re-enqueueable**

Write this regression in `tests/test_jobs.py`:

```python
def test_enqueue_creates_new_job_after_nonretryable_failure(job_database):
    with seeded_service(job_database) as (service, _, _, profile, source):
        original = service.enqueue(source.id, profile.id, "pipeline-v1")
        claim = service.claim("worker-a")
        service.fail(
            claim.id,
            "worker-a",
            error_code="invalid_model",
            error_message="bundle digest mismatch",
            retry_eligible=False,
        )
        replacement = service.enqueue(source.id, profile.id, "pipeline-v1")
        assert replacement.id != original.id
        assert replacement.attempt_count == 0
```

Change enqueue deduplication to return an existing `queued`, `running`, `succeeded`, or retry-eligible `failed` job. Only a non-retryable failed or cancelled job permits an explicit rescan/re-enqueue.

- [ ] **Step 7: Run schema and protocol tests**

Run: `pytest tests/test_persistence.py tests/test_jobs.py tests/test_worker_protocol.py tests/test_worker.py -q`

Expected: migration, immutable first claim, no-model reclaim, all-or-none schema, and replacement job tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add backend/src/das_sentinel/persistence/migrations/versions/0002_job_model_and_review_request.py backend/src/das_sentinel/persistence/models.py backend/src/das_sentinel/services/jobs.py backend/src/das_sentinel/api/schemas.py backend/src/das_sentinel/api/routes/internal.py backend/src/das_sentinel/worker/main.py tests/test_persistence.py tests/test_jobs.py tests/test_worker_protocol.py tests/test_worker.py
git commit -m "feat: bind model snapshots to jobs"
```

---

### Task 5: Execute, Fuse, and Persist the Bound Supervised Model

**Files:**
- Create: `tests/test_model_inference_flow.py`
- Modify: `backend/src/das_sentinel/classifiers/supervised.py`
- Modify: `backend/src/das_sentinel/pipeline.py`
- Modify: `backend/src/das_sentinel/worker/main.py`
- Modify: `backend/src/das_sentinel/api/schemas.py`
- Modify: `backend/src/das_sentinel/services/results.py`
- Modify: `backend/src/das_sentinel/persistence/repositories.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_worker.py`
- Test: `tests/test_events_api.py`
- Test: `tests/test_supervised.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Produces: `load_model_bundle_bytes(raw: bytes, expected_feature_names: Sequence[str], expected_sha256: str) -> ModelBundle`.
- Extends: `AnalysisContext` with `processing_profile_id`, `model_version_id`, and `model_version`.
- Extends: `analyze_file(..., context: AnalysisContext | None = None, model_bundle: ModelBundle | None = None) -> FileAnalysis`.
- Extends: `WorkerResultRequest.model_version_id: str | None` and validates it against `Job.model_version_id`.
- Invariant: `method == "fusion"` exactly when `model_version_id` is present; otherwise `method == "rules"`.

- [ ] **Step 1: Write failing pipeline fusion test**

```python
# tests/test_pipeline.py
class _PredictingBundle:
    def predict_probabilities(self, features):
        assert features["site_id"] == "site-a"
        return {
            Label.EARTHQUAKE: 0.01,
            Label.BLAST: 0.01,
            Label.TRAFFIC: 0.97,
            Label.NOISE: 0.01,
        }


def test_pipeline_uses_bound_model_and_records_model_evidence(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(
        pipeline,
        "render_event_artifacts",
        lambda data, event, features, out_dir: captured.append(data)
        or _empty_artifact_manifest(data, event, features, out_dir),
    )
    context = AnalysisContext(
        site_id="site-a",
        processing_profile_id="profile-a",
        profile_version="1",
        pipeline_version="v1",
        model_version_id="model-a",
        model_version="7",
    )
    result = analyze_file(
        _write_boundary_file(tmp_path / "source.h5"),
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "artifacts",
        context,
        _PredictingBundle(),
    )
    assert result.events[0].label == Label.TRAFFIC
    assert captured[0].model_version == "7"
    assert captured[0].model_output["probabilities"]["traffic"] == 0.97
```

- [ ] **Step 2: Run the pipeline test and verify it fails**

Run: `pytest tests/test_pipeline.py::test_pipeline_uses_bound_model_and_records_model_evidence -q`

Expected: `AnalysisContext` and `analyze_file` do not accept model identity/bundle arguments.

- [ ] **Step 3: Implement model-aware pipeline fusion**

Add defaults `processing_profile_id="unknown-profile"`, `model_version_id=None`, and `model_version=None` to `AnalysisContext`, with an all-or-none validator for the model pair. Pass `processing_profile_id` into `fuse_candidates` in Task 7; do not change UUID behavior in this step.

Inside the classification loop:

```python
model_probabilities = (
    None
    if model_bundle is None
    else model_bundle.predict_probabilities(features)
)
decision = fuse_classification(
    rule_result.scores,
    model_probabilities,
    quality_flags,
)
model_output = None
if model_probabilities is not None:
    model_output = {
        "model_version_id": run_context.model_version_id,
        "probabilities": {
            label.value: probability
            for label, probability in model_probabilities.items()
        },
        "fused_scores": {
            label.value: score for label, score in decision.scores.items()
        },
        "reasons": list(decision.reasons),
    }
```

Set `EventArtifactData.model_version` to `run_context.model_version or "rules-only"` and `model_output` to the mapping above.

- [ ] **Step 4: Write failing safe bundle byte-loading and worker failure tests**

```python
# tests/test_supervised.py
def test_load_model_bundle_bytes_verifies_expected_digest(trained_bundle_bytes):
    raw, feature_names = trained_bundle_bytes
    digest = hashlib.sha256(raw).hexdigest()
    loaded = load_model_bundle_bytes(raw, feature_names, digest)
    assert loaded.feature_names == tuple(feature_names)
    with pytest.raises(ValueError, match="SHA-256"):
        load_model_bundle_bytes(raw, feature_names, "0" * 64)
```

```python
# tests/test_worker.py
def test_bound_corrupt_model_fails_without_rules_fallback(tmp_path):
    claim = _claim(tmp_path, model_version_id="model-1", model_version="1")
    Path(claim.model_artifact_path).write_bytes(b"corrupt")
    protocol = FakeProtocol([claim])
    worker = _worker(tmp_path, protocol=protocol)
    outcome = worker.run_once()
    assert outcome.error_code == "invalid_model"
    assert protocol.failures[0]["retry_eligible"] is False
    assert protocol.submissions == []
```

Extend existing `_claim` test helper with optional model fields and place valid model paths under `settings.app_data_dir / "artifacts" / "models"`.

- [ ] **Step 5: Implement held-byte bundle loading and invalid-model classification**

Refactor existing `load_model_bundle(path, ...)` so it reads and verifies bytes, then delegates to:

```python
def load_model_bundle_bytes(
    raw: bytes,
    expected_feature_names: Sequence[str],
    expected_sha256: str,
) -> ModelBundle:
    if len(raw) > _MAX_BUNDLE_BYTES:
        raise ValueError("model bundle is too large")
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("model bundle SHA-256 does not match")
    metadata, model_bytes = _read_bundle(raw)
    _validate_skops_archive(model_bytes)
    return _load_verified_components(metadata, model_bytes, expected_feature_names)
```

Move the existing metadata/skops checks into `_load_verified_components` without weakening bundle size, member count, member size, total size, compression-ratio, schema, or sklearn-version validation.

In the worker, open the claimed model with `_open_anchored_source(model_root, model_path)`, read no more than the existing bundle maximum, and call `load_model_bundle_bytes(raw, claim.model_feature_names, claim.model_bundle_sha256)` exactly once while the anchored handle remains open. Define `InvalidModelError` in `pipeline.py`; wrap bundle prediction exceptions in it, and wrap worker path/digest/archive/schema failures in the same type. `_failure_code` returns `invalid_model`, and `_report_failure` accepts an explicit `retry_eligible` argument that is `False` only for this error.

- [ ] **Step 6: Carry model identity through worker payload and persistence**

Add to `WorkerResultRequest`:

```python
model_version_id: str | None = Field(default=None, max_length=36)
```

Set it from the claim in `_result_payload`; set each classification method to `"fusion"` when non-null and `"rules"` otherwise. In `ResultService._validate_identity`, require exact equality with `job.model_version_id` and reject method/model mismatches. In `_persist_validated`, pass the ID to both:

```python
run = self.repositories.runs.add(
    job_id=job.id,
    model_version_id=payload.model_version_id,
    source_file_id=source.id,
    processing_profile_id=profile.id,
    pipeline_version=job.pipeline_version,
    requested_backend=payload.requested_backend,
    actual_backend=payload.actual_backend,
    backend_fallback_reason=payload.backend_fallback_reason,
    status="succeeded",
    timings=payload.timings_ms,
)
```

```python
self.repositories.events.add_classification(
    row.id,
    label=classification.label,
    decision_score=classification.decision_score,
    method=classification.method,
    reasons=classification.reasons,
    model_version_id=payload.model_version_id,
)
```

- [ ] **Step 7: Add one full published-model integration test**

In `tests/test_model_inference_flow.py`, define these local fixtures so the test has no dependency on another test module:

```python
LABELS = ("earthquake", "blast", "traffic", "noise")
CONTEXT_FEATURES = {"site_id", "cable_type", "device_config"}


def _training_features(site_id: str, label_index: int, index: int):
    values = {
        name: 0.0
        for name in FEATURE_NAMES
        if name not in CONTEXT_FEATURES
    }
    values.update({
        "duration_s": label_index * 100.0 + index,
        "rms": label_index + index / 100.0,
        "site_id": site_id,
        "cable_type": "urban",
        "device_config": "cpu",
    })
    return values
```

`_seed_published_model(sessions, artifact_root, inference_path)` creates one urban UTC site and current profile, a training source/run, 25 reviewed events for each label over five dates using `_training_features`, then calls real `TrainingService.train` and `publish`. It also registers `inference_path` as a second source and enqueues one `pipeline-v1` job, returning `(published_model, job_id)`.

`_write_inference_hdf5(path)` writes 400 samples × 8 channels at 200 Hz with `RawData`, `RawDataTime`, `OutputDataRate`, `SpatialSamplingInterval`, `StartLocusIndex`, and `RawDataUnit`. `_fixed_model_event` returns one score-0.95 candidate covering samples 100:160 and channels 0:4. Use the real CPU preprocessing, feature extraction, bundle loading, classification fusion, artifact rendering, worker result schema, and result service; patch only the four detector callables so STA/LTA returns `_fixed_model_event` and the other three return no candidates.

Seed, train, publish, claim through `JobService`, convert the returned dataclass through `claim_response(...).model_dump(mode="json")` and `WorkerClaim.from_dict`, execute a real `Worker` with a capturing protocol, validate its submitted dictionary with `WorkerResultRequest`, persist it with `ResultService`, then assert:

```python
assert claim.model_version_id == published.id
assert persisted_run.model_version_id == published.id
assert persisted_classification.model_version_id == published.id
assert persisted_classification.method == "fusion"
assert evidence_json["model_version"] == published.version
assert evidence_json["model_output"]["model_version_id"] == published.id
assert set(evidence_json["model_output"]["probabilities"]) == {
    "earthquake", "blast", "traffic", "noise"
}
```

The test must use real `save_model_bundle`, `load_model_bundle_bytes`, `ModelBundle.predict_probabilities`, `fuse_classification`, worker payload validation, and `ResultService`; patch only detector output and image rendering duration.

- [ ] **Step 8: Run focused model flow tests**

Run: `pytest tests/test_supervised.py tests/test_pipeline.py tests/test_worker.py tests/test_events_api.py tests/test_artifacts.py tests/test_model_inference_flow.py -q`

Expected: rules-only behavior remains unchanged, bound model flow records one consistent model ID, and corrupt models fail non-retryably.

- [ ] **Step 9: Commit Task 5**

```bash
git add backend/src/das_sentinel/classifiers/supervised.py backend/src/das_sentinel/pipeline.py backend/src/das_sentinel/worker/main.py backend/src/das_sentinel/api/schemas.py backend/src/das_sentinel/services/results.py backend/src/das_sentinel/persistence/repositories.py tests/test_model_inference_flow.py tests/test_pipeline.py tests/test_worker.py tests/test_events_api.py tests/test_supervised.py tests/test_artifacts.py
git commit -m "feat: run published models in analysis jobs"
```

---

### Task 6: Preserve Real Feature Mappings in Training and Similarity

**Files:**
- Modify: `backend/src/das_sentinel/domain.py`
- Modify: `backend/src/das_sentinel/services/training.py`
- Modify: `backend/src/das_sentinel/api/routes/events.py`
- Modify: `frontend/src/api.ts`
- Test: `tests/test_training_service.py`
- Test: `tests/test_events_api.py`
- Test: `tests/test_similarity.py`

**Interfaces:**
- Produces: `EventRecord.features: dict[str, float | str]` and `TrainingRow.features: dict[str, float | str]`.
- Consumes: existing `FeatureVectorizer`, which ignores only `site_id`, `cable_type`, and `device_config` and validates approved numeric features.

- [ ] **Step 1: Write failing real-feature regression tests**

```python
# tests/test_training_service.py
def _real_training_features(site_id: str, label_index: int, index: int):
    block = preprocessed_block(
        np.tile(np.linspace(-1.0, 1.0, 80, dtype=np.float32)[:, None], (1, 4))
    )
    event = EventRecord(
        event_id=f"feature-{label_index}-{index}",
        start_sample=1_010,
        end_sample=1_070,
        peak_sample=1_040,
        channel_start=0,
        channel_stop=4,
        evidence=(),
    )
    features = extract_features(
        block,
        event,
        context={
            "site_id": site_id,
            "cable_type": "urban",
            "device_config": "cpu",
        },
    )
    features["duration_s"] = label_index * 100.0 + index
    features["rms"] = label_index + index / 100.0
    return features


def test_reviewed_rows_keep_extract_features_string_context(training_env):
    sessions, artifact_root = training_env
    site_id = _seed_site(sessions)
    with sessions() as session:
        rows = TrainingService(session, artifact_root)._reviewed_rows(site_id)
    assert rows[0].features["site_id"] == site_id
    assert rows[0].features["cable_type"] == "urban"
    FeatureVectorizer().transform(row.features for row in rows)
```

Change `_seed_site` in the same test file to use `cable_type="urban"` and call `_real_training_features(site.id, label_index, index)` instead of constructing its two-field feature dictionary. Keep the existing label separation and five review dates.

```python
# tests/test_events_api.py
def _seed_events_with_extract_features_output(event_api, count: int) -> str:
    client, _, sessions, _, ids = event_api
    block = preprocessed_block(
        np.tile(np.linspace(-1.0, 1.0, 80, dtype=np.float32)[:, None], (1, 4))
    )
    record = EventRecord(
        event_id="feature-template",
        start_sample=1_010,
        end_sample=1_070,
        peak_sample=1_040,
        channel_start=0,
        channel_stop=4,
        evidence=(),
    )
    template = extract_features(
        block,
        record,
        context={"site_id": ids["site_a"], "cable_type": "urban", "device_config": "cpu"},
    )
    created = []
    with sessions.begin() as session:
        repositories = Repositories(session)
        for index in range(count):
            event = repositories.events.add(
                run_id=ids["run_a"],
                event_uid=f"real-feature-{index}",
                start_sample=100 + index * 10,
                end_sample=105 + index * 10,
                channel_start=0,
                channel_stop=4,
                occurred_at=datetime(2026, 8, 1, 0, index, tzinfo=UTC),
                features={**template, "rms": float(index + 1)},
            )
            repositories.events.add_classification(
                event.id, label="traffic", decision_score=0.9, method="rules"
            )
            created.append(event.id)
    return created[0]


def test_similar_events_accept_real_string_context_features(event_api):
    client, _, _, _, _ = event_api
    event_id = _seed_events_with_extract_features_output(event_api, count=6)
    response = client.get(f"/api/v1/events/{event_id}/similar?limit=5")
    assert response.status_code == 200
    assert len(response.json()) == 5
```

- [ ] **Step 2: Run regressions and verify 500 failures**

Run: `pytest tests/test_training_service.py::test_reviewed_rows_keep_extract_features_string_context tests/test_events_api.py::test_similar_events_accept_real_string_context_features -q`

Expected: `float("site-id")` or equivalent conversion fails before `FeatureVectorizer` sees the mapping.

- [ ] **Step 3: Remove premature float conversions**

Use direct decoded mappings in both services:

```python
features={str(name): value for name, value in json.loads(event.features_json).items()}
```

Use the same expression for similarity peers and the query event. Do not add a second allowlist; `FeatureVectorizer` remains the single schema boundary. Update Python and TypeScript feature types to `float | str` and `number | string` respectively.

- [ ] **Step 4: Run feature, training, and API tests**

Run: `pytest tests/test_features.py tests/test_similarity.py tests/test_supervised.py tests/test_training_service.py tests/test_events_api.py -q`

Expected: real extracted mappings train/vectorize/query without 500 responses; unknown numeric keys remain rejected by `FeatureVectorizer`.

- [ ] **Step 5: Commit Task 6**

```bash
git add backend/src/das_sentinel/domain.py backend/src/das_sentinel/services/training.py backend/src/das_sentinel/api/routes/events.py frontend/src/api.ts tests/test_training_service.py tests/test_events_api.py tests/test_similarity.py
git commit -m "fix: preserve contextual event features"
```

---

### Task 7: Profile-Specific Event Identity and Final-Window Energy Detection

**Files:**
- Modify: `backend/src/das_sentinel/fusion.py`
- Modify: `backend/src/das_sentinel/pipeline.py`
- Modify: `backend/src/das_sentinel/worker/main.py`
- Modify: `backend/src/das_sentinel/detectors/energy.py`
- Test: `tests/test_fusion.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_worker.py`
- Test: `tests/test_energy.py`

**Interfaces:**
- Extends: `fuse_candidates(..., processing_profile_id: str, *, source_fingerprint: str | None = None)`.
- Consumes: Task 5 `AnalysisContext.processing_profile_id` populated from `WorkerClaim.processing_profile_id`.
- Invariant: same source/profile/pipeline evidence has a deterministic UUID; changing immutable profile ID changes the UUID and artifact directory.

- [ ] **Step 1: Write failing UUID identity tests**

```python
# tests/test_fusion.py
def test_event_uid_is_stable_within_profile_and_changes_between_profiles(profile):
    candidate = Candidate(
        detector="energy", start_sample=10, end_sample=20, peak_sample=15,
        channel_start=2, channel_stop=6, score=0.95,
    )
    first = fuse_candidates(
        [candidate], profile, 100.0, "site-a", "pipeline-v1", "profile-a",
        source_fingerprint="a" * 64,
    )[0]
    repeat = fuse_candidates(
        [candidate], profile, 100.0, "site-a", "pipeline-v1", "profile-a",
        source_fingerprint="a" * 64,
    )[0]
    changed = fuse_candidates(
        [candidate], profile, 100.0, "site-a", "pipeline-v1", "profile-b",
        source_fingerprint="a" * 64,
    )[0]
    assert first.event_id == repeat.event_id
    assert first.event_id != changed.event_id
```

- [ ] **Step 2: Write failing final energy window test**

```python
# tests/test_energy.py
def test_energy_keeps_event_in_last_complete_rms_window(profile):
    data = np.full((800, 5), 0.01, dtype=np.float32)
    time = np.arange(610, 800, dtype=np.float32) / 100.0
    data[610:800, :4] += (3.0 * np.sin(2 * np.pi * 8.0 * time))[:, None]
    candidates = detect_energy(preprocessed_block(data), profile)
    assert len(candidates) == 1
    assert candidates[0].end_sample >= 1_799
```

- [ ] **Step 3: Run both tests and verify they fail**

Run: `pytest tests/test_fusion.py::test_event_uid_is_stable_within_profile_and_changes_between_profiles tests/test_energy.py::test_energy_keeps_event_in_last_complete_rms_window -q`

Expected: fusion has no profile parameter and the final energy event is masked away.

- [ ] **Step 4: Add profile ID to UUID5 input**

Change `_event_from_evidence` to build the stable name from all immutable identity fields:

```python
event_name = (
    f"site={site_id}|profile={processing_profile_id}|pipeline={pipeline_version}|"
    f"source={source_fingerprint or 'unknown'}|peak={peak_candidate.peak_sample}|"
    f"channels={channel_start}:{channel_stop}"
)
```

Pass `run_context.processing_profile_id` from `analyze_file`; populate it from `claim.processing_profile_id` in the worker. Update all direct `fuse_candidates` calls and tests with a stable profile ID.

- [ ] **Step 5: Remove only the invalid final-window mask**

Delete this line from `detect_energy`:

```python
active[-window:] = False
```

Do not change RMS indexing, candidate sample offsets, or the pipeline canonical peak ownership filter.

- [ ] **Step 6: Add a two-profile pipeline artifact regression**

In `tests/test_pipeline.py`, analyze the same synthetic source twice with identical detector output and contexts whose `processing_profile_id` values are `profile-a` and `profile-b`. Assert two distinct event IDs, two distinct final artifact directories, and deterministic IDs on a repeat of `profile-a`.

- [ ] **Step 7: Run detector and pipeline suites**

Run: `pytest tests/test_energy.py tests/test_fusion.py tests/test_pipeline.py tests/test_worker.py tests/test_artifacts.py -q`

Expected: middle, boundary, and final-window events pass; same-profile identity stays stable; different profiles do not conflict.

- [ ] **Step 8: Commit Task 7**

```bash
git add backend/src/das_sentinel/fusion.py backend/src/das_sentinel/pipeline.py backend/src/das_sentinel/worker/main.py backend/src/das_sentinel/detectors/energy.py tests/test_fusion.py tests/test_pipeline.py tests/test_worker.py tests/test_energy.py
git commit -m "fix: isolate events by processing profile"
```

---

### Task 8: Idempotent Reviews and Race-Free Event Detail UI

**Files:**
- Modify: `backend/src/das_sentinel/api/schemas.py`
- Modify: `backend/src/das_sentinel/api/routes/events.py`
- Modify: `backend/src/das_sentinel/persistence/repositories.py`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/pages/EventsPage.tsx`
- Modify: `frontend/src/components/ReviewForm.tsx`
- Test: `tests/test_events_api.py`
- Test: `frontend/src/pages/EventsPage.test.tsx`

**Interfaces:**
- Extends: `ReviewCreate.request_id: UUID | None`; old clients may omit it.
- Extends: `ReviewRepository.add(..., request_id: str | None = None)` and `get_by_request(event_id, request_id)`.
- Extends: frontend `api.get(path, params, signal?)` and internal request function with `AbortSignal`.
- Invariant: a repeated `(event_id, request_id)` returns the original review; a new request ID appends a new review.

- [ ] **Step 1: Write failing API idempotency tests**

```python
# tests/test_events_api.py
def test_review_request_id_is_idempotent_and_new_id_appends(event_api):
    client, _, _, token, ids = event_api
    event_id = ids["event_1"]
    request_id = "11111111-1111-4111-8111-111111111111"
    payload = {"label": "earthquake", "note": "clear", "request_id": request_id}
    first = client.post(
        f"/api/v1/events/{event_id}/reviews",
        headers={"X-CSRF-Token": token}, json=payload,
    )
    repeat = client.post(
        f"/api/v1/events/{event_id}/reviews",
        headers={"X-CSRF-Token": token}, json=payload,
    )
    second = client.post(
        f"/api/v1/events/{event_id}/reviews",
        headers={"X-CSRF-Token": token},
        json={**payload, "request_id": "22222222-2222-4222-8222-222222222222"},
    )
    assert repeat.status_code == 200
    assert repeat.json()["id"] == first.json()["id"]
    assert second.json()["id"] != first.json()["id"]
```

Change the route decorator to document both 201 for creation and 200 for a duplicate replay; return a `Response` status selected by `_append_review`.

- [ ] **Step 2: Run API test and verify it fails**

Run: `pytest tests/test_events_api.py::test_review_request_id_is_idempotent_and_new_id_appends -q`

Expected: strict schema rejects `request_id` or two reviews are inserted.

- [ ] **Step 3: Implement repository and API idempotency**

```python
class ReviewCreate(StrictSchema):
    label: Literal["earthquake", "blast", "traffic", "noise"]
    note: str | None = Field(default=None, max_length=2000)
    request_id: UUID | None = None
```

In the same write transaction, query by `(event_id, str(request_id))` first. Return the existing row without comparing mutable payload fields; the request ID identifies the original submission. On insert, persist the string request ID. Catch only a matching unique-constraint race, roll back to a fresh transaction, and return the existing row; do not hide unrelated `IntegrityError` values.

- [ ] **Step 4: Write failing inverse-response and double-submit UI tests**

```tsx
// frontend/src/pages/EventsPage.test.tsx
it("keeps the latest selected detail when responses finish in reverse order", async () => {
  const first = deferredResponse(DETAIL_ONE);
  const second = deferredResponse(DETAIL_TWO);
  useTwoEventHandlers(first.promise, second.promise);
  renderApp("/events");
  await userEvent.click(await screen.findByText("事件一"));
  await userEvent.click(await screen.findByText("事件二"));
  second.resolve();
  expect(await screen.findByText("事件二详情")).toBeVisible();
  first.resolve();
  expect(screen.queryByText("事件一详情")).not.toBeInTheDocument();
});


it("double click sends one review request with one UUID", async () => {
  const requests: unknown[] = [];
  useEventHandlers();
  server.use(http.post("/api/v1/events/event-1/reviews", async ({ request }) => {
    requests.push(await request.json());
    await delay(50);
    return HttpResponse.json(REVIEW, { status: 201 });
  }));
  renderApp("/events");
  await userEvent.click(await screen.findByText("2026-07-26 10:42:16"));
  await userEvent.selectOptions(await screen.findByLabelText("人工标签"), "earthquake");
  const save = screen.getByRole("button", { name: "保存复核" });
  await Promise.all([userEvent.click(save), userEvent.click(save)]);
  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({ request_id: expect.any(String) });
});
```

Define `deferredResponse`, `useTwoEventHandlers`, `DETAIL_ONE`, `DETAIL_TWO`, and `REVIEW` in the test file with concrete two-event summaries and MSW handlers; the deferred handler resolves only after its corresponding promise.

- [ ] **Step 5: Run frontend tests and verify they fail**

Run: `npm test -- --run src/pages/EventsPage.test.tsx`

Working directory: `frontend`

Expected: the late first response overwrites detail or the form issues two POSTs.

- [ ] **Step 6: Implement request sequencing and abort support**

Extend `request` and `api.get` with an optional `AbortSignal` and pass it to `fetch`. In `EventsPage`, keep a `useRef(0)` sequence and `useRef<AbortController | null>` controller. Each `openEvent` increments the sequence, aborts the old controller, and only sets detail/similar/error if its captured sequence still equals the current sequence. Abort on unmount. Render `ReviewForm` only when `selected.id === detail.id`.

```tsx
const requestSequence = useRef(0);
const activeController = useRef<AbortController | null>(null);

const openEvent = useCallback((event: EventSummary) => {
  const sequence = ++requestSequence.current;
  activeController.current?.abort();
  const controller = new AbortController();
  activeController.current = controller;
  setSelected(event);
  setDetail(null);
  setSimilar([]);
  void api.get<EventDetail>(`/api/v1/events/${event.id}`, undefined, controller.signal)
    .then((value) => {
      if (sequence === requestSequence.current) setDetail(value);
    })
    .catch((cause) => {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      if (sequence === requestSequence.current) setError("无法加载事件详情");
    });
}, []);
```

Apply the same guard to similar-event responses.

- [ ] **Step 7: Implement busy state and stable retry request ID**

In `ReviewForm`, add `busy` and `requestIdRef`. On a fresh submit, set `requestIdRef.current ??= crypto.randomUUID()`, set busy before awaiting, include `request_id`, and disable the button while busy. On success clear the ref so the next user edit creates a new ID; on failure retain it so an explicit retry uses the same ID. Reset the ref and saved/error state when `eventId` changes.

- [ ] **Step 8: Run backend and frontend integrity tests**

Run: `pytest tests/test_events_api.py tests/test_persistence.py -q`

Run: `npm test -- --run src/pages/EventsPage.test.tsx && npm run build`

Working directory for the second command: `frontend`

Expected: API duplicate replay produces one row, inverse responses never retarget the form, double clicks produce one POST, and TypeScript builds.

- [ ] **Step 9: Commit Task 8**

```bash
git add backend/src/das_sentinel/api/schemas.py backend/src/das_sentinel/api/routes/events.py backend/src/das_sentinel/persistence/repositories.py frontend/src/api.ts frontend/src/pages/EventsPage.tsx frontend/src/components/ReviewForm.tsx tests/test_events_api.py frontend/src/pages/EventsPage.test.tsx
git commit -m "fix: keep reviews bound to selected events"
```

---

### Task 9: Phase 1 CI, End-to-End Gates, and Operations Evidence

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `e2e/review-flow.spec.ts`
- Modify: `e2e/seed_stack.py`
- Modify: `docs/operations.md`
- Modify: `README.md`
- Test: full Python, frontend, Compose, CPU container, and GPU image build commands.

**Interfaces:**
- Produces: four CI jobs named `python`, `frontend`, `cpu-container`, and `gpu-image-build`.
- Consumes: Tasks 1-8 complete behavior and existing Playwright seed stack.
- Invariant: CI never reports a real CUDA smoke pass without a GPU runner.

- [ ] **Step 1: Add a failing CI contract test**

```python
# append to tests/test_compose.py
def test_ci_declares_required_release_jobs() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    assert set(workflow["jobs"]) >= {
        "python", "frontend", "cpu-container", "gpu-image-build"
    }
```

- [ ] **Step 2: Run the contract and verify it fails**

Run: `pytest tests/test_compose.py::test_ci_declares_required_release_jobs -q`

Expected: `.github/workflows/ci.yml` does not exist.

- [ ] **Step 3: Create the CI workflow with exact gates**

Configure these exact job steps; standard checkout/setup actions precede the shown `run` steps:

```yaml
python:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: "3.12" }
    - run: sudo apt-get update && sudo apt-get install -y build-essential curl unzip
    - run: docker/build-pysqlite3-wheel.sh "$RUNNER_TEMP/wheels"
    - run: python -m pip install "$RUNNER_TEMP"/wheels/*.whl
    - run: python -m pip install -e ".[dev]"
    - run: python -m pip check
    - run: python -c "from das_sentinel.persistence import validate_selected_sqlite_runtime; assert validate_selected_sqlite_runtime() == (3, 53, 3)"
    - run: ruff check .
    - run: pytest -q
frontend:
  runs-on: ubuntu-latest
  defaults: { run: { working-directory: frontend } }
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with: { node-version: "24", cache: npm, cache-dependency-path: frontend/package-lock.json }
    - run: npm ci
    - run: npm test -- --run
    - run: npm run build
cpu-container:
  runs-on: ubuntu-latest
  env: { COMPOSE_PROJECT_NAME: das-sentinel-ci }
  steps:
    - uses: actions/checkout@v4
    - run: docker build -f docker/Dockerfile.cpu -t das-sentinel:ci .
    - name: Write isolated Compose environment
      run: |
        mkdir -p "$RUNNER_TEMP/das-data"
        printf '%s\n' \
          "DAS_HOST_DATA_ROOT=$RUNNER_TEMP/das-data" \
          "DAS_SESSION_SECRET=ci-session-secret-0123456789abcdef" \
          "DAS_WORKER_TOKEN=ci-worker-token-0123456789abcdefghi" \
          "DAS_SECURE_COOKIES=false" \
          "DAS_HTTP_PORT=8000" > .env
    - run: docker compose -f compose.yaml -f compose.cpu.yaml config
    - run: docker compose -f compose.yaml -f compose.cpu.yaml up -d api
    - name: Wait for JSON health
      run: |
        for attempt in $(seq 1 60); do
          python -c "import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/healthz')) == {'status':'ok'}" && exit 0
          sleep 2
        done
        exit 1
    - run: docker compose -f compose.yaml -f compose.cpu.yaml exec -T api das-sentinel admin init --password ci-admin-password
    - run: docker compose -f compose.yaml -f compose.cpu.yaml up -d worker
    - run: docker compose -f compose.yaml -f compose.cpu.yaml exec -T api python -c "import pysqlite3.dbapi2 as db; assert db.sqlite_version == '3.53.3'"
    - if: always()
      run: docker compose -f compose.yaml -f compose.cpu.yaml logs --no-color
    - if: always()
      run: docker compose -f compose.yaml -f compose.cpu.yaml down --volumes --remove-orphans
gpu-image-build:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - run: docker build -f docker/Dockerfile.gpu -t das-sentinel-gpu:ci .
```

Do not invoke `smoke_test_cuda()` in the GPU build-only job.

- [ ] **Step 4: Extend Playwright flow for selection and duplicate review safety**

Change `write_event_file(path, start_time_utc)` and create two source files whose starts differ by one hour so their event occurrence strings are unique. Scan and drain both jobs. Add these Playwright scenarios after the existing login helper is factored into `login(page)`:

```ts
test("late detail response cannot replace the latest selection", async ({ page }) => {
  await login(page);
  const events = await page.evaluate(async () =>
    fetch("/api/v1/events?limit=100").then((response) => response.json()),
  ) as Array<{ id: string; occurred_at: string }>;
  expect(events.length).toBeGreaterThanOrEqual(2);
  const first = events[0];
  const second = events[1];
  await page.route(`**/api/v1/events/${first.id}`, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    await route.continue();
  });
  await page.getByText(formatQueueTime(first.occurred_at), { exact: true }).click();
  await page.getByText(formatQueueTime(second.occurred_at), { exact: true }).click();
  await expect(page.getByRole("heading", { name: formatQueueTime(second.occurred_at) })).toBeVisible();
  await page.waitForTimeout(700);
  await expect(page.getByRole("heading", { name: formatQueueTime(second.occurred_at) })).toBeVisible();
});


test("replaying one review request id appends one audit row", async ({ page }) => {
  await login(page);
  const result = await page.evaluate(async () => {
    const session = await fetch("/api/v1/session").then((response) => response.json());
    const [event] = await fetch("/api/v1/events?limit=1").then((response) => response.json());
    const before = await fetch(`/api/v1/events/${event.id}`).then((response) => response.json());
    const requestId = "33333333-3333-4333-8333-333333333333";
    const submit = () => fetch(`/api/v1/events/${event.id}/reviews`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": session.csrf_token },
      body: JSON.stringify({ label: "earthquake", note: "idempotent e2e", request_id: requestId }),
    });
    const responses = await Promise.all([submit(), submit()]);
    const after = await fetch(`/api/v1/events/${event.id}`).then((response) => response.json());
    return {
      statuses: responses.map((response) => response.status).sort(),
      added: after.review_history.length - before.review_history.length,
    };
  });
  expect(result.statuses).toEqual([200, 201]);
  expect(result.added).toBe(1);
});
```

Define `formatQueueTime(iso)` in the spec as `iso.replace("T", " ").replace(/(\.\d+)?Z$/, "")`, matching the UI's UTC test formatting.

- [ ] **Step 5: Update operations and README claims**

Document:

- CPU Compose first-boot command and `/healthz` JSON check.
- GPU image build separately from target-GPU runtime smoke.
- Published model path: train, publish, new claim binding, `fusion` method visibility.
- Repair flow for `invalid_model`: repair/publish, then rescan/re-enqueue; no automatic retry.
- Review request idempotency and audit append behavior.
- The React Router RSC advisory exception: BrowserRouter + REST does not use unstable RSC APIs; introducing RSC before an upgrade is prohibited.

- [ ] **Step 6: Run the complete non-performance verification**

Run: `ruff check .`

Run: `pytest -q`

Run: `npm ci && npm test -- --run && npm run build`

Working directory for the third command: `frontend`

Run when Playwright browsers are available: `npm ci && npx playwright test`

Working directory: `e2e`

Expected baseline: at least the prior `562 passed, 3 skipped, 1 deselected` plus all new tests; exact count may increase. No failures are acceptable.

- [ ] **Step 7: Run real container verification where available**

Run: `docker build -f docker/Dockerfile.cpu -t das-sentinel:phase1 .`

Run: `docker compose -f compose.yaml -f compose.cpu.yaml config`

Run the CI job's fresh-volume boot sequence and capture: image digest, container SQLite version, migration success, `/healthz` JSON, admin initialization, and worker polling log.

Run: `docker build -f docker/Dockerfile.gpu -t das-sentinel-gpu:phase1 .`

On a target NVIDIA host only: `docker run --rm --gpus all --entrypoint python das-sentinel-gpu:phase1 -c "from das_sentinel.signal.backend import smoke_test_cuda; smoke_test_cuda()"`

If Docker or NVIDIA hardware is unavailable, state exactly which command was not executed and do not convert static tests into runtime evidence.

- [ ] **Step 8: Request code review and fix only substantiated findings**

Invoke `superpowers:requesting-code-review` over the full diff from `339d2dc` to current HEAD. For every finding, reproduce it with a focused test or concrete trace before changing code. Re-run the affected task's focused suite after each correction.

- [ ] **Step 9: Commit Task 9**

```bash
git add .github/workflows/ci.yml e2e/review-flow.spec.ts e2e/seed_stack.py docs/operations.md README.md tests/test_compose.py
git commit -m "ci: gate DAS Sentinel phase one hardening"
```

---

## Phase 1 Completion Gate

Before starting Phase 2, invoke `superpowers:verification-before-completion` and collect fresh output for:

1. `ruff check .`
2. `pytest -q`
3. `cd frontend && npm test -- --run && npm run build`
4. CPU image build and fresh-volume Compose startup when Docker is available.
5. GPU image build when Docker is available; real CuPy smoke only on an NVIDIA runner.
6. `git diff --check 339d2dc..HEAD` and a review of only task-related files.

Do not mark the overall project complete after this gate. Continue with `docs/superpowers/plans/2026-08-06-das-sentinel-optimization-phase-2.md` for the measured performance target and deferred frontend work.
