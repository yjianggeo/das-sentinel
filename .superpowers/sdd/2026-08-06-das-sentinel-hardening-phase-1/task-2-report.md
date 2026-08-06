# Task 2 Report: Safe SQLite Build, Pre-Migration Gate, and NAS Compose Defaults

## Scope and files changed

- Added `validate_selected_sqlite_runtime()` and invoked it as the first executable statement in `upgrade_database()`.
- Added `preflight_from_environment()` and the `python -m das_sentinel.runtime` entry module; the API entrypoint now preflights, migrates, then starts the Task 1 factory with `--factory`.
- Added the hashed, reusable SQLite 3.53.3 static `pysqlite3` wheel builder and its dedicated Docker stage. The runtime image installs that wheel before the application and asserts `SELECT sqlite_version()` returns `3.53.3`.
- Removed `pysqlite3-binary` from project dependencies.
- Changed Compose raw-data bind sources to `${DAS_HOST_DATA_ROOT}`, sets container `DAS_DATA_ROOT=/data`, uses loopback API binding, `unless-stopped` restart policies, and a JSON-aware health check. CPU and GPU workers receive the same NAS defaults.
- Updated `.env.example` and first-boot/NAS security operations guidance. `/data` is explicitly excluded from writes and database placement.
- Added focused persistence, runtime, Compose, and host-safe CLI test coverage. The test-only SQLite wrappers model the required 3.53.3 target because the prepared Windows verifier exposes SQLite 3.50.4; unsafe-driver tests retain the real rejection path.

## Local verification copy

Fresh local copy: `C:\Users\Administrator\AppData\Local\Temp\das-sentinel-task2-verify-20260806`.

It was mechanically seeded with `robocopy` from `C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-src-20260805` (excluding `.git`, `.pytest_cache`, and `node_modules`), then synchronized with all seven committed Task 1 files and each Task 2 test/production state before execution. Production edits remained solely in `Z:\das-event-platform\.worktrees\das-sentinel-hardening`.

Before final tests, `Get-FileHash -Algorithm SHA256` reported equality for all 21 selected Task 1/Task 2 inputs in the Z: worktree and local verifier:

```text
SHA256 equality verified for 21 selected inputs
```

## TDD RED/GREEN evidence

1. Selected-driver and entrypoint tests:

```text
RED: C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_persistence.py tests/test_runtime.py -q
ERROR collecting tests/test_persistence.py
ImportError: cannot import name 'validate_selected_sqlite_runtime'
```

After implementing the gate and entrypoint, the first GREEN run exposed the verifier's real SQLite `3.50.4` (`2 failed, 21 passed, 24 errors`) at the new intentional gate. Added test-only safe-version DBAPI wrappers for integration migration tests; the unsafe selected-driver test still overrides the driver to `3.51.1`.

```text
GREEN: C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_persistence.py tests/test_runtime.py -q
47 passed in 8.58s
```

2. Compose/image/defaults tests:

```text
RED: C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_compose.py -q
4 failed, 7 passed in 0.15s
```

The failures identified the old `DAS_DATA_ROOT` source, public port binding, missing host variable, and missing wheel script.

```text
GREEN: C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_compose.py -q
11 passed in 0.05s
```

The GPU defaults were then returned to their pre-change state, tested, and reapplied to preserve strict test-first ordering:

```text
RED: C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_compose.py -q
1 failed, 11 passed in 0.12s
GREEN: C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_compose.py -q
12 passed in 0.05s
```

3. Runtime preflight behavior:

```text
RED: C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_runtime.py -q
FAILED test_runtime_preflight_loads_settings_and_validates_selected_sqlite
AttributeError: module 'das_sentinel.runtime' has no attribute 'preflight_from_environment'
GREEN: C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_runtime.py -q
6 passed in 3.25s
```

4. CLI suite uses the same target-runtime test harness. Before that test-only adaptation, it correctly rejected the host's `3.50.4` runtime (`3 failed, 6 passed`); after adaptation:

```text
GREEN: C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_cli.py -q
9 passed in 2.19s
```

## Focused and final verification

```text
C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_persistence.py tests/test_runtime.py tests/test_compose.py tests/test_cli.py -q
69 passed in 9.51s
```

Docker availability check:

```text
docker command unavailable; Docker build and run verification not executed
```

Consequently, neither required Docker command was executed:

```text
docker build -f docker/Dockerfile.cpu -t das-sentinel:phase1 .
docker run --rm --entrypoint python das-sentinel:phase1 -c "import pysqlite3.dbapi2 as db; assert db.sqlite_version == '3.53.3'"
```

## Self-review

- Reviewed the path-scoped diff against the Task 2 brief: every changed production/configuration line is required by the listed SQLite preflight, static wheel, NAS Compose, environment, or operations requirements.
- Confirmed `upgrade_database()` validates the selected SQLAlchemy driver before URL/Alembic construction, and the API script runs preflight before migration.
- Confirmed the runtime still uses Python 3.12, retains the frontend stage and non-root user, and does not alter REST or single-writer architecture.
- Confirmed raw data remains read-only in all overlays; SQLite/artifacts remain on `/app-data`.

## Concern

Docker is unavailable on this host, so the static wheel/image build cannot be empirically executed here. The reproducible script includes the brief's exact source URLs, hashes, static build command, and in-image SQLite assertion; Python/static configuration coverage is green.

## Fix round 1: runtime mutation coverage and release gates

### Files changed

- `tests/test_runtime.py`: added `test_runtime_preflight_rejects_unsafe_selected_sqlite` and organized the import block required by Ruff.
- `tests/test_health.py`: applied the branch-local Ruff import ordering fix.

No production files changed. The test configures valid runtime settings, selects a real unsafe SQLite driver version (`3.51.1`) through `das_sentinel.persistence.db._sqlite_driver`, and asserts that `preflight_from_environment()` raises the real `SQLiteRuntimeError` containing `3.51.3`. If `validate_selected_sqlite_runtime()` is removed from `preflight_from_environment()`, the function returns settings and the test fails; this directly protects the pre-migration safety call rather than asserting a mock was called.

### Focused test and local-copy mutation check

Initial current-worktree behavior, after syncing `tests/test_runtime.py` to the local verifier:

```text
C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_runtime.py::test_runtime_preflight_rejects_unsafe_selected_sqlite -q
.                                                                        [100%]
1 passed in 3.02s
```

Only in `C:\Users\Administrator\AppData\Local\Temp\das-sentinel-task2-verify-20260806\backend\src\das_sentinel\runtime.py`, removed `validate_selected_sqlite_runtime()` from `preflight_from_environment()` and ran:

```text
C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_runtime.py::test_runtime_preflight_rejects_unsafe_selected_sqlite -q
FAILED test_runtime_preflight_rejects_unsafe_selected_sqlite
Failed: DID NOT RAISE SQLiteRuntimeError
1 failed in 3.39s
```

Restored the local runtime file mechanically from the Z: worktree and verified exact restored bytes before rerunning:

```text
runtime_restore_sha256=CE62BBF83C00C756ABF3182752D7F96D44330D8BCD45FD5654DBFCBDF2AA2547
C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_runtime.py::test_runtime_preflight_rejects_unsafe_selected_sqlite -q
1 passed in 3.01s
```

After the small Ruff-only import arrangement, the focused test and changed file lint both passed:

```text
C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_runtime.py::test_runtime_preflight_rejects_unsafe_selected_sqlite -q
1 passed in 3.06s
C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m ruff check tests/test_runtime.py
All checks passed!
```

### Hash equality and final Python verification

The current Task 1/Task 2 inputs plus all non-generated frontend sources were synchronized to the local verifier. SHA-256 equality was verified before final gates:

```text
SHA256 equality verified for 21 Task inputs and 29 branch frontend inputs
```

```text
C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m pytest tests/test_persistence.py tests/test_runtime.py tests/test_compose.py tests/test_cli.py -q
70 passed in 9.21s
```

### Ruff and frontend release gates

The repository Ruff gate was rerun in the hash-verified local verifier after the targeted branch-local `tests/test_health.py` import sort:

```text
C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-20260805\Scripts\python.exe -m ruff check .
All checks passed!
```

The changed runtime test also passes its focused Ruff gate (`ruff check tests/test_runtime.py`: `All checks passed!`).

The branch frontend sources and package lock (29 non-generated files) SHA-256 match the supplied installed-dependency mirror. With `C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin` prepended to `PATH`, these direct gates ran in `C:\Users\Administrator\AppData\Local\Temp\das-sentinel-audit-src-20260805\frontend`:

```text
node_modules\.bin\vitest.cmd run
Test Files  5 passed (5)
Tests  11 passed (11)

node_modules\.bin\tsc.cmd -b
exit 0

node_modules\.bin\vite.cmd build
vite v7.3.6 building client environment for production...
✓ 54 modules transformed.
✓ built in 791ms
frontend gates passed: vitest=0 tsc=0 vite=0
```

`npm ci` was not run because `npm` is unavailable; the existing installed dependency tree plus hash-matched frontend source/lock evidence was used for the equivalent direct test and build gates.

### Commit

Runtime safety/test commit: `6ede315 test: cover runtime SQLite preflight`.

### Concerns

- Docker remains unavailable as recorded above.
