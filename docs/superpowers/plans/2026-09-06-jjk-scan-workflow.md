# JJK Directory Scan Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blocking whole-site scan with an explicitly selected, recursively scanned JJK directory, live progress, duplicate-start protection, and one automatic stability recheck.

**Architecture:** Add safe root-relative directory navigation and discovery progress primitives, then wrap the existing stable-file tracker and idempotent enqueue path in an application-memory `ScanCoordinator`. The API returns a scan snapshot immediately and the React jobs page polls it while a focused directory dialog restricts every selection to the configured site root.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy/SQLite, `asyncio`, Pydantic 2, Pytest; React 19, TypeScript 5.8, Vite 7, Vitest, Testing Library, MSW.

## Global Constraints

- NAS access remains read-only; never create, modify, move, or delete JJK/HDF5 source data.
- The Web UI must require one non-root directory and recursively scan all of its real subdirectories.
- Directory input is a root-relative POSIX-style path returned by the server; reject absolute paths, `..`, symlinks, and Windows reparse points.
- A site can have at most one active scan; duplicate POST requests return the existing scan and never start another traversal.
- The automatic stability interval is exactly the `StableFileTracker` interval, default 10 seconds.
- Scan state remains in the single API process; no database table, migration, new process, or new dependency is allowed.
- API restart interruption is explicit and recoverable by starting a new scan.
- Existing file probing, fingerprinting, source persistence, processing-profile binding, job idempotency, event processing, GPU behavior, and analysis results must remain unchanged.
- No scan cancellation, multi-directory selection, arbitrary server filesystem browser, or dependency upgrade is in scope.

---

## File Structure

- Modify `backend/src/das_sentinel/services/discovery.py`: root-relative directory resolution/listing and per-pass traversal counters.
- Create `backend/src/das_sentinel/services/scans.py`: one in-memory scan state machine and background orchestration boundary.
- Modify `backend/src/das_sentinel/api/schemas.py`: strict directory and scan response contracts.
- Modify `backend/src/das_sentinel/api/routes/scans.py`: authenticated directory, start, active-scan, and status endpoints.
- Modify `backend/src/das_sentinel/api/app.py`: construct and close the scan coordinator.
- Modify `backend/src/das_sentinel/cli.py`: submit one automatic scan and poll its status.
- Modify `frontend/src/api.ts`: TypeScript contracts matching the API schemas.
- Create `frontend/src/components/ScanDirectoryDialog.tsx`: lazy, server-backed folder navigation.
- Create `frontend/src/components/ScanStatusCard.tsx`: pure phase-to-Chinese-status rendering.
- Modify `frontend/src/pages/JobsPage.tsx`: selection, start, disable, polling, restart recovery, and job refresh.
- Modify `frontend/src/styles.css`: scoped directory dialog and scan status presentation.
- Modify `tests/test_discovery.py`, `tests/test_scan_api.py`, `tests/test_cli.py`: backend RED/GREEN coverage.
- Create `tests/test_scan_service.py`: coordinator state-machine tests.
- Create `frontend/src/components/ScanDirectoryDialog.test.tsx` and `frontend/src/components/ScanStatusCard.test.tsx`: focused component tests.
- Modify `frontend/src/pages/JobsPage.test.tsx`: scan integration tests.
- Modify `docs/operations.md`: new Web and CLI operating procedure.

---

### Task 1: Safe directory navigation and discovery progress

**Files:**
- Modify: `backend/src/das_sentinel/services/discovery.py`
- Modify: `tests/test_discovery.py`

**Interfaces:**
- Produces: `DiscoveryProgress(checked_files: int, found_files: int)`.
- Produces: `ScanDirectoryEntry(name: str, path: str)` and `ScanDirectoryListing(path: str, parent_path: str | None, directories: tuple[ScanDirectoryEntry, ...])`.
- Produces: `resolve_scan_directory(site: DiscoverySite, relative_path: str, *, allow_root: bool = False) -> Path`.
- Produces: `list_scan_directories(site: DiscoverySite, relative_path: str | None = None) -> ScanDirectoryListing`.
- Extends: `discover_batch(..., recursive: bool | None = None, on_progress: Callable[[DiscoveryProgress], None] | None = None) -> tuple[DiscoveredFile, ...]`.
- Preserves: callers that omit the two new keyword arguments retain profile-controlled recursion and current return values.

- [ ] **Step 1: Write failing safe-navigation tests**

Add tests that create `root/jjk甲/day-01`, `root/ordinary`, a file at root, and—where the platform supports it—a symlink/reparse entry. Assert root listing is shallow, Unicode-safe, sorted, and returns only relative directory values:

```python
def test_scan_directory_listing_is_shallow_sorted_and_root_relative(tmp_path: Path):
    root = tmp_path / "root"
    (root / "jjk甲" / "day-01").mkdir(parents=True)
    (root / "ordinary").mkdir()
    (root / "ignore.txt").write_text("not a directory", encoding="utf-8")

    listing = list_scan_directories(_site(root))

    assert listing.path == ""
    assert listing.parent_path is None
    assert [(item.name, item.path) for item in listing.directories] == [
        ("jjk甲", "jjk甲"),
        ("ordinary", "ordinary"),
    ]
    nested = list_scan_directories(_site(root), "jjk甲")
    assert nested.parent_path == ""
    assert nested.directories == (ScanDirectoryEntry("day-01", "jjk甲/day-01"),)
```

Add parameterized rejection cases for `../outside`, `/absolute`, the root when `allow_root=False`, a missing folder, a regular file, and a symlink/reparse path. Assert exception text is controlled and does not include an outside absolute path.

- [ ] **Step 2: Run the navigation tests to verify RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py -k 'scan_directory' -q
```

Expected: collection or assertions fail because `ScanDirectoryEntry`, `ScanDirectoryListing`, `resolve_scan_directory`, and `list_scan_directories` do not exist.

- [ ] **Step 3: Implement minimal root-relative directory helpers**

Use `PurePosixPath` only for the wire value, join its validated parts onto `site.data_root`, and finish with the existing no-reparse/root-containment checks:

```python
@dataclass(frozen=True)
class ScanDirectoryEntry:
    name: str
    path: str


@dataclass(frozen=True)
class ScanDirectoryListing:
    path: str
    parent_path: str | None
    directories: tuple[ScanDirectoryEntry, ...]


def resolve_scan_directory(
    site: DiscoverySite,
    relative_path: str,
    *,
    allow_root: bool = False,
) -> Path:
    wire_path = PurePosixPath(relative_path)
    if wire_path.is_absolute() or ".." in wire_path.parts:
        raise DiscoveryPathError("scan directory must remain under the site root")
    parts = tuple(part for part in wire_path.parts if part not in ("", "."))
    if not parts and not allow_root:
        raise DiscoveryPathError("select a directory below the site root")
    root = _validated_root(site.data_root)
    return _validate_under_root(root, root.joinpath(*parts), directory=True)
```

`list_scan_directories` calls `resolve_scan_directory(..., allow_root=True)`, performs one `os.scandir`, omits non-directories and reparse points, sorts by `(casefold, original)`, and converts `Path.relative_to(root).parts` back to `PurePosixPath` strings. Convert enumeration `OSError` to a generic `DiscoveryPathError("cannot enumerate scan directory")` without the absolute path.

- [ ] **Step 4: Write failing traversal-counter tests**

Add three ordinary files, including two `.h5` candidates in a nested directory, and assert manual recursion overrides a profile with `recursive_discovery=False` while progress is monotonic:

```python
def test_discovery_reports_checked_and_h5_counts_with_recursive_override(tmp_path: Path):
    root = tmp_path / "root"
    selected = root / "selected"
    nested = selected / "nested"
    nested.mkdir(parents=True)
    write_zd_h5(selected / "a.h5")
    write_zd_h5(nested / "b.H5")
    (nested / "note.txt").write_text("x", encoding="utf-8")
    progress: list[DiscoveryProgress] = []

    discover_batch(
        _site(root, recursive=False),
        selected,
        tracker=StableFileTracker(),
        now=0,
        recursive=True,
        on_progress=progress.append,
    )

    assert progress[-1] == DiscoveryProgress(checked_files=3, found_files=2)
    assert progress == sorted(progress, key=lambda item: item.checked_files)
```

- [ ] **Step 5: Run the counter test to verify RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py -k 'reports_checked' -q
```

Expected: FAIL because `DiscoveryProgress` and the new `discover_batch` keyword arguments do not exist.

- [ ] **Step 6: Implement traversal progress without changing discovery output**

Add the immutable counter and update `_candidate_files` after each regular file inspection:

```python
@dataclass(frozen=True)
class DiscoveryProgress:
    checked_files: int = 0
    found_files: int = 0


def discover_batch(
    site: DiscoverySite,
    path: Path,
    *,
    tracker: StableFileTracker,
    now: float,
    recursive: bool | None = None,
    on_progress: Callable[[DiscoveryProgress], None] | None = None,
) -> tuple[DiscoveredFile, ...]:
    use_recursive = site.recursive_discovery if recursive is None else recursive
    candidates = _candidate_files(
        root,
        scan_path,
        recursive=use_recursive,
        on_progress=on_progress,
    )
```

Count every ordinary file as checked and increment `found_files` only for case-insensitive `.h5`. Call the callback with immutable cumulative values; do not count directories, reparse points, or other filesystem entries.

- [ ] **Step 7: Run focused and full discovery tests**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py -q
```

Expected: all discovery tests pass, including existing recursion, no-follow, replacement-race, fingerprint, and watch tests.

- [ ] **Step 8: Commit Task 1**

```powershell
git add backend/src/das_sentinel/services/discovery.py tests/test_discovery.py
git commit -m "feat(discovery): add safe scan directory navigation"
```

---

### Task 2: In-memory automatic scan coordinator

**Files:**
- Create: `backend/src/das_sentinel/services/scans.py`
- Create: `tests/test_scan_service.py`

**Interfaces:**
- Consumes: Task 1 `DiscoveryProgress`, `DiscoverySite`, `discover_batch`, `resolve_scan_directory`, `StableFileTracker`, and existing `enqueue_discovered`.
- Produces: `ScanPhase = Literal["initial_scan", "stability_wait", "verification_scan", "enqueueing", "completed", "failed"]`.
- Produces: immutable `ScanSnapshot` with the exact fields used by Python and TypeScript below.
- Produces: `ScanCoordinator.start(site, scan_path, relative_path) -> tuple[ScanSnapshot, bool]`, where the boolean means a new session was created.
- Produces: `get(scan_id)`, `active(site_id)`, and async `close()`.

- [ ] **Step 1: Write failing state-machine and automatic-recheck tests**

Create a temporary database/site and selected subdirectory with one valid H5. Inject a fake async wait that advances a fake monotonic clock, then start once and poll `get()` until terminal:

```python
def test_scan_runs_two_passes_and_enqueues_without_second_start(scan_context):
    async def scenario():
        coordinator, site, selected = scan_context
        first, created = coordinator.start(site, selected, "jjk/day")
        assert created is True
        terminal = await wait_for_terminal(coordinator, first.id)
        assert terminal.phase == "completed"
        assert terminal.initial_found_files == 1
        assert terminal.verification_found_files == 1
        assert terminal.stable_files == 1
        assert terminal.submitted_files == 1
        assert terminal.skipped_files == 0
        await coordinator.close()

    asyncio.run(scenario())
```

Add tests that pause the injected wait with an `asyncio.Event` and assert: the `stability_wait` snapshot exposes remaining seconds; a second start for the same site returns the same ID with `created is False`; another site gets a different ID; one changed or unsupported file increases `skipped_files` without failing stable files; an enumeration error produces `failed`; and both completion and failure make `active(site_id)` return `None`.

- [ ] **Step 2: Run coordinator tests to verify RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_scan_service.py -q
```

Expected: FAIL because `das_sentinel.services.scans` does not exist.

- [ ] **Step 3: Implement immutable snapshots and atomic start/get operations**

Use a `threading.Lock` only around small in-memory dictionary/snapshot operations. Keep one strong task reference per site and one most-recent snapshot per site:

```python
ScanPhase = Literal[
    "initial_scan",
    "stability_wait",
    "verification_scan",
    "enqueueing",
    "completed",
    "failed",
]


@dataclass(frozen=True)
class ScanSnapshot:
    id: str
    site_id: str
    path: str
    phase: ScanPhase
    initial_checked_files: int = 0
    initial_found_files: int = 0
    verification_checked_files: int = 0
    verification_found_files: int = 0
    stable_files: int = 0
    submitted_files: int = 0
    skipped_files: int = 0
    stability_seconds_remaining: int = 0
    error_message: str | None = None
```

`start()` must atomically return an existing nonterminal scan before allocating a UUID. For a new scan, store the initial snapshot, create exactly one `asyncio.Task`, save the task reference, and return `(snapshot, True)`.

- [ ] **Step 4: Implement the two-pass runner and countdown**

The runner creates a session-local tracker with the same stable interval, sends each blocking pass through `asyncio.to_thread`, and uses `dataclasses.replace` for phase/counter updates:

```python
tracker = StableFileTracker(stable_seconds=self.stable_seconds)
await asyncio.to_thread(
    discover_batch,
    site,
    scan_path,
    tracker=tracker,
    now=self.clock(),
    recursive=True,
    on_progress=lambda value: self._set_pass_progress(scan_id, "initial", value),
)
for remaining in range(math.ceil(self.stable_seconds), 0, -1):
    self._replace(scan_id, phase="stability_wait", stability_seconds_remaining=remaining)
    await self.sleep(min(1.0, self.stable_seconds))
```

Use a monotonic deadline so a fractional or delayed sleep never shortens the required tracker interval. The second pass supplies the same tracker and `recursive=True`. Set `stable_files = len(files)` and `skipped_files = max(0, verification_found_files - stable_files)` before enqueueing.

Run existing `_enqueue` logic inside `async with write_coordinator.lock`, using a new SQLAlchemy session in `asyncio.to_thread`; set `submitted_files = len(returned_snapshots)`. Catch `DiscoveryPathError`, `DiscoveryConfigurationError`, and unexpected exceptions, log the diagnostic stack, and store only one of these controlled messages:

```python
SCAN_READ_ERROR = "无法读取所选目录，请检查 NAS 连接和目录权限后重试"
SCAN_CONFIG_ERROR = "站点配置已变化或不可用，请检查站点设置后重试"
SCAN_UNKNOWN_ERROR = "扫描失败，请检查服务日志后重试"
```

Always remove the task from the active-site map in `finally`; retain the final snapshot by scan ID. `close()` cancels and awaits all active tasks so application shutdown does not leave warnings.

- [ ] **Step 5: Run coordinator tests and discovery regression**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_scan_service.py tests/test_discovery.py -q
```

Expected: both files pass; the automatic test proves one `start()` performs both observations and creates one idempotent job.

- [ ] **Step 6: Commit Task 2**

```powershell
git add backend/src/das_sentinel/services/scans.py tests/test_scan_service.py
git commit -m "feat(scans): add automatic background scan coordinator"
```

---

### Task 3: Directory and scan HTTP API

**Files:**
- Modify: `backend/src/das_sentinel/api/schemas.py`
- Modify: `backend/src/das_sentinel/api/routes/scans.py`
- Modify: `backend/src/das_sentinel/api/app.py`
- Modify: `tests/test_scan_api.py`

**Interfaces:**
- Consumes: Task 1 listing helpers and Task 2 `ScanCoordinator`/`ScanSnapshot`.
- Produces: `GET /api/v1/sites/{site_id}/scan-directories`.
- Produces: async `POST /api/v1/sites/{site_id}/scan` accepting `{"path": "root/relative"}`.
- Produces: `GET /api/v1/sites/{site_id}/active-scan` and `GET /api/v1/scans/{scan_id}`.
- Produces: JSON fields exactly matching the `ScanSnapshot` names from Task 2.

- [ ] **Step 1: Rewrite scan API tests for directory selection and async semantics**

Move fixture H5 files below `raw/jjk/day`, set the test coordinator interval to `0.01`, and replace the old “POST twice” test with:

```python
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
    assert len(client.get("/api/v1/jobs").json()) == 2
```

Add tests for shallow root/subdirectory responses; root/blank scan rejection; outside, absolute, missing, and regular-file path rejection; duplicate start returning the same ID and HTTP 200; active scan lookup; unknown scan 404; disabled/unknown site; GET authentication; POST authentication plus CSRF; and terminal empty-directory success.

- [ ] **Step 2: Run API tests to verify RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_scan_api.py -q
```

Expected: FAIL because the old route returns a job list synchronously and the three GET endpoints do not exist.

- [ ] **Step 3: Add strict API schemas**

Replace optional `ScanRequest.path` with a stripped nonblank field and add response models:

```python
class ScanRequest(StrictSchema):
    path: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
    ]


class ScanDirectoryEntryResponse(StrictSchema):
    name: str
    path: str


class ScanDirectoryResponse(StrictSchema):
    site_id: str
    site_name: str
    path: str
    parent_path: str | None
    directories: list[ScanDirectoryEntryResponse]


class ScanResponse(StrictSchema):
    id: str
    site_id: str
    path: str
    phase: Literal[
        "initial_scan", "stability_wait", "verification_scan",
        "enqueueing", "completed", "failed",
    ]
    initial_checked_files: int = Field(ge=0)
    initial_found_files: int = Field(ge=0)
    verification_checked_files: int = Field(ge=0)
    verification_found_files: int = Field(ge=0)
    stable_files: int = Field(ge=0)
    submitted_files: int = Field(ge=0)
    skipped_files: int = Field(ge=0)
    stability_seconds_remaining: int = Field(ge=0)
    error_message: str | None
```

- [ ] **Step 4: Implement routes with existing authentication and controlled errors**

Use `AuthenticatedAdmin` for all GETs and `CsrfAdmin` for POST. Load site/list directories in `run_in_threadpool`. Before starting, load the site and resolve the chosen non-root directory in one helper; map missing site to 404, disabled/configuration errors to 409, and path errors to 422. Return the coordinator snapshot through one `scan_response()` mapper:

```python
@router.post("/sites/{site_id}/scan", response_model=ScanResponse)
async def scan_site(
    site_id: str,
    payload: ScanRequest,
    response: Response,
    request: Request,
    admin: CsrfAdmin,
) -> ScanResponse:
    del admin
    site, scan_path = await run_in_threadpool(
        _load_scan_target, request.app.state.session_factory, site_id, payload.path
    )
    snapshot, created = request.app.state.scan_coordinator.start(
        site, scan_path, payload.path
    )
    response.status_code = 202 if created else 200
    return scan_response(snapshot)
```

Directory listing errors must not echo absolute paths. `GET active-scan` returns `ScanResponse | None`; `GET scans/{id}` returns 404 for an unknown/expired in-memory ID.

- [ ] **Step 5: Wire lifecycle construction and shutdown**

In `create_app`, construct one coordinator after `write_coordinator`:

```python
app.state.scan_coordinator = ScanCoordinator(
    app.state.session_factory,
    settings.app_data_dir / "artifacts",
    settings.pipeline_version,
    app.state.write_coordinator,
)
```

In `lifespan` `finally`, `await app.state.scan_coordinator.close()` before `engine.dispose()`. Remove `app.state.discovery_trackers`; automatic scans always use a session-local tracker.

- [ ] **Step 6: Run API, auth-lifecycle, and schema tests**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_scan_api.py tests/test_auth_api.py tests/test_discovery.py tests/test_scan_service.py -q
```

Expected: all selected tests pass, TestClient shutdown leaves no pending-task warning, and legacy discovery behavior remains green.

- [ ] **Step 7: Commit Task 3**

```powershell
git add backend/src/das_sentinel/api/schemas.py backend/src/das_sentinel/api/routes/scans.py backend/src/das_sentinel/api/app.py tests/test_scan_api.py
git commit -m "feat(api): expose asynchronous directory scan progress"
```

---

### Task 4: CLI uses the server-owned automatic scan

**Files:**
- Modify: `backend/src/das_sentinel/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/site-calibration.md`

**Interfaces:**
- Consumes: Task 3 scan POST and `GET /api/v1/scans/{scan_id}`.
- Produces: `das-sentinel scan SITE_ID PATH`, where PATH is a non-root site-relative path or an absolute path beneath the site root.
- Produces: `das-sentinel watch SITE_ID PATH` with the same required path rule.

- [ ] **Step 1: Write failing one-POST CLI tests**

Extend `FakeApi` so payload values may be response queues, then assert the CLI fetches the site, normalizes the path, POSTs exactly once, polls, and prints completion:

```python
def test_scan_posts_once_and_polls_until_automatic_recheck_finishes(monkeypatch):
    api = FakeApi(payloads={
        ("GET", "/api/v1/sites/site-1"): {"id": "site-1", "data_path": "C:/data"},
        ("POST", "/api/v1/sites/site-1/scan"): scan_snapshot("scan-1", "stability_wait"),
        ("GET", "/api/v1/scans/scan-1"): scan_snapshot(
            "scan-1", "completed", submitted_files=2
        ),
    })
    monkeypatch.setattr("das_sentinel.cli._connect", lambda: api)
    monkeypatch.setattr("das_sentinel.cli.time.sleep", lambda _: None)

    result = runner.invoke(app, ["scan", "site-1", "jjk/day"])

    assert result.exit_code == 0
    assert sum(path.endswith("/scan") for _, path, _ in api.requests) == 1
    assert "扫描完成" in result.stdout
```

Add tests that missing PATH shows usage error; root/`..` is rejected before POST; an absolute in-root path becomes a POSIX relative path; `failed` exits 1 with the server message; and `watch --once` starts and completes one server-owned scan without a second POST.

- [ ] **Step 2: Run CLI tests to verify RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_cli.py -k 'scan or watch' -q
```

Expected: FAIL because PATH is optional, `--settle-seconds` owns the wait, and the command submits two POST requests.

- [ ] **Step 3: Implement path normalization and status polling**

Make the positional path mandatory, remove `settle_seconds`, and add private helpers with exact responsibilities:

```python
ACTIVE_SCAN_PHASES = {
    "initial_scan", "stability_wait", "verification_scan", "enqueueing"
}


def _relative_scan_path(client: ApiClient, site_id: str, value: str) -> str:
    site = client.get(f"/api/v1/sites/{site_id}").json()
    root = Path(site["data_path"])
    candidate = Path(value)
    relative = candidate.relative_to(root) if candidate.is_absolute() else candidate
    if not relative.parts or ".." in relative.parts or relative == Path("."):
        raise typer.BadParameter("PATH 必须是站点根目录下的具体数据目录")
    return PurePosixPath(*relative.parts).as_posix()


def _wait_for_scan(client: ApiClient, snapshot: dict[str, Any]) -> dict[str, Any]:
    while snapshot["phase"] in ACTIVE_SCAN_PHASES:
        typer.echo(_scan_status_message(snapshot))
        time.sleep(1.0)
        snapshot = client.get(f"/api/v1/scans/{snapshot['id']}").json()
    return snapshot
```

Print the same Chinese phase meanings as the Web page. On `failed`, echo `error_message` and raise `typer.Exit(1)`. `watch` requires PATH, waits for each scan terminal state, then applies `poll_seconds`; `--once` exits after the first terminal state.

- [ ] **Step 4: Update the calibration command example**

In `docs/site-calibration.md`, retain `das-sentinel scan <site-id> <path>` but state PATH is required, must be below the site root, and the server automatically performs the 10-second stability recheck. Remove any instruction to call scan twice or scan a whole root.

- [ ] **Step 5: Run complete CLI tests**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_cli.py -q
```

Expected: every CLI test passes and the recorded requests prove exactly one scan POST per logical run.

- [ ] **Step 6: Commit Task 4**

```powershell
git add backend/src/das_sentinel/cli.py tests/test_cli.py docs/site-calibration.md
git commit -m "feat(cli): follow automatic scan sessions"
```

---

### Task 5: Server-backed scan directory dialog

**Files:**
- Modify: `frontend/src/api.ts`
- Create: `frontend/src/components/ScanDirectoryDialog.tsx`
- Create: `frontend/src/components/ScanDirectoryDialog.test.tsx`
- Modify: `frontend/src/styles.css`

**Interfaces:**
- Consumes: Task 3 directory endpoint.
- Produces: `ScanDirectoryListing`, `ScanDirectoryEntry`, and `ScanStatus` TypeScript interfaces matching Python field names.
- Produces: `ScanDirectoryDialog({site, onCancel, onSelect})`, where `onSelect(path: string)` can never receive the root empty string.

- [ ] **Step 1: Add TypeScript API contracts**

Add exact matching types to `frontend/src/api.ts`:

```ts
export interface ScanDirectoryEntry {
  name: string;
  path: string;
}

export interface ScanDirectoryListing {
  site_id: string;
  site_name: string;
  path: string;
  parent_path: string | null;
  directories: ScanDirectoryEntry[];
}

export type ScanPhase =
  | "initial_scan"
  | "stability_wait"
  | "verification_scan"
  | "enqueueing"
  | "completed"
  | "failed";

export interface ScanStatus {
  id: string;
  site_id: string;
  path: string;
  phase: ScanPhase;
  initial_checked_files: number;
  initial_found_files: number;
  verification_checked_files: number;
  verification_found_files: number;
  stable_files: number;
  submitted_files: number;
  skipped_files: number;
  stability_seconds_remaining: number;
  error_message: string | null;
}
```

- [ ] **Step 2: Write failing directory dialog tests**

Mock the root and `jjk20240608` directory endpoints with MSW. Test opening at root, entering a directory, returning to root, selecting only a non-root directory, canceling, loading state, and a controlled API error:

```tsx
it("navigates server directories and selects the current non-root path", async () => {
  render(<ScanDirectoryDialog site={SITE} onCancel={vi.fn()} onSelect={onSelect} />);
  await userEvent.click(await screen.findByRole("button", { name: "jjk20240608" }));
  expect(await screen.findByText("JJK NAS Archive / jjk20240608")).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "选择当前目录" }));
  expect(onSelect).toHaveBeenCalledWith("jjk20240608");
});
```

Assert the root “选择当前目录” button is disabled and that query parameters are values returned by the previous server response, not client-built absolute paths.

- [ ] **Step 3: Run component test to verify RED**

Run:

```powershell
npm test -- --run src/components/ScanDirectoryDialog.test.tsx
```

Expected: FAIL because the component does not exist.

- [ ] **Step 4: Implement the focused dialog**

Use the existing `.dialog-backdrop`/`.dialog` pattern, an `AbortController` per navigation request, and accessible labels:

```tsx
type Props = {
  site: Site;
  onCancel: () => void;
  onSelect: (path: string) => void;
};

export default function ScanDirectoryDialog({ site, onCancel, onSelect }: Props) {
  const [listing, setListing] = useState<ScanDirectoryListing | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  // load("") on mount; load(entry.path) on entry click; abort stale requests.
  return (
    <div className="dialog-backdrop">
      <div className="dialog scan-directory-dialog" role="dialog" aria-label={`选择 ${site.name} 扫描目录`}>
        {/* breadcrumb, parent button, directory buttons, error, cancel/select */}
        <button disabled={!listing?.path} onClick={() => listing && onSelect(listing.path)}>
          选择当前目录
        </button>
      </div>
    </div>
  );
}
```

Do not expose `site.data_path` or add a manual text field. Add only scoped styles for a scrollable directory list, breadcrumb wrapping, action spacing, disabled buttons, and a maximum dialog width appropriate for long Unicode names.

- [ ] **Step 5: Run dialog tests and production TypeScript build**

Run:

```powershell
npm test -- --run src/components/ScanDirectoryDialog.test.tsx
npm run build
```

Expected: dialog tests pass and TypeScript/Vite build completes without unused variables or type mismatches.

- [ ] **Step 6: Commit Task 5**

```powershell
git add frontend/src/api.ts frontend/src/components/ScanDirectoryDialog.tsx frontend/src/components/ScanDirectoryDialog.test.tsx frontend/src/styles.css
git commit -m "feat(web): add safe scan directory picker"
```

---

### Task 6: Jobs page progress, disabled state, polling, and recovery

**Files:**
- Create: `frontend/src/components/ScanStatusCard.tsx`
- Create: `frontend/src/components/ScanStatusCard.test.tsx`
- Modify: `frontend/src/pages/JobsPage.tsx`
- Modify: `frontend/src/pages/JobsPage.test.tsx`
- Modify: `frontend/src/styles.css`

**Interfaces:**
- Consumes: Task 5 `ScanStatus` and `ScanDirectoryDialog`.
- Produces: pure `isActiveScan(scan: ScanStatus) -> boolean` and `ScanStatusCard({scan})`.
- Persists: `sessionStorage["das-sentinel.active-scans"]` as a JSON object from site ID to scan ID only.

- [ ] **Step 1: Write failing status-card copy tests**

Create one table-driven test per phase, including exact required Chinese copy and counters:

```tsx
it.each([
  ["initial_scan", "正在扫描：已检查 1250 个文件，发现 86 个 H5 数据文件"],
  ["stability_wait", "请等待稳定性复检，剩余 8 秒，无需再次点击"],
  ["verification_scan", "正在进行稳定性复检：已复检 64 个文件，确认 58 个稳定文件"],
  ["enqueueing", "正在为稳定文件创建分析任务"],
])("renders %s", (phase, expected) => {
  render(<ScanStatusCard scan={{ ...SCAN, phase }} />);
  expect(screen.getByText(new RegExp(expected))).toBeVisible();
});
```

Add `completed` cases for nonzero and empty results, plus `failed` with the controlled server message.

- [ ] **Step 2: Run status-card tests to verify RED**

Run:

```powershell
npm test -- --run src/components/ScanStatusCard.test.tsx
```

Expected: FAIL because the component does not exist.

- [ ] **Step 3: Implement the pure status card**

Map each phase without network or timer logic. `completed` must display first/verification found counts, stable, submitted, and skipped; when both found counts are zero, render `未发现可处理的 H5 数据文件`. Give the changing message `role="status"` and failure `role="alert"`.

- [ ] **Step 4: Write failing Jobs page integration tests**

Expand MSW handlers and add tests proving:

```tsx
it("disables the site scan button and never posts twice while active", async () => {
  let posts = 0;
  // sites/jobs/directory handlers; POST returns INITIAL_SCAN; status remains active.
  renderApp("/jobs");
  await userEvent.click(await screen.findByRole("button", { name: "选择目录并扫描 JJK NAS Archive" }));
  await userEvent.click(await screen.findByRole("button", { name: "jjk20240608" }));
  await userEvent.click(screen.getByRole("button", { name: "选择当前目录" }));
  await userEvent.click(screen.getByRole("button", { name: "开始扫描" }));
  expect(screen.getByRole("button", { name: "正在扫描 JJK NAS Archive" })).toBeDisabled();
  expect(posts).toBe(1);
});
```

Also prove: POST body is `{"path":"jjk20240608"}`; polling changes wait → verification → completed; completion stops polling and refreshes `/jobs`; an active scan returned at initial page load disables the button; a stored ID returning 404 produces the restart-interruption message and clears storage; failure restores the selection button; other sites remain enabled; all timers are cleaned up on unmount.

- [ ] **Step 5: Run Jobs page tests to verify RED**

Run:

```powershell
npm test -- --run src/pages/JobsPage.test.tsx
```

Expected: FAIL because the page still POSTs `{}` directly, has no dialog, and has no scan state or polling.

- [ ] **Step 6: Implement scan state orchestration in JobsPage**

Maintain `Record<string, ScanStatus>` by site ID and one selected/pending directory. On site load, request each `/active-scan`; reconcile any IDs saved in session storage. Start only with the server-selected nonempty path:

```tsx
async function startScan(site: Site, path: string) {
  const scan = await api.post<ScanStatus>(`/api/v1/sites/${site.id}/scan`, { path });
  setScans((current) => ({ ...current, [site.id]: scan }));
  rememberActiveScan(site.id, scan.id);
}
```

Use one effect with a one-second timeout after each completed polling batch, not overlapping `setInterval` requests. For every active snapshot, fetch `/api/v1/scans/{id}`; on transition to terminal, clear its saved ID and call `refresh()` exactly once. On `ApiError(404)` for a previously saved scan ID, clear storage and show `扫描因服务重启而中断，请重新扫描`.

Render one button per site as:

```tsx
<button disabled={active} onClick={() => setChoosingSite(site)}>
  {active ? `正在扫描 ${site.name}` : `选择目录并扫描 ${site.name}`}
</button>
```

After directory selection, show the root-relative scope and a separate “开始扫描” confirmation. Never start merely by entering or selecting a directory. Render `ScanStatusCard` below its site controls.

- [ ] **Step 7: Add scoped Jobs page styles and run frontend regression**

Add `.scan-controls`, `.scan-scope`, `.scan-status-card`, `.scan-directory-list`, and disabled-button styles without changing unrelated artifact/event layout. Run:

```powershell
npm test -- --run
npm run build
```

Expected: all existing and new frontend tests pass, build succeeds, and the jobs page remains usable at narrow widths.

- [ ] **Step 8: Commit Task 6**

```powershell
git add frontend/src/components/ScanStatusCard.tsx frontend/src/components/ScanStatusCard.test.tsx frontend/src/pages/JobsPage.tsx frontend/src/pages/JobsPage.test.tsx frontend/src/styles.css
git commit -m "feat(web): show automatic scan progress"
```

---

### Task 7: Operations documentation, full verification, and local activation

**Files:**
- Modify: `docs/operations.md`
- Verify only: all files changed in Tasks 1–6
- Runtime artifact only: `frontend/dist/` (gitignored build output)

**Interfaces:**
- Consumes: complete backend, frontend, and CLI workflow.
- Produces: verified local service at `http://127.0.0.1:18000/jobs` using the feature worktree and production app-data settings.

- [ ] **Step 1: Document the exact operator workflow and restart limitation**

Add a Chinese “JJK 目录扫描” section to `docs/operations.md` with these steps:

1. Open 任务 → 选择目录并扫描站点；
2. enter the required concrete JJK/date directory and confirm the displayed relative scope;
3. click 开始扫描 once;
4. wait through 正在扫描 → 请等待稳定性复检 → 正在复检 → 创建任务 → 完成;
5. do not click again; the button is disabled and the second check is automatic;
6. if interrupted by API restart or NAS disconnect, reconnect and start a new scan;
7. CLI examples use `das-sentinel scan <site-id> <relative-path>` and never omit PATH.

State that actual source data is read-only and that a folder containing many minute files creates or reuses one analysis job per stable supported file.

- [ ] **Step 2: Run focused backend and frontend suites**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_discovery.py tests/test_scan_service.py tests/test_scan_api.py tests/test_cli.py -q
npm test -- --run src/components/ScanDirectoryDialog.test.tsx src/components/ScanStatusCard.test.tsx src/pages/JobsPage.test.tsx
```

Expected: every focused test passes with no pending task, unhandled request, or React `act()` warning.

- [ ] **Step 3: Run complete repository gates**

Run from the worktree root, then the frontend directory:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m ruff check .
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest -q
Set-Location frontend
npm test -- --run
npm run build
Set-Location ..
git diff --check
```

Expected: Ruff passes; backend is at least the 750-pass baseline plus new tests with only documented skips/performance deselection; all frontend tests and production build pass; diff check is clean.

- [ ] **Step 4: Inspect and activate the local API/frontend safely**

First inspect the existing listener and inherited environment without exposing secret values:

```powershell
& 'D:\das-sentinel-runtime\task4-final-runtime-restart.ps1' `
  -Repo 'D:\das-sentinel-runtime\waterfall-range-previews-git-20260901-2\.worktrees\jjk-scan-progress' `
  -NewDist 'D:\das-sentinel-runtime\waterfall-range-previews-git-20260901-2\.worktrees\jjk-scan-progress\frontend\dist' `
  -InspectOnly
```

Verify exactly one `uvicorn das_sentinel.api.app:create_runtime_app` listener owns port 18000, the inherited Python path/web-dist targets are expected, and secret/token lengths are 64. Then rerun the same command without `-InspectOnly`. The script performs health-gated rollback if the new API does not become healthy.

- [ ] **Step 5: Verify live routes and actual NAS directory browsing without bulk enqueue**

Read `/openapi.json` and assert it includes the directory, active-scan, and scan-status GET routes plus the POST scan route. In the authenticated in-app browser, open `/jobs`, open `JJK NAS Archive`, and navigate the real read-only tree to:

`jjk20240608 / 20240610`

Verify the root lists all ordinary directories, the breadcrumb and back button work, and selecting the current directory only stages the relative path. Do **not** click “开始扫描” on this production directory during verification: it contains many large minute files and would intentionally enqueue bulk analysis work.

- [ ] **Step 6: Exercise the complete two-pass UI against isolated test data**

Use the normal Pytest/TestClient and MSW integration suites from Steps 2–3 as the destructive-isolation boundary: they create temporary app-data, a temporary site root, and synthetic H5 files, then delete only their own temporary directories. Confirm the tests observed all four user requirements: selected subdirectory only, live counters/messages, disabled button, and automatic second check with exactly one POST.

This is the full behavioral acceptance path; live production NAS verification remains browse-only to honor the read-only/no-bulk-side-effect constraint.

- [ ] **Step 7: Commit documentation and record final evidence**

```powershell
git add docs/operations.md
git commit -m "docs: explain guided JJK directory scans"
git status --short --branch
git log --oneline --decorate -8
```

Expected: worktree is clean; the branch contains the design, plan, focused feature commits, and operations documentation. Do not push, merge, or delete the worktree unless the user separately requests it.

---

## Plan Self-Review

- **Spec coverage:** Task 1 covers safe all-directory browsing, explicit root-relative selection, recursive traversal, and counters. Tasks 2–3 cover immediate return, six phases, automatic tracker-owned recheck, duplicate-site protection, errors, lifecycle, and authenticated APIs. Tasks 5–6 cover the folder dialog, exact Chinese messages, disabled button, non-overlapping polling, refresh recovery, restart interruption, and job refresh. Task 4 keeps CLI behavior coherent. Task 7 covers documentation, full regression, safe live activation, actual NAS browsing, and isolated end-to-end scan proof.
- **Placeholder scan:** Every public type, endpoint, phase, counter, error message, test command, commit, and runtime check is specified; there are no deferred implementation markers or generic “handle errors” steps.
- **Type consistency:** Python `ScanSnapshot`, Pydantic `ScanResponse`, TypeScript `ScanStatus`, status card props, Jobs page state, and CLI dictionaries use the same 13 field names. `ScanDirectoryListing` and entry `path` values are root-relative POSIX strings in every layer.
- **Scope check:** No migration, scan cancellation, multi-directory selection, dependency change, worker protocol change, database scan history, arbitrary filesystem browser, or production NAS write is included.
