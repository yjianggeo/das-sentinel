# DAS Sentinel Phase 2 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the fixed 1000-channel CPU benchmark below 60 seconds without changing results, then make event browsing, session handling, dashboard counts, and artifact interaction reliable for long-running NAS deployments.

**Architecture:** Split preprocessing into an immutable source preparation plus band-specific inverse transforms so each temporal source range performs bad-channel analysis, detrending, common-mode removal, and forward FFT once. Keep detector and fusion interfaces stable. Move collection-scale filtering and dashboard aggregation to existing FastAPI/SQLite query boundaries while the React client owns explicit loading/error/empty/retry state.

**Tech Stack:** Python 3.12, NumPy/CuPy FFT, Pytest performance markers, FastAPI/SQLAlchemy, React 19, TypeScript, Vitest/MSW, Playwright.

## Global Constraints

- Begin only after Phase 1's verification gate passes.
- Use the fixed benchmark shape `1000 channels × 1000 Hz × 60 seconds`, two repeat runs, mean wall time `< 60 seconds`, and peak RSS `< 4 GB`.
- Keep CPU/CUDA parity at `rtol=1e-4, atol=1e-5`.
- Candidate fields, fused event UUIDs, labels, artifact input hashes, and repeat-run output must remain unchanged.
- STA/LTA uses the history-extended block; energy, coherence, and trajectory use the logical block. Never share prepared spectra across those different time ranges.
- Optimize shared preparation and forward FFT first. If the measured mean remains at or above 60 seconds, stop and write a focused coherence optimization design before changing moveout logic.
- Do not add React Server Components while React Router remains on the documented BrowserRouter + REST advisory exception.
- Every production change follows a failing test, minimal implementation, focused verification, and commit.

---

## File Structure

- Modify `backend/src/das_sentinel/signal/preprocess.py`: add immutable `PreparedSignal`, `prepare_signal`, and `filter_prepared_signal`; keep `preprocess_block` as the compatibility wrapper.
- Modify `backend/src/das_sentinel/pipeline.py`: cache one prepared signal per temporal source range and derive per-band `PreprocessedBlock` values.
- Create `backend/src/das_sentinel/api/routes/dashboard.py`: authoritative aggregate counts and site-timezone “today” calculation.
- Modify existing frontend pages/components in place; do not replace the application shell or routing library.
- Extend `scripts/benchmark_baseline.py` and `docs/acceptance-report.md` for before/after evidence without committing generated HDF5 or artifact directories.

---

### Task 1: Share Base Preprocessing and Forward FFT Per Temporal Block

**Files:**
- Modify: `backend/src/das_sentinel/signal/preprocess.py`
- Modify: `backend/src/das_sentinel/pipeline.py`
- Test: `tests/test_preprocess.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_backend.py`
- Test: `tests/test_performance.py`

**Interfaces:**
- Produces: immutable `PreparedSignal(data, spectrum, bad_channel_mask, backend_name, sample_offset, sample_rate_hz, channel_distance_m)`.
- Produces: `prepare_signal(block: DasBlock, profile: ProcessingProfile, backend: ArrayBackend) -> PreparedSignal`.
- Produces: `filter_prepared_signal(prepared: PreparedSignal, low_hz: float, high_hz: float, backend: ArrayBackend) -> PreprocessedBlock`.
- Preserves: `preprocess_block(block, profile, backend)` by delegating to the two new functions.

- [ ] **Step 1: Write failing prepared-signal parity test**

```python
# tests/test_preprocess.py
def _shared_block() -> DasBlock:
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 0.1, size=(400, 8)).astype(np.float32)
    data[:, 7] = 0.0
    data[10, 6] = np.nan
    metadata = DasMetadata(
        source_path="synthetic.h5",
        start_time_utc=datetime(2026, 8, 6, tzinfo=UTC),
        sample_rate_hz=100.0,
        channel_distance_m=np.arange(8, dtype=np.float32) * 4.0,
        unit="strain_rate",
    )
    return DasBlock(data=data, metadata=metadata, sample_offset=1_000)


def test_prepare_then_filter_matches_preprocess_block():
    synthetic_block = _shared_block()
    profile = ProcessingProfile()
    backend = select_backend("cpu")
    expected = preprocess_block(synthetic_block, profile, backend)
    prepared = prepare_signal(synthetic_block, profile, backend)
    actual = filter_prepared_signal(
        prepared,
        profile.stalta_freq_low_hz,
        profile.stalta_freq_high_hz,
        backend,
    )
    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual.bad_channel_mask, expected.bad_channel_mask)
    assert actual.sample_offset == expected.sample_offset
```

- [ ] **Step 2: Write failing forward-FFT count test**

```python
# tests/test_pipeline.py
def test_detect_block_uses_one_forward_fft_per_temporal_source(monkeypatch):
    backend = select_backend("cpu")
    calls = 0
    original = backend.xp.fft.rfft

    def counted_rfft(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(backend.xp.fft, "rfft", counted_rfft)
    metadata = DasMetadata(
        source_path="synthetic.h5",
        start_time_utc=datetime(2026, 8, 6, tzinfo=UTC),
        sample_rate_hz=100.0,
        channel_distance_m=np.arange(8, dtype=np.float32) * 4.0,
        unit="strain_rate",
    )
    block = DasBlock(
        data=np.random.default_rng(42).normal(0.0, 0.1, size=(400, 8)).astype(np.float32),
        metadata=metadata,
        sample_offset=1_000,
    )
    pipeline._detect_block(
        block,
        ProcessingProfile(),
        backend,
        {name: 0.0 for name in pipeline._TIMING_NAMES},
        logical_start=block.sample_offset + 100,
    )
    assert calls == 2
```

The expected count is two: one history-extended STA/LTA source and one sliced logical source for the other detectors, regardless of their four distinct frequency bands.

- [ ] **Step 3: Run both tests and verify they fail**

Run: `pytest tests/test_preprocess.py::test_prepare_then_filter_matches_preprocess_block tests/test_pipeline.py::test_detect_block_uses_one_forward_fft_per_temporal_source -q`

Expected: new preprocessing functions are missing and current pipeline calls forward FFT once per band.

- [ ] **Step 4: Split preparation from band filtering**

```python
@dataclass(frozen=True)
class PreparedSignal:
    data: Any
    spectrum: Any
    bad_channel_mask: np.ndarray
    backend_name: str
    sample_offset: int
    sample_rate_hz: float
    channel_distance_m: np.ndarray
```

`prepare_signal` must perform, in this exact order: `find_bad_channels`, backend float32 copy, finite-value replacement, `_linear_detrend`, optional good-channel common-mode subtraction, then `xp.fft.rfft(data, axis=0)`. `filter_prepared_signal` must validate matching backend/range, apply `_raised_cosine_band_gain`, execute only `irfft`, apply `_robust_scale`, and return `PreprocessedBlock`.

```python
def preprocess_block(block, profile, backend):
    prepared = prepare_signal(block, profile, backend)
    return filter_prepared_signal(
        prepared,
        profile.stalta_freq_low_hz,
        profile.stalta_freq_high_hz,
        backend,
    )
```

- [ ] **Step 5: Cache by temporal source identity in `_detect_block`**

Create exactly two possible prepared entries keyed by `(sample_offset, sample_count)` from the existing `block` and `logical_block`. For each detector group, call `filter_prepared_signal` with its band. Delete each derived band block after its detectors finish; keep the prepared spectrum only until all bands for that source finish, then clear the backend cache once.

Do not change `_detector_band`, `_logical_detector_block`, detector call order, candidate order, or canonical peak ownership.

- [ ] **Step 6: Add output parity assertions**

Extend `tests/test_pipeline.py` to compare the post-refactor result to a checked-in deterministic expected structure containing candidate detector, start/end/peak, channel bounds, score rounded to 12 places, event UUID, label, and artifact input SHA-256. Extend `tests/test_backend.py` to compare CUDA and CPU `filter_prepared_signal` for two different bands using `rtol=1e-4, atol=1e-5`, skipping only when CUDA is genuinely unavailable.

- [ ] **Step 7: Run signal and pipeline suites**

Run: `pytest tests/test_preprocess.py tests/test_backend.py tests/test_stalta.py tests/test_energy.py tests/test_coherence.py tests/test_trajectory.py tests/test_pipeline.py -q`

Expected: exact structural parity, numerical tolerance parity, and two forward FFT calls for the two temporal source ranges.

- [ ] **Step 8: Commit Task 1**

```bash
git add backend/src/das_sentinel/signal/preprocess.py backend/src/das_sentinel/pipeline.py tests/test_preprocess.py tests/test_pipeline.py tests/test_backend.py tests/test_performance.py
git commit -m "perf: reuse DAS block preprocessing"
```

---

### Task 2: Measure the Fixed Benchmark and Enforce the Performance Gate

**Files:**
- Modify: `scripts/benchmark_baseline.py`
- Modify: `tests/test_performance.py`
- Modify: `docs/acceptance-report.md`

**Interfaces:**
- Produces: benchmark JSON fields `mean_wall_seconds`, `max_peak_rss_bytes`, and `parity`; existing per-run stage timings remain in the report.
- Consumes: Task 1 result structure and the existing deterministic HDF5 generator.

- [ ] **Step 1: Write failing benchmark summary tests**

```python
# tests/test_performance.py
def test_benchmark_summary_enforces_mean_memory_and_parity():
    runs = [
        {"wall_seconds": 55.0, "peak_rss_bytes": 900_000_000, "event_ids": ["a"], "event_labels": ["traffic"]},
        {"wall_seconds": 57.0, "peak_rss_bytes": 950_000_000, "event_ids": ["a"], "event_labels": ["traffic"]},
    ]
    summary = summarize_runs(runs)
    assert summary == {
        "mean_wall_seconds": 56.0,
        "max_peak_rss_bytes": 950_000_000,
        "parity": True,
    }


def test_benchmark_gate_rejects_slow_run():
    with pytest.raises(PerformanceGateError, match="60"):
        enforce_gate({"mean_wall_seconds": 60.0, "max_peak_rss_bytes": 1, "parity": True})
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_performance.py::test_benchmark_summary_enforces_mean_memory_and_parity tests/test_performance.py::test_benchmark_gate_rejects_slow_run -q`

Expected: summary and gate helpers do not exist.

- [ ] **Step 3: Implement deterministic summary and gate helpers**

```python
class PerformanceGateError(RuntimeError):
    pass


def summarize_runs(runs: list[dict]) -> dict:
    identities = {
        (tuple(run["event_ids"]), tuple(run["event_labels"])) for run in runs
    }
    return {
        "mean_wall_seconds": round(
            sum(run["wall_seconds"] for run in runs) / len(runs), 2
        ),
        "max_peak_rss_bytes": max(run["peak_rss_bytes"] for run in runs),
        "parity": len(identities) == 1,
    }


def enforce_gate(summary: dict) -> None:
    if not summary["parity"]:
        raise PerformanceGateError("repeat-run output parity failed")
    if summary["max_peak_rss_bytes"] >= 4 * 1024**3:
        raise PerformanceGateError("peak RSS must remain below 4 GB")
    if summary["mean_wall_seconds"] >= 60.0:
        raise PerformanceGateError("mean wall time must remain below 60 seconds")
```

Include event IDs in `run_once`, render the summary in Markdown and JSON, and add `--enforce` so the process exits nonzero on a failed gate.

- [ ] **Step 4: Run the fixed benchmark twice**

Run: `python scripts/benchmark_baseline.py --device cpu --out docs/acceptance-report.md --enforce`

Expected: two runs of exactly `1000 × 1000 Hz × 60 s`, identical event IDs/labels, maximum RSS below 4 GB, and mean below 60 seconds.

- [ ] **Step 5: Apply the conditional stop rule honestly**

If the command passes, commit the report and continue to Task 3. If the command fails only because mean wall time is at or above 60 seconds, do not alter coherence in this task: save the stage timings and open a new brainstorming/design document limited to `detect_coherence` moveout batching. Phase 2 remains incomplete until that focused design is approved and its implementation brings the same benchmark below 60 seconds.

- [ ] **Step 6: Commit Task 2 after a passing gate**

```bash
git add scripts/benchmark_baseline.py tests/test_performance.py docs/acceptance-report.md
git commit -m "test: enforce DAS CPU performance budget"
```

---

### Task 3: Server-Side Event Filters, Loading More, and Explicit Page States

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/pages/EventsPage.tsx`
- Modify: `frontend/src/components/EventQueue.tsx`
- Modify: `frontend/src/pages/EventsPage.test.tsx`
- Modify: `frontend/src/styles.css`
- Test: `tests/test_events_api.py`

**Interfaces:**
- Consumes: existing backend `/api/v1/events` query parameters `site_id`, `start_utc`, `end_utc`, `machine_label`, `reviewed`, `limit`, and `offset`.
- Produces: controlled `EventFilters` with `machineLabel` and `reviewed`; each filter change reloads offset 0 from the server.
- Produces: “加载更多” while the latest page contains 100 rows; append by event ID without duplication.

- [ ] **Step 1: Write failing server-filter and pagination UI test**

```tsx
// frontend/src/pages/EventsPage.test.tsx
it("sends filters to the API and loads the next page", async () => {
  const queries: URL[] = [];
  server.use(http.get("/api/v1/events", ({ request }) => {
    const url = new URL(request.url);
    queries.push(url);
    const offset = Number(url.searchParams.get("offset") ?? "0");
    return HttpResponse.json(offset === 0 ? makeSummaries(100) : makeSummaries(2, 100));
  }));
  renderApp("/events");
  await userEvent.selectOptions(await screen.findByLabelText("机器分类"), "traffic");
  await userEvent.selectOptions(screen.getByLabelText("复核状态"), "false");
  await userEvent.click(await screen.findByRole("button", { name: "加载更多" }));
  expect(queries.some((url) =>
    url.searchParams.get("machine_label") === "traffic" &&
    url.searchParams.get("reviewed") === "false"
  )).toBe(true);
  expect(queries.at(-1)?.searchParams.get("offset")).toBe("100");
  expect(screen.getAllByRole("button", { name: /^事件 / })).toHaveLength(102);
});
```

- [ ] **Step 2: Write failing error-before-loading test**

```tsx
it("shows a retryable list error instead of an endless loading state", async () => {
  server.use(http.get("/api/v1/events", () => HttpResponse.json({ detail: "failed" }, { status: 500 })));
  renderApp("/events");
  expect(await screen.findByRole("alert")).toHaveTextContent("无法加载事件");
  expect(screen.getByRole("button", { name: "重试" })).toBeVisible();
  expect(screen.queryByText("正在加载…")).not.toBeInTheDocument();
});
```

- [ ] **Step 3: Run the tests and verify they fail**

Run: `npm test -- --run src/pages/EventsPage.test.tsx`

Working directory: `frontend`

Expected: filters remain local to the first 100 events, no next-page request exists, and initial errors are hidden behind `events === null`.

- [ ] **Step 4: Implement controlled filters and page state**

Define:

```ts
export interface EventFilters {
  machineLabel: "" | EventSummary["machine_label"];
  reviewed: "" | "true" | "false";
}

type CollectionState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; items: EventSummary[]; hasMore: boolean; loadingMore: boolean };
```

Move filter state to `EventsPage`, pass values/callbacks into `EventQueue`, remove local `.filter`, and request `limit=100`, current filter query values, and an explicit offset. Give every queue item `aria-label={"事件 " + event.id}` for stable keyboard/test identity. On filter change, abort the old collection request, clear selected detail if it is no longer in the new results, and load offset 0. On “加载更多”, append only unseen IDs and set `hasMore = page.length === 100`.

- [ ] **Step 5: Preserve backend bounded pagination tests**

Run: `pytest tests/test_events_api.py::test_event_list_filters_and_stable_bounded_pagination tests/test_events_api.py::test_event_list_select_count_is_constant_across_page_sizes -q`

Expected: backend bounds and stable order remain unchanged.

- [ ] **Step 6: Run frontend tests and build**

Run: `npm test -- --run src/pages/EventsPage.test.tsx && npm run build`

Working directory: `frontend`

Expected: filter query, pagination, loading, error, empty, and retry states pass and TypeScript builds.

- [ ] **Step 7: Commit Task 3**

```bash
git add frontend/src/api.ts frontend/src/pages/EventsPage.tsx frontend/src/components/EventQueue.tsx frontend/src/pages/EventsPage.test.tsx frontend/src/styles.css tests/test_events_api.py
git commit -m "feat: page and filter the event queue"
```

---

### Task 4: Central 401 Handling and Authoritative Dashboard Aggregates

**Files:**
- Create: `backend/src/das_sentinel/api/routes/dashboard.py`
- Modify: `backend/src/das_sentinel/api/app.py`
- Modify: `backend/src/das_sentinel/api/schemas.py`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/auth.tsx`
- Modify: `frontend/src/pages/DashboardPage.tsx`
- Create: `frontend/src/pages/DashboardPage.test.tsx`
- Modify: `frontend/src/pages/LoginPage.test.tsx`
- Create: `tests/test_dashboard_api.py`

**Interfaces:**
- Produces: `GET /api/v1/dashboard` returning `job_counts`, `today_event_count`, and `machine_label_counts`.
- Produces: `setUnauthorizedHandler(handler: (() => void) | null)` in frontend API; it fires once for an authenticated request returning 401.
- Invariant: logout failure leaves authenticated state intact and visible as an error; a confirmed server 401 clears session state centrally.

- [ ] **Step 1: Write failing dashboard timezone/complete-count tests**

```python
# tests/test_dashboard_api.py
@pytest.fixture
def dashboard_api(tmp_path, monkeypatch):
    data = tmp_path / "data"
    app_data = tmp_path / "app-data"
    data.mkdir()
    app_data.mkdir()
    settings = Settings(
        data_root=data,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'dashboard.sqlite3'}",
        session_secret="s" * 32,
        worker_token="w" * 40,
        secure_cookies=False,
    )
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    with sessions.begin() as session:
        Repositories(session).admins.add(password_hash=PasswordHasher().hash("correct horse"))
    now_utc = datetime(2026, 8, 6, 0, 30, tzinfo=UTC)
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    monkeypatch.setattr("das_sentinel.api.routes.dashboard.utc_now", lambda: now_utc)
    with TestClient(create_app(settings)) as client:
        client.post("/api/v1/session", json={"password": "correct horse"})
        yield client, sessions, now_utc
    engine.dispose()


def _seed_dashboard_site(
    sessions, *, name: str, timezone: str, occurred_at: datetime,
    label: str, count: int = 1,
) -> None:
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name=name,
            data_path=f"/data/{name}",
            timezone=timezone,
            cable_type="urban",
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=ProcessingProfile().model_dump(mode="json"),
            is_current=True,
        )
        source = repositories.source_files.add(
            site_id=site.id,
            path=f"/data/{name}/source.h5",
            fingerprint=hashlib.sha256(name.encode()).hexdigest(),
            size_bytes=1,
            mtime_ns=1,
        )
        run = repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="v1",
            requested_backend="cpu",
            actual_backend="cpu",
            status="succeeded",
        )
        for index in range(count):
            event = repositories.events.add(
                run_id=run.id,
                event_uid=f"{name}-{index}",
                start_sample=index * 10,
                end_sample=index * 10 + 5,
                channel_start=0,
                channel_stop=2,
                occurred_at=occurred_at + timedelta(seconds=index),
            )
            repositories.events.add_classification(
                event.id,
                label=label,
                decision_score=0.9,
                method="rules",
            )


def test_dashboard_uses_each_site_timezone_and_all_events(dashboard_api):
    client, sessions, now_utc = dashboard_api
    _seed_dashboard_site(
        sessions, name="shanghai", timezone="Asia/Shanghai",
        occurred_at=now_utc, label="earthquake",
    )
    _seed_dashboard_site(
        sessions, name="los-angeles", timezone="America/Los_Angeles",
        occurred_at=now_utc, label="noise",
    )
    _seed_dashboard_site(
        sessions, name="history", timezone="UTC",
        occurred_at=now_utc - timedelta(days=2), label="traffic", count=101,
    )
    response = client.get("/api/v1/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert payload["today_event_count"] == 2
    assert payload["machine_label_counts"]["traffic"] == 101
```

Import the concrete dependencies used above in the new test file. Calling route-local `utc_now()` makes the monkeypatched instant cover two local calendar dates without relying on wall clock.

- [ ] **Step 2: Run the dashboard test and verify it fails**

Run: `pytest tests/test_dashboard_api.py -q`

Expected: dashboard route is absent.

- [ ] **Step 3: Implement aggregate schema and route**

```python
class DashboardResponse(StrictSchema):
    job_counts: dict[str, int]
    today_event_count: int
    machine_label_counts: dict[str, int]
```

Use SQL grouped counts for job status and latest classification labels over the full event table. For today count, load enabled sites, compute each site's local midnight boundaries with `ZoneInfo(site.timezone)`, convert boundaries to UTC, and count joined events for that site in `[start_utc, end_utc)`. Register the authenticated route under `/api/v1`; never derive counts from a limited event list.

- [ ] **Step 4: Write failing frontend 401/logout/dashboard tests**

```tsx
// frontend/src/pages/DashboardPage.test.tsx
it("renders authoritative dashboard counts", async () => {
  server.use(http.get("/api/v1/dashboard", () => HttpResponse.json({
    job_counts: { queued: 3, running: 2, failed: 1 },
    today_event_count: 145,
    machine_label_counts: { traffic: 120, earthquake: 25 },
  })));
  renderApp("/");
  expect(await screen.findByText("145")).toBeVisible();
  expect(screen.getByText(/车辆：120/)).toBeVisible();
});
```

```tsx
// frontend/src/pages/LoginPage.test.tsx
it("keeps the session when logout fails", async () => {
  server.use(http.delete("/api/v1/session", () => HttpResponse.json({ detail: "failed" }, { status: 500 })));
  renderApp("/");
  await userEvent.click(await screen.findByRole("button", { name: "退出" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("退出失败");
  expect(screen.queryByRole("button", { name: "登录" })).not.toBeInTheDocument();
});


it("returns to login once when an authenticated request becomes unauthorized", async () => {
  let dashboardRequests = 0;
  server.use(http.get("/api/v1/dashboard", () => {
    dashboardRequests += 1;
    return HttpResponse.json({ detail: "Unauthorized" }, { status: 401 });
  }));
  renderApp("/");
  expect(await screen.findByRole("button", { name: "登录" })).toBeVisible();
  expect(dashboardRequests).toBe(1);
});
```

- [ ] **Step 5: Implement centralized unauthorized handling**

Store one module-level unauthorized callback in `api.ts`. After parsing an HTTP 401, invoke it before throwing `ApiError`. Register/unregister the callback in `AuthProvider`. Change `logout()` so it clears CSRF/authenticated state only after a successful DELETE; expose logout error state to the app shell instead of using `finally`.

- [ ] **Step 6: Replace client-side dashboard derivation**

`DashboardPage` makes one `/api/v1/dashboard` request and renders the response. Remove the jobs/events `Promise.all`, local UTC date calculation, 100-event wording, and `useMemo` aggregation.

- [ ] **Step 7: Run backend/frontend tests and build**

Run: `pytest tests/test_dashboard_api.py tests/test_auth_api.py tests/test_events_api.py -q`

Run: `npm test -- --run src/pages/DashboardPage.test.tsx src/pages/LoginPage.test.tsx && npm run build`

Working directory for the second command: `frontend`

Expected: site-local today counts, all-event label counts, centralized 401 transition, and failed-logout state pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add backend/src/das_sentinel/api/routes/dashboard.py backend/src/das_sentinel/api/app.py backend/src/das_sentinel/api/schemas.py frontend/src/api.ts frontend/src/auth.tsx frontend/src/pages/DashboardPage.tsx frontend/src/pages/DashboardPage.test.tsx frontend/src/pages/LoginPage.test.tsx tests/test_dashboard_api.py
git commit -m "fix: make dashboard and session state authoritative"
```

---

### Task 5: Accessible Artifact Interaction and Visible Image Failures

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/components/ArtifactViewer.tsx`
- Create: `frontend/src/components/ArtifactViewer.test.tsx`
- Modify: `frontend/src/pages/EventsPage.tsx`
- Modify: `frontend/src/pages/EventsPage.test.tsx`
- Modify: `frontend/src/styles.css`
- Test: `e2e/review-flow.spec.ts`

**Interfaces:**
- Produces: WAI-ARIA tab behavior with ArrowLeft/ArrowRight/Home/End, roving `tabIndex`, and linked `tabpanel`.
- Produces: modal image dialog that focuses its close button, closes on Escape, restores trigger focus, and prevents click-through.
- Produces: `api.getBlob(path, signal?) -> Promise<Blob>` so artifact HTTP 401/409 remains observable.
- Produces: per-artifact load/decode error with an explicit “重试图件” request.

- [ ] **Step 1: Write failing keyboard and image-error tests**

```tsx
// frontend/src/components/ArtifactViewer.test.tsx
it("supports arrow-key tabs and an accessible tabpanel", async () => {
  render(<ArtifactViewer artifacts={ARTIFACTS} />);
  const waterfall = screen.getByRole("tab", { name: "瀑布图" });
  waterfall.focus();
  await userEvent.keyboard("{ArrowRight}");
  expect(screen.getByRole("tab", { name: "PSD" })).toHaveFocus();
  expect(screen.getByRole("tabpanel")).toHaveAccessibleName("PSD");
});


it("shows and retries an image load failure", async () => {
  let requests = 0;
  server.use(http.get("/api/v1/artifacts/art-1", () => {
    requests += 1;
    return new HttpResponse(new Blob(["png"], { type: "image/png" }));
  }));
  render(<ArtifactViewer artifacts={ARTIFACTS} />);
  fireEvent.error(await screen.findByAltText("事件瀑布图"));
  expect(screen.getByRole("alert")).toHaveTextContent("图件加载失败");
  await userEvent.click(screen.getByRole("button", { name: "重试图件" }));
  await waitFor(() => expect(requests).toBe(2));
});
```

In the test setup, run an MSW server, define concrete `ARTIFACTS` containing `art-1` waterfall and `art-2` PSD entries, and stub `URL.createObjectURL`/`URL.revokeObjectURL` with deterministic `blob:artifact-N` values.

- [ ] **Step 2: Write failing fullscreen focus test**

```tsx
it("traps fullscreen entry and restores trigger focus on Escape", async () => {
  render(<ArtifactViewer artifacts={ARTIFACTS} />);
  const trigger = screen.getByRole("button", { name: "全屏查看" });
  await userEvent.click(trigger);
  expect(screen.getByRole("button", { name: "关闭全屏" })).toHaveFocus();
  await userEvent.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});
```

- [ ] **Step 3: Run tests and verify they fail**

Run: `npm test -- --run src/components/ArtifactViewer.test.tsx`

Working directory: `frontend`

Expected: tabs do not move by keyboard, dialog has no close focus/Escape restoration, and image errors are invisible.

- [ ] **Step 4: Implement observable blob loading, tabs, dialog, and retry state**

Add `api.getBlob` using the same credential, 401 callback, and `ApiError` path as JSON requests:

```ts
async function requestBlob(path: string, signal?: AbortSignal): Promise<Blob> {
  const response = await fetch(new URL(path, window.location.origin), {
    method: "GET",
    credentials: "same-origin",
    signal,
  });
  if (!response.ok) {
    throw await apiErrorFromResponse(response);
  }
  return response.blob();
}
```

`apiErrorFromResponse` is the single helper used by JSON and blob requests; it invokes Task 4's unauthorized callback on 401 before returning `ApiError`.

Give each tab stable IDs `artifact-tab-${kind}` and panels `artifact-panel-${kind}`. Only the active tab has `tabIndex={0}`. Key handling computes the next index and calls both `setActiveKind` and `.focus()`.

For fullscreen, render an explicit close button inside the dialog, record the trigger ref, focus close on open, listen for Escape only while open, restore trigger on close, and call `event.stopPropagation()` on dialog content. Set `aria-modal="true"` and label the dialog with the active artifact name.

Track an artifact load state and `retryNumber`. An effect aborts the prior request, calls `api.getBlob(artifactUrl(active.id), signal)`, creates an object URL only for the latest active artifact/retry, and revokes it on replacement/unmount. An `ApiError(409)` renders “图件完整性校验失败，请联系管理员”; other non-401 HTTP/decode failures render “图件加载失败”. Image `onError` enters the decode-failure state. “重试图件” increments the counter and performs a new HTTP request. Reset error when the active artifact changes.

- [ ] **Step 5: Cover 401 and 409 artifact-visible errors**

Add these event-page MSW regressions:

```tsx
it("returns to login when an artifact request becomes unauthorized", async () => {
  useEventHandlers();
  server.use(http.get("/api/v1/artifacts/art-1", () =>
    HttpResponse.json({ detail: "Unauthorized" }, { status: 401 })
  ));
  renderApp("/events");
  await userEvent.click(await screen.findByText("2026-07-26 10:42:16"));
  expect(await screen.findByRole("button", { name: "登录" })).toBeVisible();
});


it("keeps event detail visible when artifact integrity fails", async () => {
  useEventHandlers();
  server.use(http.get("/api/v1/artifacts/art-1", () =>
    HttpResponse.json({ detail: "Artifact integrity check failed" }, { status: 409 })
  ));
  renderApp("/events");
  await userEvent.click(await screen.findByText("2026-07-26 10:42:16"));
  expect(await screen.findByText("图件完整性校验失败，请联系管理员")).toBeVisible();
  expect(screen.getByRole("button", { name: "重试图件" })).toBeVisible();
  expect(screen.getByText("规则判断")).toBeVisible();
});
```

- [ ] **Step 6: Run frontend and Playwright verification**

Run: `npm test -- --run && npm run build`

Working directory: `frontend`

Run when browsers are available: `npm test`

Working directory: `e2e`

Expected: all component/page tests pass; Playwright confirms tab keyboard navigation, Escape close, focus restoration, and visible image failure.

- [ ] **Step 7: Commit Task 5**

```bash
git add frontend/src/api.ts frontend/src/components/ArtifactViewer.tsx frontend/src/components/ArtifactViewer.test.tsx frontend/src/pages/EventsPage.tsx frontend/src/pages/EventsPage.test.tsx frontend/src/styles.css e2e/review-flow.spec.ts
git commit -m "fix: make event artifacts accessible and recoverable"
```

---

## Phase 2 Completion Gate

Invoke `superpowers:verification-before-completion` and run fresh:

1. `ruff check .`
2. `pytest -q`
3. `python scripts/benchmark_baseline.py --device cpu --out docs/acceptance-report.md --enforce`
4. `cd frontend && npm test -- --run && npm run build`
5. `cd e2e && npm test` when Playwright browsers are available.
6. CPU Compose fresh-volume startup and GPU image/runtime evidence under the same availability rules as Phase 1.
7. `git diff --check 339d2dc..HEAD` and `superpowers:requesting-code-review` over the complete hardening branch.

Completion requires the benchmark's measured mean below 60 seconds and peak RSS below 4 GB. If Task 2 reaches the coherence design stop rule, the project remains in progress until that approved focused plan passes the same gate.
