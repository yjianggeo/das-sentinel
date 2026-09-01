# Waterfall Range Preview and Filter Band Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show each filtered waterfall's actual detector frequency band and let an authenticated reviewer generate temporary raw/filtered waterfall previews for an exact Distance interval without changing audit data.

**Architecture:** Extract detector-band and Distance-selection rules into a pure shared module used by the processing pipeline and a new preview service. Keep NAS I/O and preview rendering behind dedicated authenticated event endpoints; keep temporary PNGs only as browser Blob URLs managed by a focused React hook. Existing event details and immutable artifacts remain available when the source file is offline.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2, NumPy/SciPy, h5py, Matplotlib, React 19, TypeScript, Vitest/Testing Library, Playwright, SQLite, NAS/UNC paths.

## Global Constraints

- Preserve the IANA timezone value and user-facing text `Asia/Shanghai`; do not introduce `Asia/Beijing` or change UTC/local-time semantics.
- Do not add a database migration, dependency, persistent preview file, cache, or derived artifact row.
- The event's processing-profile snapshot is authoritative; never use the site's current profile for a historical event.
- A preview may change only `waterfall_raw` and `waterfall_filtered`; waveform, PSD, STFT, detection, fusion, features, classification, and reviews remain unchanged.
- A preview request reads one fixed event time window once and returns both PNGs in one response.
- Revalidate the source path below `Settings.data_root` and compare its canonical discovery fingerprint before and after reading.
- Preserve the user's pre-existing uncommitted `frontend/tsconfig.tsbuildinfo`; never stage, revert, or overwrite it.
- Add no broad refactor. Every changed production line must serve filter-band labelling, temporary Distance preview, required resource cleanup, or explicit preservation of station-timezone formatting.

---

## File Structure

- Create `backend/src/das_sentinel/waterfalls.py`: pure detector-band resolution, Distance-to-channel selection, and the shared preview preprocessing entry point.
- Modify `backend/src/das_sentinel/pipeline.py`: consume the shared waterfall rules instead of private duplicates and pass filter context into permanent artifact rendering.
- Modify `backend/src/das_sentinel/artifacts.py`: accept explicit waterfall plot context, put the filter band in filtered PNG titles, and expose an in-memory waterfall renderer used by previews.
- Create `backend/src/das_sentinel/services/waterfall_previews.py`: load an immutable event/run/profile/source binding, read metadata or one bounded signal block, validate fingerprints, and return option/preview values without writes.
- Create `backend/src/das_sentinel/api/routes/waterfall_previews.py`: authenticated GET options and CSRF-protected POST preview endpoints.
- Modify `backend/src/das_sentinel/api/schemas.py` and `backend/src/das_sentinel/api/app.py`: strict request/response types and router/service registration.
- Modify `frontend/src/api.ts`: preview types and abortable POST support.
- Create `frontend/src/components/useWaterfallPreview.ts`: option loading, submit/reset, Base64-to-Blob conversion, abort handling, stale-response rejection, and URL cleanup.
- Create `frontend/src/components/WaterfallRangeControls.tsx`: accessible numeric form and status/error presentation.
- Modify `frontend/src/components/ArtifactViewer.tsx`: use preview URLs only for the two waterfall kinds, display the band/detector, and keep fullscreen/download behavior aligned with the active image.
- Modify `frontend/src/pages/EventsPage.tsx`: pass `eventId` to ArtifactViewer and explicitly preserve `detail.site_timezone` in the detail heading formatter.
- Modify `frontend/src/styles.css`: compact responsive control-row styling.
- Test with `tests/test_waterfalls.py`, `tests/test_artifacts.py`, `tests/test_pipeline.py`, `tests/test_waterfall_preview_api.py`, `frontend/src/api.test.ts`, `frontend/src/components/ArtifactViewer.test.tsx`, `frontend/src/pages/EventsPage.test.tsx`, and `e2e/review-flow.spec.ts`.

---

### Task 1: Shared waterfall semantics and permanent filter-band titles

**Files:**
- Create: `backend/src/das_sentinel/waterfalls.py`
- Modify: `backend/src/das_sentinel/pipeline.py:480-545,250-355`
- Modify: `backend/src/das_sentinel/artifacts.py:25-48,147-170,322-370`
- Create: `tests/test_waterfalls.py`
- Modify: `tests/test_artifacts.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `WaterfallFilter(detector: str, low_hz: float, high_hz: float)`.
- Produces: `resolve_waterfall_filter(evidence: Sequence[Candidate], profile: ProcessingProfile) -> WaterfallFilter`.
- Produces: `channel_slice_for_distance(distances_m: np.ndarray, start_m: float, stop_m: float) -> slice`.
- Produces: `usable_channel_indices(distances_m: np.ndarray, bad_channel_mask: np.ndarray, start_m: float, stop_m: float) -> np.ndarray`.
- Produces: `preprocess_for_waterfall(block: DasBlock, profile: ProcessingProfile, backend: ArrayBackend, filter_context: WaterfallFilter) -> PreprocessedBlock`.
- Produces: `detector_display_name(detector: str) -> str`.
- Produces: `WaterfallPlotContext` and `render_waterfall_png(...) -> bytes` for Task 2.
- Consumes: existing `Candidate`, `ProcessingProfile`, `preprocess_block`, Matplotlib rendering, and deterministic primary-evidence ordering.

- [ ] **Step 1: Write pure failing tests for detector-band resolution**

Create `tests/test_waterfalls.py` with literal expectations, including a non-default profile so the test cannot pass through hardcoded defaults:

```python
def test_resolves_primary_detector_band_from_run_profile():
    profile = ProcessingProfile(stalta_freq_low_hz=3.5, stalta_freq_high_hz=27.25)
    evidence = (
        candidate("coherence", score=0.70, peak_sample=80),
        candidate("stalta:coincident", score=0.91, peak_sample=90),
    )

    result = resolve_waterfall_filter(evidence, profile)

    assert result == WaterfallFilter("stalta", 3.5, 27.25)


def test_energy_filter_spans_the_configured_energy_bank():
    profile = ProcessingProfile(energy_bands_hz=((0.75, 4.0), (6.0, 33.0)))
    result = resolve_waterfall_filter(
        (candidate("energy", score=0.8, peak_sample=20),), profile
    )
    assert result == WaterfallFilter("energy", 0.75, 33.0)
```

Also cover the existing stable tie order `(-score, peak_sample, detector)` and detector suffix removal at the first `:`.

- [ ] **Step 2: Run the detector-band tests and verify RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_waterfalls.py -q
```

Expected: collection fails because `das_sentinel.waterfalls` and `WaterfallFilter` do not exist.

- [ ] **Step 3: Implement the pure shared filter context**

Create `waterfalls.py` with the exact public surface:

```python
@dataclass(frozen=True)
class WaterfallFilter:
    detector: str
    low_hz: float
    high_hz: float


def resolve_waterfall_filter(
    evidence: Sequence[Candidate], profile: ProcessingProfile
) -> WaterfallFilter:
    primary = min(
        evidence,
        key=lambda item: (-item.score, item.peak_sample, item.detector),
    ).detector.split(":", 1)[0]
    low_hz, high_hz = detector_band(profile, primary)
    return WaterfallFilter(primary, low_hz, high_hz)
```

Implement `detector_band` with the four mappings from the design and raise `ValueError` for empty evidence or an unsupported detector. Move the `_detector_band`, `_preprocess_for_band`, `_preprocess_for_detector`, and `_primary_detector` behavior out of `pipeline.py`; make the pipeline import and use the new functions.

- [ ] **Step 4: Add failing Distance-selection tests**

Add literal tests for inclusive bounds, a non-monotonic intermediate value, and bad-channel removal:

```python
def test_distance_selection_is_inclusive_and_removes_bad_channels():
    distances = np.array([8.0, 12.0, 16.0, 20.0, 24.0])
    bad = np.array([False, False, True, False, False])
    assert channel_slice_for_distance(distances, 12.0, 20.0) == slice(1, 4)
    np.testing.assert_array_equal(
        usable_channel_indices(distances, bad, 12.0, 20.0),
        np.array([1, 3]),
    )
```

Test finite values, `start_m < stop_m`, fully in-range requests, and the no-selected/no-usable-channel errors.

- [ ] **Step 5: Verify Distance tests RED, then implement the minimum pure functions**

Run the new tests and confirm missing-function failures. Implement selection using an inclusive Boolean mask; return a contiguous read slice from the first to last selected source index, then use `usable_channel_indices` after preprocessing to remove out-of-range and bad channels. Do not assume distances are ascending.

- [ ] **Step 6: Write failing permanent-title and pipeline-context tests**

Extend `tests/test_artifacts.py` to assert the pure title builder and a valid PNG:

```python
def test_filtered_waterfall_title_names_detector_and_exact_band():
    context = WaterfallPlotContext(
        sample_rate_hz=1000.0,
        site_timezone="Asia/Shanghai",
        profile_version="7",
        pipeline_version="v1",
        model_version="rules-only",
        filter=WaterfallFilter("stalta", 2.5, 40.25),
    )
    assert filtered_waterfall_title(context) == (
        "Filtered waterfall | Band-pass 2.5–40.25 Hz | "
        "primary detector=STA/LTA | profile=7 pipeline=v1 model=rules-only | "
        "1000 Hz | input unit"
    )
```

Extend the pipeline test that captures `EventArtifactData` and assert its `waterfall_filter` equals the primary detector context derived from the supplied profile.

- [ ] **Step 7: Verify title/context tests RED, then implement rendering changes**

Add `WaterfallPlotContext`, a detector display-name map, finite frequency formatting without trailing zeros, and:

```python
def render_waterfall_png(
    values: np.ndarray,
    dates: np.ndarray,
    distances: np.ndarray,
    context: WaterfallPlotContext,
    *,
    filtered: bool,
    distance_note: str | None = None,
) -> bytes:
    ...
```

Permanent raw and filtered artifacts call the same renderer; only the filtered title contains `Band-pass` and `primary detector`. Add `waterfall_filter` to `EventArtifactData`, resolve it once per event in `pipeline.py`, and preserve the existing UTC/local axes, robust color limits, padding, and evidence JSON.

- [ ] **Step 8: Run Task 1 verification**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_waterfalls.py tests/test_artifacts.py tests/test_pipeline.py -q
```

Expected: all selected tests pass, including existing raw/filtered separation and PNG border checks.

- [ ] **Step 9: Commit Task 1**

```powershell
git add backend/src/das_sentinel/waterfalls.py backend/src/das_sentinel/pipeline.py backend/src/das_sentinel/artifacts.py tests/test_waterfalls.py tests/test_artifacts.py tests/test_pipeline.py
git diff --cached --check
git commit -m "feat: label waterfall filter bands"
```

---

### Task 2: Read-only NAS preview service and authenticated API

**Files:**
- Create: `backend/src/das_sentinel/services/waterfall_previews.py`
- Create: `backend/src/das_sentinel/api/routes/waterfall_previews.py`
- Modify: `backend/src/das_sentinel/api/schemas.py`
- Modify: `backend/src/das_sentinel/api/app.py`
- Create: `tests/test_waterfall_preview_api.py`
- Modify: `tests/test_auth_api.py`

**Interfaces:**
- Consumes: Task 1 `WaterfallFilter`, `resolve_waterfall_filter`, Distance functions, `preprocess_for_waterfall`, `WaterfallPlotContext`, and `render_waterfall_png`.
- Produces: `WaterfallOptionsResponse` at `GET /api/v1/events/{event_id}/waterfall-options`.
- Produces: `WaterfallPreviewRequest` and `WaterfallPreviewResponse` at `POST /api/v1/events/{event_id}/waterfall-preview`.
- Produces: `WaterfallPreviewService.options(session, event_id) -> WaterfallOptions` and `.render(session, event_id, start_m, stop_m) -> WaterfallPreview`.

- [ ] **Step 1: Write strict schema tests and API contract tests first**

Add schemas with these exact fields:

```python
class WaterfallOptionsResponse(StrictSchema):
    automatic_distance_start_m: FiniteFloat
    automatic_distance_stop_m: FiniteFloat
    available_distance_start_m: FiniteFloat
    available_distance_stop_m: FiniteFloat
    primary_detector: str
    primary_detector_display: str
    filter_low_hz: FiniteFloat = Field(ge=0)
    filter_high_hz: FiniteFloat = Field(gt=0)
    distance_unit: Literal["m"] = "m"


class WaterfallPreviewRequest(StrictSchema):
    distance_start_m: FiniteFloat
    distance_stop_m: FiniteFloat

    @model_validator(mode="after")
    def validate_order(self) -> "WaterfallPreviewRequest":
        if self.distance_stop_m <= self.distance_start_m:
            raise ValueError("distance_stop_m must exceed distance_start_m")
        return self


class WaterfallPreviewResponse(StrictSchema):
    distance_start_m: FiniteFloat
    distance_stop_m: FiniteFloat
    channel_count: StrictInt = Field(ge=1)
    primary_detector: str
    primary_detector_display: str
    filter_low_hz: FiniteFloat = Field(ge=0)
    filter_high_hz: FiniteFloat = Field(gt=0)
    raw_png_base64: str = Field(min_length=8, max_length=16 * 1024 * 1024)
    filtered_png_base64: str = Field(min_length=8, max_length=16 * 1024 * 1024)
```

Seed a real small ZD HDF5 file, event, evidence, run, and non-current historical profile. First assert both endpoints are 404 because the routes do not exist. Add authentication tests for GET 401 and POST 401/403 without a valid CSRF token.

- [ ] **Step 2: Run contract/authentication tests and verify RED**

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_waterfall_preview_api.py tests/test_auth_api.py -q
```

Expected: route-contract tests fail with 404 while existing authentication tests remain green.

- [ ] **Step 3: Implement immutable binding and options lookup**

Create immutable internal values:

```python
@dataclass(frozen=True)
class WaterfallOptions:
    automatic_distance_start_m: float
    automatic_distance_stop_m: float
    available_distance_start_m: float
    available_distance_stop_m: float
    filter: WaterfallFilter


@dataclass(frozen=True)
class WaterfallPreview:
    distance_start_m: float
    distance_stop_m: float
    channel_count: int
    filter: WaterfallFilter
    raw_png: bytes
    filtered_png: bytes
```

The service must load `Event -> ProcessingRun -> SourceFile`, the exact `ProcessingProfile` referenced by `run.processing_profile_id`, and persisted evidence payloads. Reconstruct candidates using their persisted `peak_sample`, channel bounds, detector, and score. Resolve and validate the source path below `Settings.data_root`, compare `source.fingerprint`, `run.source_fingerprint`, and `services.discovery.fingerprint(path)`, then use `ZdHdf5Reader.read_metadata` for available and automatic ranges. Do not query the current site profile.

- [ ] **Step 4: Add options tests for historical profile and real distances**

Make the seeded site's current profile different from the run profile. Assert the response uses the run profile's literal band and metadata distances:

```python
assert response.json() == {
    "automatic_distance_start_m": 12.0,
    "automatic_distance_stop_m": 20.0,
    "available_distance_start_m": 8.0,
    "available_distance_stop_m": 24.0,
    "primary_detector": "stalta",
    "primary_detector_display": "STA/LTA",
    "filter_low_hz": 3.5,
    "filter_high_hz": 27.25,
    "distance_unit": "m",
}
```

Also assert 404 for missing event and 409 for missing, outside-root, or fingerprint-mismatched source without returning the source path in `detail`.

- [ ] **Step 5: Verify options tests RED, then add the options route**

Create a router with prefix `/events`; GET uses `AuthenticatedAdmin`. Register a single `WaterfallPreviewService` on `app.state` in `create_app`, configured with `settings.data_root` and a `ZdHdf5Reader`. Keep the endpoint synchronous so FastAPI executes metadata I/O in its thread pool.

- [ ] **Step 6: Write failing real preview tests**

Post an inclusive range and verify both Base64 values decode to PNG signatures, the actual good-channel bounds/count are returned, and a counting reader observed one `read_block` call:

```python
body = response.json()
assert base64.b64decode(body["raw_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
assert base64.b64decode(body["filtered_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
assert body["channel_count"] == 2
assert counting_reader.read_block_calls == 1
```

Snapshot table row counts and the artifact directory tree before and after; assert both are identical.

- [ ] **Step 7: Implement one-read preview generation**

The service must:

```python
read_slice = channel_slice_for_distance(metadata.channel_distance_m, start_m, stop_m)
block = reader.read_block(path, window_start, window_stop, read_slice)
filtered = preprocess_for_waterfall(block, profile, cpu_backend, filter_context)
indices = usable_channel_indices(
    block.metadata.channel_distance_m,
    filtered.bad_channel_mask,
    start_m,
    stop_m,
)
```

Use CPU for this bounded API preview so it never competes with or changes worker GPU selection. Generate dates from the block start time and sample rate. Call `render_waterfall_png` twice using the same selected raw/filtered arrays, distances, dates, and plot context; add an `Artificial Distance X–Y m` note. Check the fingerprint immediately before and after the read/render operation. Encode bytes with `base64.b64encode(...).decode("ascii")` only in the route response mapping.

- [ ] **Step 8: Add validation and controlled-failure tests**

Cover non-finite values, reversed bounds, partially or wholly out-of-range bounds, no selected channel, all selected channels bad, source removal, and a fingerprint change between pre/post checks. Assert 422 for user ranges, 409 for source conflicts, no path disclosure, and no writes.

- [ ] **Step 9: Run Task 2 verification**

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_waterfall_preview_api.py tests/test_auth_api.py tests/test_events_api.py tests/test_artifacts.py -q
```

Expected: all selected tests pass and existing event/artifact APIs remain compatible.

- [ ] **Step 10: Commit Task 2**

```powershell
git add backend/src/das_sentinel/services/waterfall_previews.py backend/src/das_sentinel/api/routes/waterfall_previews.py backend/src/das_sentinel/api/schemas.py backend/src/das_sentinel/api/app.py tests/test_waterfall_preview_api.py tests/test_auth_api.py
git diff --cached --check
git commit -m "feat: generate temporary waterfall previews"
```

---

### Task 3: Accessible Distance controls and browser-only preview lifecycle

**Files:**
- Modify: `frontend/src/api.ts`
- Create: `frontend/src/components/useWaterfallPreview.ts`
- Create: `frontend/src/components/WaterfallRangeControls.tsx`
- Modify: `frontend/src/components/ArtifactViewer.tsx`
- Modify: `frontend/src/pages/EventsPage.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/api.test.ts`
- Modify: `frontend/src/components/ArtifactViewer.test.tsx`
- Modify: `frontend/src/pages/EventsPage.test.tsx`
- Modify: `e2e/review-flow.spec.ts`

**Interfaces:**
- Consumes: Task 2 option/preview JSON contracts.
- Produces: `api.post<T>(path, body, signal?)` with AbortSignal propagation.
- Produces: `useWaterfallPreview(eventId)` with `options`, `preview`, `busy`, `error`, `generate`, and `reset`.
- Produces: `ArtifactViewer({ eventId, artifacts })`.

- [ ] **Step 1: Write an abortable POST test and verify RED**

Extend `frontend/src/api.test.ts`:

```typescript
it("forwards AbortSignal for JSON POST requests", async () => {
  const controller = new AbortController();
  const pending = api.post("/api/v1/events/e1/waterfall-preview", {
    distance_start_m: 12,
    distance_stop_m: 20,
  }, controller.signal);
  controller.abort();
  await expect(pending).rejects.toMatchObject({ name: "AbortError" });
});
```

Run `npm test -- --run src/api.test.ts`; expected TypeScript/test failure because `post` accepts only two arguments.

- [ ] **Step 2: Add strict frontend API types and signal support**

Add interfaces mirroring Task 2 exactly and change only `post`:

```typescript
post: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
  request<T>("POST", path, body, undefined, signal),
```

Run the API test again; expected PASS.

- [ ] **Step 3: Write ArtifactViewer behavior tests before UI code**

In `ArtifactViewer.test.tsx`, provide `eventId="event-1"` and mock real API responses. Add tests that assert:

- options render `可用 8–24 m`, default inputs `12` and `20`, `滤波瀑布图（3.5–27.25 Hz）`, and `主检测器：STA/LTA`;
- invalid client input shows a literal error and sends no POST;
- one valid submit swaps both waterfall image/download URLs to Blob URLs but leaves waveform/PSD/STFT on artifact API URLs;
- reset restores artifact API URLs and revokes both preview URLs;
- a second preview revokes the first pair;
- event change/unmount aborts requests, ignores late results, and revokes all created URLs;
- a 409/422/500 response preserves the currently displayed image and shows `无法生成临时预览`.

Name the production mutation each test catches in a short test comment; assertions must target rendered behavior and URL lifecycle, not mock existence.

At the same RED stage, extend `e2e/review-flow.spec.ts` with the real seeded-stack workflow shown in Task 4. The E2E test must be written before the controls exist, so it fails on the missing band label/range form rather than being added after the implementation.

- [ ] **Step 4: Run ArtifactViewer tests and verify RED**

```powershell
npm test -- --run src/components/ArtifactViewer.test.tsx
```

Expected: component prop/type and missing-control failures.

Run the seeded Playwright spec once as well; expected: missing band label/range controls. Record this E2E RED result before production UI changes.

- [ ] **Step 5: Implement the focused preview hook**

Create `useWaterfallPreview.ts` with this public return shape:

```typescript
export interface WaterfallPreviewUrls {
  waterfall_raw: string;
  waterfall_filtered: string;
}

export function useWaterfallPreview(eventId: string): {
  options: WaterfallOptions | null;
  optionsError: string;
  preview: WaterfallPreviewUrls | null;
  busy: boolean;
  error: string;
  generate(startM: number, stopM: number): Promise<void>;
  reset(): void;
}
```

Use one controller and monotonically increasing sequence for options/preview requests. Decode each Base64 string into a `Blob({type: "image/png"})`; if decoding or the second URL creation fails, revoke anything already created before surfacing the error. Revoke the previous pair only after a complete new pair is ready. Cleanup aborts requests and revokes URLs.

- [ ] **Step 6: Implement controls and ArtifactViewer integration**

Create labelled `type="number"` inputs with `step="any"`, visible bounds, generate/reset buttons, `aria-busy`, `role="status"`, and `role="alert"`. Keep input strings local so partially typed numbers do not become `NaN` state.

`ArtifactViewer` chooses the active source as:

```typescript
const previewUrl =
  active.kind === "waterfall_raw"
    ? preview?.waterfall_raw
    : active.kind === "waterfall_filtered"
      ? preview?.waterfall_filtered
      : undefined;
```

Use the preview URL for the image, fullscreen image, and download link only when defined. Existing permanent artifact loading, retry, keyboard tabs, focus trap, and error behavior remain unchanged.

- [ ] **Step 7: Preserve explicit station timezone and pass event identity**

Change `EventsPage.tsx` to:

```tsx
<h2>{formatOccurredAt(detail.occurred_at, detail.site_timezone)}</h2>
<ArtifactViewer eventId={detail.id} artifacts={detail.artifacts} />
```

The first line is required because the current HEAD omits the required second formatter argument; relying on the host timezone would violate the already approved `Asia/Shanghai` behavior. Strengthen `EventsPage.test.tsx` with a timezone different from the test host so this mutation fails deterministically.

- [ ] **Step 8: Add compact responsive styling**

Add only selectors under `.waterfall-range-controls`: a wrapping flex/grid row, bounded numeric inputs, status/error placement, and mobile stacking. Do not change the established artifact/fullscreen geometry.

- [ ] **Step 9: Run Task 3 tests and production build**

```powershell
npm test -- --run src/api.test.ts src/components/ArtifactViewer.test.tsx src/pages/EventsPage.test.tsx
npm run build
```

Run the isolated seeded stack and `e2e/review-flow.spec.ts` after the component suite. Expected: selected unit/component tests and the real preview E2E pass; TypeScript and Vite build succeed with no missing formatter argument.

- [ ] **Step 10: Commit Task 3**

```powershell
git add frontend/src/api.ts frontend/src/components/useWaterfallPreview.ts frontend/src/components/WaterfallRangeControls.tsx frontend/src/components/ArtifactViewer.tsx frontend/src/pages/EventsPage.tsx frontend/src/styles.css frontend/src/api.test.ts frontend/src/components/ArtifactViewer.test.tsx frontend/src/pages/EventsPage.test.tsx e2e/review-flow.spec.ts
git diff --cached --check
git commit -m "feat: preview waterfall distance ranges"
```

---

### Task 4: End-to-end acceptance, full regression, and runtime verification

**Files:**
- No repository source files expected.
- Runtime data is read-only for acceptance snapshots.

**Interfaces:**
- Consumes: Tasks 1–3 complete API and UI.
- Produces: verified Distance preview workflow against a real synthetic HDF5 source and the current JJK runtime event.

- [ ] **Step 1: Run the already-RED/GREEN Playwright workflow fresh**

Task 3 has already added this test before the UI implementation:

```typescript
await expect(page.getByRole("tab", { name: "滤波瀑布图（2–40 Hz）" })).toBeVisible();
await expect(page.getByText("主检测器：STA/LTA")).toBeVisible();
await page.getByLabel("起始距离（m）").fill("12");
await page.getByLabel("结束距离（m）").fill("20");
await page.getByRole("tab", { name: "瀑布图", exact: true }).click();
const originalRawDownload = await page.getByRole("link", { name: "下载" }).getAttribute("href");
await page.getByRole("button", { name: "生成预览" }).click();
await expect(page.getByText("人工范围：12–20 m")).toBeVisible();
await expect(page.getByRole("link", { name: "下载" })).toHaveAttribute("href", /^blob:/);
await page.getByRole("button", { name: "恢复自动范围" }).click();
await expect(page.getByRole("link", { name: "下载" })).toHaveAttribute("href", originalRawDownload!);
```

Capture the raw and filtered image URLs before/after preview and assert both change. Capture waveform/PSD/STFT URLs and assert they do not change. Reuse the existing fullscreen geometry assertions on a preview image.

- [ ] **Step 2: Confirm the real fixture and full E2E behavior**

The existing synthetic source contains eight channels at 8–36 m and a 10 Hz STA/LTA burst; do not mock the preview endpoints. Run the existing project Playwright command for `e2e/review-flow.spec.ts`. Expected: band label, one-request two-image preview, reset, fullscreen geometry, review flow, and stale-detail tests all pass. If the literal event automatic range differs, stop and diagnose the detector output rather than weakening the exact assertions.

- [ ] **Step 3: Run fresh full verification from an exact local-disk copy**

Mechanically synchronize tracked files to a temporary/local D-drive verification copy, compare SHA-256 for every changed file, then run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest -q
npm test -- --run
npm run build
```

Expected: the full backend suite, full frontend suite, and production build exit 0. Record exact counts and warnings; do not claim success from earlier task-level runs.

- [ ] **Step 4: Verify the current JJK event without mutating audit state**

Before preview, record read-only counts for events/artifacts and the current event artifact directory file names/hashes. In the authenticated browser select event `c58e73c0-8a51-4db2-99a9-e3d9b8a2d610`, confirm:

- time remains `2024-05-27 15:28:50` with `Asia/Shanghai` semantics;
- the filtered tab and nearby text show the historical run's actual band/detector;
- a valid smaller Distance interval produces two temporary images;
- filtered preview title contains the band and detector;
- both preview Distance axes use the artificial interval;
- waveform, PSD, and STFT stay on original artifacts;
- reset returns both waterfall tabs to original artifact URLs;
- full-screen image remains contained and the close button does not overlap.

After reset, repeat the DB counts and artifact file/hash snapshot; expected: byte-for-byte and row-for-row unchanged.

- [ ] **Step 5: Request whole-branch review and integrate only after approval**

Generate a review package from the plan starting commit through HEAD. The reviewer must check filter-profile provenance, path/fingerprint safety, no DB/artifact writes, one-read behavior, Base64/Blob cleanup, station-timezone preservation, and all UI abort/stale-response paths. Fix every Critical or Important finding with a failing test and scoped re-review. After the final fresh verification is green, merge the feature branch to `main`, push `origin/main`, and verify the remote SHA plus `GET /healthz`.

---

## Plan Self-Review

- **Spec coverage:** Task 1 covers filter provenance and permanent titles; Task 2 covers options, one-read previews, validation, security, NAS failures, and immutability; Task 3 covers numeric UI, two-waterfall-only replacement, cancellation, reset, Blob cleanup, fullscreen/download, and station timezone preservation; Task 4 covers real HDF5, current JJK acceptance, no-write proof, full regression, review, merge, and push.
- **Placeholder scan:** Every test, command, interface, error class, and commit point is specified; no deferred implementation step or unspecified generic error handling remains.
- **Type consistency:** `WaterfallFilter`, `WaterfallOptionsResponse`, `WaterfallPreviewRequest`, `WaterfallPreviewResponse`, `WaterfallPreviewUrls`, and `ArtifactViewer(eventId, artifacts)` use the same names and units throughout all tasks.
- **Scope check:** The plan does not add drag selection, station processing limits, persistent previews, caches, migrations, new dependencies, or recomputation of waveform/PSD/STFT.
