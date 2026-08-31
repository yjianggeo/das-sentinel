# JJK Time and Artifact Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display the JJK `UTC8_202405271528` sample as Beijing time 15:28 everywhere while preserving UTC internally, and render every event image completely without fullscreen controls covering it.

**Architecture:** Normalize only the ZD files whose `UTC<offset>` filename wall clock agrees with the first raw timestamp, then propagate the existing site IANA timezone through feature extraction and event API responses. Keep the artifact component structure unchanged: fix Matplotlib canvas padding at export and use CSS flow layout for the fullscreen button and contained image.

**Tech Stack:** Python 3.12, h5py, NumPy, Matplotlib, FastAPI, SQLAlchemy, pytest, React 18, TypeScript, Vitest, Playwright, CSS.

## Global Constraints

- Preserve timezone-aware UTC values in domain objects and database columns.
- Do not hardcode `Asia/Shanghai`; use the configured `Site.timezone` for display and classification features.
- Do not change detection thresholds, classification thresholds, or processing profiles.
- Preserve existing behavior for files without a matching `UTC<offset>_<timestamp>` filename marker.
- Use test-driven development: observe every new regression test fail before changing production code.
- Before replacing current validation data, copy the SQLite database and move the exact current event artifact directory to a recoverable backup location.

---

### Task 1: Normalize JJK UTC8 source time and localize the time-of-day feature

**Files:**
- Modify: `tests/test_zd_hdf5.py`
- Modify: `backend/src/das_sentinel/readers/zd_hdf5.py`
- Modify: `tests/test_features.py`
- Modify: `backend/src/das_sentinel/features.py`
- Modify: `backend/src/das_sentinel/pipeline.py`

**Interfaces:**
- Consumes: `Path | BinaryIO`, HDF5 `RawDataTime`, and filename convention `UTC<offset>_<YYYYMMDDHHMM[SS]>`.
- Produces: `ZdHdf5Reader._time_at(times, index, path) -> datetime` returning timezone-aware UTC, and feature context keys `start_time_utc: datetime` plus `site_timezone: str`.

- [ ] **Step 1: Write a failing reader regression test**

Add imports and a test proving that a matching `UTC8` wall clock is converted while ordinary files retain their current UTC interpretation:

```python
from datetime import UTC, datetime


def test_interprets_matching_utc8_filename_clock_as_station_local_time(tmp_path):
    path = tmp_path / "ZD_3400m_8m_4m_5000Hz_1000Hz_UTC8_202607260000.h5"
    write_zd_h5(path)

    reader = ZdHdf5Reader()
    metadata = reader.read_metadata(path)
    block = reader.read_block(path, 10, 11)

    assert metadata.start_time_utc == datetime(2026, 7, 25, 16, 0, tzinfo=UTC)
    assert block.metadata.start_time_utc == datetime(
        2026, 7, 25, 16, 0, 0, 100_000, tzinfo=UTC
    )


def test_does_not_apply_filename_offset_when_wall_clocks_disagree(tmp_path):
    path = tmp_path / "ZD_3400m_8m_4m_5000Hz_1000Hz_UTC8_202607260100.h5"
    write_zd_h5(path)

    assert ZdHdf5Reader().read_metadata(path).start_time_utc == datetime(
        2026, 7, 26, 0, 0, tzinfo=UTC
    )
```

- [ ] **Step 2: Run the reader tests and verify RED**

Run from a local-disk test copy to avoid NAS collection latency:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_zd_hdf5.py -q
```

Expected: the matching `UTC8` test fails with actual `2026-07-26T00:00:00+00:00`; the mismatch and existing reader tests pass.

- [ ] **Step 3: Implement conservative filename offset normalization**

Change all `_time_at` calls to pass `path`. Implement helpers with these exact semantics:

```python
@staticmethod
def _source_name(path: Path | BinaryIO) -> str:
    name = path.name if isinstance(path, Path) else getattr(path, "name", "")
    return Path(name).name if isinstance(name, (str, Path)) else ""

@classmethod
def _filename_utc_offset(
    cls, path: Path | BinaryIO, raw_start: datetime
) -> timedelta | None:
    match = re.search(
        r"_UTC(?P<offset>[+-]?\d{1,2})_(?P<stamp>\d{12}(?:\d{2})?)(?!\d)",
        cls._source_name(path),
        re.IGNORECASE,
    )
    if match is None:
        return None
    stamp = match.group("stamp")
    format_string = "%Y%m%d%H%M%S" if len(stamp) == 14 else "%Y%m%d%H%M"
    filename_clock = datetime.strptime(stamp, format_string)
    raw_clock = raw_start.replace(tzinfo=None, microsecond=0)
    if len(stamp) == 12:
        raw_clock = raw_clock.replace(second=0)
    if filename_clock != raw_clock:
        return None
    return timedelta(hours=int(match.group("offset")))

@classmethod
def _time_at(
    cls, times: h5py.Dataset, index: int, path: Path | BinaryIO
) -> datetime:
    raw_start = datetime.fromtimestamp(int(times[0]) / 1_000_000, tz=UTC)
    raw_time = datetime.fromtimestamp(int(times[index]) / 1_000_000, tz=UTC)
    return raw_time - (cls._filename_utc_offset(path, raw_start) or timedelta())
```

Import `timedelta`. Reuse `_source_name` from `_spacing_from_filename` so held streams continue to retain filename hints.

- [ ] **Step 4: Verify the reader is GREEN**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_zd_hdf5.py -q
```

Expected: all reader tests pass, including held-stream spacing inference.

- [ ] **Step 5: Write a failing station-local feature test**

Add to `tests/test_features.py`:

```python
from datetime import UTC, datetime


def test_time_of_day_uses_configured_station_timezone():
    block = preprocessed_block(np.ones((10, 2), dtype=np.float32))
    event = EventRecord("local-time", 1_000, 1_010, 1_005, 0, 2, ())

    features = extract_features(
        block,
        event,
        context={
            "start_time_utc": datetime(2024, 5, 27, 7, 28, tzinfo=UTC),
            "site_timezone": "Asia/Shanghai",
        },
    )

    assert features["time_of_day_hours"] == pytest.approx(15 + 28 / 60)
```

- [ ] **Step 6: Run the feature test and verify RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_features.py::test_time_of_day_uses_configured_station_timezone -q
```

Expected: FAIL with `7.466...` instead of `15.466...`.

- [ ] **Step 7: Localize the feature using the pipeline site timezone**

Import `ZoneInfo` in `features.py` and replace the current assignment with:

```python
start_time = details.get("start_time_utc")
site_timezone = details.get("site_timezone")
if isinstance(start_time, datetime):
    local_time = (
        start_time.astimezone(ZoneInfo(site_timezone))
        if isinstance(site_timezone, str)
        else start_time
    )
    values["time_of_day_hours"] = (
        local_time.hour + local_time.minute / 60 + local_time.second / 3600
    )
```

Pass `"site_timezone": run_context.site_timezone` beside `start_time_utc` in `pipeline.py`.

- [ ] **Step 8: Verify Task 1 and commit**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_zd_hdf5.py tests/test_features.py -q
git add -- tests/test_zd_hdf5.py backend/src/das_sentinel/readers/zd_hdf5.py tests/test_features.py backend/src/das_sentinel/features.py backend/src/das_sentinel/pipeline.py
git commit -m "fix: normalize JJK UTC8 acquisition times"
```

Expected: all selected tests pass and only Task 1 files are committed.

---

### Task 2: Carry the site timezone through the event API and format it in the UI

**Files:**
- Modify: `tests/test_events_api.py`
- Modify: `backend/src/das_sentinel/api/schemas.py`
- Modify: `backend/src/das_sentinel/api/routes/events.py`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/components/EventQueue.tsx`
- Modify: `frontend/src/pages/EventsPage.test.tsx`
- Modify: `e2e/review-flow.spec.ts`

**Interfaces:**
- Consumes: `Site.timezone`, `Event.occurred_at` as ISO UTC, and `EventSummaryResponse`.
- Produces: `EventSummaryResponse.site_timezone: str`, TypeScript `EventSummary.site_timezone: string`, and `formatOccurredAt(iso, timeZone) -> string`.

- [ ] **Step 1: Add failing API assertions for station timezone**

In the existing list/detail event API test, assert both response shapes include the seeded site's timezone:

```python
all_events = client.get("/api/v1/events").json()
assert all_events[0]["site_timezone"] == "Asia/Shanghai"

detail = client.get(f"/api/v1/events/{ids['event_1']}").json()
assert detail["site_timezone"] == "Asia/Shanghai"
```

- [ ] **Step 2: Run the API test and verify RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_events_api.py -q
```

Expected: FAIL because `site_timezone` is absent.

- [ ] **Step 3: Add `site_timezone` to event summaries without N+1 list queries**

Add `site_timezone: str` to `EventSummaryResponse`. Import `Site` in the events route, join it in `list_events`, select `Site.timezone`, and pass that value to `_list_summary`. Extend `_summary` to resolve the source's `Site` once when needed and pass its timezone. Construct every summary with:

```python
site_timezone=site_timezone,
```

Do not change UTC filtering or ordering.

- [ ] **Step 4: Verify API tests are GREEN**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_events_api.py tests/test_similar_api.py -q
```

Expected: all selected API tests pass.

- [ ] **Step 5: Write a failing frontend timezone display test**

Add `site_timezone: "Asia/Shanghai"` to the event fixtures in `EventsPage.test.tsx`, change the expected queue and heading clock for `2026-07-26T10:42:16Z` to `2026-07-26 18:42:16`, and add a focused test:

```tsx
it("formats event UTC timestamps in the station timezone", async () => {
  useEventHandlers();
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));

  expect(
    await screen.findByRole("heading", { name: "2026-07-26 18:42:16" }),
  ).toBeVisible();
});
```

Update the other fixture-time expectations mechanically to use their UTC+8 equivalents; do not alter selection/race assertions.

- [ ] **Step 6: Run the frontend test and verify RED**

Run:

```powershell
npm test -- --run frontend/src/pages/EventsPage.test.tsx
```

Expected: FAIL because `formatOccurredAt` still strips `Z` without timezone conversion.

- [ ] **Step 7: Implement deterministic station-time formatting**

Add `site_timezone: string` to `EventSummary`. Change the formatter and all calls:

```tsx
export function formatOccurredAt(iso: string, timeZone: string): string {
  const parts = new Intl.DateTimeFormat("zh-CN", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date(iso));
  const value = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${value.year}-${value.month}-${value.day} ${value.hour}:${value.minute}:${value.second}`;
}
```

Queue items call `formatOccurredAt(event.occurred_at, event.site_timezone)` and the detail heading uses `detail.site_timezone`. Update the e2e helper and call sites to accept each event's timezone.

- [ ] **Step 8: Verify Task 2 and commit**

Run:

```powershell
npm test -- --run frontend/src/pages/EventsPage.test.tsx
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_events_api.py tests/test_similar_api.py -q
git add -- tests/test_events_api.py backend/src/das_sentinel/api/schemas.py backend/src/das_sentinel/api/routes/events.py frontend/src/api.ts frontend/src/components/EventQueue.tsx frontend/src/pages/EventsPage.test.tsx e2e/review-flow.spec.ts
git commit -m "fix: display events in station timezone"
```

Expected: selected backend and frontend tests pass.

---

### Task 3: Preserve PNG margins and remove fullscreen overlap

**Files:**
- Modify: `tests/test_artifacts.py`
- Modify: `backend/src/das_sentinel/artifacts.py`
- Modify: `frontend/src/styles.css`
- Modify: `e2e/review-flow.spec.ts`

**Interfaces:**
- Consumes: existing `_png(figure) -> bytes` and existing fullscreen DOM classes.
- Produces: PNGs with white outer margins and a fullscreen layout whose close button rectangle never intersects the image rectangle.

- [ ] **Step 1: Write a failing PNG boundary test**

Import Pillow and add:

```python
from PIL import Image


def test_png_artifacts_leave_a_clean_margin_around_plot_content(tmp_path):
    from das_sentinel.artifacts import render_event_artifacts

    manifest = render_event_artifacts(_artifact_data(), _event(), {}, tmp_path)
    for kind in ("waterfall_raw", "waterfall_filtered", "waveform", "psd", "stft"):
        pixels = np.asarray(Image.open(manifest.files[kind].path).convert("RGB"))
        border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
        assert np.all(border >= 250), kind
```

- [ ] **Step 2: Run the artifact test and verify RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_artifacts.py::test_png_artifacts_leave_a_clean_margin_around_plot_content -q
```

Expected: FAIL for at least the waterfall image because plot text or axes reach the current canvas boundary.

- [ ] **Step 3: Export figures with a tight padded boundary**

Change only `_png`:

```python
figure.savefig(
    buffer,
    format="png",
    dpi=120,
    bbox_inches="tight",
    pad_inches=0.2,
    metadata={"Software": "das-sentinel"},
)
```

- [ ] **Step 4: Verify PNG generation is GREEN**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_artifacts.py -q
```

Expected: all artifact tests pass without warnings.

- [ ] **Step 5: Add failing fullscreen geometry assertions**

Extend the existing Playwright fullscreen flow immediately after opening the dialog:

```ts
const dialog = page.getByRole("dialog");
const imageBox = await dialog.locator("img").boundingBox();
const closeBox = await page.getByRole("button", { name: "关闭全屏" }).boundingBox();
expect(imageBox).not.toBeNull();
expect(closeBox).not.toBeNull();
expect(closeBox!.y + closeBox!.height).toBeLessThanOrEqual(imageBox!.y);
expect(imageBox!.y).toBeGreaterThanOrEqual(0);
expect(imageBox!.y + imageBox!.height).toBeLessThanOrEqual(
  page.viewportSize()!.height,
);
```

Use the current live browser to record the equivalent pre-fix geometry. Expected RED evidence: the button bottom is below the image top and the rectangles overlap.

- [ ] **Step 6: Implement the smallest CSS-only fullscreen layout fix**

Replace the fullscreen rules with:

```css
.artifact-fullscreen {
  position: fixed;
  inset: 0;
  padding: 1rem;
  overflow: auto;
  background: rgba(0, 0, 0, 0.92);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.artifact-fullscreen-dialog {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  max-width: 100%;
  max-height: 100%;
}

.artifact-fullscreen-dialog img {
  display: block;
  max-width: 100%;
  max-height: calc(100vh - 5rem);
  object-fit: contain;
}

.artifact-fullscreen-close {
  position: static;
  justify-self: end;
  margin-bottom: 0.5rem;
}
```

Preserve the existing close-button colors and focus behavior declarations.

- [ ] **Step 7: Verify Task 3 and commit**

Run:

```powershell
npm test -- --run frontend/src/components/ArtifactViewer.test.tsx
npm run build
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_artifacts.py -q
git add -- tests/test_artifacts.py backend/src/das_sentinel/artifacts.py frontend/src/styles.css e2e/review-flow.spec.ts
git commit -m "fix: show complete event artifacts"
```

Expected: unit tests and production build pass; do not stage generated `frontend/tsconfig.tsbuildinfo`.

---

### Task 4: Regression verification and recoverable reprocessing of the current JJK validation event

**Files:**
- No repository source files.
- Runtime data: `D:\das-sentinel-runtime\app-data`.
- Source: `Z:\jjk20240527\20240527\ZD_3400m_8m_4m_5000Hz_1000Hz_UTC8_202405271528.h5`.

**Interfaces:**
- Consumes: verified application build and current validation event `66ad21b8-dfa6-4612-840e-3dac2ab5c599` at site `cce64999-eb01-4b47-8544-1301ebd6b33c`.
- Produces: a newly processed event whose UTC time is 07:28 and station display time is 15:28, with regenerated artifacts.

- [ ] **Step 1: Run focused and full automated verification**

Run backend tests from the local-disk copy after synchronizing the changed backend and test files:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest -q
npm test -- --run
npm run build
```

Expected: all backend and frontend tests pass and the frontend production build succeeds.

- [ ] **Step 2: Back up runtime state and resolve exact cleanup targets**

Stop the API and worker sessions. Create a timestamped directory below `D:\das-sentinel-runtime\backups`, copy the SQLite database into it, and query the database for the source, jobs, run, event, and artifact paths tied to event ID `66ad21b8-dfa6-4612-840e-3dac2ab5c599`. Verify every artifact path resolves below `D:\das-sentinel-runtime\app-data\artifacts` before moving that exact event directory into the backup.

Expected: database backup exists, the exact event artifact directory is recoverable, and no site/admin row is included in cleanup targets.

- [ ] **Step 3: Remove only the generated validation source record in one transaction**

Enable SQLite foreign keys, delete the single `source_files` row discovered in Step 2, and commit. Rely on declared cascades to remove only that source's jobs, processing run, event, classifications, evidence, reviews, and artifact metadata. Query all affected tables afterward to verify the old event ID and old source ID no longer exist while site `cce64999-eb01-4b47-8544-1301ebd6b33c` and the admin row remain.

Expected: only current validation-derived records are absent; the backup makes the operation reversible.

- [ ] **Step 4: Restart, rescan, and wait for reprocessing**

Restart Uvicorn on `127.0.0.1:18000` and the worker using the existing virtual environment and app-data configuration. Trigger a scan of the exact JJK source path, wait until the new job reaches `succeeded`, and record the new event ID.

Expected: a single new event and six artifacts are produced without time-related metadata warnings.

- [ ] **Step 5: Verify the API and browser result**

Verify through the authenticated event API that:

```text
occurred_at starts with 2024-05-27T07:28
site_timezone equals Asia/Shanghai
time_of_day_hours is approximately 15.48
```

Open `/events`, select the regenerated event, and verify the queue, heading, waterfall station-local axis, and waveform station-local axis show `2024-05-27 15:28`. Open every artifact tab and fullscreen view. Measure the image and close-button rectangles and verify they do not overlap and the image is within the viewport.

- [ ] **Step 6: Commit any verification-only test adjustments, then push**

Run:

```powershell
git diff --check
git status --short
git push origin main
```

Expected: only the pre-existing generated `frontend/tsconfig.tsbuildinfo` remains unstaged, and all new commits are present on `origin/main`.

---

## Plan Self-Review

- Spec coverage: Tasks 1–4 cover UTC normalization, station-local feature calculation, API timezone propagation, frontend formatting, PNG margins, fullscreen geometry, recoverable current-data reprocessing, browser verification, and push.
- Placeholder scan: no deferred implementation or unspecified test step remains.
- Type consistency: `site_timezone` is used consistently in Python response models, TypeScript event interfaces, formatter calls, feature context, and runtime checks.
