# Event Plot Time Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide at least three seconds of real display data for event waterfalls and waveforms, with precise relative-time labels and non-destructive historical previews.

**Architecture:** Keep analysis inputs and event-only spectral slices unchanged. A shared sample-window helper selects display context; shared rendering uses a local whole-second reference and relative ticks. Extend the existing preview response with a waveform, without new jobs or persisted evidence.

**Tech Stack:** Existing Python/NumPy/Matplotlib/FastAPI, React/TypeScript, pytest and Vitest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-08-event-plot-time-window-design.md` (user approved).

## Global Constraints

- 原始瀑布图、滤波瀑布图和波形使用相同的显示时间窗，真实数据允许时至少为 3 秒。
- 时长定义为 `(stop - start) / fs`，不是最后一个采样点时间减去第一个采样点时间。
- 不改变事件检测、融合、分类、特征提取、PSD、STFT 的分析范围与计算输入。
- 不涉及修改 NAS 原始数据、覆盖历史 PNG、修改其校验和、数据库迁移、跨文件拼接或重新检测。
- 沿用认证、CSRF、源指纹复检及现有的 30,000 样本、256 通道读取上限；在读取大数组前完成范围检查。
- 响应仅增加必填 `waveform_png_base64`，采用与另两张 PNG 一样的长度限制。
- Tasks execute sequentially in the existing linked worktree; preserve unrelated work. No push, merge, global install or dependency upgrade.
- Keep cancelled production jobs cancelled. Do not start a worker or create analysis jobs for verification.

## File and execution map

Repository: `D:/das-sentinel-runtime/waterfall-range-previews-git-20260901-2/.worktrees/jjk-scan-progress`.
Python: `D:/das-sentinel-runtime/.venv/Scripts/python.exe`.
Node tools: `C:/Program Files/nodejs/npm.cmd`; run frontend commands from `frontend/`.
Use `apply_patch` for edits. Before each implementation, run its new tests and record the expected failure; after implementation record passing results. Commit only owned files. Full regression runs once after all tasks; individual tasks run their complete affected test files, not the repository-wide suite.

Responsibilities:

- `backend/src/das_sentinel/plot_windows.py`: pure source-bounded display-window selection.
- `backend/src/das_sentinel/artifacts.py`: common time axes, reusable waveform rendering, display/analysis separation.
- `backend/src/das_sentinel/pipeline.py`: prepare extra display-only input only when necessary.
- `backend/src/das_sentinel/services/waterfall_previews.py`, API schema/route: one read, three transient plots, explicit automatic selection.
- Existing frontend preview hook, viewer and controls: atomic three-image lifecycle and automatic/manual distance requests.
- Focused tests plus the existing Chinese beginner manual: regression and operating instructions.

### Task 1: Shared window selection and precise plotting primitives

**Files:**
- Create: `backend/src/das_sentinel/plot_windows.py`, `tests/test_plot_windows.py`.
- Modify: `backend/src/das_sentinel/artifacts.py`, `tests/test_artifacts.py`.

**Interfaces:**
- Produce `event_display_window(start_sample: int, end_sample: int, sample_count: int, pad_seconds: float, sample_rate_hz: float) -> tuple[int, int]`.
- Produce public `render_waveform_png(values, dates, context: WaterfallPlotContext, event: EventRecord, *, sample_offset: int, window_note: str | None = None, distance_note: str | None = None) -> bytes`. `dates[0]` is the absolute time at global `sample_offset`.
- Extend `render_waterfall_png` with optional keyword `window_note`; preserve existing positional arguments and `filtered_waterfall_title` contract.
- Keep `_waveform_png` as a small compatibility wrapper calling the shared renderer with `sample_offset=data.sample_offset + samples.start`. Task 2 and Task 3 consume the public primitives.

- [ ] Step 1: Write literal window tests and a real Matplotlib label regression.

```python
@pytest.mark.parametrize('start,end,total,pad,fs,want', [
    (450, 460, 1000, 0, 100, (305, 605)),
    (0, 10, 1000, 0, 100, (0, 300)),
    (990, 1000, 1000, 0, 100, (700, 1000)),
    (450, 850, 1000, 0, 100, (450, 850)),
    (40, 50, 100, 0, 100, (0, 100)),
    (100, 101, 300, 0, 10.5, (85, 117)),
    (450, 460, 2000, 5, 100, (0, 960)),
])
def test_display_window(start, end, total, pad, fs, want):
    assert event_display_window(start, end, total, pad, fs) == want
```

Also exercise invalid/non-finite rates, negative padding and invalid event bounds. For `_time_axes`, create a three-second 100 Hz datetime array starting at `2026-09-08T15:59:59.234Z`, draw the canvas, and assert visible labels are unique, the reference includes local `2026-09-08 23:59:59` and `Asia/Shanghai`, and tick numeric values equal elapsed seconds from that integer reference within date-coordinate precision. Compare two fresh axes receiving the same dates for identical ticks and limits. Include a subsecond and single-sample source; assert the latter has sample-period bounds, not day-sized mesh edges.

- [ ] Step 2: Run new window tests and focused axis regressions; confirm failure before code.

```powershell
& 'D:/das-sentinel-runtime/.venv/Scripts/python.exe' -m pytest tests/test_plot_windows.py tests/test_artifacts.py -q
```

- [ ] Step 3: Implement the pure helper, preserving the existing rounded configured padding and using a ceil only for the minimum duration.

```python
padding = max(0, round(pad_seconds * sample_rate_hz))
start = max(0, start_sample - padding)
stop = min(sample_count, end_sample + padding)
target = min(sample_count, max(stop - start, math.ceil(3 * sample_rate_hz)))
missing = target - (stop - start)
start = max(0, min(start - missing // 2, sample_count - target))
return start, start + target
```

Validate before this core. Keep absolute Matplotlib coordinates, but create ticks in relative seconds using a bounded numeric locator and install fixed absolute tick positions. Derive formatter precision from tick spacing (not the sample rate), with enough digits for uniqueness; remove negative zero. Top axis becomes only a reference label; bottom carries relative seconds. Preserve whole date and IANA zone. Rendering follows existing English plot-label style with equivalent explicit meaning; Chinese explanations remain in the UI/manual. Example plot strings: `Local reference: 2026-09-08 23:59:59 (Asia/Shanghai)` and `Time relative to reference (s)`.

Extract existing waveform drawing into the public renderer without changing event/candidate lines, the representative median algorithm, or PSD/STFT. Add short/window notes without breaking the literal existing filter-title helper. For single-sample waterfall time edges pass the actual sampling interval; preserve distance-edge behavior.

- [ ] Step 4: Run all `test_plot_windows.py` and `test_artifacts.py`, Ruff on touched Python, and `git diff --check`; report RED/GREEN evidence and any baseline warnings.
- [ ] Step 5: Commit the task as `feat(plots): add bounded windows and precise relative time axes`.

### Task 2: Formal plots with isolated display context

**Files:** `backend/src/das_sentinel/artifacts.py`, `backend/src/das_sentinel/pipeline.py`, `tests/test_artifacts.py`, `tests/test_pipeline.py`.

**Interfaces:**
- Consume Task 1 `event_display_window` and shared renderers.
- Extend `render_event_artifacts(data, event, features, out_dir, *, display_data: EventArtifactData | None = None, source_sample_count: int | None = None)`; original `data` remains the analysis/evidence input.
- Reuse existing `EventArtifactData` for truthful display-only data with replaced arrays/timing/mask. Do not manufacture this analysis-heavy type in historical preview; Task 3 calls the narrow renderer.

- [ ] Step 1: Add rendering regression that captures renderer inputs at the PNG boundary while exercising real slice selection. Assert two waterfalls and waveform receive the same entire display interval and event-selected good channels; PSD/STFT still receive only the original event slice. Evidence JSON `source_range` and input hashes still refer to original analysis data.

```python
# With existing _artifact_data() fixture and _event():
assert recorded['display_shape'] == (160, 3)
assert recorded['waveform_length'] == 160
assert recorded['psd_length'] == 60
assert recorded['stft_length'] == 60
assert evidence['source_range']['sample_start'] == 1020
```

Add a pipeline test with the existing boundary fixture and zero artifact padding: the analysis read remains `[184:197]`; at 10 Hz the display read expands to `[176:206]`. Capture real feature input and renderer inputs, assert event labels/features unchanged, and ensure all default-context paths add no extra source read. Tests for original `capture_artifacts` fakes must accept the new keyword arguments while retaining their existing original-input assertions.

- [ ] Step 2: Run affected tests and observe short-window/isolation failures before changing production code.
- [ ] Step 3: Keep feature extraction/classification in their present order, then prepare display input independently.

```python
display_start, display_stop = event_display_window(
    event.start_sample, event.end_sample, sample_count,
    profile.artifact_pad_seconds, metadata.sample_rate_hz,
)
display_data = artifact_data
if (display_start, display_stop) != (window_start, window_stop):
    display_raw = reader.read_block(reader_source, display_start, display_stop)
    display_filtered = _preprocess_for_band(
        display_raw, profile, backend,
        (waterfall_filter.low_hz, waterfall_filter.high_hz),
    )
    display_data = replace(
        artifact_data,
        raw_data=display_raw.data,
        filtered_data=np.asarray(backend.asnumpy(display_filtered.data), dtype=np.float32),
        sample_offset=display_raw.sample_offset,
        start_time_utc=display_raw.metadata.start_time_utc,
        bad_channel_mask=display_filtered.bad_channel_mask,
    )
```

Account for actual read/preprocess time without changing timing keys. Preserve source fingerprint checks and artifact transaction rollback. In rendering, keep `_event_selection(data,event)` for the analysis beam and evidence, but separately use all display samples with the same event channel interval and its display good-channel mask for three plots. Do not let expanded display arrays reach spectra. Put actual display sample interval and duration on all three plots; only call it a short source when full source length proves that fact. Unknown-source standalone renderer fixtures may describe available data without falsely asserting source length.

- [ ] Step 4: Run complete artifacts/pipeline/window test files and Ruff; inspect source-fingerprint and rollback regressions remain passing.
- [ ] Step 5: Commit `fix(plots): expand display context without altering analysis`.

### Task 3: Historical three-image preview with automatic range

**Files:** `backend/src/das_sentinel/services/waterfall_previews.py`, `backend/src/das_sentinel/api/schemas.py`, `backend/src/das_sentinel/api/routes/waterfall_previews.py`, `tests/test_waterfall_preview_api.py`.

**Interfaces:**
- Request `{}` means automatic event channels; a manual request supplies both finite increasing distance values. Partial/null-only pairs are invalid.
- Response adds `waveform_png_base64: str` with `min_length=8, max_length=16 * 1024 * 1024`.
- `WaterfallPreview` gains `waveform_png: bytes`; service `render(session,event_id,start_m: float | None = None,stop_m: float | None = None)` supports exact automatic selection.
- Consume Task 1 window helper and public waveform renderer.

- [ ] Step 1: Extend existing real TestClient/HDF5 fixture tests to expect three PNG signatures and unchanged DB, source and registered artifact contents.

```python
response = client.post(f'/api/v1/events/{event_id}/waterfall-preview', json={}, headers=csrf_headers)
assert response.status_code == 200
for name in ('raw_png_base64', 'filtered_png_base64', 'waveform_png_base64'):
    assert base64.b64decode(response.json()[name]).startswith(b'\x89PNG\r\n\x1a\n')
```

Use existing fixture login/header helpers. Add automatic single-channel and non-monotonic distances, manual nonzero channel offset, short-source full reads, three-second lower bound, limits checked before read, missing-source/fingerprint failure, and partial/null requests. Historical waveform peak must match highest-score candidate and earliest tied peak; assert absolute marker time against hand-derived timestamp.

- [ ] Step 2: Run API test file and record failures attributable to missing waveform/automatic mode/minimum window.
- [ ] Step 3: Extend schema and route, then the existing service without a new endpoint or task.

```python
automatic = start_m is None and stop_m is None
if automatic:
    read_slice = slice(binding.event.channel_start, binding.event.channel_stop)
else:
    read_slice = channel_slice_for_distance(metadata.channel_distance_m, start_m, stop_m)
sample_start, sample_stop = event_display_window(
    binding.event.start_sample, binding.event.end_sample, sample_count,
    binding.profile.artifact_pad_seconds, metadata.sample_rate_hz,
)
self._validate_read_bounds(sample_start, sample_stop, read_slice)
```

Retain source/binding validations. Automatic mode selects only good channels in the read block without filtering by min/max distances; manual mode retains existing distance selection. Use the selected filtered channels' median for waveform. Construct genuine `EventRecord` time markers from saved event bounds and candidates, with `peak_sample=min(candidates,key=lambda c:(-c.score,c.peak_sample)).peak_sample`; do not invoke fusion, detection or fabricate analysis provenance. Use global sample_offset exactly once. Return only after all images and final fingerprint check succeed. Preserve per-band and automatic/artificial distance notes. Update old two-image tests rather than deleting their safety assertions.

- [ ] Step 4: Run complete preview API tests plus related artifacts/window tests and Ruff. All resource caps and authentication responses must retain their previous behavior.
- [ ] Step 5: Commit `feat(preview): redraw historical waveform with shared time window`.

### Task 4: Atomic three-image frontend and automatic/manual controls

**Files:** `frontend/src/api.ts`, `frontend/src/components/useWaterfallPreview.ts`, `frontend/src/components/ArtifactViewer.tsx`, `frontend/src/components/WaterfallRangeControls.tsx`, `frontend/src/components/ArtifactViewer.test.tsx` (plus `frontend/src/api.test.ts` only if request helper coverage needs updating).

**Interfaces:**
- `WaterfallPreview` adds required `waveform_png_base64: string`.
- `WaterfallPreviewUrls` adds `waveform: string`.
- `generate(startM?: number, stopM?: number): Promise<void>` sends `{}` if neither argument is supplied, otherwise the manual pair.

- [ ] Step 1: Change existing preview fixture to include the waveform and replace the test that asserts waveform stays original with a test proving waveform is replaced while PSD/STFT remain original. Assert the active preview's download URL matches the image. Add automatic untouched/reset request `{}`, edited-input manual payload, single-channel auto success, empty input rejection, third-image decode failure rollback and all-three URL cleanup.

```typescript
expect(requestBody).toEqual({}); // untouched automatic mode
await user.click(screen.getByRole('tab', { name: '波形' }));
expect(screen.getByRole('img', { name: waveformAlt })).toHaveAttribute('src', waveformPreviewUrl);
expect(screen.getByRole('link', { name: '下载' })).toHaveAttribute('href', waveformPreviewUrl);
```

Use the real labels/alt text already defined by the viewer. Assert observed requests/DOM/resource lifecycle, not implementation source text. Existing superseded-response, authentication and event-switch tests remain.

- [ ] Step 2: Run `npm.cmd test -- --run src/components/ArtifactViewer.test.tsx` and capture expected RED.
- [ ] Step 3: Add waveform URL creation and cleanup as part of the existing atomic `previewUrls` function. In catch paths revoke newly created URLs only; do not discard previous good preview. Select the waveform preview in the viewer and keep original download behavior for other tabs. Mark temporary raw/filtered/waveform consistently in Chinese.

```typescript
const body = startM === undefined && stopM === undefined
  ? {}
  : { distance_start_m: startM, distance_stop_m: stopM };
```

Keep a simple `manual` boolean in range controls, reset on options/event change and reset button; any input edit sets it true. Untouched generation bypasses manual equal-range rejection. Manual mode rejects blank strings before numeric conversion and retains current bounds/order validation. Preserve busy-disabled generation/reset controls; do not add a server cancellation mechanism. Input editing does not rename/reinterpret an already displayed preview.

- [ ] Step 4: Run all frontend tests and a production build into a unique staging directory (do not overwrite the live `frontend/dist` while developing): `npm.cmd run build -- --outDir dist-plot-time-window`. Do not commit generated output.
- [ ] Step 5: Commit `feat(ui): synchronize waveform and waterfall previews`.

### Task 5: User guidance, visual verification and final regression

**Files:** Modify the existing Chinese beginner guide identified by `rg --files docs` and its relevant preview/time-axis sections only. Put generated validation images/reports in this plan's ignored SDD workspace, not the evidence store.

**Interfaces:** All preceding public behavior is frozen; new correctness findings go through a scoped fix dispatch and review, not unreviewed controller edits.

- [ ] Step 1: Document the actual workflow: open an existing event, keep automatic distance or edit both values, generate once, switch among three tabs, read top reference plus relative seconds, download current view, reset to original. Explain source-short exception and unchanged PSD/STFT/features. Include the `.234` second example.
- [ ] Step 2: Render synthetic examples using the production public functions: three-second, midnight-crossing, full short source and single-sample. Inspect all PNGs via image viewer, including labels, notes, legend, margins and time alignment. Do not alter source data or production artifacts.
- [ ] Step 3: Run full backend once and frontend tests/build on final code, plus targeted Ruff and `git diff --check`.

```powershell
& 'D:/das-sentinel-runtime/.venv/Scripts/python.exe' -m pytest -q
& 'C:/Program Files/nodejs/npm.cmd' test -- --run
& 'C:/Program Files/nodejs/npm.cmd' run build -- --outDir dist-plot-time-window
```

The last two commands run in `frontend/`. Record exact pass counts and baseline warnings. Commit only updated human documentation and any separately reviewed verification fixture if needed.
- [ ] Step 4: Controller obtains final whole-change review of Tasks 1–5, addresses blocking findings and records final evidence. No merge/push is part of this plan.
- [ ] Step 5: After review, safely restart only the scoped API with the staged frontend using the inspected existing runtime helper. Keep worker stopped and queue unchanged. If the helper cannot safely perform this action, report the exact constraint rather than changing secrets/configuration or bypassing a rejection.
- [ ] Step 6: Verify `/health` and delivered frontend bundle, then generate one authenticated existing-event temporary preview and verify three PNGs/UI. Use only original source reads, never enqueue analysis. Confirm database job counts still have zero queued/running. Report implemented/deployed state distinctly and preserve historical PNGs.

## Self-review and execution choice

Coverage: spec windows/axes → Task 1; analysis isolation → Task 2; immutable historical previews/limits → Task 3; three-image lifecycle → Task 4; documentation/visual/regression/deployment → Task 5. Shared signatures are defined at their producing tasks. Source-short notes are part of rendered images, not extra response fields.

Use the user's previously selected subagent-driven workflow in this same session; do not ask again between tasks. Controllers keep a plan-scoped ledger, provide each implementer only its extracted brief and constraints, and obtain task-scoped spec/quality review before moving on.
