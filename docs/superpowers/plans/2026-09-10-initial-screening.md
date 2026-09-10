# Initial Screening V2 Implementation Plan

> 历史第一阶段计划：其中生产保留 v1 的限制，已由用户后续确认的[版本2上线决定](../../production-screening-v2.md)取代。以下保留当时的评估计划。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement and evaluate improved non-AI DAS initial screening and versioned event measurements, with usable local review UI.

**Architecture:** Preserve v1 code/model inputs as the production baseline. Add opt-in pure v2 detector/measurement modules, a read-only offline replay tool, and an authenticated endpoint serving verified measurement sidecars.

**Tech Stack:** Existing Python 3.12+, NumPy/SciPy, FastAPI, SQLite, React/TypeScript/Vitest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-10-initial-screening-design.md`

## Global Constraints

- Work only in existing `fix/jjk-scan-progress` worktree; preserve all pre-existing edits. No commit/push/merge, destructive cleanup, formal job creation, worker restart, or review/label writes.
- Legacy detector/feature/model behavior remains unchanged by default. New detectors are opt-in offline candidates until independently evaluated.
- Do not append new keys to legacy `features`; use measurement sidecars under `app_data_dir/screening-measurements/v2/<event-id>.json`.
- All measurement values are finite floats or null with an explicit status; no propagation estimate means unavailable, not zero velocity.
- Tests first for new behavior. Save commands and outputs in task reports. Use `D:/das-sentinel-runtime/.venv/Scripts/python.exe -m pytest`; no global installs.
- APIs do not read source waveforms; sidecars must match database event/source identity. Existing feature scroll and bottom alignment remain.
- Dataset comparison is exploratory; report labeled detection counts and unreviewed candidates, never infer continuous-data false-positive/false-negative rates from selected events alone.

### Task 1: Opt-in v2 screening algorithms

**Files:** Create `backend/src/das_sentinel/screening.py`, `tests/test_screening.py`. Do not edit legacy detectors, config, fusion or pipeline.

**Interfaces:**
```python
@dataclass(frozen=True)
class ScreeningOptions:
    spatial_window_m: float = 256.0
    spatial_overlap: float = 0.5
    off_ratio: float = 2.0
    release_gap_seconds: float = 0.15
    short_sta_seconds: float = 0.1
    short_lta_seconds: float = 2.0
    coincidence_tolerance_seconds: float = 0.03
    common_mode_removal: bool = False
    enable_coherence: bool = True
    enable_trajectory: bool = True

@dataclass(frozen=True)
class ScreeningResult:
    candidates: tuple[Candidate, ...]
    events: tuple[EventRecord, ...]
    diagnostics: dict[str, Any]

def screen_block(block: DasBlock, profile: ProcessingProfile,
                 backend: ArrayBackend,
                 options: ScreeningOptions | None = None) -> ScreeningResult: ...
```

Result sample coordinates remain absolute relative to source; channel indices refer to original full block channels. `diagnostics` records algorithm version, options, detector timings/counts, effective sample coverage/cold-start interval and skipped geometry. IDs deterministic from input range/options and evidence. Reject invalid options.

- [x] Write failing synthetic tests for strong ratio, scattered vs local supported channels, release hysteresis, short-scale cold start, same-block two coherent pulses, near-zero moveout soft flag and long-cable local trajectory. Make helper blocks directly with `DasMetadata/DasBlock` and seeded noise. Confirm import/test failures before code.
- [x] Implement local physical-distance windows, per-window robust amplitude support, two STA/LTA scales with on/off hysteresis and time-tolerant coincidence; strong threshold uses profile.stalta_strong_ratio. Record detector evidence ratios and support rather than calling scores probabilities. Reuse existing preprocessing but honor options.common_mode_removal independently from legacy defaults.
- [x] Add multi-band energy/background comparison; STA/LTA and energy must share amplitude-family semantics in fusion. Keep energy-only strong candidates possible. Add local spatial quality evidence without pretending overlapping neighbors are independent stations.
- [x] Compute local semblance curves with existing `_moveout_semblance` or equivalent bounded vectorized calculation, pick multiple separated peaks, retain near-zero candidates with ambiguity diagnostics. Screen independently of amplitude branch; exclude only invalid data, not arbitrary quality warnings.
- [x] Apply existing trajectory detector on physically feasible local windows, map indices back; skip infeasible windows before expensive search. Bound processing and suppress overlap duplicates; do not treat downhole apparent motion as a definitive traffic label.
- [x] Verify focused tests and legacy detector/fusion tests. Record report and expose diagnostics sufficient for per-detector ablations.

### Task 2: Versioned scientific measurements

**Files:** Create `backend/src/das_sentinel/measurements.py`, `tests/test_measurements.py`. Do not edit legacy features, model vectorizer, API or UI.

**Interfaces:**
```python
def measure_event(block: DasBlock, event: EventRecord,
                  profile: ProcessingProfile) -> dict[str, Any]: ...
# {'schema_version':'2', 'values':{name: float | None},
#  'status':{name: 'measured' | reason}, 'context':{...}}
```

Values include `input_rms`, `snr_db`, `channel_snr_median_db`, `channel_snr_p90_db`, `valid_channel_count`, `valid_channel_fraction`, `bad_channel_fraction`, `clipping_fraction`, `trigger_duration_s`, `energy_duration_5_95_s`, `rise_time_s`, `pulse_count`, `pulse_interval_s`, `dominant_frequency_hz`, `spectral_centroid_hz`, `spectral_entropy`, `frequency_resolution_hz`, `band_fraction_low`, `band_fraction_mid`, `band_fraction_high`, `active_channel_fraction`, `longest_active_span_m`, `distance_extent_m`, `lagged_correlation`, `slowness_s_per_m`, `apparent_velocity_mps`, `direction`, `moveout_residual_s`. Additional minimal quality metrics allowed if justified in report.

Context includes effective frequency band, sampling rate, input unit, background seconds available, event/time/channel window, common-mode treatment (none for measurements), algorithm version and quality flags. Do not classify, mutate input, or inspect path names as features.

- [x] Write tests using seeded background plus one/two finite sinusoidal bursts; an isolated 2-second 10Hz burst is one pulse, not40. Test stronger amplitude increases SNR, insufficient baseline yieldsnull/reason, unresolvable propagation produces no0velocity, invalid channels, and evidence changes do not change measurement values.
- [x] Use finite-channel filtering without discarding a channel solely for real high-amplitude event energy. Detrend and use fixed 0.5–80Hz bounded by Nyquist; report actual band. Keep amplitude calibration honest.
- [x] Estimate pre-event noise from [-5,-0.5] seconds when at least1second usable; apply identical filtering to noise/event. Use smoothed Hilbert envelope with prominence/minimum spacing or supported segments for independent pulses; handle constant envelope and clipped windows explicitly.
- [x] Compute energy-duration and frequency/space measurements with meaningful resolution. Do not report precise low-frequency information if too short. Correlation uses bounded local channel pairs and finite lags, reports fit quality, and returns null slow/velocity if not supported. Clipping is null without known ADC limits.
- [x] Return stable finite JSON, use availability flags for missing/undefined values, run focused and legacy feature/model tests, save report including any remaining measurement limitations.

### Task 3: Read-only dataset replay and reproducible evidence

**Files:** Create `backend/src/das_sentinel/screening_evaluation.py`, `tests/test_screening_evaluation.py`, `scripts/evaluate_screening.py`.

**Interfaces:** Consume screen_block/ScreeningOptions and measure_event from Tasks1–2. Snapshot uses event id, source path/fingerprint/stat, sample/channel interval, latest review and split group (source/date); JSON-safe context is explicit.

```python
def snapshot_events(database_path: Path) -> list[dict[str, Any]]: ...
def select_samples(rows: list[dict[str, Any]], noise_limit: int = 16,
                   unreviewed_limit: int = 8) -> list[dict[str, Any]]: ...
def run_evaluation(database_path: Path, output_dir: Path,
                   *, measurement_dir: Path | None = None,
                   event_limit: int | None = None) -> dict[str, Any]: ...
```

- [x] Test latest-review dedup, grouping, deterministic stratified sample inclusion of earthquakes/traffic and energy-only, mismatch/error not counted as detection failure, interval matching and artifact JSON escaping using small temporary SQLite/HDF5 fixtures.
- [x] Read SQLite through mode=ro. Snapshot baseline source stat and evidence. Select all earthquakes/traffic, bounded stratified noise/unreviewed and all energy-only; process grouped sources in bounded windows with at least10sec preceding context and20sec central coverage if available.
- [x] Run same window through v1 `_detect_block` and fusion and v2 screen_block. Record preprocessing/algorithm options, elapsed times, all raw candidates and accepted/rejected groups, matching to reviewed intervals by temporal+spatial intersection; mark boundary coverage and read failures.
- [x] Record common-mode on/off and amplitude/spatial family ablations without falsely deleting matches for missing files. Include deterministic non-event background windows with unknown ground truth, label those candidates unreviewed. Compute detection counts for each reviewed class, unique detector contributions, candidate counts, duplicates and timings, not claimed accuracy.
- [x] Measure selected historical events with v2; produce per-event sidecar containing `event_id`, `source_fingerprint`, `start_sample`, `end_sample`, `channel_start`, `channel_stop`, `measurements`. Atomic local JSON writes, no DB updates. HTML report shows old/new matching, per-class counts and measurement examples with escaped content.
- [x] Run real bounded evaluation, preserve frozen snapshot and skipped reason. Save a separate full measurement-only pass if practical so existing events can display v2 without rerunning detectors.

### Task 4: Authenticated API, review UI and integrated validation

**Files:** Create `backend/src/das_sentinel/services/event_measurements.py`, `tests/test_event_measurements_api.py`; modify `api/routes/events.py`, `frontend/src/api.ts`, `frontend/src/components/EvidencePanel.tsx`, focused tests, `docs/operations.md`.

**Interface:** `GET /api/v1/events/{event_id}/measurements` returns `{available:false}` if no sidecar, else `{available:true, measurements:<Task2 envelope>}`. 404 for unknown event. Resolve using app.state settings app_data_dir; never take filesystem path from request. Reject mismatched source/window, symlinks, oversize/malformed/nonfinite files; unavailable data must not break existing event detail.

- [x] Tests first: unauthenticated/unknown/missing, matching sidecar, stale fingerprint/window, path safety/oversized/invalid float. Verify GET performs no DB writes.
- [x] UI fetches on event change with abort+identity guard; ready data display grouped Chinese names and units, missing values Chinese reasons. Keep legacy values accessible under“历史特征”, add“新版测量”tab only if available; no fabricated v2values for old data. Preserve scroll and form-bottom alignment; do not submit reviews in browser validation.
- [x] Test stale event fetch response, successful grouped render, missing status and fallback. Run backend focused regressions and full frontend suite/build. Independent final review covers new code only against saved dirty baseline.
- [x] Update local API/frontend for read-only sidecar display; worker untouched. Verify current login/session, real selected event v2 values, unchanged legacy28values, and measured bottom alignment. Save deployment and review evidence; document that production detector remainsv1 awaiting blind review of offline differences.

## Final implementation and verification — 2026-09-10

All four implementation tasks are complete. Independent reviews and corrections are recorded in `.superpowers/sdd/2026-09-10-initial-screening/`. The original design remains an offline evaluation: production detector v1 was not replaced.

- Final frozen-code replay: 59 selected historical event windows plus 4 background windows, no errors, 2080.99 seconds end-to-end in two bounded offline processes. All 11 reviewed earthquakes overlap a candidate from both versions, but fused window instances rise from 87 to 783. These figures do not establish improved precision, event boundaries or reduced review burden.
- Versioned measurements: 277/277 completed and verified against database event/source identities. API and frontend deployed locally; actual authenticated browser values and 28 unchanged historical values checked. No manual review submitted.
- Validation: full backend 1045 passed, 4 skipped, 1 performance test deselected; full frontend 137 passed and build passed. Final terminal spectral-bin fix separately verified by 35 measurement and legacy-feature tests. Ruff and Git whitespace checks pass.
- Task4 geometry verification limit: the live browser is a narrow stacked viewport; actual independent scrolling and absence of horizontal overflow were measured there. Wide-layout bottom alignment is covered by existing automated tests, not a newly captured wide-viewport measurement. Existing EventsPage/styles changes were preserved byte-for-byte against the task baseline.
- Deployment recovered an observed nested Vite output-directory hazard; future builds must use a staging --outDir outside frontend/dist. The operations guide and deployment manifest record this. Worker PID/start, 442 jobs, 277 events and 172 review-history rows remained unchanged; no Git integration.

Results and artifacts: `D:/das-sentinel-runtime/manual-work/initial-screening-20260910/paired-v2-reviewed/report.html`, `evaluation-summary.md`, `evaluation-interpretation.md`, `review-packets/index.html`, `sidecar-validation.json`, `deployment/browser-verification.json`, and `final-state-verification.json`. Preliminary runs are retained as preliminary evidence, not substituted for this completed reviewed run.
