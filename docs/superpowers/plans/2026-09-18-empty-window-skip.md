# Empty Window Skip Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for bounded independent implementation and review tasks. Preserve the existing dirty worktree; no commits or worktree removal.

**Goal:** Skip newly claimed empty 30-second windows before feature measurement or output generation, while preserving existing artifacts and human evidence.

**Architecture:** Analyzer opt-in early return; per-claim authorization and authenticated transactional skip endpoint; compact tombstones prevent repeated scans and retain ETA throughput.

**Tech Stack:** Python, FastAPI, SQLAlchemy/Alembic, NumPy/SciPy, pytest; existing Windows deployment.

**Spec:** `docs/superpowers/specs/2026-09-18-empty-window-skip-design.md`

## Global Constraints

- Exact existing qualified-region semantics; no ML or detector threshold changes.
- Only 30-second queued/future jobs are eligible; existing running jobs, images, reviews, historical non-noise imports and multi-analysis histories are protected.
- Delete no raw files or legacy events/reviews. Skip is not a training label.
- Work in the existing branch and preserve unrelated modifications. No Git commits/push/merge.

### Task 1: Analyzer and worker early exit

Files: `whole_window.py`, `worker/window_main.py`, `tests/test_whole_window.py`, `tests/test_window_worker.py`.

Interface: `analyze_window(..., skip_empty=False)` returns `{"skipped": true, "reason": "no_qualified_regions"}` only after normal source/coverage validation and exact `merge_regions` returns zero. Otherwise return the original result. Default unchanged.

Claim request adds `supports_empty_skip: true`; response contains `skip_empty: bool`, `plan_sha256: str`, `skip_policy_version: str | null`. Only pass analyzer `skip_empty=True` when authorized. HTTP client `skip(job_id, worker_id, attempt_count, plan_sha256)` posts owned fields plus `plan_sha256`, `reason="no_qualified_regions"` to `/{id}/skip`. A successful worker outcome is `skipped`; no result publication. Final heartbeat required before skip. `source_profile_changed` and `skip_not_allowed` 409 map to ResultSubmissionError, preserving worker loop isolation.

- [x] Add failing tests: qualified-empty early return creates no directory; protected/default and qualified signal produce normal artifacts; valid claim uses skip, normal claim uses result, lost lease cannot skip.
- [x] Run focused tests and record expected failures.
- [x] Implement minimal flow above; preserve existing analyzers lacking new kwarg when skip flag absent.
- [x] Run `python -m pytest tests/test_whole_window.py tests/test_window_worker.py tests/test_window_regions.py -q`; report evidence and obtain review.

### Task 2: Persistence, eligibility and API

Files: `persistence/window_models.py`, new migration `0006_window_skips.py`, `services/event_windows.py`, `api/window_schemas.py`, `api/routes/event_windows.py`, new `tests/test_window_skips.py`.

Interface: `WindowAnalysis.skip_policy_version` nullable string. `WindowSkip` stores analysis_id PK, window_id/site_id/start_utc/window_seconds/plan_sha256/policy_version/worker_id/attempt_count/finished_at/reason. Unique window+plan+policy, finish indexed. `EMPTY_SKIP_POLICY = "empty-regions-v1"`. `WindowService.claim(worker_id, supports_empty_skip=False)`; `skip(aid, worker_id, attempt_count, plan_sha256, reason="no_qualified_regions")` returns `{"id": aid, "status": "skipped"}`.

- [x] Test empty skip removes only authorized task/window; old worker/running claim, reviewed/generated/versioned/positive-import windows protected; source change, cancellation, lease loss rejected; identical replay success; conflicting replay rejected; noise-only draft deletion safe; enqueue same plan no ghost, changed plan allowed.
- [x] Run tests before implementation.
- [x] Add nullable column and tombstone table with inspect/checkfirst migration for fresh/deployed DBs. Split WindowOwned inheritance to avoid claim capability leaking into owned endpoints.
- [x] Move plan hashing before window creation and honor tombstone. Compute eligibility on claim and recheck on skip. Include current latest legacy review protection in addition to frozen import data.
- [x] Run targeted service/API/planning/migration regressions; independent review.

### Task 3: ETA, integration and rollout

Files: `services/job_estimate.py`, `tests/test_job_estimate.py`, `tests/test_window_worker_api.py`, `docs/whole-window-events.md`, `docs/operations.md`.

- [x] Test recent skipped timestamps contribute to throughput and activity; merge newest 31 completed/suppressed timestamps within the existing two-hour lookback.
- [x] End-to-end worker/API skip test verifies no publication or artifacts and safe duplicate skip endpoint.
- [x] Review entire feature diff, run focused regression suite once.
- [x] Backup SQLite and capture protected IDs/artifacts/reviews. Additive migration, warm API with temporary claim gate, drain current worker then start new worker and release gate. Validate unchanged old output/history and live skip/normal completion/ETA. Document exact outcomes.

## Verification examples

```python
assert service.claim("old")["skip_empty"] is False
assert service.claim("new", supports_empty_skip=True)["skip_empty"] is True
assert service.skip(aid, "new", 1, plan_sha256)["status"] == "skipped"
assert session.get(EventWindow, wid) is None
assert enqueue_windows(session, site.id, profile.id, [same_plan]) == []
```
