# First Trained JJK Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Preserve the existing worktree and all unrelated edits. User authorized execution through training; do not pause for execution-choice approval.

**Goal:** Produce a trained, evaluated and reloadable first JJK candidate model.

**Architecture:** Repair shared grouped class coverage without weakening training gates. Freeze a common physical cohort and holdout, compare feature baselines and a frozen DASNet encoder with a locally trained classifier, then save evidence and a local candidate.

**Tech Stack:** Python, NumPy, SciPy, scikit-learn/skops, isolated CPU PyTorch/torchvision.

**Spec:** docs/superpowers/specs/2026-09-14-first-trained-model-design.md

## Global Constraints

- Keep original labels, full-manifest groups and existing uncommitted Chinese documentation.
- Never use fixed holdout metrics to select a model or change preprocessing.
- Never alter production dependencies, publish gates, or activate an unqualified candidate.
- No Git push/merge, no restoration of cancelled jobs; keep current worker running.

### Task 1: Shared feasible group folds

Files: classifiers/group_folds.py; classifiers/supervised.py; services/training_readiness.py; focused tests.

Interface: `make_group_folds(labels, groups, *, n_splits=5, seed=20260726)` returns list of NumPy train/test index pairs; raises ValueError when coverage cannot be obtained.

- [x] Add failing regression with five rare-class groups where SGKF omits a class; assert complete test coverage, each class on both sides, and no group overlap.
- [x] Implement SGKF-first with bounded deterministic constrained fallback; no feature access. Reject fewer than n_splits groups for any class. Example contract: `folds = make_group_folds(labels, groups); assert all(set(labels[te]) == set(labels) for _, te in folds)`.
- [x] Route trainer and readiness through this exact helper, retaining TrainingGateError and blocker behavior.
- [x] Run group, supervised, readiness and model API tests; inspect diff without commits to unrelated work.

### Task 2: Reproducible frozen encoder

Files: classifiers/dasnet_transfer.py; tests/test_dasnet_transfer.py; isolated runtime and source checkpoint outside repository.

Interface: load verified checkpoint into frozen ResNet50; `encode_sample(arrays, metadata, event_id)` returns finite fixed-size ROI embedding. Expose preparation/ROI validation separately for lightweight tests.

- [x] Test true time/channel orientation, local ROI pooling, masked suspended channels, nonfinite/invalid rate rejection, deterministic dimensions.
- [x] Implement strict safe checkpoint load with original layer1 stride; no implicit downloads. Use raw/2–10/>10 Hz strain-rate views, clipped channel standardization and ImageNet transform.
- [x] Verify real checkpoint keys, actual sample shape and CPU/memory cost before batch processing.
- [x] Keep heavy dependencies optional and isolated; execute focused tests in that environment and normal suite without torch.

### Task 3: Common cohort export and experiment

Files: reproducible scripts under manual-work/first-model-20260914; docs/first-trained-model.md.

- [x] Verify each actual source/ROI coverage and fixed grid with existing waveform exporter; record all failures before fitting. Reuse verified samples and immutable resampled context where safe.
- [x] Save final cohort and original frozen outer/inner group assignment hashes. Assert `set(test_groups).isdisjoint(dev_groups)` and every inner train/validation class coverage before fit.
- [x] Fit predeclared feature and frozen-embedding candidates using inner folds only, with preprocessing fitted inside each fold. Save per-fold predictions and scores.
- [x] Lock selected recipe to JSON before loading test labels for metrics. Fit development model and evaluate fixed test once; save confusion matrix and class metrics.
- [x] Refit final recipe on all eligible labels; save model, schema, provenance, safe reload prediction smoke test. Register supported RF baseline through existing lifecycle as candidate if it remains useful; never mislabel a new architecture as RF.

### Task 4: Review and handoff

- [x] Independent correctness/leakage review; fix substantiated issues and rerun appropriate checks.
- [x] Verify actual local model artifact and candidate listing, health, unchanged worker/labels and baseline preservation.
- [x] Finish Chinese model card and progress with measured results, uncertainty and exact artifact paths.
