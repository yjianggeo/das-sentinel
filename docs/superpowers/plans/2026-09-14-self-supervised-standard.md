# JJK Standard Self-Supervised Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. User approved execution through training; keep working without repeated approval.

**Goal:** Train, compare, save and verify one non-low-frequency JJK self-supervised encoder and four-class candidate.

**Architecture:** Independently sampled, date-isolated standard waveform windows train a compact masked convolutional autoencoder. Frozen ROI features support a paired labelled-budget experiment against the same random encoder and an existing feature baseline.

**Tech Stack:** Existing reader Python environment; isolated CPU PyTorch 2.8/torchvision0.23; NumPy/SciPy/sklearn/skops.

**Spec:** docs/superpowers/specs/2026-09-14-self-supervised-standard-design.md

## Global Constraints

- Existing worktree, branch, labels, prior experiments, API models and worker preserved; no Git push/merge or publication.
- Only standard60s/100Hz/832×6000; model input0.5–40Hz; valid and recognition masks mandatory.
- Entire supervised dates±611s excluded from SSL pool; date-separated SSL validation; exact source/geometry checks before training.
- Prior81 test is historical, never new independent evidence; no classifier selection on its scores.
- CPU4threads, batch32, maximum20epochs; lower batch only for measured resource necessity.

### Task 1: Source audit and actual unlabeled export

Files: manual-work/self-supervised-standard-20260914/source-pool-audit.json; ssl_export.py; test_ssl_export.py; training-datasets/jjk-self-supervised-standard-20260914.

Interface: samples-index.json contains complete, samples[{window_id, status, path, split, utc_day, reason}], no class labels. Successful directories use existing waveform_samples schema, annotations empty, supervised_eligible=false.

- [x] Audit source pool and grid, freeze eligible dates and label-free deterministic sampling rule before reads.
- [x] Test duplicate time exclusion, exact supervised-day/context exclusion, no random splitting within a date, and missing-context rejection.
- [x] Implement using existing build_sample/save_sample/load_sample; reference grid equal, no continuity changes. Sample candidates evenly in time per date, require source+10s context, cap24train/8validation. Record every failure.
- [x] Export actual windows, revalidate saved arrays/metadata/provenance, freeze final index SHA before fitting.

### Task 2: Lightweight masked encoder and ROI representation

Files: backend/src/das_sentinel/classifiers/self_supervised.py; tests/test_self_supervised.py.

Interfaces: prepare_standard(arrays,metadata)->float32[832,6000]; make_model(seed)->optional Torch module with encode([B,2,64,512])->[B,64,8,64]; block_mask(valid,rng)->bool[B,1,64,512]; masked_loss(model,x,valid,mask)->scalar; encode_roi(model,prepared,roi,recognition,batch_size=16)->float32[128]. ROI=(channel_start,channel_stop,time_start,time_stop), half-open original indices.

- [x] Write failing tests for rejecting low_frequency/wrong grid/nonfinite data, masked-only loss, visibility handling, excluded channels, exact ROI pooling/tail padding and safe checkpoint deterministic reload.
- [x] Implement exact spec architecture and preprocessing; Torch imports remain optional so main API environment can import the module.
- [x] Verify synthetic20Hz signal preservation and no temporal resizing, fixed feature shape, finite deterministic inference, and honest zero-valid rejection.
- [x] Benchmark real patches in isolated environment and report time/RSS before bulk training.

### Task 3: Pretraining and paired classification experiment

Files: manual-work/self-supervised-standard-20260914/train_ssl.py, compare_ssl.py, predict_ssl.py, tests for leakage/roundtrip/selection.

- [x] Freeze config/source/supervised/SSL split/code hashes; assert all supervised dates excluded from SSL windows and read context.
- [x] Build16 patches/window with deterministic seed; train AdamW20epoch/patience5 using masked validation MSE only. Save initialization, best tensor weights, loss curve and reload evidence.
- [x] Encode same554 reviewed ROIs with initial and pretrained encoder; validate IDs/dimension/finite/hash, save immutable caches.
- [x] Fit seven fixed recipes for25/50/100% nested labelled budgets under original four development group folds. Standardize only fold training. Save per-event predictions and actual budget counts.
- [x] Lock best SSL recipe by100% OOF score; conduct one historical81 sample comparison against predeclared controls, no retuning. Refit selected SSL head on554 labels, save JSON parameters with float64 training/replay.
- [x] Fresh-process prediction from an actual NPZ sample must match cached inference; retain all artifacts and no API publication.

### Task 4: Independent review and handoff

Files: docs/self-supervised-standard-model.md; this plan; .superpowers/sdd/2026-09-14-self-supervised-standard/progress.md.

- [x] Independent review of data isolation, mask loss, ROI semantics, code and checksums; fix substantiated issues before final claims.
- [x] Plot measured reconstruction/label-efficiency results and visually inspect output; distinguish reconstruction performance, classification benefit and historical-test limitations.
- [x] Verify original edits/labels/model status/worker preserved and document exact saved model, input contract, test results and inference command in Chinese.
