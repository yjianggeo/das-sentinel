import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_threshold_uses_development_positives_and_includes_ties():
    from das_sentinel.window_benchmark import classification_metrics, recall_threshold

    threshold = recall_threshold([1, 0, 1, 0, 1], [.9, .85, .8, .8, .1], target=2 / 3)
    assert threshold == .8
    metrics = classification_metrics([1, 0, 1, 0, 1], [.9, .85, .8, .8, .1], threshold)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (2, 2, 1, 0)
    assert metrics["recall"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="positive"):
        recall_threshold([0, 0], [.1, .2])


def test_average_precision_is_invariant_to_tied_row_order():
    from das_sentinel.window_benchmark import classification_metrics

    assert classification_metrics([1, 0], [.5, .5], .5)["average_precision"] == .5
    assert classification_metrics([0, 1], [.5, .5], .5)["average_precision"] == .5


def make_manifest(tmp_path):
    from das_sentinel.window_training import TASK, split_rows

    distance = np.arange(8, dtype="<f8") * 4.1
    grid_hash = hashlib.sha256(distance.tobytes()).hexdigest()
    grid = {"version": "native-channel-union-v1", "channels": 8, "spacing_m": 4.1,
            "start_locus": 0, "grid_sha256": grid_hash}
    rows = []
    for day in range(1, 4):
        for positive in range(2):
            path = tmp_path / f"{day}-{positive}.npz"
            wave = np.ones((8, 3000), np.float32)
            wave[:, 1000:1600] *= 1 + positive * 5
            np.savez(path, waveform=wave, distance_m=distance, relative_seconds=np.arange(3000) / 100,
                     valid_channel_mask=np.ones(8, bool), recognition_channel_mask=np.ones(8, bool),
                     region_id=np.arange(8) % 3)
            date = f"2020-01-{day:02}"
            rows.append({"window_id": path.stem, "date": date, "catalog_ids": [],
                         "read_start_utc": date + "T00:00:00+00:00",
                         "read_end_utc": date + "T00:02:00+00:00", "sources": [],
                         "grid_sha256": grid_hash, "grid_normalization": grid["version"],
                         "unit": "strain", "window_seconds": 30,
                         "labels": {"earthquake": "present" if positive else "absent"},
                         "label_origin": "individual_review", "waveform": str(path),
                         "waveform_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    split_rows(rows)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 2, "task": TASK, "site_id": "synthetic",
                                "training_grid": grid, "rows": rows}))
    return path


def test_rejects_cross_split_group_and_changed_arrays(tmp_path):
    from das_sentinel.window_benchmark import run_benchmark, validate_manifest

    path = make_manifest(tmp_path)
    original = json.loads(path.read_text())
    modified = json.loads(path.read_text())
    modified["rows"][0]["split"] = "dev" if modified["rows"][0]["split"] != "dev" else "test"
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="split"):
        validate_manifest(path)
    path.write_text(json.dumps(original))
    from pathlib import Path

    Path(original["rows"][0]["waveform"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        run_benchmark(path, tmp_path / "changed", epochs=1, candidates=("tiny_pooled",))


@pytest.mark.parametrize("candidate", ["tiny_pooled", "residual_pooled", "residual_waveform"])
def test_real_benchmark_preserves_split_and_reloads_selected_model(tmp_path, candidate):
    from das_sentinel.classifiers.window_model import prepare
    from das_sentinel.window_benchmark import (
        candidate_inputs,
        load_candidate,
        run_benchmark,
    )

    path = make_manifest(tmp_path)
    frozen = path.read_bytes()
    output = tmp_path / "result"
    report = run_benchmark(path, output, epochs=2, candidates=(candidate,))
    assert path.read_bytes() == frozen
    assert report["manifest_sha256"] == hashlib.sha256(frozen).hexdigest()
    assert report["publication_ready"] is False
    result = report["candidates"][candidate]
    assert result["selected_epoch"] in (1, 2)
    assert result["test"]["known"] == 2
    assert result["test"]["tp"] + result["test"]["fn"] == 1
    model = load_candidate(output / candidate)
    meta = json.loads((output / candidate / "model.json").read_text())
    rows = json.loads(path.read_text())["rows"]
    predictions = json.loads((output / candidate / "predictions.json").read_text())
    row = next(r for r in rows if r["split"] == "test")
    from das_sentinel.classifiers.window_resnet import prepare_waveform

    preparer = prepare_waveform if candidate == "residual_waveform" else prepare
    with np.load(row["waveform"]) as arrays, torch.no_grad():
        probability = model(*preparer(arrays)).sigmoid().item()
    saved = next(p for p in predictions if p["window_id"] == row["window_id"])
    assert probability == pytest.approx(saved["probability"])
    assert saved["split"] == "test"
    with np.load(row["waveform"]) as arrays:
        wrong_time = {key: arrays[key] for key in arrays}
    wrong_time["relative_seconds"] *= 2
    with pytest.raises(ValueError, match="time grid"):
        candidate_inputs(wrong_time, meta)
    with pytest.raises(FileExistsError):
        run_benchmark(path, output, epochs=1, candidates=("tiny_pooled",))
    (output / candidate / "weights.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash"):
        load_candidate(output / candidate)


def test_review_update_keeps_frozen_split_but_cannot_change_sources(tmp_path):
    from das_sentinel.window_benchmark import validate_manifest
    from das_sentinel.window_training import split_rows

    source = make_manifest(tmp_path)
    updated = json.loads(source.read_text())
    updated["rows"] += [{**r, "window_id": r["window_id"] + "-copy"}
                        for r in updated["rows"]]
    split_rows(updated["rows"])
    source.write_text(json.dumps(updated))
    updated["split_reference"] = {
        "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()
    }
    row = next(r for r in updated["rows"]
               if r["split"] == "test" and r["labels"]["earthquake"] == "present")
    row["labels"] = {"earthquake": "absent"}
    row["review_id"] = "new-human-review"
    path = tmp_path / "review-updated.json"
    path.write_text(json.dumps(updated))
    # No new independent test is implied: preserve the original groups exactly.
    assert validate_manifest(path)["rows"] == updated["rows"]
    row["waveform_sha256"] = "changed"
    path.write_text(json.dumps(updated))
    with pytest.raises(ValueError, match="reference"):
        validate_manifest(path)
    updated["split_reference"]["sha256"] = "wrong"
    path.write_text(json.dumps(updated))
    with pytest.raises(ValueError, match="reference"):
        validate_manifest(path)


def test_cropped_benchmark_reloads_with_the_same_input_contract(tmp_path):
    from das_sentinel.window_benchmark import (
        candidate_inputs,
        load_candidate,
        run_benchmark,
    )

    path = make_manifest(tmp_path)
    output = tmp_path / "cropped"
    report = run_benchmark(path, output, epochs=1,
                           candidates=("residual_waveform",), distance_start_m=13.)
    folder = output / "residual_waveform"
    meta = json.loads((folder / "model.json").read_text())
    assert meta["distance_start_m"] == 13.
    assert meta["input_channels"] == 4
    assert report["protocol"]["distance_start_m"] == 13.
    cached = np.load(output / "cache/0-waveform.npy")
    assert cached.shape == (2, 4, 3000)
    rows = json.loads(path.read_text())["rows"]
    row = next(r for r in rows if r["split"] == "test")
    model = load_candidate(folder)
    with np.load(row["waveform"]) as arrays, torch.inference_mode():
        probability = model(*candidate_inputs(arrays, meta)).sigmoid().item()
    saved = next(p for p in json.loads((folder / "predictions.json").read_text())
                 if p["window_id"] == row["window_id"])
    assert probability == pytest.approx(saved["probability"])
    assert saved["split"] == "test"


def test_external_cache_preserves_predictions_and_never_overwrites(tmp_path):
    from das_sentinel.window_benchmark import run_benchmark

    path = make_manifest(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    cache = tmp_path / "ssd/cache"
    run_benchmark(path, first, epochs=1, candidates=("residual_waveform",), distance_start_m=13.)
    run_benchmark(path, second, epochs=1, candidates=("residual_waveform",),
                  distance_start_m=13., cache_directory=cache)
    assert (cache / "0-waveform.npy").exists()
    assert not (second / "cache").exists()
    assert ((first / "residual_waveform/predictions.json").read_bytes()
            == (second / "residual_waveform/predictions.json").read_bytes())
    original = (cache / "0-waveform.npy").read_bytes()
    with pytest.raises(FileExistsError):
        run_benchmark(path, tmp_path / "third", epochs=1,
                      candidates=("residual_waveform",), cache_directory=cache)
    assert (cache / "0-waveform.npy").read_bytes() == original


def test_class_group_sampling_keeps_duplicate_windows_from_dominating():
    from das_sentinel.window_benchmark import training_sampling_weights

    rows = [
        {"split": "train", "group": group, "labels": {"earthquake": label}}
        for label, group, count in (
            ("present", "event-a", 9), ("present", "event-b", 1),
            ("absent", "noise-a", 1), ("absent", "noise-b", 3),
        ) for _ in range(count)
    ]
    rows.append({"split": "test", "group": "held-out", "labels": {"earthquake": "present"}})
    weights = training_sampling_weights(rows, list(range(14)))
    # Each class has half the mass; each of its two groups has one quarter.
    assert weights[:9].sum().item() == pytest.approx(.25)
    assert weights[9].item() == pytest.approx(.25)
    assert weights[10].item() == pytest.approx(.25)
    assert weights[11:14].sum().item() == pytest.approx(.25)
    rows[-1]["labels"]["earthquake"] = "absent"
    assert torch.equal(weights, training_sampling_weights(rows, list(range(14))))
    with pytest.raises(ValueError, match="train"):
        training_sampling_weights(rows, list(range(15)))


def test_group_balanced_benchmark_repeats_fixed_steps_reproducibly(tmp_path):
    from das_sentinel.window_benchmark import run_benchmark

    path = make_manifest(tmp_path)
    frozen = path.read_bytes()
    first, second = tmp_path / "balanced-first", tmp_path / "balanced-second"
    for output in (first, second):
        report = run_benchmark(path, output, epochs=2, candidates=("residual_waveform",),
                               sampling="class_group_balanced")
        assert report["protocol"]["sampling"] == "class_group_balanced"
        assert report["protocol"]["positive_loss_weight"] == 1.
        history = json.loads((output / "residual_waveform/history.json").read_text())
        assert len(history) == 2
        assert all(sum(entry["sampled_classes"].values()) == 2 for entry in history)
        assert report["candidates"]["residual_waveform"]["test"]["known"] == 2
    assert path.read_bytes() == frozen
    assert ((first / "residual_waveform/predictions.json").read_bytes()
            == (second / "residual_waveform/predictions.json").read_bytes())
    with pytest.raises(ValueError, match="sampling"):
        run_benchmark(path, tmp_path / "bad-sampling", sampling="test_labels")


def test_waveform_only_cache_matches_online_float32_inputs(tmp_path):
    from das_sentinel.classifiers.window_resnet import prepare_waveform
    from das_sentinel.window_benchmark import run_benchmark

    path = make_manifest(tmp_path)
    output = tmp_path / "waveform-only"
    run_benchmark(path, output, epochs=1, candidates=("residual_waveform",))
    assert not list((output / "cache").glob("*-pooled.npy"))
    rows = json.loads(path.read_text())["rows"]
    for index, row in enumerate(rows):
        with np.load(row["waveform"]) as arrays:
            expected = prepare_waveform(arrays)[0][0].numpy()
        cached = np.load(output / f"cache/{index}-waveform.npy", allow_pickle=False)
        assert cached.dtype == np.float32
        assert np.array_equal(cached, expected[:2])


def test_two_plane_cache_reconstructs_identical_masked_waveform_and_prediction(tmp_path):
    from das_sentinel.classifiers.window_resnet import prepare_waveform
    from das_sentinel.window_benchmark import CachedWindows, make_model

    path = make_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    row = manifest["rows"][0]
    with np.load(row["waveform"]) as saved:
        arrays = {key: saved[key] for key in saved}
    arrays["waveform"] = np.random.default_rng(42).normal(size=(8, 3000)).astype(np.float32)
    arrays["valid_channel_mask"][1] = False
    arrays["recognition_channel_mask"][3] = False
    np.savez(row["waveform"], **arrays)
    from pathlib import Path

    row["waveform_sha256"] = hashlib.sha256(Path(row["waveform"]).read_bytes()).hexdigest()
    cached = CachedWindows(manifest, tmp_path / "two-planes", waveform=True, pooled=False)
    expected = prepare_waveform(arrays)
    stored = np.load(cached.folder / "0-waveform.npy", allow_pickle=False)
    assert stored.shape == (2, 8, 3000)
    assert stored.dtype == np.float32
    assert np.array_equal(stored, expected[0][0, :2].numpy())
    restored = cached.inputs(0, "residual_waveform")
    assert all(np.array_equal(a.numpy(), b.numpy()) for a, b in zip(restored, expected))
    torch.manual_seed(42)
    model = make_model("residual_waveform").eval()
    with torch.inference_mode():
        assert torch.equal(model(*restored), model(*expected))


def test_waveform_only_cache_does_not_prepare_pooled_features(tmp_path, monkeypatch):
    import das_sentinel.window_benchmark as benchmark
    from das_sentinel.classifiers.window_resnet import prepare_waveform

    path = make_manifest(tmp_path)
    manifest = json.loads(path.read_text())

    def unexpected_pooled(_arrays):
        raise AssertionError("waveform-only cache must not prepare pooled features")

    monkeypatch.setattr(benchmark, "prepare", unexpected_pooled)
    cached = benchmark.CachedWindows(manifest, tmp_path / "waveform-only", waveform=True, pooled=False)

    assert not list(cached.folder.glob("*-pooled.npy"))
    for index, row in enumerate(cached.rows):
        with np.load(row["waveform"], allow_pickle=False) as arrays:
            expected = prepare_waveform(arrays)
        assert all(torch.equal(left, right) for left, right in
                   zip(cached.inputs(index, "residual_waveform"), expected))


@pytest.mark.parametrize("waveform", [False, True])
def test_pooled_cache_preserves_inputs_with_or_without_waveform(tmp_path, waveform):
    from das_sentinel.classifiers.window_model import prepare
    from das_sentinel.classifiers.window_resnet import prepare_waveform
    from das_sentinel.window_benchmark import CachedWindows

    path = make_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    cached = CachedWindows(manifest, tmp_path / "pooled", waveform=waveform, pooled=True)

    for index, row in enumerate(cached.rows):
        with np.load(row["waveform"], allow_pickle=False) as arrays:
            expected_pooled = prepare(arrays)
            expected_waveform = prepare_waveform(arrays) if waveform else None
        for candidate in ("tiny_pooled", "residual_pooled"):
            assert all(torch.equal(left, right) for left, right in
                       zip(cached.inputs(index, candidate), expected_pooled))
        if waveform:
            assert all(torch.equal(left, right) for left, right in
                       zip(cached.inputs(index, "residual_waveform"), expected_waveform))
        else:
            assert not (cached.folder / f"{index}-waveform.npy").exists()


@pytest.mark.parametrize("invalid", ["grid", "mask", "nonfinite"])
def test_waveform_only_cache_still_rejects_invalid_input(tmp_path, invalid):
    from pathlib import Path

    from das_sentinel.window_benchmark import CachedWindows

    path = make_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    row = manifest["rows"][0]
    with np.load(row["waveform"], allow_pickle=False) as saved:
        arrays = dict(saved)
    if invalid == "grid":
        arrays["relative_seconds"][1] = 10
    elif invalid == "mask":
        arrays["recognition_channel_mask"][:] = False
    else:
        arrays["waveform"][0, 0] = np.nan
    np.savez(row["waveform"], **arrays)
    row["waveform_sha256"] = hashlib.sha256(Path(row["waveform"]).read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="grid mismatch" if invalid == "grid" else "invalid whole-window waveform"):
        CachedWindows(manifest, tmp_path / "invalid", waveform=True, pooled=False)
