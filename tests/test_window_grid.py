import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from das_sentinel.services.discovery import fingerprint


def sample(tmp_path, name, channels, *, spacing=4.1, nominal=4.0):
    source = tmp_path / f"{name}_3400m_8m_4m_source.h5"
    with h5py.File(source, "w") as handle:
        acquisition = handle.create_group("Acquisition")
        acquisition.attrs["SpatialSamplingInterval"] = spacing
        acquisition.attrs["SpatialSamplingIntervalUnit"] = "m"
        raw = acquisition.create_group("Raw[0]")
        raw.attrs["StartLocusIndex"] = 0
        raw.attrs["RawDataUnit"] = "strain"
        raw.create_dataset("RawData", shape=(8, channels), dtype="f4")
        raw.create_dataset("RawDataTime", shape=(8,), dtype="i8")
    waveform = tmp_path / f"{name}.npz"
    wave = np.arange(channels * 3000, dtype=np.float32).reshape(channels, 3000)
    distance = np.arange(channels, dtype="<f8") * nominal
    np.savez(waveform, waveform=wave, distance_m=distance, relative_seconds=np.arange(3000) / 100,
             valid_channel_mask=np.ones(channels, bool), recognition_channel_mask=np.ones(channels, bool),
             region_id=np.zeros(channels, dtype=np.int8))
    row = {"window_id": name, "waveform": str(waveform), "waveform_sha256": hashlib.sha256(waveform.read_bytes()).hexdigest(),
           "grid_sha256": hashlib.sha256(distance.tobytes()).hexdigest(), "unit": "strain", "window_seconds": 30,
           "input_sha256": "input-" + name, "labels": {"earthquake": "present"},
           "recognition_range_m": {"start": 4.05, "stop": None},
           "sources": [{"path": str(source), "size_bytes": source.stat().st_size,
                        "mtime_ns": source.stat().st_mtime_ns, "fingerprint": fingerprint(source),
                        "channel_start": 0, "channel_stop": channels}]}
    return row, wave


def test_aligns_actual_coordinates_without_interpolation_and_masks_missing_channels(tmp_path):
    from das_sentinel.window_grid import normalize_rows

    short, original = sample(tmp_path, "short", 2)
    long, _ = sample(tmp_path, "long", 3)
    before = json.dumps(short, sort_keys=True)
    normalized, grid = normalize_rows([short, long], tmp_path / "cache")
    assert json.dumps(short, sort_keys=True) == before
    assert grid["channels"] == 3 and grid["spacing_m"] == 4.1
    assert len({r["grid_sha256"] for r in normalized}) == 1
    with np.load(normalized[0]["waveform"]) as data:
        np.testing.assert_array_equal(data["waveform"][:2], original)
        assert not data["waveform"][2].any()
        np.testing.assert_allclose(data["distance_m"], [0, 4.1, 8.2])
        np.testing.assert_array_equal(data["valid_channel_mask"], [True, True, False])
        np.testing.assert_array_equal(data["recognition_channel_mask"], [False, True, False])
    assert normalized[0]["source_waveform"] == short["waveform"]
    assert normalized[0]["source_grid_sha256"] == short["grid_sha256"]
    assert normalized[0]["input_sha256"] == short["input_sha256"]
    assert normalized[0]["labels"] == short["labels"]
    assert hashlib.sha256(Path(short["waveform"]).read_bytes()).hexdigest() == short["waveform_sha256"]
    repeated, same_grid = normalize_rows([short, long], tmp_path / "cache")
    assert repeated == normalized and same_grid == grid
    subset, subset_grid = normalize_rows([short], tmp_path / "cache")
    assert subset_grid["grid_sha256"] == grid["grid_sha256"]
    assert subset_grid["channels"] == 3
    assert subset[0]["waveform"] == normalized[0]["waveform"]


def test_native_raw_geometry_has_priority_and_bad_metadata_is_rejected(tmp_path):
    from das_sentinel.window_grid import native_geometry

    row, _ = sample(tmp_path, "raw", 3)
    source = row["sources"][0]
    with h5py.File(source["path"], "r+") as handle:
        handle["Acquisition/Raw[0]"].attrs["SpatialSamplingInterval"] = 2.5
    path = Path(source["path"])
    source.update(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                  fingerprint=fingerprint(path))
    assert native_geometry(source, tmp_path / "cache")["spacing_m"] == 2.5
    with h5py.File(source["path"], "r+") as handle:
        handle["Acquisition/Raw[0]"].attrs["SpatialSamplingInterval"] = -1
    source.update(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                  fingerprint=fingerprint(path))
    with pytest.raises(ValueError, match="spacing"):
        native_geometry(source, tmp_path / "cache")


def test_rejects_changed_sources_different_spacing_and_corrupt_arrays(tmp_path):
    from das_sentinel.window_grid import normalize_rows

    first, _ = sample(tmp_path, "first", 2)
    other, _ = sample(tmp_path, "other", 3, spacing=5.0)
    with pytest.raises(ValueError, match="spacing"):
        normalize_rows([first, other], tmp_path / "cache")
    first["waveform_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        normalize_rows([first], tmp_path / "cache")
    assert not (tmp_path / "cache/grid.json").exists()
    first["sources"][0]["mtime_ns"] -= 1
    with pytest.raises(ValueError, match="source changed"):
        normalize_rows([first], tmp_path / "cache")


def test_immutable_cache_publish_is_idempotent_and_rejects_conflicts(tmp_path):
    from das_sentinel.window_grid import _save_json

    path = tmp_path / "entry.json"
    _save_json(path, {"value": 1})
    before = path.stat().st_ino
    with path.open("rb"):
        _save_json(path, {"value": 1})
    assert path.stat().st_ino == before
    with pytest.raises(ValueError, match="cache conflict"):
        _save_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_normalization_publishes_one_identical_cache(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from das_sentinel.window_grid import normalize_rows

    rows = [sample(tmp_path, "short", 2)[0], sample(tmp_path, "long", 3)[0]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: normalize_rows(rows, tmp_path / "cache"), range(4)))
    assert all(result == results[0] for result in results)
    assert len(list((tmp_path / "cache/arrays").glob("*.npz"))) == 2
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_normalization_preserves_excluded_waveforms_and_recomputes_region_boundaries(tmp_path):
    from das_sentinel.window_grid import normalize_rows

    row, wave = sample(tmp_path, "regions", 126, spacing=4.1)
    row["recognition_range_m"] = {"start": 300, "stop": 499}
    normalized, _ = normalize_rows([row], tmp_path / "cache")
    with np.load(normalized[0]["waveform"]) as data:
        np.testing.assert_array_equal(data["waveform"], wave)
        expected = (data["distance_m"] >= 300) & (data["distance_m"] <= 499)
        np.testing.assert_array_equal(data["recognition_channel_mask"], expected)
        assert data["region_id"][72] == 0
        assert data["region_id"][73] == 1
        assert data["region_id"][122] == 2


def test_normalized_grid_is_bound_to_trained_model_and_validated_at_training(tmp_path):
    pytest.importorskip("torch")
    from das_sentinel.classifiers.window_model import load_bundle
    from das_sentinel.window_grid import normalize_rows
    from das_sentinel.window_training import TASK, split_rows, train

    rows = []
    for day in range(1, 4):
        for positive in (False, True):
            row, _ = sample(tmp_path, f"{day}-{positive}", 2 + day % 2)
            row.update(date=f"2020-01-0{day}", catalog_ids=[],
                       read_start_utc=f"2020-01-0{day}T00:00:00+00:00",
                       read_end_utc=f"2020-01-0{day}T00:00:30+00:00",
                       labels={"earthquake": "present" if positive else "absent"})
            rows.append(row)
    rows, grid = normalize_rows(rows, tmp_path / "cache")
    split_rows(rows)
    manifest = {"schema_version": 2, "task": TASK, "site_id": "test", "rows": rows,
                "training_grid": grid}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    train(path, tmp_path / "model", epochs=1)
    metadata = json.loads((tmp_path / "model/model.json").read_text())
    assert metadata["training_grid"] == grid
    assert load_bundle(tmp_path / "model", grid["grid_sha256"], unit="strain")
    manifest["training_grid"]["channels"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="grid"):
        train(path, tmp_path / "invalid", epochs=1)


def test_normalization_cache_miss_reuses_validated_npz_arrays(tmp_path, monkeypatch):
    from collections import Counter

    from das_sentinel.window_grid import normalize_rows

    short, wave = sample(tmp_path, "short", 2)
    long, _ = sample(tmp_path, "long", 3)
    np.savez(short["waveform"], waveform=wave, distance_m=np.array([0, 4], dtype="<f8"),
        relative_seconds=np.arange(3000) / 100, valid_channel_mask=np.array([False, True]),
        recognition_channel_mask=np.array([True, True]), region_id=np.array([9, 9], dtype="i1"),
        annotation=np.array(["extra array"]))
    short["waveform_sha256"] = hashlib.sha256(Path(short["waveform"]).read_bytes()).hexdigest()
    reads = Counter()
    original = np.lib.npyio.NpzFile.__getitem__
    def counted(self, key):
        if self.fid.name == short["waveform"]:
            reads[key] += 1
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counted)
    normalized, grid = normalize_rows([short, long], tmp_path / "cache")
    with np.load(normalized[0]["waveform"], allow_pickle=False) as arrays:
        assert arrays.files == ["waveform", "distance_m", "relative_seconds", "valid_channel_mask",
                                "recognition_channel_mask", "region_id"]
        np.testing.assert_array_equal(arrays["waveform"], np.vstack([wave, np.zeros((1, 3000), dtype="f4")]))
        np.testing.assert_array_equal(arrays["distance_m"], np.array([0, 4.1, 8.2], dtype="<f8"))
        np.testing.assert_array_equal(arrays["relative_seconds"], np.arange(3000) / 100)
        np.testing.assert_array_equal(arrays["valid_channel_mask"], [False, True, False])
        np.testing.assert_array_equal(arrays["recognition_channel_mask"], [False, True, False])
        np.testing.assert_array_equal(arrays["region_id"], [0, 0, 0])
    assert normalized[0]["input_sha256"] == short["input_sha256"]
    assert normalized[0]["labels"] == short["labels"]
    assert grid["channels"] == 3 and grid["spacing_m"] == 4.1
    assert reads == Counter(waveform=1, distance_m=1, relative_seconds=1, valid_channel_mask=1)
