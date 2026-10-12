import hashlib
import importlib

import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.whole_window import analyze_window, inspect_sources, plan_windows
from das_sentinel.window_inference import normalize_prediction_row

from tests.test_whole_window import BASE, write

LEGACY_ARRAY_KEYS = ("waveform", "distance_m", "relative_seconds", "valid_channel_mask",
                     "recognition_channel_mask", "region_id")


def module():
    assert importlib.util.find_spec("das_sentinel.model_scan_input"), "model-only input missing"
    return importlib.import_module("das_sentinel.model_scan_input")


@pytest.mark.parametrize("dtype", ["<f4", ">f4", "<f8", ">f8"])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_array_hash_preserves_legacy_bytes_dtype_shape_and_key_order(dtype, noncontiguous):
    api = module()
    from das_sentinel.model_scan_contract import PROTOCOL, canonical, utc

    data = np.arange(24, dtype=dtype).reshape(4, 6)
    if noncontiguous:
        data = data[:, ::2]
    arrays = {key: data for key in reversed(LEGACY_ARRAY_KEYS)}
    expected = hashlib.sha256(canonical({"protocol": PROTOCOL, "start_utc": utc(BASE).isoformat(),
                                        "unit": "strain_rate"}).encode())
    for key in LEGACY_ARRAY_KEYS:
        array = np.ascontiguousarray(arrays[key])
        expected.update(canonical([key, array.dtype.str, array.shape]).encode())
        expected.update(array.tobytes())
    assert api.arrays_sha256(arrays, BASE, "strain_rate") == expected.hexdigest()


def test_contiguous_array_hash_does_not_copy_waveform_bytes():
    import tracemalloc

    api = module()
    waveform = np.ones((128, 3000), dtype="f8")
    arrays = {key: waveform for key in LEGACY_ARRAY_KEYS}
    tracemalloc.start()
    try:
        digest = api.arrays_sha256(arrays, BASE, "strain_rate")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(digest) == 64
    assert peak < waveform.nbytes // 2, "hashing copied a contiguous waveform"


def test_array_hash_preserves_empty_array_bytes():
    api = module()
    from das_sentinel.model_scan_contract import PROTOCOL, canonical, utc

    arrays = {key: np.empty((0, 3), dtype=">f8") for key in LEGACY_ARRAY_KEYS}
    expected = hashlib.sha256(canonical({"protocol": PROTOCOL, "start_utc": utc(BASE).isoformat(),
                                        "unit": "strain_rate"}).encode())
    for key in LEGACY_ARRAY_KEYS:
        expected.update(canonical([key, ">f8", (0, 3)]).encode())
    assert api.arrays_sha256(arrays, BASE, "strain_rate") == expected.hexdigest()


def test_model_only_input_exactly_matches_existing_review_array(tmp_path):
    api = module()
    path = write(tmp_path / "source.h5")
    profile = ProcessingProfile(device="cpu", scan_distance_start_m=300)
    result = analyze_window([path], BASE, profile, tmp_path / "reference")
    distance = np.arange(10, dtype="<f8") * 100
    grid = {"version": "native-channel-union-v1", "spacing_m": 100., "start_locus": 0,
            "channels": 10, "grid_sha256": hashlib.sha256(distance.tobytes()).hexdigest()}
    metadata = {"unit": result["unit"], "distance_start_m": 300, "training_grid": grid}
    waveform = tmp_path / "reference/waveform.npz"
    row = {"window_id": "test", "window_seconds": 30, "input_sha256": result["input_sha256"],
           "sources": result["read_sources"], "unit": result["unit"], "grid_sha256": result["grid_sha256"],
           "waveform": str(waveform), "waveform_sha256": hashlib.sha256(waveform.read_bytes()).hexdigest()}
    normalized = normalize_prediction_row(row, metadata, tmp_path / "old-grid")
    plan = next(p for p in plan_windows(inspect_sources([path])) if p["start_utc"] == BASE.isoformat())
    arrays, provenance = api.read_scan_input(plan, metadata, tmp_path / "scan-cache")
    with np.load(normalized["waveform"], allow_pickle=False) as expected:
        for key in expected.files:
            np.testing.assert_array_equal(arrays[key], expected[key])
    assert arrays["waveform"].shape == (10, 3000)
    assert not arrays["valid_channel_mask"][-2:].any()
    assert len(provenance["model_input_sha256"]) == 64
    assert not list((tmp_path / "scan-cache").rglob("*.npz"))


def test_mismatched_unit_and_gap_are_not_background(tmp_path):
    api = module()
    path = write(tmp_path / "source.h5")
    plan = next(p for p in plan_windows(inspect_sources([path])) if p["start_utc"] == BASE.isoformat())
    with pytest.raises(ValueError, match="unit"):
        api.read_scan_input(plan, {"unit": "wrong"}, tmp_path / "cache")
    with pytest.raises(ValueError, match="coverage"):
        api.read_scan_input(plan | {"coverage": "incomplete"}, {"unit": "strain_rate"}, tmp_path / "cache")


def test_incompatible_native_spacing_is_rejected_before_waveform_read(tmp_path, monkeypatch):
    api = module()
    path = write(tmp_path / "source.h5")
    plan = next(p for p in plan_windows(inspect_sources([path])) if p["start_utc"] == BASE.isoformat())
    metadata = {"unit": "strain_rate", "distance_start_m": 300, "training_grid": {
        "version": "native-channel-union-v1", "spacing_m": 200., "start_locus": 0, "channels": 10}}
    def forbidden(*args, **kwargs):
        pytest.fail("incompatible geometry must not read a waveform")
    monkeypatch.setattr(api, "read_low_frequency_window", forbidden)
    with pytest.raises(ValueError, match="native geometry"):
        api.read_scan_input(plan, metadata, tmp_path / "cache")
