from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from das_sentinel.domain import DasBlock, DasMetadata
from das_sentinel.readers.continuous_low_frequency import ContinuousLowFrequencyWindow
from das_sentinel.services import waveform_samples

START = datetime(2026, 6, 16, 1, 2, tzinfo=UTC)


def _window(start, rate, seconds, margin, *, flat=False):
    count = round((seconds + 2 * margin) * rate)
    data = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
    if flat:
        data[:, 0] = 7
    distances = np.array([0.0, 300.0, 500.0])
    block = DasBlock(
        data,
        DasMetadata("source.h5", start - timedelta(seconds=margin), rate,
                    distances, "strain_rate"),
        sample_offset=0,
    )
    return ContinuousLowFrequencyWindow(
        block, round(margin * rate), round((margin + seconds) * rate),
        ({"path": "source.h5", "fingerprint": "abc"},), (),
    )


def test_build_standard_sample_has_fixed_coordinates_masks_and_roi(monkeypatch):
    calls = []

    def fake_read(paths, start, end, *args, **kwargs):
        calls.append((paths, start, end, args, kwargs))
        return _window(start, 100, 60, 0, flat=True)

    monkeypatch.setattr(waveform_samples, "read_low_frequency_window", fake_read)
    annotations = ({
        "event_id": "event-1", "review_id": 42, "label": "blast",
        "start_utc": (START + timedelta(seconds=10)).isoformat(),
        "end_utc": (START + timedelta(seconds=20)).isoformat(),
        "channel_start": 1, "channel_stop": 9,
        "legacy_features": {"peak": 3.0},
    }, {
        "event_id": "outside", "start_utc": (START - timedelta(seconds=5)).isoformat(),
        "end_utc": START.isoformat(), "channel_start": 0, "channel_stop": 1,
    })
    arrays, metadata = waveform_samples.build_sample(
        ["source.h5"], START, "standard", np.array([0.0, 300.0, 500.0]),
        "strain_rate", suspended_end_m=299, borehole_start_m=499,
        annotations=annotations,
    )

    assert calls[0][-1] == {"margin_seconds": 0.0, "sample_rate_hz": 100}
    assert arrays["waveform"].shape == (3, 6000)
    assert arrays["waveform"].dtype == np.float32
    np.testing.assert_array_equal(arrays["distance_m"], [0, 300, 500])
    np.testing.assert_allclose(arrays["relative_seconds"][[0, -1]], [0, 59.99])
    np.testing.assert_array_equal(arrays["valid_channel_mask"], [False, True, True])
    np.testing.assert_array_equal(arrays["region_id"], [0, 1, 2])
    np.testing.assert_array_equal(arrays["recognition_channel_mask"], [False, True, True])
    assert metadata["schema_version"] == 1
    assert metadata["start_utc"] == START.isoformat()
    assert metadata["end_utc"] == (START + timedelta(seconds=60)).isoformat()
    assert metadata["shape"] == [3, 6000]
    assert metadata["suspended_end_m"] == 299
    assert metadata["borehole_start_m"] == 499
    assert [a["event_id"] for a in metadata["annotations"]] == ["event-1"]
    roi = metadata["annotations"][0]
    assert (roi["time_start_index"], roi["time_stop_index"]) == (1000, 2000)
    assert (roi["sample_channel_start"], roi["sample_channel_stop"]) == (1, 3)


def test_build_rejects_misaligned_start_and_reference_mismatch(monkeypatch):
    monkeypatch.setattr(
        waveform_samples, "read_low_frequency_window",
        lambda paths, start, end, *args, **kwargs: _window(start, 100, 60, 0),
    )
    with pytest.raises(ValueError, match="采样网格|aligned"):
        waveform_samples.build_sample(
            ["x"], START + timedelta(microseconds=1), "standard", [0, 300, 500],
            "strain_rate", suspended_end_m=299, borehole_start_m=499,
        )
    with pytest.raises(ValueError, match="参考距离"):
        waveform_samples.build_sample(
            ["x"], START, "standard", [0, 301, 500], "strain_rate",
            suspended_end_m=299, borehole_start_m=499,
        )


def test_low_frequency_keeps_raw_crop_and_reuses_analysis_filter(monkeypatch):
    window = _window(START, 5, 600, 300)
    monkeypatch.setattr(
        waveform_samples, "read_low_frequency_window",
        lambda *args, **kwargs: window,
    )

    class Analysis:
        filtered_data = np.full((3000, 3), 9, dtype=np.float32)
        valid_channel_mask = np.array([True, False, True])

    calls = []
    monkeypatch.setattr(
        waveform_samples, "analyze_low_frequency",
        lambda block, first, stop: calls.append((block, first, stop)) or Analysis(),
    )
    arrays, metadata = waveform_samples.build_sample(
        ["x"], START, "low_frequency", [0, 300, 500], "strain_rate",
        suspended_end_m=299, borehole_start_m=499,
    )
    assert calls[0][1:] == (1500, 4500)
    assert arrays["waveform"].shape == (3, 3000)
    assert arrays["filtered_waveform"].shape == (3, 3000)
    assert np.all(arrays["filtered_waveform"][[0, 2]] == 9)
    assert np.all(arrays["filtered_waveform"][1] == 0)
    np.testing.assert_array_equal(arrays["valid_channel_mask"], [True, False, True])
    assert metadata["sample_rate_hz"] == 5


def test_low_frequency_masks_channel_flat_in_target_despite_varying_context(monkeypatch):
    rate = 5
    count = 1200 * rate
    time = np.arange(count) / rate
    data = np.column_stack([
        np.sin(2 * np.pi * 0.03 * time),
        np.sin(2 * np.pi * 0.05 * time),
    ]).astype(np.float32)
    data[1500:4500, 0] = 7.0
    block = DasBlock(
        data,
        DasMetadata(
            "source.h5", START - timedelta(seconds=300), 5,
            np.array([300.0, 500.0]), "strain_rate",
        ),
        sample_offset=0,
    )
    window = ContinuousLowFrequencyWindow(block, 1500, 4500, (), ())
    monkeypatch.setattr(
        waveform_samples, "read_low_frequency_window", lambda *args, **kwargs: window
    )

    arrays, _ = waveform_samples.build_sample(
        ["x"], START, "low_frequency", [300, 500], "strain_rate",
        suspended_end_m=299, borehole_start_m=499,
    )

    assert np.all(arrays["waveform"][0] == 7)
    assert arrays["valid_channel_mask"].tolist() == [False, True]
    assert arrays["recognition_channel_mask"].tolist() == [False, True]
    assert np.all(arrays["filtered_waveform"][0] == 0)


def test_save_load_round_trip_refuses_overwrite_and_detects_bad_contract(tmp_path):
    arrays = {
        "waveform": np.ones((2, 6000), np.float32),
        "distance_m": np.array([1, 2.0]),
        "relative_seconds": np.arange(6000) / 100,
        "valid_channel_mask": np.array([True, False]),
        "region_id": np.array([0, 2], np.uint8),
        "recognition_channel_mask": np.array([False, False]),
    }
    metadata = {
        "schema_version": 1, "view": "standard", "sample_rate_hz": 100,
        "start_utc": START.isoformat(),
        "end_utc": (START + timedelta(seconds=60)).isoformat(),
        "unit": "strain_rate", "shape": [2, 6000], "sources": [],
        "warnings": [], "annotations": [], "target_event_id": "e1",
        "suspended_end_m": 1.5, "borehole_start_m": 2.0,
    }
    directory = tmp_path / "sample"
    waveform_samples.save_sample(directory, arrays, metadata)
    loaded_arrays, loaded_metadata = waveform_samples.load_sample(directory)
    for key, expected in arrays.items():
        np.testing.assert_array_equal(loaded_arrays[key], expected)
    assert loaded_metadata == metadata
    with pytest.raises(FileExistsError):
        waveform_samples.save_sample(directory, arrays, metadata)

    bad = tmp_path / "bad"
    bad.mkdir()
    np.savez(bad / "arrays.npz", waveform=np.ones((2, 4)))
    (bad / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="数组|contract"):
        waveform_samples.load_sample(tmp_path / "bad")


@pytest.mark.parametrize("corruption", [
    "wrong_rate", "reverse_time", "relative_grid", "integer_mask",
    "suspended_recognition", "extra_object", "extra_array",
])
def test_save_rejects_corrupt_standard_contract(corruption, tmp_path):
    arrays = {
        "waveform": np.ones((2, 6000), np.float32),
        "distance_m": np.array([1.0, 2.0]),
        "relative_seconds": np.arange(6000) / 100,
        "valid_channel_mask": np.array([True, False]),
        "region_id": np.array([0, 2], np.uint8),
        "recognition_channel_mask": np.array([False, False]),
    }
    metadata = {
        "schema_version": 1, "view": "standard", "sample_rate_hz": 100,
        "start_utc": START.isoformat(),
        "end_utc": (START + timedelta(seconds=60)).isoformat(),
        "unit": "strain_rate", "shape": [2, 6000], "sources": [],
        "warnings": [], "annotations": [], "suspended_end_m": 1.5,
        "borehole_start_m": 2.0,
    }
    if corruption == "wrong_rate": metadata["sample_rate_hz"] = 123
    elif corruption == "reverse_time": metadata["end_utc"] = START.isoformat()
    elif corruption == "relative_grid": arrays["relative_seconds"] = np.arange(6000)[::-1]
    elif corruption == "integer_mask": arrays["valid_channel_mask"] = np.array([2, 3])
    elif corruption == "suspended_recognition": arrays["recognition_channel_mask"][0] = True
    elif corruption == "extra_object": arrays["payload"] = np.array([object()], dtype=object)
    elif corruption == "extra_array": arrays["unexpected"] = np.array([1])
    with pytest.raises(ValueError):
        waveform_samples.save_sample(tmp_path / corruption, arrays, metadata)


def test_save_requires_low_frequency_filter_and_valid_annotation_roi(tmp_path):
    arrays = {
        "waveform": np.ones((1, 3000), np.float32),
        "distance_m": np.array([500.0]),
        "relative_seconds": np.arange(3000) / 5,
        "valid_channel_mask": np.array([True]), "region_id": np.array([2], np.uint8),
        "recognition_channel_mask": np.array([True]),
    }
    metadata = {
        "schema_version": 1, "view": "low_frequency", "sample_rate_hz": 5,
        "start_utc": START.isoformat(),
        "end_utc": (START + timedelta(seconds=600)).isoformat(), "unit": "x",
        "shape": [1, 3000], "sources": [], "warnings": [],
        "suspended_end_m": 299.0, "borehole_start_m": 499.0,
        "annotations": [{"time_start_index": 4, "time_stop_index": 3,
                         "sample_channel_start": 0, "sample_channel_stop": 1}],
    }
    with pytest.raises(ValueError):
        waveform_samples.save_sample(tmp_path / "missing-filter", arrays, metadata)
    arrays["filtered_waveform"] = arrays["waveform"].copy()
    with pytest.raises(ValueError):
        waveform_samples.save_sample(tmp_path / "bad-roi", arrays, metadata)
