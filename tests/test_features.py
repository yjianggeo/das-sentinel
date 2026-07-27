import numpy as np
import pytest
from das_sentinel.classifiers.rules import score_rules
from das_sentinel.domain import Candidate, EventRecord, Label
from das_sentinel.features import FEATURE_NAMES, extract_features

from tests.synthetic import preprocessed_block


def test_extracts_stable_features_from_event_global_ranges_with_block_offset():
    fs = 100.0
    time = np.arange(500) / fs
    data = np.zeros((500, 6), dtype=np.float32)
    data[100:300, 1:5] = np.sin(2 * np.pi * 12.0 * time[:200])[:, None]
    block = preprocessed_block(data, sample_offset=1_000, sample_rate_hz=fs)
    event = EventRecord(
        "event-1", 1_100, 1_300, 1_200, 1, 5,
        (Candidate("coherence", 1_100, 1_300, 1_200, 1, 5, 0.9,
                   {"slowness_s_per_m": 0.001, "apparent_velocity_mps": 1_000.0,
                    "direction": 1.0, "semblance": 0.8}),),
    )

    features = extract_features(block, event, fs=fs)

    assert tuple(features) == FEATURE_NAMES
    assert features["duration_s"] == pytest.approx(2.0)
    assert features["dominant_frequency_hz"] == pytest.approx(12.0, rel=0.1)
    assert features["distance_extent_m"] == pytest.approx(3.0)
    assert features["best_slowness_s_per_m"] == pytest.approx(0.001)


def test_feature_extraction_clips_event_window_and_excludes_bad_channels():
    data = np.zeros((100, 4), dtype=np.float32)
    data[:, 0] = 100.0
    data[10:30, 1:3] = 2.0
    block = preprocessed_block(
        data, sample_offset=100, bad_channel_mask=np.array([True, False, False, False])
    )
    event = EventRecord("edge", 50, 130, 90, -2, 8, ())

    features = extract_features(block, event, fs=100.0)

    assert features["duration_s"] == pytest.approx(0.8)
    assert features["rms"] == pytest.approx(np.sqrt(160.0 / 90.0), rel=0.05)
    assert features["distance_extent_m"] == pytest.approx(2.0)


def test_short_or_zero_energy_events_return_finite_stable_features():
    block = preprocessed_block(np.zeros((2, 2), dtype=np.float32))
    event = EventRecord("quiet", 1_000, 1_002, 1_001, 0, 2, ())

    features = extract_features(block, event, fs=100.0)

    assert tuple(features) == FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values() if isinstance(value, float))


def test_opposite_phase_channels_preserve_welch_dominant_frequency():
    fs = 100.0
    time = np.arange(400) / fs
    signal = np.sin(2 * np.pi * 12.0 * time).astype(np.float32)
    block = preprocessed_block(np.column_stack((signal, -signal)), sample_rate_hz=fs)
    event = EventRecord("opposite", 1_000, 1_400, 1_200, 0, 2, ())

    features = extract_features(block, event)

    assert features["dominant_frequency_hz"] == pytest.approx(12.0, rel=0.1)


def test_impulse_peak_mean_ratio_supports_blast_rule_evidence():
    data = np.zeros((100, 2), dtype=np.float32)
    data[50] = 10.0
    block = preprocessed_block(data)
    event = EventRecord("pulse", 1_000, 1_100, 1_050, 0, 2, ())

    features = extract_features(block, event)

    assert features["peak_median_ratio"] > 5.0
    assert score_rules(features, {}).scores[Label.BLAST] == 1.0
