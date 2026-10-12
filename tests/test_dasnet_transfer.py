"""Contracts for the optional frozen DASNet transfer adapter."""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest


def sample():
    time = np.arange(6000, dtype=np.float64) / 100
    trace = (np.sin(2 * np.pi * 0.5 * time)
             + np.sin(2 * np.pi * 5 * time)
             + np.sin(2 * np.pi * 20 * time)).astype(np.float32)
    waveform = np.tile(trace, (832, 1))
    recognition = np.arange(832) >= 75
    arrays = {
        "waveform": waveform,
        "distance_m": np.arange(832, dtype=np.float64) * 4,
        "valid_channel_mask": np.ones(832, dtype=bool),
        "recognition_channel_mask": recognition,
        "region_id": np.where(np.arange(832) < 75, 0,
                              np.where(np.arange(832) < 125, 1, 2)).astype(np.uint8),
    }
    metadata = {
        "view": "standard", "sample_rate_hz": 100,
        "shape": [832, 6000], "unit": "(nm/m)/s * Hz/m",
        "annotations": [{"event_id": "event-a", "time_start_index": 2000,
                         "time_stop_index": 3000, "sample_channel_start": 80,
                         "sample_channel_stop": 120}],
    }
    return arrays, metadata


def test_frequency_views_preserve_time_axis_and_ignore_excluded_channel_signal():
    from das_sentinel.classifiers.dasnet_transfer import prepare_views

    arrays, metadata = sample()
    arrays["waveform"][0] *= 1000
    views = prepare_views(arrays, metadata)
    assert views.shape == (3, 6000, 832)
    assert views.dtype == np.float32
    assert np.all(views[:, :, :75] == 0)
    time = np.arange(6000) / 100
    interior = slice(200, -200)
    for view, frequency in [(1, 5), (2, 20)]:
        correlation = np.corrcoef(views[view, interior, 80],
                                  np.sin(2 * np.pi * frequency * time[interior]))[0, 1]
        assert correlation > 0.999
    assert np.isfinite(views).all()
    assert np.max(np.abs(views)) <= 3


@pytest.mark.parametrize("change", ["rate", "unit", "shape", "nonfinite", "mask", "region"])
def test_incompatible_waveforms_fail_before_encoding(change):
    from das_sentinel.classifiers.dasnet_transfer import prepare_views

    arrays, metadata = sample()
    if change == "rate":
        metadata["sample_rate_hz"] = 5
    elif change == "unit":
        metadata["unit"] = "strain"
    elif change == "shape":
        arrays["waveform"] = arrays["waveform"][:, :-1]
    elif change == "nonfinite":
        arrays["waveform"][100, 100] = np.nan
    elif change == "mask":
        arrays["recognition_channel_mask"][0] = True
    else:
        arrays["recognition_channel_mask"][0] = True
        arrays["region_id"][0] = 2
    with pytest.raises(ValueError):
        prepare_views(arrays, metadata)


def test_roi_pool_uses_local_time_distance_and_usable_channels():
    from das_sentinel.classifiers.dasnet_transfer import pool_roi

    # Two feature planes, two time cells, four distance cells. The target
    # covers the first time cell and only the last two usable distance cells.
    features = np.array([[[100, 200, 3, 5], [300, 400, 7, 9]],
                         [[1000, 2000, 30, 50], [3000, 4000, 70, 90]]], dtype=np.float32)
    mask = np.array([False, False, True, True])
    result = pool_roi(features, (2, 4), (0, 1, 0, 4), mask)
    np.testing.assert_array_equal(result, [4, 40, 5, 50])
    later = pool_roi(features, (2, 4), (1, 2, 2, 4), mask)
    np.testing.assert_array_equal(later, [8, 80, 9, 90])


def test_tiny_roi_retains_positive_overlap_without_expanding_to_whole_window():
    from das_sentinel.classifiers.dasnet_transfer import pool_roi

    features = np.array([[[2, 10], [20, 100]]], dtype=np.float32)
    mask = np.ones(8, dtype=bool)
    result = pool_roi(features, (8, 8), (1, 2, 1, 2), mask)
    np.testing.assert_array_equal(result, [2, 2])


@pytest.mark.parametrize("change", ["missing", "duplicate", "empty", "masked", "float"])
def test_review_roi_must_identify_one_nonempty_usable_target(change):
    from das_sentinel.classifiers.dasnet_transfer import validate_roi

    arrays, metadata = sample()
    annotation = metadata["annotations"][0]
    if change == "missing":
        annotation["event_id"] = "other"
    elif change == "duplicate":
        metadata["annotations"].append(copy.deepcopy(annotation))
    elif change == "empty":
        annotation["time_stop_index"] = annotation["time_start_index"]
    elif change == "masked":
        annotation["sample_channel_start"], annotation["sample_channel_stop"] = 0, 20
    else:
        annotation["time_start_index"] = 1.5
    with pytest.raises(ValueError):
        validate_roi(arrays, metadata, "event-a")


def test_unverified_checkpoint_is_rejected_without_optional_torch(tmp_path):
    from das_sentinel.classifiers.dasnet_transfer import load_encoder

    checkpoint = tmp_path / "bad.pth"
    checkpoint.write_bytes(b"not the published weights")
    with pytest.raises(ValueError, match="SHA256"):
        load_encoder(checkpoint)


@pytest.mark.skipif(not os.environ.get("DASNET_TEST_CHECKPOINT"),
                    reason="requires separately verified optional DASNet runtime and checkpoint")
def test_real_checkpoint_encoder_is_frozen_repeatable_and_label_independent():
    from das_sentinel.classifiers.dasnet_transfer import load_encoder

    encoder = load_encoder(os.environ["DASNET_TEST_CHECKPOINT"], threads=2)
    arrays, metadata = sample()
    before = arrays["waveform"][100].copy()
    seen = {}

    def inspect_input(_module, args):
        seen["highpass_trace"] = args[0][0, 2, :, 128].numpy().copy()

    hook = encoder.backbone.conv1.register_forward_pre_hook(inspect_input)
    result = encoder.encode_sample(arrays, metadata, "event-a")
    hook.remove()
    # Temporal resizing to 1024 pixels would suppress this >10 Hz input view.
    time = np.arange(6000) / 100
    assert seen["highpass_trace"].shape == time.shape
    assert np.corrcoef(seen["highpass_trace"][200:-200],
                       np.sin(2 * np.pi * 20 * time[200:-200]))[0, 1] > 0.999
    metadata["annotations"][0]["label"] = "noise"
    repeated = encoder.encode_sample(arrays, metadata, "event-a")
    assert result.shape == (3072,)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) > 0
    np.testing.assert_array_equal(result, repeated)
    np.testing.assert_array_equal(arrays["waveform"][100], before)
    assert not encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())
    assert encoder.provenance["checkpoint_sha256"] == (
        "f1b0e1ee00a5a4a818b4bcfb70205bcbafd0db38acfa6d04537456576ae9c590"
    )
