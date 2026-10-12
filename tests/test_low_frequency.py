from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from das_sentinel.domain import DasBlock, DasMetadata
from das_sentinel.low_frequency import analyze_low_frequency


def _analyze(block, start=1500, stop=4500):
    return analyze_low_frequency(block, start, stop)


def _block(seconds=1200, channels=12, noise=0.02):
    data = np.random.default_rng(20260914).normal(
        0, noise, (round(seconds * 5), channels)
    ).astype(np.float32)
    return DasBlock(data, DasMetadata(
        source_path="synthetic-continuous",
        start_time_utc=datetime(2026, 9, 14, tzinfo=UTC),
        sample_rate_hz=5.0,
        channel_distance_m=500.0 + np.arange(channels) * 4,
        unit="strain_rate",
    ), sample_offset=12345)


def _pulse(block, start, stop, frequency=0.03, amplitude=1.0, channels=slice(None)):
    first, last = round(start * 5), round(stop * 5)
    time = np.arange(last - first) / 5
    wave = amplitude * np.hanning(last - first) * np.sin(2 * np.pi * frequency * time)
    block.data[first:last, channels] += wave[:, None].astype(np.float32)
    return block


@pytest.mark.parametrize("phase", [0.0, np.pi / 2])
def test_long_context_preserves_100_second_signal_without_twenty_second_detrending(phase):
    block = _block(noise=0)
    time = np.arange(block.data.shape[0]) / 5
    wave = np.sin(2 * np.pi * 0.01 * time + phase)
    block.data[:] = wave[:, None]
    result = _analyze(block)
    # The declared Butterworth cutoff has half-amplitude after forward/backward filtering.
    expected = 0.5 * wave[1500:4500]
    assert np.corrcoef(result.filtered_data[:, 0], expected)[0, 1] > 0.995
    assert np.sqrt(np.mean(result.filtered_data[:, 0] ** 2)) == pytest.approx(
        np.sqrt(np.mean(expected ** 2)), rel=0.08
    )
    assert result.measurements["dominant_frequency_hz"] == pytest.approx(0.01, abs=1 / 600)


def test_synchronous_slow_wave_packet_is_kept_with_original_amplitude_and_seconds():
    block = _pulse(_block(), 450, 750, frequency=0.01)
    original = block.data.copy()
    result = _analyze(block)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert 100 < candidate["peak_seconds"] < 450
    assert candidate["start_seconds"] < 225 < candidate["end_seconds"]
    assert candidate["end_seconds"] - candidate["start_seconds"] >= 20
    assert candidate["coherence"] > 0.95
    assert candidate["active_channel_fraction"] > 0.9
    assert candidate["snr_db"] > 6
    assert 0 <= candidate["score"] <= 1
    assert "label" not in candidate
    assert candidate["energy_duration_s"] > 50
    assert candidate["frequency_resolution_hz"] > 0
    assert 0.005 <= candidate["dominant_frequency_hz"] <= 0.02
    assert result.filtered_data.shape == (3000, 12)
    assert np.isfinite(result.filtered_data).all()
    np.testing.assert_array_equal(block.data, original)
    scaled = _analyze(replace(block, data=(block.data * 1e-9).astype(np.float32)))
    np.testing.assert_allclose(scaled.filtered_data, result.filtered_data * 1e-9,
                               rtol=2e-5, atol=1e-14)
    assert len(scaled.candidates) == 1
    assert scaled.candidates[0]["snr_db"] == pytest.approx(candidate["snr_db"], abs=1e-3)
    assert scaled.measurements["band_rms"] == pytest.approx(
        result.measurements["band_rms"] * 1e-9, rel=1e-5
    )


def test_stationary_low_frequency_background_does_not_generate_candidates():
    block = _block(noise=0)
    time = np.arange(block.data.shape[0]) / 5
    wave = np.sin(2 * np.pi * 0.03 * time) + 0.2 * np.sin(2 * np.pi * 0.15 * time)
    block.data[:] = wave[:, None]
    result = _analyze(block)
    assert result.candidates == ()
    assert result.measurements["snr_db"] == pytest.approx(0, abs=0.4)
    assert result.measurements["coherence"] > 0.99


def test_single_energetic_channel_is_not_spatial_support_or_a_bad_channel():
    block = _pulse(_block(), 450, 750, amplitude=100, channels=slice(0, 1))
    result = _analyze(block)
    assert result.valid_channel_mask.all()
    assert result.candidates == ()
    assert np.sqrt(np.mean(result.filtered_data[:, 0] ** 2)) > 10


def test_nonfinite_and_flat_channels_are_masked_without_removing_strong_valid_channels():
    block = _pulse(_block(), 450, 750)
    block.data[:, 0] = np.nan
    block.data[10, 1] = np.inf
    block.data[:, 2] = 3
    block.data[:, 3] *= 100
    result = _analyze(block)
    assert result.valid_channel_mask.tolist() == [False] * 3 + [True] * 9
    assert len(result.candidates) == 1
    assert np.all(result.filtered_data[:, :3] == 0)
    assert result.measurements["valid_channel_count"] == 9
    assert np.isfinite(result.filtered_data).all()
    json.dumps(result.measurements, allow_nan=False)
    json.dumps(result.candidates, allow_nan=False)


@pytest.mark.parametrize("start,stop", [(500, 4500), (1500, 5500)])
def test_incomplete_filter_context_is_explicit_and_cannot_auto_detect(start, stop):
    result = _analyze(_pulse(_block(), 450, 750), start, stop)
    assert "insufficient_filter_context" in result.warnings
    assert result.candidates == ()


def test_event_already_active_at_target_start_is_clipped_and_warned():
    block = _pulse(_block(), 270, 600, frequency=0.05)
    result = _analyze(block)
    assert result.candidates
    assert result.candidates[0]["start_seconds"] == 0
    assert "candidate_clipped_start" in result.warnings


def test_wave_packet_continuing_past_target_end_is_clipped_and_warned():
    result = _analyze(_pulse(_block(), 800, 990, frequency=0.05))
    assert len(result.candidates) == 1
    assert result.candidates[0]["end_seconds"] == 600
    assert "candidate_clipped_end" in result.warnings


@pytest.mark.parametrize("gap_seconds,expected_count", [(10, 1), (120, 2)])
def test_long_wave_packets_merge_only_across_short_gaps(gap_seconds, expected_count):
    block = _pulse(_block(), 430, 510, frequency=0.1)
    _pulse(block, 510 + gap_seconds, 590 + gap_seconds, frequency=0.1)
    result = _analyze(block)
    assert len(result.candidates) == expected_count
    if expected_count == 2:
        assert result.candidates[0]["end_seconds"] < result.candidates[1]["start_seconds"]


def test_missing_real_baseline_does_not_invent_snr():
    result = _analyze(_pulse(_block(), 450, 750), 400, 4500)
    assert result.measurements["snr_db"] is None
    assert result.measurements["background_start_seconds"] == -80
    assert result.measurements["background_end_seconds"] == -30
    assert result.candidates == ()
    assert "insufficient_background" in result.warnings


def test_too_few_valid_channels_explicitly_warns_that_detection_cannot_run():
    result = _analyze(_pulse(_block(channels=3), 450, 750))
    assert result.valid_channel_mask.all()
    assert result.candidates == ()
    assert result.measurements["band_rms"] > 0
    assert "insufficient_spatial_support_channels" in result.warnings


def test_low_frequency_subband_fractions_and_resolution_are_measured():
    block = _block(noise=0)
    time = np.arange(block.data.shape[0]) / 5
    block.data[:] = (np.sin(2 * np.pi * 0.03 * time)
                     + 2 * np.sin(2 * np.pi * 0.075 * time)
                     + 3 * np.sin(2 * np.pi * 0.2 * time))[:, None]
    measurements = _analyze(block).measurements
    assert measurements["background_start_seconds"] == -200
    assert measurements["background_end_seconds"] == -30
    assert measurements["frequency_resolution_hz"] == pytest.approx(1 / 600)
    assert measurements["dominant_frequency_hz"] == pytest.approx(0.2, abs=1 / 600)
    fractions = [measurements[key] for key in (
        "band_fraction_0_01_0_05", "band_fraction_0_05_0_1", "band_fraction_0_1_0_4"
    )]
    assert sum(fractions) == pytest.approx(1)
    np.testing.assert_allclose(fractions, [1 / 14, 4 / 14, 9 / 14], atol=0.015)


def test_twenty_second_target_does_not_claim_resolved_lowest_subband():
    result = _analyze(_block(), 1500, 1600)
    assert result.measurements["frequency_resolution_hz"] == pytest.approx(0.05)
    assert result.measurements["dominant_frequency_hz"] is None
    assert result.measurements["band_fraction_0_01_0_05"] is None
    assert "insufficient_low_frequency_resolution" in result.warnings
    assert result.candidates == ()


def test_short_candidate_keeps_fft_spacing_but_does_not_claim_low_frequency_dominant():
    result = _analyze(_pulse(_block(), 430, 510, frequency=0.1))
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    duration = candidate["end_seconds"] - candidate["start_seconds"]
    assert 20 <= duration < 100
    assert candidate["dominant_frequency_hz"] is None
    assert candidate["frequency_resolution_hz"] == pytest.approx(1 / duration)
    assert "candidate_insufficient_low_frequency_resolution" in result.warnings
    assert candidate["score"] > 0
    assert candidate["snr_db"] > 6


def test_all_invalid_channels_return_unavailable_measurements_and_finite_array():
    block = _block(noise=0)
    result = _analyze(block)
    assert not result.valid_channel_mask.any()
    assert result.candidates == ()
    assert result.measurements["band_rms"] is None
    assert result.measurements["snr_db"] is None
    assert result.measurements["dominant_frequency_hz"] is None
    assert "no_valid_channels" in result.warnings
    assert np.isfinite(result.filtered_data).all()
    json.dumps(result.measurements, allow_nan=False)


@pytest.mark.parametrize("start,stop", [(-1, 4500), (1500, 6001), (1500, 1500), (1.5, 20)])
def test_invalid_target_ranges_are_rejected(start, stop):
    with pytest.raises(ValueError, match="analysis"):
        _analyze(_block(), start, stop)


def test_wrong_sampling_rate_is_rejected_instead_of_mislabelling_filter():
    block = _block()
    with pytest.raises(ValueError, match="5 Hz"):
        _analyze(replace(block, metadata=replace(block.metadata, sample_rate_hz=10)))
