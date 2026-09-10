from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate, DasBlock, DasMetadata, EventRecord
from das_sentinel.measurements import measure_event


def _event_block(
    amplitude: float = 1.0,
    *,
    event_start: int = 500,
    event_stop: int = 700,
    frequency_hz: float = 10.0,
) -> tuple[DasBlock, EventRecord]:
    fs = 100.0
    rng = np.random.default_rng(20260910)
    data = rng.normal(0.0, 0.02, size=(900, 8)).astype(np.float32)
    time = np.arange(event_stop - event_start) / fs
    taper = np.hanning(event_stop - event_start)
    burst = amplitude * taper * np.sin(2.0 * np.pi * frequency_hz * time)
    data[event_start:event_stop, 1:7] += burst[:, None].astype(np.float32)
    block = DasBlock(
        data=data,
        metadata=DasMetadata(
            source_path="ignored/source-name.h5",
            start_time_utc=datetime(2026, 9, 10, tzinfo=UTC),
            sample_rate_hz=fs,
            channel_distance_m=np.arange(8, dtype=np.float64) * 10.0,
            unit="strain_rate",
        ),
        sample_offset=1_000,
    )
    event = EventRecord(
        event_id="event-1",
        start_sample=1_000 + event_start,
        end_sample=1_000 + event_stop,
        peak_sample=1_000 + (event_start + event_stop) // 2,
        channel_start=0,
        channel_stop=8,
        evidence=(),
    )
    return block, event


def test_measurement_contract_reports_one_windowed_sinusoidal_pulse():
    block, event = _event_block()

    result = measure_event(block, event, ProcessingProfile())

    assert result["schema_version"] == "2"
    assert result["values"]["trigger_duration_s"] == pytest.approx(2.0)
    assert result["values"]["pulse_count"] == 1.0
    assert result["values"]["dominant_frequency_hz"] == pytest.approx(10.0, abs=0.6)
    assert result["values"]["frequency_resolution_hz"] == pytest.approx(0.5)
    assert result["values"]["input_rms"] > 0.1
    assert result["values"]["clipping_fraction"] is None
    assert result["status"]["clipping_fraction"] == "adc_limits_unavailable"
    assert result["context"]["effective_frequency_band_hz"] == [
        0.5,
        pytest.approx(49.5),
    ]
    assert result["context"]["sampling_rate_hz"] == 100.0
    assert result["context"]["input_unit"] == "strain_rate"
    assert result["context"]["common_mode_treatment"] == "none"
    assert result["context"]["background_seconds_available"] == pytest.approx(4.5)
    json.dumps(result, allow_nan=False)


def test_short_event_keeps_known_resolution_and_supported_spectral_measurements():
    block, event = _event_block(event_start=500, event_stop=550)

    result = measure_event(block, event, ProcessingProfile())

    assert result["values"]["frequency_resolution_hz"] == pytest.approx(2.0)
    assert result["values"]["dominant_frequency_hz"] == pytest.approx(10.0, abs=2.0)
    assert result["status"]["spectral_centroid_hz"] == "measured"
    assert result["status"]["spectral_entropy"] == "measured"
    assert (
        result["status"]["band_fraction_low"] == "insufficient_low_frequency_resolution"
    )
    assert result["status"]["band_fraction_mid"] == "measured"
    assert result["status"]["band_fraction_high"] == "measured"
    assert "coarse_frequency_resolution" in result["context"]["quality_flags"]


def test_upper_cutoff_bin_is_included_in_high_band_energy_partition():
    fs = 200.0
    rng = np.random.default_rng(19)
    data = rng.normal(0.0, 1e-4, size=(1_800, 4)).astype(np.float32)
    time = np.arange(400) / fs
    data[1_000:1_400] += np.sin(2.0 * np.pi * 80.0 * time)[:, None].astype(
        np.float32
    )
    block = DasBlock(
        data,
        DasMetadata(
            "ignored.h5",
            datetime(2026, 9, 10, tzinfo=UTC),
            fs,
            np.arange(4, dtype=np.float64) * 10.0,
            "strain_rate",
        ),
        0,
    )
    event = EventRecord("cutoff", 1_000, 1_400, 1_200, 0, 4, ())

    result = measure_event(block, event, ProcessingProfile())
    fractions = [
        result["values"][name]
        for name in ("band_fraction_low", "band_fraction_mid", "band_fraction_high")
    ]

    assert result["values"]["dominant_frequency_hz"] == pytest.approx(80.0)
    assert result["values"]["band_fraction_high"] > 0.99
    assert sum(fractions) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("fs", "cutoff_hz", "terminal_band"),
    ((20.0, 9.9, "band_fraction_mid"), (8.0, 3.96, "band_fraction_low")),
)
def test_effective_cutoff_bin_belongs_to_terminal_available_band(
    fs, cutoff_hz, terminal_band
):
    event_samples = 400
    event_start = round(5.0 * fs)
    event_stop = event_start + event_samples
    total_samples = event_stop + round(2.0 * fs)
    rng = np.random.default_rng(21)
    data = rng.normal(0.0, 1e-6, size=(total_samples, 4)).astype(np.float32)
    time = np.arange(event_samples) / fs
    data[event_start:event_stop] += np.sin(
        2.0 * np.pi * cutoff_hz * time
    )[:, None].astype(np.float32)
    block = DasBlock(
        data,
        DasMetadata(
            "ignored.h5",
            datetime(2026, 9, 10, tzinfo=UTC),
            fs,
            np.arange(4, dtype=np.float64) * 10.0,
            "strain_rate",
        ),
        0,
    )
    event = EventRecord(
        "terminal-cutoff", event_start, event_stop, event_start + 200, 0, 4, ()
    )

    result = measure_event(block, event, ProcessingProfile())

    assert result["values"]["dominant_frequency_hz"] == pytest.approx(cutoff_hz)
    assert result["values"][terminal_band] > 0.99


def test_bands_above_nyquist_are_unavailable_and_effective_bounds_are_reported():
    fs = 8.0
    rng = np.random.default_rng(20)
    data = rng.normal(0.0, 0.01, size=(72, 4)).astype(np.float32)
    time = np.arange(16) / fs
    data[40:56] += np.sin(2.0 * np.pi * 2.0 * time)[:, None].astype(np.float32)
    block = DasBlock(
        data,
        DasMetadata(
            "ignored.h5",
            datetime(2026, 9, 10, tzinfo=UTC),
            fs,
            np.arange(4, dtype=np.float64) * 10.0,
            "strain_rate",
        ),
        0,
    )
    event = EventRecord("low-fs", 40, 56, 48, 0, 4, ())

    result = measure_event(block, event, ProcessingProfile())

    assert result["status"]["band_fraction_low"] == "measured"
    assert result["values"]["band_fraction_mid"] is None
    assert result["status"]["band_fraction_mid"] == "band_outside_nyquist"
    assert result["values"]["band_fraction_high"] is None
    assert result["status"]["band_fraction_high"] == "band_outside_nyquist"
    assert result["context"]["effective_spectral_bands_hz"] == {
        "low": [0.5, pytest.approx(3.96)],
        "mid": None,
        "high": None,
    }


def test_band_with_no_fourier_bin_is_unavailable_instead_of_zero():
    block, event = _event_block(amplitude=0.0, event_start=500, event_stop=504)
    data = block.data.copy()
    data[500:504] += np.asarray((0.0, 1.0, -1.0, 0.0), dtype=np.float32)[
        :, None
    ]

    result = measure_event(replace(block, data=data), event, ProcessingProfile())

    assert result["values"]["band_fraction_mid"] is None
    assert (
        result["status"]["band_fraction_mid"]
        == "band_unresolved_at_frequency_resolution"
    )
    assert result["status"]["band_fraction_high"] == "measured"


def test_extremely_short_event_does_not_invent_pulse_or_lag_support():
    block, event = _event_block(amplitude=0.0, event_start=500, event_stop=502)
    data = block.data.copy()
    data[500:502] += np.asarray((1.0, -1.0), dtype=np.float32)[:, None]

    result = measure_event(replace(block, data=data), event, ProcessingProfile())

    assert result["values"]["pulse_count"] == 0.0
    assert result["values"]["lagged_correlation"] is None
    assert result["status"]["lagged_correlation"] == "insufficient_lag_overlap"


def test_filter_context_reports_available_post_event_guard():
    block, event = _event_block()
    shortened = replace(block, data=block.data[:750].copy())

    complete = measure_event(block, event, ProcessingProfile())
    incomplete = measure_event(shortened, event, ProcessingProfile())

    assert complete["context"]["post_event_filter_guard_seconds_available"] == 2.0
    assert (
        "incomplete_post_event_filter_guard" not in complete["context"]["quality_flags"]
    )
    assert incomplete["context"]["post_event_filter_guard_seconds_available"] == 0.5
    assert (
        "incomplete_post_event_filter_guard" in incomplete["context"]["quality_flags"]
    )


def test_stronger_physical_amplitude_increases_input_rms_and_window_snr():
    weak_block, event = _event_block(0.2)
    strong_block, _ = _event_block(2.0)

    weak = measure_event(weak_block, event, ProcessingProfile())["values"]
    strong = measure_event(strong_block, event, ProcessingProfile())["values"]

    assert strong["input_rms"] > weak["input_rms"] * 5.0
    assert strong["snr_db"] > weak["snr_db"] + 10.0
    assert strong["channel_snr_median_db"] > weak["channel_snr_median_db"]


def test_input_rms_uses_fixed_band_without_amplitude_normalization():
    block, event = _event_block(1.0)
    offset_block = replace(block, data=(block.data + np.float32(100.0)))

    baseline_rms = measure_event(block, event, ProcessingProfile())["values"][
        "input_rms"
    ]
    offset_rms = measure_event(offset_block, event, ProcessingProfile())["values"][
        "input_rms"
    ]

    assert offset_rms == pytest.approx(baseline_rms, rel=0.03)


def test_measurement_availability_is_invariant_to_input_unit_scale():
    block, event = _event_block(1.0)
    reference = measure_event(block, event, ProcessingProfile())
    scaled = measure_event(
        replace(block, data=(block.data * np.float32(1e-8)).astype(np.float32)),
        event,
        ProcessingProfile(),
    )

    checked = (
        "snr_db",
        "channel_snr_median_db",
        "channel_snr_p90_db",
        "pulse_count",
        "dominant_frequency_hz",
        "spectral_centroid_hz",
        "active_channel_fraction",
        "distance_extent_m",
    )
    assert scaled["values"]["valid_channel_count"] == reference["values"][
        "valid_channel_count"
    ]
    assert scaled["values"]["input_rms"] == pytest.approx(
        reference["values"]["input_rms"] * 1e-8, rel=1e-5
    )
    for name in checked:
        assert scaled["status"][name] == reference["status"][name]
        assert scaled["values"][name] == pytest.approx(
            reference["values"][name], rel=1e-5, abs=1e-10
        )


def test_insufficient_pre_event_baseline_is_explicitly_unavailable():
    block, event = _event_block(event_start=50, event_stop=250)

    result = measure_event(block, event, ProcessingProfile())

    for name in ("snr_db", "channel_snr_median_db", "channel_snr_p90_db"):
        assert result["values"][name] is None
        assert result["status"][name] == "insufficient_background"
    assert result["context"]["background_seconds_available"] == 0.0
    assert result["values"]["dominant_frequency_hz"] == pytest.approx(10.0, abs=0.6)


def test_nonfinite_and_flat_channels_are_excluded_without_rejecting_event_energy():
    block, event = _event_block(amplitude=20.0)
    data = block.data.copy()
    data[:, 0] = np.nan
    data[:, 1] = 0.0
    block = replace(block, data=data)

    result = measure_event(block, event, ProcessingProfile())

    assert result["values"]["valid_channel_count"] == 6.0
    assert result["values"]["valid_channel_fraction"] == pytest.approx(0.75)
    assert result["values"]["bad_channel_fraction"] == pytest.approx(0.25)
    assert result["status"]["input_rms"] == "measured"
    assert result["values"]["input_rms"] > 1.0


@pytest.mark.parametrize("invalid_distance", (float("nan"), float("inf")))
def test_invalid_geometry_preserves_nonspatial_measurements(invalid_distance):
    block, event = _event_block()
    distances = block.metadata.channel_distance_m.copy()
    distances[3] = invalid_distance
    block = replace(block, metadata=replace(block.metadata, channel_distance_m=distances))

    result = measure_event(block, event, ProcessingProfile())

    assert result["values"]["valid_channel_count"] == 8.0
    for name in (
        "input_rms",
        "snr_db",
        "pulse_count",
        "dominant_frequency_hz",
        "active_channel_fraction",
    ):
        assert result["status"][name] == "measured"
        assert result["values"][name] is not None
    for name in (
        "longest_active_span_m",
        "distance_extent_m",
        "lagged_correlation",
        "slowness_s_per_m",
        "apparent_velocity_mps",
        "direction",
        "moveout_residual_s",
    ):
        assert result["values"][name] is None
        assert result["status"][name] == "invalid_channel_geometry"
    assert "invalid_channel_geometry" in result["context"]["quality_flags"]
    json.dumps(result, allow_nan=False)


def test_detector_evidence_does_not_change_scientific_measurements():
    block, event = _event_block()
    evidence = Candidate(
        detector="fabricated",
        start_sample=event.start_sample,
        end_sample=event.end_sample,
        peak_sample=event.peak_sample,
        channel_start=event.channel_start,
        channel_stop=event.channel_stop,
        score=999.0,
        metrics={"slowness_s_per_m": -0.002, "apparent_velocity_mps": 500.0},
        quality_flags=("zero_slowness_common_mode",),
    )

    without_evidence = measure_event(block, event, ProcessingProfile())
    with_evidence = measure_event(
        block,
        replace(event, evidence=(evidence,), reasons=("changed",), decision_score=1.0),
        ProcessingProfile(),
    )

    assert with_evidence == without_evidence


def test_synchronous_channels_do_not_create_zero_velocity_propagation():
    block, event = _event_block()

    result = measure_event(block, event, ProcessingProfile())

    assert result["values"]["lagged_correlation"] > 0.8
    for name in (
        "slowness_s_per_m",
        "apparent_velocity_mps",
        "direction",
        "moveout_residual_s",
    ):
        assert result["values"][name] is None
        assert result["status"][name] == "unresolved_zero_lag"


def test_two_separated_bursts_are_two_pulses_with_one_interval():
    block, event = _event_block(amplitude=0.0, event_start=500, event_stop=800)
    data = block.data.copy()
    fs = block.metadata.sample_rate_hz
    for start, stop in ((520, 580), (680, 740)):
        time = np.arange(stop - start) / fs
        pulse = np.hanning(stop - start) * np.sin(2.0 * np.pi * 10.0 * time)
        data[start:stop, 1:7] += pulse[:, None].astype(np.float32)

    result = measure_event(replace(block, data=data), event, ProcessingProfile())

    assert result["values"]["pulse_count"] == 2.0
    assert result["values"]["pulse_interval_s"] == pytest.approx(1.6, abs=0.2)


def test_rise_time_uses_ten_percent_crossing_before_pulse_support():
    block, event = _event_block(event_start=500, event_stop=900)

    result = measure_event(block, event, ProcessingProfile())

    assert result["values"]["rise_time_s"] == pytest.approx(1.18, abs=0.12)


def test_bounded_local_lags_measure_supported_moveout():
    block, event = _event_block(amplitude=0.0, event_start=500, event_stop=800)
    data = block.data.copy()
    fs = block.metadata.sample_rate_hz
    pulse_time = np.arange(100) / fs
    pulse = np.hanning(100) * np.sin(2.0 * np.pi * 8.0 * pulse_time)
    for channel in range(8):
        start = 540 + channel
        data[start : start + pulse.size, channel] += pulse.astype(np.float32)

    result = measure_event(replace(block, data=data), event, ProcessingProfile())

    assert result["values"]["lagged_correlation"] > 0.8
    assert result["values"]["slowness_s_per_m"] == pytest.approx(0.001, abs=0.0002)
    assert result["values"]["apparent_velocity_mps"] == pytest.approx(1_000.0, rel=0.2)
    assert result["values"]["direction"] == 1.0
    assert result["values"]["moveout_residual_s"] <= 0.005


@pytest.mark.parametrize("delay_samples", (3, 4))
def test_search_boundary_lag_does_not_publish_truncated_velocity(delay_samples):
    block, event = _event_block(amplitude=0.0, event_start=500, event_stop=800)
    data = block.data.copy()
    time = np.arange(100) / block.metadata.sample_rate_hz
    pulse = np.hanning(100) * np.sin(2.0 * np.pi * 8.0 * time)
    for channel in range(8):
        start = 520 + delay_samples * channel
        data[start : start + pulse.size, channel] += pulse.astype(np.float32)

    result = measure_event(replace(block, data=data), event, ProcessingProfile())

    assert result["values"]["lagged_correlation"] > 0.5
    for name in (
        "slowness_s_per_m",
        "apparent_velocity_mps",
        "direction",
        "moveout_residual_s",
    ):
        assert result["values"][name] is None
        assert result["status"][name] == "lag_peak_at_search_boundary"


def test_one_nonzero_pair_among_zero_lags_is_not_resolved_moveout():
    block, event = _event_block(amplitude=0.0, event_start=500, event_stop=800)
    data = block.data.copy()
    time = np.arange(100) / block.metadata.sample_rate_hz
    pulse = np.hanning(100) * np.sin(2.0 * np.pi * 8.0 * time)
    for channel in range(8):
        start = 540 + int(channel >= 4)
        data[start : start + pulse.size, channel] += pulse.astype(np.float32)

    result = measure_event(replace(block, data=data), event, ProcessingProfile())

    assert result["values"]["lagged_correlation"] > 0.8
    for name in (
        "slowness_s_per_m",
        "apparent_velocity_mps",
        "direction",
        "moveout_residual_s",
    ):
        assert result["values"][name] is None
        assert result["status"][name] == "insufficient_nonzero_lags"
