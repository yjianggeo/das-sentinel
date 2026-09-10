from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
from scipy.signal import butter, detrend, hilbert, sosfiltfilt

from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import DasBlock, EventRecord

MEASUREMENT_NAMES = (
    "input_rms",
    "snr_db",
    "channel_snr_median_db",
    "channel_snr_p90_db",
    "valid_channel_count",
    "valid_channel_fraction",
    "bad_channel_fraction",
    "clipping_fraction",
    "trigger_duration_s",
    "energy_duration_5_95_s",
    "rise_time_s",
    "pulse_count",
    "pulse_interval_s",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_entropy",
    "frequency_resolution_hz",
    "band_fraction_low",
    "band_fraction_mid",
    "band_fraction_high",
    "active_channel_fraction",
    "longest_active_span_m",
    "distance_extent_m",
    "lagged_correlation",
    "slowness_s_per_m",
    "apparent_velocity_mps",
    "direction",
    "moveout_residual_s",
)


def _set(
    values: dict[str, float | None], status: dict[str, str], name: str, value: float
) -> None:
    number = float(value)
    if np.isfinite(number):
        values[name] = number
        status[name] = "measured"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _merge_runs(
    runs: list[tuple[int, int]], max_gap: int, min_length: int
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, stop in runs:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return [(start, stop) for start, stop in merged if stop - start >= min_length]


def _filtered_analysis(data: np.ndarray, fs: float, high_hz: float) -> np.ndarray:
    centered = (
        detrend(data, axis=0, type="linear")
        if data.shape[0] > 1
        else data - data.mean(axis=0)
    )
    sos = butter(4, (0.5, high_hz), btype="bandpass", fs=fs, output="sos")
    padlen = min(max(0, data.shape[0] - 1), max(1, round(fs)))
    return sosfiltfilt(sos, centered, axis=0, padlen=padlen)


def _measure_temporal(
    event_data: np.ndarray,
    noise_data: np.ndarray | None,
    fs: float,
    values: dict[str, float | None],
    status: dict[str, str],
) -> None:
    analytic = np.abs(hilbert(event_data, axis=0))
    envelope = np.sqrt(np.mean(np.square(analytic), axis=1))
    smooth_samples = min(event_data.shape[0], max(3, round(0.1 * fs)))
    envelope = np.convolve(
        envelope, np.ones(smooth_samples) / smooth_samples, mode="same"
    )
    peak = float(envelope.max(initial=0.0))
    baseline = 0.0
    noise_spread = 0.0
    if noise_data is not None and noise_data.size:
        noise_envelope = np.sqrt(
            np.mean(np.square(np.abs(hilbert(noise_data, axis=0))), axis=1)
        )
        baseline = float(np.median(noise_envelope))
        noise_spread = float(1.4826 * np.median(np.abs(noise_envelope - baseline)))

    if peak <= baseline:
        for name in ("rise_time_s", "pulse_count", "pulse_interval_s"):
            status[name] = "constant_envelope"
    else:
        threshold = max(
            baseline + 4.0 * noise_spread, baseline + 0.2 * (peak - baseline)
        )
        pulse_runs = _merge_runs(
            _runs(envelope >= threshold),
            max_gap=max(1, round(0.15 * fs)),
            min_length=max(1, round(0.1 * fs)),
        )
        _set(values, status, "pulse_count", len(pulse_runs))
        if pulse_runs:
            centers = np.asarray(
                [
                    start + int(np.argmax(envelope[start:stop]))
                    for start, stop in pulse_runs
                ]
            )
            if centers.size > 1:
                _set(values, status, "pulse_interval_s", np.mean(np.diff(centers)) / fs)
            else:
                status["pulse_interval_s"] = "single_pulse"
            main_index = max(
                range(len(pulse_runs)),
                key=lambda index: float(
                    envelope[pulse_runs[index][0] : pulse_runs[index][1]].max()
                ),
            )
            main_start, main_stop = pulse_runs[main_index]
            peak_index = main_start + int(np.argmax(envelope[main_start:main_stop]))
            search_start = pulse_runs[main_index - 1][1] if main_index else 0
            leading = envelope[search_start : peak_index + 1]
            dynamic = peak - baseline
            low_threshold = baseline + 0.1 * dynamic
            high_threshold = baseline + 0.9 * dynamic
            below_low = np.flatnonzero(leading < low_threshold)
            if below_low.size:
                low_crossing = int(below_low[-1] + 1)
                high = np.flatnonzero(leading[low_crossing:] >= high_threshold)
            else:
                low_crossing = 0
                high = np.asarray([], dtype=int)
            if below_low.size and high.size:
                _set(values, status, "rise_time_s", int(high[0]) / fs)
            elif not below_low.size:
                status["rise_time_s"] = "rise_start_not_observed"
            else:
                status["rise_time_s"] = "rise_not_resolved"
        else:
            for name in ("pulse_interval_s", "rise_time_s"):
                status[name] = "no_supported_pulse"

    energy = np.sum(np.square(event_data), axis=1)
    total_energy = float(energy.sum())
    if total_energy > 0.0:
        cumulative = np.cumsum(energy) / total_energy
        start_5 = int(np.searchsorted(cumulative, 0.05))
        stop_95 = int(np.searchsorted(cumulative, 0.95))
        _set(values, status, "energy_duration_5_95_s", max(0, stop_95 - start_5) / fs)
    else:
        status["energy_duration_5_95_s"] = "no_event_energy"


def _measure_spectrum(
    event_data: np.ndarray,
    fs: float,
    high_hz: float,
    values: dict[str, float | None],
    status: dict[str, str],
) -> None:
    samples = event_data.shape[0]
    spectral_names = (
        "dominant_frequency_hz",
        "spectral_centroid_hz",
        "spectral_entropy",
        "frequency_resolution_hz",
        "band_fraction_low",
        "band_fraction_mid",
        "band_fraction_high",
    )
    if samples < 2:
        for name in spectral_names:
            status[name] = "insufficient_frequency_resolution"
        return

    _set(values, status, "frequency_resolution_hz", fs / samples)
    tapered = event_data * np.hanning(samples)[:, None]
    frequency = np.fft.rfftfreq(samples, d=1.0 / fs)
    power = np.mean(np.square(np.abs(np.fft.rfft(tapered, axis=0))), axis=1)
    usable = (frequency >= 0.5) & (frequency <= high_hz)
    frequency, power = frequency[usable], power[usable]
    total = float(power.sum())
    if not power.size or total <= 0.0:
        for name in spectral_names:
            if name != "frequency_resolution_hz":
                status[name] = "no_spectral_energy"
        return

    _set(values, status, "dominant_frequency_hz", frequency[int(np.argmax(power))])
    _set(values, status, "spectral_centroid_hz", np.sum(frequency * power) / total)
    probabilities = power[power > 0.0] / total
    if probabilities.size > 1:
        _set(
            values,
            status,
            "spectral_entropy",
            -np.sum(probabilities * np.log(probabilities)) / np.log(probabilities.size),
        )
    else:
        status["spectral_entropy"] = "insufficient_spectral_bins"

    bands = {
        "band_fraction_low": (0.5, 5.0),
        "band_fraction_mid": (5.0, 20.0),
        "band_fraction_high": (20.0, 80.0),
    }
    for name, (low, nominal_high) in bands.items():
        if high_hz <= low:
            status[name] = "band_outside_nyquist"
        elif name == "band_fraction_low" and samples / fs < 2.0:
            status[name] = "insufficient_low_frequency_resolution"
        else:
            effective_high = min(nominal_high, high_hz)
            in_band = (frequency >= low) & (frequency < effective_high)
            if high_hz <= nominal_high:
                in_band = (frequency >= low) & (frequency <= effective_high)
            if not in_band.any():
                status[name] = "band_unresolved_at_frequency_resolution"
            else:
                _set(values, status, name, power[in_band].sum() / total)


def _best_local_lag(
    first: np.ndarray, second: np.ndarray, max_lag: int, min_overlap: int
) -> tuple[int, float, bool] | None:
    if first.size < min_overlap or second.size < min_overlap:
        return None
    max_lag = min(max_lag, first.size - min_overlap, second.size - min_overlap)
    first = first - first.mean()
    second = second - second.mean()
    best_lag, best_correlation = 0, -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = first[-lag:], second[:lag]
        elif lag > 0:
            left, right = first[:-lag], second[lag:]
        else:
            left, right = first, second
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = (
            float(np.dot(left, right) / denominator) if denominator > 0.0 else -1.0
        )
        if correlation > best_correlation:
            best_lag, best_correlation = lag, correlation
    return best_lag, best_correlation, max_lag > 0 and abs(best_lag) == max_lag


def _measure_propagation(
    event_data: np.ndarray,
    channel_indices: np.ndarray,
    distances: np.ndarray,
    active: np.ndarray,
    fs: float,
    max_slowness: float,
    values: dict[str, float | None],
    status: dict[str, str],
) -> None:
    active_positions = np.flatnonzero(active)
    propagation_names = (
        "slowness_s_per_m",
        "apparent_velocity_mps",
        "direction",
        "moveout_residual_s",
    )
    if active_positions.size < 2:
        status["lagged_correlation"] = "insufficient_active_channels"
        for name in propagation_names:
            status[name] = "insufficient_active_channels"
        return

    observations: list[tuple[float, float, float, bool]] = []
    eligible_pairs = 0
    min_overlap = max(3, round(0.1 * fs))
    for left_pos, right_pos in pairwise(active_positions):
        if channel_indices[right_pos] - channel_indices[left_pos] > 4:
            continue
        delta_distance = float(distances[right_pos] - distances[left_pos])
        if delta_distance == 0.0:
            continue
        eligible_pairs += 1
        max_lag = max(1, int(np.floor(max_slowness * abs(delta_distance) * fs)))
        result = _best_local_lag(
            event_data[:, left_pos],
            event_data[:, right_pos],
            max_lag,
            min_overlap,
        )
        if result is None:
            continue
        lag, correlation, at_boundary = result
        if np.isfinite(correlation):
            observations.append((delta_distance, lag / fs, correlation, at_boundary))

    if not observations:
        reason = "insufficient_lag_overlap" if eligible_pairs else "no_local_channel_pairs"
        status["lagged_correlation"] = reason
        for name in propagation_names:
            status[name] = reason
        return

    correlations = np.asarray([item[2] for item in observations])
    _set(values, status, "lagged_correlation", np.median(correlations))
    boundary_supported = [item for item in observations if item[2] >= 0.5 and item[3]]
    supported = [item for item in observations if item[2] >= 0.5 and not item[3]]
    if len(supported) < 3:
        reason = (
            "lag_peak_at_search_boundary"
            if boundary_supported
            else "insufficient_correlated_pairs"
        )
        for name in propagation_names:
            status[name] = reason
        return

    delta_distance = np.asarray([item[0] for item in supported])
    delay = np.asarray([item[1] for item in supported])
    weights = np.asarray([item[2] for item in supported])
    if np.all(delay == 0.0):
        for name in propagation_names:
            status[name] = "unresolved_zero_lag"
        return
    if np.count_nonzero(delay) <= delay.size / 2:
        for name in propagation_names:
            status[name] = "insufficient_nonzero_lags"
        return
    slope = float(
        np.sum(weights * delta_distance * delay)
        / np.sum(weights * np.square(delta_distance))
    )
    total_span = float(np.ptp(distances[active_positions]))
    if total_span <= 0.0:
        for name in propagation_names:
            status[name] = "no_spatial_extent"
        return
    resolution = 0.5 / (fs * total_span)
    if abs(slope) < resolution:
        for name in propagation_names:
            status[name] = "unresolved_zero_lag"
        return

    residual = float(
        np.sqrt(np.average(np.square(delay - slope * delta_distance), weights=weights))
    )
    nonzero = delay[delay != 0.0]
    consistency = (
        float(np.mean(np.sign(nonzero) == np.sign(slope))) if nonzero.size else 0.0
    )
    if abs(slope) > max_slowness or residual > 1.5 / fs or consistency < 0.7:
        for name in propagation_names:
            status[name] = "inconsistent_moveout"
        return

    _set(values, status, "slowness_s_per_m", slope)
    _set(values, status, "apparent_velocity_mps", 1.0 / abs(slope))
    _set(values, status, "direction", np.sign(slope))
    _set(values, status, "moveout_residual_s", residual)


def measure_event(
    block: DasBlock, event: EventRecord, profile: ProcessingProfile
) -> dict[str, Any]:
    """Measure a raw event window without classification or amplitude normalization."""
    values: dict[str, float | None] = {name: None for name in MEASUREMENT_NAMES}
    status = {name: "unavailable" for name in MEASUREMENT_NAMES}
    quality_flags: list[str] = []
    samples, channel_count = block.data.shape
    fs = float(block.metadata.sample_rate_hz)

    requested_start = event.start_sample - block.sample_offset
    requested_stop = event.end_sample - block.sample_offset
    event_start = min(samples, max(0, requested_start))
    event_stop = min(samples, max(event_start, requested_stop))
    requested_channel_start = event.channel_start
    requested_channel_stop = event.channel_stop
    channel_start = min(channel_count, max(0, requested_channel_start))
    channel_stop = min(channel_count, max(channel_start, requested_channel_stop))

    if event_start != requested_start:
        quality_flags.append("event_window_clipped_start")
    if event_stop != requested_stop:
        quality_flags.append("event_window_clipped_end")
    if (
        channel_start != requested_channel_start
        or channel_stop != requested_channel_stop
    ):
        quality_flags.append("channel_window_clipped")

    finite_fs = np.isfinite(fs) and fs > 1.0
    high_hz = min(80.0, 0.99 * (fs / 2.0)) if finite_fs else None
    band_available = high_hz is not None and high_hz > 0.5
    background_start = (
        max(0, requested_start - round(5.0 * fs)) if finite_fs else event_start
    )
    background_stop = (
        min(samples, max(background_start, requested_start - round(0.5 * fs)))
        if finite_fs
        else event_start
    )
    background_seconds = (
        max(0.0, (background_stop - background_start) / fs) if finite_fs else 0.0
    )
    background_available = background_seconds >= 1.0
    if not background_available:
        quality_flags.append("insufficient_background")
    post_guard_requested_s = 2.0
    post_guard_stop = (
        min(samples, event_stop + round(post_guard_requested_s * fs))
        if finite_fs
        else event_stop
    )
    post_guard_available_s = (
        max(0.0, (post_guard_stop - event_stop) / fs) if finite_fs else 0.0
    )
    if post_guard_available_s < post_guard_requested_s:
        quality_flags.append("incomplete_post_event_filter_guard")

    context: dict[str, Any] = {
        "algorithm_version": "measurements-v2.0",
        "effective_frequency_band_hz": [0.5, float(high_hz)]
        if band_available
        else None,
        "effective_spectral_bands_hz": (
            {
                "low": [0.5, float(min(5.0, high_hz))]
                if high_hz > 0.5
                else None,
                "mid": [5.0, float(min(20.0, high_hz))]
                if high_hz > 5.0
                else None,
                "high": [20.0, float(high_hz)] if high_hz > 20.0 else None,
            }
            if band_available
            else {"low": None, "mid": None, "high": None}
        ),
        "sampling_rate_hz": fs if finite_fs else None,
        "input_unit": str(block.metadata.unit),
        "background_seconds_available": float(background_seconds),
        "post_event_filter_guard_seconds_requested": post_guard_requested_s,
        "post_event_filter_guard_seconds_available": float(post_guard_available_s),
        "background_window_samples": {
            "start": int(block.sample_offset + background_start),
            "end": int(block.sample_offset + background_stop),
        },
        "event_window_samples": {
            "requested_start": int(event.start_sample),
            "requested_end": int(event.end_sample),
            "measured_start": int(block.sample_offset + event_start),
            "measured_end": int(block.sample_offset + event_stop),
        },
        "event_window_seconds": {
            "start": float(event_start / fs) if finite_fs else None,
            "end": float(event_stop / fs) if finite_fs else None,
        },
        "channel_window": {
            "requested_start": int(requested_channel_start),
            "requested_stop": int(requested_channel_stop),
            "measured_start": int(channel_start),
            "measured_stop": int(channel_stop),
        },
        "common_mode_treatment": "none",
        "quality_flags": quality_flags,
    }
    if finite_fs and requested_stop >= requested_start:
        _set(
            values,
            status,
            "trigger_duration_s",
            (requested_stop - requested_start) / fs,
        )

    selected_count = max(0, channel_stop - channel_start)
    if selected_count == 0:
        quality_flags.append("empty_channel_window")
        for name in MEASUREMENT_NAMES:
            if status[name] == "unavailable":
                status[name] = "empty_channel_window"
        status["clipping_fraction"] = "adc_limits_unavailable"
        return {
            "schema_version": "2",
            "values": values,
            "status": status,
            "context": context,
        }

    analysis_start = min(background_start, event_start)
    analysis_stop = post_guard_stop
    selected = np.asarray(
        block.data[analysis_start:analysis_stop, channel_start:channel_stop],
        dtype=np.float64,
    )
    valid = np.isfinite(selected).all(axis=0)
    if selected.shape[0]:
        valid &= np.ptp(np.where(np.isfinite(selected), selected, 0.0), axis=0) > 0.0
    else:
        valid[:] = False

    valid_count = int(valid.sum())
    _set(values, status, "valid_channel_count", valid_count)
    _set(values, status, "valid_channel_fraction", valid_count / selected_count)
    _set(values, status, "bad_channel_fraction", 1.0 - valid_count / selected_count)
    status["clipping_fraction"] = "adc_limits_unavailable"
    if not valid_count:
        quality_flags.append("no_valid_channels")
        for name in MEASUREMENT_NAMES:
            if status[name] == "unavailable":
                status[name] = "no_valid_channels"
        return {
            "schema_version": "2",
            "values": values,
            "status": status,
            "context": context,
        }

    raw_event = np.asarray(
        block.data[event_start:event_stop, channel_start:channel_stop], dtype=np.float64
    )[:, valid]
    if not raw_event.shape[0]:
        quality_flags.append("empty_event_window")
        for name in MEASUREMENT_NAMES:
            if status[name] == "unavailable":
                status[name] = "empty_event_window"
        return {
            "schema_version": "2",
            "values": values,
            "status": status,
            "context": context,
        }

    if not band_available:
        quality_flags.append("frequency_band_unavailable")
        for name in MEASUREMENT_NAMES:
            if status[name] == "unavailable":
                status[name] = "frequency_band_unavailable"
        return {
            "schema_version": "2",
            "values": values,
            "status": status,
            "context": context,
        }

    filtered = _filtered_analysis(selected[:, valid], fs, float(high_hz))
    event_left = event_start - analysis_start
    event_right = event_stop - analysis_start
    event_data = filtered[event_left:event_right]
    _set(values, status, "input_rms", np.sqrt(np.mean(np.square(event_data))))
    noise_data = None
    if background_available:
        noise_left = background_start - analysis_start
        noise_right = background_stop - analysis_start
        noise_data = filtered[noise_left:noise_right]

    event_power = np.mean(np.square(event_data), axis=0)
    noise_power: np.ndarray | None = None
    if noise_data is not None and noise_data.size:
        noise_power = np.mean(np.square(noise_data), axis=0)
        positive_noise = noise_power > 0.0
        if positive_noise.any():
            channel_snr = 10.0 * np.log10(
                event_power[positive_noise] / noise_power[positive_noise]
            )
            _set(values, status, "channel_snr_median_db", np.median(channel_snr))
            _set(values, status, "channel_snr_p90_db", np.percentile(channel_snr, 90.0))
            _set(
                values,
                status,
                "snr_db",
                10.0
                * np.log10(
                    np.mean(event_power[positive_noise])
                    / np.mean(noise_power[positive_noise])
                ),
            )
        else:
            for name in ("snr_db", "channel_snr_median_db", "channel_snr_p90_db"):
                status[name] = "zero_background_energy"
    else:
        for name in ("snr_db", "channel_snr_median_db", "channel_snr_p90_db"):
            status[name] = "insufficient_background"

    _measure_temporal(event_data, noise_data, fs, values, status)
    if event_data.shape[0] / fs < 2.0:
        quality_flags.append("coarse_frequency_resolution")
    _measure_spectrum(event_data, fs, float(high_hz), values, status)

    channel_rms = np.sqrt(event_power)
    relative_active = channel_rms >= 0.2 * float(channel_rms.max(initial=0.0))
    if noise_power is not None:
        active = relative_active & (event_power >= 2.0 * noise_power)
    else:
        active = relative_active
    _set(values, status, "active_channel_fraction", np.mean(active))
    channel_indices = np.arange(channel_start, channel_stop)[valid]
    distances = np.asarray(
        block.metadata.channel_distance_m[channel_start:channel_stop], dtype=np.float64
    )[valid]
    geometry_valid = bool(np.isfinite(distances).all())
    if not geometry_valid:
        quality_flags.append("invalid_channel_geometry")
        for name in (
            "longest_active_span_m",
            "distance_extent_m",
            "lagged_correlation",
            "slowness_s_per_m",
            "apparent_velocity_mps",
            "direction",
            "moveout_residual_s",
        ):
            status[name] = "invalid_channel_geometry"
        return {
            "schema_version": "2",
            "values": values,
            "status": status,
            "context": context,
        }
    active_positions = np.flatnonzero(active)
    if active_positions.size:
        _set(values, status, "distance_extent_m", np.ptp(distances[active_positions]))
        active_channel_indices = channel_indices[active_positions]
        breaks = np.flatnonzero(np.diff(active_channel_indices) > 1) + 1
        longest = 0.0
        for run in np.split(active_positions, breaks):
            if run.size:
                longest = max(longest, float(np.ptp(distances[run])))
        _set(values, status, "longest_active_span_m", longest)
    else:
        status["distance_extent_m"] = "no_active_channels"
        status["longest_active_span_m"] = "no_active_channels"

    _measure_propagation(
        event_data,
        channel_indices,
        distances,
        active,
        fs,
        float(profile.slowness_max_s_per_m),
        values,
        status,
    )

    for name, value in values.items():
        if value is not None and not np.isfinite(value):
            values[name] = None
            status[name] = "nonfinite_result"
    return {
        "schema_version": "2",
        "values": values,
        "status": status,
        "context": context,
    }
