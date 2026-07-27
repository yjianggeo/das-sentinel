from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np

from das_sentinel.domain import EventRecord
from das_sentinel.signal.preprocess import PreprocessedBlock

FEATURE_NAMES = (
    "duration_s", "rise_time_s", "pulse_count", "pulse_interval_s", "time_of_day_hours",
    "dominant_frequency_hz", "bandwidth_hz", "band_energy_ratio_low_mid",
    "band_energy_ratio_mid_high", "spectral_centroid_hz", "spectral_entropy",
    "distance_extent_m", "energy_centroid_m", "shallow_deep_energy_ratio", "continuity",
    "best_slowness_s_per_m", "apparent_velocity_mps", "direction", "coherence",
    "zero_slowness_flag", "rms", "kurtosis", "peak_median_ratio",
    "cross_channel_correlation", "repetition_rate_hz", "site_id", "cable_type",
    "device_config",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _event_window(block: PreprocessedBlock, event: EventRecord) -> tuple[slice, np.ndarray]:
    samples, channels = np.asarray(block.data).shape
    start = max(0, event.start_sample - block.sample_offset)
    stop = min(samples, max(start, event.end_sample - block.sample_offset))
    channel_start = min(channels, max(0, event.channel_start))
    channel_stop = min(channels, max(channel_start, event.channel_stop))
    indices = np.arange(channel_start, channel_stop)
    return slice(start, stop), indices[~block.bad_channel_mask[indices]]


def _spectrum(data: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    if data.shape[0] < 2 or data.shape[1] == 0 or fs <= 0:
        return np.zeros(1), np.zeros(1)
    window = min(data.shape[0], max(8, round(fs * 2)))
    step = max(1, window // 2)
    starts = range(0, data.shape[0] - window + 1, step)
    segments = [data[start : start + window] for start in starts] or [data]
    powers = []
    for segment in segments:
        taper = np.hanning(segment.shape[0])
        powers.append(np.mean(np.abs(np.fft.rfft(segment * taper[:, None], axis=0)) ** 2, axis=1))
    return np.fft.rfftfreq(segments[0].shape[0], d=1.0 / fs), np.mean(powers, axis=0)


def _propagation(event: EventRecord) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for evidence in sorted(event.evidence, key=lambda item: item.score, reverse=True):
        for key, value in evidence.metrics.items():
            metrics.setdefault(key, _finite(value))
    slowness = metrics.get("slowness_s_per_m", 0.0)
    return {
        "best_slowness_s_per_m": slowness,
        "apparent_velocity_mps": metrics.get("apparent_velocity_mps", 0.0),
        "direction": metrics.get("direction", 0.0),
        "coherence": metrics.get("semblance", metrics.get("coherence", 0.0)),
        "zero_slowness_flag": float(("slowness_s_per_m" in metrics and abs(slowness) <= np.finfo(float).eps) or any(
            "zero_slowness" in flag for evidence in event.evidence for flag in evidence.quality_flags
        )),
    }


def extract_features(
    block: PreprocessedBlock,
    event: EventRecord,
    fs: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, float | str]:
    """Extract a stable, finite event schema without reading source paths as features."""
    sample_rate = _finite(fs if fs is not None else block.sample_rate_hz)
    window, channels = _event_window(block, event)
    data = np.asarray(block.data, dtype=np.float64)[window][:, channels]
    duration = max(0.0, (event.end_sample - event.start_sample) / sample_rate) if sample_rate else 0.0
    values: dict[str, float | str] = {name: 0.0 for name in FEATURE_NAMES}
    values["duration_s"] = duration
    details = context or {}
    values["site_id"] = str(details.get("site_id", "unknown"))
    values["cable_type"] = str(details.get("cable_type", "unknown"))
    values["device_config"] = str(details.get("device_config", "unknown"))
    if data.size == 0:
        return values

    energy = np.square(data)
    channel_energy = energy.sum(axis=0)
    envelope = np.sqrt(np.mean(energy, axis=1))
    peak = float(envelope.max(initial=0.0))
    epsilon = np.finfo(float).eps
    if peak > epsilon:
        high = int(np.flatnonzero(envelope >= 0.9 * peak)[0])
        low = int(np.flatnonzero(envelope[: high + 1] >= 0.1 * peak)[0])
        crossings = np.flatnonzero((envelope[1:] >= 0.5 * peak) & (envelope[:-1] < 0.5 * peak)) + 1
        values["rise_time_s"] = (high - low) / sample_rate if sample_rate else 0.0
        values["pulse_count"] = float(crossings.size)
        values["pulse_interval_s"] = float(np.mean(np.diff(crossings)) / sample_rate) if crossings.size > 1 and sample_rate else 0.0
        values["repetition_rate_hz"] = float(crossings.size / duration) if duration else 0.0

    frequency, power = _spectrum(data, sample_rate)
    total_power = float(power.sum())
    if total_power > epsilon:
        dominant = int(np.argmax(power))
        values["dominant_frequency_hz"] = float(frequency[dominant])
        values["bandwidth_hz"] = float(np.sqrt(np.average((frequency - np.average(frequency, weights=power)) ** 2, weights=power)))
        values["spectral_centroid_hz"] = float(np.average(frequency, weights=power))
        probabilities = power[power > 0] / total_power
        values["spectral_entropy"] = float(-np.sum(probabilities * np.log(probabilities)) / np.log(max(2, power.size)))
        bands = [power[(frequency >= low) & (frequency < high)].sum() for low, high in ((0.5, 5.0), (5.0, 20.0), (20.0, np.inf))]
        values["band_energy_ratio_low_mid"] = float(bands[0] / max(bands[1], epsilon))
        values["band_energy_ratio_mid_high"] = float(bands[1] / max(bands[2], epsilon))

    distances = np.asarray(block.channel_distance_m)[channels]
    values["distance_extent_m"] = float(np.ptp(distances)) if distances.size else 0.0
    total_energy = float(channel_energy.sum())
    if total_energy > epsilon:
        values["energy_centroid_m"] = float(np.average(distances, weights=channel_energy))
        split = float(np.median(distances))
        shallow = channel_energy[distances <= split].sum()
        deep = channel_energy[distances > split].sum()
        values["shallow_deep_energy_ratio"] = float(shallow / max(deep, epsilon))
        active = channel_energy >= 0.2 * channel_energy.max()
        values["continuity"] = float(active.sum() / active.size)
    values["rms"] = float(np.sqrt(np.mean(energy)))
    centered = data - np.mean(data)
    variance = float(np.mean(np.square(centered)))
    values["kurtosis"] = float(np.mean(centered**4) / (variance**2)) if variance > epsilon else 0.0
    mean_absolute_amplitude = float(np.mean(np.abs(data)))
    values["peak_median_ratio"] = peak / mean_absolute_amplitude if mean_absolute_amplitude > epsilon else 0.0
    if data.shape[1] > 1 and np.all(np.std(data, axis=0) > epsilon):
        correlation = np.corrcoef(data.T)
        upper = correlation[np.triu_indices(data.shape[1], 1)]
        values["cross_channel_correlation"] = float(np.mean(upper[np.isfinite(upper)])) if np.isfinite(upper).any() else 0.0
    values.update(_propagation(event))
    start_time = details.get("start_time_utc")
    if isinstance(start_time, datetime):
        values["time_of_day_hours"] = start_time.hour + start_time.minute / 60 + start_time.second / 3600
    return {name: values[name] for name in FEATURE_NAMES}
