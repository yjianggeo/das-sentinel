from __future__ import annotations

from typing import Any

import numpy as np

from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate
from das_sentinel.signal.preprocess import PreprocessedBlock


def _array_module(block: PreprocessedBlock) -> Any:
    if block.backend_name == "cpu":
        return np
    if block.backend_name == "cuda":
        import cupy

        return cupy
    raise ValueError(f"unknown array backend: {block.backend_name}")


def _to_numpy(value: Any, backend_name: str) -> np.ndarray:
    if backend_name == "cuda":
        import cupy

        return cupy.asnumpy(value)
    return np.asarray(value)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    transitions = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1), strict=True))


def _band_metrics(segment: Any, sample_rate_hz: float, bands: tuple[tuple[float, float], ...], xp: Any) -> tuple[Any, Any]:
    centered = segment - xp.mean(segment, axis=0, keepdims=True)
    spectrum = xp.fft.rfft(centered, axis=0)
    power = xp.mean(xp.square(xp.abs(spectrum)), axis=1)
    frequency = xp.fft.rfftfreq(segment.shape[0], d=1.0 / sample_rate_hz)
    band_powers = xp.asarray([xp.sum(power[(frequency >= low) & (frequency < high)]) for low, high in bands])
    normalized = band_powers / xp.maximum(xp.sum(band_powers), xp.finfo(segment.dtype).eps)
    entropy = -xp.sum(xp.where(normalized > 0, normalized * xp.log(normalized), 0.0))
    return normalized, entropy / xp.log(len(bands))


def detect_energy(block: PreprocessedBlock, profile: ProcessingProfile) -> list[Candidate]:
    """Detect sustained, coincident high-energy intervals and their spectral evidence."""
    xp = _array_module(block)
    data = block.data
    samples, channels = data.shape
    window = max(1, round(profile.energy_window_seconds * block.sample_rate_hz))
    if samples <= 2 * window or channels == 0:
        return []

    power = xp.square(data)
    cumulative = xp.concatenate((xp.zeros((1, channels), dtype=data.dtype), xp.cumsum(power, axis=0)))
    rms = xp.sqrt((cumulative[window:] - cumulative[:-window]) / window)
    baseline = xp.median(rms, axis=0)
    ratio = rms / xp.maximum(baseline, xp.finfo(data.dtype).eps)
    good = xp.asarray(~block.bad_channel_mask)
    active_channels = (ratio >= profile.energy_ratio_threshold) & good[None, :]
    coincidence = xp.sum(active_channels, axis=1)
    active = coincidence >= profile.min_coincidence
    active[:window] = False
    active[-window:] = False
    active_cpu = _to_numpy(active, block.backend_name)

    min_duration = max(1, round(profile.min_duration_seconds * block.sample_rate_hz))
    candidates: list[Candidate] = []
    for start, stop in _runs(active_cpu):
        if stop - start < min_duration:
            continue
        peak_relative = int(_to_numpy(xp.argmax(xp.max(ratio[start:stop], axis=1)), block.backend_name))
        peak_index = start + peak_relative
        peak_ratio = float(_to_numpy(xp.max(ratio[peak_index]), block.backend_name))
        channel_hits = xp.any(active_channels[start:stop], axis=0)
        channel_indices = np.flatnonzero(_to_numpy(channel_hits, block.backend_name))
        if channel_indices.size < profile.min_coincidence:
            continue
        segment = data[start + window - 1 : stop + window - 1, channel_indices]
        centered = segment - xp.mean(segment, axis=0, keepdims=True)
        variance = xp.mean(xp.square(centered))
        kurtosis = xp.mean(xp.power(centered, 4)) / xp.maximum(
            xp.square(variance), xp.finfo(data.dtype).eps
        ) - 3.0
        peak_median = xp.max(xp.abs(segment)) / xp.maximum(
            xp.median(xp.abs(segment)), xp.finfo(data.dtype).eps
        )
        band_powers, entropy = _band_metrics(segment, block.sample_rate_hz, profile.energy_bands_hz, xp)
        score = 0.60 + 0.40 * np.clip(
            (peak_ratio - profile.energy_ratio_threshold) / profile.energy_ratio_threshold, 0.0, 1.0
        )
        metrics = {
            "peak_ratio": peak_ratio,
            "peak_coincidence": float(_to_numpy(coincidence[peak_index], block.backend_name)),
            "excess_kurtosis": float(_to_numpy(kurtosis, block.backend_name)),
            "peak_median_ratio": float(_to_numpy(peak_median, block.backend_name)),
            "spectral_entropy": float(_to_numpy(entropy, block.backend_name)),
        }
        metrics.update(
            {
                f"band_power_{index}": float(_to_numpy(value, block.backend_name))
                for index, value in enumerate(band_powers)
            }
        )
        candidates.append(
            Candidate(
                detector="energy",
                start_sample=block.sample_offset + start + window - 1,
                end_sample=block.sample_offset + stop + window - 1,
                peak_sample=block.sample_offset + peak_index + window - 1,
                channel_start=int(channel_indices[0]),
                channel_stop=int(channel_indices[-1]) + 1,
                score=float(score),
                metrics=metrics,
            )
        )
    return candidates
