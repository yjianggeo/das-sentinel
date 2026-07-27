from __future__ import annotations

from typing import Any

import numpy as np

from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate
from das_sentinel.signal.backend import ArrayBackend
from das_sentinel.signal.preprocess import PreprocessedBlock


def _as_float(value: Any, backend: ArrayBackend) -> float:
    return float(backend.asnumpy(value))


def _coherence_score(semblance: float, threshold: float) -> float:
    return float(0.60 + 0.40 * np.clip((semblance - threshold) / threshold, 0.0, 1.0))


def _selects_moveout(
    *,
    score: float,
    slowness: float,
    best_score: float,
    best_slowness: float,
) -> bool:
    """Select strictly higher coherence, then the smaller equivalent moveout."""
    if score > best_score:
        return True
    if score < best_score:
        return False
    return abs(slowness) < abs(best_slowness)


def _trailing_sum(values: Any, width: int, xp: Any) -> Any:
    cumulative = xp.concatenate((xp.zeros(1, dtype=values.dtype), xp.cumsum(values)))
    return cumulative[width:] - cumulative[:-width]


def _moveout_semblance(data: Any, delays: Any, width: int, xp: Any) -> tuple[Any, Any]:
    """Return debiased sample coherence and the aligned stack for one moveout."""
    samples, channels = data.shape
    time = xp.arange(samples)[:, None]
    source_index = time + delays[None, :]
    valid = (source_index >= 0) & (source_index < samples)
    values = data[xp.clip(source_index, 0, samples - 1), xp.arange(channels)[None, :]]
    values = xp.where(valid, values, 0.0)
    valid_channels = xp.sum(valid, axis=1)
    individual_energy = xp.sum(xp.square(values), axis=1)
    stack = xp.sum(values, axis=1)
    denominator = valid_channels * individual_energy
    window_numerator = _trailing_sum(xp.square(stack), width, xp)
    window_denominator = _trailing_sum(denominator, width, xp)
    raw = xp.where(
        window_denominator > xp.finfo(data.dtype).eps,
        window_numerator / window_denominator,
        0.0,
    )
    valid_windows = _trailing_sum((valid_channels == channels).astype(data.dtype), width, xp) == width
    return xp.where(valid_windows, xp.clip(raw - 1.0 / channels, 0.0, 1.0), 0.0), stack


def detect_coherence(
    block: PreprocessedBlock, profile: ProcessingProfile, backend: ArrayBackend
) -> list[Candidate]:
    """Find the most coherent integer-sample spatial moveout in a block."""
    if profile.coherence_threshold <= 0:
        raise ValueError("coherence_threshold must be positive")
    if block.backend_name != backend.name:
        raise ValueError("block and backend must use the same array backend")

    valid_indices = np.flatnonzero(~block.bad_channel_mask)
    samples = block.data.shape[0]
    if samples == 0 or valid_indices.size < profile.min_coincidence:
        return []

    xp = backend.xp
    data = block.data[:, valid_indices]
    window = min(samples, max(1, round(profile.coherence_window_seconds * block.sample_rate_hz)))
    distances = backend.asarray(block.channel_distance_m[valid_indices], dtype=data.dtype)
    distances = distances - xp.min(distances)
    slownesses = xp.linspace(
        -profile.slowness_max_s_per_m,
        profile.slowness_max_s_per_m,
        profile.slowness_points,
        dtype=data.dtype,
    )
    slownesses[profile.slowness_points // 2] = 0.0

    best_semblance = xp.asarray(-1.0, dtype=data.dtype)
    best_peak = xp.asarray(0, dtype=xp.int64)
    best_slowness = xp.asarray(0.0, dtype=data.dtype)
    best_stack = xp.zeros(samples, dtype=data.dtype)
    for slowness in slownesses:
        delays = xp.rint(slowness * distances * block.sample_rate_hz).astype(xp.int64)
        semblance, stack = _moveout_semblance(data, delays, window, xp)
        peak = xp.argmax(semblance)
        score = semblance[peak]
        score_value = _as_float(score, backend)
        best_value = _as_float(best_semblance, backend)
        slowness_value = _as_float(slowness, backend)
        best_slowness_value = _as_float(best_slowness, backend)
        if _selects_moveout(
            score=score_value,
            slowness=slowness_value,
            best_score=best_value,
            best_slowness=best_slowness_value,
        ):
            best_semblance = score
            best_peak = peak
            best_slowness = slowness
            best_stack = stack

    semblance = _as_float(best_semblance, backend)
    if semblance < profile.coherence_threshold:
        return []

    slowness = _as_float(best_slowness, backend)
    coherence_start = int(backend.asnumpy(best_peak))
    peak = coherence_start + int(backend.asnumpy(xp.argmax(xp.abs(best_stack[coherence_start : coherence_start + window]))))
    near_zero = abs(slowness) <= profile.slowness_max_s_per_m / profile.slowness_points
    direction = 0.0 if near_zero else float(np.sign(slowness))
    apparent_velocity = 0.0 if near_zero else 1.0 / abs(slowness)
    half_window = max(1, window // 2)
    score = _coherence_score(semblance, profile.coherence_threshold)
    return [
        Candidate(
            detector="coherence",
            start_sample=block.sample_offset + max(0, peak - half_window),
            end_sample=block.sample_offset + min(samples, peak + half_window + 1),
            peak_sample=block.sample_offset + peak,
            channel_start=int(valid_indices[0]),
            channel_stop=int(valid_indices[-1]) + 1,
            score=score,
            metrics={
                "semblance": semblance,
                "slowness_s_per_m": slowness,
                "apparent_velocity_mps": apparent_velocity,
                "direction": direction,
            },
            quality_flags=("zero_slowness_common_mode",) if near_zero else (),
        )
    ]
