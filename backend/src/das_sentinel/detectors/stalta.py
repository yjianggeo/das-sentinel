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


def _trailing_mean(values: Any, width: int, xp: Any) -> Any:
    cumulative = xp.concatenate((xp.zeros((1, values.shape[1]), dtype=values.dtype), xp.cumsum(values, axis=0)))
    return (cumulative[width:] - cumulative[:-width]) / width


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    transitions = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1), strict=True))


def detect_stalta(block: PreprocessedBlock, profile: ProcessingProfile) -> list[Candidate]:
    """Detect coincident squared-amplitude STA/LTA transients."""
    xp = _array_module(block)
    data = block.data
    samples, channels = data.shape
    sta = max(1, round(profile.sta_seconds * block.sample_rate_hz))
    lta = max(sta + 1, round(profile.lta_seconds * block.sample_rate_hz))
    if samples <= 2 * lta or channels == 0:
        return []

    power = xp.square(data)
    sta_mean = _trailing_mean(power, sta, xp)
    lta_mean = _trailing_mean(power, lta, xp)
    aligned_sta = sta_mean[lta - sta :]
    epsilon = xp.finfo(data.dtype).eps
    ratios = aligned_sta / xp.maximum(lta_mean, epsilon)
    good = xp.asarray(~block.bad_channel_mask)
    active_channels = (ratios >= profile.stalta_threshold) & good[None, :]
    coincidence = xp.sum(active_channels, axis=1)
    active = coincidence >= profile.min_coincidence

    margin = lta
    active[-margin:] = False
    active_cpu = _to_numpy(active, block.backend_name)
    min_duration = max(1, round(profile.min_duration_seconds * block.sample_rate_hz))
    candidates: list[Candidate] = []
    for start, stop in _runs(active_cpu):
        if stop - start < min_duration:
            continue
        run_ratios = ratios[start:stop]
        good_ratios = xp.where(good[None, :], run_ratios, 0.0)
        peak_relative = int(_to_numpy(xp.argmax(xp.max(good_ratios, axis=1)), block.backend_name))
        peak_index = start + peak_relative
        peak_ratio = float(_to_numpy(xp.max(good_ratios[peak_relative]), block.backend_name))
        run_channels = active_channels[start:stop]
        channel_hits = xp.any(run_channels, axis=0)
        channel_indices = np.flatnonzero(_to_numpy(channel_hits, block.backend_name))
        if channel_indices.size < profile.min_coincidence:
            continue
        score = 0.60 + 0.40 * np.clip(
            (peak_ratio - profile.stalta_threshold) / profile.stalta_threshold, 0.0, 1.0
        )
        candidates.append(
            Candidate(
                detector="stalta",
                start_sample=block.sample_offset + start + lta - 1,
                end_sample=block.sample_offset + stop + lta - 1,
                peak_sample=block.sample_offset + peak_index + lta - 1,
                channel_start=int(channel_indices[0]),
                channel_stop=int(channel_indices[-1]) + 1,
                score=float(score),
                metrics={
                    "peak_ratio": peak_ratio,
                    "peak_coincidence": float(_to_numpy(coincidence[peak_index], block.backend_name)),
                },
            )
        )
    return candidates
