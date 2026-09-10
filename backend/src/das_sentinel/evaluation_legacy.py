"""Historical detector baseline for explicit offline comparisons only.

The production pipeline does not import or select this implementation.
"""
from __future__ import annotations

import time
from dataclasses import replace
from datetime import timedelta

import numpy as np

from das_sentinel.config import ProcessingProfile
from das_sentinel.detectors.coherence import detect_coherence
from das_sentinel.detectors.energy import detect_energy
from das_sentinel.detectors.stalta import detect_stalta
from das_sentinel.detectors.trajectory import detect_trajectories
from das_sentinel.domain import Candidate, DasBlock
from das_sentinel.signal.backend import ArrayBackend
from das_sentinel.signal.preprocess import (
    PreprocessedBlock,
    filter_prepared_signal,
    prepare_signal,
)
from das_sentinel.waterfalls import detector_band


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _detect_block(
    block: DasBlock,
    profile: ProcessingProfile,
    backend: ArrayBackend,
    timings: dict[str, float],
    *,
    logical_start: int | None = None,
) -> list[Candidate]:
    found: list[Candidate] = []
    logical_block = _logical_detector_block(block, logical_start)
    grouped: dict[tuple[int, int], tuple[DasBlock, list[str]]] = {}
    for detector in ("stalta", "energy", "coherence", "trajectory"):
        detector_source = block if detector == "stalta" else logical_block
        key = (
            detector_source.sample_offset,
            detector_source.data.shape[0],
        )
        if key not in grouped:
            grouped[key] = (detector_source, [])
        grouped[key][1].append(detector)
    for detector_source, detectors in grouped.values():
        preprocess_started = time.perf_counter()
        prepared_signal = prepare_signal(detector_source, profile, backend)
        timings["preprocess"] += _elapsed_ms(preprocess_started)
        bands: dict[tuple[float, float], list[str]] = {}
        for detector in detectors:
            bands.setdefault(detector_band(profile, detector), []).append(detector)
        for band, band_detectors in bands.items():
            preprocess_started = time.perf_counter()
            prepared_block = filter_prepared_signal(
                prepared_signal, band[0], band[1], backend
            )
            timings["preprocess"] += _elapsed_ms(preprocess_started)
            for detector in band_detectors:
                detector_block = prepared_block
                if detector == "trajectory" and detector_block.backend_name != "cpu":
                    detector_block = replace(
                        detector_block,
                        data=np.asarray(
                            backend.asnumpy(detector_block.data), dtype=np.float32
                        ),
                        backend_name="cpu",
                    )
                detect_started = time.perf_counter()
                found.extend(
                    _run_detector(detector, detector_block, profile, backend)
                )
                timings["detect"] += _elapsed_ms(detect_started)
                del detector_block
            del prepared_block
        del prepared_signal
    if hasattr(backend, "clear_cache"):
        backend.clear_cache()
    return found


def _logical_detector_block(
    block: DasBlock, logical_start: int | None
) -> DasBlock:
    if logical_start is None or logical_start == block.sample_offset:
        return block
    block_stop = block.sample_offset + block.data.shape[0]
    if not block.sample_offset <= logical_start < block_stop:
        raise ValueError("logical block start is outside the read block")
    relative_start = logical_start - block.sample_offset
    return replace(
        block,
        data=block.data[relative_start:],
        metadata=replace(
            block.metadata,
            start_time_utc=block.metadata.start_time_utc
            + timedelta(
                seconds=relative_start / block.metadata.sample_rate_hz
            ),
        ),
        sample_offset=logical_start,
    )


def _run_detector(
    detector: str,
    block: PreprocessedBlock,
    profile: ProcessingProfile,
    backend: ArrayBackend,
) -> list[Candidate]:
    if detector == "stalta":
        return detect_stalta(block, profile)
    if detector == "energy":
        return detect_energy(block, profile)
    if detector == "coherence":
        return detect_coherence(block, profile, backend)
    if detector == "trajectory":
        return detect_trajectories(block, profile)
    raise ValueError(f"unknown detector: {detector}")


def _deduplicate_candidates(
    candidates: list[Candidate],
    profile: ProcessingProfile,
    sample_rate_hz: float,
) -> list[Candidate]:
    peak_tolerance = max(1, round(profile.merge_gap_seconds * sample_rate_hz))
    remaining = sorted(
        candidates,
        key=lambda item: (
            item.detector,
            item.peak_sample,
            item.start_sample,
            item.channel_start,
        ),
    )
    selected: list[Candidate] = []
    while remaining:
        group = [remaining.pop(0)]
        index = 0
        while index < len(remaining):
            candidate = remaining[index]
            if any(
                _overlap_copy(candidate, member, peak_tolerance)
                for member in group
            ):
                group.append(remaining.pop(index))
                index = 0
            else:
                index += 1
        selected.append(
            min(
                group,
                key=lambda item: (
                    bool(item.quality_flags),
                    -item.score,
                    item.peak_sample,
                    item.start_sample,
                    item.end_sample,
                    item.channel_start,
                    item.channel_stop,
                ),
            )
        )
    return sorted(
        selected,
        key=lambda item: (
            item.start_sample,
            item.end_sample,
            item.channel_start,
            item.detector,
            item.peak_sample,
        ),
    )


def _overlap_copy(
    left: Candidate, right: Candidate, peak_tolerance: int
) -> bool:
    return (
        left.detector == right.detector
        and abs(left.peak_sample - right.peak_sample) <= peak_tolerance
        and max(left.start_sample, right.start_sample)
        <= min(left.end_sample, right.end_sample)
        and max(left.channel_start, right.channel_start)
        < min(left.channel_stop, right.channel_stop)
    )
