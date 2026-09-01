from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate, DasBlock
from das_sentinel.signal.backend import ArrayBackend
from das_sentinel.signal.preprocess import PreprocessedBlock, preprocess_block


@dataclass(frozen=True)
class WaterfallFilter:
    detector: str
    low_hz: float
    high_hz: float


def detector_band(
    profile: ProcessingProfile, detector: str
) -> tuple[float, float]:
    if detector == "stalta":
        return profile.stalta_freq_low_hz, profile.stalta_freq_high_hz
    if detector == "energy":
        return (
            min(low for low, _ in profile.energy_bands_hz),
            max(high for _, high in profile.energy_bands_hz),
        )
    if detector == "coherence":
        return profile.coherence_freq_low_hz, profile.coherence_freq_high_hz
    if detector == "trajectory":
        return profile.trajectory_freq_low_hz, profile.trajectory_freq_high_hz
    raise ValueError(f"unknown detector frequency bank: {detector}")


def resolve_waterfall_filter(
    evidence: Sequence[Candidate], profile: ProcessingProfile
) -> WaterfallFilter:
    if not evidence:
        raise ValueError("waterfall filter requires evidence")
    primary = min(
        evidence,
        key=lambda item: (-item.score, item.peak_sample, item.detector),
    ).detector.split(":", 1)[0]
    low_hz, high_hz = detector_band(profile, primary)
    return WaterfallFilter(primary, low_hz, high_hz)


def _distance_mask(
    distances_m: np.ndarray, start_m: float, stop_m: float
) -> np.ndarray:
    distances = np.asarray(distances_m, dtype=float)
    if distances.ndim != 1 or not np.isfinite(distances).all():
        raise ValueError("distances_m must be a finite one-dimensional array")
    if not np.isfinite(start_m) or not np.isfinite(stop_m):
        raise ValueError("start_m and stop_m must be finite")
    if start_m > stop_m:
        raise ValueError("start_m must not exceed stop_m")
    return (distances >= start_m) & (distances <= stop_m)


def channel_slice_for_distance(
    distances_m: np.ndarray, start_m: float, stop_m: float
) -> slice:
    selected = np.flatnonzero(_distance_mask(distances_m, start_m, stop_m))
    if selected.size == 0:
        raise ValueError("distance range has no selected channels")
    return slice(int(selected[0]), int(selected[-1]) + 1)


def usable_channel_indices(
    distances_m: np.ndarray,
    bad_channel_mask: np.ndarray,
    start_m: float,
    stop_m: float,
) -> np.ndarray:
    selected = _distance_mask(distances_m, start_m, stop_m)
    bad = np.asarray(bad_channel_mask, dtype=bool)
    if bad.shape != selected.shape:
        raise ValueError("bad_channel_mask must match channel distances")
    usable = np.flatnonzero(selected & ~bad)
    if usable.size == 0:
        raise ValueError("distance range has no usable channels")
    return usable


def preprocess_for_waterfall(
    block: DasBlock,
    profile: ProcessingProfile,
    backend: ArrayBackend,
    filter_context: WaterfallFilter,
) -> PreprocessedBlock:
    band_profile = profile.model_copy(
        update={
            "stalta_freq_low_hz": filter_context.low_hz,
            "stalta_freq_high_hz": filter_context.high_hz,
        }
    )
    return preprocess_block(block, band_profile, backend)


def detector_display_name(detector: str) -> str:
    names = {
        "stalta": "STA/LTA",
        "energy": "Energy",
        "coherence": "Coherence",
        "trajectory": "Trajectory",
    }
    try:
        return names[detector]
    except KeyError as error:
        raise ValueError(f"unknown detector: {detector}") from error
