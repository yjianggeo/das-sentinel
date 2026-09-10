from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks

from das_sentinel.config import ProcessingProfile
from das_sentinel.detectors.coherence import _moveout_semblance
from das_sentinel.detectors.trajectory import detect_trajectories
from das_sentinel.domain import Candidate, DasBlock, EventRecord
from das_sentinel.signal.backend import ArrayBackend
from das_sentinel.signal.preprocess import (
    PreprocessedBlock,
    filter_prepared_signal,
    prepare_signal,
)

ALGORITHM_VERSION = "screening-v2"
_MAX_COHERENCE_CHANNELS = 64


@dataclass(frozen=True)
class ScreeningOptions:
    spatial_window_m: float = 256.0
    spatial_overlap: float = 0.5
    off_ratio: float = 2.0
    release_gap_seconds: float = 0.15
    short_sta_seconds: float = 0.1
    short_lta_seconds: float = 2.0
    coincidence_tolerance_seconds: float = 0.03
    common_mode_removal: bool = False
    enable_coherence: bool = True
    enable_trajectory: bool = True


@dataclass(frozen=True)
class ScreeningResult:
    candidates: tuple[Candidate, ...]
    events: tuple[EventRecord, ...]
    diagnostics: dict[str, Any]


def _validate_options(options: ScreeningOptions, profile: ProcessingProfile) -> None:
    positive = (
        options.spatial_window_m,
        options.off_ratio,
        options.short_sta_seconds,
        options.short_lta_seconds,
    )
    nonnegative = (
        options.release_gap_seconds,
        options.coincidence_tolerance_seconds,
    )
    if (
        any(not math.isfinite(value) or value <= 0 for value in positive)
        or any(not math.isfinite(value) or value < 0 for value in nonnegative)
        or not math.isfinite(options.spatial_overlap)
        or not 0 <= options.spatial_overlap < 1
        or options.short_lta_seconds <= options.short_sta_seconds
        or options.off_ratio >= profile.stalta_threshold
    ):
        raise ValueError("invalid screening option values")
    if not all(
        isinstance(value, bool)
        for value in (
            options.common_mode_removal,
            options.enable_coherence,
            options.enable_trajectory,
        )
    ):
        raise ValueError("invalid screening option boolean")


def _validate_block(block: DasBlock, backend: ArrayBackend) -> None:
    sample_rate_hz = block.metadata.sample_rate_hz
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if backend.name not in {"cpu", "cuda"}:
        raise ValueError(f"unknown array backend: {backend.name}")


def _effective_band(
    low_hz: float,
    high_hz: float,
    sample_rate_hz: float,
) -> tuple[float, float] | None:
    effective_high_hz = min(high_hz, sample_rate_hz / 2.0)
    if low_hz >= effective_high_hz:
        return None
    return float(low_hz), float(effective_high_hz)


def _option_fingerprint(options: ScreeningOptions) -> float:
    encoded = json.dumps(asdict(options), sort_keys=True, separators=(",", ":")).encode()
    return float(int(hashlib.sha256(encoded).hexdigest()[:13], 16))


def _physical_windows(
    distances: np.ndarray,
    options: ScreeningOptions,
) -> tuple[list[np.ndarray], dict[str, int]]:
    finite = np.isfinite(distances)
    finite_indices = np.flatnonzero(finite)
    skipped = {
        "nonfinite_channels": int((~finite).sum()),
        "spatial_windows_too_small": 0,
        "trajectory_infeasible_windows": 0,
    }
    if finite_indices.size == 0:
        return [], skipped

    finite_distances = distances[finite_indices].astype(np.float64, copy=False)
    low = float(finite_distances.min())
    high = float(finite_distances.max())
    if high - low <= options.spatial_window_m:
        return [finite_indices], skipped

    step = options.spatial_window_m * (1.0 - options.spatial_overlap)
    bucket = np.floor((finite_distances - low) / step)
    starts = list(np.unique(low + bucket * step))
    final_start = max(low, high - options.spatial_window_m)
    starts.append(final_start)
    windows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for start in sorted(starts):
        indices = finite_indices[
            (finite_distances >= start)
            & (finite_distances <= start + options.spatial_window_m)
        ]
        key = tuple(int(index) for index in indices)
        if not key or key in seen:
            continue
        seen.add(key)
        windows.append(indices)
    return windows, skipped


def _invalid_channel_mask(data: np.ndarray) -> np.ndarray:
    finite = np.isfinite(data).all(axis=0)
    flat = np.ones(data.shape[1], dtype=bool)
    for channel in np.flatnonzero(finite):
        flat[channel] = np.ptp(data[:, channel]) == 0.0
    return (~finite | flat).astype(bool, copy=False)


def _normalize_block_scale(block: DasBlock) -> tuple[DasBlock, float]:
    """Power-of-two scale normalization preserves ratios across physical units."""
    maximum = 0.0
    chunk_rows = 4096
    for start in range(0, block.data.shape[0], chunk_rows):
        chunk = block.data[start : start + chunk_rows]
        finite = np.isfinite(chunk)
        if finite.any():
            maximum = max(
                maximum,
                float(np.max(np.abs(chunk), where=finite, initial=0.0)),
            )
    if maximum == 0.0:
        return block, 1.0
    _, exponent = math.frexp(maximum)
    divisor = math.ldexp(1.0, exponent)
    normalized = np.asarray(block.data / divisor, dtype=np.float32)
    return replace(block, data=normalized), divisor


def _as_numpy(value: Any, backend: ArrayBackend) -> np.ndarray:
    return np.asarray(backend.asnumpy(value))


def _preceding_ratio(
    data: Any,
    current_width: int,
    background_width: int,
    backend: ArrayBackend,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare a trailing current window with the immediately preceding background."""
    xp = backend.xp
    samples, channels = data.shape
    total_width = current_width + background_width
    if samples < total_width or channels == 0:
        empty = np.empty((0, channels), dtype=np.float32)
        return empty, np.empty(0, dtype=np.int64), empty, empty
    power = xp.square(data)
    cumulative = xp.concatenate(
        (
            xp.zeros((1, channels), dtype=power.dtype),
            xp.cumsum(power, axis=0, dtype=power.dtype),
        ),
        axis=0,
    )
    current_end = cumulative[total_width:]
    current_start = cumulative[background_width : samples + 1 - current_width]
    background_end = current_start
    background_start = cumulative[: samples + 1 - total_width]
    current = (current_end - current_start) / current_width
    background = (background_end - background_start) / background_width
    epsilon = xp.finfo(data.dtype).eps
    ratio = current / xp.maximum(background, epsilon)
    sample_indices = np.arange(total_width - 1, samples, dtype=np.int64)
    return (
        _as_numpy(ratio, backend),
        sample_indices,
        _as_numpy(current, backend),
        _as_numpy(background, backend),
    )


def _hysteresis_runs(
    on: np.ndarray,
    off: np.ndarray,
    release_gap: int,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last_supported = -1
    for index in range(on.size):
        if start is None:
            if not on[index]:
                continue
            start = index
            last_supported = index
            continue
        if off[index]:
            last_supported = index
        if index - last_supported > release_gap:
            runs.append((start, last_supported + 1))
            start = index if on[index] else None
            last_supported = index if start is not None else -1
    if start is not None:
        runs.append((start, last_supported + 1))
    return runs


def _ratio_score(
    ratio: float,
    normal_ratio: float,
    strong_ratio: float,
    profile: ProcessingProfile,
) -> float:
    if ratio < normal_ratio:
        return 0.0
    if strong_ratio <= normal_ratio:
        return float(profile.fusion_strong_score)
    progress = np.clip(
        (ratio - normal_ratio) / (strong_ratio - normal_ratio),
        0.0,
        1.0,
    )
    return float(
        profile.fusion_normal_score
        + progress * (profile.fusion_strong_score - profile.fusion_normal_score)
    )


def _ratio_candidates(
    *,
    ratios: np.ndarray,
    sample_indices: np.ndarray,
    current_power: np.ndarray,
    background_power: np.ndarray,
    window_channels: np.ndarray,
    good_local: np.ndarray,
    block: PreprocessedBlock,
    profile: ProcessingProfile,
    options: ScreeningOptions,
    detector: str,
    threshold: float,
    strong_threshold: float,
    extra_metrics: dict[str, float],
) -> list[Candidate]:
    if ratios.shape[0] == 0 or int(good_local.sum()) < profile.min_coincidence:
        return []
    ratios = np.where(np.isfinite(ratios), ratios, 0.0)
    ratios[:, ~good_local] = 0.0
    tolerance = round(options.coincidence_tolerance_seconds * block.sample_rate_hz)
    tolerant_ratios = (
        maximum_filter1d(
            ratios,
            size=2 * tolerance + 1,
            axis=0,
            mode="constant",
            cval=0.0,
        )
        if tolerance
        else ratios
    )
    on_hits = tolerant_ratios >= threshold
    off_hits = tolerant_ratios >= options.off_ratio
    on_support = on_hits.sum(axis=1)
    off_support = off_hits.sum(axis=1)
    on = on_support >= profile.min_coincidence
    off = off_support >= profile.min_coincidence
    release = round(options.release_gap_seconds * block.sample_rate_hz)
    minimum = max(1, round(profile.min_duration_seconds * block.sample_rate_hz))
    valid_count = int(good_local.sum())
    option_fingerprint = _option_fingerprint(options)

    robust = np.zeros(ratios.shape[0], dtype=np.float64)
    for index in np.flatnonzero(on):
        values = tolerant_ratios[index, on_hits[index] & good_local]
        robust[index] = float(np.median(values)) if values.size else 0.0

    candidates: list[Candidate] = []
    for start, stop in _hysteresis_runs(on, off, release):
        active_indices = np.flatnonzero(on[start:stop]) + start
        if active_indices.size == 0:
            continue
        first_active = int(active_indices[0])
        last_active = int(active_indices[-1])
        last_supported = stop - 1
        if sample_indices[last_supported] - sample_indices[first_active] + 1 < minimum:
            continue
        peak_index = int(active_indices[np.argmax(robust[active_indices])])
        channel_hits = np.any(on_hits[first_active : last_active + 1], axis=0) & good_local
        local_channels = np.flatnonzero(channel_hits)
        if local_channels.size < profile.min_coincidence:
            continue
        original_channels = window_channels[local_channels]
        peak_support = int(on_support[peak_index])
        peak_ratio = float(robust[peak_index])
        metrics = {
            "sample_rate_hz": float(block.sample_rate_hz),
            "robust_peak_ratio": peak_ratio,
            "peak_support_channels": float(peak_support),
            "local_valid_channels": float(valid_count),
            "local_support_fraction": float(peak_support / valid_count),
            "current_power": float(np.median(current_power[peak_index, channel_hits])),
            "background_power": float(np.median(background_power[peak_index, channel_hits])),
            "normal_ratio_threshold": float(threshold),
            "strong_ratio_threshold": float(strong_threshold),
            "options_fingerprint": option_fingerprint,
            **extra_metrics,
        }
        candidates.append(
            Candidate(
                detector=detector,
                start_sample=block.sample_offset + int(sample_indices[first_active]),
                end_sample=block.sample_offset + int(sample_indices[last_supported]) + 1,
                peak_sample=block.sample_offset + int(sample_indices[peak_index]),
                channel_start=int(original_channels.min()),
                channel_stop=int(original_channels.max()) + 1,
                score=_ratio_score(peak_ratio, threshold, strong_threshold, profile),
                metrics=metrics,
            )
        )
    return candidates


def _contains(outer: Candidate, inner: Candidate) -> bool:
    return (
        outer.start_sample <= inner.start_sample
        and outer.end_sample >= inner.end_sample
        and outer.channel_start <= inner.channel_start
        and outer.channel_stop >= inner.channel_stop
    )


def _is_duplicate(candidate: Candidate, selected: Candidate, peak_tolerance: int) -> bool:
    close_peak = abs(candidate.peak_sample - selected.peak_sample) <= peak_tolerance
    return close_peak and _contains(selected, candidate)


def _suppress_duplicates(
    candidates: Iterable[Candidate],
    peak_tolerance: int,
) -> list[Candidate]:
    selected: list[Candidate] = []
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.peak_sample,
            candidate.channel_start,
            candidate.channel_stop,
            candidate.detector,
        ),
    )
    for candidate in ordered:
        if any(
            prior.detector == candidate.detector
            and _is_duplicate(candidate, prior, peak_tolerance)
            for prior in selected
        ):
            continue
        selected.append(candidate)
    return sorted(selected, key=_candidate_key)


def _candidate_key(candidate: Candidate) -> tuple[object, ...]:
    return (
        candidate.start_sample,
        candidate.end_sample,
        candidate.channel_start,
        candidate.channel_stop,
        candidate.detector,
        candidate.peak_sample,
        candidate.score,
        tuple(sorted(candidate.metrics.items())),
        candidate.quality_flags,
    )


def _detect_stalta_scales(
    block: PreprocessedBlock,
    windows: list[np.ndarray],
    profile: ProcessingProfile,
    options: ScreeningOptions,
    backend: ArrayBackend,
) -> list[Candidate]:
    scales = (
        ("short", options.short_sta_seconds, options.short_lta_seconds),
        ("long", profile.sta_seconds, profile.lta_seconds),
    )
    candidates: list[Candidate] = []
    for scale_name, sta_seconds, lta_seconds in scales:
        sta = max(1, round(sta_seconds * block.sample_rate_hz))
        lta = max(sta + 1, round(lta_seconds * block.sample_rate_hz))
        for window_channels in windows:
            good_local = ~block.bad_channel_mask[window_channels]
            if int(good_local.sum()) < profile.min_coincidence:
                continue
            ratios, indices, current, background = _preceding_ratio(
                block.data[:, window_channels], sta, lta, backend
            )
            candidates.extend(
                _ratio_candidates(
                    ratios=ratios,
                    sample_indices=indices,
                    current_power=current,
                    background_power=background,
                    window_channels=window_channels,
                    good_local=good_local,
                    block=block,
                    profile=profile,
                    options=options,
                    detector=f"stalta:{scale_name}",
                    threshold=profile.stalta_threshold,
                    strong_threshold=profile.stalta_strong_ratio,
                    extra_metrics={
                        "sta_seconds": float(sta_seconds),
                        "lta_seconds": float(lta_seconds),
                    },
                )
            )
    tolerance = round(options.coincidence_tolerance_seconds * block.sample_rate_hz)
    return _suppress_duplicates(candidates, tolerance)


def _detect_energy_bands(
    prepared: Any,
    windows: list[np.ndarray],
    profile: ProcessingProfile,
    options: ScreeningOptions,
    backend: ArrayBackend,
) -> tuple[list[Candidate], list[int], list[tuple[float, float]]]:
    candidates: list[Candidate] = []
    skipped_bands: list[int] = []
    effective_bands: list[tuple[float, float]] = []
    current_width = max(1, round(profile.energy_window_seconds * prepared.sample_rate_hz))
    background_width = max(
        current_width,
        round(options.short_lta_seconds * prepared.sample_rate_hz),
    )
    for band_index, (low_hz, requested_high_hz) in enumerate(profile.energy_bands_hz):
        band = _effective_band(low_hz, requested_high_hz, prepared.sample_rate_hz)
        if band is None:
            skipped_bands.append(band_index)
            continue
        low_hz, high_hz = band
        effective_bands.append(band)
        band_block = filter_prepared_signal(prepared, low_hz, high_hz, backend)
        for window_channels in windows:
            good_local = ~band_block.bad_channel_mask[window_channels]
            if int(good_local.sum()) < profile.min_coincidence:
                continue
            ratios, indices, current, background = _preceding_ratio(
                band_block.data[:, window_channels],
                current_width,
                background_width,
                backend,
            )
            candidates.extend(
                _ratio_candidates(
                    ratios=ratios,
                    sample_indices=indices,
                    current_power=current,
                    background_power=background,
                    window_channels=window_channels,
                    good_local=good_local,
                    block=band_block,
                    profile=profile,
                    options=options,
                    detector=f"energy:band-{band_index}",
                    threshold=profile.energy_ratio_threshold,
                    strong_threshold=2.0 * profile.energy_ratio_threshold,
                    extra_metrics={
                        "band_index": float(band_index),
                        "band_low_hz": float(low_hz),
                        "band_high_hz": float(high_hz),
                        "energy_window_seconds": float(profile.energy_window_seconds),
                    },
                )
            )
    tolerance = round(options.coincidence_tolerance_seconds * prepared.sample_rate_hz)
    return _suppress_duplicates(candidates, tolerance), skipped_bands, effective_bands


def _sample_channels(indices: np.ndarray, maximum: int) -> np.ndarray:
    if indices.size <= maximum:
        return indices
    positions = np.linspace(0, indices.size - 1, maximum)
    return indices[np.unique(np.rint(positions).astype(int))]


def _coherence_candidates(
    block: PreprocessedBlock,
    windows: list[np.ndarray],
    profile: ProcessingProfile,
    options: ScreeningOptions,
    backend: ArrayBackend,
) -> tuple[list[Candidate], int]:
    xp = backend.xp
    candidates: list[Candidate] = []
    skipped = 0
    decimation = max(
        1,
        math.floor(block.sample_rate_hz / (4.0 * profile.coherence_freq_high_hz)),
    )
    effective_rate = block.sample_rate_hz / decimation
    width = max(1, round(profile.coherence_window_seconds * effective_rate))
    option_fingerprint = _option_fingerprint(options)
    slownesses = np.linspace(
        -profile.slowness_max_s_per_m,
        profile.slowness_max_s_per_m,
        profile.slowness_points,
        dtype=np.float64,
    )
    slownesses[profile.slowness_points // 2] = 0.0
    zero_tolerance = profile.slowness_max_s_per_m / max(1, profile.slowness_points - 1) / 2.0

    for window_channels in windows:
        good = window_channels[~block.bad_channel_mask[window_channels]]
        if good.size < profile.min_coincidence:
            skipped += 1
            continue
        channels = _sample_channels(good, _MAX_COHERENCE_CHANNELS)
        data = block.data[::decimation, channels]
        if data.shape[0] < width:
            skipped += 1
            continue
        distances = block.channel_distance_m[channels].astype(np.float64, copy=False)
        distances = distances - distances.min()
        best_curve = xp.full(data.shape[0] - width + 1, -1.0, dtype=data.dtype)
        best_slowness = xp.zeros_like(best_curve)
        for slowness in slownesses:
            delays = xp.rint(
                backend.asarray(slowness * distances * effective_rate)
            ).astype(xp.int64)
            curve, _ = _moveout_semblance(data, delays, width, xp)
            replace = curve > best_curve
            tied_better = (curve == best_curve) & (
                abs(float(slowness)) < xp.abs(best_slowness)
            )
            choose = replace | tied_better
            best_curve = xp.where(choose, curve, best_curve)
            best_slowness = xp.where(choose, float(slowness), best_slowness)

        curve_cpu = _as_numpy(best_curve, backend)
        slowness_cpu = _as_numpy(best_slowness, backend)
        peaks, properties = find_peaks(
            curve_cpu,
            height=profile.coherence_threshold,
            distance=max(1, width // 2),
        )
        if peaks.size == 0 and curve_cpu.size:
            maximum = int(np.argmax(curve_cpu))
            if curve_cpu[maximum] >= profile.coherence_threshold:
                peaks = np.array([maximum], dtype=int)
                properties = {"peak_heights": np.array([curve_cpu[maximum]])}

        for peak_index, semblance in zip(
            peaks,
            properties.get("peak_heights", curve_cpu[peaks]),
            strict=True,
        ):
            selected_slowness = float(slowness_cpu[peak_index])
            delays = xp.rint(
                backend.asarray(selected_slowness * distances * effective_rate)
            ).astype(xp.int64)
            _, stack = _moveout_semblance(data, delays, width, xp)
            search = stack[peak_index : peak_index + width]
            relative_peak = int(_as_numpy(xp.argmax(xp.abs(search)), backend))
            peak_decimated = int(peak_index) + relative_peak
            peak_sample = min(block.data.shape[0] - 1, peak_decimated * decimation)
            start_sample = min(block.data.shape[0] - 1, int(peak_index) * decimation)
            end_sample = min(
                block.data.shape[0],
                (int(peak_index) + width) * decimation,
            )
            near_zero = abs(selected_slowness) <= zero_tolerance
            metrics = {
                "sample_rate_hz": float(block.sample_rate_hz),
                "semblance": float(semblance),
                "slowness_s_per_m": selected_slowness,
                "local_valid_channels": float(good.size),
                "coherence_channels_used": float(channels.size),
                "temporal_decimation": float(decimation),
                "options_fingerprint": option_fingerprint,
            }
            if not near_zero:
                metrics["apparent_velocity_mps"] = 1.0 / abs(selected_slowness)
                metrics["direction"] = float(np.sign(selected_slowness))
            candidates.append(
                Candidate(
                    detector="coherence",
                    start_sample=block.sample_offset + start_sample,
                    end_sample=block.sample_offset + end_sample,
                    peak_sample=block.sample_offset + peak_sample,
                    channel_start=int(channels.min()),
                    channel_stop=int(channels.max()) + 1,
                    score=_ratio_score(
                        float(semblance),
                        profile.coherence_threshold,
                        2.0 * profile.coherence_threshold,
                        profile,
                    ),
                    metrics=metrics,
                    quality_flags=("near_zero_moveout_ambiguous",) if near_zero else (),
                )
            )
    tolerance = round(options.coincidence_tolerance_seconds * block.sample_rate_hz)
    return _suppress_duplicates(candidates, tolerance), skipped


def _trajectory_velocity_bounds(
    distances: np.ndarray,
    sample_count: int,
    sample_rate_hz: float,
    profile: ProcessingProfile,
) -> tuple[float, float] | None:
    if distances.size < 2 or sample_count < 2:
        return None
    span = float(np.ptp(distances))
    if not math.isfinite(span) or span <= 0:
        return None
    time_step = max(
        1,
        math.ceil(sample_rate_hz / (2.0 * profile.trajectory_freq_high_hz)),
    )
    minimum_velocity = max(
        profile.trajectory_velocity_min_mps,
        span * sample_rate_hz / (sample_count - 1),
    )
    maximum_velocity = min(
        profile.trajectory_velocity_max_mps,
        span * sample_rate_hz / time_step,
    )
    if minimum_velocity >= maximum_velocity:
        return None
    return minimum_velocity, maximum_velocity


def _trajectory_candidates(
    block: PreprocessedBlock,
    windows: list[np.ndarray],
    profile: ProcessingProfile,
    options: ScreeningOptions,
) -> tuple[list[Candidate], int]:
    candidates: list[Candidate] = []
    skipped = 0
    option_fingerprint = _option_fingerprint(options)
    for window_channels in windows:
        good_count = int((~block.bad_channel_mask[window_channels]).sum())
        if good_count < profile.min_coincidence:
            skipped += 1
            continue
        bounds = _trajectory_velocity_bounds(
            block.channel_distance_m[window_channels],
            block.data.shape[0],
            block.sample_rate_hz,
            profile,
        )
        if bounds is None:
            skipped += 1
            continue
        local_profile = profile.model_copy(
            update={
                "trajectory_velocity_min_mps": bounds[0],
                "trajectory_velocity_max_mps": bounds[1],
            }
        )
        local_block = PreprocessedBlock(
            data=block.data[:, window_channels],
            bad_channel_mask=block.bad_channel_mask[window_channels].copy(),
            backend_name=block.backend_name,
            sample_offset=block.sample_offset,
            sample_rate_hz=block.sample_rate_hz,
            channel_distance_m=block.channel_distance_m[window_channels].copy(),
        )
        for candidate in detect_trajectories(local_block, local_profile):
            start_local = min(candidate.channel_start, window_channels.size - 1)
            stop_local = min(max(candidate.channel_stop - 1, 0), window_channels.size - 1)
            metrics = dict(candidate.metrics)
            metrics["sample_rate_hz"] = float(block.sample_rate_hz)
            metrics["local_valid_channels"] = float(good_count)
            metrics["options_fingerprint"] = option_fingerprint
            candidates.append(
                Candidate(
                    detector="trajectory",
                    start_sample=candidate.start_sample,
                    end_sample=candidate.end_sample,
                    peak_sample=candidate.peak_sample,
                    channel_start=int(window_channels[start_local]),
                    channel_stop=int(window_channels[stop_local]) + 1,
                    score=candidate.score,
                    metrics=metrics,
                    quality_flags=candidate.quality_flags,
                )
            )
    tolerance = round(options.coincidence_tolerance_seconds * block.sample_rate_hz)
    return _suppress_duplicates(candidates, tolerance), skipped


def _screening_family(candidate: Candidate) -> str:
    base = candidate.detector.split(":", 1)[0]
    if base in {"stalta", "energy"}:
        return "amplitude"
    return base


def _same_event(left: Candidate, right: Candidate, merge_gap: int) -> bool:
    temporal = max(left.start_sample, right.start_sample) <= min(
        left.end_sample, right.end_sample
    ) + merge_gap
    spatial = max(left.channel_start, right.channel_start) < min(
        left.channel_stop, right.channel_stop
    )
    return temporal and spatial


def _passes_fusion_gate(
    evidence: tuple[Candidate, ...],
    profile: ProcessingProfile,
) -> bool:
    if any(candidate.score >= profile.fusion_strong_score for candidate in evidence):
        return True
    families: set[str] = set()
    for candidate in evidence:
        if candidate.score < profile.fusion_normal_score:
            continue
        families.add(_screening_family(candidate))
    return len(families) >= 2


def _event_from_evidence(
    evidence: tuple[Candidate, ...],
) -> EventRecord:
    ordered = tuple(sorted(evidence, key=_candidate_key))
    peak_candidate = min(
        ordered,
        key=lambda candidate: (-candidate.score, candidate.peak_sample, _candidate_key(candidate)),
    )
    payload = [
        {
            "detector": candidate.detector,
            "start": candidate.start_sample,
            "end": candidate.end_sample,
            "peak": candidate.peak_sample,
            "channel_start": candidate.channel_start,
            "channel_stop": candidate.channel_stop,
            "score": candidate.score,
            "metrics": sorted(candidate.metrics.items()),
            "quality_flags": candidate.quality_flags,
        }
        for candidate in ordered
    ]
    event_id = str(
        uuid5(
            NAMESPACE_URL,
            f"{ALGORITHM_VERSION}|"
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
    )
    return EventRecord(
        event_id=event_id,
        start_sample=min(candidate.start_sample for candidate in ordered),
        end_sample=max(candidate.end_sample for candidate in ordered),
        peak_sample=peak_candidate.peak_sample,
        channel_start=min(candidate.channel_start for candidate in ordered),
        channel_stop=max(candidate.channel_stop for candidate in ordered),
        evidence=ordered,
    )


def fuse_screening_candidates(
    candidates: Iterable[Candidate],
    profile: ProcessingProfile,
) -> tuple[EventRecord, ...]:
    """Fuse v2 evidence while keeping STA/LTA and energy in one amplitude family."""
    remaining = sorted(candidates, key=_candidate_key)
    sample_rates = {
        float(candidate.metrics["sample_rate_hz"])
        for candidate in remaining
        if "sample_rate_hz" in candidate.metrics
        and math.isfinite(float(candidate.metrics["sample_rate_hz"]))
        and float(candidate.metrics["sample_rate_hz"]) > 0
    }
    sample_rate_hz = max(sample_rates, default=0.0)
    merge_gap = round(profile.merge_gap_seconds * sample_rate_hz)
    groups: list[tuple[Candidate, ...]] = []
    while remaining:
        group = [remaining.pop(0)]
        index = 0
        while index < len(remaining):
            candidate = remaining[index]
            if all(_same_event(candidate, member, merge_gap) for member in group):
                group.append(remaining.pop(index))
                index = 0
            else:
                index += 1
        groups.append(tuple(group))
    events = [_event_from_evidence(group) for group in groups if _passes_fusion_gate(group, profile)]
    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.start_sample,
                event.end_sample,
                event.channel_start,
                event.channel_stop,
                event.event_id,
            ),
        )
    )


def screen_block(
    block: DasBlock,
    profile: ProcessingProfile,
    backend: ArrayBackend,
    options: ScreeningOptions | None = None,
) -> ScreeningResult:
    """Run the opt-in v2 screen without changing legacy detector behavior."""
    options = options or ScreeningOptions()
    _validate_options(options, profile)
    _validate_block(block, backend)
    sample_rate_hz = block.metadata.sample_rate_hz
    short_cold_samples = round(
        (options.short_sta_seconds + options.short_lta_seconds) * sample_rate_hz
    )
    full_cold_samples = round((profile.sta_seconds + profile.lta_seconds) * sample_rate_hz)
    sample_count = block.data.shape[0]
    coverage_samples = max(0, sample_count - short_cold_samples)
    diagnostics: dict[str, Any] = {
        "algorithm_version": ALGORITHM_VERSION,
        "options": asdict(options),
        "score_semantics": "threshold-normalized detector evidence; not a probability",
        "cold_start_seconds": short_cold_samples / sample_rate_hz,
        "full_lta_cold_start_seconds": full_cold_samples / sample_rate_hz,
        "effective_sample_coverage": {
            "start_sample": block.sample_offset + min(sample_count, short_cold_samples),
            "end_sample": block.sample_offset + sample_count,
            "fully_conditioned_start_sample": block.sample_offset
            + min(sample_count, full_cold_samples),
            "covered_samples": coverage_samples,
            "fraction": coverage_samples / sample_count if sample_count else 0.0,
        },
        "skipped_geometry": {},
        "detectors": {},
    }
    if sample_count < 2 or block.data.shape[1] == 0:
        diagnostics["skipped_geometry"] = {
            "nonfinite_channels": 0,
            "spatial_windows_too_small": 0,
            "trajectory_infeasible_windows": 0,
        }
        return ScreeningResult(candidates=(), events=(), diagnostics=diagnostics)

    screening_profile = profile.model_copy(
        update={"common_mode_removal": options.common_mode_removal}
    )
    preparation_started = perf_counter()
    normalized_block, input_scale_divisor = _normalize_block_scale(block)
    prepared = prepare_signal(normalized_block, screening_profile, backend)
    rms_warning_count = int(prepared.bad_channel_mask.sum())
    invalid_mask = _invalid_channel_mask(normalized_block.data)
    prepared = replace(prepared, bad_channel_mask=invalid_mask)
    diagnostics["preprocessing_seconds"] = perf_counter() - preparation_started
    diagnostics["channel_quality"] = {
        "rms_warning_channels": rms_warning_count,
        "invalid_excluded_channels": int(invalid_mask.sum()),
        "input_scale_divisor": input_scale_divisor,
    }
    windows, skipped_geometry = _physical_windows(
        prepared.channel_distance_m,
        options,
    )
    usable_windows: list[np.ndarray] = []
    for window in windows:
        if int((~prepared.bad_channel_mask[window]).sum()) < profile.min_coincidence:
            skipped_geometry["spatial_windows_too_small"] += 1
        else:
            usable_windows.append(window)
    diagnostics["spatial_window_count"] = len(usable_windows)

    all_candidates: list[Candidate] = []
    stalta_started = perf_counter()
    stalta: list[Candidate] = []
    stalta_band = _effective_band(
        profile.stalta_freq_low_hz,
        profile.stalta_freq_high_hz,
        sample_rate_hz,
    )
    if stalta_band is not None:
        stalta_block = filter_prepared_signal(prepared, *stalta_band, backend)
        stalta = _detect_stalta_scales(
            stalta_block,
            usable_windows,
            profile,
            options,
            backend,
        )
    all_candidates.extend(stalta)
    diagnostics["detectors"]["stalta"] = {
        "seconds": perf_counter() - stalta_started,
        "count": len(stalta),
        "short_count": sum(candidate.detector == "stalta:short" for candidate in stalta),
        "long_count": sum(candidate.detector == "stalta:long" for candidate in stalta),
        "status": "processed" if stalta_band is not None else "skipped",
        "effective_band_hz": stalta_band,
    }
    if stalta_band is None:
        diagnostics["detectors"]["stalta"]["reason"] = "frequency_band_unavailable"

    energy_started = perf_counter()
    energy, energy_skipped_bands, energy_effective_bands = _detect_energy_bands(
        prepared,
        usable_windows,
        profile,
        options,
        backend,
    )
    all_candidates.extend(energy)
    diagnostics["detectors"]["energy"] = {
        "seconds": perf_counter() - energy_started,
        "count": len(energy),
        "status": "processed" if energy_effective_bands else "skipped",
        "processed_bands": len(energy_effective_bands),
        "effective_bands_hz": energy_effective_bands,
        "skipped_bands": energy_skipped_bands,
    }
    if not energy_effective_bands:
        diagnostics["detectors"]["energy"]["reason"] = "frequency_bands_unavailable"

    coherence: list[Candidate] = []
    coherence_started = perf_counter()
    coherence_skipped = 0
    coherence_band = _effective_band(
        profile.coherence_freq_low_hz,
        profile.coherence_freq_high_hz,
        sample_rate_hz,
    )
    if options.enable_coherence and coherence_band is not None:
        coherence_block = filter_prepared_signal(prepared, *coherence_band, backend)
        coherence, coherence_skipped = _coherence_candidates(
            coherence_block,
            usable_windows,
            profile,
            options,
            backend,
        )
        all_candidates.extend(coherence)
    coherence_status = (
        "disabled"
        if not options.enable_coherence
        else "processed"
        if coherence_band is not None
        else "skipped"
    )
    diagnostics["detectors"]["coherence"] = {
        "seconds": perf_counter() - coherence_started,
        "count": len(coherence),
        "skipped_windows": coherence_skipped,
        "enabled": options.enable_coherence,
        "status": coherence_status,
        "effective_band_hz": coherence_band,
    }
    if coherence_status == "skipped":
        diagnostics["detectors"]["coherence"]["reason"] = (
            "frequency_band_unavailable"
        )

    trajectories: list[Candidate] = []
    trajectory_started = perf_counter()
    trajectory_skipped = 0
    trajectory_band = _effective_band(
        profile.trajectory_freq_low_hz,
        profile.trajectory_freq_high_hz,
        sample_rate_hz,
    )
    if options.enable_trajectory and trajectory_band is not None:
        trajectory_block = filter_prepared_signal(prepared, *trajectory_band, backend)
        if trajectory_block.backend_name == "cuda":
            trajectory_block = replace(
                trajectory_block,
                data=backend.asnumpy(trajectory_block.data),
                backend_name="cpu",
            )
        trajectories, trajectory_skipped = _trajectory_candidates(
            trajectory_block,
            usable_windows,
            profile,
            options,
        )
        all_candidates.extend(trajectories)
    trajectory_status = (
        "disabled"
        if not options.enable_trajectory
        else "processed"
        if trajectory_band is not None
        else "skipped"
    )
    skipped_geometry["trajectory_infeasible_windows"] = trajectory_skipped
    diagnostics["skipped_geometry"] = skipped_geometry
    diagnostics["detectors"]["trajectory"] = {
        "seconds": perf_counter() - trajectory_started,
        "count": len(trajectories),
        "skipped_windows": trajectory_skipped,
        "enabled": options.enable_trajectory,
        "status": trajectory_status,
        "effective_band_hz": trajectory_band,
    }
    if trajectory_status == "skipped":
        diagnostics["detectors"]["trajectory"]["reason"] = (
            "frequency_band_unavailable"
        )

    ordered_candidates = tuple(sorted(all_candidates, key=_candidate_key))
    fusion_started = perf_counter()
    events = fuse_screening_candidates(ordered_candidates, profile)
    diagnostics["fusion_seconds"] = perf_counter() - fusion_started
    diagnostics["candidate_count"] = len(ordered_candidates)
    diagnostics["event_count"] = len(events)
    return ScreeningResult(
        candidates=ordered_candidates,
        events=events,
        diagnostics=diagnostics,
    )
