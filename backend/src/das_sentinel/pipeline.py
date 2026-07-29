from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from das_sentinel.artifacts import (
    ArtifactManifest,
    EventArtifactData,
    render_event_artifacts,
)
from das_sentinel.classifiers.fusion import fuse_classification
from das_sentinel.classifiers.rules import RuleResult, score_rules
from das_sentinel.config import ProcessingProfile
from das_sentinel.detectors.coherence import detect_coherence
from das_sentinel.detectors.energy import detect_energy
from das_sentinel.detectors.stalta import detect_stalta
from das_sentinel.detectors.trajectory import detect_trajectories
from das_sentinel.domain import Candidate, DasBlock, EventRecord
from das_sentinel.features import extract_features
from das_sentinel.fusion import fuse_candidates
from das_sentinel.readers.base import DasReader
from das_sentinel.signal.backend import ArrayBackend
from das_sentinel.signal.preprocess import PreprocessedBlock, preprocess_block

_TIMING_NAMES = ("read", "preprocess", "detect", "fuse", "classify", "render")


class SourceChangedError(RuntimeError):
    """Raised when source identity changes during one analysis run."""


@dataclass(frozen=True)
class AnalysisContext:
    site_id: str = "unknown-site"
    site_timezone: str = "UTC"
    profile_version: str = "profile-v1"
    pipeline_version: str = "pipeline-v1"
    cable_type: str = "unknown"
    device_config: str = "unknown"
    rule_profile: Mapping[str, Any] = field(default_factory=dict)
    rule_profile_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.site_id,
            self.site_timezone,
            self.profile_version,
            self.pipeline_version,
            self.cable_type,
            self.device_config,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("analysis context string fields must be non-empty")
        try:
            ZoneInfo(self.site_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("site_timezone must be a valid IANA timezone") from error
        if not isinstance(self.rule_profile, Mapping):
            raise TypeError("rule_profile must be a mapping")
        try:
            canonical = json.dumps(
                dict(self.rule_profile),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            snapshot = json.loads(canonical)
        except (TypeError, ValueError) as error:
            raise ValueError("rule_profile must be canonical JSON data") from error
        object.__setattr__(self, "rule_profile", MappingProxyType(snapshot))
        object.__setattr__(
            self,
            "rule_profile_sha256",
            hashlib.sha256(canonical.encode()).hexdigest(),
        )


@dataclass(frozen=True)
class FileAnalysis:
    source_path: str
    source_fingerprint: str
    actual_backend: str
    backend_fallback_reason: str | None
    metadata_warnings: tuple[str, ...]
    events: tuple[EventRecord, ...]
    audit_candidate_count: int
    timings_ms: dict[str, float] = field(compare=False)
    artifacts: tuple[ArtifactManifest, ...]


def analyze_file(
    path: str | Path,
    reader: DasReader,
    profile: ProcessingProfile,
    backend: ArrayBackend,
    artifact_root: str | Path,
    context: AnalysisContext | None = None,
) -> FileAnalysis:
    """Analyze one DAS file in overlapping blocks and render review artifacts."""
    source = Path(path)
    run_context = context or AnalysisContext()
    started = time.perf_counter()
    timings = {name: 0.0 for name in _TIMING_NAMES}

    read_started = time.perf_counter()
    source_fingerprint = _source_fingerprint(source)
    metadata = reader.read_metadata(source)
    sample_count = reader.sample_count(source)
    timings["read"] += _elapsed_ms(read_started)
    _validate_metadata(profile, metadata.sample_rate_hz)
    if sample_count <= 0:
        raise ValueError("DAS source must contain at least one sample")

    candidates: list[Candidate] = []
    block_samples = max(1, round(profile.block_seconds * metadata.sample_rate_hz))
    overlap_samples = round(profile.overlap_seconds * metadata.sample_rate_hz)
    step_samples = block_samples - overlap_samples
    if step_samples <= 0:
        raise ValueError("block overlap leaves no forward progress")

    for sample_start in range(0, sample_count, step_samples):
        sample_stop = min(sample_count, sample_start + block_samples)
        read_started = time.perf_counter()
        block = reader.read_block(source, sample_start, sample_stop)
        timings["read"] += _elapsed_ms(read_started)
        candidates.extend(_detect_block(block, profile, backend, timings))
        if sample_stop == sample_count:
            break

    _ensure_source_unchanged(source, source_fingerprint)
    candidates = _deduplicate_candidates(
        candidates, profile, metadata.sample_rate_hz
    )
    fuse_started = time.perf_counter()
    review_events = fuse_candidates(
        candidates,
        profile,
        metadata.sample_rate_hz,
        run_context.site_id,
        run_context.pipeline_version,
        source_fingerprint=source_fingerprint,
    )
    timings["fuse"] += _elapsed_ms(fuse_started)

    events: list[EventRecord] = []
    artifacts: list[ArtifactManifest] = []
    profile_payload = {
        **profile.model_dump(mode="json"),
        "rule_profile": dict(run_context.rule_profile),
        "profile_version": run_context.profile_version,
        "rule_profile_sha256": run_context.rule_profile_sha256,
    }
    for event in review_events:
        window_start, window_stop = _event_window(
            event,
            sample_count,
            profile.artifact_pad_seconds,
            metadata.sample_rate_hz,
        )
        read_started = time.perf_counter()
        raw_block = reader.read_block(source, window_start, window_stop)
        timings["read"] += _elapsed_ms(read_started)

        primary_detector = _primary_detector(event)
        preprocess_started = time.perf_counter()
        filtered_block = _preprocess_for_detector(
            raw_block, profile, backend, primary_detector
        )
        filtered_data = np.asarray(
            backend.asnumpy(filtered_block.data), dtype=np.float32
        )
        feature_block = replace(
            filtered_block, data=filtered_data, backend_name="cpu"
        )
        timings["preprocess"] += _elapsed_ms(preprocess_started)

        classify_started = time.perf_counter()
        event_start_utc = raw_block.metadata.start_time_utc + timedelta(
            seconds=(event.start_sample - raw_block.sample_offset)
            / metadata.sample_rate_hz
        )
        features = extract_features(
            feature_block,
            event,
            context={
                "site_id": run_context.site_id,
                "cable_type": run_context.cable_type,
                "device_config": run_context.device_config,
                "start_time_utc": event_start_utc,
            },
        )
        rule_result = score_rules(features, run_context.rule_profile)
        quality_flags = tuple(
            sorted(
                {
                    flag
                    for evidence in event.evidence
                    for flag in evidence.quality_flags
                }
            )
        )
        decision = fuse_classification(rule_result.scores, None, quality_flags)
        classified = replace(
            event,
            features=dict(features),
            label=decision.label,
            decision_score=decision.decision_score,
            reasons=decision.reasons,
        )
        timings["classify"] += _elapsed_ms(classify_started)
        events.append(classified)

        render_started = time.perf_counter()
        artifact_data = EventArtifactData(
            raw_data=raw_block.data,
            filtered_data=filtered_data,
            sample_offset=raw_block.sample_offset,
            sample_rate_hz=raw_block.metadata.sample_rate_hz,
            channel_distance_m=raw_block.metadata.channel_distance_m,
            bad_channel_mask=filtered_block.bad_channel_mask,
            start_time_utc=raw_block.metadata.start_time_utc,
            site_timezone=run_context.site_timezone,
            profile=profile_payload,
            profile_version=run_context.profile_version,
            pipeline_version=run_context.pipeline_version,
            model_version="rules-only",
            backend_name=backend.name,
            rule_output=_rule_output(rule_result, run_context),
            model_output=None,
        )
        _ensure_source_unchanged(source, source_fingerprint)
        artifacts.append(
            render_event_artifacts(
                artifact_data, classified, features, Path(artifact_root)
            )
        )
        _ensure_source_unchanged(source, source_fingerprint)
        timings["render"] += _elapsed_ms(render_started)

    _ensure_source_unchanged(source, source_fingerprint)
    timings["total"] = _elapsed_ms(started)
    return FileAnalysis(
        source_path=str(source),
        source_fingerprint=source_fingerprint,
        actual_backend=backend.name,
        backend_fallback_reason=backend.unavailable_reason,
        metadata_warnings=metadata.warnings,
        events=tuple(events),
        audit_candidate_count=len(candidates),
        timings_ms=timings,
        artifacts=tuple(artifacts),
    )


def _detect_block(
    block: DasBlock,
    profile: ProcessingProfile,
    backend: ArrayBackend,
    timings: dict[str, float],
) -> list[Candidate]:
    prepared: dict[tuple[float, float], PreprocessedBlock] = {}

    def prepared_for(detector: str) -> PreprocessedBlock:
        band = _detector_band(profile, detector)
        if band not in prepared:
            preprocess_started = time.perf_counter()
            prepared[band] = _preprocess_for_band(block, profile, backend, band)
            timings["preprocess"] += _elapsed_ms(preprocess_started)
        return prepared[band]

    found: list[Candidate] = []
    prepared_block = prepared_for("stalta")
    detect_started = time.perf_counter()
    found.extend(detect_stalta(prepared_block, profile))
    timings["detect"] += _elapsed_ms(detect_started)

    prepared_block = prepared_for("energy")
    detect_started = time.perf_counter()
    found.extend(detect_energy(prepared_block, profile))
    timings["detect"] += _elapsed_ms(detect_started)

    prepared_block = prepared_for("coherence")
    detect_started = time.perf_counter()
    found.extend(detect_coherence(prepared_block, profile, backend))
    timings["detect"] += _elapsed_ms(detect_started)

    prepared_block = prepared_for("trajectory")
    if prepared_block.backend_name != "cpu":
        prepared_block = replace(
            prepared_block,
            data=np.asarray(backend.asnumpy(prepared_block.data), dtype=np.float32),
            backend_name="cpu",
        )
    detect_started = time.perf_counter()
    found.extend(detect_trajectories(prepared_block, profile))
    timings["detect"] += _elapsed_ms(detect_started)
    return found


def _detector_band(
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


def _preprocess_for_band(
    block: DasBlock,
    profile: ProcessingProfile,
    backend: ArrayBackend,
    band: tuple[float, float],
) -> PreprocessedBlock:
    band_profile = profile.model_copy(
        update={
            "stalta_freq_low_hz": band[0],
            "stalta_freq_high_hz": band[1],
        }
    )
    return preprocess_block(block, band_profile, backend)


def _preprocess_for_detector(
    block: DasBlock,
    profile: ProcessingProfile,
    backend: ArrayBackend,
    detector: str,
) -> PreprocessedBlock:
    return _preprocess_for_band(
        block, profile, backend, _detector_band(profile, detector)
    )


def _primary_detector(event: EventRecord) -> str:
    evidence = min(
        event.evidence,
        key=lambda item: (-item.score, item.peak_sample, item.detector),
    )
    return evidence.detector.split(":", 1)[0]


def _event_window(
    event: EventRecord,
    sample_count: int,
    pad_seconds: float,
    sample_rate_hz: float,
) -> tuple[int, int]:
    padding = max(0, round(pad_seconds * sample_rate_hz))
    start = max(0, event.start_sample - padding)
    stop = min(sample_count, event.end_sample + padding)
    if start >= stop:
        raise ValueError("fused event does not overlap the source file")
    return start, stop


def _rule_output(
    result: RuleResult, context: AnalysisContext
) -> dict[str, Any]:
    return {
        "scores": {label.value: score for label, score in result.scores.items()},
        "reasons": {
            label.value: reasons for label, reasons in result.reasons.items()
        },
        "rule_profile": dict(context.rule_profile),
        "profile_version": context.profile_version,
        "rule_profile_sha256": context.rule_profile_sha256,
    }


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


def _validate_metadata(
    profile: ProcessingProfile, sample_rate_hz: float
) -> None:
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("metadata sample rate must be positive and finite")
    nyquist = sample_rate_hz / 2.0
    bands = (
        ("STA/LTA", profile.stalta_freq_low_hz, profile.stalta_freq_high_hz),
        (
            "coherence",
            profile.coherence_freq_low_hz,
            profile.coherence_freq_high_hz,
        ),
        (
            "trajectory",
            profile.trajectory_freq_low_hz,
            profile.trajectory_freq_high_hz,
        ),
        *(
            (f"energy[{index}]", low, high)
            for index, (low, high) in enumerate(profile.energy_bands_hz)
        ),
    )
    if not profile.energy_bands_hz:
        raise ValueError("at least one energy band is required")
    for name, low, high in bands:
        if (
            not np.isfinite(low)
            or not np.isfinite(high)
            or low <= 0
            or high <= low
            or high > nyquist
        ):
            raise ValueError(
                f"{name} band must satisfy 0 < low < high <= Nyquist ({nyquist:g} Hz)"
            )


def _ensure_source_unchanged(path: Path, expected: str) -> None:
    try:
        current = _source_fingerprint(path)
    except OSError as error:
        raise SourceChangedError("DAS source changed during analysis") from error
    if current != expected:
        raise SourceChangedError("DAS source changed during analysis")


def _source_fingerprint(path: Path) -> str:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    payload = json.dumps(
        (str(resolved), stat.st_size, stat.st_mtime_ns),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1_000
