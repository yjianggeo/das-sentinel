from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO
from uuid import NAMESPACE_URL, UUID, uuid5
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from das_sentinel.artifacts import (
    ArtifactConflictError,
    ArtifactFile,
    ArtifactManifest,
    EventArtifactData,
    _process_lock,
    render_event_artifacts,
)
from das_sentinel.classifiers.fusion import fuse_classification
from das_sentinel.classifiers.rules import RuleResult, score_rules
from das_sentinel.classifiers.supervised import ModelBundle
from das_sentinel.config import PIPELINE_VERSION, ProcessingProfile
from das_sentinel.domain import Candidate, DasBlock, EventRecord
from das_sentinel.features import extract_features
from das_sentinel.measurements import measure_event
from das_sentinel.plot_windows import event_display_window
from das_sentinel.readers.base import DasReader
from das_sentinel.screening import (
    ScreeningOptions,
    fuse_screening_candidates,
    screen_block,
)
from das_sentinel.signal.backend import ArrayBackend
from das_sentinel.signal.preprocess import PreprocessedBlock
from das_sentinel.signal.scan_distance import mask_scan_block, scan_channel_mask
from das_sentinel.waterfalls import (
    WaterfallFilter,
    preprocess_for_waterfall,
    resolve_waterfall_filter,
)

_TIMING_NAMES = ("read", "preprocess", "detect", "fuse", "classify", "measure", "render")


class SourceChangedError(RuntimeError):
    """Raised when source identity changes during one analysis run."""


class InvalidModelError(RuntimeError):
    """Raised when a bound supervised model cannot be safely used."""


@dataclass(frozen=True)
class AnalysisContext:
    site_id: str = "unknown-site"
    site_timezone: str = "UTC"
    processing_profile_id: str = "unknown-profile"
    profile_version: str = "profile-v1"
    pipeline_version: str = PIPELINE_VERSION
    model_version_id: str | None = None
    model_version: str | None = None
    cable_type: str = "unknown"
    device_config: str = "unknown"
    rule_profile: Mapping[str, Any] = field(default_factory=dict)
    rule_profile_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.site_id,
            self.site_timezone,
            self.processing_profile_id,
            self.profile_version,
            self.pipeline_version,
            self.cable_type,
            self.device_config,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("analysis context string fields must be non-empty")
        if (self.model_version_id is None) != (self.model_version is None):
            raise ValueError("model version ID and version must be provided together")
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
        object.__setattr__(self, "rule_profile", _freeze_json(snapshot))
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


@dataclass(frozen=True)
class BoundSource:
    """A source path whose no-follow file handle is held by the caller."""

    path: Path
    stream: BinaryIO
    fingerprint: str
    size_bytes: int
    mtime_ns: int

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def suffix(self) -> str:
        return self.path.suffix

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


def analyze_file(
    path: str | Path | BoundSource,
    reader: DasReader,
    profile: ProcessingProfile,
    backend: ArrayBackend,
    artifact_root: str | Path,
    context: AnalysisContext | None = None,
    model_bundle: ModelBundle | None = None,
) -> FileAnalysis:
    """Analyze one DAS file in overlapping blocks and render review artifacts."""
    reader_source: Path | BinaryIO
    if isinstance(path, BoundSource):
        source: Path | BoundSource = path
        reader_source = path.stream
        source_path = path.path
        source_fingerprint = path.fingerprint
        _ensure_source_unchanged(source, source_fingerprint)
    else:
        source = Path(path)
        reader_source = source
        source_path = source
        source_fingerprint = _source_fingerprint(source)
    run_context = context or AnalysisContext()
    started = time.perf_counter()
    timings = {name: 0.0 for name in _TIMING_NAMES}

    read_started = time.perf_counter()
    metadata = reader.read_metadata(reader_source)
    sample_count = reader.sample_count(reader_source)
    timings["read"] += _elapsed_ms(read_started)
    _validate_metadata(profile, metadata.sample_rate_hz)
    scan_channel_mask(metadata.channel_distance_m, profile)
    if sample_count <= 0:
        raise ValueError("DAS source must contain at least one sample")

    candidates: list[Candidate] = []
    block_samples = max(1, round(profile.block_seconds * metadata.sample_rate_hz))
    overlap_samples = round(profile.overlap_seconds * metadata.sample_rate_hz)
    step_samples = block_samples - overlap_samples
    if step_samples <= 0:
        raise ValueError("block overlap leaves no forward progress")

    options = ScreeningOptions()
    # Condition every preceding-background detector before its canonical window.
    history_samples = 0
    for sta_seconds, lta_seconds in (
        (profile.sta_seconds, profile.lta_seconds),
        (options.short_sta_seconds, options.short_lta_seconds),
    ):
        sta_samples = max(1, round(sta_seconds * metadata.sample_rate_hz))
        lta_samples = max(sta_samples + 1, round(lta_seconds * metadata.sample_rate_hz))
        history_samples = max(history_samples, sta_samples + lta_samples)
    energy_samples = max(1, round(profile.energy_window_seconds * metadata.sample_rate_hz))
    history_samples = max(
        history_samples,
        energy_samples + max(
            energy_samples, round(options.short_lta_seconds * metadata.sample_rate_hz)
        ),
    )
    for logical_start in range(0, sample_count, step_samples):
        sample_start = max(0, logical_start - history_samples)
        sample_stop = min(sample_count, logical_start + block_samples)
        canonical_stop = (
            sample_count
            if sample_stop == sample_count
            else min(sample_count, logical_start + step_samples)
        )
        read_started = time.perf_counter()
        block = reader.read_block(reader_source, sample_start, sample_stop)
        timings["read"] += _elapsed_ms(read_started)
        detect_started = time.perf_counter()
        screening = screen_block(block, profile, backend)
        screening_ms = _elapsed_ms(detect_started)
        preprocess_ms = 1000.0 * screening.diagnostics.get("preprocessing_seconds", 0.0)
        block_fuse_ms = 1000.0 * screening.diagnostics.get("fusion_seconds", 0.0)
        timings["preprocess"] += preprocess_ms
        timings["fuse"] += block_fuse_ms
        timings["detect"] += max(0.0, screening_ms - preprocess_ms - block_fuse_ms)
        candidates.extend(
            candidate
            for candidate in screening.candidates
            if logical_start <= candidate.peak_sample < canonical_stop
        )
        if sample_stop == sample_count:
            break

    _ensure_source_unchanged(source, source_fingerprint)
    fuse_started = time.perf_counter()
    review_events = [
        replace(
            event,
            event_id=str(
                uuid5(
                    NAMESPACE_URL,
                    json.dumps(
                        {
                            "site_id": run_context.site_id,
                            "source_fingerprint": source_fingerprint,
                            "processing_profile_id": run_context.processing_profile_id,
                            "profile_version": run_context.profile_version,
                            "pipeline_version": run_context.pipeline_version,
                            "screening_event_id": event.event_id,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            ),
        )
        for event in fuse_screening_candidates(candidates, profile)
    ]
    timings["fuse"] += _elapsed_ms(fuse_started)

    events: list[EventRecord] = []
    artifacts: list[ArtifactManifest] = []
    transaction = (
        _ArtifactTransaction(Path(artifact_root), source, source_fingerprint)
        if review_events
        else None
    )
    profile_payload = {
        **profile.model_dump(mode="json"),
        "rule_profile": dict(run_context.rule_profile),
        "profile_version": run_context.profile_version,
        "rule_profile_sha256": run_context.rule_profile_sha256,
    }
    try:
        for event in review_events:
            window_start, window_stop = _event_window(
                event,
                sample_count,
                profile.artifact_pad_seconds,
                metadata.sample_rate_hz,
            )
            read_started = time.perf_counter()
            raw_block = reader.read_block(
                reader_source,
                window_start,
                window_stop,
            )
            timings["read"] += _elapsed_ms(read_started)

            waterfall_filter = resolve_waterfall_filter(event.evidence, profile)
            preprocess_started = time.perf_counter()
            filtered_block = _preprocess_for_band(
                raw_block,
                profile,
                backend,
                (waterfall_filter.low_hz, waterfall_filter.high_hz),
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
                    "site_timezone": run_context.site_timezone,
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
            try:
                model_probabilities = (
                    None
                    if model_bundle is None
                    else model_bundle.predict_probabilities(features)
                )
            except Exception as error:
                raise InvalidModelError(
                    "bound model prediction failed"
                ) from error
            decision = fuse_classification(
                rule_result.scores,
                model_probabilities,
                quality_flags,
            )
            model_output = None
            if model_probabilities is not None:
                model_output = {
                    "model_version_id": run_context.model_version_id,
                    "probabilities": {
                        label.value: probability
                        for label, probability in model_probabilities.items()
                    },
                    "fused_scores": {
                        label.value: score
                        for label, score in decision.scores.items()
                    },
                    "reasons": list(decision.reasons),
                }
            classified = replace(
                event,
                features=dict(features),
                label=decision.label,
                decision_score=decision.decision_score,
                reasons=decision.reasons,
            )
            timings["classify"] += _elapsed_ms(classify_started)
            events.append(classified)

            read_started = time.perf_counter()
            measurement_block = reader.read_block(
                reader_source,
                max(0, event.start_sample - round(7.0 * metadata.sample_rate_hz)),
                min(sample_count, event.end_sample + round(2.0 * metadata.sample_rate_hz)),
            )
            timings["read"] += _elapsed_ms(read_started)
            measure_started = time.perf_counter()
            measurements = measure_event(measurement_block, event, profile)
            timings["measure"] += _elapsed_ms(measure_started)
            del measurement_block

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
                model_version=run_context.model_version or "rules-only",
                backend_name=backend.name,
                rule_output=_rule_output(rule_result, run_context),
                waterfall_filter=waterfall_filter,
                model_output=model_output,
                source_fingerprint=source_fingerprint,
                measurements=measurements,
            )
            timings["render"] += _elapsed_ms(render_started)
            display_start, display_stop = event_display_window(
                event.start_sample,
                event.end_sample,
                sample_count,
                profile.artifact_pad_seconds,
                metadata.sample_rate_hz,
            )
            display_data = artifact_data
            if (display_start, display_stop) != (window_start, window_stop):
                read_started = time.perf_counter()
                display_raw = reader.read_block(reader_source, display_start, display_stop)
                timings["read"] += _elapsed_ms(read_started)
                preprocess_started = time.perf_counter()
                display_filtered = _preprocess_for_band(
                    display_raw,
                    profile,
                    backend,
                    (waterfall_filter.low_hz, waterfall_filter.high_hz),
                )
                display_data = replace(
                    artifact_data,
                    raw_data=display_raw.data,
                    filtered_data=np.asarray(backend.asnumpy(display_filtered.data), dtype=np.float32),
                    sample_offset=display_raw.sample_offset,
                    start_time_utc=display_raw.metadata.start_time_utc,
                    bad_channel_mask=display_filtered.bad_channel_mask,
                )
                timings["preprocess"] += _elapsed_ms(preprocess_started)
            render_started = time.perf_counter()
            _ensure_source_unchanged(source, source_fingerprint)
            if transaction is None:
                raise RuntimeError("artifact transaction is unavailable")
            transaction.add(
                render_event_artifacts(
                    artifact_data,
                    classified,
                    features,
                    transaction.staging_root,
                    display_data=display_data,
                    source_sample_count=sample_count,
                )
            )
            _ensure_source_unchanged(source, source_fingerprint)
            timings["render"] += _elapsed_ms(render_started)
        if transaction is not None:
            artifacts.extend(transaction.commit())
    except BaseException:
        if transaction is not None:
            transaction._rollback()
        raise
    finally:
        if transaction is not None:
            try:
                transaction.close()
            except BaseException:
                transaction._rollback()
                raise

    timings["total"] = _elapsed_ms(started)
    return FileAnalysis(
        source_path=str(source_path),
        source_fingerprint=source_fingerprint,
        actual_backend=backend.name,
        backend_fallback_reason=backend.unavailable_reason,
        metadata_warnings=metadata.warnings,
        events=tuple(events),
        audit_candidate_count=len(candidates),
        timings_ms=timings,
        artifacts=tuple(artifacts),
    )


def _preprocess_for_band(
    block: DasBlock,
    profile: ProcessingProfile,
    backend: ArrayBackend,
    band: tuple[float, float],
) -> PreprocessedBlock:
    return preprocess_for_waterfall(
        mask_scan_block(block, profile),
        profile,
        backend,
        WaterfallFilter("stalta", band[0], band[1]),
    )


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


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


class _ArtifactTransaction:
    def __init__(
        self,
        artifact_root: Path,
        source: Path | BoundSource,
        source_fingerprint: str,
    ) -> None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.final_root = artifact_root.resolve(strict=True)
        self.source = source
        self.source_fingerprint = source_fingerprint
        self.staging_root = Path(
            tempfile.mkdtemp(prefix=".das-staging-", dir=self.final_root)
        ).resolve(strict=True)
        if self.staging_root.parent != self.final_root:
            raise ValueError("artifact staging path escapes final root")
        self._staged: list[ArtifactManifest] = []
        self._promoted: list[Path] = []

    def add(self, manifest: ArtifactManifest) -> None:
        self._staged.append(manifest)

    def commit(self) -> list[ArtifactManifest]:
        completed: list[ArtifactManifest] = []
        lock_path = self.final_root / ".pipeline-promote.lock"
        with _process_lock(lock_path):
            try:
                _ensure_source_unchanged(
                    self.source, self.source_fingerprint
                )
                for manifest in self._staged:
                    final_manifest, created = _promote_event_directory(
                        manifest, self.staging_root, self.final_root
                    )
                    if created:
                        self._promoted.append(
                            self.final_root / manifest.event_id
                        )
                    completed.append(final_manifest)
                    _ensure_source_unchanged(
                        self.source, self.source_fingerprint
                    )
                _ensure_source_unchanged(
                    self.source, self.source_fingerprint
                )
                self._promoted.clear()
            except BaseException:
                self._rollback()
                raise
        return completed

    def close(self) -> None:
        _remove_staging_directory(self.staging_root, self.final_root)

    def _rollback(self) -> None:
        for event_dir in reversed(self._promoted):
            _remove_promoted_directory(event_dir, self.final_root)
        self._promoted.clear()


def _promote_event_directory(
    manifest: ArtifactManifest, staging_root: Path, final_root: Path
) -> tuple[ArtifactManifest, bool]:
    event_id = str(UUID(manifest.event_id))
    staging_event = (staging_root / event_id).resolve(strict=True)
    if staging_event.parent != staging_root:
        raise ValueError("staged artifact path escapes staging root")
    final_event = final_root / event_id
    if final_event.exists():
        return _manifest_at_final(manifest, final_event, final_root), False
    created = False
    try:
        os.replace(staging_event, final_event)
        created = True
    except OSError:
        if not final_event.exists():
            raise
        return _manifest_at_final(manifest, final_event, final_root), False
    try:
        return _manifest_at_final(manifest, final_event, final_root), True
    except BaseException:
        if created:
            _remove_promoted_directory(final_event, final_root)
        raise


def _manifest_at_final(
    staged: ArtifactManifest, final_event: Path, final_root: Path
) -> ArtifactManifest:
    resolved_event = final_event.resolve(strict=True)
    if resolved_event != final_event or resolved_event.parent != final_root:
        raise ValueError("final artifact path escapes final root")
    files: dict[str, ArtifactFile] = {}
    for kind, staged_file in staged.files.items():
        final_file = (resolved_event / staged_file.path.name).resolve(strict=True)
        if final_file.parent != resolved_event:
            raise ValueError("final artifact file escapes event directory")
        digest = hashlib.sha256(final_file.read_bytes()).hexdigest()
        if digest != staged_file.sha256:
            raise ArtifactConflictError(
                f"artifact already exists with different bytes: {kind}"
            )
        files[kind] = ArtifactFile(
            kind=kind,
            path=final_file,
            media_type=staged_file.media_type,
            sha256=digest,
        )
    return ArtifactManifest(staged.event_id, files)


def _remove_staging_directory(staging_root: Path, final_root: Path) -> None:
    if not staging_root.exists():
        return
    if (
        staging_root.parent != final_root
        or not staging_root.name.startswith(".das-staging-")
    ):
        raise ValueError("refusing to remove unvalidated artifact staging path")
    shutil.rmtree(staging_root)


def _remove_promoted_directory(event_dir: Path, final_root: Path) -> None:
    if not event_dir.exists():
        return
    if event_dir.parent != final_root:
        raise ValueError("refusing to remove artifact outside final root")
    UUID(event_dir.name)
    shutil.rmtree(event_dir)


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


def _ensure_source_unchanged(
    path: Path | BoundSource,
    expected: str,
) -> None:
    try:
        if isinstance(path, BoundSource):
            stat_result = os.fstat(path.stream.fileno())
            if (
                stat_result.st_size != path.size_bytes
                or stat_result.st_mtime_ns != path.mtime_ns
            ):
                raise SourceChangedError(
                    "DAS source changed during analysis"
                )
            current = _source_fingerprint_from_stat(path.path, stat_result)
        else:
            current = _source_fingerprint(path)
    except OSError as error:
        raise SourceChangedError("DAS source changed during analysis") from error
    if current != expected:
        raise SourceChangedError("DAS source changed during analysis")


def _source_fingerprint(path: Path) -> str:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return _source_fingerprint_from_stat(resolved, stat)


def _source_fingerprint_from_stat(
    path: Path,
    stat_result: os.stat_result,
) -> str:
    payload = json.dumps(
        (str(path), stat_result.st_size, stat_result.st_mtime_ns),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1_000
