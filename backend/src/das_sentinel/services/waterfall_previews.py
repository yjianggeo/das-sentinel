from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from das_sentinel.artifacts import (
    WaterfallPlotContext,
    render_waterfall_png,
    render_waveform_png,
)
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate, DasMetadata, EventRecord
from das_sentinel.persistence.models import (
    Event,
    ModelVersion,
    ProcessingRun,
    Site,
    SourceFile,
)
from das_sentinel.persistence.models import ProcessingProfile as ProfileRow
from das_sentinel.plot_windows import event_display_window
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.services.discovery import fingerprint
from das_sentinel.signal.backend import select_backend
from das_sentinel.waterfalls import (
    WaterfallFilter,
    channel_slice_for_distance,
    preprocess_for_waterfall,
    resolve_waterfall_filter,
    usable_channel_indices,
)

_MAX_PREVIEW_SAMPLES = 30_000
_MAX_PREVIEW_CHANNELS = 2048


class WaterfallPreviewConflict(RuntimeError):
    pass


class WaterfallPreviewRangeError(ValueError):
    pass


@dataclass(frozen=True)
class WaterfallOptions:
    automatic_distance_start_m: float
    automatic_distance_stop_m: float
    available_distance_start_m: float
    available_distance_stop_m: float
    filter: WaterfallFilter


@dataclass(frozen=True)
class WaterfallPreview:
    distance_start_m: float
    distance_stop_m: float
    channel_count: int
    filter: WaterfallFilter
    raw_png: bytes
    filtered_png: bytes
    waveform_png: bytes


@dataclass(frozen=True)
class _Binding:
    event: Event
    candidates: tuple[Candidate, ...]
    source_path: Path
    source_fingerprint: str
    profile: ProcessingProfile
    profile_version: int
    pipeline_version: str
    model_version: str
    site_timezone: str
    filter: WaterfallFilter


class WaterfallPreviewService:
    def __init__(self, data_root: Path, reader: ZdHdf5Reader) -> None:
        self.data_root = Path(data_root)
        self.reader = reader

    def options(self, session: Session, event_id: str) -> WaterfallOptions:
        binding = self._binding(session, event_id)
        metadata = self._metadata(binding)
        sample_count = self._sample_count(binding)
        self._validate_event_binding(binding, metadata, sample_count)
        self._verify_fingerprint(binding.source_path, binding.source_fingerprint)
        start_m, stop_m = self._automatic_range(binding, metadata.channel_distance_m)
        return WaterfallOptions(
            automatic_distance_start_m=start_m,
            automatic_distance_stop_m=stop_m,
            available_distance_start_m=float(np.min(metadata.channel_distance_m)),
            available_distance_stop_m=float(np.max(metadata.channel_distance_m)),
            filter=binding.filter,
        )

    def render(
        self,
        session: Session,
        event_id: str,
        start_m: float | None = None,
        stop_m: float | None = None,
    ) -> WaterfallPreview:
        binding = self._binding(session, event_id)
        metadata = self._metadata(binding)
        sample_count = self._sample_count(binding)
        self._validate_event_binding(binding, metadata, sample_count)
        automatic = start_m is None and stop_m is None
        if not automatic:
            if start_m is None or stop_m is None:
                raise WaterfallPreviewRangeError("both distance bounds must be supplied")
            available_start = float(np.min(metadata.channel_distance_m))
            available_stop = float(np.max(metadata.channel_distance_m))
            self._validate_range(start_m, stop_m, available_start, available_stop)
        try:
            read_slice = (
                slice(binding.event.channel_start, binding.event.channel_stop)
                if automatic
                else channel_slice_for_distance(metadata.channel_distance_m, start_m, stop_m)
            )
            sample_start, sample_stop = event_display_window(
                binding.event.start_sample,
                binding.event.end_sample,
                sample_count,
                binding.profile.artifact_pad_seconds,
                metadata.sample_rate_hz,
            )
            self._validate_read_bounds(sample_start, sample_stop, read_slice)
        except ValueError as error:
            raise WaterfallPreviewRangeError("requested range is not previewable") from error
        try:
            block = self.reader.read_block(
                binding.source_path,
                sample_start,
                sample_stop,
                read_slice,
            )
        except (OSError, ValueError) as error:
            raise WaterfallPreviewConflict("waterfall preview source is unavailable") from error
        try:
            filtered = preprocess_for_waterfall(
                block,
                binding.profile,
                select_backend("cpu"),
                binding.filter,
            )
        except ValueError as error:
            raise WaterfallPreviewConflict("event binding is inconsistent") from error
        try:
            if automatic:
                indices = np.flatnonzero(~np.asarray(filtered.bad_channel_mask, dtype=bool))
                if indices.size == 0:
                    raise ValueError("event range has no usable channels")
            else:
                indices = usable_channel_indices(
                    block.metadata.channel_distance_m,
                    filtered.bad_channel_mask,
                    start_m,
                    stop_m,
                )
        except ValueError as error:
            raise WaterfallPreviewRangeError("requested range is not previewable") from error
        dates = np.array(
            [
                block.metadata.start_time_utc
                + timedelta(seconds=index / block.metadata.sample_rate_hz)
                for index in range(block.data.shape[0])
            ]
        )
        context = WaterfallPlotContext(
            sample_rate_hz=block.metadata.sample_rate_hz,
            site_timezone=binding.site_timezone,
            profile_version=str(binding.profile_version),
            pipeline_version=binding.pipeline_version,
            model_version=binding.model_version,
            filter=binding.filter,
        )
        distances = block.metadata.channel_distance_m[indices]
        actual_start_m = float(np.min(distances))
        actual_stop_m = float(np.max(distances))
        distance_mode = "Automatic event range" if automatic else "Artificial Distance"
        note = f"{distance_mode} {actual_start_m:g}–{actual_stop_m:g} m"
        window_note = (
            f"Display samples {sample_start}:{sample_stop} | "
            f"duration {(sample_stop - sample_start) / metadata.sample_rate_hz:g} s"
        )
        if sample_count / metadata.sample_rate_hz < 3:
            window_note += " | Source shorter than 3 s; showing actual duration"
        event = EventRecord(
            event_id=binding.event.id,
            start_sample=binding.event.start_sample,
            end_sample=binding.event.end_sample,
            peak_sample=min(
                binding.candidates, key=lambda candidate: (-candidate.score, candidate.peak_sample)
            ).peak_sample,
            channel_start=binding.event.channel_start,
            channel_stop=binding.event.channel_stop,
            evidence=binding.candidates,
        )
        selected_filtered = np.asarray(filtered.data)[:, indices]
        try:
            raw_png = render_waterfall_png(
                block.data[:, indices],
                dates,
                distances,
                context,
                filtered=False,
                window_note=window_note,
                distance_note=note,
            )
            filtered_png = render_waterfall_png(
                selected_filtered,
                dates,
                distances,
                context,
                filtered=True,
                window_note=window_note,
                distance_note=note,
            )
            waveform_png = render_waveform_png(
                np.median(selected_filtered, axis=1),
                dates,
                context,
                event,
                sample_offset=block.sample_offset,
                window_note=window_note,
                distance_note=note,
            )
        except (ValueError, OSError) as error:
            raise WaterfallPreviewConflict("waterfall preview source is unavailable") from error
        self._verify_fingerprint(binding.source_path, binding.source_fingerprint)
        return WaterfallPreview(
            distance_start_m=actual_start_m,
            distance_stop_m=actual_stop_m,
            channel_count=int(indices.size),
            filter=binding.filter,
            raw_png=raw_png,
            filtered_png=filtered_png,
            waveform_png=waveform_png,
        )

    def _binding(self, session: Session, event_id: str) -> _Binding:
        event = session.get(Event, event_id)
        if event is None:
            raise LookupError("event not found")
        run = session.get(ProcessingRun, event.processing_run_id)
        source = None if run is None else session.get(SourceFile, run.source_file_id)
        profile_row = (
            None if run is None else session.get(ProfileRow, run.processing_profile_id)
        )
        if run is None or source is None or profile_row is None:
            raise WaterfallPreviewConflict("event binding is inconsistent")
        site = session.get(Site, source.site_id)
        model = (
            None
            if run.model_version_id is None
            else session.get(ModelVersion, run.model_version_id)
        )
        if site is None or (run.model_version_id is not None and model is None):
            raise WaterfallPreviewConflict("event binding is inconsistent")
        if (
            profile_row.site_id != source.site_id
            or run.profile_version != profile_row.version
            or (model is not None and model.site_id != source.site_id)
        ):
            raise WaterfallPreviewConflict("event binding is inconsistent")
        try:
            profile = ProcessingProfile.model_validate(json.loads(profile_row.payload_json))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise WaterfallPreviewConflict("event profile is unavailable") from error
        source_path = self._source_path(source, site)
        if source.fingerprint != run.source_fingerprint:
            raise WaterfallPreviewConflict("event source changed")
        self._verify_fingerprint(source_path, source.fingerprint)
        try:
            candidates = self._candidates(event)
            self._validate_fused_event(event, candidates)
            filter_context = resolve_waterfall_filter(candidates, profile)
        except ValueError as error:
            raise WaterfallPreviewConflict("event evidence is unavailable") from error
        return _Binding(
            event=event,
            candidates=candidates,
            source_path=source_path,
            source_fingerprint=source.fingerprint,
            profile=profile,
            profile_version=profile_row.version,
            pipeline_version=run.pipeline_version,
            model_version="rules-only" if model is None else model.version,
            site_timezone=site.timezone,
            filter=filter_context,
        )

    def _source_path(self, source: SourceFile, site: Site) -> Path:
        try:
            root = self.data_root.resolve(strict=True)
            site_root = Path(site.data_path).resolve(strict=True)
            path = Path(source.path).resolve(strict=True)
            if not site_root.is_relative_to(root) or not path.is_relative_to(site_root):
                raise ValueError
            return path
        except (OSError, RuntimeError, ValueError) as error:
            raise WaterfallPreviewConflict("waterfall preview source is unavailable") from error

    @staticmethod
    def _candidates(event: Event) -> tuple[Candidate, ...]:
        candidates = []
        for evidence in event.evidence:
            try:
                payload = json.loads(evidence.payload_json)
                candidates.append(
                    Candidate(
                        detector=evidence.detector,
                        start_sample=int(payload["start_sample"]),
                        end_sample=int(payload["end_sample"]),
                        peak_sample=int(payload["peak_sample"]),
                        channel_start=int(payload["channel_start"]),
                        channel_stop=int(payload["channel_stop"]),
                        score=float(evidence.score),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise WaterfallPreviewConflict("event evidence is unavailable") from error
        if not candidates:
            raise WaterfallPreviewConflict("event evidence is unavailable")
        return tuple(candidates)

    def _metadata(self, binding: _Binding):
        try:
            return self.reader.read_metadata(binding.source_path)
        except (OSError, ValueError) as error:
            raise WaterfallPreviewConflict("waterfall preview source is unavailable") from error

    def _sample_count(self, binding: _Binding) -> int:
        try:
            count = self.reader.sample_count(binding.source_path)
        except (OSError, ValueError) as error:
            raise WaterfallPreviewConflict("waterfall preview source is unavailable") from error
        if count <= 0:
            raise WaterfallPreviewConflict("waterfall preview source is unavailable")
        return count

    def _automatic_range(self, binding: _Binding, distances: np.ndarray) -> tuple[float, float]:
        start = binding.event.channel_start
        stop = binding.event.channel_stop
        if start < 0 or stop <= start or stop > len(distances):
            raise WaterfallPreviewConflict("event channel bounds are unavailable")
        selected = np.asarray(distances[start:stop], dtype=float)
        if selected.size == 0 or not np.isfinite(selected).all():
            raise WaterfallPreviewConflict("event channel bounds are unavailable")
        return float(np.min(selected)), float(np.max(selected))

    @staticmethod
    def _validate_read_bounds(
        sample_start: int, sample_stop: int, channels: slice
    ) -> None:
        if sample_stop - sample_start > _MAX_PREVIEW_SAMPLES:
            raise WaterfallPreviewRangeError("event window exceeds preview limit")
        if channels.stop - channels.start > _MAX_PREVIEW_CHANNELS:
            raise WaterfallPreviewRangeError("channel range exceeds preview limit")

    @staticmethod
    def _validate_range(
        start_m: float,
        stop_m: float,
        automatic_start: float,
        automatic_stop: float,
    ) -> None:
        if not np.isfinite(start_m) or not np.isfinite(stop_m):
            raise WaterfallPreviewRangeError("distance range must be finite")
        if stop_m <= start_m:
            raise WaterfallPreviewRangeError("distance_stop_m must exceed distance_start_m")
        if start_m < automatic_start or stop_m > automatic_stop:
            raise WaterfallPreviewRangeError("distance range must remain within the source")

    @staticmethod
    def _validate_fused_event(
        event: Event, candidates: tuple[Candidate, ...]
    ) -> None:
        if any(
            candidate.start_sample < 0
            or candidate.end_sample <= candidate.start_sample
            or not candidate.start_sample <= candidate.peak_sample < candidate.end_sample
            or candidate.channel_start < 0
            or candidate.channel_stop <= candidate.channel_start
            or not np.isfinite(candidate.score)
            for candidate in candidates
        ):
            raise WaterfallPreviewConflict("event evidence is unavailable")
        if (
            event.start_sample != min(item.start_sample for item in candidates)
            or event.end_sample != max(item.end_sample for item in candidates)
            or event.channel_start != min(item.channel_start for item in candidates)
            or event.channel_stop != max(item.channel_stop for item in candidates)
        ):
            raise WaterfallPreviewConflict("event binding is inconsistent")

    @staticmethod
    def _validate_event_binding(
        binding: _Binding,
        metadata: DasMetadata,
        sample_count: int,
    ) -> None:
        distances = metadata.channel_distance_m
        if not np.isfinite(metadata.sample_rate_hz) or metadata.sample_rate_hz <= 0:
            raise WaterfallPreviewConflict("event binding is inconsistent")
        expected_occurred_at = metadata.start_time_utc + timedelta(
            seconds=binding.event.start_sample / metadata.sample_rate_hz
        )
        if (
            distances.ndim != 1
            or distances.size == 0
            or not np.isfinite(distances).all()
            or binding.event.channel_stop > distances.size
            or binding.event.end_sample > sample_count
            or binding.event.occurred_at != expected_occurred_at
        ):
            raise WaterfallPreviewConflict("event binding is inconsistent")

    @staticmethod
    def _verify_fingerprint(path: Path, expected: str) -> None:
        try:
            current = fingerprint(path)
        except (OSError, ValueError) as error:
            raise WaterfallPreviewConflict("waterfall preview source is unavailable") from error
        if current != expected:
            raise WaterfallPreviewConflict("event source changed")
