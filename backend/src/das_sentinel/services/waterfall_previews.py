from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from das_sentinel.artifacts import WaterfallPlotContext, render_waterfall_png
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate
from das_sentinel.persistence.models import (
    Event,
    ProcessingRun,
    Site,
    SourceFile,
)
from das_sentinel.persistence.models import ProcessingProfile as ProfileRow
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


@dataclass(frozen=True)
class _Binding:
    event: Event
    source_path: Path
    source_fingerprint: str
    profile: ProcessingProfile
    profile_version: int
    pipeline_version: str
    site_timezone: str
    filter: WaterfallFilter


class WaterfallPreviewService:
    def __init__(self, data_root: Path, reader: ZdHdf5Reader) -> None:
        self.data_root = Path(data_root)
        self.reader = reader

    def options(self, session: Session, event_id: str) -> WaterfallOptions:
        binding = self._binding(session, event_id)
        metadata = self._metadata(binding)
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
        start_m: float,
        stop_m: float,
    ) -> WaterfallPreview:
        binding = self._binding(session, event_id)
        metadata = self._metadata(binding)
        automatic_start, automatic_stop = self._automatic_range(
            binding, metadata.channel_distance_m
        )
        self._validate_range(start_m, stop_m, automatic_start, automatic_stop)
        try:
            read_slice = channel_slice_for_distance(
                metadata.channel_distance_m, start_m, stop_m
            )
            block = self.reader.read_block(
                binding.source_path,
                binding.event.start_sample,
                binding.event.end_sample,
                read_slice,
            )
            filtered = preprocess_for_waterfall(
                block,
                binding.profile,
                select_backend("cpu"),
                binding.filter,
            )
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
            model_version="rules-only",
            filter=binding.filter,
        )
        distances = block.metadata.channel_distance_m[indices]
        note = f"Artificial Distance {start_m:g}–{stop_m:g} m"
        try:
            raw_png = render_waterfall_png(
                block.data[:, indices],
                dates,
                distances,
                context,
                filtered=False,
                distance_note=note,
            )
            filtered_png = render_waterfall_png(
                np.asarray(filtered.data)[:, indices],
                dates,
                distances,
                context,
                filtered=True,
                distance_note=note,
            )
        except (ValueError, OSError) as error:
            raise WaterfallPreviewConflict("waterfall preview source is unavailable") from error
        self._verify_fingerprint(binding.source_path, binding.source_fingerprint)
        return WaterfallPreview(
            distance_start_m=float(start_m),
            distance_stop_m=float(stop_m),
            channel_count=int(indices.size),
            filter=binding.filter,
            raw_png=raw_png,
            filtered_png=filtered_png,
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
            raise LookupError("event binding not found")
        if profile_row.site_id != source.site_id:
            raise WaterfallPreviewConflict("event binding is inconsistent")
        try:
            profile = ProcessingProfile.model_validate(json.loads(profile_row.payload_json))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise WaterfallPreviewConflict("event profile is unavailable") from error
        source_path = self._source_path(source)
        if source.fingerprint != run.source_fingerprint:
            raise WaterfallPreviewConflict("event source changed")
        self._verify_fingerprint(source_path, source.fingerprint)
        try:
            filter_context = resolve_waterfall_filter(self._candidates(event), profile)
        except ValueError as error:
            raise WaterfallPreviewConflict("event evidence is unavailable") from error
        return _Binding(
            event=event,
            source_path=source_path,
            source_fingerprint=source.fingerprint,
            profile=profile,
            profile_version=profile_row.version,
            pipeline_version=run.pipeline_version,
            site_timezone=self._site_timezone(session, source.site_id),
            filter=filter_context,
        )

    def _site_timezone(self, session: Session, site_id: str) -> str:
        site = session.get(Site, site_id)
        if site is None:
            raise LookupError("event site not found")
        return site.timezone

    def _source_path(self, source: SourceFile) -> Path:
        try:
            root = self.data_root.resolve(strict=True)
            path = Path(source.path).resolve(strict=True)
            if not path.is_relative_to(root):
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

    def _automatic_range(self, binding: _Binding, distances: np.ndarray) -> tuple[float, float]:
        candidates = self._candidates(binding.event)
        start = min(candidate.channel_start for candidate in candidates)
        stop = max(candidate.channel_stop for candidate in candidates)
        if start < 0 or stop <= start or stop > len(distances):
            raise WaterfallPreviewConflict("event channel bounds are unavailable")
        return float(distances[start]), float(distances[stop - 1])

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
            raise WaterfallPreviewRangeError("distance range must remain within the event")

    @staticmethod
    def _verify_fingerprint(path: Path, expected: str) -> None:
        try:
            current = fingerprint(path)
        except (OSError, ValueError) as error:
            raise WaterfallPreviewConflict("waterfall preview source is unavailable") from error
        if current != expected:
            raise WaterfallPreviewConflict("event source changed")
