from __future__ import annotations

import hashlib
import os
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from das_sentinel.api.schemas import (
    WorkerArtifactResult,
    WorkerEventResult,
    WorkerResultRequest,
)
from das_sentinel.persistence.models import (
    Event,
    Job,
    ProcessingProfile,
    SourceFile,
    utc_now,
)
from das_sentinel.persistence.repositories import Repositories
from das_sentinel.services.discovery import (
    DiscoveryPathError,
    _AnchoredSource,
    _open_anchored_source,
)
from das_sentinel.services.jobs import (
    JobConflictError,
    JobNotFoundError,
    JobService,
    JobSnapshot,
    JobValidationError,
)

MAX_RESULT_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_RESULT_ARTIFACT_FILES = 600
ARTIFACT_KINDS = frozenset(
    {
        "waterfall_raw",
        "waterfall_filtered",
        "waveform",
        "psd",
        "stft",
        "evidence",
    }
)


class ResultService:
    """Persist one worker result and complete its job in the caller transaction."""

    def __init__(self, session: Session, artifact_root: Path) -> None:
        self.session = session
        self.artifact_root = Path(artifact_root).absolute()
        self.repositories = Repositories(session)
        self.jobs = JobService(session, self.artifact_root)

    def persist(
        self,
        job_id: str,
        payload: WorkerResultRequest,
    ) -> JobSnapshot:
        job = self.session.get(Job, job_id)
        if job is None:
            raise JobNotFoundError(f"job not found: {job_id}")
        if job.status != "running":
            raise JobConflictError("job is not running")
        if job.worker_id != payload.worker_id:
            raise JobConflictError("worker is not the job owner")
        source = self.session.get(SourceFile, job.source_file_id)
        profile = self.session.get(
            ProcessingProfile,
            job.processing_profile_id,
        )
        if source is None:
            raise JobNotFoundError(f"source file not found: {job.source_file_id}")
        if profile is None:
            raise JobNotFoundError(
                f"processing profile not found: {job.processing_profile_id}"
            )
        self._validate_identity(job, source, profile, payload)
        events_by_uid = self._validate_manifest(payload)
        with ExitStack() as artifact_handles:
            artifact_rows, held_artifacts = self._validate_artifact_files(
                payload,
                artifact_handles,
            )
            return self._persist_validated(
                job,
                source,
                profile,
                payload,
                events_by_uid,
                artifact_rows,
                held_artifacts,
            )

    def _persist_validated(
        self,
        job: Job,
        source: SourceFile,
        profile: ProcessingProfile,
        payload: WorkerResultRequest,
        events_by_uid: dict[str, WorkerEventResult],
        artifact_rows: list[tuple[WorkerArtifactResult, str]],
        held_artifacts: list[
            tuple[_AnchoredSource, str, tuple[Path, int, int]]
        ],
    ) -> JobSnapshot:
        now = utc_now()
        run = self.repositories.runs.add(
            job_id=job.id,
            model_version_id=payload.model_version_id,
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version=job.pipeline_version,
            requested_backend=payload.requested_backend,
            actual_backend=payload.actual_backend,
            backend_fallback_reason=payload.backend_fallback_reason,
            status="succeeded",
            timings=payload.timings_ms,
        )
        run.started_at = now
        run.finished_at = now
        persisted_events: dict[str, Event] = {}
        for event in payload.events:
            occurred_at = payload.source_start_time_utc + timedelta(
                seconds=event.start_sample / payload.sample_rate_hz
            )
            row = self.repositories.events.add(
                run_id=run.id,
                event_uid=event.event_uid,
                start_sample=event.start_sample,
                end_sample=event.end_sample,
                channel_start=event.channel_start,
                channel_stop=event.channel_stop,
                occurred_at=occurred_at,
                features=dict(event.features),
            )
            persisted_events[event.event_uid] = row
            for evidence in event.evidence:
                self.repositories.events.add_evidence(
                    row.id,
                    detector=evidence.detector,
                    score=evidence.score,
                    payload=evidence.payload,
                )
            classification = event.classification
            self.repositories.events.add_classification(
                row.id,
                label=classification.label,
                decision_score=classification.decision_score,
                method=classification.method,
                reasons=classification.reasons,
                model_version_id=payload.model_version_id,
            )
        if set(persisted_events) != set(events_by_uid):
            raise JobValidationError("result event identities are inconsistent")
        for artifact, stored_path in artifact_rows:
            persisted_event = persisted_events[artifact.event_uid]
            self.repositories.artifacts.add(
                event_id=persisted_event.id,
                kind=artifact.kind,
                path=stored_path,
                media_type=artifact.media_type,
                sha256=artifact.sha256,
            )
        completed = self.jobs.complete(
            job.id,
            payload.worker_id,
            payload.profile_version,
            payload.pipeline_version,
            (),
            now=now,
        )
        self._verify_held_artifacts(held_artifacts)
        return completed

    @staticmethod
    def _validate_identity(
        job: Job,
        source: SourceFile,
        profile: ProcessingProfile,
        payload: WorkerResultRequest,
    ) -> None:
        if payload.source_file_id != source.id:
            raise JobConflictError("source file does not match claimed job")
        if payload.processing_profile_id != profile.id:
            raise JobConflictError("processing profile does not match claimed job")
        if payload.model_version_id != job.model_version_id:
            raise JobConflictError("model version does not match claimed job")
        if payload.profile_version != profile.version:
            raise JobConflictError("profile version does not match claimed job")
        if payload.pipeline_version != job.pipeline_version:
            raise JobConflictError("pipeline version does not match claimed job")
        if payload.source_fingerprint != source.fingerprint:
            raise JobConflictError("source fingerprint does not match claimed job")
        expected_method = (
            "fusion" if payload.model_version_id is not None else "rules"
        )
        if any(
            event.classification.method != expected_method
            for event in payload.events
        ):
            raise JobValidationError(
                "classification method does not match model identity"
            )
        if (
            payload.actual_backend == "cuda"
            and payload.backend_fallback_reason is not None
        ):
            raise JobValidationError(
                "CUDA results cannot declare a CPU fallback reason"
            )
        if (
            payload.backend_fallback_reason is not None
            and payload.actual_backend != "cpu"
        ):
            raise JobValidationError("fallback results must use the CPU backend")

    @staticmethod
    def _validate_manifest(
        payload: WorkerResultRequest,
    ) -> dict[str, WorkerEventResult]:
        if len(payload.artifacts) > MAX_RESULT_ARTIFACT_FILES:
            raise JobValidationError(
                "result contains too many artifact files"
            )
        events_by_uid: dict[str, WorkerEventResult] = {}
        for event in payload.events:
            if event.event_uid in events_by_uid:
                raise JobValidationError("result contains duplicate event UIDs")
            events_by_uid[event.event_uid] = event
        artifact_kinds: dict[str, set[str]] = {
            event_uid: set() for event_uid in events_by_uid
        }
        for artifact in payload.artifacts:
            if artifact.event_uid not in artifact_kinds:
                raise JobValidationError("artifact refers to an unknown event")
            kinds = artifact_kinds[artifact.event_uid]
            if artifact.kind in kinds:
                raise JobValidationError("artifact kind is duplicated for an event")
            kinds.add(artifact.kind)
            expected_media = (
                "application/json"
                if artifact.kind == "evidence"
                else "image/png"
            )
            if artifact.media_type != expected_media:
                raise JobValidationError("artifact media type does not match kind")
        if any(kinds != ARTIFACT_KINDS for kinds in artifact_kinds.values()):
            raise JobValidationError(
                "every result event must register exactly six artifact kinds"
            )
        return events_by_uid

    def _validate_artifact_files(
        self,
        payload: WorkerResultRequest,
        handles: ExitStack,
    ) -> tuple[
        list[tuple[WorkerArtifactResult, str]],
        list[tuple[_AnchoredSource, str, tuple[Path, int, int]]],
    ]:
        rows: list[tuple[WorkerArtifactResult, str]] = []
        held: list[
            tuple[_AnchoredSource, str, tuple[Path, int, int]]
        ] = []
        for artifact in payload.artifacts:
            candidate = Path(artifact.path)
            if not candidate.is_absolute() or ".." in candidate.parts:
                raise JobValidationError(
                    "artifact path must be absolute and normalized"
                )
            candidate = candidate.absolute()
            try:
                relative = candidate.relative_to(self.artifact_root)
            except ValueError as error:
                raise JobValidationError(
                    "artifact path escapes configured artifact root"
                ) from error
            try:
                source = handles.enter_context(
                    _open_anchored_source(
                        self.artifact_root,
                        candidate,
                        required_suffix=None,
                        hold_ancestors=os.name == "nt",
                    )
                )
                snapshot = source.snapshot()
            except (DiscoveryPathError, OSError) as error:
                raise JobValidationError(
                    "artifact path must use an existing no-follow regular file chain"
                ) from error
            if snapshot.size_bytes > MAX_RESULT_ARTIFACT_BYTES:
                raise JobValidationError("artifact file exceeds the size limit")
            try:
                digest = _stream_sha256(source.stream)
            except OSError as error:
                raise JobValidationError(
                    "artifact file is not readable"
                ) from error
            if digest != artifact.sha256:
                raise JobValidationError("artifact SHA-256 does not match file")
            rows.append((artifact, relative.as_posix()))
            held.append((source, digest, snapshot.signature))
        return rows, held

    @staticmethod
    def _verify_held_artifacts(
        held_artifacts: list[
            tuple[_AnchoredSource, str, tuple[Path, int, int]]
        ],
    ) -> None:
        for source, expected_digest, expected_signature in held_artifacts:
            try:
                if source.snapshot().signature != expected_signature:
                    raise JobValidationError(
                        "artifact file changed while persisting result"
                    )
                if _stream_sha256(source.stream) != expected_digest:
                    raise JobValidationError(
                        "artifact file changed while persisting result"
                    )
            except OSError as error:
                raise JobValidationError(
                    "artifact file changed while persisting result"
                ) from error


def _stream_sha256(
    stream,
    *,
    maximum_bytes: int = MAX_RESULT_ARTIFACT_BYTES,
) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    remaining = maximum_bytes
    try:
        while True:
            chunk = stream.read(min(1024 * 1024, remaining + 1))
            if not chunk:
                return digest.hexdigest()
            if len(chunk) > remaining:
                raise JobValidationError(
                    "artifact file exceeds the size limit"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    finally:
        stream.seek(0)
