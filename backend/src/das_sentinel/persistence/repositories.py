from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from das_sentinel.persistence.models import (
    MACHINE_LABELS,
    SUPERVISED_LABELS,
    Admin,
    Artifact,
    Classification,
    DetectionEvidence,
    Event,
    Job,
    ModelVersion,
    ProcessingProfile,
    ProcessingRun,
    Review,
    Site,
    SourceFile,
    normalize_utc,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ModelT = TypeVar("ModelT")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str, *, field: str = "SHA-256") -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _label(value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"unsupported label: {value}")
    return value


class _Repository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _add(self, row: ModelT) -> ModelT:
        self.session.add(row)
        self.session.flush()
        return row


class AdminRepository(_Repository):
    def add(self, *, password_hash: str, disabled: bool = False) -> Admin:
        return self._add(Admin(password_hash=password_hash, disabled=disabled))

    def get(self, admin_id: int = 1) -> Admin | None:
        return self.session.get(Admin, admin_id)


class SiteRepository(_Repository):
    def add(
        self,
        *,
        name: str,
        data_path: str,
        timezone: str,
        cable_type: str,
        enabled: bool = True,
    ) -> Site:
        return self._add(
            Site(
                name=name,
                data_path=data_path,
                timezone=timezone,
                cable_type=cable_type,
                enabled=enabled,
            )
        )

    def get(self, site_id: str) -> Site | None:
        return self.session.get(Site, site_id)

    def list(self) -> list[Site]:
        return list(self.session.scalars(select(Site).order_by(Site.name)))

    def set_enabled(self, site_id: str, enabled: bool) -> Site:
        site = self._require(site_id)
        site.enabled = enabled
        self.session.flush()
        return site

    def set_current_profile(self, site_id: str, profile_id: str) -> Site:
        site = self._require(site_id)
        profile = self.session.get(ProcessingProfile, profile_id)
        if profile is None or profile.site_id != site.id:
            raise ValueError("profile does not belong to site")
        statement = select(ProcessingProfile).where(
            ProcessingProfile.site_id == site_id,
            ProcessingProfile.is_current.is_(True),
        )
        for current in self.session.scalars(statement):
            current.is_current = False
        self.session.flush()
        profile.is_current = True
        self.session.flush()
        return site

    def _require(self, site_id: str) -> Site:
        site = self.get(site_id)
        if site is None:
            raise LookupError(f"site not found: {site_id}")
        return site


class ProfileRepository(_Repository):
    def add(
        self,
        *,
        site_id: str,
        version: int,
        payload: dict[str, Any],
        is_current: bool = False,
    ) -> ProcessingProfile:
        if version < 1:
            raise ValueError("profile version must be positive")
        snapshot = canonical_json(payload)
        return self._add(
            ProcessingProfile(
                site_id=site_id,
                version=version,
                payload_json=snapshot,
                is_current=is_current,
            )
        )

    def get(self, profile_id: str) -> ProcessingProfile | None:
        return self.session.get(ProcessingProfile, profile_id)

    def list_for_site(self, site_id: str) -> list[ProcessingProfile]:
        statement = (
            select(ProcessingProfile)
            .where(ProcessingProfile.site_id == site_id)
            .order_by(ProcessingProfile.version)
        )
        return list(self.session.scalars(statement))


class SourceFileRepository(_Repository):
    def add(
        self,
        *,
        site_id: str,
        path: str,
        fingerprint: str,
        size_bytes: int,
        mtime_ns: int,
    ) -> SourceFile:
        return self._add(
            SourceFile(
                site_id=site_id,
                path=path,
                fingerprint=_sha256(fingerprint, field="source fingerprint SHA-256"),
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
            )
        )

    def get(self, source_file_id: str) -> SourceFile | None:
        return self.session.get(SourceFile, source_file_id)

    def find(self, *, site_id: str, fingerprint: str) -> SourceFile | None:
        statement = select(SourceFile).where(
            SourceFile.site_id == site_id,
            SourceFile.fingerprint == fingerprint,
        )
        return self.session.scalar(statement)


class JobRepository(_Repository):
    def add(
        self,
        *,
        source_file_id: str,
        processing_profile_id: str,
        pipeline_version: str,
        status: str = "queued",
    ) -> Job:
        _require_matching_source_profile(
            self.session, source_file_id, processing_profile_id
        )
        return self._add(
            Job(
                source_file_id=source_file_id,
                processing_profile_id=processing_profile_id,
                pipeline_version=pipeline_version,
                status=status,
            )
        )

    def get(self, job_id: str) -> Job | None:
        return self.session.get(Job, job_id)

    def list_by_status(self, status: str) -> list[Job]:
        statement = (
            select(Job).where(Job.status == status).order_by(Job.created_at, Job.id)
        )
        return list(self.session.scalars(statement))


class RunRepository(_Repository):
    def add(
        self,
        *,
        source_file_id: str,
        processing_profile_id: str,
        pipeline_version: str,
        requested_backend: str,
        actual_backend: str,
        job_id: str | None = None,
        model_version_id: str | None = None,
        backend_fallback_reason: str | None = None,
        status: str = "pending",
        timings: dict[str, float] | None = None,
    ) -> ProcessingRun:
        source, profile = _require_matching_source_profile(
            self.session, source_file_id, processing_profile_id
        )
        return self._add(
            ProcessingRun(
                job_id=job_id,
                source_file_id=source.id,
                processing_profile_id=profile.id,
                model_version_id=model_version_id,
                source_fingerprint=source.fingerprint,
                profile_version=profile.version,
                pipeline_version=pipeline_version,
                requested_backend=requested_backend,
                actual_backend=actual_backend,
                backend_fallback_reason=backend_fallback_reason,
                status=status,
                timings_json=canonical_json(timings or {}),
            )
        )

    def get(self, run_id: str) -> ProcessingRun | None:
        return self.session.get(ProcessingRun, run_id)

    def find_by_key(
        self,
        *,
        source_file_id: str,
        processing_profile_id: str,
        pipeline_version: str,
    ) -> ProcessingRun | None:
        statement = select(ProcessingRun).where(
            ProcessingRun.source_file_id == source_file_id,
            ProcessingRun.processing_profile_id == processing_profile_id,
            ProcessingRun.pipeline_version == pipeline_version,
        )
        return self.session.scalar(statement)


class EventRepository(_Repository):
    def add(
        self,
        *,
        run_id: str,
        event_uid: str,
        start_sample: int,
        end_sample: int,
        channel_start: int,
        channel_stop: int,
        occurred_at: datetime,
        features: dict[str, float | str] | None = None,
    ) -> Event:
        return self._add(
            Event(
                processing_run_id=run_id,
                event_uid=event_uid,
                start_sample=start_sample,
                end_sample=end_sample,
                channel_start=channel_start,
                channel_stop=channel_stop,
                occurred_at=normalize_utc(occurred_at),
                features_json=canonical_json(features or {}),
            )
        )

    def get(self, event_id: str) -> Event | None:
        return self.session.get(Event, event_id)

    def list_for_run(self, run_id: str) -> list[Event]:
        statement = (
            select(Event)
            .where(Event.processing_run_id == run_id)
            .order_by(Event.occurred_at, Event.id)
        )
        return list(self.session.scalars(statement))

    def add_evidence(
        self,
        event_id: str,
        *,
        detector: str,
        score: float,
        payload: dict[str, Any],
    ) -> DetectionEvidence:
        event = self._require(event_id)
        row = DetectionEvidence(
            detector=detector,
            score=score,
            payload_json=canonical_json(payload),
        )
        event.evidence.append(row)
        self.session.flush()
        return row

    def add_classification(
        self,
        event_id: str,
        *,
        label: str,
        decision_score: float,
        method: str,
        reasons: list[str] | tuple[str, ...] = (),
        model_version_id: str | None = None,
        created_at: datetime | None = None,
    ) -> Classification:
        event = self._require(event_id)
        row = Classification(
            label=_label(label, MACHINE_LABELS),
            decision_score=decision_score,
            method=method,
            reasons_json=canonical_json(list(reasons)),
            model_version_id=model_version_id,
        )
        if created_at is not None:
            row.created_at = normalize_utc(created_at)
        event.classifications.append(row)
        self.session.flush()
        return row

    def _require(self, event_id: str) -> Event:
        event = self.get(event_id)
        if event is None:
            raise LookupError(f"event not found: {event_id}")
        return event


class ReviewRepository(_Repository):
    def add(
        self,
        event_id: str,
        *,
        label: str,
        note: str | None = None,
        admin_id: int | None = None,
        created_at: datetime | None = None,
    ) -> Review:
        event = self.session.get(Event, event_id)
        if event is None:
            raise LookupError(f"event not found: {event_id}")
        row = Review(
            admin_id=admin_id,
            label=_label(label, SUPERVISED_LABELS),
            note=note,
        )
        if created_at is not None:
            row.created_at = normalize_utc(created_at)
        event.reviews.append(row)
        self.session.flush()
        return row

    def list_for_event(self, event_id: str) -> list[Review]:
        statement = (
            select(Review)
            .where(Review.event_id == event_id)
            .order_by(Review.created_at, Review.id)
        )
        return list(self.session.scalars(statement))


class ModelRepository(_Repository):
    def add(
        self,
        *,
        site_id: str,
        version: str,
        artifact_path: str,
        bundle_sha256: str,
        status: str,
        dataset_sha256: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> ModelVersion:
        if dataset_sha256 is not None:
            _sha256(dataset_sha256, field="dataset SHA-256")
        return self._add(
            ModelVersion(
                site_id=site_id,
                version=version,
                artifact_path=artifact_path,
                bundle_sha256=_sha256(bundle_sha256),
                dataset_sha256=dataset_sha256,
                status=status,
                metrics_json=canonical_json(metrics or {}),
            )
        )

    def get(self, model_id: str) -> ModelVersion | None:
        return self.session.get(ModelVersion, model_id)

    def list_for_site(self, site_id: str) -> list[ModelVersion]:
        statement = (
            select(ModelVersion)
            .where(ModelVersion.site_id == site_id)
            .order_by(ModelVersion.created_at, ModelVersion.id)
        )
        return list(self.session.scalars(statement))

    def set_status(
        self,
        model_id: str,
        *,
        status: str,
        published_at: datetime | None = None,
    ) -> ModelVersion:
        model = self.get(model_id)
        if model is None:
            raise LookupError(f"model not found: {model_id}")
        model.status = status
        model.published_at = (
            normalize_utc(published_at) if published_at is not None else None
        )
        self.session.flush()
        return model


class ArtifactRepository(_Repository):
    def add(
        self,
        *,
        event_id: str,
        kind: str,
        path: str,
        media_type: str,
        sha256: str,
    ) -> Artifact:
        return self._add(
            Artifact(
                event_id=event_id,
                kind=kind,
                path=path,
                media_type=media_type,
                sha256=_sha256(sha256),
            )
        )

    def get(self, artifact_id: str) -> Artifact | None:
        return self.session.get(Artifact, artifact_id)

    def list_for_event(self, event_id: str) -> list[Artifact]:
        statement = (
            select(Artifact)
            .where(Artifact.event_id == event_id)
            .order_by(Artifact.kind)
        )
        return list(self.session.scalars(statement))


class Repositories:
    def __init__(self, session: Session) -> None:
        self.admins = AdminRepository(session)
        self.sites = SiteRepository(session)
        self.profiles = ProfileRepository(session)
        self.source_files = SourceFileRepository(session)
        self.jobs = JobRepository(session)
        self.runs = RunRepository(session)
        self.events = EventRepository(session)
        self.reviews = ReviewRepository(session)
        self.models = ModelRepository(session)
        self.artifacts = ArtifactRepository(session)


def _require_matching_source_profile(
    session: Session,
    source_file_id: str,
    processing_profile_id: str,
) -> tuple[SourceFile, ProcessingProfile]:
    source = session.get(SourceFile, source_file_id)
    if source is None:
        raise LookupError(f"source file not found: {source_file_id}")
    profile = session.get(ProcessingProfile, processing_profile_id)
    if profile is None:
        raise LookupError(f"processing profile not found: {processing_profile_id}")
    if source.site_id != profile.site_id:
        raise ValueError("source file and processing profile belong to different sites")
    return source, profile
