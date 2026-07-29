from __future__ import annotations

import json
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from das_sentinel.persistence.models import (
    Job,
    ProcessingProfile,
    Site,
    SourceFile,
    freeze_json,
    normalize_utc,
    utc_now,
)

MAX_ATTEMPTS = 3
STALE_AFTER = timedelta(minutes=2)
MAX_ERROR_CODE_LENGTH = 128
MAX_ERROR_MESSAGE_LENGTH = 4096
JOB_STATUSES = frozenset({"queued", "running", "succeeded", "failed", "cancelled"})


class JobServiceError(RuntimeError):
    """Base class for job errors exposed at the API boundary."""


class JobNotFoundError(JobServiceError):
    pass


class JobConflictError(JobServiceError):
    pass


class JobValidationError(JobServiceError):
    pass


@dataclass(frozen=True)
class JobSnapshot:
    id: str
    source_file_id: str
    processing_profile_id: str
    pipeline_version: str
    status: str
    attempt_count: int
    worker_id: str | None
    heartbeat_at: datetime | None
    progress_numerator: int
    progress_denominator: int
    error_code: str | None
    error_message: str | None
    retry_eligible: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class JobClaim:
    id: str
    source_file_id: str
    source_path: str
    source_fingerprint: str
    site_id: str
    site_timezone: str
    cable_type: str
    processing_profile_id: str
    profile_version: int
    profile_payload: Mapping[str, Any]
    pipeline_version: str
    attempt_count: int
    worker_id: str
    heartbeat_at: datetime


class JobService:
    """Transaction-scoped state machine for persistent jobs.

    Methods flush changes so constraints fail at the call boundary. They never
    commit; callers own the surrounding transaction.
    """

    def __init__(self, session: Session, artifact_root: Path) -> None:
        self.session = session
        self.artifact_root = Path(artifact_root)

    def enqueue(
        self,
        source_file_id: str,
        processing_profile_id: str,
        pipeline_version: str,
    ) -> JobSnapshot:
        pipeline_version = _bounded_nonblank(
            pipeline_version,
            field="pipeline version",
            maximum=128,
        )
        source = self.session.get(SourceFile, source_file_id)
        if source is None:
            raise JobNotFoundError(f"source file not found: {source_file_id}")
        profile = self.session.get(ProcessingProfile, processing_profile_id)
        if profile is None:
            raise JobNotFoundError(
                f"processing profile not found: {processing_profile_id}"
            )
        if source.site_id != profile.site_id:
            raise JobValidationError(
                "source file and processing profile belong to different sites"
            )
        site = self.session.get(Site, source.site_id)
        if site is None:
            raise JobNotFoundError(f"site not found: {source.site_id}")
        if not site.enabled:
            raise JobValidationError("site is disabled")

        statement = (
            select(Job)
            .where(
                Job.source_file_id == source.id,
                Job.processing_profile_id == profile.id,
                Job.pipeline_version == pipeline_version,
            )
            .order_by(Job.created_at, Job.id)
        )
        existing = self.session.scalars(statement).first()
        if existing is not None:
            return _snapshot(existing)

        row = Job(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version=pipeline_version,
        )
        self.session.add(row)
        self.session.flush()
        return _snapshot(row)

    def get(self, job_id: str) -> JobSnapshot:
        return _snapshot(self._require(job_id))

    def list(self, *, status: str | None = None) -> list[JobSnapshot]:
        if status is not None and status not in JOB_STATUSES:
            raise JobValidationError(f"unsupported job status: {status}")
        statement = select(Job)
        if status is not None:
            statement = statement.where(Job.status == status)
        statement = statement.order_by(Job.created_at, Job.id)
        return [_snapshot(row) for row in self.session.scalars(statement)]

    def claim(
        self,
        worker_id: str,
        *,
        now: datetime | None = None,
    ) -> JobClaim | None:
        worker_id = _bounded_nonblank(worker_id, field="worker ID", maximum=200)
        claimed_at = _utc(now)
        statement = (
            select(Job)
            .where(
                Job.status == "queued",
                Job.retry_eligible.is_(True),
                Job.attempt_count < MAX_ATTEMPTS,
            )
            .order_by(Job.created_at, Job.id)
            .limit(1)
        )
        row = self.session.scalar(statement)
        if row is None:
            return None

        row.status = "running"
        row.attempt_count += 1
        row.worker_id = worker_id
        row.heartbeat_at = claimed_at
        row.updated_at = claimed_at
        row.error_code = None
        row.error_message = None
        self.session.flush()
        return self._claim(row)

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        numerator: int,
        denominator: int,
        *,
        now: datetime | None = None,
    ) -> JobSnapshot:
        if (
            isinstance(numerator, bool)
            or isinstance(denominator, bool)
            or not isinstance(numerator, int)
            or not isinstance(denominator, int)
            or numerator < 0
            or denominator < 0
            or numerator > denominator
        ):
            raise JobValidationError(
                "progress must satisfy 0 <= numerator <= denominator"
            )
        row = self._require_running_owner(job_id, worker_id)
        heartbeat_at = _utc(now)
        row.progress_numerator = numerator
        row.progress_denominator = denominator
        row.heartbeat_at = heartbeat_at
        row.updated_at = heartbeat_at
        self.session.flush()
        return _snapshot(row)

    def complete(
        self,
        job_id: str,
        worker_id: str,
        profile_version: int,
        pipeline_version: str,
        artifact_paths: Sequence[str | Path] = (),
        *,
        now: datetime | None = None,
    ) -> JobSnapshot:
        row = self._require_running_owner(job_id, worker_id)
        profile = self.session.get(ProcessingProfile, row.processing_profile_id)
        if profile is None:
            raise JobNotFoundError(
                f"processing profile not found: {row.processing_profile_id}"
            )
        if (
            isinstance(profile_version, bool)
            or not isinstance(profile_version, int)
            or profile.version != profile_version
        ):
            raise JobConflictError("profile version does not match claimed job")
        if row.pipeline_version != pipeline_version:
            raise JobConflictError("pipeline version does not match claimed job")

        for artifact_path in artifact_paths:
            self._validate_artifact_path(artifact_path)

        finished_at = _utc(now)
        row.status = "succeeded"
        row.worker_id = None
        row.heartbeat_at = None
        row.retry_eligible = False
        row.error_code = None
        row.error_message = None
        row.updated_at = finished_at
        self.session.flush()
        return _snapshot(row)

    def fail(
        self,
        job_id: str,
        worker_id: str,
        *,
        error_code: str,
        error_message: str,
        retry_eligible: bool = True,
        now: datetime | None = None,
    ) -> JobSnapshot:
        error_code = _bounded_nonblank(
            error_code,
            field="error code",
            maximum=MAX_ERROR_CODE_LENGTH,
        )
        error_message = _bounded_nonblank(
            error_message,
            field="error message",
            maximum=MAX_ERROR_MESSAGE_LENGTH,
        )
        if not isinstance(retry_eligible, bool):
            raise JobValidationError("retry eligibility must be a boolean")
        row = self._require_running_owner(job_id, worker_id)
        failed_at = _utc(now)
        row.status = "failed"
        row.worker_id = None
        row.heartbeat_at = None
        row.error_code = error_code
        row.error_message = error_message
        row.retry_eligible = retry_eligible and row.attempt_count < MAX_ATTEMPTS
        row.updated_at = failed_at
        self.session.flush()
        return _snapshot(row)

    def retry(
        self,
        job_id: str,
        *,
        now: datetime | None = None,
    ) -> JobSnapshot:
        row = self._require(job_id)
        if (
            row.status != "failed"
            or not row.retry_eligible
            or row.attempt_count >= MAX_ATTEMPTS
        ):
            raise JobConflictError("job is not eligible for retry")
        changed_at = _utc(now)
        row.status = "queued"
        row.worker_id = None
        row.heartbeat_at = None
        row.progress_numerator = 0
        row.progress_denominator = 0
        row.error_code = None
        row.error_message = None
        row.retry_eligible = True
        row.updated_at = changed_at
        self.session.flush()
        return _snapshot(row)

    def cancel(
        self,
        job_id: str,
        *,
        now: datetime | None = None,
    ) -> JobSnapshot:
        row = self._require(job_id)
        if row.status not in {"queued", "running"}:
            raise JobConflictError("job cannot be cancelled from its current state")
        changed_at = _utc(now)
        row.status = "cancelled"
        row.worker_id = None
        row.heartbeat_at = None
        row.retry_eligible = False
        row.updated_at = changed_at
        self.session.flush()
        return _snapshot(row)

    def recover_stale(
        self,
        *,
        now: datetime | None = None,
    ) -> list[str]:
        recovered_at = _utc(now)
        cutoff = recovered_at - STALE_AFTER
        statement = (
            select(Job)
            .where(
                Job.status == "running",
                or_(Job.heartbeat_at.is_(None), Job.heartbeat_at < cutoff),
            )
            .order_by(Job.created_at, Job.id)
        )
        recovered: list[str] = []
        for row in self.session.scalars(statement):
            recovered.append(row.id)
            row.worker_id = None
            row.heartbeat_at = None
            row.updated_at = recovered_at
            if row.attempt_count < MAX_ATTEMPTS:
                row.status = "queued"
                row.retry_eligible = True
                row.error_code = None
                row.error_message = None
            else:
                row.status = "failed"
                row.retry_eligible = False
                row.error_code = "stale_worker"
                row.error_message = "worker heartbeat expired after maximum attempts"
        self.session.flush()
        return recovered

    def _require(self, job_id: str) -> Job:
        row = self.session.get(Job, job_id)
        if row is None:
            raise JobNotFoundError(f"job not found: {job_id}")
        return row

    def _require_running_owner(self, job_id: str, worker_id: str) -> Job:
        row = self._require(job_id)
        if row.status != "running":
            raise JobConflictError("job is not running")
        if not worker_id or row.worker_id != worker_id:
            raise JobConflictError("worker is not the job owner")
        return row

    def _claim(self, row: Job) -> JobClaim:
        source = self.session.get(SourceFile, row.source_file_id)
        profile = self.session.get(ProcessingProfile, row.processing_profile_id)
        if source is None:
            raise JobNotFoundError(f"source file not found: {row.source_file_id}")
        if profile is None:
            raise JobNotFoundError(
                f"processing profile not found: {row.processing_profile_id}"
            )
        site = self.session.get(Site, source.site_id)
        if site is None:
            raise JobNotFoundError(f"site not found: {source.site_id}")
        if row.worker_id is None or row.heartbeat_at is None:
            raise RuntimeError("claimed job is missing worker ownership")
        payload = freeze_json(json.loads(profile.payload_json))
        return JobClaim(
            id=row.id,
            source_file_id=source.id,
            source_path=source.path,
            source_fingerprint=source.fingerprint,
            site_id=site.id,
            site_timezone=site.timezone,
            cable_type=site.cable_type,
            processing_profile_id=profile.id,
            profile_version=profile.version,
            profile_payload=payload,
            pipeline_version=row.pipeline_version,
            attempt_count=row.attempt_count,
            worker_id=row.worker_id,
            heartbeat_at=row.heartbeat_at,
        )

    def _validate_artifact_path(self, submitted: str | Path) -> Path:
        candidate = Path(submitted)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise JobValidationError("artifact path must be absolute and normalized")
        try:
            root = self.artifact_root.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise JobValidationError(
                "configured artifact root must be an existing directory"
            ) from error
        if not root.is_dir():
            raise JobValidationError(
                "configured artifact root must be an existing directory"
            )
        try:
            lexical = candidate.absolute()
            if not lexical.is_relative_to(root):
                raise JobValidationError("artifact path escapes configured root")
            relative = lexical.relative_to(root)
            current = root
            for component in relative.parts:
                current = current / component
                metadata = current.lstat()
                attributes = getattr(metadata, "st_file_attributes", 0)
                reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                if stat.S_ISLNK(metadata.st_mode) or attributes & reparse_flag:
                    raise JobValidationError(
                        "artifact path must not traverse a symlink or junction"
                    )
            resolved = candidate.resolve(strict=True)
        except JobValidationError:
            raise
        except (OSError, RuntimeError) as error:
            raise JobValidationError(
                "artifact path must identify an existing regular file"
            ) from error
        if not resolved.is_relative_to(root):
            raise JobValidationError("artifact path escapes configured root")
        try:
            metadata = resolved.stat(follow_symlinks=False)
        except OSError as error:
            raise JobValidationError(
                "artifact path must identify an existing regular file"
            ) from error
        if not stat.S_ISREG(metadata.st_mode):
            raise JobValidationError(
                "artifact path must identify an existing regular file"
            )
        return resolved


def _snapshot(row: Job) -> JobSnapshot:
    return JobSnapshot(
        id=row.id,
        source_file_id=row.source_file_id,
        processing_profile_id=row.processing_profile_id,
        pipeline_version=row.pipeline_version,
        status=row.status,
        attempt_count=row.attempt_count,
        worker_id=row.worker_id,
        heartbeat_at=row.heartbeat_at,
        progress_numerator=row.progress_numerator,
        progress_denominator=row.progress_denominator,
        error_code=row.error_code,
        error_message=row.error_message,
        retry_eligible=row.retry_eligible,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return utc_now()
    try:
        return normalize_utc(value)
    except ValueError as error:
        raise JobValidationError("timestamp must be timezone-aware") from error


def _bounded_nonblank(value: str, *, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise JobValidationError(f"{field} must be a string")
    stripped = value.strip()
    if not stripped or len(stripped) > maximum:
        raise JobValidationError(
            f"{field} must contain between 1 and {maximum} characters"
        )
    return stripped
