from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
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
MAX_SQLITE_INTEGER = 2**63 - 1
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
    source_size_bytes: int
    source_mtime_ns: int
    site_id: str
    site_data_root: str
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
            or denominator > MAX_SQLITE_INTEGER
        ):
            raise JobValidationError(
                "progress must use strict integers satisfying "
                "0 <= numerator <= denominator <= 2**63 - 1"
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

        with _hold_artifact_paths(self.artifact_root, artifact_paths):
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
            source_size_bytes=source.size_bytes,
            source_mtime_ns=source.mtime_ns,
            site_id=site.id,
            site_data_root=site.data_path,
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


class _ArtifactHandle:
    def __init__(self, value: int, close) -> None:
        self.value = value
        self._close = close

    def close(self) -> None:
        if self.value is not None:
            value = self.value
            self.value = None
            self._close(value)


@contextmanager
def _hold_artifact_paths(
    artifact_root: Path,
    submitted_paths: Sequence[str | Path],
):
    if not submitted_paths:
        yield
        return

    root = Path(artifact_root)
    if ".." in root.parts:
        raise JobValidationError("artifact root must be absolute and normalized")
    root = root.absolute()
    with ExitStack() as handles:
        try:
            root_handle = _open_artifact_component(
                root,
                None,
                expect_directory=True,
            )
        except OSError as error:
            raise JobValidationError(
                "artifact root must be an existing non-reparse directory"
            ) from error
        handles.callback(root_handle.close)

        for submitted in submitted_paths:
            candidate = Path(submitted)
            if not candidate.is_absolute() or ".." in candidate.parts:
                raise JobValidationError(
                    "artifact path must be absolute and normalized"
                )
            candidate = candidate.absolute()
            try:
                relative = candidate.relative_to(root)
            except ValueError as error:
                raise JobValidationError(
                    "artifact path escapes configured artifact root"
                ) from error
            if not relative.parts:
                raise JobValidationError(
                    "artifact path must identify an existing regular file"
                )

            parent_handle = root_handle
            current = root
            for index, component in enumerate(relative.parts):
                current = current / component
                expect_directory = index < len(relative.parts) - 1
                try:
                    opened = _open_artifact_component(
                        current,
                        parent_handle,
                        expect_directory=expect_directory,
                    )
                except OSError as error:
                    kind = "parent directory" if expect_directory else "regular file"
                    raise JobValidationError(
                        f"artifact path must use an existing non-reparse {kind}"
                    ) from error
                handles.callback(opened.close)
                parent_handle = opened
        yield


def _open_artifact_component(
    path: Path,
    parent_handle: _ArtifactHandle | None,
    *,
    expect_directory: bool,
) -> _ArtifactHandle:
    if os.name == "nt":
        return _open_windows_artifact_component(path, expect_directory)
    return _open_posix_artifact_component(
        path,
        parent_handle,
        expect_directory,
    )


def _open_posix_artifact_component(
    path: Path,
    parent_handle: _ArtifactHandle | None,
    expect_directory: bool,
) -> _ArtifactHandle:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    if expect_directory:
        flags |= os.O_DIRECTORY
    opened = path if parent_handle is None else path.name
    descriptor = os.open(
        opened,
        flags,
        dir_fd=None if parent_handle is None else parent_handle.value,
    )
    try:
        mode = os.fstat(descriptor).st_mode
        valid = stat.S_ISDIR(mode) if expect_directory else stat.S_ISREG(mode)
        if not valid:
            raise OSError("artifact component has the wrong file type")
        return _ArtifactHandle(descriptor, os.close)
    except BaseException:
        os.close(descriptor)
        raise


def _open_windows_artifact_component(
    path: Path,
    expect_directory: bool,
) -> _ArtifactHandle:
    import ctypes
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ByHandleFileInformation),
    ]
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    generic_read = 0x80000000
    share_read_write = 0x00000001 | 0x00000002
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    file_flag_open_reparse_point = 0x00200000
    handle = create_file(
        str(path),
        generic_read,
        share_read_write,
        None,
        open_existing,
        file_flag_backup_semantics | file_flag_open_reparse_point,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        information = ByHandleFileInformation()
        if not get_information(handle, ctypes.byref(information)):
            raise ctypes.WinError(ctypes.get_last_error())
        file_attribute_directory = 0x00000010
        file_attribute_reparse_point = 0x00000400
        is_directory = bool(
            information.file_attributes & file_attribute_directory
        )
        is_reparse = bool(
            information.file_attributes & file_attribute_reparse_point
        )
        if is_reparse or is_directory != expect_directory:
            raise OSError(
                "artifact component has the wrong type or is a reparse point"
            )
        result = _ArtifactHandle(handle, close_handle)
        handle = None
        return result
    finally:
        if handle is not None:
            close_handle(handle)
