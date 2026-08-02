from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.api.schemas import JobEnqueueRequest, JobResponse
from das_sentinel.services.jobs import (
    JOB_STATUSES,
    JobConflictError,
    JobNotFoundError,
    JobService,
    JobServiceError,
    JobSnapshot,
    JobValidationError,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def enqueue_job(
    payload: JobEnqueueRequest,
    request: Request,
    admin: CsrfAdmin,
) -> JobResponse:
    del admin
    try:
        async with request.app.state.write_coordinator.lock:
            result = await run_in_threadpool(
                _enqueue,
                request.app.state.session_factory,
                request.app.state.settings.app_data_dir / "artifacts",
                payload,
            )
    except JobServiceError as error:
        raise job_http_exception(error) from error
    return job_response(result)


@router.get("", response_model=list[JobResponse])
def list_jobs(
    request: Request,
    admin: AuthenticatedAdmin,
    job_status: str | None = Query(default=None, alias="status"),
) -> list[JobResponse]:
    del admin
    if job_status is not None and job_status not in JOB_STATUSES:
        raise job_http_exception(
            JobValidationError(f"unsupported job status: {job_status}")
        )
    with request.app.state.session_factory() as session:
        service = JobService(
            session,
            request.app.state.settings.app_data_dir / "artifacts",
        )
        return [job_response(item) for item in service.list(status=job_status)]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
) -> JobResponse:
    del admin
    try:
        with request.app.state.session_factory() as session:
            service = JobService(
                session,
                request.app.state.settings.app_data_dir / "artifacts",
            )
            result = service.get(job_id)
    except JobServiceError as error:
        raise job_http_exception(error) from error
    return job_response(result)


@router.post("/{job_id}/retry", response_model=JobResponse)
async def retry_job(
    job_id: str,
    request: Request,
    admin: CsrfAdmin,
) -> JobResponse:
    del admin
    try:
        async with request.app.state.write_coordinator.lock:
            result = await run_in_threadpool(
                _transition,
                request.app.state.session_factory,
                request.app.state.settings.app_data_dir / "artifacts",
                "retry",
                job_id,
            )
    except JobServiceError as error:
        raise job_http_exception(error) from error
    return job_response(result)


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: str,
    request: Request,
    admin: CsrfAdmin,
) -> JobResponse:
    del admin
    try:
        async with request.app.state.write_coordinator.lock:
            result = await run_in_threadpool(
                _transition,
                request.app.state.session_factory,
                request.app.state.settings.app_data_dir / "artifacts",
                "cancel",
                job_id,
            )
    except JobServiceError as error:
        raise job_http_exception(error) from error
    return job_response(result)


def _enqueue(sessions, artifact_root, payload: JobEnqueueRequest) -> JobSnapshot:
    with sessions.begin() as session:
        return JobService(session, artifact_root).enqueue(
            payload.source_file_id,
            payload.processing_profile_id,
            payload.pipeline_version,
        )


def _transition(
    sessions,
    artifact_root,
    operation: str,
    job_id: str,
) -> JobSnapshot:
    with sessions.begin() as session:
        service = JobService(session, artifact_root)
        if operation == "retry":
            return service.retry(job_id)
        return service.cancel(job_id)


def job_response(snapshot: JobSnapshot) -> JobResponse:
    return JobResponse(
        id=snapshot.id,
        source_file_id=snapshot.source_file_id,
        processing_profile_id=snapshot.processing_profile_id,
        pipeline_version=snapshot.pipeline_version,
        status=snapshot.status,
        attempt_count=snapshot.attempt_count,
        worker_id=snapshot.worker_id,
        heartbeat_at=snapshot.heartbeat_at,
        progress_numerator=snapshot.progress_numerator,
        progress_denominator=snapshot.progress_denominator,
        error_code=snapshot.error_code,
        error_message=snapshot.error_message,
        retry_eligible=snapshot.retry_eligible,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
    )


def job_http_exception(error: JobServiceError) -> HTTPException:
    if isinstance(error, JobNotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        )
    if isinstance(error, JobConflictError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=str(error),
    )
