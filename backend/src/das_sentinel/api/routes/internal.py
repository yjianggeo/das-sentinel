from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.concurrency import run_in_threadpool

from das_sentinel.api.routes.jobs import job_http_exception, job_response
from das_sentinel.api.schemas import (
    JobClaimResponse,
    JobResponse,
    WorkerClaimRequest,
    WorkerCompleteRequest,
    WorkerFailRequest,
    WorkerHeartbeatRequest,
    WorkerResultRequest,
)
from das_sentinel.services.jobs import (
    JobClaim,
    JobService,
    JobServiceError,
    JobSnapshot,
)
from das_sentinel.services.results import ResultService

router = APIRouter(prefix="/internal/jobs", tags=["internal-jobs"])


def require_worker_token(
    request: Request,
    x_worker_token: Annotated[str | None, Header()] = None,
) -> None:
    expected = request.app.state.settings.worker_token
    supplied = x_worker_token or ""
    if not supplied or not hmac.compare_digest(expected, supplied):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )


WorkerToken = Annotated[None, Depends(require_worker_token)]


@router.post(
    "/claim",
    response_model=JobClaimResponse,
    responses={status.HTTP_204_NO_CONTENT: {"description": "No queued job"}},
)
async def claim_job(
    payload: WorkerClaimRequest,
    request: Request,
    worker_token: WorkerToken,
) -> JobClaimResponse | Response:
    del worker_token
    try:
        async with request.app.state.write_coordinator.lock:
            result = await run_in_threadpool(
                _claim,
                request.app.state.session_factory,
                request.app.state.settings.app_data_dir / "artifacts",
                payload.worker_id,
            )
    except JobServiceError as error:
        raise job_http_exception(error) from error
    if result is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return claim_response(result)


@router.post("/{job_id}/heartbeat", response_model=JobResponse)
async def heartbeat_job(
    job_id: str,
    payload: WorkerHeartbeatRequest,
    request: Request,
    worker_token: WorkerToken,
) -> JobResponse:
    del worker_token
    try:
        async with request.app.state.write_coordinator.lock:
            result = await run_in_threadpool(
                _heartbeat,
                request.app.state.session_factory,
                request.app.state.settings.app_data_dir / "artifacts",
                job_id,
                payload,
            )
    except JobServiceError as error:
        raise job_http_exception(error) from error
    return job_response(result)


@router.post("/{job_id}/complete", response_model=JobResponse)
async def complete_job(
    job_id: str,
    payload: WorkerCompleteRequest,
    request: Request,
    worker_token: WorkerToken,
) -> JobResponse:
    del worker_token
    try:
        async with request.app.state.write_coordinator.lock:
            result = await run_in_threadpool(
                _complete,
                request.app.state.session_factory,
                request.app.state.settings.app_data_dir / "artifacts",
                job_id,
                payload,
            )
    except JobServiceError as error:
        raise job_http_exception(error) from error
    return job_response(result)


@router.post("/{job_id}/fail", response_model=JobResponse)
async def fail_job(
    job_id: str,
    payload: WorkerFailRequest,
    request: Request,
    worker_token: WorkerToken,
) -> JobResponse:
    del worker_token
    try:
        async with request.app.state.write_coordinator.lock:
            result = await run_in_threadpool(
                _fail,
                request.app.state.session_factory,
                request.app.state.settings.app_data_dir / "artifacts",
                job_id,
                payload,
            )
    except JobServiceError as error:
        raise job_http_exception(error) from error
    return job_response(result)


@router.post("/{job_id}/result", response_model=JobResponse)
async def submit_job_result(
    job_id: str,
    payload: WorkerResultRequest,
    request: Request,
    worker_token: WorkerToken,
) -> JobResponse:
    del worker_token
    try:
        async with request.app.state.write_coordinator.lock:
            result = await run_in_threadpool(
                _persist_result,
                request.app.state.session_factory,
                request.app.state.settings.app_data_dir / "artifacts",
                job_id,
                payload,
            )
    except JobServiceError as error:
        raise job_http_exception(error) from error
    return job_response(result)


def _claim(sessions, artifact_root, worker_id: str) -> JobClaim | None:
    with sessions.begin() as session:
        return JobService(session, artifact_root).claim(worker_id)


def _heartbeat(
    sessions,
    artifact_root,
    job_id: str,
    payload: WorkerHeartbeatRequest,
) -> JobSnapshot:
    with sessions.begin() as session:
        return JobService(session, artifact_root).heartbeat(
            job_id,
            payload.worker_id,
            payload.numerator,
            payload.denominator,
        )


def _complete(
    sessions,
    artifact_root,
    job_id: str,
    payload: WorkerCompleteRequest,
) -> JobSnapshot:
    with sessions.begin() as session:
        return JobService(session, artifact_root).complete(
            job_id,
            payload.worker_id,
            payload.profile_version,
            payload.pipeline_version,
            payload.artifact_paths,
        )


def _fail(
    sessions,
    artifact_root,
    job_id: str,
    payload: WorkerFailRequest,
) -> JobSnapshot:
    with sessions.begin() as session:
        return JobService(session, artifact_root).fail(
            job_id,
            payload.worker_id,
            error_code=payload.error_code,
            error_message=payload.error_message,
            retry_eligible=payload.retry_eligible,
        )


def _persist_result(
    sessions,
    artifact_root,
    job_id: str,
    payload: WorkerResultRequest,
) -> JobSnapshot:
    with sessions.begin() as session:
        return ResultService(session, artifact_root).persist(job_id, payload)


def claim_response(claim: JobClaim) -> JobClaimResponse:
    return JobClaimResponse(
        id=claim.id,
        source_file_id=claim.source_file_id,
        source_path=claim.source_path,
        source_fingerprint=claim.source_fingerprint,
        source_size_bytes=claim.source_size_bytes,
        source_mtime_ns=claim.source_mtime_ns,
        site_id=claim.site_id,
        site_data_root=claim.site_data_root,
        site_timezone=claim.site_timezone,
        cable_type=claim.cable_type,
        processing_profile_id=claim.processing_profile_id,
        profile_version=claim.profile_version,
        profile_payload=_thaw(claim.profile_payload),
        pipeline_version=claim.pipeline_version,
        attempt_count=claim.attempt_count,
        worker_id=claim.worker_id,
        heartbeat_at=claim.heartbeat_at,
    )


def _thaw(value):
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _thaw(item) for key, item in value.items()}
    return value
