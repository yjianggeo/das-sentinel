from __future__ import annotations

from pathlib import Path
from time import monotonic

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from das_sentinel.api.auth import CsrfAdmin
from das_sentinel.api.routes.jobs import job_response
from das_sentinel.api.schemas import JobResponse, ScanRequest
from das_sentinel.services.discovery import (
    DiscoveryConfigurationError,
    DiscoveryPathError,
    StableFileTracker,
    discover_batch,
    enqueue_discovered,
    load_discovery_site,
)

router = APIRouter(tags=["scans"])


@router.post(
    "/sites/{site_id}/scan",
    response_model=list[JobResponse],
    status_code=status.HTTP_201_CREATED,
)
async def scan_site(
    site_id: str,
    payload: ScanRequest,
    request: Request,
    admin: CsrfAdmin,
) -> list[JobResponse]:
    del admin
    trackers = request.app.state.discovery_trackers
    tracker = trackers.setdefault(site_id, StableFileTracker())
    try:
        files = await run_in_threadpool(
            _discover,
            request.app.state.session_factory,
            site_id,
            payload.path,
            tracker,
        )
    except DiscoveryConfigurationError as error:
        if "not found" in str(error):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found",
            ) from error
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except DiscoveryPathError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    try:
        async with request.app.state.write_coordinator.lock:
            snapshots = await run_in_threadpool(
                _enqueue,
                request.app.state.session_factory,
                request.app.state.settings.app_data_dir / "artifacts",
                site_id,
                files,
                request.app.state.settings.pipeline_version,
            )
    except DiscoveryConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    return [job_response(snapshot) for snapshot in snapshots]


def _discover(sessions, site_id: str, path: str | None, tracker: StableFileTracker):
    with sessions() as session:
        site = load_discovery_site(session, site_id)
    return discover_batch(
        site,
        Path(path) if path is not None else site.data_root,
        tracker=tracker,
        now=monotonic(),
    )


def _enqueue(sessions, artifact_root, site_id: str, files, pipeline_version: str):
    if not files:
        return ()
    with sessions.begin() as session:
        site = load_discovery_site(session, site_id)
        return enqueue_discovered(
            session,
            site,
            files,
            pipeline_version,
            artifact_root=artifact_root,
        )
