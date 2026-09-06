from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.api.schemas import (
    ScanDirectoryEntryResponse,
    ScanDirectoryResponse,
    ScanRequest,
    ScanResponse,
)
from das_sentinel.persistence.models import Site
from das_sentinel.services.discovery import (
    DiscoveryConfigurationError,
    DiscoveryPathError,
    list_scan_directories,
    load_discovery_site,
    resolve_scan_directory,
)
from das_sentinel.services.scans import ScanSnapshot

router = APIRouter(tags=["scans"])


@router.get(
    "/sites/{site_id}/scan-directories",
    response_model=ScanDirectoryResponse,
)
async def scan_directories(
    site_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
    path: str | None = Query(default=None, max_length=1024),
) -> ScanDirectoryResponse:
    del admin
    try:
        return await run_in_threadpool(
            _load_scan_directories,
            request.app.state.session_factory,
            site_id,
            path,
        )
    except (DiscoveryConfigurationError, DiscoveryPathError) as error:
        raise _discovery_http_exception(error) from error


@router.post("/sites/{site_id}/scan", response_model=ScanResponse)
async def scan_site(
    site_id: str,
    payload: ScanRequest,
    response: Response,
    request: Request,
    admin: CsrfAdmin,
) -> ScanResponse:
    del admin
    try:
        site, scan_path = await run_in_threadpool(
            _load_scan_target,
            request.app.state.session_factory,
            site_id,
            payload.path,
        )
    except (DiscoveryConfigurationError, DiscoveryPathError) as error:
        raise _discovery_http_exception(error) from error
    snapshot, created = request.app.state.scan_coordinator.start(
        site,
        scan_path,
        payload.path,
    )
    response.status_code = (
        status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK
    )
    return scan_response(snapshot)


@router.get(
    "/sites/{site_id}/active-scan",
    response_model=ScanResponse | None,
)
def active_scan(
    site_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
) -> ScanResponse | None:
    del admin
    snapshot = request.app.state.scan_coordinator.active(site_id)
    return None if snapshot is None else scan_response(snapshot)


@router.get("/scans/{scan_id}", response_model=ScanResponse)
def get_scan(
    scan_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
) -> ScanResponse:
    del admin
    snapshot = request.app.state.scan_coordinator.get(scan_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found",
        )
    return scan_response(snapshot)


def _load_scan_directories(
    sessions,
    site_id: str,
    path: str | None,
) -> ScanDirectoryResponse:
    with sessions() as session:
        row = session.get(Site, site_id)
        site = load_discovery_site(session, site_id)
        listing = list_scan_directories(site, path)
        assert row is not None
        return ScanDirectoryResponse(
            site_id=site.site_id,
            site_name=row.name,
            path=listing.path,
            parent_path=listing.parent_path,
            directories=[
                ScanDirectoryEntryResponse(name=item.name, path=item.path)
                for item in listing.directories
            ],
        )


def _load_scan_target(sessions, site_id: str, path: str):
    with sessions() as session:
        site = load_discovery_site(session, site_id)
    return site, resolve_scan_directory(site, path)


def _discovery_http_exception(
    error: DiscoveryConfigurationError | DiscoveryPathError,
) -> HTTPException:
    if isinstance(error, DiscoveryConfigurationError):
        if "not found" in str(error):
            return HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found",
            )
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        )
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="Invalid scan directory",
    )


def scan_response(snapshot: ScanSnapshot) -> ScanResponse:
    return ScanResponse(
        id=snapshot.id,
        site_id=snapshot.site_id,
        path=snapshot.path,
        phase=snapshot.phase,
        initial_checked_files=snapshot.initial_checked_files,
        initial_found_files=snapshot.initial_found_files,
        verification_checked_files=snapshot.verification_checked_files,
        verification_found_files=snapshot.verification_found_files,
        stable_files=snapshot.stable_files,
        submitted_files=snapshot.submitted_files,
        skipped_files=snapshot.skipped_files,
        stability_seconds_remaining=snapshot.stability_seconds_remaining,
        error_message=snapshot.error_message,
    )
