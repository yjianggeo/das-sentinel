import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.api.routes.internal import WorkerToken
from das_sentinel.api.window_schemas import (
    WindowClaim,
    WindowCoverageRequest,
    WindowFail,
    WindowFavoriteSet,
    WindowOwned,
    WindowReviewCreate,
    WindowSkipCreate,
)
from das_sentinel.services.event_windows import WindowError, WindowService

router = APIRouter(tags=["event-windows"])


def call(request, method, *args, **kwargs):
    try:
        with request.app.state.session_factory.begin() as session:
            return getattr(
                WindowService(
                    session,
                    request.app.state.settings.app_data_dir / "window-artifacts",
                ),
                method,
            )(*args, **kwargs)
    except WindowError as error:
        raise HTTPException(error.status_code, str(error)) from error


async def write(request, method, *args, **kwargs):
    async with request.app.state.write_coordinator.lock:
        return await run_in_threadpool(call, request, method, *args, **kwargs)


@router.get("/event-windows")
def windows(
    request: Request,
    admin: AuthenticatedAdmin,
    site_id: str | None = None,
    review_status: Literal["unreviewed", "reviewed", "reviewed_earthquake", "reviewed_noise", "all"] = "unreviewed",
    recommended: bool = False,
    candidate_only: bool = False,
    model_recommended: bool = False,
    model_scan_id: str | None = None,
    favorites_only: bool = False,
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    window_seconds: Literal["30", "60"] = "30",
    sort: Literal["default", "borehole_coverage"] = "default",
):
    return call(
        request,
        "list",
        site_id=site_id,
        review_status=review_status,
        recommended=recommended,
        candidate_only=candidate_only,
        model_recommended=model_recommended,
        model_scan_id=model_scan_id,
        favorites_only=favorites_only,
        admin_id=admin.id,
        limit=limit,
        offset=offset,
        window_seconds=int(window_seconds),
        sort=sort,
    )


@router.get("/event-windows/{window_id}")
def detail(window_id: str, request: Request, admin: AuthenticatedAdmin, model_scan_id: str | None = None):
    return call(request, "detail", window_id, model_scan_id=model_scan_id, admin_id=admin.id)


async def measure_spatial_coverage(request, window_id, analysis_id, input_sha256):
    measured = await run_in_threadpool(call, request, "prepare_spatial_coverage",
        window_id, analysis_id, input_sha256)
    return await write(request, "save_spatial_coverage", window_id, measured)


@router.post("/event-windows/{window_id}/spatial-coverage")
async def spatial_coverage(
    window_id: str, payload: WindowCoverageRequest, request: Request, admin: CsrfAdmin,
):
    return await measure_spatial_coverage(request, window_id, **payload.model_dump())


async def schedule_spatial_coverage(request, window_id, analysis_id, input_sha256):
    try:
        await measure_spatial_coverage(request, window_id, analysis_id, input_sha256)
    except Exception:
        logging.getLogger(__name__).exception("Cannot measure spatial coverage for %s", analysis_id)


@router.put("/event-windows/{window_id}/favorite")
async def favorite(
    window_id: str, payload: WindowFavoriteSet, request: Request, admin: CsrfAdmin,
):
    return await write(request, "set_favorite", window_id, payload.favorite, admin_id=admin.id)


@router.get("/event-windows/{window_id}/evidence")
def evidence(
    window_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
    analysis_id: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return call(
        request,
        "evidence",
        window_id,
        analysis_id=analysis_id,
        limit=limit,
        offset=offset,
    )


@router.get("/event-windows/{window_id}/artifacts/{kind}")
def artifact(
    window_id: str,
    kind: str,
    request: Request,
    admin: AuthenticatedAdmin,
    analysis_id: str | None = None,
):
    path, media_type = call(request, "artifact", window_id, kind, analysis_id)
    return FileResponse(path, media_type=media_type)


@router.get("/event-windows/{window_id}/waterfall-preview")
def waterfall_preview(
    window_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
    analysis_id: str,
    filtered: bool = False,
):
    payload = call(request, "waterfall_preview", window_id, analysis_id, filtered)
    return Response(content=payload, media_type="image/png")


@router.get("/event-windows/{window_id}/region-preview")
def region_preview(
    window_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
    analysis_id: str,
    region_id: int = Query(ge=1),
    filtered: bool = False,
    show_regions: bool = True,
):
    payload = call(request, "region_preview", window_id, analysis_id, region_id, filtered, show_regions)
    return Response(content=payload, media_type="image/png")


@router.post("/event-windows/{window_id}/reviews")
async def review(
    window_id: str, payload: WindowReviewCreate, request: Request, admin: CsrfAdmin
):
    return await write(
        request, "review", window_id, **payload.model_dump(), admin_id=admin.id
    )


@router.get("/window-jobs")
def jobs(
    request: Request,
    admin: AuthenticatedAdmin,
    site_id: str | None = None,
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    | None = None,
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    window_seconds: Literal["30", "60"] = "30",
):
    return call(
        request, "jobs", site_id=site_id, status=status, limit=limit, offset=offset,
        window_seconds=int(window_seconds),
    )


@router.post("/window-jobs/{analysis_id}/cancel")
async def cancel(analysis_id: str, request: Request, admin: CsrfAdmin):
    return await write(request, "cancel", analysis_id)


@router.post("/window-jobs/{analysis_id}/retry")
async def retry(analysis_id: str, request: Request, admin: CsrfAdmin):
    return await write(request, "retry", analysis_id)


@router.post("/internal/window-jobs/claim")
async def claim(payload: WindowClaim, request: Request, worker: WorkerToken):
    result = await write(request, "claim", **payload.model_dump())
    return Response(status_code=204) if result is None else result


@router.post("/internal/window-jobs/{analysis_id}/heartbeat")
async def heartbeat(
    analysis_id: str, payload: WindowOwned, request: Request, worker: WorkerToken
):
    return await write(request, "heartbeat", analysis_id, **payload.model_dump())


@router.post("/internal/window-jobs/{analysis_id}/result")
async def result(
    analysis_id: str, payload: WindowOwned, request: Request, worker: WorkerToken
):
    completed = await write(request, "result", analysis_id, **payload.model_dump())
    await schedule_spatial_coverage(request, completed["window_id"], analysis_id, completed["input_sha256"])
    try:
        await run_in_threadpool(request.app.state.window_inference_service.prediction,
                                completed['window_id'], analysis_id=completed['id'])
    except Exception:
        logging.getLogger(__name__).exception('Cannot schedule inference for %s', analysis_id)
    return completed


@router.post("/internal/window-jobs/{analysis_id}/fail")
async def fail(
    analysis_id: str, payload: WindowFail, request: Request, worker: WorkerToken
):
    return await write(request, "fail", analysis_id, **payload.model_dump())


@router.post("/internal/window-jobs/{analysis_id}/skip")
async def skip(
    analysis_id: str, payload: WindowSkipCreate, request: Request, worker: WorkerToken
):
    return await write(request, "skip", analysis_id, **payload.model_dump())
