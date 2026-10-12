"""Authenticated model-first scan administration and fenced worker operations."""

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.api.model_scan_schemas import (
    DiscoveryDone,
    DiscoveryPage,
    Materialized,
    ModelScanCreate,
    PlanDone,
    Prediction,
    ScanClaim,
    ScanFail,
    ScanOwned,
)
from das_sentinel.api.routes.event_windows import schedule_spatial_coverage
from das_sentinel.api.routes.internal import WorkerToken
from das_sentinel.model_scan_contract import DIRECTORIES, utc
from das_sentinel.persistence.models import ProcessingProfile, Site
from das_sentinel.services.discovery import _validate_under_root, _validated_root
from das_sentinel.services.event_windows import WindowError
from das_sentinel.services.model_scans import ModelScanService
from das_sentinel.window_inference import read_active

router = APIRouter(tags=["model-scans"])


def call(request, method, *args, **kwargs):
    try:
        with request.app.state.session_factory.begin() as session:
            return getattr(ModelScanService(session, request.app.state.settings.app_data_dir), method)(*args, **kwargs)
    except WindowError as error:
        raise HTTPException(error.status_code, str(error)) from error
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(422, "invalid_model_scan_payload") from error


async def write(request, method, *args, **kwargs):
    async with request.app.state.write_coordinator.lock:
        return await run_in_threadpool(call, request, method, *args, **kwargs)


def prepare_create(request, site_id, payload):
    with request.app.state.session_factory() as session:
        site = session.get(Site, site_id)
        profile = session.scalar(select(ProcessingProfile).where(
            ProcessingProfile.site_id == site_id, ProcessingProfile.is_current == True))
        if site is None or not site.enabled or profile is None:
            raise HTTPException(422, "site_unavailable")
        root = _validated_root(Path(site.data_path))
        profile_id = profile.id
    directories = []
    for value in payload.directories:
        parts = PurePosixPath(value).parts
        if (not parts or parts[0] not in DIRECTORIES or ".." in parts or "\\" in value
                or PureWindowsPath(value).anchor or PurePosixPath(value).is_absolute()):
            raise HTTPException(422, "directory_not_authorized")
        directories.append(str(_validate_under_root(root, root.joinpath(*parts), directory=True)))
    models = request.app.state.settings.app_data_dir / "window-models"
    model = read_active(models, site_id)
    if model is None or model["id"] != str(payload.expected_model_id):
        raise HTTPException(409, "active_model_changed")
    exclusion_path = request.app.state.settings.app_data_dir / "model-scan-exclusions.json"
    if not exclusion_path.is_file():
        raise HTTPException(409, "model_scan_exclusions_missing")
    exclusions = [{"start_utc": utc(row["start_utc"]).isoformat(), "window_seconds": row["window_seconds"]}
                  for row in json.loads(exclusion_path.read_text(encoding="utf8"))["event_windows"]
                  if row["site_id"] == site_id and row["window_seconds"] == 30]
    ranges = None
    if payload.time_ranges:
        merged = []
        for start, end in sorted((r.start_utc, r.end_utc) for r in payload.time_ranges):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        ranges = [{"start_utc": utc(start).isoformat(), "end_utc": utc(end).isoformat()} for start, end in merged]
    return profile_id, sorted(set(directories)), model, exclusions, ranges


@router.post("/sites/{site_id}/model-scans", status_code=201)
async def create(site_id: str, payload: ModelScanCreate, request: Request, admin: CsrfAdmin):
    try:
        profile_id, directories, model, exclusions, ranges = await run_in_threadpool(prepare_create, request, site_id, payload)
    except (ValueError, OSError, KeyError) as error:
        raise HTTPException(422, "invalid_model_scan_configuration") from error
    async with request.app.state.write_coordinator.lock:
        # Recheck local deployment pointer and current profile before the short insert.
        current = await run_in_threadpool(read_active, request.app.state.settings.app_data_dir / "window-models", site_id)
        if current != model:
            raise HTTPException(409, "active_model_changed")
        return await run_in_threadpool(call, request, "create_checked", site_id, profile_id,
            payload.request_id, directories, model, exclusions, time_ranges=ranges)


@router.get("/model-scans")
def scans(request: Request, admin: AuthenticatedAdmin, site_id: str | None = None,
          limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    return call(request, "list", site_id=site_id, limit=limit, offset=offset)


@router.get("/model-scans/{scan_id}")
def detail(scan_id: str, request: Request, admin: AuthenticatedAdmin):
    return call(request, "get", scan_id)


@router.post("/model-scans/{scan_id}/{action}")
async def control(scan_id: str, action: Literal["pause", "resume", "cancel", "retry"], request: Request, admin: CsrfAdmin):
    return await write(request, "control", scan_id, action)


@router.post("/internal/model-scans/claim")
async def claim(payload: ScanClaim, request: Request, worker: WorkerToken):
    result = await write(request, "claim", payload.worker_id)
    return Response(status_code=204) if result is None else result


@router.post("/internal/model-scans/{unit_id}/heartbeat")
async def heartbeat(unit_id: str, payload: ScanOwned, request: Request, worker: WorkerToken):
    return await write(request, "heartbeat", unit_id, **payload.model_dump())


@router.post("/internal/model-scans/{unit_id}/prediction")
async def prediction(unit_id: str, payload: Prediction, request: Request, worker: WorkerToken):
    return await write(request, "record_prediction", unit_id, **payload.model_dump())


@router.post("/internal/model-scans/{unit_id}/discovery-page")
async def discovery_page(unit_id: str, payload: DiscoveryPage, request: Request, worker: WorkerToken):
    return await write(request, "commit_discovery_page", unit_id, **payload.model_dump())


@router.post("/internal/model-scans/{unit_id}/discovery-done")
async def discovery_done(unit_id: str, payload: DiscoveryDone, request: Request, worker: WorkerToken):
    return await write(request, "finish_discovery", unit_id, **payload.model_dump())


@router.post("/internal/model-scans/{unit_id}/plan-done")
async def plan_done(unit_id: str, payload: PlanDone, request: Request, worker: WorkerToken):
    return await write(request, "finish_plan", unit_id, **payload.model_dump())


@router.post("/internal/model-scans/{unit_id}/materialized")
async def materialized(unit_id: str, payload: Materialized, request: Request, worker: WorkerToken):
    validated = await run_in_threadpool(call, request, "validate_materialization", unit_id, **payload.model_dump())
    completed = await write(request, "finish_materialization", unit_id, payload.worker_id, payload.attempt_count, validated)
    if completed["status"] == "ready":
        await schedule_spatial_coverage(request, completed["window_id"], completed["analysis_id"],
                                       payload.binding["input_sha256"])
    return completed


@router.post("/internal/model-scans/{unit_id}/fail")
async def fail(unit_id: str, payload: ScanFail, request: Request, worker: WorkerToken):
    return await write(request, "fail", unit_id, **payload.model_dump())
