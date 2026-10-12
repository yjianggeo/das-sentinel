from __future__ import annotations

import asyncio
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles

from das_sentinel.api.routes import health_router
from das_sentinel.api.routes.artifacts import router as artifacts_router
from das_sentinel.api.routes.auth import router as auth_router
from das_sentinel.api.routes.dashboard import router as dashboard_router
from das_sentinel.api.routes.event_windows import router as event_windows_router
from das_sentinel.api.routes.events import router as events_router
from das_sentinel.api.routes.internal import router as internal_router
from das_sentinel.api.routes.jobs import router as jobs_router
from das_sentinel.api.routes.low_frequency import router as low_frequency_router
from das_sentinel.api.routes.model_scans import router as model_scans_router
from das_sentinel.api.routes.models import router as models_router
from das_sentinel.api.routes.review_batches import router as review_batches_router
from das_sentinel.api.routes.scan_coverage import router as scan_coverage_router
from das_sentinel.api.routes.scans import router as scans_router
from das_sentinel.api.routes.sites import router as sites_router
from das_sentinel.api.routes.waterfall_previews import (
    router as waterfall_previews_router,
)
from das_sentinel.api.routes.window_inference import router as window_inference_router
from das_sentinel.api.routes.window_training import router as window_training_router
from das_sentinel.config import Settings
from das_sentinel.persistence import (
    AsyncWriteCoordinator,
    check_database_startup,
    create_local_engine,
    session_factory,
)
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.runtime import probe_storage
from das_sentinel.services.jobs import JobService
from das_sentinel.services.low_frequency_scans import LowFrequencyScanService
from das_sentinel.services.scans import ScanCoordinator
from das_sentinel.services.waterfall_previews import WaterfallPreviewService
from das_sentinel.services.window_inference import WindowInferenceService
from das_sentinel.services.window_training import WindowTrainingService


class ApiSessionSecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        path = request.url.path
        if path == "/api/v1/session" and request.method == "POST":
            request.session.clear()
        response = await call_next(request)
        if path.startswith("/api/v1"):
            response.headers["Cache-Control"] = "no-store"
        return response


def create_app(settings: Settings) -> FastAPI:
    engine = create_local_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            check_database_startup(engine)
            probe_storage(settings)
            await run_in_threadpool(
                _recover_stale_jobs,
                app.state.session_factory,
                settings.app_data_dir / "artifacts",
            )
            yield
        finally:
            await app.state.scan_coordinator.close()
            app.state.low_frequency_service.close()
            app.state.window_training_service.close()
            await run_in_threadpool(app.state.window_inference_service.close)
            engine.dispose()

    app = FastAPI(title="DAS Sentinel", version="0.1.0", lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, _validation_error_response)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory(engine)
    app.state.write_coordinator = AsyncWriteCoordinator()
    app.state.window_inference_service = WindowInferenceService(
        app.state.session_factory, settings.app_data_dir,
    )
    app.state.window_training_service = WindowTrainingService(
        app.state.session_factory, settings.app_data_dir,
    )
    app.state.scan_coordinator = ScanCoordinator(
        app.state.session_factory,
        settings.app_data_dir / "artifacts",
        settings.pipeline_version,
        app.state.write_coordinator,
    )
    app.state.auth_limiter = asyncio.Semaphore(1)
    app.state.low_frequency_service = LowFrequencyScanService(
        settings.app_data_dir / "low-frequency-scans", settings.data_root,
    )
    app.state.waterfall_preview_service = WaterfallPreviewService(
        settings.data_root,
        ZdHdf5Reader(),
    )
    app.add_middleware(ApiSessionSecurityMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie="session",
        same_site="strict",
        https_only=settings.secure_cookies,
    )
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(dashboard_router, prefix="/api/v1")
    app.include_router(sites_router, prefix="/api/v1")
    app.include_router(jobs_router, prefix="/api/v1")
    app.include_router(low_frequency_router, prefix="/api/v1")
    app.include_router(events_router, prefix="/api/v1")
    app.include_router(event_windows_router, prefix="/api/v1")
    app.include_router(window_training_router, prefix="/api/v1")
    app.include_router(window_inference_router, prefix="/api/v1")
    app.include_router(review_batches_router, prefix="/api/v1")
    app.include_router(waterfall_previews_router, prefix="/api/v1")
    app.include_router(artifacts_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    app.include_router(model_scans_router, prefix="/api/v1")
    app.include_router(scans_router, prefix="/api/v1")
    app.include_router(scan_coverage_router, prefix="/api/v1")
    app.include_router(internal_router, prefix="/api/v1")
    app.include_router(health_router)
    _mount_spa(app)
    return app


def create_runtime_app() -> FastAPI:
    return create_app(Settings())  # type: ignore[call-arg]


async def _validation_error_response(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    del request
    return JSONResponse(
        {"detail": _json_safe(error.errors())},
        status_code=422,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _web_dist() -> Path:
    override = os.environ.get("DAS_WEB_DIST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "frontend" / "dist"


def _mount_spa(app: FastAPI) -> None:
    dist = _web_dist().resolve()
    index = dist / "index.html"
    if not index.is_file():
        return
    assets = dist / "assets"
    if assets.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=assets),
            name="web-assets",
        )

    async def spa_fallback(request: Request, exc: Exception) -> Response:
        # Only GET misses fall back to the SPA; API misses stay JSON 404 and
        # every other HTTP error keeps its original status and detail.
        if isinstance(exc, StarletteHTTPException) and exc.status_code == 404:
            full_path = request.url.path.lstrip("/")
            if full_path.startswith("api/") or request.method != "GET":
                return JSONResponse({"detail": "Not found"}, status_code=404)
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file() and candidate.is_relative_to(dist):
                return FileResponse(candidate)
            return FileResponse(index)
        if isinstance(exc, StarletteHTTPException):
            return JSONResponse(
                {"detail": exc.detail},
                status_code=exc.status_code,
                headers=getattr(exc, "headers", None),
            )
        raise exc

    # A 404 handler (not a catch-all route) so dynamically added routes keep
    # working and no route ordering traps exist.
    app.add_exception_handler(StarletteHTTPException, spa_fallback)


def _recover_stale_jobs(sessions, artifact_root) -> list[str]:
    with sessions.begin() as session:
        return JobService(session, artifact_root).recover_stale()
