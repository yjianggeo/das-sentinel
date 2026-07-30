from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import Response

from das_sentinel.api.routes.artifacts import router as artifacts_router
from das_sentinel.api.routes.auth import router as auth_router
from das_sentinel.api.routes.events import router as events_router
from das_sentinel.api.routes.internal import router as internal_router
from das_sentinel.api.routes.jobs import router as jobs_router
from das_sentinel.api.routes.sites import router as sites_router
from das_sentinel.config import Settings
from das_sentinel.persistence import (
    AsyncWriteCoordinator,
    check_database_startup,
    create_local_engine,
    session_factory,
)
from das_sentinel.services.jobs import JobService


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
            await run_in_threadpool(
                _recover_stale_jobs,
                app.state.session_factory,
                settings.app_data_dir / "artifacts",
            )
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="DAS Sentinel", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory(engine)
    app.state.write_coordinator = AsyncWriteCoordinator()
    app.state.auth_limiter = asyncio.Semaphore(1)
    app.add_middleware(ApiSessionSecurityMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie="session",
        same_site="strict",
        https_only=settings.secure_cookies,
    )
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(sites_router, prefix="/api/v1")
    app.include_router(jobs_router, prefix="/api/v1")
    app.include_router(events_router, prefix="/api/v1")
    app.include_router(artifacts_router, prefix="/api/v1")
    app.include_router(internal_router, prefix="/api/v1")
    return app


def _recover_stale_jobs(sessions, artifact_root) -> list[str]:
    with sessions.begin() as session:
        return JobService(session, artifact_root).recover_stale()
