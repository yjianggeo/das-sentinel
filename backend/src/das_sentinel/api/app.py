from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

from das_sentinel.api.routes.auth import router as auth_router
from das_sentinel.api.routes.sites import router as sites_router
from das_sentinel.config import Settings
from das_sentinel.persistence import (
    AsyncWriteCoordinator,
    check_database_startup,
    create_local_engine,
    session_factory,
)


def create_app(settings: Settings) -> FastAPI:
    engine = create_local_engine(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            check_database_startup(engine)
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="DAS Sentinel", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory(engine)
    app.state.write_coordinator = AsyncWriteCoordinator()
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie="session",
        same_site="strict",
        https_only=settings.secure_cookies,
    )
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(sites_router, prefix="/api/v1")
    return app
