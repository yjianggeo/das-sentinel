from __future__ import annotations

import json
from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.api.schemas import ModelResponse
from das_sentinel.persistence.models import ModelVersion
from das_sentinel.services.training import (
    PublishGateError,
    TrainingDataError,
    TrainingService,
)

router = APIRouter(tags=["models"])
ModelStatus = Literal["candidate", "active", "retired", "invalid"]


@router.get(
    "/sites/{site_id}/models",
    response_model=list[ModelResponse],
)
def list_models(
    site_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
) -> list[ModelResponse]:
    del admin
    with request.app.state.session_factory() as session:
        service = _service(request, session)
        return [_response(model) for model in service.list_for_site(site_id)]


@router.post(
    "/sites/{site_id}/models",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
)
async def train_model_route(
    site_id: str,
    request: Request,
    admin: CsrfAdmin,
) -> ModelResponse:
    del admin
    try:
        async with request.app.state.write_coordinator.lock:
            return await run_in_threadpool(
                _train,
                request.app.state.session_factory,
                _artifact_root(request),
                site_id,
            )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Site not found",
        ) from error
    except TrainingDataError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@router.get(
    "/models/{model_id}",
    response_model=ModelResponse,
)
def get_model(
    model_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
) -> ModelResponse:
    del admin
    with request.app.state.session_factory() as session:
        try:
            model = _service(request, session).get(model_id)
        except LookupError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found",
            ) from error
        return _response(model)


@router.post(
    "/models/{model_id}/publish",
    response_model=ModelResponse,
)
async def publish_model(
    model_id: str,
    request: Request,
    admin: CsrfAdmin,
) -> ModelResponse:
    del admin
    try:
        async with request.app.state.write_coordinator.lock:
            return await run_in_threadpool(
                _publish,
                request.app.state.session_factory,
                _artifact_root(request),
                model_id,
            )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        ) from error
    except PublishGateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


@router.post(
    "/sites/{site_id}/models/rollback",
    response_model=ModelResponse,
)
async def rollback_model(
    site_id: str,
    request: Request,
    admin: CsrfAdmin,
) -> ModelResponse:
    del admin
    try:
        async with request.app.state.write_coordinator.lock:
            return await run_in_threadpool(
                _rollback,
                request.app.state.session_factory,
                _artifact_root(request),
                site_id,
            )
    except PublishGateError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error


def _artifact_root(request: Request):
    return request.app.state.settings.app_data_dir / "artifacts"


def _service(request: Request, session) -> TrainingService:
    return TrainingService(session, _artifact_root(request))


def _train(sessions, artifact_root, site_id: str) -> ModelResponse:
    with sessions.begin() as session:
        return _response(TrainingService(session, artifact_root).train(site_id=site_id))


def _publish(sessions, artifact_root, model_id: str) -> ModelResponse:
    with sessions.begin() as session:
        return _response(TrainingService(session, artifact_root).publish(model_id))


def _rollback(sessions, artifact_root, site_id: str) -> ModelResponse:
    with sessions.begin() as session:
        return _response(TrainingService(session, artifact_root).rollback(site_id))


def _response(model: ModelVersion) -> ModelResponse:
    return ModelResponse(
        id=model.id,
        site_id=model.site_id,
        version=model.version,
        status=cast("ModelStatus", model.status),
        dataset_sha256=model.dataset_sha256,
        bundle_sha256=model.bundle_sha256,
        metrics=json.loads(model.metrics_json),
        created_at=model.created_at,
        published_at=model.published_at,
    )
