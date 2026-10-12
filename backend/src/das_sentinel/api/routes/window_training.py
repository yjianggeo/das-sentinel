from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.services.window_training import WindowTrainingError

router = APIRouter(tags=["window-training"])


class TrainingCreate(BaseModel):
    site_id: str = Field(min_length=1, max_length=36)


def call(request, method, site_id):
    try:
        return getattr(request.app.state.window_training_service, method)(site_id)
    except WindowTrainingError as error:
        raise HTTPException(error.status_code, str(error)) from error


@router.get("/window-training/readiness")
def readiness(request: Request, admin: AuthenticatedAdmin, site_id: str):
    return call(request, "readiness", site_id)


@router.get("/window-training/jobs")
def jobs(request: Request, admin: AuthenticatedAdmin, site_id: str):
    return call(request, "jobs", site_id)


@router.post("/window-training/jobs", status_code=202)
def create(payload: TrainingCreate, request: Request, admin: CsrfAdmin):
    return call(request, "create", payload.site_id)
