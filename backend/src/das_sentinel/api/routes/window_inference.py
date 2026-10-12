import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.services.event_windows import WindowError

router = APIRouter(tags=['window-inference'])
logger = logging.getLogger(__name__)


class PredictionRetry(BaseModel):
    analysis_id: str = Field(min_length=1, max_length=36)
    input_sha256: str = Field(pattern='^[0-9a-f]{64}$')


def call(request, method, *args, **kwargs):
    try:
        return getattr(request.app.state.window_inference_service, method)(*args, **kwargs)
    except WindowError as error:
        raise HTTPException(error.status_code, str(error)) from error
    except (ValueError, OSError, KeyError) as error:
        logger.exception('Window model unavailable')
        raise HTTPException(503, '整窗模型暂不可用，请稍后重试。') from error


@router.get('/window-models/active')
def active(request: Request, admin: AuthenticatedAdmin, site_id: str):
    return call(request, 'active', site_id)


@router.get('/event-windows/{window_id}/prediction')
def prediction(window_id: str, request: Request, admin: AuthenticatedAdmin,
               analysis_id: str | None = None, input_sha256: str | None = None):
    return call(request, 'prediction', window_id, analysis_id=analysis_id, input_sha256=input_sha256)


@router.post('/event-windows/{window_id}/prediction')
def retry(window_id: str, payload: PredictionRetry, request: Request, admin: CsrfAdmin):
    return call(request, 'prediction', window_id, **payload.model_dump(), retry=True)
