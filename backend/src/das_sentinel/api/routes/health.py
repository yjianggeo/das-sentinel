from fastapi import APIRouter, HTTPException, Request, status

from das_sentinel.persistence import check_database_startup
from das_sentinel.runtime import probe_storage

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz(request: Request) -> dict[str, str]:
    try:
        check_database_startup(request.app.state.engine)
        probe_storage(request.app.state.settings)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        ) from error
    return {"status": "ok", "pipeline_version": request.app.state.settings.pipeline_version}
