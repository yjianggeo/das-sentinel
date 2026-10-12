from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from das_sentinel.api.auth import AuthenticatedAdmin
from das_sentinel.api.schemas import EventSummaryResponse
from das_sentinel.services.review_batches import get_review_batch

router = APIRouter(prefix="/review-batches", tags=["review-batches"])


class CatalogWindowResponse(BaseModel):
    catalog_id: str
    phase: str
    start_utc: str
    end_utc: str
    source_url: str


class ReviewBatchGroupResponse(BaseModel):
    id: str
    representative: EventSummaryResponse
    representatives: list[EventSummaryResponse] = Field(min_length=1, max_length=3)
    catalog_window: CatalogWindowResponse | None
    members: list[EventSummaryResponse]
    start_utc: str
    end_utc: str
    priority_score: float
    reasons: list[str]
    selection_kind: Literal["earthquake", "diverse", "audit"]
    similar_earthquake_event_id: str | None


class ReviewBatchResponse(BaseModel):
    site_id: str
    total_unreviewed: int
    total_groups: int
    offset: int
    limit: int
    reference_counts: dict[str, int]
    catalog_window_count: int
    warnings: list[str]
    groups: list[ReviewBatchGroupResponse]


@router.get("", response_model=ReviewBatchResponse)
def review_batches(
    request: Request,
    admin: AuthenticatedAdmin,
    site_id: str = Query(min_length=1, max_length=36),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict:
    del admin
    with request.app.state.session_factory() as session:
        try:
            return get_review_batch(
                session,
                request.app.state.settings.app_data_dir,
                site_id=site_id,
                offset=offset,
                limit=limit,
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail="Site not found") from error
