from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import and_, case, func, or_, select

from das_sentinel.api.auth import AuthenticatedAdmin
from das_sentinel.api.schemas import DashboardResponse
from das_sentinel.persistence.model_scan_models import ModelScan
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowReview,
)

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(
    request: Request,
    admin: AuthenticatedAdmin,
) -> DashboardResponse:
    del admin
    # Materialization keeps SQLite on the covering index, away from large result rows.
    analyses = select(
        WindowAnalysis.id, WindowAnalysis.status, WindowAnalysis.input_sha256,
    ).cte("analysis_summary").prefix_with("MATERIALIZED", dialect="sqlite")
    current_analysis = analyses.c.id == EventWindow.latest_analysis_id
    window_status = func.coalesce(analyses.c.status, "planned")
    latest_review_id = (
        select(WindowReview.id)
        .where(WindowReview.window_id == EventWindow.id)
        .order_by(WindowReview.created_at.desc(), WindowReview.id.desc())
        .limit(1)
        .correlate(EventWindow)
        .scalar_subquery()
    )
    bound_review = and_(
        WindowReview.analysis_id == analyses.c.id,
        WindowReview.input_sha256 == analyses.c.input_sha256,
    )
    earthquake = func.json_extract(WindowReview.labels_json, "$.earthquake")
    blast = func.json_extract(WindowReview.labels_json, "$.blast")
    review_class = case(
        (and_(bound_review, or_(earthquake == "present", blast == "present")), "earthquake"),
        (and_(bound_review, earthquake == "absent", or_(blast == "absent", blast.is_(None))), "noise"),
        else_="pending",
    )
    with request.app.state.session_factory() as session:
        window_status_counts = dict(session.execute(
            select(window_status, func.count())
            .select_from(EventWindow)
            .outerjoin(analyses, current_analysis)
            .where(EventWindow.window_seconds == 30)
            .group_by(window_status)
        ).all())
        model_scan_status_counts = dict(session.execute(
            select(ModelScan.status, func.count()).group_by(ModelScan.status)
        ).all())
        review_counts = {"earthquake": 0, "noise": 0, "pending": 0}
        review_counts.update(session.execute(
            select(review_class, func.count())
            .select_from(EventWindow)
            .join(analyses, current_analysis)
            .outerjoin(WindowReview, WindowReview.id == latest_review_id)
            .where(EventWindow.window_seconds == 30, analyses.c.status == "succeeded")
            .group_by(review_class)
        ).all())
    return DashboardResponse(
        window_status_counts=window_status_counts,
        model_scan_status_counts=model_scan_status_counts,
        review_counts=review_counts,
    )
