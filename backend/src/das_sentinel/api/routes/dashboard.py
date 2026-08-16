from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from sqlalchemy import func, select

from das_sentinel.api.auth import AuthenticatedAdmin
from das_sentinel.api.schemas import DashboardResponse
from das_sentinel.persistence.models import (
    Classification,
    Event,
    Job,
    ProcessingRun,
    Site,
    SourceFile,
)

router = APIRouter(tags=["dashboard"])


def utc_now() -> datetime:
    return datetime.now(UTC)


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(
    request: Request,
    admin: AuthenticatedAdmin,
) -> DashboardResponse:
    del admin
    latest_classification_id = (
        select(Classification.id)
        .where(Classification.event_id == Event.id)
        .order_by(Classification.created_at.desc(), Classification.id.desc())
        .limit(1)
        .correlate(Event)
        .scalar_subquery()
    )
    latest_label = func.coalesce(
        select(Classification.label)
        .where(Classification.id == latest_classification_id)
        .scalar_subquery(),
        "unknown",
    )
    with request.app.state.session_factory() as session:
        now = utc_now()
        job_counts = dict(
            session.execute(
                select(Job.status, func.count()).group_by(Job.status)
            ).all()
        )
        machine_label_counts = dict(
            session.execute(
                select(latest_label, func.count())
                .select_from(Event)
                .group_by(latest_label)
            ).all()
        )
        today_event_count = sum(
            _site_today_event_count(session, site, now=now)
            for site in session.scalars(select(Site).where(Site.enabled.is_(True)))
        )
    return DashboardResponse(
        job_counts=job_counts,
        today_event_count=today_event_count,
        machine_label_counts=machine_label_counts,
    )


def _site_today_event_count(session, site: Site, *, now: datetime) -> int:
    local_now = now.astimezone(ZoneInfo(site.timezone))
    timezone = ZoneInfo(site.timezone)
    start_utc = datetime.combine(local_now.date(), time.min, timezone).astimezone(UTC)
    end_utc = datetime.combine(
        local_now.date() + timedelta(days=1),
        time.min,
        timezone,
    ).astimezone(UTC)
    return session.scalar(
        select(func.count())
        .select_from(Event)
        .join(ProcessingRun, ProcessingRun.id == Event.processing_run_id)
        .join(SourceFile, SourceFile.id == ProcessingRun.source_file_id)
        .where(
            SourceFile.site_id == site.id,
            Event.occurred_at >= start_utc,
            Event.occurred_at < end_utc,
        )
    ) or 0
