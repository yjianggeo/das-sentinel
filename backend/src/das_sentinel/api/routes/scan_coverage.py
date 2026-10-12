from __future__ import annotations

from datetime import date as CalendarDate

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from das_sentinel.api.auth import AuthenticatedAdmin
from das_sentinel.persistence.models import Site
from das_sentinel.services.scan_coverage import day_coverage, year_coverage

router = APIRouter(prefix="/scan-coverage", tags=["scan-coverage"])


class CoverageDay(BaseModel):
    date: str
    scanned_seconds: int


class YearCoverageResponse(BaseModel):
    year: int
    timezone: str
    available_years: list[int]
    days: list[CoverageDay]
    total_scanned_seconds: int
    scanned_days: int


class CoverageInterval(BaseModel):
    start_at: str
    end_at: str


class DayCoverageResponse(BaseModel):
    date: str
    timezone: str
    scanned_seconds: int
    intervals: list[CoverageInterval]


def _require_site(session: Session, site_id: str | None) -> None:
    if site_id is not None and session.scalar(select(Site.id).where(Site.id == site_id)) is None:
        raise HTTPException(status_code=404, detail="Site not found")


@router.get("", response_model=YearCoverageResponse)
def scan_coverage(request: Request, admin: AuthenticatedAdmin,
                  year: int | None = Query(None, ge=1900, le=2100),
                  site_id: str | None = None) -> dict[str, object]:
    del admin
    with request.app.state.session_factory() as session:
        _require_site(session, site_id)
        return year_coverage(session, year, site_id)


@router.get("/day", response_model=DayCoverageResponse)
def scan_coverage_day(request: Request, admin: AuthenticatedAdmin,
                      date: str = Query(pattern=r"^\d{4}-\d{2}-\d{2}$"),
                      site_id: str | None = None) -> dict[str, object]:
    del admin
    try:
        day = CalendarDate.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid calendar date") from None
    if not 1900 <= day.year <= 2100:
        raise HTTPException(status_code=422, detail="Year must be between 1900 and 2100")
    with request.app.state.session_factory() as session:
        _require_site(session, site_id)
        return day_coverage(session, day, site_id)
