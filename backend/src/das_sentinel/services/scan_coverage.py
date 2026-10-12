from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone

from sqlalchemy import Integer, cast, func, literal_column, select, union
from sqlalchemy.orm import Session

from das_sentinel.persistence.model_scan_models import ModelScan, ModelScanReceipt
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowSkip,
)

BEIJING = timezone(timedelta(hours=8))
WINDOW_SECONDS = 30


def _evidence_queries(site_id: str | None, start: datetime | None = None,
                     end: datetime | None = None):
    # Both existing analysis indexes include SQLite's rowid. Keep status/SHA checks
    # in a narrow materialized projection so window lookup never reads large rows.
    analysis_rowid = literal_column("window_analyses.rowid")
    analyses = select(
        analysis_rowid.label("analysis_rowid"), WindowAnalysis.status, WindowAnalysis.input_sha256,
    ).cte("coverage_analysis_summary").prefix_with("MATERIALIZED", dialect="sqlite")
    succeeded_summary = select(1).where(
        analyses.c.analysis_rowid == analysis_rowid,
        analyses.c.status == "succeeded",
        analyses.c.input_sha256 != "",
    ).exists()
    succeeded_analysis = select(1).select_from(WindowAnalysis).where(
        WindowAnalysis.window_id == EventWindow.id, succeeded_summary,
    ).exists()
    # Materialize before site/time filters so SQLite scans the receipt covering index
    # instead of choosing a scan_id lookup that fetches large result payload rows.
    receipts = select(
        ModelScanReceipt.scan_id, ModelScanReceipt.start_utc,
        ModelScanReceipt.score, ModelScanReceipt.model_input_sha256,
    ).cte("coverage_receipt_summary").prefix_with("MATERIALIZED", dialect="sqlite")
    queries = [
        (select(EventWindow.start_utc).where(
            EventWindow.window_seconds == WINDOW_SECONDS, succeeded_analysis,
        ), EventWindow.site_id, EventWindow.start_utc),
        (select(receipts.c.start_utc).join(ModelScan, ModelScan.id == receipts.c.scan_id)
            .where(receipts.c.score.between(0, 1),
                   func.length(receipts.c.model_input_sha256) == 64),
         ModelScan.site_id, receipts.c.start_utc),
        (select(WindowSkip.start_utc).where(
            WindowSkip.window_seconds == WINDOW_SECONDS,
            WindowSkip.policy_version == "empty-regions-v1",
            WindowSkip.reason == "no_qualified_regions",
        ), WindowSkip.site_id, WindowSkip.start_utc),
    ]
    statements = []
    for statement, site, timestamp in queries:
        if site_id is not None:
            statement = statement.where(site == site_id)
        if start is not None:
            statement = statement.where(timestamp >= start)
        if end is not None:
            statement = statement.where(timestamp < end)
        statements.append(statement)
    return statements


def _available_years(session: Session, site_id: str | None) -> list[int]:
    # Deduplicate small year projections before combining the three evidence sources.
    years = union(*[
        statement.with_only_columns(cast(func.strftime(
            "%Y", statement.selected_columns.start_utc, "+8 hours",
        ), Integer).label("year")).distinct()
        for statement in _evidence_queries(site_id)
    ]).subquery()
    return list(session.scalars(select(years.c.year).order_by(years.c.year)))


def year_coverage(session: Session, year: int | None = None,
                  site_id: str | None = None) -> dict[str, object]:
    available_years = _available_years(session, site_id)
    if year is None:
        year = available_years[-1] if available_years else datetime.now(BEIJING).year
    first_day, next_year = date(year, 1, 1), date(year + 1, 1, 1)
    start = datetime.combine(first_day, time(), BEIJING).astimezone(UTC)
    end = datetime.combine(next_year, time(), BEIJING).astimezone(UTC)
    # All evidence describes aligned 30-second windows. UNION counts each time slot once,
    # independently of its site, batch, analysis history, or evidence source.
    slots = union(*_evidence_queries(site_id, start, end)).subquery()
    acquisition_date = func.date(slots.c.start_utc, "+8 hours")
    daily_seconds = dict(session.execute(select(
        acquisition_date, func.count() * WINDOW_SECONDS,
    ).group_by(acquisition_date)).all())
    days = []
    for offset in range((next_year - first_day).days):
        day = (first_day + timedelta(days=offset)).isoformat()
        days.append({"date": day, "scanned_seconds": daily_seconds.get(day, 0)})
    return {"year": year, "timezone": "Asia/Shanghai", "available_years": available_years,
        "days": days, "total_scanned_seconds": sum(daily_seconds.values()),
        "scanned_days": len(daily_seconds)}


def day_coverage(session: Session, day: date,
                 site_id: str | None = None) -> dict[str, object]:
    start = datetime.combine(day, time(), BEIJING).astimezone(UTC)
    end = start + timedelta(days=1)
    slots = union(*_evidence_queries(site_id, start, end)).subquery()
    starts = list(session.scalars(select(slots.c.start_utc).order_by(slots.c.start_utc)))
    merged: list[tuple[datetime, datetime]] = []
    for timestamp in starts:
        interval_end = timestamp + timedelta(seconds=WINDOW_SECONDS)
        if merged and timestamp <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval_end))
        else:
            merged.append((timestamp, interval_end))
    return {"date": day.isoformat(), "timezone": "Asia/Shanghai",
        "scanned_seconds": len(starts) * WINDOW_SECONDS,
        "intervals": [{"start_at": begin.astimezone(BEIJING).isoformat(),
            "end_at": finish.astimezone(BEIJING).isoformat()} for begin, finish in merged]}
