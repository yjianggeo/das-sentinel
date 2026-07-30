from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import exists, func, select

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.api.schemas import (
    EventArtifactResponse,
    EventDetailResponse,
    EventEvidenceResponse,
    EventRunResponse,
    EventSummaryResponse,
    MachineClassificationResponse,
    ReviewCreate,
    ReviewResponse,
)
from das_sentinel.persistence.models import (
    MACHINE_LABELS,
    Classification,
    Event,
    ProcessingRun,
    Review,
    SourceFile,
)
from das_sentinel.persistence.repositories import Repositories

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=list[EventSummaryResponse])
def list_events(
    request: Request,
    admin: AuthenticatedAdmin,
    site_id: str | None = Query(default=None, max_length=36),
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    machine_label: str | None = Query(default=None, max_length=16),
    reviewed: bool | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=100_000),
) -> list[EventSummaryResponse]:
    del admin
    start_utc = _utc_query(start_utc, "start_utc")
    end_utc = _utc_query(end_utc, "end_utc")
    if start_utc is not None and end_utc is not None and end_utc <= start_utc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="end_utc must be later than start_utc",
        )
    if machine_label is not None and machine_label not in MACHINE_LABELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="unsupported machine label",
        )

    latest_classification_id = (
        select(func.max(Classification.id))
        .where(Classification.event_id == Event.id)
        .correlate(Event)
        .scalar_subquery()
    )
    statement = (
        select(Event)
        .join(ProcessingRun, ProcessingRun.id == Event.processing_run_id)
        .join(SourceFile, SourceFile.id == ProcessingRun.source_file_id)
    )
    if site_id is not None:
        statement = statement.where(SourceFile.site_id == site_id)
    if start_utc is not None:
        statement = statement.where(Event.occurred_at >= start_utc)
    if end_utc is not None:
        statement = statement.where(Event.occurred_at <= end_utc)
    if machine_label is not None:
        statement = statement.join(
            Classification,
            Classification.id == latest_classification_id,
        ).where(Classification.label == machine_label)
    if reviewed is not None:
        has_review = exists(
            select(Review.id).where(Review.event_id == Event.id)
        )
        statement = statement.where(has_review if reviewed else ~has_review)
    statement = statement.order_by(
        Event.occurred_at.desc(),
        Event.id.desc(),
    ).limit(limit).offset(offset)

    with request.app.state.session_factory() as session:
        rows = list(session.scalars(statement))
        return [_summary(session, event) for event in rows]


@router.get("/{event_id}", response_model=EventDetailResponse)
def get_event(
    event_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
) -> EventDetailResponse:
    del admin
    with request.app.state.session_factory() as session:
        event = session.get(Event, event_id)
        if event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Event not found",
            )
        run = session.get(ProcessingRun, event.processing_run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Event run not found",
            )
        source = session.get(SourceFile, run.source_file_id)
        if source is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Event source not found",
            )
        summary = _summary(session, event, site_id=source.site_id)
        return EventDetailResponse(
            **summary.model_dump(),
            features=json.loads(event.features_json),
            evidence=[
                EventEvidenceResponse(
                    detector=item.detector,
                    score=item.score,
                    payload=json.loads(item.payload_json),
                )
                for item in event.evidence
            ],
            machine_history=[
                MachineClassificationResponse(
                    id=item.id,
                    label=item.label,
                    decision_score=item.decision_score,
                    method=item.method,
                    reasons=json.loads(item.reasons_json),
                    model_version_id=item.model_version_id,
                    created_at=item.created_at,
                )
                for item in event.classifications
            ],
            review_history=[_review(item) for item in event.reviews],
            artifacts=[
                EventArtifactResponse(
                    id=item.id,
                    kind=item.kind,
                    media_type=item.media_type,
                )
                for item in sorted(event.artifacts, key=lambda row: row.kind)
            ],
            run=EventRunResponse(
                id=run.id,
                source_file_id=run.source_file_id,
                source_fingerprint=run.source_fingerprint,
                processing_profile_id=run.processing_profile_id,
                requested_backend=run.requested_backend,
                actual_backend=run.actual_backend,
                backend_fallback_reason=run.backend_fallback_reason,
                profile_version=run.profile_version,
                pipeline_version=run.pipeline_version,
                model_version_id=run.model_version_id,
                timings_ms=json.loads(run.timings_json),
                started_at=run.started_at,
                finished_at=run.finished_at,
            ),
        )


@router.post(
    "/{event_id}/reviews",
    response_model=ReviewResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_review(
    event_id: str,
    payload: ReviewCreate,
    request: Request,
    admin: CsrfAdmin,
) -> ReviewResponse:
    try:
        async with request.app.state.write_coordinator.lock:
            review = await run_in_threadpool(
                _append_review,
                request.app.state.session_factory,
                event_id,
                payload,
                admin.id,
            )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event not found",
        ) from error
    return review


def _append_review(
    sessions,
    event_id: str,
    payload: ReviewCreate,
    admin_id: int,
) -> ReviewResponse:
    with sessions.begin() as session:
        row = Repositories(session).reviews.add(
            event_id,
            label=payload.label,
            note=payload.note,
            admin_id=admin_id,
        )
        return _review(row)


def _summary(
    session,
    event: Event,
    *,
    site_id: str | None = None,
) -> EventSummaryResponse:
    if site_id is None:
        run = session.get(ProcessingRun, event.processing_run_id)
        source = None if run is None else session.get(SourceFile, run.source_file_id)
        if source is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Event source not found",
            )
        site_id = source.site_id
    latest = event.latest_review
    return EventSummaryResponse(
        id=event.id,
        event_uid=event.event_uid,
        site_id=site_id,
        occurred_at=event.occurred_at,
        start_sample=event.start_sample,
        end_sample=event.end_sample,
        channel_start=event.channel_start,
        channel_stop=event.channel_stop,
        machine_label=event.machine_label,
        latest_review=None if latest is None else _review(latest),
    )


def _review(row: Review) -> ReviewResponse:
    return ReviewResponse(
        id=row.id,
        label=row.label,
        note=row.note,
        created_at=row.created_at,
    )


def _utc_query(value: datetime | None, field: str) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field} must be timezone-aware",
        )
    return value.astimezone(UTC)
