from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select

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
from das_sentinel.classifiers.similarity import SimilarityIndex
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
MachineLabelValue = Literal[
    "earthquake",
    "blast",
    "traffic",
    "noise",
    "unknown",
]


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

    latest_machine_label = (
        select(Classification.label)
        .where(Classification.event_id == Event.id)
        .order_by(
            Classification.created_at.desc(),
            Classification.id.desc(),
        )
        .limit(1)
        .correlate(Event)
        .scalar_subquery()
    )
    latest_review_id = (
        select(Review.id)
        .where(Review.event_id == Event.id)
        .order_by(
            Review.created_at.desc(),
            Review.id.desc(),
        )
        .limit(1)
        .correlate(Event)
        .scalar_subquery()
    )
    statement = (
        select(
            Event,
            SourceFile.site_id,
            func.coalesce(latest_machine_label, "unknown"),
            Review,
        )
        .join(ProcessingRun, ProcessingRun.id == Event.processing_run_id)
        .join(SourceFile, SourceFile.id == ProcessingRun.source_file_id)
        .outerjoin(Review, Review.id == latest_review_id)
    )
    if site_id is not None:
        statement = statement.where(SourceFile.site_id == site_id)
    if start_utc is not None:
        statement = statement.where(Event.occurred_at >= start_utc)
    if end_utc is not None:
        statement = statement.where(Event.occurred_at <= end_utc)
    if machine_label is not None:
        statement = statement.where(
            func.coalesce(latest_machine_label, "unknown") == machine_label
        )
    if reviewed is not None:
        statement = statement.where(
            latest_review_id.is_not(None)
            if reviewed
            else latest_review_id.is_(None)
        )
    statement = statement.order_by(
        Event.occurred_at.desc(),
        Event.id.desc(),
    ).limit(limit).offset(offset)

    with request.app.state.session_factory() as session:
        rows = session.execute(statement).all()
        return [
            _list_summary(
                event,
                row_site_id,
                cast(MachineLabelValue, label),
                latest_review,
            )
            for event, row_site_id, label, latest_review in rows
        ]


@router.get(
    "/{event_id}/similar",
    response_model=list[EventSummaryResponse],
)
def similar_events(
    event_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
    limit: int = Query(default=5, ge=1, le=20),
) -> list[EventSummaryResponse]:
    del admin
    with request.app.state.session_factory() as session:
        event = session.get(Event, event_id)
        if event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Event not found",
            )
        run = session.get(ProcessingRun, event.processing_run_id)
        source = None if run is None else session.get(SourceFile, run.source_file_id)
        if source is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Event source not found",
            )
        statement = (
            select(Event)
            .join(ProcessingRun, ProcessingRun.id == Event.processing_run_id)
            .join(SourceFile, SourceFile.id == ProcessingRun.source_file_id)
            .where(SourceFile.site_id == source.site_id)
            .order_by(Event.id)
        )
        peers = [
            SimpleNamespace(
                event_id=peer.id,
                features={
                    str(name): float(value)
                    for name, value in json.loads(peer.features_json).items()
                },
            )
            for peer in session.scalars(statement)
        ]
        try:
            index = SimilarityIndex().fit(peers)
            result = index.query(
                {
                    str(name): float(value)
                    for name, value in json.loads(event.features_json).items()
                },
                limit=limit + 1,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(error),
            ) from error
        neighbours = []
        for neighbour in result.neighbors:
            if neighbour.event_id == event_id:
                continue
            peer = session.get(Event, neighbour.event_id)
            if peer is None:
                continue
            neighbours.append(_summary(session, peer, site_id=source.site_id))
            if len(neighbours) >= limit:
                break
        return neighbours


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
        machine_label=cast(MachineLabelValue, event.machine_label),
        latest_review=None if latest is None else _review(latest),
    )


def _list_summary(
    event: Event,
    site_id: str,
    machine_label: MachineLabelValue,
    latest_review: Review | None,
) -> EventSummaryResponse:
    return EventSummaryResponse(
        id=event.id,
        event_uid=event.event_uid,
        site_id=site_id,
        occurred_at=event.occurred_at,
        start_sample=event.start_sample,
        end_sample=event.end_sample,
        channel_start=event.channel_start,
        channel_stop=event.channel_stop,
        machine_label=machine_label,
        latest_review=None if latest_review is None else _review(latest_review),
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
