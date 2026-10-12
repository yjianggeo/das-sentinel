from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from das_sentinel.api.auth import AuthenticatedAdmin, CsrfAdmin
from das_sentinel.persistence.models import Site

router = APIRouter(prefix="/low-frequency/scans", tags=["low-frequency"])


class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site_id: str = Field(min_length=1, max_length=36)
    relative_path: str = Field(min_length=1, max_length=512)
    start_utc: datetime
    end_utc: datetime
    distance_start_m: float = Field(
        default=499.0, ge=0, strict=True, allow_inf_nan=False
    )
    distance_stop_m: float | None = Field(
        default=None, ge=0, strict=True, allow_inf_nan=False
    )

    @field_validator("start_utc", "end_utc")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("起止时间必须带时区")
        return value

    @model_validator(mode="after")
    def ranges(self):
        if not 600 <= (self.end_utc - self.start_utc).total_seconds() <= 1200:
            raise ValueError("低频有效扫描窗必须为600–1200秒，两侧另需连续保护数据")
        if (
            self.distance_stop_m is not None
            and self.distance_stop_m <= self.distance_start_m
        ):
            raise ValueError("结束距离必须大于起始距离")
        return self


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["earthquake", "blast", "traffic", "noise", "unknown"]
    notes: str = Field(default="", max_length=4000)
    request_id: UUID


@router.get("")
def list_scans(request: Request, admin: AuthenticatedAdmin):
    del admin
    return request.app.state.low_frequency_service.list()[:50]


@router.post("", status_code=202)
def create_scan(payload: ScanRequest, request: Request, admin: CsrfAdmin):
    del admin
    with request.app.state.session_factory() as session:
        site = session.get(Site, payload.site_id)
        if site is None:
            raise HTTPException(404, "站点不存在")
        if not site.enabled:
            raise HTTPException(409, "站点已停用")
        site_root = Path(site.data_path)
        site_timezone = site.timezone
    try:
        return request.app.state.low_frequency_service.create(
            {**payload.model_dump(mode="json"), "site_timezone": site_timezone},
            site_root,
        )
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error


@router.get("/{scan_id}")
def get_scan(scan_id: str, request: Request, admin: AuthenticatedAdmin):
    del admin
    try:
        return request.app.state.low_frequency_service.get(scan_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error


@router.get("/{scan_id}/plots/{kind}")
def get_plot(scan_id: str, kind: str, request: Request, admin: AuthenticatedAdmin):
    del admin
    try:
        path = request.app.state.low_frequency_service.plot(scan_id, kind)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    return FileResponse(path, media_type="image/png")


@router.post("/{scan_id}/candidates/{candidate_id}/reviews", status_code=201)
def review_candidate(
    scan_id: str,
    candidate_id: str,
    payload: ReviewRequest,
    request: Request,
    admin: CsrfAdmin,
):
    del admin
    try:
        return request.app.state.low_frequency_service.review(
            scan_id, candidate_id, payload.label, payload.notes, str(payload.request_id)
        )
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
