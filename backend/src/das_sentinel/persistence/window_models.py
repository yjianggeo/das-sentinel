"""Whole-window records; legacy event and review semantics remain separate."""

from datetime import datetime

from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from das_sentinel.persistence.models import Base, UTCDateTime, new_id, utc_now


class EventWindow(Base):
    __tablename__ = "event_windows"
    __table_args__ = (
        UniqueConstraint("site_id", "start_utc", "window_seconds", name="uq_window_site_duration"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"), index=True)
    start_utc: Mapped[datetime] = mapped_column(UTCDateTime())
    end_utc: Mapped[datetime] = mapped_column(UTCDateTime())
    window_seconds: Mapped[int] = mapped_column(Integer, default=60, server_default="60")
    priority: Mapped[int] = mapped_column(Integer, default=0)
    catalog_matches_json: Mapped[str] = mapped_column(Text, default="[]")
    legacy_event_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    latest_analysis_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


class WindowAnalysis(Base):
    __tablename__ = "window_analyses"
    __table_args__ = (
        UniqueConstraint("window_id", "plan_sha256", name="uq_window_analysis_plan"),
        Index("ix_window_analysis_list", "id", "profile_id", "status", "input_sha256",
              text("json_extract(result_json, '$.candidate')"),
              text("json_extract(result_json, '$.score')")),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    window_id: Mapped[str] = mapped_column(ForeignKey("event_windows.id"), index=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey("processing_profiles.id"))
    plan_json: Mapped[str] = mapped_column(Text)
    plan_sha256: Mapped[str] = mapped_column(String(64))
    algorithm_version: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), index=True)
    worker_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    skip_policy_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    input_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class WindowSpatialCoverage(Base):
    """Versioned measurements bound to immutable analysis input, never labels."""

    __tablename__ = "window_spatial_coverage"
    analysis_id: Mapped[str] = mapped_column(
        ForeignKey("window_analyses.id", ondelete="CASCADE"), primary_key=True,
    )
    input_sha256: Mapped[str] = mapped_column(String(64))
    algorithm_version: Mapped[str] = mapped_column(String(100))
    score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    payload_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class WindowFavorite(Base):
    """A bookmark belongs to the stable window, independently of review labels."""

    __tablename__ = "window_favorites"
    window_id: Mapped[str] = mapped_column(
        ForeignKey("event_windows.id", ondelete="CASCADE"), primary_key=True,
    )
    admin_id: Mapped[int] = mapped_column(
        ForeignKey("admins.id", ondelete="CASCADE"), primary_key=True, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class WindowReview(Base):
    __tablename__ = "window_reviews"
    __table_args__ = (
        UniqueConstraint("window_id", "request_id", name="uq_window_review_request"),
        Index("ix_window_review_latest_bound", "window_id", "created_at", "id",
              "analysis_id", "input_sha256", "labels_json"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    window_id: Mapped[str] = mapped_column(ForeignKey("event_windows.id"), index=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("window_analyses.id"))
    input_sha256: Mapped[str] = mapped_column(String(64))
    labels_json: Mapped[str] = mapped_column(Text)
    note: Mapped[str] = mapped_column(Text, default="")
    request_id: Mapped[str] = mapped_column(String(100))
    admin_id: Mapped[int] = mapped_column(ForeignKey("admins.id"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class WindowLabelImport(Base):
    """Frozen legacy evidence authorized for conservative positive-only conversion."""

    __tablename__ = "window_label_imports"
    window_id: Mapped[str] = mapped_column(ForeignKey("event_windows.id"), primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("window_analyses.id"))
    payload_json: Mapped[str] = mapped_column(Text)
    admin_id: Mapped[int] = mapped_column(ForeignKey("admins.id"))
    applied_review_id: Mapped[str | None] = mapped_column(
        ForeignKey("window_reviews.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class WindowSkip(Base):
    """Compact completion/deduplication receipt, never a training label."""

    __tablename__ = "window_skips"
    __table_args__ = (
        UniqueConstraint("window_id", "plan_sha256", "policy_version", name="uq_window_skip_plan"),
    )
    analysis_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    window_id: Mapped[str] = mapped_column(String(36))
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"))
    start_utc: Mapped[datetime] = mapped_column(UTCDateTime())
    window_seconds: Mapped[int] = mapped_column(Integer)
    plan_sha256: Mapped[str] = mapped_column(String(64))
    policy_version: Mapped[str] = mapped_column(String(100))
    worker_id: Mapped[str] = mapped_column(String(200))
    attempt_count: Mapped[int] = mapped_column(Integer)
    finished_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, index=True)
    reason: Mapped[str] = mapped_column(String(100))
