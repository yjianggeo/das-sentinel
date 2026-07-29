from __future__ import annotations

import json
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

SUPERVISED_LABELS = ("earthquake", "blast", "traffic", "noise")
MACHINE_LABELS = (*SUPERVISED_LABELS, "unknown")


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


def normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    impl = String(32)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> str | None:
        if value is None:
            return None
        return (
            normalize_utc(value)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )

    def process_result_value(self, value: str | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return datetime.fromisoformat(value).astimezone(UTC)


def freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(freeze_json(item) for item in value)
    return value


class Base(DeclarativeBase):
    pass


class Admin(Base):
    __tablename__ = "admins"
    __table_args__ = (CheckConstraint("id = 1", name="ck_admin_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    password_hash: Mapped[str] = mapped_column(String(512))
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class Site(Base):
    __tablename__ = "sites"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    data_path: Mapped[str] = mapped_column(Text)
    timezone: Mapped[str] = mapped_column(String(128))
    cable_type: Mapped[str] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    profiles: Mapped[list[ProcessingProfile]] = relationship(
        back_populates="site",
        foreign_keys=lambda: ProcessingProfile.site_id,
        cascade="all, delete-orphan",
    )


class ProcessingProfile(Base):
    __tablename__ = "processing_profiles"
    __table_args__ = (
        UniqueConstraint("site_id", "version", name="uq_profile_site_version"),
        Index(
            "uq_profile_current_site",
            "site_id",
            unique=True,
            sqlite_where=text("is_current = 1"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    site_id: Mapped[str] = mapped_column(
        ForeignKey("sites.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    site: Mapped[Site] = relationship(back_populates="profiles", foreign_keys=[site_id])

    @property
    def payload(self) -> MappingProxyType[str, Any]:
        return freeze_json(json.loads(self.payload_json))


class SourceFile(Base):
    __tablename__ = "source_files"
    __table_args__ = (
        UniqueConstraint("site_id", "fingerprint", name="uq_source_site_fingerprint"),
        CheckConstraint(
            "length(fingerprint) = 64 AND fingerprint NOT GLOB '*[^0-9a-f]*'",
            name="ck_source_fingerprint_sha256",
        ),
        Index("ix_source_site_path", "site_id", "path"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    path: Mapped[str] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    mtime_ns: Mapped[int] = mapped_column(Integer)
    discovered_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','running','succeeded','failed','cancelled')",
            name="ck_job_status",
        ),
        Index("ix_jobs_status_created", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_file_id: Mapped[str] = mapped_column(
        ForeignKey("source_files.id", ondelete="CASCADE")
    )
    processing_profile_id: Mapped[str] = mapped_column(
        ForeignKey("processing_profiles.id")
    )
    pipeline_version: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    progress_numerator: Mapped[int] = mapped_column(Integer, default=0)
    progress_denominator: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_eligible: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class ProcessingRun(Base):
    __tablename__ = "processing_runs"
    __table_args__ = (
        UniqueConstraint(
            "source_file_id",
            "processing_profile_id",
            "pipeline_version",
            name="uq_run_source_profile_pipeline",
        ),
        CheckConstraint(
            "status IN ('pending','running','succeeded','failed')",
            name="ck_run_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_id: Mapped[str | None] = mapped_column(
        ForeignKey("jobs.id"), unique=True, nullable=True
    )
    source_file_id: Mapped[str] = mapped_column(ForeignKey("source_files.id"))
    processing_profile_id: Mapped[str] = mapped_column(
        ForeignKey("processing_profiles.id")
    )
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    source_fingerprint: Mapped[str] = mapped_column(String(64))
    profile_version: Mapped[int] = mapped_column(Integer)
    pipeline_version: Mapped[str] = mapped_column(String(128))
    requested_backend: Mapped[str] = mapped_column(String(16))
    actual_backend: Mapped[str] = mapped_column(String(16))
    backend_fallback_reason: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), default="pending")
    timings_json: Mapped[str] = mapped_column(Text, default="{}")
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("processing_run_id", "event_uid", name="uq_event_run_uid"),
        CheckConstraint(
            "start_sample >= 0 AND end_sample > start_sample",
            name="ck_event_sample_range",
        ),
        CheckConstraint(
            "channel_start >= 0 AND channel_stop > channel_start",
            name="ck_event_channel_range",
        ),
        Index("ix_events_run_occurred", "processing_run_id", "occurred_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    processing_run_id: Mapped[str] = mapped_column(
        ForeignKey("processing_runs.id", ondelete="CASCADE")
    )
    event_uid: Mapped[str] = mapped_column(String(128))
    start_sample: Mapped[int] = mapped_column(Integer)
    end_sample: Mapped[int] = mapped_column(Integer)
    channel_start: Mapped[int] = mapped_column(Integer)
    channel_stop: Mapped[int] = mapped_column(Integer)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime())
    features_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    evidence: Mapped[list[DetectionEvidence]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        order_by=lambda: (DetectionEvidence.created_at, DetectionEvidence.id),
    )
    classifications: Mapped[list[Classification]] = relationship(
        back_populates="event",
        passive_deletes=True,
        order_by=lambda: (Classification.created_at, Classification.id),
    )
    reviews: Mapped[list[Review]] = relationship(
        back_populates="event",
        passive_deletes=True,
        order_by=lambda: (Review.created_at, Review.id),
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
    )

    @property
    def machine_label(self) -> str:
        if not self.classifications:
            return "unknown"
        return max(
            self.classifications,
            key=lambda item: (item.created_at, item.id),
        ).label

    @property
    def latest_review(self) -> Review | None:
        if not self.reviews:
            return None
        return max(
            self.reviews,
            key=lambda item: (item.created_at, item.id),
        )


class DetectionEvidence(Base):
    __tablename__ = "detection_evidence"
    __table_args__ = (Index("ix_evidence_event_detector", "event_id", "detector"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    detector: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    event: Mapped[Event] = relationship(back_populates="evidence")


class Classification(Base):
    __tablename__ = "classifications"
    __table_args__ = (
        CheckConstraint(
            "label IN ('earthquake','blast','traffic','noise','unknown')",
            name="ck_classification_label",
        ),
        Index("ix_classifications_event_created", "event_id", "created_at", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"))
    label: Mapped[str] = mapped_column(String(16))
    decision_score: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(64))
    reasons_json: Mapped[str] = mapped_column(Text, default="[]")
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    event: Mapped[Event] = relationship(back_populates="classifications")


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        CheckConstraint(
            "label IN ('earthquake','blast','traffic','noise')",
            name="ck_review_label",
        ),
        Index("ix_reviews_event_created", "event_id", "created_at", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"))
    admin_id: Mapped[int | None] = mapped_column(ForeignKey("admins.id"), nullable=True)
    label: Mapped[str] = mapped_column(String(16))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    event: Mapped[Event] = relationship(back_populates="reviews")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("site_id", "version", name="uq_model_site_version"),
        CheckConstraint(
            "status IN ('candidate','active','retired','invalid')",
            name="ck_model_status",
        ),
        CheckConstraint(
            "length(bundle_sha256) = 64 AND bundle_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_model_bundle_sha256",
        ),
        Index("ix_models_site_status", "site_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"))
    version: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="candidate")
    artifact_path: Mapped[str] = mapped_column(Text)
    bundle_sha256: Mapped[str] = mapped_column(String(64))
    dataset_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint("event_id", "kind", name="uq_artifact_event_kind"),
        CheckConstraint(
            "length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_artifact_sha256",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(64))
    path: Mapped[str] = mapped_column(Text)
    media_type: Mapped[str] = mapped_column(String(128))
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    event: Mapped[Event] = relationship(back_populates="artifacts")
