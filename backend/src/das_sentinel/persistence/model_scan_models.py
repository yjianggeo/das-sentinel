"""Durable model-first scans, separate from review labels and rule skips."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from das_sentinel.persistence.models import Base, UTCDateTime, new_id, utc_now


class ModelScan(Base):
    __tablename__ = "model_scans"
    __table_args__ = (UniqueConstraint("site_id", "request_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.id"), index=True)
    profile_id: Mapped[str] = mapped_column(ForeignKey("processing_profiles.id"))
    request_id: Mapped[str] = mapped_column(String(100))
    config_json: Mapped[str] = mapped_column(Text)
    config_sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class ModelScanUnit(Base):
    __tablename__ = "model_scan_units"
    __table_args__ = (
        UniqueConstraint("scan_id", "kind", "unit_key"),
        Index("ix_model_scan_unit_summary", "scan_id", "status", "finished_at", "kind",
              "started_at", "window_count", "heartbeat_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    scan_id: Mapped[str] = mapped_column(ForeignKey("model_scans.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    unit_key: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text)
    pages_json: Mapped[str] = mapped_column(Text, default="{}")
    window_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    worker_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class ModelScanSource(Base):
    __tablename__ = "model_scan_sources"
    __table_args__ = (UniqueConstraint("scan_id", "path"),
                      Index("ix_model_scan_source_time", "scan_id", "start_utc", "end_utc"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    scan_id: Mapped[str] = mapped_column(ForeignKey("model_scans.id"), index=True)
    discovery_unit_id: Mapped[str] = mapped_column(ForeignKey("model_scan_units.id"))
    path: Mapped[str] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[str] = mapped_column(Text)
    start_utc: Mapped[datetime] = mapped_column(UTCDateTime())
    end_utc: Mapped[datetime] = mapped_column(UTCDateTime())
    context_only: Mapped[bool] = mapped_column(Boolean, default=False)


class ModelScanReceipt(Base):
    __tablename__ = "model_scan_receipts"
    __table_args__ = (
        UniqueConstraint("scan_id", "start_utc"),
        Index("ix_model_scan_receipt_coverage", "start_utc", "scan_id", "score", "model_input_sha256"),
        Index("ix_model_scan_receipt_ready_window", "window_id", "status", "analysis_id",
              "input_sha256", "created_at", "id", "final_score", "scan_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    scan_id: Mapped[str] = mapped_column(ForeignKey("model_scans.id"), index=True)
    infer_unit_id: Mapped[str] = mapped_column(ForeignKey("model_scan_units.id"), index=True)
    start_utc: Mapped[datetime] = mapped_column(UTCDateTime())
    inference_key: Mapped[str] = mapped_column(String(64), index=True)
    reused_from: Mapped[str | None] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_input_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_json: Mapped[str] = mapped_column(Text)
    result_sha256: Mapped[str] = mapped_column(String(64))
    window_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    analysis_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    analysis_attempt_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
