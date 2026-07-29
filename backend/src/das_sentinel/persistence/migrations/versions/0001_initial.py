"""Create the initial DAS Sentinel persistence schema."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "admins",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("disabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_admin_singleton"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "sites",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("data_path", sa.Text(), nullable=False),
        sa.Column("timezone", sa.String(length=128), nullable=False),
        sa.Column("cable_type", sa.String(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "processing_profiles",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("site_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("site_id", "version", name="uq_profile_site_version"),
    )
    op.create_index(
        "ix_processing_profiles_site_id",
        "processing_profiles",
        ["site_id"],
        unique=False,
    )
    op.create_index(
        "uq_profile_current_site",
        "processing_profiles",
        ["site_id"],
        unique=True,
        sqlite_where=sa.text("is_current = 1"),
    )
    op.execute(
        """
        CREATE TRIGGER processing_profiles_immutable
        BEFORE UPDATE OF site_id, version, payload_json ON processing_profiles
        WHEN NEW.site_id IS NOT OLD.site_id
          OR NEW.version IS NOT OLD.version
          OR NEW.payload_json IS NOT OLD.payload_json
        BEGIN
            SELECT RAISE(ABORT, 'processing profile identity and payload are immutable');
        END
        """
    )
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("site_id", sa.String(length=36), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("dataset_sha256", sa.String(length=64), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("published_at", sa.String(length=32), nullable=True),
        sa.CheckConstraint(
            "status IN ('candidate','active','retired','invalid')",
            name="ck_model_status",
        ),
        sa.CheckConstraint(
            "length(bundle_sha256) = 64 AND bundle_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_model_bundle_sha256",
        ),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("site_id", "version", name="uq_model_site_version"),
    )
    op.create_index(
        "ix_models_site_status",
        "model_versions",
        ["site_id", "status"],
        unique=False,
    )
    op.create_table(
        "source_files",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("site_id", sa.String(length=36), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("mtime_ns", sa.Integer(), nullable=False),
        sa.Column("discovered_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "length(fingerprint) = 64 AND fingerprint NOT GLOB '*[^0-9a-f]*'",
            name="ck_source_fingerprint_sha256",
        ),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "site_id", "fingerprint", name="uq_source_site_fingerprint"
        ),
    )
    op.create_index(
        "ix_source_site_path",
        "source_files",
        ["site_id", "path"],
        unique=False,
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("source_file_id", sa.String(length=36), nullable=False),
        sa.Column("processing_profile_id", sa.String(length=36), nullable=False),
        sa.Column("pipeline_version", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.String(length=200), nullable=True),
        sa.Column("heartbeat_at", sa.String(length=32), nullable=True),
        sa.Column("progress_numerator", sa.Integer(), nullable=False),
        sa.Column("progress_denominator", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_eligible", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.Column("updated_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued','running','succeeded','failed','cancelled')",
            name="ck_job_status",
        ),
        sa.ForeignKeyConstraint(["processing_profile_id"], ["processing_profiles.id"]),
        sa.ForeignKeyConstraint(
            ["source_file_id"], ["source_files.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_jobs_status_created",
        "jobs",
        ["status", "created_at"],
        unique=False,
    )
    op.create_table(
        "processing_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("source_file_id", sa.String(length=36), nullable=False),
        sa.Column("processing_profile_id", sa.String(length=36), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("source_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("profile_version", sa.Integer(), nullable=False),
        sa.Column("pipeline_version", sa.String(length=128), nullable=False),
        sa.Column("requested_backend", sa.String(length=16), nullable=False),
        sa.Column("actual_backend", sa.String(length=16), nullable=False),
        sa.Column("backend_fallback_reason", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("timings_json", sa.Text(), nullable=False),
        sa.Column("started_at", sa.String(length=32), nullable=True),
        sa.Column("finished_at", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending','running','succeeded','failed')",
            name="ck_run_status",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(["processing_profile_id"], ["processing_profiles.id"]),
        sa.ForeignKeyConstraint(["source_file_id"], ["source_files.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
        sa.UniqueConstraint(
            "source_file_id",
            "processing_profile_id",
            "pipeline_version",
            name="uq_run_source_profile_pipeline",
        ),
    )
    op.create_table(
        "events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("processing_run_id", sa.String(length=36), nullable=False),
        sa.Column("event_uid", sa.String(length=128), nullable=False),
        sa.Column("start_sample", sa.Integer(), nullable=False),
        sa.Column("end_sample", sa.Integer(), nullable=False),
        sa.Column("channel_start", sa.Integer(), nullable=False),
        sa.Column("channel_stop", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.String(length=32), nullable=False),
        sa.Column("features_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "start_sample >= 0 AND end_sample > start_sample",
            name="ck_event_sample_range",
        ),
        sa.CheckConstraint(
            "channel_start >= 0 AND channel_stop > channel_start",
            name="ck_event_channel_range",
        ),
        sa.ForeignKeyConstraint(
            ["processing_run_id"], ["processing_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("processing_run_id", "event_uid", name="uq_event_run_uid"),
    )
    op.create_index(
        "ix_events_run_occurred",
        "events",
        ["processing_run_id", "occurred_at"],
        unique=False,
    )
    op.create_table(
        "detection_evidence",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("detector", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_evidence_event_detector",
        "detection_evidence",
        ["event_id", "detector"],
        unique=False,
    )
    op.create_table(
        "classifications",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("decision_score", sa.Float(), nullable=False),
        sa.Column("method", sa.String(length=64), nullable=False),
        sa.Column("reasons_json", sa.Text(), nullable=False),
        sa.Column("model_version_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "label IN ('earthquake','blast','traffic','noise','unknown')",
            name="ck_classification_label",
        ),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_classifications_event_created",
        "classifications",
        ["event_id", "created_at", "id"],
        unique=False,
    )
    op.execute(
        """
        CREATE TRIGGER classifications_append_only_update
        BEFORE UPDATE ON classifications
        BEGIN
            SELECT RAISE(ABORT, 'classifications are append-only');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER classifications_append_only_delete
        BEFORE DELETE ON classifications
        BEGIN
            SELECT RAISE(ABORT, 'classifications are append-only');
        END
        """
    )
    op.create_table(
        "reviews",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("admin_id", sa.Integer(), nullable=True),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "label IN ('earthquake','blast','traffic','noise')",
            name="ck_review_label",
        ),
        sa.ForeignKeyConstraint(["admin_id"], ["admins.id"]),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_reviews_event_created",
        "reviews",
        ["event_id", "created_at", "id"],
        unique=False,
    )
    op.execute(
        """
        CREATE TRIGGER reviews_append_only_update
        BEFORE UPDATE ON reviews
        BEGIN
            SELECT RAISE(ABORT, 'reviews are append-only');
        END
        """
    )
    op.execute(
        """
        CREATE TRIGGER reviews_append_only_delete
        BEFORE DELETE ON reviews
        BEGIN
            SELECT RAISE(ABORT, 'reviews are append-only');
        END
        """
    )
    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("media_type", sa.String(length=128), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_artifact_sha256",
        ),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id", "kind", name="uq_artifact_event_kind"),
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS reviews_append_only_delete")
    op.execute("DROP TRIGGER IF EXISTS reviews_append_only_update")
    op.execute("DROP TRIGGER IF EXISTS classifications_append_only_delete")
    op.execute("DROP TRIGGER IF EXISTS classifications_append_only_update")
    op.execute("DROP TRIGGER IF EXISTS processing_profiles_immutable")
    op.drop_table("artifacts")
    op.drop_index("ix_reviews_event_created", table_name="reviews")
    op.drop_table("reviews")
    op.drop_index("ix_classifications_event_created", table_name="classifications")
    op.drop_table("classifications")
    op.drop_index("ix_evidence_event_detector", table_name="detection_evidence")
    op.drop_table("detection_evidence")
    op.drop_index("ix_events_run_occurred", table_name="events")
    op.drop_table("events")
    op.drop_table("processing_runs")
    op.drop_index("ix_jobs_status_created", table_name="jobs")
    op.drop_table("jobs")
    op.drop_index("ix_source_site_path", table_name="source_files")
    op.drop_table("source_files")
    op.drop_index("ix_models_site_status", table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("uq_profile_current_site", table_name="processing_profiles")
    op.drop_index("ix_processing_profiles_site_id", table_name="processing_profiles")
    op.drop_table("processing_profiles")
    op.drop_table("sites")
    op.drop_table("admins")
