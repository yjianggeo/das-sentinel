"""Add immutable job model binding and review request idempotency."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_job_model_and_review_request"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE jobs ADD COLUMN model_version_id "
        "VARCHAR(36) REFERENCES model_versions(id)"
    )
    op.add_column("reviews", sa.Column("request_id", sa.String(36), nullable=True))
    op.create_index(
        "uq_review_event_request",
        "reviews",
        ["event_id", "request_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_review_event_request", table_name="reviews")
    op.drop_column("reviews", "request_id")
    op.drop_column("jobs", "model_version_id")
