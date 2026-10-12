"""Add whole-window tables without changing legacy records."""

from alembic import op
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowReview,
)

revision = "0003_event_windows"
down_revision = "0002_job_model_and_review_request"
branch_labels = None
depends_on = None


def upgrade():
    for table in (
        EventWindow.__table__,
        WindowAnalysis.__table__,
        WindowReview.__table__,
    ):
        table.create(op.get_bind(), checkfirst=True)
    op.execute(
        """CREATE TRIGGER window_analysis_plan_immutable BEFORE UPDATE OF window_id, profile_id, plan_json, plan_sha256, algorithm_version ON window_analyses BEGIN SELECT RAISE(ABORT, 'window analysis plan is immutable'); END"""
    )
    op.execute(
        """CREATE TRIGGER window_review_immutable BEFORE UPDATE ON window_reviews BEGIN SELECT RAISE(ABORT, 'window review is immutable'); END"""
    )
    op.execute(
        """CREATE TRIGGER window_identity_immutable BEFORE UPDATE OF site_id, start_utc, end_utc ON event_windows BEGIN SELECT RAISE(ABORT, 'window identity is immutable'); END"""
    )


def downgrade():
    for table in (
        WindowReview.__table__,
        WindowAnalysis.__table__,
        EventWindow.__table__,
    ):
        table.drop(op.get_bind())
