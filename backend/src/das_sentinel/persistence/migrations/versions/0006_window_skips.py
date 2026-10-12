"""Add opt-in skip authorization and compact empty-window completion receipts."""
from alembic import op
from das_sentinel.persistence.window_models import WindowSkip
from sqlalchemy import Column, String, inspect

revision = "0006_window_skips"
down_revision = "0005_window_duration"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    if "skip_policy_version" not in {c["name"] for c in inspect(bind).get_columns("window_analyses")}:
        op.add_column("window_analyses", Column("skip_policy_version", String(100), nullable=True))
    WindowSkip.__table__.create(bind, checkfirst=True)


def downgrade():
    raise RuntimeError("Empty-window receipts require an explicit preserving rollback")
