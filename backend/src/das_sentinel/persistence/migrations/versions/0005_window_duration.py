"""Preserve 60-second history while adding distinct half-minute window identities."""

from alembic import op
from sqlalchemy import Column, Integer, inspect

revision = "0005_window_duration"
down_revision = "0004_window_label_imports"
branch_labels = None
depends_on = None


def upgrade():
    # Fresh 0003 installs use current model metadata; deployed 0003 tables need a rebuild.
    if "window_seconds" not in {c["name"] for c in inspect(op.get_bind()).get_columns("event_windows")}:
        with op.batch_alter_table("event_windows") as batch:
            batch.add_column(Column("window_seconds", Integer, nullable=False, server_default="60"))
            batch.drop_constraint("uq_window_site_minute", type_="unique")
            batch.create_unique_constraint("uq_window_site_duration", ["site_id", "start_utc", "window_seconds"])
    op.execute("DROP TRIGGER IF EXISTS window_identity_immutable")
    op.execute(
        "CREATE TRIGGER window_identity_immutable BEFORE UPDATE OF "
        "site_id, start_utc, end_utc, window_seconds ON event_windows "
        "BEGIN SELECT RAISE(ABORT, 'window identity is immutable'); END"
    )


def downgrade():
    raise RuntimeError("30-second data requires an explicit preserving rollback, not schema downgrade")
