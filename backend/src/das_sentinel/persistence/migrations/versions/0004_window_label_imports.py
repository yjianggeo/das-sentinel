"""Preserve authorized legacy label conversion independently of whole-window truth."""

from alembic import op
from das_sentinel.persistence.window_models import WindowLabelImport

revision = "0004_window_label_imports"
down_revision = "0003_event_windows"
branch_labels = None
depends_on = None


def upgrade():
    WindowLabelImport.__table__.create(op.get_bind(), checkfirst=True)
    op.execute(
        "CREATE TRIGGER window_label_import_immutable BEFORE UPDATE OF "
        "window_id, analysis_id, payload_json, admin_id, created_at ON window_label_imports "
        "BEGIN SELECT RAISE(ABORT, 'window label import is immutable'); END"
    )


def downgrade():
    WindowLabelImport.__table__.drop(op.get_bind())
