"""Add independent bookmarks without changing analyses or human labels."""

from alembic import op
from das_sentinel.persistence.window_models import WindowFavorite

revision = "0008_window_favorites"
down_revision = "0007_model_scans"
branch_labels = None
depends_on = None


def upgrade():
    WindowFavorite.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("Window favorites require an explicit preserving rollback")
