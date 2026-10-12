"""Cache spatial measurements separately from immutable waveform artifacts."""

from alembic import op
from das_sentinel.persistence.window_models import WindowSpatialCoverage

revision = "0009_window_spatial_coverage"
down_revision = "0008_window_favorites"
branch_labels = None
depends_on = None


def upgrade():
    WindowSpatialCoverage.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("Spatial measurements require an explicit preserving rollback")
