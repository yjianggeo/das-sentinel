"""Add model-first scan journals without rewriting existing review records."""

from alembic import op
from das_sentinel.persistence.model_scan_models import (
    ModelScan,
    ModelScanReceipt,
    ModelScanSource,
    ModelScanUnit,
)

revision = "0007_model_scans"
down_revision = "0006_window_skips"
branch_labels = None
depends_on = None


def upgrade():
    for model in (ModelScan, ModelScanUnit, ModelScanSource, ModelScanReceipt):
        model.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("Model scan receipts require an explicit preserving rollback")
