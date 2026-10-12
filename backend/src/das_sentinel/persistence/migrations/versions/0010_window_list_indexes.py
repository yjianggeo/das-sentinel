"""Cover window-list metadata without rewriting waveform or review records."""

from alembic import op
from das_sentinel.persistence.model_scan_models import ModelScanReceipt
from das_sentinel.persistence.window_models import WindowAnalysis, WindowReview
from sqlalchemy.schema import CreateIndex

revision = "0010_window_list_indexes"
down_revision = "0009_window_spatial_coverage"
branch_labels = None
depends_on = None


def upgrade():
    for table, name in (
        (WindowAnalysis.__table__, "ix_window_analysis_list"),
        (ModelScanReceipt.__table__, "ix_model_scan_receipt_ready_window"),
        (WindowReview.__table__, "ix_window_review_latest_bound"),
    ):
        index = next(index for index in table.indexes if index.name == name)
        # SQLite expression indexes are not exposed by SQLAlchemy's reflection.
        op.get_bind().execute(CreateIndex(index, if_not_exists=True))
        op.get_bind().exec_driver_sql("ANALYZE " + name)


def downgrade():
    raise RuntimeError("List indexes require an explicit preserving rollback")
