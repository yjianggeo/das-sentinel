"""Cover task summary metadata without reading large unit payloads."""

from alembic import op
from das_sentinel.persistence.model_scan_models import ModelScanUnit
from sqlalchemy.schema import CreateIndex

revision = "0011_model_scan_summary_index"
down_revision = "0010_window_list_indexes"
branch_labels = None
depends_on = None


def upgrade():
    index = next(index for index in ModelScanUnit.__table__.indexes
                 if index.name == "ix_model_scan_unit_summary")
    op.get_bind().execute(CreateIndex(index, if_not_exists=True))
    op.get_bind().exec_driver_sql("ANALYZE ix_model_scan_unit_summary")


def downgrade():
    raise RuntimeError("Task summary index requires an explicit preserving rollback")
