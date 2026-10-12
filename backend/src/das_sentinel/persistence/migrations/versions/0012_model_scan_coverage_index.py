"""Cover acquisition-time success evidence without reading model receipt payloads."""

from alembic import op
from das_sentinel.persistence.model_scan_models import ModelScanReceipt
from sqlalchemy.schema import CreateIndex

revision = "0012_model_scan_coverage_index"
down_revision = "0011_model_scan_summary_index"
branch_labels = None
depends_on = None


def upgrade():
    index = next(index for index in ModelScanReceipt.__table__.indexes
                 if index.name == "ix_model_scan_receipt_coverage")
    op.get_bind().execute(CreateIndex(index, if_not_exists=True))
    op.get_bind().exec_driver_sql("ANALYZE ix_model_scan_receipt_coverage")


def downgrade():
    op.drop_index("ix_model_scan_receipt_coverage", table_name="model_scan_receipts", if_exists=True)
