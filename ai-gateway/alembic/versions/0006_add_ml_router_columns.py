"""add learned router columns to traces

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-16
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("traces", sa.Column("ml_pick", sa.String(length=128), nullable=True))
    op.add_column("traces", sa.Column("ml_status", sa.String(length=16), nullable=True))
    op.add_column("traces", sa.Column("ml_prob", sa.Float(), nullable=True))
    op.add_column("traces", sa.Column("ml_similarity", sa.Float(), nullable=True))
    op.create_index(op.f("ix_traces_ml_status"), "traces", ["ml_status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_traces_ml_status"), table_name="traces")
    op.drop_column("traces", "ml_similarity")
    op.drop_column("traces", "ml_prob")
    op.drop_column("traces", "ml_status")
    op.drop_column("traces", "ml_pick")