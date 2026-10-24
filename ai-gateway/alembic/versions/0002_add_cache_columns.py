"""add semantic cache columns to traces

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-24
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("traces", sa.Column("cache_status", sa.String(length=16), nullable=True))
    op.add_column("traces", sa.Column("cache_reason", sa.String(length=32), nullable=True))
    op.add_column("traces", sa.Column("cache_similarity", sa.Float(), nullable=True))
    op.add_column("traces", sa.Column("cache_entry_id", sa.String(length=32), nullable=True))
    op.add_column("traces", sa.Column("saved_tokens", sa.Integer(), nullable=True))
    op.add_column(
        "traces",
        sa.Column("saved_cost_usd", sa.Numeric(precision=12, scale=8), nullable=True),
    )
    op.create_index(op.f("ix_traces_cache_status"), "traces", ["cache_status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_traces_cache_status"), table_name="traces")
    op.drop_column("traces", "saved_cost_usd")
    op.drop_column("traces", "saved_tokens")
    op.drop_column("traces", "cache_entry_id")
    op.drop_column("traces", "cache_similarity")
    op.drop_column("traces", "cache_reason")
    op.drop_column("traces", "cache_status")