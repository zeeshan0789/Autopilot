"""create traces table

Revision ID: 0001
Revises:
Create Date: 2026-10-24

Baseline: the traces table exactly as created by Base.metadata.create_all in step 4.1.
Databases that already have this table must be marked as migrated with
"alembic stamp 0001" instead of running this migration.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "traces",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("api_key_id", sa.String(length=16), nullable=True),
        sa.Column("requested_model", sa.String(length=128), nullable=False),
        sa.Column("served_model", sa.String(length=128), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column(
            "failed_attempts",
            postgresql.JSONB(none_as_null=True, astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Numeric(precision=12, scale=8), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_traces_created_at"), "traces", ["created_at"], unique=False)
    op.create_index(op.f("ix_traces_request_id"), "traces", ["request_id"], unique=False)
    op.create_index(op.f("ix_traces_api_key_id"), "traces", ["api_key_id"], unique=False)
    op.create_index(op.f("ix_traces_status"), "traces", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_traces_status"), table_name="traces")
    op.drop_index(op.f("ix_traces_api_key_id"), table_name="traces")
    op.drop_index(op.f("ix_traces_request_id"), table_name="traces")
    op.drop_index(op.f("ix_traces_created_at"), table_name="traces")
    op.drop_table("traces")