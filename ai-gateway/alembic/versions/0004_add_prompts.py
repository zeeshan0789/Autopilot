"""add managed prompts and prompt columns on traces

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-28
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prompts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("prompt_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("user_template", sa.Text(), nullable=True),
        sa.Column("weight", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["prompt_id"], ["prompts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prompt_id", "version"),
    )
    op.create_index(op.f("ix_prompt_versions_prompt_id"), "prompt_versions", ["prompt_id"], unique=False)
    op.add_column("traces", sa.Column("prompt_name", sa.String(length=64), nullable=True))
    op.add_column("traces", sa.Column("prompt_version", sa.Integer(), nullable=True))
    op.create_index(op.f("ix_traces_prompt_name"), "traces", ["prompt_name"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_traces_prompt_name"), table_name="traces")
    op.drop_column("traces", "prompt_version")
    op.drop_column("traces", "prompt_name")
    op.drop_index(op.f("ix_prompt_versions_prompt_id"), table_name="prompt_versions")
    op.drop_table("prompt_versions")
    op.drop_table("prompts")