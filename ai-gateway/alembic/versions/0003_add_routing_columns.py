"""add routing policy columns to traces

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-26
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("traces", sa.Column("routed_model", sa.String(length=128), nullable=True))
    op.add_column("traces", sa.Column("policy_rule", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("traces", "policy_rule")
    op.drop_column("traces", "routed_model")