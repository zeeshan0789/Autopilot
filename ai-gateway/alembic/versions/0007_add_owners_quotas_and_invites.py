"""add owners, quotas and invites

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-22
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users: a status and two limits. The old "viewer" role becomes an ordinary member, and the
    # admin gets limits so large that they do not matter (the admin runs the platform).
    op.add_column("users", sa.Column("status", sa.String(length=16), server_default=sa.text("'active'"), nullable=False))
    op.add_column("users", sa.Column("daily_request_limit", sa.Integer(), server_default=sa.text("200"), nullable=False))
    op.add_column("users", sa.Column("max_tokens_cap", sa.Integer(), server_default=sa.text("1024"), nullable=False))
    op.alter_column("users", "role", server_default=sa.text("'member'"))
    op.execute("UPDATE users SET role = 'member' WHERE role = 'viewer'")
    op.execute("UPDATE users SET daily_request_limit = 1000000, max_tokens_cap = 100000 WHERE role = 'admin'")

    # Owners. Existing rows keep user_id NULL (no owner) until scripts/claim_legacy.py assigns them.
    for table in ("api_keys", "prompts", "traces"):
        op.add_column(table, sa.Column("user_id", sa.Integer(), nullable=True))
        op.create_index(op.f(f"ix_{table}_user_id"), table, ["user_id"], unique=False)
    op.create_foreign_key(None, "api_keys", "users", ["user_id"], ["id"], ondelete="CASCADE")
    op.create_foreign_key(None, "prompts", "users", ["user_id"], ["id"], ondelete="CASCADE")

    # A prompt name is unique per owner (and among the prompts without an owner)
    op.drop_constraint("prompts_name_key", "prompts", type_="unique")
    op.create_unique_constraint("prompts_user_id_name_key", "prompts", ["user_id", "name"])
    op.create_index(
        "uq_prompts_legacy_name", "prompts", ["name"], unique=True, postgresql_where=sa.text("user_id IS NULL")
    )

    op.create_table(
        "invites",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code_hash", sa.String(length=64), nullable=False),
        sa.Column("note", sa.String(length=200), nullable=True),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("used_by", sa.Integer(), nullable=True),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["used_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code_hash"),
    )


def downgrade() -> None:
    op.drop_table("invites")
    op.drop_index("uq_prompts_legacy_name", table_name="prompts")
    op.drop_constraint("prompts_user_id_name_key", "prompts", type_="unique")
    # Fails when two users have a prompt with the same name: remove one of them first
    op.create_unique_constraint("prompts_name_key", "prompts", ["name"])
    op.drop_constraint("prompts_user_id_fkey", "prompts", type_="foreignkey")
    op.drop_constraint("api_keys_user_id_fkey", "api_keys", type_="foreignkey")
    for table in ("traces", "prompts", "api_keys"):
        op.drop_index(op.f(f"ix_{table}_user_id"), table_name=table)
        op.drop_column(table, "user_id")
    op.alter_column("users", "role", server_default=sa.text("'admin'"))
    op.drop_column("users", "max_tokens_cap")
    op.drop_column("users", "daily_request_limit")
    op.drop_column("users", "status")