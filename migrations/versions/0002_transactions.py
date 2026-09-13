"""categories, labels, transactions + seed data

Revision ID: 0002_transactions
Revises: 0001_users
Create Date: 2026-09-13
"""

import sqlalchemy as sa
from alembic import op

from app.models import seed

revision = "0002_transactions"
down_revision = "0001_users"
branch_labels = None
depends_on = None

DIRECTION_CHECK = (
    "(direction = 'spend' AND category_id IS NOT NULL AND label_id IS NULL) OR "
    "(direction = 'earn' AND category_id IS NULL AND label_id IS NOT NULL)"
)


def upgrade() -> None:
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("is_investment", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "labels",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("is_spending", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("description", sa.String(500), nullable=False),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("category_id", sa.Integer(), sa.ForeignKey("categories.id"), nullable=True),
        sa.Column("label_id", sa.Integer(), sa.ForeignKey("labels.id"), nullable=True),
        sa.Column("is_draft", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("draft_id", sa.Integer(), sa.ForeignKey("transactions.id"), nullable=True),
        sa.Column("source", sa.String(8), nullable=False, server_default="manual"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("amount > 0", name="ck_transactions_amount_positive"),
        sa.CheckConstraint(DIRECTION_CHECK, name="ck_transactions_direction"),
    )
    seed(op.get_bind())


def downgrade() -> None:
    op.drop_table("transactions")
    op.drop_table("labels")
    op.drop_table("categories")
