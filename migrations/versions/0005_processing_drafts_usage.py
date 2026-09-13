"""drafts run detached from the request (processing/error states) + llm_usage cost log

Revision ID: 0005_processing_drafts_usage
Revises: 0004_uploads_drafts
Create Date: 2026-09-13
"""

import sqlalchemy as sa
from alembic import op

revision = "0005_processing_drafts_usage"
down_revision = "0004_uploads_drafts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("drafts", "status", type_=sa.String(16))
    op.add_column("drafts", sa.Column("error", sa.String(500), nullable=True))
    op.create_table(
        "llm_usage",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.String(16), nullable=False),
        sa.Column("model", sa.String(120), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("draft_id", sa.Integer(), sa.ForeignKey("drafts.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("llm_usage")
    op.drop_column("drafts", "error")
    op.alter_column("drafts", "status", type_=sa.String(8))
