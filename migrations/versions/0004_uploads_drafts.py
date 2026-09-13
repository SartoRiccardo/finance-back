"""uploads, drafts, app_settings + retarget transactions.draft_id

V2 created draft_id with a placeholder FK to transactions.id (drafts didn't
exist yet); V5 stores drafts.id in it, so the FK moves to drafts.

Revision ID: 0004_uploads_drafts
Revises: 0003_taxonomy_colors
Create Date: 2026-09-13
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_uploads_drafts"
down_revision = "0003_taxonomy_colors"
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "uploads",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("stored_path", sa.String(200), nullable=False),
        sa.Column("original_name", sa.String(), nullable=True),
        sa.Column("mime_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "drafts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.String(8), nullable=False, server_default="photo"),
        sa.Column("upload_id", sa.Uuid(), sa.ForeignKey("uploads.id"), nullable=True),
        sa.Column("email_meta", JSON, nullable=True),
        sa.Column("raw_llm_output", JSON, nullable=True),
        sa.Column("status", sa.String(8), nullable=False, server_default="open"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "app_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("llm_provider", sa.String(16), nullable=False, server_default="google"),
        sa.Column("llm_model", sa.String(120), nullable=False, server_default="gemini-2.5-flash"),
    )
    # V2's placeholder FK: draft_id holds drafts.id from V5 on.
    op.drop_constraint("transactions_draft_id_fkey", "transactions", type_="foreignkey")
    op.create_foreign_key(
        "fk_transactions_draft_id_drafts", "transactions", "drafts", ["draft_id"], ["id"]
    )


def downgrade() -> None:
    op.drop_constraint("fk_transactions_draft_id_drafts", "transactions", type_="foreignkey")
    op.create_foreign_key(
        "transactions_draft_id_fkey", "transactions", "transactions", ["draft_id"], ["id"]
    )
    op.drop_table("app_settings")
    op.drop_table("drafts")
    op.drop_table("uploads")
