"""app_settings.custom_prompt: free-text additions appended to the extraction prompt

Revision ID: 0006_custom_prompt
Revises: 0005_processing_drafts_usage
Create Date: 2026-09-13
"""

import sqlalchemy as sa
from alembic import op

revision = "0006_custom_prompt"
down_revision = "0005_processing_drafts_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("app_settings", sa.Column("custom_prompt", sa.String(2000), nullable=True))


def downgrade() -> None:
    op.drop_column("app_settings", "custom_prompt")
