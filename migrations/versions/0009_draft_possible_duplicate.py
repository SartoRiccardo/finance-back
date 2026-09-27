"""drafts.possible_duplicate flag (V10a duplicate-check post-pass)

Existing rows are simply unflagged — the server default covers them.

Revision ID: 0009_draft_possible_duplicate
Revises: 0008_custom_prompts_list
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from alembic import op

revision = "0009_draft_possible_duplicate"
down_revision = "0008_custom_prompts_list"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("drafts", sa.Column(
        "possible_duplicate", sa.Boolean(), nullable=False, server_default=sa.false(),
    ))


def downgrade() -> None:
    op.drop_column("drafts", "possible_duplicate")
