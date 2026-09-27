"""custom_prompt blob → custom_prompts jsonb list (per-rule editors in the UI)

The old free-text blob was hand-written as "- " bullets, so the backfill splits
on those; multi-line rules survive intact.

Revision ID: 0008_custom_prompts_list
Revises: 0007_api_keys
Create Date: 2026-09-27
"""

import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008_custom_prompts_list"
down_revision = "0007_api_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("app_settings", sa.Column("custom_prompts", JSONB(), nullable=True))
    conn = op.get_bind()
    if row := conn.execute(sa.text("SELECT custom_prompt FROM app_settings WHERE id = 1")).first():
        rules = [p.strip().removeprefix("- ").strip() for p in (row[0] or "").split("\n- ")]
        rules = [r for r in rules if r]
        if rules:
            conn.execute(
                sa.text("UPDATE app_settings SET custom_prompts = CAST(:rules AS jsonb) WHERE id = 1"),
                {"rules": json.dumps(rules)},
            )
    op.drop_column("app_settings", "custom_prompt")


def downgrade() -> None:
    op.add_column("app_settings", sa.Column("custom_prompt", sa.String(2000), nullable=True))
    conn = op.get_bind()
    if row := conn.execute(sa.text("SELECT custom_prompts FROM app_settings WHERE id = 1")).first():
        if rules := row[0]:
            conn.execute(
                sa.text("UPDATE app_settings SET custom_prompt = :text WHERE id = 1"),
                {"text": "\n".join(f"- {r}" for r in rules)},
            )
    op.drop_column("app_settings", "custom_prompts")
