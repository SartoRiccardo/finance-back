"""taxonomy colors

Revision ID: 0003_taxonomy_colors
Revises: 0002_transactions
Create Date: 2026-09-13
"""

import sqlalchemy as sa
from alembic import op

from app.models import CATEGORY_COLORS, LABEL_COLORS

revision = "0003_taxonomy_colors"
down_revision = "0002_transactions"


def upgrade() -> None:
    op.add_column("categories", sa.Column("color", sa.String(7), nullable=True))
    op.add_column("labels", sa.Column("color", sa.String(7), nullable=True))
    for name, color in CATEGORY_COLORS.items():
        op.execute(
            sa.text("UPDATE categories SET color = :c WHERE name = :n").bindparams(c=color, n=name)
        )
    for name, color in LABEL_COLORS.items():
        op.execute(
            sa.text("UPDATE labels SET color = :c WHERE name = :n").bindparams(c=color, n=name)
        )


def downgrade() -> None:
    op.drop_column("labels", "color")
    op.drop_column("categories", "color")
