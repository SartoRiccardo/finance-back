"""Load the V3 fixture spreadsheet (the user's 2026 numbers) into the dev DB
so the Dashboard demo matches the sprint table to the cent.

Usage: uv run python scripts/seed_fixture.py [--year 2026]
Idempotent: re-running replaces the fixture rows it previously inserted.
"""

import argparse
import asyncio
import datetime
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run by path, not -m

from sqlalchemy import delete, select

from app.config import get_settings
from app.db import SessionLocal
from app.models import Category, Label, Transaction, User

# (month, spent, invested, earned) — the sprint's reference table
FIXTURE = [
    (1, "531.58", "647.41", "1094.00"), (2, "479.56", "598.24", "1405.00"),
    (3, "814.06", "643.49", "1396.00"), (4, "365.47", "625.95", "1405.00"),
    (5, "467.09", "662.63", "1456.00"), (6, "330.47", "538.75", "1470.00"),
    (7, "835.55", "257.46", "2565.00"), (8, "129.46", "660.30", "1433.00"),
    (9, "98.38", "569.37", "1445.00"), (10, "1083.52", "403.00", "0.00"),
    (11, "33.00", "390.00", "0.00"), (12, "33.52", "403.00", "0.00"),
]


async def main(year: int) -> None:
    async with SessionLocal() as s:
        user = (
            await s.scalars(select(User).where(User.email == get_settings().master_admin_email))
        ).one_or_none()
        if user is None:
            raise SystemExit(f"No user {get_settings().master_admin_email} — log in once first")
        cats = {c.name: c.id for c in (await s.scalars(select(Category)))}
        salary = (await s.scalars(select(Label).where(Label.name == "salary"))).one().id

        await s.execute(delete(Transaction).where(Transaction.description == "V3 fixture"))
        for m, spent, invested, earned in FIXTURE:
            day = datetime.date(year, m, 15)
            s.add_all([
                Transaction(user_id=user.id, date=day, description="V3 fixture",
                            amount=Decimal(spent), direction="spend", category_id=cats["Misc"]),
                Transaction(user_id=user.id, date=day, description="V3 fixture",
                            amount=Decimal(invested), direction="spend", category_id=cats["Investment"]),
                *([Transaction(user_id=user.id, date=day, description="V3 fixture",
                               amount=Decimal(earned), direction="earn", label_id=salary)]
                  if Decimal(earned) else []),
            ])
        await s.commit()
    print(f"Seeded the 2026 fixture for {year} — GET /api/reports/yearly?year={year}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=datetime.date.today().year)
    asyncio.run(main(p.parse_args().year))
