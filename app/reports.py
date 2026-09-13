"""Dashboard aggregations.

The functions taking (db, user_id, ...) are pure aggregation — no endpoints, no
HTTP — so later sprints (V4) reuse them directly. The router below is the only
HTTP-aware part.
"""

import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Integer, and_, case, cast, extract, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.db import get_db
from app.models import Category, Transaction, User

ZERO = Decimal("0.00")


async def monthly_sums(
    db: AsyncSession, user_id, date_from: datetime.date, date_to: datetime.date
) -> dict:
    """{(year, month): (spent, invested, earned)} for one user's non-draft txns in range.

    spent excludes investment spends; invested is them; earned is earn rows.
    """
    is_inv = func.coalesce(Category.is_investment, False)
    spend = Transaction.direction == "spend"
    year = cast(extract("year", Transaction.date), Integer).label("y")
    month = cast(extract("month", Transaction.date), Integer).label("m")
    rows = await db.execute(
        select(
            year,
            month,
            func.sum(case((and_(spend, is_inv.is_(False)), Transaction.amount), else_=0)).label("spent"),
            func.sum(case((and_(spend, is_inv.is_(True)), Transaction.amount), else_=0)).label("invested"),
            func.sum(case((~spend, Transaction.amount), else_=0)).label("earned"),
        )
        .join(Category, Transaction.category_id == Category.id, isouter=True)
        .where(
            Transaction.user_id == user_id,
            Transaction.is_draft.is_(False),
            Transaction.date >= date_from,
            Transaction.date <= date_to,
        )
        .group_by("y", "m")
    )
    return {
        (int(r.y), int(r.m)): (Decimal(str(r.spent)), Decimal(str(r.invested)), Decimal(str(r.earned)))
        for r in rows
    }


def totals_of(months: list[dict]) -> dict:
    """{spent, invested, earned, total_cash, liquid_cash} summed over month dicts."""
    t = {k: sum((m[k] for m in months), ZERO) for k in ("spent", "invested", "earned")}
    t["total_cash"] = t["earned"] - t["spent"]
    t["liquid_cash"] = t["total_cash"] - t["invested"]
    return t


async def range_sums(
    db: AsyncSession, user_id, date_from: datetime.date, date_to: datetime.date
) -> dict:
    """Totals over an arbitrary date range (inclusive)."""
    sums = await monthly_sums(db, user_id, date_from, date_to)
    return totals_of([
        {"spent": s, "invested": i, "earned": e} for s, i, e in sums.values()
    ])


async def yearly(db: AsyncSession, user_id, year: int) -> dict:
    """The spreadsheet's yearly roundup: 12 zero-filled months + YTD totals.

    Cumulative sums are within the year only — each January restarts.
    """
    sums = await monthly_sums(db, user_id, datetime.date(year, 1, 1), datetime.date(year, 12, 31))
    months, cum_total, cum_liquid = [], ZERO, ZERO
    for m in range(1, 13):
        spent, invested, earned = sums.get((year, m), (ZERO, ZERO, ZERO))
        monthly_total = earned - spent
        monthly_liquid = monthly_total - invested
        cum_total += monthly_total
        cum_liquid += monthly_liquid
        months.append({
            "month": m,
            "spent": spent,
            "invested": invested,
            "earned": earned,
            "monthly_total": monthly_total,
            "monthly_liquid": monthly_liquid,
            "cum_total": cum_total,
            "cum_liquid": cum_liquid,
        })
    return {"year": year, "months": months, "totals": totals_of(months)}


# --- endpoints ---


router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/yearly")
async def yearly_report(
    year: int | None = Query(None, ge=1900, le=2100),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await yearly(db, user.id, year or datetime.date.today().year)


@router.get("/summary")
async def summary_report(
    date_from: datetime.date = Query(alias="from"),
    date_to: datetime.date = Query(alias="to"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if date_from > date_to:
        raise HTTPException(status_code=422, detail="from must be <= to")
    return await range_sums(db, user.id, date_from, date_to)
