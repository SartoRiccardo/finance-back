"""V3 dashboard reports. The money-math fixture runs on Postgres only — sqlite
float sums can drift a hair, and the spreadsheet match must be exact to the cent.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.models import Category, Label, Transaction, User

YEAR = 2026

# The brief's reference table: (month, spent, invested, earned)
FIXTURE = [
    (1, "531.58", "647.41", "1094.00"), (2, "479.56", "598.24", "1405.00"),
    (3, "814.06", "643.49", "1396.00"), (4, "365.47", "625.95", "1405.00"),
    (5, "467.09", "662.63", "1456.00"), (6, "330.47", "538.75", "1470.00"),
    (7, "835.55", "257.46", "2565.00"), (8, "129.46", "660.30", "1433.00"),
    (9, "98.38", "569.37", "1445.00"), (10, "1083.52", "403.00", "0.00"),
    (11, "33.00", "390.00", "0.00"), (12, "33.52", "403.00", "0.00"),
]

# The brief's 8 columns: (spent, invested, earned, monthly_total, monthly_liquid, cum_total, cum_liquid)
EXPECTED = [
    ("531.58", "647.41", "1094.00", "562.42", "-84.99", "562.42", "-84.99"),
    ("479.56", "598.24", "1405.00", "925.44", "327.20", "1487.86", "242.21"),
    ("814.06", "643.49", "1396.00", "581.94", "-61.55", "2069.80", "180.66"),
    ("365.47", "625.95", "1405.00", "1039.53", "413.58", "3109.33", "594.24"),
    ("467.09", "662.63", "1456.00", "988.91", "326.28", "4098.24", "920.52"),
    ("330.47", "538.75", "1470.00", "1139.53", "600.78", "5237.77", "1521.30"),
    ("835.55", "257.46", "2565.00", "1729.45", "1471.99", "6967.22", "2993.29"),
    ("129.46", "660.30", "1433.00", "1303.54", "643.24", "8270.76", "3636.53"),
    ("98.38", "569.37", "1445.00", "1346.62", "777.25", "9617.38", "4413.78"),
    ("1083.52", "403.00", "0.00", "-1083.52", "-1486.52", "8533.86", "2927.26"),
    ("33.00", "390.00", "0.00", "-33.00", "-423.00", "8500.86", "2504.26"),
    ("33.52", "403.00", "0.00", "-33.52", "-436.52", "8467.34", "2067.74"),
]

KEYS = ("spent", "invested", "earned", "monthly_total", "monthly_liquid", "cum_total", "cum_liquid")


TOTAL_KEYS = ("spent", "invested", "earned", "total_cash", "liquid_cash")


def money(row: dict, keys=KEYS) -> list[Decimal]:
    """JSON numbers/strings back to exact cents (Decimal comparison ignores 0.0 vs 0.00)."""
    return [Decimal(str(row[k])) for k in keys]


async def seed_fixture(db) -> None:
    """One row per bucket per month — row-level split is our choice, sums must match."""
    async with db() as s:
        cats = {c.name: c.id for c in (await s.scalars(select(Category)))}
        salary = (await s.scalars(select(Label).where(Label.name == "salary"))).one().id
        user_id = (await s.scalars(select(User))).one().id
        for m, spent, invested, earned in FIXTURE:
            day = datetime.date(YEAR, m, 15)
            s.add_all([
                Transaction(user_id=user_id, date=day, description="groceries",
                            amount=Decimal(spent), direction="spend", category_id=cats["Misc"]),
                Transaction(user_id=user_id, date=day, description="etf",
                            amount=Decimal(invested), direction="spend", category_id=cats["Investment"]),
                *([Transaction(user_id=user_id, date=day, description="salary", amount=Decimal(earned),
                               direction="earn", label_id=salary)] if Decimal(earned) else []),
            ])
        await s.commit()


@pytest.mark.anyio
async def test_reports_require_auth(client):
    paths = (
        "/api/reports/yearly",
        "/api/reports/summary?from=2026-01-01&to=2026-12-31",
        "/api/reports/by-category?from=2026-01-01&to=2026-12-31",
        "/api/reports/category-series?year=2026",
    )
    for path in paths:
        assert (await client.get(path)).status_code == 401


@pytest.mark.anyio
async def test_yearly_defaults_to_current_year_and_validates(client):
    await client.get("/api/auth/dev-login")
    data = (await client.get("/api/reports/yearly")).json()
    assert data["year"] == datetime.date.today().year
    assert len(data["months"]) == 12  # empty year: 12 zero rows, not fewer
    assert (await client.get("/api/reports/yearly", params={"year": 1800})).status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize("db", ["postgres"], indirect=True)  # exact-cent Numeric math
async def test_yearly_matches_spreadsheet_to_the_cent(client, db):
    await client.get("/api/auth/dev-login")
    await seed_fixture(db)

    data = (await client.get("/api/reports/yearly", params={"year": YEAR})).json()
    assert data["year"] == YEAR and len(data["months"]) == 12
    for i, (row, want) in enumerate(zip(data["months"], EXPECTED)):
        assert row["month"] == i + 1
        assert money(row) == [Decimal(v) for v in want], f"month {i + 1}"

    t = data["totals"]
    assert money(t, TOTAL_KEYS) == [
        sum(Decimal(f[1]) for f in FIXTURE),
        sum(Decimal(f[2]) for f in FIXTURE),
        sum(Decimal(f[3]) for f in FIXTURE),
        Decimal("8467.34"),  # total_cash = Dec cum_total
        Decimal("2067.74"),  # liquid_cash = Dec cum_liquid
    ]

    # a draft spend inside a counted month changes nothing
    async with db() as s:
        misc = (await s.scalars(select(Category).where(Category.name == "Misc"))).one().id
        s.add(Transaction(
            user_id=(await s.scalars(select(User))).one().id, date=datetime.date(YEAR, 6, 2),
            description="photo draft", amount=Decimal("999.99"), direction="spend",
            category_id=misc, is_draft=True, source="photo",
        ))
        await s.commit()
    again = (await client.get("/api/reports/yearly", params={"year": YEAR})).json()
    assert again["months"] == data["months"]

    # another year: all zeros, all 12 months still present
    other = (await client.get("/api/reports/yearly", params={"year": YEAR - 1})).json()
    assert other["year"] == YEAR - 1
    assert [money(row) for row in other["months"]] == [[Decimal(0)] * 7] * 12
    assert money(other["totals"], TOTAL_KEYS) == [Decimal(0)] * 5


@pytest.mark.anyio
@pytest.mark.parametrize("db", ["postgres"], indirect=True)
async def test_summary_over_partial_range(client, db):
    await client.get("/api/auth/dev-login")
    await seed_fixture(db)

    feb_apr = FIXTURE[1:4]  # Feb..Apr inclusive
    spent = sum(Decimal(f[1]) for f in feb_apr)
    invested = sum(Decimal(f[2]) for f in feb_apr)
    earned = sum(Decimal(f[3]) for f in feb_apr)
    s = (
        await client.get("/api/reports/summary", params={"from": f"{YEAR}-02-01", "to": f"{YEAR}-04-30"})
    ).json()
    assert money(s, TOTAL_KEYS) == [spent, invested, earned, earned - spent, earned - spent - invested]

    r = await client.get("/api/reports/summary", params={"from": f"{YEAR}-12-31", "to": f"{YEAR}-01-01"})
    assert r.status_code == 422
    assert (await client.get("/api/reports/summary", params={"to": f"{YEAR}-01-01"})).status_code == 422


# --- V4 insights: by-category + category-series ---

# (date, category, amount, is_draft) — the last row must be invisible to both endpoints
V4_SPENDS = [
    ("2025-12-31", "Misc", "77.77", False),  # year boundary: belongs to 2025 only
    ("2026-01-05", "Misc", "120.50", False),
    ("2026-01-20", "Misc", "30.25", False),
    ("2026-01-20", "Takeout", "55.00", False),
    ("2026-01-20", "Investment", "200.00", False),
    ("2026-02-10", "Misc", "10.00", False),
    ("2026-03-01", "Misc", "999.99", True),  # draft
]


async def seed_v4(db) -> None:
    async with db() as s:
        cats = {c.name: c.id for c in (await s.scalars(select(Category)))}
        user_id = (await s.scalars(select(User))).one().id
        for day, cat, amount, draft in V4_SPENDS:
            s.add(Transaction(
                user_id=user_id, date=datetime.date.fromisoformat(day),
                description=f"{cat} spend", amount=Decimal(amount),
                direction="spend", category_id=cats[cat], is_draft=draft,
            ))
        await s.commit()


def spent(row: dict) -> list[Decimal]:
    return [Decimal(str(v)) for v in row["months"]]


@pytest.mark.anyio
@pytest.mark.parametrize("db", ["postgres"], indirect=True)  # exact-cent Numeric math
async def test_by_category_orders_spends_and_skips_drafts(client, db):
    await client.get("/api/auth/dev-login")
    await seed_v4(db)

    data = (await client.get(
        "/api/reports/by-category", params={"from": "2026-01-01", "to": "2026-02-28"}
    )).json()
    assert [(r["name"], r["is_investment"], Decimal(str(r["total"]))) for r in data] == [
        ("Investment", True, Decimal("200.00")),
        ("Misc", False, Decimal("160.75")),  # two txns summed
        ("Takeout", False, Decimal("55.00")),
    ]
    assert all(isinstance(r["category_id"], int) for r in data)

    # the 2025 row and the draft row are out of range/invisible; earns have no category
    empty = (await client.get(
        "/api/reports/by-category", params={"from": "2026-03-01", "to": "2026-03-31"}
    )).json()
    assert empty == []  # only the draft lives in March

    r = await client.get("/api/reports/by-category", params={"from": "2026-02-28", "to": "2026-01-01"})
    assert r.status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize("db", ["postgres"], indirect=True)  # exact-cent Numeric math
async def test_category_series_zero_fills_twelve_months(client, db):
    await client.get("/api/auth/dev-login")
    await seed_v4(db)

    data = (await client.get("/api/reports/category-series", params={"year": 2026})).json()
    assert {r["name"] for r in data} == {"Misc", "Takeout", "Investment"}
    by_name = {r["name"]: r for r in data}
    assert all(r["is_investment"] == (r["name"] == "Investment") for r in data)
    for row in data:
        assert len(row["months"]) == 12
    assert spent(by_name["Misc"]) == [Decimal("150.75"), Decimal("10.00"), *([Decimal(0)] * 10)]
    assert spent(by_name["Takeout"]) == [Decimal("55.00"), *([Decimal(0)] * 11)]
    assert spent(by_name["Investment"]) == [Decimal("200.00"), *([Decimal(0)] * 11)]  # draft row not added

    # year boundary: the 2025-12-31 spend shows up in 2025's December, not 2026's January
    prev = (await client.get("/api/reports/category-series", params={"year": 2025})).json()
    prev_misc = next(r for r in prev if r["name"] == "Misc")
    assert spent(prev_misc) == [Decimal(0)] * 11 + [Decimal("77.77")]

    # empty year: categories may vanish entirely rather than show all-zero rows
    assert (await client.get("/api/reports/category-series", params={"year": 2024})).json() == []
    assert (await client.get("/api/reports/category-series", params={"year": 1800})).status_code == 422
