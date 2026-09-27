from decimal import Decimal
import datetime

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from app.models import Transaction, User

CATEGORY_NAMES = [
    "Education", "Sewing", "Chinese Plushies", "Investment", "Girlfriend",
    "Social life", "Ingredients", "Takeout", "Self Care", "Transport", "Misc",
]
LABEL_NAMES = ["fumofumo", "payback", "salary", "tip"]  # alphabetical


async def cid(client, name: str) -> int:
    cats = (await client.get("/api/categories")).json()
    return next(c["id"] for c in cats if c["name"] == name)


async def lid(client, name: str) -> int:
    labels = (await client.get("/api/labels")).json()
    return next(l["id"] for l in labels if l["name"] == name)


async def add_txn(client, **over) -> dict:
    body = {
        "date": "2026-09-10",
        "description": "Coffee",
        "amount": "3.50",
        "direction": "earn",
        "label_id": await lid(client, "tip"),
    } | over
    r = await client.post("/api/transactions", json=body)
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.anyio
async def test_seeds_present(client):
    await client.get("/api/auth/dev-login")
    cats = (await client.get("/api/categories")).json()
    assert [c["name"] for c in cats] == CATEGORY_NAMES
    assert [c["sort_order"] for c in cats] == list(range(11))
    plushies = next(c for c in cats if c["name"] == "Chinese Plushies")
    assert plushies["is_investment"] is True
    assert plushies["description"] == "smallplushies.com stock"
    assert next(c for c in cats if c["name"] == "Education")["is_investment"] is False

    labels = (await client.get("/api/labels")).json()
    assert [l["name"] for l in labels] == LABEL_NAMES
    assert all(l["is_spending"] is False for l in labels)


@pytest.mark.anyio
async def test_requires_auth(client):
    for method, path in [
        ("get", "/api/categories"), ("post", "/api/categories"),
        ("get", "/api/labels"), ("post", "/api/labels"),
        ("patch", "/api/labels/1"), ("delete", "/api/labels/1"),
        ("get", "/api/transactions"), ("post", "/api/transactions"),
    ]:
        r = await client.request(method.upper(), path, json={"date": "2026-09-01"})
        assert r.status_code == 401, f"{method} {path}: {r.status_code}"


@pytest.mark.anyio
async def test_category_crud_and_409_guard(client):
    await client.get("/api/auth/dev-login")
    transport = await cid(client, "Transport")

    r = await client.post("/api/categories", json={
        "name": "Vinyl", "description": "Records", "is_investment": True, "sort_order": 99,
    })
    assert r.status_code == 201, r.text
    vinyl = r.json()
    assert vinyl["is_investment"] is True and vinyl["sort_order"] == 99

    assert (await client.post("/api/categories", json={"name": "Vinyl"})).status_code == 409
    # different case is a different name
    assert (await client.post("/api/categories", json={"name": "vinyl"})).status_code == 201

    r = await client.patch(f"/api/categories/{vinyl['id']}", json={"sort_order": 5, "name": "Vinyls"})
    assert r.status_code == 200
    assert r.json()["name"] == "Vinyls" and r.json()["sort_order"] == 5
    assert (
        await client.patch(f"/api/categories/{vinyl['id']}", json={"name": "Transport"})
    ).status_code == 409
    assert (await client.patch("/api/categories/999999", json={"name": "X"})).status_code == 404

    # unused -> deletable
    assert (await client.delete(f"/api/categories/{vinyl['id']}")).status_code == 204

    # referenced -> 409, until nothing references it anymore
    txn = await add_txn(client, direction="spend", category_id=transport, label_id=None)
    assert (await client.delete(f"/api/categories/{transport}")).status_code == 409
    assert (await client.delete(f"/api/transactions/{txn['id']}")).status_code == 204
    assert (await client.delete(f"/api/categories/{transport}")).status_code == 204
    assert (await client.delete("/api/categories/999999")).status_code == 404


@pytest.mark.anyio
async def test_label_crud_and_409_guard(client):
    await client.get("/api/auth/dev-login")

    r = await client.post("/api/labels", json={"name": "cashback", "color": "#22c55e"})
    assert r.status_code == 201, r.text
    cashback = r.json()
    assert cashback["is_spending"] is False and cashback["color"] == "#22c55e"
    assert cashback["name"] in [l["name"] for l in (await client.get("/api/labels")).json()]

    r = await client.post("/api/labels", json={"name": "cashback"})
    assert r.status_code == 409
    assert r.json()["detail"] == "Label 'cashback' already exists"

    r = await client.patch(f"/api/labels/{cashback['id']}", json={"name": "cash"})
    assert r.status_code == 200 and r.json()["name"] == "cash"
    assert (
        await client.patch(f"/api/labels/{cashback['id']}", json={"name": "tip"})
    ).status_code == 409
    assert (await client.patch("/api/labels/999999", json={"name": "X"})).status_code == 404

    r = await client.patch(f"/api/labels/{cashback['id']}", json={"is_spending": True})
    assert r.status_code == 200 and r.json()["is_spending"] is True

    # referenced -> 409 with the count, until nothing references it anymore
    tip = await lid(client, "tip")
    txn = await add_txn(client, label_id=tip)
    r = await client.delete(f"/api/labels/{tip}")
    assert r.status_code == 409
    assert r.json()["detail"] == "Label in use by 1 transactions"
    assert (await client.delete(f"/api/transactions/{txn['id']}")).status_code == 204
    assert (await client.delete(f"/api/labels/{tip}")).status_code == 204
    assert (await client.delete("/api/labels/999999")).status_code == 404


@pytest.mark.anyio
@pytest.mark.parametrize("over", [
    {"direction": "spend", "category_id": None, "label_id": None},  # spend without category
    {"direction": "spend", "label_id": 1, "category_id": 1},        # spend with a label
    {"direction": "earn", "label_id": None, "category_id": None},   # earn without label
    {"direction": "earn", "label_id": 1, "category_id": 1},         # earn with a category
    {"direction": "spend", "category_id": 999999},                  # unknown category
    {"direction": "earn", "label_id": 999999},                      # unknown label
    {"amount": "0"},                                                # not positive
    {"amount": "-1"},
    {"amount": "1.234"},                                            # more than 2 dp
    {"description": ""},                                            # empty description
])
async def test_invalid_transactions_rejected_422(client, over, db):
    await client.get("/api/auth/dev-login")
    body = {
        "date": "2026-09-10", "description": "Coffee", "amount": "3.50",
        "direction": "earn", "label_id": await lid(client, "tip"),
    } | over
    r = await client.post("/api/transactions", json=body)
    assert r.status_code == 422, r.text
    async with db() as s:
        assert (await s.scalar(select(func.count()).select_from(Transaction))) == 0


@pytest.mark.anyio
async def test_transaction_crud_filters_pagination(client):
    await client.get("/api/auth/dev-login")
    earn = await add_txn(client, date="2026-09-01", description="Coffee tip")
    spend = await add_txn(
        client, direction="spend", category_id=await cid(client, "Transport"),
        label_id=None, date="2026-09-10", description="Train ticket", amount="12.34",
    )

    # order: date desc, id desc; nested category/label
    page = (await client.get("/api/transactions")).json()
    assert page["total"] == 2
    assert [t["id"] for t in page["items"]] == [spend["id"], earn["id"]]
    row = page["items"][0]
    assert Decimal(str(row["amount"])) == Decimal("12.34")
    assert row["category"]["name"] == "Transport"
    assert row["category"]["is_investment"] is False
    assert row["label"] is None
    assert page["items"][1]["label"]["name"] == "tip"
    assert page["items"][1]["category"] is None

    # patch money + category (stays a spend)
    plushies = await cid(client, "Chinese Plushies")
    r = await client.patch(f"/api/transactions/{spend['id']}", json={
        "description": "Plushie restock", "amount": "99.99",
        "category_id": plushies, "date": "2026-09-12",
    })
    assert r.status_code == 200, r.text
    assert Decimal(str(r.json()["amount"])) == Decimal("99.99")
    assert r.json()["category"]["name"] == "Chinese Plushies"
    assert (await client.get("/api/transactions")).json()["items"][0]["description"] == "Plushie restock"

    # filters
    assert (await client.get("/api/transactions", params={"from": "2026-09-02"})).json()["total"] == 1
    assert (await client.get("/api/transactions", params={"to": "2026-09-01"})).json()["total"] == 1
    assert (await client.get("/api/transactions", params={"from": "2026-09-01"})).json()["total"] == 2
    assert (await client.get("/api/transactions", params={"direction": "spend"})).json()["total"] == 1
    assert (await client.get("/api/transactions", params={"direction": "earn"})).json()["total"] == 1
    assert (await client.get("/api/transactions", params={"category_id": plushies})).json()["total"] == 1
    assert (await client.get("/api/transactions", params={"q": "PLUSH"})).json()["total"] == 1
    assert (await client.get("/api/transactions", params={"q": "zzz"})).json()["total"] == 0

    # pagination: 3 more earns -> total 5
    for _ in range(3):
        await add_txn(client)
    page = (await client.get("/api/transactions", params={"limit": 2, "offset": 1})).json()
    assert page["total"] == 5 and len(page["items"]) == 2
    assert (await client.get("/api/transactions", params={"limit": 0})).status_code == 422
    assert (await client.get("/api/transactions", params={"offset": -1})).status_code == 422

    # direction switch validated against the merged row; the dropped ref must be
    # cleared explicitly — the API applies exactly what you send, like the DB CHECK
    r = await client.patch(f"/api/transactions/{spend['id']}", json={
        "direction": "earn", "label_id": await lid(client, "salary"), "category_id": None,
    })
    assert r.status_code == 200, r.text
    assert r.json()["direction"] == "earn" and r.json()["category"] is None
    assert (await client.get("/api/transactions", params={"category_id": plushies})).json()["total"] == 0
    r = await client.patch(f"/api/transactions/{spend['id']}", json={"direction": "spend"})
    assert r.status_code == 422  # spend without a category

    # delete
    assert (await client.delete(f"/api/transactions/{earn['id']}")).status_code == 204
    assert (await client.get("/api/transactions")).json()["total"] == 4
    assert (await client.delete(f"/api/transactions/{earn['id']}")).status_code == 404


@pytest.mark.anyio
async def test_draft_rows_never_listed(client, db):
    await client.get("/api/auth/dev-login")
    await add_txn(client)
    misc = await cid(client, "Misc")
    async with db() as s:
        user = (await s.execute(select(User))).scalar_one()
        s.add(Transaction(
            user_id=user.id, date=datetime.date(2026, 9, 9), description="photo draft",
            amount=Decimal("5"), direction="spend", category_id=misc,
            is_draft=True, source="photo",
        ))
        await s.commit()

    page = (await client.get("/api/transactions")).json()
    assert page["total"] == 1
    assert all("photo draft" not in t["description"] for t in page["items"])


@pytest.mark.anyio
async def test_schema_enforces_direction_and_money_at_db_level(client, db):
    """The DB CHECK constraints hold even when the API layer is bypassed."""
    await client.get("/api/auth/dev-login")
    misc = await cid(client, "Misc")
    async with db() as s:
        user_id = (await s.execute(select(User))).scalar_one().id

        def sql(direction="spend", amount="5.00", category_id=None, label_id=None):
            return text(
                "INSERT INTO transactions (user_id, date, description, amount, direction,"
                " category_id, label_id) VALUES (:u, '2026-09-10', 'x', :a, :d, :c, :l)"
            ).bindparams(u=user_id, a=Decimal(amount), d=direction, c=category_id, l=label_id)

        bad = [
            dict(direction="spend", label_id=1),      # spend with a label, no category
            dict(direction="earn", category_id=misc),  # earn with a category, no label
            dict(amount="-1.00", category_id=misc),    # non-positive amount
        ]
        for kwargs in bad:
            with pytest.raises(IntegrityError):
                await s.execute(sql(**kwargs))
            await s.rollback()  # PG aborts the transaction on a failed statement

        # exact cents round-trip — the invariant V3 money math relies on
        await s.execute(sql(amount="999999999.99", category_id=misc))
        row = (await s.execute(select(Transaction))).scalar_one()
        assert row.amount == Decimal("999999999.99")


@pytest.mark.anyio
async def test_taxonomy_colors(client):
    await client.get("/api/auth/dev-login")

    cats = (await client.get("/api/categories")).json()
    edu = next(c for c in cats if c["name"] == "Education")
    assert edu["color"] == "#3b82f6"
    plushies = next(c for c in cats if c["name"] == "Chinese Plushies")
    assert plushies["color"] == "#ec4899"

    labels = (await client.get("/api/labels")).json()
    tip = next(l for l in labels if l["name"] == "tip")
    assert tip["color"] == "#eab308"

    r = await client.patch(f"/api/categories/{edu['id']}", json={"color": "#ef4444"})
    assert r.status_code == 200 and r.json()["color"] == "#ef4444"

    r = await client.patch(f"/api/categories/{edu['id']}", json={"color": "red"})
    assert r.status_code == 422

    r = await client.patch(f"/api/labels/{tip['id']}", json={"color": "#84cc16"})
    assert r.status_code == 200 and r.json()["color"] == "#84cc16"

    r = await client.patch("/api/labels/99999", json={"color": "#84cc16"})
    assert r.status_code == 404
