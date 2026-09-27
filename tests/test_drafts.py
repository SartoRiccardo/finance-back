"""V5 photo drafts. The LLM dependency is always faked — no model calls in CI."""

import asyncio
import base64
import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.llm import LLMError, Usage, get_llm
from app.models import Draft, Transaction, Upload

# 1x1 transparent png
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

GOOD_ROWS = {
    "rows": [
        {"date": "2026-09-10", "description": "Coop run", "amount": 23.45, "category": "ingredients"},
        {"date": "2026-09-11", "description": "Moon rocks", "amount": 7, "category": "Moon rocks ltd"},
    ]
}


class FakeLLM:
    """Stands in for both real clients."""

    def __init__(self, payload=None, error=None, delay=0, usage=None):
        self.provider, self.model = "google", "fake-model"
        self.payload, self.error, self.delay = payload, error, delay
        self.usage = usage or Usage()
        self.calls = []

    async def complete_structured(self, schema, contents):
        self.calls.append((schema, contents))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise LLMError(self.error)
        return self.payload, self.usage


def use_llm(client, fake):
    """Swap the LLM dependency on this client's app (conftest keeps the app private)."""
    client._transport.app.dependency_overrides[get_llm] = lambda: fake
    return client


@pytest.fixture
def upload_dir(tmp_path):
    return tmp_path / "uploads"  # not tmp_path itself: the test DB file lives there now


@pytest.fixture
def dclient(make_client, upload_dir):
    return make_client(upload_dir=str(upload_dir))


async def upload_receipt(client, **over):
    r = await client.post(
        "/api/uploads", files={"file": over.pop("file", ("receipt.png", PNG, "image/png"))}
    )
    assert r.status_code == 201, r.text
    return r.json()["upload_id"]


async def wait_draft(client, draft_id: int) -> dict:
    """Extraction is detached; poll until it leaves processing (1s cap)."""
    for _ in range(200):
        body = (await client.get(f"/api/drafts/{draft_id}")).json()
        if body["status"] != "processing":
            return body
        await asyncio.sleep(0.005)
    raise AssertionError("draft never left processing")


async def make_draft(client, payload=GOOD_ROWS) -> dict:
    upload_id = await upload_receipt(client)
    use_llm(client, FakeLLM(payload))
    r = await client.post("/api/drafts/from-upload", json={"upload_id": upload_id})
    assert r.status_code == 201, r.text
    return await wait_draft(client, r.json()["id"])


async def cid(client, name: str) -> int:
    cats = (await client.get("/api/categories")).json()
    return next(c["id"] for c in cats if c["name"] == name)


@pytest.mark.anyio
async def test_custom_prompts_are_appended_to_extraction_prompt(dclient):
    await dclient.get("/api/auth/dev-login")
    r = await dclient.put("/api/settings", json={
        "llm_provider": "google", "llm_model": "gemini-2.5-flash",
        "custom_prompts": [
            "apple vinegar goes in Self Care",
            "a discount applies to the item above:\nITEM A -> 3,00\nDISCOUNT -> -1,00",
        ],
    })
    assert r.status_code == 200, r.text

    fake = FakeLLM(GOOD_ROWS)
    upload_id = await upload_receipt(dclient)
    use_llm(dclient, fake)
    r = await dclient.post("/api/drafts/from-upload", json={"upload_id": upload_id})
    assert (await wait_draft(dclient, r.json()["id"]))["status"] == "open"

    prompt = fake.calls[0][1][0]  # (schema, contents) → the text part
    # each rule concatenated as a bullet, multi-line rules intact
    assert "- apple vinegar goes in Self Care" in prompt
    assert "- a discount applies to the item above:\nITEM A -> 3,00\nDISCOUNT -> -1,00" in prompt
    # appended after the JSON-shape instructions, not replacing them
    assert prompt.index("Respond with ONLY a JSON object") < prompt.index("apple vinegar")


@pytest.mark.anyio
async def test_draft_endpoints_require_auth(client):
    for method, path in [
        ("post", "/api/uploads"), ("get", "/api/uploads/00000000-0000-0000-0000-000000000000"),
        ("post", "/api/drafts/from-upload"), ("get", "/api/drafts"), ("get", "/api/drafts/1"),
        ("post", "/api/drafts/1/rows"), ("patch", "/api/drafts/1/rows/1"),
        ("delete", "/api/drafts/1/rows/1"), ("post", "/api/drafts/1/approve"),
        ("delete", "/api/drafts/1"),
    ]:
        kwargs = {"files": {"file": ("r.png", PNG, "image/png")}} if path == "/api/uploads" else {}
        r = await client.request(method.upper(), path, json={"upload_id": str(uuid.uuid4())}, **kwargs)
        assert r.status_code == 401, f"{method} {path}: {r.status_code}"


@pytest.mark.anyio
async def test_upload_validates_mime_and_size_and_stores(dclient, db, upload_dir):
    await dclient.get("/api/auth/dev-login")
    upload_id = await upload_receipt(dclient)

    # stored as <uuid>.png under UPLOAD_DIR, row records origin + size
    files = list(upload_dir.iterdir())
    assert [f.name for f in files] == [f"{upload_id}.png"] and files[0].read_bytes() == PNG
    async with db() as s:
        up = (await s.scalars(select(Upload))).one()
        assert (str(up.id), up.mime_type, up.size_bytes, up.original_name) == (
            upload_id, "image/png", len(PNG), "receipt.png",
        )
        # the image itself is servable (draft thumbnails)
        r = await dclient.get(f"/api/uploads/{upload_id}")
        assert r.status_code == 200 and r.content == PNG

    r = await dclient.post("/api/uploads", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 422, r.text
    r = await dclient.post("/api/uploads", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 422
    r = await dclient.post(
        "/api/uploads", files={"file": ("big.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")}
    )
    assert r.status_code == 413
    r = await dclient.post("/api/drafts/from-upload", json={"upload_id": str(uuid.uuid4())})
    assert r.status_code == 404


@pytest.mark.anyio
async def test_from_upload_extracts_maps_categories_hides_drafts(dclient, db):
    await dclient.get("/api/auth/dev-login")
    draft = await make_draft(dclient)

    assert draft["source"] == "photo" and draft["status"] == "open"
    rows = sorted(draft["rows"], key=lambda r: r["id"])
    assert [r["description"] for r in rows] == ["Coop run", "Moon rocks"]
    assert [Decimal(str(r["amount"])) for r in rows] == [Decimal("23.45"), Decimal("7.00")]
    # case-insensitive contains → Ingredients; unknown name → Misc
    assert [r["category"]["name"] for r in rows] == ["Ingredients", "Misc"]
    assert all(r["direction"] == "spend" for r in rows)

    # real transactions rows with is_draft + draft_id set, invisible to the ledger
    page = (await dclient.get("/api/transactions")).json()
    assert page["total"] == 0
    async with db() as s:
        txns = (await s.scalars(select(Transaction))).all()
        assert len(txns) == 2
        assert all(
            t.is_draft and t.source == "photo" and t.draft_id == draft["id"] for t in txns
        )

    listing = (await dclient.get("/api/drafts")).json()
    assert [d["id"] for d in listing] == [draft["id"]]
    detail = (await dclient.get(f"/api/drafts/{draft['id']}")).json()
    assert len(detail["rows"]) == 2
    assert (await dclient.get("/api/drafts/999999")).status_code == 404


@pytest.mark.anyio
async def test_open_draft_invisible_to_transactions_and_reports(dclient):
    await dclient.get("/api/auth/dev-login")
    before = (await dclient.get("/api/reports/yearly", params={"year": 2026})).json()
    await make_draft(dclient)

    assert (await dclient.get("/api/transactions")).json()["total"] == 0
    params = {"from": "2026-09-01", "to": "2026-09-30"}
    summary = (await dclient.get("/api/reports/summary", params=params)).json()
    assert Decimal(str(summary["spent"])) == 0
    bycat = (await dclient.get("/api/reports/by-category", params=params)).json()
    assert bycat == []
    after = (await dclient.get("/api/reports/yearly", params={"year": 2026})).json()
    assert after["months"] == before["months"]


@pytest.mark.anyio
@pytest.mark.parametrize("payload", [
    {"rows": []},
    {"rows": "nope"},
    {"rows": [{"description": "x", "amount": 1, "category": "Misc"}]},  # no date
    {"rows": [{"date": "someday", "description": "x", "amount": 1, "category": "Misc"}]},
    {"rows": [{"date": "2026-09-10", "description": "x", "amount": -3, "category": "Misc"}]},
])
async def test_malformed_llm_output_leaves_an_errored_empty_draft(dclient, db, upload_dir, payload):
    await dclient.get("/api/auth/dev-login")
    upload_id = await upload_receipt(dclient)
    use_llm(dclient, FakeLLM(payload))
    r = await dclient.post("/api/drafts/from-upload", json={"upload_id": upload_id})
    assert r.status_code == 201, r.text  # processing draft returned immediately

    draft = await wait_draft(dclient, r.json()["id"])
    assert draft["status"] == "error" and draft["rows"] == []

    async with db() as s:
        assert (await s.scalars(select(Transaction))).all() == []
        stored = (await s.scalars(select(Draft))).one()
        assert stored.status == "error" and stored.error
    # the image survived too — a retry doesn't need a re-upload
    assert [f.name for f in upload_dir.iterdir()] == [f"{upload_id}.png"]


@pytest.mark.anyio
async def test_llm_provider_error_lands_in_the_draft(dclient, db):
    await dclient.get("/api/auth/dev-login")
    upload_id = await upload_receipt(dclient)
    use_llm(dclient, FakeLLM(error="Gemini call failed: 429"))
    r = await dclient.post("/api/drafts/from-upload", json={"upload_id": upload_id})
    assert r.status_code == 201, r.text
    draft = await wait_draft(dclient, r.json()["id"])
    assert draft["status"] == "error" and "429" in draft["error"]
    # still listed — a failed read is visible, not lost
    assert [d["id"] for d in (await dclient.get("/api/drafts")).json()] == [draft["id"]]
    # and it can be discarded to clear the list (approve stays blocked)
    assert (await dclient.post(f"/api/drafts/{draft['id']}/approve")).status_code == 409
    assert (await dclient.delete(f"/api/drafts/{draft['id']}")).status_code == 204
    assert (await dclient.get("/api/drafts")).json() == []


@pytest.mark.anyio
async def test_empty_api_key_fails_at_extraction_not_boot(make_client, upload_dir):
    """Real client path with empty keys: the app serves, only the extraction complains."""
    client = make_client(
        upload_dir=str(upload_dir), google_api_key="", openrouter_api_key="", llm_provider="google"
    )
    await client.get("/api/auth/dev-login")
    assert (await client.get("/api/health")).json() == {"status": "ok"}
    upload_id = await upload_receipt(client)
    r = await client.post("/api/drafts/from-upload", json={"upload_id": upload_id})
    assert r.status_code == 201, r.text
    draft = await wait_draft(client, r.json()["id"])
    assert draft["status"] == "error" and "GOOGLE_API_KEY" in draft["error"]


@pytest.mark.anyio
async def test_processing_draft_is_listed_and_not_approvable(dclient):
    await dclient.get("/api/auth/dev-login")
    upload_id = await upload_receipt(dclient)
    use_llm(dclient, FakeLLM(GOOD_ROWS, delay=0.3))  # long enough to catch processing
    r = await dclient.post("/api/drafts/from-upload", json={"upload_id": upload_id})
    assert r.status_code == 201 and r.json()["status"] == "processing" and r.json()["rows"] == []
    draft_id = r.json()["id"]

    listing = (await dclient.get("/api/drafts")).json()
    assert [d["id"] for d in listing] == [draft_id]
    assert listing[0]["status"] == "processing"

    assert (await dclient.post(f"/api/drafts/{draft_id}/approve")).status_code == 409

    body = await wait_draft(dclient, draft_id)
    assert body["status"] == "open" and len(body["rows"]) == 2


@pytest.mark.anyio
async def test_usage_recorded_with_cost(dclient, monkeypatch):
    """The extraction logs an llm_usage row; the picker's prices price it."""
    await dclient.get("/api/auth/dev-login")

    async def fake_prices():
        return {"fake-model": (0.3, 2.5)}

    monkeypatch.setattr("app.llm._openrouter_price_map", fake_prices)
    upload_id = await upload_receipt(dclient)
    use_llm(dclient, FakeLLM(GOOD_ROWS, usage=Usage(input_tokens=100, output_tokens=50)))
    r = await dclient.post("/api/drafts/from-upload", json={"upload_id": upload_id})
    draft = await wait_draft(dclient, r.json()["id"])

    page = (await dclient.get("/api/llm/usage")).json()
    assert page["totals"]["calls"] == 1
    assert page["totals"]["input_tokens"] == 100 and page["totals"]["output_tokens"] == 50
    # (100 in × $0.3 + 50 out × $2.5) / 1M
    assert float(page["totals"]["cost_usd"]) == pytest.approx(0.000155, abs=1e-9)
    row = page["recent"][0]
    assert (row["provider"], row["model"], row["draft_id"]) == ("google", "fake-model", draft["id"])


@pytest.mark.anyio
async def test_approve_publishes_rows_deletes_file_moves_reports(dclient, db, upload_dir):
    await dclient.get("/api/auth/dev-login")
    draft = await make_draft(dclient)
    params = {"from": "2026-09-01", "to": "2026-09-30"}
    assert Decimal(str((await dclient.get("/api/reports/summary", params=params)).json()["spent"])) == 0

    r = await dclient.post(f"/api/drafts/{draft['id']}/approve")
    assert r.status_code == 200 and r.json()["status"] == "approved"

    # rows are real transactions now, newest date first
    page = (await dclient.get("/api/transactions")).json()
    assert page["total"] == 2
    assert [Decimal(str(t["amount"])) for t in page["items"]] == [Decimal("7.00"), Decimal("23.45")]
    summary = Decimal(str((await dclient.get("/api/reports/summary", params=params)).json()["spent"]))
    assert summary == Decimal("30.45")  # the dashboard math changed

    # is_draft flipped, draft_id + source kept for provenance
    async with db() as s:
        txns = (await s.scalars(select(Transaction))).all()
        assert all(not t.is_draft and t.draft_id == draft["id"] and t.source == "photo" for t in txns)

    # the file is gone from disk; the upload + draft rows stay
    assert list(upload_dir.iterdir()) == []
    async with db() as s:
        assert len((await s.scalars(select(Upload))).all()) == 1
        assert (await s.get(Draft, draft["id"])).status == "approved"
    assert (await dclient.get("/api/drafts")).json() == []
    assert (await dclient.get(f"/api/uploads/{draft['upload_id']}")).status_code == 410

    # a closed draft can't be re-approved or edited
    assert (await dclient.post(f"/api/drafts/{draft['id']}/approve")).status_code == 409
    assert (await dclient.delete(f"/api/drafts/{draft['id']}")).status_code == 409
    r = await dclient.post(
        f"/api/drafts/{draft['id']}/rows",
        json={"date": "2026-09-12", "description": "x", "amount": "1",
              "direction": "spend", "category_id": await cid(dclient, "Misc")},
    )
    assert r.status_code == 409


@pytest.mark.anyio
async def test_discard_deletes_rows_and_file(dclient, db, upload_dir):
    await dclient.get("/api/auth/dev-login")
    draft = await make_draft(dclient)

    r = await dclient.delete(f"/api/drafts/{draft['id']}")
    assert r.status_code == 204
    assert list(upload_dir.iterdir()) == []
    async with db() as s:
        assert (await s.scalars(select(Transaction))).all() == []  # rows deleted
        assert len((await s.scalars(select(Upload))).all()) == 1
        assert (await s.get(Draft, draft["id"])).status == "rejected"  # kept for provenance
    assert (await dclient.get("/api/drafts")).json() == []
    assert (await dclient.get(f"/api/transactions")).json()["total"] == 0


@pytest.mark.anyio
async def test_draft_row_crud_validates_like_transactions(dclient):
    await dclient.get("/api/auth/dev-login")
    draft = await make_draft(dclient)
    did, ingredients = draft["id"], await cid(dclient, "Ingredients")
    labels = (await dclient.get("/api/labels")).json()
    tip = next(l["id"] for l in labels if l["name"] == "tip")

    r = await dclient.post(f"/api/drafts/{did}/rows", json={
        "date": "2026-09-12", "description": "Tip jar", "amount": "1.00",
        "direction": "spend", "category_id": ingredients, "label_id": None,
    })
    assert r.status_code == 201, r.text
    row = r.json()
    assert len((await dclient.get(f"/api/drafts/{did}")).json()["rows"]) == 3

    r = await dclient.patch(
        f"/api/drafts/{did}/rows/{row['id']}", json={"description": "Tip jar extra", "amount": "2.50"}
    )
    assert r.status_code == 200 and Decimal(str(r.json()["amount"])) == Decimal("2.50")

    # V2 rules: positive amount, real category, spend needs a category
    for over in [{"amount": "0"}, {"amount": "1.234"}, {"category_id": 999999}, {"category_id": None}]:
        assert (
            await dclient.patch(f"/api/drafts/{did}/rows/{row['id']}", json=over)
        ).status_code == 422, over

    # earn allowed with a label, no category
    r = await dclient.patch(
        f"/api/drafts/{did}/rows/{row['id']}",
        json={"direction": "earn", "label_id": tip, "category_id": None},
    )
    assert r.status_code == 200 and r.json()["direction"] == "earn"

    # unknown row / another draft's row → 404
    assert (await dclient.patch("/api/drafts/999999/rows/1", json={"amount": "1"})).status_code == 404

    r = await dclient.delete(f"/api/drafts/{did}/rows/{row['id']}")
    assert r.status_code == 204
    assert len((await dclient.get(f"/api/drafts/{did}")).json()["rows"]) == 2
    assert (await dclient.delete(f"/api/drafts/{did}/rows/{row['id']}")).status_code == 404
