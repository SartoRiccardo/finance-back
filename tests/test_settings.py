"""V5.1 model picker: settings round-trip + live provider catalog.

Both upstreams are faked — google-genai via a fake module in sys.modules,
OpenRouter via respx. No real API calls in CI.
"""

import base64
import sys
import types

import pytest
import respx
from httpx import Response
from sqlalchemy import select

from app.llm import _catalog_cache
from app.models import AppSetting

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
ROWS = {"rows": [
    {"date": "2026-09-10", "description": "Coop run", "amount": 23.45, "category": "Ingredients"},
]}


@pytest.fixture(autouse=True)
def _fresh_catalog():
    _catalog_cache.clear()
    yield
    _catalog_cache.clear()


@pytest.fixture
def dclient(make_client, tmp_path):
    return make_client(upload_dir=str(tmp_path))


def fake_genai_module(monkeypatch, models=None, error=None):
    """Swap google.genai for a fake whose Client.aio.models.list() yields `models`."""
    calls = []

    class FakePager:
        def __init__(self):
            self.i = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if error:
                raise error
            if self.i >= len(models):
                raise StopAsyncIteration
            m = models[self.i]
            self.i += 1
            return m

    class FakeAioModels:
        def list(self):
            calls.append("list")
            return FakePager()

    genai = types.SimpleNamespace(Client=lambda api_key: types.SimpleNamespace(
        aio=types.SimpleNamespace(models=FakeAioModels())
    ))
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=genai))
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    return calls


def model(name, display_name=None):
    return types.SimpleNamespace(name=name, display_name=display_name)


@pytest.mark.anyio
async def test_settings_endpoints_require_auth(client):
    assert (await client.get("/api/settings")).status_code == 401
    assert (await client.put("/api/settings", json={
        "llm_provider": "google", "llm_model": "gemini-2.5-flash"
    })).status_code == 401
    assert (await client.get("/api/llm/models")).status_code == 401


@pytest.mark.anyio
async def test_settings_round_trip_persists_row(dclient, db):
    await dclient.get("/api/auth/dev-login")

    # env defaults before any PUT (startup never ran in tests, so no row yet)
    r = await dclient.get("/api/settings")
    assert r.status_code == 200
    assert r.json() == {"llm_provider": "google", "llm_model": "gemini-2.5-flash"}

    body = {"llm_provider": "openrouter", "llm_model": "qwen/qwen3-vl-235b"}
    r = await dclient.put("/api/settings", json=body)
    assert r.status_code == 200 and r.json() == body
    assert (await dclient.get("/api/settings")).json() == body

    async with db() as s:
        row = await s.get(AppSetting, 1)
        assert (row.llm_provider, row.llm_model) == (body["llm_provider"], body["llm_model"])


@pytest.mark.anyio
@pytest.mark.parametrize("body", [
    {"llm_provider": "anthropic", "llm_model": "claude"},  # unknown provider
    {"llm_provider": "google", "llm_model": ""},           # empty model
    {"llm_provider": "google"},                            # missing model
    {"llm_model": "gemini-2.5-flash"},                     # missing provider
])
async def test_settings_validation_rejects_and_persists_nothing(dclient, db, body):
    await dclient.get("/api/auth/dev-login")
    r = await dclient.put("/api/settings", json=body)
    assert r.status_code == 422, r.text
    async with db() as s:
        assert (await s.scalars(select(AppSetting))).all() == []


@pytest.mark.anyio
async def test_google_models_q_filter_normalize_cap(dclient, monkeypatch):
    await dclient.get("/api/auth/dev-login")
    models = (
        [model(f"models/gemini-flash-{i:02d}", f"Gemini Flash {i:02d}") for i in range(25)]
        + [model("models/gemini-2.5-pro", "Gemini 2.5 Pro")]
        + [model("models/other-1")]  # no display_name → name falls back to id
    )
    calls = fake_genai_module(monkeypatch, models)

    r = await dclient.get("/api/llm/models", params={"q": "FLASH"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 20  # 25 matches, capped
    assert body[0] == {"id": "gemini-flash-00", "name": "Gemini Flash 00"}
    assert all("flash" in m["id"] for m in body)

    # matches on name too; no display_name falls back to the raw id
    r = await dclient.get("/api/llm/models", params={"q": "2.5 pro"})
    assert r.json() == [{"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"}]
    r = await dclient.get("/api/llm/models", params={"q": "other"})
    assert r.json() == [{"id": "other-1", "name": "models/other-1"}]

    # one upstream fetch serves all three listings (cache)
    unfiltered = (await dclient.get("/api/llm/models")).json()
    assert len(unfiltered) == 20 and unfiltered[0]["id"] == "gemini-flash-00"
    assert calls == ["list"]


@pytest.mark.anyio
@respx.mock
async def test_openrouter_models_q_filter(dclient):
    await dclient.get("/api/auth/dev-login")
    await dclient.put("/api/settings", json={
        "llm_provider": "openrouter", "llm_model": "qwen/qwen3-vl-235b"
    })
    route = respx.get("https://openrouter.ai/api/v1/models").mock(return_value=Response(200, json={
        "data": [
            {"id": "qwen/qwen3-vl-235b", "name": "Qwen3 VL 235B"},
            {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
            {"id": "no-name-field"},  # name falls back to id
        ]
    }))

    r = await dclient.get("/api/llm/models", params={"q": "qwen"})
    assert r.status_code == 200, r.text
    assert r.json() == [{"id": "qwen/qwen3-vl-235b", "name": "Qwen3 VL 235B"}]
    assert route.call_count == 1  # cached across the unfiltered retry below
    assert (await dclient.get("/api/llm/models")).json() == [
        {"id": "qwen/qwen3-vl-235b", "name": "Qwen3 VL 235B"},
        {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
        {"id": "no-name-field", "name": "no-name-field"},
    ]
    assert route.call_count == 1


@pytest.mark.anyio
@respx.mock
async def test_models_upstream_error_is_a_clean_502(dclient, monkeypatch):
    await dclient.get("/api/auth/dev-login")

    respx.get("https://openrouter.ai/api/v1/models").mock(return_value=Response(500))
    await dclient.put("/api/settings", json={"llm_provider": "openrouter", "llm_model": "m"})
    r = await dclient.get("/api/llm/models")
    assert r.status_code == 502 and "OpenRouter model list failed" in r.json()["detail"]

    await dclient.put("/api/settings", json={"llm_provider": "google", "llm_model": "m"})
    fake_genai_module(monkeypatch, error=RuntimeError("quota blown"))
    r = await dclient.get("/api/llm/models")
    assert r.status_code == 502 and "Gemini model list failed" in r.json()["detail"]


@pytest.mark.anyio
async def test_extraction_uses_provider_and_model_from_settings(dclient, db, monkeypatch):
    """The fake client stands in for the real one, but the real get_llm resolves it —
    so the provider/model recorded are the ones PUT /api/settings wrote."""
    await dclient.get("/api/auth/dev-login")
    body = {"llm_provider": "openrouter", "llm_model": "qwen/qwen3-vl-235b"}
    assert (await dclient.put("/api/settings", json=body)).status_code == 200

    seen = []

    class RecordingClient:
        def __init__(self, api_key, model):
            seen.append((api_key, model))

        async def complete_structured(self, schema, contents):
            return ROWS

    monkeypatch.setattr("app.llm.OpenRouterClient", RecordingClient)

    r = await dclient.post(
        "/api/uploads", files={"file": ("receipt.png", PNG, "image/png")}
    )
    assert r.status_code == 201, r.text
    r = await dclient.post("/api/drafts/from-upload", json={"upload_id": r.json()["upload_id"]})
    assert r.status_code == 201, r.text
    assert [row["description"] for row in r.json()["rows"]] == ["Coop run"]
    # the fake got exactly one call; the model is the one PUT wrote (key comes from env —
    # not asserted, it may be the developer's real one)
    assert len(seen) == 1 and seen[0][1] == "qwen/qwen3-vl-235b"
