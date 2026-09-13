"""V5.1 model picker: settings round-trip + live provider catalog.

Both upstreams are faked — google-genai via a fake module in sys.modules,
OpenRouter via respx. No real API calls in CI.
"""

import asyncio
import base64
import sys
import types

import pytest
import respx
from httpx import Response
from sqlalchemy import select

from app.llm import Usage, _catalog_cache
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
        # the real AsyncModels.list() is a coroutine resolving to the pager —
        # mimic it, or a missing `await` in production code passes these tests
        def list(self):
            calls.append("list")

            async def _call():
                return FakePager()

            return _call()

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
    assert r.json() == {"llm_provider": "google", "llm_model": "gemini-2.5-flash", "custom_prompt": ""}

    body = {"llm_provider": "openrouter", "llm_model": "qwen/qwen3-vl-235b",
            "custom_prompt": "apple vinegar goes in Self Care"}
    r = await dclient.put("/api/settings", json=body)
    assert r.status_code == 200 and r.json() == body
    assert (await dclient.get("/api/settings")).json() == body

    async with db() as s:
        row = await s.get(AppSetting, 1)
        assert (row.llm_provider, row.llm_model) == (body["llm_provider"], body["llm_model"])
        assert row.custom_prompt == "apple vinegar goes in Self Care"


@pytest.mark.anyio
@pytest.mark.parametrize("body", [
    {"llm_provider": "anthropic", "llm_model": "claude"},  # unknown provider
    {"llm_provider": "google", "llm_model": ""},           # empty model
    {"llm_provider": "google"},                            # missing model
    {"llm_model": "gemini-2.5-flash"},                     # missing provider
    {"llm_provider": "google", "llm_model": "gemini-2.5-flash",
     "custom_prompt": "x" * 2001},                         # prompt over the column cap
])
async def test_settings_validation_rejects_and_persists_nothing(dclient, db, body):
    await dclient.get("/api/auth/dev-login")
    r = await dclient.put("/api/settings", json=body)
    assert r.status_code == 422, r.text
    async with db() as s:
        assert (await s.scalars(select(AppSetting))).all() == []


@pytest.mark.anyio
async def test_custom_prompt_preserved_by_partial_put_and_clearable(dclient):
    """A picker save carries no custom_prompt — it must survive; only '' (or blank) clears."""
    await dclient.get("/api/auth/dev-login")
    assert (await dclient.put("/api/settings", json={
        "llm_provider": "google", "llm_model": "gemini-2.5-flash",
        "custom_prompt": "  apple vinegar goes in Self Care  ",
    })).status_code == 200
    assert (await dclient.get("/api/settings")).json()["custom_prompt"] == \
        "apple vinegar goes in Self Care"  # stored stripped

    assert (await dclient.put("/api/settings", json={
        "llm_provider": "openrouter", "llm_model": "qwen/qwen3-vl-235b",
    })).json() == {"llm_provider": "openrouter", "llm_model": "qwen/qwen3-vl-235b",
                   "custom_prompt": "apple vinegar goes in Self Care"}

    assert (await dclient.put("/api/settings", json={
        "llm_provider": "openrouter", "llm_model": "qwen/qwen3-vl-235b", "custom_prompt": " ",
    })).json()["custom_prompt"] == ""


@pytest.mark.anyio
async def test_google_models_q_filter_normalize_cap(dclient, monkeypatch):
    await dclient.get("/api/auth/dev-login")
    models = (
        [model(f"models/gemini-flash-{i:02d}", f"Gemini Flash {i:02d}") for i in range(25)]
        + [model("models/gemini-2.5-pro", "Gemini 2.5 Pro")]
        + [model("models/gemini-shiny")]  # no display_name → name falls back to id
        + [model("models/gemini-embedding-001", "Embeddings"),  # gemini prefix, not chat
           model("models/gemini-2.5-flash-tts", "TTS")]  # → filtered out
    )
    calls = fake_genai_module(monkeypatch, models)

    async def fake_prices():
        return {"gemini-flash-00": (0.3, 2.5), "gemini-shiny": (None, None)}

    monkeypatch.setattr("app.llm._openrouter_price_map", fake_prices)

    r = await dclient.get("/api/llm/models", params={"q": "FLASH"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 20  # 25 matches, capped
    assert body[0] == {
        "id": "gemini-flash-00", "name": "Gemini Flash 00",
        "input_cost": 0.3, "output_cost": 2.5,  # joined from OpenRouter's listing
    }
    assert all("flash" in m["id"] for m in body)

    # matches on name too; no display_name falls back to the raw id; no price → None
    r = await dclient.get("/api/llm/models", params={"q": "2.5 pro"})
    assert r.json() == [{
        "id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "input_cost": None, "output_cost": None,
    }]
    r = await dclient.get("/api/llm/models", params={"q": "shiny"})
    assert r.json() == [{"id": "gemini-shiny", "name": "models/gemini-shiny",
                         "input_cost": None, "output_cost": None}]

    # non-chat models sharing the gemini prefix (tts/embedding) never reach the picker
    assert (await dclient.get("/api/llm/models", params={"q": "embedding"})).json() == []
    assert (await dclient.get("/api/llm/models", params={"q": "tts"})).json() == []

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
            {"id": "qwen/qwen3-vl-235b", "name": "Qwen3 VL 235B",
             "architecture": {"input_modalities": ["text", "image"]},
             "pricing": {"prompt": "0.0000002", "completion": "0.0000006"}},
            {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash",
             "architecture": {"input_modalities": ["text", "image"]},
             "pricing": {"prompt": "0", "completion": "0"}},  # free variant
            {"id": "text-only-model", "name": "Text Only",
             "architecture": {"input_modalities": ["text"]},
             "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"id": "no-name-field",  # name falls back to id; no pricing → n/a
             "architecture": {"input_modalities": ["text", "image"]}},
        ]
    }))

    r = await dclient.get("/api/llm/models", params={"q": "qwen"})
    assert r.status_code == 200, r.text
    assert r.json() == [{"id": "qwen/qwen3-vl-235b", "name": "Qwen3 VL 235B",
                         "input_cost": 0.2, "output_cost": 0.6}]
    assert route.call_count == 1  # cached across the unfiltered retry below
    assert (await dclient.get("/api/llm/models")).json() == [
        {"id": "qwen/qwen3-vl-235b", "name": "Qwen3 VL 235B",
         "input_cost": 0.2, "output_cost": 0.6},
        {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash",
         "input_cost": 0.0, "output_cost": 0.0},
        {"id": "no-name-field", "name": "no-name-field",
         "input_cost": None, "output_cost": None},
    ]
    assert route.call_count == 1


@pytest.mark.anyio
@respx.mock
async def test_openrouter_models_sorted_by_intelligence(make_client):
    """Smartest first; variants/aliases/pinned versions inherit the line's rank; unranked last."""
    dclient = make_client(openrouter_api_key="test-key")
    await dclient.get("/api/auth/dev-login")
    vision = {"architecture": {"input_modalities": ["text", "image"]}}
    respx.get("https://openrouter.ai/api/v1/models").mock(return_value=Response(200, json={
        "data": [
            {"id": "a/glm-5:free", "name": "GLM 5 (free)", "canonical_slug": "a/glm-5-20260901", **vision},
            {"id": "b/small", "name": "Small", "canonical_slug": "b/small-20260801", **vision},
            {"id": "~a/glm-5", "name": "GLM 5", "canonical_slug": "a/glm-5-20260915", **vision},
            {"id": "c/unranked", "name": "Unranked", **vision},  # no canonical_slug
        ]
    }))
    benchmarks = respx.get("https://openrouter.ai/api/v1/benchmarks").mock(
        return_value=Response(200, json={"data": [
            {"model_permaslug": "b/small-20260801", "intelligence_index": 50},
            {"model_permaslug": "a/glm-5-20260901", "intelligence_index": 40},
            {"model_permaslug": "a/glm-5-20260915", "intelligence_index": 45},
        ]})
    )

    await dclient.put("/api/settings", json={"llm_provider": "openrouter", "llm_model": "a/glm-5"})
    r = await dclient.get("/api/llm/models")
    assert r.status_code == 200, r.text
    # newest pinned glm-5 version's score (45) wins for the whole line; variant+alias
    # join it via _base_slug (tie → stable catalog order); unranked last
    assert [m["id"] for m in r.json()] == ["b/small", "a/glm-5:free", "~a/glm-5", "c/unranked"]
    assert all(set(m) == {"id", "name", "input_cost", "output_cost"} for m in r.json())
    assert benchmarks.call_count == 1  # cached alongside the catalog


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
        provider = "openrouter"

        def __init__(self, api_key, model):
            seen.append((api_key, model))
            self.model = model

        async def complete_structured(self, schema, contents):
            return ROWS, Usage()

    monkeypatch.setattr("app.llm.OpenRouterClient", RecordingClient)

    r = await dclient.post(
        "/api/uploads", files={"file": ("receipt.png", PNG, "image/png")}
    )
    assert r.status_code == 201, r.text
    r = await dclient.post("/api/drafts/from-upload", json={"upload_id": r.json()["upload_id"]})
    assert r.status_code == 201, r.text
    # extraction is detached — wait for it to land
    for _ in range(200):
        draft = (await dclient.get(f"/api/drafts/{r.json()['id']}")).json()
        if draft["status"] != "processing":
            break
        await asyncio.sleep(0.005)
    assert [row["description"] for row in draft["rows"]] == ["Coop run"]
    # the fake got exactly one call; the model is the one PUT wrote (key comes from env —
    # not asserted, it may be the developer's real one)
    assert len(seen) == 1 and seen[0][1] == "qwen/qwen3-vl-235b"
