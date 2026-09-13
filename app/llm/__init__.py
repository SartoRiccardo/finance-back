"""The only place in the codebase where a model is called.

V6 (email pipeline) reuses `complete_structured`; V5.1 (model picker) flips the
app_settings row and searches provider catalogs here — it never talks to a
provider itself except to list models.
"""

import base64
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DecimalException
from typing import Literal, Protocol

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_app_settings, get_current_user
from app.config import Settings
from app.db import SessionLocal, get_db
from app.models import AppSetting, LLMUsage

# A prompt/email is str; an image is (bytes, mime_type).
Part = str | tuple[bytes, str]


class LLMError(Exception):
    """Provider failure, missing key, or unparseable output — surfaced as 502."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(Protocol):
    provider: str
    model: str

    async def complete_structured(self, schema: dict, contents: list[Part]) -> tuple[dict, Usage]: ...


def rows_schema(category_names: list[str]) -> dict:
    """Extraction contract shared by both providers and both pipelines (V5 photo, V6 email)."""
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "Transaction date, YYYY-MM-DD"},
                        "description": {"type": "string"},
                        "amount": {"type": "number", "exclusiveMinimum": 0, "description": "positive, in euros"},
                        "category": {"type": "string", "enum": category_names},
                    },
                    "required": ["date", "description", "amount", "category"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    }


def _parts_for_gemini(contents: list[Part]):
    from google.genai import types

    return [
        types.Part.from_bytes(data=data, mime_type=mime) if isinstance(c, tuple) else types.Part(text=c)
        for c in contents
    ]


class GoogleClient:
    provider = "google"

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    async def complete_structured(self, schema: dict, contents: list[Part]) -> tuple[dict, Usage]:
        if not self.api_key:
            raise LLMError("GOOGLE_API_KEY is empty — add it to api/.env")
        # Imported here so the app never boots (or tests never run) on this import.
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=_parts_for_gemini(contents))],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=schema,  # plain JSON schema, shared with OpenRouter
                ),
            )
        except Exception as exc:
            raise LLMError(f"Gemini call failed: {exc}") from exc
        try:
            raw = json.loads(response.text)
        except (TypeError, ValueError) as exc:
            raise LLMError(f"Gemini returned non-JSON output: {response.text!r}") from exc
        meta = getattr(response, "usage_metadata", None)
        return raw, Usage(
            getattr(meta, "prompt_token_count", None),
            getattr(meta, "candidates_token_count", None),
        )


def _openrouter_part(c: Part) -> dict:
    """One chat-completions content part: text, or a tuple as a base64 data URL."""
    if isinstance(c, str):
        return {"type": "text", "text": c}
    data, mime = c
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"},
    }


class OpenRouterClient:
    URL = "https://openrouter.ai/api/v1/chat/completions"
    provider = "openrouter"

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    async def complete_structured(self, schema: dict, contents: list[Part]) -> tuple[dict, Usage]:
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is empty — add it to api/.env")
        content = [_openrouter_part(c) for c in contents]
        try:
            async with httpx.AsyncClient(timeout=120) as http:
                resp = await http.post(
                    self.URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": content}],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"name": "extraction", "strict": True, "schema": schema},
                        },
                    },
                )
                resp.raise_for_status()
                body = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenRouter call failed: {exc}") from exc
        try:
            raw = json.loads(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"OpenRouter returned malformed output: {body!r}") from exc
        used = body.get("usage") or {}
        toks = Usage(used.get("prompt_tokens"), used.get("completion_tokens"))
        return raw, toks


async def current_llm(db: AsyncSession, settings: Settings) -> tuple[str, str]:
    """The (provider, model) the app_settings row holds, falling back to env defaults."""
    row = await db.get(AppSetting, 1)
    return (
        (row.llm_provider, row.llm_model) if row
        else (settings.llm_provider, settings.llm_model)
    )


async def get_llm(
    db: AsyncSession = Depends(get_db), settings: Settings = Depends(get_app_settings)
) -> LLMClient:
    """Resolve provider+model from the app_settings row, falling back to env defaults.

    Never raises at startup or in tests: an empty key surfaces only when a real
    call is made. Tests override this dependency with a fake client.
    """
    provider, model = await current_llm(db, settings)
    if provider == "google":
        return GoogleClient(settings.google_api_key, model)
    if provider == "openrouter":
        return OpenRouterClient(settings.openrouter_api_key, model)
    raise HTTPException(status_code=502, detail=f"Unknown LLM provider {provider!r} in app_settings")


async def ensure_app_settings(settings: Settings) -> None:
    """Create the single app_settings row from env defaults if missing (runs on startup)."""
    async with SessionLocal() as db:
        if await db.get(AppSetting, 1) is None:
            db.add(AppSetting(
                id=1, llm_provider=settings.llm_provider, llm_model=settings.llm_model
            ))
            await db.commit()


# --- V5.1 model picker: settings round-trip + live provider catalog ---


class LLMSettingsOut(BaseModel):
    llm_provider: str
    llm_model: str


class LLMSettingsIn(BaseModel):
    # Literal + length bounds are the validation: invalid → FastAPI's 422.
    llm_provider: Literal["google", "openrouter"]
    llm_model: str = Field(min_length=1, max_length=120)  # String(120) column


router = APIRouter(tags=["settings"], dependencies=[Depends(get_current_user)])


@router.get("/settings", response_model=LLMSettingsOut)
async def read_llm_settings(
    db: AsyncSession = Depends(get_db), settings: Settings = Depends(get_app_settings)
):
    provider, model = await current_llm(db, settings)
    return LLMSettingsOut(llm_provider=provider, llm_model=model)


@router.put("/settings", response_model=LLMSettingsOut)
async def update_llm_settings(body: LLMSettingsIn, db: AsyncSession = Depends(get_db)):
    row = await db.get(AppSetting, 1)
    if row is None:  # startup didn't run (or row was dropped) — create it now
        row = AppSetting(id=1)
        db.add(row)
    row.llm_provider, row.llm_model = body.llm_provider, body.llm_model
    await db.commit()
    return LLMSettingsOut(llm_provider=row.llm_provider, llm_model=row.llm_model)


CATALOG_TTL = 300  # seconds
_catalog_cache: dict[str, tuple[float, list[dict]]] = {}


def _per_million(price) -> float | None:
    """OpenRouter prices are per-token USD strings; the picker reads better per million."""
    try:
        return round(float(price) * 1_000_000, 4)
    except (TypeError, ValueError):
        return None


async def _openrouter_models() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get("https://openrouter.ai/api/v1/models")
            resp.raise_for_status()
            data = resp.json()["data"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise LLMError(f"OpenRouter model list failed: {exc}") from exc
    out = []
    for m in data:
        if not {"text", "image"} <= set((m.get("architecture") or {}).get("input_modalities") or []):
            continue  # picker is for receipt vision models only
        price = m.get("pricing") or {}
        out.append({
            "id": m["id"],
            "name": m.get("name") or m["id"],
            "input_cost": _per_million(price.get("prompt")),
            "output_cost": _per_million(price.get("completion")),
            # internal: joins the model to its intelligence benchmark (popped before return)
            "slug": m.get("canonical_slug") or m["id"],
        })
    return out


async def _openrouter_price_map() -> dict[str, tuple[float | None, float | None]]:
    """Google model id → (in, out) per-M USD, via OpenRouter's listing of the same model.

    Google's ListModels carries no pricing, so this is the nearest fetchable number
    (OpenRouter may add a small cut vs Google's own price). Missing id → None → "n/a".
    """
    try:
        return {m["id"].removeprefix("google/"): (m["input_cost"], m["output_cost"])
                for m in await _openrouter_models()}
    except LLMError:
        return {}  # OpenRouter down → the google listing still works, costs just show n/a


def _base_slug(slug: str) -> str:
    """Collapse slug spellings to the model line: pinned versions, variants, aliases."""
    slug = slug.strip("~")  # provider aliases like ~z-ai/glm-flash-latest
    slug = slug.split(":")[0]  # :free/:batch variants
    return re.sub(r"-\d{8}$", "", slug)  # canonical slugs are date-pinned


async def _intelligence_index(api_key: str) -> dict[str, float]:
    """Base model slug → Artificial Analysis intelligence index, for ranking the picker.

    Benchmarks key on `model_permaslug` (date-pinned) while the catalog carries
    unpinned ids — both sides go through _base_slug to meet in the middle. Needs an
    OpenRouter key and is rate-limited hard, so it's only fetched inside the cached
    catalog path. Any failure → {} → catalog order (newest first).
    """
    if not api_key:
        return {}
    url = "https://openrouter.ai/api/v1/benchmarks?source=artificial-analysis&task_type=intelligence"
    try:
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.get(url, headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            index: dict[str, float] = {}
            for m in resp.json()["data"]:
                if m.get("model_permaslug") and m.get("intelligence_index") is not None:
                    key = _base_slug(m["model_permaslug"])
                    # several pinned versions of one line may be listed — keep the best
                    index[key] = max(index.get(key, -1), m["intelligence_index"])
            return index
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return {}


async def _google_models(api_key: str) -> list[dict]:
    from google import genai  # lazy, same reason as the extraction client

    client = genai.Client(api_key=api_key)
    try:
        pager = await client.aio.models.list()  # coroutine → AsyncPager
        page = [m async for m in pager]
    except Exception as exc:  # anything from the SDK → one clean failure shape
        raise LLMError(f"Gemini model list failed: {exc}") from exc
    prices = await _openrouter_price_map()
    out = []
    for m in page:
        gid = m.name.removeprefix("models/")
        # ListModels has no modality info; gemini-* chat models are text+image input,
        # minus the tts/embedding variants that share the gemini prefix.
        if "gemini" not in gid or "tts" in gid or "embedding" in gid:
            continue
        in_cost, out_cost = prices.get(gid, (None, None))
        out.append({
            "id": gid,
            "name": m.display_name or m.name,
            "input_cost": in_cost,
            "output_cost": out_cost,
        })
    return out


async def _catalog(provider: str, settings: Settings) -> list[dict]:
    """Full model list for a provider, cached in memory only — never persisted."""
    if hit := _catalog_cache.get(provider):
        if time.monotonic() - hit[0] < CATALOG_TTL:
            return hit[1]
    if provider == "google":
        models = await _google_models(settings.google_api_key)
    elif provider == "openrouter":
        models = await _openrouter_models()
        index = await _intelligence_index(settings.openrouter_api_key)
        # smartest first; variants/aliases/pinned versions meet in _base_slug,
        # unranked models keep catalog order at the end (stable sort)
        models.sort(key=lambda m: index.get(_base_slug(m["slug"]), -1), reverse=True)
        for m in models:
            m.pop("slug", None)  # internal join key — not part of the response contract
    else:
        raise LLMError(f"Unknown LLM provider {provider!r}")
    _catalog_cache[provider] = (time.monotonic(), models)
    return models


@router.get("/llm/models")
async def list_llm_models(
    q: str = "",
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """Live provider search for the picker, searched against the provider set in settings."""
    provider, _ = await current_llm(db, settings)
    try:
        models = await _catalog(provider, settings)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    if needle := q.strip().lower():
        models = [m for m in models if needle in m["id"].lower() or needle in m["name"].lower()]
    return models[:20]


# --- usage log: the ironic cost of our readings ---


async def _prices(provider: str, settings: Settings) -> dict[str, tuple[float | None, float | None]]:
    """model id → (in, out) USD per-M tokens, from the cached catalogs."""
    try:
        if provider == "openrouter":
            return {m["id"]: (m["input_cost"], m["output_cost"]) for m in await _openrouter_models()}
        if provider == "google":
            return await _openrouter_price_map()  # nearest fetchable number for google models
    except LLMError:
        pass
    return {}


async def usage_cost(provider: str, model: str, usage: Usage, settings: Settings) -> Decimal | None:
    """USD for one call at write-time prices; None when tokens or prices are unknown."""
    if usage.input_tokens is None and usage.output_tokens is None:
        return None
    in_price, out_price = (await _prices(provider, settings)).get(model, (None, None))
    if in_price is None or out_price is None:
        return None
    try:
        cost = (
            Decimal(usage.input_tokens or 0) * Decimal(str(in_price))
            + Decimal(usage.output_tokens or 0) * Decimal(str(out_price))
        ) / Decimal(1_000_000)
    except DecimalException:
        return None
    return cost.quantize(Decimal("0.000001"))


class UsageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: Decimal | None
    draft_id: int | None
    created_at: datetime


class UsagePage(BaseModel):
    totals: dict  # {calls, input_tokens, output_tokens, cost_usd}
    recent: list[UsageOut]


@router.get("/llm/usage", response_model=UsagePage)
async def read_llm_usage(db: AsyncSession = Depends(get_db)):
    calls, in_toks, out_toks, cost = (await db.execute(
        select(
            func.count(LLMUsage.id),
            func.sum(LLMUsage.input_tokens),
            func.sum(LLMUsage.output_tokens),
            func.sum(LLMUsage.cost_usd),
        )
    )).one()
    recent = (await db.scalars(
        select(LLMUsage).order_by(LLMUsage.id.desc()).limit(50)
    )).all()
    return UsagePage(
        totals={
            "calls": calls,
            "input_tokens": in_toks,
            "output_tokens": out_toks,
            "cost_usd": cost,
        },
        recent=recent,
    )
