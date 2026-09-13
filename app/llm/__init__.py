"""The only place in the codebase where a model is called.

V6 (email pipeline) reuses `complete_structured`; V7 (model search) flips the
app_settings row via this module's helpers — it never talks to a provider itself.
"""

import base64
import json
from typing import Protocol

import httpx
from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_app_settings
from app.config import Settings
from app.db import SessionLocal, get_db
from app.models import AppSetting

# A prompt/email is str; an image is (bytes, mime_type).
Part = str | tuple[bytes, str]


class LLMError(Exception):
    """Provider failure, missing key, or unparseable output — surfaced as 502."""


class LLMClient(Protocol):
    async def complete_structured(self, schema: dict, contents: list[Part]) -> dict: ...


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
    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    async def complete_structured(self, schema: dict, contents: list[Part]) -> dict:
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
            return json.loads(response.text)
        except (TypeError, ValueError) as exc:
            raise LLMError(f"Gemini returned non-JSON output: {response.text!r}") from exc


class OpenRouterClient:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        self.api_key, self.model = api_key, model

    async def complete_structured(self, schema: dict, contents: list[Part]) -> dict:
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is empty — add it to api/.env")
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}
            if isinstance(c, tuple) else {"type": "text", "text": c}
            for c in contents
        ]
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
            return json.loads(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"OpenRouter returned malformed output: {body!r}") from exc


async def get_llm(
    db: AsyncSession = Depends(get_db), settings: Settings = Depends(get_app_settings)
) -> LLMClient:
    """Resolve provider+model from the app_settings row, falling back to env defaults.

    Never raises at startup or in tests: an empty key surfaces only when a real
    call is made. Tests override this dependency with a fake client.
    """
    row = await db.get(AppSetting, 1)
    provider = row.llm_provider if row else settings.llm_provider
    model = row.llm_model if row else settings.llm_model
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
