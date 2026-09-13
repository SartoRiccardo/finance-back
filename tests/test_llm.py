"""Real-client request shaping — the fake-client tests bypass this branch entirely."""

import base64
import json

import httpx
import pytest
import respx

from app.llm import OpenRouterClient, Usage, _openrouter_part


def test_openrouter_part_text_and_image():
    assert _openrouter_part("hi") == {"type": "text", "text": "hi"}
    part = _openrouter_part((b"\x89PNG", "image/png"))
    assert part["type"] == "image_url"
    assert part["image_url"]["url"] == f"data:image/png;base64,{base64.b64encode(b'\x89PNG').decode()}"


@pytest.mark.anyio
@respx.mock
async def test_openrouter_complete_structured_sends_image_and_parses_rows():
    route = respx.post(OpenRouterClient.URL).mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"rows": []})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
    )
    client = OpenRouterClient("k", "m")
    out, usage = await client.complete_structured({"type": "object"}, ["p", (b"img", "image/jpeg")])
    assert out == {"rows": []} and usage == Usage(10, 5)
    assert client.provider == "openrouter"
    body = json.loads(route.calls[0].request.content)
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
