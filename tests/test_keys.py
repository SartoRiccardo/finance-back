"""V7 API keys: mint → use → revoke, hash/full key never in a response, throttled last_used_at."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.auth import hash_key
from app.models import ApiKey


def bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


async def mint(client, name="cli"):
    r = await client.post("/api/keys", json={"name": name})
    assert r.status_code == 201, r.text
    return r


@pytest.mark.anyio
async def test_key_endpoints_require_auth(client):
    assert (await client.get("/api/keys")).status_code == 401
    assert (await client.post("/api/keys", json={"name": "x"})).status_code == 401
    assert (await client.delete("/api/keys/1")).status_code == 401


@pytest.mark.anyio
async def test_mint_use_revoke_and_no_secret_leak(client, db):
    await client.get("/api/auth/dev-login")
    r = await mint(client)
    key = r.json()["key"]
    digest = hash_key(key)
    assert key.startswith("pf_") and len(key) == 35  # pf_ + 32 hex

    # the full key works on a protected route; the cookie still wins when present
    me = await client.get("/api/auth/me", headers=bearer(key))
    assert me.status_code == 200 and me.json()["email"] == "admin@test.dev"
    assert (await client.get("/api/auth/me")).status_code == 200
    assert (await client.get("/api/auth/me", headers={"Authorization": "Bearer ghp_x"})).status_code == 200

    client.cookies.clear()  # from here on the key is the only credential

    listed = await client.get("/api/keys", headers=bearer(key))
    rows = listed.json()
    assert len(rows) == 1
    assert set(rows[0]) == {"id", "name", "key_prefix", "created_at", "last_used_at", "revoked_at"}
    assert rows[0]["key_prefix"] == key[:12] and rows[0]["revoked_at"] is None

    # only the hash is stored, nothing else
    async with db() as s:
        row = (await s.scalars(select(ApiKey))).one()
        assert row.key_hash == digest and len(row.key_hash) == 64

    rev = await client.delete(f"/api/keys/{rows[0]['id']}", headers=bearer(key))
    assert rev.status_code == 200

    # revoked keys are dead forever
    dead = await client.get("/api/auth/me", headers=bearer(key))
    assert dead.status_code == 401
    final = await client.get("/api/keys")
    assert final.status_code == 401

    # neither the full key nor its hash appears in ANY response — except the POST
    # body, the one place the full key is allowed to exist
    for resp in (r, me, listed, rev, dead, final):
        assert digest not in resp.text
        if resp is not r:
            assert key not in resp.text


@pytest.mark.anyio
async def test_garbage_and_foreign_bearer_are_401(client):
    for header in (
        "Bearer ghp_notours",                    # foreign format
        f"pf_{'z' * 32}",                        # no scheme
        f"Bearer pf_{'g' * 32}",                 # right shape, not hex
        f"Bearer pf_{hash_key('x')}",            # hash of something — mints nothing
        "Bearer",                                # empty
        "Basic dXNlcjpwYXNz",                    # wrong scheme
    ):
        r = await client.get("/api/auth/me", headers={"Authorization": header})
        assert r.status_code == 401, header


@pytest.mark.anyio
async def test_last_used_at_updates_at_most_once_a_minute(client, db):
    await client.get("/api/auth/dev-login")
    key = (await mint(client)).json()["key"]
    client.cookies.clear()

    async with db() as s:  # untouched at mint time
        assert (await s.scalars(select(ApiKey))).one().last_used_at is None

    await client.get("/api/auth/me", headers=bearer(key))
    first = (await client.get("/api/keys", headers=bearer(key))).json()[0]["last_used_at"]
    assert first is not None

    # inside the window: no write
    await client.get("/api/auth/me", headers=bearer(key))
    assert (await client.get("/api/keys", headers=bearer(key))).json()[0]["last_used_at"] == first

    # outside the window: written again
    async with db() as s:
        row = (await s.scalars(select(ApiKey))).one()
        row.last_used_at = datetime.now(UTC) - timedelta(minutes=2)
        await s.commit()
    await client.get("/api/auth/me", headers=bearer(key))
    assert (await client.get("/api/keys", headers=bearer(key))).json()[0]["last_used_at"] != first


@pytest.mark.anyio
async def test_revoke_unknown_key_404(client):
    await client.get("/api/auth/dev-login")
    assert (await client.delete("/api/keys/999")).status_code == 404
