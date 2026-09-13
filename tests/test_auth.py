import pytest
import respx
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import Response
from sqlalchemy import func, select

from app.models import User

TOKEN = "https://oauth2.googleapis.com/token"
JWKS = "https://www.googleapis.com/oauth2/v3/certs"


def mock_google(google, email="admin@test.dev"):
    respx.post(TOKEN).mock(return_value=Response(200, json={
        "id_token": google.id_token(email), "access_token": "at",
    }))
    respx.get(JWKS).mock(return_value=Response(200, json=google.jwks))


async def login(client, google, email="admin@test.dev"):
    mock_google(google, email)
    await client.get("/api/auth/google/login")
    state = client.cookies.get("pf_oauth")
    return await client.get(f"/api/auth/google/callback?code=abc&state={state}")


async def user_count(db) -> int:
    async with db() as s:
        return (await s.execute(select(func.count()).select_from(User))).scalar_one()


@pytest.mark.anyio
async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.anyio
@respx.mock
async def test_google_login_redirects_with_state(client):
    r = await client.get("/api/auth/google/login")
    assert r.status_code == 302
    assert r.headers["location"].startswith("https://accounts.google.com/o/oauth2/v2/auth")
    assert "scope=openid+email+profile" in r.headers["location"]
    assert client.cookies.get("pf_oauth")


@pytest.mark.anyio
@respx.mock
async def test_full_login_creates_user_and_sets_cookie(client, google, db):
    r = await login(client, google)
    assert r.status_code == 302
    assert r.headers["location"] == "http://front.test"
    assert client.cookies.get("pf_session")

    assert await user_count(db) == 1

    me = await client.get("/api/auth/me")
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == "admin@test.dev"
    assert body["name"] == "Test Admin"


@pytest.mark.anyio
@respx.mock
async def test_second_login_reuses_user(client, google, db):
    await login(client, google)
    first_id = (await client.get("/api/auth/me")).json()["id"]
    await login(client, google)
    assert (await client.get("/api/auth/me")).json()["id"] == first_id
    assert await user_count(db) == 1


@pytest.mark.anyio
@respx.mock
async def test_non_whitelisted_email_forbidden_and_no_row(client, google, db):
    r = await login(client, google, email="intruder@evil.test")
    assert r.status_code == 403
    assert client.cookies.get("pf_session") is None
    assert await user_count(db) == 0


@pytest.mark.anyio
async def test_me_without_cookie_is_401(client):
    assert (await client.get("/api/auth/me")).status_code == 401


@pytest.mark.anyio
@respx.mock
async def test_logout_clears_session(client, google):
    await login(client, google)
    assert (await client.get("/api/auth/me")).status_code == 200
    assert (await client.post("/api/auth/logout")).status_code == 200
    assert client.cookies.get("pf_session") is None
    assert (await client.get("/api/auth/me")).status_code == 401


@pytest.mark.anyio
@respx.mock
async def test_callback_rejects_bad_state(client, google):
    mock_google(google)
    await client.get("/api/auth/google/login")
    r = await client.get("/api/auth/google/callback?code=abc&state=tampered")
    assert r.status_code == 400
    assert (await client.get("/api/auth/me")).status_code == 401


@pytest.mark.anyio
@respx.mock
async def test_callback_rejects_foreign_signature(client, google):
    # Token signed by a different key than the JWKS we serve.
    respx.post(TOKEN).mock(return_value=Response(200, json={
        "id_token": google.id_token("admin@test.dev"), "access_token": "at",
    }))
    google.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    respx.get(JWKS).mock(return_value=Response(200, json=google.jwks))
    await client.get("/api/auth/google/login")
    state = client.cookies.get("pf_oauth")
    r = await client.get(f"/api/auth/google/callback?code=abc&state={state}")
    assert r.status_code == 502


@pytest.mark.anyio
async def test_dev_login_creates_user_and_cookie(client, db):
    r = await client.get("/api/auth/dev-login")
    assert r.status_code == 302
    assert r.headers["location"] == "http://front.test"
    assert client.cookies.get("pf_session")
    assert await user_count(db) == 1
    assert (await client.get("/api/auth/me")).json()["email"] == "admin@test.dev"


@pytest.mark.anyio
async def test_dev_login_disabled_in_production(make_client, db):
    prod = make_client(env="production", dev_auth_bypass=True)
    assert (await prod.get("/api/auth/dev-login")).status_code == 404
    assert await user_count(db) == 0
