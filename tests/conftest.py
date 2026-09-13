import base64
import time
import uuid

import asyncpg
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool, StaticPool

from app.config import Settings
from app.db import get_db
from app.main import create_app
from app.models import Base, seed

TEST_SETTINGS = {
    "database_url": "sqlite+aiosqlite://",
    "session_secret": "test-secret",
    "master_admin_email": "admin@test.dev",
    "google_client_id": "test-client-id",
    "google_client_secret": "test-client-secret",
    "frontend_url": "http://front.test",
    "env": "local",
    "dev_auth_bypass": True,
}

# Dedicated scratch database on the dev Postgres server; never touches the dev `pf` DB.
PG_URL = "postgresql+asyncpg://pf:pf@localhost:5432/pf_test"
PG_DSN = "postgresql://pf:pf@localhost:5432/pf"


async def _pg_engine():
    """Engine on the scratch Postgres DB, or None if the dev server is unreachable."""
    try:
        admin = await asyncpg.connect(PG_DSN)
    except (OSError, asyncpg.PostgresError):
        return None
    try:
        await admin.execute("CREATE DATABASE pf_test")
    except asyncpg.DuplicateDatabaseError:
        pass
    await admin.close()
    return create_async_engine(PG_URL, poolclass=NullPool)


@pytest.fixture(params=["sqlite", "postgres"], ids=["sqlite", "pg"])
async def db(request):
    """Every feature test runs on in-memory sqlite AND on real Postgres (when reachable)."""
    if request.param == "sqlite":
        engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    else:
        engine = await _pg_engine()
        if engine is None:
            pytest.skip("dev Postgres not reachable")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(seed)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def make_client(db):
    """Async client bound to an in-memory sqlite DB; overrides get_db."""

    def _make(**overrides):
        app = create_app(Settings(**(TEST_SETTINGS | overrides)))

        async def override():
            async with db() as session:
                yield session

        app.dependency_overrides[get_db] = override
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    return _make


@pytest.fixture
def client(make_client):
    return make_client()


def _b64uint(n: int) -> str:
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class FakeGoogle:
    """Stands in for Google's token + JWKS endpoints (used with respx)."""

    def __init__(self, client_id: str = "test-client-id"):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.client_id = client_id

    @property
    def jwks(self) -> dict:
        pub = self.key.public_key().public_numbers()
        return {
            "keys": [{
                "kty": "RSA", "use": "sig", "alg": "RS256", "kid": "test-key",
                "n": _b64uint(pub.n), "e": _b64uint(pub.e),
            }]
        }

    def id_token(self, email: str, expired: bool = False) -> str:
        now = int(time.time())
        claims = {
            "iss": "https://accounts.google.com",
            "aud": self.client_id,
            "sub": "1234567890",
            "email": email,
            "email_verified": True,
            "name": "Test Admin",
            "picture": "http://pic.test/a.png",
            "iat": now,
            "exp": now - 10 if expired else now + 300,
        }
        return jwt.encode(claims, self.key, algorithm="RS256", headers={"kid": "test-key"})


@pytest.fixture
def google():
    return FakeGoogle()
