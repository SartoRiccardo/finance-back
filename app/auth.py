import logging
import secrets
import uuid
from datetime import timedelta
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import get_db
from app.models import User

log = logging.getLogger("pf.auth")

SESSION_COOKIE = "pf_session"
STATE_COOKIE = "pf_oauth"
SESSION_MAX_AGE = int(timedelta(days=30).total_seconds())

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUER = "https://accounts.google.com"
SCOPE = "openid email profile"

router = APIRouter(prefix="/auth", tags=["auth"])


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def _serializer(settings: Settings, salt: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.session_secret, salt=salt)


def _session_cookie(settings: Settings, response: Response, user: User) -> None:
    token = _serializer(settings, "session").dumps(str(user.id))
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=settings.is_prod,
        path="/",
    )


def _read_session(settings: Settings, token: str | None) -> uuid.UUID | None:
    if not token:
        return None
    try:
        value = _serializer(settings, "session").loads(token, max_age=SESSION_MAX_AGE)
        return uuid.UUID(value)
    except (BadSignature, SignatureExpired, ValueError):
        return None


async def _load_user(db: AsyncSession, settings: Settings, token: str | None) -> User | None:
    user_id = _read_session(settings, token)
    return await db.get(User, user_id) if user_id else None


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> User:
    user = await _load_user(db, settings, request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def _exchange_code(settings: Settings, code: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        id_token = resp.json().get("id_token")
        if not id_token:
            raise HTTPException(status_code=502, detail="No id_token from Google")
        jwks = (await client.get(GOOGLE_JWKS_URL)).json()

    return _verify_id_token(settings, id_token, jwks)


def _verify_id_token(settings: Settings, id_token: str, jwks: dict) -> dict:
    try:
        kid = jwt.get_unverified_header(id_token).get("kid")
        key = next(k for k in jwks.get("keys", []) if k.get("kid") == kid)
        return jwt.decode(
            id_token,
            jwt.PyJWK(key).key,
            algorithms=["RS256"],
            audience=settings.google_client_id,
            issuer=GOOGLE_ISSUER,
        )
    except (jwt.PyJWTError, StopIteration) as exc:
        raise HTTPException(status_code=502, detail="Invalid id_token") from exc


async def _upsert_user(db: AsyncSession, email: str, name: str | None, picture: str | None) -> User:
    user = (
        await db.execute(select(User).where(User.email == email.lower()))
    ).scalar_one_or_none()
    if user is None:
        user = User(email=email.lower(), name=name, picture_url=picture)
        db.add(user)
        await db.commit()
    return user


@router.get("/google/login")
async def google_login(request: Request, settings: Settings = Depends(get_app_settings)):
    state = secrets.token_urlsafe(32)
    params = urlencode({
        "client_id": settings.google_client_id,
        "redirect_uri": str(request.url_for("google_callback")),
        "response_type": "code",
        "scope": SCOPE,
        "state": state,
    })
    response = Response(status_code=302, headers={"Location": f"{GOOGLE_AUTH_URL}?{params}"})
    response.set_cookie(
        STATE_COOKIE, state, max_age=600, httponly=True, samesite="lax",
        secure=settings.is_prod, path="/",
    )
    return response


@router.get("/google/callback", name="google_callback")
async def google_callback(
    request: Request,
    code: str,
    state: str,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    if state != request.cookies.get(STATE_COOKIE):
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    claims = await _exchange_code(settings, code, str(request.url_for("google_callback")))

    email = claims.get("email")
    if not email or email.lower() != settings.master_admin_email.lower():
        # Whitelist enforced before any row is created.
        raise HTTPException(status_code=403, detail="Not authorized")

    user = await _upsert_user(db, email, claims.get("name"), claims.get("picture"))
    response = Response(status_code=302, headers={"Location": settings.frontend_url})
    response.delete_cookie(STATE_COOKIE, path="/")
    _session_cookie(settings, response, user)
    return response


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"status": "ok"}


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "picture_url": user.picture_url,
    }


def register_dev_login(app) -> None:
    """Called only when settings.dev_auth_enabled — disabled means the route is never registered (404)."""

    r = APIRouter(prefix="/auth", tags=["auth"])

    @r.get("/dev-login", include_in_schema=False)
    async def dev_login(
        db: AsyncSession = Depends(get_db),
        settings: Settings = Depends(get_app_settings),
    ):
        user = await _upsert_user(db, settings.master_admin_email, "Dev Admin", None)
        response = Response(status_code=302, headers={"Location": settings.frontend_url})
        _session_cookie(settings, response, user)
        return response

    app.include_router(r, prefix="/api")
    log.warning(
        "DEV AUTH BYPASS active: GET /api/auth/dev-login logs in %s — never enable in production",
        settings.master_admin_email,
    )
