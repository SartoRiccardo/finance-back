import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI, Request

from app.auth import register_dev_login, router as auth_router
from app.config import Settings, get_settings
from app.db import SessionLocal
from app.drafts import router as drafts_router
from app.email_ingest import start_email_poller
from app.ledger import router as ledger_router
from app.keys import router as keys_router
from app.llm import ensure_app_settings, router as llm_router
from app.reports import router as reports_router

log = logging.getLogger("pf")

# pf.* logs at INFO (poller confirmations, "email → draft N"); without a handler
# Python's last-resort default only surfaces WARNING+, hiding exactly those lines.
_pf_handler = logging.StreamHandler()
_pf_handler.setFormatter(logging.Formatter("%(levelname)s  [%(name)s] %(message)s"))
log.addHandler(_pf_handler)
log.setLevel(logging.INFO)

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


def upgrade_head() -> None:
    command.upgrade(Config(str(ALEMBIC_INI)), "head")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Runs env.py, which spins its own event loop — hence the thread.
        await asyncio.to_thread(upgrade_head)
        await ensure_app_settings(settings)
        poller = start_email_poller(settings)  # None (inert) without INGEST_IMAP_HOST
        yield
        if poller:
            poller.cancel()

    app = FastAPI(title="Personal Finance API", lifespan=lifespan)
    app.state.settings = settings
    app.state.db_factory = SessionLocal  # detached work (draft extraction) uses this

    @app.middleware("http")
    async def no_store_for_api(request: Request, call_next):
        # session-derived responses (auth state, the ledger) must never be cached
        # by Cloudflare or the browser — a stale cached /api/auth/me is exactly
        # what causes "login works but I still get bounced back to the login page"
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    app.include_router(auth_router, prefix="/api")
    app.include_router(ledger_router, prefix="/api")
    app.include_router(reports_router, prefix="/api")
    app.include_router(drafts_router, prefix="/api")
    app.include_router(llm_router, prefix="/api")
    app.include_router(keys_router, prefix="/api")
    if settings.dev_auth_enabled:
        register_dev_login(app, settings)

    return app


app = create_app()
