import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI

from app.auth import register_dev_login, router as auth_router
from app.config import Settings, get_settings
from app.drafts import router as drafts_router
from app.ledger import router as ledger_router
from app.llm import ensure_app_settings
from app.reports import router as reports_router

log = logging.getLogger("pf")

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
        yield

    app = FastAPI(title="Personal Finance API", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    app.include_router(auth_router, prefix="/api")
    app.include_router(ledger_router, prefix="/api")
    app.include_router(reports_router, prefix="/api")
    app.include_router(drafts_router, prefix="/api")
    if settings.dev_auth_enabled:
        register_dev_login(app, settings)

    return app


app = create_app()
