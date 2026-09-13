from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://pf:pf@localhost:5432/pf"
    session_secret: str = "dev-secret-change-me"
    master_admin_email: str = "admin@pf.local"
    google_client_id: str = ""
    google_client_secret: str = ""
    frontend_url: str = "http://localhost:5173"
    env: str = "local"
    dev_auth_bypass: bool = False

    @property
    def dev_auth_enabled(self) -> bool:
        return self.dev_auth_bypass and self.env != "production"

    @property
    def is_prod(self) -> bool:
        return self.env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
