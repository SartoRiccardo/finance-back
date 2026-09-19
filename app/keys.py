"""API keys: mint once, show the full key once, store only the hash, never log either."""

import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user, hash_key
from app.db import get_db
from app.models import ApiKey, User

router = APIRouter(tags=["keys"])


class KeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)  # String(120) column


class KeyCreatedOut(BaseModel):
    """The only response that ever carries `key` — the full secret, shown exactly once."""

    id: int
    name: str
    key_prefix: str
    created_at: datetime
    key: str


class KeyOut(BaseModel):
    id: int
    name: str
    key_prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


@router.post("/keys", status_code=201, response_model=KeyCreatedOut)
async def create_key(
    body: KeyIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    key = "pf_" + secrets.token_hex(16)  # 32 hex chars
    row = ApiKey(user_id=user.id, name=body.name, key_prefix=key[:12], key_hash=hash_key(key))
    db.add(row)
    await db.commit()
    await db.refresh(row)  # server default filled created_at
    return KeyCreatedOut(
        id=row.id, name=row.name, key_prefix=row.key_prefix, created_at=row.created_at, key=key
    )


@router.get("/keys", response_model=list[KeyOut])
async def list_keys(
    db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)
):
    return (await db.scalars(
        select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.id.desc())
    )).all()


@router.delete("/keys/{key_id}")
async def revoke_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    row = (await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Key not found")
    if row.revoked_at is None:  # soft revoke — idempotent
        row.revoked_at = datetime.now(UTC)
        await db.commit()
    return {"status": "ok"}
