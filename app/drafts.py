"""V5 photo drafts: upload → LLM extraction → reviewable rows → approve/discard.

`create_draft_from_content` is the shared pipeline — V6's email pipeline calls it
with a str payload instead of image bytes. Endpoints stay thin.
"""

import uuid
from datetime import date, datetime
from decimal import Decimal, DecimalException
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_app_settings, get_current_user
from app.config import Settings
from app.db import get_db
from app.ledger import TxnIn, TxnOut, TxnPatch, _check_refs
from app.llm import LLMClient, LLMError, get_llm, rows_schema
from app.models import Category, Draft, Transaction, Upload, User

router = APIRouter(tags=["drafts"], dependencies=[Depends(get_current_user)])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGE_EXTS = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
    "image/heic": ".heic", "image/heif": ".heif", "image/gif": ".gif",
}
MAX_AMOUNT = Decimal("9999999999.99")  # Numeric(12,2) ceiling

# V6 adds "email": <prompt> and passes a str payload; everything else is shared.
EXTRACT_PROMPTS = {
    "photo": (
        "This is a photo of a receipt or invoice. Extract every purchased line item as a row: "
        "the item's date (fall back to the receipt date), a short description, the line amount "
        "in euros as a positive number, and the best-fitting category from the allowed values. "
        "Skip totals, payment methods and change."
    ),
}


class DraftOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    status: str
    upload_id: uuid.UUID | None
    created_at: datetime
    rows: list[TxnOut]


class FromUploadIn(BaseModel):
    upload_id: uuid.UUID


def _rows_from_llm(raw, cat_ids: dict[str, int], misc_id: int) -> list[dict]:
    """LLM output → Transaction kwargs. Anything malformed aborts the whole draft."""
    if not isinstance(raw, dict) or not isinstance(raw.get("rows"), list) or not raw["rows"]:
        raise LLMError('LLM returned no usable rows — expected {"rows": [...]}')
    rows = []
    for i, r in enumerate(raw["rows"]):
        try:
            description = str(r["description"]).strip()
            day = date.fromisoformat(str(r["date"]))
            amount = Decimal(str(r["amount"]))
            category = str(r.get("category", "")).strip().lower()
        except (KeyError, TypeError, ValueError, DecimalException):
            raise LLMError(f"LLM row {i} is malformed: {r!r}") from None
        if not description:
            raise LLMError(f"LLM row {i} has an empty description")
        if not (0 < amount <= MAX_AMOUNT):
            raise LLMError(f"LLM row {i} amount must be positive and fit 12 digits: {amount}")
        rows.append({
            "date": day,
            "description": description[:500],
            "amount": amount.quantize(Decimal("0.01")),
            "direction": "spend",
            # case-insensitive contains; unknown → Misc
            "category_id": next((cid for name, cid in cat_ids.items() if category in name), misc_id),
            "label_id": None,
        })
    return rows


async def create_draft_from_content(
    db: AsyncSession,
    user: User,
    llm: LLMClient,
    source: str,
    payload: str | tuple[bytes, str],
    *,
    upload_id: uuid.UUID | None = None,
    email_meta: dict | None = None,
) -> Draft:
    """Extract rows from image bytes or (V6) email text and persist an open draft.

    One LLM call → validated rows → draft + transactions with is_draft=true.
    Raises LLMError before anything is written if the model output is unusable.
    """
    prompt = EXTRACT_PROMPTS.get(source)
    if prompt is None:
        raise LLMError(f"Unknown draft source {source!r}")
    categories = (await db.scalars(select(Category).order_by(Category.sort_order))).all()
    cat_ids = {c.name.lower(): c.id for c in categories}

    raw = await llm.complete_structured(
        rows_schema([c.name for c in categories]), [prompt, payload]
    )
    rows = _rows_from_llm(raw, cat_ids, cat_ids["misc"])

    draft = Draft(source=source, upload_id=upload_id, email_meta=email_meta, raw_llm_output=raw)
    db.add(draft)
    await db.flush()  # draft.id for the row FK
    db.add_all(
        Transaction(user_id=user.id, is_draft=True, source=source, draft_id=draft.id, **r)
        for r in rows
    )
    await db.commit()
    await db.refresh(draft, ["rows"])  # the collection isn't loaded on a brand-new draft
    return draft


async def _draft_or_404(db: AsyncSession, draft_id: int) -> Draft:
    draft = await db.get(Draft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="Not found")
    return draft


def _check_open(draft: Draft) -> None:
    if draft.status != "open":
        raise HTTPException(status_code=409, detail=f"Draft is {draft.status}")


def _upload_file(settings: Settings, upload: Upload | None) -> Path | None:
    return Path(settings.upload_dir) / upload.stored_path if upload else None


def _delete_upload_file(settings: Settings, upload: Upload | None) -> None:
    if (path := _upload_file(settings, upload)) is not None:
        path.unlink(missing_ok=True)


# --- uploads ---


@router.post("/uploads", status_code=201)
async def create_upload(
    file: UploadFile,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=422, detail="file must be an image (image/*)")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file must be 10MB or less")

    ext = IMAGE_EXTS.get(file.content_type) or Path(file.filename or "").suffix.lower() or ".img"
    upload_id = uuid.uuid4()
    upload = Upload(
        id=upload_id,
        stored_path=f"{upload_id}{ext}",
        original_name=file.filename,
        mime_type=file.content_type,
        size_bytes=len(data),
    )
    directory = Path(settings.upload_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / upload.stored_path).write_bytes(data)
    db.add(upload)
    await db.commit()
    return {"upload_id": str(upload_id)}


@router.get("/uploads/{upload_id}")
async def get_upload_file(
    upload_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """The stored image, for draft thumbnails."""
    upload = await db.get(Upload, upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail="Not found")
    if not (path := _upload_file(settings, upload)) or not path.exists():
        raise HTTPException(status_code=410, detail="File was deleted")
    return FileResponse(path, media_type=upload.mime_type)


# --- drafts ---


@router.post("/drafts/from-upload", status_code=201, response_model=DraftOut)
async def create_draft_from_upload(
    body: FromUploadIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    llm: LLMClient = Depends(get_llm),
):
    upload = await db.get(Upload, body.upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail="Not found")
    if not (path := _upload_file(settings, upload)) or not path.exists():
        raise HTTPException(status_code=410, detail="Uploaded file is gone")
    try:
        return await create_draft_from_content(
            db, user, llm, "photo", (path.read_bytes(), upload.mime_type), upload_id=upload.id
        )
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None


@router.get("/drafts", response_model=list[DraftOut])
async def list_drafts(db: AsyncSession = Depends(get_db)):
    return (
        await db.scalars(select(Draft).where(Draft.status == "open").order_by(Draft.id.desc()))
    ).all()


@router.get("/drafts/{draft_id}", response_model=DraftOut)
async def get_draft(draft_id: int, db: AsyncSession = Depends(get_db)):
    return await _draft_or_404(db, draft_id)


# --- draft rows (same validation as V2 transaction rows) ---


async def _row_or_404(db: AsyncSession, draft: Draft, row_id: int, user: User) -> Transaction:
    row = await db.get(Transaction, row_id)
    if row is None or row.draft_id != draft.id or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    return row


@router.post("/drafts/{draft_id}/rows", status_code=201, response_model=TxnOut)
async def add_draft_row(
    draft_id: int,
    body: TxnIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await _draft_or_404(db, draft_id)
    _check_open(draft)
    await _check_refs(db, body)
    row = Transaction(
        user_id=user.id, is_draft=True, source=draft.source, draft_id=draft.id,
        **body.model_dump(),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.patch("/drafts/{draft_id}/rows/{row_id}", response_model=TxnOut)
async def patch_draft_row(
    draft_id: int,
    row_id: int,
    body: TxnPatch,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await _draft_or_404(db, draft_id)
    _check_open(draft)
    row = await _row_or_404(db, draft, row_id, user)
    changes = body.model_dump(exclude_unset=True)
    try:  # re-validate the merged row against the direction constraints, like V2
        merged = TxnIn(**{k: getattr(row, k) for k in TxnIn.model_fields} | changes)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from None
    await _check_refs(db, merged)
    for key, value in changes.items():
        setattr(row, key, value)
    await db.commit()
    await db.refresh(row)
    return row


@router.delete("/drafts/{draft_id}/rows/{row_id}", status_code=204)
async def delete_draft_row(
    draft_id: int,
    row_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await _draft_or_404(db, draft_id)
    _check_open(draft)
    row = await _row_or_404(db, draft, row_id, user)
    await db.delete(row)
    await db.commit()


# --- close out ---


@router.post("/drafts/{draft_id}/approve", response_model=DraftOut)
async def approve_draft(
    draft_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    draft = await _draft_or_404(db, draft_id)
    _check_open(draft)
    for row in draft.rows:
        row.is_draft = False  # draft_id + source stay for provenance
    draft.status = "approved"
    _delete_upload_file(settings, draft.upload)
    await db.commit()
    return draft


@router.delete("/drafts/{draft_id}", status_code=204)
async def discard_draft(
    draft_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    draft = await _draft_or_404(db, draft_id)
    _check_open(draft)
    await db.execute(delete(Transaction).where(Transaction.draft_id == draft.id))
    draft.status = "rejected"  # draft + upload rows kept; only the file and its rows go
    _delete_upload_file(settings, draft.upload)
    await db.commit()
