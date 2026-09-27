"""V5 photo drafts: upload → LLM extraction → reviewable rows → approve/discard.

Extraction runs in a task detached from the request: leaving the page mid-read
surfaces as a "processing" draft in the list, never as a lost request.
`create_draft_from_content` is the shared pipeline — V6's email pipeline calls it
with a str payload instead of image bytes. Endpoints stay thin.
"""

import asyncio
import logging
import uuid
from datetime import date, datetime
from decimal import Decimal, DecimalException
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth import get_app_settings, get_current_user
from app.config import Settings
from app.db import get_db
from app.ledger import TxnIn, TxnOut, TxnPatch, _check_refs
from app.llm import LLMClient, LLMError, bool_schema, get_llm, rows_schema, usage_cost
from app.models import AppSetting, Category, Draft, LLMUsage, Transaction, Upload, User

router = APIRouter(tags=["drafts"], dependencies=[Depends(get_current_user)])

log = logging.getLogger("pf.drafts")

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
    "email": (
        "This is an emailed receipt or order confirmation. Extract every purchased line item as a row: "
        "the item's date (fall back to the order date), a short description, the line amount "
        "in euros as a positive number, and the best-fitting category from the allowed values. "
        "Skip totals, shipping, signatures, footers and marketing text."
    ),
}


class DraftOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    status: str
    error: str | None = None
    upload_id: uuid.UUID | None
    # email drafts only: {"from", "date" (ISO 8601), "subject"}; null for photo drafts
    email_meta: dict | None = None
    created_at: datetime
    possible_duplicate: bool = False
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
    user_id: uuid.UUID,
    llm: LLMClient,
    settings: Settings,
    source: str,
    payload: str | tuple[bytes, str],
    *,
    draft: Draft | None = None,
    upload_id: uuid.UUID | None = None,
    email_meta: dict | None = None,
) -> Draft:
    """Extract rows from image bytes or (V6) email text and fill a draft.

    One LLM call → validated rows → draft (status open) + transactions with
    is_draft=true + an llm_usage row. Raises LLMError before anything is written
    if the model output is unusable — the caller decides what that means.
    """
    prompt = EXTRACT_PROMPTS.get(source)
    if prompt is None:
        raise LLMError(f"Unknown draft source {source!r}")
    categories = (await db.scalars(select(Category).order_by(Category.sort_order))).all()
    cat_ids = {c.name.lower(): c.id for c in categories}

    # Routes that ignore response_format still read the prompt — carry the shape there too,
    # with each category's description so the model can match on meaning, not just names.
    menu = " | ".join(f"{c.name} ({c.description})" if c.description else c.name for c in categories)
    shape = (
        "Respond with ONLY a JSON object (no markdown, no prose) shaped "
        '{"rows": [{"date": "YYYY-MM-DD", "description": "short item", "amount": 12.34, '
        '"category": "<best fit: ' + menu + '">}]}'
    )
    contents = [f"{prompt}\n\n{shape}", payload]
    # The owner's own quirks (e.g. what odd items are) go last — they refine the mapping.
    row = await db.get(AppSetting, 1)
    rules = [r.strip() for r in ((row.custom_prompts if row else None) or []) if r.strip()]
    if rules:
        contents[0] += "\n\nOwner's mapping rules (they win when a category fits):\n" + "\n".join(
            f"- {r}" for r in rules
        )
    raw, usage = await llm.complete_structured(rows_schema([c.name for c in categories]), contents)
    rows = _rows_from_llm(raw, cat_ids, cat_ids["misc"])

    draft = draft or Draft(source=source, upload_id=upload_id, email_meta=email_meta)
    draft.raw_llm_output = raw
    draft.status = "open"
    draft.error = None
    if draft not in db:
        db.add(draft)
    await db.flush()  # draft.id for the row FK
    db.add_all(
        Transaction(user_id=user_id, is_draft=True, source=source, draft_id=draft.id, **r)
        for r in rows
    )
    db.add(LLMUsage(
        provider=llm.provider, model=llm.model,
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        cost_usd=await usage_cost(llm.provider, llm.model, usage, settings),
        draft_id=draft.id,
    ))
    await db.commit()
    await db.refresh(draft, ["rows"])  # the collection isn't loaded on a brand-new draft
    await _flag_duplicate(db, user_id, llm, settings, draft)
    return draft


def _fmt_rows(rows) -> str:
    return "\n".join(f"{r.date} | {r.description} | {r.amount}" for r in rows)


async def _flag_duplicate(
    db: AsyncSession, user_id: uuid.UUID, llm: LLMClient, settings: Settings, draft: Draft
) -> None:
    """V10a post-pass: one conditional second LLM call — does the ledger already hold this?

    Zero cost when no approved transactions share the rows' dates. Tolerate-and-continue
    like the email poller: a broken check leaves the draft open and unflagged, never
    fails the already-committed draft.
    """
    try:
        dates = {r.date for r in draft.rows}
        existing = (await db.scalars(select(Transaction).where(
            Transaction.user_id == user_id,
            Transaction.is_draft.is_(False),
            Transaction.date.in_(dates),
        ))).all()
        if not existing:
            return  # nothing on those days to duplicate — no LLM call at all
        raw, usage = await llm.complete_structured(bool_schema(), [(
            "A receipt is about to be added to a personal ledger. Decide whether its "
            "purchases are already recorded there. Answer with ONLY a JSON object "
            '(no markdown, no prose): {"duplicate": true|false} — true only when the '
            "existing rows record the same purchase(s) as the extracted rows, not "
            "merely a same-day same-category coincidence.\n\n"
            f"EXTRACTED:\n{_fmt_rows(draft.rows)}\nEXISTING:\n{_fmt_rows(existing)}"
        )])
        db.add(LLMUsage(
            provider=llm.provider, model=llm.model,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cost_usd=await usage_cost(llm.provider, llm.model, usage, settings),
            draft_id=draft.id,
        ))
        if isinstance(raw, dict) and raw.get("duplicate") is True:
            draft.possible_duplicate = True
        await db.commit()
    except Exception as exc:
        log.warning("draft %s duplicate check skipped: %s", draft.id, exc)


# --- detached extraction: the draft is visible as "processing" while this runs ---


_background: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """Fire-and-forget task, pinned so the loop can't GC it mid-flight."""
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


def get_db_factory(request: Request) -> async_sessionmaker:
    """The session factory detached work must use (overridden in tests)."""
    return request.app.state.db_factory


async def _run_extraction(
    draft_id: int,
    user_id: uuid.UUID,
    llm: LLMClient,
    source: str,
    payload: str | tuple[bytes, str],
    db_factory: async_sessionmaker,
    settings: Settings,
) -> None:
    """Fill a processing draft. Any failure lands in draft.error — nothing is lost."""
    try:
        async with db_factory() as db:
            draft = await db.get(Draft, draft_id)
            await create_draft_from_content(
                db, user_id, llm, settings, source, payload, draft=draft
            )
    except Exception as exc:  # LLMError, storage hiccups, cancelled uploads — everything
        async with db_factory() as db:
            if (draft := await db.get(Draft, draft_id)) is not None:
                draft.status, draft.error = "error", str(exc)[:500]
                await db.commit()


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
    db_factory: async_sessionmaker = Depends(get_db_factory),
):
    """Store the image bytes in the draft now, extract in the background.

    Returns a processing draft immediately — navigating away can't lose it.
    """
    upload = await db.get(Upload, body.upload_id)
    if upload is None:
        raise HTTPException(status_code=404, detail="Not found")
    if not (path := _upload_file(settings, upload)) or not path.exists():
        raise HTTPException(status_code=410, detail="Uploaded file is gone")
    draft = Draft(source="photo", upload_id=upload.id, status="processing")
    db.add(draft)
    await db.commit()
    await db.refresh(draft, ["rows"])  # empty collection for DraftOut
    _spawn(_run_extraction(
        draft.id, user.id, llm, "photo", (path.read_bytes(), upload.mime_type),
        db_factory, settings,
    ))
    return draft


@router.get("/drafts", response_model=list[DraftOut])
async def list_drafts(db: AsyncSession = Depends(get_db)):
    """Open drafts plus the ones still reading / freshly failed, for the list UI."""
    return (
        await db.scalars(
            select(Draft)
            .where(Draft.status.in_(("processing", "open", "error")))
            .order_by(Draft.id.desc())
        )
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
    if draft.status not in ("open", "error"):  # a failed read is discarding-able too
        raise HTTPException(status_code=409, detail=f"Draft is {draft.status}")
    await db.execute(delete(Transaction).where(Transaction.draft_id == draft.id))
    draft.status = "rejected"  # draft + upload rows kept; only the file and its rows go
    _delete_upload_file(settings, draft.upload)
    await db.commit()
