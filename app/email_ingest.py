"""V6 email ingestion: receipts emailed to the +finance alias become drafts.

The poller runs in-process (started by main's lifespan only when INGEST_IMAP_HOST
is set — unset means fully inert) and pushes every alias-matching message
through the same `create_draft_from_content` pipeline photo drafts use. Dedupe
is unseen + \\Seen-marking: a message is marked read only after its draft is
committed, so a crash before that point replays it on the next poll.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from typing import Protocol

from imap_tools import AND, MailBox, MailMessageFlags
from markdownify import markdownify
from sqlalchemy import func, select

from app.config import Settings
from app.db import SessionLocal
from app.drafts import MAX_UPLOAD_BYTES, create_draft_from_content
from app.llm import LLMClient, get_llm
from app.models import User

log = logging.getLogger("pf.email")

MAX_ATTACHMENT_CHARS = 50_000  # combined attachment text fed to the LLM


class EmailIngestionSource(Protocol):
    """The whole transport interface — a webhook impl later just implements fetch_new."""

    async def fetch_new(self) -> list["InboundEmail"]: ...


@dataclass
class InboundEmail:
    message_id: str
    sender: str
    recipients: list[str]
    date: datetime
    subject: str
    html: str | None = None
    text: str | None = None
    # (filename, content_type, data) — beyond the specced fields because the payload
    # builder needs the raw parts for PDF extraction; a webhook transport can carry them just the same.
    attachments: list[tuple[str, str, bytes]] = field(default_factory=list)


def pdf_text(data: bytes) -> str:
    """All extractable text of one PDF. No OCR — an image-only (scanned) PDF yields ''."""
    from pypdf import PdfReader  # lazy, like the LLM SDKs

    try:
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
    except Exception:  # corrupted PDF → nothing extractable, same as scanned
        return ""


def is_pdf(name: str, content_type: str) -> bool:
    return content_type.startswith("application/pdf") or name.lower().endswith(".pdf")


def matches_alias(addrs: list[str], alias: str) -> bool:
    """True when To/Cc/Delivered-To contains the alias (e.g. me+finance@gmail.com)."""
    needle = alias.lower()
    return bool(needle) and any(needle in a.lower() for a in addrs if a)


def build_payload(email: InboundEmail) -> str | None:
    """Metadata + markdown body + extracted attachment text — the one str the LLM reads.

    None = nothing extractable (no body, no usable attachments): skip without an LLM call.
    Non-PDF and >10MB attachments ignored; image-only PDFs yield no text and are skipped.
    """
    body = (email.text or "").strip()
    if email.html:
        body = markdownify(email.html).strip() or body
    parts = [f"From: {email.sender}", f"Date: {email.date.isoformat()}", f"Subject: {email.subject}"]
    budget = MAX_ATTACHMENT_CHARS
    for name, ctype, data in email.attachments:
        if budget <= 0:
            break
        if not is_pdf(name, ctype) or len(data) > MAX_UPLOAD_BYTES:
            continue
        text = pdf_text(data).strip()[:budget]
        if not text:
            continue
        budget -= len(text)
        parts += ["", f"--- Attachment: {name or 'receipt.pdf'} ---", text]
    if not body and len(parts) == 3:
        return None
    return "\n".join([*parts, "", body])


class ImapIngestionSource:
    """IMAP via app password (Gmail). Opens/closes a connection per operation —
    no idle-connection state to go stale; a dead server just logs and retries next poll."""

    def __init__(self, settings: Settings):
        self.s = settings

    def _login(self) -> MailBox:
        return MailBox(self.s.ingest_imap_host, self.s.ingest_imap_port).login(
            self.s.ingest_imap_user, self.s.ingest_imap_password, initial_folder="INBOX"
        )

    async def fetch_new(self) -> list[InboundEmail]:
        return await asyncio.to_thread(self._fetch_sync)  # sync imap-tools off the loop

    def _fetch_sync(self) -> list[InboundEmail]:
        alias = self.s.ingest_mail_alias
        out = []
        with self._login() as box:
            # mark_seen=False: \Seen is ours to set, only after the draft commits
            for msg in box.fetch(AND(seen=False), mark_seen=False):
                addrs = [*(msg.to or []), *(msg.cc or []), *msg.headers_values("delivered-to")]
                if not matches_alias(addrs, alias):
                    continue  # not addressed to the alias — never touched, stays unread
                out.append(InboundEmail(
                    message_id=msg.uid,
                    sender=msg.from_ or "",
                    recipients=[*(msg.to or []), *(msg.cc or [])],
                    date=msg.date or datetime.now(UTC),
                    subject=msg.subject or "",
                    html=msg.html or None,
                    text=msg.text or None,
                    attachments=[
                        (a.filename or "", a.content_type or "", a.payload) for a in msg.attachments
                    ],
                ))
        return out

    async def mark_seen(self, message_id: str) -> None:
        await asyncio.to_thread(self._mark_seen_sync, message_id)

    def _mark_seen_sync(self, uid: str) -> None:
        with self._login() as box:
            box.flag(uid, MailMessageFlags.SEEN, True)


async def process_email(
    settings: Settings, db_factory, llm: LLMClient, email: InboundEmail
):
    """One message through the shared pipeline. Returns the Draft, or None to skip.

    Skip (no draft, hence no \\Seen — re-polling a skip is free): sender is not a
    whitelisted user's email, or the message has nothing extractable.
    """
    payload = build_payload(email)
    if payload is None:
        return None
    async with db_factory() as db:
        user = (await db.scalars(
            select(User).where(func.lower(User.email) == email.sender.lower())
        )).first()
        if user is None:
            return None
        draft = await create_draft_from_content(
            db, user.id, llm, settings, "email", payload,
            email_meta={  # the pinned response contract: from/date(ISO)/subject
                "from": email.sender,
                "date": email.date.isoformat(),
                "subject": email.subject,
            },
        )
    log.info("email %s → draft %s (%d rows)", email.message_id, draft.id, len(draft.rows))
    return draft


async def poll_once(settings: Settings, db_factory, source: EmailIngestionSource) -> None:
    """One pass: fetch → whitelist → shared pipeline → mark seen on success."""
    try:
        emails = await source.fetch_new()
    except Exception as exc:
        log.warning("email poll failed: %s", exc)
        return
    if not emails:
        return
    async with db_factory() as db:
        try:
            llm = await get_llm(db, settings)
        except Exception as exc:
            log.warning("email poll skipped, no usable LLM: %s", exc)
            return
    for email in emails:
        try:
            draft = await process_email(settings, db_factory, llm, email)
        except Exception as exc:
            # stays UNSEEN — the next poll retries it (crash-safe by construction)
            log.warning("email %s failed, will retry next poll: %s", email.message_id, exc)
            continue
        if draft is not None and (mark := getattr(source, "mark_seen", None)):
            await mark(email.message_id)  # only here: the draft is committed


async def poll_forever(settings: Settings, db_factory) -> None:
    source = ImapIngestionSource(settings)
    while True:
        await poll_once(settings, db_factory, source)
        await asyncio.sleep(settings.ingest_poll_seconds)


def start_email_poller(settings: Settings, db_factory=SessionLocal) -> asyncio.Task | None:
    """The in-process poller task. INGEST_IMAP_HOST unset ⇒ None: no task, no errors."""
    if not settings.ingest_imap_host:
        return None
    log.info("email ingestion polling %s every %ss", settings.ingest_imap_host, settings.ingest_poll_seconds)
    return asyncio.create_task(poll_forever(settings, db_factory))
