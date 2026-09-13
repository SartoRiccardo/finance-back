"""V6 email ingestion. The transport is faked — no IMAP, no real keys in CI."""

import asyncio
import base64
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.config import Settings
from app.email_ingest import (
    InboundEmail,
    build_payload,
    matches_alias,
    pdf_text,
    poll_once,
    start_email_poller,
)
from app.models import Draft, User
from conftest import TEST_SETTINGS
from test_drafts import GOOD_ROWS, FakeLLM

# A 623-byte one-page PDF whose text layer says "Order confirmation / Total 23.45 EUR".
TINY_PDF = base64.b64decode(
    "JVBERi0xLjQKMSAwIG9iago8PCAvVHlwZSAvQ2F0YWxvZyAvUGFnZXMgMiAwIFIgPj4KZW5kb2JqCjIgMCBvYmoKPDwgL1R5cGUgL1BhZ2VzIC9LaWRzIFszIDAgUl0gL0NvdW50IDEgPj4KZW5kb2JqCjMgMCBvYmoKPDwgL1R5cGUgL1BhZ2UgL1BhcmVudCAyIDAgUiAvTWVkaWFCb3ggWzAgMCA2MTIgNzkyXSAvUmVzb3VyY2VzIDw8IC9Gb250IDw8IC9GMSA1IDAgUiA+PiA+PiAvQ29udGVudHMgNCAwIFIgPj4KZW5kb2JqCjQgMCBvYmoKPDwgL0xlbmd0aCA3OSA+PgpzdHJlYW0KQlQgL0YxIDE0IFRmIDcyIDcyMCBUZCAoT3JkZXIgY29uZmlybWF0aW9uKSBUaiAwIC0yMCBUZCAoVG90YWwgMjMuNDUgRVVSKSBUaiBFVAplbmRzdHJlYW0KZW5kb2JqCjUgMCBvYmoKPDwgL1R5cGUgL0ZvbnQgL1N1YnR5cGUgL1R5cGUxIC9CYXNlRm9udCAvSGVsdmV0aWNhID4+CmVuZG9iagp4cmVmCjAgNgowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMDkgMDAwMDAgbiAKMDAwMDAwMDA1OCAwMDAwMCBuIAowMDAwMDAwMTE1IDAwMDAwIG4gCjAwMDAwMDAyNDEgMDAwMDAgbiAKMDAwMDAwMDM3MCAwMDAwMCBuIAp0cmFpbGVyCjw8IC9TaXplIDYgL1Jvb3QgMSAwIFIgPj4Kc3RhcnR4cmVmCjQ0MAolJUVPRgo="
)

INGEST_OFF = {  # feature inert by default in tests, whatever the developer's .env says
    "ingest_imap_host": "", "ingest_imap_port": 993, "ingest_imap_user": "",
    "ingest_imap_password": "", "ingest_mail_alias": "me+finance@test.dev",
    "ingest_poll_seconds": 60,
}

RECEIPT_HTML = """<html><body><h2>Order confirmation</h2>
<p>See <a href="https://shop.test/order/1">your order</a>.</p>
<table><tr><th>Item</th><th>Price</th></tr><tr><td>Yarn</td><td>23.45 EUR</td></tr></table>
<ul><li>outer<ul><li>inner</li></ul></li></ul>
</body></html>"""


def an_email(**over) -> InboundEmail:
    defaults = dict(
        message_id="uid-1", sender="me@test.dev", recipients=["me+finance@test.dev"],
        date=datetime(2026, 9, 12, 10, 30, tzinfo=timezone.utc), subject="Your order",
        html=RECEIPT_HTML, text=None, attachments=[],
    )
    return InboundEmail(**(defaults | over))


class FakeSource:
    """In-memory transport; records \\Seen marks."""

    def __init__(self, emails):
        self.emails, self.seen = list(emails), []

    async def fetch_new(self):
        return list(self.emails)

    async def mark_seen(self, message_id):
        self.seen.append(message_id)


async def add_user(db, addr="me@test.dev"):
    async with db() as s:
        s.add(User(email=addr))
        await s.commit()


async def poll(db, source, llm, monkeypatch, **settings_over):
    async def fake_llm(db_, settings_):
        return llm

    monkeypatch.setattr("app.email_ingest.get_llm", fake_llm)
    await poll_once(Settings(**(TEST_SETTINGS | INGEST_OFF | settings_over)), db, source)


@pytest.mark.anyio
async def test_markdownify_survives_links_tables_nested_lists():
    md = build_payload(an_email())
    assert "[your order](https://shop.test/order/1)" in md
    assert "| Item | Price |" in md and "| Yarn | 23.45 EUR |" in md
    lines = md.splitlines()
    inner = next(i for i, l in enumerate(lines) if l.strip() == "+ inner")
    assert lines[inner].startswith("  ")  # nested list survives indented


@pytest.mark.anyio
async def test_email_becomes_draft_via_shared_pipeline(client, db, monkeypatch):
    await client.get("/api/auth/dev-login")
    await add_user(db)
    fake = FakeLLM(GOOD_ROWS)
    source = FakeSource([an_email()])
    await poll(db, source, fake, monkeypatch)

    assert source.seen == ["uid-1"]  # draft committed → and only now marked
    payload = fake.calls[0][1][1]  # (prompt, payload) → the email text
    assert "From: me@test.dev" in payload and "Subject: Your order" in payload
    assert "| Yarn | 23.45 EUR |" in payload  # markdownified body, not HTML

    listing = (await client.get("/api/drafts")).json()
    assert len(listing) == 1
    draft = listing[0]
    assert draft["source"] == "email" and draft["status"] == "open"
    assert draft["email_meta"] == {
        "from": "me@test.dev", "date": "2026-09-12T10:30:00+00:00", "subject": "Your order",
    }
    assert [r["description"] for r in draft["rows"]] == ["Coop run", "Moon rocks"]

    async with db() as s:
        stored = (await s.scalars(select(Draft))).one()
        assert stored.raw_llm_output == GOOD_ROWS  # stored alongside email_meta
        assert stored.email_meta == draft["email_meta"]

    # a photo draft keeps the contract's null
    photo = Draft(source="photo", status="open")
    async with db() as s:
        s.add(photo)
        await s.commit()
    detail = (await client.get(f"/api/drafts/{photo.id}")).json()
    assert detail["email_meta"] is None


@pytest.mark.anyio
async def test_non_whitelisted_sender_skipped_silently(db, monkeypatch):
    source = FakeSource([an_email(sender="stranger@evil.dev")])
    await poll(db, source, FakeLLM(GOOD_ROWS), monkeypatch)

    assert source.seen == []  # no draft → no \Seen → it re-polls for free
    async with db() as s:
        assert (await s.scalars(select(Draft))).all() == []


@pytest.mark.anyio
async def test_seen_only_after_draft_commits(db, monkeypatch):
    await add_user(db)
    source = FakeSource([an_email()])
    await poll(db, source, FakeLLM(error="Gemini call failed: 429"), monkeypatch)
    assert source.seen == []  # failed extraction → retried next poll
    async with db() as s:
        assert (await s.scalars(select(Draft))).all() == []

    await poll(db, source, FakeLLM(GOOD_ROWS), monkeypatch)  # same message again
    assert source.seen == ["uid-1"]


@pytest.mark.anyio
async def test_malformed_content_skips_cleanly(db, monkeypatch):
    fake = FakeLLM(GOOD_ROWS)
    source = FakeSource([an_email(html=None, text=None, attachments=[])])
    await poll(db, source, fake, monkeypatch)  # no raise

    assert fake.calls == []  # nothing extractable → no LLM call at all
    assert source.seen == []
    async with db() as s:
        assert (await s.scalars(select(Draft))).all() == []


@pytest.mark.anyio
async def test_attachment_text_lands_in_payload_under_separator(db, monkeypatch):
    scanned_pdf, good_pdf = b"image-only", b"real text layer"
    big_pdf = b"x" * (10 * 1024 * 1024 + 1)
    calls = []

    def fake_pdf_text(data):
        calls.append(data)
        return "" if data == scanned_pdf else "Item total 12.00 EUR"

    monkeypatch.setattr("app.email_ingest.pdf_text", fake_pdf_text)
    source_email = an_email(text="body text", attachments=[
        ("receipt.pdf", "application/pdf", good_pdf),
        ("notes.txt", "text/plain", b"ignore me"),
        ("huge.pdf", "application/pdf", big_pdf),  # >10MB → never extracted
        ("scan.pdf", "application/pdf", scanned_pdf),  # no OCR → nothing → skipped
    ])
    payload = build_payload(source_email)

    assert "--- Attachment: receipt.pdf ---" in payload
    assert "Item total 12.00 EUR" in payload
    assert "notes.txt" not in payload and "ignore me" not in payload
    assert "huge.pdf" not in payload and "scan.pdf" not in payload
    assert big_pdf not in calls  # size gate runs before extraction
    # combined attachment text is capped at 50k chars
    monkeypatch.setattr("app.email_ingest.pdf_text", lambda data: "y" * 60_000)
    payload = build_payload(an_email(html=None, text="b", attachments=[("a.pdf", "application/pdf", b"d")]))
    assert payload.count("y") == 50_000


@pytest.mark.anyio
async def test_real_pdf_extracts_through_pypdf_wrapper():
    text = pdf_text(TINY_PDF)
    assert "Order confirmation" in text and "23.45" in text
    assert pdf_text(b"not a pdf") == ""  # corrupted → nothing extractable, no raise


def test_alias_matching():
    alias = "me+finance@test.dev"
    assert matches_alias(["me+finance@test.dev"], alias)
    assert matches_alias(["a@x.dev"], "me+finance@test.dev") is False
    assert matches_alias(["Me+Finance@Test.dev"], alias)  # case-insensitive
    assert matches_alias(["someone else@x.dev"], alias) is False
    assert matches_alias(["me@x.dev", "me+finance@test.dev"], alias)


@pytest.mark.anyio
async def test_poller_disabled_without_host():
    settings = Settings(**(TEST_SETTINGS | INGEST_OFF))
    assert start_email_poller(settings) is None  # fully inert: no task, no errors

    settings = Settings(**(TEST_SETTINGS | INGEST_OFF | {"ingest_imap_host": "imap.test.dev"}))
    task = start_email_poller(settings)
    assert isinstance(task, asyncio.Task)
    task.cancel()  # never ran a poll — no network in this test
