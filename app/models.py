import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Uuid,
    func,
    insert,
    inspect,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str | None]
    picture_url: Mapped[str | None]
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Category(TimestampMixin, Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str | None]
    is_investment: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    sort_order: Mapped[int] = mapped_column(default=0, server_default="0")
    color: Mapped[str | None] = mapped_column(String(7))


class Label(TimestampMixin, Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    is_spending: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    color: Mapped[str | None] = mapped_column(String(7))


class Transaction(TimestampMixin, Base):
    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("amount > 0", name="ck_transactions_amount_positive"),
        CheckConstraint(
            "(direction = 'spend' AND category_id IS NOT NULL AND label_id IS NULL) OR "
            "(direction = 'earn' AND category_id IS NULL AND label_id IS NOT NULL)",
            name="ck_transactions_direction",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    date: Mapped[date] = mapped_column(Date)
    description: Mapped[str] = mapped_column(String(500))
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    direction: Mapped[str] = mapped_column(String(8))
    category_id: Mapped[int | None] = mapped_column(ForeignKey("categories.id"))
    label_id: Mapped[int | None] = mapped_column(ForeignKey("labels.id"))
    # Draft columns are filled by later sprints (V5/V6); unused until then.
    is_draft: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    draft_id: Mapped[int | None] = mapped_column(ForeignKey("drafts.id"))
    source: Mapped[str] = mapped_column(String(8), default="manual", server_default="manual")

    category: Mapped[Category | None] = relationship(lazy="selectin")
    label: Mapped[Label | None] = relationship(lazy="selectin")


# jsonb on Postgres (spec), plain JSON on the sqlite test DB.
JsonDict = JSON().with_variant(JSONB(), "postgresql")


class Upload(TimestampMixin, Base):
    __tablename__ = "uploads"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    stored_path: Mapped[str] = mapped_column(String(200))
    original_name: Mapped[str | None]
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int]


class Draft(TimestampMixin, Base):
    __tablename__ = "drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(8), default="photo", server_default="photo")
    upload_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("uploads.id"))
    # email_meta / source='email' are V6's; nullable until then.
    email_meta: Mapped[dict | None] = mapped_column(JsonDict)
    raw_llm_output: Mapped[dict | None] = mapped_column(JsonDict)
    # processing → open|error (extraction runs detached from the request)
    status: Mapped[str] = mapped_column(String(16), default="open", server_default="open")
    error: Mapped[str | None] = mapped_column(String(500))

    upload: Mapped[Upload | None] = relationship(lazy="selectin")
    rows: Mapped[list[Transaction]] = relationship(foreign_keys="Transaction.draft_id", lazy="selectin")


class LLMUsage(TimestampMixin, Base):
    """One row per extraction call — the ironic cost of our readings."""

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(120))
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    # USD; computed from picker prices at write time (None when no price is known)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    draft_id: Mapped[int | None] = mapped_column(ForeignKey("drafts.id"))


class AppSetting(Base):
    """Single row (id=1) holding runtime-flippable config; created from env defaults on startup."""

    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    llm_provider: Mapped[str] = mapped_column(String(16), default="google", server_default="google")
    llm_model: Mapped[str] = mapped_column(String(120), default="gemini-2.5-flash", server_default="gemini-2.5-flash")
    # free-text additions appended to the extraction prompt (e.g. where odd items belong)
    custom_prompt: Mapped[str | None] = mapped_column(String(2000))


SEED_CATEGORIES = [
    ("Education", "Tech & singing, personal learning", False),
    ("Sewing", "Fabric, patterns, tools", False),
    ("Chinese Plushies", "smallplushies.com stock", True),
    ("Investment", "ETFs and similar", True),
    ("Girlfriend", "Gifts, dates", False),
    ("Social life", "Going out with friends", False),
    ("Ingredients", "Groceries and cooking", False),
    ("Takeout", "Restaurants and delivery", False),
    ("Self Care", "Health, grooming, therapy", False),
    ("Transport", "Trains, fuel, tickets", False),
    ("Misc", "Catch-all", False),
]
SEED_LABELS = ["tip", "fumofumo", "payback", "salary"]

# Preset hex colors (same palette as front/src/lib/palette.ts) — the single
# source for both fresh seeds and the 0003 backfill on existing databases.
CATEGORY_COLORS = {
    "Education": "#3b82f6",  # blue
    "Sewing": "#8b5cf6",  # violet
    "Chinese Plushies": "#ec4899",  # pink
    "Investment": "#10b981",  # emerald
    "Girlfriend": "#f43f5e",  # rose
    "Social life": "#f59e0b",  # amber
    "Ingredients": "#84cc16",  # lime
    "Takeout": "#f97316",  # orange
    "Self Care": "#14b8a6",  # teal
    "Transport": "#0ea5e9",  # sky
    "Misc": "#6b7280",  # gray
}
LABEL_COLORS = {
    "tip": "#eab308",  # yellow
    "fumofumo": "#d946ef",  # fuchsia
    "payback": "#06b6d4",  # cyan
    "salary": "#22c55e",  # green
}


def seed(bind) -> None:
    """Insert seed rows. bind is a Connection (migrations) or Session (tests).

    Columns the physical table doesn't have yet are skipped — migration 0002
    seeds before 0003 adds `color`; 0003 backfills it afterwards.
    """

    def rows(table, data):
        cols = {c["name"] for c in inspect(bind).get_columns(table)}
        return [{k: v for k, v in row.items() if k in cols} for row in data]

    bind.execute(
        insert(Category),
        rows(
            "categories",
            [
                {
                    "name": n,
                    "description": d,
                    "is_investment": inv,
                    "sort_order": i,
                    "color": CATEGORY_COLORS.get(n),
                }
                for i, (n, d, inv) in enumerate(SEED_CATEGORIES)
            ],
        ),
    )
    bind.execute(
        insert(Label),
        rows(
            "labels",
            [{"name": n, "is_spending": False, "color": LABEL_COLORS.get(n)} for n in SEED_LABELS],
        ),
    )