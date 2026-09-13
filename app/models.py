import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
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
)
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


class Label(TimestampMixin, Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    is_spending: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


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
    draft_id: Mapped[int | None] = mapped_column(ForeignKey("transactions.id"))
    source: Mapped[str] = mapped_column(String(8), default="manual", server_default="manual")

    category: Mapped[Category | None] = relationship(lazy="selectin")
    label: Mapped[Label | None] = relationship(lazy="selectin")


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


def seed(bind) -> None:
    """Insert seed rows. bind is a Connection (migrations) or Session (tests)."""
    bind.execute(
        insert(Category),
        [
            {"name": n, "description": d, "is_investment": inv, "sort_order": i}
            for i, (n, d, inv) in enumerate(SEED_CATEGORIES)
        ],
    )
    bind.execute(insert(Label), [{"name": n, "is_spending": False} for n in SEED_LABELS])