import datetime
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.db import get_db
from app.models import Category, Label, Transaction, User

router = APIRouter(tags=["ledger"], dependencies=[Depends(get_current_user)])


class CategoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    is_investment: bool
    sort_order: int
    color: str | None


class LabelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    is_spending: bool
    color: str | None


class CategoryIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = None
    is_investment: bool = False
    sort_order: int = 0
    color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")


class CategoryPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    description: str | None = None
    is_investment: bool | None = None
    sort_order: int | None = None
    color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")


class LabelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    is_spending: bool = False
    color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")


class LabelPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")


class TxnIn(BaseModel):
    date: datetime.date
    description: str = Field(min_length=1, max_length=500)
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    direction: Literal["spend", "earn"]
    category_id: int | None = None
    label_id: int | None = None

    @model_validator(mode="after")
    def _direction(self):
        spend = self.direction == "spend" and self.category_id is not None and self.label_id is None
        earn = self.direction == "earn" and self.label_id is not None and self.category_id is None
        if not (spend or earn):
            raise ValueError(
                "spend requires category_id and no label_id; earn requires label_id and no category_id"
            )
        return self


class TxnPatch(BaseModel):
    date: datetime.date | None = None
    description: str | None = Field(None, min_length=1, max_length=500)
    amount: Decimal | None = Field(None, gt=0, max_digits=12, decimal_places=2)
    direction: Literal["spend", "earn"] | None = None
    category_id: int | None = None
    label_id: int | None = None


class TxnOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    date: datetime.date
    description: str
    amount: Decimal
    direction: str
    category_id: int | None
    label_id: int | None
    category: CategoryOut | None
    label: LabelOut | None


class TxnPage(BaseModel):
    items: list[TxnOut]
    total: int


async def _get_or_404(db: AsyncSession, model, id: int):
    row = await db.get(model, id)
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")
    return row


async def _check_refs(db: AsyncSession, body: TxnIn) -> None:
    if body.category_id is not None and not await db.get(Category, body.category_id):
        raise HTTPException(status_code=422, detail="category_id does not exist")
    if body.label_id is not None and not await db.get(Label, body.label_id):
        raise HTTPException(status_code=422, detail="label_id does not exist")


# --- categories ---


@router.get("/categories", response_model=list[CategoryOut])
async def list_categories(db: AsyncSession = Depends(get_db)):
    return (
        await db.scalars(select(Category).order_by(Category.sort_order, Category.id))
    ).all()


@router.post("/categories", status_code=201, response_model=CategoryOut)
async def create_category(body: CategoryIn, db: AsyncSession = Depends(get_db)):
    if await db.scalar(select(Category).where(Category.name == body.name)):
        raise HTTPException(status_code=409, detail="Category name already exists")
    cat = Category(**body.model_dump())
    db.add(cat)
    await db.commit()
    await db.refresh(cat)
    return cat


@router.patch("/categories/{category_id}", response_model=CategoryOut)
async def update_category(
    category_id: int, body: CategoryPatch, db: AsyncSession = Depends(get_db)
):
    cat = await _get_or_404(db, Category, category_id)
    changes = body.model_dump(exclude_unset=True)
    if (name := changes.get("name")) and name != cat.name:
        if await db.scalar(select(Category).where(Category.name == name)):
            raise HTTPException(status_code=409, detail="Category name already exists")
    for key, value in changes.items():
        setattr(cat, key, value)
    await db.commit()
    await db.refresh(cat)
    return cat


@router.delete("/categories/{category_id}", status_code=204)
async def delete_category(category_id: int, db: AsyncSession = Depends(get_db)):
    cat = await _get_or_404(db, Category, category_id)
    used = await db.scalar(
        select(func.count()).select_from(Transaction).where(Transaction.category_id == category_id)
    )
    if used:
        raise HTTPException(status_code=409, detail="Category is used by transactions")
    await db.delete(cat)
    await db.commit()


# --- labels ---


@router.get("/labels", response_model=list[LabelOut])
async def list_labels(db: AsyncSession = Depends(get_db)):
    return (await db.scalars(select(Label).order_by(Label.name))).all()


@router.post("/labels", status_code=201, response_model=LabelOut)
async def create_label(body: LabelCreate, db: AsyncSession = Depends(get_db)):
    if await db.scalar(select(Label).where(Label.name == body.name)):
        raise HTTPException(status_code=409, detail=f"Label '{body.name}' already exists")
    label = Label(**body.model_dump())
    db.add(label)
    await db.commit()
    await db.refresh(label)
    return label


@router.patch("/labels/{label_id}", response_model=LabelOut)
async def update_label(label_id: int, body: LabelPatch, db: AsyncSession = Depends(get_db)):
    label = await _get_or_404(db, Label, label_id)
    changes = body.model_dump(exclude_unset=True)
    if (name := changes.get("name")) and name != label.name:
        if await db.scalar(select(Label).where(Label.name == name)):
            raise HTTPException(status_code=409, detail=f"Label '{name}' already exists")
    for key, value in changes.items():
        setattr(label, key, value)
    await db.commit()
    await db.refresh(label)
    return label


@router.delete("/labels/{label_id}", status_code=204)
async def delete_label(label_id: int, db: AsyncSession = Depends(get_db)):
    label = await _get_or_404(db, Label, label_id)
    used = await db.scalar(
        select(func.count()).select_from(Transaction).where(Transaction.label_id == label_id)
    )
    if used:
        raise HTTPException(status_code=409, detail=f"Label in use by {used} transactions")
    await db.delete(label)
    await db.commit()


# --- transactions ---


@router.get("/transactions", response_model=TxnPage)
async def list_transactions(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    date_from: datetime.date | None = Query(None, alias="from"),
    date_to: datetime.date | None = Query(None, alias="to"),
    category_id: int | None = None,
    direction: Literal["spend", "earn"] | None = None,
    q: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    stmt = select(Transaction).where(
        Transaction.user_id == user.id, Transaction.is_draft.is_(False)
    )
    if date_from:
        stmt = stmt.where(Transaction.date >= date_from)
    if date_to:
        stmt = stmt.where(Transaction.date <= date_to)
    if category_id is not None:
        stmt = stmt.where(Transaction.category_id == category_id)
    if direction:
        stmt = stmt.where(Transaction.direction == direction)
    if q:
        stmt = stmt.where(Transaction.description.ilike(f"%{q}%"))

    total = await db.scalar(select(func.count()).select_from(stmt.subquery()))
    items = (
        await db.scalars(
            stmt.order_by(Transaction.date.desc(), Transaction.id.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return {"items": items, "total": total}


@router.post("/transactions", status_code=201, response_model=TxnOut)
async def create_transaction(
    body: TxnIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _check_refs(db, body)
    txn = Transaction(user_id=user.id, **body.model_dump())
    db.add(txn)
    await db.commit()
    await db.refresh(txn)
    return txn


@router.patch("/transactions/{transaction_id}", response_model=TxnOut)
async def update_transaction(
    transaction_id: int,
    body: TxnPatch,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    txn = await _get_or_404(db, Transaction, transaction_id)
    if txn.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    changes = body.model_dump(exclude_unset=True)
    try:  # re-validate the merged row against the direction constraints
        merged = TxnIn(**{k: getattr(txn, k) for k in TxnIn.model_fields} | changes)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_context=False, include_input=False),
        ) from None
    await _check_refs(db, merged)
    for key, value in changes.items():
        setattr(txn, key, value)
    await db.commit()
    await db.refresh(txn)
    return txn


@router.delete("/transactions/{transaction_id}", status_code=204)
async def delete_transaction(
    transaction_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    txn = await _get_or_404(db, Transaction, transaction_id)
    if txn.user_id != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(txn)
    await db.commit()
