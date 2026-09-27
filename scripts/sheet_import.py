"""Generate a Postgres SQL file that bulk-imports the Google-Sheets TSVs.

Usage: python3 sheet_import.py ~/Desktop/spending.csv ~/Desktop/entrate.csv ~/Desktop/import-pf.sql

Sheet format (tab-separated, Italian locale): dates like "01 ott 2026",
amounts like "€ 1.050,00". Spending rows need a category (Type), earning rows
a label (Fonte) — mapped to the app's taxonomy below; anything unmapped aborts.
Rows with an empty amount are sheet placeholders and are skipped.

The SQL is safe to run before or after the first Google login: it resolves the
user by email and refuses to run if they don't exist yet.
"""

import csv
import re
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

USER_EMAIL = "rikki.sarto@gmail.com"

# sheet Type (spend) → app category name
CATEGORIES = {
    "Cibo:Ingredienti": "Ingredients",
    "Cibo:Takeout": "Takeout",
    "Cucito": "Sewing",
    "Investimento": "Investment",
    "Istruzione": "Education",
    "Misc": "Misc",
    "Plushie Cinesi": "Chinese Plushies",
    "Self Care": "Self Care",
    "Socialità": "Social life",
    "Trasporto": "Transport",
}
# sheet Fonte (earn) → app label name
LABELS = {"Stipendio": "salary", "Rimborso": "payback", "Mancia": "tip"}

MONTHS = dict(zip(
    "gen feb mar apr mag giu lug ago set ott nov dic".split(), range(1, 13)
))
AMOUNT_RE = re.compile(r"[\d.,]+")  # strips '€', spaces (incl. nbsp) — leaves 1.050,00


def parse_date(text: str) -> date:
    day, mon, year = text.strip().split()
    return date(int(year), MONTHS[mon[:3]], int(day))


def parse_amount(text: str) -> Decimal:
    raw = AMOUNT_RE.search(text.replace("\xa0", " ")).group()
    return Decimal(raw.replace(".", "").replace(",", "."))


def sql_str(text: str) -> str:
    return "'" + text.strip()[:500].replace("'", "''") + "'"


def read_rows(path: Path, direction: str) -> list[tuple]:
    rows = list(csv.reader(open(path, encoding="utf-8"), delimiter="\t"))
    header, data = rows[0], rows[1:]
    amount_col, kind_col = ("Costo", "Type") if direction == "spend" else ("Guadagno", "Fonte")
    ai, ki = header.index(amount_col), header.index(kind_col)
    mapping = CATEGORIES if direction == "spend" else LABELS
    out, skipped = [], 0
    for r in data:
        if not r[ai].strip():
            skipped += 1  # sheet placeholder (future salary etc.)
            continue
        kind = r[ki].strip()
        if kind not in mapping:
            sys.exit(f"unmapped {kind_col} {kind!r} in {path.name} — add it to the mapping")
        out.append((parse_date(r[0]), r[1], parse_amount(r[ai]), mapping[kind]))
    print(f"{path.name}: {len(out)} rows imported, {skipped} empty-amount placeholders skipped")
    return out


def insert(rows: list[tuple], direction: str) -> str:
    """One multi-row INSERT; the kind name resolves to a category (spend) or label (earn) id."""
    kind_table = "categories" if direction == "spend" else "labels"
    other = "NULL" if direction == "spend" else "NULL"
    lines = []
    for d, desc, amt, kind in rows:
        kind_id = f"(SELECT id FROM {kind_table} WHERE name = {sql_str(kind)})"
        lines.append(
            f"({user}, '{d.isoformat()}', {sql_str(desc)}, {amt}, '{direction}', "
            + (f"{kind_id}, NULL)" if direction == "spend" else f"NULL, {kind_id})")
        )
    return ",\n".join(lines)


spends = read_rows(Path(sys.argv[1]), "spend")
earns = read_rows(Path(sys.argv[2]), "earn")
out_path = Path(sys.argv[3])

user = f"(SELECT id FROM users WHERE email = '{USER_EMAIL}')"

sql = f"""-- Bulk import from the Google Sheets export ({date.today().isoformat()})
-- spending.csv: {len(spends)} rows, entrate.csv: {len(earns)} rows (empty-amount placeholders skipped)
-- Run inside the app's DB: docker compose exec -T db psql -U pf -d pf < import-pf.sql

BEGIN;

-- the user row only exists after the first Google login — fail loudly, import nothing
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM users WHERE email = '{USER_EMAIL}') THEN
    RAISE EXCEPTION 'No user {USER_EMAIL} yet — log in to the app once, then re-run';
  END IF;
END $$;

INSERT INTO transactions (user_id, date, description, amount, direction, category_id, label_id)
VALUES
{insert(spends, "spend")};

INSERT INTO transactions (user_id, date, description, amount, direction, category_id, label_id)
VALUES
{insert(earns, "earn")};

COMMIT;
"""

out_path.write_text(sql, encoding="utf-8")
print(f"{out_path}: {len(spends)} spends = €{sum(r[2] for r in spends)}, "
      f"{len(earns)} earns = €{sum(r[2] for r in earns)}")
