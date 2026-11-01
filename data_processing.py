"""
utils/data_processing.py
========================

Everything between "user drops an Excel file" and "clean ledger ready for
reporting":

    read_upload()          -> raw sheets (header row auto-detected)
    auto_map_columns()     -> {standard_field: source_column}  (+ saved mappings)
    standardise_frame()    -> canonical long-format TB rows
    build_ledger()         -> per account x period: movement, closing, opening
    validate_data()        -> list of issues (unbalanced TB, unclassified, gaps ...)
    generate_sample_files()-> demo workbooks for testing

Canonical (standardised) columns
--------------------------------
    company, cost_centre, year, period, account_code, account_name,
    debit, credit, net, opening_balance, source

    * net  = debit - credit   (debit positive / credit negative)
    * year = FISCAL year label (the calendar year in which the fiscal year ENDS)
    * period = fiscal period 1..12 (period 1 = first month of the fiscal year)

TB basis
--------
    "closing"  : amounts are cumulative balances at period end
                 (standard ERP trial balance; P&L accounts are year-to-date)
    "movement" : amounts are activity for the period only
                 (optionally with an Opening Balance column)

Ledger columns (output of build_ledger)
---------------------------------------
    company, cost_centre, year, period, pkey (= year*100+period),
    account_code, account_name, category, sub_category, is_pl,
    movement (period activity), closing (balance at period end; YTD for P&L),
    opening (= closing - movement), debit, credit (period columns as supplied)
"""
from __future__ import annotations

import calendar
import difflib
import hashlib
import io
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from account_classifier import CATEGORY_OF, PL_CATEGORIES, UNCLASSIFIED

CONFIG_DIR = Path(__file__).resolve().parent / "config"
MAPPINGS_FILE = CONFIG_DIR / "column_mappings.json"
SETTINGS_FILE = CONFIG_DIR / "settings.json"

NOT_MAPPED = "— not in file —"
NO_CC = "(none)"

# --------------------------------------------------------------------------- #
# Standard fields and the header aliases used for auto-detection
# --------------------------------------------------------------------------- #
FIELD_DEFS: dict[str, dict] = {
    "account_code": {"label": "Account Code", "required": True, "aliases": [
        "account code", "acc code", "a c code", "ac code", "gl code", "gl account", "account no", "account number",
        "acct no", "acct code", "code", "ledger code", "gl", "account id", "acc no", "account", "gl no",
        "nominal code", "account num"]},
    "account_name": {"label": "Account Name", "required": False, "aliases": [
        "account name", "account description", "description", "gl name", "ledger name", "name",
        "account title", "particulars", "gl description", "acct name", "nominal name", "ledger"]},
    "debit": {"label": "Debit", "required": False, "aliases": [
        "debit", "dr", "debit amount", "closing debit", "debit balance", "dr amount", "debits",
        "debit kes", "debit ksh", "dr balance", "period debit"]},
    "credit": {"label": "Credit", "required": False, "aliases": [
        "credit", "cr", "credit amount", "closing credit", "credit balance", "cr amount", "credits",
        "credit kes", "credit ksh", "cr balance", "period credit"]},
    "balance": {"label": "Net Balance (Dr + / Cr -)", "required": False, "aliases": [
        "balance", "net balance", "closing balance", "amount", "net amount", "ending balance",
        "balance dr cr", "net", "closing", "balance kes", "movement", "net movement"]},
    "opening_balance": {"label": "Opening Balance", "required": False, "aliases": [
        "opening balance", "opening", "op bal", "balance b f", "b f", "brought forward",
        "beginning balance", "opening bal", "initial balance", "start balance"]},
    "year": {"label": "Year (fiscal)", "required": False, "aliases": [
        "year", "fiscal year", "financial year", "fy", "yr", "period year", "fin year"]},
    "month": {"label": "Month / Period", "required": False, "aliases": [
        "month", "period", "month no", "period no", "fiscal period", "mth", "month number", "month name",
        "period number"]},
    "date": {"label": "Period Date", "required": False, "aliases": [
        "date", "period end", "period ending", "as at", "as of", "period date", "month end", "report date"]},
    "company": {"label": "Company", "required": False, "aliases": [
        "company", "entity", "company name", "legal entity", "company code", "business unit", "entity name"]},
    "cost_centre": {"label": "Cost Centre / Department", "required": False, "aliases": [
        "cost centre", "cost center", "department", "dept", "cost centre code", "cc", "profit centre",
        "profit center", "division", "analytic account", "segment", "branch", "cost centre name"]},
    "budget": {"label": "Budget", "required": False, "aliases": [
        "budget", "budget amount", "plan", "budgeted", "target", "budget kes", "forecast"]},
}
FIELD_ORDER = list(FIELD_DEFS.keys())

MONTH_LOOKUP: dict[str, int] = {}
for _i in range(1, 13):
    MONTH_LOOKUP[calendar.month_name[_i].lower()] = _i
    MONTH_LOOKUP[calendar.month_abbr[_i].lower()] = _i
MONTH_LOOKUP["sept"] = 9


# --------------------------------------------------------------------------- #
# Settings & mapping persistence
# --------------------------------------------------------------------------- #
DEFAULT_SETTINGS = {
    "basis": "closing",            # closing | movement
    "budget_basis": "movement",    # closing | movement
    "budget_sign": "natural",      # natural (all positive) | signed (Dr + / Cr -)
    "fy_start_month": 1,           # 1 = January ... 7 = July
    "numeric_month_is": "calendar",  # calendar | fiscal
    "carry_forward_pl": True,      # movement basis: roll prior-year P&L into equity
    "units": 1,                    # 1 | 1000 | 1000000
    "currency": "KES",
    "default_company": "Company",
}


def _read_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, json.JSONDecodeError):
        pass
    return json.loads(json.dumps(default))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    tmp.replace(path)


def load_settings() -> dict:
    s = _read_json(SETTINGS_FILE, DEFAULT_SETTINGS)
    for k, v in DEFAULT_SETTINGS.items():
        s.setdefault(k, v)
    return s


def save_settings(settings: dict) -> None:
    _write_json(SETTINGS_FILE, settings)


def load_mapping_store() -> dict:
    """{"by_signature": {sig: {field: column}}, "learned_aliases": {normalised header: field}}"""
    store = _read_json(MAPPINGS_FILE, {"by_signature": {}, "learned_aliases": {}})
    store.setdefault("by_signature", {})
    store.setdefault("learned_aliases", {})
    return store


def save_mapping(columns: list[str], mapping: dict[str, str | None]) -> None:
    """Persist the mapping for this exact column layout and learn the header aliases."""
    store = load_mapping_store()
    sig = column_signature(columns)
    clean = {f: c for f, c in mapping.items() if c and c != NOT_MAPPED}
    store["by_signature"][sig] = clean
    for field, col in clean.items():
        store["learned_aliases"][normalise_header(col)] = field
    _write_json(MAPPINGS_FILE, store)


def column_signature(columns: list) -> str:
    key = "|".join(sorted(normalise_header(c) for c in columns))
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Reading uploads
# --------------------------------------------------------------------------- #
def normalise_header(h) -> str:
    s = str(h).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _all_aliases(store: dict | None = None) -> set[str]:
    al = {a for f in FIELD_DEFS.values() for a in f["aliases"]}
    if store:
        al |= set(store.get("learned_aliases", {}).keys())
    return al


def detect_header_row(raw: pd.DataFrame, store: dict | None = None, scan_rows: int = 30) -> int:
    """Return the 0-based row index most likely to hold the column headers."""
    aliases = _all_aliases(store)
    best_row, best_score = 0, 0
    for i in range(min(scan_rows, len(raw))):
        cells = [normalise_header(v) for v in raw.iloc[i].tolist() if pd.notna(v) and str(v).strip()]
        score = sum(1 for c in cells if c in aliases)
        # small bonus for rows where most cells are text (typical header row)
        if score > best_score:
            best_row, best_score = i, score
    return best_row if best_score >= 2 else 0


def read_upload(file_name: str, data: bytes) -> dict[str, pd.DataFrame]:
    """
    Read every sheet of an Excel/CSV upload with NO header (header row is
    detected separately so title rows above the table are tolerated).
    """
    name = file_name.lower()
    bio = io.BytesIO(data)
    if name.endswith(".csv"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                bio.seek(0)
                return {"CSV": pd.read_csv(bio, header=None, dtype=object, encoding=enc)}
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV file")
    engine = "xlrd" if name.endswith(".xls") else None
    sheets = pd.read_excel(bio, sheet_name=None, header=None, dtype=object, engine=engine)
    return {str(k): v for k, v in sheets.items() if not v.dropna(how="all").empty}


def apply_header(raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    """Use `header_row` as the header; drop fully empty rows/columns; make names unique."""
    df = raw.iloc[header_row + 1:].copy()
    headers, seen = [], {}
    for i, h in enumerate(raw.iloc[header_row].tolist()):
        h = str(h).strip() if pd.notna(h) and str(h).strip() else f"Column {i + 1}"
        if h in seen:
            seen[h] += 1
            h = f"{h} ({seen[h]})"
        else:
            seen[h] = 0
        headers.append(h)
    df.columns = headers
    df = df.dropna(how="all")
    df = df.loc[:, [c for c in df.columns if not (c.startswith("Column ") and df[c].isna().all())]]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Column auto-mapping
# --------------------------------------------------------------------------- #
def auto_map_columns(columns: list[str], store: dict | None = None) -> dict[str, str | None]:
    """
    Map source columns to standard fields.

    1. Exact saved mapping for this column layout (signature) -> used as-is.
    2. Otherwise score every (field, column) pair:
         learned alias 100 | exact alias 95 | alias contained in header 60-70 |
         fuzzy similarity * 55
       and assign greedily (best score first, one column per field).
    """
    store = store or load_mapping_store()
    sig = column_signature(columns)
    saved = store.get("by_signature", {}).get(sig)
    if saved:
        out = {f: (saved.get(f) if saved.get(f) in columns else None) for f in FIELD_ORDER}
        if out.get("account_code"):
            return out

    learned = store.get("learned_aliases", {})
    candidates = []
    for col in columns:
        n = normalise_header(col)
        if not n:
            continue
        tokens = set(n.split())
        for field, fdef in FIELD_DEFS.items():
            score = 0.0
            if learned.get(n) == field:
                score = 100
            elif n in fdef["aliases"]:
                score = 95
            else:
                for a in fdef["aliases"]:
                    a_tokens = set(a.split())
                    if len(a) >= 3 and a_tokens <= tokens:
                        score = max(score, 60 + min(10, len(a)))
                ratio = max(difflib.SequenceMatcher(None, n, a).ratio() for a in fdef["aliases"])
                score = max(score, ratio * 55)
            if score >= 45:
                candidates.append((score, field, col))
    candidates.sort(key=lambda x: (-x[0], FIELD_ORDER.index(x[1])))
    mapping: dict[str, str | None] = {f: None for f in FIELD_ORDER}
    used: set[str] = set()
    for score, field, col in candidates:
        if mapping[field] is None and col not in used:
            mapping[field] = col
            used.add(col)
    # If both a debit/credit pair and a balance column matched, keep debit/credit
    if mapping["debit"] and mapping["credit"] and mapping["balance"]:
        n = normalise_header(mapping["balance"])
        if n not in FIELD_DEFS["balance"]["aliases"][:4]:
            mapping["balance"] = None
    return mapping


# --------------------------------------------------------------------------- #
# Value parsers
# --------------------------------------------------------------------------- #
def to_number(s: pd.Series) -> pd.Series:
    """
    Robust numeric parser: handles 1,234.50 | (1,234.50) | 1,234.50- | 1,234 Cr | 1,234 Dr | blanks.
    Credit markers (brackets, trailing minus, 'Cr') produce negative values.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype("string").str.strip()
    t = t.fillna("")
    is_neg = t.str.match(r"^\(.*\)$") | t.str.endswith("-") | t.str.upper().str.endswith("CR")
    clean = (t.str.replace(r"(?i)\s*(cr|dr)\.?$", "", regex=True)
              .str.replace(r"[()\s,]", "", regex=True)
              .str.replace(r"-$", "", regex=True)
              .str.replace(r"^[A-Za-z$£€]+\.?", "", regex=True))  # currency prefixes like KES / Ksh.
    v = pd.to_numeric(clean.replace("", np.nan), errors="coerce").astype(float)
    return pd.Series(np.where(is_neg.fillna(False).to_numpy(), -v.abs(), v), index=s.index, dtype=float)


def clean_code(v) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s


def parse_year(v) -> int | None:
    """2025 | 2025.0 | 'FY2025' | '2024/25' | '2024-2025' -> fiscal year (end year)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.year
    s = str(v).strip()
    m = re.search(r"(\d{4})\s*[/\-]\s*(\d{2,4})", s)
    if m:
        end = m.group(2)
        return int(m.group(1)[:2] + end) if len(end) == 2 else int(end)
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return int(m.group(0))
    m = re.fullmatch(r"(?:fy)?\s*(\d{2})", s.lower())
    if m:
        return 2000 + int(m.group(1))
    return None


def parse_month(v) -> tuple[int | None, str]:
    """
    Return (month_number, kind) where kind is 'calendar' (month name / date)
    or 'numeric' (plain number - meaning set by the 'numeric_month_is' setting).
    """
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None, ""
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.month, "calendar"
    if isinstance(v, (int, float, np.integer, np.floating)):
        iv = int(v)
        return (iv, "numeric") if 1 <= iv <= 12 else (None, "")
    s = str(v).strip().lower()
    if re.fullmatch(r"\d{1,2}(\.0+)?", s):
        iv = int(float(s))
        return (iv, "numeric") if 1 <= iv <= 12 else (None, "")
    m = re.fullmatch(r"(?:p|per|period|m|month)\s*0?(\d{1,2})", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return int(m.group(1)), "numeric"
    for name, num in sorted(MONTH_LOOKUP.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{name}", s):
            return num, "calendar"
    try:
        d = pd.to_datetime(s, errors="raise", dayfirst=True)
        return d.month, "calendar"
    except (ValueError, TypeError, OverflowError):
        return None, ""


def calendar_to_fiscal(cal_month: int, cal_year: int | None, fy_start: int) -> tuple[int, int | None]:
    """Calendar month/year -> (fiscal period, fiscal year). Fiscal year = year in which FY ends."""
    period = (cal_month - fy_start) % 12 + 1
    if cal_year is None:
        return period, None
    fy = cal_year + (1 if fy_start > 1 and cal_month >= fy_start else 0)
    return period, fy


def fiscal_to_calendar_month(period: int, fy_start: int) -> int:
    return (fy_start - 1 + period - 1) % 12 + 1


def period_label(year: int, period: int, fy_start: int = 1) -> str:
    mon = calendar.month_abbr[fiscal_to_calendar_month(int(period), fy_start)]
    if fy_start == 1:
        return f"{mon}-{int(year)}"
    return f"FY{int(year)} P{int(period):02d} ({mon})"


def infer_period_from_text(text: str) -> tuple[int | None, int | None]:
    """Guess (year, calendar month) from a file or sheet name e.g. 'TB Mar 2025.xlsx'."""
    t = str(text)
    year = None
    m = re.search(r"(19|20)\d{2}(?:\s*[/\-]\s*\d{2,4})?", t)
    if m:
        year = parse_year(m.group(0))
    month = None
    tl = t.lower()
    for name, num in sorted(MONTH_LOOKUP.items(), key=lambda x: -len(x[0])):
        if re.search(rf"(?<![a-z]){name}(?![a-z])", tl):
            month = num
            break
    if month is None:
        m2 = re.search(r"(19|20)\d{2}[\-_\.](\d{1,2})(?!\d)", t)
        if m2 and 1 <= int(m2.group(2)) <= 12:
            month = int(m2.group(2))
    return year, month


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #
TOTAL_ROW_PATTERN = re.compile(r"\b(?:grand\s+)?totals?\b|^net\s+(?:profit|loss)|^difference$", re.I)


def standardise_frame(df: pd.DataFrame, mapping: dict[str, str | None], defaults: dict,
                      settings: dict, source: str = "") -> tuple[pd.DataFrame, list[dict]]:
    """
    Convert a mapped source frame into canonical TB rows.

    defaults: {"company": str, "year": int|None, "month": int|None (calendar), "cost_centre": str|None}
    Returns (frame, issues)
    """
    issues: list[dict] = []
    m = {k: v for k, v in mapping.items() if v and v != NOT_MAPPED and v in df.columns}
    fy_start = int(settings.get("fy_start_month", 1))
    numeric_is = settings.get("numeric_month_is", "calendar")

    if "account_code" not in m:
        raise ValueError(f"{source}: 'Account Code' column must be mapped.")
    has_dc = "debit" in m or "credit" in m
    if not has_dc and "balance" not in m and "budget" not in m:
        raise ValueError(f"{source}: map either Debit/Credit columns or a Net Balance column.")

    out = pd.DataFrame(index=df.index)
    out["account_code"] = df[m["account_code"]].map(clean_code)
    out["account_name"] = (df[m["account_name"]].astype("string").str.strip().fillna("")
                           if "account_name" in m else "")

    # --- remove blank / subtotal rows ------------------------------------------------
    blank = out["account_code"].eq("") | out["account_code"].str.lower().isin(["nan", "none"])
    totals = out["account_code"].str.contains(TOTAL_ROW_PATTERN, na=False) | (
        blank & out["account_name"].astype(str).str.contains(TOTAL_ROW_PATTERN, na=False))
    n_blank, n_tot = int((blank & ~totals).sum()), int(totals.sum())
    keep = ~(blank | totals)
    if n_tot:
        issues.append({"level": "info", "title": "Total rows removed",
                       "message": f"{source}: {n_tot} subtotal/total row(s) were excluded."})
    if n_blank:
        dropped = df.loc[blank & ~totals]
        amount_cols = [m[c] for c in ("debit", "credit", "balance") if c in m]
        has_amounts = dropped[amount_cols].apply(to_number).abs().sum().sum() if amount_cols else 0
        issues.append({"level": "warning" if has_amounts else "info", "title": "Rows without account code",
                       "message": f"{source}: {n_blank} row(s) without an account code were excluded"
                                  + (f" (carrying amounts totalling {has_amounts:,.2f})." if has_amounts else "."),
                       "detail": dropped.head(200) if has_amounts else None})
    df = df.loc[keep]
    out = out.loc[keep]

    # --- amounts ---------------------------------------------------------------------
    def num(field):
        raw = df[m[field]]
        v = to_number(raw)
        bad = raw.notna() & raw.astype(str).str.strip().ne("") & v.isna()
        if bad.any():
            issues.append({"level": "warning", "title": "Non-numeric amounts",
                           "message": f"{source}: {int(bad.sum())} non-numeric value(s) in '{m[field]}' treated as 0.",
                           "detail": df.loc[bad, [m["account_code"], m[field]]].head(200)})
        return v.fillna(0.0)

    if has_dc:
        dr = num("debit") if "debit" in m else pd.Series(0.0, index=df.index)
        cr = num("credit") if "credit" in m else pd.Series(0.0, index=df.index)
        # Some exports show credits as negatives in the credit column - normalise to positive
        cr = cr.abs() if (cr < 0).sum() > (cr > 0).sum() else cr
        out["debit"], out["credit"] = dr, cr
        out["net"] = dr - cr
    elif "balance" in m:
        bal = num("balance")
        out["net"] = bal
        out["debit"], out["credit"] = bal.clip(lower=0), (-bal).clip(lower=0)
    else:
        out["net"], out["debit"], out["credit"] = 0.0, 0.0, 0.0
    out["opening_balance"] = to_number(df[m["opening_balance"]]) if "opening_balance" in m else np.nan
    out["budget"] = num("budget") if "budget" in m else np.nan

    # --- company / cost centre ----------------------------------------------------------
    def_company = str(defaults.get("company") or settings.get("default_company") or "Company")
    if "company" in m:
        out["company"] = df[m["company"]].astype("string").str.strip().fillna(def_company).replace("", def_company)
    else:
        out["company"] = def_company
    if "cost_centre" in m:
        out["cost_centre"] = df[m["cost_centre"]].astype("string").str.strip().fillna(NO_CC).replace(
            {"": NO_CC, "nan": NO_CC, "None": NO_CC})
    else:
        out["cost_centre"] = defaults.get("cost_centre") or NO_CC

    # --- year / period ------------------------------------------------------------------
    years = pd.Series([None] * len(df), index=df.index, dtype=object)
    periods = pd.Series([None] * len(df), index=df.index, dtype=object)
    if "date" in m:
        d = pd.to_datetime(df[m["date"]], errors="coerce", dayfirst=True)
        ok = d.notna()
        for idx in d[ok].index:
            p, fy = calendar_to_fiscal(d[idx].month, d[idx].year, fy_start)
            periods[idx], years[idx] = p, fy
    if "year" in m:
        y = df[m["year"]].map(parse_year)
        years = years.where(years.notna(), y)
    if "month" in m:
        parsed = df[m["month"]].map(parse_month)
        for idx, (mn, kind) in parsed.items():
            if mn is None or periods[idx] is not None:
                continue
            if kind == "numeric" and numeric_is == "fiscal":
                periods[idx] = mn
            else:
                periods[idx] = calendar_to_fiscal(mn, None, fy_start)[0]
    if defaults.get("year"):
        years = years.where(years.notna(), int(defaults["year"]))
    if periods.isna().all() and defaults.get("month"):
        periods[:] = calendar_to_fiscal(int(defaults["month"]), None, fy_start)[0]
    elif periods.isna().all():
        periods[:] = 12  # annual trial balance -> treat as year-end (period 12)
        issues.append({"level": "info", "title": "Annual TB assumed",
                       "message": f"{source}: no month/period found - treated as a year-end (period 12) trial balance."})
    out["year"], out["period"] = years, periods
    miss = out["year"].isna() | out["period"].isna()
    if miss.any():
        issues.append({"level": "error", "title": "Missing year/period",
                       "message": f"{source}: {int(miss.sum())} row(s) have no identifiable year/period and were excluded. "
                                  "Map a Year/Month column or set a default year in the mapping panel.",
                       "detail": out.loc[miss].head(200)})
        out = out.loc[~miss]
    out["year"] = out["year"].astype(int)
    out["period"] = out["period"].astype(int)
    out["source"] = source
    cols = ["company", "cost_centre", "year", "period", "account_code", "account_name",
            "debit", "credit", "net", "opening_balance", "budget", "source"]
    return out[cols].reset_index(drop=True), issues


# --------------------------------------------------------------------------- #
# Ledger construction
# --------------------------------------------------------------------------- #
KEYS = ["company", "cost_centre", "account_code"]


def account_master(std: pd.DataFrame) -> pd.DataFrame:
    """Unique accounts with the most frequent non-blank name."""
    if std.empty:
        return pd.DataFrame(columns=["account_code", "account_name"])
    s = std[["account_code", "account_name"]].copy()
    s["account_name"] = s["account_name"].astype(str).replace({"nan": "", "<NA>": ""})
    named = s[s["account_name"].str.strip() != ""]
    names = (named.groupby("account_code")["account_name"]
             .agg(lambda x: x.value_counts().index[0]) if not named.empty else pd.Series(dtype=str))
    codes = pd.DataFrame({"account_code": sorted(s["account_code"].unique(), key=_code_sort_key)})
    codes["account_name"] = codes["account_code"].map(names).fillna("")
    return codes


def _code_sort_key(c: str):
    d = re.sub(r"\D", "", str(c))
    return (0, int(d), str(c)) if d else (1, 0, str(c))


def build_ledger(std: pd.DataFrame, classification: pd.DataFrame, basis: str = "closing",
                 natural_sign: bool = False) -> pd.DataFrame:
    """
    Aggregate canonical rows to account x period and derive movement / closing.

    * The grid is completed so every account has a row in every period the
      company reported (absent accounts in a TB = nil balance).
    * closing basis : closing = amount; movement = change vs previous period
                      (P&L accounts reset at the start of each fiscal year).
    * movement basis: movement = amount; P&L closing = YTD cumulative;
                      B/S closing = opening balance (when supplied) + cumulative movement.
    * natural_sign  : budgets keyed with income as positive numbers are flipped to credit (negative).
    """
    cols_out = ["company", "cost_centre", "year", "period", "pkey", "account_code", "account_name",
                "category", "sub_category", "is_pl", "movement", "closing", "opening", "debit", "credit"]
    if std is None or std.empty:
        return pd.DataFrame(columns=cols_out)

    agg = (std.groupby(KEYS + ["year", "period"], as_index=False)
              .agg(debit=("debit", "sum"), credit=("credit", "sum"), net=("net", "sum"),
                   opening_balance=("opening_balance", lambda x: x.sum(min_count=1))))

    # --- complete the grid -----------------------------------------------------------
    comp_periods = agg[["company", "year", "period"]].drop_duplicates()
    comp_accts = agg[KEYS].drop_duplicates()
    grid = comp_accts.merge(comp_periods, on="company", how="inner")
    led = grid.merge(agg, on=KEYS + ["year", "period"], how="left")
    for c in ("debit", "credit", "net"):
        led[c] = led[c].fillna(0.0)

    cls = classification[["account_code", "account_name", "category", "sub_category"]]
    led = led.merge(cls, on="account_code", how="left")
    led["sub_category"] = led["sub_category"].fillna(UNCLASSIFIED)
    led["category"] = led["sub_category"].map(CATEGORY_OF).fillna(UNCLASSIFIED)
    led["account_name"] = led["account_name"].fillna("")
    led["is_pl"] = led["category"].isin(PL_CATEGORIES)

    if natural_sign:
        # natural budgets: income keyed positive -> credit; expenses positive -> debit
        led.loc[led["category"] == "Income", "net"] *= -1
        led.loc[led["category"].isin(["Liability", "Equity"]), "net"] *= -1

    led = led.sort_values(KEYS + ["year", "period"]).reset_index(drop=True)
    g = led.groupby(KEYS, sort=False)

    if basis == "closing":
        led["closing"] = led["net"]
        prev = g["closing"].shift(1)
        prev_year = g["year"].shift(1)
        first_opening = led["opening_balance"].fillna(0.0)
        bs_mov = np.where(prev.isna(), led["closing"] - first_opening, led["closing"] - prev)
        pl_mov = np.where(prev_year.eq(led["year"]), led["closing"] - prev, led["closing"])
        led["movement"] = np.where(led["is_pl"], pl_mov, bs_mov)
    else:
        led["movement"] = led["net"]
        pl_close = led.groupby(KEYS + ["year"], sort=False)["movement"].cumsum()
        # B/S: segments start at each supplied opening balance
        anchor = led["opening_balance"].notna()
        seg = anchor.astype(int).groupby([led[k] for k in KEYS]).cumsum()
        anchor_val = led["opening_balance"].where(anchor)
        anchor_val = anchor_val.groupby([led[k] for k in KEYS] + [seg]).transform("first").fillna(0.0)
        bs_close = anchor_val + led.groupby([led[k] for k in KEYS] + [seg])["movement"].cumsum()
        led["closing"] = np.where(led["is_pl"], pl_close, bs_close)

    led["opening"] = led["closing"] - led["movement"]
    led["pkey"] = led["year"] * 100 + led["period"]
    return led[cols_out]


def budget_frame_from_actual(std: pd.DataFrame) -> pd.DataFrame:
    """Extract a Budget column carried inside an actual TB as its own canonical frame."""
    if "budget" not in std or std["budget"].isna().all():
        return pd.DataFrame(columns=std.columns)
    b = std[std["budget"].notna()].copy()
    b["net"] = b["budget"]
    b["debit"], b["credit"] = b["net"].clip(lower=0), (-b["net"]).clip(lower=0)
    b["opening_balance"] = np.nan
    return b


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_data(std: pd.DataFrame, ledger: pd.DataFrame, classification: pd.DataFrame,
                  basis: str, carry_forward: bool = True) -> list[dict]:
    issues: list[dict] = []
    if std is None or std.empty:
        return [{"level": "error", "title": "No data", "message": "No trial balance rows were loaded."}]

    # 1. TB balances per company / period
    tb = (std.groupby(["company", "year", "period"])
             .agg(total_debit=("debit", "sum"), total_credit=("credit", "sum"), net=("net", "sum"))
             .reset_index())
    tb["difference"] = tb["net"].round(2)
    tol = np.maximum(1.0, tb["total_debit"].abs() * 1e-6)
    unbalanced = tb[tb["difference"].abs() > tol]
    if not unbalanced.empty:
        issues.append({"level": "error", "title": "Unbalanced trial balance",
                       "message": f"{len(unbalanced)} company-period(s) where total debits ≠ total credits. "
                                  "Reports will show the difference in the Balance Sheet check line.",
                       "detail": unbalanced})
    else:
        issues.append({"level": "success", "title": "Trial balance balances",
                       "message": f"All {len(tb)} company-period trial balances agree (Dr = Cr)."})

    # 2. Unclassified / fallback-classified accounts
    unc = classification[classification["sub_category"] == UNCLASSIFIED]
    if not unc.empty:
        issues.append({"level": "error", "title": "Unclassified accounts",
                       "message": f"{len(unc)} account(s) could not be classified - fix on the Classification page.",
                       "detail": unc})
    fb = classification[classification["rule_source"].str.startswith("First-digit", na=False)]
    if not fb.empty:
        issues.append({"level": "warning", "title": "Fallback classification used",
                       "message": f"{len(fb)} account(s) were classified only by their first digit - please review.",
                       "detail": fb})

    # 3. Missing account names
    nameless = classification[classification["account_name"].astype(str).str.strip().isin(["", "nan", "<NA>"])]
    if not nameless.empty:
        issues.append({"level": "warning", "title": "Missing account names",
                       "message": f"{len(nameless)} account code(s) have no name.", "detail": nameless})

    # 4. Same code with different names
    names = std[std["account_name"].astype(str).str.strip() != ""].groupby("account_code")["account_name"].nunique()
    multi = names[names > 1]
    if not multi.empty:
        det = (std[std["account_code"].isin(multi.index)][["account_code", "account_name", "source"]]
               .drop_duplicates().sort_values("account_code"))
        issues.append({"level": "info", "title": "Account codes with multiple names",
                       "message": f"{len(multi)} code(s) appear with different names across files (most frequent name used).",
                       "detail": det})

    # 5. Duplicate rows (same key appearing more than once - aggregated)
    dup = std.duplicated(subset=["company", "cost_centre", "year", "period", "account_code"], keep=False)
    if dup.any():
        issues.append({"level": "warning", "title": "Duplicate account lines",
                       "message": f"{int(dup.sum())} rows share the same company/cost centre/period/account and were summed. "
                                  "Check the same TB was not uploaded twice.",
                       "detail": std[dup].sort_values(["account_code", "year", "period"]).head(500)})

    # 6. Period gaps within a year
    gaps = []
    for (comp, yr), grp in std.groupby(["company", "year"]):
        ps = sorted(grp["period"].unique())
        if len(ps) > 1:
            missing = [p for p in range(min(ps), max(ps) + 1) if p not in ps]
            if missing:
                gaps.append({"company": comp, "year": yr, "missing_periods": ", ".join(map(str, missing))})
    if gaps:
        issues.append({"level": "warning", "title": "Missing months",
                       "message": "Some years have gaps in the monthly sequence - movements across a gap combine months.",
                       "detail": pd.DataFrame(gaps)})

    # 7. Companies with different period coverage (consolidation risk)
    cov = std.groupby("company")[["year", "period"]].apply(lambda x: frozenset(zip(x["year"], x["period"])))
    if cov.nunique() > 1:
        issues.append({"level": "warning", "title": "Different period coverage across companies",
                       "message": "Companies do not cover identical periods - consolidated figures for a period "
                                  "only include companies that reported it."})

    # 8. Suspense accounts carrying balances
    if ledger is not None and not ledger.empty:
        last = ledger[ledger["pkey"] == ledger["pkey"].max()]
        sus = last[last["account_name"].str.contains("suspense|clearing|unallocated|unknown", case=False, na=False)
                   & (last["closing"].abs() > 0.5)]
        if not sus.empty:
            issues.append({"level": "warning", "title": "Suspense / clearing balances",
                           "message": f"{len(sus)} suspense or clearing account(s) carry balances at the latest period.",
                           "detail": sus[["company", "cost_centre", "account_code", "account_name", "closing"]]})

        # 9. Abnormal balances (assets in credit, liabilities in debit, income debit ...)
        abn_rule = (((last["category"] == "Asset") & (last["closing"] < -0.5)
                     & ~last["account_name"].str.contains("depreciation|amortis|amortiz|provision|impairment",
                                                          case=False, na=False))
                    | ((last["category"].isin(["Liability"])) & (last["closing"] > 0.5))
                    | ((last["category"] == "Income") & (last["closing"] > 0.5))
                    | ((last["category"] == "Expense") & (last["closing"] < -0.5)))
        abn = last[abn_rule]
        if not abn.empty:
            issues.append({"level": "info", "title": "Contra / abnormal balances",
                           "message": f"{len(abn)} account(s) carry a balance opposite to their normal side "
                                      "at the latest period (e.g. overdrawn bank, debit-balance creditor).",
                           "detail": abn[["company", "account_code", "account_name", "sub_category", "closing"]]})

        # 10. Balance sheet check per period
        diffs = []
        for (comp, pk), grp in ledger.groupby(["company", "pkey"]):
            d = grp["closing"].sum()
            if basis == "movement" and carry_forward:
                prior = ledger[(ledger["company"] == comp) & ledger["is_pl"] & (ledger["year"] < pk // 100)]
                d += prior["movement"].sum()
            if abs(d) > 1:
                diffs.append({"company": comp, "period": pk, "difference": round(d, 2)})
        if diffs and unbalanced.empty:
            issues.append({"level": "warning", "title": "Balance sheet does not balance",
                           "message": "Assets ≠ Equity + Liabilities for some periods (check opening balances / basis setting).",
                           "detail": pd.DataFrame(diffs)})
    return issues


# --------------------------------------------------------------------------- #
# Sample data (demo + test)
# --------------------------------------------------------------------------- #
SAMPLE_ACCOUNTS = [
    ("1010", "Land & Buildings"), ("1020", "Plant & Machinery"), ("1030", "Motor Vehicles"),
    ("1090", "Accumulated Depreciation"), ("1510", "Inventory - Raw Materials"),
    ("1520", "Inventory - Finished Goods"), ("1610", "Trade Debtors"), ("1710", "Prepayments"),
    ("1720", "VAT Receivable"), ("1810", "Bank - KCB Current Account"), ("1820", "Bank - Equity USD Account"),
    ("1830", "M-Pesa Float"), ("1840", "Petty Cash"), ("2010", "Trade Creditors"),
    ("2110", "Corporation Tax Payable"), ("2210", "Bank Overdraft"), ("2310", "Accruals"),
    ("2320", "PAYE Payable"), ("2330", "VAT Payable"), ("2510", "Long-term Bank Loan"),
    ("3010", "Share Capital"), ("3210", "Retained Earnings"), ("4010", "Sales - Local"),
    ("4020", "Sales - Export"), ("4810", "Interest Income"), ("4820", "Other Income"),
    ("5010", "Raw Materials Consumed"), ("5020", "Direct Labour"), ("5030", "Factory Overheads"),
    ("6010", "Salaries & Wages"), ("6020", "NSSF, SHIF & Housing Levy - Employer"),
    ("6310", "Rent & Rates"), ("6320", "Electricity & Water"), ("6330", "Marketing & Advertising"),
    ("6340", "Transport & Fuel"), ("6350", "Repairs & Maintenance"), ("6360", "Professional Fees"),
    ("6370", "Bank Charges"), ("6380", "Insurance"), ("6390", "Office Expenses"),
    ("7510", "Depreciation"), ("7710", "Interest on Loans"), ("7910", "Income Tax Expense"),
]


def _simulate_ledger(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate 34 months (Jan-2023 .. Oct-2025) of balanced double entries.
    Returns (closing balances by month & cost centre, monthly budget).
    """
    rng = np.random.default_rng(seed)
    names = dict(SAMPLE_ACCOUNTS)
    bal: dict[tuple[str, str], float] = {}

    def post(code, amt, cc=NO_CC):
        bal[(code, cc)] = bal.get((code, cc), 0.0) + amt

    # Opening position 1 Jan 2023 (Dr +, Cr -)
    for code, amt in {"1010": 45e6, "1020": 28e6, "1030": 9e6, "1090": -14e6, "1510": 6.5e6, "1520": 4.2e6,
                      "1610": 11e6, "1710": 0.9e6, "1810": 7.4e6, "1820": 2.1e6, "1830": 0.35e6, "1840": 0.05e6,
                      "2010": -8.6e6, "2110": -1.2e6, "2310": -0.9e6, "2510": -30e6, "3010": -20e6}.items():
        post(code, amt)
    post("3210", -sum(bal.values()))  # balancing retained earnings

    rows, months = [], [(y, mth) for y in (2023, 2024, 2025) for mth in range(1, 13)][:34]
    growth = {2023: 1.0, 2024: 1.12, 2025: 1.23}
    pl_codes = [c for c, _ in SAMPLE_ACCOUNTS if c[0] in "4567"]
    for y, mth in months:
        if mth == 1 and y > 2023:  # year-end close: P&L -> retained earnings
            for (code, cc), v in list(bal.items()):
                if code in pl_codes and v:
                    post("3210", v)
                    bal[(code, cc)] = 0.0
        seas = 1 + 0.12 * np.sin((mth - 3) / 12 * 2 * np.pi)
        g = growth[y] * seas
        local, export = 14.5e6 * g * rng.uniform(.9, 1.1), 3.8e6 * g * rng.uniform(.8, 1.2)
        post("4010", -local, "Sales"); post("4020", -export, "Sales")
        post("2330", -local * .16); post("1610", local * 1.16 + export)
        post("1810", bal[("1610", NO_CC)] * .82); post("1610", -bal[("1610", NO_CC)] * .82)
        rm = (local + export) * rng.uniform(.34, .38)
        post("5010", rm, "Production"); post("1510", -rm)
        purchases = rm * rng.uniform(.93, 1.06)
        post("1510", purchases); post("1720", purchases * .16); post("2010", -purchases * 1.16)
        pay = -bal[("2010", NO_CC)] * .78
        post("2010", pay); post("1810", -pay)
        dl = 1.2e6 * g; fo = 0.8e6 * g * rng.uniform(.9, 1.15)
        post("5020", dl, "Production"); post("5030", fo, "Production")
        sal = {"Head Office": 1.2e6, "Sales": 0.8e6, "Production": 0.5e6}
        for cc, base in sal.items():
            s = base * growth[y]
            post("6010", s, cc); post("6020", s * .075, cc)
        gross_pay = sum(sal.values()) * growth[y] + dl
        post("2320", -gross_pay * .22)
        post("1810", -(gross_pay * (1 - .22) + sum(sal.values()) * growth[y] * .075 + fo))
        paye_paid = -bal[("2320", NO_CC)] * .5
        post("2320", paye_paid); post("1810", -paye_paid)
        opex = {"6310": (0.85e6, "Head Office"), "6320": (0.42e6, "Production"), "6330": (0.6e6, "Sales"),
                "6340": (0.55e6, "Sales"), "6350": (0.3e6, "Production"), "6360": (0.25e6, "Head Office"),
                "6370": (0.06e6, "Head Office"), "6380": (0.2e6, "Head Office"), "6390": (0.18e6, "Head Office")}
        for code, (base, cc) in opex.items():
            a = base * growth[y] * rng.uniform(.8, 1.25)
            post(code, a, cc); post("1810", -a * .7); post("2310", -a * .3)
        acc_paid = -bal[("2310", NO_CC)] * .6
        post("2310", acc_paid); post("1810", -acc_paid)
        dep = 0.62e6 * (1 + 0.01 * (mth + 12 * (y - 2023)))
        post("7510", dep, "Head Office"); post("1090", -dep)
        intr = -bal[("2510", NO_CC)] * .14 / 12
        post("7710", intr, "Head Office"); post("1810", -intr)
        post("4810", -0.04e6, "Head Office"); post("1810", 0.04e6)
        if rng.uniform() < .3:
            oi = rng.uniform(.05e6, .3e6); post("4820", -oi, "Head Office"); post("1830", oi)
        if mth % 3 == 0:
            post("2510", 2.0e6); post("1810", -2.0e6)  # loan repayment
        if mth in (4, 10):
            capex = rng.uniform(1.5e6, 4.5e6); post("1020", capex); post("1810", -capex)
        if mth == 8 and y == 2024:
            post("1030", 6.5e6); post("1810", -6.5e6)
        vat_net = -bal[("2330", NO_CC)] - bal[("1720", NO_CC)]
        post("2330", -bal[("2330", NO_CC)]); post("1720", -bal[("1720", NO_CC)]); post("1810", -max(vat_net, 0))
        if vat_net < 0:
            post("1720", -vat_net)  # carry forward VAT credit
        pbt_ytd = -sum(v for (c, _), v in bal.items() if c in pl_codes and c != "7910")
        tax = max(pbt_ytd, 0) * .30 - bal.get(("7910", "Head Office"), 0.0)  # true-up YTD provision
        post("7910", tax, "Head Office"); post("2110", -tax)
        if mth in (4, 6, 9, 12):
            inst = -bal[("2110", NO_CC)] * .7
            post("2110", inst); post("1810", -inst)
        if mth == 6 and y in (2023, 2024):
            post("3210", 5e6); post("1810", -5e6)  # dividend
        if bal[("1810", NO_CC)] < 0:  # move negative bank to overdraft
            od = -bal[("1810", NO_CC)]
            post("1810", od); post("2210", -od)
        elif bal.get(("2210", NO_CC), 0) < 0 and bal[("1810", NO_CC)] > 5e6:
            rep = min(-bal[("2210", NO_CC)], bal[("1810", NO_CC)] - 5e6)
            post("2210", rep); post("1810", -rep)
        for (code, cc), v in bal.items():
            if abs(v) > 0.005:
                rows.append({"year": y, "month": mth, "cost_centre": cc, "code": code,
                             "name": names[code], "balance": round(v, 2)})
    closing = pd.DataFrame(rows)

    # Budget for 2025 (monthly, natural sign, by cost centre)
    brows = []
    pl25 = closing[(closing["year"] == 2024) & (closing["month"] == 12) & closing["code"].isin(pl_codes)]
    for _, r in pl25.iterrows():
        if r["code"] == "7910":
            continue
        uplift = 1.15 if r["code"][0] == "4" else 1.10
        for mth in range(1, 13):
            seas = 1 + 0.10 * np.sin((mth - 3) / 12 * 2 * np.pi) if r["code"][0] in "45" else 1
            brows.append({"Company": "Demo Kenya Ltd", "Year": 2025, "Month": mth, "Cost Centre": r["cost_centre"],
                          "Account Code": r["code"], "Account Name": r["name"],
                          "Budget Amount": round(abs(r["balance"]) / 12 * uplift * seas, 0)})
    bud = pd.DataFrame(brows)
    # Budgeted tax at 30% of budgeted monthly profit before tax
    sgn = np.where(bud["Account Code"].str[0] == "4", 1, -1)
    pbt = (bud["Budget Amount"] * sgn).groupby(bud["Month"]).sum()
    tax = pd.DataFrame({"Company": "Demo Kenya Ltd", "Year": 2025, "Month": pbt.index, "Cost Centre": "Head Office",
                        "Account Code": "7910", "Account Name": "Income Tax Expense",
                        "Budget Amount": (pbt.clip(lower=0) * .30).round(0).to_numpy()})
    return closing, pd.concat([bud, tax], ignore_index=True)


def generate_sample_files(seed: int = 7) -> dict[str, bytes]:
    """
    Build three demo workbooks:
      1. Sample_TB_Annual_2023_2024.xlsx  - multi-year annual closing TB (title rows + non-standard headers)
      2. Sample_TB_Monthly_2025.xlsx      - current year month-wise closing TB by cost centre
      3. Sample_Budget_2025.xlsx          - monthly budget (natural sign) by cost centre
    """
    closing, budget = _simulate_ledger(seed)

    def dr_cr(df):
        df = df.copy()
        df["Debit"] = df["balance"].clip(lower=0).round(2)
        df["Credit"] = (-df["balance"]).clip(lower=0).round(2)
        return df

    files: dict[str, bytes] = {}
    # 1. annual
    ann = closing[closing["month"] == 12].groupby(["year", "code", "name"], as_index=False)["balance"].sum()
    ann = dr_cr(ann[ann["balance"].abs() > 0.005])
    ann = ann.rename(columns={"year": "Fin Year", "code": "GL Code", "name": "Description"})
    ann.insert(0, "Entity", "Demo Kenya Ltd")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for yr in (2023, 2024):
            part = ann[ann["Fin Year"] == yr][["Entity", "Fin Year", "GL Code", "Description", "Debit", "Credit"]]
            title = pd.DataFrame([[f"Demo Kenya Ltd - Trial Balance as at 31 December {yr}"], ["Amounts in KES"], [None]])
            title.to_excel(xw, sheet_name=f"TB {yr}", header=False, index=False)
            part.to_excel(xw, sheet_name=f"TB {yr}", startrow=3, index=False)
            tot = pd.DataFrame([["", "", "", "TOTAL", part["Debit"].sum(), part["Credit"].sum()]])
            tot.to_excel(xw, sheet_name=f"TB {yr}", startrow=4 + len(part), header=False, index=False)
    files["Sample_TB_Annual_2023_2024.xlsx"] = bio.getvalue()

    # 2. monthly 2025
    mon = dr_cr(closing[closing["year"] == 2025])
    mon["Month"] = mon["month"].map(lambda x: calendar.month_abbr[x])
    mon["Company"] = "Demo Kenya Ltd"
    mon = mon.rename(columns={"year": "Year", "cost_centre": "Cost Centre", "code": "Account Code",
                              "name": "Account Name"})
    mon["Cost Centre"] = mon["Cost Centre"].replace(NO_CC, "")
    bio = io.BytesIO()
    mon[["Company", "Year", "Month", "Cost Centre", "Account Code", "Account Name", "Debit", "Credit"]].to_excel(
        bio, sheet_name="Monthly TB 2025", index=False)
    files["Sample_TB_Monthly_2025.xlsx"] = bio.getvalue()

    # 3. budget
    bio = io.BytesIO()
    budget.to_excel(bio, sheet_name="Budget 2025", index=False)
    files["Sample_Budget_2025.xlsx"] = bio.getvalue()
    return files
