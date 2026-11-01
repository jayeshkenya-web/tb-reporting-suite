"""
utils/account_classifier.py
===========================

Configurable classification of Trial Balance accounts into the reporting
hierarchy used by every report in the suite:

    Category      ->  Sub-category (drives report line placement)
    ------------      ------------------------------------------------
    Asset             Non-Current Assets, Inventory, Trade Receivables,
                      Other Current Assets, Cash & Bank
    Liability         Non-Current Liabilities, Borrowings - Current,
                      Trade Payables, Tax Payable, Other Current Liabilities
    Equity            Share Capital, Other Reserves, Retained Earnings
    Income            Revenue, Other Income
    Expense           Cost of Sales, Staff Costs, Operating Expenses,
                      Depreciation & Amortisation, Finance Costs, Income Tax

Classification layers (evaluated in the order set in ``rules["priority"]``,
first match wins):

    1. override     - manual per-account assignment (saved from the UI)
    2. code_range   - numeric ranges on the first N digits of the account code
    3. keyword      - ordered keyword list matched against the account name
    4. first_digit  - fallback: first digit of code -> category -> default sub
    5. Unclassified - nothing matched (flagged in validation)

Only the *sub-category* is stored for a rule; the category is derived from it,
which makes inconsistent category/sub-category pairs impossible.

Rules and overrides persist as JSON in ``config/`` so the next upload is
classified automatically.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parent / "config"
RULES_FILE = CONFIG_DIR / "classification_rules.json"
OVERRIDES_FILE = CONFIG_DIR / "account_overrides.json"

# --------------------------------------------------------------------------- #
# Reporting hierarchy
# --------------------------------------------------------------------------- #
SUB_CATEGORIES: dict[str, list[str]] = {
    "Asset": ["Non-Current Assets", "Inventory", "Trade Receivables",
              "Other Current Assets", "Cash & Bank"],
    "Liability": ["Non-Current Liabilities", "Borrowings - Current", "Trade Payables",
                  "Tax Payable", "Other Current Liabilities"],
    "Equity": ["Share Capital", "Other Reserves", "Retained Earnings"],
    "Income": ["Revenue", "Other Income"],
    "Expense": ["Cost of Sales", "Staff Costs", "Operating Expenses",
                "Depreciation & Amortisation", "Finance Costs", "Income Tax"],
}
UNCLASSIFIED = "Unclassified"
CATEGORIES = list(SUB_CATEGORIES.keys())
CATEGORY_OF: dict[str, str] = {s: c for c, subs in SUB_CATEGORIES.items() for s in subs}
CATEGORY_OF[UNCLASSIFIED] = UNCLASSIFIED
ALL_SUB_CATEGORIES: list[str] = [s for subs in SUB_CATEGORIES.values() for s in subs] + [UNCLASSIFIED]
PL_CATEGORIES = ("Income", "Expense")
BS_CATEGORIES = ("Asset", "Liability", "Equity")

# Default sub-category used by the first-digit fallback
DEFAULT_SUB_FOR_CATEGORY = {
    "Asset": "Other Current Assets",
    "Liability": "Other Current Liabilities",
    "Equity": "Other Reserves",
    "Income": "Revenue",
    "Expense": "Operating Expenses",
}

# --------------------------------------------------------------------------- #
# Default rules (edit in the UI or in config/classification_rules.json)
# --------------------------------------------------------------------------- #
DEFAULT_RULES: dict = {
    "version": 1,
    "priority": ["override", "code_range", "keyword", "first_digit"],
    # Number of leading digits of the account code compared with the ranges.
    # Codes shorter than this are right-padded with zeros ("101" -> "1010").
    "code_digits": 4,
    "code_ranges": [
        {"from": 1000, "to": 1499, "sub_category": "Non-Current Assets"},
        {"from": 1500, "to": 1599, "sub_category": "Inventory"},
        {"from": 1600, "to": 1699, "sub_category": "Trade Receivables"},
        {"from": 1700, "to": 1799, "sub_category": "Other Current Assets"},
        {"from": 1800, "to": 1999, "sub_category": "Cash & Bank"},
        {"from": 2000, "to": 2099, "sub_category": "Trade Payables"},
        {"from": 2100, "to": 2199, "sub_category": "Tax Payable"},
        {"from": 2200, "to": 2299, "sub_category": "Borrowings - Current"},
        {"from": 2300, "to": 2499, "sub_category": "Other Current Liabilities"},
        {"from": 2500, "to": 2999, "sub_category": "Non-Current Liabilities"},
        {"from": 3000, "to": 3099, "sub_category": "Share Capital"},
        {"from": 3100, "to": 3199, "sub_category": "Other Reserves"},
        {"from": 3200, "to": 3999, "sub_category": "Retained Earnings"},
        {"from": 4000, "to": 4799, "sub_category": "Revenue"},
        {"from": 4800, "to": 4999, "sub_category": "Other Income"},
        {"from": 5000, "to": 5999, "sub_category": "Cost of Sales"},
        {"from": 6000, "to": 6299, "sub_category": "Staff Costs"},
        {"from": 6300, "to": 7499, "sub_category": "Operating Expenses"},
        {"from": 7500, "to": 7699, "sub_category": "Depreciation & Amortisation"},
        {"from": 7700, "to": 7899, "sub_category": "Finance Costs"},
        {"from": 7900, "to": 7999, "sub_category": "Income Tax"},
        {"from": 8000, "to": 9999, "sub_category": "Operating Expenses"},
    ],
    # Ordered: first keyword found in the (lower-case) account name wins.
    # Specific phrases are listed before generic words on purpose
    # (e.g. "accumulated depreciation" before "depreciation", "bank charges" before "bank").
    "keywords": [
        {"keywords": ["accumulated depreciation", "accumulated amortisation", "accumulated amortization",
                      "provision for depreciation", "acc. dep", "acc dep"], "sub_category": "Non-Current Assets"},
        {"keywords": ["depreciation", "amortisation", "amortization", "impairment"],
         "sub_category": "Depreciation & Amortisation"},
        {"keywords": ["corporation tax expense", "income tax expense", "tax charge", "deferred tax expense",
                      "current tax expense"], "sub_category": "Income Tax"},
        {"keywords": ["corporation tax payable", "income tax payable", "tax payable", "instalment tax",
                      "installment tax", "corporation tax"], "sub_category": "Tax Payable"},
        {"keywords": ["interest expense", "interest paid", "interest on loan", "loan interest",
                      "overdraft interest", "finance cost", "finance charge", "interest on overdraft",
                      "lease interest"], "sub_category": "Finance Costs"},
        {"keywords": ["interest income", "interest received", "dividend income", "gain on disposal",
                      "exchange gain", "forex gain", "rental income", "other income", "sundry income",
                      "commission income", "discount received"], "sub_category": "Other Income"},
        {"keywords": ["overdraft", "short term loan", "short-term loan", "current portion"],
         "sub_category": "Borrowings - Current"},
        {"keywords": ["long term loan", "long-term loan", "term loan", "bank loan", "debenture",
                      "lease liability", "deferred tax liability", "director loan", "directors loan",
                      "shareholder loan"], "sub_category": "Non-Current Liabilities"},
        {"keywords": ["bank charges", "bank fees", "transaction charges"], "sub_category": "Operating Expenses"},
        {"keywords": ["staff advance", "salary advance", "vat receivable", "input vat", "withholding tax receivable",
                      "wht receivable", "prepayment", "prepaid"], "sub_category": "Other Current Assets"},
        {"keywords": ["cash", "bank", "m-pesa", "mpesa", "petty cash", "float"],
         "sub_category": "Cash & Bank"},
        {"keywords": ["salaries", "salary", "wages", "payroll", "staff", "nssf", "nhif", "shif",
                      "housing levy", "gratuity", "pension", "leave pay", "bonus", "allowance",
                      "medical cover", "staff welfare", "casual labour", "casuals"], "sub_category": "Staff Costs"},
        {"keywords": ["cost of sales", "cost of goods", "purchases", "raw material", "direct labour",
                      "direct labor", "factory overhead", "freight inward", "carriage inward",
                      "production", "consumables used"], "sub_category": "Cost of Sales"},
        {"keywords": ["sales", "revenue", "turnover", "fees earned", "service income", "income from"],
         "sub_category": "Revenue"},
        {"keywords": ["inventory", "stock", "work in progress", "wip", "goods in transit", "finished goods"],
         "sub_category": "Inventory"},
        {"keywords": ["debtors", "receivable", "trade debtors", "accounts receivable", "customer"],
         "sub_category": "Trade Receivables"},
        {"keywords": ["prepayment", "prepaid", "deposit paid", "advance to", "vat receivable",
                      "input vat", "withholding tax receivable", "wht receivable", "staff advance",
                      "other receivable"], "sub_category": "Other Current Assets"},
        {"keywords": ["creditors", "payable", "accounts payable", "trade creditors", "supplier"],
         "sub_category": "Trade Payables"},
        {"keywords": ["accrual", "accrued", "vat payable", "output vat", "paye", "withholding tax payable",
                      "wht payable", "deferred income", "customer deposit", "dividend payable",
                      "provision"], "sub_category": "Other Current Liabilities"},
        {"keywords": ["share capital", "ordinary shares", "paid up capital", "issued capital"],
         "sub_category": "Share Capital"},
        {"keywords": ["retained earnings", "retained profit", "accumulated profit", "accumulated loss",
                      "profit and loss account", "revenue reserve"], "sub_category": "Retained Earnings"},
        {"keywords": ["reserve", "revaluation", "share premium", "capital reserve"],
         "sub_category": "Other Reserves"},
        {"keywords": ["land", "building", "plant", "machinery", "equipment", "motor vehicle", "furniture",
                      "fixtures", "computer", "intangible", "goodwill", "software", "right of use",
                      "investment property", "capital work in progress", "leasehold"],
         "sub_category": "Non-Current Assets"},
        {"keywords": ["loan", "borrowing"], "sub_category": "Non-Current Liabilities"},
        {"keywords": ["expense", "rent", "rates", "electricity", "water", "utilities", "marketing",
                      "advertising", "transport", "travel", "fuel", "repairs", "maintenance",
                      "professional fees", "audit fees", "legal", "consultancy", "insurance",
                      "telephone", "internet", "office", "printing", "stationery", "licence",
                      "license", "subscription", "security", "cleaning", "entertainment",
                      "donation", "bad debt", "penalt", "commission", "training"],
         "sub_category": "Operating Expenses"},
    ],
    # Fallback: first digit of the account code -> category (default sub-category applied)
    "first_digit": {"1": "Asset", "2": "Liability", "3": "Equity", "4": "Income",
                    "5": "Expense", "6": "Expense", "7": "Expense", "8": "Expense", "9": "Expense"},
}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _read_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, json.JSONDecodeError):
        pass
    return copy.deepcopy(default)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_rules() -> dict:
    """Load saved rules, filling any missing keys from the defaults."""
    rules = _read_json(RULES_FILE, DEFAULT_RULES)
    for key, val in DEFAULT_RULES.items():
        rules.setdefault(key, copy.deepcopy(val))
    return rules


def save_rules(rules: dict) -> None:
    _write_json(RULES_FILE, rules)


def reset_rules() -> dict:
    rules = copy.deepcopy(DEFAULT_RULES)
    save_rules(rules)
    return rules


def load_overrides() -> dict[str, str]:
    """Manual overrides: {account_code: sub_category}."""
    data = _read_json(OVERRIDES_FILE, {})
    return {str(k): v for k, v in data.items() if v in CATEGORY_OF}


def save_overrides(overrides: dict[str, str]) -> None:
    _write_json(OVERRIDES_FILE, {str(k): v for k, v in sorted(overrides.items()) if v in CATEGORY_OF})


# --------------------------------------------------------------------------- #
# Classification engine
# --------------------------------------------------------------------------- #
def _code_number(code: str, digits: int) -> int | None:
    """Leading `digits` digits of the code as an int, right-padded with zeros."""
    d = re.sub(r"\D", "", str(code))
    if not d:
        return None
    d = d[:digits] if len(d) >= digits else d.ljust(digits, "0")
    return int(d)


def classify_account(code: str, name: str, rules: dict, overrides: dict[str, str]) -> tuple[str, str]:
    """Return (sub_category, rule_source) for a single account."""
    code = str(code).strip()
    lname = str(name or "").lower()
    digits = int(rules.get("code_digits", 4) or 4)

    for layer in rules.get("priority", DEFAULT_RULES["priority"]):
        if layer == "override":
            sub = overrides.get(code)
            if sub in CATEGORY_OF:
                return sub, "Manual override"
        elif layer == "code_range":
            num = _code_number(code, digits)
            if num is None:
                continue
            for r in rules.get("code_ranges", []):
                try:
                    lo, hi = int(r["from"]), int(r["to"])
                except (KeyError, TypeError, ValueError):
                    continue
                if lo <= num <= hi and r.get("sub_category") in CATEGORY_OF:
                    return r["sub_category"], f"Code range {lo}-{hi}"
        elif layer == "keyword":
            if not lname:
                continue
            for r in rules.get("keywords", []):
                for kw in r.get("keywords", []):
                    kw = str(kw).strip().lower()
                    if kw and kw in lname and r.get("sub_category") in CATEGORY_OF:
                        return r["sub_category"], f"Keyword '{kw}'"
        elif layer == "first_digit":
            d = re.sub(r"\D", "", code)[:1]
            cat = rules.get("first_digit", {}).get(d)
            if cat in DEFAULT_SUB_FOR_CATEGORY:
                return DEFAULT_SUB_FOR_CATEGORY[cat], f"First-digit fallback ({d})"
    return UNCLASSIFIED, "No rule matched"


def classify_accounts(accounts: pd.DataFrame, rules: dict | None = None,
                      overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Classify a frame of unique accounts.

    Parameters
    ----------
    accounts : DataFrame with columns ``account_code`` and ``account_name``.

    Returns
    -------
    DataFrame: account_code, account_name, category, sub_category, rule_source
    """
    rules = rules or load_rules()
    overrides = overrides if overrides is not None else load_overrides()
    acc = accounts[["account_code", "account_name"]].drop_duplicates("account_code").copy()
    res = [classify_account(c, n, rules, overrides) for c, n in zip(acc["account_code"], acc["account_name"])]
    acc["sub_category"] = [r[0] for r in res]
    acc["rule_source"] = [r[1] for r in res]
    acc["category"] = acc["sub_category"].map(CATEGORY_OF)
    return acc[["account_code", "account_name", "category", "sub_category", "rule_source"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Helpers for the rules editor UI
# --------------------------------------------------------------------------- #
def ranges_to_frame(rules: dict) -> pd.DataFrame:
    df = pd.DataFrame(rules.get("code_ranges", []), columns=["from", "to", "sub_category"])
    return df


def frame_to_ranges(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.dropna(subset=["from", "to", "sub_category"]).iterrows():
        try:
            out.append({"from": int(r["from"]), "to": int(r["to"]), "sub_category": str(r["sub_category"])})
        except (TypeError, ValueError):
            continue
    return out


def keywords_to_frame(rules: dict) -> pd.DataFrame:
    rows = [{"keywords": ", ".join(r.get("keywords", [])), "sub_category": r.get("sub_category")}
            for r in rules.get("keywords", [])]
    return pd.DataFrame(rows, columns=["keywords", "sub_category"])


def frame_to_keywords(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.dropna(subset=["keywords", "sub_category"]).iterrows():
        kws = [k.strip().lower() for k in str(r["keywords"]).split(",") if k.strip()]
        if kws:
            out.append({"keywords": kws, "sub_category": str(r["sub_category"])})
    return out
