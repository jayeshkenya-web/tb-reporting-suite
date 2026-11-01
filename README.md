# Trial Balance → Financial & Management Reporting Suite

A Streamlit app that turns one or more Excel/CSV trial balances (multi-year and/or month-wise, actuals and budgets) into a full pack of financial statements and management reports, with Excel and PDF export.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Then click **🧪 Load sample data** → **🚀 Process data** to see every report working, or drop in your own files.

## Project structure

```
app.py                       Streamlit UI (upload, mapping, classification, validation, reports, export)
utils/data_processing.py     Reading, header detection, column auto-mapping, cleaning, ledger build, validation, sample data
utils/account_classifier.py  Configurable account classification (overrides → code ranges → keywords → first digit)
utils/report_generator.py    All reports + Excel and PDF export
config/                      Saved settings, column mappings, classification rules and overrides (JSON)
sample_data/                 Demo trial balances (annual 2023-24, monthly 2025 by cost centre, budget 2025)
requirements.txt
```

## Expected input

The layout is one row per account (per period, per cost centre). Headers are matched automatically, and you can re-map them in the UI. The app ignores title rows above the table and *Total* rows.

| Field | Required | Accepted headers (examples) |
|---|---|---|
| Account Code | ✅ | Account Code, GL Code, A/C No |
| Account Name | recommended | Account Name, Description |
| Debit + Credit **or** Net Balance | ✅ | Debit/Dr, Credit/Cr, Balance, Closing Balance |
| Opening Balance | optional | Opening, B/F |
| Year | optional | Year, FY, `2024/25` (= FY2025) |
| Month / Period | optional | Jan, January, 3, P03 |
| Period Date | optional | Period End, As at |
| Company | optional | Company, Entity |
| Cost Centre | optional | Cost Centre, Department, Branch |
| Budget | optional | Budget, Plan |

The parser understands amounts in the forms `(1,234)`, `1,234-`, `1,234 Cr`, `KES 1,234`. If there is no Year column, the year is taken from the file or sheet name (`TB_2024.xlsx`, `Mar 2025`). A file with no month is treated as a year-end (annual) TB.

**Example (month-wise, closing basis):**

| Company | Year | Month | Cost Centre | Account Code | Account Name | Debit | Credit |
|---|---|---|---|---|---|---|---|
| Demo Kenya Ltd | 2025 | Jan | | 1810 | Bank - KCB | 4,108,461 | |
| Demo Kenya Ltd | 2025 | Jan | Sales | 4010 | Sales - Local | | 17,514,471 |

## Key settings (sidebar → ⚙️ Settings)

* **TB basis.** Use *Closing balances* for standard ERP TBs, where P&L accounts are YTD. Use *Period movements* for activity-only files, optionally with an Opening Balance column. The app derives both movement and closing balances either way.
* **Fiscal year start.** For example, July for a June year-end. The fiscal year label is the year in which the fiscal year **ends**.
* **Budget.** Upload budgets as separate files with role = *Budget*. Set whether the amounts are monthly or cumulative, and whether they use the natural sign (all positive) or Dr+/Cr−.
* **Units.** Choose units, thousands or millions, and the currency label.

## Classification

Each account gets a **sub-category**, and the category follows from it automatically:

* Assets: Non-Current Assets · Inventory · Trade Receivables · Other Current Assets · Cash & Bank
* Liabilities: Non-Current Liabilities · Borrowings – Current · Trade Payables · Tax Payable · Other Current Liabilities
* Equity: Share Capital · Other Reserves · Retained Earnings
* Income: Revenue · Other Income
* Expenses: Cost of Sales · Staff Costs · Operating Expenses · Depreciation & Amortisation · Finance Costs · Income Tax

The rules are tried in order: manual override, then code range, then keyword, then the first-digit fallback. Edit and save them on the Classification page. They are stored in `config/` and re-applied to future uploads. You can export the rules as JSON and import them on another machine.

## Reports

**Financial:** P&L (monthly, quarterly, multi-year, with optional account detail); Balance Sheet with balancing check; Cash Flow (indirect method, reconciled to the cash movement); Trial Balance summary and detail; Schedule of Accounts and a period-by-period ledger for any account.

**Management:** Variance analysis (actual vs prior year and vs budget, by line or account, for a month, quarter, YTD or full year); key ratios (liquidity, profitability, leverage, efficiency); trend analysis; cost-centre P&L; top 10 expenses and income; working capital with DSO, DIO, DPO and cash conversion cycle; monthly movement of major accounts.

**Validation:** Unbalanced TBs, unclassified or fallback-classified accounts, missing names, duplicate lines, month gaps, different period coverage across companies, suspense balances, contra balances, and balance sheet checks.

**Export:** Excel pack (formatted, one sheet per report, index with hyperlinks) and PDF pack (landscape A4, contents page). Normalised ledger and classification are available as CSV.

## Notes and limitations

* Multi-company figures are a simple aggregation. Intercompany eliminations are not performed.
* The cost-centre filter applies to P&L-type reports. The Balance Sheet, Cash Flow, ratios and working capital always use the whole entity.
* Closing-basis TBs only give the net movement per account. Debits and credits in schedules are a split of that net movement.
* The cash flow is derived from balance sheet movements. Capex is shown net of disposals. Bank overdrafts are treated as financing; classify them as *Cash & Bank* if you treat them as cash equivalents under IAS 7.
* The first period ever loaded on a closing basis has no prior balance, so the cash flow starts from the second period.
* Month, quarter and YTD prior-year comparisons need prior-year **monthly** TBs. Annual TBs support full-year comparisons.
