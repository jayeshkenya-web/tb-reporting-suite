"""
utils/report_generator.py
=========================

Builds every financial and management report from the normalised ledger
(see data_processing.build_ledger) and exports them to Excel and PDF.

Sign conventions used in the reports
------------------------------------
* Ledger amounts are debit-positive.
* P&L statements show the credit-positive view: income positive, expenses
  negative, so every subtotal is a simple sum and a POSITIVE variance is always
  FAVOURABLE (more income or less expense).
* Balance sheet: assets positive; equity and liabilities positive.
* Cash flow: inflows positive, outflows negative.

Every statement DataFrame has:
    "Line" column, one column per period/entity, and a hidden "_type" column
    (header | line | account | subtotal | total | check | note) used for styling
    on screen, in Excel and in PDF.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from account_classifier import UNCLASSIFIED
from data_processing import NO_CC, period_label

# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


@dataclass
class ReportContext:
    """All inputs a report needs. Build once per rerun from the sidebar filters."""
    ledger: pd.DataFrame            # entity-level ledger (company filter applied, ALL cost centres)
    pl_ledger: pd.DataFrame         # cost-centre filtered ledger (used for P&L-type reports)
    budget: pd.DataFrame | None     # budget ledger (cost-centre filtered) or None
    year: int                       # selected fiscal year
    period: int                     # selected (as-at) fiscal period
    basis: str = "closing"
    fy_start: int = 1
    carry_forward_pl: bool = True
    has_opening: bool = False
    entity_label: str = ""
    units_label: str = "KES"
    cost_centre_label: str = "All cost centres"

    @property
    def pkey(self) -> int:
        return self.year * 100 + self.period

    def label(self, year: int, period: int) -> str:
        return period_label(year, period, self.fy_start)

    @property
    def as_at_label(self) -> str:
        return self.label(self.year, self.period)


def available_periods(ledger: pd.DataFrame) -> list[tuple[int, int]]:
    if ledger is None or ledger.empty:
        return []
    p = ledger[["year", "period"]].drop_duplicates().sort_values(["year", "period"])
    return [tuple(map(int, x)) for x in p.to_numpy()]


# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #
PL_ALL = ["Revenue", "Cost of Sales", "Other Income", "Staff Costs", "Operating Expenses",
          "Depreciation & Amortisation", "Finance Costs", "Income Tax"]
PL_LAYOUT = [
    ("Revenue", "line", ["Revenue"]),
    ("Cost of Sales", "line", ["Cost of Sales"]),
    ("Gross Profit", "subtotal", ["Revenue", "Cost of Sales"]),
    ("Other Income", "line", ["Other Income"]),
    ("Staff Costs", "line", ["Staff Costs"]),
    ("Operating Expenses", "line", ["Operating Expenses"]),
    ("EBITDA", "subtotal", ["Revenue", "Cost of Sales", "Other Income", "Staff Costs", "Operating Expenses"]),
    ("Depreciation & Amortisation", "line", ["Depreciation & Amortisation"]),
    ("Operating Profit (EBIT)", "subtotal", PL_ALL[:6]),
    ("Finance Costs", "line", ["Finance Costs"]),
    ("Profit Before Tax", "subtotal", PL_ALL[:7]),
    ("Income Tax", "line", ["Income Tax"]),
    ("Profit After Tax", "total", PL_ALL),
]
CA = ["Inventory", "Trade Receivables", "Other Current Assets", "Cash & Bank"]
CL = ["Trade Payables", "Tax Payable", "Borrowings - Current", "Other Current Liabilities"]
EQ = ["Share Capital", "Other Reserves", "Retained Earnings", "__PRIOR_PL", "__CURR_PL"]
BS_LAYOUT = [
    ("ASSETS", "header", []),
    ("Non-Current Assets", "line", ["Non-Current Assets"]),
    ("Current Assets", "header", []),
    ("Inventory", "line", ["Inventory"]),
    ("Trade Receivables", "line", ["Trade Receivables"]),
    ("Other Current Assets", "line", ["Other Current Assets"]),
    ("Cash & Bank", "line", ["Cash & Bank"]),
    ("Total Current Assets", "subtotal", CA),
    ("TOTAL ASSETS", "total", ["Non-Current Assets"] + CA),
    ("EQUITY AND LIABILITIES", "header", []),
    ("Share Capital", "line", ["Share Capital"]),
    ("Other Reserves", "line", ["Other Reserves"]),
    ("Retained Earnings", "line", ["Retained Earnings"]),
    ("Prior Years' Profit (derived)", "line", ["__PRIOR_PL"]),
    ("Profit for the Period (YTD)", "line", ["__CURR_PL"]),
    ("Total Equity", "subtotal", EQ),
    ("Non-Current Liabilities", "line", ["Non-Current Liabilities"]),
    ("Current Liabilities", "header", []),
    ("Trade Payables", "line", ["Trade Payables"]),
    ("Tax Payable", "line", ["Tax Payable"]),
    ("Borrowings - Current", "line", ["Borrowings - Current"]),
    ("Other Current Liabilities", "line", ["Other Current Liabilities"]),
    ("Total Current Liabilities", "subtotal", CL),
    ("Total Liabilities", "subtotal", ["Non-Current Liabilities"] + CL),
    ("TOTAL EQUITY AND LIABILITIES", "total", EQ + ["Non-Current Liabilities"] + CL),
]
ASSET_SUBS = {"Non-Current Assets"} | set(CA)


def _bs_sign(sub: str) -> int:
    return 1 if sub in ASSET_SUBS else -1


def _build_statement(values: pd.DataFrame, layout, sign_fn, detail: pd.DataFrame | None = None,
                     hide_zero_lines: tuple[str, ...] = ()) -> pd.DataFrame:
    """
    values : index = sub_category, columns = report columns (debit-positive sums)
    detail : index = (sub_category, account_code, account_name), same columns
    """
    cols = list(values.columns)
    rows = []
    for label, kind, subs in layout:
        if kind == "header":
            rows.append({"Line": label, **{c: np.nan for c in cols}, "_type": "header"})
            continue
        vals = {c: 0.0 for c in cols}
        for s in subs:
            if s in values.index:
                for c in cols:
                    vals[c] += sign_fn(s) * float(values.at[s, c])
        if kind == "line" and label in hide_zero_lines and all(abs(v) < 0.005 for v in vals.values()):
            continue
        rows.append({"Line": label, **vals, "_type": kind})
        if kind == "line" and detail is not None and not detail.empty:
            for s in subs:
                if s not in detail.index.get_level_values(0):
                    continue
                d = detail.loc[s]
                for (code, name), r in d.iterrows():
                    if r.abs().sum() < 0.005:
                        continue
                    rows.append({"Line": f"    {code}  {name}", **{c: sign_fn(s) * float(r[c]) for c in cols},
                                 "_type": "account"})
    return pd.DataFrame(rows, columns=["Line"] + cols + ["_type"])


def _pivot(df: pd.DataFrame, value: str, col: str, col_order: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame(columns=col_order, dtype=float)
        return empty, empty
    v = df.pivot_table(index="sub_category", columns=col, values=value, aggfunc="sum", fill_value=0.0)
    d = df.pivot_table(index=["sub_category", "account_code", "account_name"], columns=col, values=value,
                       aggfunc="sum", fill_value=0.0)
    return v.reindex(columns=col_order, fill_value=0.0), d.reindex(columns=col_order, fill_value=0.0)


def _add_change_cols(df: pd.DataFrame, cur: str, prev: str) -> pd.DataFrame:
    df = df.copy()
    df["Change"] = df[cur] - df[prev]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["Change %"] = np.where(df[prev].abs() > 0.005, df["Change"] / df[prev].abs(), np.nan)
    df.loc[df["_type"] == "header", ["Change", "Change %"]] = np.nan
    df.loc[df["_type"] == "check", ["Change %"]] = np.nan
    return df[[c for c in df.columns if c != "_type"] + ["_type"]]


# --------------------------------------------------------------------------- #
# Period helpers
# --------------------------------------------------------------------------- #
def _year_periods(ledger: pd.DataFrame, year: int, upto: int | None = None) -> list[int]:
    ps = sorted(ledger.loc[ledger["year"] == year, "period"].unique().tolist())
    return [int(p) for p in ps if upto is None or p <= upto]


def _years(ledger: pd.DataFrame, upto: int) -> list[int]:
    return [int(y) for y in sorted(ledger["year"].unique()) if y <= upto]


def _year_col(ledger: pd.DataFrame, y: int, upto: int | None = None) -> str:
    ps = _year_periods(ledger, y, upto)
    if not ps:
        return f"FY{y}"
    return f"FY{y}" if max(ps) == 12 else f"FY{y} YTD P{max(ps):02d}"


def window_rows(ctx: ReportContext, scope: str, year: int | None = None, period: int | None = None,
                df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows of `df` (default pl_ledger) whose movements fall in the scope window."""
    df = ctx.pl_ledger if df is None else df
    y, p = year or ctx.year, period or ctx.period
    if scope == "Month":
        m = (df["year"] == y) & (df["period"] == p)
    elif scope == "Quarter":
        q0 = (p - 1) // 3 * 3 + 1
        m = (df["year"] == y) & df["period"].between(q0, p)
    elif scope == "Full Year":
        m = df["year"] == y
    else:  # YTD
        m = (df["year"] == y) & (df["period"] <= p)
    return df[m]


def scope_label(ctx: ReportContext, scope: str, year: int | None = None, period: int | None = None) -> str:
    y, p = year or ctx.year, period or ctx.period
    if scope == "Month":
        return ctx.label(y, p)
    if scope == "Quarter":
        q0 = (p - 1) // 3 * 3 + 1
        return f"Q{(p - 1) // 3 + 1} FY{y} (P{q0:02d}-P{p:02d})"
    if scope == "Full Year":
        return f"FY{y}"
    return f"YTD FY{y} to {ctx.label(y, p)}"


# --------------------------------------------------------------------------- #
# 1. Profit & Loss
# --------------------------------------------------------------------------- #
def profit_and_loss(ctx: ReportContext, view: str = "Monthly", detailed: bool = False,
                    like_for_like: bool = True) -> pd.DataFrame:
    """
    view: Monthly (periods of selected year + YTD) | Quarterly (+ YTD) |
          Yearly (multi-year; like_for_like restricts every year to the same periods).
    """
    led = ctx.pl_ledger[ctx.pl_ledger["is_pl"]]
    if view == "Yearly":
        # Prior years: full year, or same periods as the selected YTD when like_for_like and the
        # prior year has monthly data. Current year: YTD to the selected period.
        df = led[led["year"] <= ctx.year]
        parts, order = [], []
        for y in _years(df, ctx.year):
            yd = df[df["year"] == y]
            ps = set(yd["period"].unique())
            if y == ctx.year:
                yd = yd[yd["period"] <= ctx.period]
                lab = _year_col(df, y, ctx.period)
            elif like_for_like and ctx.period < 12 and ctx.period in ps and min(ps) == 1:
                yd = yd[yd["period"] <= ctx.period]
                lab = f"FY{y} YTD P{ctx.period:02d}"
            else:
                lab = _year_col(df, y)
            parts.append(yd.assign(col=lab))
            order.append(lab)
        df = pd.concat(parts) if parts else led.iloc[0:0].assign(col=[])
        v, d = _pivot(df, "movement", "col", order)
        out = _build_statement(v, PL_LAYOUT, lambda s: -1, d if detailed else None)
        if len(order) >= 2 and order[-1].split(" ", 1)[1:] == order[-2].split(" ", 1)[1:]:
            out = _add_change_cols(out, order[-1], order[-2])
        return out
    df = led[(led["year"] == ctx.year) & (led["period"] <= ctx.period)].copy()
    if view == "Quarterly":
        df["col"] = df["period"].map(lambda p: f"Q{(p - 1) // 3 + 1}")
        order = sorted(df["col"].unique())
    else:
        df["col"] = df["period"].map(lambda p: ctx.label(ctx.year, p))
        order = [ctx.label(ctx.year, p) for p in _year_periods(df, ctx.year)]
    v, d = _pivot(df, "movement", "col", order)
    if not v.empty or order:
        v["YTD"] = v.sum(axis=1)
        if not d.empty:
            d["YTD"] = d.sum(axis=1)
    return _build_statement(v, PL_LAYOUT, lambda s: -1, d if detailed else None)


def pl_metrics(movements: pd.DataFrame) -> dict[str, float]:
    """Key P&L subtotals (credit-positive) from ledger rows."""
    s = movements[movements["is_pl"]].groupby("sub_category")["movement"].sum()
    g = lambda k: -float(s.get(k, 0.0))  # noqa: E731
    rev, cos = g("Revenue"), g("Cost of Sales")
    gp = rev + cos
    ebitda = gp + g("Other Income") + g("Staff Costs") + g("Operating Expenses")
    ebit = ebitda + g("Depreciation & Amortisation")
    pbt = ebit + g("Finance Costs")
    pat = pbt + g("Income Tax")
    return {"Revenue": rev, "Cost of Sales": -cos, "Gross Profit": gp, "Other Income": g("Other Income"),
            "Operating Expenses": -(g("Staff Costs") + g("Operating Expenses")), "EBITDA": ebitda,
            "Depreciation": -g("Depreciation & Amortisation"), "EBIT": ebit,
            "Finance Costs": -g("Finance Costs"), "PBT": pbt, "Income Tax": -g("Income Tax"), "PAT": pat}


# --------------------------------------------------------------------------- #
# 2. Balance Sheet
# --------------------------------------------------------------------------- #
def bs_columns(ctx: ReportContext, view: str) -> list[tuple[str, int, int]]:
    """(label, year, period) as-at points for Balance Sheet / ratios / working capital."""
    led = ctx.ledger
    if view == "Yearly":
        cols = []
        for y in _years(led, ctx.year):
            ps = _year_periods(led, y, ctx.period if y == ctx.year else None)
            if ps:
                cols.append((ctx.label(y, max(ps)), y, max(ps)))
        return cols
    ps = _year_periods(led, ctx.year, ctx.period)
    if view == "Quarterly":
        sel = [p for p in ps if p % 3 == 0]
        if ps and ps[-1] not in sel:
            sel.append(ps[-1])
        ps = sel
    return [(ctx.label(ctx.year, p), ctx.year, p) for p in ps]


def _bs_values(ctx: ReportContext, points: list[tuple[str, int, int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    led = ctx.ledger
    order = [p[0] for p in points]
    frames, extra = [], {}
    for label, y, p in points:
        part = led[(led["year"] == y) & (led["period"] == p)].copy()
        part["col"] = label
        frames.append(part)
        curr = part.loc[part["is_pl"], "closing"].sum()
        prior = 0.0
        if ctx.basis == "movement" and ctx.carry_forward_pl:
            prior = led.loc[led["is_pl"] & (led["year"] < y), "movement"].sum()
        extra[label] = {"__CURR_PL": curr, "__PRIOR_PL": prior}
    df = pd.concat(frames) if frames else led.iloc[0:0].assign(col=[])
    bs = df[~df["is_pl"]]
    v, d = _pivot(bs, "closing", "col", order)
    for key in ("__CURR_PL", "__PRIOR_PL"):
        v.loc[key] = [extra[c][key] for c in order]
    if UNCLASSIFIED not in v.index:
        v.loc[UNCLASSIFIED] = 0.0
    return v, d


def balance_sheet(ctx: ReportContext, view: str = "Monthly", detailed: bool = False) -> pd.DataFrame:
    points = bs_columns(ctx, view)
    v, d = _bs_values(ctx, points)
    out = _build_statement(v, BS_LAYOUT, _bs_sign, d if detailed else None,
                           hide_zero_lines=("Prior Years' Profit (derived)",))
    ta = out.loc[out["Line"] == "TOTAL ASSETS"].iloc[0]
    tel = out.loc[out["Line"] == "TOTAL EQUITY AND LIABILITIES"].iloc[0]
    cols = [p[0] for p in points]
    unc = {c: float(v.at[UNCLASSIFIED, c]) for c in cols}
    if any(abs(x) > 0.005 for x in unc.values()):
        out.loc[len(out)] = {"Line": "Unclassified balances (net, Dr +)", **unc, "_type": "note"}
    out.loc[len(out)] = {"Line": "Check: Assets - (Equity + Liabilities)",
                         **{c: round(ta[c] - tel[c], 2) for c in cols}, "_type": "check"}
    if len(cols) >= 2 and view == "Yearly":
        out = _add_change_cols(out, cols[-1], cols[-2])
    return out


# --------------------------------------------------------------------------- #
# 3. Cash Flow Statement (indirect method)
# --------------------------------------------------------------------------- #
def _cash_flow_column(ctx: ReportContext, rows: pd.DataFrame) -> dict[str, float]:
    """Compute one cash-flow column from the ledger rows of a window (entity level)."""
    led = ctx.ledger
    mv = rows.groupby("sub_category")["movement"].sum()
    m = lambda k: float(mv.get(k, 0.0))  # noqa: E731  (debit-positive movement)
    pl_subs = PL_ALL
    pbt = -sum(m(s) for s in pl_subs if s != "Income Tax")
    dep, fin, tax = m("Depreciation & Amortisation"), m("Finance Costs"), m("Income Tax")

    # Year-end close (closing basis): P&L balances transferred into retained earnings show as a
    # non-cash movement in Retained Earnings at the first period of a new year.
    transfer = 0.0
    if ctx.basis == "closing":
        for y in sorted(rows["year"].unique()):
            yp = _year_periods(led, int(y))
            prev = _year_periods(led, int(y) - 1)
            if yp and prev and ((rows["year"] == y) & (rows["period"] == yp[0])).any():
                transfer += led.loc[led["is_pl"] & (led["year"] == y - 1) & (led["period"] == prev[-1]),
                                    "closing"].sum()
    ops_before_wc = pbt + dep + fin
    wc = {"(Increase)/decrease in inventory": -m("Inventory"),
          "(Increase)/decrease in trade receivables": -m("Trade Receivables"),
          "(Increase)/decrease in other current assets": -m("Other Current Assets"),
          "Increase/(decrease) in trade payables": -m("Trade Payables"),
          "Increase/(decrease) in other current liabilities": -m("Other Current Liabilities")}
    cash_gen = ops_before_wc + sum(wc.values())
    tax_paid = -tax - m("Tax Payable")
    unclassified = -m(UNCLASSIFIED)
    net_ops = cash_gen + tax_paid + unclassified
    capex = -(m("Non-Current Assets") + dep)
    fin_lines = {"Proceeds from / (repayment of) long-term borrowings": -m("Non-Current Liabilities"),
                 "Proceeds from / (repayment of) short-term borrowings": -m("Borrowings - Current"),
                 "Share capital & reserve movements": -(m("Share Capital") + m("Other Reserves")),
                 "Dividends paid & other retained earnings movements": -m("Retained Earnings") + transfer,
                 "Finance costs paid": -fin}
    net_fin = sum(fin_lines.values())
    net_change = net_ops + capex + net_fin
    cash_end = rows.loc[rows["sub_category"] == "Cash & Bank"]
    # closing cash at end of window = closing of the last period in the window
    last_key = rows["pkey"].max() if not rows.empty else None
    end_cash = float(cash_end.loc[cash_end["pkey"] == last_key, "closing"].sum()) if last_key else 0.0
    actual_change = m("Cash & Bank")
    out = {"Profit before tax": pbt, "Depreciation & amortisation": dep, "Finance costs": fin,
           "Operating profit before working capital changes": ops_before_wc, **wc,
           "Cash generated from operations": cash_gen, "Income tax paid": tax_paid,
           "Movement in unclassified accounts": unclassified, "Net cash from operating activities": net_ops,
           "Net (purchase)/disposal of non-current assets": capex, "Net cash used in investing activities": capex,
           **fin_lines, "Net cash from financing activities": net_fin,
           "Net increase/(decrease) in cash & bank": net_change,
           "Cash & bank at beginning of period": end_cash - actual_change,
           "Cash & bank at end of period": end_cash,
           "Check: unreconciled difference": round(actual_change - net_change, 2)}
    return out


CF_LAYOUT = [
    ("CASH FLOWS FROM OPERATING ACTIVITIES", "header"),
    ("Profit before tax", "line"),
    ("Adjustments for:", "header"),
    ("Depreciation & amortisation", "line"),
    ("Finance costs", "line"),
    ("Operating profit before working capital changes", "subtotal"),
    ("(Increase)/decrease in inventory", "line"),
    ("(Increase)/decrease in trade receivables", "line"),
    ("(Increase)/decrease in other current assets", "line"),
    ("Increase/(decrease) in trade payables", "line"),
    ("Increase/(decrease) in other current liabilities", "line"),
    ("Cash generated from operations", "subtotal"),
    ("Income tax paid", "line"),
    ("Movement in unclassified accounts", "line"),
    ("Net cash from operating activities", "total"),
    ("CASH FLOWS FROM INVESTING ACTIVITIES", "header"),
    ("Net (purchase)/disposal of non-current assets", "line"),
    ("Net cash used in investing activities", "total"),
    ("CASH FLOWS FROM FINANCING ACTIVITIES", "header"),
    ("Proceeds from / (repayment of) long-term borrowings", "line"),
    ("Proceeds from / (repayment of) short-term borrowings", "line"),
    ("Share capital & reserve movements", "line"),
    ("Dividends paid & other retained earnings movements", "line"),
    ("Finance costs paid", "line"),
    ("Net cash from financing activities", "total"),
    ("Net increase/(decrease) in cash & bank", "total"),
    ("Cash & bank at beginning of period", "line"),
    ("Cash & bank at end of period", "subtotal"),
    ("Check: unreconciled difference", "check"),
]


def cash_flow(ctx: ReportContext, view: str = "Monthly") -> pd.DataFrame:
    led = ctx.ledger
    first_key = int(led["pkey"].min()) if not led.empty else 0
    cols: list[tuple[str, pd.DataFrame]] = []
    if view == "Yearly":
        for y in _years(led, ctx.year):
            rows = led[(led["year"] == y) & ((led["year"] < ctx.year) | (led["period"] <= ctx.period))]
            cols.append((_year_col(led, y, ctx.period if y == ctx.year else None), rows))
    else:
        yr = led[(led["year"] == ctx.year) & (led["period"] <= ctx.period)]
        if view == "Quarterly":
            for q in sorted({(p - 1) // 3 + 1 for p in yr["period"].unique()}):
                cols.append((f"Q{q}", yr[yr["period"].between(q * 3 - 2, q * 3)]))
        else:
            for p in _year_periods(yr, ctx.year):
                cols.append((ctx.label(ctx.year, p), yr[yr["period"] == p]))
        cols.append(("YTD", yr))
    data, notes = {}, []
    for label, rows in cols:
        # In closing basis the very first period has no prior balance -> movements are not cash flows
        if ctx.basis == "closing" and not ctx.has_opening and (rows["pkey"] == first_key).any():
            data[label] = {k: np.nan for k, _ in CF_LAYOUT}
            notes.append(label)
        else:
            data[label] = _cash_flow_column(ctx, rows)
    out_rows = []
    for line, kind in CF_LAYOUT:
        out_rows.append({"Line": line, **{c: (np.nan if kind == "header" else data[c].get(line, np.nan))
                                          for c in data}, "_type": kind})
    out = pd.DataFrame(out_rows, columns=["Line"] + list(data.keys()) + ["_type"])
    if notes:
        out.loc[len(out)] = {"Line": f"Note: no opening balances before {', '.join(notes)} - cash flow not "
                                     "computable for the first period loaded.", **{c: np.nan for c in data},
                             "_type": "note"}
    return out


# --------------------------------------------------------------------------- #
# 4. Trial Balance summary & schedules
# --------------------------------------------------------------------------- #
def trial_balance(ctx: ReportContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary by category/sub-category, detailed TB) at the selected period (closing balances)."""
    led = ctx.ledger
    cur = led[led["pkey"] == ctx.pkey]
    det = (cur.groupby(["account_code", "account_name", "category", "sub_category"], as_index=False)["closing"].sum())
    det = det[det["closing"].abs() > 0.005]
    det["Debit"] = det["closing"].clip(lower=0)
    det["Credit"] = (-det["closing"]).clip(lower=0)
    det = det.rename(columns={"account_code": "Account Code", "account_name": "Account Name",
                              "category": "Category", "sub_category": "Sub-category", "closing": "Net (Dr +)"})
    det = det[["Account Code", "Account Name", "Category", "Sub-category", "Debit", "Credit", "Net (Dr +)"]]
    tot = {"Account Code": "", "Account Name": "TOTAL", "Category": "", "Sub-category": "",
           "Debit": det["Debit"].sum(), "Credit": det["Credit"].sum(), "Net (Dr +)": det["Net (Dr +)"].sum()}
    det_out = pd.concat([det, pd.DataFrame([tot])], ignore_index=True)
    summ = (det.groupby(["Category", "Sub-category"], as_index=False)
               .agg(Accounts=("Account Code", "count"), Debit=("Debit", "sum"), Credit=("Credit", "sum"),
                    Net=("Net (Dr +)", "sum")))
    cat_order = {"Asset": 0, "Liability": 1, "Equity": 2, "Income": 3, "Expense": 4, UNCLASSIFIED: 5}
    summ = summ.sort_values(by="Category", key=lambda s: s.map(cat_order), kind="stable")
    summ.loc[len(summ)] = ["TOTAL", "", summ["Accounts"].sum(), summ["Debit"].sum(), summ["Credit"].sum(),
                           summ["Net"].sum()]
    return summ.reset_index(drop=True), det_out


def schedule_of_accounts(ctx: ReportContext, scope: str = "YTD") -> pd.DataFrame:
    """Per-account opening, debits, credits and closing for the window, grouped by sub-category."""
    rows = window_rows(ctx, scope, df=ctx.ledger)
    if rows.empty:
        return pd.DataFrame()
    last_key = rows["pkey"].max()
    first_key = rows["pkey"].min()
    g = rows.groupby(["sub_category", "account_code", "account_name"])
    mv = g["movement"].sum()
    close = rows[rows["pkey"] == last_key].groupby(["sub_category", "account_code", "account_name"])["closing"].sum()
    open_ = rows[rows["pkey"] == first_key].groupby(["sub_category", "account_code", "account_name"])["opening"].sum()
    if ctx.basis == "movement":
        dr, cr = g["debit"].sum(), g["credit"].sum()
    else:  # closing basis: only net movement is known
        dr, cr = mv.clip(lower=0), (-mv).clip(lower=0)
    df = pd.DataFrame({"Opening": open_, "Debits": dr, "Credits": cr, "Net Movement": mv,
                       "Closing": close}).fillna(0.0).reset_index()
    df = df[(df[["Opening", "Debits", "Credits", "Closing"]].abs().sum(axis=1)) > 0.005]
    from account_classifier import ALL_SUB_CATEGORIES
    order = {s: i for i, s in enumerate(ALL_SUB_CATEGORIES)}
    out = []
    for sub in sorted(df["sub_category"].unique(), key=lambda s: order.get(s, 99)):
        part = df[df["sub_category"] == sub]
        out.append({"Line": sub.upper(), "_type": "header"})
        for _, r in part.iterrows():
            out.append({"Line": f"    {r['account_code']}  {r['account_name']}", "Opening": r["Opening"],
                        "Debits": r["Debits"], "Credits": r["Credits"], "Net Movement": r["Net Movement"],
                        "Closing": r["Closing"], "_type": "account"})
        out.append({"Line": f"Total {sub}", **part[["Opening", "Debits", "Credits", "Net Movement", "Closing"]].sum()
                    .to_dict(), "_type": "subtotal"})
    res = pd.DataFrame(out, columns=["Line", "Opening", "Debits", "Credits", "Net Movement", "Closing", "_type"])
    return res


def account_ledger(ctx: ReportContext, account_code: str) -> pd.DataFrame:
    """Period-by-period ledger (opening / movement / closing) for one account (all periods loaded)."""
    a = ctx.ledger[ctx.ledger["account_code"] == account_code]
    if a.empty:
        return pd.DataFrame()
    g = a.groupby(["year", "period"], as_index=False)[["opening", "debit", "credit", "movement", "closing"]].sum()
    g.insert(0, "Period", [ctx.label(y, p) for y, p in zip(g["year"], g["period"])])
    g = g.drop(columns=["year", "period"])
    if ctx.basis == "closing":
        g = g.drop(columns=["debit", "credit"])
    return g.rename(columns={"opening": "Opening", "debit": "Debits", "credit": "Credits",
                             "movement": "Movement", "closing": "Closing"})


# --------------------------------------------------------------------------- #
# 5. Variance analysis
# --------------------------------------------------------------------------- #
def variance_analysis(ctx: ReportContext, scope: str = "Month",
                      level: str = "Line") -> pd.DataFrame:
    """
    Actual vs Prior Year (same window) and vs Budget (if loaded).
    level: Line (P&L lines) | Account
    Positive variance = favourable (credit-positive presentation).
    """
    act = window_rows(ctx, scope)
    py = window_rows(ctx, scope, year=ctx.year - 1)
    bud = window_rows(ctx, scope, df=ctx.budget) if ctx.budget is not None and not ctx.budget.empty else None

    def by_level(df):
        if df is None or df.empty:
            return pd.Series(dtype=float)
        df = df[df["is_pl"]]
        if level == "Account":
            return -df.groupby(["sub_category", "account_code", "account_name"])["movement"].sum()
        s = df.groupby("sub_category")["movement"].sum()
        out = {}
        for label, kind, subs in PL_LAYOUT:
            out[label] = -sum(float(s.get(x, 0.0)) for x in subs)
        return pd.Series(out)

    a, p = by_level(act), by_level(py)
    data = {"Actual": a, "Prior Year": p}
    if bud is not None:
        data["Budget"] = by_level(bud)
    df = pd.DataFrame(data).fillna(0.0)
    if level == "Line":
        df = df.reindex([x[0] for x in PL_LAYOUT])
    else:
        df = df[(df.abs().sum(axis=1)) > 0.005]

    def pct(var, base):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(base.abs() > 0.005, var / base.abs(), np.nan)
    df["Var vs PY"] = df["Actual"] - df["Prior Year"]
    df["Var vs PY %"] = pct(df["Var vs PY"], df["Prior Year"])
    if "Budget" in df:
        df["Var vs Budget"] = df["Actual"] - df["Budget"]
        df["Var vs Budget %"] = pct(df["Var vs Budget"], df["Budget"])
    df["Status"] = np.where(df.get("Var vs Budget", df["Var vs PY"]) >= 0, "Favourable", "Adverse")
    df = df.reset_index()
    if level == "Line":
        df = df.rename(columns={"index": "Line"})
        kinds = {x[0]: x[1] for x in PL_LAYOUT}
        df["_type"] = df["Line"].map(kinds)
    else:
        df = df.rename(columns={"sub_category": "Sub-category", "account_code": "Account Code",
                                "account_name": "Account Name"})
        df = df.sort_values("Var vs PY", key=lambda s: s.abs(), ascending=False)
        df["_type"] = "line"
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Ratios
# --------------------------------------------------------------------------- #
RATIO_DEFS = [
    # (group, name, unit, formula)
    ("Liquidity", "Current ratio", "x", "Current assets / Current liabilities"),
    ("Liquidity", "Quick ratio", "x", "(Current assets - Inventory) / Current liabilities"),
    ("Liquidity", "Cash ratio", "x", "Cash & bank / Current liabilities"),
    ("Profitability", "Gross profit margin", "%", "Gross profit / Revenue"),
    ("Profitability", "EBITDA margin", "%", "EBITDA / Revenue"),
    ("Profitability", "Operating (EBIT) margin", "%", "EBIT / Revenue"),
    ("Profitability", "Net profit margin", "%", "PAT / Revenue"),
    ("Profitability", "Return on assets (annualised)", "%", "PAT x 12/periods / Total assets"),
    ("Profitability", "Return on equity (annualised)", "%", "PAT x 12/periods / Total equity"),
    ("Leverage", "Debt to equity", "x", "Total borrowings / Total equity"),
    ("Leverage", "Debt ratio", "%", "Total liabilities / Total assets"),
    ("Leverage", "Interest cover", "x", "EBIT / Finance costs"),
    ("Leverage", "Net debt / EBITDA (annualised)", "x", "(Borrowings - Cash) / annualised EBITDA"),
    ("Efficiency", "Receivable days (DSO)", "days", "Trade receivables / Revenue x days"),
    ("Efficiency", "Inventory days (DIO)", "days", "Inventory / Cost of sales x days"),
    ("Efficiency", "Payable days (DPO)", "days", "Trade payables / Cost of sales x days"),
    ("Efficiency", "Cash conversion cycle", "days", "DSO + DIO - DPO"),
    ("Efficiency", "Asset turnover (annualised)", "x", "Revenue x 12/periods / Total assets"),
]


def _position(ctx: ReportContext, y: int, p: int) -> dict[str, float]:
    """Balance-sheet position (positive amounts) and YTD P&L metrics at (y, p)."""
    led = ctx.ledger
    cur = led[(led["year"] == y) & (led["period"] == p)]
    s = cur[~cur["is_pl"]].groupby("sub_category")["closing"].sum()
    g = lambda k: float(s.get(k, 0.0))  # noqa: E731
    pos = {k: g(k) for k in ("Non-Current Assets", *CA)}
    pos.update({k: -g(k) for k in ("Non-Current Liabilities", *CL, "Share Capital", "Other Reserves",
                                   "Retained Earnings")})
    curr_pl = -cur.loc[cur["is_pl"], "closing"].sum()
    prior = 0.0
    if ctx.basis == "movement" and ctx.carry_forward_pl:
        prior = -led.loc[led["is_pl"] & (led["year"] < y), "movement"].sum()
    pos["Total current assets"] = sum(pos[k] for k in CA)
    pos["Total current liabilities"] = sum(pos[k] for k in CL)
    pos["Total assets"] = pos["Non-Current Assets"] + pos["Total current assets"]
    pos["Total liabilities"] = pos["Non-Current Liabilities"] + pos["Total current liabilities"]
    pos["Total equity"] = pos["Share Capital"] + pos["Other Reserves"] + pos["Retained Earnings"] + curr_pl + prior
    pos["Borrowings"] = pos["Non-Current Liabilities"] + pos["Borrowings - Current"]
    ytd = led[(led["year"] == y) & (led["period"] <= p)]
    pos.update(pl_metrics(ytd))
    pos["periods"] = p
    return pos


def _ratios_from_position(x: dict[str, float]) -> dict[str, float]:
    def div(a, b):
        return a / b if b not in (0, 0.0) and abs(b) > 1e-9 else np.nan
    ann = 12 / max(x["periods"], 1)
    days = 365 * x["periods"] / 12
    cos = x["Cost of Sales"]
    r = {
        "Current ratio": div(x["Total current assets"], x["Total current liabilities"]),
        "Quick ratio": div(x["Total current assets"] - x["Inventory"], x["Total current liabilities"]),
        "Cash ratio": div(x["Cash & Bank"], x["Total current liabilities"]),
        "Gross profit margin": div(x["Gross Profit"], x["Revenue"]),
        "EBITDA margin": div(x["EBITDA"], x["Revenue"]),
        "Operating (EBIT) margin": div(x["EBIT"], x["Revenue"]),
        "Net profit margin": div(x["PAT"], x["Revenue"]),
        "Return on assets (annualised)": div(x["PAT"] * ann, x["Total assets"]),
        "Return on equity (annualised)": div(x["PAT"] * ann, x["Total equity"]),
        "Debt to equity": div(x["Borrowings"], x["Total equity"]),
        "Debt ratio": div(x["Total liabilities"], x["Total assets"]),
        "Interest cover": div(x["EBIT"], x["Finance Costs"]),
        "Net debt / EBITDA (annualised)": div(x["Borrowings"] - x["Cash & Bank"], x["EBITDA"] * ann),
        "Receivable days (DSO)": div(x["Trade Receivables"], x["Revenue"]) * days,
        "Inventory days (DIO)": div(x["Inventory"], cos) * days,
        "Payable days (DPO)": div(x["Trade Payables"], cos) * days,
        "Asset turnover (annualised)": div(x["Revenue"] * ann, x["Total assets"]),
    }
    r["Cash conversion cycle"] = (np.nansum([r["Receivable days (DSO)"], r["Inventory days (DIO)"]])
                                  - np.nan_to_num(r["Payable days (DPO)"]))
    return r


def financial_ratios(ctx: ReportContext, view: str = "Monthly") -> pd.DataFrame:
    points = bs_columns(ctx, view)
    vals = {label: _ratios_from_position(_position(ctx, y, p)) for label, y, p in points}
    rows = []
    for grp, name, unit, formula in RATIO_DEFS:
        rows.append({"Category": grp, "Ratio": name, "Unit": unit,
                     **{c: vals[c].get(name, np.nan) for c in vals}, "Formula": formula})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 7. Trend, top-10, cost centre, working capital, movements
# --------------------------------------------------------------------------- #
def trend_data(ctx: ReportContext, view: str = "Monthly") -> pd.DataFrame:
    """Long-format P&L metrics per period (Monthly: every period loaded; Yearly: per fiscal year)."""
    led = ctx.pl_ledger[ctx.pl_ledger["pkey"] <= ctx.pkey]
    out = []
    if view == "Yearly":
        for y in _years(led, ctx.year):
            m = pl_metrics(led[led["year"] == y])
            out.append({"Period": _year_col(led, y), "order": y, **m})
    else:
        for (y, p), grp in led.groupby(["year", "period"]):
            m = pl_metrics(grp)
            out.append({"Period": ctx.label(y, p), "order": y * 100 + p, **m})
    df = pd.DataFrame(out).sort_values("order") if out else pd.DataFrame()
    if not df.empty:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["GP margin %"] = np.where(df["Revenue"] != 0, df["Gross Profit"] / df["Revenue"] * 100, np.nan)
            df["EBITDA margin %"] = np.where(df["Revenue"] != 0, df["EBITDA"] / df["Revenue"] * 100, np.nan)
            df["Net margin %"] = np.where(df["Revenue"] != 0, df["PAT"] / df["Revenue"] * 100, np.nan)
    return df


def top_n(ctx: ReportContext, scope: str = "YTD", category: str = "Expense", n: int = 10) -> pd.DataFrame:
    rows = window_rows(ctx, scope)
    rows = rows[rows["category"] == category]
    if category == "Expense":
        rows = rows[rows["sub_category"] != "Income Tax"]  # tax is not a controllable expense
    sign = 1 if category == "Expense" else -1
    g = (rows.groupby(["account_code", "account_name", "sub_category"], as_index=False)["movement"].sum())
    g["Amount"] = g["movement"] * sign
    total = g["Amount"].sum()
    g = g.sort_values("Amount", ascending=False).head(n)
    g["% of total"] = g["Amount"] / total if total else np.nan
    g["Cumulative %"] = g["% of total"].cumsum()
    g.insert(0, "Rank", range(1, len(g) + 1))
    return g.rename(columns={"account_code": "Account Code", "account_name": "Account Name",
                             "sub_category": "Sub-category"}).drop(columns="movement").reset_index(drop=True)


def cost_centres(ledger: pd.DataFrame) -> list[str]:
    if ledger is None or ledger.empty:
        return []
    return sorted(ledger["cost_centre"].astype(str).unique())


def cost_centre_pl(ctx: ReportContext, scope: str = "YTD", detailed: bool = False) -> pd.DataFrame:
    rows = window_rows(ctx, scope)
    rows = rows[rows["is_pl"]]
    order = cost_centres(rows)
    v, d = _pivot(rows, "movement", "cost_centre", order)
    v["Total"] = v.sum(axis=1)
    if not d.empty:
        d["Total"] = d.sum(axis=1)
    out = _build_statement(v, PL_LAYOUT, lambda s: -1, d if detailed else None)
    return out.rename(columns={NO_CC: "Unallocated"})


def working_capital(ctx: ReportContext, view: str = "Monthly") -> pd.DataFrame:
    points = bs_columns(ctx, view)
    data = {}
    for label, y, p in points:
        x = _position(ctx, y, p)
        r = _ratios_from_position(x)
        twc = x["Inventory"] + x["Trade Receivables"] - x["Trade Payables"]
        data[label] = {
            "Inventory": x["Inventory"], "Trade receivables": x["Trade Receivables"],
            "Other current assets": x["Other Current Assets"], "Trade payables": -x["Trade Payables"],
            "Other current liabilities (incl. tax)": -(x["Other Current Liabilities"] + x["Tax Payable"]),
            "Operating working capital": twc + x["Other Current Assets"]
            - x["Other Current Liabilities"] - x["Tax Payable"],
            "Cash & bank": x["Cash & Bank"], "Current borrowings": -x["Borrowings - Current"],
            "Net working capital": x["Total current assets"] - x["Total current liabilities"],
            "Current ratio (x)": r["Current ratio"], "DSO (days)": r["Receivable days (DSO)"],
            "DIO (days)": r["Inventory days (DIO)"], "DPO (days)": r["Payable days (DPO)"],
            "Cash conversion cycle (days)": r["Cash conversion cycle"],
        }
    kinds = {"Operating working capital": "subtotal", "Net working capital": "total"}
    rows = [{"Line": k, **{c: data[c][k] for c in data}, "_type": kinds.get(k, "line")}
            for k in (next(iter(data.values())).keys() if data else [])]
    return pd.DataFrame(rows, columns=["Line"] + list(data.keys()) + ["_type"])


def major_account_movements(ctx: ReportContext, n: int = 10, by: str = "Balance") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Top-n accounts (by absolute closing balance or absolute YTD movement) and their monthly
    movement / closing for the selected year. Returns (movement pivot, closing pivot).
    """
    led = ctx.ledger[(ctx.ledger["year"] == ctx.year) & (ctx.ledger["period"] <= ctx.period)]
    if led.empty:
        return pd.DataFrame(), pd.DataFrame()
    key = ["account_code", "account_name"]
    if by == "Balance":
        rank = led[led["pkey"] == led["pkey"].max()].groupby(key)["closing"].sum().abs()
    else:
        rank = led.groupby(key)["movement"].sum().abs()
    top = rank.sort_values(ascending=False).head(n).index
    sel = led.set_index(key).loc[top].reset_index()
    sel["Account"] = sel["account_code"] + "  " + sel["account_name"]
    sel["Period"] = sel["period"].map(lambda p: ctx.label(ctx.year, p))
    order = [ctx.label(ctx.year, p) for p in sorted(sel["period"].unique())]
    mv = sel.pivot_table(index="Account", columns="Period", values="movement", aggfunc="sum").reindex(columns=order)
    cl = sel.pivot_table(index="Account", columns="Period", values="closing", aggfunc="sum").reindex(columns=order)
    acc_order = [f"{c}  {nm}" for c, nm in top]
    mv = mv.reindex(acc_order)
    mv["YTD movement"] = mv.sum(axis=1)
    return mv.reset_index(), cl.reindex(acc_order).reset_index()


# --------------------------------------------------------------------------- #
# Report pack (used for export)
# --------------------------------------------------------------------------- #
@dataclass
class ReportItem:
    sheet: str
    title: str
    df: pd.DataFrame
    kind: str = "statement"      # statement | table | ratios
    subtitle: str = ""
    pct_cols: list[str] = field(default_factory=list)


def build_report_pack(ctx: ReportContext, selected: list[str] | None = None, view: str = "Monthly",
                      scope: str = "YTD") -> list[ReportItem]:
    """Build every report (or the `selected` names) for export."""
    has_cc = len([c for c in cost_centres(ctx.pl_ledger) if c != NO_CC]) > 0
    builders = {
        "Profit & Loss": lambda: [ReportItem("P&L", f"Profit & Loss Statement ({view})",
                                             profit_and_loss(ctx, view, detailed=False),
                                             pct_cols=["Change %"]),
                                  ReportItem("P&L Detailed", f"Profit & Loss - Detailed ({view})",
                                             profit_and_loss(ctx, view, detailed=True), pct_cols=["Change %"])],
        "P&L Multi-year": lambda: [ReportItem("P&L Multi-year", "Profit & Loss - Multi-year comparison (like-for-like)",
                                              profit_and_loss(ctx, "Yearly", like_for_like=True),
                                              pct_cols=["Change %"])],
        "Balance Sheet": lambda: [ReportItem("Balance Sheet", f"Balance Sheet ({view})", balance_sheet(ctx, view)),
                                  ReportItem("BS Multi-year", "Balance Sheet - Multi-year",
                                             balance_sheet(ctx, "Yearly"), pct_cols=["Change %"])],
        "Cash Flow": lambda: [ReportItem("Cash Flow", f"Cash Flow Statement - Indirect Method ({view})",
                                         cash_flow(ctx, view))],
        "Trial Balance": lambda: [ReportItem("TB Summary", f"Trial Balance Summary as at {ctx.as_at_label}",
                                             trial_balance(ctx)[0], kind="table"),
                                  ReportItem("TB Detail", f"Trial Balance as at {ctx.as_at_label}",
                                             trial_balance(ctx)[1], kind="table")],
        "Schedules": lambda: [ReportItem("Schedules", f"Schedule of Accounts - {scope_label(ctx, scope)}",
                                         schedule_of_accounts(ctx, scope))],
        "Variance": lambda: [ReportItem("Variance", f"Variance Analysis - {scope_label(ctx, scope)}",
                                        variance_analysis(ctx, scope, "Line"),
                                        pct_cols=["Var vs PY %", "Var vs Budget %"]),
                             ReportItem("Variance by Account", f"Variance by Account - {scope_label(ctx, scope)}",
                                        variance_analysis(ctx, scope, "Account"), kind="table",
                                        pct_cols=["Var vs PY %", "Var vs Budget %"])],
        "Ratios": lambda: [ReportItem("Ratios", f"Key Financial Ratios ({view})", financial_ratios(ctx, view),
                                      kind="ratios")],
        "Trend": lambda: [ReportItem("Trend", "Trend Analysis - P&L metrics by period",
                                     trend_data(ctx, "Monthly").drop(columns=["order"], errors="ignore"),
                                     kind="table")],
        "Cost Centre P&L": (lambda: [ReportItem("Cost Centre P&L", f"Cost Centre P&L - {scope_label(ctx, scope)}",
                                                cost_centre_pl(ctx, scope))]) if has_cc else (lambda: []),
        "Top 10": lambda: [ReportItem("Top 10 Expenses", f"Top 10 Expenses - {scope_label(ctx, scope)}",
                                      top_n(ctx, scope, "Expense"), kind="table",
                                      pct_cols=["% of total", "Cumulative %"]),
                           ReportItem("Top 10 Income", f"Top 10 Income - {scope_label(ctx, scope)}",
                                      top_n(ctx, scope, "Income"), kind="table",
                                      pct_cols=["% of total", "Cumulative %"])],
        "Working Capital": lambda: [ReportItem("Working Capital", f"Working Capital Analysis ({view})",
                                               working_capital(ctx, view))],
        "Account Movements": lambda: [ReportItem("Account Movements",
                                                 f"Monthly Movement of Major Accounts - FY{ctx.year}",
                                                 major_account_movements(ctx)[0], kind="table")],
    }
    items: list[ReportItem] = []
    for name, fn in builders.items():
        if selected is None or name in selected:
            try:
                items.extend(fn())
            except Exception as exc:  # keep the pack usable if one report fails
                items.append(ReportItem(name[:31], f"{name} - could not be generated",
                                        pd.DataFrame({"Error": [str(exc)]}), kind="table"))
    sub = f"{ctx.entity_label} | {ctx.cost_centre_label} | As at {ctx.as_at_label} | Amounts in {ctx.units_label}"
    for it in items:
        it.subtitle = sub
    return items


ALL_REPORTS = ["Profit & Loss", "P&L Multi-year", "Balance Sheet", "Cash Flow", "Trial Balance", "Schedules",
               "Variance", "Ratios", "Trend", "Cost Centre P&L", "Top 10", "Working Capital", "Account Movements"]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_amount(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if not isinstance(v, (int, float, np.integer, np.floating)):
        return str(v)
    if abs(v) < 0.5:
        return "-"
    return f"({abs(v):,.0f})" if v < 0 else f"{v:,.0f}"


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or not isinstance(v, (int, float, np.floating)):
        return ""
    return f"{v * 100:,.1f}%"


def fmt_ratio(v, unit: str) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "n/a"
    if unit == "%":
        return f"{v * 100:,.1f}%"
    if unit == "days":
        return f"{v:,.0f}"
    return f"{v:,.2f}x"


# --------------------------------------------------------------------------- #
# Excel export
# --------------------------------------------------------------------------- #
def export_to_excel(items: list[ReportItem], meta: dict | None = None) -> bytes:
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    navy, light, grey = "1F3864", "D9E2F3", "F2F2F2"
    num_fmt = '#,##0;(#,##0);"-"'
    bio = io.BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        # Cover / index sheet
        cover = pd.DataFrame({"Report": [i.title for i in items], "Sheet": [""] * len(items)})
        cover.to_excel(xw, sheet_name="Index", startrow=5, index=False)
        sheet_names = []
        for it in items:
            name = "".join(ch for ch in it.sheet if ch not in '[]:*?/\\')[:31] or "Sheet"
            base, k = name, 1
            while name in used:
                k += 1
                name = f"{base[:28]}_{k}"
            used.add(name)
            sheet_names.append(name)
            df = it.df.copy() if it.df is not None else pd.DataFrame()
            types = df.pop("_type") if "_type" in df.columns else pd.Series(["line"] * len(df))
            df.to_excel(xw, sheet_name=name, startrow=3, index=False)
            ws = xw.sheets[name]
            ws["A1"] = it.title
            ws["A1"].font = Font(bold=True, size=14, color=navy)
            ws["A2"] = it.subtitle
            ws["A2"].font = Font(italic=True, size=9, color="595959")
            hdr_row = 4
            for ci in range(1, len(df.columns) + 1):
                c = ws.cell(row=hdr_row, column=ci)
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor=navy)
                c.alignment = Alignment(horizontal="center" if ci > 1 else "left", vertical="center", wrap_text=True)
            unit_col = list(df.columns).index("Unit") if "Unit" in df.columns else None
            thin = Side(style="thin", color="7F7F7F")
            for ri, t in enumerate(types.tolist(), start=hdr_row + 1):
                unit = df.iloc[ri - hdr_row - 1, unit_col] if unit_col is not None else None
                for ci, col in enumerate(df.columns, start=1):
                    c = ws.cell(row=ri, column=ci)
                    if isinstance(c.value, (int, float)) and not isinstance(c.value, bool):
                        if col in it.pct_cols:
                            c.number_format = "0.0%"
                        elif unit == "%":
                            c.number_format = "0.0%"
                        elif unit == "days":
                            c.number_format = "#,##0"
                        elif unit == "x":
                            c.number_format = '0.00"x"'
                        else:
                            c.number_format = num_fmt
                    if t in ("subtotal", "total", "check"):
                        c.font = Font(bold=True, color="C00000" if t == "check" else "000000")
                        c.border = Border(top=thin, bottom=Side(style="double" if t == "total" else "thin",
                                                                color="7F7F7F"))
                        if t == "total":
                            c.fill = PatternFill("solid", fgColor=light)
                    elif t == "header":
                        c.font = Font(bold=True, color=navy)
                        c.fill = PatternFill("solid", fgColor=grey)
                    elif t == "account":
                        c.font = Font(size=9, color="404040")
                    elif t == "note":
                        c.font = Font(italic=True, size=9, color="7F7F7F")
                # TB tables: bold the TOTAL row
                if it.kind == "table" and any(str(ws.cell(row=ri, column=k).value).upper() == "TOTAL"
                                              for k in range(1, min(3, len(df.columns)) + 1)):
                    for ci in range(1, len(df.columns) + 1):
                        ws.cell(row=ri, column=ci).font = Font(bold=True)
            for ci, col in enumerate(df.columns, start=1):
                if ci == 1:
                    width = max([len(str(col))] + [len(str(v)) for v in df.iloc[:, 0].tolist()[:500]]) + 2
                    width = min(max(width, 18), 60)
                else:
                    width = min(max(len(str(col)) + 2, 14), 40)
                ws.column_dimensions[get_column_letter(ci)].width = width
            ws.freeze_panes = ws.cell(row=hdr_row + 1, column=2)
            ws.sheet_view.showGridLines = False
            ws.page_setup.orientation = "landscape"
            ws.page_setup.fitToWidth = 1
            ws.sheet_properties.pageSetUpPr.fitToPage = True
            ws.page_setup.fitToHeight = 0
        # finish index
        wsi = xw.sheets["Index"]
        wsi["A1"] = "Financial & Management Reporting Pack"
        wsi["A1"].font = Font(bold=True, size=16, color=navy)
        meta = meta or {}
        wsi["A2"] = items[0].subtitle if items else ""
        wsi["A3"] = f"Generated {datetime.now():%d %b %Y %H:%M}" + (f" | {meta.get('note')}" if meta.get("note") else "")
        for ci in (1, 2):
            cell = wsi.cell(row=6, column=ci)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor=navy)
        for r, name in enumerate(sheet_names, start=7):
            cell = wsi.cell(row=r, column=2, value=name)
            cell.hyperlink = f"#'{name}'!A1"
            cell.font = Font(color="0563C1", underline="single")
        wsi.column_dimensions["A"].width = 70
        wsi.column_dimensions["B"].width = 24
        wsi.sheet_view.showGridLines = False
    return bio.getvalue()


# --------------------------------------------------------------------------- #
# PDF export
# --------------------------------------------------------------------------- #
def export_to_pdf(items: list[ReportItem], title: str = "Financial & Management Reporting Pack") -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

    navy = colors.HexColor("#1F3864")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=navy, fontSize=15, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#595959"))
    cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=7, leading=8.5, alignment=TA_LEFT)
    cell_b = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
    bio = io.BytesIO()
    page = landscape(A4)
    doc = SimpleDocTemplate(bio, pagesize=page, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=12 * mm,
                            bottomMargin=12 * mm, title=title)

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(12 * mm, 7 * mm, f"{title} - generated {datetime.now():%d %b %Y %H:%M}")
        canvas.drawRightString(page[0] - 12 * mm, 7 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    story = [Spacer(1, 40 * mm), Paragraph(title, ParagraphStyle("t", parent=h1, fontSize=24, leading=28)),
             Spacer(1, 4 * mm), Paragraph(items[0].subtitle if items else "", styles["Normal"]),
             Spacer(1, 8 * mm), Paragraph("<b>Contents</b>", styles["Normal"])]
    for i, it in enumerate(items, 1):
        story.append(Paragraph(f"{i}. {it.title}", sub))
    story.append(PageBreak())

    usable = page[0] - 24 * mm
    max_num_cols = 13
    for it in items:
        df = it.df.copy() if it.df is not None else pd.DataFrame()
        types = df.pop("_type").tolist() if "_type" in df.columns else ["line"] * len(df)
        if "Formula" in df.columns:
            df = df.drop(columns=["Formula"])
        if df.empty:
            story += [Paragraph(it.title, h1), Paragraph("No data for this selection.", sub), PageBreak()]
            continue
        cols = list(df.columns)
        text_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        num_cols = [c for c in cols if c not in text_cols]
        unit_series = df["Unit"] if "Unit" in df.columns else None
        chunks = [num_cols[i:i + max_num_cols] for i in range(0, len(num_cols), max_num_cols)] or [[]]
        for ci, chunk in enumerate(chunks):
            show = text_cols + chunk
            hdr = it.title + (f" (cont. {ci + 1}/{len(chunks)})" if len(chunks) > 1 else "")
            story.append(Paragraph(hdr, h1))
            story.append(Paragraph(it.subtitle, sub))
            story.append(Spacer(1, 3 * mm))
            data = [[Paragraph(f"<b>{c}</b>", ParagraphStyle("hh", parent=cell, textColor=colors.white))
                     for c in show]]
            for ri in range(len(df)):
                t = types[ri]
                row = []
                for c in show:
                    v = df.iloc[ri][c]
                    if c in num_cols:
                        if c in it.pct_cols:
                            s = fmt_pct(v)
                        elif unit_series is not None:
                            s = fmt_ratio(v, unit_series.iloc[ri])
                        else:
                            s = fmt_amount(v)
                        row.append(s)
                    else:
                        txt = "" if pd.isna(v) else str(v).replace("&", "&amp;").replace("<", "&lt;")
                        txt = txt.replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")
                        row.append(Paragraph(txt, cell_b if t in ("header", "subtotal", "total", "check") else cell))
                data.append(row)
            n_text = len(text_cols)
            text_w = min(95 * mm, usable * (0.45 if n_text == 1 else 0.6))
            num_w = (usable - text_w) / max(len(chunk), 1) if chunk else 0
            if not chunk:
                widths = [usable / max(n_text, 1)] * n_text
            else:
                widths = [text_w / max(n_text, 1)] * n_text + [min(num_w, 32 * mm)] * len(chunk)
            tbl = Table(data, colWidths=widths, repeatRows=1)
            ts = [("BACKGROUND", (0, 0), (-1, 0), navy),
                  ("FONTSIZE", (0, 0), (-1, -1), 7 if len(chunk) <= 9 else 6),
                  ("ALIGN", (n_text, 1), (-1, -1), "RIGHT"),
                  ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                  ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.8, navy)]
            for ri, t in enumerate(types, start=1):
                if t in ("subtotal", "total", "check"):
                    ts += [("FONTNAME", (0, ri), (-1, ri), "Helvetica-Bold"),
                           ("LINEABOVE", (n_text, ri), (-1, ri), 0.5, colors.grey)]
                    if t == "total":
                        ts += [("BACKGROUND", (0, ri), (-1, ri), colors.HexColor("#D9E2F3")),
                               ("LINEBELOW", (n_text, ri), (-1, ri), 1, colors.grey)]
                    if t == "check":
                        ts += [("TEXTCOLOR", (0, ri), (-1, ri), colors.HexColor("#C00000"))]
                elif t == "header":
                    ts += [("BACKGROUND", (0, ri), (-1, ri), colors.HexColor("#F2F2F2"))]
                elif t == "account":
                    ts += [("TEXTCOLOR", (n_text, ri), (-1, ri), colors.HexColor("#404040"))]
                elif ri % 2 == 0 and it.kind != "statement":
                    ts += [("BACKGROUND", (0, ri), (-1, ri), colors.HexColor("#FAFAFA"))]
            tbl.setStyle(TableStyle(ts))
            story += [tbl, PageBreak()]
    if story and isinstance(story[-1], PageBreak):
        story.pop()
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return bio.getvalue()
