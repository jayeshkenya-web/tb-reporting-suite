"""
app.py - Trial Balance Financial & Management Reporting Suite
=============================================================

Run:   streamlit run app.py

Workflow
--------
1. Upload & Mapping      drop one or more Excel/CSV trial balances (multi-year and/or month-wise,
                         actuals and budgets); columns are auto-mapped, adjustable, and remembered.
2. Classification        accounts are classified automatically (code ranges -> keywords -> fallback);
                         override any account; edit and save the rules.
3. Validation            unbalanced TB, unclassified accounts, gaps, duplicates, suspense balances ...
4. Financial Reports     P&L, Balance Sheet, Cash Flow (indirect), TB summary, schedules.
5. Management Reports    variance, ratios, trends, cost-centre P&L, top 10, working capital, movements.
6. Export                full pack to Excel (multi-sheet, formatted) and PDF.
"""
from __future__ import annotations

import calendar
import io
import json
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import account_classifier as ac
from utils import data_processing as dp
from utils import report_generator as rg

st.set_page_config(page_title="TB Financial Reporting Suite", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# Validated categorical palette (fixed order - colour follows the entity, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: rgba(42,120,214,0.06); border: 1px solid rgba(42,120,214,0.18);
    border-radius: 10px; padding: 10px 14px;}
div[data-testid="stMetricLabel"] p {font-size: 0.8rem; opacity: 0.8;}
.app-title {font-size: 1.7rem; font-weight: 700; margin-bottom: 0;}
.app-sub {opacity: 0.7; margin-top: 0; margin-bottom: 1rem;}
.small-note {font-size: 0.8rem; opacity: 0.7;}
</style>
""", unsafe_allow_html=True)

PAGES = ["📤 Upload & Mapping", "🏷️ Account Classification", "✅ Data Validation",
         "📑 Financial Reports", "📈 Management Reports", "⬇️ Export"]
ROLE_ACTUAL, ROLE_BUDGET = "Actual", "Budget"


# =========================================================================== #
# Session state
# =========================================================================== #
def init_state():
    ss = st.session_state
    ss.setdefault("files", {})            # {file_name: bytes}
    ss.setdefault("settings", dp.load_settings())
    ss.setdefault("rules", ac.load_rules())
    ss.setdefault("overrides", ac.load_overrides())
    ss.setdefault("std", None)            # canonical actual rows
    ss.setdefault("bstd", None)           # canonical budget rows
    ss.setdefault("load_issues", [])
    ss.setdefault("classification", None)
    ss.setdefault("ledger", None)
    ss.setdefault("bledger", None)
    ss.setdefault("issues", [])
    ss.setdefault("uploader_key", 0)


init_state()
ss = st.session_state


@st.cache_data(show_spinner=False)
def cached_read(name: str, data: bytes) -> dict[str, pd.DataFrame]:
    return dp.read_upload(name, data)


@st.cache_data(show_spinner=False)
def cached_samples() -> dict[str, bytes]:
    return dp.generate_sample_files()


def run_pipeline():
    """Classify -> build ledgers -> validate. Called after processing or any rule/setting change."""
    std, bstd, s = ss.std, ss.bstd, ss.settings
    if std is None or std.empty:
        return
    master = dp.account_master(pd.concat([std, bstd]) if bstd is not None and not bstd.empty else std)
    cls = ac.classify_accounts(master, ss.rules, ss.overrides)
    ss.classification = cls
    ss.ledger = dp.build_ledger(std, cls, s["basis"])
    if bstd is not None and not bstd.empty:
        ss.bledger = dp.build_ledger(bstd, cls, s["budget_basis"], natural_sign=s["budget_sign"] == "natural")
    else:
        ss.bledger = None
    ss.issues = ss.load_issues + dp.validate_data(std, ss.ledger, cls, s["basis"], s["carry_forward_pl"])


def has_data() -> bool:
    return ss.ledger is not None and not ss.ledger.empty


# =========================================================================== #
# Display helpers
# =========================================================================== #
def show_statement(df: pd.DataFrame, pct_cols: tuple = ("Change %",), height: int | None = None, key=None):
    """Render a statement frame (with _type) with accounting formatting and bold totals."""
    if df is None or df.empty:
        st.info("No data for this selection.")
        return
    types = df["_type"].tolist() if "_type" in df.columns else ["line"] * len(df)
    view = df.drop(columns=["_type"], errors="ignore").reset_index(drop=True)
    fmt = {}
    for c in view.columns:
        if pd.api.types.is_numeric_dtype(view[c]):
            fmt[c] = rg.fmt_pct if c in pct_cols else rg.fmt_amount

    def row_style(row):
        t = types[row.name]
        if t == "total":
            return ["font-weight: 700; background-color: rgba(42,120,214,0.12); border-top: 1px solid #888"] * len(row)
        if t == "subtotal":
            return ["font-weight: 700; border-top: 1px solid #bbb"] * len(row)
        if t == "header":
            return ["font-weight: 700; background-color: rgba(128,128,128,0.10)"] * len(row)
        if t == "check":
            return ["font-weight: 700; color: #d03b3b"] * len(row)
        if t in ("account", "note"):
            return ["color: gray; font-size: 0.85em"] * len(row)
        return [""] * len(row)

    num_cols = list(fmt.keys())
    disp = view.copy()
    for c in num_cols:
        disp[c] = [fmt[c](v) for v in view[c]]
    disp = disp.fillna("").astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    right = [c for c in disp.columns if c not in ("Line", "Status", "Sub-category", "Account Name", "Category")]
    styler = disp.style.apply(row_style, axis=1).set_properties(subset=right, **{"text-align": "right"})
    h = height or min(36 * (len(view) + 1) + 4, 1100)
    st.dataframe(styler, hide_index=True, width="stretch", height=h, key=key,
                 column_config={c: st.column_config.TextColumn(c, alignment="right") for c in right}
                 | {"Line": st.column_config.TextColumn("Line", pinned=True)})


def show_table(df: pd.DataFrame, pct_cols: tuple = (), height: int | None = None):
    if df is None or df.empty:
        st.info("No data for this selection.")
        return
    view = df.drop(columns=["_type"], errors="ignore")
    fmt = {c: (rg.fmt_pct if c in pct_cols else rg.fmt_amount)
           for c in view.columns if pd.api.types.is_numeric_dtype(view[c]) and c not in ("Rank", "Accounts")}
    disp = view.copy()
    for c, f in fmt.items():
        disp[c] = [f(v) for v in view[c]]
    disp = disp.fillna("").astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    h = height or min(36 * (len(view) + 1) + 4, 700)
    st.dataframe(disp, hide_index=True, width="stretch", height=h,
                 column_config={c: st.column_config.TextColumn(c, alignment="right") for c in fmt})


def compact(v: float) -> str:
    """Compact KPI number: 234.7M / 49.6K, brackets for negatives."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    a = abs(v)
    t = f"{a / 1e9:,.2f}B" if a >= 1e9 else f"{a / 1e6:,.1f}M" if a >= 1e6 else f"{a / 1e3:,.1f}K" if a >= 1e3 else f"{a:,.0f}"
    return f"({t})" if v < 0 else t


def style_fig(fig: go.Figure, height: int = 380, yaxis_title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
                      title=dict(y=0.98, yanchor="top", x=0, xanchor="left", font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, title=None),
                      bargap=0.25, bargroupgap=0.06)
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=True, zerolinecolor="rgba(128,128,128,0.4)",
                     title=yaxis_title, tickformat=",.0f")
    fig.update_traces(selector=dict(type="scatter"), line=dict(width=2), marker=dict(size=8))
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


def statement_to_long(df: pd.DataFrame, lines: list[str], exclude_cols=("YTD", "Change", "Change %", "Total")):
    cols = [c for c in df.columns if c not in ("Line", "_type") and c not in exclude_cols]
    sub = df[df["Line"].isin(lines)][["Line"] + cols]
    return sub.melt(id_vars="Line", var_name="Period", value_name="Amount")


def units_label() -> str:
    u = int(ss.settings.get("units", 1))
    cur = ss.settings.get("currency", "KES")
    return {1: cur, 1000: f"{cur} '000", 1_000_000: f"{cur} million"}.get(u, cur)


# =========================================================================== #
# Sidebar: navigation, filters, settings
# =========================================================================== #
with st.sidebar:
    st.markdown("### 📊 TB Reporting Suite")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()

    ctx: rg.ReportContext | None = None
    view, scope = "Monthly", "YTD"
    if has_data():
        st.markdown("**Filters**")
        led = ss.ledger
        companies = sorted(led["company"].unique())
        sel_comp = st.multiselect("Company", companies, default=companies,
                                  help="Multiple companies are aggregated (simple sum - no intercompany eliminations).")
        sel_comp = sel_comp or companies
        ccs = rg.cost_centres(led)
        sel_cc = st.multiselect("Cost centre", ccs, default=[],
                                help="Leave empty for all. Applies to P&L-based reports; Balance Sheet, "
                                     "Cash Flow, Ratios and Working Capital always use the whole entity.")
        periods = rg.available_periods(led[led["company"].isin(sel_comp)])
        years = sorted({y for y, _ in periods}, reverse=True)
        year = st.selectbox("Fiscal year", years, index=0)
        ps = sorted({p for y, p in periods if y == year})
        fy = int(ss.settings["fy_start_month"])
        period = st.selectbox("As at period", ps, index=len(ps) - 1, format_func=lambda p: dp.period_label(year, p, fy))
        view = st.radio("Column view", ["Monthly", "Quarterly", "Yearly"], horizontal=True,
                        help="Monthly/Quarterly = periods of the selected year; Yearly = multi-year comparison.")
        scope = st.radio("Analysis window", ["Month", "Quarter", "YTD", "Full Year"], index=2, horizontal=True,
                         help="Window used by variance, top 10, cost-centre P&L and schedules.")

        units = int(ss.settings.get("units", 1))
        ent = led[led["company"].isin(sel_comp)].copy()
        plf = ent[ent["cost_centre"].isin(sel_cc)] if sel_cc else ent
        bud = ss.bledger
        if bud is not None:
            bud = bud[bud["company"].isin(sel_comp)]
            bud = bud[bud["cost_centre"].isin(sel_cc)] if sel_cc else bud
        if units != 1:
            amt = ["movement", "closing", "opening", "debit", "credit"]
            ent[amt] = ent[amt] / units
            plf = plf.copy()
            plf[amt] = plf[amt] / units
            if bud is not None:
                bud = bud.copy()
                bud[amt] = bud[amt] / units
        ctx = rg.ReportContext(
            ledger=ent, pl_ledger=plf, budget=bud, year=int(year), period=int(period),
            basis=ss.settings["basis"], fy_start=fy, carry_forward_pl=bool(ss.settings["carry_forward_pl"]),
            has_opening=bool(ss.std["opening_balance"].notna().any()),
            entity_label=", ".join(sel_comp) if len(sel_comp) <= 3 else f"{len(sel_comp)} companies (aggregated)",
            units_label=units_label(),
            cost_centre_label=("Cost centres: " + ", ".join(sel_cc)) if sel_cc else "All cost centres")
        st.divider()

    with st.expander("⚙️ Settings", expanded=not has_data()):
        s = dict(ss.settings)
        s["basis"] = st.selectbox(
            "Actual TB amounts are", ["closing", "movement"], index=["closing", "movement"].index(s["basis"]),
            format_func=lambda x: {"closing": "Closing balances (cumulative / YTD)",
                                   "movement": "Period movements (activity)"}[x],
            help="Most ERP trial balances are closing balances. Choose 'movements' if each file shows only "
                 "the activity for the period (optionally with an Opening Balance column).")
        s["fy_start_month"] = st.selectbox("Fiscal year starts in", list(range(1, 13)), index=s["fy_start_month"] - 1,
                                           format_func=lambda m: calendar.month_name[m],
                                           help="Fiscal year label = calendar year in which the year ENDS.")
        s["numeric_month_is"] = st.radio("Numeric month values (1-12) mean", ["calendar", "fiscal"],
                                         index=["calendar", "fiscal"].index(s["numeric_month_is"]), horizontal=True)
        s["carry_forward_pl"] = st.checkbox(
            "Roll prior-year profit into equity (movement basis)", value=s["carry_forward_pl"],
            help="Only for movement-basis data without year-end closing entries.")
        s["budget_basis"] = st.selectbox("Budget amounts are", ["movement", "closing"],
                                         index=["movement", "closing"].index(s["budget_basis"]),
                                         format_func=lambda x: {"movement": "Monthly amounts",
                                                                "closing": "Cumulative YTD amounts"}[x])
        s["budget_sign"] = st.selectbox("Budget sign convention", ["natural", "signed"],
                                        index=["natural", "signed"].index(s["budget_sign"]),
                                        format_func=lambda x: {"natural": "All positive (income & costs +)",
                                                               "signed": "Dr + / Cr - (like the TB)"}[x])
        c1, c2 = st.columns(2)
        s["currency"] = c1.text_input("Currency", s["currency"])
        s["units"] = c2.selectbox("Show in", [1, 1000, 1_000_000], index=[1, 1000, 1_000_000].index(int(s["units"])),
                                  format_func=lambda u: {1: "Units", 1000: "Thousands", 1_000_000: "Millions"}[u])
        s["default_company"] = st.text_input("Default company name", s["default_company"],
                                             help="Used when a file has no Company column.")
        if st.button("Save settings & refresh", width="stretch"):
            reprocess = (s["fy_start_month"] != ss.settings["fy_start_month"]
                         or s["numeric_month_is"] != ss.settings["numeric_month_is"])
            ss.settings = s
            dp.save_settings(s)
            if reprocess and ss.files:
                st.warning("Fiscal-year settings changed - click **Process data** on the Upload page again.")
            run_pipeline()
            st.rerun()
    st.caption("Settings, column mappings and classification rules are saved in the `config/` folder.")


# =========================================================================== #
# Header
# =========================================================================== #
st.markdown("## Trial Balance → Financial & Management Reporting")
if ctx is not None:
    st.markdown(f'<p class="app-sub">{ctx.entity_label} · {ctx.cost_centre_label} · As at '
                f'<b>{ctx.as_at_label}</b> · Amounts in {ctx.units_label}</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="app-sub">Upload trial balances to generate the full reporting pack.</p>',
                unsafe_allow_html=True)


def need_data():
    if not has_data():
        st.info("No processed data yet. Go to **📤 Upload & Mapping**, upload trial balance file(s) "
                "(or load the sample data) and click **Process data**.")
        st.stop()


# =========================================================================== #
# PAGE 1 - Upload & mapping
# =========================================================================== #
def page_upload():
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        ups = st.file_uploader("Drag & drop trial balance files (Excel .xlsx/.xlsm/.xls or .csv) - "
                               "multi-year, month-wise, budgets; several files at once",
                               type=["xlsx", "xlsm", "xls", "csv"], accept_multiple_files=True,
                               key=f"uploader_{ss.uploader_key}")
        for f in ups or []:
            ss.files[f.name] = f.getvalue()
    with c2:
        st.write("")
        if st.button("🧪 Load sample data", width="stretch",
                     help="Loads a 2-year annual TB, a 2025 month-wise TB by cost centre and a 2025 budget."):
            ss.files.update(cached_samples())
            st.rerun()
        zbio = io.BytesIO()
        with zipfile.ZipFile(zbio, "w", zipfile.ZIP_DEFLATED) as z:
            for n, b in cached_samples().items():
                z.writestr(n, b)
        st.download_button("⬇️ Sample files (.zip)", zbio.getvalue(), "sample_trial_balances.zip",
                           "application/zip", width="stretch")
    with c3:
        st.write("")
        if st.button("🗑️ Clear all", width="stretch"):
            for k in ("std", "bstd", "classification", "ledger", "bledger"):
                ss[k] = None
            ss.files, ss.issues, ss.load_issues = {}, [], []
            ss.uploader_key += 1
            st.rerun()

    with st.expander("ℹ️ Expected data structure", expanded=not ss.files):
        st.markdown("""
**One row per account (per period, per cost centre).** Column names are flexible - they are matched
automatically and you can re-map them below. Title rows above the table and *Total* rows are ignored.

| Field | Required | Examples of accepted headers | Notes |
|---|---|---|---|
| Account Code | ✅ | Account Code, GL Code, A/C No | Drives classification |
| Account Name | recommended | Account Name, Description | Used for keyword classification |
| Debit / Credit | ✅ *or* Net Balance | Debit, Dr, Closing Debit / Credit, Cr | Brackets, `Cr` suffix, trailing `-` understood |
| Net Balance | alternative | Balance, Closing Balance, Amount | Debit positive, credit negative |
| Opening Balance | optional | Opening, B/F | Movement-basis files |
| Year | optional* | Year, FY, Financial Year (`2024/25` → 2025) | Fiscal year = year in which it ends |
| Month / Period | optional | Month (Jan, January, 3, P03), Period | No month → annual (year-end) TB |
| Period Date | optional | Date, Period End, As at | Alternative to Year + Month |
| Company | optional | Company, Entity | Multi-company support |
| Cost Centre | optional | Cost Centre, Department, Branch | Enables cost-centre P&L |
| Budget | optional | Budget, Plan | Or upload a separate budget file (role = Budget) |

\\* If there is no Year column, the year is read from the file/sheet name (e.g. `TB_2024.xlsx`, `Mar 2025`)
or entered below.

**Example (month-wise, closing basis):**
""")
        st.dataframe(pd.DataFrame({
            "Company": ["Demo Kenya Ltd"] * 4, "Year": [2025] * 4, "Month": ["Jan"] * 4,
            "Cost Centre": ["", "", "Sales", "Head Office"], "Account Code": ["1810", "2010", "4010", "6010"],
            "Account Name": ["Bank - KCB", "Trade Creditors", "Sales - Local", "Salaries & Wages"],
            "Debit": [4_108_461, 0, 0, 1_290_000], "Credit": [0, 2_578_445, 17_514_471, 0]}), hide_index=True)

    if not ss.files:
        return

    store = dp.load_mapping_store()
    st.subheader(f"Files ({len(ss.files)})")
    fy = int(ss.settings["fy_start_month"])
    sheet_cfgs = []
    for fname, data in list(ss.files.items()):
        try:
            sheets = cached_read(fname, data)
        except Exception as exc:
            st.error(f"**{fname}** could not be read: {exc}")
            continue
        for sname, raw in sheets.items():
            sid = f"{fname}::{sname}"
            k = lambda x: f"{sid}::{x}"  # noqa: E731
            det_hdr = dp.detect_header_row(raw, store)
            with st.expander(f"📄 {fname} — sheet **{sname}** ({len(raw):,} rows)", expanded=False):
                top = st.columns([1, 1, 1, 2])
                include = top[0].checkbox("Include", value=True, key=k("inc"))
                hdr = top[1].number_input("Header row", min_value=1, max_value=max(len(raw), 1),
                                          value=det_hdr + 1, key=k("hdr"),
                                          help="Detected automatically - change if the preview looks wrong.") - 1
                df = dp.apply_header(raw, hdr)
                cols = list(df.columns)
                auto = dp.auto_map_columns(cols, store)
                guess_budget = ("budget" in f"{fname} {sname}".lower()) or (
                    auto.get("budget") and not (auto.get("debit") or auto.get("credit") or auto.get("balance")))
                role = top[2].selectbox("Role", [ROLE_ACTUAL, ROLE_BUDGET], index=1 if guess_budget else 0,
                                        key=k("role"))
                saved = dp.column_signature(cols) in store.get("by_signature", {})
                top[3].markdown(f"<div class='small-note'><br>{'💾 Saved mapping applied' if saved else '🔎 Auto-detected mapping'}"
                                f" · {len(cols)} columns</div>", unsafe_allow_html=True)

                st.markdown("**Column mapping**")
                options = [dp.NOT_MAPPED] + cols
                mapping = {}
                mcols = st.columns(4)
                for i, field in enumerate(dp.FIELD_ORDER):
                    fdef = dp.FIELD_DEFS[field]
                    cur = auto.get(field)
                    mapping[field] = mcols[i % 4].selectbox(
                        fdef["label"] + (" *" if fdef["required"] else ""), options,
                        index=options.index(cur) if cur in options else 0, key=k(f"map_{field}"))
                mapping = {f: (None if v == dp.NOT_MAPPED else v) for f, v in mapping.items()}

                iy, im = dp.infer_period_from_text(f"{fname} {sname}")
                d1, d2, d3 = st.columns(3)
                def_company = d1.text_input("Company (if no column)", ss.settings["default_company"],
                                            key=k("dcomp"), disabled=bool(mapping.get("company")))
                def_year = d2.number_input("Year (if no column)", min_value=0, max_value=2100,
                                           value=int(iy or 0), key=k("dyear"),
                                           disabled=bool(mapping.get("year") or mapping.get("date")),
                                           help="0 = not set. Inferred from the file/sheet name when possible.")
                months = [0] + list(range(1, 13))
                def_month = d3.selectbox("Month (if no column)", months, index=months.index(im or 0),
                                         format_func=lambda m: "Annual / year-end" if m == 0 else calendar.month_name[m],
                                         key=k("dmonth"), disabled=bool(mapping.get("month") or mapping.get("date")))
                st.caption("Preview (first rows after the header):")
                st.dataframe(df.head(8), hide_index=True, width="stretch")
                sheet_cfgs.append(dict(sid=sid, fname=fname, sname=sname, include=include, df=df, cols=cols,
                                       mapping=mapping, role=role,
                                       defaults={"company": def_company, "year": def_year or None,
                                                 "month": def_month or None}))

    c1, c2 = st.columns([1, 3])
    remember = c2.checkbox("Remember these column mappings for next time", value=True)
    if c1.button("🚀 Process data", type="primary", width="stretch"):
        actual, budget, issues, errors = [], [], [], []
        with st.spinner("Processing trial balances..."):
            for cfg in sheet_cfgs:
                if not cfg["include"]:
                    continue
                try:
                    std, iss = dp.standardise_frame(cfg["df"], cfg["mapping"], cfg["defaults"], ss.settings,
                                                    source=f"{cfg['fname']} / {cfg['sname']}")
                except Exception as exc:
                    errors.append(f"{cfg['fname']} / {cfg['sname']}: {exc}")
                    continue
                issues += iss
                if remember:
                    dp.save_mapping(cfg["cols"], cfg["mapping"])
                if cfg["role"] == ROLE_BUDGET:
                    if cfg["mapping"].get("budget"):
                        std["net"] = std["budget"]
                    budget.append(std)
                else:
                    actual.append(std)
                    b = dp.budget_frame_from_actual(std)
                    if not b.empty:
                        budget.append(b)
        for e in errors:
            st.error(e)
        if not actual:
            st.error("No actual trial balance rows were processed - check the mappings.")
            return
        ss.std = pd.concat(actual, ignore_index=True)
        ss.bstd = pd.concat(budget, ignore_index=True) if budget else None
        ss.load_issues = issues
        run_pipeline()
        n_err = sum(1 for i in ss.issues if i["level"] == "error")
        st.success(f"Processed {len(ss.std):,} TB rows · {ss.std['account_code'].nunique()} accounts · "
                   f"{len(rg.available_periods(ss.ledger))} periods"
                   + (f" · budget rows {len(ss.bstd):,}" if ss.bstd is not None else "")
                   + (f" · ⚠️ {n_err} validation error(s) - see Data Validation" if n_err else ""))
        st.rerun()

    if has_data():
        st.success(f"✅ Data loaded: {len(ss.std):,} rows · {ss.std['account_code'].nunique()} accounts · "
                   f"periods {', '.join(dp.period_label(y, p, fy) for y, p in rg.available_periods(ss.ledger)[:3])}"
                   f"{' …' if len(rg.available_periods(ss.ledger)) > 3 else ''}. Use the sidebar to navigate.")


# =========================================================================== #
# PAGE 2 - Classification
# =========================================================================== #
def page_classification():
    need_data()
    cls = ss.classification.copy()
    latest = ss.ledger[ss.ledger["pkey"] == ss.ledger["pkey"].max()].groupby("account_code")["closing"].sum()
    cls["latest_balance"] = cls["account_code"].map(latest).fillna(0.0)

    m = st.columns(5)
    m[0].metric("Accounts", len(cls))
    m[1].metric("Code-range rules", int(cls["rule_source"].str.startswith("Code range").sum()))
    m[2].metric("Keyword rules", int(cls["rule_source"].str.startswith("Keyword").sum()))
    m[3].metric("Manual overrides", int((cls["rule_source"] == "Manual override").sum()))
    m[4].metric("Unclassified / fallback", int(((cls["sub_category"] == ac.UNCLASSIFIED)
                                                | cls["rule_source"].str.startswith("First-digit")).sum()))

    tab1, tab2, tab3, tab4 = st.tabs(["Account mapping", "Code-range rules", "Keyword rules", "Priority & backup"])
    with tab1:
        f1, f2 = st.columns([1, 2])
        show = f1.selectbox("Show", ["All accounts", "Needs review (unclassified / fallback)", "Manual overrides"])
        q = f2.text_input("Search code or name")
        view = cls
        if show.startswith("Needs"):
            view = view[(view["sub_category"] == ac.UNCLASSIFIED) | view["rule_source"].str.startswith("First-digit")]
        elif show.startswith("Manual"):
            view = view[view["rule_source"] == "Manual override"]
        if q:
            view = view[view["account_code"].str.contains(q, case=False) | view["account_name"].str.contains(q, case=False)]
        st.caption("Change the **Sub-category** to override an account (the category follows automatically). "
                   "Overrides are saved and re-applied to future uploads.")
        edited = st.data_editor(
            view[["account_code", "account_name", "sub_category", "category", "rule_source", "latest_balance"]],
            hide_index=True, width="stretch", height=min(36 * (len(view) + 1) + 4, 650), key="cls_editor",
            column_config={
                "account_code": st.column_config.TextColumn("Code", disabled=True),
                "account_name": st.column_config.TextColumn("Account name", disabled=True),
                "sub_category": st.column_config.SelectboxColumn("Sub-category", options=ac.ALL_SUB_CATEGORIES,
                                                                 required=True),
                "category": st.column_config.TextColumn("Category", disabled=True),
                "rule_source": st.column_config.TextColumn("Classified by", disabled=True),
                "latest_balance": st.column_config.NumberColumn("Latest balance (Dr +)", format="localized",
                                                                disabled=True),
            })
        b1, b2 = st.columns([1, 1])
        if b1.button("💾 Save overrides & refresh reports", type="primary", width="stretch"):
            orig = view.set_index("account_code")["sub_category"]
            changed = 0
            for _, r in edited.iterrows():
                if r["sub_category"] != orig.get(r["account_code"]):
                    ss.overrides[str(r["account_code"])] = r["sub_category"]
                    changed += 1
            ac.save_overrides(ss.overrides)
            run_pipeline()
            st.success(f"Saved {changed} change(s).")
            st.rerun()
        if b2.button("↩️ Clear all manual overrides", width="stretch"):
            ss.overrides = {}
            ac.save_overrides({})
            run_pipeline()
            st.rerun()

    with tab2:
        st.caption(f"Ranges compare the first **{ss.rules['code_digits']}** digits of the account code "
                   "(shorter codes are right-padded with zeros). First matching range wins.")
        rng = st.data_editor(ac.ranges_to_frame(ss.rules), num_rows="dynamic", hide_index=True, width="stretch",
                             key="rng_editor", column_config={
                                 "from": st.column_config.NumberColumn("From", step=1, format="%d"),
                                 "to": st.column_config.NumberColumn("To", step=1, format="%d"),
                                 "sub_category": st.column_config.SelectboxColumn(
                                     "Sub-category", options=ac.ALL_SUB_CATEGORIES[:-1])})
        digits = st.number_input("Digits compared", 1, 12, int(ss.rules["code_digits"]))
        if st.button("💾 Save code-range rules", type="primary"):
            ss.rules["code_ranges"] = ac.frame_to_ranges(rng)
            ss.rules["code_digits"] = int(digits)
            ac.save_rules(ss.rules)
            run_pipeline()
            st.rerun()

    with tab3:
        st.caption("Comma-separated keywords matched (case-insensitive) inside the account name. "
                   "Rows are evaluated top-down; put specific phrases above generic words.")
        kw = st.data_editor(ac.keywords_to_frame(ss.rules), num_rows="dynamic", hide_index=True, width="stretch",
                            key="kw_editor", column_config={
                                "keywords": st.column_config.TextColumn("Keywords", width="large"),
                                "sub_category": st.column_config.SelectboxColumn(
                                    "Sub-category", options=ac.ALL_SUB_CATEGORIES[:-1])})
        if st.button("💾 Save keyword rules", type="primary"):
            ss.rules["keywords"] = ac.frame_to_keywords(kw)
            ac.save_rules(ss.rules)
            run_pipeline()
            st.rerun()

    with tab4:
        st.caption("Order in which the layers are tried (select in the desired order).")
        layers = ["override", "code_range", "keyword", "first_digit"]
        pr = st.multiselect("Priority", layers, default=[x for x in ss.rules["priority"] if x in layers])
        if st.button("💾 Save priority"):
            ss.rules["priority"] = pr or layers
            ac.save_rules(ss.rules)
            run_pipeline()
            st.rerun()
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Download rules & overrides (JSON)",
                           json.dumps({"rules": ss.rules, "overrides": ss.overrides}, indent=2),
                           "classification_rules_backup.json", "application/json", width="stretch")
        up = c2.file_uploader("Import rules JSON", type=["json"], label_visibility="collapsed")
        if up is not None and c2.button("Import", width="stretch"):
            try:
                data = json.loads(up.getvalue())
                ss.rules = data.get("rules", ss.rules)
                ss.overrides = data.get("overrides", ss.overrides)
                ac.save_rules(ss.rules)
                ac.save_overrides(ss.overrides)
                run_pipeline()
                st.rerun()
            except Exception as exc:
                st.error(f"Invalid file: {exc}")
        if c3.button("Reset rules to defaults", width="stretch"):
            ss.rules = ac.reset_rules()
            run_pipeline()
            st.rerun()


# =========================================================================== #
# PAGE 3 - Validation
# =========================================================================== #
def page_validation():
    need_data()
    issues = ss.issues
    counts = {lvl: sum(1 for i in issues if i["level"] == lvl) for lvl in ("error", "warning", "info", "success")}
    m = st.columns(4)
    m[0].metric("❌ Errors", counts["error"])
    m[1].metric("⚠️ Warnings", counts["warning"])
    m[2].metric("ℹ️ Information", counts["info"])
    m[3].metric("✅ Checks passed", counts["success"])
    order = {"error": 0, "warning": 1, "info": 2, "success": 3}
    for i, iss in enumerate(sorted(issues, key=lambda x: order.get(x["level"], 9))):
        fn = {"error": st.error, "warning": st.warning, "info": st.info, "success": st.success}[iss["level"]]
        fn(f"**{iss['title']}** — {iss['message']}")
        det = iss.get("detail")
        if isinstance(det, pd.DataFrame) and not det.empty:
            with st.expander(f"Details ({len(det)} rows)"):
                st.dataframe(det, hide_index=True, width="stretch")

    st.subheader("Trial balance check by period")
    tb = (ss.std.groupby(["company", "year", "period"])
          .agg(rows=("account_code", "count"), total_debit=("debit", "sum"), total_credit=("credit", "sum"))
          .reset_index())
    tb["difference"] = (tb["total_debit"] - tb["total_credit"]).round(2)
    tb["status"] = np.where(tb["difference"].abs() <= 1, "✅ Balanced", "❌ Unbalanced")
    tb.insert(1, "period_label", [dp.period_label(y, p, int(ss.settings["fy_start_month"]))
                                  for y, p in zip(tb["year"], tb["period"])])
    st.dataframe(tb.style.format({c: rg.fmt_amount for c in ("total_debit", "total_credit", "difference")}),
                 hide_index=True, width="stretch")


# =========================================================================== #
# PAGE 4 - Financial reports
# =========================================================================== #
def kpi_row(c: rg.ReportContext):
    ytd = rg.window_rows(c, "YTD")
    m = rg.pl_metrics(ytd)
    py_rows = rg.window_rows(c, "YTD", year=c.year - 1)
    pm = rg.pl_metrics(py_rows) if not py_rows.empty else None
    pos = rg._position(c, c.year, c.period)

    def delta(k):
        if pm is None or not pm.get(k):
            return None
        return f"{(m[k] - pm[k]) / abs(pm[k]) * 100:+.1f}% vs PY"
    k = st.columns(6)
    k[0].metric("Revenue (YTD)", compact(m["Revenue"]), delta("Revenue"))
    k[1].metric("Gross margin", f"{m['Gross Profit'] / m['Revenue'] * 100:.1f}%" if m["Revenue"] else "n/a")
    k[2].metric("EBITDA (YTD)", compact(m["EBITDA"]), delta("EBITDA"))
    k[3].metric("Profit after tax (YTD)", compact(m["PAT"]), delta("PAT"))
    k[4].metric("Cash & bank", compact(pos["Cash & Bank"]))
    k[5].metric("Net working capital", compact(pos["Total current assets"] - pos["Total current liabilities"]))


def page_financial(c: rg.ReportContext, view: str, scope: str):
    need_data()
    kpi_row(c)
    t = st.tabs(["Profit & Loss", "Balance Sheet", "Cash Flow", "Trial Balance", "Schedule of Accounts"])

    with t[0]:
        o1, o2 = st.columns([1, 1])
        detailed = o1.toggle("Show account detail", key="pl_det")
        lfl = o2.toggle("Like-for-like YTD (yearly view)", value=False, key="pl_lfl",
                        help="Restrict prior years to the same periods as the selected YTD (needs monthly data).")
        pl = rg.profit_and_loss(c, view, detailed=detailed, like_for_like=lfl)
        st.markdown(f"#### Profit & Loss Statement — {view}")
        show_statement(pl)
        long = statement_to_long(pl, ["Revenue", "Gross Profit", "EBITDA", "Profit After Tax"])
        if not long.empty:
            fig = px.bar(long, x="Period", y="Amount", color="Line", barmode="group",
                         color_discrete_sequence=SERIES, title="Revenue and profit by period")
            st.plotly_chart(style_fig(fig, yaxis_title=c.units_label), width="stretch")

    with t[1]:
        detailed = st.toggle("Show account detail", key="bs_det")
        bs = rg.balance_sheet(c, view, detailed=detailed)
        st.markdown(f"#### Balance Sheet — {view}")
        diff = bs.loc[bs["_type"] == "check"].drop(columns=["Line", "_type", "Change", "Change %"], errors="ignore")
        if (diff.abs() > 1).any(axis=None):
            st.warning("The balance sheet does not balance for some columns - review unclassified accounts, "
                       "the TB basis setting and the Data Validation page.")
        show_statement(bs)
        comp = statement_to_long(bs, ["Non-Current Assets", "Total Current Assets", "Total Equity",
                                      "Non-Current Liabilities", "Total Current Liabilities"])
        if not comp.empty:
            comp["Side"] = np.where(comp["Line"].isin(["Non-Current Assets", "Total Current Assets"]),
                                    "Assets", "Equity & liabilities")
            fig = px.bar(comp, x="Period", y="Amount", color="Line", facet_col="Side",
                         color_discrete_sequence=SERIES, title="Balance sheet composition")
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(style_fig(fig, 400, c.units_label), width="stretch")

    with t[2]:
        cf = rg.cash_flow(c, view)
        st.markdown(f"#### Cash Flow Statement (indirect method) — {view}")
        st.caption("Derived from balance-sheet movements. Bank overdrafts are presented as financing "
                   "(classify them as *Cash & Bank* if you treat them as cash equivalents under IAS 7).")
        show_statement(cf)
        col = "YTD" if "YTD" in cf.columns else [x for x in cf.columns if x not in ("Line", "_type")][-1]
        lines = ["Net cash from operating activities", "Net cash used in investing activities",
                 "Net cash from financing activities"]
        vals = cf.set_index("Line").loc[lines, col]
        open_c = cf.set_index("Line").loc["Cash & bank at beginning of period", col]
        if not vals.isna().any():
            fig = go.Figure(go.Waterfall(
                x=["Opening cash", "Operating", "Investing", "Financing", "Closing cash"],
                measure=["absolute", "relative", "relative", "relative", "total"],
                y=[open_c, *vals.tolist(), 0], connector=dict(line=dict(color="rgba(128,128,128,0.5)")),
                increasing=dict(marker=dict(color=SERIES[0])), decreasing=dict(marker=dict(color=SERIES[1])),
                totals=dict(marker=dict(color=SERIES[6])), text=[rg.fmt_amount(v) for v in [open_c, *vals, None]],
                textposition="outside"))
            fig.update_layout(title=f"Cash bridge — {col}", showlegend=False)
            st.plotly_chart(style_fig(fig, 380, c.units_label), width="stretch")

    with t[3]:
        summ, det = rg.trial_balance(c)
        tot = summ.iloc[-1]
        k = st.columns(3)
        k[0].metric("Total debits", rg.fmt_amount(tot["Debit"]))
        k[1].metric("Total credits", rg.fmt_amount(tot["Credit"]))
        k[2].metric("Difference", rg.fmt_amount(tot["Debit"] - tot["Credit"]),
                    "Balanced" if abs(tot["Debit"] - tot["Credit"]) < 1 else "UNBALANCED",
                    delta_color="off" if abs(tot["Debit"] - tot["Credit"]) < 1 else "inverse")
        st.markdown(f"#### Trial Balance Summary — as at {c.as_at_label}")
        show_table(summ)
        with st.expander("Detailed trial balance", expanded=False):
            show_table(det, height=600)

    with t[4]:
        st.markdown(f"#### Schedule of Accounts — {rg.scope_label(c, scope)}")
        if c.basis == "closing":
            st.caption("Closing-balance TBs give net movement only: debits/credits shown are the net movement split.")
        show_statement(rg.schedule_of_accounts(c, scope), pct_cols=())
        st.markdown("#### Account ledger (period by period)")
        accts = (c.ledger[["account_code", "account_name"]].drop_duplicates()
                 .sort_values("account_code", key=lambda s: s.map(dp._code_sort_key)))
        opts = (accts["account_code"] + "  " + accts["account_name"]).tolist()
        pick = st.selectbox("Account", opts)
        if pick:
            al = rg.account_ledger(c, pick.split("  ")[0])
            a1, a2 = st.columns([3, 2])
            with a1:
                show_table(al, height=420)
            with a2:
                fig = px.line(al, x="Period", y="Closing", markers=True, color_discrete_sequence=SERIES,
                              title="Closing balance (Dr +)")
                st.plotly_chart(style_fig(fig, 420, c.units_label), width="stretch")


# =========================================================================== #
# PAGE 5 - Management reports
# =========================================================================== #
def page_management(c: rg.ReportContext, view: str, scope: str):
    need_data()
    t = st.tabs(["Variance Analysis", "Key Ratios", "Trend Analysis", "Cost Centre P&L", "Top 10",
                 "Working Capital", "Account Movements"])

    with t[0]:
        o1, o2 = st.columns([1, 1])
        sc = o1.radio("Window", ["Month", "Quarter", "YTD", "Full Year"],
                      index=["Month", "Quarter", "YTD", "Full Year"].index(scope), horizontal=True, key="var_sc")
        level = o2.radio("Level", ["Line", "Account"], horizontal=True, key="var_lvl")
        st.markdown(f"#### Variance Analysis — {rg.scope_label(c, sc)}")
        if rg.window_rows(c, sc, year=c.year - 1).empty:
            st.info(f"No prior-year data for this window (FY{c.year - 1} may be loaded as an annual TB only) - "
                    "use the *Full Year* window or load prior-year monthly TBs for month/quarter/YTD comparisons.")
        if c.budget is None or c.budget.empty:
            st.caption("No budget loaded - upload a budget file (role = Budget) to add budget variances.")
        var = rg.variance_analysis(c, sc, level)
        st.caption("Positive variance = favourable (higher income or lower cost).")
        pct = ("Var vs PY %", "Var vs Budget %")
        if level == "Line":
            show_statement(var, pct_cols=pct)
        else:
            show_table(var, pct_cols=pct, height=520)
        vcol = "Var vs Budget" if "Var vs Budget" in var.columns else "Var vs PY"
        chart = var if level == "Account" else var[var["_type"] == "line"]
        lab = "Line" if level == "Line" else "Account Name"
        chart = chart.reindex(chart[vcol].abs().sort_values(ascending=False).index).head(12)
        if not chart.empty:
            chart = chart.assign(Effect=np.where(chart[vcol] >= 0, "Favourable", "Adverse"))
            fig = px.bar(chart, x=vcol, y=lab, orientation="h", color="Effect",
                         color_discrete_map={"Favourable": GOOD, "Adverse": CRITICAL},
                         title=f"Largest variances ({vcol.replace('Var vs ', 'vs ')})")
            fig.update_yaxes(autorange="reversed", tickformat=None)
            fig.update_xaxes(tickformat=",.0f")
            st.plotly_chart(style_fig(fig, 420), width="stretch")

    with t[1]:
        rat = rg.financial_ratios(c, view)
        st.markdown(f"#### Key Financial Ratios — {view}")
        disp = rat.copy()
        pcols = [x for x in rat.columns if x not in ("Category", "Ratio", "Unit", "Formula")]
        for pc in pcols:
            disp[pc] = [rg.fmt_ratio(v, u) for v, u in zip(rat[pc], rat["Unit"])]
        st.dataframe(disp.drop(columns=["Unit"]), hide_index=True, width="stretch",
                     height=36 * (len(disp) + 1) + 4)
        if len(pcols) > 1:
            sel = st.multiselect("Chart ratios", rat["Ratio"].tolist(), default=["Current ratio", "Quick ratio"])
            if sel:
                long = rat[rat["Ratio"].isin(sel)].melt(id_vars=["Ratio", "Unit"], value_vars=pcols,
                                                        var_name="Period", value_name="Value")
                units_sel = long["Unit"].unique()
                if len(units_sel) > 1:
                    st.caption("Selected ratios use different units - charted separately.")
                for u in units_sel:
                    part = long[long["Unit"] == u].copy()
                    if u == "%":
                        part["Value"] *= 100
                    fig = px.line(part, x="Period", y="Value", color="Ratio", markers=True,
                                  color_discrete_sequence=SERIES, title=f"Ratio trend ({u})")
                    fig = style_fig(fig, 340, {"%": "%", "x": "times", "days": "days"}[u])
                    fig.update_yaxes(tickformat=",.2f" if u == "x" else ",.0f")
                    st.plotly_chart(fig, width="stretch")

    with t[2]:
        tv = st.radio("Granularity", ["Monthly", "Yearly"], horizontal=True, key="trend_v")
        td = rg.trend_data(c, tv)
        st.markdown("#### Trend Analysis")
        if td.empty:
            st.info("No data.")
        else:
            metrics = st.multiselect("Metrics", ["Revenue", "Gross Profit", "EBITDA", "PBT", "PAT",
                                                 "Operating Expenses", "Cost of Sales"],
                                     default=["Revenue", "Gross Profit", "EBITDA", "PAT"])
            if metrics:
                long = td.melt(id_vars="Period", value_vars=metrics, var_name="Metric", value_name="Amount")
                fig = px.line(long, x="Period", y="Amount", color="Metric", markers=True,
                              color_discrete_sequence=SERIES, title="P&L trend")
                st.plotly_chart(style_fig(fig, 400, c.units_label), width="stretch")
            ml = td.melt(id_vars="Period", value_vars=["GP margin %", "EBITDA margin %", "Net margin %"],
                         var_name="Margin", value_name="%")
            fig = px.line(ml, x="Period", y="%", color="Margin", markers=True, color_discrete_sequence=SERIES,
                          title="Margin trend (%)")
            fig = style_fig(fig, 320, "%")
            fig.update_yaxes(tickformat=",.1f")
            st.plotly_chart(fig, width="stretch")
            if tv == "Monthly" and len(td) > 1:
                st.caption("Annual-only TBs appear as a single year-end point; month-wise TBs give a monthly series.")
            show_table(td.drop(columns=["order"]), pct_cols=(), height=360)

    with t[3]:
        ccs = [x for x in rg.cost_centres(c.pl_ledger) if x != dp.NO_CC]
        if not ccs:
            st.info("No cost-centre / department column in the loaded TBs.")
        else:
            det = st.toggle("Show account detail", key="cc_det")
            st.markdown(f"#### Cost Centre P&L — {rg.scope_label(c, scope)}")
            ccpl = rg.cost_centre_pl(c, scope, det)
            show_statement(ccpl)
            long = statement_to_long(ccpl, ["Revenue", "Gross Profit", "EBITDA", "Profit After Tax"],
                                     exclude_cols=("Total",))
            fig = px.bar(long, x="Period", y="Amount", color="Line", barmode="group", color_discrete_sequence=SERIES,
                         title="Contribution by cost centre", labels={"Period": "Cost centre"})
            st.plotly_chart(style_fig(fig, 400, c.units_label), width="stretch")

    with t[4]:
        st.markdown(f"#### Top 10 — {rg.scope_label(c, scope)}")
        a, b = st.columns(2)
        for col, cat, title, color in ((a, "Expense", "Top 10 expenses (excl. income tax)", SERIES[1]),
                                       (b, "Income", "Top 10 income accounts", SERIES[0])):
            with col:
                top = rg.top_n(c, scope, cat)
                st.markdown(f"**{title}**")
                if top.empty:
                    st.info("No data.")
                    continue
                fig = px.bar(top, x="Amount", y="Account Name", orientation="h",
                             color_discrete_sequence=[color], hover_data=["Account Code", "Sub-category"])
                fig.update_yaxes(autorange="reversed", tickformat=None)
                fig.update_xaxes(tickformat=",.0f")
                st.plotly_chart(style_fig(fig, 380), width="stretch")
                show_table(top, pct_cols=("% of total", "Cumulative %"), height=400)

    with t[5]:
        wc = rg.working_capital(c, view)
        st.markdown(f"#### Working Capital Analysis — {view}")
        disp = wc.copy()
        num = [x for x in wc.columns if x not in ("Line", "_type")]
        is_metric = disp["Line"].str.contains(r"\(x\)|\(days\)")
        for col in num:
            disp[col] = [f"{v:,.2f}" if "(x)" in ln else (f"{v:,.0f}" if "(days)" in ln else rg.fmt_amount(v))
                         for v, ln in zip(wc[col], wc["Line"])]
        show_statement(disp, pct_cols=())
        long = statement_to_long(wc, ["Operating working capital", "Net working capital", "Cash & bank"])
        fig = px.line(long, x="Period", y="Amount", color="Line", markers=True, color_discrete_sequence=SERIES,
                      title="Working capital")
        st.plotly_chart(style_fig(fig, 340, c.units_label), width="stretch")
        dl = statement_to_long(wc[is_metric], ["DSO (days)", "DIO (days)", "DPO (days)",
                                               "Cash conversion cycle (days)"])
        fig = px.line(dl, x="Period", y="Amount", color="Line", markers=True, color_discrete_sequence=SERIES,
                      title="Working-capital days")
        st.plotly_chart(style_fig(fig, 320, "days"), width="stretch")

    with t[6]:
        o1, o2 = st.columns(2)
        n = o1.slider("Number of accounts", 5, 25, 10)
        by = o2.radio("Rank by", ["Balance", "Movement"], horizontal=True,
                      help="Balance = largest absolute closing balance; Movement = largest absolute YTD movement.")
        mv, cl = rg.major_account_movements(c, n, by)
        st.markdown(f"#### Monthly movement of major accounts — FY{c.year} (Dr + / Cr -)")
        show_table(mv, height=min(36 * (len(mv) + 1) + 4, 700))
        if not cl.empty:
            pick = st.multiselect("Chart closing balances", cl["Account"].tolist(), default=cl["Account"].tolist()[:4])
            if pick:
                long = cl[cl["Account"].isin(pick)].melt(id_vars="Account", var_name="Period", value_name="Closing")
                fig = px.line(long, x="Period", y="Closing", color="Account", markers=True,
                              color_discrete_sequence=SERIES, title="Closing balance by month (Dr +)")
                st.plotly_chart(style_fig(fig, 400, c.units_label), width="stretch")


# =========================================================================== #
# PAGE 6 - Export
# =========================================================================== #
def page_export(c: rg.ReportContext, view: str, scope: str):
    need_data()
    st.markdown("#### Export reporting pack")
    st.caption(f"Uses the current sidebar filters: {c.entity_label} · {c.cost_centre_label} · as at "
               f"{c.as_at_label} · column view **{view}** · analysis window **{scope}** · {c.units_label}.")
    sel = st.multiselect("Reports to include", rg.ALL_REPORTS, default=rg.ALL_REPORTS)
    c1, c2 = st.columns(2)
    if c1.button("Build Excel pack", type="primary", width="stretch"):
        with st.spinner("Building Excel workbook..."):
            items = rg.build_report_pack(c, sel, view, scope)
            ss["xlsx_bytes"] = rg.export_to_excel(items)
            ss["pack_sig"] = (c.entity_label, c.cost_centre_label, c.year, c.period, view, scope, c.units_label,
                              tuple(sel))
    if c2.button("Build PDF pack", type="primary", width="stretch"):
        with st.spinner("Building PDF..."):
            items = rg.build_report_pack(c, sel, view, scope)
            ss["pdf_bytes"] = rg.export_to_pdf(items)
            ss["pack_sig"] = (c.entity_label, c.cost_centre_label, c.year, c.period, view, scope, c.units_label,
                              tuple(sel))
    stamp = f"{c.year}_P{c.period:02d}"
    sig = (c.entity_label, c.cost_centre_label, c.year, c.period, view, scope, c.units_label, tuple(sel))
    if ss.get("pack_sig") != sig:  # filters changed since last build -> drop stale files
        ss["xlsx_bytes"] = ss["pdf_bytes"] = None
    if ss.get("xlsx_bytes"):
        c1.download_button("⬇️ Download Excel", ss["xlsx_bytes"], f"Financial_Reports_{stamp}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    if ss.get("pdf_bytes"):
        c2.download_button("⬇️ Download PDF", ss["pdf_bytes"], f"Financial_Reports_{stamp}.pdf",
                           "application/pdf", width="stretch")
    st.divider()
    st.markdown("#### Data extracts")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Normalised ledger (CSV)", c.ledger.to_csv(index=False).encode(), "normalised_ledger.csv",
                       "text/csv", width="stretch")
    d2.download_button("⬇️ Account classification (CSV)", ss.classification.to_csv(index=False).encode(),
                       "account_classification.csv", "text/csv", width="stretch")


# =========================================================================== #
# Router
# =========================================================================== #
if page == PAGES[0]:
    page_upload()
elif page == PAGES[1]:
    page_classification()
elif page == PAGES[2]:
    page_validation()
elif page == PAGES[3]:
    page_financial(ctx, view, scope)
elif page == PAGES[4]:
    page_management(ctx, view, scope)
else:
    page_export(ctx, view, scope)
