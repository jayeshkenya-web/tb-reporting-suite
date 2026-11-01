"""
utils/dashboard.py
==================

Interactive, presentation-quality visuals for the Executive Dashboard page.

All figures are Plotly (hover, zoom, click-to-drill) and use one validated, colour-blind-safe
categorical palette in a fixed order, a single y-axis per chart, thin marks and a recessive grid.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import report_generator as rg
from data_processing import NO_CC

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GOOD, WARN, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
NAVY = "#1F3864"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#d03b3b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]
FONT = "Segoe UI, Inter, -apple-system, sans-serif"


def _layout(fig: go.Figure, title: str = "", height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=15, family=FONT)),
        height=height, margin=dict(l=8, r=8, t=(78 if legend else 48) if title else 10, b=8),
        font=dict(family=FONT, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left", title=None),
        hoverlabel=dict(font=dict(family=FONT)))
    fig.update_xaxes(showgrid=False, title=None, linecolor="rgba(128,128,128,0.35)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.16)", zeroline=True, zerolinecolor="rgba(128,128,128,0.4)",
                     title=None, tickformat="~s")
    return fig


def compact(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    a = abs(v)
    t = (f"{a / 1e9:,.2f}B" if a >= 1e9 else f"{a / 1e6:,.1f}M" if a >= 1e6
         else f"{a / 1e3:,.1f}K" if a >= 1e3 else f"{a:,.0f}")
    return f"-{t}" if v < 0 else t


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def monthly_metrics(ctx: rg.ReportContext, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """P&L metrics for every period of the selected year up to the as-at period."""
    df = ctx.pl_ledger if df is None else df
    if df is None or df.empty:
        return pd.DataFrame()
    yr = df[(df["year"] == ctx.year) & (df["period"] <= ctx.period)]
    rows = []
    for p, grp in yr.groupby("period"):
        rows.append({"period": int(p), "Period": ctx.label(ctx.year, int(p)), **rg.pl_metrics(grp)})
    return pd.DataFrame(rows).sort_values("period") if rows else pd.DataFrame()


def monthly_balances(ctx: rg.ReportContext) -> pd.DataFrame:
    rows = []
    for p in sorted(ctx.ledger.loc[(ctx.ledger["year"] == ctx.year) & (ctx.ledger["period"] <= ctx.period),
                                   "period"].unique()):
        pos = rg._position(ctx, ctx.year, int(p))
        rows.append({"period": int(p), "Period": ctx.label(ctx.year, int(p)), "Cash": pos["Cash & Bank"],
                     "NWC": pos["Total current assets"] - pos["Total current liabilities"],
                     "Receivables": pos["Trade Receivables"], "Inventory": pos["Inventory"],
                     "Payables": pos["Trade Payables"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# KPI cards (HTML with inline SVG sparklines)
# --------------------------------------------------------------------------- #
def sparkline_svg(values: list[float], color: str = SERIES[0], w: int = 100, h: int = 34) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pts = [(i * (w - 4) / (len(vals) - 1) + 2, h - 3 - (v - lo) / rng * (h - 8)) for i, v in enumerate(vals)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    area = path + f" L{pts[-1][0]:.1f},{h} L{pts[0][0]:.1f},{h} Z"
    lx, ly = pts[-1]
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<path d="{area}" fill="{color}" fill-opacity="0.12"/>'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>'
            f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" fill="{color}" stroke="white" stroke-width="1.5"/></svg>')


def kpi_card(title: str, value: str, delta: float | None = None, delta_label: str = "", spark: str = "",
             higher_is_better: bool = True, sub: str = "") -> str:
    d_html = ""
    if delta is not None and not np.isnan(delta):
        good = (delta >= 0) == higher_is_better
        col = GOOD if good else CRITICAL
        arrow = "▲" if delta >= 0 else "▼"
        d_html = (f'<span class="kpi-delta" style="color:{col}">{arrow} {abs(delta) * 100:.1f}%</span>'
                  f'<span class="kpi-dlabel"> {html.escape(delta_label)}</span>')
    return (f'<div class="kpi-card"><div class="kpi-title">{html.escape(title)}</div>'
            f'<div class="kpi-row"><div><div class="kpi-value">{html.escape(value)}</div>'
            f'<div class="kpi-sub">{d_html or html.escape(sub)}</div></div>'
            f'<div class="kpi-spark">{spark}</div></div></div>')


KPI_CSS = f"""
<style>
.kpi-grid {{display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 6px 0 18px 0;}}
.kpi-card {{border-radius: 14px; padding: 14px 16px 10px 16px; background: linear-gradient(180deg, rgba(42,120,214,0.07), rgba(42,120,214,0.02));
           border: 1px solid rgba(42,120,214,0.18); box-shadow: 0 1px 2px rgba(16,24,40,0.05);
           transition: transform .15s ease, box-shadow .15s ease;}}
.kpi-card:hover {{transform: translateY(-2px); box-shadow: 0 6px 16px rgba(16,24,40,0.10);}}
.kpi-title {{font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .7; font-weight: 600;}}
.kpi-row {{display:flex; justify-content: space-between; align-items: flex-end; gap: 8px;}}
.kpi-value {{font-size: 1.75rem; font-weight: 700; line-height: 1.25;}}
.kpi-sub {{font-size: 0.8rem; min-height: 1.2em; opacity: .9; white-space: nowrap;}}
.kpi-delta {{font-weight: 700;}} .kpi-dlabel {{opacity: .65;}}
.hero {{border-radius: 16px; padding: 18px 22px; margin-bottom: 14px; color: white;
        background: linear-gradient(120deg, {NAVY} 0%, #2a78d6 60%, #1baf7a 140%);}}
.hero h2 {{margin: 0; color: white; font-size: 1.45rem;}} .hero p {{margin: 4px 0 0 0; opacity: .85;}}
.insight {{border-left: 4px solid #2a78d6; background: rgba(42,120,214,0.06); padding: 8px 12px; border-radius: 8px;
           margin: 4px 0; font-size: .9rem;}}
</style>
"""


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_pl_waterfall(m: dict, title: str) -> go.Figure:
    steps = [("Revenue", m["Revenue"], "absolute"), ("Cost of sales", -m["Cost of Sales"], "relative"),
             ("Gross profit", 0, "total"), ("Other income", m["Other Income"], "relative"),
             ("Operating costs", -m["Operating Expenses"], "relative"), ("EBITDA", 0, "total"),
             ("Depreciation", -m["Depreciation"], "relative"), ("Finance costs", -m["Finance Costs"], "relative"),
             ("Income tax", -m["Income Tax"], "relative"), ("Profit after tax", 0, "total")]
    totals = {"Gross profit": m["Gross Profit"], "EBITDA": m["EBITDA"], "Profit after tax": m["PAT"]}
    text = [compact(totals.get(s[0], s[1])) for s in steps]
    fig = go.Figure(go.Waterfall(
        x=[s[0] for s in steps], y=[s[1] for s in steps], measure=[s[2] for s in steps], text=text,
        textposition="outside", cliponaxis=False,
        connector=dict(line=dict(color="rgba(128,128,128,0.45)", width=1, dash="dot")),
        increasing=dict(marker=dict(color=SERIES[2])), decreasing=dict(marker=dict(color=SERIES[1])),
        totals=dict(marker=dict(color=SERIES[0])),
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>"))
    return _layout(fig, title, 400, legend=False)


def fig_trend(mm: pd.DataFrame, bm: pd.DataFrame | None, metric: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=mm["Period"], y=mm[metric], name=f"Actual {metric.lower()}", marker_color=SERIES[0],
                marker_line_width=0, hovertemplate="%{x}<br>Actual: %{y:,.0f}<extra></extra>")
    if bm is not None and not bm.empty and metric in bm:
        b = bm.set_index("Period").reindex(mm["Period"])[metric]
        fig.add_scatter(x=mm["Period"], y=b, name="Budget", mode="lines+markers",
                        line=dict(color=SERIES[1], width=2, dash="dash"), marker=dict(size=8),
                        hovertemplate="%{x}<br>Budget: %{y:,.0f}<extra></extra>")
    fig.update_layout(bargap=0.3, hovermode="x unified", clickmode="event+select")
    return _layout(fig, title, 380)


def fig_margins(mm: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    rev = mm["Revenue"].replace(0, np.nan)
    for i, (k, lab) in enumerate([("Gross Profit", "Gross margin"), ("EBITDA", "EBITDA margin"),
                                  ("PAT", "Net margin")]):
        fig.add_scatter(x=mm["Period"], y=mm[k] / rev * 100, name=lab, mode="lines+markers",
                        line=dict(color=SERIES[i], width=2, shape="spline"), marker=dict(size=8),
                        hovertemplate=f"{lab}: %{{y:.1f}}%<extra></extra>")
    fig.update_layout(hovermode="x unified")
    fig = _layout(fig, "Margin trend (%)", 340)
    fig.update_yaxes(ticksuffix="%", tickformat=".0f")
    return fig


def fig_expense_sunburst(rows: pd.DataFrame, title: str) -> go.Figure:
    """Click a segment to drill from sub-category down to accounts; click the centre to go back."""
    ex = rows[(rows["category"] == "Expense") & (rows["sub_category"] != "Income Tax")]
    g = ex.groupby(["sub_category", "account_code", "account_name"], as_index=False)["movement"].sum()
    g = g[g["movement"] > 0.5]
    if g.empty:
        return _layout(go.Figure(), title, 420, legend=False)
    subs = g.groupby("sub_category")["movement"].sum().sort_values(ascending=False)
    ids, labels, parents, values, colors = ["root"], ["Total costs"], [""], [float(subs.sum())], ["rgba(0,0,0,0)"]
    for i, (s, v) in enumerate(subs.items()):
        c = SERIES[i % len(SERIES)]
        ids.append(s); labels.append(s); parents.append("root"); values.append(float(v)); colors.append(c)
        for _, r in g[g["sub_category"] == s].sort_values("movement", ascending=False).iterrows():
            ids.append(f"{s}/{r['account_code']}"); labels.append(f"{r['account_name']}")
            parents.append(s); values.append(float(r["movement"])); colors.append(c)
    fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
                                marker=dict(colors=colors, line=dict(color="white", width=2)),
                                insidetextorientation="radial", maxdepth=2,
                                hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percentRoot:.1%} of costs"
                                              "<extra></extra>"))
    return _layout(fig, title, 440, legend=False)


def fig_bs_treemap(ctx: rg.ReportContext) -> go.Figure:
    cur = ctx.ledger[(ctx.ledger["pkey"] == ctx.pkey) & ~ctx.ledger["is_pl"]]
    asset_subs = {"Non-Current Assets", "Inventory", "Trade Receivables", "Other Current Assets", "Cash & Bank"}
    g = cur.groupby(["category", "sub_category", "account_code", "account_name"], as_index=False)["closing"].sum()
    g["value"] = np.where(g["sub_category"].isin(asset_subs), g["closing"], -g["closing"])
    g = g[g["value"] > 0.5]
    profit = -ctx.ledger.loc[(ctx.ledger["pkey"] == ctx.pkey) & ctx.ledger["is_pl"], "closing"].sum()
    ids, labels, parents, values, colors = [], [], [], [], []
    sides = {"Asset": ("Assets", SERIES[0]), "Liability": ("Liabilities", SERIES[1]), "Equity": ("Equity", SERIES[2])}
    for cat, (side, col) in sides.items():
        part = g[g["category"] == cat]
        extra = profit if cat == "Equity" and profit > 0 else 0
        tot = part["value"].sum() + extra
        if tot <= 0:
            continue
        ids.append(side); labels.append(side); parents.append(""); values.append(float(tot)); colors.append(col)
        for s, sp in part.groupby("sub_category"):
            sid = f"{side}/{s}"
            ids.append(sid); labels.append(s); parents.append(side); values.append(float(sp["value"].sum()))
            colors.append(col)
            for _, r in sp.iterrows():
                ids.append(f"{sid}/{r['account_code']}"); labels.append(r["account_name"]); parents.append(sid)
                values.append(float(r["value"])); colors.append(col)
        if extra:
            ids.append(f"{side}/Profit for the period"); labels.append("Profit for the period (YTD)")
            parents.append(side); values.append(float(extra)); colors.append(col)
    fig = go.Figure(go.Treemap(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
                               marker=dict(colors=colors, line=dict(color="white", width=2)), maxdepth=2,
                               texttemplate="<b>%{label}</b><br>%{value:,.3s}",
                               hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percentParent:.1%} of %{parent}"
                                             "<extra></extra>", pathbar=dict(visible=True)))
    return _layout(fig, f"Balance sheet map — as at {ctx.as_at_label} (click to drill)", 440, legend=False)


def fig_cc_heatmap(ctx: rg.ReportContext, metric: str = "Contribution") -> go.Figure | None:
    led = ctx.pl_ledger[(ctx.pl_ledger["year"] == ctx.year) & (ctx.pl_ledger["period"] <= ctx.period)
                        & ctx.pl_ledger["is_pl"] & (ctx.pl_ledger["cost_centre"] != NO_CC)]
    if led.empty or led["cost_centre"].nunique() < 2:
        return None
    if metric == "Revenue":
        led = led[led["sub_category"] == "Revenue"]
    elif metric == "Costs":
        led = led[led["category"] == "Expense"]
    sign = 1 if metric == "Costs" else -1
    pv = (led.pivot_table(index="cost_centre", columns="period", values="movement", aggfunc="sum", fill_value=0) * sign)
    cols = [ctx.label(ctx.year, int(p)) for p in pv.columns]
    z = pv.to_numpy()
    zmax = float(np.nanmax(np.abs(z))) or 1.0
    diverging = metric == "Contribution" and (z < 0).any()
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=list(pv.index), xgap=3, ygap=3,
        colorscale=DIVERGING if diverging else [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)],
        zmid=0 if diverging else None, zmin=-zmax if diverging else 0, zmax=zmax,
        text=[[compact(v) for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z:,.0f}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0, tickformat="~s")))
    fig = _layout(fig, f"{metric} by cost centre and month", 120 + 46 * len(pv.index), legend=False)
    fig.update_yaxes(tickformat=None, gridcolor="rgba(0,0,0,0)", zeroline=False)
    return fig


def fig_gauges(ctx: rg.ReportContext) -> go.Figure:
    r = rg._ratios_from_position(rg._position(ctx, ctx.year, ctx.period))
    specs = [  # (label, value, max, suffix, bands [(to, colour)], higher_is_better)
        ("Current ratio", r["Current ratio"], 4, "x", [(1, "#f6d0d0"), (1.5, "#fdebc4"), (4, "#d2efd2")]),
        ("Gross margin", (r["Gross profit margin"] or 0) * 100, 80, "%", [(20, "#f6d0d0"), (35, "#fdebc4"), (80, "#d2efd2")]),
        ("Net margin", (r["Net profit margin"] or 0) * 100, 40, "%", [(5, "#f6d0d0"), (10, "#fdebc4"), (40, "#d2efd2")]),
        ("Debt / equity", r["Debt to equity"], 3, "x", [(1, "#d2efd2"), (2, "#fdebc4"), (3, "#f6d0d0")]),
    ]
    fig = go.Figure()
    for i, (lab, val, mx, suf, bands) in enumerate(specs):
        val = 0 if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
        if val > mx:  # widen the dial so the needle stays on scale; extend the last band
            mx = float(np.ceil(val * 1.15))
            bands = bands[:-1] + [(mx, bands[-1][1])]
        lo, steps = 0, []
        for to, col in bands:
            steps.append(dict(range=[lo, to], color=col))
            lo = to
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=val, domain=dict(row=0, column=i),
            title=dict(text=lab, font=dict(size=13)),
            number=dict(suffix=suf, valueformat=".2f" if suf == "x" else ".1f", font=dict(size=24)),
            gauge=dict(axis=dict(range=[0, mx], tickwidth=1, tickcolor="rgba(128,128,128,0.5)"),
                       bar=dict(color=NAVY, thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                       steps=steps)))
    fig.update_layout(grid=dict(rows=1, columns=4, pattern="independent"))
    return _layout(fig, f"Financial health — as at {ctx.as_at_label}", 230, legend=False)


def fig_budget_variance(mm: pd.DataFrame, bm: pd.DataFrame, metric: str) -> go.Figure:
    b = bm.set_index("Period").reindex(mm["Period"])[metric].to_numpy()
    var = mm[metric].to_numpy() - b
    if metric in ("Cost of Sales", "Operating Expenses"):
        var = -var
    fig = go.Figure(go.Bar(x=mm["Period"], y=var, marker_color=np.where(var >= 0, GOOD, CRITICAL),
                           text=[compact(v) for v in var], textposition="outside", cliponaxis=False,
                           customdata=np.where(var >= 0, "Favourable", "Adverse"),
                           hovertemplate="%{x}<br>%{customdata}: %{y:,.0f}<extra></extra>"))
    return _layout(fig, f"{metric}: variance vs budget (green = favourable)", 320, legend=False)


def fig_cash(bal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=bal["Period"], y=bal["Cash"], name="Cash & bank", mode="lines+markers", fill="tozeroy",
                    line=dict(color=SERIES[0], width=2, shape="spline"), marker=dict(size=8),
                    fillcolor="rgba(42,120,214,0.12)", hovertemplate="%{x}<br>Cash: %{y:,.0f}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "Cash & bank balance", 320, legend=False)


def fig_wc_days(ctx: rg.ReportContext) -> go.Figure:
    rows = []
    for p in sorted(ctx.ledger.loc[(ctx.ledger["year"] == ctx.year) & (ctx.ledger["period"] <= ctx.period),
                                   "period"].unique()):
        r = rg._ratios_from_position(rg._position(ctx, ctx.year, int(p)))
        rows.append((ctx.label(ctx.year, int(p)), r["Receivable days (DSO)"], r["Inventory days (DIO)"],
                     r["Payable days (DPO)"]))
    df = pd.DataFrame(rows, columns=["Period", "DSO", "DIO", "DPO"])
    fig = go.Figure()
    for i, (k, lab) in enumerate([("DSO", "Receivable days"), ("DIO", "Inventory days"), ("DPO", "Payable days")]):
        fig.add_scatter(x=df["Period"], y=df[k], name=lab, mode="lines+markers", line=dict(color=SERIES[i], width=2),
                        marker=dict(size=8), hovertemplate=f"{lab}: %{{y:.0f}} days<extra></extra>")
    fig.update_layout(hovermode="x unified")
    fig = _layout(fig, "Working-capital days", 320)
    fig.update_yaxes(tickformat=".0f", ticksuffix=" d")
    return fig


def insights(ctx: rg.ReportContext, mm: pd.DataFrame, bm: pd.DataFrame | None) -> list[str]:
    """A few plain-language observations generated from the numbers (no judgement calls)."""
    out = []
    if mm.empty:
        return out
    ytd = mm[["Revenue", "Gross Profit", "EBITDA", "PAT"]].sum()
    if len(mm) >= 2:
        last, prev = mm.iloc[-1], mm.iloc[-2]
        if prev["Revenue"]:
            ch = (last["Revenue"] - prev["Revenue"]) / abs(prev["Revenue"])
            out.append(f"Revenue in {last['Period']} was {compact(last['Revenue'])}, "
                       f"{'up' if ch >= 0 else 'down'} {abs(ch) * 100:.1f}% on {prev['Period']}.")
        best = mm.loc[mm["PAT"].idxmax()]
        out.append(f"Best month for profit after tax: {best['Period']} ({compact(best['PAT'])}).")
    if bm is not None and not bm.empty:
        b = bm[bm["period"] <= ctx.period][["Revenue", "PAT"]].sum()
        if b["Revenue"]:
            v = (ytd["Revenue"] - b["Revenue"]) / abs(b["Revenue"])
            out.append(f"YTD revenue is {abs(v) * 100:.1f}% {'above' if v >= 0 else 'below'} budget "
                       f"({compact(ytd['Revenue'])} vs {compact(b['Revenue'])}).")
    if ytd["Revenue"]:
        out.append(f"YTD gross margin {ytd['Gross Profit'] / ytd['Revenue'] * 100:.1f}%, "
                   f"EBITDA margin {ytd['EBITDA'] / ytd['Revenue'] * 100:.1f}%, "
                   f"net margin {ytd['PAT'] / ytd['Revenue'] * 100:.1f}%.")
    return out
