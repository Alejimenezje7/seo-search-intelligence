"""
views/buying.py
---------------
Buying & Trading — Demand Intelligence view.

Surfaces rising and falling product demand from organic search signals,
helping the buying team spot which categories and products are gaining
or losing consumer interest.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.filters import (
    add_brand_column,
    add_category_column,
    render_country_selector,
    render_filters,
    PRODUCT_CATEGORIES,
)
from src.processor import (
    compute_wow,
    compute_wow_by_category,
    top_gainers,
    top_decliners,
)
from src.utils import (
    apply_bw,
    build_display_table,
    C_BLACK, C_MID,
    fmt_delta, fmt_pct,
)


# ── KPI strip ──────────────────────────────────────────────────────────────────

def _kpi_strip(df: pd.DataFrame, cat_wow: pd.DataFrame) -> None:
    total_impr = int(df["impressions"].sum()) if not df.empty else 0

    top_rising = top_falling = "—"
    ones_to_watch = 0

    if not cat_wow.empty:
        rising  = cat_wow[cat_wow["impressions_pct"].notna() & (cat_wow["impressions_pct"] > 0)]
        falling = cat_wow[cat_wow["impressions_pct"].notna() & (cat_wow["impressions_pct"] < 0)]
        if not rising.empty:
            top_rising  = rising.nlargest(1, "impressions_pct")["product_category"].iloc[0]
        if not falling.empty:
            top_falling = falling.nsmallest(1, "impressions_pct")["product_category"].iloc[0]

    try:
        nb = df[df["brand_type"] == "Non-Brand"]
        if not nb.empty:
            wow_nb = compute_wow(nb, min_clicks=5, min_impressions=20)
            gainers = top_gainers(wow_nb, "impressions", n=200)
            ones_to_watch = len(gainers[gainers["impressions_pct"].notna() & (gainers["impressions_pct"] > 20)])
    except Exception:
        pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Search Demand",    f"{total_impr:,}", help="Total impressions in loaded data")
    c2.metric("Top Rising Category",    top_rising,        help="Category with highest WoW impression growth")
    c3.metric("Top Declining Category", top_falling,       help="Category with highest WoW impression drop")
    c4.metric("Ones to Watch",          f"{ones_to_watch}", help="Non-brand keywords with >20% impression growth WoW")


# ── Category demand chart ───────────────────────────────────────────────────────

def _category_chart(cat_wow: pd.DataFrame) -> None:
    st.subheader("Demand by Category — Week-over-Week")
    st.caption("Search impression change per product category. Positive bars = growing consumer interest.")

    if cat_wow.empty:
        st.info("Not enough data for category comparison.")
        return

    chart = cat_wow[cat_wow["product_category"] != "Other"].copy()
    chart = chart.sort_values("impressions_delta")
    chart["color"] = chart["impressions_delta"].apply(lambda v: C_BLACK if v >= 0 else C_MID)
    chart["label"] = chart["impressions_delta"].apply(
        lambda v: f"▲ {int(v):,}" if v >= 0 else f"▼ {abs(int(v)):,}"
    )

    fig = go.Figure(go.Bar(
        x=chart["impressions_delta"],
        y=chart["product_category"],
        orientation="h",
        marker_color=chart["color"],
        text=chart["label"],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Impressions Delta (Current Week vs Previous Week)",
        yaxis_title="",
        xaxis=dict(zeroline=True, zerolinecolor=C_MID, zerolinewidth=1),
    )
    apply_bw(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


# ── Ones to Watch ───────────────────────────────────────────────────────────────

def _ones_to_watch(df: pd.DataFrame, top_n: int) -> None:
    st.subheader("Ones to Watch — Rising Search Demand")
    st.caption(
        "Non-brand keywords with the strongest week-over-week impression growth. "
        "High growth = increasing consumer interest — a leading signal for buying decisions."
    )

    nb = df[df["brand_type"] == "Non-Brand"]
    if nb.empty:
        st.info("No non-brand data available.")
        return

    try:
        wow = compute_wow(nb, min_clicks=5, min_impressions=20)
    except Exception as exc:
        st.error(f"Could not compute WoW: {exc}")
        return

    if wow.empty:
        st.info("Not enough data for this period.")
        return

    gainers = top_gainers(wow, "impressions", n=top_n)
    if gainers.empty:
        st.info("No rising keywords found this period.")
        return

    # Attach category
    if "product_category" in df.columns:
        kw_cat = df[["keyword", "product_category"]].drop_duplicates("keyword")
        gainers = gainers.merge(kw_cat, on="keyword", how="left")

    extra = ["product_category"] if "product_category" in gainers.columns else []
    disp_cols = ["keyword"] + extra + ["impressions_prev", "impressions_curr", "impressions_delta", "impressions_pct"]
    disp_cols = [c for c in disp_cols if c in gainers.columns]
    display = gainers[disp_cols].copy()
    display["impressions_delta"] = display["impressions_delta"].apply(fmt_delta)
    display["impressions_pct"]   = display["impressions_pct"].apply(fmt_pct)
    display = display.rename(columns={
        "keyword":           "Keyword",
        "product_category":  "Category",
        "impressions_prev":  "Prev Week",
        "impressions_curr":  "This Week",
        "impressions_delta": "Change",
        "impressions_pct":   "% Growth",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)


# ── Cooling demand ──────────────────────────────────────────────────────────────

def _cooling_demand(df: pd.DataFrame, top_n: int) -> None:
    st.caption("Non-brand keywords losing search traction — review inventory exposure.")

    nb = df[df["brand_type"] == "Non-Brand"]
    if nb.empty:
        return

    try:
        wow = compute_wow(nb, min_clicks=5, min_impressions=20)
    except Exception:
        return

    if wow.empty:
        return

    decliners = top_decliners(wow, "impressions", n=top_n)
    if decliners.empty:
        st.info("No significantly declining keywords detected.")
        return

    if "product_category" in df.columns:
        kw_cat = df[["keyword", "product_category"]].drop_duplicates("keyword")
        decliners = decliners.merge(kw_cat, on="keyword", how="left")

    extra = ["product_category"] if "product_category" in decliners.columns else []
    disp_cols = ["keyword"] + extra + ["impressions_prev", "impressions_curr", "impressions_delta", "impressions_pct"]
    disp_cols = [c for c in disp_cols if c in decliners.columns]
    display = decliners[disp_cols].copy()
    display["impressions_delta"] = display["impressions_delta"].apply(fmt_delta)
    display["impressions_pct"]   = display["impressions_pct"].apply(fmt_pct)
    display = display.rename(columns={
        "keyword":           "Keyword",
        "product_category":  "Category",
        "impressions_prev":  "Prev Week",
        "impressions_curr":  "This Week",
        "impressions_delta": "Change",
        "impressions_pct":   "% Change",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)


# ── Category deep-dive tabs ────────────────────────────────────────────────────

def _category_detail(df: pd.DataFrame, top_n: int) -> None:
    st.subheader("Category Deep Dive")
    st.caption("Top growing and declining keywords within each product category.")

    if "product_category" not in df.columns:
        return

    cats = [c for c in PRODUCT_CATEGORIES.keys() if c in df["product_category"].unique()]
    if not cats:
        st.info("No categorized keywords found in the loaded data.")
        return

    tabs = st.tabs(cats)
    for tab, cat in zip(tabs, cats):
        with tab:
            subset = df[df["product_category"] == cat]
            if subset.empty:
                st.info(f"No data for {cat}.")
                continue
            try:
                wow = compute_wow(subset, min_clicks=2, min_impressions=10)
            except Exception as exc:
                st.error(f"Could not compute WoW: {exc}")
                continue
            if wow.empty:
                st.info("Not enough history for WoW comparison.")
                continue

            col_g, col_d = st.columns(2)
            with col_g:
                st.caption(f"▲ Rising demand — {cat}")
                g = top_gainers(wow, "impressions", n=top_n)
                if g.empty:
                    st.info("No gainers.")
                else:
                    st.dataframe(build_display_table(g, "impressions"), use_container_width=True, hide_index=True)
            with col_d:
                st.caption(f"▼ Cooling demand — {cat}")
                d = top_decliners(wow, "impressions", n=top_n)
                if d.empty:
                    st.info("No decliners.")
                else:
                    st.dataframe(build_display_table(d, "impressions"), use_container_width=True, hide_index=True)


# ── Main render ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.title("Buying & Trading — Demand Intelligence")
    st.caption(
        "Organic search as a leading indicator of consumer demand. "
        "Rising impressions signal growing product interest before purchase."
    )

    if df.empty:
        st.warning("No data loaded. Click **⚡ Quick** or **🔄 Full** in the sidebar.")
        return

    df = add_brand_column(df)
    df = add_category_column(df)

    with st.container(border=True):
        df = render_country_selector(df, key="buy_country")

    _, top_n = render_filters(df, prefix="buy")

    try:
        nb_df   = df[df["brand_type"] == "Non-Brand"]
        cat_wow = compute_wow_by_category(nb_df)
    except Exception:
        cat_wow = pd.DataFrame()

    _kpi_strip(df, cat_wow)
    st.divider()
    _category_chart(cat_wow)
    st.divider()
    _ones_to_watch(df, top_n)
    st.divider()
    with st.expander("▼ Cooling Demand — keywords losing search traction", expanded=False):
        _cooling_demand(df, top_n)
    st.divider()
    _category_detail(df, top_n)
