"""
views/activation.py
-------------------
Digital Activation — Campaign Signal Intelligence view.

Monitors how commercial campaign keywords perform in organic search,
helping the activation team understand which messages drive search behavior
and which markets respond best to campaign activity.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.filters import (
    add_brand_column,
    add_campaign_column,
    render_country_selector,
    render_filters,
    CAMPAIGN_CATEGORIES,
)
from src.processor import (
    compute_wow,
    top_gainers,
    top_decliners,
)
from src.utils import (
    apply_bw,
    style_pct_cols,
    C_BLACK, C_MID, C_XLIGHT,
    fmt_delta, fmt_pct,
)


# ── KPI strip ──────────────────────────────────────────────────────────────────

def _kpi_strip(campaign_df: pd.DataFrame) -> None:
    if campaign_df.empty:
        st.info("No campaign keywords detected in the loaded data. Refresh data or broaden the market filter.")
        return

    total_kws  = campaign_df["keyword"].nunique() if "keyword" in campaign_df.columns else 0
    total_impr = int(campaign_df["impressions"].sum())
    total_clks = int(campaign_df["clicks"].sum())
    avg_ctr    = (
        round(campaign_df["clicks"].sum() / campaign_df["impressions"].sum() * 100, 2)
        if campaign_df["impressions"].sum() > 0 else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Campaign Keywords",  f"{total_kws:,}",    help="Unique keywords matching campaign categories")
    c2.metric("Total Impressions",  f"{total_impr:,}",   help="Search demand for all campaign terms")
    c3.metric("Total Clicks",       f"{total_clks:,}")
    c4.metric("Avg CTR",            f"{avg_ctr:.2f}%",   help="Click-through rate on campaign keywords")


# ── Campaign category performance table ────────────────────────────────────────

def _category_performance(campaign_df: pd.DataFrame) -> None:
    st.subheader("Campaign Category Performance")
    st.caption("Search volume and engagement per campaign theme.")

    if campaign_df.empty or "campaign_category" not in campaign_df.columns:
        st.info("No campaign data available.")
        return

    agg = (
        campaign_df
        .groupby("campaign_category", as_index=False)
        .agg(
            keywords    =("keyword",      "nunique"),
            impressions =("impressions",  "sum"),
            clicks      =("clicks",       "sum"),
        )
    )
    agg["ctr"] = agg.apply(
        lambda r: f"{r['clicks'] / r['impressions'] * 100:.2f}%" if r["impressions"] > 0 else "0%",
        axis=1,
    )
    agg = agg.sort_values("impressions", ascending=False)
    agg = agg.rename(columns={
        "campaign_category": "Campaign Type",
        "keywords":          "Unique Keywords",
        "impressions":       "Impressions",
        "clicks":            "Clicks",
        "ctr":               "CTR",
    })
    st.dataframe(agg, use_container_width=True, hide_index=True)

    # Horizontal bar — impressions by category
    bar_data = agg.head(10)
    fig = go.Figure(go.Bar(
        y=bar_data["Campaign Type"],
        x=bar_data["Impressions"],
        orientation="h",
        marker_color=C_BLACK,
        text=bar_data["Impressions"].apply(lambda v: f"{int(v):,}"),
        textposition="outside",
    ))
    fig.update_layout(xaxis_title="Impressions", yaxis_title="")
    apply_bw(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)


# ── Campaign WoW trends ─────────────────────────────────────────────────────────

def _campaign_wow(campaign_df: pd.DataFrame, top_n: int) -> None:
    st.subheader("Campaign Keywords — Week-over-Week Trend")
    st.caption("Which campaign terms are gaining or losing search traction this week.")

    if campaign_df.empty:
        st.info("No campaign data available.")
        return

    tab_rising, tab_declining = st.tabs(["▲ Rising Campaign Terms", "▼ Declining Campaign Terms"])

    for tab, fn, label in [
        (tab_rising,    top_gainers,   "Rising"),
        (tab_declining, top_decliners, "Declining"),
    ]:
        with tab:
            try:
                wow = compute_wow(campaign_df, min_clicks=2, min_impressions=10)
            except Exception as exc:
                st.error(f"Could not compute WoW: {exc}")
                continue
            if wow.empty:
                st.info("Not enough data for WoW comparison.")
                continue

            result = fn(wow, "impressions", n=top_n)
            if result.empty:
                st.info(f"No {label.lower()} campaign keywords this period.")
                continue

            # Attach campaign category
            if "campaign_category" in campaign_df.columns:
                kw_cat = campaign_df[["keyword", "campaign_category"]].drop_duplicates("keyword")
                result = result.merge(kw_cat, on="keyword", how="left")

            extra = ["campaign_category"] if "campaign_category" in result.columns else []
            disp_cols = ["keyword"] + extra + [
                "impressions_prev", "impressions_curr", "impressions_delta", "impressions_pct"
            ]
            disp_cols = [c for c in disp_cols if c in result.columns]
            display = result[disp_cols].copy()
            display["impressions_delta"] = display["impressions_delta"].apply(fmt_delta)
            display["impressions_pct"]   = display["impressions_pct"].apply(fmt_pct)
            display = display.rename(columns={
                "keyword":            "Keyword",
                "campaign_category":  "Campaign Type",
                "impressions_prev":   "Prev Week",
                "impressions_curr":   "This Week",
                "impressions_delta":  "Change",
                "impressions_pct":    "% Change",
            })
            st.dataframe(style_pct_cols(display), use_container_width=True, hide_index=True)


# ── Campaign category deep-dive tabs ──────────────────────────────────────────

def _category_tabs(campaign_df: pd.DataFrame, top_n: int) -> None:
    st.subheader("Campaign Type — Keyword Detail")
    st.caption("Top-performing keywords within each campaign category.")

    if campaign_df.empty or "campaign_category" not in campaign_df.columns:
        st.info("No campaign data.")
        return

    active_cats = [c for c in CAMPAIGN_CATEGORIES.keys() if c in campaign_df["campaign_category"].unique()]
    if not active_cats:
        st.info("No active campaign categories found in the loaded data.")
        return

    tabs = st.tabs(active_cats)
    for tab, cat in zip(tabs, active_cats):
        with tab:
            subset = campaign_df[campaign_df["campaign_category"] == cat]
            if subset.empty:
                st.info(f"No data for {cat}.")
                continue

            top_kws = (
                subset
                .groupby("keyword", as_index=False)
                .agg(impressions=("impressions", "sum"), clicks=("clicks", "sum"))
                .sort_values("impressions", ascending=False)
                .head(top_n)
            )
            top_kws["ctr"] = top_kws.apply(
                lambda r: f"{r['clicks'] / r['impressions'] * 100:.2f}%"
                if r["impressions"] > 0 else "0%",
                axis=1,
            )

            col_table, col_chart = st.columns([2, 1])
            with col_table:
                st.dataframe(
                    top_kws.rename(columns={
                        "keyword": "Keyword", "impressions": "Impressions",
                        "clicks": "Clicks", "ctr": "CTR",
                    }),
                    use_container_width=True, hide_index=True,
                )
            with col_chart:
                chart_data = top_kws.head(10)
                fig = go.Figure(go.Bar(
                    y=chart_data["keyword"],
                    x=chart_data["impressions"],
                    orientation="h",
                    marker_color=C_BLACK,
                ))
                apply_bw(fig, height=280)
                st.plotly_chart(fig, use_container_width=True)


# ── Custom campaign keyword search ─────────────────────────────────────────────

def _custom_search(df: pd.DataFrame) -> None:
    st.subheader("Search Any Campaign Term")
    st.caption("Look up impressions and clicks for any keyword — e.g. a product launch, sale name, or campaign tag.")

    with st.container(border=True):
        search = st.text_input(
            "Keyword / campaign term",
            placeholder="e.g.  black friday,  copa america,  ultraboost 24",
            key="act_search",
        )

    if not search:
        return

    results = df[df["keyword"].str.contains(search, case=False, na=False)]
    if results.empty:
        st.info(f"No keywords found matching '{search}'.")
        return

    st.caption(f"{results['keyword'].nunique()} unique keywords match '{search}'")

    summary = (
        results
        .groupby("keyword", as_index=False)
        .agg(impressions=("impressions", "sum"), clicks=("clicks", "sum"))
        .sort_values("impressions", ascending=False)
        .head(50)
    )
    summary["ctr"] = summary.apply(
        lambda r: f"{r['clicks'] / r['impressions'] * 100:.2f}%" if r["impressions"] > 0 else "0%",
        axis=1,
    )
    st.dataframe(
        summary.rename(columns={
            "keyword": "Keyword", "impressions": "Impressions",
            "clicks": "Clicks", "ctr": "CTR",
        }),
        use_container_width=True, hide_index=True,
    )


# ── Main render ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.title("Digital Activation — Campaign Signal Intelligence")
    st.caption(
        "Monitor how commercial campaign keywords perform in organic search. "
        "Rising impressions = your campaigns are driving search behavior."
    )

    if df.empty:
        st.warning("No data loaded. Click **⚡ Quick** or **🔄 Full** in the sidebar.")
        return

    df = add_brand_column(df)
    df = add_campaign_column(df)

    with st.container(border=True):
        df = render_country_selector(df, key="act_country")

    _, top_n = render_filters(df, prefix="act")

    campaign_df = df[df["campaign_category"].notna()].copy()

    _kpi_strip(campaign_df)
    st.divider()
    _category_performance(campaign_df)
    st.divider()
    _campaign_wow(campaign_df, top_n)
    st.divider()
    _category_tabs(campaign_df, top_n)
    st.divider()
    _custom_search(df)
