"""
views/overview.py
-----------------
Dashboard homepage — Search Pulse summary.

Sections:
  1. KPI strip            — total clicks / impressions / anomalies
  2. 28-day trend chart   — B&W dual-axis
  3. Brand Trends         — top growing Brand vs Non-Brand queries side by side
  4. Country Performance  — WoW per market table + bar chart
  5. Anomaly callout      — flagged keywords with reason text
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.anomaly import anomaly_summary, flag_anomalies
from src.filters import (
    add_brand_column,
    domain_label,
    render_filters,
    render_country_selector,
    render_brand_selector,
)
from src.processor import (
    compute_wow,
    compute_wow_by_domain,
    daily_trend,
    top_gainers,
    top_decliners,
)
from src.utils import apply_bw, BW_PALETTE, C_BLACK, C_MID, C_XLIGHT, build_display_table


# ── KPI strip ──────────────────────────────────────────────────────────────────

def _kpi_strip(wow: pd.DataFrame, summary: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)

    total_clicks_curr = int(wow["clicks_curr"].sum())
    total_clicks_prev = int(wow["clicks_prev"].sum())
    click_pct = round((total_clicks_curr - total_clicks_prev) / total_clicks_prev * 100, 1) if total_clicks_prev else 0

    total_imp_curr = int(wow["impressions_curr"].sum())
    total_imp_prev = int(wow["impressions_prev"].sum())
    imp_pct = round((total_imp_curr - total_imp_prev) / total_imp_prev * 100, 1) if total_imp_prev else 0

    c1.metric("Clicks (curr week)", f"{total_clicks_curr:,}", f"{click_pct:+.1f}%")
    c2.metric("Impressions (curr week)", f"{total_imp_curr:,}", f"{imp_pct:+.1f}%")
    c3.metric("Keywords tracked", f"{summary['total_keywords']:,}")
    c4.metric("Anomalies detected", f"{summary['anomalies']}")
    c5.metric("Spikes / Drops", f"{summary['spikes']} ↑  {summary['drops']} ↓")


# ── 28-day trend chart ─────────────────────────────────────────────────────────

def _trend_chart(df: pd.DataFrame) -> None:
    st.subheader("28-day Search Pulse")
    daily = daily_trend(df, ["date"])
    if daily.empty:
        st.info("No daily data available.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["clicks"],
        name="Clicks", mode="lines",
        line=dict(color=C_BLACK, width=2),
        fill="tozeroy", fillcolor="rgba(0,0,0,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["impressions"],
        name="Impressions", mode="lines",
        line=dict(color=C_MID, width=1.5, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        yaxis=dict(title="Clicks", gridcolor=C_XLIGHT),
        yaxis2=dict(title="Impressions", overlaying="y", side="right", showgrid=False),
    )
    apply_bw(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)


# ── Brand Trends ───────────────────────────────────────────────────────────────

def _brand_trends(df: pd.DataFrame, top_n: int) -> None:
    st.subheader("Brand vs Non-Brand — Top Growing Queries")
    st.caption("Identifies rising search intent split by branded and non-branded queries.")

    tab_brand, tab_nonbrand = st.tabs(["Brand Keywords", "Non-Brand Keywords"])

    for tab, brand_type in [(tab_brand, "Brand"), (tab_nonbrand, "Non-Brand")]:
        with tab:
            subset = df[df["brand_type"] == brand_type]
            if subset.empty:
                st.info(f"No {brand_type} data available.")
                continue

            try:
                wow = compute_wow(subset)
            except Exception as exc:
                st.error(f"Could not compute WoW: {exc}")
                continue
            if wow.empty:
                st.info("Not enough data for this period.")
                continue

            gainers  = top_gainers(wow, metric="clicks", n=top_n)
            decliners = top_decliners(wow, metric="clicks", n=top_n)

            col_g, col_d = st.columns(2)
            with col_g:
                st.markdown("**▲ Growing**")
                if gainers.empty:
                    st.info("No gainers.")
                else:
                    st.dataframe(
                        build_display_table(gainers, "clicks"),
                        use_container_width=True,
                        hide_index=True,
                    )
            with col_d:
                st.markdown("**▼ Declining**")
                if decliners.empty:
                    st.info("No decliners.")
                else:
                    st.dataframe(
                        build_display_table(decliners, "clicks"),
                        use_container_width=True,
                        hide_index=True,
                    )

            # Impressions growing — separate expander for extra depth
            with st.expander(f"▲ Growing by Impressions — {brand_type}"):
                imp_gainers = top_gainers(wow, metric="impressions", n=top_n)
                if imp_gainers.empty:
                    st.info("No data.")
                else:
                    st.dataframe(
                        build_display_table(imp_gainers, "impressions"),
                        use_container_width=True,
                        hide_index=True,
                    )


# ── Country Performance ────────────────────────────────────────────────────────

def _country_performance(df: pd.DataFrame) -> None:
    st.subheader("Country Performance — Week-over-Week")
    st.caption("Understand which markets are driving growth or showing drops.")

    country_wow = compute_wow_by_domain(df)
    if country_wow.empty:
        st.info("Not enough data for country comparison.")
        return

    country_wow["market"] = country_wow["domain"].apply(domain_label)

    # Table
    display = country_wow[[
        "market", "clicks_prev", "clicks_curr", "clicks_delta", "clicks_pct",
        "impressions_curr", "impressions_delta",
    ]].copy()

    display["clicks_delta"]      = display["clicks_delta"].apply(
        lambda v: f"▲ {int(v):,}" if v > 0 else f"▼ {abs(int(v)):,}"
    )
    display["clicks_pct"]        = display["clicks_pct"].apply(
        lambda v: f"▲ {v:.1f}%" if v and v > 0 else (f"▼ {abs(v):.1f}%" if v else "new")
    )
    display["impressions_delta"] = display["impressions_delta"].apply(
        lambda v: f"▲ {int(v):,}" if v > 0 else f"▼ {abs(int(v)):,}"
    )

    display = display.rename(columns={
        "market":            "Market",
        "clicks_prev":       "Clicks Prev",
        "clicks_curr":       "Clicks Curr",
        "clicks_delta":      "Clicks Δ",
        "clicks_pct":        "% Chg",
        "impressions_curr":  "Impressions",
        "impressions_delta": "Impr Δ",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Bar chart — click delta by market
    chart_df = country_wow.copy()
    chart_df["market"] = chart_df["domain"].apply(domain_label)
    chart_df = chart_df.sort_values("clicks_delta")
    chart_df["bar_color"] = chart_df["clicks_delta"].apply(
        lambda v: C_BLACK if v >= 0 else C_MID
    )

    fig = go.Figure(go.Bar(
        x=chart_df["clicks_delta"],
        y=chart_df["market"],
        orientation="h",
        marker_color=chart_df["bar_color"],
        text=chart_df["clicks_delta"].apply(lambda v: f"{v:+,}"),
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Click Delta (Curr Week vs Prev Week)",
        yaxis_title="",
        xaxis=dict(zeroline=True, zerolinecolor=C_MID, zerolinewidth=1),
    )
    apply_bw(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)


# ── Anomaly callout ────────────────────────────────────────────────────────────

def _anomaly_section(wow_flagged: pd.DataFrame) -> None:
    anomalies = wow_flagged[wow_flagged["is_anomaly"] == True]
    if anomalies.empty:
        return

    st.subheader(f"Anomaly Alerts — {len(anomalies)} keyword(s) flagged")

    for _, row in anomalies.iterrows():
        icon = "▼" if row.get("anomaly_type") == "drop" else "▲"
        with st.expander(f"{icon} {row['keyword']}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Clicks", f"{int(row['clicks_curr']):,}", f"{int(row['clicks_delta']):+,}")
            col2.metric("Impressions", f"{int(row['impressions_curr']):,}", f"{int(row['impressions_delta']):+,}")
            col3.write(f"**Reason:** {row['anomaly_reason']}")


# ── Main render ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.title("Search Pulse — Overview")
    st.caption("WoW comparison uses last two full ISO weeks (Mon–Sun).")

    if df.empty:
        st.warning("No data loaded. Click **⚡ Quick** or **🔄 Full** in the sidebar.")
        return

    df = add_brand_column(df)

    # ── In-page selectors ────────────────────────────────────────────────────
    with st.container(border=True):
        col_country, col_brand = st.columns([2, 1])
        with col_country:
            df = render_country_selector(df, key="ov_country")
        with col_brand:
            df = render_brand_selector(df, key="ov_brand")

    _, top_n = render_filters(df, prefix="overview")

    wow      = compute_wow(df)
    wow_flag = flag_anomalies(wow)
    summary  = anomaly_summary(wow_flag)

    _kpi_strip(wow_flag, summary)
    st.divider()
    _trend_chart(df)
    st.divider()
    _brand_trends(df, top_n)
    st.divider()
    _country_performance(df)
    st.divider()
    _anomaly_section(wow_flag)
