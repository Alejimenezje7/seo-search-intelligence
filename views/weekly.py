"""
views/weekly.py
---------------
Week-over-Week deep-dive.

Sections:
  1. Period header
  2. Brand / Non-Brand tabs  — top gainers + decliners for each type
  3. Country breakdown       — delta table + scatter per market
  4. Full keyword table      — sortable, searchable
  5. Scatter chart           — current vs previous (B&W)
  6. Export
"""

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly import flag_anomalies
from src.filters import (
    add_brand_column,
    domain_label,
    keyword_search,
    render_filters,
    render_country_selector,
    render_brand_selector,
)
from src.processor import (
    compute_wow,
    compute_wow_by_domain,
    get_last_two_full_weeks,
    top_decliners,
    top_gainers,
)
from src.utils import (
    apply_bw,
    build_display_table,
    C_BLACK,
    C_MID,
    C_LIGHT,
    C_XLIGHT,
    fmt_delta,
    fmt_pct,
)


# ── Period banner ──────────────────────────────────────────────────────────────

def _period_banner(df: pd.DataFrame) -> None:
    try:
        _, _, curr_start, curr_end, prev_start, prev_end = get_last_two_full_weeks(df)
        st.info(
            f"**Current week:** {curr_start:%b %d} – {curr_end:%b %d, %Y}  ·  "
            f"**Previous week:** {prev_start:%b %d} – {prev_end:%b %d, %Y}"
        )
    except Exception:
        st.info("Could not determine week ranges from the available data.")


# ── Brand / Non-Brand tabs ─────────────────────────────────────────────────────

def _brand_tabs(df: pd.DataFrame, metric: str, top_n: int) -> None:
    st.subheader("Brand vs Non-Brand Analysis")

    tab_all, tab_brand, tab_nonbrand = st.tabs(["All Keywords", "Brand", "Non-Brand"])

    for tab, label, subset in [
        (tab_all,      "All",       df),
        (tab_brand,    "Brand",     df[df["brand_type"] == "Brand"]),
        (tab_nonbrand, "Non-Brand", df[df["brand_type"] == "Non-Brand"]),
    ]:
        with tab:
            if subset.empty:
                st.info(f"No {label} data.")
                continue

            try:
                wow = compute_wow(subset)
            except Exception as exc:
                st.error(f"Could not compute WoW for {label}: {exc}")
                continue
            wow = flag_anomalies(wow)

            if wow.empty:
                st.info(f"Not enough data to compare weeks for {label} keywords.")
                continue

            gainers   = top_gainers(wow, metric=metric, n=top_n)
            decliners = top_decliners(wow, metric=metric, n=top_n)

            col_g, col_d = st.columns(2)

            with col_g:
                st.markdown(f"**▲ Top Growing — {label}**")
                if gainers.empty:
                    st.info("No gainers.")
                else:
                    st.dataframe(
                        build_display_table(gainers, metric),
                        use_container_width=True,
                        hide_index=True,
                    )
                    anomaly_count = gainers.get("is_anomaly", pd.Series(False)).sum()
                    if anomaly_count:
                        st.caption(f"⚡ {anomaly_count} anomal{'y' if anomaly_count == 1 else 'ies'} in this list")

            with col_d:
                st.markdown(f"**▼ Top Declining — {label}**")
                if decliners.empty:
                    st.info("No decliners.")
                else:
                    st.dataframe(
                        build_display_table(decliners, metric),
                        use_container_width=True,
                        hide_index=True,
                    )


# ── Country breakdown ──────────────────────────────────────────────────────────

def _country_section(df: pd.DataFrame, metric: str) -> None:
    st.subheader("Country Breakdown")

    country_wow = compute_wow_by_domain(df)
    if country_wow.empty:
        st.info("Not enough data for country comparison.")
        return

    country_wow["market"] = country_wow["domain"].apply(domain_label)

    # Table
    display = country_wow[[
        "market",
        f"{metric}_prev", f"{metric}_curr",
        f"{metric}_delta", f"{metric}_pct",
    ]].copy()
    display[f"{metric}_delta"] = display[f"{metric}_delta"].apply(fmt_delta)
    display[f"{metric}_pct"]   = display[f"{metric}_pct"].apply(fmt_pct)
    display = display.rename(columns={
        "market":             "Market",
        f"{metric}_prev":    "Prev Week",
        f"{metric}_curr":    "Curr Week",
        f"{metric}_delta":   "Change",
        f"{metric}_pct":     "% Chg",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Horizontal bar chart
    chart_df = country_wow.copy()
    chart_df["market"] = chart_df["domain"].apply(domain_label)
    chart_df = chart_df.sort_values(f"{metric}_delta")
    bar_colors = [C_BLACK if v >= 0 else C_MID for v in chart_df[f"{metric}_delta"]]

    fig = go.Figure(go.Bar(
        x=chart_df[f"{metric}_delta"],
        y=chart_df["market"],
        orientation="h",
        marker_color=bar_colors,
        text=chart_df[f"{metric}_delta"].apply(lambda v: f"{v:+,.0f}"),
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title=f"{metric.capitalize()} Delta",
        yaxis_title="",
        xaxis=dict(zeroline=True, zerolinecolor=C_MID, zerolinewidth=1),
    )
    apply_bw(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)


# ── Scatter chart ──────────────────────────────────────────────────────────────

def _scatter(wow: pd.DataFrame, metric: str) -> None:
    st.subheader(f"{metric.capitalize()} — Current vs Previous Week")

    plot_df = wow[
        (wow[f"{metric}_curr"] > 0) | (wow[f"{metric}_prev"] > 0)
    ].copy()

    if plot_df.empty:
        st.info("Not enough data for scatter plot.")
        return

    plot_df["is_anomaly"] = plot_df.get("is_anomaly", False)
    plot_df["marker_color"] = plot_df["is_anomaly"].map({True: C_BLACK, False: C_LIGHT})
    plot_df["marker_size"]  = plot_df["is_anomaly"].map({True: 10, False: 6})

    fig = go.Figure()

    for is_anomaly, label, color, size in [
        (False, "Normal",  C_LIGHT, 6),
        (True,  "Anomaly", C_BLACK, 10),
    ]:
        subset = plot_df[plot_df["is_anomaly"] == is_anomaly]
        fig.add_trace(go.Scatter(
            x=subset[f"{metric}_prev"],
            y=subset[f"{metric}_curr"],
            mode="markers",
            name=label,
            text=subset["keyword"],
            marker=dict(color=color, size=size, line=dict(color=C_BLACK, width=0.5)),
        ))

    # y = x reference line (no change)
    max_val = max(plot_df[f"{metric}_curr"].max(), plot_df[f"{metric}_prev"].max())
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color=C_MID, dash="dash", width=1),
    )
    fig.update_layout(
        xaxis_title=f"Prev Week {metric.capitalize()}",
        yaxis_title=f"Curr Week {metric.capitalize()}",
    )
    apply_bw(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)


# ── Export ─────────────────────────────────────────────────────────────────────

def _export(wow: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        wow.to_excel(writer, sheet_name="WoW_All", index=False)
        top_gainers(wow, "clicks", 50).to_excel(writer, sheet_name="Gainers_Clicks", index=False)
        top_gainers(wow, "impressions", 50).to_excel(writer, sheet_name="Gainers_Impressions", index=False)
        top_decliners(wow, "clicks", 50).to_excel(writer, sheet_name="Decliners_Clicks", index=False)
    buffer.seek(0)
    st.download_button(
        label="Export WoW report (.xlsx)",
        data=buffer,
        file_name="wow_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── Main render ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.title("Week-over-Week Analysis")

    if df.empty:
        st.warning("No data loaded.")
        return

    df = add_brand_column(df)

    # ── In-page selectors ────────────────────────────────────────────────────
    with st.container(border=True):
        col_country, col_brand, col_metric = st.columns([2, 1, 1])
        with col_country:
            df = render_country_selector(df, key="ww_country")
        with col_brand:
            df = render_brand_selector(df, key="ww_brand")
        with col_metric:
            metric = st.selectbox(
                "Metric",
                options=["Clicks", "Impressions"],
                key="wow_metric",
            )
            metric = metric.lower()

    _, top_n = render_filters(df, prefix="weekly")

    _period_banner(df)

    st.divider()
    _brand_tabs(df, metric, top_n)

    st.divider()
    _country_section(df, metric)

    # Full keyword table with search
    st.divider()
    st.subheader("Full Keyword Table")
    wow_all = compute_wow(df)
    wow_all = flag_anomalies(wow_all)
    wow_all = keyword_search(wow_all, key="wow_kw_search")

    st.dataframe(
        build_display_table(wow_all, metric),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    _scatter(wow_all, metric)
    st.divider()
    _export(wow_all)
