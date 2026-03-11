"""
views/mtd.py
------------
Month-to-Date comparison view.
B&W theme, ▲/▼ delta display, brand/non-brand breakdown.
"""

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.anomaly import flag_anomalies
from src.filters import (
    add_brand_column,
    keyword_search,
    render_filters,
    render_country_selector,
    render_brand_selector,
)
from src.processor import (
    compute_mtd,
    daily_trend,
    get_mtd_ranges,
    top_decliners,
    top_gainers,
)
from src.utils import apply_bw, build_display_table, C_BLACK, C_MID, C_LIGHT, C_XLIGHT, fmt_delta, fmt_pct


# ── Period banner ──────────────────────────────────────────────────────────────

def _period_banner(df: pd.DataFrame) -> None:
    try:
        _, _, curr_start, curr_end, prev_start, prev_end = get_mtd_ranges(df)
        st.info(
            f"**Current MTD:** {curr_start:%b %d} – {curr_end:%b %d, %Y}  ·  "
            f"**vs. Previous Month (same days):** {prev_start:%b %d} – {prev_end:%b %d, %Y}"
        )
    except Exception:
        st.info("Could not determine MTD ranges from the available data.")


# ── KPI totals ─────────────────────────────────────────────────────────────────

def _kpi_totals(mtd: pd.DataFrame) -> None:
    total_clicks_curr = int(mtd["clicks_curr"].sum())
    total_clicks_prev = int(mtd["clicks_prev"].sum())
    click_pct = round((total_clicks_curr - total_clicks_prev) / total_clicks_prev * 100, 1) if total_clicks_prev else 0

    total_imp_curr = int(mtd["impressions_curr"].sum())
    total_imp_prev = int(mtd["impressions_prev"].sum())
    imp_pct = round((total_imp_curr - total_imp_prev) / total_imp_prev * 100, 1) if total_imp_prev else 0

    avg_pos_curr = round(mtd["position_curr"].mean(), 1) if not mtd.empty else 0
    avg_pos_prev = round(mtd["position_prev"].mean(), 1) if not mtd.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clicks MTD",        f"{total_clicks_curr:,}", f"{click_pct:+.1f}%")
    c2.metric("Impressions MTD",   f"{total_imp_curr:,}",    f"{imp_pct:+.1f}%")
    c3.metric("Avg Position",      f"{avg_pos_curr}",        f"{avg_pos_curr - avg_pos_prev:+.1f}")
    c4.metric("Active keywords",   f"{len(mtd[mtd['clicks_curr'] > 0]):,}")


# ── Brand breakdown ────────────────────────────────────────────────────────────

def _brand_tabs(df: pd.DataFrame, metric: str, top_n: int) -> None:
    st.subheader("Brand vs Non-Brand — MTD")

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
                mtd = compute_mtd(subset)
            except Exception as exc:
                st.error(f"Could not compute MTD: {exc}")
                continue
            if mtd.empty:
                st.info("Not enough data for this period.")
                continue

            gainers   = top_gainers(mtd, metric=metric, n=top_n)
            decliners = top_decliners(mtd, metric=metric, n=top_n)

            col_g, col_d = st.columns(2)
            with col_g:
                st.markdown(f"**▲ Growing — {label}**")
                st.dataframe(
                    build_display_table(gainers, metric) if not gainers.empty else pd.DataFrame(),
                    use_container_width=True, hide_index=True,
                )
            with col_d:
                st.markdown(f"**▼ Declining — {label}**")
                st.dataframe(
                    build_display_table(decliners, metric) if not decliners.empty else pd.DataFrame(),
                    use_container_width=True, hide_index=True,
                )


# ── Cumulative trend ───────────────────────────────────────────────────────────

def _cumulative_chart(df: pd.DataFrame) -> None:
    st.subheader("Cumulative Clicks — Current MTD vs Previous Period")

    try:
        curr_df, prev_df, *_ = get_mtd_ranges(df)
    except Exception:
        st.info("Not enough data for trend chart.")
        return

    def _cum(period_df: pd.DataFrame, label: str) -> pd.DataFrame:
        daily = daily_trend(period_df, ["date"]).sort_values("date")
        daily["day"] = range(1, len(daily) + 1)
        daily["cumulative"] = daily["clicks"].cumsum()
        daily["series"] = label
        return daily[["day", "cumulative", "series"]]

    curr_cum = _cum(curr_df, "Current MTD")
    prev_cum = _cum(prev_df, "Previous Period")
    combined = pd.concat([curr_cum, prev_cum])

    fig = go.Figure()
    for series, color, dash in [("Current MTD", C_BLACK, "solid"), ("Previous Period", C_LIGHT, "dot")]:
        subset = combined[combined["series"] == series]
        fig.add_trace(go.Scatter(
            x=subset["day"], y=subset["cumulative"],
            name=series, mode="lines+markers",
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=4),
        ))

    fig.update_layout(xaxis_title="Day of Month", yaxis_title="Cumulative Clicks")
    apply_bw(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)


# ── Export ─────────────────────────────────────────────────────────────────────

def _export(mtd: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        mtd.to_excel(writer, sheet_name="MTD_All", index=False)
        top_gainers(mtd, "clicks", 50).to_excel(writer, sheet_name="MTD_Gainers", index=False)
        top_decliners(mtd, "clicks", 50).to_excel(writer, sheet_name="MTD_Decliners", index=False)
    buffer.seek(0)
    st.download_button(
        label="Export MTD report (.xlsx)",
        data=buffer,
        file_name="mtd_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── Main render ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.title("Month-to-Date Analysis")
    st.caption("Compares current month (day 1 → yesterday) vs the same day-range in the previous month.")

    if df.empty:
        st.warning("No data loaded.")
        return

    df = add_brand_column(df)

    # ── In-page selectors ────────────────────────────────────────────────────
    with st.container(border=True):
        col_country, col_brand, col_metric = st.columns([2, 1, 1])
        with col_country:
            df = render_country_selector(df, key="mtd_country")
        with col_brand:
            df = render_brand_selector(df, key="mtd_brand")
        with col_metric:
            metric = st.selectbox(
                "Metric",
                options=["Clicks", "Impressions"],
                key="mtd_metric",
            )
            metric = metric.lower()

    _, top_n = render_filters(df, prefix="mtd")

    _period_banner(df)

    mtd_all = compute_mtd(df)
    mtd_all = flag_anomalies(mtd_all)
    _kpi_totals(mtd_all)

    st.divider()
    _brand_tabs(df, metric, top_n)

    st.divider()
    _cumulative_chart(df)

    st.divider()
    _export(mtd_all)
