"""
views/opportunities.py
----------------------
SEO Oportunidades — five actionable sections for the adidas LatAm team.

Tabs:
    1. 🎯 Quick Wins       — pos 4-10, high impressions, easy traffic gains
    2. 📉 CTR Gap          — top-10 keywords whose CTR is far below benchmark
    3. 🕳️ Content Gaps     — high impressions, near-zero clicks (titles/content)
    4. 🌎 Market Health    — composite score per country
    5. 🆕 New Keywords     — keywords emerging in the last 14 days
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.filters import (
    add_brand_column,
    domain_label,
    render_brand_selector,
    render_country_selector,
)
# ── CTR benchmarks ─────────────────────────────────────────────────────────────
# Industry-average organic CTR by SERP position (positions 1–10).
_CTR_BENCHMARKS = [28.5, 15.7, 11.0, 7.5, 5.5, 4.0, 3.5, 3.0, 2.5, 2.0]


def _expected_ctr(position: float) -> float:
    """Return the expected CTR (%) for a given average position."""
    idx = max(0, min(9, round(position) - 1))
    return _CTR_BENCHMARKS[idx]


# ── Shared aggregation helper ──────────────────────────────────────────────────

def _aggregate_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily rows into one row per (keyword, domain) using:
      - weighted average position  (weight = impressions)
      - sum of clicks / impressions
      - CTR = clicks / impressions
    """
    df = df.copy()
    df["impr_x_pos"] = df["impressions"] * df["position"]

    agg = (
        df.groupby(["keyword", "domain"], sort=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            impr_x_pos=("impr_x_pos", "sum"),
        )
        .reset_index()
    )
    agg["position"] = agg["impr_x_pos"] / agg["impressions"].clip(lower=1)
    agg["ctr_pct"] = agg["clicks"] / agg["impressions"].clip(lower=1) * 100
    return agg.drop(columns=["impr_x_pos"])


# ── Tab 1 — Quick Wins ─────────────────────────────────────────────────────────

def _quick_wins(df: pd.DataFrame) -> None:
    st.markdown("## 🎯 Quick Wins")
    st.caption(
        "Keywords ranked **position 4–10** with high impressions. "
        "A small ranking improvement → big traffic gain."
    )

    agg = _aggregate_keywords(df)
    agg = add_brand_column(agg)

    # Filter to positions 4–10
    qw = agg[(agg["position"] >= 3.5) & (agg["position"] <= 10.5)].copy()

    if qw.empty:
        st.info("No Quick Win candidates found in the current data.")
        return

    # Estimated extra clicks if keyword moved to position 3
    target_ctr = _CTR_BENCHMARKS[2]  # pos 3 = 11.0 %
    qw["est_gain_clicks"] = (
        qw["impressions"] * (target_ctr - qw["ctr_pct"]) / 100
    ).clip(lower=0).round(0).astype(int)

    qw["expected_ctr"] = qw["position"].apply(_expected_ctr)
    qw["Market"] = qw["domain"].apply(domain_label)

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            qw = render_country_selector(qw.rename(columns={"domain": "domain"}), key="qw_country")
        with c2:
            qw = render_brand_selector(qw, key="qw_brand")

    if qw.empty:
        st.info("No results for the selected filters.")
        return

    qw_display = (
        qw[["keyword", "Market", "brand_type", "position", "ctr_pct",
            "expected_ctr", "impressions", "clicks", "est_gain_clicks"]]
        .sort_values("est_gain_clicks", ascending=False)
        .head(50)
        .rename(columns={
            "keyword":        "Keyword",
            "brand_type":     "Type",
            "position":       "Avg Pos",
            "ctr_pct":        "CTR %",
            "expected_ctr":   "Expected CTR %",
            "impressions":    "Impressions",
            "clicks":         "Clicks",
            "est_gain_clicks": "Est. Extra Clicks (→ pos 3)",
        })
    )

    qw_display["Avg Pos"] = qw_display["Avg Pos"].round(1)
    qw_display["CTR %"] = qw_display["CTR %"].round(2)
    qw_display["Expected CTR %"] = qw_display["Expected CTR %"].round(1)

    st.dataframe(qw_display, use_container_width=True, hide_index=True)

    total_gain = int(qw["est_gain_clicks"].sum())
    st.success(
        f"**{len(qw_display)} oportunidades** encontradas · "
        f"Ganancia estimada si suben a pos 3: **{total_gain:,} clics adicionales**"
    )

    csv = qw_display.to_csv(index=False).encode()
    st.download_button("⬇ Descargar CSV", csv, "quick_wins.csv", "text/csv", key="qw_dl")


# ── Tab 2 — CTR Gap ────────────────────────────────────────────────────────────

def _ctr_gap(df: pd.DataFrame) -> None:
    st.markdown("## 📉 CTR Gap")
    st.caption(
        "Keywords in **top 10** whose actual CTR is below **60 % of the expected benchmark**. "
        "Usually signals a weak title or meta description."
    )

    agg = _aggregate_keywords(df)
    agg = add_brand_column(agg)

    # Only top-10 with enough impressions
    top10 = agg[(agg["position"] <= 10.5) & (agg["impressions"] >= 50)].copy()
    top10["expected_ctr"] = top10["position"].apply(_expected_ctr)
    top10["ctr_ratio"] = top10["ctr_pct"] / top10["expected_ctr"].clip(lower=0.1)
    gap = top10[top10["ctr_ratio"] < 0.60].copy()

    if gap.empty:
        st.info("No CTR Gap candidates found. ¡Bien hecho!")
        return

    gap["lost_clicks_est"] = (
        (gap["expected_ctr"] - gap["ctr_pct"]) * gap["impressions"] / 100
    ).clip(lower=0).round(0).astype(int)

    gap["Market"] = gap["domain"].apply(domain_label)

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            gap = render_country_selector(gap, key="ctr_country")
        with c2:
            gap = render_brand_selector(gap, key="ctr_brand")

    if gap.empty:
        st.info("No results for the selected filters.")
        return

    gap_display = (
        gap[["keyword", "Market", "brand_type", "position", "ctr_pct",
             "expected_ctr", "ctr_ratio", "impressions", "lost_clicks_est"]]
        .sort_values("lost_clicks_est", ascending=False)
        .head(50)
        .rename(columns={
            "keyword":          "Keyword",
            "brand_type":       "Type",
            "position":         "Avg Pos",
            "ctr_pct":          "CTR %",
            "expected_ctr":     "Expected CTR %",
            "ctr_ratio":        "CTR / Expected",
            "impressions":      "Impressions",
            "lost_clicks_est":  "Clics Perdidos Est.",
        })
    )

    gap_display["Avg Pos"] = gap_display["Avg Pos"].round(1)
    gap_display["CTR %"] = gap_display["CTR %"].round(2)
    gap_display["Expected CTR %"] = gap_display["Expected CTR %"].round(1)
    gap_display["CTR / Expected"] = (gap_display["CTR / Expected"] * 100).round(0).astype(int).astype(str) + " %"

    st.dataframe(gap_display, use_container_width=True, hide_index=True)

    total_lost = int(gap["lost_clicks_est"].sum())
    st.warning(
        f"**{len(gap_display)} keywords** con CTR por debajo del benchmark · "
        f"Clics perdidos estimados: **{total_lost:,}** · "
        "Acción: revisar título y meta description."
    )

    csv = gap_display.to_csv(index=False).encode()
    st.download_button("⬇ Descargar CSV", csv, "ctr_gap.csv", "text/csv", key="ctr_dl")


# ── Tab 3 — Content Gaps ───────────────────────────────────────────────────────

def _content_gaps(df: pd.DataFrame) -> None:
    st.markdown("## 🕳️ Content Gaps")
    st.caption(
        "Keywords with **≥ 100 impressions** but a CTR **< 0.5 %**. "
        "Google shows your page, but users don't click — content or page is misaligned."
    )

    agg = _aggregate_keywords(df)
    agg = add_brand_column(agg)

    gaps = agg[(agg["impressions"] >= 100) & (agg["ctr_pct"] < 0.5)].copy()

    if gaps.empty:
        st.info("No Content Gaps found. ¡Excelente!")
        return

    # Classify action
    def _action(row: pd.Series) -> str:
        if row["position"] <= 20:
            return "🔧 Optimizar título / meta"
        return "📝 Crear / mejorar contenido"

    gaps["Acción"] = gaps.apply(_action, axis=1)
    gaps["Market"] = gaps["domain"].apply(domain_label)

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            gaps = render_country_selector(gaps, key="cg_country")
        with c2:
            gaps = render_brand_selector(gaps, key="cg_brand")

    if gaps.empty:
        st.info("No results for the selected filters.")
        return

    gaps_display = (
        gaps[["keyword", "Market", "brand_type", "position", "impressions",
              "clicks", "ctr_pct", "Acción"]]
        .sort_values("impressions", ascending=False)
        .head(50)
        .rename(columns={
            "keyword":    "Keyword",
            "brand_type": "Type",
            "position":   "Avg Pos",
            "impressions": "Impressions",
            "clicks":     "Clicks",
            "ctr_pct":    "CTR %",
        })
    )

    gaps_display["Avg Pos"] = gaps_display["Avg Pos"].round(1)
    gaps_display["CTR %"] = gaps_display["CTR %"].round(3)

    st.dataframe(gaps_display, use_container_width=True, hide_index=True)

    optimize_count = int((gaps["Acción"] == "🔧 Optimizar título / meta").sum())
    create_count = int((gaps["Acción"] == "📝 Crear / mejorar contenido").sum())

    c1, c2 = st.columns(2)
    c1.metric("🔧 Optimizar título / meta", optimize_count)
    c2.metric("📝 Crear contenido", create_count)

    csv = gaps_display.to_csv(index=False).encode()
    st.download_button("⬇ Descargar CSV", csv, "content_gaps.csv", "text/csv", key="cg_dl")


# ── Tab 4 — Market Health Score ────────────────────────────────────────────────

def _market_health(df: pd.DataFrame) -> None:
    st.markdown("## 🌎 Market Health Score")
    st.caption(
        "Composite score per country: **40 % avg position** (inverted) + "
        "**40 % avg CTR** + **20 % top-10 coverage**. "
        "Score 0–100, higher = healthier."
    )

    agg = _aggregate_keywords(df)

    if agg.empty:
        st.info("No data available.")
        return

    # Per-domain metrics
    def _score_domain(grp: pd.DataFrame) -> pd.Series:
        n = len(grp)
        avg_pos = grp["position"].mean()
        avg_ctr = grp["ctr_pct"].mean()
        top10_pct = (grp["position"] <= 10).sum() / max(n, 1) * 100

        # Normalize position: pos 1 = 100, pos 100 = 0
        pos_score = max(0.0, (100 - avg_pos) / 99 * 100)
        # Normalize CTR: 0–28.5 % range → 0–100
        ctr_score = min(100.0, avg_ctr / 28.5 * 100)

        composite = 0.40 * pos_score + 0.40 * ctr_score + 0.20 * top10_pct

        return pd.Series({
            "Avg Position": round(avg_pos, 1),
            "Avg CTR %": round(avg_ctr, 2),
            "Top-10 Coverage %": round(top10_pct, 1),
            "Health Score": round(composite, 1),
            "Keywords": n,
        })

    health = (
        agg.groupby("domain")
        .apply(_score_domain)
        .reset_index()
    )
    health["Market"] = health["domain"].apply(domain_label)
    health = health.drop(columns=["domain"]).sort_values("Health Score", ascending=False)

    # Color-code the score
    def _score_color(score: float) -> str:
        if score >= 70:
            return "🟢"
        if score >= 50:
            return "🟡"
        return "🔴"

    health["Status"] = health["Health Score"].apply(_score_color)

    display_cols = ["Status", "Market", "Health Score", "Avg Position",
                    "Avg CTR %", "Top-10 Coverage %", "Keywords"]
    st.dataframe(health[display_cols], use_container_width=True, hide_index=True)

    # Bar chart
    import plotly.express as px

    fig = px.bar(
        health.sort_values("Health Score"),
        x="Health Score",
        y="Market",
        orientation="h",
        color="Health Score",
        color_continuous_scale=["#FF4444", "#FFAA00", "#22CC88"],
        range_color=[0, 100],
        text="Health Score",
        labels={"Health Score": "Score (0–100)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        height=350,
        showlegend=False,
        coloraxis_showscale=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=60, t=20, b=20),
        xaxis=dict(range=[0, 110]),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 5 — New Keywords ───────────────────────────────────────────────────────

def _new_keywords(df: pd.DataFrame) -> None:
    st.markdown("## 🆕 New Keywords")
    st.caption(
        "Keywords that **appeared in the last 14 days** but were **absent before** that window. "
        "Early-stage demand signals worth monitoring."
    )

    if "date" not in df.columns or df.empty:
        st.info("Date column not available.")
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        st.info("No valid date data available.")
        return

    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=14)

    recent = df[df["date"] > cutoff]
    older = df[df["date"] <= cutoff]

    recent_kws = set(recent["keyword"].unique())
    older_kws = set(older["keyword"].unique())
    new_kws = recent_kws - older_kws

    if not new_kws:
        st.info("No brand-new keywords detected in the last 14 days. (All recent keywords also appeared in older data.)")
        return

    # Aggregate new keywords from recent data only
    new_df = recent[recent["keyword"].isin(new_kws)].copy()
    new_df["impr_x_pos"] = new_df["impressions"] * new_df["position"]

    agg = (
        new_df.groupby(["keyword", "domain"], sort=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            impr_x_pos=("impr_x_pos", "sum"),
            first_seen=("date", "min"),
        )
        .reset_index()
    )
    agg["position"] = agg["impr_x_pos"] / agg["impressions"].clip(lower=1)
    agg["ctr_pct"] = agg["clicks"] / agg["impressions"].clip(lower=1) * 100
    agg = agg.drop(columns=["impr_x_pos"])
    agg = add_brand_column(agg)
    agg["Market"] = agg["domain"].apply(domain_label)

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            agg = render_country_selector(agg, key="nk_country")
        with c2:
            agg = render_brand_selector(agg, key="nk_brand")

    if agg.empty:
        st.info("No results for the selected filters.")
        return

    display = (
        agg[["keyword", "Market", "brand_type", "first_seen", "impressions",
             "clicks", "ctr_pct", "position"]]
        .sort_values("impressions", ascending=False)
        .head(100)
        .rename(columns={
            "keyword":    "Keyword",
            "brand_type": "Type",
            "first_seen": "Primera vez",
            "impressions": "Impressions",
            "clicks":     "Clicks",
            "ctr_pct":    "CTR %",
            "position":   "Avg Pos",
        })
    )

    display["Avg Pos"] = display["Avg Pos"].round(1)
    display["CTR %"] = display["CTR %"].round(2)
    display["Primera vez"] = display["Primera vez"].dt.strftime("%Y-%m-%d")

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.info(
        f"**{len(new_kws)} new keywords** detected in the last 14 days "
        f"(window: {cutoff.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')})."
    )

    csv = display.to_csv(index=False).encode()
    st.download_button("⬇ Descargar CSV", csv, "new_keywords.csv", "text/csv", key="nk_dl")


# ── Main render ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.title("🔍 Oportunidades SEO")
    st.caption("Análisis accionable para maximizar visibilidad y tráfico orgánico de adidas LatAm.")

    if df is None or df.empty:
        st.warning("No hay datos disponibles. Usa el botón **Refresh** en la barra lateral.")
        return

    # Ensure required columns exist
    required = {"keyword", "impressions", "clicks", "position", "domain"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Columnas faltantes en los datos: {missing}")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Quick Wins",
        "📉 CTR Gap",
        "🕳️ Content Gaps",
        "🌎 Market Health",
        "🆕 New Keywords",
    ])

    with tab1:
        _quick_wins(df)

    with tab2:
        _ctr_gap(df)

    with tab3:
        _content_gaps(df)

    with tab4:
        _market_health(df)

    with tab5:
        _new_keywords(df)
