"""
views/buying.py
---------------
Buying & Trading — Demand Intelligence view.

Surfaces rising and falling product demand from organic search signals,
helping the buying team spot which categories and products are gaining
or losing consumer interest.

Supports two comparison modes selectable in the UI:
  - WoW  : last full ISO week vs the week before
  - MoM  : last complete calendar month vs the month before
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
from src.insights import build_buying_context, get_buying_insights, render_email_button
from src.processor import (
    classify_search_signal,
    compute_wow,
    compute_mom,
    compute_wow_by_category,
    compute_mom_by_category,
    get_last_two_full_weeks,
    get_last_two_full_months,
    top_gainers,
    top_decliners,
)
from src.utils import (
    apply_bw,
    build_display_table,
    style_pct_cols,
    C_BLACK, C_MID,
    fmt_delta, fmt_int, fmt_pct,
)


# ── KPI strip ──────────────────────────────────────────────────────────────────

def _kpi_strip(df: pd.DataFrame, cat_data: pd.DataFrame, period_label: str) -> None:
    total_impr = int(df["impressions"].sum()) if not df.empty else 0

    top_rising = top_falling = "—"
    ones_to_watch = 0

    if not cat_data.empty:
        rising  = cat_data[cat_data["impressions_pct"].notna() & (cat_data["impressions_pct"] > 0)]
        falling = cat_data[cat_data["impressions_pct"].notna() & (cat_data["impressions_pct"] < 0)]
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
    c1.metric("Total Search Demand",    f"{total_impr:,}",   help="Total impressions in loaded data")
    c2.metric(f"Top Rising ({period_label})",   top_rising,  help="Category with highest impression growth")
    c3.metric(f"Top Declining ({period_label})", top_falling, help="Category with highest impression drop")
    c4.metric("Ones to Watch (WoW)",    f"{ones_to_watch}",  help="Non-brand keywords with >20% impression growth WoW")


# ── Category demand chart ───────────────────────────────────────────────────────

def _category_chart(cat_data: pd.DataFrame, period_label: str) -> None:
    st.subheader(f"Demand by Category — {period_label}")
    st.caption(
        f"Search impression change per product category ({period_label}). "
        "Positive bars = growing consumer interest."
    )

    if cat_data.empty:
        st.info("Not enough data for category comparison.")
        return

    chart = cat_data[cat_data["product_category"] != "Other"].copy()
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
        xaxis_title=f"Impressions Delta ({period_label})",
        yaxis_title="",
        xaxis=dict(zeroline=True, zerolinecolor=C_MID, zerolinewidth=1),
    )
    apply_bw(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


# ── Ones to Watch ───────────────────────────────────────────────────────────────

def _ones_to_watch(
    df: pd.DataFrame,
    top_n: int,
    period_label: str,
    compute_fn,
) -> None:
    st.subheader(f"Ones to Watch — Rising Search Demand ({period_label})")
    st.caption(
        "Non-brand keywords with the strongest impression growth. "
        "High growth = increasing consumer interest — a leading signal for buying decisions."
    )

    nb = df[df["brand_type"] == "Non-Brand"]
    if nb.empty:
        st.info("No non-brand data available.")
        return

    try:
        wow = compute_fn(nb, min_clicks=5, min_impressions=20)
    except Exception as exc:
        st.error(f"Could not compute comparison: {exc}")
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

    # ── SEO vs Tendencia signal ───────────────────────────────────────────────
    gainers["Signal"] = gainers.apply(
        lambda r: classify_search_signal(
            r["keyword"],
            r.get("impressions_prev", 0),
            r.get("impressions_pct", 0),
        ),
        axis=1,
    )

    # Signal filter + legend
    col_filter, col_legend = st.columns([2, 3])
    with col_filter:
        signal_filter = st.radio(
            "Filtrar por señal:",
            ["Todos", "🔍 SEO", "📈 Tendencia"],
            horizontal=True,
            key="buy_signal_filter",
        )
    with col_legend:
        st.caption(
            "🔍 **SEO** — crecimiento atribuible a visibilidad orgánica / trabajo SEO · "
            "📈 **Tendencia** — pico de demanda impulsado por eventos, moda o temporalidad"
        )

    # Apply filter
    if signal_filter != "Todos":
        gainers = gainers[gainers["Signal"] == signal_filter]
    if gainers.empty:
        st.info(f"No hay keywords con señal **{signal_filter}** este período.")
        return

    # Badge strip
    n_seo  = int((gainers["Signal"] == "🔍 SEO").sum())
    n_tend = int((gainers["Signal"] == "📈 Tendencia").sum())
    st.markdown(
        f"<span style='background:#111;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em;'>🔍 SEO: {n_seo}</span>&nbsp;&nbsp;"
        f"<span style='background:#1a7a1a;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em;'>📈 Tendencia: {n_tend}</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    extra = ["product_category"] if "product_category" in gainers.columns else []
    disp_cols = ["keyword", "Signal"] + extra + ["impressions_prev", "impressions_curr", "impressions_delta", "impressions_pct"]
    disp_cols = [c for c in disp_cols if c in gainers.columns]
    display = gainers[disp_cols].copy()
    display["impressions_prev"]  = display["impressions_prev"].apply(fmt_int)
    display["impressions_curr"]  = display["impressions_curr"].apply(fmt_int)
    display["impressions_delta"] = display["impressions_delta"].apply(fmt_delta)
    display["impressions_pct"]   = display["impressions_pct"].apply(fmt_pct)
    display = display.rename(columns={
        "keyword":           "Keyword",
        "Signal":            "Señal",
        "product_category":  "Category",
        "impressions_prev":  "Prev Period",
        "impressions_curr":  "This Period",
        "impressions_delta": "Change",
        "impressions_pct":   "% Growth",
    })
    st.dataframe(style_pct_cols(display), use_container_width=True, hide_index=True)


# ── Cooling demand ──────────────────────────────────────────────────────────────

def _cooling_demand(
    df: pd.DataFrame,
    top_n: int,
    period_label: str,
    compute_fn,
) -> None:
    st.caption(
        f"Non-brand keywords losing search traction ({period_label}) — "
        "🔍 **SEO** drop = possible ranking loss, review page optimisation · "
        "📈 **Tendencia** drop = trend cycle ending, adjust inventory exposure."
    )

    nb = df[df["brand_type"] == "Non-Brand"]
    if nb.empty:
        return

    try:
        wow = compute_fn(nb, min_clicks=5, min_impressions=20)
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

    # ── SEO vs Tendencia signal ───────────────────────────────────────────────
    decliners["Signal"] = decliners.apply(
        lambda r: classify_search_signal(
            r["keyword"],
            r.get("impressions_prev", 0),
            r.get("impressions_pct", 0),
        ),
        axis=1,
    )

    extra = ["product_category"] if "product_category" in decliners.columns else []
    disp_cols = ["keyword", "Signal"] + extra + ["impressions_prev", "impressions_curr", "impressions_delta", "impressions_pct"]
    disp_cols = [c for c in disp_cols if c in decliners.columns]
    display = decliners[disp_cols].copy()
    display["impressions_prev"]  = display["impressions_prev"].apply(fmt_int)
    display["impressions_curr"]  = display["impressions_curr"].apply(fmt_int)
    display["impressions_delta"] = display["impressions_delta"].apply(fmt_delta)
    display["impressions_pct"]   = display["impressions_pct"].apply(fmt_pct)
    display = display.rename(columns={
        "keyword":           "Keyword",
        "Signal":            "Señal",
        "product_category":  "Category",
        "impressions_prev":  "Prev Period",
        "impressions_curr":  "This Period",
        "impressions_delta": "Change",
        "impressions_pct":   "% Change",
    })
    st.dataframe(style_pct_cols(display), use_container_width=True, hide_index=True)


# ── Category deep-dive tabs ────────────────────────────────────────────────────

def _category_detail(
    df: pd.DataFrame,
    top_n: int,
    period_label: str,
    compute_fn,
) -> None:
    st.subheader(f"Category Deep Dive — {period_label}")
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
                wow = compute_fn(subset, min_clicks=2, min_impressions=10)
            except Exception as exc:
                st.error(f"Could not compute comparison: {exc}")
                continue
            if wow.empty:
                st.info("Not enough history for comparison.")
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


# ── AI Buying Analyst panel ────────────────────────────────────────────────────

def _buying_ai_section(cat_wow: pd.DataFrame, df: pd.DataFrame, period_label: str) -> None:
    """Render the AI-powered buying recommendations panel using the Claude API."""
    from config import ANTHROPIC_API_KEY

    st.subheader("🛒 AI Buying Analyst — Señales de Demanda")
    st.caption(
        "El agente analiza las señales de demanda orgánica por categoría y genera "
        "recomendaciones de buying, inventario y trading para tu equipo comercial. "
        "Powered by Claude (Anthropic)."
    )

    if not ANTHROPIC_API_KEY:
        st.info(
            "**Para activar AI Buying Analyst**, añade tu API key de Anthropic a los "
            "secrets de Streamlit Cloud:\n\n"
            "```toml\n[ai]\nanthropic_api_key = \"sk-ant-...\"\n```\n\n"
            "Puedes generar una key en [console.anthropic.com](https://console.anthropic.com)."
        )
        return

    cache_key     = "buy_ai_insights_text"
    data_hash_key = "buy_ai_insights_hash"
    data_hash = (
        f"{period_label}-{len(df)}-{int(df['impressions'].sum())}"
        if not df.empty and "impressions" in df.columns
        else f"{period_label}-empty"
    )
    if st.session_state.get(data_hash_key) != data_hash:
        st.session_state.pop(cache_key, None)
        st.session_state[data_hash_key] = data_hash

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        generate = st.button(
            "⚡ Generar Insights de Buying",
            key="buy_gen_ai_insights",
            type="primary",
        )
    with col_note:
        if cache_key in st.session_state:
            st.caption("✅ Insight generado — clic nuevamente para refrescar.")

    if generate:
        with st.spinner("🛒 Analizando señales de demanda por categoría..."):
            try:
                context = build_buying_context(cat_wow, df, period=period_label)
                result  = get_buying_insights(context, ANTHROPIC_API_KEY)
                st.session_state[cache_key] = result
            except Exception as exc:
                st.error(f"Error al conectar con la API de Claude: {exc}")
                return

    if cache_key in st.session_state:
        with st.container(border=True):
            st.markdown(st.session_state[cache_key])


# ── Main render ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    # Read mode from session_state so the title is correct on first render too
    _saved_mode = st.session_state.get("buy_period_mode", "📅 WoW — semana vs semana")
    _is_mom_early = str(_saved_mode).startswith("📆")

    if _is_mom_early:
        st.title("Buying & Trading — Month-over-Month Demand Intelligence")
        st.caption(
            "Organic search demand trend — month-over-month comparison of the last two complete months."
        )
    else:
        st.title("Buying & Trading — Week-over-Week Demand Intelligence")
        st.caption(
            "Organic search as a leading indicator of consumer demand — week-over-week view. "
            "Rising impressions signal growing product interest before purchase."
        )

    if df.empty:
        st.warning("No data loaded. Click **⚡ Quick** or **🔄 Full** in the sidebar.")
        return

    df = add_brand_column(df)
    df = add_category_column(df)

    # ── Filters + period mode selector ───────────────────────────────────────
    with st.container(border=True):
        col_country, col_mode = st.columns([2, 1])
        with col_country:
            df = render_country_selector(df, key="buy_country")
        with col_mode:
            mode = st.radio(
                "Período de análisis:",
                ["📅 WoW — semana vs semana", "📆 MoM — mes vs mes"],
                horizontal=False,
                key="buy_period_mode",
                help=(
                    "WoW: última semana completa vs la semana anterior\n"
                    "MoM: último mes completo vs el mes anterior"
                ),
            )

    is_mom       = mode.startswith("📆")
    period_label = "MoM" if is_mom else "WoW"
    compute_fn   = compute_mom           if is_mom else compute_wow
    cat_fn       = compute_mom_by_category if is_mom else compute_wow_by_category

    # ── Period info banner ────────────────────────────────────────────────────
    try:
        if is_mom:
            _, _, cs, ce, ps, pe = get_last_two_full_months(df)
            st.info(
                f"📆 **Month-over-Month** — "
                f"Comparando **{cs.strftime('%B %Y')}** ({cs} → {ce}) "
                f"vs **{ps.strftime('%B %Y')}** ({ps} → {pe})"
            )
        else:
            _, _, cs, ce, ps, pe = get_last_two_full_weeks(df)
            st.info(
                f"📅 **Week-over-Week** — "
                f"Comparando semana **{cs} → {ce}** "
                f"vs semana **{ps} → {pe}**"
            )
    except Exception:
        pass

    _, top_n = render_filters(df, prefix="buy")

    try:
        nb_df   = df[df["brand_type"] == "Non-Brand"]
        cat_data = cat_fn(nb_df)
    except Exception:
        cat_data = pd.DataFrame()

    _kpi_strip(df, cat_data, period_label)
    st.divider()
    _category_chart(cat_data, period_label=f"{'Month-over-Month' if is_mom else 'Week-over-Week'}")
    st.divider()
    _ones_to_watch(df, top_n, period_label=period_label, compute_fn=compute_fn)
    st.divider()
    with st.expander(
        f"▼ Cooling Demand — keywords losing search traction ({period_label})",
        expanded=False,
    ):
        _cooling_demand(df, top_n, period_label=period_label, compute_fn=compute_fn)
    st.divider()
    _category_detail(df, top_n, period_label=period_label, compute_fn=compute_fn)
    st.divider()
    _buying_ai_section(cat_data, df, period_label=period_label)
    st.divider()
    render_email_button(
        f"Buying & Trading — Demand Intelligence ({period_label})",
        build_buying_context(cat_data, df, period=period_label),
        key_suffix="buying",
        insights_cache_key="buy_ai_insights_text",
    )
