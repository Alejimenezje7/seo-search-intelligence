"""
views/activation.py
-------------------
Digital Activation — Campaign Signal Intelligence view.

Monitors how commercial campaign keywords perform in organic search,
helping the activation team understand which messages drive search behavior
and which markets respond best to campaign activity.
"""

from datetime import date as _date
import calendar as _cal

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
from src.insights import build_activation_context, render_email_button
from src.processor import (
    compute_wow,
    top_gainers,
    top_decliners,
)
from src.utils import (
    apply_bw,
    style_pct_cols,
    C_BLACK, C_MID, C_XLIGHT,
    fmt_delta, fmt_int, fmt_pct,
)


# ── Commercial Event Radar ─────────────────────────────────────────────────────
# Each entry: event name → (peak_month, peak_day, [keyword_patterns])
# peak_day is the approximate day the event occurs / peaks each year.
# Keyword patterns are plain substrings matched case-insensitively.

_COMMERCIAL_EVENTS = {
    "Día de la Madre":  (5,  11, ["dia de la madre", "día de la madre", "dia madre",
                                   "regalo mama", "regalos mama", "regalo madre"]),
    "Cyber Day":        (5,  27, ["cyber day", "cyberday"]),
    "Hot Sale":         (5,  26, ["hot sale", "hotsale"]),
    "Día del Padre":    (6,  21, ["dia del padre", "día del padre", "dia padre",
                                   "regalo papa", "regalos papa", "regalo padre"]),
    "Cyber WoW":        (10,  5, ["cyber wow", "cyberwow"]),
    "Buen Fin":         (11, 15, ["buen fin", "buenfin"]),
    "11.11 Solteros":   (11, 11, ["11.11"]),
    "Black Friday":     (11, 28, ["black friday", "viernes negro", "blackfriday"]),
    "Cyber Monday":     (12,  1, ["cyber monday", "cybermonday"]),
    "Navidad":          (12, 24, ["navidad", "christmas", "regalo navidad", "regalos navidad"]),
    "Rebajas Enero":    (1,   5, ["rebajas enero", "liquidación enero", "liquidacion enero"]),
}


def _next_event_date(today: _date, month: int, day: int) -> _date:
    """Return the next calendar occurrence of an annual event."""
    last_day = _cal.monthrange(today.year, month)[1]
    clamped   = min(day, last_day)
    candidate = _date(today.year, month, clamped)
    if candidate < today:
        last_day_ny = _cal.monthrange(today.year + 1, month)[1]
        candidate   = _date(today.year + 1, month, min(day, last_day_ny))
    return candidate


def _alert_level(days_away: int, impressions: int, wow_pct) -> tuple[str, str, str]:
    """
    Return (emoji, label, css_color) for the event alert level.
    Priority: proximity + active signal.
    """
    has_signal = impressions > 0
    growing    = wow_pct is not None and wow_pct > 0

    if days_away <= 21 and has_signal:
        return "🔴", "URGENTE", "#cc2200"
    if days_away <= 21:
        return "🔴", "PRÓXIMO", "#cc2200"
    if days_away <= 45 and has_signal and growing:
        return "🟠", "PRECALENTANDO", "#cc6600"
    if days_away <= 60 and has_signal:
        return "🟡", "SEÑAL TEMPRANA", "#b38600"
    if days_away <= 90 or has_signal:
        return "🟢", "EN RADAR", "#1a7a1a"
    return "⚪", "SIN SEÑAL", "#aaaaaa"


def _event_radar(df: pd.DataFrame) -> None:
    st.subheader("📡 Radar de Eventos Comerciales")
    st.caption(
        "Detecta cuándo los consumidores empiezan a buscar keywords de eventos comerciales — "
        "anticípate con contenido SEO y activación digital antes que la competencia. "
        "Señales basadas en impresiones orgánicas WoW por evento."
    )

    if df.empty or "keyword" not in df.columns:
        st.info("Carga datos para activar el Radar de Eventos.")
        return

    today = _date.today()
    kw_lower = df["keyword"].str.lower()

    event_signals = []
    for ev_name, (ev_month, ev_day, ev_patterns) in _COMMERCIAL_EVENTS.items():
        next_date = _next_event_date(today, ev_month, ev_day)
        days_away = (next_date - today).days

        # Match keywords
        mask = pd.Series(False, index=df.index)
        for pat in ev_patterns:
            mask |= kw_lower.str.contains(pat, na=False, regex=False)
        event_df = df[mask]

        impressions_curr = 0
        impressions_prev = 0
        wow_pct          = None
        keywords_found   = 0
        top_kws          = pd.DataFrame()

        if not event_df.empty:
            keywords_found = event_df["keyword"].nunique()
            try:
                wow = compute_wow(event_df, min_clicks=0, min_impressions=0)
                if not wow.empty:
                    impressions_curr = int(wow["impressions_curr"].sum())
                    impressions_prev = int(wow["impressions_prev"].sum())
                    if impressions_prev > 0:
                        wow_pct = round(
                            (impressions_curr - impressions_prev) / impressions_prev * 100, 1
                        )
                    elif impressions_curr > 0:
                        wow_pct = 100.0   # new signal — no prior baseline
                    top_kws = (
                        wow.nlargest(8, "impressions_curr")
                        [["keyword", "impressions_curr", "impressions_prev", "impressions_pct"]]
                        .copy()
                    )
                else:
                    impressions_curr = int(event_df["impressions"].sum())
            except Exception:
                impressions_curr = int(event_df["impressions"].sum())

        emoji, label, color = _alert_level(days_away, impressions_curr, wow_pct)

        event_signals.append({
            "name":             ev_name,
            "next_date":        next_date,
            "days_away":        days_away,
            "impressions_curr": impressions_curr,
            "impressions_prev": impressions_prev,
            "wow_pct":          wow_pct,
            "keywords_found":   keywords_found,
            "top_kws":          top_kws,
            "emoji":            emoji,
            "label":            label,
            "color":            color,
        })

    # Sort: active signals first, then by proximity
    event_signals.sort(key=lambda e: (0 if e["impressions_curr"] > 0 else 1, e["days_away"]))

    # ── Summary strip — only events with signal or within 120 days ─────────────
    visible = [e for e in event_signals if e["impressions_curr"] > 0 or e["days_away"] <= 120]

    if not visible:
        st.info("No hay eventos en los próximos 120 días ni señales activas. Carga más datos.")
        return

    # Cards — 3 per row
    for row_start in range(0, len(visible), 3):
        cols = st.columns(3)
        for col, ev in zip(cols, visible[row_start : row_start + 3]):
            with col:
                with st.container(border=True):
                    # Alert badge
                    st.markdown(
                        f"<span style='background:{ev['color']};color:#fff;padding:2px 10px;"
                        f"border-radius:4px;font-size:0.72rem;font-weight:700;"
                        f"letter-spacing:0.06em;'>{ev['emoji']} {ev['label']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{ev['name']}**")
                    st.caption(
                        f"📅 {ev['next_date'].strftime('%d %b %Y')} · "
                        f"**{ev['days_away']} días**"
                    )
                    if ev["impressions_curr"] > 0:
                        c1, c2 = st.columns(2)
                        c1.metric("Impresiones", f"{ev['impressions_curr']:,}")
                        if ev["wow_pct"] is not None:
                            arrow = "▲" if ev["wow_pct"] >= 0 else "▼"
                            c2.metric("WoW", f"{arrow} {abs(ev['wow_pct']):.0f}%")
                        st.caption(f"🔑 {ev['keywords_found']} keyword(s) detectado(s)")
                    else:
                        st.caption("_Sin señal en datos actuales_")

    # ── Keyword detail per active event ────────────────────────────────────────
    active = [e for e in event_signals if not e["top_kws"].empty]
    if active:
        st.markdown("##### 🔍 Detalle de keywords por evento")
        for ev in active:
            kw_count = ev["keywords_found"]
            header   = f"{ev['emoji']} **{ev['name']}** — {kw_count} keyword(s)  ·  {ev['days_away']}d para el evento"
            with st.expander(header, expanded=(ev["days_away"] <= 45)):
                detail = ev["top_kws"].rename(columns={
                    "keyword":          "Keyword",
                    "impressions_curr": "Esta Semana",
                    "impressions_prev": "Semana Ant.",
                    "impressions_pct":  "WoW %",
                }).copy()
                # Format WoW %
                if "WoW %" in detail.columns:
                    detail["WoW %"] = detail["WoW %"].apply(
                        lambda v: (
                            f"▲ {v:.1f}%" if (v is not None and not pd.isna(v) and v >= 0)
                            else f"▼ {abs(v):.1f}%" if (v is not None and not pd.isna(v))
                            else "—"
                        )
                    )
                st.dataframe(detail, use_container_width=True, hide_index=True)
                if ev["wow_pct"] is not None and ev["wow_pct"] > 0 and ev["days_away"] <= 60:
                    st.info(
                        f"💡 **Señal activa {ev['days_away']}d antes del evento** — "
                        "considera activar contenido SEO, landing pages y paid media ahora."
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
            display["impressions_prev"]  = display["impressions_prev"].apply(fmt_int)
            display["impressions_curr"]  = display["impressions_curr"].apply(fmt_int)
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

    _event_radar(df)
    st.divider()
    _kpi_strip(campaign_df)
    st.divider()
    _category_performance(campaign_df)
    st.divider()
    _campaign_wow(campaign_df, top_n)
    st.divider()
    _category_tabs(campaign_df, top_n)
    st.divider()
    _custom_search(df)
    st.divider()
    render_email_button(
        "Digital Activation — Campaign Signals",
        build_activation_context(campaign_df),
        key_suffix="activation",
    )
