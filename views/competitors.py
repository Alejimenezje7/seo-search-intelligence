"""
views/competitors.py
---------------------
Competitor Intelligence — adidas LatAm vs Nike, Puma, New Balance, Reebok, UA.

Data source: Ahrefs v3 REST API (requires AHREFS_API_KEY in secrets / env var).
GSC data is used alongside Ahrefs estimates for enriched keyword gap analysis.

Tabs:
    1. 🏆 Benchmark SEO     — Domain Rating, organic traffic, keywords, backlinks
    2. 🥊 Competencia        — Keyword overlap, gap and share vs organic competitors
    3. 🔍 Keywords Gap       — Competitor keywords adidas doesn't rank for (opportunities)
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    AHREFS_API_KEY,
    AHREFS_COMPETITORS,
    AHREFS_MARKETS,
)
from src import ahrefs as ah

# ── Brand colours ──────────────────────────────────────────────────────────────
_ADIDAS_COLOR = "#000000"
_COMP_COLORS  = {c["label"]: c["color"] for c in AHREFS_COMPETITORS}
_COMP_COLORS["adidas"] = _ADIDAS_COLOR


# ── Setup guard ────────────────────────────────────────────────────────────────

def _api_key_guard() -> str | None:
    """Return API key or render a setup card and return None."""
    key = AHREFS_API_KEY
    if key:
        return key

    st.warning(
        "**Ahrefs API Key no configurada.** "
        "Para activar esta sección agrega tu clave en Streamlit Cloud secrets:"
    )
    st.code(
        "[ahrefs]\napi_key = \"tu-api-key-de-ahrefs-aqui\"",
        language="toml",
    )
    st.caption(
        "También puedes usar la variable de entorno `AHREFS_API_KEY` "
        "cuando corres la app localmente."
    )
    return None


# ── Market selector ────────────────────────────────────────────────────────────

def _market_selector(key: str = "comp_market") -> tuple[str, str, str] | None:
    """
    Render market picker.  Returns (market_label, ahrefs_domain, iso_country)
    or None if no market is configured.
    """
    options = list(AHREFS_MARKETS.keys())
    if not options:
        st.info("No hay mercados configurados en AHREFS_MARKETS.")
        return None

    market = st.selectbox("Mercado", options, key=key)
    cfg    = AHREFS_MARKETS[market]
    return market, cfg["domain"], cfg["country"]


# ── Competitor selector ─────────────────────────────────────────────────────────

def _competitor_selector(key: str = "comp_sel") -> list[dict]:
    """Multi-select of competitors to include in analysis."""
    all_labels = [c["label"] for c in AHREFS_COMPETITORS]
    selected   = st.multiselect(
        "Competidores",
        options=all_labels,
        default=all_labels[:4],   # Nike, Puma, NB, Reebok by default
        key=key,
    )
    return [c for c in AHREFS_COMPETITORS if c["label"] in selected]


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Benchmark SEO
# ─────────────────────────────────────────────────────────────────────────────

def _tab_benchmark(api_key: str) -> None:
    st.markdown("## 🏆 Benchmark SEO")
    st.caption(
        "Comparación de autoridad y visibilidad orgánica entre adidas y sus principales "
        "competidores en el mercado seleccionado. Fuente: **Ahrefs** (datos estimados)."
    )

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            result = _market_selector("bench_market")
        with c2:
            competitors = _competitor_selector("bench_comp")

    if result is None or not competitors:
        return

    market_label, adidas_domain, country = result

    # Build domain list: adidas first, then competitors
    all_domains = [adidas_domain] + [c["domain"] for c in competitors]
    all_labels  = ["adidas"] + [c["label"] for c in competitors]
    all_colors  = [_ADIDAS_COLOR] + [c["color"] for c in competitors]

    with st.spinner(f"Cargando datos Ahrefs para {market_label}…"):
        df = ah.fetch_batch_metrics(
            domains=tuple(all_domains),
            labels=tuple(all_labels),
            country=country,
            api_key=api_key,
        )

    if df.empty:
        st.error(
            "No se obtuvieron datos. Verifica que tu API key tenga permisos para "
            "**Site Explorer** y que los dominios sean correctos."
        )
        return

    # ── KPI strip — adidas vs best competitor ─────────────────────────────────
    adidas_row = df[df["label"] == "adidas"].iloc[0] if not df[df["label"] == "adidas"].empty else None
    if adidas_row is not None:
        comp_df  = df[df["label"] != "adidas"]
        top_comp = comp_df.sort_values("org_traffic", ascending=False).iloc[0] if not comp_df.empty else None

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Domain Rating · adidas",
            f"{adidas_row['domain_rating']:.0f}" if adidas_row["domain_rating"] else "—",
            delta=f"{(adidas_row['domain_rating'] or 0) - (top_comp['domain_rating'] or 0):+.0f} vs {top_comp['label']}" if top_comp is not None else None,
        )
        k2.metric(
            "Tráfico Orgánico · adidas",
            f"{int(adidas_row['org_traffic'] or 0):,}",
            delta=f"{int((adidas_row['org_traffic'] or 0) - (top_comp['org_traffic'] or 0)):+,} vs {top_comp['label']}" if top_comp is not None else None,
        )
        k3.metric(
            "Keywords Orgánicas · adidas",
            f"{int(adidas_row['org_keywords'] or 0):,}",
        )
        k4.metric(
            "Dominios de Referencia · adidas",
            f"{int(adidas_row['refdomains'] or 0):,}",
        )

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # Domain Rating
    with col_left:
        st.markdown("#### Domain Rating")
        fig_dr = px.bar(
            df.sort_values("domain_rating"),
            x="domain_rating",
            y="label",
            orientation="h",
            color="label",
            color_discrete_map={row["label"]: all_colors[i] for i, row in df.iterrows()},
            text="domain_rating",
            labels={"domain_rating": "DR (0–100)", "label": ""},
        )
        fig_dr.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        _style_chart(fig_dr, xmax=105)
        st.plotly_chart(fig_dr, use_container_width=True)

    # Organic Traffic
    with col_right:
        st.markdown("#### Tráfico Orgánico Estimado / mes")
        fig_tr = px.bar(
            df.sort_values("org_traffic"),
            x="org_traffic",
            y="label",
            orientation="h",
            color="label",
            color_discrete_map={row["label"]: all_colors[i] for i, row in df.iterrows()},
            text="org_traffic",
            labels={"org_traffic": "Visitas org. (est.)", "label": ""},
        )
        fig_tr.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
        )
        _style_chart(fig_tr)
        st.plotly_chart(fig_tr, use_container_width=True)

    # Keywords breakdown
    st.markdown("#### Keywords Orgánicas por Grupo de Posición")
    df_kw = df[["label", "org_keywords_1_3", "org_keywords_4_10"]].copy().fillna(0)
    df_kw["org_keywords_11plus"] = (
        df["org_keywords"].fillna(0) - df["org_keywords_1_3"].fillna(0) - df["org_keywords_4_10"].fillna(0)
    ).clip(lower=0)

    df_kw_long = df_kw.melt(
        id_vars="label",
        value_vars=["org_keywords_1_3", "org_keywords_4_10", "org_keywords_11plus"],
        var_name="Grupo",
        value_name="Keywords",
    )
    group_labels = {
        "org_keywords_1_3":    "Top 3",
        "org_keywords_4_10":   "Pos 4–10",
        "org_keywords_11plus": "Pos 11+",
    }
    group_colors = {
        "Top 3":    "#000000",
        "Pos 4–10": "#555555",
        "Pos 11+":  "#BBBBBB",
    }
    df_kw_long["Grupo"] = df_kw_long["Grupo"].map(group_labels)

    fig_kw = px.bar(
        df_kw_long,
        x="Keywords",
        y="label",
        color="Grupo",
        orientation="h",
        barmode="stack",
        color_discrete_map=group_colors,
        labels={"label": "", "Keywords": "N° Keywords"},
    )
    _style_chart(fig_kw, height=300)
    st.plotly_chart(fig_kw, use_container_width=True)

    # Referring domains
    st.markdown("#### Dominios de Referencia (Backlinks)")
    fig_rd = px.bar(
        df.sort_values("refdomains"),
        x="refdomains",
        y="label",
        orientation="h",
        color="label",
        color_discrete_map={row["label"]: all_colors[i] for i, row in df.iterrows()},
        text="refdomains",
        labels={"refdomains": "Ref. Domains", "label": ""},
    )
    fig_rd.update_traces(texttemplate="%{text:,}", textposition="outside")
    _style_chart(fig_rd, height=300)
    st.plotly_chart(fig_rd, use_container_width=True)

    # Raw table
    with st.expander("Ver tabla completa"):
        display = df.rename(columns={
            "label":             "Marca",
            "domain":            "Dominio",
            "domain_rating":     "DR",
            "org_traffic":       "Tráfico Org.",
            "org_keywords":      "Keywords",
            "org_keywords_1_3":  "Top 3",
            "org_keywords_4_10": "Pos 4–10",
            "refdomains":        "Ref. Domains",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Descargar CSV", csv, "benchmark_seo.csv", "text/csv", key="bench_dl")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Competencia Orgánica
# ─────────────────────────────────────────────────────────────────────────────

def _tab_competencia(api_key: str) -> None:
    st.markdown("## 🥊 Competencia Orgánica")
    st.caption(
        "Keywords compartidas, brecha vs adidas y cuánto tráfico captura cada competidor "
        "en el mercado seleccionado. Fuente: **Ahrefs Site Explorer**."
    )

    with st.container(border=True):
        result = _market_selector("comp_market2")

    if result is None:
        return

    market_label, adidas_domain, country = result

    with st.spinner(f"Analizando competencia para {adidas_domain} en {market_label}…"):
        df = ah.fetch_organic_competitors(
            target=adidas_domain,
            country=country,
            api_key=api_key,
            limit=20,
        )

    if df.empty:
        st.info(
            f"No se encontraron competidores orgánicos para **{adidas_domain}** "
            f"en **{market_label}**. Verifica que la fecha de snapshot esté disponible."
        )
        return

    # ── KPI strip ─────────────────────────────────────────────────────────────
    total_common    = int(df["keywords_common"].sum())
    total_gap_them  = int(df["keywords_competitor"].sum())
    top_rival       = df.iloc[0]["competitor_domain"] if not df.empty else "—"
    top_rival_kws   = int(df.iloc[0]["keywords_common"]) if not df.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Competidores detectados", len(df))
    k2.metric("Mayor overlap con", top_rival, delta=f"{top_rival_kws:,} keywords en común")
    k3.metric("Total keywords compartidas", f"{total_common:,}")
    k4.metric("Keywords ventaja rival (gap)", f"{total_gap_them:,}", delta="adidas no rankea estos", delta_color="inverse")

    st.divider()

    # ── Bubble chart: overlap vs traffic ──────────────────────────────────────
    st.markdown("#### Mapa de Competencia — Overlap de Keywords vs Tráfico")
    st.caption(
        "Burbuja más grande = más keywords compartidas con adidas. "
        "Posición derecha-arriba = rival con más tráfico y más overlap."
    )

    fig_bubble = px.scatter(
        df.head(15),
        x="keywords_common",
        y="traffic",
        size="keywords_competitor",
        text="competitor_domain",
        color="domain_rating",
        color_continuous_scale=["#CCCCCC", "#000000"],
        labels={
            "keywords_common":     "Keywords en común con adidas",
            "traffic":             "Tráfico orgánico estimado",
            "keywords_competitor": "Keywords gap (ellos > adidas)",
            "domain_rating":       "DR",
        },
        size_max=55,
    )
    fig_bubble.update_traces(
        textposition="top center",
        textfont=dict(size=10),
    )
    fig_bubble.update_layout(
        height=480,
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        margin=dict(l=0, r=0, t=20, b=20),
        coloraxis_colorbar=dict(title="DR"),
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ── Stacked bar: keywords overlap breakdown ────────────────────────────────
    st.markdown("#### Distribución de Keywords por Competidor")
    st.caption(
        "**En común**: ambos rankean · "
        "**Solo ellos**: ellos rankean, adidas NO (= gap) · "
        "**Solo adidas**: adidas rankea, ellos no (= ventaja)"
    )

    df_bar = df.head(12)[["competitor_domain", "keywords_common", "keywords_competitor", "keywords_target"]].copy()
    df_long = df_bar.melt(
        id_vars="competitor_domain",
        value_vars=["keywords_common", "keywords_competitor", "keywords_target"],
        var_name="Tipo",
        value_name="Keywords",
    )
    type_labels = {
        "keywords_common":     "En común",
        "keywords_competitor": "Solo ellos (gap)",
        "keywords_target":     "Solo adidas (ventaja)",
    }
    type_colors = {
        "En común":            "#777777",
        "Solo ellos (gap)":    "#E11B22",
        "Solo adidas (ventaja)": "#000000",
    }
    df_long["Tipo"] = df_long["Tipo"].map(type_labels)

    fig_bar = px.bar(
        df_long,
        x="Keywords",
        y="competitor_domain",
        color="Tipo",
        orientation="h",
        barmode="group",
        color_discrete_map=type_colors,
        labels={"competitor_domain": "", "Keywords": "N° Keywords"},
    )
    _style_chart(fig_bar, height=420)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Detail table ──────────────────────────────────────────────────────────
    st.markdown("#### Tabla detallada")
    df_display = df[["competitor_domain", "domain_rating", "traffic",
                      "keywords_common", "keywords_competitor", "keywords_target", "share"]].copy()
    df_display.columns = ["Competidor", "DR", "Tráfico Est.",
                          "Keywords Comunes", "Gap (ellos)", "Ventaja (adidas)", "Share (%)"]
    df_display["Share (%)"] = df_display["Share (%)"].round(2)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Descargar CSV", csv, "competencia_organica.csv", "text/csv", key="comp_dl")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Keywords Gap
# ─────────────────────────────────────────────────────────────────────────────

def _tab_keywords_gap(gsc_df: pd.DataFrame, api_key: str) -> None:
    st.markdown("## 🔍 Keywords Gap")
    st.caption(
        "Keywords en las que un competidor rankea (top-20) pero adidas NO aparece en tu GSC. "
        "Son oportunidades directas de contenido o link building. "
        "Fuente: **Ahrefs** + **Google Search Console** (datos reales de adidas)."
    )

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            result = _market_selector("gap_market")
        with c2:
            comp_options = [c["label"] for c in AHREFS_COMPETITORS]
            chosen_label = st.selectbox("Analizar competidor", comp_options, key="gap_comp")

    if result is None:
        return

    market_label, adidas_domain, country = result
    chosen_comp = next((c for c in AHREFS_COMPETITORS if c["label"] == chosen_label), None)
    if chosen_comp is None:
        return

    min_vol = st.slider(
        "Volumen mínimo de búsqueda mensual",
        min_value=50, max_value=5000, value=200, step=50,
        key="gap_min_vol",
    )

    if st.button(f"🔍 Cargar keywords de {chosen_label}", type="primary", key="gap_load"):
        with st.spinner(f"Descargando top keywords de {chosen_comp['domain']} en {market_label}…"):
            comp_kws_df = ah.fetch_top_organic_keywords(
                target=chosen_comp["domain"],
                country=country,
                api_key=api_key,
                limit=100,
                min_volume=min_vol,
            )

        if comp_kws_df.empty:
            st.info(f"No se encontraron keywords para {chosen_comp['domain']} con esos criterios.")
            return

        # ── Build adidas keyword set from GSC data ─────────────────────────
        gsc_filtered = gsc_df.copy()
        if not gsc_filtered.empty and "domain" in gsc_filtered.columns:
            # Try to match the market's domain from GSC data
            from config import DOMAIN_LABELS
            market_domain_url = next(
                (url for url, label in DOMAIN_LABELS.items() if label == market_label),
                None,
            )
            if market_domain_url:
                gsc_filtered = gsc_filtered[gsc_filtered["domain"] == market_domain_url]

        adidas_kw_set = set()
        if not gsc_filtered.empty and "keyword" in gsc_filtered.columns:
            adidas_kw_set = set(gsc_filtered["keyword"].str.lower().unique())

        # ── Find gap keywords ──────────────────────────────────────────────
        comp_kws_df["keyword_lower"] = comp_kws_df["keyword"].str.lower()
        gap_df = comp_kws_df[~comp_kws_df["keyword_lower"].isin(adidas_kw_set)].copy()
        gap_df = gap_df.drop(columns=["keyword_lower"])

        # ── KPI strip ─────────────────────────────────────────────────────
        st.success(
            f"**{len(comp_kws_df)}** keywords analizadas de {chosen_label} · "
            f"**{len(gap_df)}** no están en tu GSC → oportunidades para adidas {market_label}"
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Keywords del competidor", f"{len(comp_kws_df):,}")
        k2.metric("Gap (adidas no rankea)", f"{len(gap_df):,}")
        if not comp_kws_df.empty and "volume" in comp_kws_df.columns:
            k3.metric(
                "Volumen total perdido",
                f"{int(gap_df['volume'].sum()):,}",
                help="Suma del volumen mensual de búsquedas donde adidas no aparece",
            )

        if gap_df.empty:
            st.info("¡Excelente! adidas ya rankea para todos los top keywords de este competidor en GSC.")
            return

        # ── Difficulty vs Volume scatter ───────────────────────────────────
        if "keyword_difficulty" in gap_df.columns and "volume" in gap_df.columns:
            st.markdown("#### Dificultad vs Volumen — prioriza aquí")
            st.caption(
                "Cuadrante **abajo-derecha** = alto volumen + baja dificultad = mayor prioridad."
            )
            fig_scatter = px.scatter(
                gap_df.dropna(subset=["keyword_difficulty", "volume"]).head(60),
                x="keyword_difficulty",
                y="volume",
                text="keyword",
                color="best_position",
                color_continuous_scale=["#22CC88", "#FFAA00", "#FF4444"],
                range_color=[1, 20],
                labels={
                    "keyword_difficulty": "Keyword Difficulty (0–100)",
                    "volume":             "Volumen mensual",
                    "best_position":      f"Pos. {chosen_label}",
                },
                hover_data=["keyword", "volume", "keyword_difficulty", "best_position"],
            )
            fig_scatter.update_traces(textposition="top center", textfont=dict(size=8))
            fig_scatter.update_layout(
                height=450,
                paper_bgcolor="white",
                plot_bgcolor="#FAFAFA",
                margin=dict(l=0, r=0, t=20, b=20),
            )
            # Quadrant lines
            kd_mid  = 40
            vol_mid = gap_df["volume"].median() if "volume" in gap_df.columns else 500
            fig_scatter.add_vline(x=kd_mid,  line_dash="dot", line_color="#CCCCCC")
            fig_scatter.add_hline(y=vol_mid, line_dash="dot", line_color="#CCCCCC")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Table ─────────────────────────────────────────────────────────
        st.markdown(f"#### Top keywords de {chosen_label} que adidas no rankea")
        cols_to_show = [c for c in ["keyword", "volume", "keyword_difficulty", "best_position", "sum_traffic", "best_position_url"] if c in gap_df.columns]
        display = gap_df[cols_to_show].head(80).rename(columns={
            "keyword":            "Keyword",
            "volume":             "Volumen/mes",
            "keyword_difficulty": "Dificultad (KD)",
            "best_position":      f"Pos. {chosen_label}",
            "sum_traffic":        "Tráfico Est. rival",
            "best_position_url":  "URL rankeando",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

        csv = gap_df.to_csv(index=False).encode()
        st.download_button(
            f"⬇ Descargar gap keywords vs {chosen_label}",
            csv,
            f"keyword_gap_{chosen_label.lower().replace(' ', '_')}.csv",
            "text/csv",
            key="gap_dl",
        )
    else:
        # Placeholder before user clicks the button
        st.info(
            f"Selecciona el mercado y competidor, luego haz clic en **Cargar keywords** "
            f"para ver las oportunidades. (Cada carga consume créditos Ahrefs — "
            f"los resultados se guardan en caché 24 h.)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_chart(fig: go.Figure, height: int = 380, xmax: int | None = None) -> None:
    """Apply consistent styling to a Plotly figure."""
    layout_kwargs: dict = dict(
        height=height,
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=60, t=20, b=20),
        font=dict(family="Inter, sans-serif", size=11),
    )
    if xmax is not None:
        layout_kwargs["xaxis"] = dict(range=[0, xmax])
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(showgrid=True, gridcolor="#F0F0F0", zeroline=False)
    fig.update_yaxes(showgrid=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.title("⚔️ Competitor Intelligence")
    st.caption(
        "Visibilidad orgánica de adidas LatAm vs Nike, Puma, New Balance y más — "
        "powered by **Ahrefs API** · datos actualizados cada 24 h."
    )

    api_key = _api_key_guard()
    if not api_key:
        return

    tab1, tab2, tab3 = st.tabs([
        "🏆 Benchmark SEO",
        "🥊 Competencia Orgánica",
        "🔍 Keywords Gap",
    ])

    with tab1:
        _tab_benchmark(api_key)

    with tab2:
        _tab_competencia(api_key)

    with tab3:
        _tab_keywords_gap(df, api_key)
