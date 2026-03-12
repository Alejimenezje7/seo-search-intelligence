"""
src/insights.py
---------------
AI-powered search intelligence insights using the Claude API (Anthropic).

The module builds a structured data summary from WoW/MTD metrics and
queries Claude to generate:
  1. Strategic recommendations (AI Analyst on Overview).
  2. Copy-pasteable executive email summaries for every view.
"""

from __future__ import annotations

import pandas as pd


# ── Data context builder ───────────────────────────────────────────────────────

def _fmt_row(row: pd.Series, metric: str) -> str:
    """Format a single WoW row as a readable bullet for the prompt."""
    kw   = row.get("keyword", "—")
    prev = int(row.get(f"{metric}_prev", 0) or 0)
    curr = int(row.get(f"{metric}_curr", 0) or 0)
    pct  = row.get(f"{metric}_pct", 0) or 0
    sign = "▲" if pct > 0 else "▼"
    return f"  • {kw}  |  prev {prev:,} → curr {curr:,}  ({sign} {abs(pct):.1f}%)"


def build_context_summary(
    wow_flagged: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
    top_n: int = 8,
) -> str:
    """
    Build a structured text block summarising key WoW metrics.
    Passed verbatim as user content to the Claude API.
    """
    from src.processor import top_gainers, top_decliners

    lines: list[str] = []

    # ── Overall performance ──────────────────────────────────────────────────
    if not wow_flagged.empty and "clicks_curr" in wow_flagged.columns:
        tc_curr = int(wow_flagged["clicks_curr"].sum())
        tc_prev = int(wow_flagged["clicks_prev"].sum())
        ti_curr = int(wow_flagged["impressions_curr"].sum())
        ti_prev = int(wow_flagged["impressions_prev"].sum())
        c_pct = round((tc_curr - tc_prev) / tc_prev * 100, 1) if tc_prev else 0
        i_pct = round((ti_curr - ti_prev) / ti_prev * 100, 1) if ti_prev else 0

        lines += [
            "=== PERFORMANCE GLOBAL (WoW) ===",
            f"• Clicks:      {tc_curr:,}  (prev {tc_prev:,})  →  {'▲' if c_pct >= 0 else '▼'} {abs(c_pct):.1f}%",
            f"• Impresiones: {ti_curr:,}  (prev {ti_prev:,})  →  {'▲' if i_pct >= 0 else '▼'} {abs(i_pct):.1f}%",
            f"• Keywords tracked: {len(wow_flagged):,}",
        ]

    # ── Top gainers / decliners ───────────────────────────────────────────────
    for metric, label in [("clicks", "CLICKS"), ("impressions", "IMPRESIONES")]:
        gainers   = top_gainers(wow_flagged,   metric, n=top_n)
        decliners = top_decliners(wow_flagged, metric, n=top_n)

        if not gainers.empty:
            lines.append(f"\n=== TOP {top_n} EN CRECIMIENTO ({label}) ===")
            for _, row in gainers.head(top_n).iterrows():
                lines.append(_fmt_row(row, metric))

        if not decliners.empty:
            lines.append(f"\n=== TOP {top_n} EN CAÍDA ({label}) ===")
            for _, row in decliners.head(top_n).iterrows():
                lines.append(_fmt_row(row, metric))

    # ── Anomalies ────────────────────────────────────────────────────────────
    if "is_anomaly" in wow_flagged.columns:
        anomalies = wow_flagged[wow_flagged["is_anomaly"] == True]
        if not anomalies.empty:
            lines.append(f"\n=== ANOMALÍAS DETECTADAS ({len(anomalies)}) ===")
            for _, row in anomalies.head(6).iterrows():
                atype  = row.get("anomaly_type",   "unknown")
                reason = row.get("anomaly_reason", "—")
                lines.append(f"  • {row['keyword']}  ({atype}):  {reason}")

    # ── Market performance ───────────────────────────────────────────────────
    if df_raw is not None and not df_raw.empty and "domain" in df_raw.columns:
        try:
            from src.processor import compute_wow_by_domain
            from src.filters  import domain_label
            mkt = compute_wow_by_domain(df_raw)
            if not mkt.empty:
                lines.append("\n=== POR MERCADO (clicks WoW) ===")
                mkt = mkt.sort_values("clicks_pct", ascending=False, na_position="last")
                for _, row in mkt.iterrows():
                    pct   = float(row.get("clicks_pct",   0) or 0)
                    delta = float(row.get("clicks_delta", 0) or 0)
                    mkt_label = domain_label(row["domain"])
                    sign  = "▲" if pct >= 0 else "▼"
                    lines.append(f"  • {mkt_label}: {sign} {abs(pct):.1f}%  (Δ {delta:+,.0f})")
        except Exception:
            pass  # market data is supplementary — never crash the main flow

    return "\n".join(lines)


# ── Claude API call ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "Eres un analista SEO senior especializado en adidas para mercados de Latinoamérica. "
    "Tu función es interpretar datos de Google Search Console (WoW) y proporcionar "
    "recomendaciones estratégicas concisas y directamente accionables para los equipos de "
    "marketing, e-commerce, buying y activación digital.\n\n"
    "Responde SIEMPRE en español. Sé directo, específico y práctico. "
    "Menciona keywords o mercados concretos cuando los datos lo respalden. "
    "Nunca inventes datos; basa cada recomendación en lo que ves en el contexto."
)

_USER_TEMPLATE = """\
Aquí está el resumen de búsqueda orgánica de adidas LatAm para la última semana:

{context}

Por favor, responde con exactamente este formato en markdown:

## 📊 Resumen Ejecutivo
_(2-3 oraciones sobre el estado general del tráfico orgánico esta semana)_

## 🚀 Oportunidades Clave
_(3-4 bullets con oportunidades específicas basadas en keywords o mercados con crecimiento)_

## ⚠️ Alertas y Riesgos
_(2-3 bullets sobre keywords en caída, anomalías o tendencias que requieren atención urgente)_

## ✅ Acciones Recomendadas
_(4-5 acciones concretas y priorizadas que el equipo puede ejecutar esta semana)_\
"""


def get_ai_recommendations(context: str, api_key: str) -> str:
    """
    Call the Claude API with the structured data context.
    Returns the full markdown response as a string.

    Raises any Anthropic exception so the caller can display a user-friendly error.
    """
    import anthropic

    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model      = "claude-3-haiku-20240307",
        max_tokens = 1200,
        system     = _SYSTEM_PROMPT,
        messages   = [
            {"role": "user", "content": _USER_TEMPLATE.format(context=context)},
        ],
    )
    return message.content[0].text


# ── Per-view context builders ──────────────────────────────────────────────────

def build_mtd_context(
    mtd: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
    top_n: int = 6,
) -> str:
    """Build a structured context string for the Month-to-Date view."""
    from src.processor import top_gainers, top_decliners

    lines: list[str] = []

    if mtd.empty or "clicks_curr" not in mtd.columns:
        return "No MTD data available."

    tc_curr = int(mtd["clicks_curr"].sum())
    tc_prev = int(mtd["clicks_prev"].sum())
    ti_curr = int(mtd["impressions_curr"].sum())
    ti_prev = int(mtd["impressions_prev"].sum())
    c_pct = round((tc_curr - tc_prev) / tc_prev * 100, 1) if tc_prev else 0
    i_pct = round((ti_curr - ti_prev) / ti_prev * 100, 1) if ti_prev else 0
    active_kws = len(mtd[mtd["clicks_curr"] > 0])

    lines += [
        "=== MONTH-TO-DATE (MTD) ANALYSIS ===",
        f"• Clicks MTD:      {tc_curr:,}  (prev {tc_prev:,})  →  {'▲' if c_pct >= 0 else '▼'} {abs(c_pct):.1f}%",
        f"• Impressions MTD: {ti_curr:,}  (prev {ti_prev:,})  →  {'▲' if i_pct >= 0 else '▼'} {abs(i_pct):.1f}%",
        f"• Active keywords: {active_kws:,}",
    ]

    for metric, label in [("clicks", "CLICKS"), ("impressions", "IMPRESSIONES")]:
        gainers   = top_gainers(mtd,   metric, n=top_n)
        decliners = top_decliners(mtd, metric, n=top_n)

        if not gainers.empty:
            lines.append(f"\n=== TOP {top_n} EN CRECIMIENTO MTD ({label}) ===")
            for _, row in gainers.head(top_n).iterrows():
                lines.append(_fmt_row(row, metric))

        if not decliners.empty:
            lines.append(f"\n=== TOP {top_n} EN CAÍDA MTD ({label}) ===")
            for _, row in decliners.head(top_n).iterrows():
                lines.append(_fmt_row(row, metric))

    return "\n".join(lines)


def build_buying_context(
    cat_wow: pd.DataFrame,
    df: pd.DataFrame,
    top_n: int = 6,
) -> str:
    """Build a structured context string for the Buying & Trading view."""
    lines: list[str] = []
    lines.append("=== BUYING & TRADING — DEMAND INTELLIGENCE ===")

    total_impr = int(df["impressions"].sum()) if not df.empty else 0
    lines.append(f"• Impresiones totales (período): {total_impr:,}")

    if not cat_wow.empty and "impressions_pct" in cat_wow.columns:
        lines.append("\n=== DEMANDA POR CATEGORÍA (WoW Impressions) ===")
        sorted_cats = cat_wow.sort_values(
            "impressions_pct", ascending=False, na_position="last"
        )
        for _, row in sorted_cats.iterrows():
            cat   = row.get("product_category", "—")
            pct   = float(row.get("impressions_pct",   0) or 0)
            delta = float(row.get("impressions_delta", 0) or 0)
            sign  = "▲" if pct >= 0 else "▼"
            lines.append(f"  • {cat}: {sign} {abs(pct):.1f}%  (Δ {delta:+,.0f})")

    try:
        from src.processor import compute_wow, top_gainers, top_decliners
        nb = df[df["brand_type"] == "Non-Brand"] if "brand_type" in df.columns else df
        if not nb.empty:
            wow_nb    = compute_wow(nb, min_clicks=5, min_impressions=20)
            gainers   = top_gainers(wow_nb,   "impressions", n=top_n)
            decliners = top_decliners(wow_nb, "impressions", n=top_n)
            if not gainers.empty:
                lines.append(f"\n=== ONES TO WATCH (keywords non-brand en alza) ===")
                for _, row in gainers.head(top_n).iterrows():
                    lines.append(_fmt_row(row, "impressions"))
            if not decliners.empty:
                lines.append(f"\n=== COOLING DEMAND (keywords perdiendo tracción) ===")
                for _, row in decliners.head(top_n).iterrows():
                    lines.append(_fmt_row(row, "impressions"))
    except Exception:
        pass

    return "\n".join(lines)


def build_activation_context(
    campaign_df: pd.DataFrame,
    top_n: int = 6,
) -> str:
    """Build a structured context string for the Digital Activation view."""
    lines: list[str] = []
    lines.append("=== DIGITAL ACTIVATION — CAMPAIGN SIGNALS ===")

    if campaign_df.empty:
        lines.append("No hay datos de keywords de campaña disponibles.")
        return "\n".join(lines)

    total_kws  = campaign_df["keyword"].nunique() if "keyword" in campaign_df.columns else 0
    total_impr = int(campaign_df["impressions"].sum())
    total_clks = int(campaign_df["clicks"].sum())
    avg_ctr    = (
        round(campaign_df["clicks"].sum() / campaign_df["impressions"].sum() * 100, 2)
        if campaign_df["impressions"].sum() > 0 else 0.0
    )
    lines += [
        f"• Campaign keywords: {total_kws:,}",
        f"• Total Impressions: {total_impr:,}",
        f"• Total Clicks:      {total_clks:,}",
        f"• Avg CTR:           {avg_ctr:.2f}%",
    ]

    if "campaign_category" in campaign_df.columns:
        lines.append("\n=== POR TIPO DE CAMPAÑA ===")
        agg = (
            campaign_df
            .groupby("campaign_category", as_index=False)
            .agg(impressions=("impressions", "sum"), clicks=("clicks", "sum"))
            .sort_values("impressions", ascending=False)
        )
        for _, row in agg.head(8).iterrows():
            cat  = row["campaign_category"]
            imp  = int(row["impressions"])
            clks = int(row["clicks"])
            ctr  = round(clks / imp * 100, 2) if imp > 0 else 0
            lines.append(f"  • {cat}: {imp:,} impressions, {clks:,} clicks ({ctr:.2f}% CTR)")

    try:
        from src.processor import compute_wow, top_gainers, top_decliners
        wow = compute_wow(campaign_df, min_clicks=2, min_impressions=10)
        if not wow.empty:
            gainers   = top_gainers(wow,   "impressions", n=top_n)
            decliners = top_decliners(wow, "impressions", n=top_n)
            if not gainers.empty:
                lines.append(f"\n=== CAMPAIGN TERMS EN CRECIMIENTO ===")
                for _, row in gainers.head(top_n).iterrows():
                    lines.append(_fmt_row(row, "impressions"))
            if not decliners.empty:
                lines.append(f"\n=== CAMPAIGN TERMS EN CAÍDA ===")
                for _, row in decliners.head(top_n).iterrows():
                    lines.append(_fmt_row(row, "impressions"))
    except Exception:
        pass

    return "\n".join(lines)


def build_explorer_context(keyword: str, matched_df: pd.DataFrame) -> str:
    """Build a structured context string for the Keyword Explorer view."""
    lines: list[str] = []
    lines.append(f"=== KEYWORD EXPLORER — '{keyword}' ===")

    if matched_df.empty:
        lines.append("No se encontraron datos para este keyword.")
        return "\n".join(lines)

    total_clks  = int(matched_df["clicks"].sum())       if "clicks"      in matched_df.columns else 0
    total_impr  = int(matched_df["impressions"].sum())  if "impressions" in matched_df.columns else 0
    unique_kws  = matched_df["keyword"].nunique()       if "keyword"     in matched_df.columns else 0
    date_min    = str(matched_df["date"].min())         if "date"        in matched_df.columns else "—"
    date_max    = str(matched_df["date"].max())         if "date"        in matched_df.columns else "—"
    avg_pos     = (
        round(matched_df["position"].mean(), 1)
        if "position" in matched_df.columns and not matched_df.empty
        else "—"
    )

    lines += [
        f"• Keywords que coinciden: {unique_kws}",
        f"• Rango de fechas: {date_min} – {date_max}",
        f"• Total Clicks:       {total_clks:,}",
        f"• Total Impressions:  {total_impr:,}",
        f"• Posición promedio:  {avg_pos}",
    ]

    if "domain" in matched_df.columns:
        try:
            from src.filters import domain_label
            lines.append("\n=== POR MERCADO ===")
            mkt_agg = (
                matched_df
                .groupby("domain", as_index=False)
                .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"))
                .sort_values("clicks", ascending=False)
            )
            for _, row in mkt_agg.iterrows():
                mkt  = domain_label(row["domain"])
                clks = int(row["clicks"])
                impr = int(row["impressions"])
                lines.append(f"  • {mkt}: {clks:,} clicks, {impr:,} impressions")
        except Exception:
            pass

    return "\n".join(lines)


# ── Email summary generation ───────────────────────────────────────────────────

_EMAIL_SYSTEM_PROMPT = (
    "Eres un analista SEO senior de adidas para mercados de Latinoamérica. "
    "Redactas correos ejecutivos claros, concisos y profesionales en español, "
    "basados en datos reales de Google Search Console. "
    "Sé directo y específico. Nunca inventes datos; usa solo lo que aparece en el contexto. "
    "El correo debe ser útil para líderes de marketing, e-commerce o management de adidas LatAm."
)

_EMAIL_TEMPLATE = """\
Aquí están los datos del reporte de búsqueda orgánica adidas LatAm — sección: {view_name}

{context}

Por favor redacta un correo ejecutivo en español con exactamente este formato
(NO uses markdown con # ni **, usa texto plano):

Asunto: [asunto del correo — máximo 12 palabras]

Hola equipo,

[2-3 oraciones de resumen ejecutivo del período analizado — usa los números del contexto]

Highlights principales:
- [dato clave 1 con número]
- [dato clave 2 con número]
- [dato clave 3 con número]
- [dato clave 4 con número — opcional]

Acciones sugeridas:
- [acción concreta 1]
- [acción concreta 2]
- [acción concreta 3]

Saludos,
Adidas Search Intelligence Platform\
"""


def get_email_summary(context: str, view_name: str, api_key: str) -> str:
    """
    Call Claude to generate a copy-pasteable Spanish executive email.
    Returns the email as a plain-text string.
    Raises on API errors (caller handles display).
    """
    import anthropic

    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model      = "claude-3-haiku-20240307",
        max_tokens = 700,
        system     = _EMAIL_SYSTEM_PROMPT,
        messages   = [
            {
                "role": "user",
                "content": _EMAIL_TEMPLATE.format(
                    view_name=view_name,
                    context=context,
                ),
            },
        ],
    )
    return message.content[0].text


def render_email_button(
    view_name: str,
    context: str,
    key_suffix: str = "",
) -> None:
    """
    Render the '📧 Preparar Correo' button and copyable text area.

    Call at the bottom of each view's render() function.

    Args:
        view_name:  Human-readable view label included in the Claude prompt.
        context:    Pre-built text summarising this view's current data.
        key_suffix: Unique suffix to prevent Streamlit widget key collisions.
    """
    import streamlit as st
    from config import ANTHROPIC_API_KEY

    st.subheader("📧 Preparar Correo Ejecutivo")
    st.caption(
        "Genera un resumen de esta sección listo para copiar y enviar por correo. "
        "Powered by Claude (Anthropic)."
    )

    if not ANTHROPIC_API_KEY:
        st.info(
            "Para activar esta función añade tu API key de Anthropic a los "
            "secrets de Streamlit Cloud: `[ai] anthropic_api_key = \"sk-ant-...\"`"
        )
        return

    cache_key = f"email_txt_{key_suffix}"
    hash_key  = f"email_hsh_{key_suffix}"
    # lightweight hash — changes whenever context changes
    data_hash = f"{len(context)}-{context[:120]}"

    if st.session_state.get(hash_key) != data_hash:
        st.session_state.pop(cache_key, None)
        st.session_state[hash_key] = data_hash

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        generate = st.button(
            "📧 Generar Correo",
            key=f"email_btn_{key_suffix}",
            type="secondary",
        )
    with col_note:
        if cache_key in st.session_state:
            st.caption("✅ Correo listo — clic para regenerar.")

    if generate:
        with st.spinner("✍️ Redactando correo ejecutivo..."):
            try:
                result = get_email_summary(context, view_name, ANTHROPIC_API_KEY)
                st.session_state[cache_key] = result
            except Exception as exc:
                st.error(f"Error al conectar con la API de Claude: {exc}")
                return

    if cache_key in st.session_state:
        st.text_area(
            "📋 Copia y pega este correo:",
            value=st.session_state[cache_key],
            height=340,
            key=f"email_ta_{key_suffix}",
        )
