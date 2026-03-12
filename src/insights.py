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


# ── Model auto-detection ───────────────────────────────────────────────────────
# Tries candidates in order and caches the first that responds successfully.
# This handles API keys that only have access to specific model families.

_CLAUDE_MODEL_CANDIDATES = [
    # Claude 4 family (2025-2026)
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-0",
    "claude-sonnet-4-0",
    "claude-haiku-4-0",
    # Claude 3.7 family (early 2025)
    "claude-3-7-sonnet-20250219",
    # Claude 3.5 family (2024) — may be deprecated
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    # Claude 3 family (2024) — legacy fallback
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

_resolved_model: str | None = None   # module-level cache


def _get_claude_model(client) -> str:
    """
    Return the first Claude model that this API key can actually call.
    Result is cached in a module-level variable so the probe only fires once
    per Streamlit worker process.
    """
    global _resolved_model
    if _resolved_model:
        return _resolved_model

    import anthropic

    last_auth_error = None

    for candidate in _CLAUDE_MODEL_CANDIDATES:
        try:
            client.messages.create(
                model=candidate,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            _resolved_model = candidate
            return candidate
        except anthropic.AuthenticationError as exc:
            # API key is invalid — no point trying other models
            raise RuntimeError(
                f"API key inválido o expirado. Verifica que el key en Streamlit secrets "
                f"sea correcto y esté activo en console.anthropic.com\n\nDetalle: {exc}"
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            # Key is valid but billing/plan issue
            raise RuntimeError(
                f"Tu API key no tiene acceso a la API de Mensajes. "
                f"Verifica que tengas créditos API activos en console.anthropic.com/billing — "
                f"los créditos de Claude.ai (web) no sirven para la API.\n\nDetalle: {exc}"
            ) from exc
        except anthropic.NotFoundError:
            # Model name doesn't exist for this key — try next
            continue
        except Exception:
            # Other errors (rate limit, network, etc.) — try next candidate
            continue

    raise RuntimeError(
        "Ningún modelo de Claude responde con este API key. "
        "Modelos probados: " + ", ".join(_CLAUDE_MODEL_CANDIDATES) + ". "
        "Ve a console.anthropic.com/settings/limits para ver los modelos disponibles "
        "en tu plan."
    )


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

## 🌎 Performance por País
_(Para CADA mercado que aparezca en los datos, escribe una línea con: bandera o nombre del país, \
variación % de clicks WoW, y una observación concreta de 1 frase sobre qué impulsó ese resultado. \
Ordena de mayor a menor crecimiento.)_

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
    model   = _get_claude_model(client)
    message = client.messages.create(
        model      = model,
        max_tokens = 1200,
        system     = _SYSTEM_PROMPT,
        messages   = [
            {"role": "user", "content": _USER_TEMPLATE.format(context=context)},
        ],
    )
    return message.content[0].text


# ── Buying & Trading AI Agent ─────────────────────────────────────────────────

_BUYING_SYSTEM_PROMPT = (
    "Eres un experto en Digital Buying & Trading para adidas Latinoamérica. "
    "Tu especialidad es interpretar señales de demanda orgánica de Google Search Console "
    "para informar decisiones de compra, inventario y trading de producto. "
    "Conoces profundamente las categorías de adidas: Running, Football, Training, "
    "Lifestyle, Outdoor y Originals. "
    "Responde SIEMPRE en español. Sé directo, específico y comercialmente relevante. "
    "Cada recomendación debe conectar el dato de búsqueda con una acción de buying "
    "o trading concreta. Nunca inventes datos; basa todo en el contexto recibido."
)

_BUYING_USER_TEMPLATE = """\
Aquí están las señales de demanda orgánica de adidas LatAm — sección Buying & Trading:

{context}

Por favor, responde con exactamente este formato en markdown:

## 📦 Diagnóstico de Demanda
_(2-3 oraciones sobre el estado general de la demanda por categoría esta semana)_

## 📈 Categorías con Mayor Momentum
_(Top 3 categorías con mayor crecimiento en búsquedas — incluye % de cambio y qué tipo \
de producto buscan los usuarios. Implicación directa para buying.)_

## 📉 Categorías bajo Presión
_(Top 2-3 categorías perdiendo demanda — implicaciones concretas para inventario o activación)_

## 🔍 Señales de Demanda No-Brand
_(Los keywords no-brand más relevantes como indicadores de intención de compra — \
qué productos están buscando los usuarios antes de comprar)_

## 🛒 Recomendaciones de Buying & Trading
_(4-5 acciones concretas para el equipo: qué categorías priorizar en compra, \
qué ajustar en inventario, qué impulsar con paid media o activación)_\
"""


def get_buying_insights(context: str, api_key: str) -> str:
    """
    Call the Claude API with a Buying & Trading specialist system prompt.
    Returns the full markdown response as a string.
    Raises on API errors (caller handles display).
    """
    import anthropic

    client  = anthropic.Anthropic(api_key=api_key)
    model   = _get_claude_model(client)
    message = client.messages.create(
        model      = model,
        max_tokens = 1400,
        system     = _BUYING_SYSTEM_PROMPT,
        messages   = [
            {"role": "user", "content": _BUYING_USER_TEMPLATE.format(context=context)},
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
    period: str = "WoW",
) -> str:
    """Build a structured context string for the Buying & Trading view."""
    lines: list[str] = []
    period_full = "Month-over-Month (último mes completo vs anterior)" if period == "MoM" else "Week-over-Week (última semana vs anterior)"
    lines.append(f"=== BUYING & TRADING — DEMAND INTELLIGENCE | Período: {period_full} ===")

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
    "Eres un comunicador ejecutivo senior de adidas para mercados de Latinoamérica. "
    "Redactas correos ejecutivos claros, concisos y de alto impacto en español. "
    "Cuando se te proporciona un análisis de IA, lo usas como fuente principal y lo "
    "condensas en una comunicación ejecutiva potente — no lo repites, lo destila. "
    "Sé específico con números y acciones. Nunca inventes datos. "
    "El correo debe ser útil para líderes de marketing, e-commerce o management de adidas LatAm. "
    "Máximo 20 líneas en total — los ejecutivos valoran la brevedad."
)

# Template when AI insights are available — uses them as primary source
_EMAIL_TEMPLATE_WITH_INSIGHTS = """\
Tengo un análisis completo de IA generado para la sección "{view_name}" de adidas LatAm.

=== ANÁLISIS IA (fuente principal — destila esto en el correo) ===
{insights}

=== DATOS DE CONTEXTO (referencia secundaria) ===
{context}

Usando el ANÁLISIS IA como tu fuente principal, redacta un correo ejecutivo en español \
listo para copiar y enviar. Destila los insights más poderosos en formato ejecutivo — \
no copies el análisis, transfórmalo en una comunicación de alto impacto.

(NO uses markdown con # ni **, usa SOLO texto plano):

Asunto: [asunto impactante — máximo 12 palabras]

Hola equipo,

[2-3 oraciones que capturen la esencia del análisis — incluye los números más importantes]

Highlights clave:
- [insight poderoso 1 del análisis con número concreto]
- [insight poderoso 2 del análisis con número concreto]
- [insight poderoso 3 del análisis con número concreto]
- [insight poderoso 4 — solo si aporta valor adicional]

Acciones prioritarias esta semana:
- [acción más urgente del análisis]
- [acción de oportunidad del análisis]
- [acción preventiva del análisis]

Saludos,
Adidas Search Intelligence Platform\
"""

# Template when no AI insights exist — uses raw data only
_EMAIL_TEMPLATE_DATA_ONLY = """\
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


def get_email_summary(
    context: str,
    view_name: str,
    api_key: str,
    insights: str | None = None,
) -> str:
    """
    Call Claude to generate a copy-pasteable Spanish executive email.

    If `insights` is provided (AI analysis already generated for this view),
    it is used as the primary source and the email becomes a distillation of
    those insights rather than a re-analysis of raw data.

    Returns the email as a plain-text string.
    Raises on API errors (caller handles display).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    model  = _get_claude_model(client)

    if insights:
        prompt = _EMAIL_TEMPLATE_WITH_INSIGHTS.format(
            view_name=view_name,
            insights=insights,
            context=context,
        )
    else:
        prompt = _EMAIL_TEMPLATE_DATA_ONLY.format(
            view_name=view_name,
            context=context,
        )

    message = client.messages.create(
        model      = model,
        max_tokens = 800,
        system     = _EMAIL_SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def render_email_button(
    view_name: str,
    context: str,
    key_suffix: str = "",
    insights_cache_key: str | None = None,
) -> None:
    """
    Render the '📧 Preparar Correo' button and copyable text area.

    Args:
        view_name:          Human-readable view label included in the Claude prompt.
        context:            Pre-built text summarising this view's current data.
        key_suffix:         Unique suffix to prevent Streamlit widget key collisions.
        insights_cache_key: session_state key where AI insights are cached for this
                            view (e.g. "ov_ai_insights_text"). When available the
                            email is distilled from the insights rather than raw data.
    """
    import streamlit as st
    from config import ANTHROPIC_API_KEY

    st.subheader("📧 Preparar Correo Ejecutivo")

    if not ANTHROPIC_API_KEY:
        st.info(
            "Para activar esta función añade tu API key de Anthropic a los "
            "secrets de Streamlit Cloud: `[ai] anthropic_api_key = \"sk-ant-...\"`"
        )
        return

    # Determine if AI insights are available for this view
    insights_text = (
        st.session_state.get(insights_cache_key)
        if insights_cache_key
        else None
    )
    has_insights = bool(insights_text)

    if has_insights:
        st.caption(
            "✨ **Basado en AI Insights** — el correo destilará el análisis de IA ya generado. "
            "Powered by Claude (Anthropic)."
        )
    else:
        st.caption(
            "Genera un resumen de esta sección listo para copiar y enviar por correo. "
            "💡 Genera primero los AI Insights para obtener un correo más potente. "
            "Powered by Claude (Anthropic)."
        )

    cache_key = f"email_txt_{key_suffix}"
    hash_key  = f"email_hsh_{key_suffix}"
    # Hash includes insights availability so switching modes invalidates cache
    insights_marker = insights_text[:80] if insights_text else "none"
    data_hash = f"{len(context)}-{context[:80]}-{insights_marker}"

    if st.session_state.get(hash_key) != data_hash:
        st.session_state.pop(cache_key, None)
        st.session_state[hash_key] = data_hash

    col_btn, col_note = st.columns([1, 4])
    with col_btn:
        btn_label = "📧 Generar Correo" if not has_insights else "✨ Generar desde Insights"
        generate = st.button(
            btn_label,
            key=f"email_btn_{key_suffix}",
            type="secondary",
        )
    with col_note:
        if cache_key in st.session_state:
            src = "insights IA" if has_insights else "datos"
            st.caption(f"✅ Correo listo (desde {src}) — clic para regenerar.")

    if generate:
        spinner_msg = (
            "✨ Destilando insights en correo ejecutivo..."
            if has_insights
            else "✍️ Redactando correo ejecutivo..."
        )
        with st.spinner(spinner_msg):
            try:
                result = get_email_summary(
                    context, view_name, ANTHROPIC_API_KEY,
                    insights=insights_text,
                )
                st.session_state[cache_key] = result
            except Exception as exc:
                st.error(f"Error al conectar con la API de Claude: {exc}")
                return

    if cache_key in st.session_state:
        st.text_area(
            "📋 Copia y pega este correo:",
            value=st.session_state[cache_key],
            height=360,
            key=f"email_ta_{key_suffix}",
        )
