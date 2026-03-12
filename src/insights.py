"""
src/insights.py
---------------
AI-powered search intelligence insights using the Claude API (Anthropic).

The module builds a structured data summary from WoW metrics and queries
Claude to generate actionable Spanish-language recommendations for the
adidas LatAm marketing / e-commerce / buying / activation teams.
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
        model      = "claude-3-5-haiku-20241022",
        max_tokens = 1200,
        system     = _SYSTEM_PROMPT,
        messages   = [
            {"role": "user", "content": _USER_TEMPLATE.format(context=context)},
        ],
    )
    return message.content[0].text
