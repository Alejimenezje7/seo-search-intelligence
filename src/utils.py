"""
src/utils.py
------------
Shared formatting helpers and chart style constants.
All views import from here so visual consistency is maintained in one place.
"""

import pandas as pd
import plotly.graph_objects as go

# ── B&W chart palette ──────────────────────────────────────────────────────────
C_BLACK   = "#000000"
C_DARK    = "#333333"
C_MID     = "#777777"
C_LIGHT   = "#BBBBBB"
C_XLIGHT  = "#E8E8E8"

BW_PALETTE = [C_BLACK, C_DARK, C_MID, C_LIGHT, C_XLIGHT]

# Base Plotly layout applied to every chart
PLOTLY_BASE = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color=C_DARK, size=12),
    xaxis=dict(gridcolor=C_XLIGHT, linecolor=C_LIGHT, zeroline=False),
    yaxis=dict(gridcolor=C_XLIGHT, linecolor=C_LIGHT, zeroline=False),
    legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=0, r=0, t=30, b=0),
)


def apply_bw(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply the B&W base layout to any Plotly figure."""
    fig.update_layout(height=height, **PLOTLY_BASE)
    return fig


# ── Delta formatters ───────────────────────────────────────────────────────────

def fmt_delta(val, unit: str = "") -> str:
    """
    Format an absolute delta with a directional arrow.
    e.g.  1234  → "▲ 1,234"
         -567   → "▼ 567"
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "new"
    if val > 0:
        return f"▲ {abs(val):,.0f}{unit}"
    if val < 0:
        return f"▼ {abs(val):,.0f}{unit}"
    return f"— {unit}"


def fmt_pct(val) -> str:
    """
    Format a percentage change with a directional arrow.
    e.g.  12.5  → "▲ 12.5%"
         -3.2   → "▼ 3.2%"
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "new"
    if val > 0:
        return f"▲ {abs(val):.1f}%"
    if val < 0:
        return f"▼ {abs(val):.1f}%"
    return "— 0%"


def fmt_pos(val) -> str:
    """
    Format an average position delta.
    Lower position number = improvement, so arrows are reversed.
    e.g.  -1.2  → "▲ 1.2" (improved)
           2.5  → "▼ 2.5" (worse)
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if val < 0:
        return f"▲ {abs(val):.1f}"   # improved
    if val > 0:
        return f"▼ {val:.1f}"        # worse
    return "—"


# ── Table builder ──────────────────────────────────────────────────────────────

def build_display_table(
    df: pd.DataFrame,
    metric: str,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert a processor output DataFrame into a clean display table.
    Adds pre-formatted Trend and % Chg columns so Streamlit renders them
    as plain text (no broken background colors).
    """
    cols = ["keyword", f"{metric}_prev", f"{metric}_curr", f"{metric}_delta", f"{metric}_pct"]
    if extra_cols:
        cols += [c for c in extra_cols if c in df.columns]
    cols = [c for c in cols if c in df.columns]

    out = df[cols].copy()

    # Replace raw delta + pct columns with formatted strings
    out[f"{metric}_delta"] = out[f"{metric}_delta"].apply(fmt_delta)
    out[f"{metric}_pct"]   = out[f"{metric}_pct"].apply(fmt_pct)

    rename = {
        "keyword":           "Keyword",
        f"{metric}_prev":    "Prev",
        f"{metric}_curr":    "Current",
        f"{metric}_delta":   "Change",
        f"{metric}_pct":     "% Chg",
        "position_curr":     "Avg Pos",
        "position_delta":    "Pos Δ",
        "brand_type":        "Type",
    }
    return out.rename(columns=rename)
