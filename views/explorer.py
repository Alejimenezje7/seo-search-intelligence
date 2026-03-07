"""
views/explorer.py
-----------------
Free-form keyword explorer — B&W charts, domain breakdown, position trend.
"""

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.filters import add_brand_column, domain_label
from src.processor import daily_trend
from src.utils import apply_bw, BW_PALETTE, C_BLACK, C_MID, C_LIGHT, C_XLIGHT


# ── Trend chart ────────────────────────────────────────────────────────────────

def _keyword_trend(df: pd.DataFrame, keyword: str) -> None:
    daily = daily_trend(df, ["date"])
    if daily.empty:
        st.info("No daily data for this keyword.")
        return

    st.subheader(f"Trend: '{keyword}'")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["clicks"],
        name="Clicks", marker_color=C_BLACK,
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["impressions"],
        name="Impressions", mode="lines",
        line=dict(color=C_MID, width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        yaxis=dict(title="Clicks"),
        yaxis2=dict(title="Impressions", overlaying="y", side="right", showgrid=False),
        barmode="overlay",
    )
    apply_bw(fig, height=300)
    st.plotly_chart(fig, use_container_width=True)


def _position_trend(df: pd.DataFrame) -> None:
    daily = (
        df.groupby("date", as_index=False)
        .apply(lambda g: pd.Series({
            "position": (
                (g["position"] * g["impressions"]).sum() / g["impressions"].sum()
                if g["impressions"].sum() > 0 else g["position"].mean()
            )
        }))
        .reset_index(drop=True)
    )
    if daily.empty:
        return

    fig = go.Figure(go.Scatter(
        x=daily["date"], y=daily["position"],
        mode="lines+markers",
        line=dict(color=C_BLACK, width=2),
        marker=dict(size=4, color=C_BLACK),
        name="Avg Position",
    ))
    fig.update_yaxes(autorange="reversed", title="Avg Position")
    fig.update_layout(xaxis_title="")
    apply_bw(fig, height=240)
    st.plotly_chart(fig, use_container_width=True)


# ── Domain breakdown ───────────────────────────────────────────────────────────

def _domain_breakdown(df: pd.DataFrame) -> None:
    agg = (
        df.groupby("domain", as_index=False)
        .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"))
        .sort_values("clicks", ascending=False)
    )
    agg["market"] = agg["domain"].apply(domain_label)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Clicks", x=agg["market"], y=agg["clicks"],
        marker_color=C_BLACK,
    ))
    fig.add_trace(go.Bar(
        name="Impressions", x=agg["market"], y=agg["impressions"],
        marker_color=C_LIGHT,
    ))
    fig.update_layout(barmode="group", xaxis_title="Market", yaxis_title="Volume")
    apply_bw(fig, height=280)
    st.plotly_chart(fig, use_container_width=True)


# ── Raw data + export ──────────────────────────────────────────────────────────

def _raw_table_and_export(df: pd.DataFrame) -> None:
    with st.expander("View raw data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="Export keyword data (.csv)",
        data=buffer,
        file_name="keyword_export.csv",
        mime="text/csv",
    )


# ── Multi-keyword comparison ───────────────────────────────────────────────────

def _multi_keyword_trend(df: pd.DataFrame, keywords: list[str]) -> None:
    frames = []
    for kw in keywords:
        subset = df[df["keyword"].str.lower() == kw.lower()]
        if subset.empty:
            continue
        daily = subset.groupby("date", as_index=False)["clicks"].sum()
        daily["keyword"] = kw
        frames.append(daily)

    if not frames:
        st.info("None of the entered keywords have data.")
        return

    combined = pd.concat(frames)

    fig = go.Figure()
    for i, kw in enumerate(combined["keyword"].unique()):
        subset = combined[combined["keyword"] == kw]
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["clicks"],
            name=kw, mode="lines+markers",
            line=dict(color=BW_PALETTE[i % len(BW_PALETTE)], width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(xaxis_title="", yaxis_title="Clicks")
    apply_bw(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)


# ── Main render ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.title("Keyword Explorer")
    st.caption("Search any keyword or topic to explore its full history and market breakdown.")

    if df.empty:
        st.warning("No data loaded.")
        return

    df = add_brand_column(df)

    mode = st.radio("Mode", ["Single keyword", "Compare keywords"], horizontal=True)

    if mode == "Single keyword":
        search = st.text_input("Enter a keyword (or partial match)", value="")
        if not search:
            st.info("Type a keyword above to start exploring.")
            return

        matched = df[df["keyword"].str.contains(search, case=False, na=False)]
        if matched.empty:
            st.warning(f"No data found for '{search}'.")
            return

        brand_label = matched["brand_type"].iloc[0] if "brand_type" in matched.columns else ""
        st.caption(
            f"{len(matched['keyword'].unique())} matching keywords  ·  "
            f"{len(matched):,} rows  ·  {brand_label}"
        )

        _keyword_trend(matched, search)
        _domain_breakdown(matched)
        _position_trend(matched)
        _raw_table_and_export(matched)

    else:
        raw_input = st.text_area(
            "Enter keywords to compare (one per line)",
            placeholder="running shoes\nadidas ultraboost\nfutbol",
            height=120,
        )
        keywords = [k.strip() for k in raw_input.splitlines() if k.strip()]
        if not keywords:
            st.info("Enter at least one keyword above.")
            return
        _multi_keyword_trend(df, keywords)
