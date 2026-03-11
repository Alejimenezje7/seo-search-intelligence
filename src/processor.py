"""
src/processor.py
----------------
Comparison engine for Week-over-Week (WoW) and Month-to-Date (MTD) analysis.

Core design principles:
  - Only compare FULL weeks (Mon–Sun) — never partial weeks.
  - Apply minimum volume thresholds before calculating percentage changes
    to avoid misleading spikes from near-zero baselines.
  - All outputs are flat DataFrames ready to be consumed by the view layer.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

from config import MIN_CLICKS_THRESHOLD, MIN_IMPRESSIONS_THRESHOLD


# ── Internal helpers ───────────────────────────────────────────────────────────

def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate clicks, impressions, and compute weighted average position
    for a given set of grouping columns.
    CTR is recomputed from the aggregated totals.

    Position is computed as a weighted average (weighted by impressions).
    We pre-compute the weighted position column to avoid lambda index issues
    across pandas versions.
    """
    work = df.copy()
    work["_weighted_pos"] = work["position"] * work["impressions"]

    agg = (
        work.groupby(group_cols, as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            _weighted_pos_sum=("_weighted_pos", "sum"),
        )
    )

    # Weighted average: sum(pos * impr) / sum(impr); fall back to 0 if impr == 0
    agg["position"] = agg.apply(
        lambda r: r["_weighted_pos_sum"] / r["impressions"] if r["impressions"] > 0 else 0.0,
        axis=1,
    )
    agg["ctr"] = agg.apply(
        lambda r: r["clicks"] / r["impressions"] if r["impressions"] > 0 else 0.0,
        axis=1,
    )
    agg = agg.drop(columns=["_weighted_pos_sum"])
    return agg


def _safe_pct(new: float, old: float) -> float | None:
    """
    Return percentage change from old → new.
    Returns None when old == 0 to signal 'undefined' rather than inf.
    """
    if old == 0:
        return None
    return round((new - old) / old * 100, 2)


def _delta_label(pct: float | None) -> str:
    """Human-readable direction label for a percentage change."""
    if pct is None:
        return "new"
    if pct > 0:
        return "up"
    if pct < 0:
        return "down"
    return "flat"


def _merge_periods(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Inner-join current and previous aggregated data on group_cols and
    compute absolute deltas + percentage changes for clicks, impressions,
    CTR, and position.
    """
    merged = pd.merge(
        current,
        previous,
        on=group_cols,
        suffixes=("_curr", "_prev"),
        how="outer",
    ).fillna(0)

    for metric in ["clicks", "impressions"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_curr"] - merged[f"{metric}_prev"]
        merged[f"{metric}_pct"] = merged.apply(
            lambda r: _safe_pct(r[f"{metric}_curr"], r[f"{metric}_prev"]), axis=1
        )

    merged["position_delta"] = merged["position_curr"] - merged["position_prev"]
    merged["ctr_delta"] = merged["ctr_curr"] - merged["ctr_prev"]

    return merged


def _apply_thresholds(
    df: pd.DataFrame,
    min_clicks: int = MIN_CLICKS_THRESHOLD,
    min_impressions: int = MIN_IMPRESSIONS_THRESHOLD,
) -> pd.DataFrame:
    """
    Keep only rows where EITHER the current OR previous period meets
    the minimum volume thresholds. This prevents percentage noise from
    tiny-baseline queries while still surfacing newly appearing keywords.
    """
    mask = (
        (df["clicks_curr"] >= min_clicks) | (df["clicks_prev"] >= min_clicks)
    ) | (
        (df["impressions_curr"] >= min_impressions) | (df["impressions_prev"] >= min_impressions)
    )
    return df[mask].copy()


# ── Week-over-Week ─────────────────────────────────────────────────────────────

def get_last_two_full_weeks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slice the unified DataFrame into two full ISO weeks (Mon–Sun).
    Returns (current_week_df, previous_week_df).
    Raises ValueError if the cache doesn't contain enough data.
    """
    today = date.today()
    # Find the most recent completed Monday
    days_since_monday = today.weekday()  # Mon=0
    last_monday = today - timedelta(days=days_since_monday)
    curr_end = last_monday - timedelta(days=1)        # Last Sunday
    curr_start = curr_end - timedelta(days=6)         # Last Monday
    prev_end = curr_start - timedelta(days=1)         # Sunday before that
    prev_start = prev_end - timedelta(days=6)

    curr = df[(df["date"].dt.date >= curr_start) & (df["date"].dt.date <= curr_end)].copy()
    prev = df[(df["date"].dt.date >= prev_start) & (df["date"].dt.date <= prev_end)].copy()

    return curr, prev, curr_start, curr_end, prev_start, prev_end


def compute_wow(
    df: pd.DataFrame,
    group_cols: list[str] = ["keyword"],
    min_clicks: int = MIN_CLICKS_THRESHOLD,
    min_impressions: int = MIN_IMPRESSIONS_THRESHOLD,
) -> pd.DataFrame:
    """
    Week-over-Week comparison.
    Returns a DataFrame with one row per keyword (+ any extra group_cols)
    containing current/previous metrics, deltas, and percentage changes.
    """
    curr_df, prev_df, *dates = get_last_two_full_weeks(df)

    curr_agg = _aggregate(curr_df, group_cols)
    prev_agg = _aggregate(prev_df, group_cols)

    merged = _merge_periods(curr_agg, prev_agg, group_cols)
    merged = _apply_thresholds(merged, min_clicks, min_impressions)
    return merged.sort_values("clicks_curr", ascending=False).reset_index(drop=True)


# ── Month-to-Date ──────────────────────────────────────────────────────────────

def get_mtd_ranges(df: pd.DataFrame) -> tuple:
    """
    Determine MTD date ranges.
    Current MTD: 1st of this month → yesterday.
    Previous MTD: same day-range one month prior.
    """
    today = date.today()
    curr_start = today.replace(day=1)
    curr_end = today - timedelta(days=1)

    # Step back one month
    if curr_start.month == 1:
        prev_start = curr_start.replace(year=curr_start.year - 1, month=12)
    else:
        prev_start = curr_start.replace(month=curr_start.month - 1)

    days_elapsed = (curr_end - curr_start).days
    prev_end = prev_start + timedelta(days=days_elapsed)

    curr = df[(df["date"].dt.date >= curr_start) & (df["date"].dt.date <= curr_end)].copy()
    prev = df[(df["date"].dt.date >= prev_start) & (df["date"].dt.date <= prev_end)].copy()

    return curr, prev, curr_start, curr_end, prev_start, prev_end


def compute_mtd(
    df: pd.DataFrame,
    group_cols: list[str] = ["keyword"],
    min_clicks: int = MIN_CLICKS_THRESHOLD,
    min_impressions: int = MIN_IMPRESSIONS_THRESHOLD,
) -> pd.DataFrame:
    """
    Month-to-Date comparison.
    Same structure as compute_wow but using MTD periods.
    """
    curr_df, prev_df, *dates = get_mtd_ranges(df)

    curr_agg = _aggregate(curr_df, group_cols)
    prev_agg = _aggregate(prev_df, group_cols)

    merged = _merge_periods(curr_agg, prev_agg, group_cols)
    merged = _apply_thresholds(merged, min_clicks, min_impressions)
    return merged.sort_values("clicks_curr", ascending=False).reset_index(drop=True)


# ── Convenience slices ─────────────────────────────────────────────────────────

def top_gainers(df_compared: pd.DataFrame, metric: str = "clicks", n: int = 20) -> pd.DataFrame:
    """Return the top N queries by absolute delta for the given metric."""
    col = f"{metric}_delta"
    return (
        df_compared[df_compared[col] > 0]
        .nlargest(n, col)
        .reset_index(drop=True)
    )


def top_decliners(df_compared: pd.DataFrame, metric: str = "clicks", n: int = 20) -> pd.DataFrame:
    """Return the top N queries by largest absolute drop for the given metric."""
    col = f"{metric}_delta"
    return (
        df_compared[df_compared[col] < 0]
        .nsmallest(n, col)
        .reset_index(drop=True)
    )


def daily_trend(df: pd.DataFrame, group_cols: list[str] = ["date"]) -> pd.DataFrame:
    """Aggregate to daily totals — useful for time-series charts."""
    return _aggregate(df, group_cols).sort_values("date")


# ── Country (domain) breakdown ─────────────────────────────────────────────────

def compute_wow_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Week-over-Week aggregated at domain level.
    Returns one row per domain with click/impression deltas.
    Useful for the country performance table.
    """
    curr_df, prev_df, curr_start, curr_end, prev_start, prev_end = get_last_two_full_weeks(df)

    curr_agg = curr_df.groupby("domain", as_index=False).agg(
        clicks_curr=("clicks", "sum"),
        impressions_curr=("impressions", "sum"),
    )
    prev_agg = prev_df.groupby("domain", as_index=False).agg(
        clicks_prev=("clicks", "sum"),
        impressions_prev=("impressions", "sum"),
    )

    merged = pd.merge(curr_agg, prev_agg, on="domain", how="outer").fillna(0)

    merged["clicks_delta"] = merged["clicks_curr"] - merged["clicks_prev"]
    merged["clicks_pct"] = merged.apply(
        lambda r: _safe_pct(r["clicks_curr"], r["clicks_prev"]), axis=1
    )
    merged["impressions_delta"] = merged["impressions_curr"] - merged["impressions_prev"]
    merged["impressions_pct"] = merged.apply(
        lambda r: _safe_pct(r["impressions_curr"], r["impressions_prev"]), axis=1
    )

    return merged.sort_values("clicks_curr", ascending=False).reset_index(drop=True)
