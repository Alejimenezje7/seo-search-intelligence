"""
src/anomaly.py
--------------
Anomaly detection for search metric changes.

Approach — Z-score on deltas:
  1. Compute absolute click/impression deltas for all keywords in the period.
  2. Calculate the Z-score of each delta against the full distribution.
  3. Flag as anomaly when |z| > threshold AND the absolute delta exceeds a
     minimum meaningful size (to avoid flagging statistical noise on tiny numbers).

This keeps the logic simple and interpretable without requiring ML infrastructure.
"""

import numpy as np
import pandas as pd

from config import (
    ANOMALY_MIN_CLICK_DELTA,
    ANOMALY_MIN_IMPRESSION_DELTA,
    ANOMALY_ZSCORE_THRESHOLD,
)


# ── Z-score flagging ───────────────────────────────────────────────────────────

def _zscore_column(series: pd.Series) -> pd.Series:
    """Return Z-scores for a numeric Series. Returns 0 if std == 0."""
    std = series.std()
    if std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def flag_anomalies(
    df_compared: pd.DataFrame,
    zscore_threshold: float = ANOMALY_ZSCORE_THRESHOLD,
    min_click_delta: int = ANOMALY_MIN_CLICK_DELTA,
    min_impression_delta: int = ANOMALY_MIN_IMPRESSION_DELTA,
) -> pd.DataFrame:
    """
    Add anomaly flag columns to a compared DataFrame produced by processor.py.

    New columns added:
      - clicks_z         : Z-score of the clicks_delta column
      - impressions_z    : Z-score of the impressions_delta column
      - is_anomaly       : True when either metric triggers the threshold
      - anomaly_type     : 'spike' | 'drop' | 'both' | None
      - anomaly_reason   : Human-readable explanation string

    Parameters
    ----------
    df_compared : DataFrame from compute_wow() or compute_mtd()
    zscore_threshold : How many standard deviations to consider anomalous
    min_click_delta : Minimum absolute click change to be eligible
    min_impression_delta : Minimum absolute impression change to be eligible
    """
    df = df_compared.copy()

    if df.empty:
        for col in ["clicks_z", "impressions_z", "is_anomaly", "anomaly_type", "anomaly_reason"]:
            df[col] = None
        return df

    df["clicks_z"] = _zscore_column(df["clicks_delta"])
    df["impressions_z"] = _zscore_column(df["impressions_delta"])

    # A metric is anomalous when its |z| exceeds the threshold AND its absolute
    # delta is large enough to be meaningful (not just statistical noise).
    click_anomaly = (df["clicks_z"].abs() >= zscore_threshold) & (
        df["clicks_delta"].abs() >= min_click_delta
    )
    impression_anomaly = (df["impressions_z"].abs() >= zscore_threshold) & (
        df["impressions_delta"].abs() >= min_impression_delta
    )

    df["is_anomaly"] = click_anomaly | impression_anomaly

    # Classify direction
    def _classify(row):
        c_spike = row["clicks_delta"] > 0 and click_anomaly[row.name]
        c_drop = row["clicks_delta"] < 0 and click_anomaly[row.name]
        i_spike = row["impressions_delta"] > 0 and impression_anomaly[row.name]
        i_drop = row["impressions_delta"] < 0 and impression_anomaly[row.name]

        if (c_spike or i_spike) and (c_drop or i_drop):
            return "mixed"
        if c_spike or i_spike:
            return "spike"
        if c_drop or i_drop:
            return "drop"
        return None

    df["anomaly_type"] = df.apply(_classify, axis=1)

    # Human-readable reason string shown in the UI
    def _reason(row):
        if not row["is_anomaly"]:
            return ""
        parts = []
        if click_anomaly[row.name]:
            direction = "up" if row["clicks_delta"] > 0 else "down"
            parts.append(
                f"Clicks {direction} {abs(row['clicks_delta']):,} "
                f"({row['clicks_z']:+.1f}σ)"
            )
        if impression_anomaly[row.name]:
            direction = "up" if row["impressions_delta"] > 0 else "down"
            parts.append(
                f"Impressions {direction} {abs(row['impressions_delta']):,} "
                f"({row['impressions_z']:+.1f}σ)"
            )
        return " | ".join(parts)

    df["anomaly_reason"] = df.apply(_reason, axis=1)

    return df


# ── Summary helpers ────────────────────────────────────────────────────────────

def anomaly_summary(df_flagged: pd.DataFrame) -> dict:
    """
    Return a quick summary dict for display in the dashboard KPI strip.
    """
    total = len(df_flagged)
    flagged = df_flagged["is_anomaly"].sum() if "is_anomaly" in df_flagged.columns else 0
    spikes = (df_flagged.get("anomaly_type", pd.Series()) == "spike").sum()
    drops = (df_flagged.get("anomaly_type", pd.Series()) == "drop").sum()
    return {
        "total_keywords": total,
        "anomalies": int(flagged),
        "spikes": int(spikes),
        "drops": int(drops),
    }
