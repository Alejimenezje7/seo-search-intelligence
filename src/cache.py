"""
src/cache.py
------------
Lightweight local data persistence layer using Parquet files.

Strategy:
  - Each extraction run saves a file named by the date range it covers.
  - On load, all Parquet files in data/raw/ are merged into a single DataFrame.
  - Duplicate rows (same date + keyword + device + domain) are dropped so that
    re-running an extraction for an overlapping range does not inflate counts.
  - The Streamlit app caches the loaded DataFrame in session state so the disk
    is only read once per session.
"""

import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# Columns that uniquely identify one row of data
_DEDUP_KEYS = ["date", "keyword", "domain"]


# ── Save ───────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, start_date: date, end_date: date) -> str:
    """
    Persist a DataFrame to a Parquet file.
    Returns the file path that was written.
    """
    if df.empty:
        logger.warning("Nothing to save — DataFrame is empty.")
        return ""

    filename = f"gsc_{start_date}_to_{end_date}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_parquet(filepath, index=False)
    logger.info("Saved %d rows → %s", len(df), filepath)
    return filepath


# ── Load ───────────────────────────────────────────────────────────────────────

def load_all() -> pd.DataFrame:
    """
    Read every Parquet file in DATA_DIR, concatenate them, deduplicate,
    and return a clean unified DataFrame.
    """
    files = sorted(Path(DATA_DIR).glob("*.parquet"))

    if not files:
        logger.warning("No cached data found in %s", DATA_DIR)
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:
            logger.error("Could not read %s: %s", f, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=_DEDUP_KEYS)
    after = len(combined)

    if before != after:
        logger.info("Deduplication removed %d duplicate rows.", before - after)

    # Ensure correct dtypes after merging files written at different times
    combined["date"] = pd.to_datetime(combined["date"])
    combined["clicks"] = combined["clicks"].astype(int)
    combined["impressions"] = combined["impressions"].astype(int)
    combined["ctr"] = combined["ctr"].astype(float)
    combined["position"] = combined["position"].astype(float)

    logger.info("Loaded %d rows from %d file(s).", after, len(files))
    return combined.sort_values("date").reset_index(drop=True)


def load_date_range(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Load all cached data and filter to the requested date range.
    Useful for feeding into the comparison engine without pulling
    more data from the API.
    """
    df = load_all()
    if df.empty:
        return df
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    return df.loc[mask].copy()


# ── Metadata helpers ───────────────────────────────────────────────────────────

def available_date_range() -> tuple[date | None, date | None]:
    """Return the earliest and latest dates found in the local cache."""
    df = load_all()
    if df.empty:
        return None, None
    return df["date"].min().date(), df["date"].max().date()


def cached_files() -> list[str]:
    """Return filenames currently in DATA_DIR."""
    return [f.name for f in sorted(Path(DATA_DIR).glob("*.parquet"))]
