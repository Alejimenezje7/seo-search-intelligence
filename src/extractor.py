"""
src/extractor.py
----------------
Google Search Console API extraction layer.
Adapted from the original script — now modular, configurable, and
ready to be called from the Streamlit app or a scheduler.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build

from config import (
    CREDENTIALS_FILE,
    DOMAINS,
    FAST_ROW_LIMIT,
    GSC_DIMENSIONS,
    GSC_ROW_LIMIT,
    OAUTH_SCOPE,
)

logger = logging.getLogger(__name__)


# ── Auth ───────────────────────────────────────────────────────────────────────

def build_service():
    """Build and return an authenticated GSC API service client."""
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE,
        scopes=[OAUTH_SCOPE],
    )
    return build("searchconsole", "v1", credentials=credentials)


def list_available_properties(service) -> list[str]:
    """Return all GSC properties accessible by this service account."""
    response = service.sites().list().execute()
    return [s["siteUrl"] for s in response.get("siteEntry", [])]


# ── Core extraction ────────────────────────────────────────────────────────────

def _fetch_pages(service, domain: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Pull all rows for a single domain over the given date range,
    handling pagination transparently.
    Returns a raw DataFrame with columns from the API response.
    """
    frames = []
    start_row = 0

    while True:
        request_body = {
            "startDate": str(start_date),
            "endDate": str(end_date),
            "dimensions": GSC_DIMENSIONS,
            "rowLimit": GSC_ROW_LIMIT,
            "startRow": start_row,
        }

        response = (
            service.searchanalytics()
            .query(siteUrl=domain, body=request_body)
            .execute()
        )

        rows = response.get("rows")
        if not rows:
            logger.info("No rows for %s starting at row %d", domain, start_row)
            break

        df = pd.DataFrame(rows)
        # Unpack the 'keys' list into named columns that match GSC_DIMENSIONS
        df[GSC_DIMENSIONS] = pd.DataFrame(df["keys"].tolist(), index=df.index)
        df["domain"] = domain
        frames.append(df)

        logger.info(
            "  %s | rows %d–%d fetched",
            domain,
            start_row,
            start_row + len(rows) - 1,
        )

        if len(rows) < GSC_ROW_LIMIT:
            break  # Last page
        start_row += GSC_ROW_LIMIT

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename, type-cast, and drop raw columns we no longer need."""
    if df.empty:
        return df

    df = df.rename(columns={"query": "keyword"})
    df = df[["date", "keyword", "clicks", "impressions", "ctr", "position", "domain"]]
    df["date"] = pd.to_datetime(df["date"])
    df["clicks"] = df["clicks"].astype(int)
    df["impressions"] = df["impressions"].astype(int)
    df["ctr"] = df["ctr"].astype(float)
    df["position"] = df["position"].astype(float)
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_domain(
    service,
    domain: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Extract and clean GSC data for a single domain."""
    logger.info("Extracting %s (%s → %s)", domain, start_date, end_date)
    raw = _fetch_pages(service, domain, start_date, end_date)
    return _clean(raw)


def extract_all_domains(
    service,
    start_date: date,
    end_date: date,
    domains: list[str] = DOMAINS,
) -> pd.DataFrame:
    """
    Extract GSC data for every domain in the portfolio and return a single
    combined DataFrame. Domains that return no data are skipped gracefully.
    """
    frames = []
    for domain in domains:
        try:
            df = extract_domain(service, domain, start_date, end_date)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            logger.error("Failed to extract %s: %s", domain, exc)

    if not frames:
        logger.warning("No data returned for any domain.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total rows extracted: %d", len(combined))
    return combined


# ── Fast parallel extractor ───────────────────────────────────────────────────

def extract_fast(
    start_date: date,
    end_date: date,
    domains: list[str] = DOMAINS,
    row_limit: int = FAST_ROW_LIMIT,
) -> pd.DataFrame:
    """
    Pull the top `row_limit` rows per domain in parallel using threads.

    Why this is much faster than extract_all_domains():
      - One API call per domain (no pagination loop)
      - All domains run concurrently via ThreadPoolExecutor
      - Each thread builds its own service client (thread-safe)

    Trade-off: only returns the top N rows by impressions per domain.
    For identifying top movers this is more than sufficient.
    """

    def _fetch_one(domain: str) -> pd.DataFrame:
        # Build a fresh client per thread — avoids shared-state issues
        svc = build_service()
        request_body = {
            "startDate": str(start_date),
            "endDate":   str(end_date),
            "dimensions": GSC_DIMENSIONS,
            "rowLimit":   min(row_limit, GSC_ROW_LIMIT),
            "startRow":   0,
        }
        try:
            response = svc.searchanalytics().query(siteUrl=domain, body=request_body).execute()
            rows = response.get("rows", [])
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df[GSC_DIMENSIONS] = pd.DataFrame(df["keys"].tolist(), index=df.index)
            df["domain"] = domain
            return _clean(df)
        except Exception as exc:
            logger.error("Fast extract failed for %s: %s", domain, exc)
            return pd.DataFrame()

    frames = []
    # Use one thread per domain — GSC API calls are I/O-bound, not CPU-bound
    with ThreadPoolExecutor(max_workers=min(len(domains), 8)) as pool:
        futures = {pool.submit(_fetch_one, d): d for d in domains}
        for future in as_completed(futures):
            result = future.result()
            if not result.empty:
                frames.append(result)
                logger.info("  ✓ %s — %d rows", futures[future], len(result))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Fast extract complete: %d total rows from %d domains", len(combined), len(frames))
    return combined


# ── Date helpers ───────────────────────────────────────────────────────────────

def last_n_days(n: int = 30) -> tuple[date, date]:
    """Return (start_date, end_date) for the last N complete days."""
    end = date.today() - timedelta(days=1)   # Yesterday (last complete day)
    start = end - timedelta(days=n - 1)
    return start, end


def last_two_full_weeks() -> tuple[date, date]:
    """
    Return a range that covers the last two full ISO weeks (Mon–Sun).
    This guarantees WoW comparisons are never made on partial weeks.
    """
    today = date.today()
    # Most recent completed Sunday
    days_since_sunday = today.weekday() + 1  # weekday(): Mon=0, Sun=6
    last_sunday = today - timedelta(days=days_since_sunday)
    # Two weeks back from that Sunday
    start = last_sunday - timedelta(days=13)  # 14 days total
    return start, last_sunday


def current_and_previous_month_range() -> tuple[date, date, date, date]:
    """
    Return date ranges for the current MTD period and the equivalent
    period in the previous month.
    Returns: (curr_start, curr_end, prev_start, prev_end)
    """
    today = date.today()
    curr_start = today.replace(day=1)
    curr_end = today - timedelta(days=1)  # Up to yesterday

    # Same day-of-month range one month prior
    if curr_start.month == 1:
        prev_start = curr_start.replace(year=curr_start.year - 1, month=12)
    else:
        prev_start = curr_start.replace(month=curr_start.month - 1)

    days_elapsed = (curr_end - curr_start).days
    prev_end = prev_start + timedelta(days=days_elapsed)

    return curr_start, curr_end, prev_start, prev_end
