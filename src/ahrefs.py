"""
src/ahrefs.py
--------------
Thin REST wrapper for the Ahrefs v3 API (https://api.ahrefs.com/v3/).

All public fetch_* functions are decorated with @st.cache_data(ttl=86_400)
so that a page reload re-uses already-fetched data and API credits are only
consumed once per 24 hours per unique set of parameters.

Authentication: Bearer token in the Authorization header.
Set via Streamlit secrets [ahrefs] api_key  or  AHREFS_API_KEY env var.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

_BASE = "https://api.ahrefs.com/v3"

# ── Module-level last-error store (one per Streamlit process) ──────────────────
# Written by every API call that fails; read by views to surface diagnostics.
_last_error: str = ""


def get_last_error() -> str:
    return _last_error


def _set_error(msg: str) -> None:
    global _last_error
    _last_error = msg
    logger.error("Ahrefs API: %s", msg)


def _clear_error() -> None:
    global _last_error
    _last_error = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def snapshot_date() -> str:
    """Return first day of the current month as a safe Ahrefs snapshot date."""
    return date.today().replace(day=1).isoformat()


def _safe_int(val) -> int | None:
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ── Connection test ────────────────────────────────────────────────────────────

def ping(api_key: str) -> tuple[bool, str]:
    """
    Make a minimal site-explorer/metrics call to verify the API key is valid
    and the endpoint is reachable.  Returns (success: bool, message: str).
    Does NOT use st.cache_data — always hits the API live.
    """
    if not api_key:
        return False, "API key vacía."

    try:
        resp = requests.get(
            f"{_BASE}/site-explorer/metrics",
            headers=_headers(api_key),
            params={
                "target":   "adidas.mx",
                "mode":     "subdomains",
                "protocol": "both",
                "date":     snapshot_date(),
                "select":   "domain_rating,org_traffic",
                "output":   "json",
            },
            timeout=15,
        )
        if resp.status_code == 200:
            m = resp.json().get("metrics", {}) or {}
            dr = m.get("domain_rating", "—")
            tr = m.get("org_traffic", "—")
            return True, f"✅ Conexión OK — adidas.mx DR={dr}, tráfico est.={tr:,}" if isinstance(tr, int) else f"✅ Conexión OK — adidas.mx DR={dr}"
        else:
            return False, f"HTTP {resp.status_code}: {resp.text[:400]}"
    except requests.Timeout:
        return False, "Timeout — la API tardó más de 15 s."
    except Exception as e:
        return False, f"Error de red: {e}"


# ── site-explorer / metrics (single domain) ────────────────────────────────────
# NOTE: /v3/batch-analysis returns 404 on non-Enterprise plans.
# We replace it with individual site-explorer/metrics calls, one per domain,
# each cached independently. Same data, fully compatible.

_SE_METRICS_SELECT = (
    "domain_rating,org_traffic,org_keywords,"
    "org_keywords_1_3,org_keywords_4_10,refdomains"
)


@st.cache_data(ttl=86_400, show_spinner=False)
def _fetch_single_metrics(
    domain: str,
    country: str,
    api_key: str,
) -> dict:
    """
    GET /v3/site-explorer/metrics for one domain.
    Cached per (domain, country) — safe to call in a loop.
    """
    params: dict = {
        "target":   domain,
        "mode":     "subdomains",
        "protocol": "both",
        "date":     snapshot_date(),
        "select":   _SE_METRICS_SELECT,
        "output":   "json",
    }
    if country:
        params["country"] = country

    try:
        resp = requests.get(
            f"{_BASE}/site-explorer/metrics",
            headers=_headers(api_key),
            params=params,
            timeout=20,
        )
        if not resp.ok:
            _set_error(f"HTTP {resp.status_code} for {domain} — {resp.text[:400]}")
            return {}
        _clear_error()
        return resp.json().get("metrics", {}) or {}
    except requests.Timeout:
        _set_error(f"Timeout fetching metrics for {domain}.")
        return {}
    except Exception as e:
        _set_error(f"Error fetching metrics for {domain}: {e}")
        return {}


def fetch_batch_metrics(
    domains: tuple[str, ...],
    labels: tuple[str, ...],
    country: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Compare multiple domains side by side via site-explorer/metrics.

    Replaces the /v3/batch-analysis endpoint (which requires Enterprise plan).
    Each domain is fetched with its own @st.cache_data call so results are
    cached individually — subsequent loads hit 0 API endpoints.

    Columns: label, domain, domain_rating, org_traffic, org_keywords,
             org_keywords_1_3, org_keywords_4_10, refdomains
    """
    if not api_key:
        _set_error("API key no configurada.")
        return pd.DataFrame()

    rows = []
    for i, domain in enumerate(domains):
        m = _fetch_single_metrics(domain, country, api_key)
        rows.append({
            "label":             labels[i] if i < len(labels) else domain,
            "domain":            domain,
            "domain_rating":     _safe_float(m.get("domain_rating")),
            "org_traffic":       _safe_int(m.get("org_traffic")),
            "org_keywords":      _safe_int(m.get("org_keywords")),
            "org_keywords_1_3":  _safe_int(m.get("org_keywords_1_3")),
            "org_keywords_4_10": _safe_int(m.get("org_keywords_4_10")),
            "refdomains":        _safe_int(m.get("refdomains")),
        })

    df = pd.DataFrame(rows)
    # Return empty if every metric is null (all API calls failed)
    metric_cols = ["domain_rating", "org_traffic", "org_keywords", "refdomains"]
    if df[metric_cols].isnull().all().all():
        return pd.DataFrame()
    return df


# ── site-explorer / organic-competitors ───────────────────────────────────────

@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_organic_competitors(
    target: str,
    country: str,
    api_key: str,
    limit: int = 15,
) -> pd.DataFrame:
    """
    GET /v3/site-explorer/organic-competitors

    Returns organic competitors for `target` domain, ranked by keyword overlap.
    Columns: competitor_domain, keywords_common, keywords_competitor,
             keywords_target, traffic, domain_rating, share
    """
    if not api_key:
        _set_error("API key no configurada.")
        return pd.DataFrame()

    try:
        resp = requests.get(
            f"{_BASE}/site-explorer/organic-competitors",
            headers=_headers(api_key),
            params={
                "target":   target,
                "mode":     "subdomains",
                "protocol": "both",
                "country":  country,
                "date":     snapshot_date(),
                "select":   "competitor_domain,keywords_common,keywords_competitor,keywords_target,traffic,domain_rating,share",
                "order_by": "keywords_common:desc",
                "limit":    limit,
                "output":   "json",
            },
            timeout=30,
        )
        if not resp.ok:
            _set_error(f"HTTP {resp.status_code} — {resp.text[:500]}")
            return pd.DataFrame()

        data = resp.json().get("competitors", [])
        _clear_error()
    except requests.Timeout:
        _set_error("Timeout en organic-competitors (>30 s).")
        return pd.DataFrame()
    except Exception as e:
        _set_error(f"Error en organic-competitors: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    for col in ["keywords_common", "keywords_competitor", "keywords_target", "traffic"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "domain_rating" in df.columns:
        df["domain_rating"] = pd.to_numeric(df["domain_rating"], errors="coerce")
    if "share" in df.columns:
        df["share"] = pd.to_numeric(df["share"], errors="coerce")

    return df


# ── site-explorer / organic-keywords ──────────────────────────────────────────

@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_top_organic_keywords(
    target: str,
    country: str,
    api_key: str,
    limit: int = 50,
    min_volume: int = 100,
) -> pd.DataFrame:
    """
    GET /v3/site-explorer/organic-keywords

    Returns top organic keywords for a domain — used for competitor gap analysis.
    Columns: keyword, volume, keyword_difficulty, best_position, sum_traffic
    """
    if not api_key:
        _set_error("API key no configurada.")
        return pd.DataFrame()

    where_filter = (
        f'{{"and":[{{"field":"volume","is":["gte",{min_volume}]}},'
        f'{{"field":"best_position","is":["lte",20]}}]}}'
    )

    try:
        resp = requests.get(
            f"{_BASE}/site-explorer/organic-keywords",
            headers=_headers(api_key),
            params={
                "target":   target,
                "mode":     "subdomains",
                "protocol": "both",
                "country":  country,
                "date":     snapshot_date(),
                "select":   "keyword,volume,keyword_difficulty,best_position,sum_traffic,best_position_url",
                "order_by": "sum_traffic:desc",
                "where":    where_filter,
                "limit":    limit,
                "output":   "json",
            },
            timeout=30,
        )
        if not resp.ok:
            _set_error(f"HTTP {resp.status_code} — {resp.text[:500]}")
            return pd.DataFrame()

        data = resp.json().get("keywords", [])
        _clear_error()
    except requests.Timeout:
        _set_error("Timeout en organic-keywords (>30 s).")
        return pd.DataFrame()
    except Exception as e:
        _set_error(f"Error en organic-keywords: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    for col in ["volume", "best_position", "sum_traffic", "keyword_difficulty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── site-explorer / metrics ────────────────────────────────────────────────────

@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_domain_metrics(
    target: str,
    country: str,
    api_key: str,
) -> dict | None:
    """
    GET /v3/site-explorer/metrics — single-domain snapshot.
    Fields: org_keywords, org_traffic, domain_rating, refdomains.
    """
    if not api_key:
        return None

    try:
        resp = requests.get(
            f"{_BASE}/site-explorer/metrics",
            headers=_headers(api_key),
            params={
                "target":   target,
                "mode":     "subdomains",
                "protocol": "both",
                "country":  country,
                "date":     snapshot_date(),
                "select":   "org_keywords,org_traffic,domain_rating,refdomains",
                "output":   "json",
            },
            timeout=15,
        )
        if not resp.ok:
            _set_error(f"HTTP {resp.status_code} — {resp.text[:500]}")
            return None
        _clear_error()
        return resp.json().get("metrics")
    except Exception as e:
        _set_error(f"Error en domain metrics: {e}")
        return None
