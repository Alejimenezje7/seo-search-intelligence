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
    Make a minimal batch-analysis call (1 domain, 1 field) to verify the
    API key is valid and the endpoint is reachable.

    Returns (success: bool, message: str).
    Does NOT use st.cache_data — always hits the API.
    """
    if not api_key:
        return False, "API key vacía."

    try:
        resp = requests.post(
            f"{_BASE}/batch-analysis",
            headers={**_headers(api_key), "Content-Type": "application/json"},
            json={
                "targets": [{"url": "adidas.mx", "mode": "subdomains", "protocol": "both"}],
                "select":  ["domain_rating"],
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            dr = data.get("results", [{}])[0].get("domain_rating")
            return True, f"✅ Conexión OK — adidas.mx DR = {dr}"
        else:
            return False, f"HTTP {resp.status_code}: {resp.text[:400]}"
    except requests.Timeout:
        return False, "Timeout — la API tardó más de 15 s."
    except Exception as e:
        return False, f"Error de red: {e}"


# ── batch-analysis ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_batch_metrics(
    domains: tuple[str, ...],
    labels: tuple[str, ...],
    country: str,
    api_key: str,
) -> pd.DataFrame:
    """
    POST /v3/batch-analysis

    Compare multiple domains side by side.
    Columns: label, domain, domain_rating, org_traffic, org_keywords,
             org_keywords_1_3, org_keywords_4_10, refdomains
    """
    if not api_key:
        _set_error("API key no configurada.")
        return pd.DataFrame()

    targets = [
        {"url": d, "mode": "subdomains", "protocol": "both"}
        for d in domains
    ]
    payload: dict = {
        "targets": targets,
        "select": [
            "domain_rating",
            "org_traffic",
            "org_keywords",
            "org_keywords_1_3",
            "org_keywords_4_10",
            "refdomains",
        ],
    }
    # country filter: only attach if provided (omitting it returns global data)
    if country:
        payload["country"] = country

    try:
        resp = requests.post(
            f"{_BASE}/batch-analysis",
            headers={**_headers(api_key), "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if not resp.ok:
            _set_error(f"HTTP {resp.status_code} — {resp.text[:500]}")
            return pd.DataFrame()

        results = resp.json().get("results", [])
        _clear_error()
    except requests.Timeout:
        _set_error("Timeout en batch-analysis (>30 s).")
        return pd.DataFrame()
    except Exception as e:
        _set_error(f"Error de red en batch-analysis: {e}")
        return pd.DataFrame()

    rows = []
    for i, row in enumerate(results):
        rows.append({
            "label":             labels[i] if i < len(labels) else domains[i],
            "domain":            domains[i],
            "domain_rating":     _safe_float(row.get("domain_rating")),
            "org_traffic":       _safe_int(row.get("org_traffic")),
            "org_keywords":      _safe_int(row.get("org_keywords")),
            "org_keywords_1_3":  _safe_int(row.get("org_keywords_1_3")),
            "org_keywords_4_10": _safe_int(row.get("org_keywords_4_10")),
            "refdomains":        _safe_int(row.get("refdomains")),
        })

    return pd.DataFrame(rows)


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
