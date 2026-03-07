"""
config.py
---------
Central configuration for the Adidas Search Intelligence Platform.
All domain lists, thresholds, and environment-driven settings live here.
"""

import json
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()


# ── Credentials ────────────────────────────────────────────────────────────────
def _resolve_credentials() -> str:
    """
    Resolve the path to the GSC service-account credentials file.

    Priority:
      1. Local file at the path given by GSC_CREDENTIALS_FILE env var (default: credentials.json)
      2. Streamlit Cloud secrets under the key [gsc_credentials]

    On Streamlit Cloud the file doesn't exist on disk, so we write the secret
    content to a temporary file and return that path.
    """
    # 1 — Local file (dev + Railway + any server deployment)
    local_path = os.getenv("GSC_CREDENTIALS_FILE", "credentials.json")
    if os.path.exists(local_path):
        return local_path

    # 2 — Streamlit Cloud secrets
    try:
        import streamlit as st  # only available when running inside Streamlit

        if hasattr(st, "secrets") and "gsc_credentials" in st.secrets:
            creds = dict(st.secrets["gsc_credentials"])
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            )
            json.dump(creds, tmp)
            tmp.flush()
            return tmp.name
    except Exception:
        pass  # streamlit not available or secret missing

    raise FileNotFoundError(
        "GSC credentials not found. "
        "Place credentials.json in the project root, set the "
        "GSC_CREDENTIALS_FILE env var, or configure [gsc_credentials] "
        "in Streamlit Cloud secrets."
    )


CREDENTIALS_FILE = _resolve_credentials()
OAUTH_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"

# ── Domains ────────────────────────────────────────────────────────────────────
# Add or remove domains here as your portfolio grows.
DOMAINS = [
    "https://www.adidas.mx/",
    "https://www.adidas.co/",
    "https://www.adidas.pe/",
    "https://www.adidas.cl/",
    "https://www.adidas.com.ar/",
    "https://www.adidas.com.br/",
    "https://www.adidas.com/cr/es",
    "https://www.adidas.com/ec/es",
]

# Friendly labels shown in the UI (must match DOMAINS order)
DOMAIN_LABELS = {
    "https://www.adidas.mx/":        "Mexico",
    "https://www.adidas.co/":        "Colombia",
    "https://www.adidas.pe/":        "Peru",
    "https://www.adidas.cl/":        "Chile",
    "https://www.adidas.com.ar/":    "Argentina",
    "https://www.adidas.com.br/":    "Brazil",
    "https://www.adidas.com/cr/es":  "Costa Rica",
    "https://www.adidas.com/ec/es":  "Ecuador",
}

# ── GSC API ────────────────────────────────────────────────────────────────────
GSC_ROW_LIMIT  = 25_000         # Max rows per API page (GSC hard limit)
GSC_DIMENSIONS = ["date", "query"]

# Fast mode: rows per domain per period (parallel, no pagination).
# 1 000 rows × 8 domains × 2 periods = 16 light API calls instead of hundreds.
FAST_ROW_LIMIT = 1_000

# ── Data Storage ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

# ── Comparison Logic ───────────────────────────────────────────────────────────
# Minimum thresholds to avoid misleading percentage spikes from near-zero baselines.
MIN_CLICKS_THRESHOLD = 10       # A query must have at least this many clicks in either period
MIN_IMPRESSIONS_THRESHOLD = 50  # Same logic for impressions

# ── Anomaly Detection ──────────────────────────────────────────────────────────
# A change is flagged as an anomaly when the absolute delta in z-score exceeds this.
ANOMALY_ZSCORE_THRESHOLD = 2.0

# Minimum absolute delta required before a query is considered for anomaly flagging.
ANOMALY_MIN_CLICK_DELTA = 20
ANOMALY_MIN_IMPRESSION_DELTA = 100

# ── Brand Keywords ─────────────────────────────────────────────────────────────
# Queries containing any of these terms are classified as "Brand".
BRAND_TERMS = ["adidas", "yeezy", "stan smith", "superstar", "ultraboost", "nmd"]

# ── UI / App ───────────────────────────────────────────────────────────────────
APP_TITLE = "Adidas Search Intelligence"
APP_ICON = "👟"
TOP_N_DEFAULT = 20              # Default rows shown in ranking tables
