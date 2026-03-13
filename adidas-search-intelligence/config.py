"""
config.py
---------
Central configuration for the Adidas Search Intelligence Platform.
All domain lists, thresholds, and environment-driven settings live here.
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ── Credentials ────────────────────────────────────────────────────────────────
# CREDENTIALS_FILE  → path to credentials.json (local / server deployments)
# CREDENTIALS_DICT  → dict with service-account info (Streamlit Cloud secrets)
#
# extractor.build_service() checks CREDENTIALS_DICT first; if None it falls
# back to loading CREDENTIALS_FILE from disk.  This avoids writing the PEM key
# to a temporary file (which corrupts the key on Streamlit Cloud).

CREDENTIALS_FILE: str = os.getenv("GSC_CREDENTIALS_FILE", "credentials.json")
CREDENTIALS_DICT: dict | None = None   # populated below when running on Streamlit Cloud

try:
    import base64 as _b64
    import json as _json
    import streamlit as st  # only importable when the Streamlit runtime is active

    if hasattr(st, "secrets") and "gsc_credentials" in st.secrets:
        sec = st.secrets["gsc_credentials"]

        if "json_b64" in sec:
            # ── Best format: base64-encoded JSON (no TOML escaping issues) ──
            # Generate with: python generate_secret.py
            raw_bytes = _b64.b64decode(str(sec["json_b64"]).strip())
            CREDENTIALS_DICT = _json.loads(raw_bytes)

        elif "json" in sec:
            # ── Fallback: raw JSON string ────────────────────────────────────
            CREDENTIALS_DICT = _json.loads(str(sec["json"]))

        else:
            # ── Legacy field-by-field format ────────────────────────────────
            _creds = dict(sec)
            if "private_key" in _creds:
                _creds["private_key"] = _creds["private_key"].replace("\\n", "\n")
            CREDENTIALS_DICT = _creds

except Exception:
    pass  # Not running inside Streamlit, or secret is missing — use file path instead

OAUTH_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"

# ── App Access Password ────────────────────────────────────────────────────────
# Protects the entire app with a password gate shown before any data renders.
# If not set, the app is accessible to anyone (use only in trusted environments).
#
# To enable on Streamlit Cloud, add to your secrets:
#   [auth]
#   password = "your-strong-password-here"
#
# To enable locally, set the env var:
#   APP_ACCESS_PASSWORD=your-password streamlit run app.py

ACCESS_PASSWORD: str | None = os.getenv("APP_ACCESS_PASSWORD")

try:
    import streamlit as _st_pw
    if hasattr(_st_pw, "secrets") and "auth" in _st_pw.secrets:
        _pwd = str(_st_pw.secrets["auth"].get("password", "")).strip()
        if _pwd:
            ACCESS_PASSWORD = _pwd
except Exception:
    pass  # Not in Streamlit runtime, or secret absent


# ── AI Insights (Anthropic Claude API) ────────────────────────────────────────
# Priority: Streamlit Cloud secrets [ai] section > ANTHROPIC_API_KEY env var.
# To enable on Streamlit Cloud add to secrets:
#   [ai]
#   anthropic_api_key = "sk-ant-..."
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

try:
    import streamlit as _st_ai
    if hasattr(_st_ai, "secrets") and "ai" in _st_ai.secrets:
        _key = str(_st_ai.secrets["ai"].get("anthropic_api_key", "")).strip()
        if _key:
            ANTHROPIC_API_KEY = _key
except Exception:
    pass  # Not running in Streamlit, or secret absent — env var fallback used

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
