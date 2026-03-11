"""
src/filters.py
--------------
Filter helpers: sidebar controls + in-page selector buttons.
"""

import streamlit as st
import pandas as pd

from config import BRAND_TERMS, DOMAIN_LABELS, TOP_N_DEFAULT


# ── Classification helpers ─────────────────────────────────────────────────────

def classify_brand(keyword: str, brand_terms: list[str] = BRAND_TERMS) -> str:
    """Return 'Brand' if any brand term is found in keyword, else 'Non-Brand'."""
    kw_lower = keyword.lower()
    for term in brand_terms:
        if term in kw_lower:
            return "Brand"
    return "Non-Brand"


def add_brand_column(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a 'brand_type' column to any DataFrame that has a 'keyword' column."""
    if "keyword" in df.columns and "brand_type" not in df.columns:
        df = df.copy()
        df["brand_type"] = df["keyword"].apply(classify_brand)
    return df


def domain_label(domain_url: str) -> str:
    """Return the friendly label for a domain URL."""
    return DOMAIN_LABELS.get(domain_url, domain_url)


# ── In-page selectors (prominent buttons at top of each view) ─────────────────

def render_country_selector(df: pd.DataFrame, key: str = "country") -> pd.DataFrame:
    """
    Render a dropdown (selectbox) for country/market selection
    directly in the page (not the sidebar).
    Returns the DataFrame filtered to the selected market.
    """
    if "domain" not in df.columns or df.empty:
        return df

    available_domains = sorted(df["domain"].unique())
    market_labels = ["All Markets"] + [domain_label(d) for d in available_domains]
    label_to_url = {domain_label(d): d for d in available_domains}

    selected = st.selectbox(
        "Market",
        options=market_labels,
        key=key,
    )

    if selected == "All Markets":
        return df

    domain_url = label_to_url.get(selected)
    if domain_url:
        return df[df["domain"] == domain_url].copy()
    return df


def render_brand_selector(df: pd.DataFrame, key: str = "brand_sel") -> pd.DataFrame:
    """
    Render a dropdown (selectbox) for Brand / Non-Brand selection
    directly in the page.
    Returns the DataFrame filtered to the selected type.
    """
    if "brand_type" not in df.columns or df.empty:
        return df

    selected = st.selectbox(
        "Keyword type",
        options=["All", "Brand", "Non-Brand"],
        key=key,
    )

    if selected == "All":
        return df
    return df[df["brand_type"] == selected].copy()


# ── Sidebar (Top-N slider only — country + brand are now in-page) ─────────────

def render_filters(df: pd.DataFrame, prefix: str = "") -> tuple[pd.DataFrame, int]:
    """
    Sidebar controls: only the Top-N slider.
    Country and brand are handled by the in-page selectors above.
    Returns (df_unchanged, top_n).
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display")

    top_n = st.sidebar.slider(
        "Rows to display",
        min_value=5,
        max_value=100,
        value=TOP_N_DEFAULT,
        step=5,
        key=f"{prefix}_topn",
    )

    return df, top_n


# ── Keyword search ─────────────────────────────────────────────────────────────

def keyword_search(df: pd.DataFrame, key: str = "kw_search") -> pd.DataFrame:
    """Render a text input that filters rows by keyword substring."""
    search = st.text_input("Search keyword", value="", key=key)
    if search and "keyword" in df.columns:
        df = df[df["keyword"].str.contains(search, case=False, na=False)]
    return df


# ── Product Category Classification (Buying & Trading) ────────────────────────

PRODUCT_CATEGORIES = {
    "Running":    ["running", "run ", "correr", "marathon", "maratón", "trail run", "jogging"],
    "Football":   ["football", "fútbol", "futbol", "soccer", "bota de", "taco ", "cleats"],
    "Training":   ["training", "gym", "fitness", "workout", "entrenamiento", "cross training"],
    "Originals":  ["originals", "stan smith", "superstar", "campus", "nmd", "gazelle", "samba", "forum"],
    "Basketball": ["basketball", "basquet", "nba", "court shoe"],
    "Outdoor":    ["outdoor", "hiking", "trail", "trekking", "montaña"],
    "Kids":       ["kids", " niño", " niña", "infantil", "junior"],
    "Ultraboost": ["ultraboost", "ultra boost", "boost"],
    "Swim/Beach": ["swim", "traje de baño", "natación", "swimwear", "playa"],
    "Tennis":     ["tennis", "tenis", "padel", "pádel"],
}


def classify_product_category(keyword: str) -> str:
    kw = keyword.lower()
    for cat, terms in PRODUCT_CATEGORIES.items():
        if any(t in kw for t in terms):
            return cat
    return "Other"


def add_category_column(df: pd.DataFrame) -> pd.DataFrame:
    if "keyword" in df.columns and "product_category" not in df.columns:
        df = df.copy()
        df["product_category"] = df["keyword"].apply(classify_product_category)
    return df


# ── Campaign Keyword Classification (Digital Activation) ──────────────────────

CAMPAIGN_CATEGORIES = {
    "Sale / Promotions": ["sale", "outlet", "rebajas", "descuento", "oferta", "promo", "liquidación"],
    "New Collection":    ["nueva colección", "new collection", "temporada", "nueva temporada"],
    "Collaborations":    ["collab", "limited edition", "edición limitada", "collaboration", "special edition"],
    "Sports Events":     ["copa del mundo", "world cup", "copa america", "euro ", "olympics", "olimpiadas", "mundial"],
    "Sustainability":    ["sustainable", "sostenible", "recycled", "reciclado", "parley", "eco "],
    "Back to School":    ["back to school", "vuelta al cole", "inicio de clases", "regreso a clases"],
}


def classify_campaign_keyword(keyword: str) -> str | None:
    kw = keyword.lower()
    for cat, terms in CAMPAIGN_CATEGORIES.items():
        if any(t in kw for t in terms):
            return cat
    return None


def add_campaign_column(df: pd.DataFrame) -> pd.DataFrame:
    """Tag keywords matching campaign categories; non-matching rows get None."""
    if "keyword" in df.columns and "campaign_category" not in df.columns:
        df = df.copy()
        df["campaign_category"] = df["keyword"].apply(classify_campaign_keyword)
    return df


# ── Internal ───────────────────────────────────────────────────────────────────

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column that exists in df, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None
