"""
app.py
------
Adidas Search Intelligence Platform — Streamlit entry point.

Run with:
    streamlit run app.py

The app:
  1. Loads cached data from data/raw/ on startup (fast, no API call).
  2. Provides a sidebar button to refresh data from the GSC API.
  3. Routes to four views via the sidebar navigation.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
# Ensure project root is on sys.path so relative imports work when running
# from any directory.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import APP_ICON, APP_TITLE, CREDENTIALS_DICT, CREDENTIALS_FILE
from src import cache
from views import explorer, mtd, overview, weekly

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ─────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None

if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = None


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading cached data…")
def load_cached_data():
    """Load all Parquet files from data/raw/ into a single DataFrame."""
    return cache.load_all()


def _do_refresh(fast: bool):
    """
    Core refresh logic shared by both buttons.
    fast=True  → parallel, top-1000 rows per domain, 2 weeks only (~15 s)
    fast=False → sequential, full pagination, 5 weeks         (~2–5 min)
    """
    from datetime import date, timedelta
    from src.extractor import build_service, extract_all_domains, extract_fast

    # ── Auth ──────────────────────────────────────────────────────────────────
    with st.spinner("Connecting to Google Search Console API…"):
        try:
            service = build_service()
        except Exception as e:
            st.sidebar.error(f"Auth failed: {e}")
            return

    # ── Date range ────────────────────────────────────────────────────────────
    today = date.today()
    # Always pull from the Monday of the previous week to cover both WoW weeks
    days_since_monday = today.weekday()
    last_monday = today - timedelta(days=days_since_monday)
    curr_week_end   = last_monday - timedelta(days=1)          # last Sunday
    curr_week_start = curr_week_end - timedelta(days=6)        # Monday before that
    prev_week_start = curr_week_start - timedelta(days=7)

    if fast:
        # Pull current week + previous week in parallel (two fast batches)
        all_frames = []
        for w_start, w_end in [(prev_week_start, curr_week_start - timedelta(days=1)),
                                (curr_week_start, curr_week_end)]:
            with st.spinner(f"Fast pull: {w_start} → {w_end} (all markets in parallel)…"):
                try:
                    df = extract_fast(start_date=w_start, end_date=w_end)
                    if not df.empty:
                        all_frames.append(df)
                except Exception as e:
                    st.sidebar.error(f"Extraction failed: {e}")
                    return
        combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
        start, end = prev_week_start, curr_week_end
    else:
        # Full extraction — 5 weeks, paginated
        end   = today - timedelta(days=1)
        start = end - timedelta(days=34)
        with st.spinner(f"Full pull: {start} → {end} (this may take a few minutes)…"):
            try:
                combined = extract_all_domains(service, start_date=start, end_date=end)
            except Exception as e:
                st.sidebar.error(f"Extraction failed: {e}")
                return

    if combined.empty:
        st.sidebar.warning("API returned no data.")
        return

    cache.save(combined, start, end)
    load_cached_data.clear()
    st.session_state["df"] = load_cached_data()
    st.session_state["last_refresh"] = date.today().isoformat()
    st.sidebar.success(f"Done — {len(combined):,} rows saved.")
    st.rerun()


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    """Inject global CSS for B&W design system."""
    st.markdown("""
    <style>
    /* ── Hide Streamlit chrome ─────────────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Page layout ───────────────────────────────────────────────── */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1200px !important;
    }

    /* ── Sidebar — dark background ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #0F0F0F !important;
        border-right: 1px solid #2A2A2A;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption { color: #AAAAAA !important; }
    section[data-testid="stSidebar"] strong { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] hr { border-color: #2A2A2A !important; }

    /* Sidebar nav radio */
    section[data-testid="stSidebar"] .stRadio label {
        color: #DDDDDD !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 4px 0 !important;
    }
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div:first-child {
        background-color: #FFFFFF !important;
        border-color: #555 !important;
    }

    /* ── KPI metric cards ──────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: #F8F8F8;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 14px 18px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #888888 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #111111 !important;
    }
    [data-testid="stMetricDelta"] svg { display: none; }
    [data-testid="stMetricDelta"] > div {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }

    /* ── Section headers ───────────────────────────────────────────── */
    h1 { font-size: 1.55rem !important; font-weight: 800 !important; color: #0F0F0F !important; letter-spacing: -0.02em; }
    h2 { font-size: 1rem !important; font-weight: 700 !important; color: #111 !important;
         border-left: 3px solid #000; padding-left: 0.55rem !important; margin-top: 0.2rem !important; }
    h3 { font-size: 0.9rem !important; font-weight: 600 !important; color: #333 !important; }

    /* ── Tabs ──────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        border-bottom: 2px solid #E8E8E8;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.83rem !important;
        font-weight: 500 !important;
        color: #777 !important;
        border-radius: 5px 5px 0 0 !important;
        padding: 6px 18px !important;
        background: transparent !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        font-weight: 700 !important;
        background: #F4F4F4 !important;
        border-bottom: 2px solid #000 !important;
    }

    /* ── Buttons ───────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        letter-spacing: 0.01em;
        transition: all 0.15s ease;
    }
    .stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

    /* ── Filter container (st.container border=True) ───────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FAFAFA !important;
        border: 1px solid #E4E4E4 !important;
        border-radius: 10px !important;
        padding: 4px 8px !important;
        margin-bottom: 0.8rem !important;
    }

    /* ── Selectbox & inputs ────────────────────────────────────────── */
    .stSelectbox label, .stTextInput label, .stRadio label {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #888 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }
    .stSelectbox [data-baseweb="select"] {
        border-radius: 8px !important;
        border-color: #E0E0E0 !important;
        background: #FFFFFF !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #999 !important;
    }
    .stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.88rem !important;
    }
    .stTextInput input {
        border-radius: 8px !important;
        border-color: #E0E0E0 !important;
        font-size: 0.88rem !important;
    }

    /* ── Dataframes ────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; border: 1px solid #EEEEEE; }

    /* ── Expanders ─────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        background: #F8F8F8 !important;
        border-radius: 6px !important;
    }

    /* ── Info / warning / success boxes ────────────────────────────── */
    [data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.84rem !important; }

    /* ── Dividers ──────────────────────────────────────────────────── */
    hr { border-color: #F2F2F2 !important; margin: 1.5rem 0 !important; }

    /* ── Caption ───────────────────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: #999 !important;
        font-size: 0.77rem !important;
    }

    /* ── Download button ───────────────────────────────────────────── */
    [data-testid="stDownloadButton"] button {
        border: 1px solid #CCCCCC !important;
        background: white !important;
        color: #333 !important;
        font-size: 0.82rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar() -> str:
    with st.sidebar:
        # ── Logo + title ───────────────────────────────────────────────
        logo_found = False
        for ext in ("logo.jpg", "logo.jpeg", "logo.png"):
            logo_path = ROOT / "assets" / ext
            if logo_path.exists():
                # Center the logo at a fixed smaller width
                col_pad_l, col_img, col_pad_r = st.columns([1, 3, 1])
                with col_img:
                    st.image(str(logo_path), use_container_width=True)
                logo_found = True
                break
        # Title always shown — below logo if present, standalone if not
        if logo_found:
            st.markdown(
                f"<p style='text-align:center; color:#CCCCCC; font-size:0.78rem;"
                f" font-weight:600; letter-spacing:0.08em; margin-top:-4px;"
                f" text-transform:uppercase;'>{APP_TITLE}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"### {APP_TITLE}")

        st.divider()

        # ── Navigation ─────────────────────────────────────────────────
        page = st.radio(
            "Navigate",
            options=["📊  Overview", "📈  Week-over-Week", "📅  Month-to-Date", "🔍  Keyword Explorer"],
            label_visibility="collapsed",
        )

        st.divider()

        # ── Data status ────────────────────────────────────────────────
        st.markdown("**Data status**")
        min_date, max_date = cache.available_date_range()
        if min_date and max_date:
            st.caption(f"📁  {min_date} → {max_date}")
            st.caption(f"🗂  {len(cache.cached_files())} file(s) cached")
        else:
            st.caption("No local cache found.")

        if st.session_state.get("last_refresh"):
            st.caption(f"🕐  Last refresh: {st.session_state['last_refresh']}")

        st.markdown(" ")
        st.markdown("**Refresh data**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⚡ Quick", use_container_width=True, type="primary",
                         help="Parallel · top 1,000 rows/market · last 2 weeks (~15 s)"):
                _do_refresh(fast=True)
        with col_b:
            if st.button("🔄 Full", use_container_width=True,
                         help="Full pagination · all rows · last 5 weeks (~2–5 min)"):
                _do_refresh(fast=False)

        st.divider()
        st.caption("SEO+ LAM · Adidas Search Intelligence")

        # ── 🔧 Credential diagnostics (remove once auth works) ─────────
        with st.expander("🔧 Auth diagnostics", expanded=False):
            import json as _dbg_json, base64 as _dbg_b64, os

            sec = st.secrets.get("gsc_credentials", {})
            sub_keys = list(sec.keys()) if sec else []
            st.caption(f"gsc_credentials sub-keys: {sub_keys}")

            if "json_b64" in sec:
                st.success("✅ json_b64 key found (base64 format)")
                try:
                    decoded = _dbg_b64.b64decode(str(sec["json_b64"]).strip())
                    parsed  = _dbg_json.loads(decoded)
                    pk = parsed.get("private_key", "")
                    st.success(f"✅ Decoded OK — keys: {list(parsed.keys())}")
                    st.caption(f"private_key has real newlines: {chr(10) in pk}")
                except Exception as exc:
                    st.error(f"❌ Decode failed: {exc}")

            elif "json" in sec:
                st.warning("⚠️ json key found (raw JSON format) — prefer json_b64")
                try:
                    parsed = _dbg_json.loads(str(sec["json"]))
                    st.success(f"✅ json.loads() OK — keys: {list(parsed.keys())}")
                except Exception as exc:
                    st.error(f"❌ json.loads() FAILED: {exc}")
            else:
                st.error("❌ Neither json_b64 nor json key found in secrets")

            st.divider()
            st.markdown("**config.py CREDENTIALS_DICT:**")
            if CREDENTIALS_DICT is not None:
                st.success("✅ Loaded successfully")
            else:
                st.error("❌ None — check secret format above")
                st.caption(f"Fallback file exists: {os.path.exists(CREDENTIALS_FILE)}")

    # Strip emoji prefix before returning clean page name
    return page.split("  ")[-1]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load data into session state on first run
    if st.session_state["df"] is None:
        st.session_state["df"] = load_cached_data()

    df = st.session_state["df"]

    page = render_sidebar()

    if page == "Overview":
        overview.render(df)
    elif page == "Week-over-Week":
        weekly.render(df)
    elif page == "Month-to-Date":
        mtd.render(df)
    elif page == "Keyword Explorer":
        explorer.render(df)


if __name__ == "__main__":
    main()
