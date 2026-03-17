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

from config import ACCESS_PASSWORD, APP_ICON, APP_TITLE, CREDENTIALS_DICT, CREDENTIALS_FILE
from src import cache
from views import activation, buying, explorer, mtd, opportunities, overview, weekly

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


# ── Authentication gate ────────────────────────────────────────────────────────

def _check_auth() -> bool:
    """
    Show a branded password gate if ACCESS_PASSWORD is configured.

    Returns True if the user is authenticated (or no password is set).
    Calls st.stop() and returns False if the login form is shown.
    """
    import hmac

    if not ACCESS_PASSWORD:
        return True  # No password configured — open access (dev / trusted env)

    if st.session_state.get("authenticated"):
        return True

    # ── Hide sidebar & chrome on the login screen ─────────────────────────────
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: none !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        max-width: 400px !important;
        margin: 10vh auto 0 !important;
        padding-top: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Login card ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='text-align:center; margin-bottom:2rem;'>
            <div style='font-size:2.6rem; font-weight:900; letter-spacing:0.12em;
                        color:#000; line-height:1;'>adidas</div>
            <div style='color:#888; font-size:0.78rem; text-transform:uppercase;
                        letter-spacing:0.14em; margin-top:0.35rem;'>
                Search Intelligence · LatAm
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            "<p style='font-weight:700; font-size:0.95rem; margin-bottom:0.6rem;'>"
            "🔐 Acceso restringido</p>",
            unsafe_allow_html=True,
        )
        pwd = st.text_input(
            "Contraseña",
            type="password",
            placeholder="Ingresa tu contraseña",
            label_visibility="collapsed",
            key="login_pwd_input",
        )
        if st.button("Entrar →", type="primary", use_container_width=True):
            if pwd and hmac.compare_digest(pwd.encode(), ACCESS_PASSWORD.encode()):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta. Intenta de nuevo.")

        st.caption("Acceso exclusivo para el equipo SEO+ · adidas LatAm")

    st.stop()
    return False  # never reached — st.stop() halts execution


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading cached data…")
def load_cached_data():
    """Load all Parquet files from data/raw/ into a single DataFrame."""
    return cache.load_all()


def _do_refresh(fast: bool):
    """
    Core refresh logic shared by both buttons.

    Both modes now cover the last 2 complete calendar months + the current
    partial month so that WoW and MoM comparisons in Buying & Trading always
    have sufficient data.

    fast=True  → parallel, top-1 000 rows/domain per batch, 3 batches  (~30 s)
    fast=False → sequential, full pagination, full 2-month range        (~3–8 min)
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

    # ── Date math — always cover 2 complete months + current partial ──────────
    today = date.today()

    # Last complete month  (e.g. Feb 1–28 when today = Mar 12)
    curr_month_start  = today.replace(day=1)
    last_month_end    = curr_month_start - timedelta(days=1)
    last_month_start  = last_month_end.replace(day=1)

    # Month before that   (e.g. Jan 1–31 when today = Mar 12)
    two_months_ago_end   = last_month_start - timedelta(days=1)
    two_months_ago_start = two_months_ago_end.replace(day=1)

    # Current partial month up to yesterday (needed for WoW current week)
    partial_end = today - timedelta(days=1)

    if fast:
        # ── 3 parallel batches — one per month-ish slice ──────────────────
        # Each batch gets the top 1 000 keywords by impressions within its
        # period, so splitting by month maximises per-period keyword coverage.
        batches = [
            (two_months_ago_start, two_months_ago_end, "mes anterior al último"),
            (last_month_start,     last_month_end,     "último mes completo"),
            (curr_month_start,     partial_end,        "mes actual parcial (WoW)"),
        ]

        all_frames = []
        for b_start, b_end, label in batches:
            if b_start > b_end:          # skip if range is nonsensical
                continue
            with st.spinner(f"⚡ Fast pull {label}: {b_start} → {b_end}…"):
                try:
                    df = extract_fast(start_date=b_start, end_date=b_end)
                    if not df.empty:
                        all_frames.append(df)
                except Exception as e:
                    st.sidebar.error(f"Extraction failed ({label}): {e}")
                    return

        combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
        start, end = two_months_ago_start, partial_end

    else:
        # ── Full paginated extraction — entire 2-month window ─────────────
        start = two_months_ago_start
        end   = partial_end
        with st.spinner(f"🔄 Full pull: {start} → {end} (this may take a few minutes)…"):
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
            options=[
                "📊  Overview",
                "📈  Week-over-Week",
                "📅  Month-to-Date",
                "🛒  Buying & Trading",
                "🎯  Digital Activation",
                "🔍  Keyword Explorer",
                "💡  Oportunidades SEO",
            ],
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
                         help="Paralelo · top 1,000 filas/mercado · 3 batches: mes anterior + último mes + mes actual (~30 s)"):
                _do_refresh(fast=True)
        with col_b:
            if st.button("🔄 Full", use_container_width=True,
                         help="Paginación completa · todas las filas · 2 meses completos + parcial (~3–8 min)"):
                _do_refresh(fast=False)

        st.divider()
        st.caption("SEO+ LAM · Adidas Search Intelligence")

        # ── Logout button (only shown when password protection is active) ──
        if ACCESS_PASSWORD and st.session_state.get("authenticated"):
            if st.button("🔒 Cerrar sesión", use_container_width=True):
                st.session_state["authenticated"] = False
                st.rerun()

    # Strip emoji prefix before returning clean page name
    return page.split("  ")[-1]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Auth gate — must pass before anything else renders ─────────────────────
    _check_auth()

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
    elif page == "Buying & Trading":
        buying.render(df)
    elif page == "Digital Activation":
        activation.render(df)
    elif page == "Keyword Explorer":
        explorer.render(df)
    elif page == "Oportunidades SEO":
        opportunities.render(df)


if __name__ == "__main__":
    main()
