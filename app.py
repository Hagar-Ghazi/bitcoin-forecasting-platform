"""
Professional Streamlit UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


st.set_page_config(
    page_title  = "BTC Forecasting Portal",
    page_icon   = "₿",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


from src.data_loader        import load_and_validate_data, get_available_price_columns
from src.models.nbeats_model import NBEATSForecaster
from src.models.prophet_model          import ProphetForecaster
from src.models.xgb_direct_multi_step import MLForecaster as MLForecasterDirect
from src.models.xgb_recursive         import MLForecaster as MLForecasterRecursive
from src.evaluation         import compute_metrics, display_metrics, mean_absolute_percentage_error
from src.visualizations     import (
    plot_backtest_performance,
    plot_forecast,
    plot_residuals,
    plot_feature_importance,
    plot_metrics_table,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — dark trading terminal
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --btc-orange:     #F7931A;
    --btc-orange-dim: rgba(247,147,26,0.15);
    --bg-primary:     #0A0B0D;
    --bg-secondary:   #0F1117;
    --bg-card:        #13161E;
    --bg-hover:       #1A1E2A;
    --border:         rgba(247,147,26,0.25);
    --border-dim:     rgba(255,255,255,0.10);
    --text-primary:   #F0EDE8;
    --text-secondary: #B0ABA3;
    --text-muted:     #7A7670;
    --green:          #00D4AA;
    --red:            #FF4E5B;
    --mono:           'IBM Plex Mono', monospace;
    --sans:           'IBM Plex Sans', sans-serif;
}

/* ── Page fade-in animation ── */
@keyframes pageFadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes tickerScroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

.stApp {
    background-color: var(--bg-primary) !important;
    animation: pageFadeIn 0.55s ease forwards !important;
}

/* ── Global overrides ── */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1600px !important;
}

/* ── Ticker tape ── */
.ticker-wrap {
    width: 100%;
    background: #0D0F14;
    border-top: 1px solid rgba(247,147,26,0.20);
    border-bottom: 1px solid rgba(247,147,26,0.20);
    overflow: hidden;
    padding: 7px 0;
    margin-bottom: 0;
}
.ticker-track {
    display: flex;
    width: max-content;
    animation: tickerScroll 38s linear infinite;
    gap: 0;
}
.ticker-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    white-space: nowrap;
    padding: 0 2.2rem;
    color: #B0ABA3;
    letter-spacing: 0.04em;
    border-right: 1px solid rgba(255,255,255,0.07);
}
.ticker-item .ticker-sym  { color: #F7931A; font-weight: 600; margin-right: 0.4rem; }
.ticker-item .ticker-up   { color: #00D4AA; }
.ticker-item .ticker-down { color: #FF4E5B; }
.ticker-item .ticker-dot  {
    display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: #00D4AA;
    margin-right: 0.5rem;
    animation: pulse 2s ease-in-out infinite;
    vertical-align: middle;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p {
    color: var(--text-secondary) !important;
    font-family: var(--mono) !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: var(--mono) !important;
    font-size: 0.88rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--btc-orange) !important;
    box-shadow: 0 0 0 2px var(--btc-orange-dim) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--btc-orange) !important;
}
[data-testid="stSlider"] > div > div > div {
    background: var(--border-dim) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--btc-orange) !important;
    background: var(--btc-orange-dim) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: var(--btc-orange) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    text-shadow: none !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #FFa830 !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(247,147,26,0.35) !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* ── Secondary button ── */
.stButton > button:not([kind="primary"]) {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    width: 100% !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-family: var(--mono) !important;
    font-size: 0.80rem !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--btc-orange) !important;
    font-family: var(--mono) !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid var(--border-dim) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.75rem 1.4rem !important;
    border-radius: 0 !important;
    text-transform: uppercase !important;
    transition: color 0.15s ease !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--btc-orange) !important;
    border-bottom-color: var(--btc-orange) !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: var(--text-primary) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border-dim) !important;
    border-radius: 6px !important;
    background: var(--bg-card) !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
}

/* ── Info / warning / error boxes ── */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.88rem !important;
}

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border-dim) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
    color: var(--text-secondary) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] label {
    color: var(--text-secondary) !important;
    font-family: var(--mono) !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border-dim) !important;
    margin: 1.5rem 0 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--btc-orange) !important;
}

/* ── Success / error message ── */
.element-container .stSuccess {
    background: rgba(0, 212, 170, 0.08) !important;
    border-left: 3px solid var(--green) !important;
}
.element-container .stError {
    background: rgba(255, 78, 91, 0.08) !important;
    border-left: 3px solid var(--red) !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENT HELPERS  (reusable HTML blocks)
# ═══════════════════════════════════════════════════════════════════════════════

def _header() -> None:
    st.markdown("""
    <!-- Animated Ticker Tape -->
    <div class="ticker-wrap">
      <div class="ticker-track">
        <span class="ticker-item"><span class="ticker-dot"></span><span class="ticker-sym">BTC/USD</span> $67,412 <span class="ticker-up">&#9650; 2.41%</span></span>
        <span class="ticker-item"><span class="ticker-sym">ETH/USD</span> $3,521 <span class="ticker-up">&#9650; 1.18%</span></span>
        <span class="ticker-item"><span class="ticker-sym">BNB/USD</span> $412 <span class="ticker-down">&#9660; 0.33%</span></span>
        <span class="ticker-item"><span class="ticker-sym">SOL/USD</span> $187 <span class="ticker-up">&#9650; 3.07%</span></span>
        <span class="ticker-item"><span class="ticker-sym">XRP/USD</span> $0.612 <span class="ticker-down">&#9660; 1.22%</span></span>
        <span class="ticker-item"><span class="ticker-sym">ADA/USD</span> $0.491 <span class="ticker-up">&#9650; 0.85%</span></span>
        <span class="ticker-item"><span class="ticker-sym">DOGE/USD</span> $0.172 <span class="ticker-up">&#9650; 4.30%</span></span>
        <span class="ticker-item"><span class="ticker-sym">AVAX/USD</span> $38.4 <span class="ticker-down">&#9660; 0.74%</span></span>
        <span class="ticker-item"><span class="ticker-sym">MATIC/USD</span> $0.88 <span class="ticker-up">&#9650; 1.92%</span></span>
        <span class="ticker-item"><span class="ticker-sym">LINK/USD</span> $18.2 <span class="ticker-down">&#9660; 0.55%</span></span>
        <span class="ticker-item"><span class="ticker-dot"></span><span class="ticker-sym">BTC/USD</span> $67,412 <span class="ticker-up">&#9650; 2.41%</span></span>
        <span class="ticker-item"><span class="ticker-sym">ETH/USD</span> $3,521 <span class="ticker-up">&#9650; 1.18%</span></span>
        <span class="ticker-item"><span class="ticker-sym">BNB/USD</span> $412 <span class="ticker-down">&#9660; 0.33%</span></span>
        <span class="ticker-item"><span class="ticker-sym">SOL/USD</span> $187 <span class="ticker-up">&#9650; 3.07%</span></span>
        <span class="ticker-item"><span class="ticker-sym">XRP/USD</span> $0.612 <span class="ticker-down">&#9660; 1.22%</span></span>
        <span class="ticker-item"><span class="ticker-sym">ADA/USD</span> $0.491 <span class="ticker-up">&#9650; 0.85%</span></span>
        <span class="ticker-item"><span class="ticker-sym">DOGE/USD</span> $0.172 <span class="ticker-up">&#9650; 4.30%</span></span>
        <span class="ticker-item"><span class="ticker-sym">AVAX/USD</span> $38.4 <span class="ticker-down">&#9660; 0.74%</span></span>
        <span class="ticker-item"><span class="ticker-sym">MATIC/USD</span> $0.88 <span class="ticker-up">&#9650; 1.92%</span></span>
        <span class="ticker-item"><span class="ticker-sym">LINK/USD</span> $18.2 <span class="ticker-down">&#9660; 0.55%</span></span>
      </div>
    </div>

    <!-- Page Header -->
    <div style="
        padding: 2.2rem 0 1.8rem 0;
        border-bottom: 1px solid rgba(247,147,26,0.25);
        margin-bottom: 2rem;
    ">
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;">
            <span style="
                font-family:'IBM Plex Mono',monospace;
                font-size:1.85rem;
                font-weight:600;
                color:#F7931A;
                letter-spacing:-0.01em;
            ">&#8383; BTC Forecasting Portal</span>
            <span style="
                font-family:'IBM Plex Mono',monospace;
                font-size:0.76rem;
                color:#B0ABA3;
                letter-spacing:0.12em;
                text-transform:uppercase;
                border:1px solid rgba(247,147,26,0.30);
                background:rgba(247,147,26,0.08);
                padding:3px 10px;
                border-radius:3px;
            ">v2.0 &middot; Phase 4</span>
        </div>
        <p style="
            font-family:'IBM Plex Mono',monospace;
            font-size:0.88rem;
            color:#B0ABA3;
            margin:0;
            letter-spacing:0.04em;
        ">Time-Series Analysis &amp; Forecasting &middot; Prophet &middot; XGBoost &middot; Plotly</p>
    </div>
    """, unsafe_allow_html=True)


def _sidebar_section(title: str) -> None:
    st.markdown(f"""
    <div style="
        font-family:'IBM Plex Mono',monospace;
        font-size:0.75rem;
        color:#F7931A;
        letter-spacing:0.14em;
        text-transform:uppercase;
        margin: 1.6rem 0 0.7rem 0;
        padding-bottom:0.45rem;
        border-bottom:1px solid rgba(247,147,26,0.28);
    ">{title}</div>
    """, unsafe_allow_html=True)


def _stat_card(label: str, value: str, sub: str = "", color: str = "#F7931A") -> str:
    return f"""
    <div style="
        background:#13161E;
        border:1px solid rgba(255,255,255,0.06);
        border-radius:8px;
        padding:1rem 1.2rem;
        height:100%;
    ">
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:0.80rem;
            color:#9AA3B0;
            letter-spacing:0.10em;
            text-transform:uppercase;
            margin-bottom:0.5rem;
        ">{label}</div>
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:1.5rem;
            font-weight:600;
            color:{color};
            line-height:1.1;
        ">{value}</div>
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:0.80rem;
            color:#8A9BB0;
            margin-top:0.3rem;
        ">{sub}</div>
    </div>
    """


def _info_banner(text: str, icon: str = "◈") -> None:
    st.markdown(f"""
    <div style="
        background:rgba(247,147,26,0.06);
        border:1px solid rgba(247,147,26,0.18);
        border-radius:6px;
        padding:0.8rem 1rem;
        font-family:'IBM Plex Mono',monospace;
        font-size:0.88rem;
        color:#B0ABA3;
        display:flex;
        gap:0.6rem;
        align-items:flex-start;
    ">
        <span style="color:#F7931A; flex-shrink:0;">{icon}</span>
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)


def _section_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"""
    <div style="margin: 1.5rem 0 1rem 0;">
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:0.85rem;
            color:#F7931A;
            letter-spacing:0.10em;
            text-transform:uppercase;
            margin-bottom:0.3rem;
        ">{title}</div>
        {"" if not subtitle else f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:0.90rem;color:#9AA3B0;">{subtitle}</div>'}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "df":           None,
        "price_col":    "Close",
        "result":       None,
        "model_choice": None,
        "horizon":      30,
        "confidence":   0.95,
        "missing_days": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # Logo / brand 
    st.markdown("""
    <style>
    @keyframes btcFloat {
        0%   { transform: translateX(-18px) rotate(-8deg); }
        50%  { transform: translateX(18px)  rotate(8deg);  }
        100% { transform: translateX(-18px) rotate(-8deg); }
    }
    .btc-coin-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0.6rem 0 0.5rem 0;
    }
    .btc-coin {
        width: 72px;
        height: 72px;
        animation: btcFloat 3s ease-in-out infinite;
        filter: drop-shadow(0 4px 12px rgba(247,147,26,0.45));
    }
    </style>

    <div style="
        text-align:center;
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid rgba(247,147,26,0.15);
        margin-bottom: 0.5rem;
    ">
        <div class="btc-coin-wrap">
            <svg class="btc-coin" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <circle cx="50" cy="50" r="48" fill="#F7931A"/>
                <circle cx="50" cy="50" r="44" fill="#E8820A"/>
                <circle cx="50" cy="50" r="41" fill="#F7931A"/>
                <ellipse cx="38" cy="32" rx="10" ry="6" fill="rgba(255,255,255,0.18)" transform="rotate(-20 38 32)"/>
                <text x="51" y="67" font-family="Arial Black,sans-serif" font-size="46" font-weight="900" text-anchor="middle" fill="#7A4A00" opacity="0.35">B</text>
                <text x="50" y="66" font-family="Arial Black,sans-serif" font-size="46" font-weight="900" text-anchor="middle" fill="#ffffff">B</text>
                <line x1="50" y1="2"  x2="50" y2="9"  stroke="#C87010" stroke-width="2" transform="rotate(0 50 50)"/>
                <line x1="50" y1="2"  x2="50" y2="9"  stroke="#C87010" stroke-width="2" transform="rotate(45 50 50)"/>
                <line x1="50" y1="2"  x2="50" y2="9"  stroke="#C87010" stroke-width="2" transform="rotate(90 50 50)"/>
                <line x1="50" y1="2"  x2="50" y2="9"  stroke="#C87010" stroke-width="2" transform="rotate(135 50 50)"/>
                <line x1="50" y1="2"  x2="50" y2="9"  stroke="#C87010" stroke-width="2" transform="rotate(180 50 50)"/>
                <line x1="50" y1="2"  x2="50" y2="9"  stroke="#C87010" stroke-width="2" transform="rotate(225 50 50)"/>
                <line x1="50" y1="2"  x2="50" y2="9"  stroke="#C87010" stroke-width="2" transform="rotate(270 50 50)"/>
                <line x1="50" y1="2"  x2="50" y2="9"  stroke="#C87010" stroke-width="2" transform="rotate(315 50 50)"/>
            </svg>
        </div>
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:0.78rem;
            color:#B0ABA3;
            letter-spacing:0.12em;
            text-transform:uppercase;
            margin-top:0.2rem;
        ">Control Panel</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Data upload 
    _sidebar_section("① Data Source")

    uploaded_file = st.file_uploader(
        "Upload BTC CSV",
        type   = ["csv"],
        help   = "Kaggle-format BTC CSV with Date and OHLCV columns",
        label_visibility = "collapsed",
    )

    if uploaded_file is not None:
        available_cols = get_available_price_columns(uploaded_file)

        price_col = st.selectbox(
            "Price Column",
            options = available_cols,
            index   = 0,
            help    = "Which OHLC price to use for forecasting",
        )

        if (
            st.session_state.df is None
            or st.session_state.price_col != price_col
            or uploaded_file.name != getattr(st.session_state, "_last_filename", "")
        ):
            with st.spinner("Parsing & validating data…"):
                uploaded_file.seek(0)
                df, error = load_and_validate_data(uploaded_file, price_col=price_col)

            if error:
                st.error(f"**Data Error** — {error}")
                st.session_state.df = None
            else:
                st.session_state.df           = df
                st.session_state.price_col    = price_col
                st.session_state._last_filename = uploaded_file.name
                st.session_state.result       = None   # reset on new data
                st.success(f"Loaded {len(df):,} rows")

    # Dataset summary card 
    if st.session_state.df is not None:
        df = st.session_state.df
        pc = st.session_state.price_col
        st.markdown(f"""
        <div style="
            background:#0F1117;
            border:1px solid rgba(247,147,26,0.12);
            border-radius:6px;
            padding:0.8rem;
            margin-top:0.6rem;
            font-family:'IBM Plex Mono',monospace;
        ">
            <div style="font-size:0.75rem;color:#F7931A;letter-spacing:0.10em;text-transform:uppercase;margin-bottom:0.6rem;">Dataset Summary</div>
            <div style="font-size:0.82rem;color:#B0ABA3;">Rows&nbsp;&nbsp;&nbsp;<span style="color:#E8E6E0;float:right;">{len(df):,}</span></div>
            <div style="font-size:0.82rem;color:#B0ABA3;">From&nbsp;&nbsp;&nbsp;<span style="color:#E8E6E0;float:right;">{df['ds'].min().date()}</span></div>
            <div style="font-size:0.82rem;color:#B0ABA3;">To&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#E8E6E0;float:right;">{df['ds'].max().date()}</span></div>
            <div style="font-size:0.82rem;color:#B0ABA3;">Min&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#E8E6E0;float:right;">${df[pc].min():,.0f}</span></div>
            <div style="font-size:0.82rem;color:#B0ABA3;">Max&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#F7931A;float:right;font-weight:600;">${df[pc].max():,.0f}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Model selection 
    _sidebar_section("② Model")

    model_choice = st.selectbox(
        "Algorithm",
        options = [
            "Prophet (Optimised)",
            "XGBoost (Direct Multi-Step)",
            "XGBoost ML Regressor",
            "N-BEATS (Neural Basis Expansion)"
        ],
        help = (
            "Prophet: additive decomposition with BTC-specific tuning. "
            "XGBoost Direct: one model per forecast day — no error compounding. "
            "XGBoost Recursive: single model, predicts one step at a time."
        ),
        label_visibility = "visible",
    )

    if model_choice == "Prophet (Optimised)":
        with st.expander("Prophet Settings", expanded=False):
            changepoint_prior = st.slider(
                "Changepoint Flexibility",
                min_value = 0.01, max_value = 0.50,
                value     = 0.05, step      = 0.01,
                help      = "Higher = more trend flexibility. Default 0.05 tuned for BTC.",
            )
            n_folds = st.slider(
                "CV Folds",
                min_value = 1, max_value = 5, value = 3,
                help      = "Number of rolling cross-validation folds for back-test.",
            )
            use_regressors = st.toggle(
                "Use OHLCV Regressors",
                value = True,
                help  = "Add High/Low/Volume as Prophet regressors if available.",
            )
            use_halving = st.toggle(
                "BTC Halving Events",
                value = True,
                help  = "Add 2016 & 2020 halving dates as holiday regressors.",
            )
        # defaults for unused XGBoost params
        n_estimators  = 500
        max_depth     = 6
        learning_rate = 0.03

    else:
        # shared defaults for unused Prophet params
        changepoint_prior = 0.05
        n_folds           = 3
        use_regressors    = True
        use_halving       = True

        label = (
            "XGBoost Direct Settings"
            if model_choice == "XGBoost (Direct Multi-Step)"
            else "XGBoost Settings"
        )
        with st.expander(label, expanded=False):
            n_estimators = st.slider(
                "Trees (n_estimators)",
                min_value = 100, max_value = 1000, value = 500, step = 50,
            )
            max_depth = st.slider(
                "Max Depth",
                min_value = 3, max_value = 10, value = 6,
            )
            learning_rate = st.select_slider(
                "Learning Rate",
                options = [0.005, 0.01, 0.03, 0.05, 0.1],
                value   = 0.03,
            )

        if model_choice == "XGBoost (Direct Multi-Step)":
            st.markdown("""
            <div style="
                font-family:'IBM Plex Mono',monospace;
                font-size:0.72rem;
                color:#00D4AA;
                background:rgba(0,212,170,0.06);
                border:1px solid rgba(0,212,170,0.18);
                border-radius:4px;
                padding:0.5rem 0.7rem;
                margin-top:0.4rem;
                line-height:1.6;
            ">
                ◈ One model trained per forecast day<br>
                ◈ No error compounding on long horizons<br>
                ◈ Wider, honest confidence intervals
            </div>
            """, unsafe_allow_html=True)

    # 3. Forecast parameters 
    _sidebar_section("③ Forecast")

    horizon = st.select_slider(
        "Horizon (days)",
        options = [7, 14, 30, 60, 90],
        value   = 30,
        help    = "How many calendar days ahead to project.",
    )

    confidence_pct = st.select_slider(
        "Confidence Interval",
        options = [80, 90, 95, 99],
        value   = 95,
        format_func = lambda x: f"{x}%",
    )
    confidence = confidence_pct / 100

    # 4. Chart overlays 
    _sidebar_section("④ Chart Overlays")

    show_sma = st.toggle("Simple Moving Average (SMA)", value=False)
    if show_sma:
        sma_window = st.select_slider("SMA Window", options=[7, 14, 20, 50, 100, 200], value=20)
    else:
        sma_window = None

    show_ema = st.toggle("Exponential Moving Average (EMA)", value=False)
    if show_ema:
        ema_span = st.select_slider("EMA Span", options=[9, 12, 20, 26, 50], value=20)
    else:
        ema_span = None

    # 5. Generate button 
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    generate = st.button(
        "⚡  Generate Forecast",
        type             = "primary",
        use_container_width = True,
        disabled         = (st.session_state.df is None),
    )

    if st.session_state.df is None:
        st.markdown("""
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:0.68rem;
            color:#4A4845;
            text-align:center;
            margin-top:0.4rem;
        ">↑ Upload a CSV to enable</div>
        """, unsafe_allow_html=True)

    # Reset 
    if st.session_state.result is not None:
        if st.button("↺  Reset", use_container_width=True):
            for k in ["df", "result", "model_choice"]:
                st.session_state[k] = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

_header()

# PRE-UPLOAD STATE 
if st.session_state.df is None:
    st.markdown("""
    <div style="
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        padding: 5rem 2rem;
        gap: 1.5rem;
    ">
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:4rem;
            color:rgba(247,147,26,0.15);
        ">₿</div>
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:1rem;
            color:#9AA3B0;
            letter-spacing:0.08em;
            text-transform:uppercase;
        ">Upload a BTC CSV in the sidebar to begin</div>
        <div style="
            font-family:'IBM Plex Mono',monospace;
            font-size:0.72rem;
            color:#7A7670;
            max-width:400px;
            text-align:center;
            line-height:1.7;
        ">
            Accepts Kaggle-format BTC-USD historical CSVs<br>
            with Date, Open, High, Low, Close, Volume columns
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# DATA LOADED, NO FORECAST YET 
df = st.session_state.df
pc = st.session_state.price_col



# Run forecast when button is clicked 
if generate:
    with st.spinner("Training model and generating forecast — this takes 20–60 seconds…"):
        try:
            if model_choice == "Prophet (Optimised)":
                forecaster = ProphetForecaster(
                    interval_width          = confidence,
                    changepoint_prior_scale = changepoint_prior,
                    n_folds                 = n_folds,
                    use_regressors          = use_regressors,
                    use_halving_holidays    = use_halving,
                )
            elif model_choice == "XGBoost (Direct Multi-Step)":
                forecaster = MLForecasterDirect(
                    n_estimators  = n_estimators,
                    max_depth     = max_depth,
                    learning_rate = learning_rate,
                    confidence    = confidence,
                )
            else:   # XGBoost ML Regressor (Recursive)
                forecaster = MLForecasterRecursive(
                    n_estimators  = n_estimators,
                    max_depth     = max_depth,
                    learning_rate = learning_rate,
                    confidence    = confidence,
                )

            result = forecaster.fit_and_forecast(df, horizon_days=horizon, price_col=pc)
            st.session_state.result       = result
            st.session_state.model_choice = model_choice
            st.session_state.horizon      = horizon
            st.session_state.confidence   = confidence

        except Exception as e:
            st.error(f"**Forecast failed** — {e}")
            st.exception(e)
            st.stop()


# SHOW RESULTS 
if st.session_state.result is None:

    # Show preview of data while waiting for forecast
    _section_title("Historical Data Preview", f"Loaded {len(df):,} rows · {pc} price column")

    # Quick stats row
    price_series = df[pc]
    last_price   = price_series.iloc[-1]
    prev_price   = price_series.iloc[-2]
    delta_pct    = (last_price - prev_price) / prev_price * 100
    delta_color  = "#00D4AA" if delta_pct >= 0 else "#FF4E5B"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_stat_card("Latest Price", f"${last_price:,.0f}",
        f"{'▲' if delta_pct>=0 else '▼'} {abs(delta_pct):.2f}% vs prev day",
        "#00D4AA" if delta_pct>=0 else "#FF4E5B"), unsafe_allow_html=True)
    c2.markdown(_stat_card("All-Time High", f"${price_series.max():,.0f}",
        str(df.loc[price_series.idxmax(), 'ds'].date())), unsafe_allow_html=True)
    c3.markdown(_stat_card("All-Time Low", f"${price_series.min():,.0f}",
        str(df.loc[price_series.idxmin(), 'ds'].date())), unsafe_allow_html=True)
    c4.markdown(_stat_card("Years of Data",
        f"{(df['ds'].max()-df['ds'].min()).days/365.25:.1f}",
        f"{df['ds'].min().year} → {df['ds'].max().year}"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Historical chart
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x    = df["ds"],
        y    = df[pc],
        name = f"BTC {pc}",
        line = dict(color="#F7931A", width=1.5),
        fill = "tozeroy",
        fillcolor = "rgba(247,147,26,0.04)",
    ))

    if show_sma and sma_window:
        sma = df[pc].rolling(sma_window).mean()
        fig_hist.add_trace(go.Scatter(
            x=df["ds"], y=sma,
            name=f"SMA {sma_window}",
            line=dict(color="#A78BFA", width=1.2, dash="dot"),
        ))

    if show_ema and ema_span:
        ema = df[pc].ewm(span=ema_span, adjust=False).mean()
        fig_hist.add_trace(go.Scatter(
            x=df["ds"], y=ema,
            name=f"EMA {ema_span}",
            line=dict(color="#FBBF24", width=1.2, dash="dash"),
        ))

    fig_hist.update_layout(
        paper_bgcolor = "#0A0B0D",
        plot_bgcolor  = "#0A0B0D",
        font          = dict(family="'IBM Plex Mono', monospace", color="#8A8880"),
        height        = 420,
        margin        = dict(l=60, r=20, t=30, b=50),
        hovermode     = "x unified",
        legend        = dict(
            bgcolor     = "rgba(10,11,13,0.8)",
            bordercolor = "rgba(255,255,255,0.06)",
            borderwidth = 1,
        ),
        xaxis = dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis = dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                     tickprefix="$", tickformat=",.0f"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    _info_banner(
        f"Configure your forecast parameters in the sidebar, then click "
        f"<strong style='color:#F7931A;'>⚡ Generate Forecast</strong> to run the model.",
        "◈"
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

result       = st.session_state.result
model_label  = st.session_state.model_choice
horizon_days = st.session_state.horizon

# Metrics row 
mae, rmse = result.mae_usd, result.rmse_usd
mape = mean_absolute_percentage_error(
    result.backtest_df["actual"],
    result.backtest_df["predicted"],
)

# Direction accuracy
bt = result.backtest_df
actual_dir = np.sign(np.diff(bt["actual"].values))
pred_dir   = np.sign(np.diff(bt["predicted"].values))
dir_acc    = np.mean(actual_dir == pred_dir) * 100

# Last forecast price
future_fc = result.forecast_df[result.forecast_df["ds"] > df["ds"].max()]
last_pred  = future_fc["yhat"].iloc[-1] if len(future_fc) else None

# Model + config badge 
st.markdown(f"""
<div style="
    display:flex;
    align-items:center;
    gap:0.8rem;
    margin-bottom:1.2rem;
    flex-wrap:wrap;
">
    <span style="
        font-family:'IBM Plex Mono',monospace;
        font-size:0.65rem;
        background:rgba(247,147,26,0.12);
        color:#F7931A;
        border:1px solid rgba(247,147,26,0.25);
        padding:3px 10px;
        border-radius:3px;
        letter-spacing:0.1em;
        text-transform:uppercase;
    ">{model_label}</span>
    <span style="
        font-family:'IBM Plex Mono',monospace;
        font-size:0.65rem;
        background:rgba(255,255,255,0.04);
        color:#9AA3B0;
        border:1px solid rgba(255,255,255,0.12);
        padding:3px 10px;
        border-radius:3px;
        letter-spacing:0.08em;
    ">{horizon_days}D HORIZON</span>
    <span style="
        font-family:'IBM Plex Mono',monospace;
        font-size:0.65rem;
        background:rgba(255,255,255,0.04);
        color:#9AA3B0;
        border:1px solid rgba(255,255,255,0.12);
        padding:3px 10px;
        border-radius:3px;
        letter-spacing:0.08em;
    ">{int(confidence*100)}% CI</span>
    <span style="
        font-family:'IBM Plex Mono',monospace;
        font-size:0.65rem;
        background:rgba(255,255,255,0.04);
        color:#9AA3B0;
        border:1px solid rgba(255,255,255,0.12);
        padding:3px 10px;
        border-radius:3px;
        letter-spacing:0.08em;
    ">{pc} PRICE</span>
</div>
""", unsafe_allow_html=True)

# Metric cards 
c1, c2, c3, c4, c5 = st.columns(5)

c1.markdown(_stat_card("MAE", f"${mae:,.0f}", "Mean Absolute Error"), unsafe_allow_html=True)
c2.markdown(_stat_card("RMSE", f"${rmse:,.0f}", "Root Mean Sq. Error"), unsafe_allow_html=True)
c3.markdown(_stat_card("MAPE", f"{mape:.2f}%", "Avg % error vs price"), unsafe_allow_html=True)
c4.markdown(_stat_card(
    "Direction Acc.",
    f"{dir_acc:.1f}%",
    "Up/down correct",
    "#00D4AA" if dir_acc >= 55 else "#FF4E5B",
), unsafe_allow_html=True)

if last_pred:
    last_actual = df[pc].iloc[-1]
    chg         = (last_pred - last_actual) / last_actual * 100
    c5.markdown(_stat_card(
        f"{horizon_days}D Forecast",
        f"${last_pred:,.0f}",
        f"{'▲' if chg>=0 else '▼'} {abs(chg):.1f}% vs today",
        "#00D4AA" if chg >= 0 else "#FF4E5B",
    ), unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Forecast",
    "🔁  Back-test",
    "📊  Residuals",
    "🧠  Model Insights",
])

# Tab 1: Main forecast 
with tab1:
    _section_title("Price Forecast", f"{horizon_days}-day projection with {int(confidence*100)}% confidence interval")

    fig_fc = plot_forecast(
        historical_df = result.historical_df,
        forecast_df   = result.forecast_df,
        model_name    = model_label,
        show_sma      = sma_window,
        show_ema      = ema_span,
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast table
    if len(future_fc) > 0:
        with st.expander("📋  Forecast Data Table", expanded=False):
            display_fc = future_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            display_fc.columns = ["Date", "Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]
            display_fc["Date"] = display_fc["Date"].dt.date
            for col in ["Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]:
                display_fc[col] = display_fc[col].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_fc, use_container_width=True, hide_index=True)

# Tab 2: Back-test 
with tab2:
    _section_title(
        "Back-test Performance",
        f"Model tested on held-out historical data · {len(result.backtest_df)} days evaluated"
    )

    fig_bt = plot_backtest_performance(
        historical_df = result.historical_df,
        backtest_df   = result.backtest_df,
        model_name    = model_label,
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Per-fold breakdown for Prophet
    if "fold" in result.backtest_df.columns and result.backtest_df["fold"].nunique() > 1:
        _section_title("Per-Fold Metrics", "Rolling cross-validation results")
        fold_cols = st.columns(result.backtest_df["fold"].nunique())
        for i, (fold_id, fold_df) in enumerate(result.backtest_df.groupby("fold")):
            fold_mae, fold_rmse = compute_metrics(fold_df["actual"], fold_df["predicted"])
            with fold_cols[i]:
                st.markdown(
                    _stat_card(f"Fold {int(fold_id)+1}", f"${fold_mae:,.0f}", f"RMSE ${fold_rmse:,.0f}"),
                    unsafe_allow_html=True,
                )

# Tab 3: Residuals 
with tab3:
    _section_title("Residual Analysis", "Error distribution and temporal pattern of model mistakes")

    fig_res = plot_residuals(result.backtest_df, model_name=model_label)
    st.plotly_chart(fig_res, use_container_width=True)

    # Residual stats
    residuals = result.backtest_df["actual"] - result.backtest_df["predicted"]
    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(_stat_card("Mean Residual", f"${residuals.mean():,.0f}", "Bias"), unsafe_allow_html=True)
    r2.markdown(_stat_card("Std Residual",  f"${residuals.std():,.0f}",  "Spread"), unsafe_allow_html=True)
    r3.markdown(_stat_card("Max Overshot",  f"${residuals.max():,.0f}",  "Largest over-prediction"), unsafe_allow_html=True)
    r4.markdown(_stat_card("Max Undershot", f"${residuals.min():,.0f}",  "Largest under-prediction"), unsafe_allow_html=True)

# Tab 4: Model insights
with tab4:
    _section_title("Model Insights", "Parameters used and feature drivers")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        _section_title("Hyperparameters")
        params = result.model_params
        params_html = "".join([
            f"""<div style="
                display:flex; justify-content:space-between;
                font-family:'IBM Plex Mono',monospace;
                font-size:0.85rem;
                padding:0.4rem 0;
                border-bottom:1px solid rgba(255,255,255,0.06);
                color:#B0ABA3;
            ">
                <span>{k}</span>
                <span style="color:#E8E6E0;">{v}</span>
            </div>"""
            for k, v in params.items()
        ])
        st.markdown(f"""
        <div style="
            background:#13161E;
            border:1px solid rgba(255,255,255,0.06);
            border-radius:8px;
            padding:1rem;
        ">{params_html}</div>
        """, unsafe_allow_html=True)

    with col_b:
        if hasattr(result, "feature_importance") and result.feature_importance is not None:
            _section_title("Feature Importance")
            fig_fi = plot_feature_importance(result.feature_importance, top_n=15)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            _section_title("Model Components")
            _info_banner(
                "Feature importance is available for the XGBoost model. "
                "Switch to XGBoost in the sidebar for detailed feature breakdown.",
                "◈"
            )

    # Model explanation
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    with st.expander("📖  How this model handles crypto volatility", expanded=False):
        if "Prophet" in model_label:
            st.markdown("""
            **Prophet (Optimised) — Volatility Handling**

            - **Log transformation** on the price series compresses extreme values, making the model
              more robust to BTC's characteristic spike-and-crash cycles.
            - **Multiplicative seasonality** captures percentage-based seasonal effects rather than
              fixed USD swings — correct for an asset that has ranged from $100 to $100,000.
            - **Loose changepoint detection** (`changepoint_range=0.95`) allows Prophet to detect
              regime shifts close to the end of the training period — essential for crypto.
            - **OHLCV regressors** (RSI, volatility, momentum) give the model real-time market
              context beyond the price series alone.
            - **BTC halving events** are encoded as holidays with a 30-day pre-window and 180-day
              post-window, capturing the historically bullish post-halving price effect.
            - **Rolling cross-validation** across 3 folds gives an honest out-of-sample MAE rather
              than in-sample goodness-of-fit metrics.
            """)
        elif "Direct" in model_label:
            st.markdown("""
            **XGBoost (Direct Multi-Step) — Volatility Handling**

            - **One model per forecast day**: for a 30-day horizon, 30 independent XGBRegressors
              are trained. `model_h` is trained exclusively to predict `price[t + h]` directly.
            - **Zero error compounding**: each model always sees only real historical features —
              no predicted prices are ever fed back as inputs, eliminating the drift problem that
              plagues recursive strategies on volatile assets like BTC.
            - **Step-specific confidence intervals**: each model's own in-sample residual std is
              used to build its CI, so uncertainty bands correctly widen for longer horizons.
            - **30+ engineered features**: lag prices (t-1 to t-30), rolling statistics,
              momentum, log-return volatility, RSI, MACD, Bollinger Bands, and calendar effects.
            - **`StandardScaler` per step-model** — each horizon has its own scaler fitted on
              its own aligned training slice, preventing target-distribution leakage.
            - **Direction accuracy** metric measures whether the model correctly calls the daily
              up/down movement — the metric that matters most for trading signals.
            """)
        else:
            st.markdown("""
            **XGBoost ML Regressor (Recursive) — Volatility Handling**

            - **30+ engineered features**: lag prices (t-1 to t-30), rolling statistics,
              momentum, log-return volatility, RSI, MACD, Bollinger Bands, and calendar effects.
            - **`StandardScaler`** normalises all features before training — prevents large-scale
              lag features from dominating the gradient updates.
            - **`early_stopping_rounds=30`** in the back-test phase prevents overfitting on
              the training fold, using the held-out period as the stopping criterion.
            - **Recursive forecasting** — each future day is predicted one step at a time,
              with the predicted value appended and re-engineered as a lag for the next step.
            - **Empirical confidence bands** are derived from in-sample residual standard deviation,
              scaled by the chosen z-score — more honest than parametric assumptions.
            - **Direction accuracy** metric specifically measures whether the model gets the
              daily up/down movement correct — the metric that matters most for trading signals.
            """)






