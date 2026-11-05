import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
user_api_key = DEFAULT_GEMINI_KEY

from data_manager import (
    load_ticker_data, update_all_tickers, get_available_tickers,
    MAG_7, INDEXES, SECTORS, MACRO_ASSETS, generate_synthetic_ratios,
    is_core_data_stale, auto_sync_core_data
)
from indicators import compute_all_metrics
from narrative_engine import analyze_market_narrative, compute_ratio_analytics
from ai_strategist import generate_ai_ideas, query_ai_agent
from confluence_engine import compute_confluence_score, compute_all_confluences
import importlib
import options_engine
importlib.reload(options_engine)
from options_engine import render_options_vol_page

st.set_page_config(
    page_title="Double Edge | Quantitative Macro & Equity Desk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- STEALTH QUANTITATIVE MACRO DESK THEME CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .tabular-num, [data-num], table, .kpi-val, .pill-num {
        font-variant-numeric: tabular-nums lining-nums;
        font-feature-settings: 'tnum' 1, 'zero' 1;
        letter-spacing: -0.015em;
    }
    
    /* Background & Main Container (Constrained Width) */
    .stApp {
        background-color: #08090C;
        color: #f1f5f9;
    }
    
    .main .block-container {
        max-width: 1200px;
        margin: 0 auto;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }
    
    /* Stealth Mint & Cherry Badges */
    .pill-green, .badge-mint {
        background: rgba(0, 245, 160, 0.10);
        color: #00F5A0;
        border: 1px solid rgba(0, 245, 160, 0.25);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 600;
        font-variant-numeric: tabular-nums lining-nums;
    }
    .pill-red, .badge-cherry {
        background: rgba(255, 46, 85, 0.10);
        color: #FF2E55;
        border: 1px solid rgba(255, 46, 85, 0.25);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 600;
        font-variant-numeric: tabular-nums lining-nums;
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.10);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 600;
    }
    .pill-cyan {
        background: rgba(0, 245, 160, 0.10);
        color: #00F5A0;
        border: 1px solid rgba(0, 245, 160, 0.25);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 600;
        font-variant-numeric: tabular-nums lining-nums;
    }

    /* Stealth Cards */
    .metric-card-lux, .stealth-card {
        background: #0D0F14;
        border: 1px solid #1B212D;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.5);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 12px;
    }
    .metric-card-lux:hover, .stealth-card:hover {
        border-color: #262E3E;
    }
    .card-label {
        font-size: 11.5px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .card-val {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        font-variant-numeric: tabular-nums lining-nums;
    }

    /* Stealth Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0D0F14;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1B212D;
        margin-bottom: 20px;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 7px 14px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 12px !important;
        border: 1px solid transparent !important;
        background-color: transparent !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: #131720 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #131720 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid rgba(0, 245, 160, 0.4) !important;
        box-shadow: 0 0 12px rgba(0, 245, 160, 0.15) !important;
    }

    /* Action Buttons */
    .stButton > button {
        background: #131720 !important;
        color: #00F5A0 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(0, 245, 160, 0.3) !important;
        border-radius: 8px !important;
        padding: 7px 16px !important;
        font-size: 12.5px !important;
        box-shadow: 0 0 10px rgba(0, 245, 160, 0.1) !important;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: #00F5A0 !important;
        color: #08090C !important;
        box-shadow: 0 0 15px rgba(0, 245, 160, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

def apply_dark_theme(fig, height=360, title=None):
    fig.update_layout(
        plot_bgcolor="#0D0F14",
        paper_bgcolor="#0D0F14",
        font=dict(family="Inter, -apple-system, sans-serif", color="#94a3b8", size=12),
        margin=dict(l=15, r=15, t=45 if title else 15, b=15),
        height=height,
        title=dict(text=title, font=dict(color="#ffffff", size=15, weight=700)) if title else None,
        xaxis=dict(gridcolor="#1B212D", zerolinecolor="#1B212D"),
        yaxis=dict(gridcolor="#1B212D", zerolinecolor="#1B212D"),
        legend=dict(bgcolor="rgba(13, 15, 20, 0.9)", bordercolor="#1B212D")
    )
    return fig

def style_table_status(val):
    if not isinstance(val, str):
        return ""
    if any(w in val for w in ["Alert", "Bearish", "Explosion", "Overbought", "Upside", "Fading"]):
        return "color: #FF2E55; font-weight: 700;"
    if any(w in val for w in ["Healthy", "Bullish", "Squeeze", "Support", "Bounce", "Downside", "Confirmed"]):
        return "color: #00F5A0; font-weight: 700;"
    return "color: #94a3b8;"

# ----------------- SIDEBAR CONTROLS & DESK STATUS -----------------
with st.sidebar:
    st.markdown('''
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <div style="width:34px; height:34px; border-radius:8px; background:#0D0F14; border:1px solid #1B212D; display:flex; align-items:center; justify-content:center; font-weight:800; color:#ffffff; font-size:13px; position:relative;">
                DE<span style="position:absolute; top:3px; right:3px; width:6px; height:6px; border-radius:50%; background:#00F5A0; box-shadow:0 0 6px #00F5A0;"></span>
            </div>
            <div>
                <div style="font-size:16px; font-weight:700; color:#ffffff; line-height:1.2;">Double Edge</div>
                <div style="font-size:11px; color:#64748b; font-weight:500;">Stealth Macro Desk</div>
            </div>
        </div>
        <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:16px; padding-bottom:14px; border-bottom:1px solid #1B212D;">
            <div style="background:#0D0F14; border:1px solid #1B212D; border-radius:8px; padding:6px 12px; font-size:11.5px; display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#ffffff; font-weight:600; display:inline-flex; align-items:center; gap:6px;">
                    <span style="width:6px; height:6px; border-radius:50%; background:#00F5A0; box-shadow:0 0 6px #00F5A0;"></span>
                    Desk Active
                </span>
                <span style="color:#64748b; font-family:monospace; font-size:10.5px;">Live</span>
            </div>
            <div style="background:#0D0F14; border:1px solid #1B212D; border-radius:8px; padding:6px 12px; font-size:11.5px; display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#00F5A0; font-weight:600; display:inline-flex; align-items:center; gap:6px;">
                    <span style="width:6px; height:6px; border-radius:50%; background:#00F5A0; box-shadow:0 0 6px #00F5A0;"></span>
                    Gemini Key Active
                </span>
                <span style="color:#64748b; font-size:10.5px;">Ready</span>
            </div>
        </div>
        <div style="font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Data Feeds</div>
    ''', unsafe_allow_html=True)
    if st.button("Sync Feeds Now", use_container_width=True):
        with st.status("Syncing core feeds from Yahoo Finance...", expanded=True) as status:
            from data_manager import sync_tickers_batch
            target_tickers = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS
            results = sync_tickers_batch(tickers=target_tickers, force=True)
            st.cache_data.clear()
            last_date = results[0].get("last_date", "Today") if results else "Today"
            status.update(label=f"Complete! Updated to {last_date}.", state="complete")
        st.toast(f"Feeds Synced! {len(results)} assets updated.")
        time.sleep(0.5)
        st.rerun()

    st.markdown('<div style="font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; margin-top:16px; margin-bottom:6px;">AI Strategist Engine</div>', unsafe_allow_html=True)
    custom_key = st.text_input("Gemini API Key", value=user_api_key, type="password", placeholder="Enter Gemini Key...", label_visibility="collapsed")
    if custom_key:
        user_api_key = custom_key

# ----------------- AUTO-SYNC & LOAD DATA -----------------
if "auto_synced_on_load" not in st.session_state:
    if is_core_data_stale():
        with st.spinner("Checking market data... auto-syncing today's prices from yfinance..."):
            auto_sync_core_data()
    st.session_state["auto_synced_on_load"] = True

@st.cache_data(ttl=300)
def get_market_analysis():
    generate_synthetic_ratios()
    return analyze_market_narrative()

@st.cache_data(ttl=600)
def get_cached_ai_output(_market_data, api_key_val):
    return generate_ai_ideas(_market_data, gemini_api_key=api_key_val)

@st.cache_data(ttl=300)
def get_cached_confluences(tickers_tuple):
    return compute_all_confluences(list(tickers_tuple))

with st.spinner("Analyzing quantitative regimes and technicals..."):
    market_data = get_market_analysis()

macro_reading = market_data.get("macro_context", "Balanced backdrop.")

# Top Banner Data
spy_df = load_ticker_data("SPY")
spy_metrics = compute_all_metrics(spy_df)

qqq_df = load_ticker_data("QQQ")
qqq_metrics = compute_all_metrics(qqq_df)

iwm_df = load_ticker_data("IWM")
iwm_metrics = compute_all_metrics(iwm_df)

macro_assets_data = market_data.get("macro_assets", {})
gld_m = macro_assets_data.get("GLD", {})
uso_m = macro_assets_data.get("USO", {})
tlt_m = macro_assets_data.get("TLT", {})
ibit_m = macro_assets_data.get("IBIT", {})

gld_df = load_ticker_data("GLD")
uso_df = load_ticker_data("USO")
tlt_df = load_ticker_data("TLT")
ibit_df = load_ticker_data("IBIT")

spy_rsp = market_data["ratios"].get("SPY_RSP", {})
iwm_qqq = market_data["ratios"].get("IWM_QQQ", {})

last_market_session = "Today"
if not spy_df.empty and "Date" in spy_df.columns:
    last_market_session = str(spy_df["Date"].iloc[-1]).split()[0]





# ----------------- SECTION 1: CORE EQUITIES & MACRO BENCHMARKS (7 UNIVERSE ASSETS) -----------------
st.markdown('''
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
    <div style="display:flex; align-items:center; gap:6px;">
        <span style="width:6px; height:6px; border-radius:50%; background:#00F5A0; display:inline-block; box-shadow:0 0 6px #00F5A0;"></span>
        <span style="font-size:11px; font-weight:700; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.5px;">Core Equities & Macro Benchmarks</span>
    </div>
    <span style="font-size:11px; color:#64748b;">Session Closes & Momentum • 7 Universe Assets</span>
</div>
''', unsafe_allow_html=True)

def generate_smooth_sparkline(prices, width=56, height=20, pad_x=3, pad_y=4):
    if not prices or len(prices) < 2:
        return "M 3 10 L 49 10", 49, 10
    n = len(prices)
    min_p = min(prices)
    max_p = max(prices)
    rng = max_p - min_p
    
    w_avail = width - pad_x * 2 - 4
    h_avail = height - pad_y * 2
    
    pts = []
    for i in range(n):
        x = pad_x + (i / (n - 1)) * w_avail
        norm = (prices[i] - min_p) / rng if rng > 1e-4 else 0.5
        y = (height - pad_y) - norm * h_avail
        pts.append((round(x, 1), round(y, 1)))
        
    d = f"M {pts[0][0]} {pts[0][1]}"
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else pts[i + 1]
        
        cp1x = round(p1[0] + (p2[0] - p0[0]) / 5.0, 1)
        cp1y = round(p1[1] + (p2[1] - p0[1]) / 5.0, 1)
        cp2x = round(p2[0] - (p3[0] - p1[0]) / 5.0, 1)
        cp2y = round(p2[1] - (p3[1] - p1[1]) / 5.0, 1)
        
        d += f" C {cp1x} {cp1y}, {cp2x} {cp2y}, {p2[0]} {p2[1]}"
        
    end_pt = pts[-1]
    return d, end_pt[0], end_pt[1]

def render_stealth_benchmark_card(ticker, name, price, pct_1d, sub1, sub2, pct_5d=0.0, prices=None):
    is_pos = pct_1d >= 0
    badge_cls = "badge-mint" if is_pos else "badge-cherry"
    sign = "+" if is_pos else ""
    spark_col = "#00F5A0" if pct_5d >= 0 else "#FF2E55"
    if prices and len(prices) >= 2:
        spark_path, spark_end_x, spark_end_y = generate_smooth_sparkline(prices)
    else:
        spark_path = "M 3 15 Q 16 12, 28 8 T 42 10 T 50 6" if pct_5d >= 0 else "M 3 7 Q 15 4, 28 13 T 44 10 T 50 14"
        spark_end_x = 50
        spark_end_y = 6 if pct_5d >= 0 else 14
    return f'''
    <div class="stealth-card" style="padding:8px 10px; display:flex; flex-direction:column; justify-content:space-between; height:100%; overflow:hidden;">
        <div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; gap:2px; overflow:hidden;">
                <span style="font-size:9px; font-weight:700; color:#f1f5f9; white-space:nowrap; letter-spacing:-0.3px; display:inline-flex; align-items:center; gap:2px;" title="{ticker} • {name}">
                    <span>{ticker}</span><span style="font-size:8px; font-weight:normal; color:#94a3b8; letter-spacing:-0.3px;">• {name}</span>
                </span>
                <span class="{badge_cls} tabular-num" style="padding:1px 3px; font-size:8px; border-radius:4px; flex-shrink:0;">{sign}{pct_1d:.2f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="tabular-num" style="font-size:19px; font-weight:800; color:#ffffff;">${price:,.2f}</span>
                <div style="width:52px; height:18px; flex-shrink:0;">
                    <svg viewBox="0 0 56 20" style="width:100%; height:100%; overflow:visible;">
                        <path d="{spark_path}" fill="none" stroke="{spark_col}" stroke-width="1.5" stroke-linecap="round" />
                        <circle cx="{spark_end_x}" cy="{spark_end_y}" r="2" fill="{spark_col}" />
                    </svg>
                </div>
            </div>
        </div>
        <div style="margin-top:8px; padding-top:6px; border-top:1px solid #1B212D; font-size:10.5px; color:#94a3b8; display:flex; flex-direction:column; gap:2px;" class="tabular-num">
            <div style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{sub1}</div>
            <div style="font-size:10px; color:#64748b; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{sub2}</div>
        </div>
    </div>
    '''

b_col1, b_col2, b_col3, b_col4, b_col5, b_col6, b_col7 = st.columns(7)

spy_closes = spy_df['Close'].tail(6).tolist() if not spy_df.empty and 'Close' in spy_df.columns else None
qqq_closes = qqq_df['Close'].tail(6).tolist() if not qqq_df.empty and 'Close' in qqq_df.columns else None
iwm_closes = iwm_df['Close'].tail(6).tolist() if not iwm_df.empty and 'Close' in iwm_df.columns else None
gld_closes = gld_df['Close'].tail(6).tolist() if not gld_df.empty and 'Close' in gld_df.columns else None
tlt_closes = tlt_df['Close'].tail(6).tolist() if not tlt_df.empty and 'Close' in tlt_df.columns else None
ibit_closes = ibit_df['Close'].tail(6).tolist() if not ibit_df.empty and 'Close' in ibit_df.columns else None
uso_closes = uso_df['Close'].tail(6).tolist() if not uso_df.empty and 'Close' in uso_df.columns else None

with b_col1:
    p1d = spy_metrics.get("pct_1d", 0)
    p5d = spy_metrics.get("pct_5d", 0)
    p5_col = "#00F5A0" if p5d >= 0 else "#FF2E55"
    st.markdown(render_stealth_benchmark_card(
        "SPY", "S&P 500", spy_metrics.get("close", 0), p1d,
        f'ATH DD: <span style="color:#FF2E55; font-weight:600;">{spy_metrics.get("drawdown_ath")}%</span> • 5D: <span style="color:{p5_col}; font-weight:600;">{p5d:+.2f}%</span>',
        "Large-Cap Core Benchmark", pct_5d=p5d, prices=spy_closes
    ), unsafe_allow_html=True)

with b_col2:
    p1d = qqq_metrics.get("pct_1d", 0)
    p5d = qqq_metrics.get("pct_5d", 0)
    p5_col = "#00F5A0" if p5d >= 0 else "#FF2E55"
    st.markdown(render_stealth_benchmark_card(
        "QQQ", "Nasdaq 100", qqq_metrics.get("close", 0), p1d,
        f'ATH DD: <span style="color:#FF2E55; font-weight:600;">{qqq_metrics.get("drawdown_ath")}%</span> • 5D: <span style="color:{p5_col}; font-weight:600;">{p5d:+.2f}%</span>',
        "Tech Growth Benchmark", pct_5d=p5d, prices=qqq_closes
    ), unsafe_allow_html=True)

with b_col3:
    p1d = iwm_metrics.get("pct_1d", 0)
    p5d = iwm_metrics.get("pct_5d", 0)
    p5_col = "#00F5A0" if p5d >= 0 else "#FF2E55"
    st.markdown(render_stealth_benchmark_card(
        "IWM", "Russell 2000", iwm_metrics.get("close", 0), p1d,
        f'ATH DD: <span style="color:#FF2E55; font-weight:600;">{iwm_metrics.get("drawdown_ath")}%</span> • 5D: <span style="color:{p5_col}; font-weight:600;">{p5d:+.2f}%</span>',
        "Small-Cap Value Benchmark", pct_5d=p5d, prices=iwm_closes
    ), unsafe_allow_html=True)

with b_col4:
    p1d = gld_m.get("pct_1d", 0)
    p5d = gld_m.get("pct_5d", 0)
    p20d = gld_m.get("pct_20d", 0)
    p5_col = "#00F5A0" if p5d >= 0 else "#FF2E55"
    st.markdown(render_stealth_benchmark_card(
        "GLD", "Gold Trust", gld_m.get("close", 0), p1d,
        f'5D: <span style="color:{p5_col}; font-weight:600;">{p5d:+.2f}%</span> • 20D: <span style="color:#94a3b8;">{p20d:+.2f}%</span>',
        "Safe Haven • Debasement", pct_5d=p5d, prices=gld_closes
    ), unsafe_allow_html=True)

with b_col5:
    p1d = tlt_m.get("pct_1d", 0)
    p5d = tlt_m.get("pct_5d", 0)
    p20d = tlt_m.get("pct_20d", 0)
    p5_col = "#00F5A0" if p5d >= 0 else "#FF2E55"
    st.markdown(render_stealth_benchmark_card(
        "TLT", "20Y Treasury", tlt_m.get("close", 0), p1d,
        f'5D: <span style="color:{p5_col}; font-weight:600;">{p5d:+.2f}%</span> • 20D: <span style="color:#94a3b8;">{p20d:+.2f}%</span>',
        "Duration • Real Yield Pressure", pct_5d=p5d, prices=tlt_closes
    ), unsafe_allow_html=True)

with b_col6:
    p1d = ibit_m.get("pct_1d", 0)
    p5d = ibit_m.get("pct_5d", 0)
    p20d = ibit_m.get("pct_20d", 0)
    p5_col = "#00F5A0" if p5d >= 0 else "#FF2E55"
    st.markdown(render_stealth_benchmark_card(
        "IBIT", "Bitcoin Trust", ibit_m.get("close", 0), p1d,
        f'5D: <span style="color:{p5_col}; font-weight:600;">{p5d:+.2f}%</span> • 20D: <span style="color:#94a3b8;">{p20d:+.2f}%</span>',
        "High-Beta Liquidity Barometer", pct_5d=p5d, prices=ibit_closes
    ), unsafe_allow_html=True)

with b_col7:
    p1d = uso_m.get("pct_1d", 0)
    p5d = uso_m.get("pct_5d", 0)
    p20d = uso_m.get("pct_20d", 0)
    p5_col = "#00F5A0" if p5d >= 0 else "#FF2E55"
    st.markdown(render_stealth_benchmark_card(
        "USO", "Crude Oil", uso_m.get("close", 0), p1d,
        f'5D: <span style="color:{p5_col}; font-weight:600;">{p5d:+.2f}%</span> • 20D: <span style="color:#94a3b8;">{p20d:+.2f}%</span>',
        "Energy & Inflation Input", pct_5d=p5d, prices=uso_closes
    ), unsafe_allow_html=True)

# ----------------- SECTION 2: QUANTITATIVE REGIME & BREADTH RATIOS (3 DEDICATED PAIRS) -----------------
st.markdown('''
<div style="margin-top:14px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
    <div style="display:flex; align-items:center; gap:6px;">
        <span style="width:6px; height:6px; border-radius:50%; background:#FF2E55; display:inline-block; box-shadow:0 0 6px #FF2E55;"></span>
        <span style="font-size:11px; font-weight:700; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.5px;">Quantitative Regime & Breadth Ratios</span>
        <span style="padding:1px 6px; border-radius:4px; background:#131720; font-size:10px; color:#94a3b8; border:1px solid #1B212D;">3 Pairs Monitored</span>
    </div>
    <span style="font-size:11px; color:#64748b;">50-Day Z-Score & Relative Velocity</span>
</div>
''', unsafe_allow_html=True)

qqq_rsp = market_data["ratios"].get("QQQ_RSP", {})

def render_stealth_ratio_card(ticker, name, subtitle, val, z_val, state, pct5d, action_text):
    is_z_pos = z_val >= 0
    badge_cls = "badge-mint" if is_z_pos else "badge-cherry"
    needle_col = "#00F5A0" if is_z_pos else "#FF2E55"
    needle_pct = max(5.0, min(95.0, ((z_val + 3.0) / 6.0) * 100))
    trend_col = "#00F5A0" if pct5d >= 0 else "#FF2E55"
    sign_5d = "+" if pct5d >= 0 else ""
    
    st_lower = state.lower()
    if any(k in st_lower for k in ["range", "neutral", "consolidat"]):
        pill_style = "color:#cbd5e1; background:#131720; border:1px solid #1B212D;"
    elif any(k in st_lower for k in ["risk-off", "bear", "narrow", "contract", "oversold", "washout", "overextend", "peaking"]):
        pill_style = "color:#FF2E55; background:rgba(255,46,85,0.1); border:1px solid rgba(255,46,85,0.25);"
    else:
        pill_style = "color:#00F5A0; background:rgba(0,245,160,0.1); border:1px solid rgba(0,245,160,0.25);"
        
    return f'''
    <div class="stealth-card" style="padding:14px; display:flex; flex-direction:column; justify-content:space-between; height:100%;">
        <div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:12px; font-weight:700; color:#ffffff;">{ticker} <span style="font-size:11px; color:#64748b; font-weight:normal;">• {name}</span></div>
                    <div style="font-size:11px; color:#64748b;">{subtitle}</div>
                </div>
                <span class="{badge_cls} tabular-num" style="font-size:12px; font-weight:700; padding:2px 8px; border-radius:6px;">{z_val:+.2f}σ</span>
            </div>
            
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:8px;">
                <span class="tabular-num" style="font-size:24px; font-weight:800; color:#ffffff;">{val:.2f}</span>
                <span style="font-size:11px; font-weight:600; padding:2px 8px; border-radius:6px; {pill_style}">{state}</span>
            </div>
            
            <!-- Z-Score Gauge Spectrum Meter (-3σ to +3σ) -->
            <div style="margin-top:8px;">
                <div style="display:flex; justify-content:space-between; font-size:10px; color:#64748b; font-family:monospace;">
                    <span>-3.0σ (Oversold)</span>
                    <span>0.0σ</span>
                    <span>+3.0σ (Extended)</span>
                </div>
                <div style="width:100%; height:6px; border-radius:9999px; background:#181E2A; position:relative; overflow:hidden; margin-top:2px;">
                    <div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:#475569;"></div>
                    <div style="position:absolute; top:0; bottom:0; left:{needle_pct:.1f}%; width:8px; margin-left:-4px; background:{needle_col}; border-radius:9999px; box-shadow:0 0 6px {needle_col};"></div>
                </div>
            </div>
        </div>
        
        <div style="margin-top:10px; padding-top:6px; border-top:1px solid #1B212D; font-size:11px; color:#94a3b8; display:flex; justify-content:space-between; align-items:center;" class="tabular-num">
            <div>5D Trend: <span style="color:{trend_col}; font-weight:600;">{sign_5d}{pct5d:.2f}%</span></div>
            <div style="font-size:11px; color:#64748b;">{action_text}</div>
        </div>
    </div>
    '''

r_col1, r_col2, r_col3 = st.columns(3)

with r_col1:
    st.markdown(render_stealth_ratio_card(
        "SPY / RSP", "Market Breadth", "Cap-Weighted vs Equal-Weight",
        spy_rsp.get("current", 0),
        spy_rsp.get("z_score_50d", 0),
        spy_rsp.get("actionable_state", "Broad Breadth Expansion"),
        spy_rsp.get("pct_5d", 0),
        spy_rsp.get("action_text", "Equal-Weight RSP Outperforming")
    ), unsafe_allow_html=True)

with r_col2:
    st.markdown(render_stealth_ratio_card(
        "QQQ / RSP", "Tech Breadth", "Tech Beta vs Broad Market",
        qqq_rsp.get("current", 0),
        qqq_rsp.get("z_score_50d", 0),
        qqq_rsp.get("actionable_state", "Broadening Beyond Mega-Tech"),
        qqq_rsp.get("pct_5d", 0),
        qqq_rsp.get("action_text", "Broadening Beyond Mega-Tech")
    ), unsafe_allow_html=True)

with r_col3:
    st.markdown(render_stealth_ratio_card(
        "IWM / QQQ", "Small vs Tech", "Small-Cap vs Mega-Cap Tech",
        iwm_qqq.get("current", 0),
        iwm_qqq.get("z_score_50d", 0),
        iwm_qqq.get("actionable_state", "Aggressive Small-Cap Momentum"),
        iwm_qqq.get("pct_5d", 0),
        iwm_qqq.get("action_text", "Small-Cap Alpha Momentum")
    ), unsafe_allow_html=True)

st.caption(f"**Cross-Asset Macro Barometer Reading:** {macro_reading}")
st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

# ----------------- MODERN TABS (Institutional Options Desk v3.0 Full Suite) -----------------
tab1, tab_agent, tab2, tab3, tab5, tab4, tab6, tab_options = st.tabs([
    "AI Macro Letter",
    "AI Agent",
    "Automated Trade Ideas",
    "Bias Matrix",
    "Indexes & Sector Flow",
    "Magnificant 7 Flow",
    "Synthetic Ratio Rotations",
    "Volatility Pricing"
])

# ================= TAB 1: AI MACRO LETTER =================
with tab1:
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.subheader("Daily Institutional Macro Letter & AI Analysis")
        st.caption("AI-Powered Cross-Asset Liquidity & Market Regime Intelligence")
    with col_t2:
        refresh_ai = st.button("Generate Fresh Briefing", use_container_width=True)

    if refresh_ai:
        st.cache_data.clear()

    ai_output = get_cached_ai_output(market_data, user_api_key)

    # 1. Regime & Capital Flow Grid (2 Compact Cards)
    col_reg, col_flow = st.columns(2)
    with col_reg:
        regime_title = ai_output.get("regime_title") or market_data.get("narrative_title", "Secular Uptrend")
        regime_detail = ai_output.get("regime_detail") or market_data.get("narrative_detail", "")
        risk_level = ai_output.get("risk_level") or market_data.get("risk_level", "Moderate")
        is_ai = ai_output.get("is_ai_derived", False)
        ai_badge = '<span class="pill-amber" style="font-size:10px; margin-left:6px; padding:2px 6px;">AI Synthesized</span>' if is_ai else ''

        reg_card_html = (
            '<div class="metric-card-lux" style="height:100%; border-top:3px solid #f59e0b;">'
            '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
            f'<div><span style="font-size:11px; font-weight:700; color:#f59e0b; text-transform:uppercase; letter-spacing:0.5px;">Market Regime</span>{ai_badge}</div>'
            f'<span class="pill-amber">{risk_level}</span>'
            '</div>'
            f'<div style="font-size:20px; font-weight:800; color:#ffffff; margin-bottom:6px;">{regime_title}</div>'
            f'<div style="font-size:13px; color:#cbd5e1; line-height:1.45;">{regime_detail}</div>'
            '</div>'
        )
        st.markdown(reg_card_html, unsafe_allow_html=True)

    with col_flow:
        baskets = ai_output.get("baskets", {})
        g_val = baskets.get("Growth", baskets.get("Growth (XLK, SMH, XLC, XLY)", baskets.get("Growth (XLK, XLC, XLY)", 0.0)))
        c_val = baskets.get("Cyclical", baskets.get("Cyclicals (XLF, XLI, XLB)", baskets.get("Cyclicals (XLI, XLF, XME)", 0.0)))
        d_val = baskets.get("Defensive", baskets.get("Defensives (XLU, XLP, XLV)", 0.0))
        e_val = baskets.get("Energy", baskets.get("Energy (XLE)", 0.0))

        def fmt_b(val):
            col_b = "#00F5A0" if val >= 0 else "#FF2E55"
            sign_b = "+" if val >= 0 else ""
            return f'<span style="font-size:16px; font-weight:800; color:{col_b};">{sign_b}{val:.2f}%</span>'

        flow_card_html = (
            '<div class="metric-card-lux" style="height:100%; border-top:3px solid #00F5A0;">'
            '<div style="font-size:11px; font-weight:700; color:#00F5A0; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px;">5-Day Sector Velocity Baskets</div>'
            '<div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:8px; text-align:center;">'
            f'<div><div style="font-size:11px; color:#64748b;">Growth</div>{fmt_b(g_val)}</div>'
            f'<div><div style="font-size:11px; color:#64748b;">Cyclical</div>{fmt_b(c_val)}</div>'
            f'<div><div style="font-size:11px; color:#64748b;">Defensive</div>{fmt_b(d_val)}</div>'
            f'<div><div style="font-size:11px; color:#64748b;">Energy</div>{fmt_b(e_val)}</div>'
            '</div>'
            '<div style="font-size:12px; color:#94a3b8; margin-top:12px; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">'
            '<b>Capital Rotation:</b> Measures 5-day velocity across cyclical vs defensive balance sheets.'
            '</div>'
            '</div>'
        )
        st.markdown(flow_card_html, unsafe_allow_html=True)

    # 2. Comprehensive Daily AI Analysis Report
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    llm_summary = ai_output.get("llm_summary")
    engine_source = ai_output.get("source", "Quantitative Confluence Desk")

    if llm_summary:
        st.markdown(
            f'<div class="metric-card-lux" style="border:1px solid rgba(0, 245, 160, 0.35); background:linear-gradient(180deg, #13161f 0%, #0d0e13 100%); margin-bottom:14px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px;">'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span style="font-size:13px; font-weight:800; color:#00F5A0; text-transform:uppercase; letter-spacing:0.5px;">Institutional Daily Market Intelligence Briefing</span>'
            f'</div>'
            f'<span class="pill-green" style="font-size:11px; font-weight:700;">{engine_source}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        import re
        chunks = re.split(r'(?:\n\s*---\s*\n|\n+)(?=###?\s+[0-9IVX]+\.)', '\n' + llm_summary)
        sections = [c.strip() for c in chunks if c.strip() and not c.strip().startswith('# DAILY') and not c.strip().startswith('**Date')]
        
        if len(sections) >= 4:
            p_tabs = st.tabs(["1. Macro Liquidity (AI)", "2. Sector Velocity (AI)", "3. Setups & Traps (AI)", "4. Desk Game Plan (AI)"])
            for idx, tab_obj in enumerate(p_tabs):
                with tab_obj:
                    st.markdown(sections[idx])
        else:
            st.markdown(llm_summary)
    else:
        synth_html = (
            f'<div class="metric-card-lux" style="border:1px solid rgba(0, 245, 160, 0.3);">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">'
            f'<span style="font-size:12px; font-weight:700; color:#00F5A0; text-transform:uppercase; letter-spacing:0.5px;">Institutional Executive Synthesis</span>'
            f'<span class="pill-green" style="font-size:11px;">{engine_source}</span>'
            f'</div>'
            f'<div style="font-size:13.5px; color:#cbd5e1; line-height:1.6;">'
            f'<div style="margin-bottom:8px;"><b style="color:#ffffff;">• Regime Mechanics:</b> <span style="color:#94a3b8;">{regime_detail}</span></div>'
            f'<div style="margin-bottom:8px;"><b style="color:#ffffff;">• Cross-Asset Flow:</b> <span style="color:#94a3b8;">{macro_reading}</span></div>'
            f'<div style="margin-bottom:8px;"><b style="color:#FF2E55;">• Desk Risk Warning:</b> <span style="color:#94a3b8;">Monitor mega-cap breadth concentration (SPY/RSP at 5-year highs). Avoid front-running unconfirmed dip buys on broken distribution names.</span></div>'
            f'</div>'
            f'<div style="margin-top:12px; font-size:12px; color:#64748b; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">'
            f'<i>Tip: Ensure your Gemini API Key is active in the sidebar to generate full daily macro briefings and activate the AI Agent.</i>'
            f'</div>'
            f'</div>'
        )
        st.markdown(synth_html, unsafe_allow_html=True)


# ================= TAB: INTERACTIVE AI QUANT AGENT =================
with tab_agent:
    col_a1, col_a2 = st.columns([3, 1])
    with col_a1:
        st.subheader("Interactive Quant Analyst & Risk Officer")
        st.caption("On-Demand AI Reasoning Across Live Regimes, Tickers, Confluences & Ratios")
    with col_a2:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state["agent_chat"] = []
            st.rerun()

    if "agent_chat" not in st.session_state:
        st.session_state["agent_chat"] = [
            {
                "role": "assistant",
                "content": f"**Greetings Trader.** I am your Quantitative Desk Analyst. I have fully indexed today's market metrics ({last_market_session}), including 20/50/200 EMAs, RSI, ATR stretches, RVOL, sector velocities, and confluence scores for the entire core universe.\n\nAsk me about any ticker, rotation setup, or market risk, or click one of the quick prompts below."
            }
        ]

    # Quick Prompts Row
    st.markdown("<div style='font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; margin-bottom:6px;'>Quick Desk Inquiries:</div>", unsafe_allow_html=True)
    qp1, qp2, qp3, qp4 = st.columns(4)
    prompt_clicked = None
    with qp1:
        if st.button("Mag 7 Divergence & Risks", use_container_width=True):
            prompt_clicked = "Analyze today's Mag 7 divergence. Which tech leaders are showing institutional accumulation vs distribution, and what are the key invalidation levels?"
    with qp2:
        if st.button("Sector Rotation Leaders", use_container_width=True):
            prompt_clicked = "Based on the 5-day sector velocities, where is institutional smart money rotating right now? Which sector is leading?"
    with qp3:
        if st.button("Top Distribution Warnings", use_container_width=True):
            prompt_clicked = "What are the biggest distribution traps and high-risk names on the desk today? Where should retail traders avoid dip buying?"
    with qp4:
        if st.button("Best High-Conviction Setups", use_container_width=True):
            prompt_clicked = "Give me the top 3 high-probability setups from today's confluence radar with exact entry rationale, targets, and stop-loss levels."

    # Render Chat History
    for msg in st.session_state["agent_chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    user_input = st.chat_input("Ask the Quant AI Agent anything about today's data, tickers, or setups...")
    active_prompt = prompt_clicked or user_input

    if active_prompt:
        st.session_state["agent_chat"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI Quant Agent analyzing live quantitative data..."):
                confluence_items = get_cached_confluences(tuple(MAG_7 + INDEXES + SECTORS + MACRO_ASSETS))
                response_text = query_ai_agent(
                    user_query=active_prompt,
                    market_data=market_data,
                    confluences=confluence_items,
                    gemini_api_key=user_api_key,
                    chat_history=st.session_state["agent_chat"]
                )
                st.markdown(response_text)
                st.session_state["agent_chat"].append({"role": "assistant", "content": response_text})

# ================= TAB 2: AUTOMATED TRADE IDEAS =================
with tab2:
    st.subheader("Algorithmic High-Probability Trade Setups")
    st.caption("Filtered by 7-Layer Confluence and 20-Year Historical Backtest Odds")

    st.markdown(
        '<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px 14px; margin-bottom:18px; font-size:12.5px; color:#94a3b8;">'
        '<b style="color:#00F5A0;">Strict Confluence Rules:</b> '
        'Green conviction scores (+15 to +100) are reserved strictly for <b>Long Setups</b>. '
        'Red conviction scores (-15 to -100) are strictly flagged for <b>Distribution Fades or De-risking</b> to protect portfolio equity. '
        'Orderly dip buys require low relative volume and intact moving average support.'
        '</div>',
        unsafe_allow_html=True
    )

    trade_cards = ai_output.get("trade_cards", [])
    if not trade_cards:
        st.info("No high-confluence setups detected in current market conditions.")
    else:
        def render_compact_trade_card_html(t):
            direction = t.get("direction", "LONG")
            is_long = "LONG" in direction or "DIP" in direction
            border_col = "#00F5A0" if is_long else "#FF2E55"
            pill_type = "green" if is_long else "red"
            
            exec_state = t.get("execution_state", "")
            is_caution = any(k in exec_state.upper() for k in ["EXTENDED", "TRAP", "DO NOT", "AWAIT", "FADE", "RISK"])
            exec_pill_type = "red" if is_caution else "green"
            
            curr_px = t.get("current_price", t.get("entry"))
            ext_metric = t.get("extension_metric", "At Support")
            ext_col = "#FF2E55" if any(k in ext_metric for k in ["-", "EXTENDED"]) else "#00F5A0"
            
            rr = t.get("risk_reward", "1 : 3.0 R:R")
            payoff_pill = "green" if is_long else "red"
            
            limit_zone = t.get("optimal_entry_zone", t.get("entry"))
            opt_label = t.get("optimal_entry_label", "20 EMA Support Zone")
            trigger = t.get("breakout_trigger", "Above 5D Pivot")
            exec_desc = t.get("execution_desc", "Bidding structural support band rather than market buying.")

            entry_bg = "rgba(0, 245, 160, 0.06)" if is_long else "rgba(255, 46, 85, 0.06)"
            entry_border = "rgba(0, 245, 160, 0.25)" if is_long else "rgba(255, 46, 85, 0.25)"
            entry_col = "#00F5A0" if is_long else "#FF2E55"

            return (
                f'<div class="metric-card-lux" style="border-left:4px solid {border_col}; padding:10px 12px; margin-bottom:12px; display:flex; flex-direction:column; justify-content:space-between; height:100%; box-sizing:border-box;">'
                f'<div>'
                f'<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px; margin-bottom:8px; gap:6px; flex-wrap:wrap;">'
                f'<div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap;">'
                f'<span style="font-size:18px; font-weight:800; color:#ffffff; letter-spacing:-0.4px;">{t["ticker"]}</span>'
                f'<span class="pill-{pill_type}" style="font-size:10px; padding:1px 6px; font-weight:700;">{t["direction"]}</span>'
                f'<span style="font-size:11px; color:#94a3b8; font-weight:600;">{t["type_label"]}</span>'
                f'{(f"<span class=\"pill-{exec_pill_type}\" style=\"font-size:10px; padding:1px 6px; font-weight:700;\">{exec_state}</span>") if exec_state else ""}'
                f'</div>'
                f'<div style="display:flex; align-items:center; gap:4px; font-size:10px;">'
                f'<span style="color:#64748b; text-transform:uppercase;">Conviction:</span>'
                f'<span class="pill-{pill_type}" style="font-size:10.5px; font-weight:700; padding:1px 6px;">Score: {t["score"]} ({t["bias_label"]})</span>'
                f'</div>'
                f'</div>'
                # Market Price vs Optimized R:R Banner
                f'<div style="display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:6px; padding:4px 8px; margin-bottom:8px; font-size:11px;">'
                f'<div style="display:flex; align-items:center; gap:4px; flex-wrap:wrap;">'
                f'<span style="color:#94a3b8;">Print:</span> <b style="color:#ffffff;">{curr_px}</b>'
                f'<span style="color:#475569; margin:0 2px;">|</span>'
                f'<span style="color:#94a3b8;">20 EMA Ext:</span> <b style="color:{ext_col};">{ext_metric}</b>'
                f'</div>'
                f'<div style="display:flex; align-items:center; gap:4px;">'
                f'<span style="font-size:9.5px; color:#64748b; text-transform:uppercase;">Payoff:</span>'
                f'<span class="pill-{payoff_pill}" style="font-size:10.5px; font-weight:800; padding:1px 5px;">{rr}</span>'
                f'</div>'
                f'</div>'
                # 4-Box Execution Strategy 2x2 Grid
                f'<div style="display:grid; grid-template-columns:repeat(2, 1fr); gap:6px; margin-bottom:8px;">'
                f'<div style="background:{entry_bg}; padding:6px 8px; border-radius:6px; border:1px solid {entry_border};">'
                f'<div style="font-size:9px; font-weight:700; color:{entry_col}; text-transform:uppercase; letter-spacing:0.3px;">Optimal Limit Entry</div>'
                f'<div style="font-size:13.5px; font-weight:800; color:#ffffff; margin-top:1px;">{limit_zone}</div>'
                f'<div style="font-size:9px; color:#94a3b8; margin-top:1px;">{opt_label}</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.02); padding:6px 8px; border-radius:6px; border:1px solid rgba(255,255,255,0.06);">'
                f'<div style="font-size:9px; font-weight:700; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.3px;">Breakout Trigger</div>'
                f'<div style="font-size:12.5px; font-weight:700; color:#ffffff; margin-top:1px;">{trigger}</div>'
                f'<div style="font-size:9px; color:#94a3b8; margin-top:1px;">Momentum Confirmation</div>'
                f'</div>'
                f'<div style="background:rgba(255, 46, 85, 0.06); padding:6px 8px; border-radius:6px; border:1px solid rgba(255, 46, 85, 0.25);">'
                f'<div style="font-size:9px; font-weight:700; color:#FF2E55; text-transform:uppercase; letter-spacing:0.3px;">Stop Invalidation</div>'
                f'<div style="font-size:13.5px; font-weight:800; color:#FF2E55; margin-top:1px;">{t["stop"]} <span style="font-size:10px; font-weight:normal; color:#fda4af;">({t["stop_pct"]})</span></div>'
                f'<div style="font-size:9px; color:#fda4af; margin-top:1px;">Below Structural Support</div>'
                f'</div>'
                f'<div style="background:rgba(0, 245, 160, 0.06); padding:6px 8px; border-radius:6px; border:1px solid rgba(0, 245, 160, 0.25);">'
                f'<div style="font-size:9px; font-weight:700; color:#00F5A0; text-transform:uppercase; letter-spacing:0.3px;">Profit Targets</div>'
                f'<div style="font-size:13.5px; font-weight:800; color:#00F5A0; margin-top:1px;">{t["target1"]} <span style="font-size:10px; font-weight:normal; color:#6ee7b7;">({t.get("target1_pct", "")})</span></div>'
                f'<div style="font-size:9px; color:#6ee7b7; margin-top:1px;">Runner: {t["target2"]} ({t["target2_pct"]})</div>'
                f'</div>'
                f'</div>'
                # Directive tip
                f'<div style="background:rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.05); border-radius:6px; padding:5px 8px; font-size:10.5px; color:#cbd5e1; margin-bottom:8px; line-height:1.35;">'
                f'<b style="color:{border_col};">Desk Directive:</b> {exec_desc}'
                f'</div>'
                f'</div>'
                # Stats Bar and Rationale
                f'<div>'
                f'<div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:6px; padding:5px 8px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:4px; font-size:10.5px;">'
                f'<span style="color:#94a3b8;">Win Rate: <b style="color:#00F5A0;">{t["win_rate"]}</b> <span style="color:#64748b;">(N={t["sample_size"]})</span></span>'
                f'<span style="color:#94a3b8;">Alpha: <b style="color:{"#00F5A0" if is_long else "#FF2E55"};">{t["median_alpha"]}</b></span>'
                f'<span style="color:#cbd5e1;"><b>Confirmations:</b> {t["confirmations"]}</span>'
                f'</div>'
                f'<div style="font-size:11px; color:#cbd5e1; margin-top:6px; line-height:1.35;">'
                f'<b style="color:#ffffff;">Desk Rationale:</b> {t["thesis"]}'
                f'</div>'
                f'</div>'
                f'</div>'
            )

        # Render trade cards side by side in pairs of 2
        for i in range(0, len(trade_cards), 2):
            cols = st.columns(2)
            with cols[0]:
                st.markdown(render_compact_trade_card_html(trade_cards[i]), unsafe_allow_html=True)
            if i + 1 < len(trade_cards):
                with cols[1]:
                    st.markdown(render_compact_trade_card_html(trade_cards[i + 1]), unsafe_allow_html=True)

    with st.expander("Quantitative Desk Rules: How We Evaluate 'Buy the Dip' Opportunities on Red Days"):
        st.markdown(
            '<div style="font-size:13px; color:#cbd5e1; line-height:1.6;">'
            '<p>A stock being down on the day does <b>not</b> automatically disqualify it from being an institutional buy opportunity. However, it must meet 4 strict mathematical criteria to prevent buying into a structural breakdown (like META):</p>'
            '<ol>'
            '<li><b style="color:#ffffff;">Secular Trend Intact:</b> Price must remain above the 50-day or 200-day moving average. Downward action must be an orderly retracement, not a trend reversal.</li>'
            '<li><b style="color:#ffffff;">Volume Exhaustion (RVOL < 1.05x):</b> Selling volume must be low and diminishing. Elevated volume (like META\'s 1.7x RVOL) indicates aggressive institutional liquidation and invalidates dip buying.</li>'
            '<li><b style="color:#ffffff;">Statistical Rubber-Band Stretch:</b> The asset must be stretched by > 1.8 ATR from its 20 EMA, or RSI must be approaching oversold territory (< 38).</li>'
            '<li><b style="color:#ffffff;">20-Year Historical Odds:</b> The asset\'s exact volatility and moving average configuration must have produced a 20-year backtested win rate > 65%.</li>'
            '</ol>'
            '</div>',
            unsafe_allow_html=True
        )

# ================= TAB 3: BIAS MATRIX =================
with tab3:
    st.subheader("Multi-Factor Bias Matrix & Confirmation Radar")
    st.markdown("""
    Instead of rigid binary rules, each asset is scored across **7 independent confirmation layers** 
    (Structural Trend, Value Pullback/Extension Zone, Volume Flow Conviction, Momentum/Divergence, Realized Vol Squeeze, Macro Flow, and **20-Year Historical Backtest Odds**).
    **Score Range:** **-100 (Deep Bearish Red)** to **+100 (Deep Bullish Green)**. 
    As independent confirmations accumulate, the conviction shifts from a faint flicker into high-intensity confirmation.
    """)

    # Select Universe for Confluence
    confluence_scope = st.radio(
        "Universe View",
        [
            "Core Universe (Mag 7 + Indexes + Sectors + Macro)",
            "Mag 7 Only",
            "Major Indexes Only",
            "Sector ETFs Only",
            "Macro Assets (GLD, USO, TLT, IBIT)"
        ],
        horizontal=True
    )

    if confluence_scope == "Mag 7 Only":
        target_universe = MAG_7
    elif confluence_scope == "Major Indexes Only":
        target_universe = INDEXES
    elif confluence_scope == "Sector ETFs Only":
        target_universe = SECTORS
    elif "Macro Assets" in confluence_scope:
        target_universe = MACRO_ASSETS
    else:
        target_universe = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS

    confluence_items = get_cached_confluences(tuple(target_universe))

    # 1. Visual Intensity Bar Chart (Plotly)
    st.markdown("#### Conviction Intensity Spectrum (-100 Bearish to +100 Bullish)")
    tickers_list = [c["ticker"] for c in confluence_items]
    scores_list = [c["score"] for c in confluence_items]
    colors_list = [c["color_hex"] for c in confluence_items]
    hover_texts = [f"{c['ticker']}: {c['bias_label']} ({c['score']})<br>Bull Confirmations: {c['bull_count']}/7<br>Bear Confirmations: {c['bear_count']}/7<br>{c['flicker_status']}" for c in confluence_items]

    fig_conf = go.Figure()
    fig_conf.add_trace(go.Bar(
        x=scores_list,
        y=tickers_list,
        orientation="h",
        marker=dict(color=colors_list),
        hovertext=hover_texts,
        hoverinfo="text"
    ))
    apply_dark_theme(fig_conf, height=max(380, len(tickers_list) * 26))
    fig_conf.update_layout(
        xaxis=dict(range=[-100, 100], title="Conviction Score (-100 Strong Red to +100 Strong Green)", zeroline=True, zerolinewidth=2, zerolinecolor="#787b86"),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # 2. Confluence Matrix Table
    st.markdown("#### Confluence Confirmation Matrix")
    conf_rows = []
    for c in confluence_items:
        m = c["metrics"]
        edge = m.get("edge_match", {})
        conf_rows.append({
            "Ticker": c["ticker"],
            "Price ($)": m.get("close"),
            "Conviction Score": c["score"],
            "Conviction Bias": c["bias_label"],
            "Bull Confirmations": f"{c['bull_count']} / 7",
            "Bear Confirmations": f"{c['bear_count']} / 7",
            "Flicker / Momentum Status": c["flicker_status"],
            "Hist. 20D Win Rate (%)": f"{edge.get('win_rate_20d', 0)}% (N={edge.get('sample_size', 0)})",
            "14D RSI": m.get("rsi"),
            "RVOL": f"{m.get('rvol')}x",
            "vs 50EMA (%)": f"{m.get('dist_ema50', 0):+.2f}%",
            "Vol Squeeze Ratio": m.get("vol_ratio"),
            "Drawdown ATH (%)": f"{m.get('drawdown_ath')}%"
        })
    conf_df = pd.DataFrame(conf_rows)
    st.dataframe(
        conf_df.style.background_gradient(subset=["Conviction Score"], cmap="RdYlGn", vmin=-80, vmax=80),
        use_container_width=True,
        height=320
    )

    # 3. Deep 7-Layer Diagnostic Breakdown for Selected Asset
    st.markdown("---")
    st.markdown("#### Asset Confirmation Layer Inspector")
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        inspector_ticker = st.selectbox("Inspect Asset Layers", tickers_list, index=0)
    
    inspect_item = next((item for item in confluence_items if item["ticker"] == inspector_ticker), None)
    if inspect_item:
        with col_sel2:
            st.markdown(f"**{inspector_ticker} Conviction:** `{inspect_item['score']}` | Status: **{inspect_item['bias_label']}** ({inspect_item['flicker_status']})")
        
        st.markdown(f"**Active Confirmations:** `{inspect_item['bull_count']} of 7 Bullish` | `{inspect_item['bear_count']} of 7 Bearish`")

        # Display 7 Layers in Clean Luxury Cards
        checklist = inspect_item["checklist"]
        for layer in checklist:
            status = layer["status"]
            pts = int(layer.get("points", 0))
            pts_color = "#00F5A0" if pts > 0 else ("#FF2E55" if pts < 0 else "#94a3b8")
            if status == "PASS_BULL":
                badge = "[BULLISH CONFIRMATION]"
                box_color = "rgba(0, 245, 160, 0.08)"
                border_color = "#00F5A0"
            elif status == "PASS_BEAR":
                badge = "[BEARISH CONFIRMATION]"
                box_color = "rgba(255, 46, 85, 0.08)"
                border_color = "#FF2E55"
            else:
                badge = "[NEUTRAL / INACTIVE]"
                box_color = "#111319"
                border_color = "rgba(255, 255, 255, 0.12)"

            layer_card_html = (
                f'<div style="background-color: {box_color}; border-left: 4px solid {border_color}; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; border-top: 1px solid rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.04); border-bottom: 1px solid rgba(255,255,255,0.04);">'
                f'<span style="font-weight: 700; font-size: 14px; color: #ffffff;">{layer["layer"]}</span> &nbsp;'
                f'<code style="font-size: 12px; background: rgba(0,0,0,0.3);">{badge}</code> &nbsp;'
                f'<span style="float: right; font-weight: 700; color: {pts_color};">{layer["points"]} pts</span>'
                f'<p style="margin: 6px 0 0 0; font-size: 13px; color: #94a3b8;">{layer["detail"]}</p>'
                f'</div>'
            )
            st.markdown(layer_card_html, unsafe_allow_html=True)

# ================= TAB 4: MAGNIFICANT 7 FLOW =================
with tab4:
    st.subheader("Magnificant 7 Flow (Technicals, Realized Vol, Drawdowns & Odds)")
    st.markdown("Every mega-cap analyzed for **Realized Volatility, Pullback Depths, RVOL, and 20-Day Historical Forward Odds**.")

    mag7_data = market_data.get("mag7", {})
    mag7_rows = []
    for ticker, d in mag7_data.items():
        edge = d.get("edge_match", {})
        mag7_rows.append({
            "Ticker": ticker,
            "Price ($)": d.get("close"),
            "1D %": d.get("pct_1d"),
            "5D %": d.get("pct_5d"),
            "20D %": d.get("pct_20d"),
            "vs 20EMA %": d.get("dist_ema20"),
            "vs 50EMA %": d.get("dist_ema50"),
            "14D RSI": d.get("rsi"),
            "RSI Divergence": d.get("rsi_divergence"),
            "RVOL": d.get("rvol"),
            "20D Realized Vol (%)": d.get("rv20"),
            "60D Realized Vol (%)": d.get("rv60"),
            "Vol Ratio (20/60)": d.get("vol_ratio"),
            "Vol Regime": d.get("vol_status"),
            "Drawdown ATH (%)": d.get("drawdown_ath"),
            "Days in Drawdown": d.get("days_in_drawdown"),
            "20D Hist. Win Rate (%)": edge.get("win_rate_20d", 0),
            "Median 20D Alpha (%)": edge.get("median_return_20d", 0),
            "Technical State": d.get("tech_status")
        })

    mag7_df = pd.DataFrame(mag7_rows)
    if not mag7_df.empty:
        st.dataframe(
            mag7_df.style.format({
                "Price ($)": "{:.2f}",
                "1D %": "{:+.2f}%",
                "5D %": "{:+.2f}%",
                "20D %": "{:+.2f}%",
                "vs 20EMA %": "{:+.2f}%",
                "vs 50EMA %": "{:+.2f}%",
                "14D RSI": "{:.1f}",
                "RVOL": "{:.2f}x",
                "20D Realized Vol (%)": "{:.1f}%",
                "60D Realized Vol (%)": "{:.1f}%",
                "Vol Ratio (20/60)": "{:.2f}",
                "Drawdown ATH (%)": "{:.2f}%",
                "20D Hist. Win Rate (%)": "{:.1f}%",
                "Median 20D Alpha (%)": "{:+.2f}%"
            })
            .background_gradient(subset=["1D %"], cmap="RdYlGn", vmin=-3, vmax=3)
            .background_gradient(subset=["5D %"], cmap="RdYlGn", vmin=-5, vmax=5)
            .background_gradient(subset=["20D %"], cmap="RdYlGn", vmin=-10, vmax=10)
            .background_gradient(subset=["vs 20EMA %"], cmap="RdYlGn", vmin=-5, vmax=5)
            .background_gradient(subset=["vs 50EMA %"], cmap="RdYlGn", vmin=-6, vmax=6)
            .background_gradient(subset=["14D RSI"], cmap="coolwarm", vmin=30, vmax=70)
            .background_gradient(subset=["RVOL"], cmap="YlOrRd", vmin=0.7, vmax=2.2)
            .background_gradient(subset=["20D Realized Vol (%)"], cmap="YlOrRd", vmin=15, vmax=50)
            .background_gradient(subset=["60D Realized Vol (%)"], cmap="YlOrRd", vmin=15, vmax=50)
            .background_gradient(subset=["Vol Ratio (20/60)"], cmap="coolwarm", vmin=0.75, vmax=1.35)
            .background_gradient(subset=["Drawdown ATH (%)"], cmap="RdYlGn", vmin=-25, vmax=0)
            .background_gradient(subset=["20D Hist. Win Rate (%)"], cmap="Greens", vmin=50, vmax=80)
            .background_gradient(subset=["Median 20D Alpha (%)"], cmap="RdYlGn", vmin=-4, vmax=8)
            .map(style_table_status, subset=["RSI Divergence", "Vol Regime", "Technical State"]),
            use_container_width=True,
            height=340
        )

    # Mag 7 Visual Comparison Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("#### 60-Day Normalized Return (% Change)")
        fig_norm = go.Figure()
        palette = ["#f59e0b", "#00F5A0", "#3b82f6", "#ec4899", "#8b5cf6", "#06b6d4", "#f97316"]
        for idx, ticker in enumerate(MAG_7):
            df_t = load_ticker_data(ticker)
            if len(df_t) >= 60:
                sub = df_t.iloc[-60:].copy()
                base_price = sub["Close"].iloc[0]
                norm_returns = (sub["Close"] / base_price - 1) * 100
                fig_norm.add_trace(go.Scatter(x=sub["Date"], y=norm_returns, mode="lines", name=ticker, line=dict(color=palette[idx % len(palette)], width=2, shape='spline', smoothing=1.3)))
        apply_dark_theme(fig_norm, height=350)
        fig_norm.update_layout(yaxis_title="Return (%)")
        st.plotly_chart(fig_norm, use_container_width=True)

    with col_chart2:
        st.markdown("#### Mag 7 Realized Volatility: 20D vs 60D")
        fig_vol = go.Figure()
        tickers = [r["Ticker"] for r in mag7_rows]
        rv20s = [r["20D Realized Vol (%)"] for r in mag7_rows]
        rv60s = [r["60D Realized Vol (%)"] for r in mag7_rows]
        fig_vol.add_trace(go.Bar(name="20D Realized Vol", x=tickers, y=rv20s, marker_color="#00F5A0"))
        fig_vol.add_trace(go.Bar(name="60D Realized Vol", x=tickers, y=rv60s, marker_color="#475569"))
        apply_dark_theme(fig_vol, height=350)
        fig_vol.update_layout(barmode="group", yaxis_title="Annualized Vol (%)")
        st.plotly_chart(fig_vol, use_container_width=True)

# ================= TAB 5: INDEXES & SECTOR FLOW =================
with tab5:
    st.subheader("Sector ETF Diagnostics & Institutional Money Flow")
    st.markdown("Tracks institutional velocity and relative strength between **Sector ETFs, Defensives, Cyclicals, and Growth**.")

    # 1. Sector ETF Technicals & Exhaustion Radar FIRST
    st.markdown("#### Sector ETF Technicals & Exhaustion Radar")
    sectors_data = market_data.get("sectors", {})
    sec_rows = []
    for ticker, d in sectors_data.items():
        sec_rows.append({
            "Sector": ticker,
            "Price ($)": d.get("close"),
            "1D %": d.get("pct_1d"),
            "5D %": d.get("pct_5d"),
            "20D %": d.get("pct_20d"),
            "14D RSI": d.get("rsi"),
            "RVOL": d.get("rvol"),
            "vs 50EMA %": d.get("dist_ema50"),
            "ATR Stretch": d.get("atr_stretch"),
            "Drawdown ATH (%)": d.get("drawdown_ath"),
            "State": d.get("tech_status")
        })
    sec_df = pd.DataFrame(sec_rows)
    if not sec_df.empty:
        st.dataframe(
            sec_df.style.format({
                "Price ($)": "{:.2f}",
                "1D %": "{:+.2f}%",
                "5D %": "{:+.2f}%",
                "20D %": "{:+.2f}%",
                "14D RSI": "{:.1f}",
                "RVOL": "{:.2f}x",
                "vs 50EMA %": "{:+.2f}%",
                "ATR Stretch": "{:+.2f}",
                "Drawdown ATH (%)": "{:.2f}%"
            })
            .background_gradient(subset=["1D %"], cmap="RdYlGn", vmin=-3, vmax=3)
            .background_gradient(subset=["5D %"], cmap="RdYlGn", vmin=-5, vmax=5)
            .background_gradient(subset=["20D %"], cmap="RdYlGn", vmin=-8, vmax=8)
            .background_gradient(subset=["vs 50EMA %"], cmap="RdYlGn", vmin=-6, vmax=6)
            .background_gradient(subset=["14D RSI"], cmap="coolwarm", vmin=30, vmax=70)
            .background_gradient(subset=["RVOL"], cmap="YlOrRd", vmin=0.7, vmax=2.2)
            .background_gradient(subset=["ATR Stretch"], cmap="coolwarm", vmin=-2.5, vmax=2.5)
            .background_gradient(subset=["Drawdown ATH (%)"], cmap="RdYlGn", vmin=-25, vmax=0)
            .map(style_table_status, subset=["State"]),
            use_container_width=True,
            height=380
        )

    # 2. 5-Day Basket Performance UNDERNEATH
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 5-Day Basket Performance & Money Flow Velocity")
    col_b1, col_b2 = st.columns([3, 2])
    with col_b1:
        baskets = market_data.get("baskets_5d", {})
        display_keys = [
            "Growth (XLK, SMH, XLC, XLY)",
            "Cyclicals (XLF, XLI, XLB)",
            "Defensives (XLU, XLP, XLV)",
            "Energy (XLE)"
        ]
        display_dict = {k: baskets[k] for k in display_keys if k in baskets}
        basket_df = pd.DataFrame(list(display_dict.items()), columns=["Basket", "5D Return (%)"])
        fig_b = px.bar(basket_df, x="5D Return (%)", y="Basket", orientation="h", color="5D Return (%)",
                       color_continuous_scale=[[0, "#FF2E55"], [0.5, "#1B212D"], [1, "#00F5A0"]])
        apply_dark_theme(fig_b, height=280)
        fig_b.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_b, use_container_width=True)

    with col_b2:
        basket_card_html = (
            '<div class="metric-card-lux" style="height:100%;">'
            '<div style="font-size:12px; font-weight:700; color:#00F5A0; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Basket Composition & Velocity</div>'
            '<div style="font-size:12.5px; color:#cbd5e1; line-height:1.55;">'
            '<div style="margin-bottom:6px;"><b style="color:#ffffff;">• Growth:</b> XLK (Tech), SMH (Semis), XLC (Comms), XLY (Cons Disc)</div>'
            '<div style="margin-bottom:6px;"><b style="color:#ffffff;">• Cyclicals:</b> XLF (Financials), XLI (Industrials), XLB (Materials)</div>'
            '<div style="margin-bottom:6px;"><b style="color:#ffffff;">• Defensives:</b> XLU (Utilities), XLP (Staples), XLV (Health)</div>'
            '<div style="margin-bottom:6px;"><b style="color:#ffffff;">• Energy:</b> XLE (Crude & Natural Gas)</div>'
            '</div>'
            '<div style="font-size:11.5px; color:#94a3b8; margin-top:10px; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">'
            'Capital rotation out of cyclicals into growth or defensives indicates institutional caution, while broad-based cyclical outperformance signals risk-on expansion.'
            '</div>'
            '</div>'
        )
        st.markdown(basket_card_html, unsafe_allow_html=True)

# ================= TAB 6: SYNTHETIC RATIO ROTATIONS =================
with tab6:
    st.subheader("Synthetic Ratio Rotations & Statistical Regimes")
    st.markdown("Quantitative measures of breadth, mega-cap concentration, and small-cap rotation.")

    ratio_choice = st.selectbox("Select Ratio to Inspect", ["SPY_RSP", "QQQ_RSP", "IWM_QQQ"], index=0)
    ratio_df = load_ticker_data(ratio_choice)

    if not ratio_df.empty:
        ratio_df["EMA20"] = ratio_df["Close"].ewm(span=20, adjust=False).mean()
        ratio_df["EMA50"] = ratio_df["Close"].ewm(span=50, adjust=False).mean()
        ratio_df["STD50"] = ratio_df["Close"].rolling(50).std()
        ratio_df["Upper2"] = ratio_df["EMA50"] + 2 * ratio_df["STD50"]
        ratio_df["Lower2"] = ratio_df["EMA50"] - 2 * ratio_df["STD50"]

        sub_ratio = ratio_df.iloc[-500:].copy()

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["Upper2"], line=dict(color="rgba(255, 46, 85, 0.7)", dash="dot"), name="+2σ Extreme"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["Lower2"], line=dict(color="rgba(0, 245, 160, 0.7)", dash="dot"), name="-2σ Extreme"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["EMA20"], line=dict(color="#06b6d4", width=1.5), name="20-Day EMA"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["EMA50"], line=dict(color="#f59e0b", width=1.5), name="50-Day EMA"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["Close"], line=dict(color="#3b82f6", width=2.5), name=ratio_choice.replace("_", "/")))

        apply_dark_theme(fig_r, height=450, title=f"{ratio_choice.replace('_', '/')} Ratio with 20/50 EMA & ±2σ Statistical Envelopes")
        fig_r.update_layout(yaxis_title="Ratio Value")
        st.plotly_chart(fig_r, use_container_width=True)

# ================= LEGACY TAB: SINGLE TICKER EXPLORER =================
if False:
    st.subheader("Deep Single Ticker Quantitative Explorer & Historical Odds")

    available_tickers = get_available_tickers()
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        selected_ticker = st.selectbox("Select Any Ticker (79 Available)", available_tickers, index=available_tickers.index("NVDA") if "NVDA" in available_tickers else 0)
    with col_sel2:
        horizon_choice = st.radio("Chart Horizon", ["1 Year", "2 Years", "5 Years", "All Historical Data"], index=3, horizontal=True)

    t_df = load_ticker_data(selected_ticker)
    t_metrics = compute_all_metrics(t_df)

    if t_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${t_metrics['close']}", f"{t_metrics['pct_1d']}% (1D)")
        c2.metric("50-Day EMA Trend", f"${t_metrics['ema50']:.2f}", f"{t_metrics['dist_ema50']:+.2f}% vs 50EMA")
        c3.metric("ATH Drawdown", f"{t_metrics['drawdown_ath']}%", f"Days: {t_metrics['days_in_drawdown']}")
        c4.metric("14D RSI", f"{t_metrics['rsi']}", f"RVOL: {t_metrics['rvol']}x")

        # Historical Edge Card (Luxury styled)
        edge = t_metrics.get("edge_match", {})
        hist_edge_card_html = (
            f'<div class="metric-card-lux" style="border-left: 4px solid #f59e0b;">'
            f'<div style="font-weight:700; font-size:16px; color:#ffffff; margin-bottom:6px;">Historical Analogue Match: {edge.get("rule_name")}</div>'
            f'<p style="color:#94a3b8; font-size:13px; margin:0 0 6px 0;"><b>Historical Sample Size:</b> {edge.get("sample_size")} instances matched in last 20 years</p>'
            f'<p style="font-size:13.5px; color:#f1f5f9; margin:0 0 6px 0;">'
            f'• <b>5-Day Win Rate:</b> {edge.get("win_rate_5d")}% &nbsp;|&nbsp; '
            f'• <b>20-Day Win Rate:</b> <span style="color:#00F5A0; font-weight:700;">{edge.get("win_rate_20d")}%</span> &nbsp;|&nbsp; '
            f'• <b>60-Day Win Rate:</b> {edge.get("win_rate_60d")}%'
            f'</p>'
            f'<p style="font-size:13px; color:#94a3b8; margin:0;">'
            f'• <b>Median 20-Day Alpha / Return:</b> <span style="color:#f59e0b; font-weight:700;">+{edge.get("median_return_20d")}%</span> &nbsp;|&nbsp; '
            f'• <b>Max Adverse Excursion (Stop guideline):</b> {edge.get("max_adverse_excursion")}%'
            f'</p>'
            f'</div>'
        )
        st.markdown(hist_edge_card_html, unsafe_allow_html=True)

        # Compute EMAs and ATH Drawdown across ALL historical data from day 1
        t_df["EMA20"] = t_df["Close"].ewm(span=20, adjust=False).mean()
        t_df["EMA50"] = t_df["Close"].ewm(span=50, adjust=False).mean()
        t_df["EMA200"] = t_df["Close"].ewm(span=200, adjust=False).mean()
        cummax = t_df["Close"].cummax()
        t_df["Drawdown_ATH"] = (t_df["Close"] - cummax) / cummax * 100

        if horizon_choice == "1 Year":
            lookback = min(len(t_df), 252)
        elif horizon_choice == "2 Years":
            lookback = min(len(t_df), 504)
        elif horizon_choice == "5 Years":
            lookback = min(len(t_df), 1260)
        else:
            lookback = len(t_df)

        t_sub = t_df.iloc[-lookback:].copy()

        # Build Synchronized 2-Row Subplot with Shared X-Axis (Synced Zoom & Pan)
        fig_combined = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
            subplot_titles=(
                f"{selected_ticker} - Daily Price & Key EMAs (20 Red, 50 Amber, 200 Blue)",
                f"{selected_ticker} - Historical Drawdown Curve from All-Time High (%)"
            )
        )

        # Row 1: Daily Price & EMAs
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["Close"], name="Close Price", line=dict(color="#ffffff", width=2)), row=1, col=1)
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["EMA20"], name="20 EMA (Red)", line=dict(color="#FF2E55", width=1.75)), row=1, col=1)
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["EMA50"], name="50 EMA (Amber)", line=dict(color="#f59e0b", width=1.75)), row=1, col=1)
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["EMA200"], name="200 EMA (Blue)", line=dict(color="#3b82f6", width=2)), row=1, col=1)

        # Row 2: Underwater Drawdown Curve from ATH
        fig_combined.add_trace(go.Scatter(
            x=t_sub["Date"],
            y=t_sub["Drawdown_ATH"],
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
            line=dict(color="#FF2E55", width=1.5),
            name="ATH Drawdown (%)"
        ), row=2, col=1)

        apply_dark_theme(fig_combined, height=620)
        fig_combined.update_xaxes(gridcolor="rgba(255, 255, 255, 0.03)", zerolinecolor="rgba(255, 255, 255, 0.08)")
        fig_combined.update_yaxes(gridcolor="rgba(255, 255, 255, 0.03)", zerolinecolor="rgba(255, 255, 255, 0.08)")
        fig_combined.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig_combined.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        # Synchronize hover crosshairs across both rows
        fig_combined.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Style subplot title labels
        for ann in fig_combined["layout"]["annotations"]:
            ann["font"] = dict(color="#ffffff", size=13, family="Plus Jakarta Sans, sans-serif")

        # Range Selector Buttons on the shared X-axis
        fig_combined.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All Time")
                ]),
                bgcolor="#111319",
                activecolor="#f59e0b"
            ),
            row=1, col=1
        )

        st.plotly_chart(fig_combined, use_container_width=True)

# ================= TAB 9: VOLATILITY PRICING =================
with tab_options:
    render_options_vol_page(get_available_tickers(), load_ticker_data)

