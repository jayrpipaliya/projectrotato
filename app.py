import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from data_manager import (
    load_ticker_data, update_all_tickers, get_available_tickers,
    MAG_7, INDEXES, SECTORS, MACRO_ASSETS, generate_synthetic_ratios
)
from indicators import compute_all_metrics
from narrative_engine import analyze_market_narrative, compute_ratio_analytics
from ai_strategist import generate_ai_ideas
from confluence_engine import compute_confluence_score, compute_all_confluences

st.set_page_config(
    page_title="AlphaRadar | Quantitative Macro & Equity Desk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- LUXURY FINTECH DARK THEME CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background & Main Container */
    .stApp {
        background-color: #08090d;
        color: #f8fafc;
    }
    
    /* Top Ticker Ribbon */
    .ticker-tape-container {
        display: flex;
        align-items: center;
        gap: 18px;
        overflow-x: auto;
        padding: 8px 16px;
        background: #050608;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        margin: -4rem -4rem 1.5rem -4rem;
        white-space: nowrap;
    }
    .ticker-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 600;
        color: #94a3b8;
    }
    .ticker-symbol {
        color: #f1f5f9;
        font-weight: 700;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.14);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.28);
        border-radius: 9999px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 700;
    }
    .pill-red {
        background: rgba(239, 68, 68, 0.14);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.28);
        border-radius: 9999px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 700;
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.14);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.28);
        border-radius: 9999px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 700;
    }
    .pill-cyan {
        background: rgba(6, 182, 212, 0.14);
        color: #06b6d4;
        border: 1px solid rgba(6, 182, 212, 0.28);
        border-radius: 9999px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 700;
    }

    /* Hero Brand Block */
    .brand-hero {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .brand-title-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-logo-box {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.35);
        font-weight: 900;
        font-size: 24px;
        color: #08090d;
    }
    .brand-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.6px;
        margin: 0;
        color: #ffffff;
    }
    .brand-title-accent {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .status-badge-live {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #111319;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        color: #cbd5e1;
    }
    .status-dot-green {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
    }

    /* Elevated Glass Cards */
    .metric-card-lux {
        background: #111319;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 12px;
    }
    .metric-card-lux:hover {
        transform: translateY(-2px);
        border-color: rgba(245, 158, 11, 0.35);
    }
    .card-label {
        font-size: 12px;
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
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .card-sub {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Regime Ribbon Card */
    .regime-card {
        background: #111319;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0d0e13;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 13.5px;
        border: none !important;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: #1c1e27 !important;
        color: #fbbf24 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: #08090d !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 8px 18px !important;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.28) !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 22px rgba(245, 158, 11, 0.45) !important;
        transform: translateY(-1px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0d12 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to format Plotly charts in the dark fintech theme
def apply_dark_theme(fig, height=360, title=None):
    fig.update_layout(
        plot_bgcolor="#111319",
        paper_bgcolor="#111319",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8", size=12),
        margin=dict(l=15, r=15, t=45 if title else 15, b=15),
        height=height,
        title=dict(text=title, font=dict(color="#ffffff", size=15, weight=700)) if title else None,
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)", zerolinecolor="rgba(255, 255, 255, 0.08)"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)", zerolinecolor="rgba(255, 255, 255, 0.08)"),
        legend=dict(bgcolor="rgba(17, 19, 25, 0.8)", bordercolor="rgba(255, 255, 255, 0.08)")
    )
    return fig

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
    <div style="background:linear-gradient(135deg, #f59e0b, #d97706); width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#08090d;">Q</div>
    <span style="font-size:18px; font-weight:800; color:#ffffff;">Alpha<span style="color:#f59e0b;">Radar</span></span>
</div>
""", unsafe_allow_html=True)

st.sidebar.subheader("Data Feeds (yfinance)")
sync_scope = st.sidebar.radio("Sync Scope", ["Core Universe (Mag 7 + Indexes + Sectors + Macro)", "All 79 Tickers in raw_csv"], index=0)

if st.sidebar.button("Sync Daily Closes Now", use_container_width=True):
    progress_bar = st.sidebar.progress(0.0)
    status_text = st.sidebar.empty()

    def on_progress(curr, total, ticker, res):
        pct = curr / total
        progress_bar.progress(pct)
        status_text.text(f"Updating {ticker} ({curr}/{total}): {res.get('status')}")

    target_tickers = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS if "Core" in sync_scope else None
    results = update_all_tickers(tickers=target_tickers, progress_callback=on_progress)
    progress_bar.progress(1.0)
    updated_count = sum(1 for r in results if r.get("status") in ["updated", "created"])
    status_text.success(f"Sync complete! {updated_count} tickers updated.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("AI Strategist Engine")
env_key = os.getenv("GEMINI_API_KEY", "")
user_api_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=env_key, 
    type="password", 
    help="Enter your Google Gemini API key to generate bespoke AI macro reports and trade setups."
)

if user_api_key:
    if user_api_key != env_key:
        if st.sidebar.button("Save Key to .env", use_container_width=True):
            with open(".env", "w", encoding="utf-8") as f:
                f.write(f"GEMINI_API_KEY={user_api_key.strip()}\n")
            st.sidebar.success("API Key saved permanently!")
            st.rerun()
    st.sidebar.caption("Status: Gemini API Key Active")
else:
    st.sidebar.caption("Status: Using Built-in Quantitative Heuristic AI")

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Local `raw_csv` + `yfinance` daily sync.")

# ----------------- LOAD DATA & METRICS -----------------
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

spy_rsp = market_data["ratios"].get("SPY_RSP", {})
iwm_qqq = market_data["ratios"].get("IWM_QQQ", {})

# ----------------- TOP TICKER TAPE RIBBON -----------------
def render_ticker_tape():
    tickers = [
        ("SPY", spy_metrics.get("close"), spy_metrics.get("pct_1d")),
        ("QQQ", qqq_metrics.get("close"), qqq_metrics.get("pct_1d")),
        ("IWM", iwm_metrics.get("close"), iwm_metrics.get("pct_1d")),
        ("GLD", gld_m.get("close"), gld_m.get("pct_1d")),
        ("TLT", tlt_m.get("close"), tlt_m.get("pct_1d")),
        ("USO", uso_m.get("close"), uso_m.get("pct_1d")),
        ("IBIT", ibit_m.get("close"), ibit_m.get("pct_1d")),
    ]
    items_html = []
    for sym, prc, chg in tickers:
        if prc is None:
            continue
        pill_class = "pill-green" if (chg or 0) >= 0 else "pill-red"
        sign = "+" if (chg or 0) >= 0 else ""
        items_html.append(
            f'<div class="ticker-item"><span class="ticker-symbol">{sym}</span><span>${prc:,.2f}</span><span class="{pill_class}">{sign}{chg:.2f}%</span></div>'
        )
    return f'<div class="ticker-tape-container">{"".join(items_html)}</div>'

st.markdown(render_ticker_tape(), unsafe_allow_html=True)

# ----------------- HERO BRAND BAR -----------------
brand_hero_html = (
    '<div class="brand-hero">'
    '<div class="brand-title-group">'
    '<div class="brand-logo-box">Q</div>'
    '<div>'
    '<h1 class="brand-title">Alpha<span class="brand-title-accent">Radar</span></h1>'
    '<span style="font-size:13px; color:#94a3b8;">Quantitative Macro & Confluence Intelligence Desk</span>'
    '</div>'
    '</div>'
    '<div class="status-badge-live">'
    '<div class="status-dot-green"></div>'
    '<span>Desk Active</span>'
    '<span style="color:#64748b;">•</span>'
    '<span style="color:#f59e0b; font-weight:700;">79 Assets Synced</span>'
    '</div>'
    '</div>'
)
st.markdown(brand_hero_html, unsafe_allow_html=True)

# ----------------- REGIME RIBBON CARD -----------------
narrative_title = market_data.get("narrative_title", "Secular Uptrend")
risk_level = market_data.get("risk_level", "Moderate")
macro_reading = market_data.get("macro_context", "Balanced backdrop.")
narrative_detail_text = market_data.get("narrative_detail", "")

regime_card_html = (
    '<div class="regime-card">'
    '<div>'
    '<div style="font-size:12px; font-weight:700; color:#f59e0b; text-transform:uppercase; letter-spacing:0.5px;">Current Market Regime</div>'
    f'<div style="font-size:18px; font-weight:800; color:#ffffff; margin-top:2px;">{narrative_title}</div>'
    f'<div style="font-size:13px; color:#94a3b8; margin-top:4px;">{narrative_detail_text}</div>'
    '</div>'
    '<div style="text-align:right;">'
    '<div style="font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase;">Risk Profile</div>'
    f'<div class="pill-amber" style="display:inline-block; margin-top:4px; font-size:12px; padding:3px 10px;">{risk_level}</div>'
    '</div>'
    '</div>'
)
st.markdown(regime_card_html, unsafe_allow_html=True)

# ----------------- TOP METRIC CARDS ROW 1: EQUITY BENCHMARKS -----------------
c1, c2, c3, c4, c5 = st.columns(5)

def render_metric_card(label, price, change, sub_text, pill_type="green"):
    sign = "+" if change >= 0 else ""
    pill_class = f"pill-{pill_type}"
    return (
        f'<div class="metric-card-lux">'
        f'<div class="card-label"><span>{label}</span><span class="{pill_class}">{sign}{change:.2f}%</span></div>'
        f'<div class="card-val">${price:,.2f}</div>'
        f'<div class="card-sub">{sub_text}</div>'
        f'</div>'
    )

with c1:
    st.markdown(render_metric_card(
        "SPY • S&P 500", spy_metrics.get("close", 0), spy_metrics.get("pct_1d", 0),
        f"ATH DD: {spy_metrics.get('drawdown_ath')}% | 20D RV: {spy_metrics.get('rv20')}%",
        "green" if spy_metrics.get("pct_1d", 0) >= 0 else "red"
    ), unsafe_allow_html=True)

with c2:
    st.markdown(render_metric_card(
        "QQQ • Nasdaq 100", qqq_metrics.get("close", 0), qqq_metrics.get("pct_1d", 0),
        f"ATH DD: {qqq_metrics.get('drawdown_ath')}% | 20D RV: {qqq_metrics.get('rv20')}%",
        "green" if qqq_metrics.get("pct_1d", 0) >= 0 else "red"
    ), unsafe_allow_html=True)

with c3:
    st.markdown(render_metric_card(
        "IWM • Russell 2000", iwm_metrics.get("close", 0), iwm_metrics.get("pct_1d", 0),
        f"ATH DD: {iwm_metrics.get('drawdown_ath')}% | 20D RV: {iwm_metrics.get('rv20')}%",
        "green" if iwm_metrics.get("pct_1d", 0) >= 0 else "red"
    ), unsafe_allow_html=True)

with c4:
    z_val = spy_rsp.get("z_score_50d", 0)
    card_c4 = (
        f'<div class="metric-card-lux">'
        f'<div class="card-label"><span>SPY / RSP (Breadth)</span><span class="pill-amber">{z_val:+.2f}σ</span></div>'
        f'<div class="card-val">{spy_rsp.get("current", 0):.4f}</div>'
        f'<div class="card-sub">5Y %ile: {spy_rsp.get("percentile_5y")}% (Mega Concentration)</div>'
        f'</div>'
    )
    st.markdown(card_c4, unsafe_allow_html=True)

with c5:
    z_rot = iwm_qqq.get("z_score_50d", 0)
    card_c5 = (
        f'<div class="metric-card-lux">'
        f'<div class="card-label"><span>IWM / QQQ (Rotation)</span><span class="pill-amber">{z_rot:+.2f}σ</span></div>'
        f'<div class="card-val">{iwm_qqq.get("current", 0):.4f}</div>'
        f'<div class="card-sub">5Y %ile: {iwm_qqq.get("percentile_5y")}% (Small vs Tech)</div>'
        f'</div>'
    )
    st.markdown(card_c5, unsafe_allow_html=True)

# ----------------- TOP METRIC CARDS ROW 2: MACRO CROSS-ASSET BAROMETERS -----------------
mc1, mc2, mc3, mc4 = st.columns(4)

with mc1:
    st.markdown(render_metric_card(
        "GLD • Gold Debasement", gld_m.get("close", 0), gld_m.get("pct_1d", 0),
        f"5D: {gld_m.get('pct_5d')}% | 20D RV: {gld_m.get('rv20')}% | Safe Haven",
        "green" if gld_m.get("pct_1d", 0) >= 0 else "red"
    ), unsafe_allow_html=True)

with mc2:
    st.markdown(render_metric_card(
        "USO • Crude Oil", uso_m.get("close", 0), uso_m.get("pct_1d", 0),
        f"5D: {uso_m.get('pct_5d')}% | 20D RV: {uso_m.get('rv20')}% | Energy Cost",
        "green" if uso_m.get("pct_1d", 0) >= 0 else "red"
    ), unsafe_allow_html=True)

with mc3:
    st.markdown(render_metric_card(
        "TLT • 20Y Treasuries", tlt_m.get("close", 0), tlt_m.get("pct_1d", 0),
        f"5D: {tlt_m.get('pct_5d')}% | 20D RV: {tlt_m.get('rv20')}% | Yield Pressure",
        "green" if tlt_m.get("pct_1d", 0) >= 0 else "red"
    ), unsafe_allow_html=True)

with mc4:
    st.markdown(render_metric_card(
        "IBIT • Bitcoin Trust", ibit_m.get("close", 0), ibit_m.get("pct_1d", 0),
        f"5D: {ibit_m.get('pct_5d')}% | 20D RV: {ibit_m.get('rv20')}% | Liquidity",
        "green" if ibit_m.get("pct_1d", 0) >= 0 else "red"
    ), unsafe_allow_html=True)

st.caption(f"**Cross-Asset Macro Barometer Reading:** {macro_reading}")

st.markdown("---")

# ----------------- MODERN TABS -----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "AI Macro Briefing",
    "AI Trade Ideas",
    "Confirmation & Confluence Radar",
    "Mag 7 Deep Dive & Realized Vol",
    "Indexes & Sector Money Flow",
    "Key Ratio & Breadth Radar",
    "Single Ticker Explorer"
])

# ================= TAB 1: AI MACRO BRIEFING =================
with tab1:
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.subheader("Daily Institutional Macro Briefing")
        st.caption("Automated Cross-Asset Liquidity & Market Regime Assessment")
    with col_t2:
        refresh_ai = st.button("Generate Fresh Briefing", use_container_width=True)

    if refresh_ai:
        st.cache_data.clear()

    ai_output = get_cached_ai_output(market_data, user_api_key)

    # 1. Regime & Capital Flow Grid (2 Compact Cards)
    col_reg, col_flow = st.columns(2)
    with col_reg:
        regime_title = ai_output.get("regime_title", "Secular Uptrend")
        regime_detail = ai_output.get("regime_detail", "")
        risk_level = ai_output.get("risk_level", "Moderate")

        reg_card_html = (
            '<div class="metric-card-lux" style="height:100%; border-top:3px solid #f59e0b;">'
            '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">'
            '<span style="font-size:11px; font-weight:700; color:#f59e0b; text-transform:uppercase; letter-spacing:0.5px;">Market Regime</span>'
            f'<span class="pill-amber">{risk_level}</span>'
            '</div>'
            f'<div style="font-size:20px; font-weight:800; color:#ffffff; margin-bottom:6px;">{regime_title}</div>'
            f'<div style="font-size:13px; color:#cbd5e1; line-height:1.45;">{regime_detail}</div>'
            '</div>'
        )
        st.markdown(reg_card_html, unsafe_allow_html=True)

    with col_flow:
        baskets = ai_output.get("baskets", {})
        g_val = baskets.get("Growth (XLK, XLC, XLY)", 0)
        c_val = baskets.get("Cyclicals (XLI, XLF, XME)", 0)
        d_val = baskets.get("Defensives (XLU, XLP, XLV)", 0)
        e_val = baskets.get("Energy (XLE)", 0)

        def fmt_b(val):
            col_b = "#10b981" if val >= 0 else "#ef4444"
            sign_b = "+" if val >= 0 else ""
            return f'<span style="font-size:16px; font-weight:800; color:{col_b};">{sign_b}{val:.2f}%</span>'

        flow_card_html = (
            '<div class="metric-card-lux" style="height:100%; border-top:3px solid #3b82f6;">'
            '<div style="font-size:11px; font-weight:700; color:#3b82f6; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:10px;">5-Day Sector Velocity Baskets</div>'
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

    # 2. Cross-Asset Barometers (4 micro-cards)
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)

    def render_macro_micro_card(ticker, name, d5, rv, desc, border_col):
        sign = "+" if d5 >= 0 else ""
        col_c = "#10b981" if d5 >= 0 else "#ef4444"
        return (
            f'<div class="metric-card-lux" style="border-left:3px solid {border_col}; padding:12px 14px; margin-bottom:0px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<span style="font-size:13px; font-weight:700; color:#ffffff;">{ticker} <span style="font-size:11px; color:#64748b; font-weight:400;">{name}</span></span>'
            f'<span style="font-size:12px; font-weight:700; color:{col_c};">{sign}{d5:.2f}% 5D</span>'
            f'</div>'
            f'<div style="font-size:11.5px; color:#cbd5e1; margin-top:4px;"><b>Signal:</b> {desc}</div>'
            f'<div style="font-size:10.5px; color:#64748b; margin-top:2px;">20D Vol: {rv}%</div>'
            f'</div>'
        )

    with c_m1:
        st.markdown(render_macro_micro_card("TLT", "Bonds", tlt_m.get("pct_5d", 0), tlt_m.get("rv20", 0), "Yield Relief (Growth Multiple Relief)", "#3b82f6"), unsafe_allow_html=True)
    with c_m2:
        st.markdown(render_macro_micro_card("GLD", "Gold", gld_m.get("pct_5d", 0), gld_m.get("rv20", 0), "Monetary Debasement & Safe-Haven Bid", "#f59e0b"), unsafe_allow_html=True)
    with c_m3:
        st.markdown(render_macro_micro_card("USO", "Crude Oil", uso_m.get("pct_5d", 0), uso_m.get("rv20", 0), "Cost-Push Inflation & Energy Drag", "#ef4444"), unsafe_allow_html=True)
    with c_m4:
        st.markdown(render_macro_micro_card("IBIT", "Bitcoin", ibit_m.get("pct_5d", 0), ibit_m.get("rv20", 0), "Speculative Liquidity & Risk Appetite", "#10b981"), unsafe_allow_html=True)

    # 3. Desk Macro Synthesis Card
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    llm_summary = ai_output.get("llm_summary")
    engine_source = ai_output.get("source", "Quantitative Confluence Desk")

    if llm_summary:
        synth_html = (
            f'<div class="metric-card-lux" style="border:1px solid rgba(245, 158, 11, 0.3); background:linear-gradient(180deg, #13161f 0%, #0d0e13 100%);">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">'
            f'<span style="font-size:12px; font-weight:700; color:#f59e0b; text-transform:uppercase; letter-spacing:0.5px;">Institutional Executive Synthesis</span>'
            f'<span class="pill-amber" style="font-size:11px;">{engine_source}</span>'
            f'</div>'
            f'<div style="font-size:13.5px; color:#cbd5e1; line-height:1.6;">'
        )
        for line in llm_summary.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    header = parts[0].replace("**", "").replace("1.", "").replace("2.", "").replace("3.", "").strip()
                    body = parts[1].strip()
                    synth_html += f'<div style="margin-bottom:8px;"><b style="color:#ffffff;">• {header}:</b> <span style="color:#94a3b8;">{body}</span></div>'
                else:
                    synth_html += f'<div style="margin-bottom:8px;">{line}</div>'
            else:
                synth_html += f'<div style="margin-bottom:6px;">{line}</div>'
        synth_html += '</div></div>'
        st.markdown(synth_html, unsafe_allow_html=True)
    else:
        synth_html = (
            f'<div class="metric-card-lux" style="border:1px solid rgba(245, 158, 11, 0.3);">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">'
            f'<span style="font-size:12px; font-weight:700; color:#f59e0b; text-transform:uppercase; letter-spacing:0.5px;">Institutional Executive Synthesis</span>'
            f'<span class="pill-amber" style="font-size:11px;">{engine_source}</span>'
            f'</div>'
            f'<div style="font-size:13.5px; color:#cbd5e1; line-height:1.6;">'
            f'<div style="margin-bottom:8px;"><b style="color:#ffffff;">• Regime Mechanics:</b> <span style="color:#94a3b8;">{regime_detail}</span></div>'
            f'<div style="margin-bottom:8px;"><b style="color:#ffffff;">• Cross-Asset Flow:</b> <span style="color:#94a3b8;">{macro_reading}</span></div>'
            f'<div style="margin-bottom:8px;"><b style="color:#ef4444;">• Desk Risk Warning:</b> <span style="color:#94a3b8;">Monitor mega-cap breadth concentration (SPY/RSP at 5-year highs). Avoid front-running unconfirmed dip buys on broken distribution names.</span></div>'
            f'</div></div>'
        )
        st.markdown(synth_html, unsafe_allow_html=True)

# ================= TAB 2: AI TRADE IDEAS =================
with tab2:
    st.subheader("Algorithmic High-Probability Trade Setups")
    st.caption("Filtered by 7-Layer Confluence and 20-Year Historical Backtest Odds")

    st.markdown(
        '<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:10px 14px; margin-bottom:18px; font-size:12.5px; color:#94a3b8;">'
        '<b style="color:#f59e0b;">Strict Confluence Rules:</b> '
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
        for t in trade_cards:
            direction = t.get("direction", "LONG")
            pill_type = "green" if "LONG" in direction else ("red" if "SHORT" in direction or "FADE" in direction else ("cyan" if "DIP" in direction else "amber"))
            border_col = t.get("card_color", "#10b981")

            t_card_html = (
                f'<div class="metric-card-lux" style="border-left:4px solid {border_col}; margin-bottom:16px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px; margin-bottom:12px;">'
                f'<div style="display:flex; align-items:center; gap:10px;">'
                f'<span style="font-size:22px; font-weight:800; color:#ffffff; letter-spacing:-0.5px;">{t["ticker"]}</span>'
                f'<span class="pill-{pill_type}">{t["direction"]}</span>'
                f'<span style="font-size:12px; color:#94a3b8; font-weight:600;">{t["type_label"]}</span>'
                f'</div>'
                f'<div style="display:flex; align-items:center; gap:8px;">'
                f'<span style="font-size:11px; color:#64748b; text-transform:uppercase;">Conviction:</span>'
                f'<span class="pill-{pill_type}" style="font-size:12px; font-weight:700;">Score: {t["score"]} ({t["bias_label"]})</span>'
                f'</div>'
                f'</div>'
                f'<div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:12px;">'
                f'<div style="background:rgba(255,255,255,0.02); padding:8px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.04);">'
                f'<div style="font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase;">Entry Price</div>'
                f'<div style="font-size:18px; font-weight:800; color:#ffffff; margin-top:2px;">{t["entry"]}</div>'
                f'<div style="font-size:11px; color:#94a3b8;">Current Execution</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.02); padding:8px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.04);">'
                f'<div style="font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase;">Stop Invalidation</div>'
                f'<div style="font-size:18px; font-weight:800; color:#ef4444; margin-top:2px;">{t["stop"]}</div>'
                f'<div style="font-size:11px; color:#ef4444;">{t["stop_pct"]} Max Risk</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.02); padding:8px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.04);">'
                f'<div style="font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase;">Target 1 (Primary)</div>'
                f'<div style="font-size:18px; font-weight:800; color:#10b981; margin-top:2px;">{t["target1"]}</div>'
                f'<div style="font-size:11px; color:#10b981;">{t["target1_pct"]} Expected</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.02); padding:8px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.04);">'
                f'<div style="font-size:11px; font-weight:600; color:#64748b; text-transform:uppercase;">Target 2 (Runner)</div>'
                f'<div style="font-size:18px; font-weight:800; color:#10b981; margin-top:2px;">{t["target2"]}</div>'
                f'<div style="font-size:11px; color:#10b981;">{t["target2_pct"]} Stretch</div>'
                f'</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:8px 12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                f'<span style="font-size:12.5px; color:#94a3b8;">20Y Win Rate: <b style="color:#10b981; font-size:14px;">{t["win_rate"]}</b> <span style="color:#64748b;">(N={t["sample_size"]})</span></span>'
                f'<span style="font-size:12.5px; color:#94a3b8;">Median 20D Alpha: <b style="color:#f59e0b;">{t["median_alpha"]}</b></span>'
                f'<span style="font-size:12px; color:#cbd5e1;"><b>Confirmations:</b> {t["confirmations"]}</span>'
                f'</div>'
                f'<div style="font-size:13px; color:#cbd5e1; margin-top:10px; line-height:1.45;">'
                f'<b style="color:#ffffff;">Desk Rationale:</b> {t["thesis"]}'
                f'</div>'
                f'</div>'
            )
            st.markdown(t_card_html, unsafe_allow_html=True)

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

# ================= TAB 3: CONFIRMATION & CONFLUENCE RADAR =================
with tab3:
    st.subheader("Multi-Factor Confirmation & Bias Intensity Radar")
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
            "20D RV (%)": f"{m.get('rv20')}%",
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
            if status == "PASS_BULL":
                badge = "[BULLISH CONFIRMATION]"
                box_color = "rgba(16, 185, 129, 0.08)"
                border_color = "#10b981"
            elif status == "PASS_BEAR":
                badge = "[BEARISH CONFIRMATION]"
                box_color = "rgba(239, 68, 68, 0.08)"
                border_color = "#ef4444"
            else:
                badge = "[NEUTRAL / INACTIVE]"
                box_color = "#111319"
                border_color = "rgba(255, 255, 255, 0.12)"

            layer_card_html = (
                f'<div style="background-color: {box_color}; border-left: 4px solid {border_color}; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; border-top: 1px solid rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.04); border-bottom: 1px solid rgba(255,255,255,0.04);">'
                f'<span style="font-weight: 700; font-size: 14px; color: #ffffff;">{layer["layer"]}</span> &nbsp;'
                f'<code style="font-size: 12px; background: rgba(0,0,0,0.3);">{badge}</code> &nbsp;'
                f'<span style="float: right; font-weight: 700; color: #f59e0b;">{layer["points"]} pts</span>'
                f'<p style="margin: 6px 0 0 0; font-size: 13px; color: #94a3b8;">{layer["detail"]}</p>'
                f'</div>'
            )
            st.markdown(layer_card_html, unsafe_allow_html=True)

# ================= TAB 4: MAG 7 DEEP DIVE =================
with tab4:
    st.subheader("Mag 7 Granular Diagnostics (Technicals, Realized Vol, Drawdowns & Odds)")
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
            "vs 50SMA %": d.get("dist_sma50"),
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
                "vs 50SMA %": "{:+.2f}%",
                "14D RSI": "{:.1f}",
                "RVOL": "{:.2f}x",
                "20D Realized Vol (%)": "{:.1f}%",
                "60D Realized Vol (%)": "{:.1f}%",
                "Vol Ratio (20/60)": "{:.2f}",
                "Drawdown ATH (%)": "{:.2f}%",
                "20D Hist. Win Rate (%)": "{:.1f}%",
                "Median 20D Alpha (%)": "{:+.2f}%"
            }).background_gradient(subset=["20D Hist. Win Rate (%)"], cmap="Greens", vmin=50, vmax=85)
              .background_gradient(subset=["20D Realized Vol (%)"], cmap="YlOrRd", vmin=15, vmax=50),
            use_container_width=True,
            height=320
        )

    # Mag 7 Visual Comparison Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("#### 60-Day Normalized Return (% Change)")
        fig_norm = go.Figure()
        palette = ["#f59e0b", "#10b981", "#3b82f6", "#ec4899", "#8b5cf6", "#06b6d4", "#f97316"]
        for idx, ticker in enumerate(MAG_7):
            df_t = load_ticker_data(ticker)
            if len(df_t) >= 60:
                sub = df_t.iloc[-60:].copy()
                base_price = sub["Close"].iloc[0]
                norm_returns = (sub["Close"] / base_price - 1) * 100
                fig_norm.add_trace(go.Scatter(x=sub["Date"], y=norm_returns, mode="lines", name=ticker, line=dict(color=palette[idx % len(palette)], width=2)))
        apply_dark_theme(fig_norm, height=350)
        fig_norm.update_layout(yaxis_title="Return (%)")
        st.plotly_chart(fig_norm, use_container_width=True)

    with col_chart2:
        st.markdown("#### Mag 7 Realized Volatility: 20D vs 60D")
        fig_vol = go.Figure()
        tickers = [r["Ticker"] for r in mag7_rows]
        rv20s = [r["20D Realized Vol (%)"] for r in mag7_rows]
        rv60s = [r["60D Realized Vol (%)"] for r in mag7_rows]
        fig_vol.add_trace(go.Bar(name="20D Realized Vol", x=tickers, y=rv20s, marker_color="#f59e0b"))
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
            "ATR Stretch": d.get("atr_stretch"),
            "20D Realized Vol (%)": d.get("rv20"),
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
                "ATR Stretch": "{:+.2f}",
                "20D Realized Vol (%)": "{:.1f}%",
                "Drawdown ATH (%)": "{:.2f}%"
            }).background_gradient(subset=["5D %"], cmap="RdYlGn", vmin=-4, vmax=4),
            use_container_width=True,
            height=360
        )

    # 2. 5-Day Basket Performance UNDERNEATH
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 5-Day Basket Performance & Money Flow Velocity")
    col_b1, col_b2 = st.columns([3, 2])
    with col_b1:
        baskets = market_data.get("baskets_5d", {})
        basket_df = pd.DataFrame(list(baskets.items()), columns=["Basket", "5D Return (%)"])
        fig_b = px.bar(basket_df, x="5D Return (%)", y="Basket", orientation="h", color="5D Return (%)",
                       color_continuous_scale="RdYlGn")
        apply_dark_theme(fig_b, height=280)
        fig_b.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_b, use_container_width=True)

    with col_b2:
        basket_card_html = (
            '<div class="metric-card-lux" style="height:100%;">'
            '<div style="font-size:12px; font-weight:700; color:#f59e0b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Basket Composition & Velocity</div>'
            '<div style="font-size:12.5px; color:#cbd5e1; line-height:1.55;">'
            '<div style="margin-bottom:6px;"><b style="color:#ffffff;">• Growth:</b> XLK (Tech), XLC (Comms), XLY (Cons Disc)</div>'
            '<div style="margin-bottom:6px;"><b style="color:#ffffff;">• Cyclicals:</b> XLI (Industrials), XLF (Financials), XME (Metals)</div>'
            '<div style="margin-bottom:6px;"><b style="color:#ffffff;">• Defensives:</b> XLU (Utilities), XLP (Staples), XLV (Health)</div>'
            '<div style="margin-bottom:6px;"><b style="color:#ffffff;">• Energy:</b> XLE (Crude & Natural Gas)</div>'
            '</div>'
            '<div style="font-size:11.5px; color:#94a3b8; margin-top:10px; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">'
            'Capital rotation out of cyclicals into growth or defensives indicates institutional caution, while broad-based cyclical outperformance signals risk-on expansion.'
            '</div>'
            '</div>'
        )
        st.markdown(basket_card_html, unsafe_allow_html=True)

# ================= TAB 6: RATIO RADAR =================
with tab6:
    st.subheader("Key Synthetic Ratio Regimes")
    st.markdown("Quantitative measures of breadth, mega-cap concentration, and small-cap rotation.")

    ratio_choice = st.selectbox("Select Ratio to Inspect", ["SPY_RSP", "QQQ_RSP", "IWM_QQQ"], index=0)
    ratio_df = load_ticker_data(ratio_choice)

    if not ratio_df.empty:
        ratio_df["SMA50"] = ratio_df["Close"].rolling(50).mean()
        ratio_df["STD50"] = ratio_df["Close"].rolling(50).std()
        ratio_df["Upper2"] = ratio_df["SMA50"] + 2 * ratio_df["STD50"]
        ratio_df["Lower2"] = ratio_df["SMA50"] - 2 * ratio_df["STD50"]

        sub_ratio = ratio_df.iloc[-500:].copy()

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["Upper2"], line=dict(color="rgba(239, 68, 68, 0.4)", dash="dot"), name="+2σ Extreme"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["Lower2"], line=dict(color="rgba(16, 185, 129, 0.4)", dash="dot"), name="-2σ Extreme"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["SMA50"], line=dict(color="#f59e0b", width=1.5), name="50-Day SMA"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["Close"], line=dict(color="#3b82f6", width=2.5), name=ratio_choice.replace("_", "/")))

        apply_dark_theme(fig_r, height=450, title=f"{ratio_choice.replace('_', '/')} Ratio with 50-Day Moving Average & ±2σ Statistical Envelopes")
        fig_r.update_layout(yaxis_title="Ratio Value")
        st.plotly_chart(fig_r, use_container_width=True)

# ================= TAB 7: SINGLE TICKER EXPLORER =================
with tab7:
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
        c2.metric("20D Realized Vol", f"{t_metrics['rv20']}%", f"Ratio: {t_metrics['vol_ratio']}")
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
            f'• <b>20-Day Win Rate:</b> <span style="color:#10b981; font-weight:700;">{edge.get("win_rate_20d")}%</span> &nbsp;|&nbsp; '
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
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["EMA20"], name="20 EMA (Red)", line=dict(color="#ef4444", width=1.75)), row=1, col=1)
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["EMA50"], name="50 EMA (Amber)", line=dict(color="#f59e0b", width=1.75)), row=1, col=1)
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["EMA200"], name="200 EMA (Blue)", line=dict(color="#3b82f6", width=2)), row=1, col=1)

        # Row 2: Underwater Drawdown Curve from ATH
        fig_combined.add_trace(go.Scatter(
            x=t_sub["Date"],
            y=t_sub["Drawdown_ATH"],
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
            line=dict(color="#ef4444", width=1.5),
            name="ATH Drawdown (%)"
        ), row=2, col=1)

        apply_dark_theme(fig_combined, height=620)
        fig_combined.update_xaxes(gridcolor="rgba(255, 255, 255, 0.04)", zerolinecolor="rgba(255, 255, 255, 0.08)")
        fig_combined.update_yaxes(gridcolor="rgba(255, 255, 255, 0.04)", zerolinecolor="rgba(255, 255, 255, 0.08)")
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
