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
    page_title="Rotato | Quantitative Macro & Equity Desk",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    if any(w in val for w in ["Alert", "Bearish", "Explosion", "Overbought", "Upside"]):
        return "color: #ef4444; font-weight: 700;"
    if any(w in val for w in ["Healthy", "Bullish", "Squeeze", "Support", "Bounce", "Downside"]):
        return "color: #10b981; font-weight: 700;"
    return "color: #94a3b8;"

# ----------------- TOP BRAND & DESK HEADER (FILAMENT STYLE) -----------------
col_h1, col_h2 = st.columns([3.5, 1.5])
with col_h1:
    st.markdown('''
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:2px;">
            <div style="width:32px; height:32px; border-radius:8px; background:#0D0F14; border:1px solid #1B212D; display:flex; align-items:center; justify-content:center; font-weight:800; color:#ffffff; font-size:15px; position:relative;">
                R<span style="position:absolute; top:3px; right:3px; width:6px; height:6px; border-radius:50%; background:#00F5A0; box-shadow:0 0 6px #00F5A0;"></span>
            </div>
            <h1 style="font-size:22px; font-weight:700; color:#ffffff; margin:0; letter-spacing:-0.5px;">Rotato</h1>
            <span style="font-size:11px; font-weight:500; color:#94a3b8; background:#131720; border:1px solid #1B212D; padding:2px 8px; border-radius:6px;">Stealth Macro Desk</span>
            <span style="font-size:11px; color:#00F5A0; font-weight:600; display:inline-flex; align-items:center; gap:4px;"><span style="width:6px; height:6px; border-radius:50%; background:#00F5A0; display:inline-block; box-shadow:0 0 6px #00F5A0;"></span> Gemini Key Active</span>
        </div>
        <div style="font-size:12px; color:#94a3b8;">Quantitative Macro & Confluence Intelligence Desk</div>
    ''', unsafe_allow_html=True)
with col_h2:
    if st.button("⚡ Sync Feeds Now", use_container_width=True):
        with st.status("Syncing core feeds from Yahoo Finance...", expanded=True) as status:
            from data_manager import sync_tickers_batch
            target_tickers = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS
            results = sync_tickers_batch(tickers=target_tickers, force=True)
            st.cache_data.clear()
            last_date = results[0].get("last_date", "Today") if results else "Today"
            status.update(label=f"✓ Complete! Updated to {last_date}.", state="complete")
        st.toast(f"Feeds Synced! {len(results)} assets updated.", icon="✅")
        time.sleep(0.5)
        st.rerun()

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

spy_rsp = market_data["ratios"].get("SPY_RSP", {})
iwm_qqq = market_data["ratios"].get("IWM_QQQ", {})

last_market_session = "Today"
if not spy_df.empty and "Date" in spy_df.columns:
    last_market_session = str(spy_df["Date"].iloc[-1]).split()[0]





def render_filament_hero_card(ticker, label, price_val, change, sub_text, is_pos=True, is_ratio=False, sigma=0.0):
    pill_class = "badge-mint" if is_pos else "badge-cherry"
    sign = "+" if is_pos else ""
    badge = f'<span class="badge-mint tabular-num">{sigma:+.2f}σ</span>' if (is_ratio and sigma >= 0) else (f'<span class="badge-cherry tabular-num">{sigma:+.2f}σ</span>' if is_ratio else f'<span class="{pill_class} tabular-num">{sign}{change:.2f}%</span>')
    stroke_color = "#00F5A0" if is_pos else "#FF2E55"
    spark_path = "M0,18 Q30,8 60,14 T120,6 T180,12 T240,4" if is_pos else "M0,6 Q30,14 60,10 T120,20 T180,16 T240,24"
    return (
        f'<div class="stealth-card" style="display:flex; flex-direction:column; justify-content:space-between; height:100%;">'
        f'<div>'
        f'<div class="card-label"><span>{ticker} <span style="color:#64748b; font-weight:normal;">• {label}</span></span>{badge}</div>'
        f'<div class="card-val tabular-num">{price_val}</div>'
        f'</div>'
        f'<div style="margin-top:14px;">'
        f'<svg style="width:100%; height:24px; overflow:visible;" viewBox="0 0 240 24" fill="none">'
        f'<path d="{spark_path}" stroke="{stroke_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
        f'<div style="font-size:11px; color:#94a3b8; border-top:1px solid #1B212D; padding-top:6px; margin-top:4px;" class="tabular-num">{sub_text}</div>'
        f'</div>'
        f'</div>'
    )

# ----------------- FILAMENT HERO STAT CARDS (4 SPAGIOUS CARDS WITH SPARKLINES) -----------------
col_h_head1, col_h_head2 = st.columns([4, 1])
with col_h_head1:
    st.markdown('<div style="font-size:11px; font-weight:600; color:#a1a1aa; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Desk Core Radar</div>', unsafe_allow_html=True)

hero_c1, hero_c2, hero_c3, hero_c4 = st.columns(4)

with hero_c1:
    spy_p = spy_metrics.get("pct_1d", 0)
    st.markdown(render_filament_hero_card(
        "SPY", "S&P 500 Core", f"${spy_metrics.get('close', 0):,.2f}",
        spy_p, f"ATH DD: {spy_metrics.get('drawdown_ath')}% • 5D: {spy_metrics.get('pct_5d', 0):+.2f}%",
        is_pos=(spy_p >= 0)
    ), unsafe_allow_html=True)

with hero_c2:
    qqq_p = qqq_metrics.get("pct_1d", 0)
    st.markdown(render_filament_hero_card(
        "QQQ", "Nasdaq 100 Tech", f"${qqq_metrics.get('close', 0):,.2f}",
        qqq_p, f"ATH DD: {qqq_metrics.get('drawdown_ath')}% • 5D: {qqq_metrics.get('pct_5d', 0):+.2f}%",
        is_pos=(qqq_p >= 0)
    ), unsafe_allow_html=True)

with hero_c3:
    s_val = spy_rsp.get("current", 0)
    s_sig = spy_rsp.get("z_score_50d", 0)
    s_state = spy_rsp.get("actionable_state", "Breadth Expansion")
    s_col = spy_rsp.get("state_color", "#34d399")
    st.markdown(render_filament_hero_card(
        "SPY / RSP", "Breadth Spread", f"{s_val:.4f}",
        spy_rsp.get("pct_5d", 0), f'<span style="color:{s_col}; font-weight:600;">{s_state}</span>',
        is_pos=(spy_rsp.get("pct_5d", 0) >= 0), is_ratio=True, sigma=s_sig
    ), unsafe_allow_html=True)

with hero_c4:
    iwm_p = iwm_metrics.get("pct_1d", 0)
    st.markdown(render_filament_hero_card(
        "IWM", "Russell 2000 Small", f"${iwm_metrics.get('close', 0):,.2f}",
        iwm_p, f"ATH DD: {iwm_metrics.get('drawdown_ath')}% • 5D: {iwm_metrics.get('pct_5d', 0):+.2f}%",
        is_pos=(iwm_p >= 0)
    ), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)
# ----------------- MODERN TABS (Institutional Options Desk v3.0 Full Suite) -----------------
tab1, tab_agent, tab2, tab3, tab4, tab5, tab6, tab7, tab_options = st.tabs([
    "📊 Daily AI Macro Briefing",
    "🤖 Interactive AI Quant Agent",
    "🎯 AI Trade Ideas",
    "📡 Confirmation & Confluence Radar",
    "⚡ Mag 7 Deep Dive & Realized Vol",
    "🔄 Indexes & Sector Money Flow",
    "📈 Key Ratio & Breadth Radar",
    "🔍 Single Ticker Explorer",
    "🎯 Cheap IV & Strike Selector"
])

# ================= TAB 1: DAILY AI MACRO BRIEFING =================
with tab1:
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.subheader("Daily Institutional Macro Briefing & AI Analysis")
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
            col_b = "#0ecb81" if val >= 0 else "#f6465d"
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

    # 2. Comprehensive Daily AI Analysis Report
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    llm_summary = ai_output.get("llm_summary")
    engine_source = ai_output.get("source", "Quantitative Confluence Desk")

    if llm_summary:
        st.markdown(
            f'<div class="metric-card-lux" style="border:1px solid rgba(245, 158, 11, 0.35); background:linear-gradient(180deg, #13161f 0%, #0d0e13 100%); margin-bottom:14px;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px;">'
            f'<div style="display:flex; align-items:center; gap:8px;">'
            f'<span style="font-size:13px; font-weight:800; color:#f59e0b; text-transform:uppercase; letter-spacing:0.5px;">Institutional Daily Market Intelligence Briefing</span>'
            f'</div>'
            f'<span class="pill-amber" style="font-size:11px; font-weight:700;">{engine_source}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        import re
        chunks = re.split(r'(?:\n\s*---\s*\n|\n+)(?=###?\s+[0-9IVX]+\.)', '\n' + llm_summary)
        sections = [c.strip() for c in chunks if c.strip() and not c.strip().startswith('# DAILY') and not c.strip().startswith('**Date')]
        
        if len(sections) >= 4:
            p_tabs = st.tabs(["🌐 1. Macro Liquidity (AI)", "⚡ 2. Sector Velocity (AI)", "🎯 3. Setups & Traps (AI)", "🛡️ 4. Desk Game Plan (AI)"])
            for idx, tab_obj in enumerate(p_tabs):
                with tab_obj:
                    st.markdown(sections[idx])
        else:
            st.markdown(llm_summary)
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
            f'</div>'
            f'<div style="margin-top:12px; font-size:12px; color:#64748b; border-top:1px solid rgba(255,255,255,0.06); padding-top:8px;">'
            f'<i>💡 Tip: Ensure your Gemini API Key is active in the sidebar to generate full daily macro briefings and activate the AI Agent.</i>'
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
        if st.button("🔍 Mag 7 Divergence & Risks", use_container_width=True):
            prompt_clicked = "Analyze today's Mag 7 divergence. Which tech leaders are showing institutional accumulation vs distribution, and what are the key invalidation levels?"
    with qp2:
        if st.button("🚀 Sector Rotation Leaders", use_container_width=True):
            prompt_clicked = "Based on the 5-day sector velocities, where is institutional smart money rotating right now? Which sector is leading?"
    with qp3:
        if st.button("⚠️ Top Distribution Warnings", use_container_width=True):
            prompt_clicked = "What are the biggest distribution traps and high-risk names on the desk today? Where should retail traders avoid dip buying?"
    with qp4:
        if st.button("🎯 Best High-Conviction Setups", use_container_width=True):
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
            border_col = t.get("card_color", "#0ecb81")
            exec_state = t.get("execution_state", "")
            curr_px = t.get("current_price", t.get("entry"))
            ext_metric = t.get("extension_metric", "At Support")
            rr = t.get("risk_reward", "1 : 3.0 R:R")
            limit_zone = t.get("optimal_entry_zone", t.get("entry"))
            trigger = t.get("breakout_trigger", "Above 5D Pivot")
            exec_desc = t.get("execution_desc", "Bidding structural support band rather than market buying.")

            t_card_html = (
                f'<div class="metric-card-lux" style="border-left:4px solid {border_col}; margin-bottom:16px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:10px; margin-bottom:12px; flex-wrap:wrap; gap:8px;">'
                f'<div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">'
                f'<span style="font-size:22px; font-weight:800; color:#ffffff; letter-spacing:-0.5px;">{t["ticker"]}</span>'
                f'<span class="pill-{pill_type}">{t["direction"]}</span>'
                f'<span style="font-size:12px; color:#94a3b8; font-weight:600;">{t["type_label"]}</span>'
                f'{(f"<span class=\"pill-amber\" style=\"font-size:11px; font-weight:700;\">{exec_state}</span>") if exec_state else ""}'
                f'</div>'
                f'<div style="display:flex; align-items:center; gap:8px;">'
                f'<span style="font-size:11px; color:#64748b; text-transform:uppercase;">Conviction:</span>'
                f'<span class="pill-{pill_type}" style="font-size:12px; font-weight:700;">Score: {t["score"]} ({t["bias_label"]})</span>'
                f'</div>'
                f'</div>'
                # Market Price vs Optimized R:R Banner
                f'<div style="display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px; margin-bottom:12px; flex-wrap:wrap; gap:8px; font-size:12px;">'
                f'<div><span style="color:#94a3b8;">Current Print:</span> <b style="color:#ffffff;">{curr_px}</b> <span style="color:#64748b; margin:0 4px;">|</span> <span style="color:#94a3b8;">Moving Avg Extension:</span> <b style="color:#f59e0b;">{ext_metric}</b></div>'
                f'<div><span style="font-size:10px; color:#64748b; text-transform:uppercase; margin-right:4px;">Optimized Payoff:</span> <span class="pill-amber" style="font-size:11px; font-weight:800;">{rr}</span></div>'
                f'</div>'
                # 4-Box Execution Strategy Grid
                f'<div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-bottom:12px;">'
                f'<div style="background:rgba(245, 158, 11, 0.05); padding:10px 12px; border-radius:8px; border:1px solid rgba(245, 158, 11, 0.25);">'
                f'<div style="font-size:10px; font-weight:700; color:#f59e0b; text-transform:uppercase;">Optimal Limit Entry</div>'
                f'<div style="font-size:16px; font-weight:800; color:#ffffff; margin-top:2px;">{limit_zone}</div>'
                f'<div style="font-size:10px; color:#94a3b8; margin-top:2px;">20 EMA Support Zone</div>'
                f'</div>'
                f'<div style="background:rgba(255,255,255,0.02); padding:10px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">'
                f'<div style="font-size:10px; font-weight:700; color:#38bdf8; text-transform:uppercase;">Breakout Trigger</div>'
                f'<div style="font-size:14px; font-weight:700; color:#ffffff; margin-top:2px;">{trigger}</div>'
                f'<div style="font-size:10px; color:#94a3b8; margin-top:2px;">Momentum Confirmation</div>'
                f'</div>'
                f'<div style="background:rgba(239, 68, 68, 0.05); padding:10px 12px; border-radius:8px; border:1px solid rgba(239, 68, 68, 0.25);">'
                f'<div style="font-size:10px; font-weight:700; color:#ef4444; text-transform:uppercase;">Stop Invalidation</div>'
                f'<div style="font-size:16px; font-weight:800; color:#ef4444; margin-top:2px;">{t["stop"]}</div>'
                f'<div style="font-size:10px; color:#fca5a5; margin-top:2px;">{t["stop_pct"]} Below Structural Support</div>'
                f'</div>'
                f'<div style="background:rgba(16, 185, 129, 0.05); padding:10px 12px; border-radius:8px; border:1px solid rgba(16, 185, 129, 0.25);">'
                f'<div style="font-size:10px; font-weight:700; color:#10b981; text-transform:uppercase;">Profit Targets</div>'
                f'<div style="font-size:16px; font-weight:800; color:#10b981; margin-top:2px;">{t["target1"]}</div>'
                f'<div style="font-size:10px; color:#6ee7b7; margin-top:2px;">Runner: {t["target2"]} ({t["target2_pct"]})</div>'
                f'</div>'
                f'</div>'
                # Directive tip
                f'<div style="background:rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px; font-size:12px; color:#cbd5e1; margin-bottom:12px;">'
                f'<span style="color:#f59e0b; margin-right:4px;">💡</span><b style="color:#f59e0b;">Desk Directive:</b> {exec_desc}'
                f'</div>'
                # Stats Bar
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
            if status == "PASS_BULL":
                badge = "[BULLISH CONFIRMATION]"
                box_color = "rgba(16, 185, 129, 0.08)"
                border_color = "#0ecb81"
            elif status == "PASS_BEAR":
                badge = "[BEARISH CONFIRMATION]"
                box_color = "rgba(239, 68, 68, 0.08)"
                border_color = "#f6465d"
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
        palette = ["#f59e0b", "#0ecb81", "#3b82f6", "#ec4899", "#8b5cf6", "#06b6d4", "#f97316"]
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
                       color_continuous_scale="RdYlGn")
        apply_dark_theme(fig_b, height=280)
        fig_b.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_b, use_container_width=True)

    with col_b2:
        basket_card_html = (
            '<div class="metric-card-lux" style="height:100%;">'
            '<div style="font-size:12px; font-weight:700; color:#f59e0b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">Basket Composition & Velocity</div>'
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

# ================= TAB 6: RATIO RADAR =================
with tab6:
    st.subheader("Key Synthetic Ratio Regimes")
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
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["Upper2"], line=dict(color="rgba(239, 68, 68, 0.4)", dash="dot"), name="+2σ Extreme"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["Lower2"], line=dict(color="rgba(16, 185, 129, 0.4)", dash="dot"), name="-2σ Extreme"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["EMA20"], line=dict(color="#06b6d4", width=1.5), name="20-Day EMA"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["EMA50"], line=dict(color="#f59e0b", width=1.5), name="50-Day EMA"))
        fig_r.add_trace(go.Scatter(x=sub_ratio["Date"], y=sub_ratio["Close"], line=dict(color="#3b82f6", width=2.5), name=ratio_choice.replace("_", "/")))

        apply_dark_theme(fig_r, height=450, title=f"{ratio_choice.replace('_', '/')} Ratio with 20/50 EMA & ±2σ Statistical Envelopes")
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
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["EMA20"], name="20 EMA (Red)", line=dict(color="#f6465d", width=1.75)), row=1, col=1)
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["EMA50"], name="50 EMA (Amber)", line=dict(color="#f59e0b", width=1.75)), row=1, col=1)
        fig_combined.add_trace(go.Scatter(x=t_sub["Date"], y=t_sub["EMA200"], name="200 EMA (Blue)", line=dict(color="#3b82f6", width=2)), row=1, col=1)

        # Row 2: Underwater Drawdown Curve from ATH
        fig_combined.add_trace(go.Scatter(
            x=t_sub["Date"],
            y=t_sub["Drawdown_ATH"],
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
            line=dict(color="#f6465d", width=1.5),
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

# ================= TAB 9: CHEAP IV & STRIKE SELECTOR =================
with tab_options:
    render_options_vol_page(get_available_tickers(), load_ticker_data)

