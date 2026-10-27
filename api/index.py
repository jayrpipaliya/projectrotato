import os
import sys
import time
import json
import numpy as np
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_manager import (
    load_ticker_data, get_available_tickers,
    MAG_7, INDEXES, SECTORS, MACRO_ASSETS, generate_synthetic_ratios
)
from indicators import compute_all_metrics
from narrative_engine import analyze_market_narrative, compute_ratio_analytics
from confluence_engine import compute_all_confluences
from ai_strategist import generate_ai_ideas

app = FastAPI(title="Rotato Quantitative Market Intelligence Desk")

def clean_json(obj):
    return json.loads(json.dumps(obj, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x)))

_CACHE = {}

def get_cached_analysis():
    now = time.time()
    if "market_data" in _CACHE and now - _CACHE.get("market_data_time", 0) < 300:
        return _CACHE["market_data"]
    generate_synthetic_ratios()
    data = analyze_market_narrative()
    _CACHE["market_data"] = data
    _CACHE["market_data_time"] = now
    return data

def get_cached_ai(market_data):
    now = time.time()
    if "ai_output" in _CACHE and now - _CACHE.get("ai_time", 0) < 600:
        return _CACHE["ai_output"]
    ai = generate_ai_ideas(market_data)
    _CACHE["ai_output"] = ai
    _CACHE["ai_time"] = now
    return ai

def get_cached_confluence(scope: str = "core"):
    now = time.time()
    cache_key = f"conf_{scope}"
    if cache_key in _CACHE and now - _CACHE.get(f"{cache_key}_time", 0) < 300:
        return _CACHE[cache_key]
    if scope == "mag7":
        universe = MAG_7
    elif scope == "indexes":
        universe = INDEXES
    elif scope == "sectors":
        universe = SECTORS
    elif scope == "macro":
        universe = MACRO_ASSETS
    else:
        universe = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS
    items = compute_all_confluences(universe)
    _CACHE[cache_key] = items
    _CACHE[f"{cache_key}_time"] = now
    return items

@app.get("/api/market-data")
def api_market_data():
    market_data = get_cached_analysis()
    ai_output = get_cached_ai(market_data)
    confluences = get_cached_confluence("core")
    
    spy_df = load_ticker_data("SPY")
    spy_metrics = compute_all_metrics(spy_df)
    qqq_df = load_ticker_data("QQQ")
    qqq_metrics = compute_all_metrics(qqq_df)
    iwm_df = load_ticker_data("IWM")
    iwm_metrics = compute_all_metrics(iwm_df)
    
    return clean_json({
        "market_data": market_data,
        "ai_output": ai_output,
        "confluences": confluences,
        "benchmarks": {
            "SPY": spy_metrics,
            "QQQ": qqq_metrics,
            "IWM": iwm_metrics
        },
        "available_tickers": get_available_tickers()
    })

@app.get("/api/confluence")
def api_confluence(scope: str = Query("core")):
    return clean_json(get_cached_confluence(scope))

@app.get("/api/ratio/{ratio_name}")
def api_ratio(ratio_name: str):
    df = load_ticker_data(ratio_name)
    if df.empty:
        return {"error": "Ratio not found"}
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["STD50"] = df["Close"].rolling(50).std()
    df["Upper2"] = df["SMA50"] + 2 * df["STD50"]
    df["Lower2"] = df["SMA50"] - 2 * df["STD50"]
    sub = df.iloc[-500:].copy()
    sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
    return clean_json({
        "dates": sub["Date"].tolist(),
        "close": [round(float(v), 4) for v in sub["Close"]],
        "sma50": [round(float(v), 4) if v == v else None for v in sub["SMA50"]],
        "upper2": [round(float(v), 4) if v == v else None for v in sub["Upper2"]],
        "lower2": [round(float(v), 4) if v == v else None for v in sub["Lower2"]]
    })

@app.get("/api/ticker/{ticker}")
def api_ticker(ticker: str, horizon: str = Query("All Historical Data")):
    df = load_ticker_data(ticker.upper())
    if df.empty:
        return {"error": "Ticker not found"}
    metrics = compute_all_metrics(df)
    
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    cummax = df["Close"].cummax()
    df["Drawdown_ATH"] = (df["Close"] - cummax) / cummax * 100
    
    if horizon == "1 Year":
        lookback = min(len(df), 252)
    elif horizon == "2 Years":
        lookback = min(len(df), 504)
    elif horizon == "5 Years":
        lookback = min(len(df), 1260)
    else:
        lookback = len(df)
        
    sub = df.iloc[-lookback:].copy()
    sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
    
    return clean_json({
        "ticker": ticker.upper(),
        "metrics": metrics,
        "dates": sub["Date"].tolist(),
        "close": [round(float(v), 2) for v in sub["Close"]],
        "ema20": [round(float(v), 2) for v in sub["EMA20"]],
        "ema50": [round(float(v), 2) for v in sub["EMA50"]],
        "ema200": [round(float(v), 2) for v in sub["EMA200"]],
        "drawdown_ath": [round(float(v), 2) for v in sub["Drawdown_ATH"]]
    })

@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTML_CONTENT

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rotato | Quantitative Macro & Equity Desk</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            background-color: #08090d;
            color: #f8fafc;
        }
        .ticker-ribbon {
            background: #050608;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }
        .lux-card {
            background: #111319;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
        }
        .tab-btn.active {
            background: #1c1e27;
            color: #fbbf24;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
        }
        .tab-btn {
            color: #94a3b8;
            transition: all 0.2s ease;
        }
        .tab-btn:hover {
            color: #ffffff;
        }
        .pill-green { background: rgba(16, 185, 129, 0.14); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.28); }
        .pill-red { background: rgba(239, 68, 68, 0.14); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.28); }
        .pill-amber { background: rgba(245, 158, 11, 0.14); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.28); }
        .pill-cyan { background: rgba(6, 182, 212, 0.14); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.28); }
        table td, table th { padding: 10px 12px; }
    </style>
</head>
<body class="min-h-screen">

    <!-- Top Ticker Ribbon -->
    <div id="tickerRibbon" class="ticker-ribbon py-2 px-6 flex items-center gap-6 overflow-x-auto text-xs font-semibold whitespace-nowrap">
        <span class="text-slate-500">Loading market tickers...</span>
    </div>

    <!-- Main Container -->
    <div class="max-w-7xl mx-auto px-6 py-6">

        <!-- Hero Brand Header -->
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center pb-5 mb-5 border-b border-white/10 gap-4">
            <div class="flex items-center gap-3.5">
                <div class="w-11 h-11 rounded-xl bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center font-black text-2xl text-slate-950 shadow-lg shadow-amber-500/30">
                    R
                </div>
                <div>
                    <h1 class="text-2xl font-extrabold text-white tracking-tight">Rot<span class="text-amber-500">ato</span></h1>
                    <p class="text-xs text-slate-400">Quantitative Macro & Confluence Intelligence Desk</p>
                </div>
            </div>
            <div class="flex items-center gap-2 bg-[#111319] border border-white/10 px-3.5 py-1.5 rounded-full text-xs font-semibold text-slate-300">
                <span class="w-2 h-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500"></span>
                <span>Desk Active</span>
                <span class="text-slate-600">•</span>
                <span class="text-amber-400 font-bold">79 Assets Synced</span>
            </div>
        </div>

        <!-- Regime Ribbon Card -->
        <div id="regimeCard" class="lux-card p-4 mb-5 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <div class="text-[11px] font-bold text-amber-500 uppercase tracking-wider">Current Market Regime</div>
                <div id="regimeTitle" class="text-lg font-extrabold text-white mt-0.5">Analyzing Market Narrative...</div>
                <div id="regimeDetail" class="text-xs text-slate-400 mt-1 max-w-4xl">Decoding multi-factor sector flows and breadth concentration...</div>
            </div>
            <div class="text-left md:text-right">
                <div class="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Risk Profile</div>
                <div id="regimeRiskBadge" class="pill-amber inline-block px-3 py-1 rounded-full text-xs font-bold mt-1">Moderate</div>
            </div>
        </div>

        <!-- Metric Cards Row 1: Equities & Ratios -->
        <div id="benchmarksRow" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5 mb-3.5">
            <!-- Dynamic Cards -->
        </div>

        <!-- Metric Cards Row 2: Macro Cross-Asset Barometers -->
        <div id="macroBarometersRow" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 mb-5">
            <!-- Dynamic Macro Cards -->
        </div>

        <!-- Navigation Tabs -->
        <div class="bg-[#0d0e13] p-1.5 rounded-xl border border-white/5 flex gap-1.5 overflow-x-auto mb-6">
            <button onclick="switchTab('tab1')" id="btn-tab1" class="tab-btn active px-4 py-2 rounded-lg text-xs font-bold">AI Macro Briefing</button>
            <button onclick="switchTab('tab2')" id="btn-tab2" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">AI Trade Ideas</button>
            <button onclick="switchTab('tab3')" id="btn-tab3" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Confirmation & Confluence</button>
            <button onclick="switchTab('tab4')" id="btn-tab4" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Mag 7 Granular Diagnostics</button>
            <button onclick="switchTab('tab5')" id="btn-tab5" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Indexes & Sector Flow</button>
            <button onclick="switchTab('tab6')" id="btn-tab6" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Key Ratio & Breadth Radar</button>
            <button onclick="switchTab('tab7')" id="btn-tab7" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Single Ticker Explorer</button>
        </div>

        <!-- Tab 1: AI Macro Briefing -->
        <div id="tab1" class="tab-pane">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div class="lux-card p-5 border-t-4 border-amber-500">
                    <div class="flex justify-between items-center mb-2">
                        <span class="text-[11px] font-bold text-amber-500 uppercase tracking-wider">Market Regime</span>
                        <span id="briefingRiskPill" class="pill-amber px-2 py-0.5 rounded text-[11px] font-bold">Moderate</span>
                    </div>
                    <div id="briefingTitle" class="text-xl font-bold text-white mb-2">Secular Uptrend</div>
                    <div id="briefingDetail" class="text-xs text-slate-300 leading-relaxed">Cross-asset flow analysis loading...</div>
                </div>
                <div class="lux-card p-5 border-t-4 border-blue-500">
                    <div class="text-[11px] font-bold text-blue-500 uppercase tracking-wider mb-2">5-Day Sector Velocity Baskets</div>
                    <div id="basketsGrid" class="grid grid-cols-4 gap-2 text-center my-3">
                        <!-- Basket stats -->
                    </div>
                    <div class="text-xs text-slate-400 pt-2 border-t border-white/5">
                        <b>Capital Rotation:</b> Measures 5-day velocity across cyclical vs defensive balance sheets.
                    </div>
                </div>
            </div>
            <div id="synthesisCard" class="lux-card p-5 border border-amber-500/30 bg-gradient-to-b from-[#13161f] to-[#0d0e13]">
                <div class="flex justify-between items-center mb-3">
                    <span class="text-xs font-bold text-amber-500 uppercase tracking-wider">Institutional Executive Synthesis</span>
                    <span id="aiSourceBadge" class="pill-amber px-2.5 py-0.5 rounded text-[11px]">Quantitative Desk</span>
                </div>
                <div id="synthesisText" class="text-sm text-slate-300 leading-relaxed space-y-2">
                    Generating institutional synthesis...
                </div>
            </div>
        </div>

        <!-- Tab 2: AI Trade Ideas -->
        <div id="tab2" class="tab-pane hidden">
            <div class="bg-white/[0.03] border border-white/5 rounded-xl p-3 mb-4 text-xs text-slate-400">
                <b class="text-amber-500">Strict Confluence Rules:</b> Green conviction scores (+15 to +100) are reserved strictly for <b>Long Setups</b>. Red conviction scores (-15 to -100) are strictly flagged for <b>Distribution Fades or De-risking</b> to protect portfolio equity.
            </div>
            <div id="tradeCardsContainer" class="space-y-4">
                <div class="text-xs text-slate-400">Loading high-probability trade setups...</div>
            </div>
        </div>

        <!-- Tab 3: Confirmation & Confluence Radar -->
        <div id="tab3" class="tab-pane hidden">
            <div class="flex flex-wrap gap-2 mb-4">
                <button onclick="loadConfluence('core', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-white/10 text-white">Core Universe</button>
                <button onclick="loadConfluence('mag7', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-[#111319] text-slate-400">Mag 7 Only</button>
                <button onclick="loadConfluence('indexes', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-[#111319] text-slate-400">Major Indexes</button>
                <button onclick="loadConfluence('sectors', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-[#111319] text-slate-400">Sector ETFs</button>
                <button onclick="loadConfluence('macro', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-[#111319] text-slate-400">Macro Assets</button>
            </div>
            <div id="confluenceChart" class="lux-card p-4 mb-5" style="min-height: 400px;"></div>
            
            <h4 class="text-sm font-bold text-white mb-2">Confluence Confirmation Matrix</h4>
            <div class="lux-card overflow-x-auto mb-6">
                <table class="w-full text-left text-xs border-collapse" id="confluenceTable">
                    <thead>
                        <tr class="border-b border-white/10 text-slate-400 bg-black/20">
                            <th>Ticker</th><th>Price</th><th>Score</th><th>Bias</th><th>Bull Confirmations</th><th>Bear Confirmations</th><th>Status</th><th>20Y Win Rate</th><th>RSI</th><th>RVOL</th><th>vs 50SMA</th><th>Drawdown ATH</th>
                        </tr>
                    </thead>
                    <tbody id="confluenceTableBody"></tbody>
                </table>
            </div>

            <div class="lux-card p-5">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-4">
                    <h4 class="text-sm font-bold text-white">Asset Confirmation Layer Inspector</h4>
                    <select id="inspectorSelect" onchange="inspectAsset(this.value)" class="bg-[#0b0d12] text-xs text-white border border-white/10 px-3 py-1.5 rounded-lg focus:outline-none focus:border-amber-500"></select>
                </div>
                <div id="inspectorDetail" class="space-y-2"></div>
            </div>
        </div>

        <!-- Tab 4: Mag 7 Granular Diagnostics -->
        <div id="tab4" class="tab-pane hidden">
            <h3 class="text-base font-bold text-white mb-1">Mag 7 Granular Diagnostics (Technicals, Realized Vol, Drawdowns & Odds)</h3>
            <p class="text-xs text-slate-400 mb-4">Every mega-cap analyzed for <b>Drawdown Depths, Moving Average Spreads, RVOL, and 20-Day Historical Forward Odds</b>.</p>
            <div class="lux-card overflow-x-auto mb-5">
                <table class="w-full text-left text-xs border-collapse" id="mag7Table">
                    <thead>
                        <tr class="border-b border-white/10 text-slate-400 bg-black/20">
                            <th>Ticker</th><th>Price</th><th>1D %</th><th>5D %</th><th>20D %</th><th>vs 20EMA</th><th>vs 50SMA</th><th>14D RSI</th><th>RVOL</th><th>Drawdown ATH</th><th>20Y Win Rate</th><th>Median 20D Alpha</th><th>Technical State</th>
                        </tr>
                    </thead>
                    <tbody id="mag7TableBody"></tbody>
                </table>
            </div>
            <div class="lux-card p-4">
                <h4 class="text-sm font-bold text-white mb-3">60-Day Normalized Return (% Change)</h4>
                <div id="mag7NormChart" style="height: 380px;"></div>
            </div>
        </div>

        <!-- Tab 5: Indexes & Sector Money Flow -->
        <div id="tab5" class="tab-pane hidden">
            <h3 class="text-base font-bold text-white mb-1">Sector ETF Diagnostics & Institutional Money Flow</h3>
            <p class="text-xs text-slate-400 mb-4">Tracks institutional velocity and relative strength across all 12 key US sector ETFs.</p>
            <div class="lux-card overflow-x-auto mb-5">
                <table class="w-full text-left text-xs border-collapse" id="sectorTable">
                    <thead>
                        <tr class="border-b border-white/10 text-slate-400 bg-black/20">
                            <th>Sector</th><th>Price</th><th>1D %</th><th>5D %</th><th>20D %</th><th>vs 50SMA</th><th>14D RSI</th><th>RVOL</th><th>ATR Stretch</th><th>Drawdown ATH</th><th>State</th>
                        </tr>
                    </thead>
                    <tbody id="sectorTableBody"></tbody>
                </table>
            </div>
            <div class="lux-card p-4">
                <h4 class="text-sm font-bold text-white mb-3">5-Day Basket Velocity Barometer</h4>
                <div id="basketChart" style="height: 300px;"></div>
            </div>
        </div>

        <!-- Tab 6: Key Synthetic Ratio Regimes -->
        <div id="tab6" class="tab-pane hidden">
            <div class="flex justify-between items-center mb-4">
                <div>
                    <h3 class="text-base font-bold text-white">Key Synthetic Ratio Regimes</h3>
                    <p class="text-xs text-slate-400">Quantitative measures of breadth, mega-cap concentration, and small-cap rotation.</p>
                </div>
                <select id="ratioSelect" onchange="loadRatio(this.value)" class="bg-[#0b0d12] text-xs text-white border border-white/10 px-3 py-1.5 rounded-lg focus:outline-none focus:border-amber-500">
                    <option value="SPY_RSP">SPY / RSP (Breadth Concentration)</option>
                    <option value="QQQ_RSP">QQQ / RSP (Tech vs Equal-Weight)</option>
                    <option value="IWM_QQQ">IWM / QQQ (Small vs Mega Growth)</option>
                </select>
            </div>
            <div class="lux-card p-4">
                <div id="ratioChart" style="height: 450px;"></div>
            </div>
        </div>

        <!-- Tab 7: Single Ticker Explorer -->
        <div id="tab7" class="tab-pane hidden">
            <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
                <div>
                    <h3 class="text-base font-bold text-white">Single Ticker Quantitative Explorer & Historical Odds</h3>
                    <p class="text-xs text-slate-400">Select any of the 79 available assets to view full OHLCV technicals and historical setup odds.</p>
                </div>
                <div class="flex items-center gap-3">
                    <select id="tickerSelect" onchange="loadSingleTicker()" class="bg-[#0b0d12] text-xs text-white border border-white/10 px-3 py-1.5 rounded-lg focus:outline-none focus:border-amber-500"></select>
                    <div class="flex bg-white/5 p-1 rounded-lg text-xs" id="horizonButtons">
                        <button onclick="setHorizon('1 Year')" class="horizon-btn px-2.5 py-1 rounded text-slate-400">1Y</button>
                        <button onclick="setHorizon('2 Years')" class="horizon-btn px-2.5 py-1 rounded text-slate-400">2Y</button>
                        <button onclick="setHorizon('5 Years')" class="horizon-btn px-2.5 py-1 rounded text-slate-400">5Y</button>
                        <button onclick="setHorizon('All Historical Data')" class="horizon-btn active bg-[#1c1e27] text-amber-400 font-bold px-2.5 py-1 rounded">All</button>
                    </div>
                </div>
            </div>
            
            <div id="singleTickerMetricsRow" class="grid grid-cols-1 sm:grid-cols-4 gap-3.5 mb-4"></div>
            <div id="singleTickerEdgeCard" class="lux-card p-4 border-l-4 border-amber-500 mb-5"></div>
            
            <div class="lux-card p-4">
                <div id="singleTickerCombinedChart" style="height: 600px;"></div>
            </div>
        </div>

    </div>

    <!-- JavaScript Application Logic -->
    <script>
        let DATA = null;
        let CURRENT_HORIZON = "All Historical Data";

        function switchTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.remove('hidden');
            document.getElementById('btn-' + tabId).classList.add('active');

            if (tabId === 'tab3') setTimeout(() => Plotly.Plots.resize('confluenceChart'), 50);
            if (tabId === 'tab4') setTimeout(() => Plotly.Plots.resize('mag7NormChart'), 50);
            if (tabId === 'tab5') setTimeout(() => Plotly.Plots.resize('basketChart'), 50);
            if (tabId === 'tab6') setTimeout(() => Plotly.Plots.resize('ratioChart'), 50);
            if (tabId === 'tab7') setTimeout(() => Plotly.Plots.resize('singleTickerCombinedChart'), 50);
        }

        async function init() {
            try {
                const res = await fetch('/api/market-data');
                DATA = await res.json();
                renderAll();
            } catch (err) {
                console.error("Failed to load market data:", err);
            }
        }

        function renderAll() {
            renderTickerTape();
            renderRegime();
            renderBenchmarks();
            renderMacroBarometers();
            renderBriefing();
            renderTradeCards();
            renderConfluence();
            renderMag7();
            renderSectors();
            loadRatio('SPY_RSP');
            populateTickersDropdown();
            loadSingleTicker();
        }

        function renderTickerTape() {
            const m = DATA.benchmarks;
            const macro = DATA.market_data.macro_assets || {};
            const list = [
                { s: 'SPY', d: m.SPY },
                { s: 'QQQ', d: m.QQQ },
                { s: 'IWM', d: m.IWM },
                { s: 'GLD', d: macro.GLD },
                { s: 'TLT', d: macro.TLT },
                { s: 'USO', d: macro.USO },
                { s: 'IBIT', d: macro.IBIT }
            ];
            let html = '';
            list.forEach(item => {
                if (!item.d) return;
                const c = item.d.pct_1d || 0;
                const pill = c >= 0 ? 'pill-green' : 'pill-red';
                const sign = c >= 0 ? '+' : '';
                html += `
                    <div class="flex items-center gap-2">
                        <span class="text-white font-bold">${item.s}</span>
                        <span>$${Number(item.d.close).toFixed(2)}</span>
                        <span class="${pill} px-1.5 py-0.5 rounded font-bold">${sign}${c.toFixed(2)}%</span>
                    </div>
                `;
            });
            document.getElementById('tickerRibbon').innerHTML = html;
        }

        function renderRegime() {
            const md = DATA.market_data;
            document.getElementById('regimeTitle').innerText = md.narrative_title || "Secular Uptrend";
            document.getElementById('regimeDetail').innerText = md.narrative_detail || "";
            document.getElementById('regimeRiskBadge').innerText = md.risk_level || "Moderate";
        }

        function renderBenchmarks() {
            const b = DATA.benchmarks;
            const r = DATA.market_data.ratios;
            const spy_rsp = r.SPY_RSP || {};
            const iwm_qqq = r.IWM_QQQ || {};

            const cards = [
                { name: "SPY • S&P 500", p: b.SPY?.close, c: b.SPY?.pct_1d, sub: `ATH DD: ${b.SPY?.drawdown_ath}% | 5D: ${b.SPY?.pct_5d > 0 ? '+' : ''}${b.SPY?.pct_5d}% | RSI: ${b.SPY?.rsi}` },
                { name: "QQQ • Nasdaq 100", p: b.QQQ?.close, c: b.QQQ?.pct_1d, sub: `ATH DD: ${b.QQQ?.drawdown_ath}% | 5D: ${b.QQQ?.pct_5d > 0 ? '+' : ''}${b.QQQ?.pct_5d}% | RSI: ${b.QQQ?.rsi}` },
                { name: "IWM • Russell 2000", p: b.IWM?.close, c: b.IWM?.pct_1d, sub: `ATH DD: ${b.IWM?.drawdown_ath}% | 5D: ${b.IWM?.pct_5d > 0 ? '+' : ''}${b.IWM?.pct_5d}% | RSI: ${b.IWM?.rsi}` },
                { 
                    name: "SPY / RSP (Breadth)", 
                    p: spy_rsp.current, 
                    sigma: spy_rsp.z_score_50d, 
                    state: spy_rsp.actionable_state || "Concentration Peaking",
                    action: spy_rsp.action_text || "Rotate into Equal-Weight (RSP)",
                    color: spy_rsp.state_color || "#f59e0b",
                    pctile: spy_rsp.percentile_5y 
                },
                { 
                    name: "IWM / QQQ (Rotation)", 
                    p: iwm_qqq.current, 
                    sigma: iwm_qqq.z_score_50d, 
                    state: iwm_qqq.actionable_state || "Small-Cap Coiling",
                    action: iwm_qqq.action_text || "Early Mean-Reversion Bid in IWM",
                    color: iwm_qqq.state_color || "#10b981",
                    pctile: iwm_qqq.percentile_5y 
                }
            ];

            let html = '';
            cards.forEach(card => {
                const isRatio = card.sigma !== undefined;
                const sign = card.c >= 0 ? '+' : '';
                const pill = isRatio ? 'pill-amber' : (card.c >= 0 ? 'pill-green' : 'pill-red');
                const badge = isRatio ? `${card.sigma > 0 ? '+' : ''}${card.sigma?.toFixed(2)}σ` : `${sign}${card.c?.toFixed(2)}%`;
                
                let subHtml = `<div class="text-[11px] text-slate-400 mt-1">${card.sub}</div>`;
                if (isRatio) {
                    subHtml = `
                        <div class="text-[11px] text-slate-400 mt-1 flex flex-col gap-0.5">
                            <div>5Y %ile: <b class="text-white">${card.pctile}%</b> • <span style="color:${card.color};" class="font-bold">${card.state}</span></div>
                            <div class="text-[10.5px] text-slate-400"><b>Action:</b> ${card.action}</div>
                        </div>
                    `;
                }

                html += `
                    <div class="lux-card p-4">
                        <div class="flex justify-between items-center text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                            <span>${card.name}</span>
                            <span class="${pill} px-2 py-0.5 rounded font-bold">${badge}</span>
                        </div>
                        <div class="text-2xl font-extrabold text-white tracking-tight">${isRatio ? Number(card.p).toFixed(4) : '$' + Number(card.p).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                        ${subHtml}
                    </div>
                `;
            });
            document.getElementById('benchmarksRow').innerHTML = html;
        }

        function renderMacroBarometers() {
            const macro = DATA.market_data.macro_assets || {};
            const items = [
                { s: 'GLD', n: 'Gold Debasement', d: macro.GLD, tag: 'Safe Haven' },
                { s: 'USO', n: 'Crude Oil', d: macro.USO, tag: 'Energy Cost' },
                { s: 'TLT', n: '20Y Treasuries', d: macro.TLT, tag: 'Yield Pressure' },
                { s: 'IBIT', n: 'Bitcoin Trust', d: macro.IBIT, tag: 'Liquidity' }
            ];
            let html = '';
            items.forEach(it => {
                const d = it.d || {};
                const c = d.pct_1d || 0;
                const d5 = d.pct_5d || 0;
                const d20 = d.pct_20d || 0;
                const pill = c >= 0 ? 'pill-green' : 'pill-red';
                const sign = c >= 0 ? '+' : '';
                html += `
                    <div class="lux-card p-4">
                        <div class="flex justify-between items-center text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
                            <span>${it.s} • ${it.n}</span>
                            <span class="${pill} px-2 py-0.5 rounded font-bold">${sign}${c.toFixed(2)}%</span>
                        </div>
                        <div class="text-2xl font-extrabold text-white tracking-tight">$${Number(d.close || 0).toFixed(2)}</div>
                        <div class="text-[11px] text-slate-400 mt-1">5D: ${d5 > 0 ? '+' : ''}${d5.toFixed(2)}% | 20D: ${d20 > 0 ? '+' : ''}${d20.toFixed(2)}% | ${it.tag}</div>
                    </div>
                `;
            });
            document.getElementById('macroBarometersRow').innerHTML = html;
        }

        function renderBriefing() {
            const ai = DATA.ai_output;
            document.getElementById('briefingTitle').innerText = ai.regime_title || "Secular Uptrend";
            document.getElementById('briefingDetail').innerText = ai.regime_detail || "";
            document.getElementById('briefingRiskPill').innerText = ai.risk_level || "Moderate";
            document.getElementById('aiSourceBadge').innerText = ai.source || "Quantitative Desk";

            const b = ai.baskets || {};
            const keys = [
                { k: 'Growth', alt: 'Growth (XLK, SMH, XLC, XLY)', n: 'Growth' },
                { k: 'Cyclical', alt: 'Cyclicals (XLF, XLI, XLB)', n: 'Cyclical' },
                { k: 'Defensive', alt: 'Defensives (XLU, XLP, XLV)', n: 'Defensive' },
                { k: 'Energy', alt: 'Energy (XLE)', n: 'Energy' }
            ];
            let bHtml = '';
            keys.forEach(it => {
                const val = (b[it.k] !== undefined) ? b[it.k] : (b[it.alt] !== undefined ? b[it.alt] : 0);
                const col = val >= 0 ? 'text-emerald-400' : 'text-red-400';
                bHtml += `
                    <div>
                        <div class="text-[11px] text-slate-500">${it.n}</div>
                        <div class="text-base font-extrabold ${col}">${val >= 0 ? '+' : ''}${Number(val).toFixed(2)}%</div>
                    </div>
                `;
            });
            document.getElementById('basketsGrid').innerHTML = bHtml;

            const summary = ai.llm_summary || "";
            if (summary) {
                let lines = summary.split('\\n').filter(l => l.trim().length > 0);
                let sHtml = '';
                lines.forEach(l => {
                    sHtml += `<div class="mb-1.5">${l.replace(/\\*\\*(.*?)\\*\\*/g, '<b class="text-white">$1</b>')}</div>`;
                });
                document.getElementById('synthesisText').innerHTML = sHtml;
            } else {
                document.getElementById('synthesisText').innerHTML = `
                    <div>• <b>Regime Mechanics:</b> ${ai.regime_detail || ""}</div>
                    <div>• <b>Cross-Asset Flow:</b> ${DATA.market_data.macro_context || ""}</div>
                    <div>• <b class="text-red-400">Desk Risk Warning:</b> Monitor mega-cap breadth concentration (SPY/RSP at 5-year highs). Avoid front-running unconfirmed dip buys on broken distribution names.</div>
                `;
            }
        }

        function renderTradeCards() {
            const cards = DATA.ai_output.trade_cards || [];
            let html = '';
            if (cards.length === 0) {
                html = '<div class="text-xs text-slate-400">No high-confluence setups detected in current market conditions.</div>';
            } else {
                cards.forEach(t => {
                    const dir = t.direction || "LONG";
                    const pill = dir.includes("LONG") ? 'pill-green' : (dir.includes("SHORT") || dir.includes("FADE") ? 'pill-red' : 'pill-amber');
                    html += `
                        <div class="lux-card p-4 border-l-4" style="border-left-color: ${t.card_color || '#10b981'}">
                            <div class="flex justify-between items-center pb-2.5 mb-3 border-b border-white/5">
                                <div class="flex items-center gap-2.5">
                                    <span class="text-xl font-extrabold text-white">${t.ticker}</span>
                                    <span class="${pill} px-2 py-0.5 rounded text-xs font-bold">${t.direction}</span>
                                    <span class="text-xs font-semibold text-slate-400">${t.type_label}</span>
                                </div>
                                <div class="flex items-center gap-2 text-xs">
                                    <span class="text-slate-500 uppercase">Conviction:</span>
                                    <span class="${pill} px-2 py-0.5 rounded font-bold">Score: ${t.score} (${t.bias_label})</span>
                                </div>
                            </div>
                            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
                                <div class="bg-white/[0.02] p-2.5 rounded-lg border border-white/5">
                                    <div class="text-[10px] font-semibold text-slate-500 uppercase">Entry Price</div>
                                    <div class="text-base font-bold text-white">${t.entry}</div>
                                </div>
                                <div class="bg-white/[0.02] p-2.5 rounded-lg border border-white/5">
                                    <div class="text-[10px] font-semibold text-slate-500 uppercase">Stop Invalidation</div>
                                    <div class="text-base font-bold text-red-400">${t.stop} <span class="text-xs font-normal">(${t.stop_pct})</span></div>
                                </div>
                                <div class="bg-white/[0.02] p-2.5 rounded-lg border border-white/5">
                                    <div class="text-[10px] font-semibold text-slate-500 uppercase">Target 1</div>
                                    <div class="text-base font-bold text-emerald-400">${t.target1} <span class="text-xs font-normal">(${t.target1_pct})</span></div>
                                </div>
                                <div class="bg-white/[0.02] p-2.5 rounded-lg border border-white/5">
                                    <div class="text-[10px] font-semibold text-slate-500 uppercase">Target 2</div>
                                    <div class="text-base font-bold text-emerald-400">${t.target2} <span class="text-xs font-normal">(${t.target2_pct})</span></div>
                                </div>
                            </div>
                            <div class="bg-white/[0.02] p-2.5 rounded-lg border border-white/5 text-xs text-slate-400 flex flex-wrap justify-between gap-2">
                                <span>20Y Win Rate: <b class="text-emerald-400 font-bold">${t.win_rate}</b> (N=${t.sample_size})</span>
                                <span>Median 20D Alpha: <b class="text-amber-400 font-bold">${t.median_alpha}</b></span>
                                <span><b>Confirmations:</b> ${t.confirmations}</span>
                            </div>
                            <div class="text-xs text-slate-300 mt-2.5 leading-relaxed">
                                <b class="text-white">Desk Rationale:</b> ${t.thesis}
                            </div>
                        </div>
                    `;
                });
            }
            document.getElementById('tradeCardsContainer').innerHTML = html;
        }

        function getStateColor(state) {
            if (!state) return 'text-slate-400';
            const s = String(state);
            if (s.includes('Alert') || s.includes('Overbought') || s.includes('Bearish') || s.includes('Exhaustion (Overbought') || s.includes('Upside') || s.includes('Fading')) {
                return 'text-red-400 font-bold';
            }
            if (s.includes('Healthy') || s.includes('Support') || s.includes('Downside') || s.includes('Bounce') || s.includes('Bullish') || s.includes('Signal') || s.includes('Confirmed') || s.includes('Breakout')) {
                return 'text-emerald-400 font-bold';
            }
            return 'text-slate-400 font-medium';
        }

        async function loadConfluence(scope, btn) {
            document.querySelectorAll('.conf-pill').forEach(p => {
                p.classList.remove('bg-white/10', 'text-white');
                p.classList.add('bg-[#111319]', 'text-slate-400');
            });
            if (btn) {
                btn.classList.add('bg-white/10', 'text-white');
                btn.classList.remove('bg-[#111319]', 'text-slate-400');
            }

            const res = await fetch(`/api/confluence?scope=${scope}`);
            const items = await res.json();
            DATA.confluences = items;
            renderConfluence();
        }

        function renderConfluence() {
            const items = DATA.confluences || [];
            const tickers = items.map(c => c.ticker);
            const scores = items.map(c => c.score);
            const colors = items.map(c => c.color_hex);
            const hover = items.map(c => `${c.ticker}: ${c.bias_label} (${c.score})<br>Bull: ${c.bull_count}/7 | Bear: ${c.bear_count}/7`);

            const trace = {
                x: scores,
                y: tickers,
                type: 'bar',
                orientation: 'h',
                marker: { color: colors },
                hovertext: hover,
                hoverinfo: 'text'
            };
            const layout = {
                plot_bgcolor: "#111319",
                paper_bgcolor: "#111319",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 60, r: 20, t: 30, b: 30 },
                xaxis: { range: [-100, 100], gridcolor: "rgba(255, 255, 255, 0.04)", zerolinecolor: "#787b86", zerolinewidth: 2, title: "Conviction Score (-100 Bearish to +100 Bullish)" },
                yaxis: { autorange: "reversed", gridcolor: "rgba(255, 255, 255, 0.04)" }
            };
            Plotly.newPlot('confluenceChart', [trace], layout, {responsive: true});

            // Table
            let tHtml = '';
            items.forEach(c => {
                const m = c.metrics || {};
                const edge = m.edge_match || {};
                const pill = c.score >= 0 ? 'pill-green' : 'pill-red';
                tHtml += `
                    <tr class="border-b border-white/5 hover:bg-white/[0.02]">
                        <td class="font-bold text-white">${c.ticker}</td>
                        <td>$${Number(m.close || 0).toFixed(2)}</td>
                        <td><span class="${pill} px-2 py-0.5 rounded font-bold">${c.score}</span></td>
                        <td class="${getStateColor(c.bias_label)} font-bold">${c.bias_label}</td>
                        <td class="text-emerald-400 font-semibold">${c.bull_count} / 7</td>
                        <td class="text-red-400 font-semibold">${c.bear_count} / 7</td>
                        <td class="${getStateColor(c.flicker_status)}">${c.flicker_status}</td>
                        <td>${edge.win_rate_20d || 0}%</td>
                        <td>${m.rsi || 0}</td>
                        <td>${m.rvol || 0}x</td>
                        <td class="${(m.dist_sma50 || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${(m.dist_sma50 || 0) >= 0 ? '+' : ''}${(m.dist_sma50 || 0).toFixed(2)}%</td>
                        <td>${m.drawdown_ath || 0}%</td>
                    </tr>
                `;
            });
            document.getElementById('confluenceTableBody').innerHTML = tHtml;

            // Inspector options
            let optHtml = '';
            tickers.forEach(t => optHtml += `<option value="${t}">${t}</option>`);
            const selectEl = document.getElementById('inspectorSelect');
            selectEl.innerHTML = optHtml;
            if (tickers.length > 0) inspectAsset(tickers[0]);
        }

        function inspectAsset(ticker) {
            const item = (DATA.confluences || []).find(c => c.ticker === ticker);
            if (!item) return;
            let html = '';
            item.checklist.forEach(layer => {
                let badge = '[NEUTRAL]';
                let border = 'border-white/10';
                let bg = 'bg-white/[0.02]';
                if (layer.status === 'PASS_BULL') {
                    badge = '[BULLISH CONFIRMATION]';
                    border = 'border-emerald-500';
                    bg = 'bg-emerald-500/10';
                } else if (layer.status === 'PASS_BEAR') {
                    badge = '[BEARISH CONFIRMATION]';
                    border = 'border-red-500';
                    bg = 'bg-red-500/10';
                }
                html += `
                    <div class="p-3 rounded-lg border-l-4 ${border} ${bg} flex justify-between items-start">
                        <div>
                            <div class="flex items-center gap-2">
                                <span class="font-bold text-white text-xs">${layer.layer}</span>
                                <span class="text-[10px] bg-black/40 px-1.5 py-0.5 rounded text-slate-300 font-mono">${badge}</span>
                            </div>
                            <p class="text-[11px] text-slate-400 mt-1">${layer.detail}</p>
                        </div>
                        <span class="text-xs font-bold text-amber-500">${layer.points} pts</span>
                    </div>
                `;
            });
            document.getElementById('inspectorDetail').innerHTML = html;
        }

        function renderMag7() {
            const mag7 = DATA.market_data.mag7 || {};
            let tHtml = '';
            Object.keys(mag7).forEach(ticker => {
                const d = mag7[ticker];
                const edge = d.edge_match || {};
                tHtml += `
                    <tr class="border-b border-white/5 hover:bg-white/[0.02]">
                        <td class="font-bold text-white">${ticker}</td>
                        <td>$${Number(d.close).toFixed(2)}</td>
                        <td class="${d.pct_1d >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${d.pct_1d >= 0 ? '+' : ''}${d.pct_1d}%</td>
                        <td class="${d.pct_5d >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${d.pct_5d >= 0 ? '+' : ''}${d.pct_5d}%</td>
                        <td class="${d.pct_20d >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${d.pct_20d >= 0 ? '+' : ''}${d.pct_20d}%</td>
                        <td class="${d.dist_ema20 >= 0 ? 'text-emerald-400' : 'text-red-400'}">${d.dist_ema20 >= 0 ? '+' : ''}${d.dist_ema20}%</td>
                        <td class="${d.dist_sma50 >= 0 ? 'text-emerald-400' : 'text-red-400'}">${d.dist_sma50 >= 0 ? '+' : ''}${d.dist_sma50}%</td>
                        <td>${d.rsi}</td>
                        <td>${d.rvol}x</td>
                        <td>${d.drawdown_ath}%</td>
                        <td class="text-emerald-400 font-bold">${edge.win_rate_20d || 0}%</td>
                        <td class="${edge.median_return_20d >= 0 ? 'text-emerald-400' : 'text-red-400'}">${edge.median_return_20d >= 0 ? '+' : ''}${edge.median_return_20d || 0}%</td>
                        <td class="${getStateColor(d.tech_status)}">${d.tech_status}</td>
                    </tr>
                `;
            });
            document.getElementById('mag7TableBody').innerHTML = tHtml;

            // Load Mag 7 Normalized Return Chart
            loadMag7NormChart();
        }

        async function loadMag7NormChart() {
            const palette = ["#f59e0b", "#10b981", "#3b82f6", "#ec4899", "#8b5cf6", "#06b6d4", "#f97316"];
            const tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'];
            
            const promises = tickers.map(t => fetch(`/api/ticker/${t}?horizon=1 Year`).then(r => r.json()).catch(() => null));
            const results = await Promise.all(promises);
            const traces = [];

            results.forEach((data, i) => {
                if (data && data.close && data.close.length >= 60) {
                    const subDates = data.dates.slice(-60);
                    const subClose = data.close.slice(-60);
                    const base = subClose[0];
                    const norm = subClose.map(c => ((c / base) - 1) * 100);
                    traces.push({
                        x: subDates,
                        y: norm,
                        name: tickers[i],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: palette[i % palette.length], width: 2 }
                    });
                }
            });
            const layout = {
                plot_bgcolor: "#111319",
                paper_bgcolor: "#111319",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 50, r: 20, t: 30, b: 30 },
                xaxis: { gridcolor: "rgba(255, 255, 255, 0.04)" },
                yaxis: { gridcolor: "rgba(255, 255, 255, 0.04)", title: "Return (%)" }
            };
            Plotly.newPlot('mag7NormChart', traces, layout, {responsive: true});
        }

        function renderSectors() {
            const sectors = DATA.market_data.sectors || {};
            let tHtml = '';
            Object.keys(sectors).forEach(ticker => {
                const d = sectors[ticker];
                tHtml += `
                    <tr class="border-b border-white/5 hover:bg-white/[0.02]">
                        <td class="font-bold text-white">${ticker}</td>
                        <td>$${Number(d.close).toFixed(2)}</td>
                        <td class="${d.pct_1d >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${d.pct_1d >= 0 ? '+' : ''}${d.pct_1d}%</td>
                        <td class="${d.pct_5d >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${d.pct_5d >= 0 ? '+' : ''}${d.pct_5d}%</td>
                        <td class="${d.pct_20d >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${d.pct_20d >= 0 ? '+' : ''}${d.pct_20d}%</td>
                        <td class="${d.dist_sma50 >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${d.dist_sma50 >= 0 ? '+' : ''}${d.dist_sma50}%</td>
                        <td>${d.rsi}</td>
                        <td>${d.rvol}x</td>
                        <td>${d.atr_stretch}</td>
                        <td>${d.drawdown_ath}%</td>
                        <td class="${getStateColor(d.tech_status)}">${d.tech_status}</td>
                    </tr>
                `;
            });
            document.getElementById('sectorTableBody').innerHTML = tHtml;

            // Basket velocity bar chart
            const baskets = DATA.market_data.baskets_5d || {};
            const displayKeys = [
                "Growth (XLK, SMH, XLC, XLY)",
                "Cyclicals (XLF, XLI, XLB)",
                "Defensives (XLU, XLP, XLV)",
                "Energy (XLE)"
            ];
            const bNames = displayKeys.filter(k => k in baskets);
            const bVals = bNames.map(k => baskets[k]);
            const bColors = bVals.map(v => v >= 0 ? '#10b981' : '#ef4444');
            const trace = {
                x: bVals,
                y: bNames,
                type: 'bar',
                orientation: 'h',
                marker: { color: bColors }
            };
            const layout = {
                plot_bgcolor: "#111319",
                paper_bgcolor: "#111319",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 80, r: 20, t: 20, b: 30 },
                xaxis: { gridcolor: "rgba(255, 255, 255, 0.04)", title: "5D Return (%)" },
                yaxis: { gridcolor: "rgba(255, 255, 255, 0.04)" }
            };
            Plotly.newPlot('basketChart', [trace], layout, {responsive: true});
        }

        async function loadRatio(ratioName) {
            const res = await fetch(`/api/ratio/${ratioName}`);
            const data = await res.json();
            if (!data.dates) return;

            const traces = [
                { x: data.dates, y: data.upper2, name: "+2σ Extreme", mode: "lines", line: { color: "rgba(239, 68, 68, 0.4)", dash: "dot" } },
                { x: data.dates, y: data.lower2, name: "-2σ Extreme", mode: "lines", line: { color: "rgba(16, 185, 129, 0.4)", dash: "dot" } },
                { x: data.dates, y: data.sma50, name: "50-Day SMA", mode: "lines", line: { color: "#f59e0b", width: 1.5 } },
                { x: data.dates, y: data.close, name: ratioName.replace('_', '/'), mode: "lines", line: { color: "#3b82f6", width: 2.5 } }
            ];
            const layout = {
                title: { text: `${ratioName.replace('_', '/')} Ratio with 50-Day Moving Average & ±2σ Envelopes`, font: { color: '#ffffff', size: 15 } },
                plot_bgcolor: "#111319",
                paper_bgcolor: "#111319",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 50, r: 20, t: 50, b: 30 },
                xaxis: { gridcolor: "rgba(255, 255, 255, 0.04)" },
                yaxis: { gridcolor: "rgba(255, 255, 255, 0.04)", title: "Ratio Value" }
            };
            Plotly.newPlot('ratioChart', traces, layout, {responsive: true});
        }

        function populateTickersDropdown() {
            const tickers = DATA.available_tickers || [];
            let html = '';
            tickers.forEach(t => {
                html += `<option value="${t}" ${t === 'NVDA' ? 'selected' : ''}>${t}</option>`;
            });
            document.getElementById('tickerSelect').innerHTML = html;
        }

        function setHorizon(h) {
            CURRENT_HORIZON = h;
            document.querySelectorAll('.horizon-btn').forEach(b => {
                b.classList.remove('active', 'bg-[#1c1e27]', 'text-amber-400', 'font-bold');
                b.classList.add('text-slate-400');
            });
            event.target.classList.add('active', 'bg-[#1c1e27]', 'text-amber-400', 'font-bold');
            event.target.classList.remove('text-slate-400');
            loadSingleTicker();
        }

        async function loadSingleTicker() {
            const ticker = document.getElementById('tickerSelect').value || 'NVDA';
            const res = await fetch(`/api/ticker/${ticker}?horizon=${encodeURIComponent(CURRENT_HORIZON)}`);
            const data = await res.json();
            if (!data.metrics) return;

            const m = data.metrics;
            const edge = m.edge_match || {};

            // Metric Cards
            document.getElementById('singleTickerMetricsRow').innerHTML = `
                <div class="lux-card p-3.5">
                    <div class="text-[11px] font-semibold text-slate-400 uppercase">Current Price</div>
                    <div class="text-xl font-bold text-white mt-1">$${m.close}</div>
                    <div class="text-xs ${m.pct_1d >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold mt-0.5">${m.pct_1d >= 0 ? '+' : ''}${m.pct_1d}% (1D)</div>
                </div>
                <div class="lux-card p-3.5">
                    <div class="text-[11px] font-semibold text-slate-400 uppercase">50-Day SMA Trend</div>
                    <div class="text-xl font-bold text-white mt-1">$${m.sma50}</div>
                    <div class="text-xs ${m.dist_sma50 >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold mt-0.5">${m.dist_sma50 >= 0 ? '+' : ''}${m.dist_sma50}% vs 50SMA</div>
                </div>
                <div class="lux-card p-3.5">
                    <div class="text-[11px] font-semibold text-slate-400 uppercase">ATH Drawdown</div>
                    <div class="text-xl font-bold text-white mt-1">${m.drawdown_ath}%</div>
                    <div class="text-xs text-slate-400 mt-0.5">${m.days_in_drawdown} Days in DD</div>
                </div>
                <div class="lux-card p-3.5">
                    <div class="text-[11px] font-semibold text-slate-400 uppercase">14D RSI & Volume</div>
                    <div class="text-xl font-bold text-white mt-1">RSI: ${m.rsi}</div>
                    <div class="text-xs text-slate-400 mt-0.5">RVOL: ${m.rvol}x</div>
                </div>
            `;

            // Edge Card
            document.getElementById('singleTickerEdgeCard').innerHTML = `
                <div class="font-bold text-sm text-white mb-1.5">Historical Analogue Match: ${edge.rule_name || "N/A"}</div>
                <div class="text-xs text-slate-400 mb-2"><b>Historical Sample Size:</b> ${edge.sample_size || 0} instances matched in last 20 years</div>
                <div class="text-xs text-slate-200 flex flex-wrap gap-4 mb-1">
                    <span>5-Day Win Rate: <b>${edge.win_rate_5d || 0}%</b></span>
                    <span>20-Day Win Rate: <b class="text-emerald-400 font-bold">${edge.win_rate_20d || 0}%</b></span>
                    <span>60-Day Win Rate: <b>${edge.win_rate_60d || 0}%</b></span>
                </div>
                <div class="text-xs text-slate-400">
                    Median 20-Day Alpha: <b class="text-amber-400 font-bold">+${edge.median_return_20d || 0}%</b> | Max Adverse Excursion: ${edge.max_adverse_excursion || 0}%
                </div>
            `;

            // Synchronized Plotly Subplot (Row 1: Price & EMAs, Row 2: Drawdown Curve)
            const tracePrice = { x: data.dates, y: data.close, name: "Close Price", type: "scatter", mode: "lines", line: { color: "#ffffff", width: 2 }, yaxis: 'y' };
            const traceEMA20 = { x: data.dates, y: data.ema20, name: "20 EMA (Red)", type: "scatter", mode: "lines", line: { color: "#ef4444", width: 1.5 }, yaxis: 'y' };
            const traceEMA50 = { x: data.dates, y: data.ema50, name: "50 EMA (Amber)", type: "scatter", mode: "lines", line: { color: "#f59e0b", width: 1.5 }, yaxis: 'y' };
            const traceEMA200 = { x: data.dates, y: data.ema200, name: "200 EMA (Blue)", type: "scatter", mode: "lines", line: { color: "#3b82f6", width: 2 }, yaxis: 'y' };
            const traceDD = { x: data.dates, y: data.drawdown_ath, name: "ATH Drawdown (%)", type: "scatter", mode: "lines", fill: "tozeroy", fillcolor: "rgba(239, 68, 68, 0.15)", line: { color: "#ef4444", width: 1.5 }, yaxis: 'y2' };

            const layout = {
                plot_bgcolor: "#111319",
                paper_bgcolor: "#111319",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 50, r: 20, t: 30, b: 30 },
                grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
                yaxis: { domain: [0.32, 1], gridcolor: "rgba(255, 255, 255, 0.04)", title: "Price ($)" },
                yaxis2: { domain: [0, 0.25], gridcolor: "rgba(255, 255, 255, 0.04)", title: "Drawdown (%)" },
                xaxis: { gridcolor: "rgba(255, 255, 255, 0.04)", anchor: 'y2' },
                hovermode: "x unified",
                legend: { orientation: "h", y: 1.05, x: 1, xanchor: "right" }
            };
            Plotly.newPlot('singleTickerCombinedChart', [tracePrice, traceEMA20, traceEMA50, traceEMA200, traceDD], layout, {responsive: true});
        }

        window.onload = init;
    </script>
</body>
</html>
"""
