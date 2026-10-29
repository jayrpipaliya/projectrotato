import os
import sys
import time
import json
from datetime import datetime
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

def get_cached_ai(market_data, api_key: Optional[str] = None):
    now = time.time()
    cache_key = f"ai_{api_key or 'default'}"
    if cache_key in _CACHE and now - _CACHE.get(f"{cache_key}_time", 0) < 600:
        return _CACHE[cache_key]
    ai = generate_ai_ideas(market_data, gemini_api_key=api_key)
    _CACHE[cache_key] = ai
    _CACHE[f"{cache_key}_time"] = now
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
    
    spy_df = load_ticker_data("SPY")
    spy_metrics = compute_all_metrics(spy_df)
    qqq_df = load_ticker_data("QQQ")
    qqq_metrics = compute_all_metrics(qqq_df)
    iwm_df = load_ticker_data("IWM")
    iwm_metrics = compute_all_metrics(iwm_df)
    
    last_date = "Today"
    if not spy_df.empty and "Date" in spy_df.columns:
        last_date = str(spy_df["Date"].iloc[-1]).split()[0]
    
    return clean_json({
        "market_data": market_data,
        "benchmarks": {
            "SPY": spy_metrics,
            "QQQ": qqq_metrics,
            "IWM": iwm_metrics
        },
        "last_market_session": last_date,
        "available_tickers": get_available_tickers()
    })

@app.get("/api/ai-briefing")
def api_ai_briefing(api_key: Optional[str] = Query(None)):
    market_data = get_cached_analysis()
    ai_output = get_cached_ai(market_data, api_key=api_key)
    return clean_json(ai_output)

@app.post("/api/agent-chat")
async def api_agent_chat(payload: dict):
    user_query = payload.get("query", "")
    api_key = payload.get("api_key") or os.getenv("GEMINI_API_KEY")
    history = payload.get("history", [])
    market_data = get_cached_analysis()
    confluences = get_cached_confluence("core")
    from ai_strategist import query_ai_agent
    answer = query_ai_agent(user_query, market_data, confluences, gemini_api_key=api_key, chat_history=history)
    return {"answer": answer}

@app.post("/api/sync-data")
def api_sync_data(payload: dict = None):
    try:
        t0 = time.time()
        scope = (payload or {}).get("scope", "core")
        from data_manager import sync_tickers_batch, get_available_tickers
        target_tickers = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS if scope == "core" else get_available_tickers()
        results = sync_tickers_batch(tickers=target_tickers, force=True)
        
        errors = [r["error"] for r in results if "error" in r]
        updated_count = sum(1 for r in results if r.get("status") in ["updated", "created"])
        sample_tickers = [r["ticker"] for r in results if "ticker" in r][:8]
        
        last_date = "N/A"
        for r in results:
            if r.get("last_date"):
                last_date = str(r["last_date"])
                break
        if last_date == "N/A":
            last_date = datetime.now().strftime("%Y-%m-%d")

        duration = round(time.time() - t0, 2)
        _CACHE.clear()

        if not sample_tickers and errors:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Data provider error: {errors[0]}",
                    "details": errors
                }
            )

        return {
            "status": "success",
            "updated_count": updated_count,
            "total": len(results) if results else len(target_tickers),
            "last_date": last_date,
            "sample_tickers": sample_tickers,
            "duration": duration,
            "timestamp": datetime.now().strftime("%I:%M:%S %p"),
            "warnings": errors if errors else []
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Sync failed: {str(e)}"
            }
        )

@app.get("/api/confluence")
def api_confluence(scope: str = Query("core")):
    return clean_json(get_cached_confluence(scope))

@app.get("/api/ratio/{ratio_name}")
def api_ratio(ratio_name: str):
    df = load_ticker_data(ratio_name)
    if df.empty:
        return {"error": "Ratio not found"}
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["STD50"] = df["Close"].rolling(50).std()
    df["Upper2"] = df["EMA50"] + 2 * df["STD50"]
    df["Lower2"] = df["EMA50"] - 2 * df["STD50"]
    sub = df.iloc[-500:].copy()
    sub["Date"] = sub["Date"].dt.strftime("%Y-%m-%d")
    return clean_json({
        "dates": sub["Date"].tolist(),
        "close": [round(float(v), 4) for v in sub["Close"]],
        "ema20": [round(float(v), 4) if v == v else None for v in sub["EMA20"]],
        "ema50": [round(float(v), 4) if v == v else None for v in sub["EMA50"]],
        "sma50": [round(float(v), 4) if v == v else None for v in sub["EMA50"]],
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
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
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

        /* Executive Synthesis Styling */
        .synth-body ul {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin: 10px 0;
            padding-left: 0;
            list-style: none;
        }
        .synth-body li {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 13px;
            line-height: 1.6;
            color: #cbd5e1;
            list-style: none;
            transition: all 0.2s ease;
        }
        .synth-body li:hover {
            background: rgba(255, 255, 255, 0.035);
            border-color: rgba(255, 255, 255, 0.1);
        }
        .synth-body li.card-long {
            border-color: rgba(16, 185, 129, 0.25);
            border-left: 4px solid #10b981;
            background: rgba(16, 185, 129, 0.04);
        }
        .synth-body li.card-warn {
            border-color: rgba(239, 68, 68, 0.25);
            border-left: 4px solid #ef4444;
            background: rgba(239, 68, 68, 0.04);
        }
        .synth-body li.card-risk {
            border-color: rgba(245, 158, 11, 0.25);
            border-left: 4px solid #f59e0b;
            background: rgba(245, 158, 11, 0.04);
        }
        .synth-body li.card-macro {
            border-color: rgba(14, 165, 233, 0.25);
            border-left: 4px solid #0ea5e9;
            background: rgba(14, 165, 233, 0.04);
        }
        .synth-body h4 {
            font-size: 12.5px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #e2e8f0;
            margin-top: 14px;
            margin-bottom: 6px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .synth-body strong, .synth-body b {
            color: #ffffff;
            font-weight: 700;
        }
        .synth-body p {
            margin-bottom: 8px;
            font-size: 13px;
            line-height: 1.65;
            color: #cbd5e1;
        }
    </style>
</head>
<body class="min-h-screen bg-[#08090d] text-[#f8fafc] flex flex-col antialiased">

    <!-- Top Ticker Ribbon -->
    <div id="tickerRibbon" class="ticker-ribbon py-2 px-6 flex items-center gap-6 overflow-x-auto text-xs font-semibold whitespace-nowrap shrink-0 border-b border-white/5">
        <span class="text-slate-500">Loading market tickers...</span>
    </div>

    <!-- App Wrapper: Left Sidebar + Main Dashboard -->
    <div class="flex flex-1 min-h-0">

        <!-- Left Sidebar (Matching Streamlit sidebar exactly) -->
        <aside id="sidebar" class="w-72 bg-[#0a0b10] border-r border-white/5 p-5 flex flex-col shrink-0 overflow-y-auto space-y-6">
            <!-- Sidebar Header Brand -->
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center font-extrabold text-base text-slate-950 shadow-md shadow-amber-500/20">
                    R
                </div>
                <span class="text-lg font-extrabold text-white tracking-tight">Rotato</span>
            </div>

            <!-- Data Feeds Section -->
            <div class="space-y-3 pt-3 border-t border-white/5">
                <div class="text-xs font-bold text-slate-300 uppercase tracking-wider">Data Feeds (yfinance)</div>
                <div class="space-y-2">
                    <label class="flex items-start gap-2.5 text-xs text-slate-300 cursor-pointer">
                        <input type="radio" name="syncScope" value="core" checked class="mt-0.5 accent-amber-500">
                        <span>Core Universe (Mag 7 + Indexes + Sectors + Macro)</span>
                    </label>
                    <label class="flex items-start gap-2.5 text-xs text-slate-400 cursor-pointer">
                        <input type="radio" name="syncScope" value="all" class="mt-0.5 accent-amber-500">
                        <span>All 79 Tickers in raw_csv</span>
                    </label>
                </div>
                <button onclick="triggerDataSync()" id="syncBtn" class="w-full bg-[#161822] hover:bg-[#1d202d] border border-white/10 hover:border-amber-500/40 text-xs font-bold text-slate-200 py-2.5 px-3 rounded-lg flex items-center justify-center gap-2 transition active:scale-[0.99]">
                    <span id="syncSpinner" class="hidden inline-block animate-spin">⚡</span>
                    <span id="syncBtnText">Sync Daily Closes Now</span>
                </button>

                <!-- Live Sync Real-Time Feedback Panel -->
                <div id="syncFeedbackBox" class="hidden p-3 rounded-xl bg-[#11131a] border border-white/10 space-y-2.5 text-xs">
                    <div class="space-y-1">
                        <div class="flex justify-between items-center text-[10px] text-slate-400">
                            <span id="syncStageTitle" class="font-bold text-amber-400">Syncing Market Feed...</span>
                            <span id="syncPctText" class="font-mono text-slate-300">0%</span>
                        </div>
                        <div class="w-full bg-slate-800/80 rounded-full h-1.5 overflow-hidden">
                            <div id="syncProgressBar" class="bg-gradient-to-r from-amber-500 to-emerald-400 h-1.5 rounded-full transition-all duration-300" style="width: 0%"></div>
                        </div>
                    </div>

                    <!-- Step-by-Step Live Log -->
                    <div id="syncLogList" class="space-y-1 text-[10.5px] font-mono">
                        <!-- Dynamically filled with checkmarks -->
                    </div>

                    <!-- Final Confirmation Badge -->
                    <div id="syncSummaryBadge" class="hidden pt-2 border-t border-white/5 space-y-1">
                        <div class="flex items-center gap-1.5 text-emerald-400 font-bold text-xs">
                            <span>✓</span> <span id="syncSuccessHeading">Feed Live & Verified</span>
                        </div>
                        <div id="syncSuccessDetails" class="text-[10px] text-slate-400 leading-tight"></div>
                    </div>
                </div>
            </div>

            <!-- AI Strategist Engine Section -->
            <div class="space-y-3 pt-3 border-t border-white/5">
                <div class="text-xs font-bold text-slate-300 uppercase tracking-wider">AI Strategist Engine</div>
                <div class="space-y-1.5">
                    <label class="text-[11px] text-slate-400 block font-medium">Gemini API Key</label>
                    <input id="sidebarApiKey" type="password" placeholder="Enter Gemini API key..." class="w-full bg-[#13151f] border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-amber-500 font-mono">
                </div>
                <button onclick="saveSidebarApiKey()" class="w-full bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs py-2 px-3 rounded-lg transition">
                    Save Key Permanently
                </button>
                <div id="sidebarApiStatus" class="text-[11px] text-slate-400 flex items-center gap-1.5">
                    <span id="apiStatusDot" class="w-2 h-2 rounded-full bg-amber-500"></span>
                    <span id="apiStatusText">Status: Checking Key...</span>
                </div>
            </div>

            <!-- Sidebar Info / Footer -->
            <div class="pt-4 border-t border-white/5 text-[11px] text-slate-500 space-y-2 mt-auto">
                <div>Data source: Local <code class="text-slate-400 bg-white/5 px-1 py-0.5 rounded">raw_csv</code> + <code class="text-slate-400 bg-white/5 px-1 py-0.5 rounded">yfinance</code> daily sync.</div>
                <div class="flex items-center gap-1.5 text-slate-400 pt-1">
                    <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                    <span>Desktop Parity Active</span>
                </div>
            </div>
        </aside>

        <!-- Main Dashboard View on the Right -->
        <main class="flex-1 min-w-0 p-6 overflow-y-auto">
            <div class="max-w-7xl mx-auto">

        <!-- Hero Brand Header -->
        <div class="flex flex-col md:flex-row justify-between items-start md:items-center pb-5 mb-5 border-b border-white/10 gap-4">
            <div class="flex items-center gap-3.5">
                <div class="w-11 h-11 rounded-xl bg-gradient-to-br from-amber-500 to-amber-600 flex items-center justify-center font-black text-2xl text-slate-950 shadow-lg shadow-amber-500/30">
                    R
                </div>
                <div>
                    <h1 class="text-2xl font-extrabold text-white tracking-tight">Rotato</h1>
                    <p class="text-xs text-slate-400">Quantitative Macro & Confluence Intelligence Desk</p>
                </div>
            </div>
            <div class="flex items-center gap-3">
                <button onclick="promptApiKey()" id="apiKeyBtn" class="bg-[#111319] hover:bg-white/10 border border-white/10 px-3 py-1.5 rounded-full text-xs font-semibold text-slate-300 flex items-center gap-1.5 transition">
                    <span>🔑</span> <span id="apiKeyStatus">Set Gemini Key</span>
                </button>
                <div class="flex items-center gap-2 bg-[#111319] border border-white/10 px-3.5 py-1.5 rounded-full text-xs font-semibold text-slate-300">
                    <span class="w-2 h-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500"></span>
                    <span>Desk Active</span>
                    <span class="text-slate-600">•</span>
                    <span id="sessionDateBadge" class="text-amber-400 font-bold">Prices: Loading...</span>
                </div>
            </div>
        </div>

        <!-- Regime Ribbon Card -->
        <div id="regimeCard" class="lux-card p-4 mb-5 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <div class="flex items-center gap-2">
                    <span class="text-[11px] font-bold text-amber-500 uppercase tracking-wider">Current Market Regime</span>
                    <span id="regimeAiBadge" class="hidden text-[10px] font-bold text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">AI Synthesized</span>
                </div>
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
            <button onclick="switchTab('tab_agent')" id="btn-tab_agent" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold flex items-center gap-1.5"><span>🤖</span> AI Quant Agent</button>
            <button onclick="switchTab('tab2')" id="btn-tab2" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">AI Trade Ideas</button>
            <button onclick="switchTab('tab3')" id="btn-tab3" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Confirmation & Confluence</button>
            <button onclick="switchTab('tab4')" id="btn-tab4" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Mag 7 Diagnostics</button>
            <button onclick="switchTab('tab5')" id="btn-tab5" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Indexes & Sector Flow</button>
            <button onclick="switchTab('tab6')" id="btn-tab6" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Key Ratio & Breadth Radar</button>
            <button onclick="switchTab('tab7')" id="btn-tab7" class="tab-btn px-4 py-2 rounded-lg text-xs font-bold">Single Ticker Explorer</button>
        </div>

        <!-- Tab Agent: Interactive AI Quant Agent -->
        <div id="tab_agent" class="tab-pane hidden">
            <div class="lux-card p-5 mb-4">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-3 mb-4 border-b border-white/10 gap-3">
                    <div>
                        <h2 class="text-lg font-bold text-white flex items-center gap-2"><span>🤖</span> Interactive Quant Analyst & Risk Officer</h2>
                        <p class="text-xs text-slate-400">On-Demand AI Reasoning Across Live Regimes, Tickers, Confluences & Ratios</p>
                    </div>
                    <button onclick="clearAgentChat()" class="text-xs text-slate-400 hover:text-white px-3 py-1 rounded bg-white/5 border border-white/10 transition">Clear Chat</button>
                </div>
                
                <!-- Quick Prompts -->
                <div class="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">Quick Desk Inquiries:</div>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 mb-4">
                    <button onclick="sendQuickPrompt('Analyze todays Mag 7 divergence. Which tech leaders are showing institutional accumulation vs distribution, and what are the key invalidation levels?')" class="p-2.5 text-left rounded-lg bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 text-xs text-slate-300 transition">
                        🔍 <b>Mag 7 Divergence & Risks</b>
                    </button>
                    <button onclick="sendQuickPrompt('Based on the 5-day sector velocities, where is institutional smart money rotating right now? Which sector is leading?')" class="p-2.5 text-left rounded-lg bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 text-xs text-slate-300 transition">
                        🚀 <b>Sector Rotation Leaders</b>
                    </button>
                    <button onclick="sendQuickPrompt('What are the biggest distribution traps and high-risk names on the desk today? Where should retail traders avoid dip buying?')" class="p-2.5 text-left rounded-lg bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 text-xs text-slate-300 transition">
                        ⚠️ <b>Top Distribution Warnings</b>
                    </button>
                    <button onclick="sendQuickPrompt('Give me the top 3 high-probability setups from todays confluence radar with exact entry rationale, targets, and stop-loss levels.')" class="p-2.5 text-left rounded-lg bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 text-xs text-slate-300 transition">
                        🎯 <b>Best High-Conviction Setups</b>
                    </button>
                </div>

                <!-- Chat History Window -->
                <div id="chatHistory" class="space-y-3 min-h-[300px] max-h-[500px] overflow-y-auto p-4 rounded-xl bg-[#090a0f] border border-white/5 mb-4">
                    <!-- Messages dynamically rendered -->
                </div>

                <!-- Chat Input Bar -->
                <form onsubmit="handleAgentSubmit(event)" class="flex gap-2">
                    <input id="chatInput" type="text" placeholder="Ask the Quant AI Agent anything about today's market data, tickers, or setups..." class="flex-1 bg-[#13151f] border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-amber-500">
                    <button type="submit" id="chatSendBtn" class="bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold px-5 py-2.5 rounded-xl text-xs flex items-center gap-1.5 transition">
                        <span>Send</span>
                    </button>
                </form>
            </div>
        </div>

        <!-- Tab 1: AI Macro Briefing -->
        <div id="tab1" class="tab-pane">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div class="lux-card p-5 border-t-4 border-amber-500">
                    <div class="flex justify-between items-center mb-2">
                        <div class="flex items-center gap-2">
                            <span class="text-[11px] font-bold text-amber-500 uppercase tracking-wider">Market Regime</span>
                            <span id="briefingAiBadge" class="hidden text-[10px] font-bold text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">AI Synthesized</span>
                        </div>
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
                <!-- Top Executive Header -->
                <div class="flex flex-wrap justify-between items-center gap-3 pb-3 mb-4 border-b border-white/5">
                    <div class="flex items-center gap-2.5">
                        <span class="text-amber-400 text-lg">🏛️</span>
                        <div>
                            <span class="text-xs font-extrabold text-amber-400 uppercase tracking-wider block">Institutional Executive Synthesis</span>
                            <span class="text-[11px] text-slate-400">Chief Quantitative Strategist & Portfolio Risk Intelligence</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-2">
                        <span id="aiBriefingTimestamp" class="text-[11px] font-mono text-slate-400 bg-white/5 px-2.5 py-1 rounded border border-white/10">Daily Session</span>
                        <span id="aiSourceBadge" class="pill-amber px-2.5 py-1 rounded text-[11px] font-bold">Quantitative Desk</span>
                    </div>
                </div>

                <!-- Navigation View Filter Pills -->
                <div class="flex flex-wrap items-center gap-1.5 mb-4 p-1.5 rounded-xl bg-white/[0.02] border border-white/5">
                    <span class="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-2">Focus Pillar:</span>
                    <button onclick="filterSynthesisSection('all')" id="synthPill-all" class="synth-nav-pill px-3 py-1 rounded-lg text-xs font-bold bg-amber-500 text-black shadow-sm transition">All 4 Pillars</button>
                    <button onclick="filterSynthesisSection('macro')" id="synthPill-macro" class="synth-nav-pill px-3 py-1 rounded-lg text-xs font-semibold bg-white/5 hover:bg-white/10 text-slate-300 transition">🌐 1. Macro Liquidity</button>
                    <button onclick="filterSynthesisSection('sectors')" id="synthPill-sectors" class="synth-nav-pill px-3 py-1 rounded-lg text-xs font-semibold bg-white/5 hover:bg-white/10 text-slate-300 transition">⚡ 2. Sector Velocity</button>
                    <button onclick="filterSynthesisSection('setups')" id="synthPill-setups" class="synth-nav-pill px-3 py-1 rounded-lg text-xs font-semibold bg-white/5 hover:bg-white/10 text-slate-300 transition">🎯 3. Setups & Traps</button>
                    <button onclick="filterSynthesisSection('risk')" id="synthPill-risk" class="synth-nav-pill px-3 py-1 rounded-lg text-xs font-semibold bg-white/5 hover:bg-white/10 text-slate-300 transition">🛡️ 4. Desk Game Plan</button>
                </div>

                <!-- Structured Synthesis Section Cards Container -->
                <div id="synthesisText" class="space-y-4">
                    <div class="flex items-center gap-2 text-slate-400 py-6 text-xs">
                        <span class="inline-block animate-spin text-amber-400">⚡</span>
                        <span>Synthesizing multi-factor institutional market intelligence...</span>
                    </div>
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
                            <th>Ticker</th><th>Price</th><th>Score</th><th>Bias</th><th>Bull Confirmations</th><th>Bear Confirmations</th><th>Status</th><th>20Y Win Rate</th><th>RSI</th><th>RVOL</th><th>vs 50EMA</th><th>Drawdown ATH</th>
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
                            <th>Ticker</th><th>Price</th><th>1D %</th><th>5D %</th><th>20D %</th><th>vs 20EMA</th><th>vs 50EMA</th><th>14D RSI</th><th>RVOL</th><th>Drawdown ATH</th><th>20Y Win Rate</th><th>Median 20D Alpha</th><th>Technical State</th>
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
                            <th>Sector</th><th>Price</th><th>1D %</th><th>5D %</th><th>20D %</th><th>vs 50EMA</th><th>14D RSI</th><th>RVOL</th><th>ATR Stretch</th><th>Drawdown ATH</th><th>State</th>
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
        </main>
    </div>

    <!-- JavaScript Application Logic -->
    <script>
        let DATA = null;
        let CURRENT_HORIZON = "All Historical Data";
        let USER_API_KEY = localStorage.getItem('rotato_gemini_key') || '';

        function updateApiKeyUI() {
            const sidebarInput = document.getElementById('sidebarApiKey');
            const textEl = document.getElementById('apiStatusText');
            const dotEl = document.getElementById('apiStatusDot');
            const headerBtn = document.getElementById('apiKeyStatus');

            if (sidebarInput && document.activeElement !== sidebarInput) {
                sidebarInput.value = USER_API_KEY;
            }
            if (headerBtn) {
                headerBtn.innerText = USER_API_KEY ? 'Gemini Key Active' : 'Set Gemini Key';
            }
            if (textEl && dotEl) {
                if (USER_API_KEY) {
                    textEl.innerText = 'Status: Gemini API Key Active';
                    textEl.className = 'text-emerald-400 font-semibold';
                    dotEl.className = 'w-2 h-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500';
                } else {
                    textEl.innerText = 'Status: Using Built-in Quantitative Heuristic AI';
                    textEl.className = 'text-slate-400';
                    dotEl.className = 'w-2 h-2 rounded-full bg-amber-500';
                }
            }
        }

        function saveSidebarApiKey() {
            const input = document.getElementById('sidebarApiKey');
            const key = input ? input.value.trim() : '';
            if (key) {
                USER_API_KEY = key;
                localStorage.setItem('rotato_gemini_key', key);
            } else {
                USER_API_KEY = '';
                localStorage.removeItem('rotato_gemini_key');
            }
            updateApiKeyUI();
            init();
        }

        function promptApiKey() {
            const key = prompt("Enter your Google Gemini API Key to enable AI Briefings & Agent Chat on Vercel:", USER_API_KEY);
            if (key !== null) {
                USER_API_KEY = key.trim();
                if (USER_API_KEY) {
                    localStorage.setItem('rotato_gemini_key', USER_API_KEY);
                } else {
                    localStorage.removeItem('rotato_gemini_key');
                }
                updateApiKeyUI();
                init();
            }
        }

        async function triggerDataSync() {
            const scope = document.querySelector('input[name="syncScope"]:checked')?.value || 'core';
            const btn = document.getElementById('syncBtn');
            const spinner = document.getElementById('syncSpinner');
            const btnText = document.getElementById('syncBtnText');
            const feedbackBox = document.getElementById('syncFeedbackBox');
            const stageTitle = document.getElementById('syncStageTitle');
            const pctText = document.getElementById('syncPctText');
            const progressBar = document.getElementById('syncProgressBar');
            const logList = document.getElementById('syncLogList');
            const summaryBadge = document.getElementById('syncSummaryBadge');
            const successHeading = document.getElementById('syncSuccessHeading');
            const successDetails = document.getElementById('syncSuccessDetails');

            btn.disabled = true;
            spinner.classList.remove('hidden');
            btnText.innerText = 'Syncing Feeds...';
            feedbackBox.classList.remove('hidden');
            summaryBadge.classList.add('hidden');
            logList.innerHTML = '';

            const appendLog = (icon, text, color = 'text-slate-300') => {
                const div = document.createElement('div');
                div.className = `flex items-center gap-1.5 ${color}`;
                div.innerHTML = `<span>${icon}</span><span>${text}</span>`;
                logList.appendChild(div);
            };

            const setProgress = (pct, stage) => {
                progressBar.style.width = pct + '%';
                pctText.innerText = pct + '%';
                stageTitle.innerText = stage;
            };

            setProgress(15, 'Contacting Yahoo Finance...');
            appendLog('⚡', 'Connecting to Yahoo Finance endpoints...', 'text-amber-300');

            await new Promise(r => setTimeout(r, 200));
            setProgress(35, 'Downloading Tickers Batch...');
            appendLog('📥', `Fetching batch quotes for ${scope === 'core' ? '28 core assets' : 'all 79 assets'}...`, 'text-blue-300');

            try {
                const res = await fetch('/api/sync-data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ scope: scope })
                });

                if (!res.ok) {
                    const errData = await res.json().catch(() => null);
                    const msg = (errData && errData.message) ? errData.message : `Server returned HTTP ${res.status}`;
                    throw new Error(msg);
                }

                const data = await res.json();

                setProgress(75, 'Re-indexing Technical Indicators...');
                appendLog('📊', `Received ${data.total} ticker daily bars. Calculating EMAs & RSI...`, 'text-indigo-300');

                await new Promise(r => setTimeout(r, 200));
                setProgress(90, 'Recalculating Synthetic Ratios...');
                appendLog('🔄', 'Updating SPY/RSP, QQQ/RSP, and IWM/QQQ spreads...', 'text-cyan-300');

                await new Promise(r => setTimeout(r, 200));
                setProgress(100, 'Sync Complete');
                appendLog('✓', `All ${data.total} assets indexed to ${data.last_date}`, 'text-emerald-400 font-bold');

                // Display success badge
                summaryBadge.classList.remove('hidden');
                successHeading.innerText = `Updated ${data.total} Assets (${data.duration}s)`;
                successDetails.innerHTML = `
                    <div>• <b>Session Date:</b> ${data.last_date} (${data.timestamp})</div>
                    <div>• <b>Verified:</b> ${(data.sample_tickers || []).slice(0, 6).join(', ')}...</div>
                `;

                // Pulse the session badge in header
                const sessionBadge = document.getElementById('sessionDateBadge');
                if (sessionBadge) {
                    sessionBadge.innerText = 'Prices: ' + data.last_date;
                    sessionBadge.classList.add('bg-emerald-500/20', 'text-emerald-300');
                    setTimeout(() => sessionBadge.classList.remove('bg-emerald-500/20', 'text-emerald-300'), 3000);
                }

                // Re-render dashboard
                await init();

            } catch (err) {
                setProgress(100, 'Sync Error');
                appendLog('✕', 'Sync failed: ' + err.message, 'text-red-400 font-bold');
            } finally {
                btn.disabled = false;
                spinner.classList.add('hidden');
                btnText.innerText = 'Sync Daily Closes Now';
            }
        }

        const LOADED_TABS = new Set(['tab1']);
        let AI_DATA = null;

        function switchTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.remove('hidden');
            document.getElementById('btn-' + tabId).classList.add('active');

            // Lazy load tabs on demand
            if (tabId === 'tab2') {
                LOADED_TABS.add('tab2');
                renderTradeCards();
            } else if (tabId === 'tab3') {
                if (!LOADED_TABS.has('tab3')) {
                    LOADED_TABS.add('tab3');
                    renderConfluence();
                } else {
                    setTimeout(() => Plotly.Plots.resize('confluenceChart'), 50);
                }
            } else if (tabId === 'tab4') {
                if (!LOADED_TABS.has('tab4')) {
                    LOADED_TABS.add('tab4');
                    renderMag7();
                } else {
                    setTimeout(() => Plotly.Plots.resize('mag7NormChart'), 50);
                }
            } else if (tabId === 'tab5') {
                if (!LOADED_TABS.has('tab5')) {
                    LOADED_TABS.add('tab5');
                    renderSectors();
                } else {
                    setTimeout(() => Plotly.Plots.resize('basketChart'), 50);
                }
            } else if (tabId === 'tab6') {
                if (!LOADED_TABS.has('tab6')) {
                    LOADED_TABS.add('tab6');
                    loadRatio('SPY_RSP');
                } else {
                    setTimeout(() => Plotly.Plots.resize('ratioChart'), 50);
                }
            } else if (tabId === 'tab7') {
                if (!LOADED_TABS.has('tab7')) {
                    LOADED_TABS.add('tab7');
                    populateTickersDropdown();
                    loadSingleTicker();
                } else {
                    setTimeout(() => Plotly.Plots.resize('singleTickerCombinedChart'), 50);
                }
            }
        }

        async function init() {
            updateApiKeyUI();
            try {
                // 1. Fetch core market numerical metrics instantly (< 300ms)
                const res = await fetch('/api/market-data');
                DATA = await res.json();
                
                // 2. Render primary dashboard shell immediately
                renderTickerTape();
                renderRegime();
                renderBenchmarks();
                renderMacroBarometers();
                renderBriefingBaskets();
                initAgentChat();

                // 3. Asynchronously load AI briefing in background without blocking UI
                loadAiBriefingAsync();

            } catch (err) {
                console.error("Failed to load market data:", err);
            }
        }

        let AGENT_CHAT_HISTORY = [
            {
                role: 'assistant',
                content: "Greetings Trader. I am your Quantitative Desk Analyst. I have indexed today's live metrics (20/50/200 EMAs, RSI, ATR stretches, RVOL, sector velocities, and confluence scores). Ask me about any ticker, setup, or rotation risk."
            }
        ];

        function initAgentChat() {
            renderChatHistory();
        }

        function renderChatHistory() {
            const container = document.getElementById('chatHistory');
            if (!container) return;
            let html = '';
            AGENT_CHAT_HISTORY.forEach(msg => {
                const isUser = msg.role === 'user';
                const bg = isUser ? 'bg-amber-500/10 border-amber-500/20 text-amber-100 ml-6' : 'bg-[#12141c] border-white/5 text-slate-200 mr-6';
                const label = isUser ? '<b class="text-amber-400 text-[11px] block mb-1">YOU</b>' : '<b class="text-slate-400 text-[11px] block mb-1">QUANT AI AGENT</b>';
                const parsedContent = window.marked ? marked.parse(msg.content) : msg.content;
                html += `
                    <div class="p-3 rounded-xl border ${bg}">
                        ${label}
                        <div class="text-xs leading-relaxed prose prose-invert max-w-none">${parsedContent}</div>
                    </div>
                `;
            });
            container.innerHTML = html;
            container.scrollTop = container.scrollHeight;
        }

        function clearAgentChat() {
            AGENT_CHAT_HISTORY = [
                {
                    role: 'assistant',
                    content: 'Chat history cleared. What quantitative setups or market dynamics would you like to review?'
                }
            ];
            renderChatHistory();
        }

        function sendQuickPrompt(promptText) {
            document.getElementById('chatInput').value = promptText;
            handleAgentSubmit(null);
        }

        async function handleAgentSubmit(e) {
            if (e && e.preventDefault) e.preventDefault();
            const input = document.getElementById('chatInput');
            const query = input.value.trim();
            if (!query) return;

            AGENT_CHAT_HISTORY.push({ role: 'user', content: query });
            input.value = '';
            renderChatHistory();

            const container = document.getElementById('chatHistory');
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'p-3 rounded-xl border bg-[#12141c] border-white/5 text-slate-400 mr-6 text-xs';
            loadingDiv.innerHTML = '<b class="text-slate-500 text-[11px] block mb-1">QUANT AI AGENT</b><span class="animate-pulse">Analyzing quantitative data & generating desk response...</span>';
            container.appendChild(loadingDiv);
            container.scrollTop = container.scrollHeight;

            try {
                const res = await fetch('/api/agent-chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        api_key: USER_API_KEY,
                        history: AGENT_CHAT_HISTORY
                    })
                });
                const data = await res.json();
                container.removeChild(loadingDiv);
                AGENT_CHAT_HISTORY.push({ role: 'assistant', content: data.answer || 'No response.' });
                renderChatHistory();
            } catch (err) {
                container.removeChild(loadingDiv);
                AGENT_CHAT_HISTORY.push({ role: 'assistant', content: '⚠️ Error connecting to AI Agent: ' + err.message });
                renderChatHistory();
            }
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
            const sessionBadge = document.getElementById('sessionDateBadge');
            if (sessionBadge) {
                sessionBadge.innerText = 'Prices: ' + (DATA.last_market_session || '2026-10-27');
            }
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
                    pct_5d: spy_rsp.pct_5d,
                    pct_20d: spy_rsp.pct_20d,
                    dist_ema20: spy_rsp.dist_ema20,
                    dist_ema50: spy_rsp.dist_ema50,
                    state: spy_rsp.actionable_state || "Concentration Peaking",
                    action: spy_rsp.action_text || "Rotate into Equal-Weight (RSP)",
                    color: spy_rsp.state_color || "#f59e0b"
                },
                { 
                    name: "IWM / QQQ (Rotation)", 
                    p: iwm_qqq.current, 
                    sigma: iwm_qqq.z_score_50d, 
                    pct_5d: iwm_qqq.pct_5d,
                    pct_20d: iwm_qqq.pct_20d,
                    dist_ema20: iwm_qqq.dist_ema20,
                    dist_ema50: iwm_qqq.dist_ema50,
                    state: iwm_qqq.actionable_state || "Small-Cap Coiling",
                    action: iwm_qqq.action_text || "Early Mean-Reversion Bid in IWM",
                    color: iwm_qqq.state_color || "#10b981"
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
                    const sign5d = (card.pct_5d || 0) >= 0 ? '+' : '';
                    const sign20d = (card.pct_20d || 0) >= 0 ? '+' : '';
                    const signEma20 = (card.dist_ema20 || 0) >= 0 ? '+' : '';
                    const signEma50 = (card.dist_ema50 || 0) >= 0 ? '+' : '';
                    subHtml = `
                        <div class="text-[11px] text-slate-400 mt-1 flex flex-col gap-0.5">
                            <div><b>5D:</b> <span class="${(card.pct_5d || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'} font-bold">${sign5d}${card.pct_5d}%</span> • <b>20D:</b> <span class="${(card.pct_20d || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'} font-bold">${sign20d}${card.pct_20d}%</span> • <span style="color:${card.color};" class="font-bold">${card.state}</span></div>
                            <div class="text-[10px] text-slate-400"><b>vs 20EMA:</b> <span class="${(card.dist_ema20 || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}">${signEma20}${card.dist_ema20}%</span> &nbsp;|&nbsp; <b>vs 50EMA:</b> <span class="${(card.dist_ema50 || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}">${signEma50}${card.dist_ema50}%</span></div>
                            <div class="text-[10.5px] text-slate-300"><b>Action:</b> ${card.action}</div>
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

        function renderBriefingBaskets() {
            const md = DATA.market_data || {};
            document.getElementById('briefingTitle').innerText = md.narrative_title || "Secular Uptrend";
            document.getElementById('briefingDetail').innerText = md.narrative_detail || "";
            document.getElementById('briefingRiskPill').innerText = md.risk_level || "Moderate";

            const b = md.baskets_5d || {};
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
        }

        async function loadAiBriefingAsync() {
            const synthBox = document.getElementById('synthesisText');
            if (synthBox) {
                synthBox.innerHTML = `
                    <div class="flex items-center gap-2 text-slate-400 py-3 text-xs">
                        <span class="inline-block animate-spin text-amber-400">⚡</span>
                        <span>AI Quant Strategist synthesizing live market regimes & setups...</span>
                    </div>
                `;
            }

            try {
                const url = USER_API_KEY ? `/api/ai-briefing?api_key=${encodeURIComponent(USER_API_KEY)}` : '/api/ai-briefing';
                const res = await fetch(url);
                AI_DATA = await res.json();
                renderAiBriefingSynthesis();
                if (LOADED_TABS.has('tab2')) {
                    renderTradeCards();
                }
            } catch (err) {
                if (synthBox) {
                    synthBox.innerHTML = `<div class="text-xs text-red-400 py-2">AI Briefing unavailable: ${err.message}</div>`;
                }
            }
        }

        let CURRENT_SYNTH_FILTER = 'all';

        function filterSynthesisSection(key) {
            CURRENT_SYNTH_FILTER = key;
            const buttons = document.querySelectorAll('.synth-nav-pill');
            for (let i = 0; i < buttons.length; i++) {
                const btn = buttons[i];
                if (btn.id === 'synthPill-' + key) {
                    btn.className = 'synth-nav-pill px-3 py-1 rounded-lg text-xs font-bold bg-amber-500 text-black shadow-sm transition';
                } else {
                    btn.className = 'synth-nav-pill px-3 py-1 rounded-lg text-xs font-semibold bg-white/5 hover:bg-white/10 text-slate-300 transition';
                }
            }
            const cards = document.querySelectorAll('.synth-section-card');
            for (let i = 0; i < cards.length; i++) {
                const card = cards[i];
                if (key === 'all' || card.getAttribute('data-section') === key) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            }
        }

        function buildStructuredSynthesisCards(summary) {
            if (!summary) return '';
            const splitRegex = /(?:\\r?\\n\\s*---\\s*\\r?\\n|\\r?\\n+)(?=###?\\s+[0-9IVX]+\\.)/;
            const chunks = summary.split(splitRegex);
            
            let cardsHtml = '';
            let metaHtml = '';
            
            const configs = [
                { key: 'macro', icon: '🌐', title: 'Macro Regime & Cross-Asset Liquidity', badge: 'AI Macro Liquidity', border: 'border-sky-500/30', bg: 'from-[#0b1322] to-[#080d17]', titleColor: 'text-sky-400' },
                { key: 'sectors', icon: '⚡', title: 'Sector Velocity & Breadth Concentration Dynamics', badge: 'AI Sector Dynamics', border: 'border-purple-500/30', bg: 'from-[#140e24] to-[#0b0814]', titleColor: 'text-purple-400' },
                { key: 'setups', icon: '🎯', title: 'Actionable Setups & Trade Confluence', badge: 'AI Confluence Setups', border: 'border-emerald-500/30', bg: 'from-[#091a14] to-[#06100d]', titleColor: 'text-emerald-400' },
                { key: 'risk', icon: '🛡️', title: 'Desk Game Plan & Tactical Risk Boundaries', badge: 'AI Tactical Boundaries', border: 'border-amber-500/30', bg: 'from-[#1a140a] to-[#100d07]', titleColor: 'text-amber-400' }
            ];
            
            let secIdx = 0;
            
            for (let i = 0; i < chunks.length; i++) {
                const chunk = chunks[i].trim();
                if (!chunk) continue;
                
                const lines = chunk.split(String.fromCharCode(10));
                const firstLine = lines[0].replace(/^[#\\s*]+/, '').trim();
                
                if (firstLine.indexOf('DAILY INSTITUTIONAL') >= 0 || firstLine.indexOf('Classification') >= 0 || firstLine.indexOf('Author:') >= 0) {
                    const dateMatch = chunk.match(/\\*\\*Date:\\*\\*\\s*([^\\n*]+)/);
                    const dateStr = dateMatch ? dateMatch[1].trim() : '';
                    metaHtml = '<div class="p-3 mb-3 rounded-xl bg-white/[0.02] border border-white/5 flex flex-wrap justify-between items-center gap-2 text-xs">' +
                               '<div class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span><span class="font-bold text-slate-200">Daily Institutional Macro Intelligence</span>' +
                               (dateStr ? '<span class="text-slate-500">•</span><span class="text-amber-400 font-mono">' + dateStr + '</span>' : '') +
                               '</div><span class="text-[11px] text-amber-400 font-bold uppercase tracking-wider bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded">100% AI Synthesized</span></div>';
                    continue;
                }
                
                let cfg = configs[secIdx % configs.length];
                if (firstLine.indexOf('1') === 0 || firstLine.indexOf('Macro Regime') >= 0 || firstLine.indexOf('Liquidity') >= 0) cfg = configs[0];
                else if (firstLine.indexOf('2') === 0 || firstLine.indexOf('Sector Velocity') >= 0 || firstLine.indexOf('Breadth') >= 0) cfg = configs[1];
                else if (firstLine.indexOf('3') === 0 || firstLine.indexOf('Actionable Setups') >= 0 || firstLine.indexOf('Trade Confluence') >= 0) cfg = configs[2];
                else if (firstLine.indexOf('4') === 0 || firstLine.indexOf('Desk Game Plan') >= 0 || firstLine.indexOf('Risk Boundaries') >= 0) cfg = configs[3];
                secIdx++;
                
                const bodyText = lines.slice(1).join(String.fromCharCode(10)).trim();
                let parsed = window.marked ? marked.parse(bodyText) : bodyText;
                
                parsed = parsed.replace(/<li>([\\s\\S]*?)<\\/li>/g, function(m, inner) {
                    const low = inner.toLowerCase();
                    let cls = '';
                    if (low.indexOf('long') >= 0 || low.indexOf('bid') >= 0 || low.indexOf('continuation') >= 0 || low.indexOf('leading') >= 0 || low.indexOf('growth') >= 0) {
                        cls = 'card-long';
                    } else if (low.indexOf('distribution') >= 0 || low.indexOf('trap') >= 0 || low.indexOf('fade') >= 0 || low.indexOf('warning') >= 0 || low.indexOf('avoid') >= 0 || low.indexOf('lagging') >= 0) {
                        cls = 'card-warn';
                    } else if (low.indexOf('invalidation') >= 0 || low.indexOf('add risk') >= 0 || low.indexOf('hedge') >= 0 || low.indexOf('trim') >= 0 || low.indexOf('parameter') >= 0) {
                        cls = 'card-risk';
                    } else if (low.indexOf('tlt') >= 0 || low.indexOf('gld') >= 0 || low.indexOf('uso') >= 0 || low.indexOf('ibit') >= 0 || low.indexOf('treasur') >= 0 || low.indexOf('gold') >= 0 || low.indexOf('crude') >= 0) {
                        cls = 'card-macro';
                    }
                    return '<li class="' + cls + '">' + inner + '</li>';
                });
                
                cardsHtml += '<div class="synth-section-card lux-card p-4 sm:p-5 mb-4 border ' + cfg.border + ' bg-gradient-to-b ' + cfg.bg + '" data-section="' + cfg.key + '">' +
                             '<div class="flex flex-wrap justify-between items-center gap-2 pb-3 mb-3 border-b border-white/5">' +
                             '<div class="flex items-center gap-2.5"><span class="text-lg">' + cfg.icon + '</span><span class="text-sm font-extrabold ' + cfg.titleColor + ' uppercase tracking-wider">' + firstLine + '</span></div>' +
                             '<div class="flex items-center gap-1.5"><span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 border border-amber-500/20 text-amber-400 uppercase">AI Synthesized</span><span class="px-2 py-0.5 rounded text-[10.5px] font-bold bg-white/5 border border-white/10 text-slate-300 uppercase">' + cfg.badge + '</span></div></div>' +
                             '<div class="synth-body space-y-2 text-xs sm:text-sm text-slate-300 leading-relaxed">' + parsed + '</div></div>';
            }
            return metaHtml + cardsHtml;
        }

        function renderAiBriefingSynthesis() {
            if (!AI_DATA) return;
            const summary = AI_DATA.llm_summary || "";
            const synthBox = document.getElementById('synthesisText');
            const sourceBadge = document.getElementById('aiSourceBadge');

            if (sourceBadge) sourceBadge.innerText = AI_DATA.source || "Google Gemini";

            // Dynamically update the Regime and Macro cards with AI's data understanding
            if (AI_DATA.regime_title) {
                const bTitle = document.getElementById('briefingTitle');
                if (bTitle) bTitle.innerText = AI_DATA.regime_title;
                const rTitle = document.getElementById('regimeTitle');
                if (rTitle) rTitle.innerText = AI_DATA.regime_title;
            }
            if (AI_DATA.regime_detail) {
                const bDetail = document.getElementById('briefingDetail');
                if (bDetail) bDetail.innerText = AI_DATA.regime_detail;
                const rDetail = document.getElementById('regimeDetail');
                if (rDetail) rDetail.innerText = AI_DATA.regime_detail;
            }
            if (AI_DATA.risk_level) {
                const bPill = document.getElementById('briefingRiskPill');
                if (bPill) bPill.innerText = AI_DATA.risk_level;
                const rBadge = document.getElementById('regimeRiskBadge');
                if (rBadge) rBadge.innerText = AI_DATA.risk_level;
            }
            if (AI_DATA.macro_context) {
                const mCtx = document.getElementById('macroContextText');
                if (mCtx) mCtx.innerText = AI_DATA.macro_context;
            }
            if (AI_DATA.is_ai_derived) {
                const bAi = document.getElementById('briefingAiBadge');
                if (bAi) bAi.classList.remove('hidden');
                const rAi = document.getElementById('regimeAiBadge');
                if (rAi) rAi.classList.remove('hidden');
            }

            if (!summary) {
                synthBox.innerHTML = `
                    <div class="synth-section-card lux-card p-5 border border-amber-500/20 bg-amber-950/10 space-y-4">
                        <div class="flex items-center justify-between pb-3 border-b border-white/5">
                            <div class="flex items-center gap-2 text-amber-400 font-bold text-xs uppercase tracking-wider">
                                <span class="animate-pulse">⚡</span>
                                <span>AI Institutional Synthesis Engine (Pillars 1 to 4)</span>
                            </div>
                            <span class="pill-amber text-[10.5px] font-bold">Awaiting AI Briefing</span>
                        </div>
                        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                            <div class="p-3.5 rounded-xl bg-black/20 border border-sky-500/20 space-y-1.5">
                                <div class="text-sky-400 font-extrabold uppercase text-[11px] flex items-center gap-1.5"><span>🌐</span> Pillar 1: Macro Liquidity</div>
                                <div class="text-slate-300">${AI_DATA.regime_title || "Synthesizing yields, gold, oil & crypto liquidity..."}</div>
                            </div>
                            <div class="p-3.5 rounded-xl bg-black/20 border border-purple-500/20 space-y-1.5">
                                <div class="text-purple-400 font-extrabold uppercase text-[11px] flex items-center gap-1.5"><span>⚡</span> Pillar 2: Sector Velocity</div>
                                <div class="text-slate-300">Tracking 5D capital rotation across Growth, Cyclicals, Defensives & Breadth...</div>
                            </div>
                            <div class="p-3.5 rounded-xl bg-black/20 border border-emerald-500/20 space-y-1.5">
                                <div class="text-emerald-400 font-extrabold uppercase text-[11px] flex items-center gap-1.5"><span>🎯</span> Pillar 3: Actionable Setups</div>
                                <div class="text-slate-300">Filtering algorithmic high-conviction long bids & distribution traps...</div>
                            </div>
                            <div class="p-3.5 rounded-xl bg-black/20 border border-amber-500/20 space-y-1.5">
                                <div class="text-amber-400 font-extrabold uppercase text-[11px] flex items-center gap-1.5"><span>🛡️</span> Pillar 4: Tactical Game Plan</div>
                                <div class="text-slate-300">Formulating allocation zones, portfolio hedges & hard invalidation triggers...</div>
                            </div>
                        </div>
                    </div>
                `;
                return;
            }

            synthBox.innerHTML = buildStructuredSynthesisCards(summary);
            filterSynthesisSection(CURRENT_SYNTH_FILTER);
        }

        function renderTradeCards() {
            const container = document.getElementById('tradeCardsContainer');
            if (!container) return;
            const cards = (AI_DATA && AI_DATA.trade_cards) ? AI_DATA.trade_cards : [];
            if (!AI_DATA) {
                container.innerHTML = `
                    <div class="flex items-center gap-2 text-slate-400 py-4 text-xs">
                        <span class="inline-block animate-spin text-amber-400">⚡</span>
                        <span>Filtering algorithmic high-probability trade setups...</span>
                    </div>
                `;
                return;
            }
            if (cards.length === 0) {
                container.innerHTML = '<div class="text-xs text-slate-400">No high-confluence setups detected in current market conditions.</div>';
                return;
            }
            let html = '';
            cards.forEach(t => {
                    const dir = t.direction || "LONG";
                    const pill = dir.includes("LONG") ? 'pill-green' : (dir.includes("SHORT") || dir.includes("FADE") ? 'pill-red' : 'pill-amber');
                    const execPill = t.execution_pill || (dir.includes("LONG") ? 'pill-green' : 'pill-amber');
                    html += `
                        <div class="lux-card p-4 sm:p-5 border-l-4" style="border-left-color: ${t.card_color || '#10b981'}">
                            <div class="flex flex-wrap justify-between items-center gap-2 pb-3 mb-3 border-b border-white/5">
                                <div class="flex flex-wrap items-center gap-2.5">
                                    <span class="text-2xl font-black text-white">${t.ticker}</span>
                                    <span class="${pill} px-2.5 py-0.5 rounded text-xs font-bold">${t.direction}</span>
                                    <span class="text-xs font-semibold text-slate-400">${t.type_label}</span>
                                    ${t.execution_state ? `<span class="${execPill} px-2.5 py-0.5 rounded text-[11px] font-bold tracking-wide">${t.execution_state}</span>` : ''}
                                </div>
                                <div class="flex items-center gap-2 text-xs">
                                    <span class="text-slate-500 uppercase">Conviction:</span>
                                    <span class="${pill} px-2 py-0.5 rounded font-bold">Score: ${t.score} (${t.bias_label})</span>
                                </div>
                            </div>

                            <!-- Market Price vs Optimized R:R Banner -->
                            <div class="flex flex-wrap items-center justify-between gap-2 p-2.5 mb-3 rounded-lg bg-white/[0.02] border border-white/5 text-xs">
                                <div class="flex items-center gap-2">
                                    <span class="text-slate-400">Current Print:</span>
                                    <span class="font-bold text-white">${t.current_price || t.entry}</span>
                                    <span class="text-slate-500">|</span>
                                    <span class="text-slate-400">Moving Avg Extension:</span>
                                    <span class="font-mono ${t.extension_metric && t.extension_metric.includes('+') ? 'text-amber-400' : 'text-slate-300'}">${t.extension_metric || 'At Support'}</span>
                                </div>
                                <div class="flex items-center gap-2">
                                    <span class="text-slate-500 uppercase text-[10px]">Optimized Payoff:</span>
                                    <span class="pill-amber px-2.5 py-0.5 rounded text-xs font-black tracking-wide">${t.risk_reward || '1 : 3.0 R:R'}</span>
                                </div>
                            </div>

                            <!-- 4-Box Execution Strategy Grid -->
                            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
                                <div class="bg-gradient-to-b from-amber-500/10 to-transparent p-3 rounded-xl border border-amber-500/25">
                                    <div class="text-[10px] font-bold text-amber-400 uppercase tracking-wider">Optimal Limit Entry</div>
                                    <div class="text-base font-extrabold text-white mt-0.5">${t.optimal_entry_zone || t.entry}</div>
                                    <div class="text-[10px] text-slate-400 mt-1">${t.optimal_entry_label || '20 EMA Support Zone'}</div>
                                </div>
                                <div class="bg-white/[0.02] p-3 rounded-xl border border-white/5">
                                    <div class="text-[10px] font-bold text-sky-400 uppercase tracking-wider">Breakout Trigger</div>
                                    <div class="text-sm font-bold text-white mt-0.5 leading-snug">${t.breakout_trigger || 'Above 5D Pivot'}</div>
                                    <div class="text-[10px] text-slate-400 mt-1">Momentum Confirmation</div>
                                </div>
                                <div class="bg-gradient-to-b from-red-500/10 to-transparent p-3 rounded-xl border border-red-500/25">
                                    <div class="text-[10px] font-bold text-red-400 uppercase tracking-wider">Stop Invalidation</div>
                                    <div class="text-base font-extrabold text-red-400 mt-0.5">${t.stop} <span class="text-xs font-normal text-red-300">(${t.stop_pct})</span></div>
                                    <div class="text-[10px] text-slate-400 mt-1">Below Structural Support</div>
                                </div>
                                <div class="bg-gradient-to-b from-emerald-500/10 to-transparent p-3 rounded-xl border border-emerald-500/25">
                                    <div class="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Profit Targets</div>
                                    <div class="text-base font-extrabold text-emerald-400 mt-0.5">${t.target1} <span class="text-xs font-normal text-emerald-300">(${t.target1_pct})</span></div>
                                    <div class="text-[10px] text-slate-400 mt-1">Runner: ${t.target2} (${t.target2_pct})</div>
                                </div>
                            </div>

                            <!-- Execution Rule Directive -->
                            <div class="p-2.5 rounded-lg bg-black/40 border border-white/5 text-xs text-slate-300 mb-3 flex items-start gap-2">
                                <span class="text-amber-400 text-sm leading-none mt-0.5">💡</span>
                                <div><b class="text-amber-400">Desk Execution Directive:</b> ${t.execution_desc || 'Bidding the structural liquidity band rather than market buying at extended highs.'}</div>
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
                        <td class="${(m.dist_ema50 ?? m.dist_sma50 ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${(m.dist_ema50 ?? m.dist_sma50 ?? 0) >= 0 ? '+' : ''}${(m.dist_ema50 ?? m.dist_sma50 ?? 0).toFixed(2)}%</td>
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
                        <td class="${(d.dist_ema50 ?? d.dist_sma50) >= 0 ? 'text-emerald-400' : 'text-red-400'}">${(d.dist_ema50 ?? d.dist_sma50) >= 0 ? '+' : ''}${d.dist_ema50 ?? d.dist_sma50}%</td>
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
                        <td class="${(d.dist_ema50 ?? d.dist_sma50) >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold">${(d.dist_ema50 ?? d.dist_sma50) >= 0 ? '+' : ''}${d.dist_ema50 ?? d.dist_sma50}%</td>
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
                { x: data.dates, y: data.ema20, name: "20-Day EMA", mode: "lines", line: { color: "#06b6d4", width: 1.5 } },
                { x: data.dates, y: data.ema50 || data.sma50, name: "50-Day EMA", mode: "lines", line: { color: "#f59e0b", width: 1.5 } },
                { x: data.dates, y: data.close, name: ratioName.replace('_', '/'), mode: "lines", line: { color: "#3b82f6", width: 2.5 } }
            ];
            const layout = {
                title: { text: `${ratioName.replace('_', '/')} Ratio with 20/50 EMA & ±2σ Envelopes`, font: { color: '#ffffff', size: 15 } },
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
                    <div class="text-[11px] font-semibold text-slate-400 uppercase">50-Day EMA Trend</div>
                    <div class="text-xl font-bold text-white mt-1">$${m.ema50 || m.sma50 || m.close}</div>
                    <div class="text-xs ${(m.dist_ema50 ?? m.dist_sma50) >= 0 ? 'text-emerald-400' : 'text-red-400'} font-semibold mt-0.5">${(m.dist_ema50 ?? m.dist_sma50) >= 0 ? '+' : ''}${m.dist_ema50 ?? m.dist_sma50}% vs 50EMA</div>
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
