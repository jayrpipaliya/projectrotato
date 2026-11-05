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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

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
    
    # 5-session (6-close) real price trajectory for smooth SVG sparklines
    sparklines = {}
    for sym in ["SPY", "QQQ", "IWM", "GLD", "TLT", "IBIT", "USO"]:
        df_sym = load_ticker_data(sym)
        if not df_sym.empty and "Close" in df_sym.columns:
            sparklines[sym] = [round(float(c), 2) for c in df_sym["Close"].tail(6).tolist()]
    
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
        "sparklines": sparklines,
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
    api_key = payload.get("api_key") or os.getenv("GEMINI_API_KEY") or DEFAULT_GEMINI_KEY
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

@app.get("/api/options/{ticker}")
def api_options(ticker: str, direction: str = Query("Bullish"), side: str = Query("Buy")):
    try:
        t_up = ticker.upper()
        dir_clean = "Bullish" if "Bullish" in str(direction) else "Bearish"
        side_clean = "Sell" if "Sell" in str(side) else "Buy"
        from options_engine import compute_realized_volatilities, fetch_term_structure, scan_cross_chain_recommendations, analyze_smile_and_skew
        df_hist = load_ticker_data(t_up)
        hvs = compute_realized_volatilities(df_hist, ticker_symbol=t_up)
        term_df, spot_price, next_earnings = fetch_term_structure(t_up, hvs_dict=hvs)
        if term_df.empty or spot_price is None:
            return {"error": f"Live options data not available for {t_up}"}
            
        chains_list = term_df.to_dict('records')
        picks = scan_cross_chain_recommendations(t_up, chains_list, spot_price, direction=dir_clean, side=side_clean)
        
        top_exp = term_df.iloc[0]['Expiration']
        smile = analyze_smile_and_skew(t_up, top_exp, spot_price, direction=dir_clean, side=side_clean)
        smile_payload = None
        if smile:
            target_df = smile['target_df']
            k_smooth = np.linspace(target_df['Log_Moneyness'].min(), target_df['Log_Moneyness'].max(), 50)
            y_smooth = np.polyval(smile['poly_coeffs_k'], k_smooth)
            x_strikes_smooth = spot_price * np.exp(k_smooth)
            top_edges = smile['top_strikes']['Bid_Edge'] if side_clean == 'Sell' else smile['top_strikes']['Ask_Edge']
            smile_payload = {
                "expiration": top_exp,
                "dte": smile['dte'],
                "exp_move": round(smile['exp_move'], 2),
                "model_strikes": [round(float(s), 2) for s in x_strikes_smooth],
                "model_iv": [round(float(iv), 2) for iv in y_smooth],
                "market_strikes": [round(float(s), 2) for s in target_df['strike']],
                "market_iv": [round(float(iv), 2) for iv in target_df['IV_Pct']],
                "market_ask_iv": [round(float(iv), 2) for iv in target_df['Ask_IV_Pct']],
                "market_bid_iv": [round(float(iv), 2) for iv in target_df['Bid_IV_Pct']],
                "top_strikes": [round(float(s), 2) for s in smile['top_strikes']['strike']],
                "top_strikes_iv": [round(float(iv), 2) for iv in smile['top_strikes']['IV_Pct']],
                "top_strikes_delta": [round(float(d), 2) for d in smile['top_strikes'].get('Est_Delta', [0]*len(smile['top_strikes']))],
                "top_strikes_edge": [round(float(e), 2) for e in top_edges]
            }
            
        return clean_json({
            "ticker": t_up,
            "direction": dir_clean,
            "side": side_clean,
            "spot_price": round(spot_price, 2),
            "next_earnings": next_earnings.strftime("%b %d, %Y") if next_earnings else None,
            "hvs": hvs,
            "term_chains": chains_list[:8],
            "picks": picks,
            "smile": smile_payload
        })
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
def index_page():
    return HTML_CONTENT

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Rotato // Stealth Quantitative Macro Desk</title>
  
  <!-- Inter Google Font with full weights -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap" rel="stylesheet">
  
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            stealth: {
              950: '#08090C', // Root background
              900: '#0D0F14', // Base KPI Card
              850: '#131720', // Elevated buttons & inputs
              800: '#181E2A', // Hover
              700: '#1F2637', // Active borders & subtle pills
              600: '#2A3449',
              border: '#1B212D',
              'border-subtle': '#141822',
              'border-focus': '#262E3E'
            },
            mint: {
              DEFAULT: '#00F5A0',
              hover: '#00DB8F',
              surface: 'rgba(0, 245, 160, 0.08)',
              glow: 'rgba(0, 245, 160, 0.22)',
              border: 'rgba(0, 245, 160, 0.3)',
              text: '#5CFFC1'
            },
            cherry: {
              DEFAULT: '#FF2E55',
              hover: '#E61E44',
              surface: 'rgba(255, 46, 85, 0.08)',
              glow: 'rgba(255, 46, 85, 0.22)',
              border: 'rgba(255, 46, 85, 0.3)',
              text: '#FF6682'
            }
          },
          fontFamily: {
            sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
          }
        }
      }
    }
  </script>
  <style>
    /* Global Typography: Inter with Tabular Numerals */
    * {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    body {
      background-color: #08090C;
      color: #f1f5f9;
    }

    /* Tabular numbers for clean financial alignment */
    .tabular-num, [data-num], table, .kpi-val, .pill-num {
      font-variant-numeric: tabular-nums lining-nums;
      font-feature-settings: 'tnum' 1, 'zero' 1, 'cv01' 1;
      letter-spacing: -0.015em;
    }

    /* Stealth Cards */
    .stealth-card, .lux-card {
      background-color: #0D0F14;
      border: 1px solid #1B212D;
      border-radius: 10px;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .stealth-card:hover, .lux-card:hover {
      border-color: #262E3E;
    }

    /* Accents */
    .badge-mint, .pill-green {
      background-color: rgba(0, 245, 160, 0.10);
      color: #00F5A0;
      border: 1px solid rgba(0, 245, 160, 0.25);
      border-radius: 6px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 600;
    }
    .badge-cherry, .pill-red {
      background-color: rgba(255, 46, 85, 0.10);
      color: #FF2E55;
      border: 1px solid rgba(255, 46, 85, 0.25);
      border-radius: 6px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 600;
    }
    .pill-cyan {
      background-color: rgba(56, 189, 248, 0.10);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 6px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 600;
    }
    .pill-amber {
      background-color: rgba(245, 158, 11, 0.10);
      color: #f59e0b;
      border: 1px solid rgba(245, 158, 11, 0.25);
      border-radius: 6px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 600;
    }

    /* Custom subtle radio styling */
    .stealth-radio {
      appearance: none;
      width: 14px;
      height: 14px;
      border: 1px solid #2A3449;
      border-radius: 50%;
      background-color: #08090C;
      display: inline-grid;
      place-content: center;
      cursor: pointer;
    }
    .stealth-radio:checked {
      border-color: #00F5A0;
      box-shadow: 0 0 8px rgba(0, 245, 160, 0.4);
    }
    .stealth-radio:checked::before {
      content: "";
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background-color: #00F5A0;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
      width: 5px;
      height: 5px;
    }
    ::-webkit-scrollbar-track {
      background: #08090C;
    }
    ::-webkit-scrollbar-thumb {
      background: #1B212D;
      border-radius: 2px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #2A3449;
    }

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
      background: #0D0F14;
      border: 1px solid #1B212D;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 13px;
      line-height: 1.6;
      color: #cbd5e1;
      list-style: none;
      transition: all 0.15s ease;
    }
    .synth-body li:hover {
      border-color: #262E3E;
    }
    .synth-body li.card-long {
      border-left: 3px solid #00F5A0;
      background: rgba(0, 245, 160, 0.05);
    }
    .synth-body li.card-warn {
      border-left: 3px solid #FF2E55;
      background: rgba(255, 46, 85, 0.05);
    }
    .synth-body li.card-risk {
      border-left: 3px solid #FF2E55;
      background: rgba(255, 46, 85, 0.05);
    }
    .synth-body li.card-macro {
      border-left: 3px solid #00F5A0;
      background: rgba(0, 245, 160, 0.05);
    }
    .synth-body li.card-neutral {
      border-left: 3px solid rgba(255, 255, 255, 0.12);
      background: rgba(255, 255, 255, 0.02);
    }
    .synth-body h4 {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #e2e8f0;
      margin-top: 14px;
      margin-bottom: 6px;
      padding-bottom: 5px;
      border-bottom: 1px solid #1B212D;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .synth-body strong, .synth-body b {
      color: #ffffff;
      font-weight: 600;
    }
    .synth-body p {
      margin-bottom: 8px;
      font-size: 13px;
      line-height: 1.6;
      color: #cbd5e1;
    }
  </style>
</head>
<body class="min-h-screen text-slate-100 flex antialiased selection:bg-mint selection:text-black">

  <!-- ================= LEFT SIDEBAR ================= -->
  <aside class="w-72 bg-stealth-950 border-r border-stealth-border flex flex-col justify-between shrink-0 min-h-screen p-5 hidden md:flex">
    
    <div class="space-y-6">
      
      <!-- Brand Logo Header -->
      <div class="flex items-center space-x-3 pb-2">
        <div class="w-9 h-9 rounded-lg bg-stealth-900 border border-stealth-border flex items-center justify-center font-bold text-lg text-white shadow-[0_0_15px_rgba(0,245,160,0.15)] relative shrink-0">
          <span class="text-white font-extrabold">R</span>
          <span class="w-1.5 h-1.5 rounded-full bg-mint absolute top-1.5 right-1.5 shadow-[0_0_6px_#00F5A0]"></span>
        </div>
        <div>
          <h1 class="font-bold text-base tracking-tight text-white leading-tight">Rotato</h1>
          <p class="text-[11px] text-slate-500 font-medium">Stealth Macro Desk</p>
        </div>
      </div>

      <!-- Desk Status & Engine Badges (Moved to Left Side Menu) -->
      <div class="space-y-2 pb-2 border-b border-stealth-border/80">
        <!-- Desk Active Pill -->
        <div class="flex items-center justify-between px-3 py-2 rounded-lg bg-stealth-900 border border-stealth-border text-xs shadow-sm">
          <div class="flex items-center space-x-2">
            <span class="relative flex h-2 w-2">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-mint opacity-75"></span>
              <span class="relative inline-flex rounded-full h-2 w-2 bg-mint"></span>
            </span>
            <span class="text-slate-200 font-semibold text-xs tracking-tight">Desk Active</span>
          </div>
          <span id="sessionDateBadge" class="text-slate-400 tabular-num text-[11px] font-mono">Prices: 2026-11-04</span>
        </div>

        <!-- Gemini Key Active Pill -->
        <button onclick="promptApiKey()" class="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-stealth-900 hover:bg-stealth-850 border border-stealth-border hover:border-mint/40 text-xs text-slate-300 transition group shadow-sm">
          <div class="flex items-center space-x-2">
            <svg class="w-3.5 h-3.5 text-mint group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
            </svg>
            <span id="aiGlobalStatusText" class="font-medium text-mint">Gemini Key Active</span>
          </div>
          <span class="w-1.5 h-1.5 rounded-full bg-mint shadow-[0_0_6px_#00F5A0]"></span>
        </button>
      </div>

      <!-- Section: DATA FEEDS -->
      <div class="space-y-2.5 pt-2">
        <div class="text-[11px] font-bold text-slate-400 tracking-wider uppercase">Data Feeds</div>
        
        <div class="space-y-2">
          <!-- Button 1: Sync from YFinance -->
          <button onclick="triggerDataSync()" id="syncBtn" class="w-full py-2.5 px-3 rounded-lg bg-stealth-900 hover:bg-stealth-850 border border-stealth-border hover:border-mint/50 text-xs font-semibold text-slate-200 hover:text-white transition-all flex items-center justify-between group shadow-sm">
            <div class="flex items-center space-x-2.5">
              <div class="w-6 h-6 rounded-md bg-stealth-800 flex items-center justify-center text-mint group-hover:scale-105 transition-transform">
                <span id="syncSpinner" class="hidden inline-block animate-spin font-mono">↻</span>
                <svg id="syncIconDefault" class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <span id="syncBtnText" class="tracking-tight">Sync from YFinance</span>
            </div>
            <span class="w-1.5 h-1.5 rounded-full bg-mint shadow-[0_0_6px_#00F5A0]"></span>
          </button>

          <!-- Button 2: Sync from CSV -->
          <button onclick="triggerDataSync()" class="w-full py-2.5 px-3 rounded-lg bg-stealth-900 hover:bg-stealth-850 border border-stealth-border hover:border-slate-600 text-xs font-semibold text-slate-300 hover:text-white transition-all flex items-center justify-between group shadow-sm">
            <div class="flex items-center space-x-2.5">
              <div class="w-6 h-6 rounded-md bg-stealth-800 flex items-center justify-center text-slate-400 group-hover:text-slate-200 group-hover:scale-105 transition-transform">
                <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <span class="tracking-tight">Sync from CSV</span>
            </div>
            <span class="w-1.5 h-1.5 rounded-full bg-slate-600"></span>
          </button>
        </div>
      </div>

      <!-- Section: AI STRATEGIST ENGINE -->
      <div class="space-y-3 pt-3 border-t border-stealth-border">
        <div class="text-[11px] font-bold text-slate-400 tracking-wider uppercase">AI Strategist Engine</div>
        
        <div>
          <label class="block text-xs text-slate-400 mb-1.5 font-medium">Gemini API Key</label>
          <div class="relative">
            <input type="password" id="sidebarApiKey" value="••••••••••••••••••••••••••••" readonly 
                   class="w-full bg-stealth-900 border border-stealth-border rounded-lg px-3 py-2 text-xs text-slate-400 font-mono focus:outline-none cursor-default" />
          </div>
        </div>

        <!-- Save Key for Session Button -->
        <button onclick="promptApiKey()" class="w-full py-2.5 px-3 rounded-lg bg-mint/10 hover:bg-mint hover:text-stealth-950 text-mint border border-mint/30 text-xs font-bold transition-all shadow-[0_0_12px_rgba(0,245,160,0.15)] flex items-center justify-center space-x-1.5">
          <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
          <span>Save Key for Session</span>
        </button>

        <div class="flex items-center space-x-2 text-xs text-mint font-medium pt-1">
          <span class="w-2 h-2 rounded-full bg-mint shadow-[0_0_8px_#00F5A0]"></span>
          <span id="sidebarEngineStatus">Status: Gemini AI Active</span>
        </div>
      </div>

    </div>

    <!-- Sidebar Bottom Status Footer -->
    <div class="pt-6 border-t border-stealth-border text-[11px] space-y-2 text-slate-500 font-medium">
      <div class="flex items-start space-x-2">
        <span class="w-1.5 h-1.5 rounded-full bg-slate-600 mt-1"></span>
        <span>Data source: Local <code class="text-slate-400 text-[10px]">raw_csv</code> + yfinance daily sync</span>
      </div>
      <div class="flex items-center space-x-2 text-slate-400">
        <span class="w-2 h-2 rounded-full bg-mint shadow-[0_0_6px_#00F5A0]"></span>
        <span class="text-slate-300 font-semibold">Desktop Parity Active</span>
      </div>
    </div>

  </aside>

  <!-- ================= MAIN VIEWPORT ================= -->
  <div class="flex-1 flex flex-col min-w-0 overflow-y-auto">

    <!-- MAIN BODY CONTENT -->
    <main class="p-6 space-y-6 max-w-[1600px] w-full">

      <!-- ================= 1. CORE EQUITIES & MACRO BENCHMARKS ================= -->
      <section class="space-y-3">
        <div class="flex items-center justify-between text-xs">
          <div class="flex items-center space-x-2">
            <span class="w-1.5 h-1.5 rounded-full bg-mint"></span>
            <h3 class="font-bold text-slate-300 tracking-wider uppercase text-[11px]">Core Equities & Macro Benchmarks</h3>
          </div>
          <span class="text-slate-500 font-medium text-[11px]">Session Closes & Momentum • 7 Universe Assets</span>
        </div>

        <!-- Responsive Grid for All 7 Primary Universe Assets -->
        <div id="benchmarksRow" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-7 gap-3">
          <!-- Dynamically populated by renderBenchmarks() with inline sparklines -->
        </div>
      </section>

      <!-- ================= 2. QUANTITATIVE REGIME & BREADTH RATIOS ================= -->
      <section class="space-y-3 pt-1">
        <div class="flex items-center justify-between text-xs">
          <div class="flex items-center space-x-2">
            <span class="w-1.5 h-1.5 rounded-full bg-cherry"></span>
            <h3 class="font-bold text-slate-300 tracking-wider uppercase text-[11px]">Quantitative Regime & Breadth Ratios</h3>
            <span class="px-1.5 py-0.5 rounded bg-stealth-850 text-[10px] text-slate-400 font-mono border border-stealth-border">3 Pairs Monitored</span>
          </div>
          <span class="text-slate-500 font-medium text-[11px]">50-Day Z-Score & Relative Velocity</span>
        </div>

        <div id="macroBarometersRow" class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <!-- Dynamically populated by renderMacroBarometers() with Z-Score Gauge Spectrum Meters -->
        </div>
      </section>

      <!-- ================= 3. NAVIGATION PILL TABS ================= -->
      <nav class="flex items-center space-x-1.5 overflow-x-auto py-1 border-b border-stealth-border text-xs scrollbar-none font-medium">
        <button onclick="switchTab('tab1')" id="btn-tab1" class="tab-btn active px-4 py-2 rounded-lg bg-stealth-850 text-white font-semibold border border-mint/40 shadow-[0_0_12px_rgba(0,245,160,0.15)] shrink-0">
          AI Macro Letter
        </button>
        <button onclick="switchTab('tab_agent')" id="btn-tab_agent" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          AI Agent
        </button>
        <button onclick="switchTab('tab2')" id="btn-tab2" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Automated Trade Ideas
        </button>
        <button onclick="switchTab('tab3')" id="btn-tab3" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Bias Matrix
        </button>
        <button onclick="switchTab('tab5')" id="btn-tab5" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Indexes & Sector Flow
        </button>
        <button onclick="switchTab('tab4')" id="btn-tab4" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Magnificent 7 Flow
        </button>
        <button onclick="switchTab('tab6')" id="btn-tab6" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Synthetic Ratio Rotations
        </button>
        <button onclick="switchTab('tab_options')" id="btn-tab_options" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Volatility Pricing
        </button>
      </nav>
<!-- Tab Agent: Interactive AI Quant Agent -->
        <div id="tab_agent" class="tab-pane hidden">
            <div class="lux-card p-5 mb-4">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-3 mb-4 border-b border-white/10 gap-3">
                    <div>
                        <h2 class="text-lg font-bold text-white flex items-center gap-2">Interactive Quant Analyst & Risk Officer</h2>
                        <p class="text-xs text-slate-400">On-Demand AI Reasoning Across Live Regimes, Tickers, Confluences & Ratios</p>
                    </div>
                    <button onclick="clearAgentChat()" class="text-xs text-slate-400 hover:text-white px-3 py-1 rounded bg-white/5 border border-white/10 transition">Clear Chat</button>
                </div>
                
                <!-- Quick Prompts -->
                <div class="text-[11px] font-bold text-slate-500 uppercase tracking-wider mb-2">Quick Desk Inquiries:</div>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 mb-4">
                    <button onclick="sendQuickPrompt('Analyze todays Mag 7 divergence. Which tech leaders are showing institutional accumulation vs distribution, and what are the key invalidation levels?')" class="p-2.5 text-left rounded-lg bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 text-xs text-slate-300 transition">
                        <b>Mag 7 Divergence & Risks</b>
                    </button>
                    <button onclick="sendQuickPrompt('Based on the 5-day sector velocities, where is institutional smart money rotating right now? Which sector is leading?')" class="p-2.5 text-left rounded-lg bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 text-xs text-slate-300 transition">
                        <b>Sector Rotation Leaders</b>
                    </button>
                    <button onclick="sendQuickPrompt('What are the biggest distribution traps and high-risk names on the desk today? Where should retail traders avoid dip buying?')" class="p-2.5 text-left rounded-lg bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 text-xs text-slate-300 transition">
                        <b>Top Distribution Warnings</b>
                    </button>
                    <button onclick="sendQuickPrompt('Give me the top 3 high-probability setups from todays confluence radar with exact entry rationale, targets, and stop-loss levels.')" class="p-2.5 text-left rounded-lg bg-white/[0.02] hover:bg-white/[0.06] border border-white/5 text-xs text-slate-300 transition">
                        <b>Best High-Conviction Setups</b>
                    </button>
                </div>

                <!-- Chat History Window -->
                <div id="chatHistory" class="space-y-3 min-h-[300px] max-h-[500px] overflow-y-auto p-4 rounded-xl bg-[#090a0f] border border-white/5 mb-4">
                    <!-- Messages dynamically rendered -->
                </div>

                <!-- Chat Input Bar -->
                <form onsubmit="handleAgentSubmit(event)" class="flex gap-2">
                    <input id="chatInput" type="text" placeholder="Ask the Quant AI Agent anything about today's market data, tickers, or setups..." class="flex-1 bg-[#161722] border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-mint">
                    <button type="submit" id="chatSendBtn" class="bg-mint hover:bg-mint/80 text-black font-bold px-5 py-2.5 rounded-xl text-xs flex items-center gap-1.5 transition">
                        <span>Send</span>
                    </button>
                </form>
            </div>
        </div>

        <!-- Tab 1: AI Macro Letter -->
        <div id="tab1" class="tab-pane">
          <!-- ================= 4. MARKET REGIME & SECTOR VELOCITY BASKETS ================= -->
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-5 mb-5">
            
            <!-- Left: MARKET REGIME SYNTHESIS (Col span 7) -->
            <div class="lg:col-span-7 stealth-card p-6 space-y-3 relative overflow-hidden border border-stealth-border hover:border-slate-700">
              <!-- Subtle Stealth Ambient Glow in Top Right Corner -->
              <div class="absolute -top-16 -right-16 w-48 h-48 bg-mint/5 rounded-full blur-2xl pointer-events-none"></div>

              <!-- Badges -->
              <div class="flex items-center justify-between">
                <div class="flex items-center space-x-2">
                  <span class="text-[11px] font-bold text-mint uppercase tracking-wider">Market Regime</span>
                  <span id="briefingAiBadge" class="px-2 py-0.5 rounded bg-stealth-850 text-[10px] text-slate-300 font-semibold border border-stealth-border">
                    AI Synthesized
                  </span>
                </div>
                <span id="briefingRiskPill" class="text-xs text-slate-400 font-medium px-2.5 py-0.5 rounded-full bg-stealth-900 border border-stealth-border">
                  Selective / Dispersion
                </span>
              </div>

              <!-- Title -->
              <h4 id="briefingTitle" class="text-xl font-bold text-white tracking-tight leading-snug">
                Selective Cyclical Rotation With Mega-Cap Tech Rebound
              </h4>

              <!-- Paragraph -->
              <p id="briefingDetail" class="text-xs text-slate-300 leading-relaxed font-normal">
                Capital is aggressively rotating out of secular tech leadership like SMH and into mega-cap anchors (AAPL, MSFT, AMZN) and cyclicals, while defensives like XLU capture steady bids. Breadth ratios show severe mean-reversion in cap-weighted benchmarks relative to equal-weight baselines.
              </p>
            </div>

            <!-- Right: 5-DAY SECTOR VELOCITY BASKETS (Col span 5) -->
            <div class="lg:col-span-5 stealth-card p-6 flex flex-col justify-between space-y-4 border border-stealth-border">
              
              <div>
                <div class="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-4">
                  5-Day Sector Velocity Baskets
                </div>

                <div id="basketsGrid" class="grid grid-cols-4 gap-2 text-center">
                  <!-- Populated dynamically by renderBriefingBaskets() -->
                </div>
              </div>

              <!-- Footer Footnote -->
              <div class="pt-3 border-t border-stealth-border text-[11px] text-slate-500 leading-normal">
                <span class="text-slate-400 font-medium">Capital Rotation:</span> Measures 5-day velocity across cyclical vs defensive balance sheets.
              </div>

            </div>

          </div>

          <!-- Executive Synthesis Card -->
          <div id="synthesisCard" class="stealth-card p-6 border border-stealth-border space-y-4">
            <div class="flex flex-wrap justify-between items-center gap-3 pb-3 border-b border-stealth-border">
              <div class="flex items-center space-x-2.5">
                <span class="w-2 h-2 rounded-full bg-mint"></span>
                <div>
                  <span class="text-xs font-bold text-mint uppercase tracking-wider block">Institutional Executive Synthesis</span>
                  <span class="text-[11px] text-slate-400">Chief Quantitative Strategist & Portfolio Risk Intelligence</span>
                </div>
              </div>
              <div class="flex items-center space-x-2">
                <span id="aiBriefingTimestamp" class="text-[11px] font-mono text-slate-400 bg-stealth-850 px-2.5 py-1 rounded border border-stealth-border">Daily Session</span>
                <span id="aiSourceBadge" class="badge-mint px-2.5 py-1 rounded text-[11px] font-bold">Quantitative Desk</span>
              </div>
            </div>

            <!-- Navigation View Filter Pills -->
            <div class="flex flex-wrap items-center gap-1.5 p-1.5 rounded-lg bg-stealth-950 border border-stealth-border">
              <span class="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-2">Focus Pillar:</span>
              <button onclick="filterSynthesisSection('all')" id="synthPill-all" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-bold bg-mint text-stealth-950 shadow-sm transition">All 4 Pillars</button>
              <button onclick="filterSynthesisSection('macro')" id="synthPill-macro" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-semibold bg-stealth-850 hover:bg-stealth-800 text-slate-300 transition">1. Macro Liquidity</button>
              <button onclick="filterSynthesisSection('sectors')" id="synthPill-sectors" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-semibold bg-stealth-850 hover:bg-stealth-800 text-slate-300 transition">2. Sector Velocity</button>
              <button onclick="filterSynthesisSection('setups')" id="synthPill-setups" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-semibold bg-stealth-850 hover:bg-stealth-800 text-slate-300 transition">3. Setups & Traps</button>
              <button onclick="filterSynthesisSection('risk')" id="synthPill-risk" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-semibold bg-stealth-850 hover:bg-stealth-800 text-slate-300 transition">4. Desk Game Plan</button>
            </div>

            <!-- Structured Synthesis Section Cards Container -->
            <div id="synthesisText" class="space-y-4">
              <div class="flex items-center space-x-2 text-slate-400 py-6 text-xs">
                <span class="inline-block animate-spin text-mint font-mono">↻</span>
                <span>Synthesizing multi-factor institutional market intelligence...</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Tab 2: Automated Trade Ideas -->
        <div id="tab2" class="tab-pane hidden">
            <div class="bg-white/[0.03] border border-white/5 rounded-xl p-3 mb-4 text-xs text-slate-400">
                <b class="text-mint font-semibold">Strict Confluence Rules:</b> Green conviction scores (+15 to +100) are reserved strictly for <b>Long Setups</b>. Red conviction scores (-15 to -100) are strictly flagged for <b>Distribution Fades or De-risking</b> to protect portfolio equity.
            </div>
            <div id="tradeCardsContainer" class="grid grid-cols-1 xl:grid-cols-2 gap-3">
                <div class="text-xs text-slate-400">Loading high-probability trade setups...</div>
            </div>
        </div>

        <!-- Tab 3: Bias Matrix -->
        <div id="tab3" class="tab-pane hidden">
            <div class="inline-flex flex-wrap bg-[#18181b] p-1 rounded-lg border border-[#27272a] gap-1 mb-4">
                <button onclick="loadConfluence('core', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-white/10 text-white">Core Universe</button>
                <button onclick="loadConfluence('mag7', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-[#111319] text-slate-400">Mag 7 Only</button>
                <button onclick="loadConfluence('indexes', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-[#111319] text-slate-400">Major Indexes</button>
                <button onclick="loadConfluence('sectors', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-[#111319] text-slate-400">Sector ETFs</button>
                <button onclick="loadConfluence('macro', this)" class="conf-pill px-3 py-1 rounded-full text-xs font-semibold bg-[#111319] text-slate-400">Macro Assets</button>
            </div>
            <div class="lux-card p-4 mb-5 overflow-hidden">
                <div id="confluenceChart" style="width: 100%; min-height: 480px;"></div>
            </div>
            
            <h4 class="text-sm font-bold text-white mb-2">Multi-Factor Bias Matrix</h4>
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
                    <select id="inspectorSelect" onchange="inspectAsset(this.value)" class="bg-[#0b0d12] text-xs text-white border border-white/10 px-3 py-1.5 rounded-lg focus:outline-none focus:border-mint"></select>
                </div>
                <div id="inspectorDetail" class="space-y-2"></div>
            </div>
        </div>

        <!-- Tab 4: Magnificant 7 Flow -->
        <div id="tab4" class="tab-pane hidden">
            <h3 class="text-base font-bold text-white mb-1">Magnificant 7 Flow (Technicals, Realized Vol, Drawdowns & Odds)</h3>
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

        <!-- Tab 6: Synthetic Ratio Rotations -->
        <div id="tab6" class="tab-pane hidden">
            <div class="flex justify-between items-center mb-4">
                <div>
                    <h3 class="text-base font-bold text-white">Synthetic Ratio Rotations</h3>
                    <p class="text-xs text-slate-400">Quantitative measures of breadth, mega-cap concentration, and small-cap rotation.</p>
                </div>
                <select id="ratioSelect" onchange="loadRatio(this.value)" class="bg-[#0b0d12] text-xs text-white border border-white/10 px-3 py-1.5 rounded-lg focus:outline-none focus:border-mint">
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

        <!-- Tab Options: Volatility Pricing Desk -->
        <div id="tab_options" class="tab-pane hidden">
            <div class="lux-card p-5 mb-5 border border-mint/20 bg-gradient-to-r from-mint/5 via-[#0d1117] to-slate-900/40">
                <div class="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
                    <div>
                        <div class="flex items-center gap-2">
                            <h2 class="text-lg font-black text-white tracking-wide">Institutional Volatility Pricing Desk</h2>
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded bg-mint/15 text-mint border border-mint/25">Live Options Vol</span>
                        </div>
                        <p class="text-xs text-slate-400 mt-1 max-w-2xl">
                            Institutional volatility architecture: Switch between <b>Buy Side (Ask Edge)</b> for underpriced cheap IV and <b>Sell Side (Bid Edge)</b> for overpriced premium harvesting. De-jumps historical volatility, models Asymmetric Log-Moneyness Skew, and executes full Bid/Ask spread inversion.
                        </p>
                    </div>
                    
                    <div class="flex flex-wrap items-center gap-3">
                        <div>
                            <label class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Ticker</label>
                            <select id="optionsTickerSelect" onchange="loadOptionsDesk()" class="bg-[#0b0d12] text-xs font-bold text-white border border-white/10 px-3 py-1.5 rounded-lg focus:outline-none focus:border-mint">
                                <option value="QQQ" selected>QQQ (Nasdaq 100)</option>
                                <option value="SPY">SPY (S&P 500)</option>
                                <option value="NVDA">NVDA (Nvidia)</option>
                                <option value="AAPL">AAPL (Apple)</option>
                                <option value="AMZN">AMZN (Amazon)</option>
                                <option value="MSFT">MSFT (Microsoft)</option>
                                <option value="META">META (Meta)</option>
                                <option value="TSLA">TSLA (Tesla)</option>
                                <option value="IWM">IWM (Russell 2000)</option>
                                <option value="GLD">GLD (Gold)</option>
                                <option value="XLF">XLF (Financials)</option>
                                <option value="SMH">SMH (Semiconductors)</option>
                                <option value="ADBE">ADBE</option>
                                <option value="AMAT">AMAT</option>
                                <option value="AMD">AMD (Advanced Micro Devices)</option>
                                <option value="AVGO">AVGO (Broadcom)</option>
                                <option value="BA">BA</option>
                                <option value="BAC">BAC</option>
                                <option value="CAT">CAT</option>
                                <option value="CIBR">CIBR</option>
                                <option value="CMCSA">CMCSA</option>
                                <option value="COP">COP</option>
                                <option value="COPX">COPX</option>
                                <option value="COST">COST (Costco)</option>
                                <option value="CRM">CRM</option>
                                <option value="CRWD">CRWD (CrowdStrike)</option>
                                <option value="CVX">CVX</option>
                                <option value="DELL">DELL (Dell)</option>
                                <option value="DIA">DIA</option>
                                <option value="DIS">DIS</option>
                                <option value="FCX">FCX</option>
                                <option value="GOOGL">GOOGL (Alphabet)</option>
                                <option value="GS">GS</option>
                                <option value="HD">HD</option>
                                <option value="IBIT">IBIT (Bitcoin ETF)</option>
                                <option value="IGV">IGV</option>
                                <option value="ITA">ITA</option>
                                <option value="JPM">JPM (JPMorgan)</option>
                                <option value="KO">KO</option>
                                <option value="KRE">KRE</option>
                                <option value="LLY">LLY (Eli Lilly)</option>
                                <option value="LRCX">LRCX</option>
                                <option value="MA">MA</option>
                                <option value="MCD">MCD</option>
                                <option value="MRVL">MRVL</option>
                                <option value="MS">MS</option>
                                <option value="MU">MU (Micron)</option>
                                <option value="NFLX">NFLX (Netflix)</option>
                                <option value="NKE">NKE</option>
                                <option value="NOW">NOW</option>
                                <option value="OIH">OIH</option>
                                <option value="ORCL">ORCL (Oracle)</option>
                                <option value="PANW">PANW</option>
                                <option value="PEP">PEP</option>
                                <option value="PFE">PFE</option>
                                <option value="PG">PG</option>
                                <option value="PLTR">PLTR (Palantir)</option>
                                <option value="QCOM">QCOM</option>
                                <option value="RSP">RSP</option>
                                <option value="SBUX">SBUX</option>
                                <option value="SLB">SLB</option>
                                <option value="STX">STX</option>
                                <option value="TGT">TGT</option>
                                <option value="TLT">TLT (20Y Treasury Bond)</option>
                                <option value="TSM">TSM</option>
                                <option value="TXN">TXN</option>
                                <option value="UNH">UNH</option>
                                <option value="USB">USB</option>
                                <option value="USO">USO</option>
                                <option value="V">V</option>
                                <option value="VRTX">VRTX</option>
                                <option value="WDC">WDC</option>
                                <option value="WFC">WFC</option>
                                <option value="WMT">WMT (Walmart)</option>
                                <option value="XBI">XBI</option>
                                <option value="XLB">XLB</option>
                                <option value="XLC">XLC</option>
                                <option value="XLE">XLE</option>
                                <option value="XLI">XLI</option>
                                <option value="XLK">XLK</option>
                                <option value="XLP">XLP</option>
                                <option value="XLU">XLU</option>
                                <option value="XLV">XLV</option>
                                <option value="XLY">XLY</option>
                                <option value="XOM">XOM</option>
                            </select>
                        </div>
                        
                        <div>
                            <label class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Execution Side</label>
                            <div class="flex bg-white/5 p-1 rounded-lg text-xs" id="optionsSideGroup">
                                <button onclick="setOptionsSide('Buy')" id="btnOptBuy" class="px-3 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30">🟢 Buy (Ask Edge)</button>
                                <button onclick="setOptionsSide('Sell')" id="btnOptSell" class="px-3 py-1 rounded font-bold transition text-slate-400">🔴 Sell (Bid Edge)</button>
                            </div>
                        </div>

                        <div>
                            <label class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Direction</label>
                            <div class="flex bg-white/5 p-1 rounded-lg text-xs" id="optionsDirGroup">
                                <button onclick="setOptionsDirection('Bullish')" id="btnOptBullish" class="px-3 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30">🟢 Bullish (Calls)</button>
                                <button onclick="setOptionsDirection('Bearish')" id="btnOptBearish" class="px-3 py-1 rounded font-bold transition text-slate-400">🔴 Bearish (Puts)</button>
                            </div>
                        </div>

                        <div class="self-end">
                            <button onclick="loadOptionsDesk(true)" class="px-3 py-1.5 rounded-lg bg-mint hover:bg-mint/80 text-black font-bold text-xs shadow-lg shadow-mint/20 transition flex items-center gap-1.5">
                                Scan Chains
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Options KPI Strip -->
            <div id="optionsKpiRow" class="grid grid-cols-2 sm:grid-cols-5 gap-3.5 mb-5">
                <!-- Populated dynamically -->
            </div>

            <!-- Cross-Chain Recommendations Header -->
            <div class="flex justify-between items-center mb-3">
                <div>
                    <h3 class="text-sm font-bold text-white flex items-center gap-2" id="optionsRecHeaderTitle">
                        Top Recommended Cheap IV Contracts Across Different Expirations
                    </h3>
                    <p class="text-[11px] text-slate-400" id="optionsRecHeaderSubtitle">Guaranteed multi-expiry diversity: Evaluates term pricing ($IV/HV$), execution edge (Ask-Edge or Bid-Edge), delta positioning, and liquidity.</p>
                </div>
            </div>

            <!-- Cross-Chain Cards Container -->
            <div id="optionsRecCards" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <!-- Dynamically populated -->
            </div>

            <!-- Volatility Smile & Skew Inspector Chart -->
            <div class="lux-card p-4 mb-4">
                <div class="flex justify-between items-center mb-3">
                    <div>
                        <h4 class="text-sm font-bold text-white flex items-center gap-1.5">
                            Volatility Smile & Fitted Asymmetric Skew Curve
                        </h4>
                        <p class="text-[11px] text-slate-400" id="optionsSmileSubtitle">Showing nearest active expiration chain, degree-3 log-moneyness skew, and 1σ expected move boundary</p>
                    </div>
                </div>
                <div id="optionsSmileChart" style="height: 420px;"></div>
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
            const statusEl = document.getElementById('aiGlobalStatusText');
            const engineStatusEl = document.getElementById('sidebarEngineStatus');
            const key = USER_API_KEY || '';
            const topInput = document.getElementById('topApiKeyInput');
            if (topInput && key) topInput.value = key;
            const sideInput = document.getElementById('sidebarApiKey');
            if (sideInput && key) sideInput.value = key;
            if (statusEl) {
                statusEl.innerText = key ? 'Gemini Key Active' : 'Gemini Key Active (Default)';
            }
            if (engineStatusEl) {
                engineStatusEl.innerText = key ? 'Status: Gemini AI Active (Custom)' : 'Status: Gemini AI Active (Default)';
            }
        }

        function saveTopApiKey() {
            const input = document.getElementById('topApiKeyInput');
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
            const hBtn = document.getElementById('headerSyncBtn');
            const hSpinner = document.getElementById('headerSyncSpinner');
            const hBtnText = document.getElementById('headerSyncBtnText');
            const sBtn = document.getElementById('syncBtn');
            const sSpinner = document.getElementById('syncSpinner');
            const sBtnText = document.getElementById('syncBtnText');

            if (hBtn) hBtn.disabled = true;
            if (sBtn) sBtn.disabled = true;
            if (hSpinner) hSpinner.classList.remove('hidden');
            if (sSpinner) sSpinner.classList.remove('hidden');
            if (hBtnText) hBtnText.innerText = 'Syncing...';
            if (sBtnText) sBtnText.innerText = 'Syncing...';

            try {
                const res = await fetch('/api/sync-data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ scope: 'core' })
                });

                if (!res.ok) {
                    throw new Error(`Sync returned HTTP ${res.status}`);
                }

                const marketRes = await fetch('/api/market-data');
                DATA = await marketRes.json();

                renderBenchmarks();
                renderMacroBarometers();
                renderBriefingBaskets();

                if (hBtnText) hBtnText.innerText = 'Synced!';
                if (sBtnText) sBtnText.innerText = 'Synced!';
                setTimeout(() => {
                    if (hBtnText) hBtnText.innerText = 'Sync Market Feeds';
                    if (sBtnText) sBtnText.innerText = 'Sync from YFinance';
                    if (hBtn) hBtn.disabled = false;
                    if (sBtn) sBtn.disabled = false;
                    if (hSpinner) hSpinner.classList.add('hidden');
                    if (sSpinner) sSpinner.classList.add('hidden');
                }, 2000);

            } catch (err) {
                console.error("Data sync error:", err);
                if (hBtnText) hBtnText.innerText = 'Sync Error';
                if (sBtnText) sBtnText.innerText = 'Sync Error';
                setTimeout(() => {
                    if (hBtnText) hBtnText.innerText = 'Sync Market Feeds';
                    if (sBtnText) sBtnText.innerText = 'Sync from YFinance';
                    if (hBtn) hBtn.disabled = false;
                    if (sBtn) sBtn.disabled = false;
                    if (hSpinner) hSpinner.classList.add('hidden');
                    if (sSpinner) sSpinner.classList.add('hidden');
                }, 2500);
            }
        }

        const LOADED_TABS = new Set(['tab1']);
        let AI_DATA = null;

        const TAB_NAMES = {
            'tab1': 'AI Macro Letter',
            'tab_agent': 'AI Agent',
            'tab2': 'Automated Trade Ideas Desk',
            'tab3': 'Bias Matrix',
            'tab5': 'Indexes & Sector Flow',
            'tab4': 'Magnificant 7 Flow',
            'tab6': 'Synthetic Ratio Rotations',
            'tab_options': 'Volatility Pricing'
        };

        function switchTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active', 'bg-stealth-850', 'text-white', 'font-semibold', 'border-mint/40', 'shadow-[0_0_12px_rgba(0,245,160,0.15)]');
                btn.classList.add('text-slate-400', 'hover:text-white', 'hover:bg-stealth-900', 'border-transparent');
            });
            const targetPane = document.getElementById(tabId);
            if (targetPane) targetPane.classList.remove('hidden');
            
            const activeBtn = document.getElementById('btn-' + tabId);
            if (activeBtn) {
                activeBtn.classList.remove('text-slate-400', 'hover:bg-stealth-900', 'border-transparent');
                activeBtn.classList.add('active', 'bg-stealth-850', 'text-white', 'font-semibold', 'border-mint/40', 'shadow-[0_0_12px_rgba(0,245,160,0.15)]');
            }

            const heading = document.getElementById('pageHeading');
            const breadcrumb = document.getElementById('breadcrumbCurrent');
            if (heading && TAB_NAMES[tabId]) heading.innerText = TAB_NAMES[tabId];
            if (breadcrumb && TAB_NAMES[tabId]) breadcrumb.innerText = TAB_NAMES[tabId];

            // Lazy load tabs on demand
            if (tabId === 'tab2') {
                LOADED_TABS.add('tab2');
                renderTradeCards();
            } else if (tabId === 'tab3') {
                if (!LOADED_TABS.has('tab3')) {
                    LOADED_TABS.add('tab3');
                    loadConfluence('core');
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
                    loadSingleTicker();
                } else {
                    setTimeout(() => Plotly.Plots.resize('singleTickerCombinedChart'), 50);
                }
            } else if (tabId === 'tab_options') {
                populateOptionsTickersDropdown();
                if (!LOADED_TABS.has('tab_options')) {
                    LOADED_TABS.add('tab_options');
                    loadOptionsDesk();
                } else {
                    setTimeout(() => Plotly.Plots.resize('optionsSmileChart'), 50);
                }
            }
        }

        let CURRENT_HERO_VIEW = 'benchmarks';

        function toggleHeroCards(view, btn) {
            CURRENT_HERO_VIEW = view;
            const bBtn = document.getElementById('heroBtnBench');
            const mBtn = document.getElementById('heroBtnMacro');
            if (bBtn) bBtn.className = view === 'benchmarks' ? 'px-3 py-1 rounded-md font-semibold bg-[#202026] text-white' : 'px-3 py-1 rounded-md font-medium text-zinc-400 hover:text-white';
            if (mBtn) mBtn.className = view === 'macro' ? 'px-3 py-1 rounded-md font-semibold bg-[#202026] text-white' : 'px-3 py-1 rounded-md font-medium text-zinc-400 hover:text-white';
            renderFilamentHeroCards();
        }

        function renderFilamentHeroCards() {
            const container = document.getElementById('filamentHeroCards');
            if (!container || !DATA) return;
            const b = DATA.benchmarks || {};
            const macro = DATA.market_data?.macro_assets || {};
            const r = DATA.market_data?.ratios || {};
            const spy_rsp = r.SPY_RSP || {};

            let cards = [];
            if (CURRENT_HERO_VIEW === 'benchmarks') {
                cards = [
                    {
                        label: 'S&P 500 Core', ticker: 'SPY',
                        val: `$${Number(b.SPY?.close || 0).toFixed(2)}`,
                        pct: b.SPY?.pct_1d || 0,
                        sub: `ATH DD: ${b.SPY?.drawdown_ath}% • 5D: ${(b.SPY?.pct_5d || 0) >= 0 ? '+' : ''}${b.SPY?.pct_5d}%`,
                        sparkPoints: 'M0,18 Q30,8 60,14 T120,6 T180,12 T240,4'
                    },
                    {
                        label: 'Nasdaq 100 Tech', ticker: 'QQQ',
                        val: `$${Number(b.QQQ?.close || 0).toFixed(2)}`,
                        pct: b.QQQ?.pct_1d || 0,
                        sub: `ATH DD: ${b.QQQ?.drawdown_ath}% • 5D: ${(b.QQQ?.pct_5d || 0) >= 0 ? '+' : ''}${b.QQQ?.pct_5d}%`,
                        sparkPoints: 'M0,16 Q30,12 60,20 T120,10 T180,15 T240,6'
                    },
                    {
                        label: 'Breadth Spread', ticker: 'SPY / RSP',
                        val: Number(spy_rsp.current || 0).toFixed(2),
                        pct: spy_rsp.pct_5d || 0,
                        isSigma: true, sigma: spy_rsp.z_score_50d || 0,
                        sub: `<span style="color:${spy_rsp.state_color || '#00F5A0'}; font-weight:600;">${spy_rsp.actionable_state || 'Breadth Expansion'}</span>`,
                        sparkPoints: 'M0,8 Q30,14 60,10 T120,18 T180,14 T240,22'
                    },
                    {
                        label: 'Russell 2000 Small', ticker: 'IWM',
                        val: `$${Number(b.IWM?.close || 0).toFixed(2)}`,
                        pct: b.IWM?.pct_1d || 0,
                        sub: `ATH DD: ${b.IWM?.drawdown_ath}% • 5D: ${(b.IWM?.pct_5d || 0) >= 0 ? '+' : ''}${b.IWM?.pct_5d}%`,
                        sparkPoints: 'M0,20 Q30,16 60,12 T120,14 T180,8 T240,4'
                    }
                ];
            } else {
                cards = [
                    {
                        label: 'Gold Debasement', ticker: 'GLD',
                        val: `$${Number(macro.GLD?.close || 0).toFixed(2)}`,
                        pct: macro.GLD?.pct_1d || 0,
                        sub: `5D: ${(macro.GLD?.pct_5d || 0) >= 0 ? '+' : ''}${macro.GLD?.pct_5d?.toFixed(2)}% • Safe Haven`,
                        sparkPoints: 'M0,14 Q30,10 60,8 T120,12 T180,6 T240,2'
                    },
                    {
                        label: '20Y Treasuries', ticker: 'TLT',
                        val: `$${Number(macro.TLT?.close || 0).toFixed(2)}`,
                        pct: macro.TLT?.pct_1d || 0,
                        sub: `5D: ${(macro.TLT?.pct_5d || 0) >= 0 ? '+' : ''}${macro.TLT?.pct_5d?.toFixed(2)}% • Duration Risk`,
                        sparkPoints: 'M0,10 Q30,12 60,16 T120,14 T180,20 T240,18'
                    },
                    {
                        label: 'Crude Oil Energy', ticker: 'USO',
                        val: `$${Number(macro.USO?.close || 0).toFixed(2)}`,
                        pct: macro.USO?.pct_1d || 0,
                        sub: `5D: ${(macro.USO?.pct_5d || 0) >= 0 ? '+' : ''}${macro.USO?.pct_5d?.toFixed(2)}% • Energy Input`,
                        sparkPoints: 'M0,6 Q30,14 60,12 T120,20 T180,18 T240,24'
                    },
                    {
                        label: 'Bitcoin Trust', ticker: 'IBIT',
                        val: `$${Number(macro.IBIT?.close || 0).toFixed(2)}`,
                        pct: macro.IBIT?.pct_1d || 0,
                        sub: `5D: ${(macro.IBIT?.pct_5d || 0) >= 0 ? '+' : ''}${macro.IBIT?.pct_5d?.toFixed(2)}% • Liquidity Proxy`,
                        sparkPoints: 'M0,18 Q30,10 60,14 T120,6 T180,10 T240,2'
                    }
                ];
            }

            let html = '';
            cards.forEach(c => {
                const isPos = c.pct >= 0;
                const pillClass = isPos ? 'pill-green' : 'pill-red';
                const sign = isPos ? '+' : '';
                const badge = c.isSigma ? `<span class="pill-cyan font-mono">${c.sigma > 0 ? '+' : ''}${c.sigma?.toFixed(2)}σ</span>` : `<span class="${pillClass}">${sign}${c.pct?.toFixed(2)}%</span>`;
                const strokeColor = isPos ? '#00F5A0' : '#FF2E55';
                html += `
                    <div class="lux-card p-5 flex flex-col justify-between">
                        <div>
                            <div class="flex items-center justify-between mb-1.5">
                                <span class="text-xs font-medium text-zinc-400">${c.ticker} • ${c.label}</span>
                                ${badge}
                            </div>
                            <div class="text-2xl font-bold text-white tracking-tight">${c.val}</div>
                        </div>
                        <div class="mt-4">
                            <svg class="sparkline-svg" viewBox="0 0 240 28" fill="none">
                                <path d="${c.sparkPoints}" stroke="${strokeColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            <div class="text-[11px] text-zinc-400 border-t border-[#232328] pt-2 mt-1 truncate">${c.sub}</div>
                        </div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        function handleGlobalSearch(e) {
            const val = document.getElementById('globalTickerSearch')?.value?.trim().toUpperCase();
            if (!val) return;
            if (e && e.key === 'Enter') {
                const tickerSelect = document.getElementById('tickerSelect');
                if (tickerSelect) {
                    tickerSelect.value = val;
                    switchTab('tab7');
                    loadSingleTicker();
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
                renderBenchmarks();
                renderMacroBarometers();
                renderBriefingBaskets();
                renderFilamentHeroCards();
                initAgentChat();
                populateTickersDropdown();
                populateOptionsTickersDropdown();

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
                AGENT_CHAT_HISTORY.push({ role: 'assistant', content: 'Error connecting to AI Agent: ' + err.message });
                renderChatHistory();
            }
        }

        function renderTickerTape() {
            const el = document.getElementById('tickerRibbon');
            if (!el) return;
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
            el.innerHTML = html;
        }

        function renderRegime() {
            const md = DATA.market_data;
            const t = document.getElementById('regimeTitle');
            if (t) t.innerText = md.narrative_title || "Secular Uptrend";
            const d = document.getElementById('regimeDetail');
            if (d) d.innerText = md.narrative_detail || "";
            const b = document.getElementById('regimeRiskBadge');
            if (b) b.innerText = md.risk_level || "Moderate";
            const sessionBadge = document.getElementById('sessionDateBadge');
            if (sessionBadge) {
                sessionBadge.innerText = 'Prices: ' + (DATA.last_market_session || '2026-10-27');
            }
        }

        function generateSmoothSparkline(prices, width = 56, height = 20, padX = 3, padY = 4) {
            if (!prices || prices.length < 2) {
                return { path: 'M 3 10 L 49 10', endX: 49, endY: 10 };
            }
            const n = prices.length;
            const minP = Math.min(...prices);
            const maxP = Math.max(...prices);
            const rng = maxP - minP;
            
            const wAvail = width - padX * 2 - 4;
            const hAvail = height - padY * 2;
            
            const pts = [];
            for (let i = 0; i < n; i++) {
                const x = padX + (i / (n - 1)) * wAvail;
                const norm = rng > 1e-4 ? (prices[i] - minP) / rng : 0.5;
                const y = (height - padY) - norm * hAvail;
                pts.push({ x: Number(x.toFixed(1)), y: Number(y.toFixed(1)) });
            }
            
            let d = `M ${pts[0].x} ${pts[0].y}`;
            for (let i = 0; i < pts.length - 1; i++) {
                const p0 = i > 0 ? pts[i - 1] : pts[i];
                const p1 = pts[i];
                const p2 = pts[i + 1];
                const p3 = i + 2 < pts.length ? pts[i + 2] : pts[i + 1];
                
                const cp1x = Number((p1.x + (p2.x - p0.x) / 5.0).toFixed(1));
                const cp1y = Number((p1.y + (p2.y - p0.y) / 5.0).toFixed(1));
                const cp2x = Number((p2.x - (p3.x - p1.x) / 5.0).toFixed(1));
                const cp2y = Number((p2.y - (p3.y - p1.y) / 5.0).toFixed(1));
                
                d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`;
            }
            const endPt = pts[pts.length - 1];
            return { path: d, endX: endPt.x, endY: endPt.y };
        }

        function renderBenchmarks() {
            const sessionBadge = document.getElementById('sessionDateBadge');
            if (sessionBadge && DATA.last_market_session) {
                sessionBadge.innerText = 'Prices: ' + DATA.last_market_session;
            }
            const b = DATA.benchmarks || {};
            const macro = DATA.market_data.macro_assets || {};

            const cards = [
                { 
                    ticker: "SPY", name: "S&P 500", 
                    price: b.SPY?.close || 0, pct1d: b.SPY?.pct_1d || 0, pct5d: b.SPY?.pct_5d || 0,
                    sub1: `ATH DD: <span class="${(b.SPY?.drawdown_ath || 0) < 0 ? 'text-cherry' : 'text-slate-300'} font-medium">${b.SPY?.drawdown_ath || 0}%</span> • 5D: <span class="${(b.SPY?.pct_5d || 0) >= 0 ? 'text-mint' : 'text-cherry'} font-medium">${(b.SPY?.pct_5d || 0) >= 0 ? '+' : ''}${b.SPY?.pct_5d || 0}%</span>`,
                    sub2: `Large-Cap Core Benchmark`
                },
                { 
                    ticker: "QQQ", name: "Nasdaq 100", 
                    price: b.QQQ?.close || 0, pct1d: b.QQQ?.pct_1d || 0, pct5d: b.QQQ?.pct_5d || 0,
                    sub1: `ATH DD: <span class="${(b.QQQ?.drawdown_ath || 0) < 0 ? 'text-cherry' : 'text-slate-300'} font-medium">${b.QQQ?.drawdown_ath || 0}%</span> • 5D: <span class="${(b.QQQ?.pct_5d || 0) >= 0 ? 'text-mint' : 'text-cherry'} font-medium">${(b.QQQ?.pct_5d || 0) >= 0 ? '+' : ''}${b.QQQ?.pct_5d || 0}%</span>`,
                    sub2: `Tech Growth Benchmark`
                },
                { 
                    ticker: "IWM", name: "Russell 2000", 
                    price: b.IWM?.close || 0, pct1d: b.IWM?.pct_1d || 0, pct5d: b.IWM?.pct_5d || 0,
                    sub1: `ATH DD: <span class="${(b.IWM?.drawdown_ath || 0) < 0 ? 'text-cherry' : 'text-slate-300'} font-medium">${b.IWM?.drawdown_ath || 0}%</span> • 5D: <span class="${(b.IWM?.pct_5d || 0) >= 0 ? 'text-mint' : 'text-cherry'} font-medium">${(b.IWM?.pct_5d || 0) >= 0 ? '+' : ''}${b.IWM?.pct_5d || 0}%</span>`,
                    sub2: `Small-Cap Value Benchmark`
                },
                { 
                    ticker: "GLD", name: "Gold Trust", 
                    price: macro.GLD?.close || 0, pct1d: macro.GLD?.pct_1d || 0, pct5d: macro.GLD?.pct_5d || 0,
                    sub1: `5D: <span class="${(macro.GLD?.pct_5d || 0) >= 0 ? 'text-mint' : 'text-cherry'} font-medium">${(macro.GLD?.pct_5d || 0) >= 0 ? '+' : ''}${Number(macro.GLD?.pct_5d || 0).toFixed(2)}%</span> • 20D: <span class="font-medium text-slate-300">${Number(macro.GLD?.pct_20d || 0).toFixed(2)}%</span>`,
                    sub2: `Safe Haven • Debasement Hedge`
                },
                { 
                    ticker: "TLT", name: "20Y Treasury", 
                    price: macro.TLT?.close || 0, pct1d: macro.TLT?.pct_1d || 0, pct5d: macro.TLT?.pct_5d || 0,
                    sub1: `5D: <span class="${(macro.TLT?.pct_5d || 0) >= 0 ? 'text-mint' : 'text-cherry'} font-medium">${(macro.TLT?.pct_5d || 0) >= 0 ? '+' : ''}${Number(macro.TLT?.pct_5d || 0).toFixed(2)}%</span> • 20D: <span class="font-medium text-slate-300">${Number(macro.TLT?.pct_20d || 0).toFixed(2)}%</span>`,
                    sub2: `Duration • Real Yield Pressure`
                },
                { 
                    ticker: "IBIT", name: "Bitcoin Trust", 
                    price: macro.IBIT?.close || 0, pct1d: macro.IBIT?.pct_1d || 0, pct5d: macro.IBIT?.pct_5d || 0,
                    sub1: `5D: <span class="${(macro.IBIT?.pct_5d || 0) >= 0 ? 'text-mint' : 'text-cherry'} font-medium">${(macro.IBIT?.pct_5d || 0) >= 0 ? '+' : ''}${Number(macro.IBIT?.pct_5d || 0).toFixed(2)}%</span> • 20D: <span class="font-medium text-slate-300">${Number(macro.IBIT?.pct_20d || 0).toFixed(2)}%</span>`,
                    sub2: `High-Beta Liquidity Barometer`
                },
                { 
                    ticker: "USO", name: "Crude Oil", 
                    price: macro.USO?.close || 0, pct1d: macro.USO?.pct_1d || 0, pct5d: macro.USO?.pct_5d || 0,
                    sub1: `5D: <span class="${(macro.USO?.pct_5d || 0) >= 0 ? 'text-mint' : 'text-cherry'} font-medium">${(macro.USO?.pct_5d || 0) >= 0 ? '+' : ''}${Number(macro.USO?.pct_5d || 0).toFixed(2)}%</span> • 20D: <span class="font-medium text-slate-300">${Number(macro.USO?.pct_20d || 0).toFixed(2)}%</span>`,
                    sub2: `Energy & Inflation Input Cost`
                }
            ];

            let html = '';
            cards.forEach(c => {
                const p1d = c.pct1d || 0;
                const isPos = p1d >= 0;
                const badgeClass = isPos ? 'badge-mint' : 'badge-cherry';
                const sign = isPos ? '+' : '';
                const p5d = c.pct5d || 0;
                const is5dPos = p5d >= 0;
                const sparkColor = is5dPos ? '#00F5A0' : '#FF2E55';
                
                // Actual smooth 5-session price trajectory
                const sparkData = (DATA.sparklines && DATA.sparklines[c.ticker]) ? DATA.sparklines[c.ticker] : null;
                let sparkPath, sparkEndX, sparkEndY;
                if (sparkData && sparkData.length >= 2) {
                    const res = generateSmoothSparkline(sparkData);
                    sparkPath = res.path;
                    sparkEndX = res.endX;
                    sparkEndY = res.endY;
                } else {
                    sparkPath = is5dPos ? 'M 3 15 Q 16 12, 28 8 T 42 10 T 50 6' : 'M 3 7 Q 15 4, 28 13 T 44 10 T 50 14';
                    sparkEndX = 50;
                    sparkEndY = is5dPos ? 6 : 14;
                }

                html += `
                    <div class="stealth-card p-2.5 space-y-1.5 overflow-hidden">
                        <div class="flex items-center justify-between gap-1 overflow-hidden min-w-0">
                            <div class="text-[9px] font-bold text-slate-200 whitespace-nowrap tracking-tighter shrink-0 flex items-center gap-0.5" title="${c.ticker} • ${c.name}">
                                <span>${c.ticker}</span><span class="text-slate-400 font-normal text-[8px] tracking-tighter">• ${c.name}</span>
                            </div>
                            <span class="${badgeClass} px-1 py-0.5 rounded text-[8px] font-semibold tabular-num shrink-0">${sign}${Number(p1d).toFixed(2)}%</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-xl font-bold text-white tracking-tight tabular-num">$${Number(c.price || 0).toFixed(2)}</span>
                            <div class="w-14 h-5 flex items-center shrink-0" title="${c.ticker} 5-Day Trend (${p5d >= 0 ? '+' : ''}${Number(p5d).toFixed(2)}%)">
                                <svg class="w-full h-full overflow-visible" viewBox="0 0 56 20">
                                    <path d="${sparkPath}" fill="none" stroke="${sparkColor}" stroke-width="1.5" stroke-linecap="round" />
                                    <circle cx="${sparkEndX}" cy="${sparkEndY}" r="2" fill="${sparkColor}" />
                                </svg>
                            </div>
                        </div>
                        <div class="pt-1.5 border-t border-stealth-border/70 space-y-0.5 text-[10.5px] tabular-num text-slate-400">
                            <div class="whitespace-nowrap overflow-hidden text-ellipsis">${c.sub1}</div>
                            <div class="text-slate-500 whitespace-nowrap overflow-hidden text-ellipsis">${c.sub2}</div>
                        </div>
                    </div>
                `;
            });
            const row = document.getElementById('benchmarksRow');
            if (row) row.innerHTML = html;
        }

        function renderMacroBarometers() {
            const r = DATA.market_data.ratios || {};
            const spy_rsp = r.SPY_RSP || {};
            const qqq_rsp = r.QQQ_RSP || {};
            const iwm_qqq = r.IWM_QQQ || {};

            const cards = [
                {
                    ticker: "SPY / RSP", name: "Market Breadth", subtitle: "Cap-Weighted vs Equal-Weight",
                    value: spy_rsp.current || 0,
                    sigma: spy_rsp.z_score_50d || 0,
                    state: spy_rsp.actionable_state || "Broad Breadth Expansion",
                    pct5d: spy_rsp.pct_5d || 0,
                    action: spy_rsp.action_text || "Equal-Weight RSP Outperforming"
                },
                {
                    ticker: "QQQ / RSP", name: "Tech Breadth", subtitle: "Tech Beta vs Broad Market",
                    value: qqq_rsp.current || 0,
                    sigma: qqq_rsp.z_score_50d || 0,
                    state: qqq_rsp.actionable_state || "Broadening Beyond Mega-Tech",
                    pct5d: qqq_rsp.pct_5d || 0,
                    action: qqq_rsp.action_text || "Broadening Beyond Mega-Tech"
                },
                {
                    ticker: "IWM / QQQ", name: "Small vs Tech", subtitle: "Small-Cap vs Mega-Cap Tech",
                    value: iwm_qqq.current || 0,
                    sigma: iwm_qqq.z_score_50d || 0,
                    state: iwm_qqq.actionable_state || "Aggressive Small-Cap Momentum",
                    pct5d: iwm_qqq.pct_5d || 0,
                    action: iwm_qqq.action_text || "Small-Cap Alpha Acceleration"
                }
            ];

            let html = '';
            cards.forEach(c => {
                const z = c.sigma || 0;
                const isZPos = z >= 0;
                const zSign = isZPos ? '+' : '';
                const zText = `${zSign}${z.toFixed(2)}σ`;
                const badgeCls = isZPos ? 'badge-mint shadow-[0_0_8px_rgba(0,245,160,0.2)]' : 'badge-cherry shadow-[0_0_8px_rgba(255,46,85,0.2)]';
                const needleColor = isZPos ? 'bg-mint shadow-[0_0_6px_#00F5A0]' : 'bg-cherry shadow-[0_0_6px_#FF2E55]';
                // Map z (-3 to +3) to percentage (5% to 95%)
                const needlePct = Math.max(5, Math.min(95, ((z + 3.0) / 6.0) * 100)).toFixed(1);

                const pct5d = c.pct5d || 0;
                const is5dPos = pct5d >= 0;
                const trendCol = is5dPos ? 'text-mint' : 'text-cherry';
                const trendSign = is5dPos ? '+' : '';

                // Pill styling for state
                let statusPillClass = 'text-xs text-mint font-semibold bg-mint/10 px-2 py-0.5 rounded border border-mint/20';
                const stLower = (c.state || '').toLowerCase();
                if (stLower.includes('range') || stLower.includes('neutral') || stLower.includes('consolidat')) {
                    statusPillClass = 'text-xs text-slate-300 font-semibold bg-stealth-850 px-2 py-0.5 rounded border border-stealth-border';
                } else if (stLower.includes('risk-off') || stLower.includes('bear') || stLower.includes('narrow') || stLower.includes('oversold') || stLower.includes('washout') || stLower.includes('overextend') || stLower.includes('peaking')) {
                    statusPillClass = 'text-xs text-cherry font-semibold bg-cherry/10 px-2 py-0.5 rounded border border-cherry/20';
                }

                html += `
                    <div class="stealth-card p-4 space-y-3 border-stealth-border hover:border-slate-700">
                        <div class="flex items-center justify-between">
                            <div>
                                <div class="text-xs font-bold text-white flex items-center space-x-1.5">
                                    <span>${c.ticker}</span>
                                    <span class="text-slate-500 font-normal text-[11px]">• ${c.name}</span>
                                </div>
                                <div class="text-[11px] text-slate-500">${c.subtitle}</div>
                            </div>
                            <span class="${badgeCls} px-2 py-1 rounded text-xs font-bold tabular-num">
                                ${zText}
                            </span>
                        </div>

                        <div class="flex items-baseline justify-between pt-1">
                            <span class="text-3xl font-bold text-white tracking-tight tabular-num">${Number(c.value).toFixed(2)}</span>
                            <span class="${statusPillClass}">
                                ${c.state}
                            </span>
                        </div>

                        <!-- Z-Score Gauge Spectrum Meter (-3σ to +3σ) -->
                        <div class="space-y-1 pt-1">
                            <div class="flex justify-between text-[10px] text-slate-500 font-mono">
                                <span>-3.0σ (Oversold)</span>
                                <span>0.0σ</span>
                                <span>+3.0σ (Extended)</span>
                            </div>
                            <div class="w-full h-1.5 rounded-full bg-stealth-800 relative overflow-hidden">
                                <div class="absolute left-1/2 top-0 bottom-0 w-0.5 bg-slate-600"></div>
                                <div class="absolute top-0 bottom-0 ${needleColor} rounded-full" style="left: ${needlePct}%; width: 8px; margin-left: -4px;"></div>
                            </div>
                        </div>

                        <div class="pt-2 border-t border-stealth-border/70 flex items-center justify-between text-xs text-slate-400">
                            <div>
                                5D Trend: <span class="${trendCol} font-semibold tabular-num">${trendSign}${Number(pct5d).toFixed(2)}%</span>
                            </div>
                            <div class="text-slate-500 text-[11px]">
                                ${c.action}
                            </div>
                        </div>
                    </div>
                `;
            });
            const row = document.getElementById('macroBarometersRow');
            if (row) row.innerHTML = html;
        }

        function renderBriefingBaskets() {
            const md = DATA.market_data || {};
            const t = document.getElementById('briefingTitle');
            if (t) t.innerText = md.narrative_title || "Selective Cyclical Rotation With Mega-Cap Tech Rebound";
            const d = document.getElementById('briefingDetail');
            if (d) d.innerText = md.narrative_detail || "Capital is aggressively rotating out of secular tech leadership and into mega-cap anchors and cyclicals.";
            const r = document.getElementById('briefingRiskPill');
            if (r) r.innerText = md.risk_level || "Selective / Dispersion";

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
                const col = val >= 0 ? 'text-mint' : 'text-cherry';
                const sign = val >= 0 ? '+' : '';
                bHtml += `
                    <div class="space-y-1">
                        <div class="text-xs text-slate-400 font-medium">${it.n}</div>
                        <div class="text-xl font-bold ${col} tabular-num tracking-tight">${sign}${Number(val).toFixed(2)}%</div>
                    </div>
                `;
            });
            const grid = document.getElementById('basketsGrid');
            if (grid) grid.innerHTML = bHtml;
        }

        async function loadAiBriefingAsync() {
            const synthBox = document.getElementById('synthesisText');
            if (synthBox) {
                synthBox.innerHTML = `
                    <div class="flex items-center gap-2 text-slate-400 py-3 text-xs">
                        <span class="inline-block animate-spin text-amber-400 font-mono">↻</span>
                        <span>AI Quant Strategist synthesizing live market regimes & setups...</span>
                    </div>
                `;
            }

            try {
                const url = USER_API_KEY ? `/api/ai-briefing?api_key=${encodeURIComponent(USER_API_KEY)}` : '/api/ai-briefing';
                const res = await fetch(url);
                if (!res.ok) {
                    const errBody = await res.text();
                    throw new Error(`Server returned HTTP ${res.status}: ${errBody.slice(0, 80)}`);
                }
                AI_DATA = await res.json();
                renderAiBriefingSynthesis();
                if (LOADED_TABS.has('tab2')) {
                    renderTradeCards();
                }
            } catch (err) {
                if (synthBox) {
                    synthBox.innerHTML = `
                        <div class="p-3 bg-cherry/10 border border-cherry/20 rounded-lg text-xs text-cherry flex items-center justify-between">
                            <span>AI Briefing unavailable: ${err.message}</span>
                            <button onclick="loadAiBriefingAsync()" class="px-2.5 py-1 bg-cherry/20 hover:bg-cherry/30 text-cherry rounded border border-cherry/30 transition text-[11px] font-medium">Retry</button>
                        </div>
                    `;
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
                    btn.className = 'synth-nav-pill px-3 py-1 rounded-md text-xs font-bold bg-mint text-stealth-950 shadow-[0_0_12px_rgba(0,245,160,0.3)] transition';
                } else {
                    btn.className = 'synth-nav-pill px-3 py-1 rounded-md text-xs font-semibold bg-stealth-850 hover:bg-stealth-800 text-slate-300 border border-stealth-border transition';
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
                { key: 'macro', icon: '', title: 'Macro Regime & Cross-Asset Liquidity', badge: 'AI Macro Liquidity', border: 'border-mint/30', bg: 'from-[#00F5A0]/5 to-transparent', titleColor: 'text-mint', isCherry: false },
                { key: 'sectors', icon: '', title: 'Sector Velocity & Breadth Concentration Dynamics', badge: 'AI Sector Dynamics', border: 'border-mint/30', bg: 'from-[#00F5A0]/5 to-transparent', titleColor: 'text-mint', isCherry: false },
                { key: 'setups', icon: '', title: 'Actionable Setups & Trade Confluence', badge: 'AI Confluence Setups', border: 'border-mint/30', bg: 'from-[#00F5A0]/5 to-transparent', titleColor: 'text-mint', isCherry: false },
                { key: 'risk', icon: '', title: 'Desk Game Plan & Tactical Risk Boundaries', badge: 'AI Tactical Boundaries', border: 'border-cherry/30', bg: 'from-[#FF2E55]/5 to-transparent', titleColor: 'text-cherry', isCherry: true }
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
                               '<div class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-mint animate-pulse"></span><span class="font-bold text-slate-200">Daily Institutional Macro Intelligence</span>' +
                               (dateStr ? '<span class="text-slate-500">•</span><span class="text-mint font-mono">' + dateStr + '</span>' : '') +
                               '</div><span class="text-[11px] text-mint font-bold uppercase tracking-wider bg-mint/10 border border-mint/25 px-2 py-0.5 rounded">100% AI Synthesized</span></div>';
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
                
                parsed = parsed.replace(new RegExp('<li>([\\s\\S]*?)<\\/li>', 'g'), function(m, inner) {
                    const low = inner.toLowerCase();
                    let cls = 'card-neutral';
                    if (low.indexOf('long') >= 0 || low.indexOf('bid') >= 0 || low.indexOf('continuation') >= 0 || low.indexOf('leading') >= 0 || low.indexOf('growth') >= 0 || low.indexOf('add risk') >= 0 || low.indexOf('capital deployment') >= 0) {
                        cls = 'card-long';
                    } else if (low.indexOf('distribution') >= 0 || low.indexOf('trap') >= 0 || low.indexOf('fade') >= 0 || low.indexOf('warning') >= 0 || low.indexOf('avoid') >= 0 || low.indexOf('lagging') >= 0 || low.indexOf('invalidation') >= 0 || low.indexOf('hedge') >= 0 || low.indexOf('trim') >= 0 || low.indexOf('portfolio defense') >= 0) {
                        cls = 'card-warn';
                    } else if (low.indexOf('tlt') >= 0 || low.indexOf('gld') >= 0 || low.indexOf('uso') >= 0 || low.indexOf('ibit') >= 0 || low.indexOf('treasur') >= 0 || low.indexOf('gold') >= 0 || low.indexOf('crude') >= 0) {
                        cls = 'card-macro';
                    }
                    return '<li class="' + cls + '">' + inner + '</li>';
                });
                
                const synthBadge = cfg.isCherry
                    ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-cherry/10 border border-cherry/25 text-cherry uppercase">AI Synthesized</span>'
                    : '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-mint/10 border border-mint/25 text-mint uppercase">AI Synthesized</span>';
                const catBadge = cfg.isCherry
                    ? '<span class="px-2 py-0.5 rounded text-[10.5px] font-bold bg-cherry/10 border border-cherry/25 text-cherry uppercase">' + cfg.badge + '</span>'
                    : '<span class="px-2 py-0.5 rounded text-[10.5px] font-bold bg-mint/10 border border-mint/25 text-mint uppercase">' + cfg.badge + '</span>';

                cardsHtml += '<div class="synth-section-card lux-card p-4 sm:p-5 mb-4 border ' + cfg.border + ' bg-gradient-to-b ' + cfg.bg + '" data-section="' + cfg.key + '">' +
                             '<div class="flex flex-wrap justify-between items-center gap-2 pb-3 mb-3 border-b border-white/5">' +
                             '<div class="flex items-center gap-2.5"><span class="text-sm font-extrabold ' + cfg.titleColor + ' uppercase tracking-wider">' + firstLine + '</span></div>' +
                             '<div class="flex items-center gap-1.5">' + synthBadge + catBadge + '</div></div>' +
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
                                <span>AI Institutional Synthesis Engine (Pillars 1 to 4)</span>
                            </div>
                            <span class="pill-amber text-[10.5px] font-bold">Awaiting AI Briefing</span>
                        </div>
                        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                            <div class="p-3.5 rounded-xl bg-black/20 border border-sky-500/20 space-y-1.5">
                                <div class="text-sky-400 font-extrabold uppercase text-[11px] flex items-center gap-1.5">Pillar 1: Macro Liquidity</div>
                                <div class="text-slate-300">${AI_DATA.regime_title || "Synthesizing yields, gold, oil & crypto liquidity..."}</div>
                            </div>
                            <div class="p-3.5 rounded-xl bg-black/20 border border-purple-500/20 space-y-1.5">
                                <div class="text-purple-400 font-extrabold uppercase text-[11px] flex items-center gap-1.5">Pillar 2: Sector Velocity</div>
                                <div class="text-slate-300">Tracking 5D capital rotation across Growth, Cyclicals, Defensives & Breadth...</div>
                            </div>
                            <div class="p-3.5 rounded-xl bg-black/20 border border-mint/20 space-y-1.5">
                                <div class="text-mint font-extrabold uppercase text-[11px] flex items-center gap-1.5">Pillar 3: Actionable Setups</div>
                                <div class="text-slate-300">Filtering algorithmic high-conviction long bids & distribution traps...</div>
                            </div>
                            <div class="p-3.5 rounded-xl bg-black/20 border border-amber-500/20 space-y-1.5">
                                <div class="text-amber-400 font-extrabold uppercase text-[11px] flex items-center gap-1.5">Pillar 4: Tactical Game Plan</div>
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
                        <span class="inline-block animate-spin text-amber-400 font-mono">↻</span>
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
                    const isLong = dir.includes("LONG") || dir.includes("DIP");
                    const borderCol = isLong ? '#00F5A0' : '#FF2E55';
                    const pill = isLong ? 'badge-mint' : 'badge-cherry';
                    const isCaution = /EXTENDED|TRAP|DO NOT|AWAIT|FADE|RISK/i.test(t.execution_state || '');
                    const execPill = isCaution ? 'badge-cherry' : 'badge-mint';
                    const extCol = (t.extension_metric && (t.extension_metric.includes('-') || isCaution)) ? 'text-cherry' : 'text-mint';
                    const payoffPill = isLong ? 'badge-mint' : 'badge-cherry';
                    const entryBg = isLong ? 'from-mint/10 to-transparent border-mint/25' : 'from-cherry/10 to-transparent border-cherry/25';
                    const entryCol = isLong ? 'text-mint' : 'text-cherry';

                    html += `
                        <div class="lux-card p-3 sm:p-3.5 border-l-4 flex flex-col justify-between h-full" style="border-left-color: ${borderCol}">
                            <div>
                                <div class="flex flex-wrap justify-between items-center gap-1.5 pb-2 mb-2 border-b border-white/5">
                                    <div class="flex flex-wrap items-center gap-2">
                                        <span class="text-lg font-black text-white">${t.ticker}</span>
                                        <span class="${pill} px-2 py-0.5 rounded text-[10px] font-bold">${t.direction}</span>
                                        <span class="text-[11px] font-semibold text-slate-400">${t.type_label}</span>
                                        ${t.execution_state ? `<span class="${execPill} px-2 py-0.5 rounded text-[10px] font-bold tracking-wide">${t.execution_state}</span>` : ''}
                                    </div>
                                    <div class="flex items-center gap-1.5 text-[10px]">
                                        <span class="text-slate-500 uppercase">Conviction:</span>
                                        <span class="${pill} px-2 py-0.5 rounded font-bold text-[10.5px]">Score: ${t.score} (${t.bias_label})</span>
                                    </div>
                                </div>

                                <!-- Market Price vs Optimized R:R Banner -->
                                <div class="flex flex-wrap items-center justify-between gap-1.5 p-1.5 px-2 mb-2 rounded-lg bg-white/[0.02] border border-white/5 text-[11px]">
                                    <div class="flex items-center gap-1.5 flex-wrap">
                                        <span class="text-slate-400">Print:</span>
                                        <span class="font-bold text-white">${t.current_price || t.entry}</span>
                                        <span class="text-slate-600">|</span>
                                        <span class="text-slate-400">20 EMA Ext:</span>
                                        <span class="font-mono ${extCol}">${t.extension_metric || 'At Support'}</span>
                                    </div>
                                    <div class="flex items-center gap-1.5">
                                        <span class="text-slate-500 uppercase text-[9.5px]">Payoff:</span>
                                        <span class="${payoffPill} px-2 py-0.5 rounded text-[10.5px] font-extrabold tracking-wide">${t.risk_reward || '1 : 3.0 R:R'}</span>
                                    </div>
                                </div>

                                <!-- 4-Box Execution Strategy 2x2 Grid -->
                                <div class="grid grid-cols-2 gap-2 mb-2">
                                    <div class="bg-gradient-to-b ${entryBg} p-2 rounded-lg border">
                                        <div class="text-[9px] font-bold ${entryCol} uppercase tracking-wider">Optimal Limit Entry</div>
                                        <div class="text-sm font-extrabold text-white mt-0.5">${t.optimal_entry_zone || t.entry}</div>
                                        <div class="text-[9px] text-slate-400 mt-0.5">${t.optimal_entry_label || '20 EMA Support Zone'}</div>
                                    </div>
                                    <div class="bg-white/[0.02] p-2 rounded-lg border border-white/5">
                                        <div class="text-[9px] font-bold text-slate-300 uppercase tracking-wider">Breakout Trigger</div>
                                        <div class="text-xs font-bold text-white mt-0.5 leading-snug">${t.breakout_trigger || 'Above 5D Pivot'}</div>
                                        <div class="text-[9px] text-slate-400 mt-0.5">Momentum Confirmation</div>
                                    </div>
                                    <div class="bg-gradient-to-b from-cherry/10 to-transparent p-2 rounded-lg border border-cherry/25">
                                        <div class="text-[9px] font-bold text-cherry uppercase tracking-wider">Stop Invalidation</div>
                                        <div class="text-sm font-extrabold text-cherry mt-0.5">${t.stop} <span class="text-[10px] font-normal text-rose-300">(${t.stop_pct})</span></div>
                                        <div class="text-[9px] text-slate-400 mt-0.5">Below Structural Support</div>
                                    </div>
                                    <div class="bg-gradient-to-b from-mint/10 to-transparent p-2 rounded-lg border border-mint/25">
                                        <div class="text-[9px] font-bold text-mint uppercase tracking-wider">Profit Targets</div>
                                        <div class="text-sm font-extrabold text-mint mt-0.5">${t.target1} <span class="text-[10px] font-normal text-emerald-300">(${t.target1_pct})</span></div>
                                        <div class="text-[9px] text-slate-400 mt-0.5">Runner: ${t.target2} (${t.target2_pct})</div>
                                    </div>
                                </div>

                                <!-- Execution Rule Directive -->
                                <div class="p-2 rounded-lg bg-black/40 border border-white/5 text-[10.5px] text-slate-300 mb-2 leading-snug">
                                    <div><b class="${isLong ? 'text-mint' : 'text-cherry'}">Desk Directive:</b> ${t.execution_desc || 'Bidding the structural liquidity band rather than market buying at extended highs.'}</div>
                                </div>
                            </div>

                            <div>
                                <div class="bg-white/[0.02] p-1.5 px-2 rounded-lg border border-white/5 text-[10.5px] text-slate-400 flex flex-wrap justify-between gap-1">
                                    <span>Win Rate: <b class="text-mint font-bold">${t.win_rate}</b> (N=${t.sample_size})</span>
                                    <span>Alpha: <b class="${isLong ? 'text-mint' : 'text-cherry'} font-bold">${t.median_alpha}</b></span>
                                    <span><b>Confirmations:</b> ${t.confirmations}</span>
                                </div>
                                <div class="text-[11px] text-slate-300 mt-1.5 leading-snug">
                                    <b class="text-white">Desk Rationale:</b> ${t.thesis}
                                </div>
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
                return 'text-cherry font-bold';
            }
            if (s.includes('Healthy') || s.includes('Support') || s.includes('Downside') || s.includes('Bounce') || s.includes('Bullish') || s.includes('Signal') || s.includes('Confirmed') || s.includes('Breakout')) {
                return 'text-mint font-bold';
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
            if (!items || !items.length) {
                loadConfluence('core');
                return;
            }
            const tickers = items.map(c => c.ticker);
            const scores = items.map(c => c.score);
            const colors = items.map(c => c.color_hex);
            const hover = items.map(c => `${c.ticker}: ${c.bias_label} (${c.score})<br>Bull: ${c.bull_count}/7 | Bear: ${c.bear_count}/7`);

            const chartHeight = Math.max(480, items.length * 26 + 100);
            const chartDiv = document.getElementById('confluenceChart');
            if (chartDiv) {
                chartDiv.style.height = chartHeight + 'px';
            }

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
                height: chartHeight,
                plot_bgcolor: "#0D0F14",
                paper_bgcolor: "#0D0F14",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 65, r: 25, t: 25, b: 60 },
                xaxis: { 
                    range: [-100, 100], 
                    gridcolor: "#1B212D", 
                    zerolinecolor: "#787b86", 
                    zerolinewidth: 2, 
                    title: {
                        text: "Conviction Score (-100 Bearish to +100 Bullish)",
                        font: { size: 11, color: "#94a3b8" }
                    }
                },
                yaxis: { autorange: "reversed", gridcolor: "#1B212D", dtick: 1 }
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
                        <td class="text-mint font-semibold">${c.bull_count} / 7</td>
                        <td class="text-cherry font-semibold">${c.bear_count} / 7</td>
                        <td class="${getStateColor(c.flicker_status)}">${c.flicker_status}</td>
                        <td>${edge.win_rate_20d || 0}%</td>
                        <td>${m.rsi || 0}</td>
                        <td>${m.rvol || 0}x</td>
                        <td class="${(m.dist_ema50 ?? m.dist_sma50 ?? 0) >= 0 ? 'text-mint' : 'text-cherry'} font-semibold">${(m.dist_ema50 ?? m.dist_sma50 ?? 0) >= 0 ? '+' : ''}${(m.dist_ema50 ?? m.dist_sma50 ?? 0).toFixed(2)}%</td>
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
                    border = 'border-mint';
                    bg = 'bg-mint/10';
                } else if (layer.status === 'PASS_BEAR') {
                    badge = '[BEARISH CONFIRMATION]';
                    border = 'border-cherry';
                    bg = 'bg-cherry/10';
                }
                const ptNum = parseInt(layer.points) || 0;
                const ptColor = ptNum > 0 ? 'text-mint' : (ptNum < 0 ? 'text-cherry' : 'text-slate-400');
                html += `
                    <div class="p-3 rounded-lg border-l-4 ${border} ${bg} flex justify-between items-start">
                        <div>
                            <div class="flex items-center gap-2">
                                <span class="font-bold text-white text-xs">${layer.layer}</span>
                                <span class="text-[10px] bg-black/40 px-1.5 py-0.5 rounded text-slate-300 font-mono">${badge}</span>
                            </div>
                            <p class="text-[11px] text-slate-400 mt-1">${layer.detail}</p>
                        </div>
                        <span class="text-xs font-bold ${ptColor}">${layer.points} pts</span>
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
                        <td class="${d.pct_1d >= 0 ? 'text-mint' : 'text-cherry'} font-semibold">${d.pct_1d >= 0 ? '+' : ''}${d.pct_1d}%</td>
                        <td class="${d.pct_5d >= 0 ? 'text-mint' : 'text-cherry'} font-semibold">${d.pct_5d >= 0 ? '+' : ''}${d.pct_5d}%</td>
                        <td class="${d.pct_20d >= 0 ? 'text-mint' : 'text-cherry'} font-semibold">${d.pct_20d >= 0 ? '+' : ''}${d.pct_20d}%</td>
                        <td class="${d.dist_ema20 >= 0 ? 'text-mint' : 'text-cherry'}">${d.dist_ema20 >= 0 ? '+' : ''}${d.dist_ema20}%</td>
                        <td class="${(d.dist_ema50 ?? d.dist_sma50) >= 0 ? 'text-mint' : 'text-cherry'}">${(d.dist_ema50 ?? d.dist_sma50) >= 0 ? '+' : ''}${d.dist_ema50 ?? d.dist_sma50}%</td>
                        <td>${d.rsi}</td>
                        <td>${d.rvol}x</td>
                        <td>${d.drawdown_ath}%</td>
                        <td class="text-mint font-bold">${edge.win_rate_20d || 0}%</td>
                        <td class="${edge.median_return_20d >= 0 ? 'text-mint' : 'text-cherry'}">${edge.median_return_20d >= 0 ? '+' : ''}${edge.median_return_20d || 0}%</td>
                        <td class="${getStateColor(d.tech_status)}">${d.tech_status}</td>
                    </tr>
                `;
            });
            document.getElementById('mag7TableBody').innerHTML = tHtml;

            // Load Mag 7 Normalized Return Chart
            loadMag7NormChart();
        }

        async function loadMag7NormChart() {
            const palette = ["#f59e0b", "#00F5A0", "#3b82f6", "#ec4899", "#8b5cf6", "#06b6d4", "#f97316"];
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
                        line: { color: palette[i % palette.length], width: 2, shape: 'spline', smoothing: 1.3 }
                    });
                }
            });
            const layout = {
                plot_bgcolor: "#0D0F14",
                paper_bgcolor: "#0D0F14",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 50, r: 20, t: 30, b: 30 },
                xaxis: { gridcolor: "#1B212D" },
                yaxis: { gridcolor: "#1B212D", title: "Return (%)" }
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
                        <td class="${d.pct_1d >= 0 ? 'text-mint' : 'text-cherry'} font-semibold">${d.pct_1d >= 0 ? '+' : ''}${d.pct_1d}%</td>
                        <td class="${d.pct_5d >= 0 ? 'text-mint' : 'text-cherry'} font-semibold">${d.pct_5d >= 0 ? '+' : ''}${d.pct_5d}%</td>
                        <td class="${d.pct_20d >= 0 ? 'text-mint' : 'text-cherry'} font-semibold">${d.pct_20d >= 0 ? '+' : ''}${d.pct_20d}%</td>
                        <td class="${(d.dist_ema50 ?? d.dist_sma50) >= 0 ? 'text-mint' : 'text-cherry'} font-semibold">${(d.dist_ema50 ?? d.dist_sma50) >= 0 ? '+' : ''}${d.dist_ema50 ?? d.dist_sma50}%</td>
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
            const bColors = bVals.map(v => v >= 0 ? '#00F5A0' : '#FF2E55');
            const trace = {
                x: bVals,
                y: bNames,
                type: 'bar',
                orientation: 'h',
                marker: { color: bColors }
            };
            const layout = {
                plot_bgcolor: "#0D0F14",
                paper_bgcolor: "#0D0F14",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 80, r: 20, t: 20, b: 30 },
                xaxis: { gridcolor: "#1B212D", title: "5D Return (%)" },
                yaxis: { gridcolor: "#1B212D" }
            };
            Plotly.newPlot('basketChart', [trace], layout, {responsive: true});
        }

        async function loadRatio(ratioName) {
            const res = await fetch(`/api/ratio/${ratioName}`);
            const data = await res.json();
            if (!data.dates) return;

            const traces = [
                { x: data.dates, y: data.upper2, name: "+2σ Extreme", mode: "lines", line: { color: "rgba(255, 46, 85, 0.7)", dash: "dot" } },
                { x: data.dates, y: data.lower2, name: "-2σ Extreme", mode: "lines", line: { color: "rgba(0, 245, 160, 0.7)", dash: "dot" } },
                { x: data.dates, y: data.ema20, name: "20-Day EMA", mode: "lines", line: { color: "#06b6d4", width: 1.5 } },
                { x: data.dates, y: data.ema50 || data.sma50, name: "50-Day EMA", mode: "lines", line: { color: "#f59e0b", width: 1.5 } },
                { x: data.dates, y: data.close, name: ratioName.replace('_', '/'), mode: "lines", line: { color: "#3b82f6", width: 2.5 } }
            ];
            const layout = {
                title: { text: `${ratioName.replace('_', '/')} Ratio with 20/50 EMA & ±2σ Envelopes`, font: { color: '#ffffff', size: 15 } },
                plot_bgcolor: "#0D0F14",
                paper_bgcolor: "#0D0F14",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 50, r: 20, t: 50, b: 30 },
                xaxis: { gridcolor: "#1B212D" },
                yaxis: { gridcolor: "#1B212D", title: "Ratio Value" }
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

        function populateOptionsTickersDropdown() {
            const select = document.getElementById('optionsTickerSelect');
            if (!select) return;
            const currentVal = select.value || 'QQQ';
            const tickers = DATA.available_tickers || [];
            const priority = ['QQQ', 'SPY', 'NVDA', 'AAPL', 'AMZN', 'MSFT', 'META', 'TSLA', 'IWM', 'GLD', 'XLF', 'SMH'];
            const tickerLabels = {
                'QQQ': 'QQQ (Nasdaq 100)',
                'SPY': 'SPY (S&P 500)',
                'NVDA': 'NVDA (Nvidia)',
                'AAPL': 'AAPL (Apple)',
                'TSLA': 'TSLA (Tesla)',
                'IWM': 'IWM (Russell 2000)',
                'GLD': 'GLD (Gold)',
                'MSFT': 'MSFT (Microsoft)',
                'AMZN': 'AMZN (Amazon)',
                'META': 'META (Meta)',
                'SMH': 'SMH (Semiconductors)',
                'XLF': 'XLF (Financials)',
                'GOOGL': 'GOOGL (Alphabet)',
                'AMD': 'AMD (Advanced Micro Devices)',
                'AVGO': 'AVGO (Broadcom)',
                'PLTR': 'PLTR (Palantir)',
                'NFLX': 'NFLX (Netflix)',
                'LLY': 'LLY (Eli Lilly)',
                'DELL': 'DELL (Dell)',
                'CRWD': 'CRWD (CrowdStrike)',
                'MU': 'MU (Micron)',
                'ORCL': 'ORCL (Oracle)',
                'JPM': 'JPM (JPMorgan)',
                'WMT': 'WMT (Walmart)',
                'COST': 'COST (Costco)',
                'TLT': 'TLT (20Y Treasury Bond)',
                'IBIT': 'IBIT (Bitcoin ETF)'
            };
            
            const universe = [...priority];
            tickers.forEach(t => {
                if (!universe.includes(t)) universe.push(t);
            });

            let html = '';
            universe.forEach(t => {
                const label = tickerLabels[t] || t;
                html += `<option value="${t}" ${t === currentVal ? 'selected' : ''}>${label}</option>`;
            });
            select.innerHTML = html;
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
                    <div class="text-xs ${m.pct_1d >= 0 ? 'text-mint' : 'text-red-400'} font-semibold mt-0.5">${m.pct_1d >= 0 ? '+' : ''}${m.pct_1d}% (1D)</div>
                </div>
                <div class="lux-card p-3.5">
                    <div class="text-[11px] font-semibold text-slate-400 uppercase">50-Day EMA Trend</div>
                    <div class="text-xl font-bold text-white mt-1">$${m.ema50 || m.sma50 || m.close}</div>
                    <div class="text-xs ${(m.dist_ema50 ?? m.dist_sma50) >= 0 ? 'text-mint' : 'text-red-400'} font-semibold mt-0.5">${(m.dist_ema50 ?? m.dist_sma50) >= 0 ? '+' : ''}${m.dist_ema50 ?? m.dist_sma50}% vs 50EMA</div>
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
                    <span>20-Day Win Rate: <b class="text-mint font-bold">${edge.win_rate_20d || 0}%</b></span>
                    <span>60-Day Win Rate: <b>${edge.win_rate_60d || 0}%</b></span>
                </div>
                <div class="text-xs text-slate-400">
                    Median 20-Day Alpha: <b class="text-amber-400 font-bold">+${edge.median_return_20d || 0}%</b> | Max Adverse Excursion: ${edge.max_adverse_excursion || 0}%
                </div>
            `;

            // Synchronized Plotly Subplot (Row 1: Price & EMAs, Row 2: Drawdown Curve)
            const tracePrice = { x: data.dates, y: data.close, name: "Close Price", type: "scatter", mode: "lines", line: { color: "#ffffff", width: 2 }, yaxis: 'y' };
            const traceEMA20 = { x: data.dates, y: data.ema20, name: "20 EMA (Red)", type: "scatter", mode: "lines", line: { color: "#FF2E55", width: 1.5 }, yaxis: 'y' };
            const traceEMA50 = { x: data.dates, y: data.ema50, name: "50 EMA (Amber)", type: "scatter", mode: "lines", line: { color: "#f59e0b", width: 1.5 }, yaxis: 'y' };
            const traceEMA200 = { x: data.dates, y: data.ema200, name: "200 EMA (Blue)", type: "scatter", mode: "lines", line: { color: "#3b82f6", width: 2 }, yaxis: 'y' };
            const traceDD = { x: data.dates, y: data.drawdown_ath, name: "ATH Drawdown (%)", type: "scatter", mode: "lines", fill: "tozeroy", fillcolor: "rgba(239, 68, 68, 0.15)", line: { color: "#FF2E55", width: 1.5 }, yaxis: 'y2' };

            const layout = {
                plot_bgcolor: "#0D0F14",
                paper_bgcolor: "#0D0F14",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 12 },
                margin: { l: 50, r: 20, t: 30, b: 30 },
                grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
                yaxis: { domain: [0.32, 1], gridcolor: "#1B212D", title: "Price ($)" },
                yaxis2: { domain: [0, 0.25], gridcolor: "#1B212D", title: "Drawdown (%)" },
                xaxis: { gridcolor: "#1B212D", anchor: 'y2' },
                hovermode: "x unified",
                legend: { orientation: "h", y: 1.05, x: 1, xanchor: "right" }
            };
            Plotly.newPlot('singleTickerCombinedChart', [tracePrice, traceEMA20, traceEMA50, traceEMA200, traceDD], layout, {responsive: true});
        }

        let CURRENT_OPTIONS_DIR = "Bullish";
        let CURRENT_OPTIONS_SIDE = "Buy";

        function setOptionsSide(side) {
            CURRENT_OPTIONS_SIDE = side;
            const btnBuy = document.getElementById('btnOptBuy');
            const btnSell = document.getElementById('btnOptSell');
            const titleEl = document.getElementById('optionsRecHeaderTitle');
            const subEl = document.getElementById('optionsRecHeaderSubtitle');

            if (side === 'Buy') {
                if (btnBuy) btnBuy.className = "px-3 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30";
                if (btnSell) btnSell.className = "px-3 py-1 rounded font-bold transition text-slate-400";
                if (titleEl) titleEl.innerHTML = `Top Recommended Cheap IV Contracts Across Different Expirations`;
                if (subEl) subEl.innerText = "Guaranteed multi-expiry diversity: Evaluates term cheapness (IV/HV), Ask-Edge (edge after crossing the spread to buy at Ask), and liquidity.";
            } else {
                if (btnSell) btnSell.className = "px-3 py-1 rounded font-bold transition bg-cherry/20 text-cherry border border-cherry/30";
                if (btnBuy) btnBuy.className = "px-3 py-1 rounded font-bold transition text-slate-400";
                if (titleEl) titleEl.innerHTML = `Top Recommended Overpriced Premium Sells (Bid Edge)`;
                if (subEl) subEl.innerText = "Harvest elevated premium: Evaluates rich term IV (IV/HV), Bid-Edge (edge after crossing spread to sell at Bid), and downside cushion.";
            }
            loadOptionsDesk();
        }

        function setOptionsDirection(dir) {
            CURRENT_OPTIONS_DIR = dir;
            const btnBull = document.getElementById('btnOptBullish');
            const btnBear = document.getElementById('btnOptBearish');
            if (dir === 'Bullish') {
                btnBull.className = "px-3 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30";
                btnBear.className = "px-3 py-1 rounded font-bold transition text-slate-400";
            } else {
                btnBear.className = "px-3 py-1 rounded font-bold transition bg-cherry/20 text-cherry border border-cherry/30";
                btnBull.className = "px-3 py-1 rounded font-bold transition text-slate-400";
            }
            loadOptionsDesk();
        }

        async function loadOptionsDesk(force = false) {
            const ticker = document.getElementById('optionsTickerSelect').value || 'QQQ';
            const kpiRow = document.getElementById('optionsKpiRow');
            const cardsContainer = document.getElementById('optionsRecCards');
            const isBuy = CURRENT_OPTIONS_SIDE === 'Buy';
            
            kpiRow.innerHTML = `<div class="col-span-full py-4 text-center text-xs text-slate-400">Scanning live option chains, IV skew, and historical volatilities for <b>${ticker}</b> (${CURRENT_OPTIONS_SIDE} Side)...</div>`;
            cardsContainer.innerHTML = `<div class="col-span-full py-8 text-center text-xs text-slate-500 animate-pulse">Calculating ${isBuy ? 'Ask-IV inversion' : 'Bid-IV premium'} and log-moneyness skew curve...</div>`;

            try {
                const res = await fetch(`/api/options/${ticker}?direction=${encodeURIComponent(CURRENT_OPTIONS_DIR)}&side=${encodeURIComponent(CURRENT_OPTIONS_SIDE)}`);
                const data = await res.json();
                if (data.error) {
                    kpiRow.innerHTML = `<div class="col-span-full text-xs text-amber-400 bg-amber-500/10 p-3 rounded-lg border border-amber-500/20">${data.error}</div>`;
                    cardsContainer.innerHTML = '';
                    return;
                }

                const h = data.hvs || {};
                const spot = data.spot_price;
                const earns = data.next_earnings || "Macro Index / ETF";

                // Render KPI Strip
                kpiRow.innerHTML = `
                    <div class="lux-card p-3.5">
                        <div class="text-[11px] font-semibold text-slate-400 uppercase">Spot Price</div>
                        <div class="text-xl font-bold text-white mt-1">$${spot.toFixed(2)}</div>
                        <div class="text-[11px] text-mint font-semibold mt-0.5">${ticker} Live Underlying</div>
                    </div>
                    <div class="lux-card p-3.5">
                        <div class="text-[11px] font-semibold text-slate-400 uppercase">10D Realized Vol</div>
                        <div class="text-xl font-bold text-white mt-1">${h.HV_10 || 20}%</div>
                        <div class="text-[11px] text-slate-400 mt-0.5">Tactical Momentum</div>
                    </div>
                    <div class="lux-card p-3.5">
                        <div class="text-[11px] font-semibold text-slate-400 uppercase">20D Realized Vol</div>
                        <div class="text-xl font-bold text-white mt-1">${h.HV_20 || 20}%</div>
                        <div class="text-[11px] text-slate-400 mt-0.5">Short-Term Baseline</div>
                    </div>
                    <div class="lux-card p-3.5 border-l-2 border-mint">
                        <div class="text-[11px] font-semibold text-mint uppercase">30D Clean HV</div>
                        <div class="text-xl font-bold text-mint mt-1">${h.HV_30_clean || h.HV_30 || 20}%</div>
                        <div class="text-[11px] text-slate-400 mt-0.5">De-Jumped (Winsorized)</div>
                    </div>
                    <div class="lux-card p-3.5">
                        <div class="text-[11px] font-semibold text-slate-400 uppercase">Earnings / Event</div>
                        <div class="text-base font-bold ${data.next_earnings ? 'text-mint' : 'text-slate-300'} mt-1 truncate">${earns}</div>
                        <div class="text-[11px] text-slate-400 mt-0.5">${data.next_earnings ? 'Binary Event Risk' : 'No Single Report'}</div>
                    </div>
                `;

                // Render Recommendations Cards
                const picks = data.picks || [];
                if (picks.length === 0) {
                    cardsContainer.innerHTML = `<div class="col-span-full text-center text-xs text-slate-400 py-6">No liquid ${isBuy ? 'cheap' : 'overpriced'} contracts identified matching criteria.</div>`;
                } else {
                    cardsContainer.innerHTML = picks.map((p, idx) => {
                        const isCall = isBuy ? (CURRENT_OPTIONS_DIR === 'Bullish') : (CURRENT_OPTIONS_DIR === 'Bearish');
                        const cType = isCall ? 'Call' : 'Put';
                        
                        let hTag = 'Tactical (10–21d)';
                        let badgeColor = 'text-mint';
                        if (p.DTE > 70) { hTag = 'Quarterly (75d+)'; badgeColor = 'text-mint'; }
                        else if (p.DTE > 45) { hTag = 'Core Monthly (60d)'; badgeColor = 'text-mint'; }
                        else if (p.DTE > 22) { hTag = 'Fast Swing (25–45d)'; badgeColor = 'text-mint'; }

                        const evTag = p.Spans_Earnings ? 'Spans Earnings' : 'Inter-Earnings';
                        const evColor = p.Spans_Earnings ? 'bg-cherry/15 text-cherry' : 'bg-mint/10 text-mint';
                        const spreadBadge = p.Spread_Quality || '🟢 Tight';

                        let edgeVal, edgeColor, edgeSign, ivVal, actionHtml, extraHtml, pickBadge, wingHtml = '';
                        
                        if (isBuy) {
                            edgeVal = p.Ask_Edge !== undefined ? p.Ask_Edge : 0;
                            edgeColor = edgeVal >= 0 ? 'text-mint' : 'text-cherry';
                            edgeSign = edgeVal >= 0 ? '+' : '';
                            ivVal = (p.Ask_IV_Pct !== undefined ? p.Ask_IV_Pct : p.IV_Pct).toFixed(1);
                            const beVal = isCall ? (p.strike + p.ask) : Math.max(p.strike - p.ask, 0.01);
                            const beDiff = ((beVal / spot) - 1) * 100;
                            pickBadge = `<span class="bg-mint/15 text-mint text-[10px] px-2 py-0.5 rounded font-bold">Pick #${idx + 1} • Buy</span>`;
                            actionHtml = `<div>Ask (Cost): <b class="text-mint">$${p.ask.toFixed(2)}</b> ($${Math.round(p.ask * 100)}/ct)</div>`;
                            extraHtml = `<div>Breakeven: <b class="text-slate-200">$${beVal.toFixed(2)} (${beDiff >= 0 ? '+' : ''}${beDiff.toFixed(1)}%)</b> • ${spreadBadge}</div>`;
                        } else {
                            edgeVal = p.Bid_Edge !== undefined ? p.Bid_Edge : 0;
                            edgeColor = edgeVal >= 0 ? 'text-mint' : 'text-cherry';
                            edgeSign = edgeVal >= 0 ? '+' : '';
                            ivVal = (p.Bid_IV_Pct !== undefined ? p.Bid_IV_Pct : p.IV_Pct).toFixed(1);
                            const cushionPct = (Math.abs(p.strike - spot) / spot) * 100;
                            pickBadge = `<span class="bg-cherry/15 text-cherry text-[10px] px-2 py-0.5 rounded font-bold">Pick #${idx + 1} • Sell</span>`;
                            actionHtml = `<div>Bid (Credit): <b class="text-mint">$${p.bid.toFixed(2)}</b> ($${Math.round(p.bid * 100)}/ct collected)</div>`;
                            extraHtml = `<div>Safety Cushion: <b class="text-slate-200">${cushionPct.toFixed(1)}% buffer</b> • ${spreadBadge}</div>`;
                            
                            if (p.Wing_Strike) {
                                wingHtml = `
                                <div class="mt-2.5 p-2 bg-white/[0.02] border border-dashed border-white/10 rounded text-[10.5px]">
                                    <div class="text-mint font-semibold flex items-center justify-between">
                                        <span>Hedged Spread Wing:</span>
                                        <span class="text-white font-mono">+$${Number(p.Wing_Strike).toFixed(0)} ${cType}</span>
                                    </div>
                                    <div class="text-slate-300 mt-1 flex justify-between">
                                        <span>Net Credit: <b class="text-mint">$${Number(p.Net_Credit).toFixed(2)}</b> ($${Math.round(p.Net_Credit*100)}/ct)</span>
                                        <span>Max Risk: <b>$${Number(p.Max_Risk).toFixed(2)}</b></span>
                                    </div>
                                    <div class="text-slate-400 text-[10px] mt-0.5 text-right">
                                        Return on Risk: <b class="text-mint font-mono">${Number(p.RoR_Pct).toFixed(1)}%</b>
                                    </div>
                                </div>`;
                            }
                        }

                        return `
                        <div class="lux-card p-4 border border-white/5 hover:border-mint/40 transition">
                            <div class="flex justify-between items-center mb-1">
                                <span class="text-base font-extrabold text-white">$${p.strike.toFixed(0)} ${cType}</span>
                                ${pickBadge}
                            </div>
                            <div class="flex items-center gap-2 mb-2">
                                <span class="text-[11px] font-semibold ${badgeColor}">${hTag}</span>
                                <span class="text-[10px] px-1.5 py-0.5 rounded ${evColor}">${evTag}</span>
                            </div>
                            <div class="bg-white/[0.03] p-2 rounded-lg flex justify-between items-center text-xs mb-3">
                                <span class="text-slate-200"><b>${p.Expiration}</b> (${p.DTE}d)</span>
                                <span class="text-[11px] ${p.Within_1_Sigma ? 'text-mint' : 'text-slate-400'}">${p.Within_1_Sigma ? 'In 1σ' : 'Out 1σ'}</span>
                            </div>
                            <div class="grid grid-cols-2 gap-2 text-xs mb-3">
                                <div><span class="text-slate-400 text-[11px]">${isBuy ? 'Ask IV:' : 'Bid IV:'}</span> <b class="text-white">${ivVal}%</b></div>
                                <div><span class="text-slate-400 text-[11px]">IV/HV:</span> <b class="text-white">${p.IV_HV_Ratio}x</b></div>
                                <div><span class="text-slate-400 text-[11px]">${isBuy ? 'Ask Edge:' : 'Bid Edge:'}</span> <b class="${edgeColor}">${edgeSign}${edgeVal.toFixed(2)}%</b></div>
                                <div><span class="text-slate-400 text-[11px]">Delta:</span> <b class="text-white">${p.Est_Delta.toFixed(2)}</b></div>
                            </div>
                            <div class="border-t border-white/5 pt-2 text-[11px] text-slate-400 space-y-0.5">
                                ${actionHtml}
                                <div>Bid / Mid: $${p.bid.toFixed(2)} / $${p.Mid_Price.toFixed(2)}</div>
                                ${extraHtml}
                                <div>Open Interest: <b class="text-slate-300">${(p.openInterest || 0).toLocaleString()}</b></div>
                                ${wingHtml}
                            </div>
                        </div>
                        `;
                    }).join('');
                }

                // Render Smile Chart
                if (data.smile) {
                    const s = data.smile;
                    document.getElementById('optionsSmileSubtitle').innerText = 
                        `Showing ${s.expiration} (${s.dte}d DTE) | Market 1σ Expected Move: ±$${s.exp_move} | ${isBuy ? 'Buy Side (Ask Edge)' : 'Sell Side (Bid Edge)'}`;

                    const traceModel = {
                        x: s.model_strikes, y: s.model_iv,
                        name: 'Model Fair Skew (Asymmetric Cubic)',
                        type: 'scatter', mode: 'lines',
                        line: { color: '#94a3b8', width: 2, dash: 'dash' }
                    };
                    const traceMarket = {
                        x: s.market_strikes, y: s.market_iv,
                        name: `${data.direction} Market Mid IV`,
                        type: 'scatter', mode: 'markers',
                        marker: { color: CURRENT_OPTIONS_DIR === 'Bullish' ? '#38bdf8' : '#f87171', size: 7, opacity: 0.8 }
                    };
                    const traceSpreadLine = isBuy ? {
                        x: s.market_strikes, y: s.market_ask_iv,
                        name: 'Effective Ask IV (Cost to Buy)',
                        type: 'scatter', mode: 'lines',
                        line: { color: 'rgba(234, 179, 8, 0.45)', width: 1.5 }
                    } : {
                        x: s.market_strikes, y: s.market_bid_iv,
                        name: 'Effective Bid IV (Selling Price)',
                        type: 'scatter', mode: 'lines',
                        line: { color: 'rgba(56, 189, 248, 0.45)', width: 1.5 }
                    };

                    const traceHighlightStrikes = {
                        x: s.top_strikes || [],
                        y: s.top_strikes_iv || [],
                        name: isBuy ? '🟢 Recommended Buy Strikes' : '🔴 Recommended Sell Strikes',
                        type: 'scatter',
                        mode: 'markers+text',
                        marker: {
                            color: isBuy ? '#00F5A0' : '#FF2E55',
                            size: 16,
                            symbol: 'circle',
                            line: { color: '#ffffff', width: 2.5 }
                        },
                        text: (s.top_strikes || []).map(st => isBuy ? `🟢 Buy $${st}` : `🔴 Sell $${st}`),
                        textposition: 'bottom center',
                        textfont: { color: isBuy ? '#00F5A0' : '#FF2E55', size: 11, family: 'Plus Jakarta Sans, sans-serif' },
                        hovertemplate: isBuy ? '<b>🟢 VALUE BUY STRIKE</b><br>Strike: $%{x}<br>Market IV: %{y:.2f}%<extra></extra>' : '<b>🔴 PREMIUM SELL STRIKE</b><br>Strike: $%{x}<br>Market IV: %{y:.2f}%<extra></extra>'
                    };

                    const layout = {
                        plot_bgcolor: "#0D0F14",
                        paper_bgcolor: "#0D0F14",
                        font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 11 },
                        margin: { l: 45, r: 25, t: 30, b: 35 },
                        xaxis: { title: "Strike Price ($)", gridcolor: "#1B212D" },
                        yaxis: { title: "Implied Volatility (%)", gridcolor: "#1B212D" },
                        legend: { orientation: "h", y: 1.12, x: 1, xanchor: "right" },
                        shapes: [
                            {
                                type: 'line', x0: spot, x1: spot, y0: 0, y1: 1, yref: 'paper',
                                line: { color: '#60a5fa', width: 1.5, dash: 'dot' }
                            },
                            {
                                type: 'rect', x0: spot - s.exp_move, x1: spot + s.exp_move, y0: 0, y1: 1, yref: 'paper',
                                fillcolor: 'rgba(96, 165, 250, 0.06)', line: { width: 0 }
                            }
                        ]
                    };
                    Plotly.newPlot('optionsSmileChart', [traceModel, traceMarket, traceSpreadLine, traceHighlightStrikes], layout, {responsive: true});
                }

            } catch (err) {
                kpiRow.innerHTML = `<div class="col-span-full text-xs text-cherry bg-cherry/10 p-3 rounded-lg border border-cherry/20">Error loading options desk: ${err.message}</div>`;
                cardsContainer.innerHTML = '';
            }
        }

        window.onload = init;
    </script>
</body>
</html>
"""
