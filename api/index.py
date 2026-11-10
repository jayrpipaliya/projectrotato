import os
import sys
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
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

app = FastAPI(title="Double Edge Quantitative Market Intelligence Desk")

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

def get_cached_ai(market_data, api_key: Optional[str] = None, force: bool = False):
    now = time.time()
    cache_key = f"ai_{api_key or 'default'}"
    if not force and cache_key in _CACHE and now - _CACHE.get(f"{cache_key}_time", 0) < 600:
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
def api_ai_briefing(api_key: Optional[str] = Query(None), force: bool = Query(False)):
    market_data = get_cached_analysis()
    ai_output = get_cached_ai(market_data, api_key=api_key, force=force)
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

@app.post("/api/sync-csv")
def api_sync_csv():
    """
    Reloads all tickers from local raw_csv files on disk, invalidating the in-memory cache,
    re-generating synthetic ratios, flushing analysis cache, and returning latest dates & counts.
    """
    try:
        t0 = time.time()
        from data_manager import (
            _IN_MEMORY_TICKER_DATA, get_available_tickers,
            load_ticker_data, generate_synthetic_ratios
        )
        # Flush in-memory cache so fresh reads occur from local disk raw_csv
        _IN_MEMORY_TICKER_DATA.clear()

        tickers = get_available_tickers()
        last_date = "N/A"
        loaded = []
        for t in tickers:
            df = load_ticker_data(t)
            if not df.empty and "Date" in df.columns:
                loaded.append(t)
                d_str = str(df["Date"].iloc[-1]).split()[0]
                if last_date == "N/A" or d_str > last_date:
                    last_date = d_str

        # Recalculate synthetic ratios from local CSVs
        generate_synthetic_ratios()

        # Clear computation cache
        _CACHE.clear()
        duration = round(time.time() - t0, 3)

        return {
            "status": "success",
            "source": "Local raw_csv files",
            "loaded_count": len(loaded),
            "total_available": len(tickers),
            "last_date": last_date,
            "sample_tickers": loaded[:8],
            "duration": duration,
            "timestamp": datetime.now().strftime("%I:%M:%S %p")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"CSV Sync failed: {str(e)}"
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

@app.get("/api/gamma/{ticker}")
def api_gamma(ticker: str, force: bool = Query(False)):
    try:
        from options_engine import compute_gamma_exposure_profile
        data = compute_gamma_exposure_profile(ticker.upper(), max_expirations=5, force_refresh=force)
        return clean_json(data)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/options/{ticker}")
def api_options(ticker: str, direction: str = Query("Calls"), side: str = Query("Buy"), expiration: str = Query(None)):
    try:
        t_up = ticker.upper()
        dir_clean = "Calls" if ("Call" in str(direction) or str(direction) == "Bullish") else "Puts"
        side_clean = "Sell" if "Sell" in str(side) else "Buy"
        from options_engine import compute_realized_volatilities, fetch_term_structure, scan_cross_chain_recommendations, analyze_smile_and_skew, compute_gamma_exposure_profile
        df_hist = load_ticker_data(t_up)
        hvs = compute_realized_volatilities(df_hist, ticker_symbol=t_up)
        term_df, spot_price, next_earnings = fetch_term_structure(t_up, hvs_dict=hvs)
        if term_df.empty or spot_price is None:
            return {"error": f"Live options data not available for {t_up}"}
            
        chains_list = term_df.to_dict('records')
        picks, ranked_chain = scan_cross_chain_recommendations(t_up, chains_list, spot_price, direction=dir_clean, side=side_clean, return_all=True)
        
        valid_exps = [str(e) for e in term_df['Expiration'].tolist()]
        if expiration and expiration in valid_exps:
            chosen_exp = expiration
        else:
            chosen_exp = valid_exps[0]
            
        smile = analyze_smile_and_skew(t_up, chosen_exp, spot_price, direction=dir_clean, side=side_clean)
        smile_payload = None
        if smile:
            target_df = smile['target_df']
            top_edges = smile['top_strikes']['Bid_Edge'] if side_clean == 'Sell' else smile['top_strikes']['Ask_Edge']
            
            # Compute term slope vs next expiration in term_df
            metrics = smile.get('metrics', {})
            try:
                exps = term_df['Expiration'].tolist()
                if chosen_exp in exps:
                    idx = exps.index(chosen_exp)
                    if idx + 1 < len(term_df):
                        next_row = term_df.iloc[idx + 1]
                        next_exp_name = str(next_row['Expiration'])
                        next_atm = float(next_row['ATM_IV'])
                        curr_atm = float(metrics.get('atm_iv', next_atm))
                        t_slope = round(next_atm - curr_atm, 1)
                        dt = pd.Timestamp(next_exp_name)
                        short_date = dt.strftime("%b %d")
                        metrics['term_slope'] = t_slope
                        if abs(t_slope) <= 0.3:
                            metrics['term_slope_text'] = f"Term flat vs {short_date}"
                        elif t_slope > 0:
                            metrics['term_slope_text'] = f"Near expiry cheap vs next (vs {short_date})"
                        else:
                            metrics['term_slope_text'] = f"Inverted / Backwardation (vs {short_date})"
                    else:
                        metrics['term_slope'] = 0.0
                        metrics['term_slope_text'] = "Front cycle"
            except Exception:
                metrics['term_slope'] = 0.0
                metrics['term_slope_text'] = "N/A"

            # Filter execution spread overlay to active tradable moneyness (eliminates deep ITM 50%+ IV spread spikes)
            if "Call" in dir_clean:
                exec_mask = (target_df['strike'] >= spot_price * 0.95) & (target_df['strike'] <= spot_price * 1.15)
            else:
                exec_mask = (target_df['strike'] <= spot_price * 1.05) & (target_df['strike'] >= spot_price * 0.85)
            exec_df = target_df[exec_mask] if exec_mask.any() else target_df

            smile_payload = {
                "expiration": chosen_exp,
                "dte": smile['dte'],
                "exp_move": round(smile['exp_move'], 2),
                "metrics": metrics,
                "structural_markers": smile.get('structural_markers', {}),
                "otm_put_curve": smile.get('otm_put_curve', {}),
                "otm_call_curve": smile.get('otm_call_curve', {}),
                "otm_puts_market": smile.get('otm_puts_market', {}),
                "otm_calls_market": smile.get('otm_calls_market', {}),
                "model_strikes": smile.get('otm_call_curve' if 'Call' in dir_clean else 'otm_put_curve', {}).get('strikes', []),
                "model_iv": smile.get('otm_call_curve' if 'Call' in dir_clean else 'otm_put_curve', {}).get('iv', []),
                "market_strikes": [round(float(s), 2) for s in exec_df['strike']],
                "market_iv": [round(float(iv), 2) for iv in exec_df['IV_Pct']],
                "market_ask_iv": [round(float(iv), 2) for iv in exec_df['Ask_IV_Pct']],
                "market_bid_iv": [round(float(iv), 2) for iv in exec_df['Bid_IV_Pct']],
                "top_strikes": [round(float(s), 2) for s in smile['top_strikes']['strike']],
                "top_strikes_iv": [round(float(iv), 2) for iv in smile['top_strikes']['IV_Pct']],
                "top_strikes_delta": [round(float(d), 2) for d in smile['top_strikes'].get('Est_Delta', [0]*len(smile['top_strikes']))],
                "top_strikes_edge": [round(float(e), 2) for e in top_edges]
            }
            
        expirations_list = [
            {
                "expiration": str(row['Expiration']),
                "dte": int(row['DTE']),
                "atm_iv": round(float(row.get('ATM_IV', 0)), 1),
                "iv_hv_ratio": round(float(row.get('IV_HV_Ratio', 1.0)), 2),
                "exp_move_pct": round(float(row.get('Expected_Move_Pct', 0)), 1),
                "exp_move_dollars": round(float(row.get('Expected_Move', 0)), 2)
            }
            for _, row in term_df.iterrows()
        ]
            
        gex_payload = None
        try:
            gex_data = compute_gamma_exposure_profile(t_up, max_expirations=5)
            if gex_data and "error" not in gex_data:
                gex_payload = gex_data
        except Exception:
            pass

        return clean_json({
            "ticker": t_up,
            "direction": dir_clean,
            "side": side_clean,
            "spot_price": round(spot_price, 2),
            "next_earnings": next_earnings.strftime("%b %d, %Y") if next_earnings else None,
            "hvs": hvs,
            "term_chains": chains_list[:8],
            "expirations": expirations_list,
            "selected_expiration": chosen_exp,
            "picks": picks,
            "ranked_chain": ranked_chain,
            "smile": smile_payload,
            "gamma": gex_payload
        })
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/options/{ticker}/smile")
def api_options_smile(ticker: str, expiration: str = Query(...), direction: str = Query("Calls"), side: str = Query("Buy"), spot: float = Query(None)):
    try:
        t_up = ticker.upper()
        dir_clean = "Calls" if ("Call" in str(direction) or str(direction) == "Bullish") else "Puts"
        side_clean = "Sell" if "Sell" in str(side) else "Buy"
        from options_engine import analyze_smile_and_skew
        spot_price = spot
        if spot_price is None or spot_price <= 0:
            df_hist = load_ticker_data(t_up)
            if not df_hist.empty:
                spot_price = float(df_hist['Close'].iloc[-1])
            else:
                import yfinance as yf
                t = yf.Ticker(t_up)
                spot_price = float(t.fast_info.last_price or t.fast_info.previous_close)
                
        smile = analyze_smile_and_skew(t_up, expiration, spot_price, direction=dir_clean, side=side_clean)
        if not smile:
            return {"error": f"Could not construct volatility smile for {expiration}"}
            
        target_df = smile['target_df']
        top_edges = smile['top_strikes']['Bid_Edge'] if side_clean == 'Sell' else smile['top_strikes']['Ask_Edge']
        
        # Calculate term slope vs adjacent cycle
        metrics = smile.get('metrics', {})
        try:
            from options_engine import fetch_term_structure
            term_df, _, _ = fetch_term_structure(t_up)
            if not term_df.empty and 'Expiration' in term_df.columns:
                exps = term_df['Expiration'].tolist()
                if expiration in exps:
                    idx = exps.index(expiration)
                    if idx + 1 < len(term_df):
                        next_row = term_df.iloc[idx + 1]
                        next_exp_name = str(next_row['Expiration'])
                        next_atm = float(next_row['ATM_IV'])
                        curr_atm = float(metrics.get('atm_iv', next_atm))
                        t_slope = round(next_atm - curr_atm, 1)
                        dt = pd.Timestamp(next_exp_name)
                        short_date = dt.strftime("%b %d")
                        metrics['term_slope'] = t_slope
                        if abs(t_slope) <= 0.3:
                            metrics['term_slope_text'] = f"Term flat vs {short_date}"
                        elif t_slope > 0:
                            metrics['term_slope_text'] = f"Near expiry cheap vs next (vs {short_date})"
                        else:
                            metrics['term_slope_text'] = f"Inverted / Backwardation (vs {short_date})"
        except Exception:
            metrics['term_slope'] = 0.0
            metrics['term_slope_text'] = "N/A"
            
        # Filter execution spread overlay to active tradable moneyness (eliminates deep ITM 50%+ IV spread spikes)
        if "Call" in dir_clean:
            exec_mask = (target_df['strike'] >= spot_price * 0.95) & (target_df['strike'] <= spot_price * 1.15)
        else:
            exec_mask = (target_df['strike'] <= spot_price * 1.05) & (target_df['strike'] >= spot_price * 0.85)
        exec_df = target_df[exec_mask] if exec_mask.any() else target_df

        return clean_json({
            "ticker": t_up,
            "direction": dir_clean,
            "side": side_clean,
            "spot_price": round(spot_price, 2),
            "smile": {
                "expiration": expiration,
                "dte": smile['dte'],
                "exp_move": round(smile['exp_move'], 2),
                "metrics": metrics,
                "structural_markers": smile.get('structural_markers', {}),
                "otm_put_curve": smile.get('otm_put_curve', {}),
                "otm_call_curve": smile.get('otm_call_curve', {}),
                "otm_puts_market": smile.get('otm_puts_market', {}),
                "otm_calls_market": smile.get('otm_calls_market', {}),
                "model_strikes": smile.get('otm_call_curve' if 'Call' in dir_clean else 'otm_put_curve', {}).get('strikes', []),
                "model_iv": smile.get('otm_call_curve' if 'Call' in dir_clean else 'otm_put_curve', {}).get('iv', []),
                "market_strikes": [round(float(s), 2) for s in exec_df['strike']],
                "market_iv": [round(float(iv), 2) for iv in exec_df['IV_Pct']],
                "market_ask_iv": [round(float(iv), 2) for iv in exec_df['Ask_IV_Pct']],
                "market_bid_iv": [round(float(iv), 2) for iv in exec_df['Bid_IV_Pct']],
                "top_strikes": [round(float(s), 2) for s in smile['top_strikes']['strike']],
                "top_strikes_iv": [round(float(iv), 2) for iv in smile['top_strikes']['IV_Pct']],
                "top_strikes_delta": [round(float(d), 2) for d in smile['top_strikes'].get('Est_Delta', [0]*len(smile['top_strikes']))],
                "top_strikes_edge": [round(float(e), 2) for e in top_edges]
            }
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
  <title>Double Edge // Stealth Quantitative Macro Desk</title>
  
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

    /* AI Terminal Chat Log Styling */
    #chatHistory table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      margin: 10px 0 14px 0;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #1B212D;
      background: #090B10;
      font-size: 11.5px;
    }
    #chatHistory th {
      background: #131720;
      color: #94a3b8;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 10px;
      letter-spacing: 0.05em;
      padding: 8px 10px;
      border-bottom: 1px solid #1B212D;
      text-align: left;
    }
    #chatHistory td {
      padding: 7px 10px;
      border-bottom: 1px solid rgba(27, 33, 45, 0.7);
      color: #e2e8f0;
      font-variant-numeric: tabular-nums;
    }
    #chatHistory tr:last-child td {
      border-bottom: none;
    }
    #chatHistory tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }
    #chatHistory blockquote {
      margin: 10px 0;
      padding: 8px 14px;
      border-left: 3px solid #00F5A0;
      background: rgba(0, 245, 160, 0.05);
      border-radius: 0 8px 8px 0;
      color: #cbd5e1;
      font-size: 12px;
    }
    #chatHistory blockquote p {
      margin: 0;
    }
    #chatHistory blockquote:has(🛑), #chatHistory blockquote:has(⚠️) {
      border-left: 3px solid #FF2E55;
      background: rgba(255, 46, 85, 0.06);
    }
    #chatHistory h2 {
      font-size: 13.5px;
      font-weight: 700;
      color: #ffffff;
      margin-top: 14px;
      margin-bottom: 6px;
    }
    #chatHistory h3 {
      font-size: 12.5px;
      font-weight: 700;
      color: #f1f5f9;
      margin-top: 14px;
      margin-bottom: 6px;
      padding-bottom: 4px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    #chatHistory h4 {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #94a3b8;
      margin-top: 10px;
      margin-bottom: 4px;
    }
    #chatHistory hr {
      border: 0;
      border-top: 1px solid #1B212D;
      margin: 12px 0;
    }
    #chatHistory strong, #chatHistory b {
      color: #ffffff;
      font-weight: 600;
    }
    #chatHistory code {
      background: rgba(255, 255, 255, 0.06);
      padding: 1px 4px;
      border-radius: 4px;
      font-size: 11px;
      color: #38bdf8;
    }

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
          <span class="text-white font-extrabold text-xs">DE</span>
          <span class="w-1.5 h-1.5 rounded-full bg-mint absolute top-1.5 right-1.5 shadow-[0_0_6px_#00F5A0]"></span>
        </div>
        <div>
          <h1 class="font-bold text-base tracking-tight text-white leading-tight">Double Edge</h1>
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
          <button onclick="triggerCsvSync()" id="syncCsvBtn" class="w-full py-2.5 px-3 rounded-lg bg-stealth-900 hover:bg-stealth-850 border border-stealth-border hover:border-mint/50 text-xs font-semibold text-slate-300 hover:text-white transition-all flex items-center justify-between group shadow-sm">
            <div class="flex items-center space-x-2.5">
              <div class="w-6 h-6 rounded-md bg-stealth-800 flex items-center justify-center text-slate-400 group-hover:text-mint group-hover:scale-105 transition-transform">
                <span id="syncCsvSpinner" class="hidden inline-block animate-spin font-mono">↻</span>
                <svg id="syncCsvIconDefault" class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <span id="syncCsvBtnText" class="tracking-tight">Sync from CSV</span>
            </div>
            <span class="w-1.5 h-1.5 rounded-full bg-slate-500 group-hover:bg-mint transition-colors"></span>
          </button>
        </div>
      </div>

      <!-- Section: AI STRATEGIST ENGINE -->
      <div class="space-y-3 pt-3 border-t border-stealth-border">
        <div class="text-[11px] font-bold text-slate-400 tracking-wider uppercase">AI Strategist Engine</div>
        
        <div>
          <label class="block text-xs text-slate-400 mb-1.5 font-medium">Gemini API Key</label>
          <div class="relative">
            <input type="password" id="sidebarApiKey" placeholder="Enter Gemini Key..." onchange="saveSidebarApiKey()"
                   class="w-full bg-stealth-900 border border-stealth-border focus:border-mint/50 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono focus:outline-none transition-colors" />
          </div>
        </div>

        <!-- Save Key for Session Button -->
        <button onclick="saveSidebarApiKey()" id="saveKeyBtn" class="w-full py-2.5 px-3 rounded-lg bg-mint/10 hover:bg-mint hover:text-stealth-950 text-mint border border-mint/30 text-xs font-bold transition-all shadow-[0_0_12px_rgba(0,245,160,0.15)] flex items-center justify-center space-x-1.5">
          <svg class="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
          <span id="saveKeyBtnText">Save Key for Session</span>
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
          Magnificant 7 Flow
        </button>
        <button onclick="switchTab('tab6')" id="btn-tab6" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Synthetic Ratio Rotations
        </button>
        <button onclick="switchTab('tab7')" id="btn-tab7" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Single Ticker Explorer
        </button>
        <button onclick="switchTab('tab_gamma')" id="btn-tab_gamma" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Dealer Gamma Desk
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
                <div id="chatHistory" class="space-y-3 min-h-[320px] max-h-[620px] overflow-y-auto p-4 rounded-xl bg-[#090a0f] border border-white/5 mb-4">
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

            <!-- Navigation View Filter Pills & Regenerate Button -->
            <div class="flex flex-wrap items-center justify-between gap-1.5 p-1.5 rounded-lg bg-stealth-950 border border-stealth-border">
              <div class="flex flex-wrap items-center gap-1.5">
                <span class="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-2">Focus Pillar:</span>
                <button onclick="filterSynthesisSection('all')" id="synthPill-all" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-bold bg-mint text-stealth-950 shadow-sm transition">All 4 Pillars</button>
                <button onclick="filterSynthesisSection('macro')" id="synthPill-macro" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-semibold bg-stealth-850 hover:bg-stealth-800 text-slate-300 transition">1. Macro Liquidity</button>
                <button onclick="filterSynthesisSection('sectors')" id="synthPill-sectors" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-semibold bg-stealth-850 hover:bg-stealth-800 text-slate-300 transition">2. Sector Velocity</button>
                <button onclick="filterSynthesisSection('setups')" id="synthPill-setups" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-semibold bg-stealth-850 hover:bg-stealth-800 text-slate-300 transition">3. Setups & Traps</button>
                <button onclick="filterSynthesisSection('risk')" id="synthPill-risk" class="synth-nav-pill px-3 py-1 rounded-md text-xs font-semibold bg-stealth-850 hover:bg-stealth-800 text-slate-300 transition">4. Desk Game Plan</button>
              </div>
              <button onclick="loadAiBriefingAsync(true)" title="Force fresh AI synthesis bypassing cache" class="px-2.5 py-1 rounded-md text-xs font-bold bg-amber-500/15 hover:bg-amber-500/25 text-amber-400 border border-amber-500/30 transition flex items-center gap-1.5 ml-auto">
                <span>↻</span> <span>Regenerate Briefing</span>
              </button>
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
                        <button onclick="setHorizon('1 Year', this)" class="horizon-btn px-2.5 py-1 rounded text-slate-400">1Y</button>
                        <button onclick="setHorizon('2 Years', this)" class="horizon-btn px-2.5 py-1 rounded text-slate-400">2Y</button>
                        <button onclick="setHorizon('5 Years', this)" class="horizon-btn px-2.5 py-1 rounded text-slate-400">5Y</button>
                        <button onclick="setHorizon('All Historical Data', this)" class="horizon-btn active bg-[#1c1e27] text-amber-400 font-bold px-2.5 py-1 rounded">All</button>
                    </div>
                </div>
            </div>
            
            <div id="singleTickerMetricsRow" class="grid grid-cols-1 sm:grid-cols-4 gap-3.5 mb-4"></div>
            <div id="singleTickerEdgeCard" class="lux-card p-4 border-l-4 border-amber-500 mb-5"></div>
            
            <div class="lux-card p-4">
                <div id="singleTickerCombinedChart" style="height: 600px;"></div>
            </div>
        </div>

        <!-- Tab Gamma: Dealer Gamma Levels Desk -->
        <div id="tab_gamma" class="tab-pane hidden">
            <!-- Header Card -->
            <div class="lux-card p-5 mb-5 border border-indigo-500/30 bg-gradient-to-r from-indigo-950/30 via-[#0d1117] to-slate-900/50">
                <div class="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
                    <div>
                        <div class="flex items-center gap-2">
                            <h2 class="text-lg font-black text-white tracking-wide">⚡ Institutional Dealer Gamma Exposure (GEX) Desk</h2>
                            <span class="text-[10px] font-bold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">Live Market Maker Hedging Map</span>
                        </div>
                        <p class="text-xs text-slate-400 mt-1 max-w-2xl">
                            Inverts Black-Scholes gamma across open interest to quantify market maker hedging inventory: <b>Call Wall (Ceiling Pin)</b>, <b>Put Wall (Support Floor)</b>, <b>Zero-Gamma Flip</b>, and aggregate <b>Net Dealer GEX</b> regime.
                        </p>
                    </div>

                    <div class="flex flex-wrap items-center gap-3">
                        <div>
                            <label class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Underlying Ticker</label>
                            <select id="gammaDeskTickerSelect" onchange="loadGammaDesk()" class="bg-[#0b0d12] text-xs font-bold text-white border border-white/10 px-3 py-1.5 rounded-lg focus:outline-none focus:border-indigo-400">
                                <!-- Populated dynamically -->
                            </select>
                        </div>

                        <div class="self-end">
                            <button onclick="loadGammaDesk(true)" class="px-3.5 py-1.5 rounded-lg bg-indigo-500 hover:bg-indigo-400 text-white font-bold text-xs shadow-lg shadow-indigo-500/20 transition flex items-center gap-1.5">
                                ↻ Recalculate GEX
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Top 5 KPI Strip -->
            <div id="gammaDeskKpiRow" class="grid grid-cols-2 sm:grid-cols-5 gap-3.5 mb-5">
                <!-- Populated dynamically: Spot, Call Wall, Put Wall, Zero Gamma Flip, Total Net GEX -->
            </div>

            <!-- Tactical Regime Directive Banner -->
            <div id="gammaDeskDirective" class="mb-5">
                <!-- Populated dynamically with regime intelligence & execution playbook -->
            </div>

            <!-- Interactive Plotly Strike GEX Bar Chart -->
            <div class="lux-card p-4 mb-5">
                <div class="flex justify-between items-center mb-2">
                    <div>
                        <h4 class="text-sm font-bold text-white flex items-center gap-2">
                            <span>Per-Strike Dealer Gamma Exposure Profile ($M per 1% Move)</span>
                            <span id="gammaDeskChartBadge" class="text-[10px] font-bold px-2 py-0.5 rounded bg-white/5 text-slate-300">Near-Term Expirations</span>
                        </h4>
                        <p class="text-[11px] text-slate-400">Green = Positive Call Gamma (Dealers Long Calls / Hedging Resistance). Red = Negative Put Gamma (Dealers Short Puts / Hedging Floor).</p>
                    </div>
                </div>
                <div id="gammaDeskChart" style="height: 420px;"></div>
            </div>

            <!-- Comprehensive Per-Strike Table Breakdown -->
            <div class="lux-card p-4 mb-5">
                <div class="flex justify-between items-center mb-3">
                    <div>
                        <h4 class="text-sm font-bold text-white">Detailed Strike Gamma & Open Interest Breakdown</h4>
                        <p class="text-[11px] text-slate-400">All liquid strikes within ±12% of spot price, including dealer net delta impact and structural wall tags.</p>
                    </div>
                    <span class="text-[10px] font-mono text-slate-400" id="gammaDeskStrikesCount">0 Strikes Mapped</span>
                </div>
                <div class="overflow-x-auto rounded-lg border border-white/10 bg-[#0b0d12]">
                    <table class="w-full text-left text-xs">
                        <thead class="bg-white/[0.03] text-[11px] uppercase tracking-wider text-slate-400 border-b border-white/10">
                            <tr>
                                <th class="py-2.5 px-3">Strike Price</th>
                                <th class="py-2.5 px-3">Dist vs Spot</th>
                                <th class="py-2.5 px-3">Call GEX ($M)</th>
                                <th class="py-2.5 px-3">Put GEX ($M)</th>
                                <th class="py-2.5 px-3">Net GEX ($M)</th>
                                <th class="py-2.5 px-3">Call OI</th>
                                <th class="py-2.5 px-3">Put OI</th>
                                <th class="py-2.5 px-3">Structural Role</th>
                            </tr>
                        </thead>
                        <tbody id="gammaDeskStrikesTbody">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Institutional Desk Guide 3-Box Educational Strip -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div class="lux-card p-4 border-t-2 border-mint">
                    <h5 class="text-xs font-bold text-mint uppercase tracking-wider mb-1.5">1. Call Wall & Dealer Pins</h5>
                    <p class="text-[11.5px] text-slate-400 leading-relaxed">
                        The strike with the largest open Call Gamma. Because dealers are typically net short these calls, they must sell futures/stock as spot rallies into the strike, creating a natural upside ceiling and pinning magnet into weekly OpEx.
                    </p>
                </div>
                <div class="lux-card p-4 border-t-2 border-cherry">
                    <h5 class="text-xs font-bold text-cherry uppercase tracking-wider mb-1.5">2. Put Wall & Cushion Floors</h5>
                    <p class="text-[11.5px] text-slate-400 leading-relaxed">
                        The strike with the largest open Put Gamma. Dealers short these puts must buy stock when price drops into the strike, creating institutional liquidity, limiting sell-offs, and triggering violent rebound squeezes.
                    </p>
                </div>
                <div class="lux-card p-4 border-t-2 border-sky-400">
                    <h5 class="text-xs font-bold text-sky-400 uppercase tracking-wider mb-1.5">3. Zero-Gamma Vol Flip</h5>
                    <p class="text-[11.5px] text-slate-400 leading-relaxed">
                        The inflection strike where cumulative dealer gamma crosses zero. Above it, market makers operate in Long Gamma (volatility suppressing). Below it, market makers flip to Short Gamma (selling drops, buying rallies), triggering fast trending expansions.
                    </p>
                </div>
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
                            <label class="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Contract Type</label>
                            <div class="flex bg-white/5 p-1 rounded-lg text-xs" id="optionsDirGroup">
                                <button onclick="setOptionsDirection('Calls')" id="btnOptCalls" class="px-3 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30">Calls</button>
                                <button onclick="setOptionsDirection('Puts')" id="btnOptPuts" class="px-3 py-1 rounded font-bold transition text-slate-400">Puts</button>
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
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-3">
                    <div>
                        <h4 class="text-sm font-bold text-white flex items-center gap-1.5">
                            Volatility Skew <span class="text-slate-500 hover:text-slate-300 cursor-pointer text-xs" title="Institutional Volatility Surface: Stitched OTM Puts (Cherry Red) & OTM Calls (Mint Green) with Risk Reversal Skew Metrics">ⓘ</span>
                        </h4>
                        <p class="text-[11px] text-slate-400" id="optionsSmileSubtitle">Showing nearest active expiration chain, monotonic PCHIP skew surface, and 1σ expected move boundary</p>
                    </div>
                    <!-- Expiration Date Slicer / Selector -->
                    <div class="flex items-center gap-2">
                        <label for="optionsExpSelect" class="text-xs font-bold text-slate-400 whitespace-nowrap">Expiration Date:</label>
                        <select id="optionsExpSelect" onchange="onExpirationChange(this.value)" class="bg-[#0b0d12] text-xs font-semibold text-white border border-white/10 px-3 py-1.5 rounded-lg focus:outline-none focus:border-mint min-w-[210px]">
                            <option value="">Nearest Active Chain</option>
                        </select>
                    </div>
                </div>

                <!-- 6 Institutional Skew KPI Cards -->
                <div id="optionsSkewKpiStrip" class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 mb-4">
                    <!-- Populated dynamically: ATM IV, 25Δ SKEW, 25Δ BUTTERFLY, 10Δ SKEW, SKEW SLOPE, TERM SLOPE -->
                </div>

                <div id="optionsSmileChart" style="height: 440px;"></div>

                <!-- Whole-Chain Tiered Discount Scanner Table -->
                <div class="mt-6 border-t border-white/10 pt-5">
                    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                        <div>
                            <h4 class="text-sm font-bold text-white flex items-center gap-2">
                                <span>Whole-Chain Value Scanner: Top Discounted Contracts</span>
                                <span id="chainDiscountBadge" class="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-mint/15 text-mint border border-mint/20">Ranked by Edge</span>
                            </h4>
                            <p class="text-xs text-slate-400 mt-0.5">Scans all expirations across the volatility surface to rank contracts with the highest mathematical edge surviving the spread.</p>
                        </div>
                        <div class="text-xs text-slate-400 hidden sm:flex items-center gap-2">
                            <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-mint"></span> Tier 1 Prime</span>
                            <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-sky-400"></span> Tier 2 High</span>
                            <span class="inline-flex items-center gap-1"><span class="w-2 h-2 rounded-full bg-purple-400"></span> Tier 3 Selective</span>
                        </div>
                    </div>
                    <div class="overflow-x-auto rounded-lg border border-white/10 bg-[#0b0d12]">
                        <table class="w-full text-left text-xs">
                            <thead class="bg-white/[0.03] text-[11px] uppercase tracking-wider text-slate-400 border-b border-white/10">
                                <tr>
                                    <th class="py-2.5 px-3">Tier / Rank</th>
                                    <th class="py-2.5 px-3">Expiration (DTE)</th>
                                    <th class="py-2.5 px-3">Strike</th>
                                    <th class="py-2.5 px-3" id="thChainCost">Ask Cost</th>
                                    <th class="py-2.5 px-3">Market IV</th>
                                    <th class="py-2.5 px-3">OI (Vol)</th>
                                    <th class="py-2.5 px-3" id="thChainEdge">Ask Edge</th>
                                    <th class="py-2.5 px-3">Delta</th>
                                </tr>
                            </thead>
                            <tbody id="optionsChainDiscountTbody" class="divide-y divide-white/5">
                                <tr><td colspan="8" class="text-center py-4 text-slate-500">Loading whole-chain candidates...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
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
            const btnText = document.getElementById('saveKeyBtnText');
            if (btnText) {
                btnText.innerText = key ? 'Key Saved!' : 'Key Cleared!';
                setTimeout(() => { btnText.innerText = 'Save Key for Session'; }, 1800);
            }
            loadAiBriefingAsync(true);
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
                loadAiBriefingAsync(true);
            }
        }

        async function triggerDataSync() {
            const hBtn = document.getElementById('headerSyncBtn');
            const hSpinner = document.getElementById('headerSyncSpinner');
            const hBtnText = document.getElementById('headerSyncBtnText');
            const sBtn = document.getElementById('syncBtn');
            const sSpinner = document.getElementById('syncSpinner');
            const sIcon = document.getElementById('syncIconDefault');
            const sBtnText = document.getElementById('syncBtnText');

            if (hBtn) hBtn.disabled = true;
            if (sBtn) sBtn.disabled = true;
            if (hSpinner) hSpinner.classList.remove('hidden');
            if (sSpinner) sSpinner.classList.remove('hidden');
            if (sIcon) sIcon.classList.add('hidden');
            if (hBtnText) hBtnText.innerText = 'Syncing...';
            if (sBtnText) sBtnText.innerText = 'Syncing YFinance...';

            try {
                const res = await fetch('/api/sync-data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ scope: 'core' })
                });

                if (!res.ok) {
                    throw new Error(`Sync returned HTTP ${res.status}`);
                }
                const syncResult = await res.json();

                const marketRes = await fetch('/api/market-data');
                DATA = await marketRes.json();

                renderBenchmarks();
                renderMacroBarometers();
                renderBriefingBaskets();
                renderRegime();
                loadAiBriefingAsync(true);

                if (hBtnText) hBtnText.innerText = 'Synced!';
                if (sBtnText) sBtnText.innerText = `YFinance Synced! (${syncResult.last_date || 'Today'})`;
                setTimeout(() => {
                    if (hBtnText) hBtnText.innerText = 'Sync Market Feeds';
                    if (sBtnText) sBtnText.innerText = 'Sync from YFinance';
                    if (hBtn) hBtn.disabled = false;
                    if (sBtn) sBtn.disabled = false;
                    if (hSpinner) hSpinner.classList.add('hidden');
                    if (sSpinner) sSpinner.classList.add('hidden');
                    if (sIcon) sIcon.classList.remove('hidden');
                }, 2200);

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
                    if (sIcon) sIcon.classList.remove('hidden');
                }, 2500);
            }
        }

        async function triggerCsvSync() {
            const sBtn = document.getElementById('syncCsvBtn');
            const sSpinner = document.getElementById('syncCsvSpinner');
            const sIcon = document.getElementById('syncCsvIconDefault');
            const sBtnText = document.getElementById('syncCsvBtnText');

            if (sBtn) sBtn.disabled = true;
            if (sSpinner) sSpinner.classList.remove('hidden');
            if (sIcon) sIcon.classList.add('hidden');
            if (sBtnText) sBtnText.innerText = 'Loading CSVs...';

            try {
                const res = await fetch('/api/sync-csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                if (!res.ok) {
                    throw new Error(`CSV sync returned HTTP ${res.status}`);
                }
                const csvResult = await res.json();

                const marketRes = await fetch('/api/market-data');
                DATA = await marketRes.json();

                renderBenchmarks();
                renderMacroBarometers();
                renderBriefingBaskets();
                renderRegime();
                if (DATA.last_market_session) {
                    const sessionBadge = document.getElementById('sessionDateBadge');
                    if (sessionBadge) sessionBadge.innerText = 'Prices: ' + DATA.last_market_session;
                }

                if (sBtnText) sBtnText.innerText = `CSVs Synced! (${csvResult.loaded_count || 0} Assets)`;
                setTimeout(() => {
                    if (sBtnText) sBtnText.innerText = 'Sync from CSV';
                    if (sBtn) sBtn.disabled = false;
                    if (sSpinner) sSpinner.classList.add('hidden');
                    if (sIcon) sIcon.classList.remove('hidden');
                }, 2200);

            } catch (err) {
                console.error("CSV sync error:", err);
                if (sBtnText) sBtnText.innerText = 'Sync Error';
                setTimeout(() => {
                    if (sBtnText) sBtnText.innerText = 'Sync from CSV';
                    if (sBtn) sBtn.disabled = false;
                    if (sSpinner) sSpinner.classList.add('hidden');
                    if (sIcon) sIcon.classList.remove('hidden');
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
            'tab7': 'Single Ticker Explorer',
            'tab_gamma': 'Dealer Gamma Desk',
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
            } else if (tabId === 'tab_gamma') {
                populateGammaDeskTickersDropdown();
                if (!LOADED_TABS.has('tab_gamma')) {
                    LOADED_TABS.add('tab_gamma');
                    loadGammaDesk();
                } else {
                    setTimeout(() => Plotly.Plots.resize('gammaDeskChart'), 50);
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
                renderRegime();
                renderFilamentHeroCards();
                initAgentChat();
                populateTickersDropdown();
                populateOptionsTickersDropdown();
                populateGammaDeskTickersDropdown();

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
                        <div class="text-xs leading-relaxed max-w-none overflow-x-auto">${parsedContent}</div>
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
            if (!DATA || !DATA.market_data) return;
            const md = DATA.market_data;
            const t = document.getElementById('briefingTitle') || document.getElementById('regimeTitle');
            if (t) t.innerText = md.narrative_title || "Secular Uptrend";
            const d = document.getElementById('briefingDetail') || document.getElementById('regimeDetail');
            if (d) d.innerText = md.narrative_detail || "";
            const b = document.getElementById('briefingRiskPill') || document.getElementById('regimeRiskBadge');
            if (b) b.innerText = md.risk_level || "Moderate";
            const sessionBadge = document.getElementById('sessionDateBadge');
            if (sessionBadge) {
                sessionBadge.innerText = 'Prices: ' + (DATA.last_market_session || 'Today');
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

        async function loadAiBriefingAsync(force = false) {
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
                let url = `/api/ai-briefing?force=${force}`;
                if (USER_API_KEY) {
                    url += `&api_key=${encodeURIComponent(USER_API_KEY)}`;
                }
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
                const sec = card.getAttribute('data-section');
                if (key === 'all' || !sec || sec === 'all' || sec === key) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            }
            const subPills = document.querySelectorAll('.synth-sub-pill');
            for (let i = 0; i < subPills.length; i++) {
                const sp = subPills[i];
                if (key === 'all' || sp.getAttribute('data-sub') === key) {
                    sp.classList.remove('opacity-25', 'scale-95');
                    sp.classList.add('opacity-100');
                } else {
                    sp.classList.add('opacity-25', 'scale-95');
                    sp.classList.remove('opacity-100');
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
                            <div class="synth-sub-pill p-3.5 rounded-xl bg-black/20 border border-sky-500/20 space-y-1.5 transition-all" data-sub="macro">
                                <div class="text-sky-400 font-extrabold uppercase text-[11px] flex items-center gap-1.5">Pillar 1: Macro Liquidity</div>
                                <div class="text-slate-300">${AI_DATA.regime_title || "Synthesizing yields, gold, oil & crypto liquidity..."}</div>
                            </div>
                            <div class="synth-sub-pill p-3.5 rounded-xl bg-black/20 border border-purple-500/20 space-y-1.5 transition-all" data-sub="sectors">
                                <div class="text-purple-400 font-extrabold uppercase text-[11px] flex items-center gap-1.5">Pillar 2: Sector Velocity</div>
                                <div class="text-slate-300">Tracking 5D capital rotation across Growth, Cyclicals, Defensives & Breadth...</div>
                            </div>
                            <div class="synth-sub-pill p-3.5 rounded-xl bg-black/20 border border-mint/20 space-y-1.5 transition-all" data-sub="setups">
                                <div class="text-mint font-extrabold uppercase text-[11px] flex items-center gap-1.5">Pillar 3: Actionable Setups</div>
                                <div class="text-slate-300">Filtering algorithmic high-conviction long bids & distribution traps...</div>
                            </div>
                            <div class="synth-sub-pill p-3.5 rounded-xl bg-black/20 border border-amber-500/20 space-y-1.5 transition-all" data-sub="risk">
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

                    const gexHtml = (t.call_wall && t.put_wall) ? `
                        <div class="flex flex-wrap items-center justify-between gap-1.5 p-1.5 px-2 mb-2 rounded-lg bg-indigo-950/20 border border-indigo-500/20 text-[10.5px]">
                            <div class="flex items-center gap-1.5 flex-wrap">
                                <span class="text-indigo-400 font-bold">⚡ GEX:</span>
                                <span class="text-slate-400">Call Wall:</span>
                                <b class="text-mint font-mono">${t.call_wall}</b>
                                <span class="text-slate-600">|</span>
                                <span class="text-slate-400">Put Wall:</span>
                                <b class="text-cherry font-mono">${t.put_wall}</b>
                                ${t.zero_gamma ? `<span class="text-slate-600">|</span><span class="text-slate-400">Zero Flip:</span><b class="text-cyan-400 font-mono">${t.zero_gamma}</b>` : ''}
                            </div>
                            <div class="flex items-center gap-1">
                                <span class="px-1.5 py-0.5 rounded text-[9.5px] font-bold ${t.gamma_regime === 'Long Gamma' ? 'bg-mint/15 text-mint' : 'bg-rose-500/15 text-rose-400'}">${t.gamma_regime || 'Dealer GEX'}</span>
                                ${t.net_gex_b ? `<span class="text-slate-400 font-mono text-[9.5px]">(${t.net_gex_b})</span>` : ''}
                            </div>
                        </div>
                    ` : '';

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

                                ${gexHtml}

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


        function setHorizon(h, btn) {
            CURRENT_HORIZON = h;
            document.querySelectorAll('.horizon-btn').forEach(b => {
                b.classList.remove('active', 'bg-[#1c1e27]', 'text-amber-400', 'font-bold');
                b.classList.add('text-slate-400');
            });
            const target = btn || (typeof event !== 'undefined' && event ? event.target : null);
            if (target) {
                target.classList.add('active', 'bg-[#1c1e27]', 'text-amber-400', 'font-bold');
                target.classList.remove('text-slate-400');
            }
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

        let CURRENT_OPTIONS_DIR = "Calls";
        let CURRENT_OPTIONS_SIDE = "Buy";
        let CURRENT_OPTIONS_EXPIRATION = "";
        let CURRENT_OPTIONS_SPOT = 0;
        let CURRENT_OPTIONS_TICKER = "QQQ";

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
            const btnCalls = document.getElementById('btnOptCalls');
            const btnPuts = document.getElementById('btnOptPuts');
            if (dir === 'Calls') {
                if (btnCalls) btnCalls.className = "px-3 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30";
                if (btnPuts) btnPuts.className = "px-3 py-1 rounded font-bold transition text-slate-400";
            } else {
                if (btnPuts) btnPuts.className = "px-3 py-1 rounded font-bold transition bg-cherry/20 text-cherry border border-cherry/30";
                if (btnCalls) btnCalls.className = "px-3 py-1 rounded font-bold transition text-slate-400";
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
                CURRENT_OPTIONS_TICKER = ticker;
                const expParam = CURRENT_OPTIONS_EXPIRATION ? `&expiration=${encodeURIComponent(CURRENT_OPTIONS_EXPIRATION)}` : '';
                const res = await fetch(`/api/options/${ticker}?direction=${encodeURIComponent(CURRENT_OPTIONS_DIR)}&side=${encodeURIComponent(CURRENT_OPTIONS_SIDE)}${expParam}`);
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
                        const isCall = (CURRENT_OPTIONS_DIR === 'Calls');
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
                                <div>
                                    Open Interest: <b class="text-slate-300">${(p.openInterest || 0).toLocaleString()}</b>
                                    ${((!p.openInterest || p.openInterest === 0) && (p.volume && p.volume > 0))
                                        ? `<span class="text-[10px] text-mint font-semibold">(${Number(p.volume).toLocaleString()} Vol • OCC Overnight Reset)</span>`
                                        : (p.volume && p.volume > 0 ? `<span class="text-[10px] text-slate-400 font-mono">(${Number(p.volume).toLocaleString()} Vol)</span>` : '')
                                    }
                                </div>
                                ${wingHtml}
                            </div>
                        </div>
                        `;
                    }).join('');
                }

                CURRENT_OPTIONS_SPOT = spot;
                CURRENT_OPTIONS_EXPIRATION = data.selected_expiration || (data.smile ? data.smile.expiration : '');

                // Populate Expiration Date Slicer
                const expSelect = document.getElementById('optionsExpSelect');
                if (expSelect && data.expirations && data.expirations.length > 0) {
                    expSelect.innerHTML = data.expirations.map(exp => `
                        <option value="${exp.expiration}" ${exp.expiration === CURRENT_OPTIONS_EXPIRATION ? 'selected' : ''}>
                            ${exp.expiration}  |  ${exp.dte}d DTE  |  ATM IV: ${exp.atm_iv}%  |  ±$${exp.exp_move_dollars}
                        </option>
                    `).join('');
                }

                // Render Smile Chart
                if (data.smile) {
                    renderSmileChart(data.smile, CURRENT_OPTIONS_DIR, isBuy, spot);
                }

                // Render Whole-Chain Tiered Discount Table
                renderWholeChainTable(data.ranked_chain, isBuy);

            } catch (err) {
                kpiRow.innerHTML = `<div class="col-span-full text-xs text-cherry bg-cherry/10 p-3 rounded-lg border border-cherry/20">Error loading options desk: ${err.message}</div>`;
                cardsContainer.innerHTML = '';
            }
        }

        function populateGammaDeskTickersDropdown() {
            const select = document.getElementById('gammaDeskTickerSelect');
            if (!select) return;
            const currentVal = select.value || 'SPY';
            const tickers = DATA.available_tickers || [];
            const priority = ['SPY', 'QQQ', 'IWM', 'NVDA', 'AAPL', 'TSLA', 'AMZN', 'MSFT', 'META', 'SMH', 'GLD', 'TLT'];
            const tickerLabels = {
                'SPY': 'SPY (S&P 500 ETF)',
                'QQQ': 'QQQ (Nasdaq 100 ETF)',
                'IWM': 'IWM (Russell 2000 ETF)',
                'NVDA': 'NVDA (Nvidia)',
                'AAPL': 'AAPL (Apple)',
                'TSLA': 'TSLA (Tesla)',
                'AMZN': 'AMZN (Amazon)',
                'MSFT': 'MSFT (Microsoft)',
                'META': 'META (Meta)',
                'SMH': 'SMH (Semiconductors)',
                'GLD': 'GLD (Gold ETF)',
                'TLT': 'TLT (20Y Treasury Bond)',
                'XLF': 'XLF (Financials ETF)',
                'GOOGL': 'GOOGL (Alphabet)',
                'AMD': 'AMD (Advanced Micro Devices)'
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

        async function loadGammaDesk(force = false) {
            const tickerSelect = document.getElementById('gammaDeskTickerSelect');
            const ticker = (tickerSelect && tickerSelect.value) ? tickerSelect.value : 'SPY';
            const kpiRow = document.getElementById('gammaDeskKpiRow');
            const directiveDiv = document.getElementById('gammaDeskDirective');
            const chartDiv = document.getElementById('gammaDeskChart');
            const countSpan = document.getElementById('gammaDeskStrikesCount');
            const tbody = document.getElementById('gammaDeskStrikesTbody');

            if (kpiRow) {
                kpiRow.innerHTML = `<div class="col-span-full py-4 text-center text-xs text-slate-400">Scanning real-time option chains & computing Black-Scholes Dealer Gamma Exposure (GEX) profile for <b>${ticker}</b>...</div>`;
            }
            if (directiveDiv) directiveDiv.innerHTML = '';
            if (tbody) tbody.innerHTML = `<tr><td colspan="8" class="text-center py-6 text-slate-500 animate-pulse">Calculating net dealer hedging delta & strike distribution...</td></tr>`;

            try {
                const url = `/api/gamma/${encodeURIComponent(ticker)}${force ? '?force=true' : ''}`;
                const res = await fetch(url);
                const g = await res.json();

                if (g.error) {
                    if (kpiRow) kpiRow.innerHTML = `<div class="col-span-full text-xs text-amber-400 bg-amber-500/10 p-3 rounded-lg border border-amber-500/20">${g.error}</div>`;
                    if (tbody) tbody.innerHTML = `<tr><td colspan="8" class="text-center py-4 text-cherry">No options data available for ${ticker}.</td></tr>`;
                    return;
                }

                const isLong = g.is_long_gamma;

                // 1. KPI Strip
                if (kpiRow) {
                    kpiRow.innerHTML = `
                        <div class="lux-card p-3 border-l-2 border-slate-400">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Spot Price</div>
                            <div class="text-lg font-bold text-white mt-0.5">$${g.spot_price.toFixed(2)}</div>
                            <div class="text-[10px] text-slate-400 mt-0.5">${g.ticker} Reference Underlying</div>
                        </div>
                        <div class="lux-card p-3 border-l-2 border-mint">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Call Wall (Ceiling Pin)</div>
                            <div class="text-lg font-bold text-mint mt-0.5">$${g.call_wall} <span class="text-xs font-normal text-slate-300">(${g.call_wall_dist_pct >= 0 ? '+' : ''}${g.call_wall_dist_pct}%)</span></div>
                            <div class="text-[10px] text-slate-400 mt-0.5">$${g.call_wall_gex_m}M GEX • Resistance Magnet</div>
                        </div>
                        <div class="lux-card p-3 border-l-2 border-cherry">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Put Wall (Support Floor)</div>
                            <div class="text-lg font-bold text-cherry mt-0.5">$${g.put_wall} <span class="text-xs font-normal text-slate-300">(${g.put_wall_dist_pct >= 0 ? '+' : ''}${g.put_wall_dist_pct}%)</span></div>
                            <div class="text-[10px] text-slate-400 mt-0.5">$${g.put_wall_gex_m}M GEX • Liquidity Cushion</div>
                        </div>
                        <div class="lux-card p-3 border-l-2 border-sky-400">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Zero Gamma Flip</div>
                            <div class="text-lg font-bold text-sky-400 mt-0.5">$${g.zero_gamma_level} <span class="text-xs font-normal text-slate-300">(${g.zero_gamma_dist_pct >= 0 ? '+' : ''}${g.zero_gamma_dist_pct}%)</span></div>
                            <div class="text-[10px] text-slate-400 mt-0.5">Vol Regime Transition Boundary</div>
                        </div>
                        <div class="lux-card p-3 border-l-2 ${isLong ? 'border-mint' : 'border-rose-500'}">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Total Net Dealer GEX</div>
                            <div class="text-lg font-bold ${isLong ? 'text-mint' : 'text-rose-400'} mt-0.5">${g.total_net_gex_billions >= 0 ? '+' : ''}${g.total_net_gex_billions.toFixed(2)}B</div>
                            <div class="text-[10px] text-slate-400 mt-0.5">${g.regime_short} • ${isLong ? 'Suppresses Volatility' : 'Amplifies Trending Vol'}</div>
                        </div>
                    `;
                }

                // 2. Tactical Directive Banner
                if (directiveDiv) {
                    if (isLong) {
                        directiveDiv.innerHTML = `
                            <div class="p-4 rounded-xl border border-mint/30 bg-mint/5 flex items-start gap-3">
                                <div class="text-xl">🛡️</div>
                                <div>
                                    <div class="text-xs font-bold text-mint uppercase tracking-wider">Tactical Desk Directive: Net Long Dealer Gamma Regime (+${g.total_net_gex_billions.toFixed(2)}B)</div>
                                    <div class="text-xs text-slate-300 mt-1 leading-relaxed">
                                        Market makers are net long options gamma. To maintain delta-neutral books, dealers must <b>buy weakness</b> and <b>sell strength</b>, acting as natural market shock-absorbers that suppress realized intraday volatility. Price action tends to pin between the <b>Put Wall ($${g.put_wall})</b> and <b>Call Wall ($${g.call_wall})</b>.
                                        <br/><span class="text-mint font-semibold mt-1 inline-block">Institutional Playbook:</span> Favor mean-reversion entries, range-bound credit harvesting, iron condors, and fading breakout attempts at extreme strike walls.
                                    </div>
                                </div>
                            </div>
                        `;
                    } else {
                        directiveDiv.innerHTML = `
                            <div class="p-4 rounded-xl border border-rose-500/30 bg-rose-500/5 flex items-start gap-3">
                                <div class="text-xl">⚠️</div>
                                <div>
                                    <div class="text-xs font-bold text-rose-400 uppercase tracking-wider">Tactical Desk Directive: Net Short Dealer Gamma Regime (${g.total_net_gex_billions.toFixed(2)}B)</div>
                                    <div class="text-xs text-slate-300 mt-1 leading-relaxed">
                                        Market makers are net short options gamma. To hedge portfolio deltas, dealers must <b>sell into falling markets</b> and <b>buy into rising markets</b>. This pro-cyclical hedging creates negative feedback cascades and sharp expansions in realized volatility.
                                        <br/><span class="text-rose-400 font-semibold mt-1 inline-block">Institutional Playbook:</span> Prepare for swift directional trend acceleration. Favor momentum debit spreads, long straddles/strangles, wider stop losses, and disciplined trend-following below the Zero-Gamma Flip line ($${g.zero_gamma_level}).
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                }

                // 3. Interactive Plotly Strike GEX Chart
                if (typeof Plotly !== 'undefined' && g.strikes && g.strikes.length > 0) {
                    const strikes = g.strikes.map(s => s.strike);
                    const callGex = g.strikes.map(s => s.call_gex);
                    const putGex = g.strikes.map(s => -s.put_gex);

                    const traceCalls = {
                        x: strikes,
                        y: callGex,
                        name: 'Call GEX ($M)',
                        type: 'bar',
                        marker: { color: '#00F5A0', opacity: 0.85 }
                    };

                    const tracePuts = {
                        x: strikes,
                        y: putGex,
                        name: 'Put GEX ($M)',
                        type: 'bar',
                        marker: { color: '#FF2E55', opacity: 0.85 }
                    };

                    const shapes = [
                        {
                            type: 'line',
                            x0: g.spot_price, x1: g.spot_price,
                            y0: 0, y1: 1, yref: 'paper',
                            line: { color: '#ffffff', width: 2, dash: 'dash' }
                        },
                        {
                            type: 'line',
                            x0: g.call_wall, x1: g.call_wall,
                            y0: 0, y1: 1, yref: 'paper',
                            line: { color: '#00F5A0', width: 2, dash: 'dot' }
                        },
                        {
                            type: 'line',
                            x0: g.put_wall, x1: g.put_wall,
                            y0: 0, y1: 1, yref: 'paper',
                            line: { color: '#FF2E55', width: 2, dash: 'dot' }
                        }
                    ];

                    const annotations = [
                        {
                            x: g.spot_price, y: 1, yref: 'paper',
                            text: 'Spot $' + g.spot_price.toFixed(2),
                            showarrow: false,
                            font: { color: '#ffffff', size: 10 },
                            xanchor: 'right', yanchor: 'bottom'
                        },
                        {
                            x: g.call_wall, y: 1, yref: 'paper',
                            text: 'Call Wall $' + g.call_wall,
                            showarrow: false,
                            font: { color: '#00F5A0', size: 10 },
                            xanchor: 'left', yanchor: 'bottom'
                        },
                        {
                            x: g.put_wall, y: 0, yref: 'paper',
                            text: 'Put Wall $' + g.put_wall,
                            showarrow: false,
                            font: { color: '#FF2E55', size: 10 },
                            xanchor: 'right', yanchor: 'top'
                        }
                    ];

                    if (g.zero_gamma_level && Math.abs(g.zero_gamma_level - g.spot_price) / g.spot_price < 0.35) {
                        shapes.push({
                            type: 'line',
                            x0: g.zero_gamma_level, x1: g.zero_gamma_level,
                            y0: 0, y1: 1, yref: 'paper',
                            line: { color: '#38bdf8', width: 1.5, dash: 'dashdot' }
                        });
                        annotations.push({
                            x: g.zero_gamma_level, y: 0.5, yref: 'paper',
                            text: 'Flip $' + (typeof g.zero_gamma_level === 'number' ? g.zero_gamma_level.toFixed(1) : g.zero_gamma_level),
                            showarrow: false,
                            font: { color: '#38bdf8', size: 9 },
                            xanchor: 'center', yanchor: 'middle'
                        });
                    }

                    const layout = {
                        title: {
                            text: `<b>${g.ticker} Dealer Gamma Exposure Profile by Strike</b> | Total Net GEX: $${g.total_net_gex_billions.toFixed(2)}B (${g.regime_short})`,
                            font: { color: '#f1f5f9', size: 13 }
                        },
                        barmode: 'relative',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        margin: { l: 45, r: 25, t: 35, b: 35 },
                        xaxis: {
                            title: { text: 'Strike Price ($)', font: { color: '#94a3b8', size: 11 } },
                            gridcolor: 'rgba(255, 255, 255, 0.04)',
                            zerolinecolor: 'rgba(255, 255, 255, 0.1)',
                            tickfont: { color: '#94a3b8', size: 10 }
                        },
                        yaxis: {
                            title: { text: 'Dollar GEX ($M / 1% move)', font: { color: '#94a3b8', size: 11 } },
                            gridcolor: 'rgba(255, 255, 255, 0.04)',
                            zerolinecolor: 'rgba(255, 255, 255, 0.15)',
                            tickfont: { color: '#94a3b8', size: 10 }
                        },
                        shapes: shapes,
                        annotations: annotations,
                        showlegend: true,
                        legend: { orientation: 'h', x: 1, xanchor: 'right', y: 1.1, font: { color: '#94a3b8', size: 10 } }
                    };

                    Plotly.newPlot('gammaDeskChart', [traceCalls, tracePuts], layout, { responsive: true, displayModeBar: false });
                }

                // 4. Tabular Per-Strike Breakdown
                if (countSpan) countSpan.innerText = `${(g.strikes || []).length} Strikes Mapped`;
                if (tbody && g.strikes && g.strikes.length > 0) {
                    const spot = g.spot_price;
                    tbody.innerHTML = g.strikes.map(s => {
                        const distPct = ((s.strike / spot) - 1) * 100;
                        const distStr = (distPct >= 0 ? '+' : '') + distPct.toFixed(1) + '%';
                        const netClass = s.net_gex >= 0 ? 'text-mint font-semibold' : 'text-cherry font-semibold';
                        const netSign = s.net_gex >= 0 ? '+' : '';

                        const targetZgStrike = g.closest_zero_gamma_strike !== undefined ? g.closest_zero_gamma_strike : g.zero_gamma_level;
                        let roleTag = '<span class="text-slate-500 text-[10px]">Normal Liquidity</span>';
                        if (Math.abs(s.strike - g.call_wall) < 0.01) {
                            roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-mint/20 text-mint border border-mint/30">CALL WALL (Pin)</span>';
                        } else if (Math.abs(s.strike - g.put_wall) < 0.01) {
                            roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">PUT WALL (Floor)</span>';
                        } else if (Math.abs(s.strike - targetZgStrike) < 0.01) {
                            roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-500/20 text-sky-400 border border-sky-500/30">ZERO FLIP</span>';
                        } else if (Math.abs(s.strike - spot) < (spot * 0.005)) {
                            roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-white/10 text-white">ATM SPOT</span>';
                        }

                        return `
                            <tr class="border-b border-white/[0.04] hover:bg-white/[0.02] transition">
                                <td class="py-2.5 px-3 font-bold text-white font-mono">$${s.strike.toFixed(1)}</td>
                                <td class="py-2.5 px-3 font-mono text-slate-400">${distStr}</td>
                                <td class="py-2.5 px-3 font-mono text-mint">+$${s.call_gex.toFixed(1)}M</td>
                                <td class="py-2.5 px-3 font-mono text-cherry">-$${s.put_gex.toFixed(1)}M</td>
                                <td class="py-2.5 px-3 font-mono ${netClass}">${netSign}$${s.net_gex.toFixed(1)}M</td>
                                <td class="py-2.5 px-3 font-mono text-slate-300">${(s.call_oi || 0).toLocaleString()}</td>
                                <td class="py-2.5 px-3 font-mono text-slate-300">${(s.put_oi || 0).toLocaleString()}</td>
                                <td class="py-2.5 px-3">${roleTag}</td>
                            </tr>
                        `;
                    }).join('');
                }

            } catch (err) {
                console.error("Gamma desk fetch error:", err);
                if (kpiRow) kpiRow.innerHTML = `<div class="col-span-full text-xs text-cherry bg-cherry/10 p-3 rounded-lg border border-cherry/20">Error loading gamma desk: ${err.message}</div>`;
            }
        }

        function renderWholeChainTable(ranked_chain, isBuy) {
            const tbody = document.getElementById("optionsChainDiscountTbody");
            const costTh = document.getElementById("thChainCost");
            const edgeTh = document.getElementById("thChainEdge");
            if (costTh) costTh.innerText = isBuy ? "Ask Cost" : "Bid Credit";
            if (edgeTh) edgeTh.innerText = isBuy ? "Ask Edge" : "Bid Edge";

            if (!tbody) return;
            if (!ranked_chain || ranked_chain.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-slate-500">No active chain contracts found.</td></tr>';
                return;
            }

            tbody.innerHTML = ranked_chain.map(r => {
                const tierBadge = r.tier_num === 1 
                    ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-mint/15 text-mint border border-mint/20">Tier 1 • Prime</span>'
                    : (r.tier_num === 2 
                        ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-500/15 text-sky-400 border border-sky-500/20">Tier 2 • High</span>'
                        : '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/15 text-purple-400 border border-purple-500/20">Tier 3 • Selective</span>');

                const edgeVal = isBuy ? r.ask_edge : r.bid_edge;
                const edgeSign = edgeVal >= 0 ? "+" : "";
                const edgeColor = edgeVal >= 0 ? "text-mint font-bold" : "text-slate-300";
                const costVal = isBuy ? r.ask_cost : r.bid_cost;
                const costStr = "$" + costVal.toFixed(2);
                const oiStr = r.oi > 0 ? r.oi.toLocaleString() : '<span class="text-slate-500">0</span> <span class="text-[10px] text-slate-400">(' + r.volume.toLocaleString() + ' vol)</span>';
                const deltaStr = Math.abs(r.delta).toFixed(2) + "Δ";
                const earnBadge = r.spans_earnings ? '<span class="ml-1 text-[9px] px-1.5 py-0.5 rounded bg-cherry/15 text-cherry font-semibold">Earn</span>' : "";

                return '<' + 'tr class="hover:bg-white/[0.02] transition border-b border-white/5 last:border-0">' +
                    '<td class="py-2.5 px-3 flex items-center gap-2"><span class="text-slate-500 font-mono text-[11px]">#' + r.rank + '</span>' + tierBadge + '</td>' +
                    '<td class="py-2.5 px-3 font-semibold text-white whitespace-nowrap">' + r.expiration + ' <span class="text-slate-400 text-[11px] font-normal">(' + r.dte + 'd)</span>' + earnBadge + '</td>' +
                    '<td class="py-2.5 px-3 font-bold text-white whitespace-nowrap">$' + Math.round(r.strike) + ' ' + r.type + '</td>' +
                    '<td class="py-2.5 px-3 font-mono font-medium text-white whitespace-nowrap">' + costStr + '</td>' +
                    '<td class="py-2.5 px-3 font-mono text-slate-300">' + r.iv.toFixed(1) + '%</td>' +
                    '<td class="py-2.5 px-3 font-mono text-slate-300 whitespace-nowrap">' + oiStr + '</td>' +
                    '<td class="py-2.5 px-3 font-mono ' + edgeColor + ' whitespace-nowrap">' + edgeSign + edgeVal.toFixed(2) + '%</td>' +
                    '<td class="py-2.5 px-3 font-mono text-slate-300">' + deltaStr + '</td>' +
                    '</tr>';
            }).join("");
        }

        function renderSmileChart(s, direction, isBuy, spot) {
            if (!s) return;
            const subEl = document.getElementById('optionsSmileSubtitle');
            const m = s.metrics || {};
            const totalContracts = m.total_contracts ? ` · ${m.total_contracts} contracts` : '';
            if (subEl) {
                subEl.innerText = `${s.expiration} · ${s.dte}DTE${totalContracts} | 1σ Expected Move: ±$${s.exp_move} | ${isBuy ? 'Buy Side (Ask Edge)' : 'Sell Side (Bid Edge)'}`;
            }

            // Populate the 6 Institutional Skew KPI Cards
            const kpiStrip = document.getElementById('optionsSkewKpiStrip');
            if (kpiStrip) {
                const atmIv = m.atm_iv != null ? m.atm_iv.toFixed(1) + '%' : '--';
                const atmStrike = m.atm_strike != null ? 'Strike ' + m.atm_strike : '--';
                
                const skew25d = m.skew_25d != null ? (m.skew_25d > 0 ? '+' : '') + m.skew_25d.toFixed(1) + ' pts' : '--';
                const skew25dColor = (m.skew_25d || 0) > 0 ? 'text-cherry' : ((m.skew_25d || 0) < 0 ? 'text-mint' : 'text-slate-200');
                const skew25dText = m.skew_25d_text || 'Risk Reversal';
                const skew25dSub = m.skew_25d_sub || '';

                const fly25d = m.fly_25d != null ? (m.fly_25d > 0 ? '+' : '') + m.fly_25d.toFixed(1) + ' pts' : '--';
                const fly25dText = m.fly_25d_text || 'Curvature';

                const skew10d = m.skew_10d != null ? (m.skew_10d > 0 ? '+' : '') + m.skew_10d.toFixed(1) + ' pts' : '--';
                const skew10dColor = (m.skew_10d || 0) > 1.2 ? 'text-cherry' : 'text-slate-200';
                const skew10dText = m.skew_10d_text || 'Tail risk';

                const skewSlope = m.skew_slope != null ? (m.skew_slope > 0 ? '+' : '') + m.skew_slope.toFixed(2) : '--';
                const skewSlopeText = m.skew_slope_text || 'IV pts per 1% of strike';

                const termSlope = m.term_slope != null ? (m.term_slope > 0 ? '+' : '') + m.term_slope.toFixed(1) + ' pts' : '--';
                const termSlopeText = m.term_slope_text || 'Term flat';

                kpiStrip.innerHTML = `
                    <div class="bg-[#12151c]/90 border border-white/5 rounded-lg p-2.5 flex flex-col justify-between">
                        <span class="text-[10px] font-bold tracking-wider text-slate-400 uppercase">ATM IV</span>
                        <div class="text-lg font-mono font-bold text-white mt-0.5">${atmIv}</div>
                        <span class="text-[10px] text-slate-400 font-medium">${atmStrike}</span>
                    </div>
                    <div class="bg-[#12151c]/90 border border-white/5 rounded-lg p-2.5 flex flex-col justify-between">
                        <span class="text-[10px] font-bold tracking-wider text-slate-400 uppercase">25Δ SKEW</span>
                        <div class="text-lg font-mono font-bold ${skew25dColor} mt-0.5">${skew25d}</div>
                        <div class="text-[10px] text-slate-300 font-semibold leading-tight">${skew25dText}</div>
                        ${skew25dSub ? `<span class="text-[9px] text-slate-500 font-mono mt-0.5">${skew25dSub}</span>` : ''}
                    </div>
                    <div class="bg-[#12151c]/90 border border-white/5 rounded-lg p-2.5 flex flex-col justify-between">
                        <span class="text-[10px] font-bold tracking-wider text-slate-400 uppercase">25Δ BUTTERFLY</span>
                        <div class="text-lg font-mono font-bold text-white mt-0.5">${fly25d}</div>
                        <span class="text-[10px] text-slate-300 font-semibold">${fly25dText}</span>
                    </div>
                    <div class="bg-[#12151c]/90 border border-white/5 rounded-lg p-2.5 flex flex-col justify-between">
                        <span class="text-[10px] font-bold tracking-wider text-slate-400 uppercase">10Δ SKEW</span>
                        <div class="text-lg font-mono font-bold ${skew10dColor} mt-0.5">${skew10d}</div>
                        <span class="text-[10px] text-slate-300 font-semibold">${skew10dText}</span>
                    </div>
                    <div class="bg-[#12151c]/90 border border-white/5 rounded-lg p-2.5 flex flex-col justify-between">
                        <span class="text-[10px] font-bold tracking-wider text-slate-400 uppercase">SKEW SLOPE</span>
                        <div class="text-lg font-mono font-bold text-white mt-0.5">${skewSlope}</div>
                        <span class="text-[10px] text-slate-400 font-medium">${skewSlopeText}</span>
                    </div>
                    <div class="bg-[#12151c]/90 border border-white/5 rounded-lg p-2.5 flex flex-col justify-between">
                        <span class="text-[10px] font-bold tracking-wider text-slate-400 uppercase">TERM SLOPE</span>
                        <div class="text-lg font-mono font-bold text-white mt-0.5">${termSlope}</div>
                        <span class="text-[10px] text-slate-400 font-medium">${termSlopeText}</span>
                    </div>
                `;
            }

            // Build Plotly Traces
            const traces = [];
            const atmIvVal = m.atm_iv || 25.0;

            // Dynamic Dual-Color Shading vs ATM Baseline (Red when above ATM, Green when under ATM)
            const fullCurveStrikes = [];
            const fullCurveIvs = [];
            if (s.otm_put_curve && s.otm_put_curve.strikes) {
                fullCurveStrikes.push(...s.otm_put_curve.strikes);
                fullCurveIvs.push(...s.otm_put_curve.iv);
            }
            if (s.otm_call_curve && s.otm_call_curve.strikes) {
                const startIdx = (fullCurveStrikes.length > 0 && Math.abs(s.otm_call_curve.strikes[0] - fullCurveStrikes[fullCurveStrikes.length - 1]) < 0.01) ? 1 : 0;
                fullCurveStrikes.push(...s.otm_call_curve.strikes.slice(startIdx));
                fullCurveIvs.push(...s.otm_call_curve.iv.slice(startIdx));
            }

            if (fullCurveStrikes.length >= 2 && atmIvVal) {
                const pts = [];
                for (let i = 0; i < fullCurveStrikes.length - 1; i++) {
                    const x1 = fullCurveStrikes[i], y1 = fullCurveIvs[i];
                    const x2 = fullCurveStrikes[i+1], y2 = fullCurveIvs[i+1];
                    pts.push({ x: x1, y: y1 });
                    if ((y1 - atmIvVal) * (y2 - atmIvVal) < 0) {
                        const t = (atmIvVal - y1) / (y2 - y1);
                        const xc = x1 + t * (x2 - x1);
                        pts.push({ x: xc, y: atmIvVal });
                    }
                }
                pts.push({ x: fullCurveStrikes[fullCurveStrikes.length - 1], y: fullCurveIvs[fullCurveIvs.length - 1] });

                const segsAbove = [];
                const segsBelow = [];
                let currSeg = [pts[0]];
                let isAbove = pts[0].y >= atmIvVal;

                for (let i = 1; i < pts.length; i++) {
                    currSeg.push(pts[i]);
                    if (Math.abs(pts[i].y - atmIvVal) < 1e-4) {
                        if (isAbove) segsAbove.push(currSeg);
                        else segsBelow.push(currSeg);
                        currSeg = [pts[i]];
                        isAbove = (i + 1 < pts.length) ? (pts[i+1].y >= atmIvVal) : !isAbove;
                    }
                }
                if (currSeg.length > 1) {
                    if (isAbove) segsAbove.push(currSeg);
                    else segsBelow.push(currSeg);
                }

                // Above ATM: Translucent Cherry Red
                segsAbove.forEach((seg, idx) => {
                    const xs = seg.map(p => p.x);
                    const ys = seg.map(p => p.y);
                    xs.push(seg[seg.length - 1].x, seg[0].x, seg[0].x);
                    ys.push(atmIvVal, atmIvVal, seg[0].y);
                    traces.push({
                        x: xs, y: ys,
                        fill: 'toself',
                        fillcolor: 'rgba(255, 46, 85, 0.16)',
                        line: { width: 0 },
                        mode: 'none',
                        type: 'scatter',
                        name: idx === 0 ? 'bid over ATM' : undefined,
                        showlegend: idx === 0,
                        legendgroup: 'shading_above',
                        hoverinfo: 'none'
                    });
                });

                // Below ATM: Translucent Mint Green
                segsBelow.forEach((seg, idx) => {
                    const xs = seg.map(p => p.x);
                    const ys = seg.map(p => p.y);
                    xs.push(seg[seg.length - 1].x, seg[0].x, seg[0].x);
                    ys.push(atmIvVal, atmIvVal, seg[0].y);
                    traces.push({
                        x: xs, y: ys,
                        fill: 'toself',
                        fillcolor: 'rgba(0, 245, 160, 0.15)',
                        line: { width: 0 },
                        mode: 'none',
                        type: 'scatter',
                        name: idx === 0 ? 'under ATM' : undefined,
                        showlegend: idx === 0,
                        legendgroup: 'shading_below',
                        hoverinfo: 'none'
                    });
                });
            }

            // 1. OTM Puts Fitted Curve (Cherry Red)
            if (s.otm_put_curve && s.otm_put_curve.strikes && s.otm_put_curve.strikes.length > 0) {
                traces.push({
                    x: s.otm_put_curve.strikes,
                    y: s.otm_put_curve.iv,
                    name: 'OTM puts',
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#FF2E55', width: 2.5 },
                    hovertemplate: '<b>OTM Put Skew</b><br>Strike: $%{x}<br>IV: %{y:.1f}%<extra></extra>'
                });
            }

            // 2. OTM Calls Fitted Curve (Mint Green)
            if (s.otm_call_curve && s.otm_call_curve.strikes && s.otm_call_curve.strikes.length > 0) {
                traces.push({
                    x: s.otm_call_curve.strikes,
                    y: s.otm_call_curve.iv,
                    name: 'OTM calls',
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#00F5A0', width: 2.5 },
                    hovertemplate: '<b>OTM Call Skew</b><br>Strike: $%{x}<br>IV: %{y:.1f}%<extra></extra>'
                });
            }

            // 3. Raw Market Scatter: OTM Puts (Cherry Red, low opacity)
            if (s.otm_puts_market && s.otm_puts_market.strikes && s.otm_puts_market.strikes.length > 0) {
                traces.push({
                    x: s.otm_puts_market.strikes,
                    y: s.otm_puts_market.iv,
                    name: 'Market Puts IV',
                    type: 'scatter',
                    mode: 'markers',
                    marker: { color: '#FF2E55', size: 5, opacity: 0.35 },
                    hovertemplate: 'Put Strike: $%{x}<br>Market IV: %{y:.1f}%<extra></extra>',
                    showlegend: false
                });
            }

            // 4. Raw Market Scatter: OTM Calls (Mint Green, low opacity)
            if (s.otm_calls_market && s.otm_calls_market.strikes && s.otm_calls_market.strikes.length > 0) {
                traces.push({
                    x: s.otm_calls_market.strikes,
                    y: s.otm_calls_market.iv,
                    name: 'Market Calls IV',
                    type: 'scatter',
                    mode: 'markers',
                    marker: { color: '#00F5A0', size: 5, opacity: 0.35 },
                    hovertemplate: 'Call Strike: $%{x}<br>Market IV: %{y:.1f}%<extra></extra>',
                    showlegend: false
                });
            }

            // 5. Key Structural Markers: ATM Dot, 25Δ Put, 25Δ Call
            const struct = s.structural_markers || {};
            const markerX = [];
            const markerY = [];
            const markerColors = [];
            const markerLines = [];
            const markerSymbols = [];
            const markerTexts = [];
            const markerHovers = [];

            if (struct.atm && struct.atm.strike) {
                markerX.push(struct.atm.strike);
                markerY.push(struct.atm.iv);
                markerColors.push('#F59E0B');
                markerLines.push('#F59E0B');
                markerSymbols.push('circle');
                markerTexts.push('');
                markerHovers.push(`<b>ATM Baseline</b><br>Strike: $${struct.atm.strike}<br>IV: ${struct.atm.iv.toFixed(1)}%<extra></extra>`);
            }
            if (struct.put_25d && struct.put_25d.strike) {
                markerX.push(struct.put_25d.strike);
                markerY.push(struct.put_25d.iv);
                markerColors.push('rgba(255, 46, 85, 0.15)');
                markerLines.push('#FF2E55');
                markerSymbols.push('circle');
                markerTexts.push('25Δ');
                markerHovers.push(`<b>25Δ Put Wing</b><br>Strike: $${struct.put_25d.strike}<br>IV: ${struct.put_25d.iv.toFixed(1)}%<extra></extra>`);
            }
            if (struct.call_25d && struct.call_25d.strike) {
                markerX.push(struct.call_25d.strike);
                markerY.push(struct.call_25d.iv);
                markerColors.push('rgba(0, 245, 160, 0.15)');
                markerLines.push('#00F5A0');
                markerSymbols.push('circle');
                markerTexts.push('25Δ');
                markerHovers.push(`<b>25Δ Call Wing</b><br>Strike: $${struct.call_25d.strike}<br>IV: ${struct.call_25d.iv.toFixed(1)}%<extra></extra>`);
            }

            if (markerX.length > 0) {
                traces.push({
                    x: markerX,
                    y: markerY,
                    name: '25Δ wings & ATM',
                    type: 'scatter',
                    mode: 'markers+text',
                    marker: {
                        color: markerColors,
                        size: 13,
                        symbol: markerSymbols,
                        line: { color: markerLines, width: 2 }
                    },
                    text: markerTexts,
                    textposition: 'top center',
                    textfont: { color: markerLines, size: 10, family: 'Plus Jakarta Sans, sans-serif' },
                    hovertemplate: markerHovers
                });
            }

            // 6. Execution Overlay: Effective Ask IV / Bid IV
            // Filter to active trading moneyness and clamp any illiquid spread spikes
            const spreadStrikes = [];
            const spreadIvs = [];
            if (s.market_strikes && (s.market_ask_iv || s.market_bid_iv)) {
                const rawIvs = isBuy ? s.market_ask_iv : s.market_bid_iv;
                const minK = isBuy ? (direction === 'Calls' ? spot * 0.95 : spot * 0.85) : spot * 0.85;
                const maxK = isBuy ? (direction === 'Calls' ? spot * 1.15 : spot * 1.05) : spot * 1.15;
                const upperIvCap = atmIvVal + 5.5;
                const lowerIvFloor = Math.max(8.0, atmIvVal - 6.5);
                s.market_strikes.forEach((k, idx) => {
                    const iv = rawIvs[idx];
                    if (k >= minK && k <= maxK && iv > 0) {
                        // Clamp outlier spikes caused by wide market-maker spreads on illiquid strikes
                        const cleanIv = Math.min(Math.max(iv, lowerIvFloor), upperIvCap);
                        spreadStrikes.push(k);
                        spreadIvs.push(cleanIv);
                    }
                });
            }

            if (spreadStrikes.length > 0) {
                const traceSpreadLine = {
                    x: spreadStrikes,
                    y: spreadIvs,
                    name: isBuy ? 'Effective Ask IV (Cost to Buy)' : 'Effective Bid IV (Selling Price)',
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: isBuy ? 'rgba(245, 158, 11, 0.75)' : 'rgba(0, 245, 160, 0.75)', width: 1.5, dash: 'dot' },
                    hovertemplate: 'Strike: $%{x}<br>' + (isBuy ? 'Ask IV: %{y:.1f}%' : 'Bid IV: %{y:.1f}%') + '<extra></extra>'
                };
                traces.push(traceSpreadLine);
            }

            // 7. Execution Overlay: Recommended Value Strikes (Clean number only, e.g. "295")
            if (s.top_strikes && s.top_strikes.length > 0) {
                traces.push({
                    x: s.top_strikes,
                    y: s.top_strikes_iv,
                    name: isBuy ? 'Recommended Buy Strikes' : 'Recommended Sell Strikes',
                    type: 'scatter',
                    mode: 'markers+text',
                    marker: {
                        color: isBuy ? '#00F5A0' : '#FF2E55',
                        size: 14,
                        symbol: 'circle',
                        line: { color: '#ffffff', width: 2.5 }
                    },
                    text: s.top_strikes.map(st => `${st}`),
                    textposition: 'bottom center',
                    textfont: { color: isBuy ? '#00F5A0' : '#FF2E55', size: 11, family: 'Plus Jakarta Sans, sans-serif' },
                    hovertemplate: isBuy ? '<b>VALUE BUY STRIKE</b><br>Strike: $%{x}<br>Market IV: %{y:.2f}%<extra></extra>' : '<b>PREMIUM SELL STRIKE</b><br>Strike: $%{x}<br>Market IV: %{y:.2f}%<extra></extra>'
                });
            }

            // Calculate appropriate y-axis range STRICTLY anchored on the Skew Curve & ATM baseline
            // Guarantees the curve fills the vertical space without deep OTM spread blowout
            const curveIvs = [];
            if (s.otm_put_curve && s.otm_put_curve.iv) curveIvs.push(...s.otm_put_curve.iv);
            if (s.otm_call_curve && s.otm_call_curve.iv) curveIvs.push(...s.otm_call_curve.iv);
            if (atmIvVal) curveIvs.push(atmIvVal);
            if (s.top_strikes_iv) curveIvs.push(...s.top_strikes_iv);
            if (spreadIvs.length > 0) curveIvs.push(...spreadIvs);

            const yaxisConfig = { 
                title: "Implied Volatility (%)", 
                gridcolor: "#1B212D",
                ticksuffix: "%"
            };

            const validCurveIvs = curveIvs.filter(v => v > 5 && v < 100);
            if (validCurveIvs.length > 0) {
                let minIv = Math.min(...validCurveIvs);
                let maxIv = Math.max(...validCurveIvs);
                let rangeSpan = maxIv - minIv;
                let pad = Math.max(rangeSpan * 0.18, 1.8);
                let yMin = Math.max(0, Math.floor(minIv - pad));
                let yMax = Math.ceil(maxIv + pad);
                
                // Ensure a comfortable vertical height (at least 6-8 pts)
                if (yMax - yMin < 6) {
                    const mid = Math.round((yMax + yMin) / 2);
                    yMin = Math.max(0, mid - 3);
                    yMax = mid + 3;
                }
                const dtick = (yMax - yMin) <= 8 ? 2 : ((yMax - yMin) <= 16 ? 2 : 4);
                yaxisConfig.range = [yMin, yMax];
                yaxisConfig.dtick = dtick;
            }

            // Shapes & Annotations
            const shapes = [];
            const annotations = [];

            // Vertical line at Spot
            if (spot) {
                shapes.push({
                    type: 'line', x0: spot, x1: spot, y0: 0, y1: 1, yref: 'paper',
                    line: { color: '#94a3b8', width: 1.5, dash: 'dash' }
                });
                annotations.push({
                    x: spot, y: 1.05, yref: 'paper',
                    text: `Spot ${spot.toFixed(2)}`,
                    showarrow: false,
                    font: { color: '#94a3b8', size: 10, family: 'Plus Jakarta Sans, sans-serif' }
                });
            }

            // Horizontal line at ATM IV
            if (atmIvVal) {
                shapes.push({
                    type: 'line', x0: 0, x1: 1, xref: 'paper', y0: atmIvVal, y1: atmIvVal,
                    line: { color: '#F59E0B', width: 1.2, dash: 'dash' }
                });
                annotations.push({
                    x: 1.0, xref: 'paper', y: atmIvVal,
                    text: `ATM ${atmIvVal.toFixed(1)}%`,
                    showarrow: false,
                    xanchor: 'right',
                    yanchor: 'bottom',
                    font: { color: '#F59E0B', size: 10, family: 'Plus Jakarta Sans, sans-serif' }
                });
            }

            // 1-sigma expected move band
            if (spot && s.exp_move) {
                shapes.push({
                    type: 'rect', x0: spot - s.exp_move, x1: spot + s.exp_move, y0: 0, y1: 1, yref: 'paper',
                    fillcolor: 'rgba(56, 189, 248, 0.04)', line: { width: 0 }
                });
            }

            const layout = {
                plot_bgcolor: "#0D0F14",
                paper_bgcolor: "#0D0F14",
                font: { family: "Plus Jakarta Sans, sans-serif", color: "#94a3b8", size: 11 },
                margin: { l: 45, r: 45, t: 30, b: 45 },
                xaxis: { title: "Strike Price ($)", gridcolor: "#1B212D" },
                yaxis: yaxisConfig,
                legend: { orientation: "h", y: -0.22, x: 0.5, xanchor: "center" },
                shapes: shapes,
                annotations: annotations
            };

            Plotly.newPlot('optionsSmileChart', traces, layout, {responsive: true});
        }

        async function onExpirationChange(expDate) {
            if (!expDate) return;
            CURRENT_OPTIONS_EXPIRATION = expDate;
            const ticker = CURRENT_OPTIONS_TICKER || document.getElementById('optionsTickerSelect').value || 'QQQ';
            const subEl = document.getElementById('optionsSmileSubtitle');
            if (subEl) {
                subEl.innerHTML = `<span class="inline-block animate-spin text-mint font-mono">↻</span> Fetching live smile & fitted asymmetric skew for ${expDate}...`;
            }

            try {
                const res = await fetch(`/api/options/${ticker}/smile?expiration=${encodeURIComponent(expDate)}&direction=${encodeURIComponent(CURRENT_OPTIONS_DIR)}&side=${encodeURIComponent(CURRENT_OPTIONS_SIDE)}&spot=${CURRENT_OPTIONS_SPOT || 0}`);
                const data = await res.json();
                if (data.error) {
                    if (subEl) subEl.innerText = `Error: ${data.error}`;
                    return;
                }
                if (data.smile) {
                    const isBuy = CURRENT_OPTIONS_SIDE === 'Buy';
                    renderSmileChart(data.smile, CURRENT_OPTIONS_DIR, isBuy, data.spot_price || CURRENT_OPTIONS_SPOT);
                }
            } catch (err) {
                if (subEl) subEl.innerText = `Error loading skew: ${err.message}`;
            }
        }

        window.onload = init;
    </script>
</body>
</html>
"""
