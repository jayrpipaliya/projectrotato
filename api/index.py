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
    MAG_7, INDEXES, SECTORS, MACRO_ASSETS, generate_synthetic_ratios,
    PROCESSED_DIR
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
    briefing_file = os.path.join(PROCESSED_DIR, "ai_briefing.json")
    if not force and os.path.exists(briefing_file):
        try:
            with open(briefing_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if saved and (saved.get("llm_summary") or saved.get("trade_cards")):
                    return clean_json(saved)
        except Exception:
            pass

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


# ================= TRADE LEDGER API ENDPOINTS =================
def _build_ledger_payload():
    from ledger_engine import get_ledger_stats, compute_mark_to_market
    stats = get_ledger_stats()
    enriched_active = []
    for t in stats.get("active_trades", []):
        mtm = compute_mark_to_market(t)
        enriched_active.append({**t, "mtm": mtm})

    return clean_json({
        "status": "success",
        "stats": {
            "total_trades": stats.get("total_trades", 0),
            "active_trades_count": stats.get("active_trades_count", 0),
            "closed_trades_count": stats.get("closed_trades_count", 0),
            "win_rate_closed": stats.get("win_rate_closed", 0.0),
            "realized_pnl": stats.get("realized_pnl", 0.0),
            "open_unrealized_pnl": stats.get("open_unrealized_pnl", 0.0),
        },
        "active_trades": enriched_active,
        "closed_trades": stats.get("closed_trades", [])
    })


@app.get("/api/ledger")
def api_get_ledger():
    try:
        return _build_ledger_payload()
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/ledger/restore")
async def api_ledger_restore(payload: dict):
    try:
        from ledger_engine import restore_trades
        restore_trades(payload.get("trades", []))
        return _build_ledger_payload()
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/ledger/add")
async def api_ledger_add(payload: dict):
    try:
        from ledger_engine import add_trade
        trd = add_trade(
            ticker=payload.get("ticker", "SPY"),
            direction=payload.get("direction", "LONG"),
            asset_type=payload.get("asset_type", "STOCK"),
            entry_price=float(payload.get("entry_price") or 0.0),
            size=float(payload.get("size") or 1.0),
            stop_loss=float(payload.get("stop_loss") or 0.0),
            target_1=float(payload.get("target_1") or 0.0),
            target_2=float(payload.get("target_2") or 0.0),
            contract_type=payload.get("contract_type"),
            strike=float(payload.get("strike")) if payload.get("strike") is not None else None,
            expiration=payload.get("expiration"),
            underlying_entry_spot=float(payload.get("underlying_entry_spot")) if payload.get("underlying_entry_spot") else None,
            spread_type=payload.get("spread_type"),
            short_strike=float(payload.get("short_strike")) if payload.get("short_strike") is not None else None,
            long_strike=float(payload.get("long_strike")) if payload.get("long_strike") is not None else None,
            short_expiry=payload.get("short_expiry"),
            long_expiry=payload.get("long_expiry"),
            net_premium=float(payload.get("net_premium")) if payload.get("net_premium") is not None else None,
            spread_width=float(payload.get("spread_width")) if payload.get("spread_width") is not None else None,
            max_risk=float(payload.get("max_risk")) if payload.get("max_risk") is not None else None,
            max_profit=float(payload.get("max_profit")) if payload.get("max_profit") is not None else None,
            notes=payload.get("notes", ""),
            ai_snapshot=payload.get("ai_snapshot", {})
        )
        return {"status": "success", "trade": clean_json(trd)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/ledger/close")
async def api_ledger_close(payload: dict):
    try:
        from ledger_engine import close_trade
        trade_id = payload.get("id")
        exit_price = float(payload.get("exit_price") or 0.0)
        outcome = payload.get("outcome", "CLOSED_MANUALLY")
        notes = payload.get("notes")
        trd = close_trade(trade_id, exit_price=exit_price, outcome=outcome, notes=notes)
        return {"status": "success", "trade": clean_json(trd) if trd else None}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/ledger/update")
async def api_ledger_update(payload: dict):
    try:
        from ledger_engine import update_trade
        trade_id = payload.get("id")
        updates = payload.get("updates", {})
        trd = update_trade(trade_id, updates)
        return {"status": "success", "trade": clean_json(trd) if trd else None}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.delete("/api/ledger/{trade_id}")
def api_ledger_delete(trade_id: str):
    try:
        from ledger_engine import delete_trade
        ok = delete_trade(trade_id)
        return {"status": "success" if ok else "not_found"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/ledger/audit")
async def api_ledger_audit(payload: dict = None):
    try:
        from ledger_engine import generate_ai_portfolio_audit
        api_key = (payload or {}).get("api_key") or os.getenv("GEMINI_API_KEY") or DEFAULT_GEMINI_KEY
        market_data = get_cached_analysis()
        report = generate_ai_portfolio_audit(market_data=market_data, gemini_api_key=api_key)
        return {"report": report}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

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

@app.get("/api/gamma/{ticker}/shift")
def api_gamma_shift(ticker: str):
    try:
        from gamma_tracker import compute_gamma_shift
        return clean_json(compute_gamma_shift(ticker))
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/gamma/{ticker}/heatmap")
def api_gamma_heatmap(ticker: str, mode: str = Query("net_gex"), days: int = Query(10)):
    try:
        from gamma_tracker import generate_gamma_heatmap_matrix
        return clean_json(generate_gamma_heatmap_matrix(ticker, mode=mode, max_days=days))
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/gamma/{ticker}/oi-history")
def api_gamma_oi_history(ticker: str, days: int = Query(5)):
    try:
        from gamma_tracker import get_historical_oi_migration
        return clean_json(get_historical_oi_migration(ticker, max_days=days))
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
    #chatHistory blockquote.danger, #chatHistory blockquote.warning {
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

        <!-- Quantitative Engine Status Pill -->
        <div class="flex items-center justify-between px-3 py-2 rounded-lg bg-stealth-900 border border-stealth-border text-xs text-slate-300 shadow-sm">
          <div class="flex items-center space-x-2">
            <svg class="w-3.5 h-3.5 text-mint" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span class="font-medium text-slate-300">Quantitative Engine</span>
          </div>
          <span class="w-1.5 h-1.5 rounded-full bg-mint shadow-[0_0_6px_#00F5A0]"></span>
        </div>
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
        <button onclick="switchTab('tab2')" id="btn-tab2" class="tab-btn active px-4 py-2 rounded-lg bg-stealth-850 text-white font-semibold border border-mint/40 shadow-[0_0_12px_rgba(0,245,160,0.15)] shrink-0">
          Automated Trade Ideas
        </button>
        <button onclick="switchTab('tab_gamma')" id="btn-tab_gamma" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Gamma Exposure Map
        </button>
        <button onclick="switchTab('tab_options')" id="btn-tab_options" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0">
          Volatility Pricing
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
        <button onclick="switchTab('tab_ledger')" id="btn-tab_ledger" class="tab-btn px-3.5 py-2 rounded-lg text-slate-400 hover:text-white hover:bg-stealth-900 transition-colors shrink-0 flex items-center gap-1.5">
          <span>Trade Ledger & Audit</span>
          <span id="ledgerTabBadge" class="hidden text-[10px] bg-mint/20 text-mint font-bold px-1.5 py-0.5 rounded-full border border-mint/30">0</span>
        </button>
      </nav>

        <!-- Tab 2: Automated Trade Ideas -->
        <div id="tab2" class="tab-pane">
            <div class="bg-white/[0.03] border border-white/5 rounded-xl p-3 mb-4 text-xs text-slate-400">
                <b class="text-mint font-semibold">Strict Confluence Rules:</b> Green conviction scores (+15 to +100) are reserved strictly for <b>Long Setups</b>. Red conviction scores (-15 to -100) are strictly flagged for <b>Distribution Fades or De-risking</b> to protect portfolio equity.
            </div>
            <div id="tradeCardsContainer" class="grid grid-cols-1 xl:grid-cols-2 gap-3">
                <div class="text-xs text-slate-400">Loading high-probability trade setups...</div>
            </div>
        </div>

        <!-- Tab Ledger: Institutional Trade Ledger & Risk Radar -->
        <div id="tab_ledger" class="tab-pane hidden">
            <!-- Header & Desk Controls -->
            <div class="lux-card p-4 sm:p-5 mb-4">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-3 mb-4 border-b border-white/10 gap-3">
                    <div>
                        <h2 class="text-lg font-bold text-white flex items-center gap-2">
                            <span>Institutional Trade Ledger</span>
                            <span class="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-mint/15 text-mint border border-mint/20">Live Forward Radar</span>
                        </h2>
                        <p class="text-xs text-slate-400">Live Mark-to-Market, Frozen Quantitative Baseline Snapshots, and Position Risk Monitoring</p>
                    </div>
                    <div class="flex items-center gap-2 w-full sm:w-auto">
                        <button onclick="loadLedger()" class="px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10 text-xs font-semibold transition flex items-center justify-center gap-1">
                            <span>Refresh</span>
                        </button>
                    </div>
                </div>


                <!-- Executive 5-KPI Stats Ribbon -->
                <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-4">
                    <div class="p-3 rounded-lg bg-white/[0.02] border border-white/5">
                        <div class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Total Logged</div>
                        <div id="kpiLedgerTotal" class="text-xl font-black text-white tabular-num mt-1">0</div>
                        <div class="text-[9.5px] text-slate-500 mt-0.5">Cumulative Setups</div>
                    </div>
                    <div class="p-3 rounded-lg bg-white/[0.02] border border-white/5 border-l-2 border-l-sky-400">
                        <div class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Active Open</div>
                        <div id="kpiLedgerActive" class="text-xl font-black text-sky-400 tabular-num mt-1">0</div>
                        <div class="text-[9.5px] text-slate-500 mt-0.5">Under Monitoring</div>
                    </div>
                    <div class="p-3 rounded-lg bg-white/[0.02] border border-white/5 border-l-2 border-l-mint" id="kpiLedgerUnrealCard">
                        <div class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Unrealized P&L</div>
                        <div id="kpiLedgerUnreal" class="text-xl font-black text-mint tabular-num mt-1">$0.00</div>
                        <div class="text-[9.5px] text-slate-500 mt-0.5">Open Risk Mark</div>
                    </div>
                    <div class="p-3 rounded-lg bg-white/[0.02] border border-white/5 border-l-2 border-l-emerald-400" id="kpiLedgerRealCard">
                        <div class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Realized Net P&L</div>
                        <div id="kpiLedgerReal" class="text-xl font-black text-white tabular-num mt-1">$0.00</div>
                        <div class="text-[9.5px] text-slate-500 mt-0.5">Closed Performance</div>
                    </div>
                    <div class="p-3 rounded-lg bg-white/[0.02] border border-white/5 col-span-2 sm:col-span-1">
                        <div class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Closed Win Rate</div>
                        <div id="kpiLedgerWinRate" class="text-xl font-black text-white tabular-num mt-1">0.0%</div>
                        <div id="kpiLedgerClosedCount" class="text-[9.5px] text-slate-500 mt-0.5">N=0 Closed Trades</div>
                    </div>
                </div>

                <!-- Sub-Navigation Navigation Pills -->
                <div class="flex border-b border-white/10 mb-4 text-xs font-semibold gap-1">
                    <button onclick="switchLedgerSub('active')" id="btnSubLedgerActive" class="px-3.5 py-1.5 border-b-2 border-mint text-white font-bold transition flex items-center gap-1.5">
                        <span>Active Positions Radar</span>
                        <span id="subBadgeActive" class="px-1.5 py-0.2 rounded-full text-[10px] bg-mint/20 text-mint">0</span>
                    </button>
                    <button onclick="switchLedgerSub('manual')" id="btnSubLedgerManual" class="px-3.5 py-1.5 border-b-2 border-transparent text-slate-400 hover:text-white transition">
                        <span>Manual Trade Logger</span>
                    </button>
                    <button onclick="switchLedgerSub('closed')" id="btnSubLedgerClosed" class="px-3.5 py-1.5 border-b-2 border-transparent text-slate-400 hover:text-white transition flex items-center gap-1.5">
                        <span>Closed Trades History</span>
                        <span id="subBadgeClosed" class="px-1.5 py-0.2 rounded-full text-[10px] bg-white/10 text-slate-400">0</span>
                    </button>
                </div>

                <!-- Sub-Pane 1: Active Positions Grid -->
                <div id="ledgerSubActive" class="space-y-3">
                    <div id="ledgerActiveList" class="grid grid-cols-1 xl:grid-cols-2 gap-3">
                        <div class="text-xs text-slate-400 p-4">Loading active positions...</div>
                    </div>
                </div>

                <!-- Sub-Pane 2: Manual Trade Logger -->
                <div id="ledgerSubManual" class="hidden">
                    <div class="p-4 rounded-xl bg-[#0b0d13] border border-white/5 max-w-3xl">
                        <h3 class="text-sm font-bold text-white mb-1">Log Custom Trade Setup</h3>
                        <p class="text-xs text-slate-400 mb-4">Enter trade parameters and auto-pull today's live confluence and backtest metrics.</p>

                        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-3">
                            <div>
                                <label class="text-[10px] uppercase font-bold text-slate-400">Ticker Symbol</label>
                                <input id="manTicker" type="text" value="NVDA" class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white uppercase font-bold focus:outline-none focus:border-mint" oninput="this.value = this.value.toUpperCase()">
                            </div>
                            <div>
                                <label class="text-[10px] uppercase font-bold text-slate-400">Direction</label>
                                <select id="manDirection" class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-mint">
                                    <option value="LONG">🟢 LONG</option>
                                    <option value="SHORT / FADE">🔴 SHORT / FADE</option>
                                    <option value="DIP BUY">🔵 DIP BUY</option>
                                </select>
                            </div>
                            <div>
                                <label class="text-[10px] uppercase font-bold text-slate-400">Instrument Type</label>
                                <select id="manAssetType" onchange="toggleManualInstrumentFields()" class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-mint">
                                    <option value="STOCK">Stock / Equity</option>
                                    <option value="OPTION">Options Contract</option>
                                </select>
                            </div>
                        </div>

                        <div class="mb-4">
                            <button onclick="autoFetchManualConfluence()" id="btnAutoFetchMan" class="px-3 py-1.5 rounded-lg bg-mint/15 hover:bg-mint/25 text-mint border border-mint/30 text-xs font-bold transition flex items-center gap-1.5">
                                <span>Auto-Fetch Current Confluence & 20-Year Odds for Ticker</span>
                            </button>
                            <div id="manAutoFetchStatus" class="text-[11px] text-slate-400 mt-1 hidden"></div>
                        </div>

                        <!-- Options specifics (toggled) -->
                        <div id="manOptionsRow" class="hidden grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3 p-3 rounded-lg bg-white/[0.02] border border-white/5">
                            <div>
                                <label class="text-[10px] uppercase font-bold text-slate-400">Contract Type</label>
                                <select id="manContractType" class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white">
                                    <option value="CALL">CALL</option>
                                    <option value="PUT">PUT</option>
                                </select>
                            </div>
                            <div>
                                <label class="text-[10px] uppercase font-bold text-slate-400">Strike Price ($)</label>
                                <input id="manStrike" type="number" step="0.5" placeholder="145.0" class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white">
                            </div>
                            <div>
                                <label class="text-[10px] uppercase font-bold text-slate-400">Expiration Date</label>
                                <input id="manExpiration" type="text" placeholder="YYYY-MM-DD" class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white">
                            </div>
                            <div>
                                <label class="text-[10px] uppercase font-bold text-slate-400">Contracts Qty</label>
                                <input id="manOptSize" type="number" min="1" value="2" class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white">
                            </div>
                        </div>

                        <!-- Stock & Options Common Execution Fields -->
                        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
                            <div>
                                <label id="manEntryLabel" class="text-[10px] uppercase font-bold text-slate-400">Entry Fill Price ($)</label>
                                <input id="manEntryPrice" type="number" step="0.01" placeholder="140.50" class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white">
                            </div>
                            <div id="manStockSizeDiv">
                                <label class="text-[10px] uppercase font-bold text-slate-400">Shares Quantity</label>
                                <input id="manStockSize" type="number" min="1" value="10" class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white">
                            </div>
                            <div>
                                <label class="text-[10px] uppercase font-bold text-cherry">Invalidation Stop ($)</label>
                                <input id="manStopLoss" type="number" step="0.01" placeholder="136.00" class="w-full mt-1 bg-[#161722] border border-cherry/30 rounded-lg px-3 py-1.5 text-xs text-cherry">
                            </div>
                            <div>
                                <label class="text-[10px] uppercase font-bold text-mint">Target 1 ($)</label>
                                <input id="manTarget1" type="number" step="0.01" placeholder="150.00" class="w-full mt-1 bg-[#161722] border border-mint/30 rounded-lg px-3 py-1.5 text-xs text-mint">
                            </div>
                        </div>

                        <div class="mb-4">
                            <label class="text-[10px] uppercase font-bold text-slate-400">Strategy Thesis / Notes</label>
                            <textarea id="manThesis" rows="2" placeholder="Strategy thesis or trade notes..." class="w-full mt-1 bg-[#161722] border border-white/10 rounded-lg p-2.5 text-xs text-white"></textarea>
                        </div>

                        <button onclick="submitManualLedgerTrade()" class="w-full py-2.5 rounded-xl bg-mint hover:bg-mint/80 text-black font-bold text-xs transition flex items-center justify-center gap-1.5 shadow-[0_0_12px_rgba(0,245,160,0.2)]">
                            <span>Commit Custom Setup to Active Ledger</span>
                        </button>
                    </div>
                </div>

                <!-- Sub-Pane 3: Closed Trades History -->
                <div id="ledgerSubClosed" class="hidden">
                    <div class="overflow-x-auto rounded-lg border border-white/10 bg-[#090a0f]">
                        <table class="w-full text-left text-xs">
                            <thead>
                                <tr class="border-b border-white/10 bg-white/[0.02] text-slate-400">
                                    <th class="p-2.5">Ticker</th>
                                    <th class="p-2.5">Instrument</th>
                                    <th class="p-2.5">Logged Date</th>
                                    <th class="p-2.5">Closed Date</th>
                                    <th class="p-2.5">Entry ($)</th>
                                    <th class="p-2.5">Exit ($)</th>
                                    <th class="p-2.5">Realized P&L</th>
                                    <th class="p-2.5">Return (%)</th>
                                    <th class="p-2.5">Outcome</th>
                                    <th class="p-2.5">Confluence WR</th>
                                </tr>
                            </thead>
                            <tbody id="ledgerClosedTbody" class="divide-y divide-white/5">
                                <tr><td colspan="10" class="p-4 text-center text-slate-500">No closed trades yet.</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
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
                        <h2 class="text-lg font-black text-white tracking-wide">Gamma Exposure Map</h2>
                        <p class="text-xs text-slate-400 mt-0.5">15-min delayed data from yfinance</p>
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
            <div id="gammaDeskKpiRow" class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5 mb-5">
                <!-- Populated dynamically: Spot, Call Wall, Put Wall, Zero Gamma Flip, Net Dealer GEX -->
            </div>

            <!-- Interactive Plotly Strike GEX Bar Chart -->
            <div class="lux-card p-4 mb-5">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                    <div>
                        <h4 class="text-sm font-bold text-white flex items-center gap-2">
                            <span>Gamma Exposure</span>
                        </h4>
                    </div>
                    <div id="gammaTimestampBadge" class="text-right text-[11px] font-mono text-slate-400"></div>
                </div>
                <div id="gammaDeskChart" style="height: 420px;"></div>
            </div>

            <!-- Gamma Terrain & Ball Rolling Physics Engine Chart -->
            <div class="lux-card p-4 mb-5 border-t-2 border-emerald-500/40">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                    <div>
                        <h4 class="text-sm font-bold text-white flex items-center gap-2">
                            <span>Gamma Terrain & Ball Rolling Physics Engine</span>
                            <span id="gammaTerrainBadge" class="text-[10px] font-bold px-2 py-0.5 rounded bg-mint/15 text-mint border border-mint/20">Gamma = The Ground • Price = The Ball</span>
                        </h4>
                        <p class="text-[11px] text-slate-400">Visualizing market maker delta-hedging friction: Valleys dampen volatility ("Push it back") • Slopes accelerate momentum ("Push it").</p>
                    </div>
                    <div id="gammaTerrainStatusBadge" class="text-right text-xs font-mono font-bold"></div>
                </div>
                <div id="gammaTerrainChart" style="height: 420px;"></div>
                <div class="p-2.5 mt-2 rounded-lg bg-slate-900/40 border border-white/5 flex flex-wrap gap-4 text-[11px] text-slate-300 items-center">
                    <div><span class="text-mint font-bold">● Positive Valley (+GEX):</span> Dealer shock absorbers active ("Push it back") ➔ <b>Mean-reversion pin action</b></div>
                    <div><span class="text-sky-400 font-bold">● Zero-Gamma Crest:</span> Tipping point inflection ➔ <b>Breakout / breakdown trigger boundary</b></div>
                    <div><span class="text-purple-400 font-bold">● Target Base:</span> Pre-positioned OTM accumulation ➔ <b>Institutional landing pad</b></div>
                    <div><span class="text-rose-400 font-bold">● Negative Slope (-GEX):</span> Dealer momentum acceleration ("Push it") ➔ <b>Fast & abrupt cascading volatility</b></div>
                </div>

                <!-- Emerging OTM Target Base Intelligence Banner -->
                <div id="gammaTargetBaseCard" class="p-3 mt-2.5 rounded-lg bg-purple-950/20 border border-purple-500/25 hidden items-start gap-2.5 text-xs text-slate-300">
                    <div class="text-base shrink-0 mt-0.5">🎯</div>
                    <div id="gammaTargetBaseText"></div>
                </div>
            </div>

            <!-- Day-over-Day Gamma Migration & Strike Heat Map -->
            <div class="lux-card p-4 mb-5 border-t-2 border-indigo-500/30">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
                    <div>
                        <h4 class="text-sm font-bold text-white flex items-center gap-2">
                            <span>Day-over-Day Gamma Migration & Strike Heat Map</span>
                            <span id="gammaMigrationStatusBadge" class="text-[10px] font-bold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">Wall Shifts & ΔGEX</span>
                        </h4>
                        <p class="text-[11px] text-slate-400">Visualizing institutional strike inventory flow, Call/Put Wall rolls, and spot price trajectory over time.</p>
                    </div>
                    <div class="flex items-center gap-2">
                        <div class="flex bg-white/5 p-1 rounded-lg text-xs" id="gammaHeatmapModeGroup">
                            <button onclick="setGammaHeatmapMode('net_gex')" id="btnHeatmapNet" class="px-2.5 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30">Total Net GEX ($M)</button>
                            <button onclick="setGammaHeatmapMode('delta_gex')" id="btnHeatmapDelta" class="px-2.5 py-1 rounded font-bold transition text-slate-400">ΔGEX Shift (Brightness = Change)</button>
                        </div>
                    </div>
                </div>

                <!-- Migration 4 KPI Cards -->
                <div id="gammaMigrationKpiRow" class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3.5">
                    <!-- Populated dynamically: Call Wall Shift, Put Wall Shift, Zero Flip Shift, Net GEX Velocity -->
                </div>

                <div id="gammaMigrationSynthesis" class="p-2.5 mb-3 rounded-lg bg-indigo-950/20 border border-indigo-500/20 text-xs text-slate-300 font-sans hidden">
                    <!-- Dynamic quantitative summary -->
                </div>

                <!-- Plotly Heatmap Container -->
                <div id="gammaHeatmapChart" style="height: 480px;"></div>
            </div>

            <!-- Multi-Day Strike Open Interest Migration Table (Historical Base Accumulation) -->
            <div class="lux-card p-4 mb-5 border-t-2 border-emerald-500/40">
                <div class="flex flex-col lg:flex-row lg:items-center justify-between gap-3 mb-3">
                    <div>
                        <div class="flex items-center gap-2">
                            <h4 class="text-sm font-bold text-white flex items-center gap-2">
                                <span>Multi-Day Strike Open Interest Migration Table</span>
                                <span class="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">Historical Base Accumulation &amp; ΔOI Matrix</span>
                            </h4>
                        </div>
                        <p class="text-[11px] text-slate-400 mt-0.5">
                            Track date-by-date Open Interest (OI) evolution across consecutive market sessions. Pinpoints where institutional market makers and smart money build pre-positioned OTM bases ("bracing for impact") or fortify put cushion floors.
                        </p>
                    </div>
                    <div class="flex flex-wrap items-center gap-2">
                        <!-- Calls / Puts Toggle -->
                        <div class="flex bg-white/5 p-1 rounded-lg text-xs" id="oiHistorySideGroup">
                            <button onclick="setOiHistorySide('calls')" id="btnOiCalls" class="px-2.5 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30">Calls (OTM Bases &amp; Pins)</button>
                            <button onclick="setOiHistorySide('puts')" id="btnOiPuts" class="px-2.5 py-1 rounded font-bold transition text-slate-400">Puts (Floors &amp; Cushions)</button>
                        </div>
                        <!-- Relevant / All Filter -->
                        <div class="flex bg-white/5 p-1 rounded-lg text-xs" id="oiHistoryFilterGroup">
                            <button onclick="setOiHistoryFilter(true)" id="btnOiRelevant" class="px-2.5 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30">Relevant Only (<span id="oiRelCount">0</span>)</button>
                            <button onclick="setOiHistoryFilter(false)" id="btnOiAll" class="px-2.5 py-1 rounded font-bold transition text-slate-400">All Strikes (<span id="oiTotalCount">0</span>)</button>
                        </div>
                    </div>
                </div>

                <!-- Top Accumulating Base Callout Strip -->
                <div id="oiHistoryTopStrip" class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3.5">
                    <!-- Populated dynamically with top Call Base & Put Floor accumulators -->
                </div>

                <!-- Multi-Day Table -->
                <div class="overflow-x-auto rounded-lg border border-white/10 bg-[#0b0d12]">
                    <table class="w-full text-left text-xs">
                        <thead id="oiHistoryTableHead" class="bg-white/[0.03] text-[11px] uppercase tracking-wider text-slate-400 border-b border-white/10">
                            <!-- Populated dynamically with date headers -->
                        </thead>
                        <tbody id="oiHistoryTableBody">
                            <tr><td colspan="10" class="text-center py-6 text-slate-500 animate-pulse">Loading multi-day historical Open Interest matrix...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Comprehensive Per-Strike Table Breakdown -->
            <div class="lux-card p-4 mb-5">
                <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                    <div>
                        <h4 class="text-sm font-bold text-white flex items-center gap-2">
                            <span>Detailed Strike Gamma & Open Interest Breakdown</span>
                            <span id="gammaFilterBadge" class="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-sky-500/15 text-sky-400 border border-sky-500/20">Relevant Strikes</span>
                        </h4>
                        <p class="text-[11px] text-slate-400">Institutional structural walls (Call/Put Walls, Zero Flip, ATM) and top dealer open interest battlegrounds.</p>
                    </div>
                    <div class="flex items-center gap-2">
                        <div class="flex bg-white/5 p-1 rounded-lg text-xs" id="gammaFilterGroup">
                            <button onclick="setGammaTableFilter(true)" id="btnGammaRelevant" class="px-2.5 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30">Relevant Only (<span id="gammaRelCount">0</span>)</button>
                            <button onclick="setGammaTableFilter(false)" id="btnGammaAll" class="px-2.5 py-1 rounded font-bold transition text-slate-400">All Strikes (<span id="gammaTotalCount">0</span>)</button>
                        </div>
                    </div>
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


        function updateApiKeyUI() {}
        function saveTopApiKey() {}
        function saveSidebarApiKey() {}
        function promptApiKey() {}

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

        const LOADED_TABS = new Set(['tab2']);
        let AI_DATA = null;

        const TAB_NAMES = {
            'tab2': 'Automated Trade Ideas Desk',
            'tab_gamma': 'Gamma Exposure Map',
            'tab_options': 'Volatility Pricing',
            'tab3': 'Bias Matrix',
            'tab5': 'Indexes & Sector Flow',
            'tab4': 'Magnificant 7 Flow',
            'tab6': 'Synthetic Ratio Rotations',
            'tab7': 'Single Ticker Explorer',
            'tab_ledger': 'Trade Ledger & Audit'
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
            if (tabId === 'tab_ledger') {
                loadLedger();
            } else if (tabId === 'tab2') {
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
                    setTimeout(() => {
                        Plotly.Plots.resize('gammaDeskChart');
                        if (document.getElementById('gammaHeatmapChart')) {
                            Plotly.Plots.resize('gammaHeatmapChart');
                        }
                    }, 50);
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
                populateTickersDropdown();
                populateOptionsTickersDropdown();
                populateGammaDeskTickersDropdown();

                // 3. Load quantitative trade setups and ledger
                loadAiBriefingAsync();
                loadLedger();

            } catch (err) {
                console.error("Failed to load market data:", err);
            }
        }

        function initAgentChat() {}
        function clearAgentChat() {}
        function sendQuickPrompt() {}
        function handleAgentSubmit() {}

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

                    const benchGexHtml = (t.benchmark_symbol && t.benchmark_put_wall) ? `
                        <div class="flex flex-wrap items-center justify-between gap-1.5 p-1 px-2 mb-2 rounded bg-sky-950/20 border border-sky-500/20 text-[10px]">
                            <div class="flex items-center gap-1.5 flex-wrap">
                                <span class="text-sky-400 font-bold">Benchmark Floor (${t.benchmark_symbol}):</span>
                                <span class="text-slate-400">Put Wall:</span>
                                <b class="text-cherry font-mono">${t.benchmark_put_wall}</b>
                                ${t.benchmark_zero_gamma ? `<span class="text-slate-600">|</span><span class="text-slate-400">Zero Flip:</span><b class="text-sky-300 font-mono">${t.benchmark_zero_gamma}</b>` : ''}
                            </div>
                            <span class="text-slate-400 font-mono text-[9px]">${t.benchmark_regime || 'Macro Hedge'}</span>
                        </div>
                    ` : '';

                    const gexHtml = (t.call_wall && t.put_wall) ? `
                        <div class="flex flex-wrap items-center justify-between gap-1.5 p-1.5 px-2 mb-1.5 rounded-lg bg-indigo-950/20 border border-indigo-500/20 text-[10.5px]">
                            <div class="flex items-center gap-1.5 flex-wrap">
                                <span class="text-indigo-400 font-bold">Stock GEX:</span>
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

                            <!-- Next-Day Institutional Defined-Risk Options Strategy (Zero Naked Risk) -->
                            ${t.options_spread ? `
                            <div class="mt-2.5 p-2.5 rounded-xl bg-purple-950/20 border border-purple-500/30 text-xs">
                                <div class="flex flex-wrap items-center justify-between gap-1.5 pb-1.5 mb-1.5 border-b border-purple-500/20">
                                    <div class="flex items-center gap-1.5 flex-wrap">
                                        <span class="px-1.5 py-0.5 rounded text-[9.5px] font-extrabold bg-purple-500/20 text-purple-300 border border-purple-500/30 tracking-wide uppercase">⚡ NEXT-DAY DEALER OPTIONS PLAY</span>
                                        <span class="text-white font-black text-[11px]">${t.options_spread.strategy_name}</span>
                                        <span class="text-[9.5px] text-slate-400 font-mono">(${t.options_spread.dte} DTE • Exp ${t.options_spread.expiration})</span>
                                    </div>
                                    <span class="px-2 py-0.5 rounded text-[10.5px] font-extrabold bg-mint/15 text-mint border border-mint/30">+${t.options_spread.roc_pct}% ROC</span>
                                </div>

                                <!-- 2-Legs Specification (Strict Defined Risk) -->
                                <div class="grid grid-cols-2 gap-2 mb-2">
                                    <div class="p-1.5 rounded bg-black/40 border border-white/5">
                                        <div class="text-[9px] uppercase font-bold text-rose-400">Sold Leg (Bid Edge)</div>
                                        <div class="text-xs font-black text-white mt-0.5">${t.options_spread.short_leg}</div>
                                        <div class="text-[9px] text-slate-400">Anchor @ ${isLong ? 'Put Wall Floor' : 'Call Wall Pin'}</div>
                                    </div>
                                    <div class="p-1.5 rounded bg-black/40 border border-white/5">
                                        <div class="text-[9px] uppercase font-bold text-mint">Bought Wing (Tail Protection)</div>
                                        <div class="text-xs font-black text-white mt-0.5">${t.options_spread.long_leg}</div>
                                        <div class="text-[9px] text-slate-400">100% Defined Risk (No Naked Risk)</div>
                                    </div>
                                </div>

                                <!-- Spread Financials -->
                                <div class="grid grid-cols-4 gap-1.5 p-1.5 rounded bg-white/[0.02] border border-white/5 text-center mb-2">
                                    <div>
                                        <div class="text-[8.5px] uppercase text-slate-400 font-bold">Net Premium</div>
                                        <div class="text-[11px] font-black text-mint mt-0.5">$${t.options_spread.net_premium.toFixed(2)}</div>
                                    </div>
                                    <div>
                                        <div class="text-[8.5px] uppercase text-slate-400 font-bold">Collateral Locked</div>
                                        <div class="text-[11px] font-black text-white mt-0.5">$${t.options_spread.max_risk.toFixed(0)}</div>
                                    </div>
                                    <div>
                                        <div class="text-[8.5px] uppercase text-slate-400 font-bold">Max Profit</div>
                                        <div class="text-[11px] font-black text-mint mt-0.5">$${t.options_spread.max_profit.toFixed(0)}</div>
                                    </div>
                                    <div>
                                        <div class="text-[8.5px] uppercase text-slate-400 font-bold">Breakeven</div>
                                        <div class="text-[11px] font-black text-cyan-300 mt-0.5">$${t.options_spread.breakeven.toFixed(2)}</div>
                                    </div>
                                </div>

                                ${(t.options_spread.pinning_ratio || t.options_spread.net_gex_billions) ? `
                                <div class="flex items-center gap-1.5 flex-wrap mb-1.5">
                                    ${t.options_spread.net_gex_billions ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-bold bg-mint/10 text-mint border border-mint/20">Net GEX: +$${t.options_spread.net_gex_billions.toFixed(2)}B</span>` : ''}
                                    ${t.options_spread.net_gex_velocity ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-bold bg-purple-500/15 text-purple-300 border border-purple-500/25">ΔGEX: ${t.options_spread.net_gex_velocity}</span>` : ''}
                                    ${t.options_spread.pinning_ratio ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-bold bg-blue-500/15 text-blue-300 border border-blue-500/25">🎯 Pin Index: ${(t.options_spread.pinning_ratio * 100).toFixed(0)}%</span>` : ''}
                                    ${t.options_spread.downside_air_pocket ? `<span class="px-1.5 py-0.5 rounded text-[9px] font-bold bg-amber-500/10 text-amber-300 border border-amber-500/20">Void: $${t.options_spread.downside_air_pocket.toFixed(0)}</span>` : ''}
                                </div>
                                ` : ''}

                                <div class="text-[10px] text-slate-300 leading-snug mb-1.5">
                                    <b class="text-purple-300">Dealer Edge:</b> ${t.options_spread.dealer_thesis}
                                </div>

                                <button onclick="prefillAndOpenSpreadLogger(${cards.indexOf(t)})" class="w-full py-1.5 rounded bg-purple-500/20 hover:bg-purple-500/30 text-purple-200 border border-purple-500/40 text-[10.5px] font-bold transition flex items-center justify-center gap-1 shadow-[0_0_10px_rgba(168,85,247,0.15)]">
                                    <span>⚡ Pre-Fill ${t.options_spread.strategy_name} in Trade Logger</span>
                                </button>
                            </div>
                            ` : ''}

                            <!-- Trade Ledger Commit Button & Drawer -->
                            <div class="mt-2.5 pt-2 border-t border-white/5 flex flex-col gap-2">
                                <button onclick="toggleCardLogger(${cards.indexOf(t)})" class="w-full py-1.5 px-3 rounded-lg bg-mint/15 hover:bg-mint/25 text-mint border border-mint/30 text-xs font-bold transition flex items-center justify-center gap-1.5">
                                    <span>Log ${t.ticker} Setup to Trade Ledger</span>
                                </button>
                                <div id="cardLogger-${cards.indexOf(t)}" class="hidden p-3 rounded-xl bg-[#090b10] border border-mint/30 space-y-2.5 text-xs">
                                    <div class="flex items-center justify-between pb-1 border-b border-white/10">
                                        <span class="font-bold text-white">Execution Parameters</span>
                                        <span class="text-[10px] text-slate-400">Locks in frozen entry snapshot</span>
                                    </div>
                                    <div class="grid grid-cols-2 gap-2">
                                        <div>
                                            <label class="text-[9.5px] uppercase font-bold text-slate-400">Instrument</label>
                                            <select id="cardInstType-${cards.indexOf(t)}" onchange="toggleCardLoggerFields(${cards.indexOf(t)})" class="w-full mt-0.5 bg-[#161722] border border-white/10 rounded px-2 py-1 text-white text-xs">
                                                <option value="STOCK">Stock / Equity</option>
                                                <option value="OPTION">Single Options Contract</option>
                                                <option value="SPREAD" ${t.options_spread ? 'selected' : ''}>Defined-Risk Vertical Spread</option>
                                            </select>
                                        </div>
                                        <div id="cardEntryContainer-${cards.indexOf(t)}" class="${t.options_spread ? 'hidden' : ''}">
                                            <label id="cardEntryLabel-${cards.indexOf(t)}" class="text-[9.5px] uppercase font-bold text-slate-400">Entry Fill ($)</label>
                                            <input id="cardEntryPrice-${cards.indexOf(t)}" type="number" step="0.01" value="${parseFloat(String(t.optimal_entry_zone || t.current_price || t.entry || '100').replace(/[^0-9.]/g, '')) || 100}" class="w-full mt-0.5 bg-[#161722] border border-white/10 rounded px-2 py-1 text-white text-xs">
                                        </div>
                                    </div>

                                    <!-- Defined-Risk Spread Specific Fields -->
                                    <div id="cardSpreadFields-${cards.indexOf(t)}" class="${t.options_spread ? '' : 'hidden'} p-2.5 rounded-lg bg-purple-950/25 border border-purple-500/30 space-y-2">
                                        <div class="grid grid-cols-2 gap-2">
                                            <div>
                                                <label class="text-[9px] uppercase font-bold text-purple-300">Spread Type</label>
                                                <select id="cardSpreadType-${cards.indexOf(t)}" class="w-full mt-0.5 bg-[#161722] border border-purple-500/30 rounded px-2 py-1 text-white text-xs">
                                                    <option value="BULL_PUT_CREDIT" ${isLong ? 'selected' : ''}>Bull Put Credit Spread</option>
                                                    <option value="BEAR_CALL_CREDIT" ${!isLong ? 'selected' : ''}>Bear Call Credit Spread</option>
                                                    <option value="BULL_CALL_DEBIT">Bull Call Debit Spread</option>
                                                    <option value="BEAR_PUT_DEBIT">Bear Put Debit Spread</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label class="text-[9px] uppercase font-bold text-purple-300">Expiration</label>
                                                <input id="cardSpreadExpiry-${cards.indexOf(t)}" type="text" value="${t.options_spread ? t.options_spread.expiration : getFutureDateStr(35)}" class="w-full mt-0.5 bg-[#161722] border border-purple-500/30 rounded px-2 py-1 text-white text-xs">
                                            </div>
                                        </div>
                                        <div class="grid grid-cols-3 gap-2">
                                            <div>
                                                <label class="text-[9px] uppercase font-bold text-rose-400">Short Strike ($)</label>
                                                <input id="cardShortStrike-${cards.indexOf(t)}" type="number" step="0.5" value="${t.options_spread ? t.options_spread.short_strike : Math.round(parseFloat(String(t.current_price || '100').replace(/[^0-9.]/g, '')) || 100)}" class="w-full mt-0.5 bg-[#161722] border border-rose-500/30 rounded px-2 py-1 text-white text-xs">
                                            </div>
                                            <div>
                                                <label class="text-[9px] uppercase font-bold text-mint">Long Strike ($)</label>
                                                <input id="cardLongStrike-${cards.indexOf(t)}" type="number" step="0.5" value="${t.options_spread ? t.options_spread.long_strike : (Math.round(parseFloat(String(t.current_price || '100').replace(/[^0-9.]/g, '')) || 100) - 5)}" class="w-full mt-0.5 bg-[#161722] border border-mint/30 rounded px-2 py-1 text-white text-xs">
                                            </div>
                                            <div>
                                                <label class="text-[9px] uppercase font-bold text-mint">Net Premium ($)</label>
                                                <input id="cardNetPrem-${cards.indexOf(t)}" type="number" step="0.05" value="${t.options_spread ? t.options_spread.net_premium : 1.35}" class="w-full mt-0.5 bg-[#161722] border border-mint/30 rounded px-2 py-1 text-mint text-xs">
                                            </div>
                                        </div>
                                        <div class="flex items-center justify-between text-[10px] text-slate-300 pt-1 border-t border-purple-500/20 font-mono">
                                            <span>Collateral: <b class="text-white">$${t.options_spread ? t.options_spread.max_risk : '365'}</b></span>
                                            <span>Max Profit: <b class="text-mint">$${t.options_spread ? t.options_spread.max_profit : '135'}</b></span>
                                            <span class="text-purple-300 font-bold">100% Defined Risk</span>
                                        </div>
                                    </div>

                                    <!-- Single Option Specific Fields -->
                                    <div id="cardOptFields-${cards.indexOf(t)}" class="hidden grid grid-cols-3 gap-2">
                                        <div>
                                            <label class="text-[9.5px] uppercase font-bold text-slate-400">Type</label>
                                            <select id="cardOptType-${cards.indexOf(t)}" class="w-full mt-0.5 bg-[#161722] border border-white/10 rounded px-2 py-1 text-white text-xs">
                                                <option value="CALL" ${isLong ? 'selected' : ''}>CALL</option>
                                                <option value="PUT" ${!isLong ? 'selected' : ''}>PUT</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label class="text-[9.5px] uppercase font-bold text-slate-400">Strike ($)</label>
                                            <input id="cardStrike-${cards.indexOf(t)}" type="number" step="0.5" value="${Math.round(parseFloat(String(t.current_price || t.entry || '100').replace(/[^0-9.]/g, '')) || 100)}" class="w-full mt-0.5 bg-[#161722] border border-white/10 rounded px-2 py-1 text-white text-xs">
                                        </div>
                                        <div>
                                            <label class="text-[9.5px] uppercase font-bold text-slate-400">Expiration</label>
                                            <input id="cardExpiry-${cards.indexOf(t)}" type="text" value="${getFutureDateStr(35)}" class="w-full mt-0.5 bg-[#161722] border border-white/10 rounded px-2 py-1 text-white text-xs">
                                        </div>
                                    </div>

                                    <div class="grid grid-cols-3 gap-2">
                                        <div>
                                            <label id="cardSizeLabel-${cards.indexOf(t)}" class="text-[9.5px] uppercase font-bold text-slate-400">${t.options_spread ? 'Spreads' : 'Shares'}</label>
                                            <input id="cardSize-${cards.indexOf(t)}" type="number" min="1" value="${t.options_spread ? 1 : 10}" class="w-full mt-0.5 bg-[#161722] border border-white/10 rounded px-2 py-1 text-white text-xs">
                                        </div>
                                        <div>
                                            <label class="text-[9.5px] uppercase font-bold text-cherry">Stop ($)</label>
                                            <input id="cardStop-${cards.indexOf(t)}" type="number" step="0.01" value="${parseFloat(String(t.stop || '0').replace(/[^0-9.]/g, '')) || 0}" class="w-full mt-0.5 bg-[#161722] border border-cherry/30 rounded px-2 py-1 text-cherry text-xs">
                                        </div>
                                        <div>
                                            <label class="text-[9.5px] uppercase font-bold text-mint">Target ($)</label>
                                            <input id="cardTp1-${cards.indexOf(t)}" type="number" step="0.01" value="${parseFloat(String(t.target1 || '0').replace(/[^0-9.]/g, '')) || 0}" class="w-full mt-0.5 bg-[#161722] border border-mint/30 rounded px-2 py-1 text-mint text-xs">
                                        </div>
                                    </div>
                                    <button onclick="commitCardTrade(${cards.indexOf(t)})" id="btnCommitCard-${cards.indexOf(t)}" class="w-full py-2 rounded-lg bg-mint text-black font-extrabold text-xs hover:bg-mint/80 transition flex items-center justify-center gap-1 shadow-[0_0_12px_rgba(0,245,160,0.2)]">
                                        <span>Confirm & Log to Ledger</span>
                                    </button>
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

        let CURRENT_GAMMA_ONLY_RELEVANT = true;
        let CURRENT_GAMMA_DATA = null;

        function setGammaTableFilter(onlyRelevant) {
            CURRENT_GAMMA_ONLY_RELEVANT = onlyRelevant;
            const btnRel = document.getElementById('btnGammaRelevant');
            const btnAll = document.getElementById('btnGammaAll');
            const badge = document.getElementById('gammaFilterBadge');
            if (btnRel && btnAll) {
                if (onlyRelevant) {
                    btnRel.className = 'px-2.5 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30';
                    btnAll.className = 'px-2.5 py-1 rounded font-bold transition text-slate-400';
                    if (badge) {
                        badge.innerText = 'Relevant Strikes';
                        badge.className = 'text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-sky-500/15 text-sky-400 border border-sky-500/20';
                    }
                } else {
                    btnRel.className = 'px-2.5 py-1 rounded font-bold transition text-slate-400';
                    btnAll.className = 'px-2.5 py-1 rounded font-bold transition bg-white/10 text-white border border-white/20';
                    if (badge) {
                        badge.innerText = 'All Mapped Strikes';
                        badge.className = 'text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-white/10 text-slate-300 border border-white/20';
                    }
                }
            }
            if (CURRENT_GAMMA_DATA) {
                renderGammaTableRows(CURRENT_GAMMA_DATA);
            }
        }

        
        let CURRENT_HEATMAP_MODE = 'net_gex';

        async function setGammaHeatmapMode(mode) {
            CURRENT_HEATMAP_MODE = mode;
            const btnNet = document.getElementById('btnHeatmapNet');
            const btnDelta = document.getElementById('btnHeatmapDelta');
            if (btnNet && btnDelta) {
                if (mode === 'net_gex') {
                    btnNet.className = 'px-2.5 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30';
                    btnDelta.className = 'px-2.5 py-1 rounded font-bold transition text-slate-400';
                } else {
                    btnDelta.className = 'px-2.5 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30';
                    btnNet.className = 'px-2.5 py-1 rounded font-bold transition text-slate-400';
                }
            }
            const sel = document.getElementById('gammaDeskTickerSelect');
            const ticker = sel ? sel.value : 'SPY';
            loadGammaMigrationAndHeatmap(ticker);
        }

        function formatSignedMoney(val, suffix = 'B', decimals = 2) {
            if (val === null || val === undefined || isNaN(val)) return '$0.00' + suffix;
            const num = Number(val);
            const sign = num >= 0 ? '+' : '-';
            return `${sign}$${Math.abs(num).toFixed(decimals)}${suffix}`;
        }

        async function loadGammaMigrationAndHeatmap(ticker) {
            try {
                // Fetch shift summary
                const shiftRes = await fetch(`/api/gamma/${encodeURIComponent(ticker)}/shift`);
                const shift = await shiftRes.json();

                const kpiRow = document.getElementById('gammaMigrationKpiRow');
                const synBox = document.getElementById('gammaMigrationSynthesis');

                if (kpiRow && shift && shift.has_history) {
                    const cwShift = Number(shift.call_wall_shift || 0);
                    const pwShift = Number(shift.put_wall_shift || 0);
                    const velShift = Number(shift.net_gex_velocity_b || 0);

                    const cwCol = Math.abs(cwShift) < 0.01 ? 'text-slate-300' : (cwShift > 0 ? 'text-mint' : 'text-cherry');
                    const pwCol = Math.abs(pwShift) < 0.01 ? 'text-slate-300' : (pwShift > 0 ? 'text-mint' : 'text-cherry');
                    const velCol = Math.abs(velShift) < 0.05 ? 'text-slate-300' : (velShift > 0 ? 'text-mint' : 'text-cherry');

                    kpiRow.innerHTML = `
                        <div class="lux-card p-3 border-l-2 border-mint">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Call Wall Migration</div>
                            <div class="text-base font-bold ${cwCol} mt-0.5">$${Number(shift.call_wall_current || 0).toFixed(1)} <span class="text-xs font-normal text-slate-400">(${cwShift >= 0 ? '+' : ''}${cwShift.toFixed(1)})</span></div>
                            <div class="text-[10px] text-slate-400 mt-0.5">${(shift.call_wall_status || '').replace('_', ' ')} • Prev $${Number(shift.call_wall_previous || 0).toFixed(1)}</div>
                        </div>
                        <div class="lux-card p-3 border-l-2 border-cherry">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Put Wall Migration</div>
                            <div class="text-base font-bold ${pwCol} mt-0.5">$${Number(shift.put_wall_current || 0).toFixed(1)} <span class="text-xs font-normal text-slate-400">(${pwShift >= 0 ? '+' : ''}${pwShift.toFixed(1)})</span></div>
                            <div class="text-[10px] text-slate-400 mt-0.5">${(shift.put_wall_status || '').replace('_', ' ')} • Prev $${Number(shift.put_wall_previous || 0).toFixed(1)}</div>
                        </div>
                        <div class="lux-card p-3 border-l-2 border-sky-400">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Zero Flip Migration</div>
                            <div class="text-base font-bold text-sky-400 mt-0.5">$${Number(shift.zero_gamma_current || 0).toFixed(1)} <span class="text-xs font-normal text-slate-400">(${Number(shift.zero_gamma_shift || 0) >= 0 ? '+' : ''}${Number(shift.zero_gamma_shift || 0).toFixed(1)})</span></div>
                            <div class="text-[10px] text-slate-400 mt-0.5">Prev $${Number(shift.zero_gamma_previous || 0).toFixed(1)}</div>
                        </div>
                        <div class="lux-card p-3 border-l-2 border-indigo-400">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Net GEX Velocity (&Delta;GEX)</div>
                            <div class="text-base font-bold ${velCol} mt-0.5">${formatSignedMoney(shift.net_gex_current_b, 'B', 2)} <span class="text-xs font-normal text-slate-400">(${formatSignedMoney(shift.net_gex_velocity_b, 'B', 2)})</span></div>
                            <div class="text-[10px] text-slate-400 mt-0.5">${shift.velocity_label || 'Stable'}</div>
                        </div>
                    `;

                    if (synBox && shift.institutional_synthesis) {
                        synBox.innerHTML = `<b>Quantitative Flow Synthesis:</b> ${shift.institutional_synthesis}`;
                        synBox.classList.remove('hidden');
                    }
                } else if (kpiRow) {
                    kpiRow.innerHTML = `<div class="col-span-full p-3.5 text-center text-xs text-slate-400 bg-white/[0.02] rounded-lg border border-white/5">Baseline snapshot recorded for <b>${ticker}</b>. Additional market sessions will build multi-day migration & heat map.</div>`;
                    if (synBox) synBox.classList.add('hidden');
                }

                // Fetch heatmap matrix
                const heatRes = await fetch(`/api/gamma/${encodeURIComponent(ticker)}/heatmap?mode=${CURRENT_HEATMAP_MODE}&days=10`);
                const heat = await heatRes.json();

                if (heat && !heat.error && heat.dates && heat.dates.length > 0) {
                    renderGammaHeatmapPlotly(heat);
                } else {
                    const hmContainer = document.getElementById('gammaHeatmapChart');
                    if (hmContainer) {
                        hmContainer.innerHTML = `<div class="flex items-center justify-center h-full text-xs text-slate-500 font-mono">Insufficient historical snapshots to render multi-day heat map for ${ticker}.</div>`;
                    }
                }

                // Fetch multi-day historical Open Interest migration matrix
                try {
                    const oiRes = await fetch(`/api/gamma/${encodeURIComponent(ticker)}/oi-history?days=5`);
                    const oiData = await oiRes.json();
                    if (oiData && oiData.has_data) {
                        renderOiHistoryTable(oiData);
                    } else {
                        const oiBody = document.getElementById('oiHistoryTableBody');
                        if (oiBody) {
                            oiBody.innerHTML = `<tr><td colspan="12" class="text-center py-4 text-slate-500">Insufficient historical sessions to display multi-day Open Interest table for ${ticker}.</td></tr>`;
                        }
                    }
                } catch (oiErr) {
                    console.error("Historical OI table load error:", oiErr);
                }
            } catch (e) {
                console.error("Gamma heatmap load error:", e);
            }
        }

        function renderGammaHeatmapPlotly(data) {
            const z = data.z_matrix;
            const x = data.dates;
            const y = data.strikes;
            const hover = data.hover_matrix;

            // Compute symmetric bounds with 95th percentile clamp
            const allAbs = [];
            for (let r = 0; r < z.length; r++) {
                for (let c = 0; c < z[r].length; c++) {
                    const v = Math.abs(z[r][c]);
                    if (v > 0.01) allAbs.push(v);
                }
            }
            allAbs.sort((a, b) => a - b);
            let maxAbs = 10;
            if (allAbs.length > 0) {
                const p95Idx = Math.floor(allAbs.length * 0.95);
                maxAbs = Math.max(allAbs[p95Idx] * 1.25, 5.0);
            }

            const customColorscale = [
                [0.0, '#FF2E55'],
                [0.45, '#4c0519'],
                [0.50, '#0e1117'],
                [0.55, '#064e3b'],
                [1.0, '#00F5A0']
            ];

            const traceHeatmap = {
                type: 'heatmap',
                x: x,
                y: y,
                z: z,
                text: hover,
                hoverinfo: 'text',
                colorscale: customColorscale,
                zmin: -maxAbs,
                zmax: maxAbs,
                colorbar: {
                    title: { text: CURRENT_HEATMAP_MODE === 'delta_gex' ? 'ΔGEX ($M)' : 'Net GEX ($M)', font: { color: '#94a3b8', size: 10 } },
                    tickfont: { color: '#94a3b8', size: 9 },
                    thickness: 12,
                    len: 0.85
                }
            };

            const traceSpots = {
                type: 'scatter',
                mode: 'lines+markers',
                x: x,
                y: data.spots,
                line: { color: '#ffffff', width: 2.5 },
                marker: { size: 7, color: '#ffffff', line: { color: '#0e1117', width: 1.5 } },
                name: 'Spot Price',
                hovertemplate: 'Spot Price: $%{y:,.2f}<extra></extra>'
            };

            const traceCW = {
                type: 'scatter',
                mode: 'markers',
                x: x,
                y: data.call_walls,
                marker: { symbol: 'diamond', size: 9, color: '#00F5A0', line: { color: '#ffffff', width: 1 } },
                name: 'Call Wall (Ceiling Pin)',
                hovertemplate: 'Call Wall Pin: $%{y:,.2f}<extra></extra>'
            };

            const tracePW = {
                type: 'scatter',
                mode: 'markers',
                x: x,
                y: data.put_walls,
                marker: { symbol: 'triangle-up', size: 10, color: '#FF2E55', line: { color: '#ffffff', width: 1 } },
                name: 'Put Wall (Support Floor)',
                hovertemplate: 'Put Wall Floor: $%{y:,.2f}<extra></extra>'
            };

            const modeTitle = CURRENT_HEATMAP_MODE === 'delta_gex' ? 'Day-over-Day ΔGEX Shift (Brightness = Change Intensity)' : 'Net Gamma Distribution by Strike ($M)';
            const layout = {
                title: {
                    text: `<b>${data.ticker} Gamma Migration Heat Map</b> | ${modeTitle}<br><span style="font-size:10px; color:#94a3b8; font-weight:normal; font-family:monospace;">White Line = Spot Trajectory • Diamonds = Call Wall • Triangles = Put Wall</span>`,
                    font: { color: '#f1f5f9', size: 13 }
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 58, r: 25, t: 65, b: 40 },
                xaxis: {
                    title: { text: 'Trading Session', font: { color: '#94a3b8', size: 11 } },
                    gridcolor: 'rgba(255, 255, 255, 0.05)',
                    tickfont: { color: '#94a3b8', size: 10 }
                },
                yaxis: {
                    title: { text: 'Option Strike Price ($)', font: { color: '#94a3b8', size: 11 } },
                    gridcolor: 'rgba(255, 255, 255, 0.05)',
                    tickfont: { color: '#94a3b8', size: 10 }
                },
                showlegend: true,
                legend: {
                    orientation: 'h',
                    x: 1,
                    xanchor: 'right',
                    y: 1.1,
                    font: { color: '#94a3b8', size: 10 }
                }
            };

            Plotly.newPlot('gammaHeatmapChart', [traceHeatmap, traceSpots, traceCW, tracePW], layout, { responsive: true, displayModeBar: false });
        }

        let CURRENT_OI_HISTORY_DATA = null;
        let CURRENT_OI_HISTORY_SIDE = 'calls'; // 'calls' or 'puts'
        let CURRENT_OI_HISTORY_RELEVANT = true;

        function setOiHistorySide(side) {
            CURRENT_OI_HISTORY_SIDE = side;
            const btnCalls = document.getElementById('btnOiCalls');
            const btnPuts = document.getElementById('btnOiPuts');
            if (btnCalls && btnPuts) {
                if (side === 'calls') {
                    btnCalls.className = 'px-2.5 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30';
                    btnPuts.className = 'px-2.5 py-1 rounded font-bold transition text-slate-400';
                } else {
                    btnPuts.className = 'px-2.5 py-1 rounded font-bold transition bg-rose-500/20 text-rose-400 border border-rose-500/30';
                    btnCalls.className = 'px-2.5 py-1 rounded font-bold transition text-slate-400';
                }
            }
            if (CURRENT_OI_HISTORY_DATA) {
                renderOiHistoryTable(CURRENT_OI_HISTORY_DATA);
            }
        }

        function setOiHistoryFilter(onlyRelevant) {
            CURRENT_OI_HISTORY_RELEVANT = onlyRelevant;
            const btnRel = document.getElementById('btnOiRelevant');
            const btnAll = document.getElementById('btnOiAll');
            if (btnRel && btnAll) {
                if (onlyRelevant) {
                    btnRel.className = 'px-2.5 py-1 rounded font-bold transition bg-mint/20 text-mint border border-mint/30';
                    btnAll.className = 'px-2.5 py-1 rounded font-bold transition text-slate-400';
                } else {
                    btnAll.className = 'px-2.5 py-1 rounded font-bold transition bg-indigo-500/20 text-indigo-300 border border-indigo-500/30';
                    btnRel.className = 'px-2.5 py-1 rounded font-bold transition text-slate-400';
                }
            }
            if (CURRENT_OI_HISTORY_DATA) {
                renderOiHistoryTable(CURRENT_OI_HISTORY_DATA);
            }
        }

        function renderOiHistoryTable(data) {
            if (!data || !data.has_data || !data.strikes) return;
            CURRENT_OI_HISTORY_DATA = data;

            const thead = document.getElementById('oiHistoryTableHead');
            const tbody = document.getElementById('oiHistoryTableBody');
            const topStrip = document.getElementById('oiHistoryTopStrip');
            const relCount = document.getElementById('oiRelCount');
            const totalCount = document.getElementById('oiTotalCount');

            if (relCount) relCount.textContent = data.relevant_strikes || 0;
            if (totalCount) totalCount.textContent = data.total_strikes || 0;

            const isCalls = CURRENT_OI_HISTORY_SIDE === 'calls';
            const dates = data.dates_labels || [];

            // 1. Render Top Strip Highlights
            if (topStrip) {
                const topC = (data.top_calls_accum && data.top_calls_accum.length > 0) ? data.top_calls_accum[0] : null;
                const topP = (data.top_puts_accum && data.top_puts_accum.length > 0) ? data.top_puts_accum[0] : null;

                let cHtml = '';
                if (topC) {
                    const cTotSign = topC.call_tot >= 0 ? '+' : '';
                    const c1dSign = topC.call_1d >= 0 ? '+' : '';
                    cHtml = `
                        <div class="lux-card p-3 border-l-2 border-mint bg-gradient-to-r from-mint/5 to-transparent">
                            <div class="flex items-center justify-between">
                                <span class="text-[10px] uppercase font-bold tracking-wider text-mint">Top Call Base Accumulation</span>
                                <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-mint/20 text-mint font-bold">${topC.call_pace}</span>
                            </div>
                            <div class="text-base font-bold text-white mt-1">
                                Strike ${topC.strike_str} <span class="text-xs font-normal text-slate-300">(${topC.dist_pct >= 0 ? '+' : ''}${topC.dist_pct}%)</span>
                            </div>
                            <div class="text-[11px] text-slate-400 mt-0.5">
                                Latest OI: <b class="text-white">${topC.call_latest.toLocaleString()}</b> • 1D Shift: <b class="text-mint">${c1dSign}${topC.call_1d.toLocaleString()}</b> • Multi-Day: <b class="text-mint">${cTotSign}${topC.call_tot.toLocaleString()} (+${topC.call_pct}%)</b>
                            </div>
                        </div>
                    `;
                }

                let pHtml = '';
                if (topP) {
                    const pTotSign = topP.put_tot >= 0 ? '+' : '';
                    const p1dSign = topP.put_1d >= 0 ? '+' : '';
                    pHtml = `
                        <div class="lux-card p-3 border-l-2 border-rose-500 bg-gradient-to-r from-rose-500/5 to-transparent">
                            <div class="flex items-center justify-between">
                                <span class="text-[10px] uppercase font-bold tracking-wider text-rose-400">Top Put Floor Fortification</span>
                                <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 font-bold">${topP.put_pace}</span>
                            </div>
                            <div class="text-base font-bold text-white mt-1">
                                Strike ${topP.strike_str} <span class="text-xs font-normal text-slate-300">(${topP.dist_pct >= 0 ? '+' : ''}${topP.dist_pct}%)</span>
                            </div>
                            <div class="text-[11px] text-slate-400 mt-0.5">
                                Latest OI: <b class="text-white">${topP.put_latest.toLocaleString()}</b> • 1D Shift: <b class="text-rose-400">${p1dSign}${topP.put_1d.toLocaleString()}</b> • Multi-Day: <b class="text-rose-400">${pTotSign}${topP.put_tot.toLocaleString()} (+${topP.put_pct}%)</b>
                            </div>
                        </div>
                    `;
                }
                topStrip.innerHTML = cHtml + pHtml;
            }

            // 2. Render Header Row
            if (thead) {
                let thHtml = `
                    <tr>
                        <th class="py-2.5 px-3">Strike</th>
                        <th class="py-2.5 px-3">Dist</th>
                        <th class="py-2.5 px-3">Role</th>
                `;
                dates.forEach((d, i) => {
                    const isLatest = (i === dates.length - 1);
                    const colLabel = isLatest ? `${d} (Today)` : d;
                    const colClass = isLatest ? 'text-white font-bold bg-white/[0.04]' : '';
                    thHtml += `<th class="py-2.5 px-3 text-right ${colClass}">${colLabel} ${isCalls ? 'Call OI' : 'Put OI'}</th>`;
                });
                thHtml += `
                        <th class="py-2.5 px-3 text-right">1-Day &Delta;OI</th>
                        <th class="py-2.5 px-3 text-right">Net &Delta;OI</th>
                        <th class="py-2.5 px-3">Accumulation Pace</th>
                    </tr>
                `;
                thead.innerHTML = thHtml;
            }

            // 3. Render Body Rows
            if (tbody) {
                const list = CURRENT_OI_HISTORY_RELEVANT
                    ? data.strikes.filter(s => s.is_relevant)
                    : data.strikes;

                if (list.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="12" class="text-center py-6 text-slate-500">No strikes matching the selected filter.</td></tr>`;
                    return;
                }

                tbody.innerHTML = list.map(s => {
                    const series = isCalls ? s.call_series : s.put_series;
                    const delta1d = isCalls ? s.call_1d : s.put_1d;
                    const deltaTot = isCalls ? s.call_tot : s.put_tot;
                    const pctTot = isCalls ? s.call_pct : s.put_pct;
                    const pace = isCalls ? s.call_pace : s.put_pace;

                    const d1dSign = delta1d >= 0 ? '+' : '';
                    const dTotSign = deltaTot >= 0 ? '+' : '';

                    const d1dClass = delta1d > 0 ? 'text-mint font-bold' : (delta1d < 0 ? 'text-rose-400 font-bold' : 'text-slate-500');
                    const dTotClass = deltaTot > 0 ? 'text-mint font-bold' : (deltaTot < 0 ? 'text-rose-400 font-bold' : 'text-slate-500');

                    let roleTag = '<span class="text-slate-500 text-[10.5px]">—</span>';
                    if (s.role_type === 'target_base') {
                        roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">🎯 TARGET BASE</span>';
                    } else if (s.role_type === 'call_wall') {
                        roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-mint/20 text-mint border border-mint/30">CALL WALL (Pin)</span>';
                    } else if (s.role_type === 'put_wall') {
                        roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">PUT WALL (Floor)</span>';
                    } else if (s.role_type === 'zero_flip') {
                        roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-500/20 text-sky-400 border border-sky-500/30">ZERO FLIP</span>';
                    } else if (s.role_type === 'atm') {
                        roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-white/10 text-white">ATM SPOT</span>';
                    }

                    let paceTag = `<span class="text-[10px] text-slate-400 font-mono">${pace}</span>`;
                    if (pace.includes('Heavy')) {
                        paceTag = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/15 text-amber-300 border border-amber-500/25">${pace}</span>`;
                    } else if (pace.includes('Accumulation') || pace.includes('Hedging')) {
                        paceTag = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-mint/15 text-mint border border-mint/20">${pace}</span>`;
                    } else if (pace.includes('Unwinding')) {
                        paceTag = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/15 text-rose-400 border border-rose-500/20">${pace}</span>`;
                    }

                    // Series cells
                    const seriesCells = series.map((val, idx) => {
                        const isLatest = (idx === series.length - 1);
                        if (val === null || val === undefined) {
                            return `<td class="py-2.5 px-3 text-right font-mono text-slate-600" title="Outside tracked strike window on this date">—</td>`;
                        }
                        const cellClass = isLatest ? 'font-bold text-white bg-white/[0.02]' : 'text-slate-300';
                        return `<td class="py-2.5 px-3 text-right font-mono ${cellClass}">${val.toLocaleString()}</td>`;
                    }).join('');

                    return `
                        <tr class="border-b border-white/[0.04] hover:bg-white/[0.02] transition">
                            <td class="py-2.5 px-3 font-bold text-white font-mono">${s.strike_str}</td>
                            <td class="py-2.5 px-3 font-mono text-slate-400">${s.dist_pct >= 0 ? '+' : ''}${s.dist_pct}%</td>
                            <td class="py-2.5 px-3">${roleTag}</td>
                            ${seriesCells}
                            <td class="py-2.5 px-3 text-right font-mono ${d1dClass}">${d1dSign}${delta1d.toLocaleString()}</td>
                            <td class="py-2.5 px-3 text-right font-mono ${dTotClass}">${dTotSign}${deltaTot.toLocaleString()} <span class="text-[10px] font-normal text-slate-400">(${dTotSign}${pctTot}%)</span></td>
                            <td class="py-2.5 px-3">${paceTag}</td>
                        </tr>
                    `;
                }).join('');
            }
        }

        function renderGammaTableRows(g) {
            const tbody = document.getElementById('gammaDeskStrikesTbody');
            if (!tbody || !g.strikes || g.strikes.length === 0) return;
            const spot = g.spot_price;
            const targetZgStrike = g.closest_zero_gamma_strike !== undefined ? g.closest_zero_gamma_strike : g.zero_gamma_level;
            
            const displayList = CURRENT_GAMMA_ONLY_RELEVANT 
                ? g.strikes.filter(s => s.is_relevant) 
                : g.strikes;

            const rowsToRender = displayList.length > 0 ? displayList : g.strikes;
            tbody.innerHTML = rowsToRender.map(s => {
                const distPct = ((s.strike / spot) - 1) * 100;
                const distStr = (distPct >= 0 ? '+' : '') + distPct.toFixed(1) + '%';
                const netClass = s.net_gex >= 0 ? 'text-mint font-semibold' : 'text-cherry font-semibold';
                const netSign = s.net_gex >= 0 ? '+' : '';

                let roleTag = '<span class="text-slate-500 text-[10px]">Normal Liquidity</span>';
                if (Math.abs(s.strike - g.call_wall) < 0.01) {
                    roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-mint/20 text-mint border border-mint/30">CALL WALL (Pin)</span>';
                } else if (Math.abs(s.strike - g.put_wall) < 0.01) {
                    roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">PUT WALL (Floor)</span>';
                } else if (Math.abs(s.strike - targetZgStrike) < 0.01) {
                    roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-sky-500/20 text-sky-400 border border-sky-500/30">ZERO FLIP</span>';
                } else if (Math.abs(s.strike - spot) < (spot * 0.0075)) {
                    roleTag = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-white/10 text-white">ATM SPOT</span>';
                } else if (s.structural_role && s.structural_role !== "Normal Liquidity") {
                    roleTag = `<span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-white/5 text-slate-300 border border-white/10">${s.structural_role}</span>`;
                }

                return `
                    <tr class="border-b border-white/[0.04] hover:bg-white/[0.02] transition">
                        <td class="py-2.5 px-3 font-bold text-white font-mono">$${s.strike.toFixed(2)}</td>
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

        async function loadGammaDesk(force = false) {
            const tickerSelect = document.getElementById('gammaDeskTickerSelect');
            const ticker = (tickerSelect && tickerSelect.value) ? tickerSelect.value : 'SPY';
            const kpiRow = document.getElementById('gammaDeskKpiRow');
            const chartDiv = document.getElementById('gammaDeskChart');
            const countSpan = document.getElementById('gammaDeskStrikesCount');
            const tbody = document.getElementById('gammaDeskStrikesTbody');

            if (kpiRow) {
                kpiRow.innerHTML = `<div class="col-span-full py-4 text-center text-xs text-slate-400">Scanning real-time option chains & computing Black-Scholes Dealer Gamma Exposure (GEX) profile for <b>${ticker}</b>...</div>`;
            }
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
                    const flipVal = typeof g.zero_gamma_level === 'number' ? g.zero_gamma_level.toFixed(2) : g.zero_gamma_level;
                    kpiRow.innerHTML = `
                        <div class="lux-card p-3 border-l-2 border-slate-400">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Spot Price</div>
                            <div class="text-lg font-bold text-white mt-0.5">$${g.spot_price.toFixed(2)}</div>
                        </div>
                        <div class="lux-card p-3 border-l-2 border-mint">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Call Wall</div>
                            <div class="text-lg font-bold text-mint mt-0.5">$${g.call_wall} <span class="text-xs font-normal text-slate-300">(${g.call_wall_dist_pct >= 0 ? '+' : ''}${g.call_wall_dist_pct}%)</span></div>
                        </div>
                        <div class="lux-card p-3 border-l-2 border-cherry">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Put Wall</div>
                            <div class="text-lg font-bold text-cherry mt-0.5">$${g.put_wall} <span class="text-xs font-normal text-slate-300">(${g.put_wall_dist_pct >= 0 ? '+' : ''}${g.put_wall_dist_pct}%)</span></div>
                        </div>
                        <div class="lux-card p-3 border-l-2 border-sky-400">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Zero Gamma Flip</div>
                            <div class="text-lg font-bold text-sky-400 mt-0.5">$${flipVal} <span class="text-xs font-normal text-slate-300">(${g.zero_gamma_dist_pct >= 0 ? '+' : ''}${g.zero_gamma_dist_pct}%)</span></div>
                        </div>
                        <div class="lux-card p-3 border-l-2 ${isLong ? 'border-mint' : 'border-rose-500'}">
                            <div class="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Net Dealer GEX</div>
                            <div class="text-lg font-bold ${isLong ? 'text-mint' : 'text-rose-400'} mt-0.5">${g.total_net_gex_billions >= 0 ? '+' : ''}${g.total_net_gex_billions.toFixed(2)}B</div>
                        </div>
                    `;
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

                    const loadedStr = g.loaded_at_display || g.loaded_at_str || 'Live Snapshot';
                    const marketStr = g.market_trade_time || 'Session Close';
                    const elapsed = typeof g.elapsed_seconds === 'number' ? g.elapsed_seconds.toFixed(2) : (g.elapsed_seconds || 0);
                    const strikesCount = g.strikes_count || (g.strikes ? g.strikes.length : 0);
                    const expsCount = (g.expirations_scanned || []).length;

                    const sessionLabel = g.market_session_label || '';
                    const tsBadge = document.getElementById('gammaTimestampBadge');
                    if (tsBadge) {
                        tsBadge.innerHTML = `
                            <div class="text-slate-300 font-mono"><span class="text-slate-500 font-sans">Exchange Print:</span> <b class="text-sky-300">${marketStr}</b>${sessionLabel ? ` <span class="text-slate-400 font-normal">(${sessionLabel})</span>` : ''}</div>
                            <div class="text-[10.5px] text-slate-400 font-mono mt-0.5"><span class="text-slate-500 font-sans">Feed Ingested:</span> <span class="text-slate-300">${loadedStr}</span>${elapsed ? ` <span class="text-slate-500">(${elapsed}s)</span>` : ''}</div>
                        `;
                    }

                    const layout = {
                        barmode: 'relative',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        margin: { l: 45, r: 25, t: 25, b: 35 },
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
                        legend: { orientation: 'h', x: 1, xanchor: 'right', y: 1.12, font: { color: '#94a3b8', size: 10 } }
                    };

                    // Auto-focus default view on the relevant strikes window
                    const relStrikes = (g.strikes || []).filter(s => s.is_relevant).map(s => s.strike);
                    if (relStrikes.length > 0) {
                        const kMin = Math.min(...relStrikes);
                        const kMax = Math.max(...relStrikes);
                        const span = kMax - kMin;
                        const pad = Math.max(span * 0.08, g.spot_price * 0.015);
                        layout.xaxis.range = [kMin - pad, kMax + pad];
                    }

                    Plotly.newPlot('gammaDeskChart', [traceCalls, tracePuts], layout, { responsive: true, displayModeBar: false });

                    // Render Gamma Terrain & Ball Rolling Physics Engine Chart
                    renderGammaTerrainChart(g);
                }

                // 4. Tabular Per-Strike Breakdown (Curated Relevant Strikes)
                CURRENT_GAMMA_DATA = g;
                const relCount = (g.strikes || []).filter(s => s.is_relevant).length;
                const totCount = (g.strikes || []).length;
                const relSpan = document.getElementById('gammaRelCount');
                const totSpan = document.getElementById('gammaTotalCount');
                if (relSpan) relSpan.innerText = relCount;
                if (totSpan) totSpan.innerText = totCount;
                if (countSpan) countSpan.innerText = `${relCount} Relevant / ${totCount} Total Mapped`;

                renderGammaTableRows(g);

                // 5. Load and Render Day-over-Day Gamma Migration & Heatmap
                loadGammaMigrationAndHeatmap(ticker);

            } catch (err) {
                console.error("Gamma desk fetch error:", err);
                if (kpiRow) kpiRow.innerHTML = `<div class="col-span-full text-xs text-cherry bg-cherry/10 p-3 rounded-lg border border-cherry/20">Error loading gamma desk: ${err.message}</div>`;
            }
        }

        function renderGammaTerrainChart(g) {
            const chartDiv = document.getElementById('gammaTerrainChart');
            if (!chartDiv || typeof Plotly === 'undefined' || !g || !g.strikes || g.strikes.length === 0) return;

            const spot = g.spot_price;
            const zg = g.zero_gamma_level || spot;
            const cw = g.call_wall || spot;
            const pw = g.put_wall || spot;
            const ticker = g.ticker || 'SPY';

            // Sort strikes by strike price
            const sorted = [...g.strikes].sort((a, b) => a.strike - b.strike);
            const rawK = sorted.map(s => s.strike);
            const rawNet = sorted.map(s => s.net_gex);

            if (rawK.length < 3) return;

            // 1D Linear Interpolation
            function interp(x, xs, ys) {
                if (x <= xs[0]) return ys[0];
                if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
                let i = 0;
                while (i < xs.length - 1 && xs[i + 1] < x) i++;
                const t = (x - xs[i]) / (xs[i + 1] - xs[i]);
                return ys[i] + t * (ys[i + 1] - ys[i]);
            }

            // Focus bounds around relevant strikes or spot +/- 15%
            const relStrikes = sorted.filter(s => s.is_relevant).map(s => s.strike);
            const kMin = relStrikes.length > 0 ? Math.min(...relStrikes) : Math.max(spot * 0.85, rawK[0]);
            const kMax = relStrikes.length > 0 ? Math.max(...relStrikes) : Math.min(spot * 1.15, rawK[rawK.length - 1]);

            const gridMin = Math.max(rawK[0], kMin - (spot * 0.04));
            const gridMax = Math.min(rawK[rawK.length - 1], kMax + (spot * 0.04));

            const N = 260;
            const denseK = [];
            const denseNet = [];
            const posTerrain = [];
            const negTerrain = [];

            for (let i = 0; i < N; i++) {
                const k = gridMin + (i / (N - 1)) * (gridMax - gridMin);
                const val = interp(k, rawK, rawNet);
                denseK.push(k);
                denseNet.push(val);
                posTerrain.push(Math.max(val, 0));
                negTerrain.push(Math.min(val, 0));
            }

            const spotGex = interp(spot, rawK, rawNet);
            const zgDistPct = zg > 0 ? (((spot - zg) / spot) * 100) : 0;
            const isAtCrest = zg > 0 && Math.abs(spot - zg) / spot < 0.008;
            const isInValley = spotGex >= 0 && !isAtCrest;

            let ballState = '';
            let ballColor = '';
            if (isAtCrest) {
                ballState = 'PERCHED ON THE CREST (Zero-Gamma Tipping Point)';
                ballColor = '#38bdf8';
            } else if (isInValley) {
                ballState = 'COILED IN +GAMMA VALLEY (Shock Absorber Active)';
                ballColor = '#00F5A0';
            } else {
                ballState = 'ROLLING DOWNHILL IN -GAMMA CHASM (Momentum Accelerator)';
                ballColor = '#FF2E55';
            }

            const statusBadge = document.getElementById('gammaTerrainStatusBadge');
            if (statusBadge) {
                const pillCol = isAtCrest ? 'text-sky-300' : (isInValley ? 'text-mint' : 'text-cherry');
                statusBadge.innerHTML = `<span class="${pillCol}">⚪ Spot: $${spot.toFixed(2)}</span>`;
            }

            const traceNeg = {
                x: denseK,
                y: negTerrain,
                mode: 'lines',
                name: 'Negative Gamma Chasm (Vol Accelerator)',
                line: { color: '#FF2E55', width: 2.5, shape: 'spline' },
                fill: 'tozeroy',
                fillcolor: 'rgba(255, 46, 85, 0.20)',
                hoverinfo: 'skip'
            };

            const tracePos = {
                x: denseK,
                y: posTerrain,
                mode: 'lines',
                name: 'Positive Gamma Valley (Shock Absorber)',
                line: { color: '#00F5A0', width: 2.5, shape: 'spline' },
                fill: 'tozeroy',
                fillcolor: 'rgba(0, 245, 160, 0.20)',
                hoverinfo: 'skip'
            };

            const traceStrikes = {
                x: rawK,
                y: rawNet,
                mode: 'markers',
                name: 'Strike Net GEX ($M)',
                marker: { size: 4.5, color: 'rgba(255, 255, 255, 0.35)' },
                hovertemplate: '<b>Strike:</b> $%{x:.1f}<br><b>Net GEX:</b> $%{y:+.1f}M<extra></extra>'
            };

            const traceBall = {
                x: [spot],
                y: [spotGex],
                mode: 'markers',
                name: `Price Ball ($${spot.toFixed(2)})`,
                marker: {
                    size: 16,
                    color: ballColor,
                    line: { width: 3, color: '#ffffff' },
                    symbol: 'circle'
                },
                hovertemplate: `<b>⚪ Price Ball: $${spot.toFixed(2)}</b><br>Ground Elevation: ${spotGex >= 0 ? '+' : ''}${spotGex.toFixed(1)}M GEX<extra></extra>`
            };

            const shapes = [
                {
                    type: 'line',
                    x0: gridMin, x1: gridMax,
                    y0: 0, y1: 0,
                    line: { color: 'rgba(148, 163, 184, 0.35)', width: 1.5, dash: 'solid' }
                },
                {
                    type: 'line',
                    x0: spot, x1: spot,
                    y0: 0, y1: 1, yref: 'paper',
                    line: { color: '#ffffff', width: 1.5, dash: 'dash' }
                }
            ];

            // No bulky description box over the ball - clean canvas
            const annotations = [];

            // OTM Target Base Landmark (Pre-Positioned Landing Pad where OI is building)
            const traces = [traceNeg, tracePos, traceStrikes, traceBall];
            if (g.target_base_strike && Math.abs(g.target_base_strike - spot) / spot < 0.40) {
                const tbK = g.target_base_strike;
                const tbOI = g.target_base_oi || 0;
                const tbGex = interp(tbK, rawK, rawNet);

                shapes.push({
                    type: 'line',
                    x0: tbK, x1: tbK,
                    y0: 0, y1: 1, yref: 'paper',
                    line: { color: '#c084fc', width: 1.5, dash: 'dash' }
                });

                annotations.push({
                    x: tbK, y: 0.94, yref: 'paper',
                    text: `🎯 Target Base $${tbK.toFixed(0)} (${(tbOI/1000).toFixed(1)}k OI)`,
                    showarrow: false,
                    font: { color: '#c084fc', size: 9 },
                    xanchor: 'center',
                    bgcolor: 'rgba(15, 23, 42, 0.90)',
                    bordercolor: '#c084fc',
                    borderwidth: 1,
                    borderpad: 3
                });

                traces.push({
                    x: [tbK],
                    y: [tbGex],
                    mode: 'markers',
                    name: `OTM Target Base ($${tbK.toFixed(0)})`,
                    marker: { size: 13, color: '#c084fc', symbol: 'diamond', line: { width: 2, color: '#ffffff' } },
                    hovertemplate: `<b>🎯 OTM TARGET BASE: $${tbK.toFixed(1)}</b><br>Accumulating Call OI: ${tbOI.toLocaleString()}<br>Dealers bracing for impact / landing pad<extra></extra>`
                });
            }

            if (zg && Math.abs(zg - spot) / spot < 0.35) {
                shapes.push({
                    type: 'line',
                    x0: zg, x1: zg,
                    y0: 0, y1: 1, yref: 'paper',
                    line: { color: '#38bdf8', width: 2, dash: 'dashdot' }
                });
                annotations.push({
                    x: zg, y: 0.94, yref: 'paper',
                    text: `⚡ Zero Flip $${zg.toFixed(1)}`,
                    showarrow: false,
                    font: { color: '#38bdf8', size: 9 },
                    xanchor: 'center',
                    bgcolor: 'rgba(15, 23, 42, 0.90)',
                    bordercolor: '#38bdf8',
                    borderwidth: 1,
                    borderpad: 3
                });
            }

            if (cw && Math.abs(cw - spot) / spot < 0.35) {
                shapes.push({
                    type: 'line',
                    x0: cw, x1: cw,
                    y0: 0, y1: 1, yref: 'paper',
                    line: { color: '#00F5A0', width: 1.5, dash: 'dot' }
                });
                annotations.push({
                    x: cw, y: 0.94, yref: 'paper',
                    text: `Call Wall $${cw.toFixed(0)}`,
                    showarrow: false,
                    font: { color: '#00F5A0', size: 9 },
                    xanchor: 'center',
                    bgcolor: 'rgba(15, 23, 42, 0.90)',
                    bordercolor: '#00F5A0',
                    borderwidth: 1,
                    borderpad: 3
                });
            }

            if (pw && Math.abs(pw - spot) / spot < 0.35) {
                shapes.push({
                    type: 'line',
                    x0: pw, x1: pw,
                    y0: 0, y1: 1, yref: 'paper',
                    line: { color: '#FF2E55', width: 1.5, dash: 'dot' }
                });
                annotations.push({
                    x: pw, y: 0.08, yref: 'paper',
                    text: `Put Wall $${pw.toFixed(0)}`,
                    showarrow: false,
                    font: { color: '#FF2E55', size: 9 },
                    xanchor: 'center',
                    bgcolor: 'rgba(15, 23, 42, 0.90)',
                    bordercolor: '#FF2E55',
                    borderwidth: 1,
                    borderpad: 3
                });
            }

            // Directional vectors (offset cleanly away from the 0 line)
            const posIndices = denseNet.map((v, idx) => v > 0 ? idx : -1).filter(idx => idx !== -1);
            if (posIndices.length > 20) {
                const mid = posIndices[Math.floor(posIndices.length / 2)];
                const yPos = Math.max(denseNet[mid] * 0.5, 40);
                annotations.push({
                    x: denseK[mid],
                    y: yPos,
                    text: "◀ <b>'PUSH IT BACK'</b> ▶<br><span style='font-size:8.5px;'>Dealers buy dips & sell rips<br>(Mean-reverting bowl)</span>",
                    showarrow: false,
                    bgcolor: "rgba(0, 245, 160, 0.12)",
                    bordercolor: "rgba(0, 245, 160, 0.4)",
                    borderwidth: 1,
                    font: { color: "#00F5A0", size: 9 }
                });
            }

            const negIndices = denseNet.map((v, idx) => v < 0 ? idx : -1).filter(idx => idx !== -1);
            if (negIndices.length > 20) {
                const midNeg = negIndices[Math.floor(negIndices.length / 2)];
                const yNeg = Math.min(denseNet[midNeg] * 0.4, -120);
                annotations.push({
                    x: denseK[midNeg],
                    y: yNeg,
                    text: "▼ <b>'PUSH IT'</b> ▼<br><span style='font-size:8.5px;'>Dealers sell with momentum<br>(Fast & abrupt cascade)</span>",
                    showarrow: false,
                    bgcolor: "rgba(255, 46, 85, 0.12)",
                    bordercolor: "rgba(255, 46, 85, 0.4)",
                    borderwidth: 1,
                    font: { color: "#FF2E55", size: 9 }
                });
            }

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 45, r: 25, t: 40, b: 35 },
                xaxis: {
                    title: { text: 'Underlying Strike Price ($)', font: { color: '#94a3b8', size: 11 } },
                    gridcolor: 'rgba(255, 255, 255, 0.04)',
                    zerolinecolor: 'rgba(255, 255, 255, 0.1)',
                    tickfont: { color: '#94a3b8', size: 10 }
                },
                yaxis: {
                    title: { text: 'Ground Elevation: Net GEX ($M / 1% move)', font: { color: '#94a3b8', size: 11 } },
                    gridcolor: 'rgba(255, 255, 255, 0.04)',
                    zerolinecolor: 'rgba(255, 255, 255, 0.15)',
                    tickfont: { color: '#94a3b8', size: 10 }
                },
                shapes: shapes,
                annotations: annotations,
                showlegend: true,
                legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: 1.14, font: { color: '#94a3b8', size: 10 } }
            };

            if (relStrikes.length > 0) {
                const span = kMax - kMin;
                const pad = Math.max(span * 0.08, spot * 0.015);
                layout.xaxis.range = [kMin - pad, kMax + pad];
            }

            Plotly.newPlot('gammaTerrainChart', traces, layout, { responsive: true, displayModeBar: false });

            // Populate Emerging OTM Target Base Banner
            const tbCard = document.getElementById('gammaTargetBaseCard');
            const tbText = document.getElementById('gammaTargetBaseText');
            if (tbCard && tbText) {
                if (g.target_base_strike) {
                    tbCard.classList.remove('hidden');
                    tbCard.classList.add('flex');
                    tbText.innerHTML = `
                        <b class="text-purple-400 font-bold">PRE-POSITIONED TARGET BASE DETECTED: $${g.target_base_strike} (+${g.target_base_dist_pct}% OTM • ${(g.target_base_oi || 0).toLocaleString()} Call OI)</b><br>
                        <span class="text-slate-300">While spot is navigating near-term negative gamma volatility, dealers and smart money are quietly building a base in <b>OTM Call Open Interest at $${g.target_base_strike}</b>. An increasing positive gamma shelf further out signals that institutions are bracing for impact and pre-positioning inventory—creating a magnetic landing pad where price is anticipated to target.</span>
                    `;
                } else {
                    tbCard.classList.add('hidden');
                    tbCard.classList.remove('flex');
                }
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
    
        // ================= TRADE LEDGER & AUDITOR JAVASCRIPT =================
        let LEDGER_DATA = null;

        function getFutureDateStr(daysAhead) {
            const d = new Date();
            d.setDate(d.getDate() + daysAhead);
            return d.toISOString().split('T')[0];
        }

        function switchLedgerSub(sub) {
            document.getElementById('ledgerSubActive').classList.add('hidden');
            document.getElementById('ledgerSubManual').classList.add('hidden');
            document.getElementById('ledgerSubClosed').classList.add('hidden');

            const btnA = document.getElementById('btnSubLedgerActive');
            const btnM = document.getElementById('btnSubLedgerManual');
            const btnC = document.getElementById('btnSubLedgerClosed');

            btnA.className = 'px-3.5 py-1.5 border-b-2 border-transparent text-slate-400 hover:text-white transition flex items-center gap-1.5';
            btnM.className = 'px-3.5 py-1.5 border-b-2 border-transparent text-slate-400 hover:text-white transition';
            btnC.className = 'px-3.5 py-1.5 border-b-2 border-transparent text-slate-400 hover:text-white transition flex items-center gap-1.5';

            if (sub === 'active') {
                document.getElementById('ledgerSubActive').classList.remove('hidden');
                btnA.className = 'px-3.5 py-1.5 border-b-2 border-mint text-white font-bold transition flex items-center gap-1.5';
            } else if (sub === 'manual') {
                document.getElementById('ledgerSubManual').classList.remove('hidden');
                btnM.className = 'px-3.5 py-1.5 border-b-2 border-mint text-white font-bold transition';
            } else if (sub === 'closed') {
                document.getElementById('ledgerSubClosed').classList.remove('hidden');
                btnC.className = 'px-3.5 py-1.5 border-b-2 border-mint text-white font-bold transition flex items-center gap-1.5';
            }
        }

        const LEDGER_STORAGE_KEY = 'rotato_trade_ledger_v1';

        function readLocalLedger() {
            try {
                const raw = localStorage.getItem(LEDGER_STORAGE_KEY);
                const parsed = raw ? JSON.parse(raw) : [];
                return Array.isArray(parsed) ? parsed : [];
            } catch (e) {
                return [];
            }
        }

        function writeLocalLedger(trades) {
            try {
                const clean = (trades || [])
                    .filter(t => t && t.id)
                    .map(t => {
                        const copy = Object.assign({}, t);
                        delete copy.mtm;
                        return copy;
                    });
                localStorage.setItem(LEDGER_STORAGE_KEY, JSON.stringify(clean));
            } catch (e) {
                console.warn("Unable to save local ledger backup:", e);
            }
        }

        function upsertLocalTrade(trd) {
            if (!trd || !trd.id) return;
            const list = readLocalLedger().filter(t => t && t.id !== trd.id);
            const copy = Object.assign({}, trd);
            delete copy.mtm;
            list.unshift(copy);
            writeLocalLedger(list);
        }

        function removeLocalTrade(id) {
            if (!id) return;
            const list = readLocalLedger().filter(t => t && t.id !== id);
            writeLocalLedger(list);
        }

        async function loadLedger() {
            try {
                const res = await fetch('/api/ledger');
                let data = await res.json();
                if (data.status !== 'success') return;

                // Reconcile ephemeral serverless /tmp storage with persistent browser localStorage backup
                const localTrades = readLocalLedger();
                if (localTrades.length > 0) {
                    const serverTrades = [...(data.active_trades || []), ...(data.closed_trades || [])];
                    const serverById = {};
                    serverTrades.forEach(t => {
                        if (t && t.id) serverById[t.id] = t;
                    });
                    const toRestore = localTrades.filter(lt => {
                        if (!lt || !lt.id) return false;
                        const st = serverById[lt.id];
                        if (!st) return true;
                        return Number(lt.updated_at || 0) > Number(st.updated_at || 0);
                    });
                    if (toRestore.length > 0) {
                        const rRes = await fetch('/api/ledger/restore', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({trades: toRestore})
                        });
                        const rData = await rRes.json();
                        if (rData.status === 'success') {
                            data = rData;
                        }
                    }
                }

                writeLocalLedger([...(data.active_trades || []), ...(data.closed_trades || [])]);
                LEDGER_DATA = data;

                const stats = data.stats || {};
                const active = data.active_trades || [];
                const closed = data.closed_trades || [];

                // 1. Update KPI ribbon
                document.getElementById('kpiLedgerTotal').innerText = stats.total_trades || 0;
                document.getElementById('kpiLedgerActive').innerText = stats.active_trades_count || 0;
                
                const unreal = stats.open_unrealized_pnl || 0;
                const signU = unreal >= 0 ? '+' : '';
                const kpiU = document.getElementById('kpiLedgerUnreal');
                kpiU.innerText = signU + '$' + unreal.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                kpiU.className = 'text-xl font-black tabular-num mt-1 ' + (unreal >= 0 ? 'text-mint' : 'text-cherry');
                
                const real = stats.realized_pnl || 0;
                const signR = real >= 0 ? '+' : '';
                const kpiR = document.getElementById('kpiLedgerReal');
                kpiR.innerText = signR + '$' + real.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                kpiR.className = 'text-xl font-black tabular-num mt-1 ' + (real >= 0 ? 'text-mint' : 'text-cherry');

                document.getElementById('kpiLedgerWinRate').innerText = (stats.win_rate_closed || 0).toFixed(1) + '%';
                document.getElementById('kpiLedgerClosedCount').innerText = 'N=' + (stats.closed_trades_count || 0) + ' Closed Trades';

                // Badges
                const badge = document.getElementById('ledgerTabBadge');
                if (badge) {
                    badge.innerText = active.length;
                    if (active.length > 0) badge.classList.remove('hidden');
                    else badge.classList.add('hidden');
                }
                const subActiveBadge = document.getElementById('subBadgeActive');
                if (subActiveBadge) subActiveBadge.innerText = active.length;
                const subClosedBadge = document.getElementById('subBadgeClosed');
                if (subClosedBadge) subClosedBadge.innerText = closed.length;

                // 2. Render Active Positions
                renderActiveLedgerCards(active);

                // 3. Render Closed History
                renderClosedLedgerTable(closed);

            } catch (err) {
                console.error("Error loading ledger:", err);
            }
        }

        function renderActiveLedgerCards(active) {
            const container = document.getElementById('ledgerActiveList');
            if (!container) return;
            if (active.length === 0) {
                container.innerHTML = `
                    <div class="col-span-full p-8 text-center text-slate-400 bg-white/[0.01] rounded-xl border border-white/5 text-xs">
                        <div class="text-slate-300 font-bold mb-1">No Active Positions Tracked</div>
                        <p class="text-[11px] text-slate-500 mb-3">Commit an algorithmic trade idea from the <b>'Automated Trade Ideas'</b> tab, or use the <b>'Manual / AI Prompt Logger'</b> above.</p>
                        <button onclick="switchTab('tab2')" class="px-3.5 py-1.5 rounded-lg bg-mint/15 text-mint border border-mint/30 font-bold text-xs hover:bg-mint/25 transition">Explore Trade Setups</button>
                    </div>
                `;
                return;
            }

            let html = '';
            active.forEach(trd => {
                const mtm = trd.mtm || {};
                const isSpread = trd.asset_type === 'SPREAD';
                const isOpt = trd.asset_type === 'OPTION';
                const dir = (trd.direction || 'LONG').toUpperCase();
                const isLong = dir.includes('LONG') || dir.includes('DIP');
                const borderCol = isSpread ? '#A855F7' : (isLong ? '#00F5A0' : '#FF2E55');
                const snap = trd.ai_snapshot || {};

                const unrealPnl = mtm.unrealized_pnl || 0;
                const unrealPct = mtm.unrealized_pct || 0;
                const pnlSign = unrealPnl >= 0 ? '+' : '';
                const pnlCol = unrealPnl >= 0 ? 'text-mint' : 'text-cherry';
                const pnlBg = unrealPnl >= 0 ? 'bg-mint/15 text-mint border-mint/30' : 'bg-cherry/15 text-cherry border-cherry/30';

                const instBadge = isSpread ? `
                    <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                        SPREAD • ${trd.spread_type || 'VERTICAL'}
                    </span>
                ` : (isOpt ? `
                    <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-500/15 text-cyan-400 border border-cyan-500/30">
                        ${trd.strike || 0} ${trd.contract_type || 'CALL'} (${trd.expiration || ''})
                    </span>
                ` : `
                    <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/15 text-amber-400 border border-amber-500/30">
                        STOCK / EQUITY
                    </span>
                `);

                let alertHtml = '';
                if (mtm.trigger_alert) {
                    const isSucc = mtm.alert_type === 'success';
                    const isErr = mtm.alert_type === 'error';
                    const aCol = isSucc ? 'text-mint border-mint/30 bg-mint/10' : (isErr ? 'text-cherry border-cherry/30 bg-cherry/10' : 'text-amber-400 border-amber-400/30 bg-amber-400/10');
                    alertHtml = `<div class="p-2 rounded-lg border text-xs font-semibold mb-2.5 ${aCol}">${mtm.trigger_alert}</div>`;
                }

                const progressPct = Math.max(0, Math.min(100, mtm.progress_pct || 0));

                html += `
                    <div class="lux-card p-4 border-l-4 flex flex-col justify-between" style="border-left-color: ${borderCol}">
                        <div>
                            <!-- Header -->
                            <div class="flex flex-wrap justify-between items-center gap-2 pb-2.5 mb-2.5 border-b border-white/5">
                                <div class="flex items-center gap-2">
                                    <span class="text-xl font-black text-white">${trd.ticker}</span>
                                    <span class="px-2 py-0.5 rounded text-[10px] font-bold ${isLong ? 'bg-mint/15 text-mint border border-mint/30' : 'bg-cherry/15 text-cherry border border-cherry/30'}">${dir}</span>
                                    ${instBadge}
                                </div>
                                <div class="flex items-center gap-2">
                                    <span class="text-[10px] text-slate-500">Logged: ${trd.logged_at || ''}</span>
                                    <span class="px-2.5 py-0.5 rounded text-xs font-extrabold border ${pnlBg}">
                                        ${pnlSign}$${unrealPnl.toFixed(2)} (${pnlSign}${unrealPct.toFixed(2)}%)
                                    </span>
                                </div>
                            </div>

                            ${alertHtml}

                            <!-- Defined Spread Legs Bar (if spread) -->
                            ${isSpread ? `
                            <div class="flex flex-wrap items-center justify-between gap-2 p-2 mb-2.5 rounded-lg bg-purple-950/20 border border-purple-500/20 text-xs">
                                <div class="flex items-center gap-2 flex-wrap">
                                    <span class="text-purple-300 font-bold">Legs:</span>
                                    <span class="text-rose-400 font-mono font-bold">Short $${trd.short_strike || 'N/A'}</span>
                                    <span class="text-slate-500">/</span>
                                    <span class="text-mint font-mono font-bold">Long $${trd.long_strike || 'N/A'}</span>
                                    <span class="text-slate-400 text-[10px]">(${trd.expiration || trd.short_expiry || ''})</span>
                                </div>
                                ${mtm.short_strike_buffer_pct !== undefined ? `
                                <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold ${mtm.short_strike_buffer_pct >= 0 ? 'bg-mint/15 text-mint' : 'bg-cherry/15 text-cherry'}">
                                    Short Barrier Buffer: ${mtm.short_strike_buffer_pct >= 0 ? '+' : ''}${mtm.short_strike_buffer_pct.toFixed(2)}%
                                </span>
                                ` : ''}
                            </div>
                            ` : ''}

                            <!-- 4-Metric Grid -->
                            <div class="grid grid-cols-4 gap-2 mb-3">
                                <div class="p-2 rounded-lg bg-white/[0.02] border border-white/5">
                                    <div class="text-[9px] uppercase font-bold text-slate-400">${isSpread ? 'Net Premium' : (isOpt ? 'Premium Paid' : 'Entry Fill')}</div>
                                    <div class="text-sm font-extrabold text-white mt-0.5">$${(trd.entry_price || 0).toFixed(2)}</div>
                                    <div class="text-[9px] text-slate-500">${trd.size} ${isSpread ? 'Spread(s)' : (isOpt ? 'Contracts' : 'Shares')}</div>
                                </div>
                                <div class="p-2 rounded-lg bg-white/[0.02] border border-white/5">
                                    <div class="text-[9px] uppercase font-bold text-slate-400">Live Spot</div>
                                    <div class="text-sm font-extrabold text-sky-400 mt-0.5">$${(mtm.live_spot || 0).toFixed(2)}</div>
                                    <div class="text-[9px] text-slate-500">${isSpread ? (mtm.est_spread_px != null ? `Live Spread: $${Number(mtm.est_spread_px).toFixed(2)}` : 'No live quotes') : (isOpt ? (mtm.est_option_px != null ? `Live Option: $${Number(mtm.est_option_px).toFixed(2)}` : 'No live quotes') : 'Underlying')}</div>
                                </div>
                                <div class="p-2 rounded-lg ${isSpread ? 'bg-purple-950/30 border border-purple-500/30' : 'bg-cherry/10 border border-cherry/25'}">
                                    <div class="text-[9px] uppercase font-bold ${isSpread ? 'text-purple-300' : 'text-cherry'}">${isSpread ? 'Collateral Locked' : 'Stop Loss'}</div>
                                    <div class="text-sm font-extrabold ${isSpread ? 'text-white' : 'text-cherry'} mt-0.5">${isSpread ? `$${(trd.max_risk || trd.position_value || 0).toFixed(2)}` : `$${(trd.stop_loss || 0).toFixed(2)}`}</div>
                                    <div class="text-[9px] ${isSpread ? 'text-purple-300' : 'text-rose-300'} font-mono">${isSpread ? '100% Defined Risk' : `Buffer: ${(mtm.dist_to_stop_pct || 0).toFixed(2)}%`}</div>
                                </div>
                                <div class="p-2 rounded-lg bg-mint/10 border border-mint/25">
                                    <div class="text-[9px] uppercase font-bold text-mint">${isSpread ? 'Max Profit (ROC)' : 'Target 1'}</div>
                                    <div class="text-sm font-extrabold text-mint mt-0.5">${isSpread ? `$${(trd.max_profit || 0).toFixed(2)}` : `$${(trd.target_1 || 0).toFixed(2)}`}</div>
                                    <div class="text-[9px] text-emerald-300 font-mono">${isSpread ? `ROC: ${pnlSign}${unrealPct.toFixed(1)}%` : `Dist: ${(mtm.dist_to_target1_pct || 0).toFixed(2)}%`}</div>
                                </div>
                            </div>

                            <!-- Target Progress Meter -->
                            <div class="mb-3">
                                <div class="flex justify-between text-[10px] text-slate-400 mb-1">
                                    <span>Target 1 Progress</span>
                                    <span class="font-bold text-white font-mono">${progressPct.toFixed(1)}%</span>
                                </div>
                                <div class="w-full bg-white/5 h-2 rounded-full overflow-hidden border border-white/10">
                                    <div class="h-full bg-mint rounded-full transition-all duration-500" style="width: ${progressPct}%"></div>
                                </div>
                            </div>

                            <!-- Frozen Baseline Dropdown -->
                            <details class="mb-3 p-2.5 rounded-lg bg-[#08090e] border border-white/5 text-[11px] text-slate-300 group">
                                <summary class="cursor-pointer font-bold text-slate-400 hover:text-white flex items-center justify-between">
                                    <span>Frozen Position Baseline at Entry (WR, Odds, GEX)</span>
                                    <span class="text-xs text-slate-500">▾</span>
                                </summary>
                                <div class="mt-2 pt-2 border-t border-white/5 grid grid-cols-2 sm:grid-cols-3 gap-2">
                                    <div>
                                        <span class="text-slate-500">20yr Win Rate:</span>
                                        <b class="text-mint font-mono">${snap.win_rate || 'N/A'} (N=${snap.sample_size || 0})</b>
                                    </div>
                                    <div>
                                        <span class="text-slate-500">Entry Confluence:</span>
                                        <b class="text-white">${snap.confluence_score || 'N/A'} (${snap.bias_label || ''})</b>
                                    </div>
                                    <div>
                                        <span class="text-slate-500">Call Wall at Entry:</span>
                                        <b class="text-mint font-mono">$${snap.call_wall || 'N/A'}</b>
                                    </div>
                                    <div>
                                        <span class="text-slate-500">Put Wall at Entry:</span>
                                        <b class="text-cherry font-mono">$${snap.put_wall || 'N/A'}</b>
                                    </div>
                                    <div class="col-span-2">
                                        <span class="text-slate-500">Zero Gamma / Net GEX:</span>
                                        <b class="text-sky-400 font-mono">${snap.zero_gamma || 'N/A'} (${snap.net_gex_b || ''})</b>
                                    </div>
                                    ${snap.benchmark_symbol && snap.benchmark_put_wall ? `
                                        <div class="col-span-2 pt-1 border-t border-white/5">
                                            <span class="text-slate-500">Benchmark Floor (${snap.benchmark_symbol}):</span>
                                            <b class="text-cherry font-mono">$${snap.benchmark_put_wall}</b>
                                            ${snap.benchmark_zero_gamma ? `<span class="text-slate-600">|</span> <span class="text-slate-500">Zero Flip:</span> <b class="text-sky-300 font-mono">$${snap.benchmark_zero_gamma}</b>` : ''}
                                        </div>
                                    ` : ''}
                                </div>
                                ${snap.thesis ? `
                                    <div class="mt-2 pt-2 border-t border-white/5 text-[10.5px] text-slate-300 leading-snug">
                                        <b class="text-white">Original Thesis:</b> ${snap.thesis}
                                    </div>
                                ` : ''}
                            </details>
                        </div>

                        <!-- Action Toolbar -->
                        <div class="pt-2.5 border-t border-white/5 flex items-center justify-between gap-2">
                            <button onclick="toggleTradeCloseDrawer('${trd.id}')" class="px-3 py-1.5 rounded-lg bg-mint/15 hover:bg-mint/25 text-mint border border-mint/30 text-xs font-bold transition flex items-center gap-1">
                                <span>Close Position</span>
                            </button>
                            <button onclick="toggleTradeAdjustDrawer('${trd.id}')" class="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10 text-xs font-semibold transition flex items-center gap-1">
                                <span>Adjust</span>
                            </button>
                            <button onclick="deleteLedgerTrade('${trd.id}')" class="px-2.5 py-1.5 rounded-lg bg-cherry/10 hover:bg-cherry/20 text-cherry border border-cherry/25 text-xs font-semibold transition">
                                <span>Delete</span>
                            </button>
                        </div>

                        <!-- Close Position Inline Drawer -->
                        <div id="closeDrawer-${trd.id}" class="hidden mt-3 p-3 rounded-xl bg-[#090b10] border border-mint/30 space-y-2 text-xs">
                            <div class="font-bold text-white flex justify-between">
                                <span>Close Position / Realize P&L</span>
                                <button onclick="toggleTradeCloseDrawer('${trd.id}')" class="text-slate-500 hover:text-white">&times;</button>
                            </div>
                            <div class="grid grid-cols-2 gap-2">
                                <div>
                                    <label class="text-[9.5px] uppercase font-bold text-slate-400">Exit Price ($)</label>
                                    <input id="closeExitPx-${trd.id}" type="number" step="0.01" value="${(isSpread ? (mtm.est_spread_px ?? trd.entry_price) : (isOpt ? (mtm.est_option_px ?? trd.entry_price) : mtm.live_spot)).toFixed(2)}" class="w-full mt-0.5 bg-[#161722] border border-white/10 rounded px-2 py-1 text-white text-xs">
                                </div>
                                <div>
                                    <label class="text-[9.5px] uppercase font-bold text-slate-400">Outcome</label>
                                    <select id="closeOutcome-${trd.id}" class="w-full mt-0.5 bg-[#161722] border border-white/10 rounded px-2 py-1 text-white text-xs">
                                        <option value="TARGET_1_HIT">Target 1 Hit</option>
                                        <option value="TARGET_2_HIT">Target 2 Runner</option>
                                        <option value="STOPPED_OUT">Stopped Out</option>
                                        <option value="CLOSED_MANUALLY">Closed Manually</option>
                                    </select>
                                </div>
                            </div>
                            <input id="closeNotes-${trd.id}" type="text" placeholder="Optional exit notes..." class="w-full bg-[#161722] border border-white/10 rounded px-2 py-1 text-white text-xs">
                            <button onclick="submitCloseTrade('${trd.id}')" class="w-full py-1.5 rounded-lg bg-mint text-black font-bold text-xs hover:bg-mint/80 transition">Confirm Realize P&L</button>
                        </div>

                        <!-- Adjust Inline Drawer -->
                        <div id="adjustDrawer-${trd.id}"
                             data-asset-type="${trd.asset_type || 'STOCK'}"
                             data-spread-type="${trd.spread_type || ''}"
                             data-spread-width="${trd.spread_width || (trd.short_strike && trd.long_strike ? Math.abs(trd.short_strike - trd.long_strike) : 5)}"
                             class="hidden mt-3 p-3 rounded-xl bg-[#090b10] border border-white/20 space-y-2 text-xs">
                            <div class="font-bold text-white flex justify-between">
                                <span>Adjust Position & Levels</span>
                                <button onclick="toggleTradeAdjustDrawer('${trd.id}')" class="text-slate-500 hover:text-white">&times;</button>
                            </div>
                            <div class="grid grid-cols-2 gap-2">
                                <div>
                                    <label class="text-[9.5px] uppercase font-bold text-slate-300">${isSpread ? ((trd.spread_type || '').includes('CREDIT') ? 'Net Credit ($)' : 'Net Debit ($)') : (isOpt ? 'Option Fill ($)' : 'Entry Fill ($)')}</label>
                                    <input id="adjustEntry-${trd.id}" type="number" step="0.01" min="0" value="${(trd.entry_price || 0).toFixed(2)}" oninput="updateAdjustPreview('${trd.id}')" class="w-full mt-0.5 bg-[#161722] border border-white/15 rounded px-2 py-1 text-white text-xs">
                                </div>
                                <div>
                                    <label class="text-[9.5px] uppercase font-bold text-slate-300">${isSpread ? 'Spreads (Qty)' : (isOpt ? 'Contracts' : 'Shares')}</label>
                                    <input id="adjustSize-${trd.id}" type="number" step="${isSpread || isOpt ? '1' : '0.01'}" min="0.01" value="${trd.size || 1}" oninput="updateAdjustPreview('${trd.id}')" class="w-full mt-0.5 bg-[#161722] border border-white/15 rounded px-2 py-1 text-white text-xs">
                                </div>
                            </div>
                            ${isSpread ? `
                            <div id="adjustSpreadPreview-${trd.id}" class="text-[10.5px] text-purple-300 font-mono bg-purple-950/20 border border-purple-500/20 rounded px-2 py-1">
                                Collateral: $${(trd.max_risk || trd.position_value || 0).toFixed(2)} | Max Profit: $${(trd.max_profit || 0).toFixed(2)} (${((trd.max_risk || 0) > 0 ? ((trd.max_profit || 0) / trd.max_risk * 100).toFixed(1) : '0.0')}% ROC)
                            </div>
                            ` : ''}
                            <div class="grid grid-cols-2 gap-2">
                                <div>
                                    <label class="text-[9.5px] uppercase font-bold text-cherry">New Stop ($)</label>
                                    <input id="adjustStop-${trd.id}" type="number" step="0.01" value="${trd.stop_loss || 0}" class="w-full mt-0.5 bg-[#161722] border border-cherry/30 rounded px-2 py-1 text-cherry text-xs">
                                </div>
                                <div>
                                    <label class="text-[9.5px] uppercase font-bold text-mint">New Target 1 ($)</label>
                                    <input id="adjustTp1-${trd.id}" type="number" step="0.01" value="${trd.target_1 || 0}" class="w-full mt-0.5 bg-[#161722] border border-mint/30 rounded px-2 py-1 text-mint text-xs">
                                </div>
                            </div>
                            <button onclick="submitAdjustTrade('${trd.id}')" class="w-full py-1.5 rounded-lg bg-white/10 text-white font-bold text-xs hover:bg-white/20 transition">Save Adjustments</button>
                        </div>
                    </div>
                `;
            });

            container.innerHTML = html;
        }

        function renderClosedLedgerTable(closed) {
            const tbody = document.getElementById('ledgerClosedTbody');
            if (!tbody) return;
            if (closed.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" class="p-4 text-center text-slate-500">No closed trades recorded yet.</td></tr>';
                return;
            }

            let html = '';
            closed.forEach(ct => {
                const realPnl = ct.realized_pnl || 0;
                const realPct = ct.realized_pct || 0;
                const sign = realPnl >= 0 ? '+' : '';
                const pnlCol = realPnl >= 0 ? 'text-mint' : 'text-cherry';

                html += `
                    <tr class="hover:bg-white/[0.02] transition">
                        <td class="p-2.5 font-bold text-white">${ct.ticker}</td>
                        <td class="p-2.5 text-slate-300 font-mono text-[11px]">${ct.asset_type === 'SPREAD' ? `SPREAD: ${ct.spread_type || 'VERT'} (Short $${ct.short_strike}/Long $${ct.long_strike})` : `${ct.asset_type || 'STOCK'} ${ct.contract_type ? ct.contract_type : ''}`}</td>
                        <td class="p-2.5 text-slate-400 text-[11px]">${ct.logged_at || ''}</td>
                        <td class="p-2.5 text-slate-400 text-[11px]">${ct.closed_at || ''}</td>
                        <td class="p-2.5 font-mono text-slate-300">$${(ct.entry_price || 0).toFixed(2)}</td>
                        <td class="p-2.5 font-mono text-slate-300">$${(ct.exit_price || 0).toFixed(2)}</td>
                        <td class="p-2.5 font-bold font-mono ${pnlCol}">${sign}$${realPnl.toFixed(2)}</td>
                        <td class="p-2.5 font-bold font-mono ${pnlCol}">${sign}${realPct.toFixed(2)}%</td>
                        <td class="p-2.5"><span class="px-2 py-0.5 rounded text-[9.5px] font-bold ${realPnl >= 0 ? 'bg-mint/15 text-mint' : 'bg-cherry/15 text-cherry'}">${ct.status || 'CLOSED'}</span></td>
                        <td class="p-2.5 text-slate-400 font-mono">${(ct.ai_snapshot && ct.ai_snapshot.win_rate) ? ct.ai_snapshot.win_rate : 'N/A'}</td>
                    </tr>
                `;
            });
            tbody.innerHTML = html;
        }

        async function runLedgerAudit() {}

        function toggleCardLogger(idx) {
            const el = document.getElementById('cardLogger-' + idx);
            if (el) el.classList.toggle('hidden');
        }

        function toggleCardLoggerFields(idx) {
            const type = document.getElementById('cardInstType-' + idx).value;
            const optRow = document.getElementById('cardOptFields-' + idx);
            const spreadRow = document.getElementById('cardSpreadFields-' + idx);
            const sizeLabel = document.getElementById('cardSizeLabel-' + idx);
            const entryLabel = document.getElementById('cardEntryLabel-' + idx);
            const entryContainer = document.getElementById('cardEntryContainer-' + idx);

            if (type === 'SPREAD') {
                if (optRow) optRow.classList.add('hidden');
                if (spreadRow) spreadRow.classList.remove('hidden');
                if (entryContainer) entryContainer.classList.add('hidden');
                if (sizeLabel) sizeLabel.innerText = 'Spreads';
            } else if (type === 'OPTION') {
                if (optRow) optRow.classList.remove('hidden');
                if (spreadRow) spreadRow.classList.add('hidden');
                if (entryContainer) entryContainer.classList.remove('hidden');
                if (sizeLabel) sizeLabel.innerText = 'Contracts';
                if (entryLabel) entryLabel.innerText = 'Premium ($)';
            } else {
                if (optRow) optRow.classList.add('hidden');
                if (spreadRow) spreadRow.classList.add('hidden');
                if (entryContainer) entryContainer.classList.remove('hidden');
                if (sizeLabel) sizeLabel.innerText = 'Shares';
                if (entryLabel) entryLabel.innerText = 'Entry Fill ($)';
            }
        }

        function prefillAndOpenSpreadLogger(idx) {
            const drawer = document.getElementById('cardLogger-' + idx);
            if (drawer && drawer.classList.contains('hidden')) {
                drawer.classList.remove('hidden');
            }
            const instSelect = document.getElementById('cardInstType-' + idx);
            if (instSelect) {
                instSelect.value = 'SPREAD';
                toggleCardLoggerFields(idx);
            }
            if (AI_DATA && AI_DATA.trade_cards && AI_DATA.trade_cards[idx]) {
                const t = AI_DATA.trade_cards[idx];
                const sp = t.options_spread;
                if (sp) {
                    const stEl = document.getElementById('cardSpreadType-' + idx);
                    if (stEl) stEl.value = sp.spread_type;
                    const skEl = document.getElementById('cardShortStrike-' + idx);
                    if (skEl) skEl.value = sp.short_strike;
                    const lkEl = document.getElementById('cardLongStrike-' + idx);
                    if (lkEl) lkEl.value = sp.long_strike;
                    const npEl = document.getElementById('cardNetPrem-' + idx);
                    if (npEl) npEl.value = sp.net_premium;
                    const expEl = document.getElementById('cardSpreadExpiry-' + idx);
                    if (expEl) expEl.value = sp.expiration;
                    const szEl = document.getElementById('cardSize-' + idx);
                    if (szEl) szEl.value = 1;
                }
            }
            drawer?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        async function commitCardTrade(idx) {
            if (!AI_DATA || !AI_DATA.trade_cards || !AI_DATA.trade_cards[idx]) return;
            const t = AI_DATA.trade_cards[idx];
            const btn = document.getElementById('btnCommitCard-' + idx);
            if (btn) btn.innerText = 'Committing...';

            const assetType = document.getElementById('cardInstType-' + idx).value;
            const size = parseFloat(document.getElementById('cardSize-' + idx).value) || 1;
            const stop = parseFloat(document.getElementById('cardStop-' + idx).value) || 0;
            const tp1 = parseFloat(document.getElementById('cardTp1-' + idx).value) || 0;
            const currSpot = parseFloat(String(t.current_price || t.entry || '100').replace(/[^0-9.]/g, '')) || 100;

            let entryPx = parseFloat(document.getElementById('cardEntryPrice-' + idx)?.value) || 0;
            let cType = null;
            let strike = null;
            let expiry = null;
            let spreadType = null;
            let shortK = null;
            let longK = null;
            let netPrem = null;
            let spreadWidth = null;
            let maxRisk = null;
            let maxProfit = null;

            if (assetType === 'SPREAD') {
                spreadType = document.getElementById('cardSpreadType-' + idx)?.value || 'BULL_PUT_CREDIT';
                shortK = parseFloat(document.getElementById('cardShortStrike-' + idx)?.value) || 0;
                longK = parseFloat(document.getElementById('cardLongStrike-' + idx)?.value) || 0;
                netPrem = parseFloat(document.getElementById('cardNetPrem-' + idx)?.value) || 1.35;
                expiry = document.getElementById('cardSpreadExpiry-' + idx)?.value || null;
                entryPx = netPrem;
                spreadWidth = Math.abs(shortK - longK) || 5.0;
                const isCredit = spreadType.includes('CREDIT');
                if (isCredit) {
                    maxRisk = Math.max(0.01, spreadWidth - netPrem) * 100 * size;
                    maxProfit = netPrem * 100 * size;
                } else {
                    maxRisk = netPrem * 100 * size;
                    maxProfit = Math.max(0.01, spreadWidth - netPrem) * 100 * size;
                }
            } else if (assetType === 'OPTION') {
                cType = document.getElementById('cardOptType-' + idx)?.value || 'CALL';
                strike = parseFloat(document.getElementById('cardStrike-' + idx)?.value) || null;
                expiry = document.getElementById('cardExpiry-' + idx)?.value || null;
            }

            const payload = {
                ticker: t.ticker,
                direction: t.direction,
                asset_type: assetType,
                entry_price: entryPx,
                size: size,
                stop_loss: stop,
                target_1: tp1,
                target_2: parseFloat(String(t.target2 || '0').replace(/[^0-9.]/g, '')) || 0,
                contract_type: cType,
                strike: strike,
                expiration: expiry,
                underlying_entry_spot: currSpot,
                spread_type: spreadType,
                short_strike: shortK,
                long_strike: longK,
                short_expiry: expiry,
                long_expiry: expiry,
                net_premium: netPrem,
                spread_width: spreadWidth,
                max_risk: maxRisk,
                max_profit: maxProfit,
                notes: (assetType === 'SPREAD' ? `Defined-Risk Spread (${spreadType}: Short $${shortK} / Long $${longK}). ` : 'Automated Setup. ') + (t.thesis || ''),
                ai_snapshot: {
                    win_rate: t.win_rate,
                    sample_size: t.sample_size,
                    median_alpha: t.median_alpha,
                    confluence_score: t.score,
                    bias_label: t.bias_label,
                    confirmations: t.confirmations,
                    thesis: t.thesis,
                    entry_price_spot: currSpot,
                    call_wall: t.call_wall,
                    put_wall: t.put_wall,
                    zero_gamma: t.zero_gamma,
                    net_gex_b: t.net_gex_b,
                    gamma_regime: t.gamma_regime,
                    benchmark_symbol: t.benchmark_symbol,
                    benchmark_put_wall: t.benchmark_put_wall,
                    benchmark_zero_gamma: t.benchmark_zero_gamma,
                    risk_reward: t.risk_reward
                }
            };

            try {
                const res = await fetch('/api/ledger/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.status === 'success') {
                    upsertLocalTrade(data.trade);
                    if (btn) btn.innerText = 'Committed to Ledger!';
                    setTimeout(() => {
                        toggleCardLogger(idx);
                        if (btn) btn.innerText = 'Confirm & Log to Ledger';
                    }, 1200);
                    loadLedger();
                } else {
                    alert('Error adding trade: ' + (data.message || 'unknown error'));
                    if (btn) btn.innerText = 'Confirm & Log to Ledger';
                }
            } catch (err) {
                alert('Failed to commit trade: ' + err.message);
                if (btn) btn.innerText = 'Confirm & Log to Ledger';
            }
        }

        function toggleManualInstrumentFields() {
            const val = document.getElementById('manAssetType').value;
            const optRow = document.getElementById('manOptionsRow');
            const stockSizeDiv = document.getElementById('manStockSizeDiv');
            const entryLabel = document.getElementById('manEntryLabel');
            if (val === 'OPTION') {
                if (optRow) optRow.classList.remove('hidden');
                if (stockSizeDiv) stockSizeDiv.classList.add('hidden');
                if (entryLabel) entryLabel.innerText = 'Option Premium ($)';
            } else {
                if (optRow) optRow.classList.add('hidden');
                if (stockSizeDiv) stockSizeDiv.classList.remove('hidden');
                if (entryLabel) entryLabel.innerText = 'Entry Fill Price ($)';
            }
        }

        let MAN_AUTO_FETCHED = null;

        async function autoFetchManualConfluence() {
            const ticker = (document.getElementById('manTicker').value || 'NVDA').toUpperCase().trim();
            const btn = document.getElementById('btnAutoFetchMan');
            const status = document.getElementById('manAutoFetchStatus');
            if (btn) btn.innerText = 'Fetching...';

            try {
                const res = await fetch('/api/confluence?scope=core');
                const list = await res.json();
                const match = (list || []).find(item => item.ticker === ticker);
                if (match) {
                    MAN_AUTO_FETCHED = match;
                    const edge = match.metrics ? match.metrics.edge_match : {};
                    const wr = edge ? edge.win_rate_20d : 'N/A';
                    if (status) {
                        status.classList.remove('hidden');
                        status.innerHTML = `<span class="text-mint font-bold">Profile Fetched:</span> Score <b>${match.score}</b> | 20yr Win Rate: <b>${wr}%</b> (N=${edge ? edge.sample_size : 0})`;
                    }
                    if (match.metrics && match.metrics.close) {
                        const pxInput = document.getElementById('manEntryPrice');
                        if (pxInput && (!pxInput.value || parseFloat(pxInput.value) <= 0)) {
                            pxInput.value = match.metrics.close.toFixed(2);
                        }
                    }
                } else {
                    if (status) {
                        status.classList.remove('hidden');
                        status.innerHTML = `<span class="text-amber-400">Ticker ${ticker} not in core cache, using discretionary settings.</span>`;
                    }
                }
                if (btn) btn.innerText = 'Auto-Fetch Current Confluence & 20-Year Odds for Ticker';
            } catch (err) {
                if (btn) btn.innerText = 'Auto-Fetch Current Confluence & 20-Year Odds for Ticker';
                console.error("Auto fetch error:", err);
            }
        }

        async function submitManualLedgerTrade() {
            const ticker = (document.getElementById('manTicker').value || 'NVDA').toUpperCase().trim();
            const direction = document.getElementById('manDirection').value;
            const assetType = document.getElementById('manAssetType').value;
            const entryPrice = parseFloat(document.getElementById('manEntryPrice').value) || 0;
            const stopLoss = parseFloat(document.getElementById('manStopLoss').value) || 0;
            const target1 = parseFloat(document.getElementById('manTarget1').value) || 0;
            const thesis = document.getElementById('manThesis').value || '';

            const isOpt = assetType === 'OPTION';
            let size = 10;
            let cType = null;
            let strike = null;
            let expiry = null;

            if (isOpt) {
                size = parseFloat(document.getElementById('manOptSize').value) || 2;
                cType = document.getElementById('manContractType').value;
                strike = parseFloat(document.getElementById('manStrike').value) || null;
                expiry = document.getElementById('manExpiration').value || getFutureDateStr(35);
            } else {
                size = parseFloat(document.getElementById('manStockSize').value) || 10;
            }

            const edge = (MAN_AUTO_FETCHED && MAN_AUTO_FETCHED.metrics) ? MAN_AUTO_FETCHED.metrics.edge_match : null;

            const payload = {
                ticker: ticker,
                direction: direction,
                asset_type: assetType,
                entry_price: entryPrice,
                size: size,
                stop_loss: stopLoss,
                target_1: target1,
                target_2: target1 * 1.05,
                contract_type: cType,
                strike: strike,
                expiration: expiry,
                underlying_entry_spot: entryPrice,
                notes: thesis,
                ai_snapshot: {
                    win_rate: edge ? (edge.win_rate_20d + '%') : 'Manual Setup',
                    sample_size: edge ? edge.sample_size : 0,
                    confluence_score: MAN_AUTO_FETCHED ? String(MAN_AUTO_FETCHED.score) : 'N/A',
                    bias_label: MAN_AUTO_FETCHED ? MAN_AUTO_FETCHED.bias_label : 'Desk Discretionary',
                    thesis: thesis,
                    entry_price_spot: entryPrice
                }
            };

            try {
                const res = await fetch('/api/ledger/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.status === 'success') {
                    upsertLocalTrade(data.trade);
                    alert(`${ticker} successfully committed to active ledger!`);
                    switchLedgerSub('active');
                    loadLedger();
                } else {
                    alert('Error: ' + (data.message || 'Unable to add trade'));
                }
            } catch (err) {
                alert('Failed to add trade: ' + err.message);
            }
        }

        function toggleTradeCloseDrawer(id) {
            const el = document.getElementById('closeDrawer-' + id);
            if (el) el.classList.toggle('hidden');
        }

        async function submitCloseTrade(id) {
            const exitPx = parseFloat(document.getElementById('closeExitPx-' + id).value) || 0;
            const outcome = document.getElementById('closeOutcome-' + id).value;
            const notes = document.getElementById('closeNotes-' + id).value;

            try {
                const res = await fetch('/api/ledger/close', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({id: id, exit_price: exitPx, outcome: outcome, notes: notes})
                });
                const data = await res.json();
                if (data.status === 'success') {
                    upsertLocalTrade(data.trade);
                    loadLedger();
                } else {
                    alert('Error closing trade: ' + (data.message || 'unknown error'));
                }
            } catch (err) {
                alert('Close request failed: ' + err.message);
            }
        }

        function toggleTradeAdjustDrawer(id) {
            const el = document.getElementById('adjustDrawer-' + id);
            if (el) el.classList.toggle('hidden');
        }

        function updateAdjustPreview(id) {
            const drawer = document.getElementById('adjustDrawer-' + id);
            const previewEl = document.getElementById('adjustSpreadPreview-' + id);
            if (!drawer || !previewEl) return;
            const spreadType = (drawer.dataset.spreadType || 'BULL_PUT_CREDIT').toUpperCase();
            const width = parseFloat(drawer.dataset.spreadWidth) || 5;
            const prem = Math.max(0, parseFloat(document.getElementById('adjustEntry-' + id).value) || 0);
            const qty = Math.max(0.01, parseFloat(document.getElementById('adjustSize-' + id).value) || 1);
            const isCredit = spreadType.includes('CREDIT');
            const maxProfit = isCredit ? (prem * 100 * qty) : (Math.max(0, width - prem) * 100 * qty);
            const maxRisk = isCredit ? (Math.max(0, width - prem) * 100 * qty) : (prem * 100 * qty);
            const roc = maxRisk > 0 ? ((maxProfit / maxRisk) * 100).toFixed(1) : '0.0';
            previewEl.innerText = `Collateral: $${maxRisk.toFixed(2)} | Max Profit: $${maxProfit.toFixed(2)} (${roc}% ROC)`;
        }

        async function submitAdjustTrade(id) {
            const drawer = document.getElementById('adjustDrawer-' + id);
            const assetType = (drawer && drawer.dataset.assetType) ? drawer.dataset.assetType.toUpperCase() : 'STOCK';
            const entryInput = document.getElementById('adjustEntry-' + id);
            const sizeInput = document.getElementById('adjustSize-' + id);
            const newEntry = entryInput ? parseFloat(entryInput.value) : NaN;
            const newSize = sizeInput ? parseFloat(sizeInput.value) : NaN;
            const newStop = parseFloat(document.getElementById('adjustStop-' + id).value) || 0;
            const newTp1 = parseFloat(document.getElementById('adjustTp1-' + id).value) || 0;

            const updates = {stop_loss: newStop, target_1: newTp1};
            if (!isNaN(newEntry) && newEntry >= 0) {
                updates.entry_price = newEntry;
                if (assetType === 'SPREAD') {
                    updates.net_premium = newEntry;
                }
            }
            if (!isNaN(newSize) && newSize > 0) {
                updates.size = newSize;
            }

            try {
                const res = await fetch('/api/ledger/update', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({id: id, updates: updates})
                });
                const data = await res.json();
                if (data.status === 'success') {
                    upsertLocalTrade(data.trade);
                    loadLedger();
                } else {
                    alert('Error updating trade: ' + (data.message || 'unknown error'));
                }
            } catch (err) {
                alert('Update request failed: ' + err.message);
            }
        }

        async function deleteLedgerTrade(id) {
            if (!confirm('Permanently delete this trade from your ledger?')) return;
            try {
                const res = await fetch('/api/ledger/' + id, {method: 'DELETE'});
                const data = await res.json();
                if (data.status === 'success') {
                    removeLocalTrade(id);
                    loadLedger();
                }
            } catch (err) {
                alert('Delete request failed: ' + err.message);
            }
        }

    </script>
</body>
</html>
"""
