import os
import sys
import time
import json
import math
import re
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import data_manager
import durable_storage

# Runtime isolation and durable cloud storage setup
runtime = Path('/tmp/double-edge-preview') if os.environ.get('VERCEL') else Path(ROOT_DIR) / '.runtime'
runtime.mkdir(parents=True, exist_ok=True)
raw_writable = Path('/tmp/raw_csv') if os.environ.get('VERCEL') else Path(ROOT_DIR) / 'raw_csv'
raw_writable.mkdir(parents=True, exist_ok=True)

for filename in ('gamma_history.json', 'opportunities.json', 'spy_rsp.csv', 'qqq_rsp.csv', 'iwm_qqq.csv'):
    source, target = Path(ROOT_DIR) / 'processed_data' / filename, runtime / filename
    if source.is_file() and (not target.exists() or source.stat().st_mtime > target.stat().st_mtime):
        shutil.copyfile(source, target)
data_manager.PROCESSED_DIR = str(runtime)

try:
    restored = durable_storage.sync_cloud_to_local(runtime, raw_writable)
    if restored > 0:
        print(f"[Storage] Restored {restored} persistent files from Vercel Blob.")
except Exception as e:
    print(f"[Storage] Cold-start cloud sync note: {e}")

from data_manager import (
    load_ticker_data, get_available_tickers,
    MAG_7, INDEXES, SECTORS, MACRO_ASSETS, generate_synthetic_ratios,
    PROCESSED_DIR
)
from indicators import compute_all_metrics
from narrative_engine import analyze_market_narrative, compute_ratio_analytics
from confluence_engine import compute_all_confluences

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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
    for sym in ["SPY", "QQQ", "IWM", "GLD", "TLT", "IBIT", "USO", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]:
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
        from gamma_tracker import load_gamma_history
        t_up = ticker.upper()
        data = compute_gamma_exposure_profile(t_up, max_expirations=5, force_refresh=force)
        hist = load_gamma_history(t_up)
        if isinstance(data, dict):
            data["history"] = hist or []
        return clean_json(data)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/gamma/{ticker}/history")
def api_gamma_history(ticker: str):
    try:
        from gamma_tracker import load_gamma_history
        hist = load_gamma_history(ticker.upper())
        return clean_json({"ticker": ticker.upper(), "history": hist or []})
    except Exception as e:
        return {"error": str(e), "history": []}

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
def api_options(ticker: str, direction: str = Query(None), side: str = Query(None), expiration: str = Query(None)):
    try:
        t_up = ticker.upper()
        from options_engine import compute_realized_volatilities, fetch_term_structure, compute_iv_rank_and_percentile, analyze_smile_and_skew
        df_hist = load_ticker_data(t_up)
        hvs = compute_realized_volatilities(df_hist, ticker_symbol=t_up)
        clean_hv = float(hvs.get('HV_30_clean', hvs.get('HV_30', 20.0)))
        hv_1y = float(hvs.get('HV_252', hvs.get('HV_1Y', 20.0)))
        term_df, spot_price, next_earnings = fetch_term_structure(t_up, hvs_dict=hvs, hv_benchmark=clean_hv)
        if term_df.empty or spot_price is None:
            return {"error": f"Live options data not available for {t_up}"}
            
        front_atm_iv = float(term_df['ATM_IV'].iloc[0])
        iv_stats = compute_iv_rank_and_percentile(t_up, current_iv_pct=front_atm_iv)
        
        # Enrich term_df with Term Slopes vs Front and vs Prev
        valid_exps = [str(e) for e in term_df['Expiration'].tolist()]
        enriched_chains = []
        for idx, row in term_df.iterrows():
            atm_iv = float(row['ATM_IV'])
            slope_front = round(atm_iv - front_atm_iv, 2)
            spread_clean = round(atm_iv - clean_hv, 2)
            spread_1y = round(atm_iv - hv_1y, 2)
            ratio_clean = round(atm_iv / max(clean_hv, 0.1), 2)
            ratio_1y = round(atm_iv / max(hv_1y, 0.1), 2)
            if idx == 0:
                slope_prev = 0.0
                struct_label = "Front Baseline"
            else:
                prev_atm = float(term_df.iloc[idx - 1]['ATM_IV'])
                slope_prev = round(atm_iv - prev_atm, 2)
                if slope_front > 0.5:
                    struct_label = "Contango"
                elif slope_front < -0.5:
                    struct_label = "Backwardation"
                else:
                    struct_label = "Flat"
            
            r_dict = row.to_dict()
            r_dict['Expiration'] = str(row['Expiration'])
            r_dict['Slope_Front'] = slope_front
            r_dict['Slope_Prev'] = slope_prev
            r_dict['Spread_HV_Clean'] = spread_clean
            r_dict['Spread_HV_1Y'] = spread_1y
            r_dict['Ratio_HV_Clean'] = ratio_clean
            r_dict['Ratio_HV_1Y'] = ratio_1y
            r_dict['Structure'] = struct_label
            r_dict['Total_OI'] = int(row.get('Total_OI', 0))
            enriched_chains.append(r_dict)
            
        # Identify High-OI Benchmark cycles (Front, 1M, 2M, 3M, Quarterly/4M, 6M, 12M)
        buckets = [
            ("Front Cycle", 1, 20),
            ("1M Benchmark", 21, 48),
            ("2M Benchmark", 49, 75),
            ("3M Benchmark", 76, 110),
            ("Quarterly (4M)", 111, 160),
            ("6M Benchmark", 161, 260),
            ("12M Benchmark", 261, 420),
        ]
        benchmark_map = {}
        for label, min_d, max_d in buckets:
            candidates = [r for r in enriched_chains if min_d <= int(r.get('DTE', 0)) <= max_d]
            if candidates:
                best_cand = max(candidates, key=lambda x: int(x.get('Total_OI', 0)))
                benchmark_map[best_cand['Expiration']] = label

        for r in enriched_chains:
            r['Benchmark_Tenor'] = benchmark_map.get(r['Expiration'], None)
            r['Is_Benchmark'] = r['Expiration'] in benchmark_map
            
        # Determine overall Term Structure Regime
        back_atm = float(term_df['ATM_IV'].iloc[-1])
        net_slope = round(back_atm - front_atm_iv, 2)
        if net_slope > 1.0:
            regime_name = f"Contango (+{net_slope} pts)"
            regime_desc = "Normal upward slope: Back-month premium"
            regime_style = "mint"
        elif net_slope < -1.0:
            regime_name = f"Backwardation ({net_slope} pts)"
            regime_desc = "Inverted slope: Front elevated / event spike"
            regime_style = "cherry"
        else:
            regime_name = f"Flat Term ({net_slope:+} pts)"
            regime_desc = "Balanced slope across horizons"
            regime_style = "slate"
            
        term_regime = {
            "name": regime_name,
            "slope": net_slope,
            "style": regime_style,
            "description": regime_desc
        }
        
        # Expiration selection for smile analysis
        if expiration and expiration in valid_exps:
            chosen_exp = expiration
        else:
            chosen_exp = valid_exps[0]
            
        smile = analyze_smile_and_skew(t_up, chosen_exp, spot_price)
        smile_payload = None
        if smile:
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
            }
            
        expirations_list = [
            {
                "expiration": str(row['Expiration']),
                "dte": int(row['DTE']),
                "atm_iv": round(float(row.get('ATM_IV', 0)), 1),
                "iv_hv_ratio": round(float(row.get('IV_HV_Ratio', 1.0)), 2),
                "exp_move_pct": round(float(row.get('Expected_Move_Pct', 0)), 1),
                "exp_move_dollars": round(float(row.get('Expected_Move', 0)), 2),
                "spans_earnings": bool(row.get('Spans_Earnings', False)),
                "structure": row.get('Structure', 'Flat'),
                "slope_front": row.get('Slope_Front', 0.0),
                "total_oi": int(row.get('Total_OI', 0)),
                "is_benchmark": bool(row.get('Is_Benchmark', False)),
                "benchmark_tenor": row.get('Benchmark_Tenor', None)
            }
            for row in enriched_chains
        ]

        # Relative Value & Forward Volatility Opportunity Scanner
        rv_opportunities = []
        n_chains = len(enriched_chains)
        import numpy as np
        for i in range(n_chains):
            # Evaluate forward spreads up to 5 maturity cycles ahead
            for j in range(i + 1, min(i + 6, n_chains)):
                r1 = enriched_chains[i]
                r2 = enriched_chains[j]
                dte1 = int(r1.get('DTE', 0))
                dte2 = int(r2.get('DTE', 0))
                if dte2 <= dte1 or (dte2 - dte1) < 4:
                    continue
                
                t1 = dte1 / 365.25
                t2 = dte2 / 365.25
                dt = t2 - t1
                
                iv1_mid = float(r1.get('ATM_IV', 20.0))
                iv2_mid = float(r2.get('ATM_IV', 20.0))
                
                # Theoretical Mid Forward Vol
                var1_mid = (iv1_mid / 100.0)**2 * t1
                var2_mid = (iv2_mid / 100.0)**2 * t2
                dvar_mid = var2_mid - var1_mid
                fwd_mid = float(np.sqrt(dvar_mid / dt) * 100.0) if dvar_mid > 0 and dt > 0 else 0.0
                
                # Executable Forward Vol (Sell Front at Bid, Buy Back at Ask)
                iv1_bid = float(r1.get('Call_Bid_IV', iv1_mid))
                iv2_ask = float(r2.get('Call_Ask_IV', iv2_mid))
                
                var1_bid = (iv1_bid / 100.0)**2 * t1
                var2_ask = (iv2_ask / 100.0)**2 * t2
                dvar_exec = var2_ask - var1_bid
                fwd_exec = float(np.sqrt(dvar_exec / dt) * 100.0) if dvar_exec > 0 and dt > 0 else fwd_mid
                
                drag = round(fwd_exec - fwd_mid, 1)
                spread_clean = round(fwd_exec - clean_hv, 1)
                spread_1y = round(fwd_exec - hv_1y, 1)
                ratio_clean = round(fwd_exec / max(clean_hv, 0.1), 2)
                ratio_1y = round(fwd_exec / max(hv_1y, 0.1), 2)
                
                c1_b = float(r1.get('Call_Bid', 0))
                c1_a = float(r1.get('Call_Ask', 0))
                c2_b = float(r2.get('Call_Bid', 0))
                c2_a = float(r2.get('Call_Ask', 0))
                
                net_debit_exec = round(c2_a - c1_b, 2)
                net_debit_mid = round(((c2_b + c2_a)/2.0) - ((c1_b + c1_a)/2.0), 2)
                slippage_cost = round(net_debit_exec - net_debit_mid, 2)
                
                th1 = abs(float(r1.get('Call_Theta', 0)))
                th2 = abs(float(r2.get('Call_Theta', 0)))
                net_theta = round(th1 - th2, 2)
                
                # Check Short Leg elevation: Front leg must be relatively expensive or higher than back leg
                front_is_elevated = (iv1_mid >= clean_hv - 0.5) or (iv1_mid - iv2_mid >= 1.5)
                front_is_near = (dte1 <= 21)
                vol_edge = round(iv1_mid - fwd_exec, 1)
                
                # Prime Trade Criteria:
                # 1. Front leg is elevated (or slopes down steeply to back leg)
                # 2. Forward window is discounted vs Clean HV or 1Y Normal HV
                # 3. Trader earns positive theta carry (front decays faster than back)
                # 4. Front leg is near-term (<= 21 DTE) so decay actually accrues
                # 5. Net debit > 0
                is_prime = bool(front_is_elevated and (spread_clean <= -2.0 or spread_1y <= -2.0) and net_theta > 0 and net_debit_exec > 0 and front_is_near)
                
                # Institutional Trade Quality Score:
                # Rewards high spread sold vs bought, high discount vs clean HV, and near-term decay focus
                quality_score = vol_edge + max(0.0, clean_hv - fwd_exec) + (10.0 if dte1 <= 10 else (5.0 if dte1 <= 21 else 0.0))
                
                rv_opportunities.append({
                    'front_exp': str(r1['Expiration']),
                    'back_exp': str(r2['Expiration']),
                    'dte1': dte1,
                    'dte2': dte2,
                    'window_days': dte2 - dte1,
                    'iv1': iv1_mid,
                    'iv2': iv2_mid,
                    'vol_edge': vol_edge,
                    'fwd_mid': round(fwd_mid, 1),
                    'fwd_exec': round(fwd_exec, 1),
                    'drag': drag,
                    'spread_clean': spread_clean,
                    'spread_1y': spread_1y,
                    'ratio_clean': ratio_clean,
                    'ratio_1y': ratio_1y,
                    'net_debit_exec': net_debit_exec,
                    'net_debit_mid': net_debit_mid,
                    'slippage_cost': slippage_cost,
                    'net_theta': net_theta,
                    'is_prime': is_prime,
                    'score': round(quality_score, 1),
                    'strike_front': float(r1.get('ATM_Strike', 0)),
                    'strike_back': float(r2.get('ATM_Strike', 0)),
                    'front_tenor': r1.get('Benchmark_Tenor'),
                    'back_tenor': r2.get('Benchmark_Tenor')
                })
        
        # Sort so prime opportunities with highest trade quality score come first
        rv_opportunities = sorted(rv_opportunities, key=lambda x: (not x['is_prime'], -x['score']))

        return clean_json({
            "ticker": t_up,
            "spot_price": round(spot_price, 2),
            "next_earnings": next_earnings.strftime("%b %d, %Y") if next_earnings else None,
            "clean_hv": round(clean_hv, 2),
            "hv_1y": round(hv_1y, 2),
            "hvs": hvs,
            "iv_stats": iv_stats,
            "term_regime": term_regime,
            "term_chains": enriched_chains,
            "expirations": expirations_list,
            "selected_expiration": chosen_exp,
            "smile": smile_payload,
            "rv_opportunities": rv_opportunities
        })
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/options/{ticker}/smile")
def api_options_smile(ticker: str, expiration: str = Query(...), direction: str = Query(None), side: str = Query(None), spot: float = Query(None)):
    try:
        t_up = ticker.upper()
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
                
        smile = analyze_smile_and_skew(t_up, expiration, spot_price)
        if not smile:
            return {"error": f"Could not construct volatility smile for {expiration}"}
            
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

        return clean_json({
            "ticker": t_up,
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
            }
        })
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/opportunities")
def api_opportunities(refresh: bool = False):
    try:
        from opportunities_engine import get_opportunities_data, scan_volatility_opportunities
        if refresh:
            return scan_volatility_opportunities()
        return get_opportunities_data()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/preview/history/{ticker}")
def api_preview_history(ticker: str):
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,14}", ticker):
        return JSONResponse({"error": "Invalid ticker."}, status_code=400)
    frame = load_ticker_data(ticker)
    if frame.empty:
        return JSONResponse({"error": "No OHLC data for this ticker."}, status_code=404)
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = []
    for date, *values in frame[columns].tail(520).itertuples(index=False, name=None):
        if all(math.isfinite(float(value)) for value in values):
            rows.append([date.strftime("%Y-%m-%d"), *map(float, values)])
    return {"ticker": ticker, "rows": rows}


def is_sync_request(request: Request):
    origin = request.headers.get("origin")
    expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
    return origin == expected and request.headers.get("x-preview-sync") == "1"


@app.post("/api/preview/sync-csv")
def api_preview_sync_csv(request: Request):
    if not is_sync_request(request):
        return JSONResponse({"message": "Use the dashboard sync button."}, status_code=403)
    result = api_sync_csv()
    if isinstance(result, JSONResponse):
        return result
    return {"message": f"CSV cache reloaded for {result['loaded_count']} tickers. The hosted build reads its packaged CSVs."}


@app.post("/api/preview/sync-yfinance")
def api_preview_sync_yfinance(request: Request):
    if not is_sync_request(request):
        return JSONResponse({"message": "Use the dashboard sync button."}, status_code=403)
    return run_yfinance_sync()


def run_yfinance_sync():
    result = api_sync_data({"scope": "core"})
    if isinstance(result, JSONResponse):
        return result

    try:
        generate_synthetic_ratios()
    except Exception as e:
        print(f"[Sync] Error generating synthetic ratios: {e}")

    _CACHE.clear()

    persisted_count = 0
    if durable_storage.get_blob_token():
        try:
            for ratio in ('spy_rsp.csv', 'qqq_rsp.csv', 'iwm_qqq.csv'):
                f = runtime / ratio
                if f.is_file() and durable_storage.persist_file_to_cloud(f, ratio):
                    persisted_count += 1
            if raw_writable.is_dir():
                for p in raw_writable.glob("*_us_d.csv"):
                    if durable_storage.persist_file_to_cloud(p, p.name):
                        persisted_count += 1
            durable_storage.persist_gamma_history(runtime)
        except Exception as e:
            print(f"[Storage] Cloud persistence warning: {e}")

    storage_note = (
        f" Persisted {persisted_count} files to Vercel Blob."
        if persisted_count > 0
        else (" Persisted to Vercel Blob." if durable_storage.get_blob_token() else " Saved to runtime cache.")
    )
    msg = (
        f"Synced {result.get('updated_count', 0)} of {result.get('total', 0)} tickers "
        f"from yfinance (latest: {result.get('last_date', 'today')}).{storage_note}"
    )
    return {"message": msg, **result}


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = os.path.join(ROOT_DIR, "public", "index.html")
    if os.path.isfile(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse('Dashboard index not found.', status_code=404)

