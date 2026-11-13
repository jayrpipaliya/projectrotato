"""
Institutional Options Volatility Desk & Strike Selection Engine
- Incorporates De-Jumped Realized Volatility (HV_clean) to prevent post-earnings distortion.
- Automatically detects upcoming earnings dates and partitions contracts (Inter-Earnings vs Spans Earnings).
- Corrects annualization mismatch (Trading Day vs Calendar Day scaling: sqrt(252/365)).
- Asymmetric Cubic Skew Modeling over Log-Moneyness (ln(K/S)).
- Full Black-Scholes Greeks with Dividend Yield (q) and Risk-Free Rate (r).
- Ask-IV Inversion: Calculates true implied volatility at the Ask (checks if edge survives the bid-ask spread).
- Computes Market Implied Expected Move (1σ price bounds).
- Outlier filtering: Cleans Yahoo 0.250007 placeholders, zero bids, wide spreads, and fractional strikes.
"""

import math
import time
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
try:
    import plotly.graph_objects as go
except Exception:
    go = None
from concurrent.futures import ThreadPoolExecutor

try:
    import streamlit as st
    st_cache = st.cache_data(ttl=600, show_spinner=False)
except Exception:
    st = None
    def st_cache(func=None, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator

try:
    from scipy.stats import norm
    norm_cdf = norm.cdf
    norm_pdf = norm.pdf
except Exception:
    def norm_cdf(x):
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
    def norm_pdf(x):
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (x ** 2))

# ----------------- IMPLIED VOLATILITY NUMERICAL SOLVER -----------------
def implied_vol_solver(price, S, K, T, r=0.045, q=0.0075, is_call=True):
    """Numerically inverts Black-Scholes to extract true IV from settlement price when feeds zero out overnight."""
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.20
    # Discounted theoretical European lower bound
    disc_intrinsic = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T)) if is_call else max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))
    if price <= disc_intrinsic:
        return 0.06
    sigma = 0.22
    for _ in range(20):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            theo = S * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        else:
            theo = K * np.exp(-r * T) * norm_cdf(-d2) - S * np.exp(-q * T) * norm_cdf(-d1)
        diff = theo - price
        if abs(diff) < 0.005 or abs(diff) / max(price, 0.1) < 0.005:
            return float(np.clip(sigma, 0.04, 3.0))
        vega = max(S * np.exp(-q * T) * np.sqrt(T) * norm_pdf(d1), 1e-4)
        step = np.clip(diff / vega, -0.4, 0.4)
        sigma -= step
        if sigma <= 0.02 or sigma > 3.0:
            break
    return float(np.clip(sigma, 0.04, 3.0))

# ----------------- DIVIDEND YIELD LOOKUP -----------------
DIVIDEND_YIELDS = {
    'SPY': 0.0120, 'QQQ': 0.0060, 'IWM': 0.0125, 'GLD': 0.0000,
    'AAPL': 0.0050, 'MSFT': 0.0070, 'NVDA': 0.0003, 'AMZN': 0.0000,
    'META': 0.0035, 'GOOGL': 0.0045, 'TSLA': 0.0000, 'XLF': 0.0145, 'SMH': 0.0065
}


def get_market_session_status():
    """
    Returns current market session status in US/Eastern:
    - Regular Market: Mon-Fri 9:30 AM to 4:15 PM EST
    - Pre-Market: Mon-Fri 4:00 AM to 9:30 AM EST
    - After-Hours: Mon-Fri 4:15 PM to 8:00 PM EST
    - Closed: Overnight & Weekends
    """
    try:
        from zoneinfo import ZoneInfo
        now_est = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_est = datetime.now()

    weekday = now_est.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    time_min = now_est.hour * 60 + now_est.minute

    if weekday in [5, 6]:
        return "WEEKEND_CLOSED", "Weekend Closed (Final Friday Settlement)"

    if 570 <= time_min < 975:
        return "LIVE_INTRADAY", "Regular Session (Live OPRA ~15m Delay)"
    elif 240 <= time_min < 570:
        return "PRE_MARKET", "Pre-Market (Overnight OCC Baseline)"
    elif 975 <= time_min < 1200:
        return "AFTER_HOURS", "Post-Market (Official Session Close)"
    else:
        return "OVERNIGHT_CLOSED", "Overnight Closed (Settlement Snapshot)"

def get_dividend_yield(ticker_symbol):
    """Returns continuous dividend yield estimate for Black-Scholes pricing."""
    return DIVIDEND_YIELDS.get(ticker_symbol.upper(), 0.0075)

# ----------------- DEALER GAMMA EXPOSURE (GEX) STRUCTURAL ENGINE -----------------
_GAMMA_PROFILE_CACHE = {}
_GAMMA_PROFILE_TTL = 300  # 5 minutes in-memory cache

def black_scholes_gamma(S, K, T, sigma, r=0.045, q=0.0075):
    """
    Computes analytical Black-Scholes gamma with dividend yield q:
    Γ = (norm_pdf(d1) * e^(-q*T)) / (S * σ * sqrt(T))
    """
    if S <= 0 or K <= 0 or T <= 0.0001 or sigma <= 0.01:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = (norm_pdf(d1) * np.exp(-q * T)) / (S * sigma * np.sqrt(T))
    return float(gamma)

def compute_gamma_exposure_profile(ticker_symbol="SPY", max_expirations=5, force_refresh=False):
    """
    Calculates institutional dealer Gamma Exposure (GEX) profile:
    - Inverts Black-Scholes gamma across all near-term strike open interest.
    - Computes Dollar GEX per strike ($ Millions per 1% move):
        Call GEX_K = (Spot^2) * Gamma * Call_OI * 100 * 0.01 / 1,000,000
        Put GEX_K  = (Spot^2) * Gamma * Put_OI  * 100 * 0.01 / 1,000,000
    - Identifies:
        1. Call Wall (Major dealer hedging resistance ceiling / pin)
        2. Put Wall (Major dealer hedging support floor / pin)
        3. Absolute Gamma Strike (Peak total liquidity cluster)
        4. Zero Gamma Flip Level (Regime inflection boundary)
        5. Net Dealer Gamma Regime (Long Gamma = Vol Suppressing, Short Gamma = Vol Accelerating)
    """
    ticker_up = ticker_symbol.upper()
    now_ts = time.time()
    try:
        from zoneinfo import ZoneInfo
        now_dt = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_dt = datetime.utcnow() - timedelta(hours=5)
    if not force_refresh and ticker_up in _GAMMA_PROFILE_CACHE:
        cached_entry = _GAMMA_PROFILE_CACHE[ticker_up]
        if now_ts - cached_entry["timestamp"] < _GAMMA_PROFILE_TTL:
            return cached_entry["data"]

    t0 = time.time()
    t = yf.Ticker(ticker_up)
    try:
        expirations = t.options
    except Exception:
        expirations = None
    if not expirations:
        return {"error": f"No options available for {ticker_up}"}

    fast_info = getattr(t, "fast_info", None)
    spot_price = None
    if fast_info:
        spot_price = getattr(fast_info, "last_price", None) or getattr(fast_info, "previous_close", None)
    if spot_price is None or spot_price <= 0:
        try:
            hist = t.history(period="5d")
            if not hist.empty:
                spot_price = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    if spot_price is None or spot_price <= 0:
        return {"error": f"Unable to determine spot price for {ticker_up}"}

    spot_price = float(spot_price)
    today = pd.Timestamp.now().normalize()
    div_yield = get_dividend_yield(ticker_up)

    # Filter to active expirations within 0 to 45 DTE (including 0 DTE same-day expirations)
    valid_exps = []
    for e in expirations:
        try:
            dte = (pd.Timestamp(e) - today).days
            if 0 <= dte <= 45:
                valid_exps.append((e, dte))
        except Exception:
            continue

    if not valid_exps:
        for e in expirations[:3]:
            try:
                dte = max((pd.Timestamp(e) - today).days, 0)
                valid_exps.append((e, dte))
            except Exception:
                continue

    valid_exps = sorted(valid_exps, key=lambda x: x[1])[:max_expirations]
    if not valid_exps:
        return {"error": f"No valid expirations found for {ticker_up}"}

    strikes_gex = {}
    min_strike = spot_price * 0.80
    max_strike = spot_price * 1.20

    def process_exp(exp_tuple):
        exp, dte = exp_tuple
        try:
            chain = t.option_chain(exp)
            T = max(dte, 0.5) / 365.25
            local_gex = {}
            local_calls = []
            local_puts = []

            calls = chain.calls
            if not calls.empty:
                # Dynamic ATM IV fallback estimation
                atm_sub = calls[(calls["strike"] >= spot_price * 0.96) & (calls["strike"] <= spot_price * 1.04)]
                valid_atm_ivs = [float(v) for v in atm_sub["impliedVolatility"].dropna() if 0.05 <= float(v) <= 2.5]
                base_iv = float(np.median(valid_atm_ivs)) if valid_atm_ivs else 0.22

                calls = calls[(calls["strike"] >= min_strike) & (calls["strike"] <= max_strike)]
                for _, row in calls.iterrows():
                    k = float(row["strike"])
                    val_oi = row.get("openInterest")
                    val_vol = row.get("volume")
                    oi = int(val_oi) if (pd.notna(val_oi) and val_oi is not None and float(val_oi) > 0) else 0
                    vol = int(val_vol) if (pd.notna(val_vol) and val_vol is not None and float(val_vol) > 0) else 0
                    # Use open interest or active contract volume when overnight clearing hasn't settled OI
                    eff_contracts = oi if oi > 0 else vol
                    if eff_contracts <= 0:
                        continue
                    val_iv = row.get("impliedVolatility")
                    if pd.isna(val_iv) or val_iv is None:
                        iv = base_iv
                    else:
                        iv = float(val_iv)
                    if iv < 0.04 or iv > 2.5:
                        iv = base_iv
                    gamma = black_scholes_gamma(spot_price, k, T, iv, q=div_yield)
                    call_dollar_gex = (spot_price ** 2) * gamma * eff_contracts * 0.01 * 100 / 1_000_000
                    if k not in local_gex:
                        local_gex[k] = {"call_gex": 0.0, "put_gex": 0.0, "call_oi": 0, "put_oi": 0}
                    local_gex[k]["call_gex"] += call_dollar_gex
                    local_gex[k]["call_oi"] += eff_contracts
                    local_calls.append((k, T, iv, eff_contracts))

            puts = chain.puts
            if not puts.empty:
                atm_sub = puts[(puts["strike"] >= spot_price * 0.96) & (puts["strike"] <= spot_price * 1.04)]
                valid_atm_ivs = [float(v) for v in atm_sub["impliedVolatility"].dropna() if 0.05 <= float(v) <= 2.5]
                base_iv = float(np.median(valid_atm_ivs)) if valid_atm_ivs else 0.22

                puts = puts[(puts["strike"] >= min_strike) & (puts["strike"] <= max_strike)]
                for _, row in puts.iterrows():
                    k = float(row["strike"])
                    val_oi = row.get("openInterest")
                    val_vol = row.get("volume")
                    oi = int(val_oi) if (pd.notna(val_oi) and val_oi is not None and float(val_oi) > 0) else 0
                    vol = int(val_vol) if (pd.notna(val_vol) and val_vol is not None and float(val_vol) > 0) else 0
                    eff_contracts = oi if oi > 0 else vol
                    if eff_contracts <= 0:
                        continue
                    val_iv = row.get("impliedVolatility")
                    if pd.isna(val_iv) or val_iv is None:
                        iv = base_iv
                    else:
                        iv = float(val_iv)
                    if iv < 0.04 or iv > 2.5:
                        iv = base_iv
                    gamma = black_scholes_gamma(spot_price, k, T, iv, q=div_yield)
                    put_dollar_gex = (spot_price ** 2) * gamma * eff_contracts * 0.01 * 100 / 1_000_000
                    if k not in local_gex:
                        local_gex[k] = {"call_gex": 0.0, "put_gex": 0.0, "call_oi": 0, "put_oi": 0}
                    local_gex[k]["put_gex"] += put_dollar_gex
                    local_gex[k]["put_oi"] += eff_contracts
                    local_puts.append((k, T, iv, eff_contracts))

            exp_max_ts = None
            if not calls.empty and "lastTradeDate" in calls.columns:
                c_max = calls["lastTradeDate"].dropna().max()
                if pd.notna(c_max):
                    exp_max_ts = c_max
            if not puts.empty and "lastTradeDate" in puts.columns:
                p_max = puts["lastTradeDate"].dropna().max()
                if pd.notna(p_max):
                    exp_max_ts = max(exp_max_ts, p_max) if exp_max_ts is not None else p_max

            return {"gex": local_gex, "calls": local_calls, "puts": local_puts, "max_trade_ts": exp_max_ts}
        except Exception:
            return {"gex": {}, "calls": [], "puts": [], "max_trade_ts": None}

    with ThreadPoolExecutor(max_workers=min(5, len(valid_exps))) as executor:
        results = list(executor.map(process_exp, valid_exps))

    all_calls = []
    all_puts = []
    latest_trade_ts = None
    for res in results:
        if not res:
            continue
        ts_cand = res.get("max_trade_ts")
        if ts_cand is not None and pd.notna(ts_cand):
            latest_trade_ts = max(latest_trade_ts, ts_cand) if latest_trade_ts is not None else ts_cand
        for k, v in res.get("gex", {}).items():
            if k not in strikes_gex:
                strikes_gex[k] = {"call_gex": 0.0, "put_gex": 0.0, "call_oi": 0, "put_oi": 0}
            strikes_gex[k]["call_gex"] += v["call_gex"]
            strikes_gex[k]["put_gex"] += v["put_gex"]
            strikes_gex[k]["call_oi"] += v["call_oi"]
            strikes_gex[k]["put_oi"] += v["put_oi"]
        all_calls.extend(res.get("calls", []))
        all_puts.extend(res.get("puts", []))

    # Fallback to liquid monthly/weekly cycles if front expirations have unfinalized overnight clearing
    if not strikes_gex and expirations and len(expirations) > len(valid_exps):
        fallback_exps = []
        for e in expirations[len(valid_exps):min(len(expirations), len(valid_exps) + 8)]:
            try:
                dte = max((pd.Timestamp(e) - today).days, 0)
                fallback_exps.append((e, dte))
            except Exception:
                continue
        if fallback_exps:
            with ThreadPoolExecutor(max_workers=min(5, len(fallback_exps))) as executor:
                fb_results = list(executor.map(process_exp, fallback_exps))
            for res in fb_results:
                if not res:
                    continue
                ts_cand = res.get("max_trade_ts")
                if ts_cand is not None and pd.notna(ts_cand):
                    latest_trade_ts = max(latest_trade_ts, ts_cand) if latest_trade_ts is not None else ts_cand
                for k, v in res.get("gex", {}).items():
                    if k not in strikes_gex:
                        strikes_gex[k] = {"call_gex": 0.0, "put_gex": 0.0, "call_oi": 0, "put_oi": 0}
                    strikes_gex[k]["call_gex"] += v["call_gex"]
                    strikes_gex[k]["put_gex"] += v["put_gex"]
                    strikes_gex[k]["call_oi"] += v["call_oi"]
                    strikes_gex[k]["put_oi"] += v["put_oi"]
                all_calls.extend(res.get("calls", []))
                all_puts.extend(res.get("puts", []))

    if not strikes_gex:
        return {"error": f"No liquid gamma profile extracted for {ticker_up}"}

    sorted_strikes = sorted(strikes_gex.keys())
    abs_gamma_strike = max(sorted_strikes, key=lambda k: strikes_gex[k]["call_gex"] + strikes_gex[k]["put_gex"])

    # Institutional Structural Wall Calibration:
    # A true Put Wall must function as an Out-Of-The-Money (OTM) hedging support floor strictly below spot (K < Spot).
    # A true Call Wall must function as an Out-Of-The-Money (OTM) hedging resistance ceiling strictly above spot (K > Spot).
    # ATM strikes within 0.35% of spot (where Black-Scholes gamma spikes to infinity) represent the
    # ATM Straddle Pin / Absolute Gamma Node, preventing artificial single-strike cannibalization (CW == PW == Spot).
    atm_band_low = spot_price * 0.9965
    atm_band_high = spot_price * 1.0035

    otm_put_cands = [k for k in sorted_strikes if k < atm_band_low]
    if not otm_put_cands:
        otm_put_cands = [k for k in sorted_strikes if k < spot_price]
    if otm_put_cands:
        put_wall = max(otm_put_cands, key=lambda k: (strikes_gex[k]["put_gex"], strikes_gex[k]["put_oi"]))
    else:
        put_wall = min(sorted_strikes)

    otm_call_cands = [k for k in sorted_strikes if k > atm_band_high]
    if not otm_call_cands:
        otm_call_cands = [k for k in sorted_strikes if k > spot_price]
    if otm_call_cands:
        call_wall = max(otm_call_cands, key=lambda k: (strikes_gex[k]["call_gex"], strikes_gex[k]["call_oi"]))
    else:
        call_wall = max(sorted_strikes)

    total_call_gex = sum(strikes_gex[k]["call_gex"] for k in sorted_strikes)
    total_put_gex = sum(strikes_gex[k]["put_gex"] for k in sorted_strikes)
    total_net_gex = total_call_gex - total_put_gex

    # Exact Zero Gamma Level solver: Spot price S* where Total Market Net Dealer Gamma Exposure flips sign
    zero_gamma_level = spot_price
    if all_calls and all_puts:
        try:
            c_K, c_T, c_iv, c_oi = map(np.array, zip(*all_calls))
            p_K, p_T, p_iv, p_oi = map(np.array, zip(*all_puts))
            r_rate = 0.045
            q_yield = div_yield

            def net_gamma_diff_at_spot(S):
                c_d1 = (np.log(S / c_K) + (r_rate - q_yield + 0.5 * c_iv**2) * c_T) / (c_iv * np.sqrt(c_T))
                c_gamma = (np.exp(-0.5 * c_d1**2) / np.sqrt(2 * np.pi) * np.exp(-q_yield * c_T)) / (S * c_iv * np.sqrt(c_T))
                tot_c = np.sum((S**2) * c_gamma * c_oi * 0.01 * 100 / 1_000_000)

                p_d1 = (np.log(S / p_K) + (r_rate - q_yield + 0.5 * p_iv**2) * p_T) / (p_iv * np.sqrt(p_T))
                p_gamma = (np.exp(-0.5 * p_d1**2) / np.sqrt(2 * np.pi) * np.exp(-q_yield * p_T)) / (S * p_iv * np.sqrt(p_T))
                tot_p = np.sum((S**2) * p_gamma * p_oi * 0.01 * 100 / 1_000_000)

                return tot_c - tot_p

            s_min = spot_price * 0.70
            s_max = spot_price * 1.30
            f_min = net_gamma_diff_at_spot(s_min)
            f_max = net_gamma_diff_at_spot(s_max)
            solved_zg = None

            if f_min * f_max < 0:
                try:
                    from scipy.optimize import brentq
                    solved_zg = float(brentq(net_gamma_diff_at_spot, s_min, s_max, xtol=0.02))
                except Exception:
                    pass

            if solved_zg is None:
                grid_s = np.linspace(s_min, s_max, 60)
                grid_vals = [net_gamma_diff_at_spot(s) for s in grid_s]
                for i in range(len(grid_vals) - 1):
                    if grid_vals[i] * grid_vals[i+1] <= 0:
                        solved_zg = float(grid_s[i] - grid_vals[i] * (grid_s[i+1] - grid_s[i]) / (grid_vals[i+1] - grid_vals[i]))
                        break

            if solved_zg is not None and abs(solved_zg - spot_price) / spot_price < 0.35:
                zero_gamma_level = solved_zg
        except Exception:
            zero_gamma_level = spot_price

    closest_zg_strike = min(sorted_strikes, key=lambda k: abs(k - zero_gamma_level))

    is_long_gamma = total_net_gex > 0
    net_gex_b = total_net_gex / 1000.0
    is_index = ticker_up in ["SPY", "QQQ", "IWM", "DIA", "RSP"]

    # Magnitude & Conviction Thresholds: A minor +$0.02B is effectively neutral compared to multi-billion dealer books
    neutral_threshold = 0.25 if is_index else 0.05   # $250M for index ETFs, $50M for single stocks
    dominant_threshold = 1.00 if is_index else 0.15  # $1.0B for index ETFs, $150M for single stocks

    abs_net = abs(net_gex_b)
    if abs_net < neutral_threshold:
        gex_intensity = "Neutral / Low Conviction"
        regime_short = "Gamma Neutral"
        regime = "Gamma Neutral (Dealers have minimal delta-hedging impact; directional trend & volume dominate)"
    elif abs_net < dominant_threshold:
        gex_intensity = "Moderate GEX"
        regime_short = "Long Gamma" if is_long_gamma else "Short Gamma"
        regime = f"{'Long' if is_long_gamma else 'Short'} Gamma (Moderate dealer hedging influence)"
    else:
        gex_intensity = "Dominant / High Conviction"
        regime_short = "Deep Long Gamma" if is_long_gamma else "Deep Short Gamma"
        regime = f"{'Long' if is_long_gamma else 'Short'} Gamma ({'Heavy Volatility Suppression & Pinning' if is_long_gamma else 'Aggressive Volatility Amplification & Cascades'})"

    cw_gex = strikes_gex[call_wall]["call_gex"]
    pw_gex = strikes_gex[put_wall]["put_gex"]
    cw_conc = round((cw_gex / total_call_gex * 100), 1) if total_call_gex > 0 else 0.0
    pw_conc = round((pw_gex / total_put_gex * 100), 1) if total_put_gex > 0 else 0.0

    # Curate high-impact / structurally relevant strikes to eliminate table noise
    atm_k = min(sorted_strikes, key=lambda k: abs(k - spot_price))
    key_targets = {call_wall, put_wall, closest_zg_strike, abs_gamma_strike, atm_k}
    top_c_k = set(sorted(sorted_strikes, key=lambda k: strikes_gex[k]["call_gex"], reverse=True)[:5])
    top_p_k = set(sorted(sorted_strikes, key=lambda k: strikes_gex[k]["put_gex"], reverse=True)[:5])
    top_net_k = set(sorted(sorted_strikes, key=lambda k: abs(strikes_gex[k]["call_gex"] - strikes_gex[k]["put_gex"]), reverse=True)[:5])
    top_oi_k = set(sorted(sorted_strikes, key=lambda k: strikes_gex[k]["call_oi"] + strikes_gex[k]["put_oi"], reverse=True)[:5])
    
    step = 5.0 if spot_price > 100 else (2.5 if spot_price > 50 else 1.0)
    round_k = {k for k in sorted_strikes if abs(k - spot_price) / spot_price <= 0.035 and (k % step == 0)}
    heavy_oi_k = {k for k in sorted_strikes if abs(k - spot_price) / spot_price <= 0.035 and (strikes_gex[k]["call_oi"] + strikes_gex[k]["put_oi"] >= 12000)}
    relevant_set = key_targets | top_c_k | top_p_k | top_net_k | top_oi_k | round_k | heavy_oi_k

    strikes_list = []
    for k in sorted_strikes:
        c_g = round(strikes_gex[k]["call_gex"], 1)
        p_g = round(strikes_gex[k]["put_gex"], 1)
        net_g = round(c_g - p_g, 1)
        is_rel = k in relevant_set

        if abs(k - call_wall) < 0.01:
            role = "Call Wall (Pin)"
        elif abs(k - put_wall) < 0.01:
            role = "Put Wall (Floor)"
        elif abs(k - closest_zg_strike) < 0.01:
            role = "Zero Flip"
        elif abs(k - atm_k) < 0.01:
            role = "ATM Spot"
        elif abs(k - abs_gamma_strike) < 0.01:
            role = "Absolute Gamma Peak"
        elif k in top_c_k and k in top_p_k:
            role = "Dual Gamma Node"
        elif k in top_c_k:
            role = "Top Call GEX"
        elif k in top_p_k:
            role = "Top Put GEX"
        elif k in top_net_k:
            role = "Net GEX Imbalance"
        elif k in top_oi_k or k in heavy_oi_k:
            role = "Major Open Interest"
        elif k in round_k:
            role = "Milestone Strike"
        else:
            role = "Normal Liquidity"

        strikes_list.append({
            "strike": round(k, 2),
            "call_gex": c_g,
            "put_gex": p_g,
            "net_gex": net_g,
            "call_oi": strikes_gex[k]["call_oi"],
            "put_oi": strikes_gex[k]["put_oi"],
            "is_relevant": is_rel,
            "structural_role": role
        })

    elapsed = round(time.time() - t0, 3)
    loaded_at_str = now_dt.strftime("%Y-%m-%d %H:%M:%S EST")
    loaded_at_display = now_dt.strftime("%b %d, %Y %I:%M:%S %p EST")

    market_trade_str = None
    if latest_trade_ts is not None:
        try:
            if hasattr(latest_trade_ts, "tz_convert"):
                m_dt = latest_trade_ts.tz_convert("America/New_York")
            else:
                m_dt = pd.to_datetime(latest_trade_ts).tz_localize("UTC").tz_convert("America/New_York")
            market_trade_str = m_dt.strftime("%b %d, %Y %I:%M %p EST")
        except Exception:
            market_trade_str = str(latest_trade_ts)[:19]

    session_code, session_label = get_market_session_status()

    profile = {
        "ticker": ticker_up,
        "spot_price": round(spot_price, 2),
        "call_wall": round(call_wall, 2),
        "call_wall_gex_m": round(strikes_gex[call_wall]["call_gex"], 1),
        "call_wall_dist_pct": round(((call_wall / spot_price) - 1) * 100, 2),
        "put_wall": round(put_wall, 2),
        "put_wall_gex_m": round(strikes_gex[put_wall]["put_gex"], 1),
        "put_wall_dist_pct": round(((put_wall / spot_price) - 1) * 100, 2),
        "absolute_gamma_strike": round(abs_gamma_strike, 2),
        "zero_gamma_level": round(zero_gamma_level, 2),
        "closest_zero_gamma_strike": round(closest_zg_strike, 2),
        "zero_gamma_dist_pct": round(((zero_gamma_level / spot_price) - 1) * 100, 2),
        "total_net_gex_billions": round(total_net_gex / 1000.0, 3),
        "total_call_gex_billions": round(total_call_gex / 1000.0, 3),
        "total_put_gex_billions": round(total_put_gex / 1000.0, 3),
        "is_long_gamma": is_long_gamma,
        "gex_intensity": gex_intensity,
        "call_wall_concentration_pct": cw_conc,
        "put_wall_concentration_pct": pw_conc,
        "regime": regime,
        "regime_short": regime_short,
        "elapsed_seconds": elapsed,
        "loaded_at_timestamp": now_ts,
        "loaded_at_str": loaded_at_str,
        "loaded_at_display": loaded_at_display,
        "market_trade_time": market_trade_str,
        "market_session_code": session_code,
        "market_session_label": session_label,
        "expirations_scanned": [e[0] for e in valid_exps],
        "strikes_count": len(sorted_strikes),
        "relevant_strikes_count": sum(1 for s in strikes_list if s["is_relevant"]),
        "strikes": strikes_list,
        "relevant_strikes": [s for s in strikes_list if s["is_relevant"]]
    }

    _GAMMA_PROFILE_CACHE[ticker_up] = {
        "timestamp": now_ts,
        "data": profile
    }

    try:
        from gamma_tracker import record_gamma_snapshot
        record_gamma_snapshot(ticker_up, profile)
    except Exception:
        pass

    return profile

def plot_gamma_exposure_profile(gamma_data):
    """
    Generates institutional Plotly Strike GEX Profile Chart:
    - Green bars for Call GEX ($M)
    - Red bars for Put GEX (-$M)
    - Markers/lines for Spot Price, Call Wall, Put Wall, Zero Gamma Flip
    """
    if not go or not gamma_data or "strikes" not in gamma_data:
        return None

    strikes = [s["strike"] for s in gamma_data["strikes"]]
    call_gex = [s["call_gex"] for s in gamma_data["strikes"]]
    put_gex = [-s["put_gex"] for s in gamma_data["strikes"]]
    spot = gamma_data["spot_price"]
    cw = gamma_data["call_wall"]
    pw = gamma_data["put_wall"]
    zg = gamma_data["zero_gamma_level"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=strikes, y=call_gex,
        name="Call GEX ($M)",
        marker_color="#00F5A0",
        opacity=0.85
    ))
    fig.add_trace(go.Bar(
        x=strikes, y=put_gex,
        name="Put GEX ($M)",
        marker_color="#FF2E55",
        opacity=0.85
    ))

    fig.add_vline(x=spot, line_width=2, line_dash="dash", line_color="#ffffff",
                  annotation_text=f"Spot ${spot:.2f}", annotation_position="top left",
                  annotation_font=dict(color="#ffffff", size=10))
    fig.add_vline(x=cw, line_width=2, line_dash="dot", line_color="#00F5A0",
                  annotation_text=f"Call Wall ${cw:.0f}", annotation_position="top right",
                  annotation_font=dict(color="#00F5A0", size=10))
    fig.add_vline(x=pw, line_width=2, line_dash="dot", line_color="#FF2E55",
                  annotation_text=f"Put Wall ${pw:.0f}", annotation_position="bottom left",
                  annotation_font=dict(color="#FF2E55", size=10))
    if zg and abs(zg - spot) / spot < 0.35:
        fig.add_vline(x=zg, line_width=1.5, line_dash="dashdot", line_color="#38bdf8",
                      annotation_text=f"Zero Gamma ${zg:.1f}", annotation_position="bottom right",
                      annotation_font=dict(color="#38bdf8", size=9))

    loaded_str = gamma_data.get("loaded_at_display") or gamma_data.get("loaded_at_str") or "Live Snapshot"
    market_str = gamma_data.get("market_trade_time")
    elapsed = gamma_data.get("elapsed_seconds", 0)
    market_part = f"Exchange Trade Print: {market_str} • " if market_str else ""

    fig.update_layout(
        title=dict(
            text=f"<b>{gamma_data['ticker']} Dealer Gamma Exposure (GEX) by Strike</b> | Net GEX: ${gamma_data['total_net_gex_billions']:+.2f}B ({gamma_data['regime_short']})<br><span style='font-size:10px; color:#94a3b8; font-weight:normal; font-family:monospace;'>{market_part}Feed Ingested: {loaded_str} (yfinance | {elapsed:.2f}s)</span>",
            font=dict(size=13, color="#f8fafc")
        ),
        xaxis_title="Strike Price ($)",
        yaxis_title="Dollar Gamma Exposure ($M / 1% move)",
        barmode="relative",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=40, r=40, t=65, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=390
    )
    
    # Auto-focus the default x-axis view on the relevant strikes window
    relevant_k = [s["strike"] for s in gamma_data.get("relevant_strikes", [])]
    if relevant_k:
        k_min = min(relevant_k)
        k_max = max(relevant_k)
        span = k_max - k_min
        pad = max(span * 0.08, spot * 0.015)
        fig.update_xaxes(range=[k_min - pad, k_max + pad])
        
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.05)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.05)")
    return fig

# ----------------- GAMMA TERRAIN & BALL PHYSICS ENGINE CHART -----------------
def plot_gamma_terrain_profile(gamma_data):
    """
    Generates the 'Gamma = The Ground, Price = The Ball' Physics Engine Chart:
    - Smooth continuous Net Gamma terrain curve across strike prices.
    - Positive Gamma Valley (+GEX in Emerald Green): Market makers dampen volatility ('Push it back').
    - Negative Gamma Slope (-GEX in Crimson Red): Market makers accelerate moves ('Push it').
    - Price Ball: Placed dynamically at current Spot price on the terrain.
    - Zero Gamma Crest: Inflection boundary where terrain tilts from valley into cliff.
    - Call Wall & Put Wall: Structural pins representing valley rims and floors.
    """
    if not go or not gamma_data or "strikes" not in gamma_data:
        return None

    strikes_list = gamma_data.get("strikes", [])
    if not strikes_list:
        return None

    spot = float(gamma_data["spot_price"])
    cw = float(gamma_data.get("call_wall", spot))
    pw = float(gamma_data.get("put_wall", spot))
    zg = float(gamma_data.get("zero_gamma_level", spot))
    ticker = gamma_data.get("ticker", "SPY")

    # Sort strikes by price
    sorted_s = sorted(strikes_list, key=lambda s: s["strike"])
    raw_k = np.array([s["strike"] for s in sorted_s], dtype=float)
    raw_net = np.array([s["net_gex"] for s in sorted_s], dtype=float)

    if len(raw_k) < 3:
        return None

    # Focus around relevant strikes or +/- 15% of spot
    relevant_k = [s["strike"] for s in gamma_data.get("relevant_strikes", [])]
    if relevant_k:
        k_min = min(relevant_k)
        k_max = max(relevant_k)
    else:
        k_min = max(spot * 0.85, raw_k[0])
        k_max = min(spot * 1.15, raw_k[-1])

    # Smooth dense sampling grid
    grid_min = max(raw_k[0], k_min - (spot * 0.04))
    grid_max = min(raw_k[-1], k_max + (spot * 0.04))
    dense_k = np.linspace(grid_min, grid_max, 300)
    dense_net = np.interp(dense_k, raw_k, raw_net)

    # Spot GEX interpolation (where the ball sits on the ground)
    spot_gex = float(np.interp(spot, raw_k, raw_net))
    zg_dist_pct = ((spot - zg) / spot) * 100 if zg > 0 else 0.0

    # Physics regime state of the ball
    is_at_crest = abs(spot - zg) / spot < 0.008 if zg > 0 else False
    is_in_valley = spot_gex >= 0 and not is_at_crest

    if is_at_crest:
        ball_state = "PERCHED ON THE CREST (Zero-Gamma Tipping Point)"
        ball_color = "#38bdf8"
        ball_border = "#ffffff"
    elif is_in_valley:
        ball_state = "COILED IN +GAMMA VALLEY (Shock Absorber Active)"
        ball_color = "#00F5A0"
        ball_border = "#ffffff"
    else:
        ball_state = "ROLLING DOWNHILL IN -GAMMA CHASM (Momentum Accelerator)"
        ball_color = "#FF2E55"
        ball_border = "#ffffff"

    # Split dense_net into positive (valley) and negative (slope) components for clean shading
    pos_terrain = np.maximum(dense_net, 0.0)
    neg_terrain = np.minimum(dense_net, 0.0)

    fig = go.Figure()

    # 1. Negative Gamma Slope (-GEX / Chasm)
    fig.add_trace(go.Scatter(
        x=dense_k, y=neg_terrain,
        mode="lines",
        name="Negative Gamma Chasm (Vol Accelerator)",
        line=dict(color="#FF2E55", width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(255, 46, 85, 0.20)",
        hoverinfo="skip"
    ))

    # 2. Positive Gamma Valley (+GEX / Shock Absorber)
    fig.add_trace(go.Scatter(
        x=dense_k, y=pos_terrain,
        mode="lines",
        name="Positive Gamma Valley (Shock Absorber)",
        line=dict(color="#00F5A0", width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(0, 245, 160, 0.20)",
        hoverinfo="skip"
    ))

    # 3. Ground Level Zero Line
    fig.add_hline(
        y=0, line_width=1.5, line_dash="solid", line_color="rgba(148, 163, 184, 0.4)",
        annotation_text="Ground Zero: $0 Net GEX Boundary",
        annotation_position="bottom right",
        annotation_font=dict(color="#64748b", size=9)
    )

    # 4. Actual Strikes Data Points
    fig.add_trace(go.Scatter(
        x=raw_k, y=raw_net,
        mode="markers",
        name="Strike Open Interest Net GEX ($M)",
        marker=dict(size=4.5, color="rgba(255, 255, 255, 0.35)", symbol="circle"),
        hovertemplate="<b>Strike:</b> $%{x:.1f}<br><b>Net GEX:</b> $%{y:+.1f}M<extra></extra>"
    ))

    # 5. THE BALL (Current Spot Price on the Ground)
    fig.add_trace(go.Scatter(
        x=[spot], y=[spot_gex],
        mode="markers",
        name=f"Price Ball (${spot:.2f})",
        marker=dict(
            size=18,
            color=ball_color,
            line=dict(width=3, color=ball_border),
            symbol="circle"
        ),
        hovertemplate=f"<b>⚪ PRICE BALL: ${spot:.2f}</b><br>Ground Elevation: {spot_gex:+.1f}M GEX<br>Status: {ball_state}<extra></extra>"
    ))

    # Ball Callout Annotation
    ay_offset = -60 if spot_gex < -150 else (-55 if spot_gex >= 0 else 55)
    fig.add_annotation(
        x=spot, y=spot_gex,
        text=f"⚪ <b>PRICE BALL: ${spot:.2f}</b><br><span style='font-size:9.5px;'>{ball_state}</span>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=ball_color,
        ax=0,
        ay=ay_offset,
        bgcolor="rgba(15, 23, 42, 0.94)",
        bordercolor=ball_color,
        borderwidth=1.5,
        borderpad=6,
        font=dict(color="#f8fafc", size=10, family="sans-serif")
    )

    # 6. Directional Dealer Hedging Vectors (Whiteboard Analogy)
    pos_indices = np.where(dense_net > 0)[0]
    if len(pos_indices) > 20:
        mid_idx = pos_indices[len(pos_indices) // 2]
        x_vec = dense_k[mid_idx]
        y_vec = max(float(dense_net[mid_idx] * 0.5), 40.0)
        fig.add_annotation(
            x=x_vec, y=y_vec,
            text="◀ <b>'PUSH IT BACK'</b> ▶<br><span style='font-size:8.5px;'>Dealers buy dips & sell rips<br>(Mean-reverting bowl)</span>",
            showarrow=False,
            bgcolor="rgba(0, 245, 160, 0.12)",
            bordercolor="rgba(0, 245, 160, 0.4)",
            borderwidth=1,
            font=dict(color="#00F5A0", size=9)
        )

    neg_indices = np.where(dense_net < 0)[0]
    if len(neg_indices) > 20:
        mid_idx_neg = neg_indices[len(neg_indices) // 2]
        x_vec_neg = dense_k[mid_idx_neg]
        y_vec_neg = min(float(dense_net[mid_idx_neg] * 0.4), -120.0)
        fig.add_annotation(
            x=x_vec_neg, y=y_vec_neg,
            text="▼ <b>'PUSH IT'</b> ▼<br><span style='font-size:8.5px;'>Dealers sell with momentum<br>(Fast & abrupt cascade)</span>",
            showarrow=False,
            bgcolor="rgba(255, 46, 85, 0.12)",
            bordercolor="rgba(255, 46, 85, 0.4)",
            borderwidth=1,
            font=dict(color="#FF2E55", size=9)
        )

    # 7. Structural Landmarks (Walls and Zero Flip)
    fig.add_vline(x=spot, line_width=1.5, line_dash="dash", line_color="#ffffff", opacity=0.6)

    if zg and abs(zg - spot) / spot < 0.35:
        fig.add_vline(
            x=zg, line_width=2, line_dash="dashdot", line_color="#38bdf8",
            annotation_text=f"⚡ Zero-Gamma Crest ${zg:.1f}", annotation_position="top left",
            annotation_font=dict(color="#38bdf8", size=9)
        )

    if cw and abs(cw - spot) / spot < 0.35:
        fig.add_vline(
            x=cw, line_width=1.5, line_dash="dot", line_color="#00F5A0",
            annotation_text=f"Call Wall Pin ${cw:.0f}", annotation_position="top right",
            annotation_font=dict(color="#00F5A0", size=9)
        )

    if pw and abs(pw - spot) / spot < 0.35:
        fig.add_vline(
            x=pw, line_width=1.5, line_dash="dot", line_color="#FF2E55",
            annotation_text=f"Put Wall Floor ${pw:.0f}", annotation_position="bottom left",
            annotation_font=dict(color="#FF2E55", size=9)
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{ticker} Gamma Ground Elevation vs Strike</b> | <i>'Gamma = The Ground, Price = The Ball'</i>",
            font=dict(size=13, color="#f8fafc")
        ),
        xaxis_title="Underlying Strike Price ($)",
        yaxis_title="Ground Elevation: Net Dealer Gamma ($M / 1% move)",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=40, r=40, t=55, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        height=410
    )

    if relevant_k:
        span = k_max - k_min
        pad = max(span * 0.08, spot * 0.015)
        fig.update_xaxes(range=[k_min - pad, k_max + pad])

    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.05)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.05)")
    return fig

# ----------------- REALIZED VOLATILITY CALCULATION (WITH DE-JUMPING) -----------------
def compute_realized_volatilities(df_history=None, ticker_symbol=None):
    """
    Computes annualized realized volatility across 10, 20, 30, and 60-day windows.
    Includes De-Jumped (Clean) HV that winsorizes >3σ outliers to prevent post-earnings distortion.
    If df_history is missing or too short, dynamically falls back to yfinance 3-month history.
    """
    if df_history is None or len(df_history) < 20:
        if ticker_symbol:
            try:
                hist = yf.Ticker(ticker_symbol).history(period="3mo")
                if not hist.empty and len(hist) >= 20:
                    df_history = hist.reset_index()
            except Exception:
                pass

    if df_history is None or len(df_history) < 20:
        return {'HV_10': 20.0, 'HV_20': 20.0, 'HV_30': 20.0, 'HV_30_clean': 20.0, 'HV_60': 20.0, 'Parkinson_30': 20.0}

    df = df_history.sort_values('Date').copy() if 'Date' in df_history.columns else df_history.copy()
    ret = df['Close'].pct_change().dropna()
    
    hv_10 = float(ret.tail(10).std() * np.sqrt(252) * 100) if len(ret) >= 10 else 20.0
    hv_20 = float(ret.tail(20).std() * np.sqrt(252) * 100) if len(ret) >= 20 else 20.0
    hv_30 = float(ret.tail(30).std() * np.sqrt(252) * 100) if len(ret) >= 30 else 20.0
    hv_60 = float(ret.tail(60).std() * np.sqrt(252) * 100) if len(ret) >= 60 else 20.0
    
    # De-Jumped HV: Winsorize outliers beyond 3 standard deviations (removes 1-day earnings gap spike)
    tail_30 = ret.tail(30).copy()
    med_std = tail_30.std()
    clean_tail_30 = tail_30.clip(lower=-3.0 * med_std, upper=3.0 * med_std)
    hv_30_clean = float(clean_tail_30.std() * np.sqrt(252) * 100)
    
    if 'High' in df.columns and 'Low' in df.columns:
        hl_ratio = np.log(df['High'] / df['Low'])
        parkinson_30 = float(np.sqrt((1.0 / (4.0 * np.log(2.0))) * (hl_ratio.tail(30)**2).mean()) * np.sqrt(252) * 100)
    else:
        parkinson_30 = hv_30

    return {
        'HV_10': round(hv_10, 2),
        'HV_20': round(hv_20, 2),
        'HV_30': round(hv_30, 2),
        'HV_30_clean': round(hv_30_clean, 2),
        'HV_60': round(hv_60, 2),
        'Parkinson_30': round(parkinson_30, 2)
    }

# ----------------- EARNINGS DATE DETECTOR -----------------
@st_cache
def get_next_earnings_date(ticker_symbol):
    """
    Fetches the next scheduled upcoming earnings report date from Yahoo Finance.
    Guarantees the returned date is strictly in the future (>= today).
    Returns None for ETFs (SPY, QQQ, etc.).
    """
    KNOWN_ETFS = {
        'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'USO', 'XLF', 'SMH', 'XLE', 
        'XLK', 'RSP', 'XBI', 'XLV', 'XLY', 'XLP', 'XLU', 'XLI', 'XLB', 'XLC', 'IBIT'
    }
    if ticker_symbol.upper() in KNOWN_ETFS:
        return None

    try:
        t = yf.Ticker(ticker_symbol)
        now = pd.Timestamp.now().normalize()

        # 1. First check official earnings_dates table (most reliable post-earnings transition)
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                idx = ed.index.tz_localize(None) if ed.index.tz is not None else ed.index
                upcoming = idx[idx >= now]
                if len(upcoming) > 0:
                    return pd.Timestamp(upcoming.min()).normalize()
        except Exception:
            pass

        # 2. Fall back to t.calendar, but ONLY if the date is strictly in the future (>= today)
        try:
            cal = t.calendar
            if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
                ed_list = cal['Earnings Date']
                for item in ed_list:
                    ts = pd.Timestamp(item).tz_localize(None).normalize()
                    if ts >= now:
                        return ts
        except Exception:
            pass
    except Exception:
        pass
    return None

# ----------------- TERM STRUCTURE SCANNER (WITH EVENT PARTITIONING) -----------------
@st_cache
def fetch_term_structure(ticker_symbol, hvs_dict=None, hv_benchmark=None):
    """
    Multithreaded fetch of active expiration chains.
    - Benchmarks against dynamic HV lookback.
    - Corrects calendar-day IV to trading-day basis (sqrt(252/365)).
    - Partitions contracts into Inter-Earnings (No IV crush) vs Spans Earnings.
    - Calculates 1-sigma Market Implied Expected Move.
    """
    if hvs_dict is None:
        if isinstance(hv_benchmark, (int, float)):
            hvs_dict = {'HV_10': hv_benchmark, 'HV_20': hv_benchmark, 'HV_30': hv_benchmark, 'HV_30_clean': hv_benchmark}
        else:
            hvs_dict = {'HV_10': 25.0, 'HV_20': 22.0, 'HV_30': 20.0, 'HV_30_clean': 20.0}
        
    next_earnings = get_next_earnings_date(ticker_symbol)
    
    try:
        t = yf.Ticker(ticker_symbol)
        expirations = t.options
        if not expirations:
            return pd.DataFrame(), None, next_earnings

        fast_info = t.fast_info
        spot_price = float(fast_info.last_price or fast_info.previous_close)
        today = pd.Timestamp.now().normalize()
        
        target_exps = [e for e in expirations if 8 <= (pd.Timestamp(e) - today).days <= 200]
        
        # Calendar to Trading-Day IV conversion factor: sqrt(252 / 365.25) = 0.8307
        CAL_TO_TRADING_FACTOR = np.sqrt(252.0 / 365.25)
        
        def process_expiration(exp):
            try:
                chain = t.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                if calls.empty or puts.empty:
                    return None
                
                exp_date = pd.Timestamp(exp)
                dte = (exp_date - today).days
                t_years = dte / 365.25
                
                atm_call = calls.iloc[(calls['strike'] - spot_price).abs().argsort()[:1]].iloc[0]
                atm_put = puts.iloc[(puts['strike'] - spot_price).abs().argsort()[:1]].iloc[0]
                
                raw_call_iv = float(atm_call.impliedVolatility * 100) if pd.notnull(atm_call.impliedVolatility) else 0.0
                raw_put_iv = float(atm_put.impliedVolatility * 100) if pd.notnull(atm_put.impliedVolatility) else 0.0
                
                # After-hours resilience: if feeds zero out IV overnight, invert Black-Scholes from lastPrice
                if raw_call_iv < 4.0 and getattr(atm_call, 'lastPrice', 0) > 0:
                    raw_call_iv = round(implied_vol_solver(atm_call.lastPrice, spot_price, atm_call.strike, t_years, is_call=True) * 100.0, 2)
                if raw_put_iv < 4.0 and getattr(atm_put, 'lastPrice', 0) > 0:
                    raw_put_iv = round(implied_vol_solver(atm_put.lastPrice, spot_price, atm_put.strike, t_years, is_call=False) * 100.0, 2)

                valid_ivs = [v for v in [raw_call_iv, raw_put_iv] if 4.0 <= v <= 250.0 and abs(v - 25.0007) > 0.01]
                if not valid_ivs:
                    return None
                raw_atm_iv = round(float(np.mean(valid_ivs)), 2)
                
                atm_iv_trading = raw_atm_iv * CAL_TO_TRADING_FACTOR
                
                # Market Implied Expected Move (1-sigma range: Spot * IV * sqrt(T))
                expected_move = round(spot_price * (raw_atm_iv / 100.0) * np.sqrt(t_years), 2)
                expected_move_pct = round((expected_move / spot_price) * 100.0, 1)
                lower_bound = round(spot_price - expected_move, 2)
                upper_bound = round(spot_price + expected_move, 2)
                
                spans_earnings = False
                if next_earnings is not None and next_earnings >= today:
                    spans_earnings = (exp_date >= next_earnings)
                    
                if dte < 18:
                    benchmark_hv = hvs_dict.get('HV_10', 25.0)
                    h_label = "10d HV"
                elif dte < 28:
                    benchmark_hv = hvs_dict.get('HV_20', 22.0)
                    h_label = "20d HV"
                else:
                    benchmark_hv = hvs_dict.get('HV_30_clean', hvs_dict.get('HV_30', 20.0))
                    h_label = "30d Clean HV"
                    
                # IV/HV Ratio and Variance Risk Premium (VRP) Calibration
                # Evaluated on true annualized market IV vs annualized HV
                ratio = round(raw_atm_iv / benchmark_hv, 2) if benchmark_hv > 0 else 1.0
                
                is_index_asset = ticker_symbol.upper() in ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'USO', 'XLF', 'SMH', 'XLE', 'XLK']
                vrp_factor = 1.12 if is_index_asset else 1.08
                fair_vrp_iv = round(benchmark_hv * vrp_factor, 2)
                vrp_ratio = round(raw_atm_iv / fair_vrp_iv, 2) if fair_vrp_iv > 0 else 1.0

                if spans_earnings:
                    event_tag = "Spans Earnings (Contains Event Jump)"
                    is_cheap = (ratio <= 1.25)
                    status = f"Event Cycle ({ratio}x {h_label})"
                else:
                    event_tag = "Inter-Earnings (No IV Crush Risk)"
                    if vrp_ratio < 0.92:
                        status = f"High Discount ({vrp_ratio}x VRP)"
                        is_cheap = True
                    elif vrp_ratio <= 1.08:
                        status = f"Fair Value ({vrp_ratio}x VRP)"
                        is_cheap = True
                    else:
                        status = f"🔴 Structurally Rich ({vrp_ratio}x VRP)"
                        is_cheap = False
                    
                return {
                    'Expiration': exp,
                    'DTE': dte,
                    'ATM_Strike': atm_call.strike,
                    'ATM_IV': raw_atm_iv,
                    'ATM_IV_Trading': round(atm_iv_trading, 2),
                    'Call_IV': round(raw_call_iv, 2),
                    'Put_IV': round(raw_put_iv, 2),
                    'IV_HV_Ratio': ratio,
                    'Benchmark_HV': benchmark_hv,
                    'VRP_Factor': vrp_factor,
                    'Fair_VRP_IV': fair_vrp_iv,
                    'VRP_Ratio': vrp_ratio,
                    'Expected_Move': expected_move,
                    'Expected_Move_Pct': expected_move_pct,
                    'Expected_Range': f"${lower_bound:.0f} - ${upper_bound:.0f}",
                    'Spans_Earnings': spans_earnings,
                    'Event_Tag': event_tag,
                    'Status': status,
                    'Is_Cheap': is_cheap,
                    'Call_Bid': atm_call.bid,
                    'Call_Ask': atm_call.ask,
                    'Put_Bid': atm_put.bid,
                    'Put_Ask': atm_put.ask
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_expiration, target_exps))
            
        rows = [r for r in results if r is not None]
        df_res = pd.DataFrame(rows).sort_values('DTE').reset_index(drop=True)
        return df_res, spot_price, next_earnings
    except Exception:
        return pd.DataFrame(), None, None

# ----------------- ASYMMETRIC LOG-MONEYNESS SKEW & GREEKS MODELER -----------------
@st_cache
def analyze_smile_and_skew(ticker_symbol, expiration, spot_price, direction="Bullish", side="Buy"):
    """
    Fetches option chain, cleans data, fits an Asymmetric Cubic Skew over Log-Moneyness ln(K/S),
    computes full Black-Scholes Greeks with dividend yield, and checks both Ask-IV slippage (Buy Side)
    and Bid-IV slippage (Sell Side).
    """
    try:
        t = yf.Ticker(ticker_symbol)
        chain = t.option_chain(expiration)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        
        min_strike = spot_price * 0.82
        max_strike = spot_price * 1.18
        
        today = pd.Timestamp.now().normalize()
        dte = max((pd.Timestamp(expiration) - today).days, 1)
        t_years = dte / 365.25
        r = 0.045  # ~4.5% risk-free rate proxy
        q = get_dividend_yield(ticker_symbol)  # continuous dividend yield

        # Outlier Filters & Per-Strike Greeks Extractor
        def clean_chain(df_sub, is_call=True):
            c = df_sub[(df_sub['strike'] >= min_strike) & (df_sub['strike'] <= max_strike)].copy()
            # After-hours resilience: If bids/asks are zeroed out by Yahoo overnight, synthesize from lastPrice row-by-row
            zero_bid_mask = (c['bid'] <= 0.0) | (c['ask'] <= 0.0)
            if 'lastPrice' in c.columns:
                valid_lp = zero_bid_mask & (c['lastPrice'] > 0)
                c.loc[valid_lp, 'bid'] = (c.loc[valid_lp, 'lastPrice'] * 0.98).round(2)
                c.loc[valid_lp, 'ask'] = (c.loc[valid_lp, 'lastPrice'] * 1.02).round(2)

            c = c[(c['bid'] > 0.0) & (c['ask'] > 0.0) & (c['ask'] >= c['bid'])]

            # If impliedVolatility is zeroed out by Yahoo overnight, invert Black-Scholes from lastPrice
            low_iv_mask = (c['impliedVolatility'] < 0.04) | (c['impliedVolatility'] > 1.8)
            if low_iv_mask.any() and 'lastPrice' in c.columns:
                c['impliedVolatility'] = c.apply(
                    lambda row: implied_vol_solver(row['lastPrice'], spot_price, row['strike'], t_years, r, q, is_call=is_call)
                    if (row['impliedVolatility'] < 0.04 or row['impliedVolatility'] > 1.8) and row.get('lastPrice', 0) > 0 else row['impliedVolatility'],
                    axis=1
                )

            c['IV_Pct'] = c['impliedVolatility'] * 100.0
            c = c[(c['IV_Pct'] >= 6.0) & (c['IV_Pct'] <= 160.0)]
            c = c[abs(c['IV_Pct'] - 25.0007) > 0.005]
            c = c.drop_duplicates(subset=['strike']).sort_values('strike')
            c['openInterest'] = pd.to_numeric(c['openInterest'], errors='coerce').fillna(0)
            c['volume'] = pd.to_numeric(c['volume'], errors='coerce').fillna(0)
            
            # Eliminate extreme bid-ask spread garbage (spread > 60% of mid)
            mid = (c['bid'] + c['ask']) / 2.0
            c = c[(c['ask'] - c['bid']) / mid.clip(lower=0.1) <= 0.60]
            if spot_price > 50:
                c = c[c['strike'].apply(lambda s: (round(s * 100) % 25 == 0) or (s % 1 == 0))]

            # Full Black-Scholes Greeks with dividend yield
            sig = c['impliedVolatility'].clip(0.05, 1.5)
            d1 = (np.log(spot_price / c['strike']) + (r - q + 0.5 * sig**2) * t_years) / (sig * np.sqrt(t_years))
            if is_call:
                c['Est_Delta'] = np.exp(-q * t_years) * norm_cdf(d1)
            else:
                c['Est_Delta'] = -np.exp(-q * t_years) * norm_cdf(-d1)
            c['Vega'] = (spot_price * np.exp(-q * t_years) * np.sqrt(t_years) * norm_pdf(d1)) / 100.0

            # Exact Black-Scholes IV Inversion for Ask and Bid
            ask_ivs = []
            bid_ivs = []
            for a_p, b_p, k in zip(c['ask'], c['bid'], c['strike']):
                a_iv = implied_vol_solver(a_p, spot_price, k, t_years, r=r, q=q, is_call=is_call) * 100.0 if a_p > 0 else 20.0
                b_iv = implied_vol_solver(b_p, spot_price, k, t_years, r=r, q=q, is_call=is_call) * 100.0 if b_p > 0 else 10.0
                ask_ivs.append(round(a_iv, 2))
                bid_ivs.append(round(b_iv, 2))
            c['Ask_IV_Pct'] = ask_ivs
            c['Bid_IV_Pct'] = bid_ivs
            c['Mid_Price'] = (c['bid'] + c['ask']) / 2.0
            c['Spread_Pct'] = (((c['ask'] - c['bid']) / c['Mid_Price'].clip(lower=0.1)) * 100.0).round(1)
            return c
            
        calls = clean_chain(calls, is_call=True)
        puts = clean_chain(puts, is_call=False)
        
        if calls.empty and puts.empty:
            return None

        # Determine ATM Strike and Baseline ATM Implied Volatility
        pool_for_atm = calls if not calls.empty else puts
        atm_strike = float(min(pool_for_atm['strike'], key=lambda k: abs(k - spot_price)))
        atm_call_iv = float(calls[calls['strike'] == atm_strike]['IV_Pct'].values[0]) if (calls['strike'] == atm_strike).any() else 25.0
        atm_put_iv = float(puts[puts['strike'] == atm_strike]['IV_Pct'].values[0]) if (puts['strike'] == atm_strike).any() else atm_call_iv
        atm_iv = round(float((atm_call_iv + atm_put_iv) / 2.0), 1)

        # Institutional OTM Stitching: OTM Puts (K <= Spot) and OTM Calls (K >= Spot)
        otm_puts = puts[puts['strike'] <= spot_price].copy() if not puts.empty else pd.DataFrame()
        if not otm_puts.empty:
            otm_puts = otm_puts[(otm_puts['IV_Pct'] >= atm_iv * 0.55) & (otm_puts['IV_Pct'] <= atm_iv * 2.8)]

        otm_calls = calls[calls['strike'] >= spot_price].copy() if not calls.empty else pd.DataFrame()
        if not otm_calls.empty:
            otm_calls = otm_calls[(otm_calls['IV_Pct'] >= atm_iv * 0.50) & (otm_calls['IV_Pct'] <= atm_iv * 2.8)]

        # Extract 25-Delta and 10-Delta structural markers
        put_25d = otm_puts.iloc[(otm_puts['Est_Delta'] - (-0.25)).abs().argsort()[:1]].iloc[0] if not otm_puts.empty else None
        call_25d = otm_calls.iloc[(otm_calls['Est_Delta'] - 0.25).abs().argsort()[:1]].iloc[0] if not otm_calls.empty else None
        put_10d = otm_puts.iloc[(otm_puts['Est_Delta'] - (-0.10)).abs().argsort()[:1]].iloc[0] if not otm_puts.empty else None

        # 6 Institutional Skew Metrics Calculation
        skew_25d = round(float(put_25d['IV_Pct'] - call_25d['IV_Pct']), 1) if (put_25d is not None and call_25d is not None) else 0.0
        fly_25d = round(float(((put_25d['IV_Pct'] + call_25d['IV_Pct']) / 2.0) - atm_iv), 1) if (put_25d is not None and call_25d is not None) else 0.0
        skew_10d = round(float(put_10d['IV_Pct'] - atm_iv), 1) if put_10d is not None else 0.0
        
        if put_25d is not None and call_25d is not None:
            pct_dist = ((call_25d['strike'] - put_25d['strike']) / spot_price) * 100.0
            skew_slope = round(float((call_25d['IV_Pct'] - put_25d['IV_Pct']) / max(pct_dist, 1.0)), 2)
            skew_25d_sub = f"{put_25d['strike']:g}P {put_25d['IV_Pct']:.1f}% vs {call_25d['strike']:g}C {call_25d['IV_Pct']:.1f}%"
        else:
            skew_slope = -0.1
            skew_25d_sub = "N/A"

        if skew_25d > 0.5:
            skew_25d_text = "Puts rich, downside bid"
        elif skew_25d < -0.5:
            skew_25d_text = "Calls rich, upside bid"
        else:
            skew_25d_text = "Flat skew / Neutral"

        if fly_25d >= 1.0:
            fly_25d_text = "Elevated wing curvature"
        elif fly_25d <= -1.0:
            fly_25d_text = "Inverted curvature"
        else:
            fly_25d_text = "Little curvature"

        if skew_10d >= 3.0:
            skew_10d_text = "Extreme crash hedge"
        elif skew_10d >= 1.2:
            skew_10d_text = "Crash-hedge tilt"
        else:
            skew_10d_text = "Moderate tail risk"

        # Fit Asymmetric Cubic Skew over Log-Moneyness ln(K/Spot)
        comb_list = []
        if not otm_puts.empty:
            comb_list.append(otm_puts[['strike', 'IV_Pct']])
        if not otm_calls.empty:
            comb_list.append(otm_calls[['strike', 'IV_Pct']])
            
        if comb_list:
            df_comb = pd.concat(comb_list).groupby('strike').mean().reset_index().sort_values('strike')
        else:
            df_comb = calls[['strike', 'IV_Pct']].drop_duplicates().sort_values('strike')

        grid_strikes = np.sort(np.unique(np.concatenate([np.linspace(df_comb['strike'].min(), df_comb['strike'].max(), 120), [spot_price]])))
        
        # Fit smooth parametric cubic skew over log-moneyness
        df_comb['m'] = np.log(df_comb['strike'] / spot_price)
        deg = min(3, len(df_comb) - 1)
        poly_coeffs = np.polyfit(df_comb['m'].values, df_comb['IV_Pct'].values, deg)
        grid_m = np.log(grid_strikes / spot_price)
        raw_grid_ivs = np.polyval(poly_coeffs, grid_m)
        grid_ivs = np.clip(raw_grid_ivs, df_comb['IV_Pct'].min() * 0.70, df_comb['IV_Pct'].max() * 1.35)

        # Split into seamless put (K <= Spot) and call (K >= Spot) curves meeting at Spot
        put_mask = grid_strikes <= spot_price
        call_mask = grid_strikes >= spot_price
        otm_put_curve = {
            'strikes': [round(float(s), 2) for s in grid_strikes[put_mask]],
            'iv': [round(float(iv), 2) for iv in grid_ivs[put_mask]]
        }
        otm_call_curve = {
            'strikes': [round(float(s), 2) for s in grid_strikes[call_mask]],
            'iv': [round(float(iv), 2) for iv in grid_ivs[call_mask]]
        }
        otm_puts_market = {
            'strikes': [round(float(s), 2) for s in otm_puts['strike']] if not otm_puts.empty else [],
            'iv': [round(float(iv), 2) for iv in otm_puts['IV_Pct']] if not otm_puts.empty else []
        }
        otm_calls_market = {
            'strikes': [round(float(s), 2) for s in otm_calls['strike']] if not otm_calls.empty else [],
            'iv': [round(float(iv), 2) for iv in otm_calls['IV_Pct']] if not otm_calls.empty else []
        }

        # Execution Desk Target Filtering
        side_clean = "Sell" if "Sell" in str(side) else "Buy"
        if "Call" in str(direction):
            is_call = True
        elif "Put" in str(direction):
            is_call = False
        else:
            is_call = (direction == "Bullish") if side_clean == "Buy" else (direction == "Bearish")
        target_df = (calls if is_call else puts).copy()
        if len(target_df) < 5:
            return None
        
        target_df['Log_Moneyness'] = np.log(target_df['strike'] / spot_price)
        
        # Evaluate Fair Fitted Skew using the smooth log-moneyness cubic model
        fitted = np.polyval(poly_coeffs, target_df['Log_Moneyness'].values)
        target_df['Fitted_IV'] = np.clip(fitted, target_df['IV_Pct'].min() * 0.70, target_df['IV_Pct'].max() * 1.35)
        target_df['IV_Discount'] = (target_df['Fitted_IV'] - target_df['IV_Pct']).round(2)  # Midpoint discount (cheapness)
        target_df['IV_Premium'] = (target_df['IV_Pct'] - target_df['Fitted_IV']).round(2)   # Midpoint premium (richness)
        
        # Spread Quality Rating
        def classify_spread(sp):
            if sp <= 5.0:
                return "Tight"
            elif sp <= 15.0:
                return "Fair"
            else:
                return "🔴 Wide"
        target_df['Spread_Quality'] = target_df['Spread_Pct'].apply(classify_spread)
        
        # Compute Ask_Edge and Bid_Edge with bounded outlier clipping
        raw_ask_edge = target_df['Fitted_IV'] - target_df['Ask_IV_Pct']
        target_df['Ask_Edge'] = np.clip(raw_ask_edge, -6.0, 6.0).round(2)
        raw_bid_edge = target_df['Bid_IV_Pct'] - target_df['Fitted_IV']
        target_df['Bid_Edge'] = np.clip(raw_bid_edge, -6.0, 6.0).round(2)
        
        # Expected Move Context
        exp_move = spot_price * (atm_iv / 100.0) * np.sqrt(t_years)
        target_df['Dist_From_Spot'] = target_df['strike'] - spot_price
        target_df['Within_1_Sigma'] = target_df['Dist_From_Spot'].abs() <= exp_move
        
        # Tradable candidate filter: reject illiquid spreads > 35%
        tradable = target_df[target_df['Spread_Pct'] <= 35.0].copy()
        if tradable.empty:
            tradable = target_df.copy()
            
        if side_clean == "Sell":
            # Target prime credit delta ~ 0.25 OTM (negative delta for Puts, positive for Calls)
            ideal_delta = -0.25 if not is_call else 0.25
            if not is_call: # Bullish: selling Puts
                delta_band = (tradable['Est_Delta'] <= -0.12) & (tradable['Est_Delta'] >= -0.45)
            else: # Bearish: selling Calls
                delta_band = (tradable['Est_Delta'] >= 0.12) & (tradable['Est_Delta'] <= 0.45)
                
            tradable['Effective_Liquidity'] = tradable[['openInterest', 'volume']].max(axis=1).fillna(0)
            liquid_band = (tradable['Effective_Liquidity'] >= 20) & (tradable['bid'] > 0.05)
            candidates = tradable[delta_band & liquid_band].copy()
            if candidates.empty:
                active_fallback = tradable[delta_band & (tradable['Effective_Liquidity'] > 0)]
                candidates = active_fallback.copy() if not active_fallback.empty else (tradable[delta_band].copy() if not tradable[delta_band].empty else tradable.copy())
                
            candidates['Composite_Score'] = (
                (candidates['Bid_Edge'] * 3.5)
                + (candidates['IV_Premium'] * 1.5)
                - (abs(candidates['Est_Delta'] - ideal_delta) * 14.0)
                + np.log10(candidates['Effective_Liquidity'].clip(lower=10)) * 1.8
                - (candidates['Spread_Pct'].clip(upper=25.0) * 0.15)
            )
            
            # Companion Defined-Risk Wing Calculation (Credit Spread Hedge)
            wing_pool = puts if not is_call else calls
            max_w = 10.0 if ticker_symbol.upper() in ['SPY', 'QQQ', 'IWM', 'DIA'] else 5.0
            target_width = min(max_w, max(spot_price * 0.015, 1.0))
            
            def calc_wing(row):
                s_k = row['strike']
                s_bid = row['bid']
                if not is_call: # Bull Put Spread: Wing is lower Put strike
                    pool = wing_pool[wing_pool['strike'] < s_k]
                    if pool.empty:
                        return pd.Series([None, None, None, None, None, None])
                    t_k = s_k - target_width
                    idx = (pool['strike'] - t_k).abs().argsort()[:1]
                    w_row = pool.iloc[idx].iloc[0]
                else: # Bear Call Spread: Wing is higher Call strike
                    pool = wing_pool[wing_pool['strike'] > s_k]
                    if pool.empty:
                        return pd.Series([None, None, None, None, None, None])
                    t_k = s_k + target_width
                    idx = (pool['strike'] - t_k).abs().argsort()[:1]
                    w_row = pool.iloc[idx].iloc[0]
                    
                w_k = float(w_row['strike'])
                w_ask = float(w_row['ask'])
                width = round(abs(s_k - w_k), 2)
                net_diff = s_bid - w_ask
                if net_diff <= 0 or width <= 0:
                    return pd.Series([None, None, None, None, None, None])
                credit = round(net_diff, 2)
                risk = round(max(0.01, width - credit), 2)
                ror = round((credit / risk) * 100.0, 1)
                return pd.Series([w_k, w_ask, width, credit, risk, ror])
                
            wing_cols = ['Wing_Strike', 'Wing_Ask', 'Spread_Width', 'Net_Credit', 'Max_Risk', 'RoR_Pct']
            candidates[wing_cols] = candidates.apply(calc_wing, axis=1)
        else:
            # Prime Delta band for buyers: ~0.25 - 0.55 (sweet spot around ~0.40)
            ideal_delta = 0.40 if is_call else -0.40
            if is_call:
                delta_band = (tradable['Est_Delta'] >= 0.20) & (tradable['Est_Delta'] <= 0.65)
            else:
                delta_band = (tradable['Est_Delta'] <= -0.20) & (tradable['Est_Delta'] >= -0.65)
                
            tradable['Effective_Liquidity'] = tradable[['openInterest', 'volume']].max(axis=1).fillna(0)
            liquid_band = (tradable['Effective_Liquidity'] >= 20) & (tradable['ask'] > 0)
            candidates = tradable[delta_band & liquid_band].copy()
            if candidates.empty:
                active_fallback = tradable[delta_band & (tradable['Effective_Liquidity'] > 0)]
                candidates = active_fallback.copy() if not active_fallback.empty else (tradable[delta_band].copy() if not tradable[delta_band].empty else tradable.copy())
                
            candidates['Composite_Score'] = (
                (candidates['Ask_Edge'] * 3.0)
                + (candidates['IV_Discount'] * 1.5)
                - (abs(candidates['Est_Delta'] - ideal_delta) * 14.0)
                + np.log10(candidates['Effective_Liquidity'].clip(lower=10)) * 1.8
                - (candidates['Spread_Pct'].clip(upper=25.0) * 0.15)
            )
            candidates['Wing_Strike'] = None
            candidates['Wing_Ask'] = None
            candidates['Spread_Width'] = None
            candidates['Net_Credit'] = None
            candidates['Max_Risk'] = None
            candidates['RoR_Pct'] = None
            
        top_strikes = candidates.sort_values('Composite_Score', ascending=False).head(4)
        
        metrics = {
            'atm_iv': atm_iv,
            'atm_strike': atm_strike,
            'skew_25d': skew_25d,
            'skew_25d_text': skew_25d_text,
            'skew_25d_sub': skew_25d_sub,
            'fly_25d': fly_25d,
            'fly_25d_text': fly_25d_text,
            'skew_10d': skew_10d,
            'skew_10d_text': skew_10d_text,
            'skew_slope': skew_slope,
            'skew_slope_text': "IV pts per 1% of strike",
            'total_contracts': int(len(chain.calls) + len(chain.puts)) if ('chain' in locals() and hasattr(chain, 'calls')) else int(len(calls) + len(puts))
        }
        structural_markers = {
            'atm': {'strike': atm_strike, 'iv': atm_iv},
            'put_25d': {'strike': float(put_25d['strike']), 'iv': float(put_25d['IV_Pct'])} if put_25d is not None else None,
            'call_25d': {'strike': float(call_25d['strike']), 'iv': float(call_25d['IV_Pct'])} if call_25d is not None else None,
            'put_10d': {'strike': float(put_10d['strike']), 'iv': float(put_10d['IV_Pct'])} if put_10d is not None else None
        }

        return {
            'calls': calls,
            'puts': puts,
            'target_df': target_df,
            'top_strikes': top_strikes,
            'spot_price': spot_price,
            'dte': dte,
            'expiration': expiration,
            'poly_coeffs_k': [round(float(c), 4) for c in poly_coeffs] if 'poly_coeffs' in locals() else [0, 0, 0],
            'exp_move': exp_move,
            'dividend_yield': q,
            'side': side_clean,
            'metrics': metrics,
            'structural_markers': structural_markers,
            'otm_put_curve': otm_put_curve,
            'otm_call_curve': otm_call_curve,
            'otm_puts_market': otm_puts_market,
            'otm_calls_market': otm_calls_market
        }
    except Exception:
        return None

# ----------------- CROSS-CHAIN RECOMMENDATION SCANNER -----------------
@st_cache
def scan_cross_chain_recommendations(ticker_symbol, cheap_expirations_list, spot_price, direction="Bullish", side="Buy", return_all=False):
    """
    Scans across date chains simultaneously.
    If side == 'Buy': identifies underpriced contracts surviving the Ask spread.
    If side == 'Sell': identifies overpriced contracts surviving the Bid spread (harvesting rich premium).
    Guarantees diversity: picks top contracts across DIFFERENT expiration dates!
    """
    all_candidates = []
    side_clean = "Sell" if "Sell" in str(side) else "Buy"
    
    # If Sell, sort expirations by highest IV/HV ratio first (harvesting richest volatility)
    # If Buy, sort by lowest IV/HV ratio first (buying cheapest relative volatility)
    sorted_exps = sorted(cheap_expirations_list[:12], key=lambda x: x.get('IV_HV_Ratio', 1.0), reverse=(side_clean == "Sell"))
    
    for exp_dict in sorted_exps:
        exp = exp_dict['Expiration']
        dte = exp_dict['DTE']
        term_ratio = exp_dict.get('IV_HV_Ratio', 1.0)
        event_tag = exp_dict.get('Event_Tag', 'Inter-Earnings')
        spans_earnings = exp_dict.get('Spans_Earnings', False)
        
        res = analyze_smile_and_skew(ticker_symbol, exp, spot_price, direction=direction, side=side_clean)
        if res and not res['top_strikes'].empty:
            for _, s_row in res['top_strikes'].iterrows():
                cand = s_row.to_dict()
                cand['Expiration'] = exp
                cand['DTE'] = dte
                cand['IV_HV_Ratio'] = term_ratio
                cand['Event_Tag'] = event_tag
                cand['Spans_Earnings'] = spans_earnings
                cand['Side'] = side_clean
                
                oi = float(cand.get('openInterest', 0) if pd.notnull(cand.get('openInterest')) else 0)
                vol = float(cand.get('volume', 0) if pd.notnull(cand.get('volume')) else 0)
                liq = max(oi, vol)
                cand['Effective_Liquidity'] = liq
                
                is_call_cand = ("Call" in str(direction)) if ("Call" in str(direction) or "Put" in str(direction)) else ((direction == "Bullish") if side_clean == "Buy" else (direction == "Bearish"))
                if side_clean == "Sell":
                    ideal_d = 0.25 if is_call_cand else -0.25
                    # Composite Cross-Chain Score for sellers: rewards high term ratio, Bid-Edge, and IV-Premium
                    cand['Cross_Chain_Score'] = (
                        (term_ratio - 0.95) * 16.0
                        + (cand.get('Bid_Edge', 0) * 3.5)
                        + (cand.get('IV_Premium', 0) * 1.5)
                        - (abs(cand.get('Est_Delta', 0) - ideal_d) * 14.0)
                        + np.log10(max(liq, 10.0)) * 2.0
                        - (cand.get('Spread_Pct', 0) * 0.1)
                    )
                else:
                    ideal_d = 0.40 if is_call_cand else -0.40
                    earnings_penalty = 1.5 if spans_earnings else 0.0
                    cand['Cross_Chain_Score'] = (
                        (1.35 - term_ratio) * 18.0
                        + (cand.get('Ask_Edge', 0) * 3.0)
                        + (cand.get('IV_Discount', 0) * 1.5)
                        - (abs(cand.get('Est_Delta', 0) - ideal_d) * 14.0)
                        + np.log10(max(liq, 10.0)) * 2.0
                        - (cand.get('Spread_Pct', 0) * 0.1)
                        - earnings_penalty
                    )
                all_candidates.append(cand)
                
    if not all_candidates:
        return ([], []) if return_all else []
        
    df_cands = pd.DataFrame(all_candidates).sort_values('Cross_Chain_Score', ascending=False)
    
    diverse_picks = []
    seen_exps = set()
    
    for _, row in df_cands.iterrows():
        if row['Expiration'] not in seen_exps:
            diverse_picks.append(row.to_dict())
            seen_exps.add(row['Expiration'])
        if len(diverse_picks) >= 4:
            break
            
    diverse_picks = sorted(diverse_picks, key=lambda x: x['DTE'])

    # Format Whole-Chain Tiered Discount Table (Top 10–12 across the whole chain)
    sort_key = 'Ask_Edge' if side_clean == 'Buy' else 'Bid_Edge'
    df_sorted = df_cands.sort_values([sort_key, 'Effective_Liquidity'], ascending=[False, False]).reset_index(drop=True)
    
    ranked_table = []
    for i, row in df_sorted.head(10).iterrows():
        tier_num = 1 if i < 3 else (2 if i < 7 else 3)
        tier_label = 'Tier 1 • Prime' if tier_num == 1 else ('Tier 2 • High' if tier_num == 2 else 'Tier 3 • Selective')
        # Determine Call vs Put from symbol, delta, and direction
        sym = str(row.get('contractSymbol', ''))
        if len(sym) >= 9 and sym[-9] in ['C', 'P']:
            c_type = "Call" if sym[-9] == 'C' else "Put"
        elif pd.notnull(row.get('Est_Delta')):
            c_type = "Call" if float(row['Est_Delta']) > 0 else "Put"
        else:
            c_type = "Call" if ("Call" in str(direction) or str(direction) == "Bullish") else "Put"
        
        ranked_table.append({
            'rank': int(i + 1),
            'tier': tier_label,
            'tier_num': tier_num,
            'expiration': str(row['Expiration']),
            'dte': int(row['DTE']),
            'strike': float(row['strike']),
            'type': c_type,
            'ask_cost': round(float(row.get('ask', 0)), 2),
            'bid_cost': round(float(row.get('bid', 0)), 2),
            'iv': round(float(row.get('IV_Pct', 0)), 1),
            'oi': int(row.get('openInterest', 0)) if pd.notnull(row.get('openInterest')) else 0,
            'volume': int(row.get('volume', 0)) if pd.notnull(row.get('volume')) else 0,
            'ask_edge': round(float(row.get('Ask_Edge', 0)), 2),
            'bid_edge': round(float(row.get('Bid_Edge', 0)), 2),
            'delta': round(float(row.get('Est_Delta', 0)), 2),
            'spans_earnings': bool(row.get('Spans_Earnings', False))
        })

    if return_all:
        return diverse_picks, ranked_table
    return diverse_picks

# ----------------- STREAMLIT UI RENDER FUNCTION -----------------
def render_options_vol_page(available_tickers, load_ticker_data_func):
    """
    Renders the Institutional Volatility Pricing Desk inside Streamlit.
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
                padding: 1.25rem 1.75rem; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 1.5rem;">
        <h2 style="margin: 0; color: #f8fafc; font-size: 1.6rem; font-weight: 700;">
            Institutional Volatility Pricing Desk
        </h2>
        <p style="margin: 0.35rem 0 0 0; color: #94a3b8; font-size: 0.95rem;">
            Institutional volatility architecture: Switch between <b>Buy Side (Ask Edge)</b> for underpriced cheap IV and <b>Sell Side (Bid Edge)</b> for overpriced premium harvesting. De-jumps historical volatility, models Asymmetric Log-Moneyness Skew, and executes full Bid/Ask spread inversion.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Top Controls Bar
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.8, 1.8])
    
    universe = ["QQQ", "SPY", "NVDA", "AAPL", "AMZN", "MSFT", "META", "TSLA", "IWM", "GLD", "XLF", "SMH"]
    for t in available_tickers:
        if t not in universe:
            universe.append(t)
            
    with ctrl_col1:
        selected_ticker = st.selectbox("Underlying Ticker", universe, index=0)
        
    with ctrl_col2:
        side_sel = st.radio("Execution Side", ["Buy Side (Ask Edge)", "🔴 Sell Side (Bid Edge)"], horizontal=True)
        side_clean = "Sell" if "Sell" in side_sel else "Buy"

    with ctrl_col3:
        direction = st.radio("Contract Type", ["Calls", "Puts"], horizontal=True)
        dir_clean = "Calls" if "Call" in direction else "Puts"

    # 2. Historical Realized Volatility Baseline (with De-Jumping)
    df_hist = load_ticker_data_func(selected_ticker)
    hvs = compute_realized_volatilities(df_hist, ticker_symbol=selected_ticker)
    
    # 3. Pull Term Structure & Upcoming Earnings
    with st.spinner(f"Scanning option chains, IV surface, and earnings dates for {selected_ticker}..."):
        term_df, spot_price, next_earnings = fetch_term_structure(selected_ticker, hvs_dict=hvs)
        
    if term_df.empty or spot_price is None:
        st.warning(f"Unable to retrieve live option chains for {selected_ticker}. Please verify ticker or try another symbol.")
        return

    # Realized Volatility KPI Strip
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.metric("Spot Price", f"${spot_price:.2f}")
    with kpi_col2:
        st.metric("10-Day Realized Vol", f"{hvs['HV_10']:.1f}%")
    with kpi_col3:
        st.metric("20-Day Realized Vol", f"{hvs['HV_20']:.1f}%")
    with kpi_col4:
        st.metric("30-Day Clean HV", f"{hvs['HV_30_clean']:.1f}%", help="De-jumped: strips 1-day earnings gap outliers.")
    with kpi_col5:
        if next_earnings:
            ed_str = next_earnings.strftime('%b %d, %Y')
            st.metric("Next Earnings", ed_str, help="Upcoming scheduled earnings report.")
        else:
            st.metric("Asset Type", "Macro Index / ETF", help="Continuous index with no single earnings report.")
        
    st.markdown("---")

    # 4. Scope Selection: Cross-Chain vs. Specific Date
    if side_clean == "Sell":
        chains_df = term_df[term_df['DTE'] >= 8].copy()
    else:
        has_cheap = term_df['Is_Cheap'].any()
        if not has_cheap:
            chains_df = term_df[term_df['DTE'] >= 8].copy()
        else:
            chains_df = term_df[term_df['Is_Cheap']].copy()

    rec_mode = st.radio(
        "Recommendation Horizon Scope:",
        [
            "Cross-Chain Scanner (Recommends Different Expiration Dates & Strikes)",
            "Single Expiration Focus (Recommends Strikes for One Selected Date)"
        ],
        horizontal=True
    )

    # ----------------- SECTION A: MULTI-DATE CROSS-CHAIN PICKS -----------------
    if "Cross-Chain" in rec_mode:
        if side_clean == "Sell":
            st.markdown("### Top Recommended Overpriced Premium Sells (Bid Edge)")
            st.caption(f"Cross-chain scan for **{selected_ticker} ({dir_clean}) [SELL SIDE]**: Identifies contracts trading at an elevated IV above model skew even after crossing the spread to sell at the Bid, capturing high variance premium.")
        else:
            st.markdown("### Top Recommended Cheap IV Contracts Across Different Expirations")
            st.caption(f"Cross-chain scan for **{selected_ticker} ({dir_clean}) [BUY SIDE]**: Evaluates term structure cheapness ($IV/HV$), Ask-Edge (edge after paying the full Ask price), delta sweet spot (~0.38–0.45), and liquidity across diverse dates.")

        with st.spinner("Scanning volatility smiles and skew across date chains..."):
            chains_list = chains_df.to_dict('records')
            cross_recommendations, ranked_chain = scan_cross_chain_recommendations(selected_ticker, chains_list, spot_price, direction=dir_clean, side=side_clean, return_all=True)

        if cross_recommendations:
            rec_cols = st.columns(len(cross_recommendations))
            for i, rec in enumerate(cross_recommendations):
                with rec_cols[i]:
                    is_call_rec = ("Call" in dir_clean)
                    c_type = "Call" if is_call_rec else "Put"
                    
                    if rec['DTE'] <= 22:
                        h_tag = "Tactical (10–21d)"
                        badge_color = "#38bdf8"
                    elif rec['DTE'] <= 45:
                        h_tag = "Fast Swing (25–45d)"
                        badge_color = "#a855f7"
                    elif rec['DTE'] <= 70:
                        h_tag = "Core Monthly (60d)"
                        badge_color = "#e3b341"
                    else:
                        h_tag = "Quarterly (75d+)"
                        badge_color = "#00F5A0"

                    event_str = "Spans Earnings" if rec.get('Spans_Earnings') else "Inter-Earnings (Safe)"
                    event_bg = "rgba(255, 46, 85, 0.15)" if rec.get('Spans_Earnings') else "rgba(0, 245, 160, 0.1)"
                    event_color = "#FF2E55" if rec.get('Spans_Earnings') else "#00F5A0"
                    prob_tag = "Inside 1σ" if rec.get('Within_1_Sigma') else "Outside 1σ"
                    spread_badge = rec.get('Spread_Quality', 'Tight')

                    if side_clean == "Sell":
                        edge_val = rec.get('Bid_Edge', 0)
                        b_color = "#00F5A0" if edge_val >= 0 else "#eab308"
                        b_sign = "+" if edge_val >= 0 else ""
                        bid_p = rec.get('bid', 0)
                        cushion = abs(rec['strike'] - spot_price) / spot_price * 100.0
                        pick_badge = f"<span style='background: rgba(255, 46, 85, 0.15); color: #FF2E55; font-size: 0.72rem; padding: 2px 7px; border-radius: 6px; font-weight: 600;'>Pick #{i+1} • Sell</span>"
                        action_html = f"<div>Bid (Credit): <b style='color: #00F5A0;'>${bid_p:.2f}</b> ($<b>{bid_p*100:.0f}</b>/ct collected)</div>"
                        edge_html = f"<div><span style='color: #8b949e;'>Bid Edge:</span> <b style='color: {b_color};'>{b_sign}{edge_val:.2f}%</b></div>"
                        iv_html = f"<div><span style='color: #8b949e;'>Bid IV:</span> <b>{rec.get('Bid_IV_Pct', rec['IV_Pct']):.1f}%</b></div>"
                        extra_html = f"<div>Safety Cushion: <b>{cushion:.1f}% buffer</b> | {spread_badge}</div>"
                        
                        wing_k = rec.get('Wing_Strike')
                        if wing_k is not None and pd.notnull(wing_k):
                            net_cr = rec.get('Net_Credit', 0)
                            max_rk = rec.get('Max_Risk', 0)
                            ror = rec.get('RoR_Pct', 0)
                            wing_box = f"<div style='margin-top:6px; padding:6px; background:rgba(255,255,255,0.03); border-radius:6px; border:1px dashed #30363d; font-size:0.75rem;'><b>Hedged Spread Wing</b>: +${wing_k:.0f} {c_type}<br>Credit: <b>${net_cr:.2f}</b> | Max Risk: <b>${max_rk:.2f}</b> | RoR: <b style='color:#00F5A0;'>{ror:.1f}%</b></div>"
                        else:
                            wing_box = ""
                    else:
                        edge_val = rec.get('Ask_Edge', 0)
                        b_color = "#00F5A0" if edge_val >= 0 else "#eab308"
                        b_sign = "+" if edge_val >= 0 else ""
                        be_val = rec['strike'] + rec['ask'] if is_call_rec else rec['strike'] - rec['ask']
                        be_diff = ((be_val / spot_price) - 1) * 100
                        pick_badge = f"<span style='background: rgba(0, 245, 160, 0.15); color: #00F5A0; font-size: 0.72rem; padding: 2px 7px; border-radius: 6px; font-weight: 600;'>Pick #{i+1} • Buy</span>"
                        action_html = f"<div>Ask (Purchase): <b style='color: #00F5A0;'>${rec['ask']:.2f}</b> ($<b>{rec['ask']*100:.0f}</b>/ct)</div>"
                        edge_html = f"<div><span style='color: #8b949e;'>Ask Edge:</span> <b style='color: {b_color};'>{b_sign}{edge_val:.2f}%</b></div>"
                        iv_html = f"<div><span style='color: #8b949e;'>Ask IV:</span> <b>{rec.get('Ask_IV_Pct', rec['IV_Pct']):.1f}%</b></div>"
                        extra_html = f"<div>Breakeven: <b>${be_val:.2f} ({be_diff:+.1f}%)</b> | {spread_badge}</div>"
                        wing_box = ""

                    st.markdown(f"""
                    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-bottom: 15px;
                                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-size: 1.25rem; font-weight: 800; color: #58a6ff;">${rec['strike']:.0f} {c_type}</span>
                            {pick_badge}
                        </div>
                        <div style="display: flex; gap: 6px; margin-bottom: 8px;">
                            <span style="font-size: 0.75rem; color: {badge_color}; font-weight: 600;">{h_tag}</span>
                            <span style="font-size: 0.72rem; background: {event_bg}; color: {event_color}; padding: 1px 5px; border-radius: 4px;">{event_str}</span>
                        </div>
                        <div style="font-size: 0.84rem; color: #f0f6fc; margin-bottom: 10px; background: rgba(255,255,255,0.03); padding: 4px 8px; border-radius: 6px; display: flex; justify-content: space-between;">
                            <span><b>{rec['Expiration']}</b> ({rec['DTE']}d)</span>
                            <span style="color: #8b949e;">{prob_tag}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.84rem; margin-bottom: 10px;">
                            {iv_html}
                            <div><span style="color: #8b949e;">IV/HV:</span> <b>{rec['IV_HV_Ratio']}x</b></div>
                            {edge_html}
                            <div><span style="color: #8b949e;">Delta:</span> <b>{rec['Est_Delta']:.2f}</b></div>
                        </div>
                        <div style="border-top: 1px solid #21262d; padding-top: 8px; font-size: 0.82rem; color: #8b949e;">
                            {action_html}
                            <div>Bid / Mid: ${rec['bid']:.2f} / ${rec['Mid_Price']:.2f}</div>
                            {extra_html}
                            <div>Open Interest: <b>{int(rec.get('openInterest', 0) or 0):,}</b> {f"<span style='font-size:10px; color:#00F5A0;'>({int(rec.get('volume', 0) or 0):,} Vol • OCC Overnight Reset)</span>" if (int(rec.get('openInterest', 0) or 0) == 0 and int(rec.get('volume', 0) or 0) > 0) else (f"<span style='font-size:10px; color:#94a3b8;'>({int(rec.get('volume', 0) or 0):,} Vol)</span>" if int(rec.get('volume', 0) or 0) > 0 else "")}</div>
                            {wing_box}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ----------------- SECTION B: VOLATILITY SMILE & SKEW INSPECTOR -----------------
    st.markdown("---")
    st.markdown("### Volatility Smile & Skew Inspector")
    st.caption("Inspect the exact smile, fitted asymmetric log-moneyness skew curve, expected move range, and value strikes for any chosen expiration date:")

    exp_col1, exp_col2 = st.columns([2.5, 1.5])
    
    display_df = chains_df.copy()
    exp_options = display_df['Expiration'].tolist()
    exp_labels = [f"{row['Expiration']}  |  {row['DTE']}d DTE  |  ATM IV: {row['ATM_IV']}%  |  IV/HV: {row['IV_HV_Ratio']}x  |  Expected Move: ±{row['Expected_Move_Pct']}%" 
                  for _, row in display_df.iterrows()]
    
    with exp_col1:
        selected_idx = st.selectbox(
            "Selected Expiration Chain to Inspect:",
            range(len(exp_options)),
            index=0,
            format_func=lambda i: exp_labels[i]
        )
        chosen_exp = exp_options[selected_idx]
        chosen_row = display_df.iloc[selected_idx]

    with exp_col2:
        ev_color = "#FF2E55" if chosen_row['Spans_Earnings'] else "#00F5A0"
        st.markdown(f"""
        <div style="background: #161b22; padding: 12px 18px; border-radius: 8px; border: 1px solid #30363d; margin-top: 1.6rem;">
            <div style="font-size: 0.85rem; color: #8b949e;">Expected Move (1σ Bound)</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #58a6ff;">±${chosen_row['Expected_Move']:.2f} (±{chosen_row['Expected_Move_Pct']}%)</div>
            <div style="font-size: 0.84rem; color: #8b949e; margin-top: 2px;">Range: <b style="color: #f0f6fc;">{chosen_row['Expected_Range']}</b></div>
            <div style="font-size: 0.80rem; color: {ev_color}; margin-top: 3px;">{chosen_row['Event_Tag']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Build Smile & Skew for the selected expiration
    with st.spinner(f"Rendering Asymmetric Log-Moneyness Volatility Smile for {chosen_exp}..."):
        smile_data = analyze_smile_and_skew(selected_ticker, chosen_exp, spot_price, direction=dir_clean, side=side_clean)
        
    if not smile_data:
        st.warning(f"Could not construct a smooth volatility smile for {chosen_exp}. Please select an adjacent expiration.")
        return

    target_df = smile_data['target_df']
    top_strikes = smile_data['top_strikes']
    
    # 6 Institutional Skew KPI Metric Cards
    metrics = smile_data.get('metrics', {})
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.metric("ATM IV", f"{metrics.get('atm_iv', 0):.1f}%", f"Strike {metrics.get('atm_strike', 0):.0f}")
    with m_col2:
        s25 = metrics.get('skew_25d', 0)
        s25_color = "inverse" if s25 > 0 else "normal"
        st.metric("25Δ SKEW", f"{s25:+.1f} pts", metrics.get('skew_25d_text', 'Risk Reversal'), delta_color=s25_color)
    with m_col3:
        st.metric("25Δ BUTTERFLY", f"{metrics.get('fly_25d', 0):+.1f} pts", metrics.get('fly_25d_text', 'Curvature'))
    with m_col4:
        st.metric("10Δ SKEW", f"{metrics.get('skew_10d', 0):+.1f} pts", metrics.get('skew_10d_text', 'Crash tilt'))
    with m_col5:
        st.metric("SKEW SLOPE", f"{metrics.get('skew_slope', 0):.2f}", "pts/1% strike")
    with m_col6:
        st.metric("TERM SLOPE", f"{chosen_row.get('IV_HV_Ratio', 1.0):.2f}x", "vs Clean HV")

    fig = go.Figure()

    # Dynamic Dual-Color Shading vs ATM Baseline (Red when above ATM, Green when under ATM)
    atm_val = metrics.get('atm_iv', 25.0)
    comb_strikes = []
    comb_ivs = []
    if 'otm_put_curve' in smile_data and smile_data['otm_put_curve']['strikes']:
        comb_strikes.extend(smile_data['otm_put_curve']['strikes'])
        comb_ivs.extend(smile_data['otm_put_curve']['iv'])
    if 'otm_call_curve' in smile_data and smile_data['otm_call_curve']['strikes']:
        start_idx = 1 if (comb_strikes and abs(smile_data['otm_call_curve']['strikes'][0] - comb_strikes[-1]) < 0.01) else 0
        comb_strikes.extend(smile_data['otm_call_curve']['strikes'][start_idx:])
        comb_ivs.extend(smile_data['otm_call_curve']['iv'][start_idx:])

    if len(comb_strikes) >= 2 and atm_val:
        pts = []
        for i in range(len(comb_strikes) - 1):
            x1, y1 = comb_strikes[i], comb_ivs[i]
            x2, y2 = comb_strikes[i+1], comb_ivs[i+1]
            pts.append((x1, y1))
            if (y1 - atm_val) * (y2 - atm_val) < 0:
                t = (atm_val - y1) / (y2 - y1)
                xc = x1 + t * (x2 - x1)
                pts.append((xc, atm_val))
        pts.append((comb_strikes[-1], comb_ivs[-1]))

        segs_above = []
        segs_below = []
        curr_seg = [pts[0]]
        is_above = pts[0][1] >= atm_val
        for i in range(1, len(pts)):
            p = pts[i]
            curr_seg.append(p)
            if abs(p[1] - atm_val) < 1e-4:
                if is_above: segs_above.append(curr_seg)
                else: segs_below.append(curr_seg)
                curr_seg = [p]
                is_above = (pts[i+1][1] >= atm_val) if i+1 < len(pts) else (not is_above)
        if len(curr_seg) > 1:
            if is_above: segs_above.append(curr_seg)
            else: segs_below.append(curr_seg)

        for idx, seg in enumerate(segs_above):
            xs = [p[0] for p in seg] + [seg[-1][0], seg[0][0], seg[0][0]]
            ys = [p[1] for p in seg] + [atm_val, atm_val, seg[0][1]]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, fill='toself', fillcolor='rgba(255, 46, 85, 0.16)',
                line=dict(width=0), mode='none', name='bid over ATM' if idx == 0 else None,
                showlegend=(idx == 0), legendgroup='shading_above', hoverinfo='none'
            ))
        for idx, seg in enumerate(segs_below):
            xs = [p[0] for p in seg] + [seg[-1][0], seg[0][0], seg[0][0]]
            ys = [p[1] for p in seg] + [atm_val, atm_val, seg[0][1]]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, fill='toself', fillcolor='rgba(0, 245, 160, 0.15)',
                line=dict(width=0), mode='none', name='under ATM' if idx == 0 else None,
                showlegend=(idx == 0), legendgroup='shading_below', hoverinfo='none'
            ))

    # OTM Puts Fitted Curve (Cherry Red)
    if 'otm_put_curve' in smile_data and smile_data['otm_put_curve']['strikes']:
        fig.add_trace(go.Scatter(
            x=smile_data['otm_put_curve']['strikes'],
            y=smile_data['otm_put_curve']['iv'],
            mode='lines',
            name='OTM Puts Skew',
            line=dict(color='#FF2E55', width=2.5)
        ))

    # OTM Calls Fitted Curve (Mint Green)
    if 'otm_call_curve' in smile_data and smile_data['otm_call_curve']['strikes']:
        fig.add_trace(go.Scatter(
            x=smile_data['otm_call_curve']['strikes'],
            y=smile_data['otm_call_curve']['iv'],
            mode='lines',
            name='OTM Calls Skew',
            line=dict(color='#00F5A0', width=2.5)
        ))

    # OTM Market Mid IV Points
    if 'otm_puts_market' in smile_data and smile_data['otm_puts_market']['strikes']:
        fig.add_trace(go.Scatter(
            x=smile_data['otm_puts_market']['strikes'],
            y=smile_data['otm_puts_market']['iv'],
            mode='markers',
            name='Market Puts IV',
            marker=dict(size=5, color='#FF2E55', opacity=0.35),
            showlegend=False
        ))
    if 'otm_calls_market' in smile_data and smile_data['otm_calls_market']['strikes']:
        fig.add_trace(go.Scatter(
            x=smile_data['otm_calls_market']['strikes'],
            y=smile_data['otm_calls_market']['iv'],
            mode='markers',
            name='Market Calls IV',
            marker=dict(size=5, color='#00F5A0', opacity=0.35),
            showlegend=False
        ))

    # Key Structural Markers: ATM Dot, 25Δ Put, 25Δ Call
    struct = smile_data.get('structural_markers', {})
    if struct.get('atm'):
        fig.add_trace(go.Scatter(
            x=[struct['atm']['strike']], y=[struct['atm']['iv']],
            mode='markers', name='ATM Baseline',
            marker=dict(size=11, color='#F59E0B', line=dict(color='#ffffff', width=1.5))
        ))
    if struct.get('put_25d'):
        fig.add_trace(go.Scatter(
            x=[struct['put_25d']['strike']], y=[struct['put_25d']['iv']],
            mode='markers+text', name='25Δ Put',
            text=['25Δ'], textposition='top center',
            marker=dict(size=12, color='rgba(255, 46, 85, 0.2)', line=dict(color='#FF2E55', width=2))
        ))
    if struct.get('call_25d'):
        fig.add_trace(go.Scatter(
            x=[struct['call_25d']['strike']], y=[struct['call_25d']['iv']],
            mode='markers+text', name='25Δ Call',
            text=['25Δ'], textposition='top center',
            marker=dict(size=12, color='rgba(0, 245, 160, 0.2)', line=dict(color='#00F5A0', width=2))
        ))

    # Effective Ask IV / Bid IV Execution Overlay (filtered to active tradable moneyness)
    is_buy = (side_clean == "Buy")
    if "Call" in dir_clean or dir_clean == "Bullish":
        exec_mask = (target_df['strike'] >= spot_price * 0.95) & (target_df['strike'] <= spot_price * 1.15)
    else:
        exec_mask = (target_df['strike'] <= spot_price * 1.05) & (target_df['strike'] >= spot_price * 0.85)
    exec_plot_df = target_df[exec_mask] if exec_mask.any() else target_df
    
    spread_trace_y = exec_plot_df['Ask_IV_Pct'] if is_buy else exec_plot_df['Bid_IV_Pct']
    spread_name = 'Effective Ask IV (Cost to Buy)' if is_buy else 'Effective Bid IV (Selling Price)'
    spread_color = 'rgba(245, 158, 11, 0.75)' if is_buy else 'rgba(0, 245, 160, 0.75)'
    fig.add_trace(go.Scatter(
        x=exec_plot_df['strike'], y=spread_trace_y,
        mode='lines', name=spread_name,
        line=dict(color=spread_color, width=1.5, dash='dot')
    ))

    # Highlight Spot Price & Expected Move Bands
    fig.add_vline(
        x=spot_price, line_width=1.5, line_dash="dash", line_color="#94a3b8",
        annotation_text=f"Spot: ${spot_price:.2f}", annotation_position="top left",
        annotation_font_color="#94a3b8"
    )
    
    # Horizontal ATM IV Line
    atm_val = metrics.get('atm_iv', 25.0)
    fig.add_hline(
        y=atm_val, line_width=1.2, line_dash="dash", line_color="#F59E0B",
        annotation_text=f"ATM: {atm_val:.1f}%", annotation_position="bottom right",
        annotation_font_color="#F59E0B"
    )

    # Expected Move Shaded Region
    fig.add_vrect(
        x0=spot_price - smile_data['exp_move'],
        x1=spot_price + smile_data['exp_move'],
        fillcolor="rgba(56, 189, 248, 0.04)",
        layer="below", line_width=0,
        annotation_text="1σ Expected Range", annotation_position="top right",
        annotation_font_color="rgba(56, 189, 248, 0.4)"
    )

    # Highlight Top Recommended Strikes (clean number only, e.g. "295")
    rec_texts = [f"{s:g}" for s in top_strikes['strike']]
    if side_clean == "Sell":
        marker_color = '#FF2E55'
        trace_label = 'Recommended Sell Strikes (Bid Edge)'
        hover_template = "<b>RECOMMENDED PREMIUM SELL</b><br>Strike: $%{x:.2f}<br>Market IV: %{y:.2f}%<extra></extra>"
    else:
        marker_color = '#00F5A0'
        trace_label = 'Recommended Buy Strikes (Ask Edge)'
        hover_template = "<b>RECOMMENDED VALUE BUY</b><br>Strike: $%{x:.2f}<br>Market IV: %{y:.2f}%<extra></extra>"

    fig.add_trace(go.Scatter(
        x=top_strikes['strike'], y=top_strikes['IV_Pct'],
        mode='markers+text',
        name=trace_label,
        marker=dict(size=15, color=marker_color, line=dict(color='#ffffff', width=2), symbol='circle'),
        text=rec_texts,
        textposition="bottom center",
        textfont=dict(color=marker_color, size=11, family="sans-serif"),
        hovertemplate=hover_template
    ))

    # Calculate appropriate y-axis range focused on the active curve
    core_ivs = []
    if 'otm_put_curve' in smile_data and smile_data['otm_put_curve']['iv']:
        core_ivs.extend(smile_data['otm_put_curve']['iv'])
    if 'otm_call_curve' in smile_data and smile_data['otm_call_curve']['iv']:
        core_ivs.extend(smile_data['otm_call_curve']['iv'])
    if 'metrics' in smile_data and 'atm_iv' in smile_data['metrics']:
        core_ivs.append(smile_data['metrics']['atm_iv'])
    if not top_strikes.empty and 'IV_Pct' in top_strikes.columns:
        core_ivs.extend(top_strikes['IV_Pct'].tolist())

    y_axis_kwargs = dict(title="Implied Volatility (%)", gridcolor="#21262d", ticksuffix="%")
    valid_ivs = [v for v in core_ivs if 4 < v < 120]
    if valid_ivs:
        min_iv = min(valid_ivs)
        max_iv = max(valid_ivs)
        pad = max((max_iv - min_iv) * 0.20, 2.0)
        y_min = max(0, int(min_iv - pad))
        y_max = int(max_iv + pad + 1)
        if y_max - y_min < 6:
            mid = (y_max + y_min) / 2
            y_min = max(0, int(mid - 3.5))
            y_max = int(mid + 3.5)
        dtick = 2 if (y_max - y_min) <= 12 else 5
        y_axis_kwargs['range'] = [y_min, y_max]
        y_axis_kwargs['dtick'] = dtick
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        height=450,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(title="Strike Price ($)", gridcolor="#21262d"),
        yaxis=y_axis_kwargs,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ----------------- WHOLE-CHAIN TIERED DISCOUNT SCANNER TABLE -----------------
    if 'ranked_chain' in locals() and ranked_chain:
        st.markdown("#### Whole-Chain Value Scanner: Top Discounted Contracts")
        st.caption("Scans across all expiration cycles simultaneously to rank contracts with the greatest mathematical edge surviving the spread.")
        
        table_rows = []
        for r in ranked_chain:
            cost_label = f"${r['ask_cost']:.2f}" if side_clean == "Buy" else f"${r['bid_cost']:.2f}"
            edge_val = r['ask_edge'] if side_clean == "Buy" else r['bid_edge']
            edge_label = f"{edge_val:+.2f}%"
            earn_flag = "Spans Earn" if r['spans_earnings'] else "Safe"
            table_rows.append({
                "Tier": r['tier'],
                "Date (DTE)": f"{r['expiration']} ({r['dte']}d)",
                "Strike": f"${r['strike']:.0f} {r['type']}",
                "Ask Cost" if side_clean == "Buy" else "Bid Credit": cost_label,
                "Market IV": f"{r['iv']:.1f}%",
                "OI (Vol)": f"{r['oi']:,} ({r['volume']:,} v)",
                "Ask Edge" if side_clean == "Buy" else "Bid Edge": edge_label,
                "Delta": f"{abs(r['delta']):.2f}Δ",
                "Event": earn_flag
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ----------------- SECTION C: SINGLE DATE STRIKE RECOMMENDATIONS -----------------
    if "Single Expiration Focus" in rec_mode or not top_strikes.empty:
        if side_clean == "Sell":
            st.markdown(f"#### Recommended Premium Sell Strikes for **{chosen_exp}** ({chosen_row['DTE']}d DTE)")
            st.caption(f"Strikes on **{chosen_exp}** offering the greatest Bid-Edge (selling at high IV even at the Bid) for credit harvesting.")
        else:
            st.markdown(f"#### Recommended Value Buy Strikes for **{chosen_exp}** ({chosen_row['DTE']}d DTE)")
            st.caption(f"Strikes on **{chosen_exp}** offering the greatest Ask-Edge (cheap even after paying the Ask price) within the prime delta band.")

        s_cols = st.columns(len(top_strikes))
        for i, (_, s_row) in enumerate(top_strikes.iterrows()):
            with s_cols[i]:
                is_call_rec = ("Call" in dir_clean)
                c_type = "Call" if is_call_rec else "Put"
                spread_badge = s_row.get('Spread_Quality', 'Tight')
                if side_clean == "Sell":
                    edge_val = s_row.get('Bid_Edge', 0)
                    b_color = "#00F5A0" if edge_val >= 0 else "#eab308"
                    b_sign = "+" if edge_val >= 0 else ""
                    bid_p = s_row.get('bid', 0)
                    cushion = abs(s_row['strike'] - spot_price) / spot_price * 100.0
                    
                    wing_k = s_row.get('Wing_Strike')
                    if wing_k is not None and pd.notnull(wing_k):
                        net_cr = s_row.get('Net_Credit', 0)
                        max_rk = s_row.get('Max_Risk', 0)
                        ror = s_row.get('RoR_Pct', 0)
                        wing_box = f"<div style='margin-top:6px; padding:6px; background:rgba(255,255,255,0.03); border-radius:6px; border:1px dashed #30363d; font-size:0.75rem;'><b>Hedged Spread Wing</b>: +${wing_k:.0f} {c_type}<br>Credit: <b>${net_cr:.2f}</b> | Max Risk: <b>${max_rk:.2f}</b> | RoR: <b style='color:#00F5A0;'>{ror:.1f}%</b></div>"
                    else:
                        wing_box = ""
                        
                    st.markdown(f"""
                    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: #f0f6fc;">${s_row['strike']:.0f} {c_type}</span>
                            <span style="background: rgba(255, 46, 85, 0.15); color: #FF2E55; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 600;">#{i+1} • Sell</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.82rem; margin-bottom: 8px;">
                            <div>Bid IV: <b>{s_row['Bid_IV_Pct']:.1f}%</b></div>
                            <div>Bid Edge: <b style="color: {b_color};">{b_sign}{edge_val:.2f}%</b></div>
                            <div>Delta: <b>{s_row['Est_Delta']:.2f}</b></div>
                            <div>Bid Credit: <b style="color: #00F5A0;">${bid_p:.2f}</b></div>
                        </div>
                        <div style="font-size: 0.78rem; color: #8b949e; border-top: 1px solid #21262d; padding-top: 6px;">
                            Mid: ${s_row['Mid_Price']:.2f} | Buffer: <b>{cushion:.1f}%</b> | {spread_badge}
                            {wing_box}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    edge_val = s_row.get('Ask_Edge', 0)
                    b_color = "#00F5A0" if edge_val >= 0 else "#eab308"
                    b_sign = "+" if edge_val >= 0 else ""
                    be_val = s_row['strike'] + s_row['ask'] if is_call_rec else s_row['strike'] - s_row['ask']
                    be_diff = ((be_val / spot_price) - 1) * 100
                    st.markdown(f"""
                    <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: #f0f6fc;">${s_row['strike']:.0f} {c_type}</span>
                            <span style="background: rgba(0, 245, 160, 0.15); color: #00F5A0; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 600;">#{i+1} • Buy</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 0.82rem; margin-bottom: 8px;">
                            <div>Ask IV: <b>{s_row['Ask_IV_Pct']:.1f}%</b></div>
                            <div>Ask Edge: <b style="color: {b_color};">{b_sign}{edge_val:.2f}%</b></div>
                            <div>Delta: <b>{s_row['Est_Delta']:.2f}</b></div>
                            <div>Ask Cost: <b style="color: #00F5A0;">${s_row['ask']:.2f}</b></div>
                        </div>
                        <div style="font-size: 0.78rem; color: #8b949e; border-top: 1px solid #21262d; padding-top: 6px;">
                            Mid: ${s_row['Mid_Price']:.2f} | Breakeven: <b>${be_val:.2f} ({be_diff:+.1f}%)</b> | {spread_badge}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ----------------- DEDICATED DEALER GAMMA DESK RENDER FUNCTION -----------------
def render_gamma_desk_page(available_tickers=None):
    """
    Renders the dedicated Dealer Gamma Exposure (GEX) & Volatility Pinning Desk in Streamlit.
    """
    if not st:
        return

    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
                padding: 1.25rem 1.75rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.25); margin-bottom: 1.5rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div>
                <h2 style="margin: 0; color: #f8fafc; font-size: 1.6rem; font-weight: 700;">
                    Institutional Dealer Gamma Exposure (GEX) Desk
                </h2>
                <p style="margin: 0.35rem 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                    Real-time dealer delta-hedging architecture: Inverts Black-Scholes gamma across open interest to identify <b>Call Wall pins</b>, <b>Put Wall hedging floors</b>, <b>Zero-Gamma flip boundaries</b>, and directional acceleration triggers.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.2, 1.0])
    universe = ["SPY", "QQQ", "NVDA", "AAPL", "AMZN", "MSFT", "META", "TSLA", "IWM", "GLD", "XLF", "SMH"]
    if available_tickers:
        for t in available_tickers:
            if t not in universe:
                universe.append(t)

    with ctrl_col1:
        selected_ticker = st.selectbox("Underlying Ticker", universe, index=0, key="gamma_desk_ticker")
    with ctrl_col2:
        max_exp = st.slider("Expirations Horizon (DTE <= 45)", min_value=2, max_value=8, value=5, key="gamma_desk_exps")
    with ctrl_col3:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        recalc = st.button("↻ Recalculate GEX", key="gamma_recalc_btn", use_container_width=True)

    with st.spinner(f"Computing institutional dealer gamma surface & wall pins for {selected_ticker}..."):
        gex_data = compute_gamma_exposure_profile(selected_ticker, max_expirations=max_exp, force_refresh=recalc)

    if not gex_data or "error" in gex_data:
        st.warning(f"Unable to extract gamma profile for {selected_ticker}: {gex_data.get('error') if gex_data else 'Unknown error'}")
        return

    spot = gex_data["spot_price"]
    cw = gex_data["call_wall"]
    cw_dist = gex_data["call_wall_dist_pct"]
    pw = gex_data["put_wall"]
    pw_dist = gex_data["put_wall_dist_pct"]
    zg = gex_data["zero_gamma_level"]
    zg_dist = gex_data["zero_gamma_dist_pct"]
    net_gex = gex_data["total_net_gex_billions"]
    is_long = gex_data["is_long_gamma"]
    regime_short = gex_data["regime_short"]

    # 2. KPI Strip
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        st.metric("Spot Price", f"${spot:.2f}", selected_ticker)
        st.caption("Live underlying reference")
    with kpi_col2:
        st.metric("Call Wall (Ceiling Pin)", f"${cw:.1f}", f"{cw_dist:+.2f}%", delta_color="normal")
        st.caption(f"${gex_data['call_wall_gex_m']:.0f}M GEX • Hedging Ceiling")
    with kpi_col3:
        st.metric("Put Wall (Floor Support)", f"${pw:.1f}", f"{pw_dist:+.2f}%", delta_color="inverse")
        st.caption(f"${gex_data['put_wall_gex_m']:.0f}M GEX • Hedging Floor")
    with kpi_col4:
        st.metric("Zero Gamma Flip", f"${zg:.1f}", f"{zg_dist:+.2f}%", delta_color="off")
        st.caption("Volatility Regime Boundary")
    with kpi_col5:
        st.metric("Total Net Dealer GEX", f"${net_gex:+.2f}B", regime_short)
        st.caption(f"{'Suppresses Volatility' if is_long else 'Amplifies Trending Vol'}")

    # 3. Tactical Desk Regime Directive Banner
    if is_long:
        st.markdown(f"""
        <div style="background: rgba(0, 245, 160, 0.08); border: 1px solid rgba(0, 245, 160, 0.25); border-left: 4px solid #00F5A0; padding: 12px 16px; border-radius: 8px; margin: 16px 0;">
            <b style="color: #00F5A0; font-size: 0.95rem;">LONG GAMMA REGIME DETECTED (+${net_gex:.2f}B Net GEX)</b><br>
            <span style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.4;">
                Market makers are long gamma and must <b>sell into rallies</b> and <b>buy into pullbacks</b> to maintain delta-neutral books. This dampens realized volatility and creates powerful mean-reversion magnet pin action between <b>Put Wall (${pw:.0f})</b> and <b>Call Wall (${cw:.0f})</b>. Limit orders at edges are favored over chasing breakouts.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(255, 46, 85, 0.08); border: 1px solid rgba(255, 46, 85, 0.25); border-left: 4px solid #FF2E55; padding: 12px 16px; border-radius: 8px; margin: 16px 0;">
            <b style="color: #FF2E55; font-size: 0.95rem;">SHORT GAMMA REGIME DETECTED (${net_gex:.2f}B Net GEX)</b><br>
            <span style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.4;">
                Market makers are short gamma and must <b>sell when spot drops</b> and <b>buy when spot rises</b> to hedge exposure. This amplifies market volatility and accelerates directional trending moves. Avoid blind dip-buying until structural support at the <b>Put Wall (${pw:.0f})</b> is confirmed.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # 4. Plotly Interactive Bar Chart
    gex_fig = plot_gamma_exposure_profile(gex_data)
    if gex_fig:
        st.plotly_chart(gex_fig, use_container_width=True)
        loaded_time_text = gex_data.get("loaded_at_display") or gex_data.get("loaded_at_str") or "Live snapshot"
        elapsed = gex_data.get("elapsed_seconds", 0)
        strikes_count = gex_data.get("strikes_count", 0)
        exps_scanned = len(gex_data.get("expirations_scanned", []))
        market_time_text = gex_data.get("market_trade_time") or "Exchange Session Close"
        session_label = gex_data.get("market_session_label", "")
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:center; font-size:11px; color:#64748b; margin-top:-14px; margin-bottom:14px; font-family:monospace; padding:0 4px; flex-wrap:wrap; gap:6px;">'
            f'<span><span style="color:#94a3b8;">Exchange Origin:</span> <b style="color:#38bdf8;">{market_time_text}</b> <span style="color:#475569;">•</span> {session_label} <span style="color:#475569;">•</span> <span style="color:#94a3b8;">OCC Open Interest:</span> Settled Overnight ({strikes_count} strikes)</span>'
            f'<span><span style="color:#94a3b8;">Feed Ingestion:</span> <b style="color:#cbd5e1;">{loaded_time_text}</b> <span style="color:#475569;">|</span> Latency: {elapsed:.2f}s</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # 4b. Gamma Terrain & Ball Rolling Physics Engine Chart ('Gamma = The Ground, Price = The Ball')
    terrain_fig = plot_gamma_terrain_profile(gex_data)
    if terrain_fig:
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        st.plotly_chart(terrain_fig, use_container_width=True)
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 8px; padding: 10px 16px; margin-top: -12px; margin-bottom: 24px;">
            <div style="display: flex; gap: 20px; flex-wrap: wrap; font-size: 11.5px; color: #cbd5e1; align-items: center;">
                <div><span style="color: #00F5A0; font-weight: 600;">● Positive Valley (+GEX):</span> Dealer shock absorbers active ('Push it back') ➔ <b>Mean-reversion pin action</b></div>
                <div><span style="color: #38bdf8; font-weight: 600;">● Zero-Gamma Crest:</span> Tipping point inflection ➔ <b>Breakout / breakdown trigger boundary</b></div>
                <div><span style="color: #FF2E55; font-weight: 600;">● Negative Slope (-GEX):</span> Dealer momentum acceleration ('Push it') ➔ <b>Fast & abrupt cascading volatility</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Detailed Strike Breakdown Table (Filtered to Relevant Strikes by default)
    strikes_list = gex_data.get("strikes", [])
    if strikes_list:
        st.markdown("#### Per-Strike Gamma Exposure & Hedging Breakdown")
        rel_count = gex_data.get("relevant_strikes_count", len([s for s in strikes_list if s.get("is_relevant")]))
        total_count = len(strikes_list)

        tb_col1, tb_col2 = st.columns([3, 1])
        with tb_col1:
            st.caption(f"Displaying institutional structural walls (Call/Put Walls, Zero Flip, ATM) and top dealer open interest battlegrounds.")
        with tb_col2:
            show_all = st.checkbox(f"Show all {total_count} mapped strikes", value=False, help="Uncheck to hide low-liquidity noise strikes and focus on high-conviction levels.")

        display_strikes = strikes_list if show_all else [s for s in strikes_list if s.get("is_relevant", True)]
        table_rows = []
        zg_strike = gex_data.get("closest_zero_gamma_strike", zg)
        for s in display_strikes:
            k = s["strike"]
            dist_pct = ((k / spot) - 1) * 100
            role = s.get("structural_role", "Normal Strike")
            if "Call Wall" in role:
                role = "" + role
            elif "Put Wall" in role:
                role = "🔴 " + role
            elif "Zero Flip" in role:
                role = role
            elif "ATM" in role:
                role = role
            elif "Dual Gamma" in role or "Peak" in role:
                role = role
            elif "Top Call" in role:
                role = role
            elif "Top Put" in role:
                role = role
            elif "Major Open Interest" in role or "Milestone" in role:
                role = role

            table_rows.append({
                "Strike ($)": f"${k:,.2f}",
                "Dist vs Spot (%)": f"{dist_pct:+.2f}%",
                "Call GEX ($M)": s["call_gex"],
                "Put GEX ($M)": s["put_gex"],
                "Net GEX ($M)": s["net_gex"],
                "Call OI": f"{s['call_oi']:,}",
                "Put OI": f"{s['put_oi']:,}",
                "Structural Role": role
            })
        df_strikes = pd.DataFrame(table_rows)
        st.dataframe(df_strikes, use_container_width=True, hide_index=True)

    # 6. Day-over-Day Gamma Migration & Strike Heat Map
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("### Day-over-Day Gamma Migration & Strike Heat Map")
    st.caption("Visualizing institutional strike inventory repositioning, Call/Put Wall roll migrations, and Net GEX velocity across trading sessions.")

    try:
        from gamma_tracker import compute_gamma_shift, plot_gamma_shift_heatmap, format_signed_currency
        shift_data = compute_gamma_shift(selected_ticker)

        if shift_data and shift_data.get("has_history"):
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            cw_shift = shift_data.get("call_wall_shift", 0.0)
            cw_col = "off" if abs(cw_shift) < 0.01 else ("normal" if cw_shift > 0 else "inverse")
            with m_col1:
                st.metric(
                    "Call Wall Migration",
                    f"${shift_data.get('call_wall_current', 0):,.1f}",
                    f"{cw_shift:+.1f} DoD ({shift_data.get('call_wall_status', '').replace('_', ' ')})",
                    delta_color=cw_col
                )
                st.caption(f"Prev: ${shift_data.get('call_wall_previous', 0):,.1f}")

            pw_shift = shift_data.get("put_wall_shift", 0.0)
            pw_col = "off" if abs(pw_shift) < 0.01 else ("normal" if pw_shift > 0 else "inverse")
            with m_col2:
                st.metric(
                    "Put Wall Migration",
                    f"${shift_data.get('put_wall_current', 0):,.1f}",
                    f"{pw_shift:+.1f} DoD ({shift_data.get('put_wall_status', '').replace('_', ' ')})",
                    delta_color=pw_col
                )
                st.caption(f"Prev: ${shift_data.get('put_wall_previous', 0):,.1f}")

            zg_shift = shift_data.get("zero_gamma_shift", 0.0)
            with m_col3:
                st.metric(
                    "Zero-Gamma Migration",
                    f"${shift_data.get('zero_gamma_current', 0):,.1f}",
                    f"{zg_shift:+.1f} DoD",
                    delta_color="off"
                )
                st.caption(f"Prev: ${shift_data.get('zero_gamma_previous', 0):,.1f}")

            vel_delta = shift_data.get("net_gex_velocity_b", 0.0)
            vel_col = "off" if abs(vel_delta) < 0.05 else ("normal" if vel_delta > 0 else "inverse")
            with m_col4:
                st.metric(
                    "Net GEX Velocity (ΔGEX)",
                    format_signed_currency(shift_data.get('net_gex_current_b', 0), "B", 2),
                    f"{vel_delta:+.2f}B DoD",
                    delta_color=vel_col
                )
                st.caption(shift_data.get("velocity_label", "Stable"))

            # Interactive Mode Toggle
            mode_c1, mode_c2 = st.columns([2.5, 1.5])
            with mode_c1:
                st.markdown(f"> **Quantitative Shift Audit:** {shift_data.get('institutional_synthesis', '')}")
            with mode_c2:
                heat_mode_choice = st.radio(
                    "Heat Map Visual Mode",
                    ["Total Net GEX ($M)", "Day-over-Day ΔGEX Shift (Brightness = Change)"],
                    index=0,
                    horizontal=True,
                    key="gamma_heatmap_mode_choice"
                )

            heat_mode = "delta_gex" if "ΔGEX" in heat_mode_choice else "net_gex"
            heatmap_fig = plot_gamma_shift_heatmap(selected_ticker, mode=heat_mode, max_days=10)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)

            # Top Accumulating Strikes expander
            top_calls = shift_data.get("top_calls_accumulating", [])
            top_puts = shift_data.get("top_puts_hedging", [])
            if top_calls or top_puts:
                with st.expander(f"Top Strike Positioning Shifts for {selected_ticker} (Where Dealers Repositioned)"):
                    tk_c1, tk_c2 = st.columns(2)
                    with tk_c1:
                        st.markdown("**🟢 Top Call Gamma Accumulation Strikes (Upside Magnets Building)**")
                        c_rows = []
                        for c_item in top_calls:
                            c_rows.append({
                                "Strike": c_item.get("strike_str"),
                                "ΔCall GEX ($M)": format_signed_currency(c_item.get('delta_call_gex_m'), "M", 1),
                                "Current Net GEX": format_signed_currency(c_item.get('curr_net_gex_m'), "M", 1),
                                "Call Open Interest": f"{c_item.get('curr_call_oi'):,}"
                            })
                        st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
                    with tk_c2:
                        st.markdown("**🔴 Top Put Gamma Hedging Strikes (Downside Demand Building)**")
                        p_rows = []
                        for p_item in top_puts:
                            p_rows.append({
                                "Strike": p_item.get("strike_str"),
                                "ΔPut GEX ($M)": format_signed_currency(p_item.get('delta_put_gex_m'), "M", 1),
                                "Current Net GEX": format_signed_currency(p_item.get('curr_net_gex_m'), "M", 1),
                                "Put Open Interest": f"{p_item.get('curr_put_oi'):,}"
                            })
                        st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(f"Gamma migration tracker initializing: {e}")

    # 7. Educational Collapsible Guide
    with st.expander("Institutional Desk Guide: How to Trade Using Gamma Levels"):
        st.markdown("""
        * **Call Wall:** The strike with the largest open Call Gamma. Market makers are typically assumed net long calls (counterparty to call sellers). As spot approaches the Call Wall, dealers must sell stock to hedge, creating a natural upside ceiling and pinning magnet into OpEx.
        * **Put Wall:** The strike with the largest open Put Gamma. Market makers are typically assumed net short puts (counterparty to protective put buyers). In a net Long Gamma regime, overall dealer dip-buying provides strong cushion and rebound support at this level.
        * **Zero Gamma Flip:** The exact inflection boundary where aggregate dealer positioning flips between positive (volatility suppressing / mean-reverting) and negative (volatility accelerating / cascade risk).
        * **Weekly OpEx Pinning:** On expiration days (especially Fridays), gamma sensitivity ($\\Gamma \\propto 1/\\sqrt{T}$) peaks, pinning price action tightly around the highest open interest strikes.
        """)
