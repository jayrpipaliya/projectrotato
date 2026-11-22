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
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

def _cache_key(arg):
    if isinstance(arg, dict):
        return tuple(sorted((k, _cache_key(v)) for k, v in arg.items()))
    if isinstance(arg, (list, tuple)):
        return tuple(_cache_key(x) for x in arg)
    return arg

def st_cache(func=None, ttl=600, **kwargs):
    def decorator(f):
        _cache = {}
        @wraps(f)
        def wrapped(*args, **kw):
            key = (_cache_key(args), _cache_key(tuple(sorted(kw.items()))))
            now = time.time()
            if key in _cache:
                val, exp_time = _cache[key]
                if now < exp_time:
                    return val
            res = f(*args, **kw)
            _cache[key] = (res, now + ttl)
            return res
        wrapped.clear = _cache.clear
        return wrapped
    if func is not None:
        return decorator(func)
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

def black_scholes_theta(S, K, T, sigma, r=0.045, q=0.0075, is_call=True):
    """Computes analytical Black-Scholes daily theta (dollar decay per calendar day)."""
    if S <= 0 or K <= 0 or T <= 0.0001 or sigma <= 0.001:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = - (S * sigma * np.exp(-q * T) * norm_pdf(d1)) / (2.0 * np.sqrt(T))
    if is_call:
        term2 = q * S * np.exp(-q * T) * norm_cdf(d1) - r * K * np.exp(-r * T) * norm_cdf(d2)
    else:
        term2 = - q * S * np.exp(-q * T) * norm_cdf(-d1) + r * K * np.exp(-r * T) * norm_cdf(-d2)
    theta_annual = term1 + term2
    return float(theta_annual / 365.25)

def black_scholes_price(S, K, T, sigma, r=0.045, q=0.0075, is_call=True):
    """Computes theoretical Black-Scholes option price with dividend yield q."""
    if S <= 0 or K <= 0 or T <= 0.0001 or sigma <= 0.001:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return float(S * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm_cdf(-d2) - S * np.exp(-q * T) * norm_cdf(-d1))

def compute_iv_rank_and_percentile(ticker_symbol, current_iv_pct=None):
    """
    Computes 1-Year (252 trading sessions) IV Rank and IV Percentile:
    - IV Rank = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV) * 100
    - IV Percentile = % of sessions over past 252 days with IV < Current IV
    """
    t_up = str(ticker_symbol).upper().strip()
    df = None
    try:
        from data_manager import load_ticker_data
        df = load_ticker_data(t_up)
    except Exception:
        pass

    if df is None or len(df) < 30:
        try:
            hist = yf.Ticker(t_up).history(period="1y")
            if not hist.empty and len(hist) >= 30:
                df = hist.reset_index()
        except Exception:
            pass

    if df is None or len(df) < 30:
        fallback_iv = current_iv_pct if current_iv_pct and current_iv_pct > 0 else 20.0
        return {
            'current_iv': round(fallback_iv, 1),
            'iv_rank': 50.0,
            'iv_percentile': 50.0,
            'iv_52w_low': round(fallback_iv * 0.6, 1),
            'iv_52w_high': round(fallback_iv * 1.8, 1),
            'vol_regime': 'Neutral Volatility',
            'style': 'slate'
        }

    ret = df['Close'].pct_change().dropna()
    rolling_21 = ret.rolling(21).std() * np.sqrt(252) * 100
    hist_1y = rolling_21.tail(252).dropna()
    if len(hist_1y) < 20:
        hist_1y = rolling_21.dropna()

    iv_min = float(hist_1y.min()) if not hist_1y.empty else 10.0
    iv_max = float(hist_1y.max()) if not hist_1y.empty else 40.0

    if current_iv_pct is None or current_iv_pct <= 0:
        current_iv_pct = float(rolling_21.iloc[-1]) if not rolling_21.empty else 20.0

    if iv_max > iv_min:
        ivr = ((current_iv_pct - iv_min) / (iv_max - iv_min)) * 100.0
    else:
        ivr = 50.0
    ivr = max(0.0, min(100.0, ivr))

    ivp = float((hist_1y < current_iv_pct).mean() * 100.0) if not hist_1y.empty else 50.0
    ivp = max(0.0, min(100.0, ivp))

    if ivp >= 80.0:
        regime = 'Elevated Premium (Top 20% Vol)'
        style = 'cherry'
    elif ivp >= 60.0:
        regime = 'Above Average Vol'
        style = 'amber'
    elif ivp <= 20.0:
        regime = 'Deep Cheap Vol (Bottom 20%)'
        style = 'mint'
    elif ivp <= 40.0:
        regime = 'Below Average Vol'
        style = 'sky'
    else:
        regime = 'Neutral Volatility'
        style = 'slate'

    return {
        'current_iv': round(current_iv_pct, 1),
        'iv_rank': round(ivr, 1),
        'iv_percentile': round(ivp, 1),
        'iv_52w_low': round(iv_min, 1),
        'iv_52w_high': round(iv_max, 1),
        'vol_regime': regime,
        'style': style
    }

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

            return {"gex": local_gex, "calls": local_calls, "puts": local_puts, "max_trade_ts": exp_max_ts, "base_iv": base_iv}
        except Exception:
            return {"gex": {}, "calls": [], "puts": [], "max_trade_ts": None, "base_iv": None}

    with ThreadPoolExecutor(max_workers=min(5, len(valid_exps))) as executor:
        results = list(executor.map(process_exp, valid_exps))

    exp_base_ivs = [res.get("base_iv") for res in results if res and res.get("base_iv")]
    current_atm_iv_pct = float(np.median(exp_base_ivs) * 100.0) if exp_base_ivs else 20.0
    iv_stats = compute_iv_rank_and_percentile(ticker_up, current_atm_iv_pct)

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
    net_put_cands = [k for k in otm_put_cands if strikes_gex[k]["put_gex"] > strikes_gex[k]["call_gex"]]
    if net_put_cands:
        put_wall = max(net_put_cands, key=lambda k: (strikes_gex[k]["put_gex"] - strikes_gex[k]["call_gex"], strikes_gex[k]["put_oi"]))
    elif otm_put_cands:
        put_wall = max(otm_put_cands, key=lambda k: (strikes_gex[k]["put_gex"], strikes_gex[k]["put_oi"]))
    else:
        put_wall = min(sorted_strikes)

    otm_call_cands = [k for k in sorted_strikes if k > atm_band_high]
    if not otm_call_cands:
        otm_call_cands = [k for k in sorted_strikes if k > spot_price]
    net_call_cands = [k for k in otm_call_cands if strikes_gex[k]["call_gex"] > strikes_gex[k]["put_gex"]]
    if net_call_cands:
        call_wall = max(net_call_cands, key=lambda k: (strikes_gex[k]["call_gex"] - strikes_gex[k]["put_gex"], strikes_gex[k]["call_oi"]))
    elif otm_call_cands:
        call_wall = max(otm_call_cands, key=lambda k: (strikes_gex[k]["call_gex"], strikes_gex[k]["call_oi"]))
    else:
        call_wall = max(sorted_strikes)

    # 4. Institutional OTM Accumulation Target Base (Landing Pad Detection):
    # Detects where smart money and dealers are building an emerging base with small yet increasing
    # Open Interest further out-of-the-money (>= 1.2% OTM). When positive gamma / OI builds out here,
    # dealers are bracing for impact, signaling a pre-positioned magnet / landing destination.
    otm_target_cands = [k for k in sorted_strikes if k >= spot_price * 1.012]
    target_base_strike = None
    target_base_oi = 0
    target_base_gex = 0.0
    target_base_dist_pct = 0.0

    if otm_target_cands:
        accum_shifts = {}
        try:
            from gamma_tracker import load_gamma_history
            h = load_gamma_history(ticker_up)
            if h and len(h) >= 2:
                prev_s = h[-2].get("strikes", {})
                curr_s = h[-1].get("strikes", {})
                for k_cand in otm_target_cands:
                    c_oi = curr_s.get(str(k_cand), {}).get("call_oi", 0) or curr_s.get(k_cand, {}).get("call_oi", 0)
                    p_oi = prev_s.get(str(k_cand), {}).get("call_oi", 0) or prev_s.get(k_cand, {}).get("call_oi", 0)
                    accum_shifts[k_cand] = max(c_oi - p_oi, 0)
        except Exception:
            pass

        def score_target_base(k):
            raw_oi = strikes_gex[k]["call_oi"]
            c_gex = max(strikes_gex[k]["call_gex"], 0.0)
            dod_boost = accum_shifts.get(k, 0) * 5
            return raw_oi + dod_boost + (c_gex * 10)

        best_target_k = max(otm_target_cands, key=score_target_base)
        if strikes_gex[best_target_k]["call_oi"] >= 3000:
            target_base_strike = round(best_target_k, 2)
            target_base_oi = int(strikes_gex[best_target_k]["call_oi"])
            target_base_gex = round(strikes_gex[best_target_k]["call_gex"], 1)
            target_base_dist_pct = round(((best_target_k / spot_price) - 1) * 100, 2)

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

    min_k = min(sorted_strikes)
    max_k = max(sorted_strikes)
    avg_step = (max_k - min_k) / max(len(sorted_strikes), 1)

    # Only assign a closest zero-gamma strike if zero_gamma_level is actually within or adjacent to the active strike grid
    if min_k - avg_step <= zero_gamma_level <= max_k + avg_step:
        cand_zg = min(sorted_strikes, key=lambda k: abs(k - zero_gamma_level))
        closest_zg_strike = cand_zg if abs(cand_zg - zero_gamma_level) <= max(avg_step * 2.0, 5.0) else None
    else:
        closest_zg_strike = None

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
    key_targets = {call_wall, put_wall, abs_gamma_strike, atm_k}
    if closest_zg_strike is not None:
        key_targets.add(closest_zg_strike)
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
        elif closest_zg_strike is not None and abs(k - closest_zg_strike) < 0.01:
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
    try:
        from gamma_tracker import get_market_session_date
        market_session_date = get_market_session_date()
    except Exception:
        market_session_date = now_dt.strftime("%Y-%m-%d")

    profile = {
        "ticker": ticker_up,
        "date": market_session_date,
        "spot_price": round(spot_price, 2),
        "call_wall": round(call_wall, 2),
        "call_wall_gex_m": round(strikes_gex[call_wall]["call_gex"], 1),
        "call_wall_dist_pct": round(((call_wall / spot_price) - 1) * 100, 2),
        "put_wall": round(put_wall, 2),
        "put_wall_gex_m": round(strikes_gex[put_wall]["put_gex"], 1),
        "put_wall_dist_pct": round(((put_wall / spot_price) - 1) * 100, 2),
        "absolute_gamma_strike": round(abs_gamma_strike, 2),
        "zero_gamma_level": round(zero_gamma_level, 2),
        "closest_zero_gamma_strike": round(closest_zg_strike, 2) if closest_zg_strike is not None else None,
        "zero_gamma_dist_pct": round(((zero_gamma_level / spot_price) - 1) * 100, 2),
        "target_base_strike": target_base_strike,
        "target_base_oi": target_base_oi,
        "target_base_gex_m": target_base_gex,
        "target_base_dist_pct": target_base_dist_pct,
        "total_net_gex_billions": round(total_net_gex / 1000.0, 3),
        "total_call_gex_billions": round(total_call_gex / 1000.0, 3),
        "total_put_gex_billions": round(total_put_gex / 1000.0, 3),
        "is_long_gamma": is_long_gamma,
        "gex_intensity": gex_intensity,
        "call_wall_concentration_pct": cw_conc,
        "put_wall_concentration_pct": pw_conc,
        "regime": regime,
        "regime_short": regime_short,
        "current_iv": iv_stats.get("current_iv"),
        "iv_rank": iv_stats.get("iv_rank"),
        "iv_percentile": iv_stats.get("iv_percentile"),
        "iv_52w_low": iv_stats.get("iv_52w_low"),
        "iv_52w_high": iv_stats.get("iv_52w_high"),
        "vol_regime": iv_stats.get("vol_regime"),
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
        return {'HV_10': 20.0, 'HV_20': 20.0, 'HV_30': 20.0, 'HV_30_clean': 20.0, 'HV_60': 20.0, 'HV_252': 20.0, 'HV_1Y': 20.0, 'Parkinson_30': 20.0}

    df = df_history.sort_values('Date').copy() if 'Date' in df_history.columns else df_history.copy()
    ret = df['Close'].pct_change().dropna()
    
    hv_10 = float(ret.tail(10).std() * np.sqrt(252) * 100) if len(ret) >= 10 else 20.0
    hv_20 = float(ret.tail(20).std() * np.sqrt(252) * 100) if len(ret) >= 20 else hv_10
    hv_30 = float(ret.tail(30).std() * np.sqrt(252) * 100) if len(ret) >= 20 else hv_20
    hv_60 = float(ret.tail(60).std() * np.sqrt(252) * 100) if len(ret) >= 40 else hv_30
    hv_252 = float(ret.tail(252).std() * np.sqrt(252) * 100) if len(ret) >= 150 else hv_60
    
    # De-Jumped HV: Robust Winsorization using Median Absolute Deviation (MAD)
    # Prevents massive single-day earnings shocks from distorting true noise volatility
    tail_30 = ret.tail(30).copy()
    median_val = float(tail_30.median())
    mad = float(np.median(np.abs(tail_30 - median_val)))
    robust_std = 1.4826 * mad if mad > 0 else float(tail_30.std())
    clean_tail_30 = tail_30.clip(lower=median_val - 3.0 * robust_std, upper=median_val + 3.0 * robust_std)
    hv_30_clean = float(clean_tail_30.std() * np.sqrt(252) * 100) if len(clean_tail_30) >= 10 else hv_30
    
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
        'HV_252': round(hv_252, 2),
        'HV_1Y': round(hv_252, 2),
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
        
        target_exps = [e for e in expirations if 1 <= (pd.Timestamp(e) - today).days <= 365]
        
        # Calendar to Trading-Day IV conversion factor: sqrt(252 / 365.25) = 0.8307
        CAL_TO_TRADING_FACTOR = np.sqrt(252.0 / 365.25)
        
        def process_expiration(exp):
            try:
                chain = t.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                if calls.empty or puts.empty:
                    return None
                
                total_call_oi = int(pd.to_numeric(calls['openInterest'], errors='coerce').fillna(0).sum())
                total_put_oi = int(pd.to_numeric(puts['openInterest'], errors='coerce').fillna(0).sum())
                total_oi = total_call_oi + total_put_oi
                
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
                    
                call_b = float(atm_call.bid) if pd.notnull(atm_call.bid) and atm_call.bid > 0 else 0.0
                call_a = float(atm_call.ask) if pd.notnull(atm_call.ask) and atm_call.ask > 0 else 0.0
                # Sanitize crossed after-hours quotes from yfinance (where stale bid > ask)
                if call_a > 0 and call_b > call_a:
                    call_b, call_a = call_a, call_b
                call_mid = round((call_b + call_a) / 2.0, 2)
                call_spread_pct = round(((call_a - call_b) / max(call_mid, 0.05)) * 100.0, 1) if call_a >= call_b else 0.0
                
                q_div = get_dividend_yield(ticker_symbol)
                call_bid_iv = round(implied_vol_solver(call_b, spot_price, atm_call.strike, t_years, q=q_div, is_call=True) * 100.0, 2) if call_b > 0 else raw_atm_iv
                call_ask_iv = round(implied_vol_solver(call_a, spot_price, atm_call.strike, t_years, q=q_div, is_call=True) * 100.0, 2) if call_a > 0 else raw_atm_iv
                if call_bid_iv > call_ask_iv:
                    call_bid_iv, call_ask_iv = call_ask_iv, call_bid_iv
                call_theta = round(black_scholes_theta(spot_price, atm_call.strike, t_years, raw_atm_iv / 100.0, q=q_div, is_call=True), 2)

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
                    'Call_Bid': call_b,
                    'Call_Ask': call_a,
                    'Call_Mid': call_mid,
                    'Call_Spread_Pct': call_spread_pct,
                    'Call_Bid_IV': call_bid_iv,
                    'Call_Ask_IV': call_ask_iv,
                    'Call_Theta': call_theta,
                    'Put_Bid': atm_put.bid,
                    'Put_Ask': atm_put.ask,
                    'Total_OI': total_oi,
                    'Call_OI': total_call_oi,
                    'Put_OI': total_put_oi
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

