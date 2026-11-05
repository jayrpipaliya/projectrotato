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
    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    if price <= intrinsic:
        return 0.15
    sigma = 0.22
    for _ in range(15):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            theo = S * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
        else:
            theo = K * np.exp(-r * T) * norm_cdf(-d2) - S * np.exp(-q * T) * norm_cdf(-d1)
        diff = theo - price
        if abs(diff) < 0.02:
            return float(np.clip(sigma, 0.06, 2.5))
        vega = max(S * np.exp(-q * T) * np.sqrt(T) * norm_pdf(d1), 1e-4)
        sigma -= diff / vega
        if sigma <= 0.02 or sigma > 3.0:
            break
    return float(np.clip(sigma, 0.06, 2.5))

# ----------------- DIVIDEND YIELD LOOKUP -----------------
DIVIDEND_YIELDS = {
    'SPY': 0.0120, 'QQQ': 0.0060, 'IWM': 0.0125, 'GLD': 0.0000,
    'AAPL': 0.0050, 'MSFT': 0.0070, 'NVDA': 0.0003, 'AMZN': 0.0000,
    'META': 0.0035, 'GOOGL': 0.0045, 'TSLA': 0.0000, 'XLF': 0.0145, 'SMH': 0.0065
}

def get_dividend_yield(ticker_symbol):
    """Returns continuous dividend yield estimate for Black-Scholes pricing."""
    return DIVIDEND_YIELDS.get(ticker_symbol.upper(), 0.0075)

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
                    
                ratio = round(atm_iv_trading / benchmark_hv, 2) if benchmark_hv > 0 else 1.0
                
                # Variance Risk Premium (VRP) Calibration: Index ETFs naturally carry ~1.12x VRP; single equities ~1.08x
                is_index_asset = ticker_symbol.upper() in ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'USO', 'XLF', 'SMH', 'XLE', 'XLK']
                vrp_factor = 1.12 if is_index_asset else 1.08
                fair_vrp_iv = round(benchmark_hv * vrp_factor, 2)
                vrp_ratio = round(atm_iv_trading / fair_vrp_iv, 2) if fair_vrp_iv > 0 else 1.0

                if spans_earnings:
                    event_tag = "Spans Earnings (Contains Event Jump)"
                    is_cheap = (ratio <= 1.25)
                    status = f"Event Cycle ({ratio}x {h_label})"
                else:
                    event_tag = "Inter-Earnings (No IV Crush Risk)"
                    if vrp_ratio < 0.92:
                        status = f"🟢 High Discount ({vrp_ratio}x VRP)"
                        is_cheap = True
                    elif vrp_ratio <= 1.08:
                        status = f"🟢 Fair Value ({vrp_ratio}x VRP)"
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
            skew_25d_sub = f"{int(put_25d['strike'])}P {put_25d['IV_Pct']:.1f}% vs {int(call_25d['strike'])}C {call_25d['IV_Pct']:.1f}%"
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

        # Fit Monotonic PCHIP Spline across combined stitched OTM strikes
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
        pchip = None
        try:
            from scipy.interpolate import PchipInterpolator
            if len(df_comb) >= 3:
                pchip = PchipInterpolator(df_comb['strike'].values, df_comb['IV_Pct'].values)
                grid_ivs = np.clip(pchip(grid_strikes), df_comb['IV_Pct'].min() * 0.75, df_comb['IV_Pct'].max() * 1.35)
        except Exception:
            pchip = None

        if pchip is None:
            p_fit = np.polyfit(df_comb['strike'].values, df_comb['IV_Pct'].values, min(2, len(df_comb)-1))
            grid_ivs = np.clip(np.polyval(p_fit, grid_strikes), df_comb['IV_Pct'].min() * 0.75, df_comb['IV_Pct'].max() * 1.35)

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
        is_call = (direction == "Bullish") if side_clean == "Buy" else (direction == "Bearish")
        target_df = (calls if is_call else puts).copy()
        if len(target_df) < 5:
            return None
        
        target_df['Log_Moneyness'] = np.log(target_df['strike'] / spot_price)
        
        # Evaluate Fair Fitted Skew using the monotonic PCHIP model (eliminates inverted bowl distortion)
        if pchip is not None:
            fitted = pchip(target_df['strike'].values)
        else:
            fitted = np.polyval(p_fit, target_df['strike'].values)

        target_df['Fitted_IV'] = np.clip(fitted, target_df['IV_Pct'].min() * 0.75, target_df['IV_Pct'].max() * 1.35)
        target_df['IV_Discount'] = target_df['Fitted_IV'] - target_df['IV_Pct']  # Midpoint discount (cheapness)
        target_df['IV_Premium'] = target_df['IV_Pct'] - target_df['Fitted_IV']   # Midpoint premium (richness)
        
        # Spread Quality Rating
        def classify_spread(sp):
            if sp <= 5.0:
                return "🟢 Tight"
            elif sp <= 15.0:
                return "Fair"
            else:
                return "🔴 Wide"
        target_df['Spread_Quality'] = target_df['Spread_Pct'].apply(classify_spread)
        
        # Clamp Ask_IV_Pct and Bid_IV_Pct so that wide bid-ask spread artifacts on illiquid wings do not explode
        target_df['Ask_IV_Pct'] = np.clip(target_df['Ask_IV_Pct'], target_df['IV_Pct'], target_df['Fitted_IV'] + 4.5)
        target_df['Bid_IV_Pct'] = np.clip(target_df['Bid_IV_Pct'], target_df['Fitted_IV'] - 4.5, target_df['IV_Pct'])
        target_df['Ask_Edge'] = (target_df['Fitted_IV'] - target_df['Ask_IV_Pct']).round(2)
        target_df['Bid_Edge'] = (target_df['Bid_IV_Pct'] - target_df['Fitted_IV']).round(2)
        
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
            target_width = max(spot_price * 0.025, 1.0)
            
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
                credit = round(max(0.01, s_bid - w_ask), 2)
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
            'poly_coeffs_k': [0, 0, 0],
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
def scan_cross_chain_recommendations(ticker_symbol, cheap_expirations_list, spot_price, direction="Bullish", side="Buy"):
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
                
                if side_clean == "Sell":
                    ideal_d = -0.25 if direction == "Bullish" else 0.25
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
                    ideal_d = 0.40 if direction == "Bullish" else -0.40
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
        return []
        
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
        side_sel = st.radio("Execution Side", ["🟢 Buy Side (Ask Edge)", "🔴 Sell Side (Bid Edge)"], horizontal=True)
        side_clean = "Sell" if "Sell" in side_sel else "Buy"

    with ctrl_col3:
        direction = st.radio("Directional Bias", ["🟢 Bullish (Calls)", "🔴 Bearish (Puts)"], horizontal=True)
        dir_clean = "Bullish" if "Bullish" in direction else "Bearish"

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
            cross_recommendations = scan_cross_chain_recommendations(selected_ticker, chains_list, spot_price, direction=dir_clean, side=side_clean)

        if cross_recommendations:
            rec_cols = st.columns(len(cross_recommendations))
            for i, rec in enumerate(cross_recommendations):
                with rec_cols[i]:
                    is_call_rec = (dir_clean == "Bullish") if side_clean == "Buy" else (dir_clean == "Bearish")
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
                    spread_badge = rec.get('Spread_Quality', '🟢 Tight')

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
    if dir_clean == "Bullish":
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
                is_call_rec = (dir_clean == "Bullish") if side_clean == "Buy" else (dir_clean == "Bearish")
                c_type = "Call" if is_call_rec else "Put"
                spread_badge = s_row.get('Spread_Quality', '🟢 Tight')
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
