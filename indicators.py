import numpy as np
import pandas as pd

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates 14-period Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_rsi_divergence(df: pd.DataFrame, window: int = 20) -> str:
    """
    Detects if there is a bullish or bearish RSI divergence in the recent window.
    - Bearish Divergence: Price printed higher high, but RSI printed lower high.
    - Bullish Divergence: Price printed lower low, but RSI printed higher low.
    """
    if len(df) < window + 10:
        return "None"

    recent = df.iloc[-window:].copy()
    prev = df.iloc[-(window * 2):-window].copy()

    price_curr_max = recent["Close"].max()
    price_prev_max = prev["Close"].max()
    rsi_curr_max = recent["RSI"].max()
    rsi_prev_max = prev["RSI"].max()

    # Bearish divergence
    if price_curr_max > price_prev_max * 1.005 and rsi_curr_max < rsi_prev_max - 2 and recent["RSI"].iloc[-1] > 60:
        return "Bearish Divergence (Upside Exhaustion)"

    price_curr_min = recent["Close"].min()
    price_prev_min = prev["Close"].min()
    rsi_curr_min = recent["RSI"].min()
    rsi_prev_min = prev["RSI"].min()

    # Bullish divergence
    if price_curr_min < price_prev_min * 0.995 and rsi_curr_min > rsi_prev_min + 2 and recent["RSI"].iloc[-1] < 45:
        return "Bullish Divergence (Downside Exhaustion)"

    return "None"

def compute_all_metrics(df: pd.DataFrame) -> dict:
    """
    Computes complete quantitative feature set for a single ticker dataframe:
    - Realized Volatility (20D, 60D, percentile, vol ratio)
    - Moving averages & ATR stretch
    - RVOL and RSI divergences
    - Drawdown & rally metrics
    - Historical probability match
    """
    if df.empty or len(df) < 65:
        return {}

    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    # 1. Moving Averages (All EMAs + benchmark SMAs)
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    
    # 2. RSI & ATR
    df["RSI"] = calculate_rsi(close, period=14)
    df["ATR20"] = calculate_atr(df, period=20)
    df["ATR_Stretch"] = (close - df["EMA20"]) / df["ATR20"].replace(0, np.nan)

    # 3. Relative Volume (RVOL)
    df["VolSMA20"] = volume.rolling(20).mean()
    df["RVOL"] = volume / df["VolSMA20"].replace(0, np.nan)

    # 4. Realized Volatility (Annualized %)
    log_ret = np.log(close / close.shift(1))
    df["RV20"] = log_ret.rolling(20).std() * np.sqrt(252) * 100
    df["RV60"] = log_ret.rolling(60).std() * np.sqrt(252) * 100
    df["Vol_Ratio"] = df["RV20"] / df["RV60"].replace(0, np.nan)

    # Realized Vol Percentile (over last 252 and 756 trading days)
    rv20_series = df["RV20"].dropna()
    curr_rv20 = rv20_series.iloc[-1] if not rv20_series.empty else np.nan
    rv20_1y = rv20_series.iloc[-252:] if len(rv20_series) >= 252 else rv20_series
    rv20_pct_1y = (rv20_1y < curr_rv20).mean() * 100 if len(rv20_1y) > 0 else 50.0

    # 5. Drawdown Analytics
    cummax = close.cummax()
    df["Drawdown_ATH"] = (close - cummax) / cummax * 100
    
    # 52-week high drawdown
    rolling_52w_max = close.rolling(252, min_periods=20).max()
    df["Drawdown_52W"] = (close - rolling_52w_max) / rolling_52w_max * 100

    # Current Drawdown duration (days since last ATH)
    is_ath = (close == cummax)
    last_ath_idx = np.where(is_ath)[0][-1] if np.any(is_ath) else 0
    days_in_dd = len(df) - 1 - last_ath_idx

    # Historical Drawdown Profile (All completed drawdowns)
    dd_series = df["Drawdown_ATH"]
    median_dd = float(dd_series[dd_series < -1.0].median()) if (dd_series < -1.0).any() else -3.0
    p75_dd = float(dd_series[dd_series < -1.0].quantile(0.25)) if (dd_series < -1.0).any() else -6.0
    max_dd = float(dd_series.min())

    # 6. Current Ticker Values
    curr_close = float(close.iloc[-1])
    curr_date = str(df["Date"].iloc[-1]).split()[0]
    curr_ema20 = float(df["EMA20"].iloc[-1])
    curr_ema50 = float(df["EMA50"].iloc[-1]) if pd.notnull(df["EMA50"].iloc[-1]) else curr_close
    curr_ema200 = float(df["EMA200"].iloc[-1]) if pd.notnull(df["EMA200"].iloc[-1]) else curr_close
    curr_sma50 = float(df["SMA50"].iloc[-1]) if pd.notnull(df["SMA50"].iloc[-1]) else curr_close
    curr_sma200 = float(df["SMA200"].iloc[-1]) if pd.notnull(df["SMA200"].iloc[-1]) else curr_close
    curr_rsi = float(df["RSI"].iloc[-1])
    curr_rvol = float(df["RVOL"].iloc[-1]) if pd.notnull(df["RVOL"].iloc[-1]) else 1.0
    curr_atr_stretch = float(df["ATR_Stretch"].iloc[-1]) if pd.notnull(df["ATR_Stretch"].iloc[-1]) else 0.0
    curr_rv60 = float(df["RV60"].iloc[-1]) if pd.notnull(df["RV60"].iloc[-1]) else curr_rv20
    curr_vol_ratio = float(df["Vol_Ratio"].iloc[-1]) if pd.notnull(df["Vol_Ratio"].iloc[-1]) else 1.0
    curr_dd_ath = float(df["Drawdown_ATH"].iloc[-1])
    curr_dd_52w = float(df["Drawdown_52W"].iloc[-1])

    # Percentage changes
    pct_1d = float((curr_close / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0.0
    pct_5d = float((curr_close / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0.0
    pct_20d = float((curr_close / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0.0
    pct_60d = float((curr_close / close.iloc[-61] - 1) * 100) if len(close) > 60 else 0.0

    # Distances to Moving Averages (EMAs + SMAs)
    dist_ema20 = (curr_close / curr_ema20 - 1) * 100
    dist_ema50 = (curr_close / curr_ema50 - 1) * 100
    dist_ema200 = (curr_close / curr_ema200 - 1) * 100
    dist_sma50 = (curr_close / curr_sma50 - 1) * 100
    dist_sma200 = (curr_close / curr_sma200 - 1) * 100

    # RSI Divergence detection
    rsi_div = detect_rsi_divergence(df, window=20)

    # Volatility Status description
    if curr_vol_ratio < 0.85:
        vol_status = "Vol Squeeze (Coiling for Breakout)"
    elif curr_vol_ratio > 1.25:
        vol_status = "Vol Explosion / Expansion"
    else:
        vol_status = "Balanced Vol Regime"

    # Exhaustion / Technical State description
    if curr_atr_stretch > 2.0 or (curr_rsi > 70 and "Bearish" in rsi_div):
        tech_status = "[Alert] Upside Exhaustion (Overbought / Pullback Risk)"
    elif curr_atr_stretch < -2.0 or (curr_rsi < 30 and "Bullish" in rsi_div):
        tech_status = "[Signal] Downside Exhaustion (Oversold Dip / Bounce Watch)"
    elif curr_dd_ath < -3.0 and curr_close > curr_sma50 and curr_rvol < 0.9:
        tech_status = "[Signal] Orderly Pullback to Support (Low Volume Dip)"
    elif curr_close > curr_ema20 and curr_ema20 > curr_sma50:
        tech_status = "Healthy Uptrend Continuation"
    else:
        tech_status = "Neutral / Consolidation"

    # 7. Historical Edge Match Engine
    edge_match = find_historical_edge(df, curr_dd_ath, curr_rsi, curr_atr_stretch, curr_rvol, curr_close > curr_sma50)

    return {
        "date": curr_date,
        "close": round(curr_close, 2),
        "pct_1d": round(pct_1d, 2),
        "pct_5d": round(pct_5d, 2),
        "pct_20d": round(pct_20d, 2),
        "pct_60d": round(pct_60d, 2),
        "ema20": round(curr_ema20, 2),
        "ema50": round(curr_ema50, 2),
        "ema200": round(curr_ema200, 2),
        "sma50": round(curr_sma50, 2),
        "sma200": round(curr_sma200, 2),
        "dist_ema20": round(dist_ema20, 2),
        "dist_ema50": round(dist_ema50, 2),
        "dist_ema200": round(dist_ema200, 2),
        "dist_sma50": round(dist_sma50, 2),
        "dist_sma200": round(dist_sma200, 2),
        "rsi": round(curr_rsi, 1),
        "rsi_divergence": rsi_div,
        "rvol": round(curr_rvol, 2),
        "atr_stretch": round(curr_atr_stretch, 2),
        # Realized Volatility suite
        "rv20": round(curr_rv20, 1),
        "rv60": round(curr_rv60, 1),
        "vol_ratio": round(curr_vol_ratio, 2),
        "rv20_pct_1y": round(rv20_pct_1y, 1),
        "vol_status": vol_status,
        # Drawdowns
        "drawdown_ath": round(curr_dd_ath, 2),
        "drawdown_52w": round(curr_dd_52w, 2),
        "days_in_drawdown": days_in_dd,
        "median_drawdown": round(median_dd, 2),
        "p75_drawdown": round(p75_dd, 2),
        "max_drawdown": round(max_dd, 2),
        "tech_status": tech_status,
        "edge_match": edge_match
    }

def find_historical_edge(df: pd.DataFrame, curr_dd: float, curr_rsi: float, curr_stretch: float, curr_rvol: float, above_50: bool) -> dict:
    """
    Scans the entire price history for similar technical setups and returns forward probabilities.
    """
    if len(df) < 250:
        return {"sample_size": 0, "status": "Insufficient history"}

    closes = df["Close"].values
    rsis = df["RSI"].values
    dd_aths = df["Drawdown_ATH"].values
    sma50s = df["SMA50"].values

    matches = []
    atr_values = df["ATR20"].replace(0, np.nan).fillna(1.0).values
    ema20_values = df["EMA20"].fillna(df["Close"]).values

    # Identify match rule based on current state
    if curr_stretch > 2.0 or curr_rsi > 70:
        rule_name = "Overbought Exhaustion Fade (RSI > 68 or ATR Stretch > 1.8)"
        cond = (rsis > 68) | (df["ATR_Stretch"].fillna(0).values > 1.8)
    elif curr_dd < -3.0 and above_50:
        rule_name = f"Uptrend Pullback ({round(curr_dd-2, 0)}% to {round(curr_dd+2, 0)}% Drawdown above 50SMA)"
        cond = (dd_aths <= curr_dd + 2.0) & (dd_aths >= curr_dd - 3.0) & (closes > sma50s)
    elif curr_stretch < -1.8 or curr_rsi < 35:
        rule_name = "Oversold Washout Bounce (RSI < 35 or ATR Stretch < -1.8)"
        cond = (rsis < 35) | (df["ATR_Stretch"].fillna(0).values < -1.8)
    else:
        rule_name = "Trend Continuation / Value Pullback to 20 EMA"
        cond = (np.abs(closes - ema20_values) / atr_values < 0.6) & (closes > sma50s)

    # We need at least 60 trading days of future data to test forward outcomes
    valid_indices = np.where(cond)[0]
    valid_indices = [i for i in valid_indices if i < len(closes) - 60]

    # Cluster filter: avoid counting adjacent days of the same signal
    filtered_indices = []
    last_idx = -999
    for idx in valid_indices:
        if idx - last_idx >= 5: # At least 5 days apart
            filtered_indices.append(idx)
            last_idx = idx

    sample_size = len(filtered_indices)
    if sample_size < 5:
        return {
            "rule_name": rule_name,
            "sample_size": sample_size,
            "win_rate_5d": 0.0,
            "win_rate_20d": 0.0,
            "win_rate_60d": 0.0,
            "median_return_20d": 0.0,
            "max_adverse_excursion": 0.0
        }

    ret_5d = [(closes[i+5] / closes[i] - 1) * 100 for i in filtered_indices]
    ret_20d = [(closes[i+20] / closes[i] - 1) * 100 for i in filtered_indices]
    ret_60d = [(closes[i+60] / closes[i] - 1) * 100 for i in filtered_indices]
    
    # Max drawdown over next 20 days (Max Adverse Excursion)
    mae_20d = []
    for i in filtered_indices:
        window_min = np.min(closes[i:i+20])
        mae_20d.append((window_min / closes[i] - 1) * 100)

    win_5d = (np.array(ret_5d) > 0).mean() * 100
    win_20d = (np.array(ret_20d) > 0).mean() * 100
    win_60d = (np.array(ret_60d) > 0).mean() * 100
    med_20d = float(np.median(ret_20d))
    avg_mae = float(np.median(mae_20d))

    return {
        "rule_name": rule_name,
        "sample_size": sample_size,
        "win_rate_5d": round(win_5d, 1),
        "win_rate_20d": round(win_20d, 1),
        "win_rate_60d": round(win_60d, 1),
        "median_return_20d": round(med_20d, 2),
        "max_adverse_excursion": round(avg_mae, 2)
    }
