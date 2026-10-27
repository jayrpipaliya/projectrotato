import numpy as np
import pandas as pd
from data_manager import load_ticker_data, MAG_7, INDEXES, SECTORS
from indicators import compute_all_metrics

def compute_confluence_score(ticker: str, market_narrative: dict = None) -> dict:
    """
    Computes a multi-factor conviction score from -100 (Deep Bearish Red) to +100 (Deep Bullish Green).
    Scores 7 independent confirmation layers:
    1. Structural Trend (Price vs 20EMA, 50SMA, 200SMA)
    2. Value / Extension Zone (Drawdown from ATH, ATR Stretch)
    3. Volume Flow (RVOL & Up/Down Volume participation)
    4. Momentum & Divergence (14D RSI, Bullish/Bearish Divergence)
    5. Realized Volatility Regime (20D Realized Vol, 20D/60D Vol Ratio Squeeze)
    6. Macro & Sector Confluence (Relative performance vs SPY and sector flows)
    7. Historical Probability & Statistical Edge (20-Year Backtested Win Rates & Median Returns)
    """
    df = load_ticker_data(ticker)
    if df.empty or len(df) < 60:
        return {}

    m = compute_all_metrics(df)
    if not m:
        return {}

    score = 0
    bull_confirmations = 0
    bear_confirmations = 0
    checklist = []

    close = m["close"]
    ema20 = m["ema20"]
    sma50 = m["sma50"]
    sma200 = m["sma200"]
    rsi = m["rsi"]
    rsi_div = m["rsi_divergence"]
    atr_stretch = m["atr_stretch"]
    rvol = m["rvol"]
    rv20 = m["rv20"]
    vol_ratio = m["vol_ratio"]
    dd_ath = m["drawdown_ath"]
    pct_5d = m["pct_5d"]

    # -------------------------------------------------------------
    # LAYER 1: STRUCTURAL TREND (Max +/- 15 pts)
    # -------------------------------------------------------------
    if close > ema20 and ema20 > sma50 and close > sma200:
        score += 15
        bull_confirmations += 1
        checklist.append({
            "layer": "Structural Trend",
            "status": "PASS_BULL",
            "points": "+15",
            "detail": f"Price (${close}) holding above 20 EMA (${ema20}), 50 SMA, and 200 SMA (Full Bullish Stack)"
        })
    elif close > sma50 and close > sma200:
        score += 10
        bull_confirmations += 1
        checklist.append({
            "layer": "Structural Trend",
            "status": "PASS_BULL",
            "points": "+10",
            "detail": f"Price above 50 SMA (${sma50}) and 200 SMA in secular uptrend"
        })
    elif close < ema20 and close < sma50 and close < sma200:
        score -= 15
        bear_confirmations += 1
        checklist.append({
            "layer": "Structural Trend",
            "status": "PASS_BEAR",
            "points": "-15",
            "detail": f"Price (${close}) broken below 20 EMA, 50 SMA, and 200 SMA (Full Bearish Stack)"
        })
    elif close < ema20 and close < sma50:
        score -= 10
        bear_confirmations += 1
        checklist.append({
            "layer": "Structural Trend",
            "status": "PASS_BEAR",
            "points": "-10",
            "detail": f"Price lost 20 EMA and 50 SMA support"
        })
    else:
        checklist.append({
            "layer": "Structural Trend",
            "status": "NEUTRAL",
            "points": "0",
            "detail": "Choppy trend structure / consolidating between key moving averages"
        })

    # -------------------------------------------------------------
    # LAYER 2: VALUE & EXTENSION ZONE (Max +/- 15 pts)
    # -------------------------------------------------------------
    if -10.0 <= dd_ath <= -3.0 and close > sma50 and atr_stretch < 0.5:
        score += 15
        bull_confirmations += 1
        checklist.append({
            "layer": "Value / Extension Zone",
            "status": "PASS_BULL",
            "points": "+15",
            "detail": f"Healthy pullback ({dd_ath}% from ATH) resting at support with ATR stretch reset ({atr_stretch} ATR)"
        })
    elif atr_stretch > 2.0 or (dd_ath > -0.5 and atr_stretch > 1.6):
        score -= 15
        bear_confirmations += 1
        checklist.append({
            "layer": "Value / Extension Zone",
            "status": "PASS_BEAR",
            "points": "-15",
            "detail": f"Overextended rubber-band (+{atr_stretch} ATR above 20 EMA) at or near ATH (Mean-reversion vulnerability)"
        })
    elif atr_stretch < -2.0:
        score += 15
        bull_confirmations += 1
        checklist.append({
            "layer": "Value / Extension Zone",
            "status": "PASS_BULL",
            "points": "+15",
            "detail": f"Deeply oversold rubber-band ({atr_stretch} ATR below 20 EMA) primed for mean-reversion snapback"
        })
    else:
        checklist.append({
            "layer": "Value / Extension Zone",
            "status": "NEUTRAL",
            "points": "0",
            "detail": f"Normal ATR stretch ({atr_stretch} ATR) within standard bands"
        })

    # -------------------------------------------------------------
    # LAYER 3: VOLUME & FLOW CONVICTION (Max +/- 15 pts)
    # -------------------------------------------------------------
    if pct_5d > 0 and rvol >= 1.2:
        score += 15
        bull_confirmations += 1
        checklist.append({
            "layer": "Volume Flow",
            "status": "PASS_BULL",
            "points": "+15",
            "detail": f"RVOL at {rvol}x on positive price progress (Active Institutional Accumulation)"
        })
    elif dd_ath < -3.0 and rvol <= 0.85:
        score += 15
        bull_confirmations += 1
        checklist.append({
            "layer": "Volume Flow",
            "status": "PASS_BULL",
            "points": "+15",
            "detail": f"Selling volume is completely drying up on the dip (RVOL {rvol}x) - Lack of aggressive selling"
        })
    elif pct_5d > 0 and rvol <= 0.70:
        score -= 15
        bear_confirmations += 1
        checklist.append({
            "layer": "Volume Flow",
            "status": "PASS_BEAR",
            "points": "-15",
            "detail": f"Price rising on anemic volume (RVOL {rvol}x) - Low volume drift / Lack of institutional sponsorship"
        })
    elif pct_5d < 0 and rvol >= 1.3:
        score -= 15
        bear_confirmations += 1
        checklist.append({
            "layer": "Volume Flow",
            "status": "PASS_BEAR",
            "points": "-15",
            "detail": f"Heavy selling pressure with RVOL {rvol}x (Institutional Distribution)"
        })
    else:
        checklist.append({
            "layer": "Volume Flow",
            "status": "NEUTRAL",
            "points": "0",
            "detail": f"Standard volume flow (RVOL {rvol}x)"
        })

    # -------------------------------------------------------------
    # LAYER 4: MOMENTUM & DIVERGENCE (Max +/- 15 pts)
    # -------------------------------------------------------------
    if "Bullish" in rsi_div or (rsi >= 40 and rsi <= 55 and pct_5d > -1.0):
        score += 15
        bull_confirmations += 1
        checklist.append({
            "layer": "Momentum & Divergence",
            "status": "PASS_BULL",
            "points": "+15",
            "detail": f"14D RSI ({rsi}) cleanly reset in constructive bounce territory; no bearish divergence"
        })
    elif "Bearish" in rsi_div:
        score -= 15
        bear_confirmations += 1
        checklist.append({
            "layer": "Momentum & Divergence",
            "status": "PASS_BEAR",
            "points": "-15",
            "detail": f"Bearish RSI Divergence detected! Price pushing higher while momentum decelerates"
        })
    elif rsi > 70:
        score -= 10
        bear_confirmations += 1
        checklist.append({
            "layer": "Momentum & Divergence",
            "status": "PASS_BEAR",
            "points": "-10",
            "detail": f"14D RSI is overbought at {rsi}"
        })
    elif rsi < 30:
        score += 10
        bull_confirmations += 1
        checklist.append({
            "layer": "Momentum & Divergence",
            "status": "PASS_BULL",
            "points": "+10",
            "detail": f"14D RSI is deeply oversold at {rsi} (Washout territory)"
        })
    else:
        checklist.append({
            "layer": "Momentum & Divergence",
            "status": "NEUTRAL",
            "points": "0",
            "detail": f"RSI neutral at {rsi}"
        })

    # -------------------------------------------------------------
    # LAYER 5: REALIZED VOLATILITY SQUEEZE (Max +/- 15 pts)
    # -------------------------------------------------------------
    if vol_ratio <= 0.85:
        score += 15
        bull_confirmations += 1
        checklist.append({
            "layer": "Realized Volatility",
            "status": "PASS_BULL",
            "points": "+15",
            "detail": f"Volatility Squeeze: 20D RV ({rv20}%) is compressed relative to 60D RV (Ratio {vol_ratio}). Coiling for expansion"
        })
    elif vol_ratio >= 1.30 and rsi > 65:
        score -= 15
        bear_confirmations += 1
        checklist.append({
            "layer": "Realized Volatility",
            "status": "PASS_BEAR",
            "points": "-15",
            "detail": f"Volatility Expansion Climax: 20D RV ({rv20}%) spiking into resistance (Ratio {vol_ratio})"
        })
    else:
        checklist.append({
            "layer": "Realized Volatility",
            "status": "NEUTRAL",
            "points": "0",
            "detail": f"Realized Volatility balanced: 20D RV at {rv20}% (Ratio {vol_ratio})"
        })

    # -------------------------------------------------------------
    # LAYER 6: MACRO & SECTOR CONFLUENCE (Max +/- 15 pts)
    # -------------------------------------------------------------
    # If SPY or QQQ is in uptrend and asset has positive relative strength
    spy_df = load_ticker_data("SPY")
    spy_close = spy_df["Close"].iloc[-1]
    spy_sma50 = spy_df["Close"].rolling(50).mean().iloc[-1]
    spy_pct_5d = (spy_close / spy_df["Close"].iloc[-6] - 1) * 100 if len(spy_df) > 5 else 0.0

    rel_strength = pct_5d - spy_pct_5d
    if rel_strength > 1.0 and spy_close > spy_sma50:
        score += 15
        bull_confirmations += 1
        checklist.append({
            "layer": "Macro Confluence",
            "status": "PASS_BULL",
            "points": "+15",
            "detail": f"Outperforming S&P 500 (+{round(rel_strength, 2)}% relative 5D alpha) in healthy market regime"
        })
    elif rel_strength < -2.0:
        score -= 15
        bear_confirmations += 1
        checklist.append({
            "layer": "Macro Confluence",
            "status": "PASS_BEAR",
            "points": "-15",
            "detail": f"Severe relative lag vs S&P 500 ({round(rel_strength, 2)}% relative 5D lag) - Capital flowing elsewhere"
        })
    else:
        checklist.append({
            "layer": "Macro Confluence",
            "status": "NEUTRAL",
            "points": "0",
            "detail": f"Performing in line with benchmark (+{round(rel_strength, 2)}% relative)"
        })

    # -------------------------------------------------------------
    # LAYER 7: HISTORICAL PROBABILITY & STATISTICAL EDGE (Max +/- 20 pts)
    # -------------------------------------------------------------
    edge = m.get("edge_match", {})
    sample_size = edge.get("sample_size", 0)
    win_rate_20d = edge.get("win_rate_20d", 0.0)
    med_ret = edge.get("median_return_20d", 0.0)
    rule_name = edge.get("rule_name", "Historical Setup")

    if sample_size >= 10:
        if win_rate_20d >= 72.0 and med_ret > 1.5:
            score += 20
            bull_confirmations += 1
            checklist.append({
                "layer": "Historical Probability & Edge",
                "status": "PASS_BULL",
                "points": "+20",
                "detail": f"High Empirical Edge: {win_rate_20d}% 20-day win rate ({sample_size} matches) with +{med_ret}% median forward return"
            })
        elif win_rate_20d >= 60.0 and med_ret > 0:
            score += 10
            bull_confirmations += 1
            checklist.append({
                "layer": "Historical Probability & Edge",
                "status": "PASS_BULL",
                "points": "+10",
                "detail": f"Positive Empirical Edge: {win_rate_20d}% win rate ({sample_size} matches) with +{med_ret}% median forward return"
            })
        elif win_rate_20d <= 35.0 or med_ret < -1.5:
            score -= 20
            bear_confirmations += 1
            checklist.append({
                "layer": "Historical Probability & Edge",
                "status": "PASS_BEAR",
                "points": "-20",
                "detail": f"Historical Failure Risk: Only {win_rate_20d}% win rate ({sample_size} matches) with {med_ret}% median return (Prone to fade/breakdown)"
            })
        elif win_rate_20d <= 45.0:
            score -= 10
            bear_confirmations += 1
            checklist.append({
                "layer": "Historical Probability & Edge",
                "status": "PASS_BEAR",
                "points": "-10",
                "detail": f"Sub-50% Empirical Odds: {win_rate_20d}% win rate across {sample_size} historical matches"
            })
        else:
            checklist.append({
                "layer": "Historical Probability & Edge",
                "status": "NEUTRAL",
                "points": "0",
                "detail": f"Neutral historical odds: {win_rate_20d}% win rate across {sample_size} matches (Coin-flip zone)"
            })
    else:
        checklist.append({
            "layer": "Historical Probability & Edge",
            "status": "NEUTRAL",
            "points": "0",
            "detail": f"Insufficient sample size ({sample_size} matches) for empirical edge"
        })

    # Bound score between -100 and +100
    final_score = int(np.clip(score, -100, 100))

    # Bias Label & Flicker Status
    if final_score >= 65:
        bias_label = "STRONG BULLISH"
        flicker_status = "Solid High Conviction Green"
        color_hex = "#00e676" # Vivid emerald green
    elif final_score >= 25:
        bias_label = "LEAN BULLISH"
        flicker_status = "Flickering Green (Building Confirmation)"
        color_hex = "#69f0ae" # Mint / lighter green
    elif final_score <= -65:
        bias_label = "STRONG BEARISH"
        flicker_status = "Solid High Conviction Red"
        color_hex = "#ff1744" # Vivid crimson red
    elif final_score <= -25:
        bias_label = "LEAN BEARISH"
        flicker_status = "Flickering Red (Early Exhaustion / Fade)"
        color_hex = "#ff5252" # Coral / lighter red
    else:
        bias_label = "NEUTRAL"
        flicker_status = "Neutral / Mixed Signals"
        color_hex = "#90a4ae" # Muted slate grey

    return {
        "ticker": ticker,
        "price": close,
        "score": final_score,
        "bias_label": bias_label,
        "flicker_status": flicker_status,
        "color_hex": color_hex,
        "bull_count": bull_confirmations,
        "bear_count": bear_confirmations,
        "total_layers": 7,
        "checklist": checklist,
        "metrics": m
    }

def compute_all_confluences(tickers: list[str]) -> list[dict]:
    """Computes conviction scores for a list of tickers, sorted from most bullish to most bearish."""
    results = []
    for t in tickers:
        res = compute_confluence_score(t)
        if res:
            results.append(res)
    return sorted(results, key=lambda x: x["score"], reverse=True)
