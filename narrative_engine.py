import pandas as pd
import numpy as np
from data_manager import load_ticker_data, SECTORS, INDEXES, MAG_7, MACRO_ASSETS
from indicators import compute_all_metrics

def compute_ratio_analytics(ratio_name: str) -> dict:
    """Computes price, 5D/20D trend, 20 EMA, 50 EMA, and actionable states for synthetic pairs."""
    df = load_ticker_data(ratio_name)
    if df.empty or len(df) < 50:
        return {}

    close = df["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    curr_close = float(close.iloc[-1])
    curr_ema20 = float(ema20.iloc[-1])
    curr_ema50 = float(ema50.iloc[-1])

    dist_ema20 = float((curr_close / curr_ema20 - 1) * 100)
    dist_ema50 = float((curr_close / curr_ema50 - 1) * 100)

    std50 = close.rolling(50).std()
    z_score_50 = (close - ema50) / std50.replace(0, np.nan)
    curr_z = float(z_score_50.iloc[-1])

    # 5-year percentile (preserved for backward compatibility)
    lookback = min(len(close), 1260)
    history_window = close.iloc[-lookback:]
    pctile_5y = float((history_window < curr_close).mean() * 100)

    pct_1d = float((curr_close / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0.0
    pct_5d = float((curr_close / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0.0
    pct_20d = float((curr_close / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0.0

    # Dynamic actionable state driven purely by 5D/20D trend and 20 EMA / 50 EMA relationship
    if "SPY" in ratio_name and "RSP" in ratio_name:
        # SPY/RSP: Cap-Weighted vs Equal-Weight (Mega Concentration vs Broad Breadth)
        if dist_ema20 > 0 and dist_ema50 > 0:
            if curr_z > 1.5 or dist_ema20 > 1.2:
                state_label = "Mega Concentration Peaking"
                action_text = "Overextended vs 20/50 EMA; Rotate into RSP"
                state_color = "#ef4444"
            elif pct_5d > 0.5:
                state_label = "Mega-Cap Tech Dominance"
                action_text = "5D/20D Trend Bullish; Stay with Cap-Weight"
                state_color = "#f59e0b"
            else:
                state_label = "Mega-Cap Bull Trend (Slowing)"
                action_text = "Holding Above 20/50 EMA; Monitor for Divergence"
                state_color = "#f59e0b"
        elif dist_ema20 < 0 and dist_ema50 < 0:
            if pct_5d < -0.5 and pct_20d < -1.0:
                state_label = "Broad Breadth Expansion"
                action_text = "Downtrend Below 20/50 EMA; Overweight RSP"
                state_color = "#10b981"
            else:
                state_label = "Equal-Weight Outperformance"
                action_text = "RSP Gaining Relative Strength vs Mega-Caps"
                state_color = "#10b981"
        elif dist_ema20 < 0 and dist_ema50 >= 0:
            state_label = "Pullback to 50 EMA Support"
            action_text = "Testing 50 EMA; RSP Rotation Building"
            state_color = "#10b981"
        else:
            state_label = "Balanced Breadth Regime"
            action_text = "Consolidating Across 20/50 EMA"
            state_color = "#94a3b8"

    elif "IWM" in ratio_name and "QQQ" in ratio_name:
        # IWM/QQQ: Small-Caps vs Mega Tech Rotation
        if dist_ema20 > 0 and dist_ema50 > 0:
            if pct_5d > 1.0 and pct_20d > 2.0:
                state_label = "Aggressive Small-Cap Rotation"
                action_text = "5D/20D Momentum Strong; Overweight IWM"
                state_color = "#10b981"
            else:
                state_label = "Small-Cap Trend Above 20/50 EMA"
                action_text = "Healthy Small-Cap Outperformance"
                state_color = "#10b981"
        elif dist_ema20 < 0 and dist_ema50 < 0:
            if curr_z < -1.5 or (pct_5d > 0.5 and dist_ema20 > -1.0):
                state_label = "Small-Cap Value Coiling"
                action_text = "Oversold vs 50 EMA; Watch for 20 EMA Reclaim"
                state_color = "#10b981"
            elif pct_5d < -0.5:
                state_label = "Tech Monopoly Regime"
                action_text = "Downtrend Below 20/50 EMA; Stay in QQQ"
                state_color = "#3b82f6"
            else:
                state_label = "Tech Leadership Steady"
                action_text = "Underweight Small-Caps; Respect Tech Trend"
                state_color = "#3b82f6"
        elif dist_ema20 >= 0 and dist_ema50 < 0:
            state_label = "Small-Cap Reversal Bid"
            action_text = "Reclaimed 20 EMA; Targeting 50 EMA Mean-Reversion"
            state_color = "#10b981"
        else:
            state_label = "Balanced Tech / Small"
            action_text = "Neutral Allocation (QQQ vs IWM)"
            state_color = "#94a3b8"
    else:
        state_label = "Range-Bound Ratio"
        action_text = "Consolidating Between 20/50 EMA"
        state_color = "#94a3b8"

    return {
        "name": ratio_name.replace("_", "/"),
        "current": round(float(curr_close), 4),
        "pct_1d": round(float(pct_1d), 2),
        "pct_5d": round(float(pct_5d), 2),
        "pct_20d": round(float(pct_20d), 2),
        "ema20": round(curr_ema20, 4),
        "ema50": round(curr_ema50, 4),
        "dist_ema20": round(dist_ema20, 2),
        "dist_ema50": round(dist_ema50, 2),
        "sma50": round(curr_ema50, 4),
        "z_score_50d": round(curr_z, 2),
        "percentile_5y": round(float(pctile_5y), 1),
        "is_stretched": abs(curr_z) >= 1.8,
        "actionable_state": state_label,
        "action_text": action_text,
        "state_color": state_color
    }

def analyze_market_narrative() -> dict:
    """
    Decodes the underlying market narrative based on:
    - Sector flow (Cyclicals vs Defensives vs Tech)
    - Breadth & Concentration Ratios (SPY/RSP, QQQ/RSP, IWM/QQQ)
    - Mag 7 Dispersion
    """
    # 1. Ratios
    spy_rsp = compute_ratio_analytics("SPY_RSP")
    qqq_rsp = compute_ratio_analytics("QQQ_RSP")
    iwm_qqq = compute_ratio_analytics("IWM_QQQ")

    # 2. Sector Metrics
    sector_results = {}
    for sec in SECTORS:
        df_sec = load_ticker_data(sec)
        metrics = compute_all_metrics(df_sec)
        if metrics:
            sector_results[sec] = metrics

    # 3. Defensive vs Cyclical vs Growth baskets
    growth_constituents = ["XLK", "SMH", "XLC", "XLY"]
    cyclical_constituents = ["XLF", "XLI", "XLB"]
    defensive_constituents = ["XLU", "XLP", "XLV"]
    energy_constituents = ["XLE"]

    growth_vals = [sector_results[s]["pct_5d"] for s in growth_constituents if s in sector_results]
    cyclical_vals = [sector_results[s]["pct_5d"] for s in cyclical_constituents if s in sector_results]
    defensive_vals = [sector_results[s]["pct_5d"] for s in defensive_constituents if s in sector_results]
    energy_vals = [sector_results[s]["pct_5d"] for s in energy_constituents if s in sector_results]

    growth_avg_5d = float(np.mean(growth_vals)) if growth_vals else 0.0
    cyclical_avg_5d = float(np.mean(cyclical_vals)) if cyclical_vals else 0.0
    defensive_avg_5d = float(np.mean(defensive_vals)) if defensive_vals else 0.0
    energy_5d = float(np.mean(energy_vals)) if energy_vals else 0.0

    # 4. Mag 7 Dispersion
    mag7_results = {}
    for m in MAG_7:
        df_m = load_ticker_data(m)
        met = compute_all_metrics(df_m)
        if met:
            mag7_results[m] = met

    # 5. Cross-Asset Macro Barometers: GLD, USO, TLT, IBIT
    macro_results = {}
    for asset in MACRO_ASSETS:
        df_a = load_ticker_data(asset)
        met_a = compute_all_metrics(df_a)
        if met_a:
            macro_results[asset] = met_a

    tlt_5d = macro_results.get("TLT", {}).get("pct_5d", 0.0)
    gld_5d = macro_results.get("GLD", {}).get("pct_5d", 0.0)
    uso_5d = macro_results.get("USO", {}).get("pct_5d", 0.0)
    ibit_5d = macro_results.get("IBIT", {}).get("pct_5d", 0.0)

    macro_notes = []
    if tlt_5d < -1.5:
        macro_notes.append("Treasury yields rising (TLT weak), creating multiple pressure on long-duration tech.")
    elif tlt_5d > 1.5:
        macro_notes.append("Bonds bidding up (TLT strong), signaling falling yield expectations or flight-to-safety.")

    if gld_5d > 1.5 and tlt_5d <= 0:
        macro_notes.append("Gold outperforming bonds (GLD strong), pricing in negative real yields or monetary debasement.")

    if uso_5d > 3.0:
        macro_notes.append("Crude oil (USO) surging, introducing commodity cost-push inflation risks.")

    if ibit_5d > 3.0:
        macro_notes.append("Bitcoin (IBIT) expanding, signaling robust speculative risk-on liquidity.")
    elif ibit_5d < -4.0:
        macro_notes.append("Bitcoin (IBIT) de-risking, warning of tightening speculative liquidity.")

    macro_context_str = " | ".join(macro_notes) if macro_notes else "Macro cross-assets (Gold, Oil, Bonds, Bitcoin) trading in balanced ranges."

    # 5. Narrative Identification
    narrative_title = "Secular Uptrend with Balanced Participation"
    narrative_detail = "Capital is moving evenly across growth, cyclicals, and defensives."
    risk_level = "Low to Moderate"

    # Check for Defensive Flight
    if defensive_avg_5d > growth_avg_5d + 2.0 and defensive_avg_5d > 1.0:
        narrative_title = "Stealth De-Risking / Defensive Flight"
        narrative_detail = (
            "Smart money is quietly parking into defensive yield proxies (Utilities, Staples, Healthcare) "
            "while trimming high-beta and growth exposure. Often precedes index consolidation or volatility expansion."
        )
        risk_level = "Elevated Caution"
    # Check for Broadening / Reflation
    elif cyclical_avg_5d > growth_avg_5d + 1.5 and iwm_qqq.get("pct_5d", 0) > 1.0:
        narrative_title = "Broadening & Cyclical Reflation (Soft Landing)"
        narrative_detail = (
            "Capital is rotating aggressively out of crowded mega-cap tech into Financials, Industrials, and Small-Caps (IWM). "
            "This indicates healthy market breadth expansion and economic optimism."
        )
        risk_level = "Bullish / Risk-On"
    # Check for Tech Climax / Narrow Breadth Squeeze
    elif growth_avg_5d > cyclical_avg_5d + 2.0 and spy_rsp.get("z_score_50d", 0) > 1.5:
        narrative_title = "Mega-Cap Tech Climax & Narrow Breadth"
        narrative_detail = (
            "Rally is highly concentrated in Mag 7 and Semiconductor names while equal-weight RSP and broader sectors lag. "
            "Narrow breadth creates fragility; watch for upside exhaustion signals."
        )
        risk_level = "High Fragility (Vulnerable to Mean-Reversion)"
    # Check for Energy / Inflation Surge
    elif energy_5d > 3.0 and growth_avg_5d < -0.5:
        narrative_title = "Inflation / Commodity Surge Squeeze"
        narrative_detail = (
            "Energy and hard assets are outperforming as rising yields or commodity prices pressure tech valuations."
        )
        risk_level = "Selective (Play Commodities/Energy, Avoid Long-Duration Tech)"
    # Check for Capitulation Dip
    elif all(sec.get("pct_5d", 0) < -3.0 for sec in sector_results.values() if isinstance(sec, dict)):
        narrative_title = "Indiscriminate Washout / Liquidation Dip"
        narrative_detail = (
            "Broad-based selling across all sectors regardless of fundamentals. Historically represents high-probability bounce entry."
        )
        risk_level = "High Volatility / Prime Asymmetric Long Dip Opportunity"

    return {
        "narrative_title": narrative_title,
        "narrative_detail": narrative_detail,
        "risk_level": risk_level,
        "macro_context": macro_context_str,
        "ratios": {
            "SPY_RSP": spy_rsp,
            "QQQ_RSP": qqq_rsp,
            "IWM_QQQ": iwm_qqq
        },
        "baskets_5d": {
            "Growth (XLK, SMH, XLC, XLY)": round(growth_avg_5d, 2),
            "Cyclicals (XLF, XLI, XLB)": round(cyclical_avg_5d, 2),
            "Defensives (XLU, XLP, XLV)": round(defensive_avg_5d, 2),
            "Energy (XLE)": round(energy_5d, 2),
            "Growth": round(growth_avg_5d, 2),
            "Cyclical": round(cyclical_avg_5d, 2),
            "Defensive": round(defensive_avg_5d, 2),
            "Energy": round(energy_5d, 2),
            "Growth (XLK, XLC, XLY)": round(growth_avg_5d, 2),
            "Growth (XLK, SMH, XLC)": round(growth_avg_5d, 2),
            "Cyclicals (XLI, XLF, XME)": round(cyclical_avg_5d, 2)
        },
        "macro_assets": macro_results,
        "sectors": sector_results,
        "mag7": mag7_results
    }
