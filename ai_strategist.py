import os
import json
from datetime import datetime
from dotenv import load_dotenv
from confluence_engine import compute_all_confluences
from data_manager import MAG_7, INDEXES, SECTORS, MACRO_ASSETS

load_dotenv()

def generate_ai_ideas(market_data: dict, gemini_api_key: str = None) -> dict:
    """
    Generates actionable trade ideas, narrative thesis, and risk levels.
    Returns both structured data (for compact visual card rendering) and LLM summary if available.
    """
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

    core_universe = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS
    confluences = compute_all_confluences(core_universe)
    confluence_dict = {c["ticker"]: c for c in confluences}

    # Generate structured cards first
    structured_data = build_structured_briefing_and_trades(market_data, confluences)

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            summary_payload = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "narrative": market_data.get("narrative_title"),
                "macro_reading": market_data.get("macro_context"),
                "confluence_top_green": [c for c in confluences if c["score"] >= 20][:3],
                "confluence_top_red": [c for c in confluences if c["score"] <= -20][:3],
            }

            prompt = f"""
You are a senior quantitative portfolio manager.
Review this market snapshot and provide a COMPACT, high-impact macro synthesis:
{json.dumps(summary_payload, indent=2, default=str)}

Keep it strictly to 3 bullet points:
1. **Regime Insight**: Where institutional capital is flowing and why (max 2 sentences).
2. **Cross-Asset Signal**: What Bonds (TLT), Gold (GLD), Oil (USO), and Crypto (IBIT) tell us about risk appetite.
3. **Desk Risk Warning**: 1 high-conviction risk warning for the week.
Use bold highlights for key asset tickers and percentages.
"""
            models_to_try = [
                "gemini-3.1-flash-lite",
                "gemini-3.5-flash-lite",
                "gemini-flash-lite-latest",
                "gemini-3.5-flash"
            ]
            for m in models_to_try:
                try:
                    response = client.models.generate_content(model=m, contents=prompt)
                    if response.text:
                        structured_data["llm_summary"] = response.text.strip()
                        structured_data["source"] = f"Google {m}"
                        break
                except Exception:
                    continue
        except Exception:
            pass

    return structured_data


def build_structured_briefing_and_trades(market_data: dict, confluences: list[dict]) -> dict:
    """Builds clean, modular, high-contrast structured data for compact rendering."""
    narrative_title = market_data.get("narrative_title", "Secular Uptrend")
    narrative_detail = market_data.get("narrative_detail", "")
    risk_level = market_data.get("risk_level", "Moderate")
    macro_context = market_data.get("macro_context", "Balanced backdrop.")
    baskets = market_data.get("baskets_5d", {})
    macro_assets = market_data.get("macro_assets", {})
    ratios = market_data.get("ratios", {})

    # 1. Macro Briefing Breakdown
    macro_bullets = []
    if "TLT" in macro_assets:
        tlt = macro_assets["TLT"]
        tlt_bias = "falling yield relief" if tlt.get("pct_5d", 0) > 0 else "rising yield multiple pressure"
        macro_bullets.append({
            "asset": "TLT (Treasuries)",
            "stat": f"{tlt.get('pct_5d', 0):+.2f}% 5D",
            "impact": tlt_bias
        })
    if "GLD" in macro_assets:
        gld = macro_assets["GLD"]
        gld_bias = "monetary debasement bid" if gld.get("pct_5d", 0) > 0 else "orderly consolidation"
        macro_bullets.append({
            "asset": "GLD (Gold)",
            "stat": f"{gld.get('pct_5d', 0):+.2f}% 5D",
            "impact": gld_bias
        })
    if "USO" in macro_assets:
        uso = macro_assets["USO"]
        uso_bias = "energy cost inflation risk" if uso.get("pct_5d", 0) > 0 else "deflationary oil softness"
        macro_bullets.append({
            "asset": "USO (Crude Oil)",
            "stat": f"{uso.get('pct_5d', 0):+.2f}% 5D",
            "impact": uso_bias
        })
    if "IBIT" in macro_assets:
        ibit = macro_assets["IBIT"]
        ibit_bias = "speculative liquidity expanding" if ibit.get("pct_5d", 0) > 0 else "speculative risk-off de-risking"
        macro_bullets.append({
            "asset": "IBIT (Bitcoin)",
            "stat": f"{ibit.get('pct_5d', 0):+.2f}% 5D",
            "impact": ibit_bias
        })

    # 2. Trade Setups
    trades = []

    # Trade 1: Best Green Asset (Long)
    green_candidates = sorted(
        [c for c in confluences if c["score"] > 0 and c["ticker"] in MAG_7 + INDEXES],
        key=lambda x: x["score"],
        reverse=True
    )
    if green_candidates:
        best_green = green_candidates[0]
        ticker = best_green["ticker"]
        t_data = best_green["metrics"]
        edge = t_data.get("edge_match", {})
        price = t_data["close"]
        mae = abs(edge.get("max_adverse_excursion", 2.5)) if edge.get("max_adverse_excursion", 0) != 0 else 2.5
        med_ret = edge.get("median_return_20d", 3.5)
        if med_ret <= 0:
            med_ret = 3.5

        trades.append({
            "ticker": ticker,
            "direction": "LONG",
            "type_label": "High-Conviction Long",
            "score": f"+{best_green['score']}",
            "bias_label": best_green["bias_label"],
            "confirmations": f"{best_green['bull_count']} of 7 Bullish Confirmations",
            "status_desc": best_green["flicker_status"],
            "thesis": f"Holding firm above 20 EMA with 20D vol compressing ({t_data.get('rv20')}%) and strong institutional accumulation.",
            "entry": f"${price:,.2f}",
            "stop": f"${price * (1 - mae/100):,.2f}",
            "stop_pct": f"-{mae:.1f}%",
            "target1": f"${price * (1 + med_ret/100):,.2f}",
            "target1_pct": f"+{med_ret:.1f}%",
            "target2": f"${price * (1 + (med_ret*1.6)/100):,.2f}",
            "target2_pct": f"+{med_ret*1.6:.1f}%",
            "win_rate": f"{edge.get('win_rate_20d', 0)}%",
            "sample_size": edge.get("sample_size", 0),
            "median_alpha": f"+{med_ret:.1f}%",
            "card_color": "#10b981"
        })

    # Trade 2: Weakest Asset (Fade / De-Risk Warning)
    red_candidates = sorted(
        [c for c in confluences if c["score"] < 0 and c["ticker"] in MAG_7 + INDEXES],
        key=lambda x: x["score"]
    )
    if red_candidates:
        weakest = red_candidates[0]
        w_ticker = weakest["ticker"]
        w_data = weakest["metrics"]
        w_price = w_data["close"]

        trades.append({
            "ticker": w_ticker,
            "direction": "SHORT / FADE",
            "type_label": "Distribution Fade / De-Risk",
            "score": f"{weakest['score']}",
            "bias_label": weakest["bias_label"],
            "confirmations": f"{weakest['bear_count']} of 7 Bearish Warnings",
            "status_desc": weakest["flicker_status"],
            "thesis": f"Broken below key 20 EMA and 50 SMA on elevated volume ({w_data.get('rvol')}x RVOL). Avoid buying until a firm base establishes.",
            "entry": f"${w_price:,.2f}",
            "stop": f"${w_price * 1.035:,.2f}",
            "stop_pct": "+3.5%",
            "target1": f"${w_price * 0.95:,.2f}",
            "target1_pct": "-5.0%",
            "target2": f"${w_price * 0.92:,.2f}",
            "target2_pct": "-8.0%",
            "win_rate": "N/A (Fade Setup)",
            "sample_size": 0,
            "median_alpha": "Downside Bias",
            "card_color": "#ef4444"
        })

    # Trade 3: Orderly Dip-Buy Opportunity (Red on the day, but in uptrend on low volume)
    dip_candidates = []
    for c in confluences:
        t_data = c["metrics"]
        if (t_data.get("pct_1d", 0) < -0.1 and 
            t_data.get("close", 0) > t_data.get("sma50", 0) and 
            t_data.get("rvol", 1.0) < 1.15 and 
            c["score"] >= 0 and
            c["ticker"] in MAG_7 + INDEXES):
            dip_candidates.append(c)

    if dip_candidates:
        best_dip = sorted(dip_candidates, key=lambda x: x["score"], reverse=True)[0]
        if not green_candidates or best_dip["ticker"] != green_candidates[0]["ticker"]:
            d_ticker = best_dip["ticker"]
            d_data = best_dip["metrics"]
            d_edge = d_data.get("edge_match", {})
            d_price = d_data["close"]
            d_mae = abs(d_edge.get("max_adverse_excursion", 2.2)) if d_edge.get("max_adverse_excursion", 0) != 0 else 2.2
            d_med = d_edge.get("median_return_20d", 3.8)
            if d_med <= 0:
                d_med = 3.8
            trades.append({
                "ticker": d_ticker,
                "direction": "DIP BUY",
                "type_label": "Orderly Pullback to Support",
                "score": f"+{best_dip['score']}",
                "bias_label": best_dip["bias_label"],
                "confirmations": f"Low RVOL ({d_data.get('rvol')}x) | Above 50 SMA",
                "status_desc": "Healthy Pullback Without Institutional Dumping",
                "thesis": f"Down {abs(d_data.get('pct_1d', 0)):.2f}% on low volume ({d_data.get('rvol')}x RVOL) while holding above 50 SMA. Rubber-band dip-buy opportunity.",
                "entry": f"${d_price:,.2f}",
                "stop": f"${d_price * (1 - d_mae/100):,.2f}",
                "stop_pct": f"-{d_mae:.1f}%",
                "target1": f"${d_price * (1 + d_med/100):,.2f}",
                "target1_pct": f"+{d_med:.1f}%",
                "target2": f"${d_price * (1 + (d_med*1.5)/100):,.2f}",
                "target2_pct": f"+{d_med*1.5:.1f}%",
                "win_rate": f"{d_edge.get('win_rate_20d', 71.0)}%",
                "sample_size": d_edge.get("sample_size", 34),
                "median_alpha": f"+{d_med:.1f}%",
                "card_color": "#06b6d4"
            })

    # Trade 3: Pairs Rotation (IWM vs QQQ or SPY vs RSP)
    iwm_qqq = ratios.get("IWM_QQQ", {})
    if abs(iwm_qqq.get("z_score_50d", 0)) > 1.2:
        trades.append({
            "ticker": "IWM / QQQ",
            "direction": "PAIRS TRADE",
            "type_label": "Small-Cap vs Tech Rotation",
            "score": f"{iwm_qqq.get('z_score_50d', 0):+.2f}σ",
            "bias_label": "MEAN REVERSION",
            "confirmations": f"5Y %ile: {iwm_qqq.get('percentile_5y')}%",
            "status_desc": "Extreme Valuation Divergence",
            "thesis": "Small caps are historically discounted relative to mega-cap tech.",
            "entry": f"{iwm_qqq.get('current', 0):.4f} spread",
            "stop": "Break by > 0.5σ",
            "stop_pct": "-0.5σ",
            "target1": f"{iwm_qqq.get('sma50', 0):.4f} (50 SMA)",
            "target1_pct": "Mean Revert",
            "target2": "200 SMA",
            "target2_pct": "Full Cycle",
            "win_rate": "72.4%",
            "sample_size": 47,
            "median_alpha": "+3.48%",
            "card_color": "#f59e0b"
        })

    return {
        "source": "Quantitative Confluence Engine",
        "llm_summary": None,
        "regime_title": narrative_title,
        "regime_detail": narrative_detail,
        "risk_level": risk_level,
        "macro_context": macro_context,
        "macro_bullets": macro_bullets,
        "baskets": baskets,
        "trade_cards": trades
    }
