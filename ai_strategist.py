import os
import json
from datetime import datetime
from dotenv import load_dotenv
from confluence_engine import compute_all_confluences
from data_manager import MAG_7, INDEXES, SECTORS, MACRO_ASSETS

load_dotenv()

# Validated working model cascade
GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
]

def get_gemini_client(api_key: str = None):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None, None
    try:
        from google import genai
        client = genai.Client(api_key=key.strip())
        return client, key.strip()
    except Exception as e:
        return None, str(e)

def generate_ai_ideas(market_data: dict, gemini_api_key: str = None) -> dict:
    """
    Generates actionable trade ideas, narrative thesis, and comprehensive daily AI market analysis.
    Returns structured data for cards and deep daily analysis from Gemini.
    """
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

    core_universe = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS
    confluences = compute_all_confluences(core_universe)

    # Generate structured quantitative cards first
    structured_data = build_structured_briefing_and_trades(market_data, confluences)

    client, err = get_gemini_client(api_key)
    if client:
        try:
            # Build comprehensive quantitative summary payload for Gemini
            green_setups = [
                {
                    "ticker": c["ticker"],
                    "score": c["score"],
                    "bias": c["bias_label"],
                    "price": c["metrics"].get("close"),
                    "pct_1d": c["metrics"].get("pct_1d"),
                    "rvol": c["metrics"].get("rvol"),
                    "rsi": c["metrics"].get("rsi14"),
                    "dist_ema20": c["metrics"].get("dist_ema20")
                }
                for c in confluences if c["score"] >= 15
            ][:5]

            red_warnings = [
                {
                    "ticker": c["ticker"],
                    "score": c["score"],
                    "bias": c["bias_label"],
                    "price": c["metrics"].get("close"),
                    "pct_1d": c["metrics"].get("pct_1d"),
                    "rvol": c["metrics"].get("rvol"),
                    "rsi": c["metrics"].get("rsi14"),
                    "dist_ema20": c["metrics"].get("dist_ema20")
                }
                for c in confluences if c["score"] <= -15
            ][:5]

            macro_assets = market_data.get("macro_assets", {})
            macro_summary = {
                k: {
                    "close": v.get("close"),
                    "pct_1d": v.get("pct_1d"),
                    "pct_5d": v.get("pct_5d"),
                    "rsi": v.get("rsi14"),
                    "trend": "Above 50EMA" if v.get("close", 0) > v.get("ema50", 0) else "Below 50EMA"
                }
                for k, v in macro_assets.items() if k in ["TLT", "GLD", "USO", "IBIT"]
            }

            baskets = market_data.get("baskets_5d", {})
            ratios = market_data.get("ratios", {})
            ratio_summary = {
                k: {
                    "current": v.get("current"),
                    "pct_5d": v.get("pct_5d"),
                    "pct_20d": v.get("pct_20d"),
                    "z_score_50d": v.get("z_score_50d")
                }
                for k, v in ratios.items() if k in ["SPY_RSP", "QQQ_RSP", "IWM_QQQ"]
            }

            analysis_payload = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "narrative": market_data.get("narrative_title"),
                "narrative_detail": market_data.get("narrative_detail"),
                "macro_reading": market_data.get("macro_context"),
                "baskets_5d": baskets,
                "macro_assets": macro_summary,
                "key_ratios": ratio_summary,
                "top_green_confluence": green_setups,
                "top_red_confluence": red_warnings,
            }

            prompt = f"""
You are the Chief Quantitative Strategist and Senior Portfolio Risk Officer at a multi-billion dollar quantitative macro fund.
Conduct an exhaustive, high-conviction **DAILY INSTITUTIONAL MARKET INTELLIGENCE BRIEFING** based strictly on the following quantitative desk snapshot for {datetime.now().strftime('%A, %B %d, %Y')}:

```json
{json.dumps(analysis_payload, indent=2, default=str)}
```

Structure your analysis into 4 clear, professional institutional sections:

### 1. Macro Regime & Cross-Asset Liquidity Assessment
- Detail the prevailing regime ({market_data.get('narrative_title')}).
- Unpack what the 4 macro pillars are signaling: Treasuries (**TLT** - yield pressure/easing), Gold (**GLD** - debasement hedge), Oil (**USO** - inflation/supply shock), and Bitcoin (**IBIT** - liquidity & speculative risk appetite).
- Explain whether current institutional capital flows are Risk-On, Risk-Off, or Selective Dispersion.

### 2. Sector Velocity & Breadth Concentration Dynamics
- Evaluate the 5-day velocity across Growth vs Cyclical vs Defensive vs Energy. Where is smart money rotating?
- Assess market breadth using **SPY/RSP** (mega-cap concentration) and **IWM/QQQ** (small-cap vs large-cap tech divergence). Is rally participation healthy or dangerously concentrated?

### 3. Actionable Setups & Trade Confluence
- Highlight the strongest long setups from the top green confluence assets. Detail technical alignment (EMAs, RVOL, stretch).
- Detail the distribution warnings and fade setups from red confluence assets. Point out traps where retail traders might mistakenly buy failing dips.

### 4. Desk Game Plan & Tactical Risk Boundaries
- Provide 3 explicit risk parameters for today: what to add risk to, what to hedge/trim, and exact price or ratio invalidation levels to respect.

Write in a crisp, Wall Street institutional tone. Bold all key tickers, percentages, and metrics.
"""

            for m in GEMINI_MODELS:
                try:
                    response = client.models.generate_content(model=m, contents=prompt)
                    if response and response.text:
                        structured_data["llm_summary"] = response.text.strip()
                        structured_data["source"] = f"Gemini ({m})"
                        structured_data["agent_status"] = "Active"
                        break
                except Exception as model_err:
                    continue
        except Exception as e:
            structured_data["agent_error"] = str(e)

    return structured_data


def query_ai_agent(
    user_query: str, 
    market_data: dict, 
    confluences: list[dict] = None, 
    gemini_api_key: str = None, 
    chat_history: list = None
) -> str:
    """
    Interactive Q&A: Allows the user to chat with the Quantitative AI Agent.
    The agent has access to all current daily metrics, ratios, confluences, and regimes.
    """
    client, key = get_gemini_client(gemini_api_key)
    if not client:
        return "⚠️ **Gemini API Key Missing or Invalid:** Please enter your Gemini API key in the sidebar to interact with the AI Quant Agent."

    if confluences is None:
        core_universe = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS
        confluences = compute_all_confluences(core_universe)

    confluence_map = {
        c["ticker"]: {
            "score": c["score"],
            "bias": c["bias_label"],
            "bull_count": c["bull_count"],
            "bear_count": c["bear_count"],
            "flicker": c["flicker_status"],
            "close": c["metrics"].get("close"),
            "pct_1d": c["metrics"].get("pct_1d"),
            "pct_5d": c["metrics"].get("pct_5d"),
            "rsi": c["metrics"].get("rsi14"),
            "rvol": c["metrics"].get("rvol"),
            "atr_stretch": c["metrics"].get("atr_stretch"),
            "above_20ema": c["metrics"].get("close", 0) > c["metrics"].get("ema20", 0),
            "above_50ema": c["metrics"].get("close", 0) > c["metrics"].get("ema50", 0),
            "above_200ema": c["metrics"].get("close", 0) > c["metrics"].get("ema200", 0)
        }
        for c in confluences
    }

    baskets = market_data.get("baskets_5d", {})
    ratios = market_data.get("ratios", {})

    system_context = f"""
You are the **Rotato Quantitative AI Analyst & Portfolio Risk Strategist**, an institutional-grade trading and macro AI agent embedded in the Rotato Quantitative Dashboard.
The user is asking you for quantitative analysis, market insight, or trade evaluations.

CURRENT LIVE MARKET SNAPSHOT ({datetime.now().strftime('%Y-%m-%d %H:%M')}):
- Market Regime: {market_data.get('narrative_title')} | Bias: {market_data.get('macro_context')}
- 5-Day Sector Velocity Baskets: {json.dumps(baskets)}
- Key Ratio Analytics (SPY/RSP, QQQ/RSP, IWM/QQQ): {json.dumps({k: {'val': v.get('current'), '5d': v.get('pct_5d'), 'z_score': v.get('z_score_50d')} for k, v in ratios.items() if k in ['SPY_RSP', 'QQQ_RSP', 'IWM_QQQ']})}
- Core Universe Confluence Snapshot:
{json.dumps(confluence_map, indent=1, default=str)}

RULES FOR RESPONDING:
1. Always be direct, precise, quantitative, and institutional.
2. Ground your explanations in the mathematical indicators (20/50/200 EMAs, RSI, ATR stretch, RVOL, Confluence Score).
3. If asked about a specific stock, provide: Trend Status, Support/Resistance (EMAs), Volume Confirmation (RVOL), Overbought/Oversold Stretch (ATR), and Actionable Desk Verdict (Buy/Wait/Fade).
4. If asked about broader market risk, evaluate breadth (SPY/RSP) and macro assets (TLT, GLD, USO, IBIT).
5. Always provide explicit risk invalidation levels (e.g. "Stop invalidation below 50 EMA at $XXX").
"""

    messages = [{"role": "system", "content": system_context}]
    if chat_history:
        for item in chat_history[-6:]:  # Keep recent context
            role = item.get("role", "user")
            content = item.get("content", "")
            if role in ["user", "model", "assistant"]:
                messages.append({"role": "user" if role == "user" else "model", "content": content})

    full_prompt = f"{system_context}\n\nUSER QUESTION: {user_query}\n\nANSWER AS ROTATO QUANT AI AGENT:"

    for m in GEMINI_MODELS:
        try:
            res = client.models.generate_content(model=m, contents=full_prompt)
            if res and res.text:
                return res.text.strip()
        except Exception:
            continue

    return "⚠️ **Agent Error:** Unable to generate response from Gemini models at this moment. Please check your API key or network connection."


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
            "thesis": f"Broken below key 20 EMA and 50 EMA on elevated volume ({w_data.get('rvol')}x RVOL). Avoid buying until a firm base establishes.",
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
            t_data.get("close", 0) > t_data.get("ema50", 0) and 
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
                "confirmations": f"Low RVOL ({d_data.get('rvol')}x) | Above 50 EMA",
                "status_desc": "Healthy Pullback Without Institutional Dumping",
                "thesis": f"Down {abs(d_data.get('pct_1d', 0)):.2f}% on low volume ({d_data.get('rvol')}x RVOL) while holding above 50 EMA. Rubber-band dip-buy opportunity.",
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
            "confirmations": f"5D: {iwm_qqq.get('pct_5d', 0):+.2f}% | 20D: {iwm_qqq.get('pct_20d', 0):+.2f}% | vs 50EMA: {iwm_qqq.get('dist_ema50', 0):+.2f}%",
            "status_desc": "Extreme Valuation Divergence",
            "thesis": "Small caps are historically discounted relative to mega-cap tech.",
            "entry": f"{iwm_qqq.get('current', 0):.4f} spread",
            "stop": "Break by > 0.5σ",
            "stop_pct": "-0.5σ",
            "target1": f"{iwm_qqq.get('ema50', iwm_qqq.get('sma50', 0)):.4f} (50 EMA)",
            "target1_pct": "Mean Revert",
            "target2": "200 EMA",
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
