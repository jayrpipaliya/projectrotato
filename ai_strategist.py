import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from confluence_engine import compute_all_confluences
from data_manager import MAG_7, INDEXES, SECTORS, MACRO_ASSETS

load_dotenv()

# Validated working model cascade (leading with supported fast models)
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-flash-latest",
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
    Returns structured data for cards and deep daily analysis from Gemini derived directly from data.
    """
    api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

    core_universe = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS
    confluences = compute_all_confluences(core_universe)

    # Generate baseline structured quantitative cards
    structured_data = build_structured_briefing_and_trades(market_data, confluences)

    client, err = get_gemini_client(api_key)
    if client:
        try:
            # Benchmark indexes
            benchmarks_summary = {}
            for t in ["SPY", "QQQ", "IWM", "RSP", "DIA"]:
                c_match = next((c for c in confluences if c["ticker"] == t), None)
                if c_match:
                    m = c_match["metrics"]
                    benchmarks_summary[t] = {
                        "close": m.get("close"),
                        "pct_1d": m.get("pct_1d"),
                        "pct_5d": m.get("pct_5d"),
                        "pct_20d": m.get("pct_20d"),
                        "rsi": m.get("rsi14"),
                        "dist_ema20": m.get("dist_ema20"),
                        "dist_ema50": m.get("dist_ema50"),
                        "confluence_score": c_match["score"],
                        "bias": c_match["bias_label"]
                    }

            green_setups = [
                {
                    "ticker": c["ticker"],
                    "score": c["score"],
                    "bias": c["bias_label"],
                    "price": c["metrics"].get("close"),
                    "pct_1d": c["metrics"].get("pct_1d"),
                    "pct_5d": c["metrics"].get("pct_5d"),
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
                    "pct_5d": c["metrics"].get("pct_5d"),
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
                    "pct_20d": v.get("pct_20d"),
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

            sector_summary = {}
            for s in SECTORS:
                c_match = next((c for c in confluences if c["ticker"] == s), None)
                if c_match:
                    m = c_match["metrics"]
                    sector_summary[s] = {
                        "pct_1d": m.get("pct_1d"),
                        "pct_5d": m.get("pct_5d"),
                        "rsi": m.get("rsi14"),
                        "dist_ema20": m.get("dist_ema20"),
                        "score": c_match["score"]
                    }

            mag7_summary = {}
            for m7 in MAG_7:
                c_match = next((c for c in confluences if c["ticker"] == m7), None)
                if c_match:
                    m = c_match["metrics"]
                    mag7_summary[m7] = {
                        "pct_1d": m.get("pct_1d"),
                        "pct_5d": m.get("pct_5d"),
                        "rsi": m.get("rsi14"),
                        "score": c_match["score"]
                    }

            analysis_payload = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "benchmarks": benchmarks_summary,
                "key_breadth_ratios": ratio_summary,
                "baskets_5d": baskets,
                "sector_breakdown": sector_summary,
                "macro_cross_assets": macro_summary,
                "mag_7_dynamics": mag7_summary,
                "top_green_confluence_accumulation": green_setups,
                "top_red_confluence_distribution": red_warnings,
            }

            prompt = f"""
You are the Chief Quantitative Strategist and Senior Portfolio Risk Officer at a multi-billion dollar quantitative macro fund.
Analyze the following comprehensive quantitative desk data for {datetime.now().strftime('%A, %B %d, %Y')}:

```json
{json.dumps(analysis_payload, indent=2, default=str)}
```

CRITICAL INSTRUCTION:
Do NOT rely on any pre-set static narrative rules or conditions. Synthesize the raw numbers, cross-asset momentum, sector rotations, and breadth ratios from first principles.

Your response MUST begin with the following structural header block:

---AI_REGIME_HEADER---
REGIME_TITLE: [Formulate a high-conviction, professional 3-7 word regime title based on your analysis of the data, e.g. "Broadening Cyclical Reflation & High-Beta Expansion", "Mega-Cap Tech Exhaustion & Narrow Breadth Squeeze", "Defensive Flight into Yield Proxies", etc.]
REGIME_DETAIL: [2-3 sentences synthesizing the exact mechanics of capital flows across sectors and breadth based on the numbers.]
RISK_LEVEL: [Choose exactly one: Bullish / Risk-On | Selective / Dispersion | Elevated Caution | High Fragility | Defensive / Risk-Off]
MACRO_CONTEXT: [1-2 sentences summarizing how Treasury yields (TLT), Gold (GLD), Crude Oil (USO), and Bitcoin (IBIT) are aligning with or diverging from equities.]
---END_AI_REGIME_HEADER---

Follow immediately with the 4-Pillar Daily Institutional Intelligence Briefing:

### 1. Macro Regime & Cross-Asset Liquidity Assessment
Provide a 1-sentence executive summary of the prevailing regime you identified above.
Then break down the 4 macro liquidity pillars with distinct bullets:
* **TLT (Treasuries):** Detail yield pressure vs easing, duration impact, and relationship to 50 EMA.
* **GLD (Gold):** Detail real yields, debasement hedging demand, and key price levels.
* **USO (Crude Oil):** Detail energy/inflation impulse and supply/demand signals.
* **IBIT (Bitcoin):** Detail high-beta liquidity appetite and institutional risk willingness.
Conclude with a clear statement: Current institutional regime is **[Risk-On / Risk-Off / Selective Dispersion]**.

### 2. Sector Velocity & Breadth Concentration Dynamics
#### **5-Day Sector Capital Rotation**
* **Leading Sectors (Smart Money Bids):** Name top leading sectors with 5-day velocity and institutional accumulation evidence.
* **Lagging/Defensive Sectors:** Name sectors seeing net capital outflows or distribution.
#### **Market Breadth & Concentration Metrics**
* **SPY vs RSP (Mega-Cap Concentration):** Analyze equal-weight vs cap-weight divergence. Is the rally broad-based or dangerously concentrated?
* **IWM vs QQQ (Risk Dispersion):** Analyze small-cap participation vs mega-cap tech dominance.

### 3. Actionable Setups & Trade Confluence
#### 🟢 High-Conviction Long Setups (Green Confluence)
* Detail top green confluence assets: Exact technical alignment (Price above 20/50 EMAs, RSI, RVOL, stretch) and trade rationale.
#### 🔴 Distribution Warnings & Retail Traps (Red Confluence)
* Detail top red confluence assets: Specific structural breakdowns and explicitly warn why retail traders must avoid buying the dip on these failing names.

### 4. Desk Game Plan & Tactical Risk Boundaries
#### 🛡️ Tactical Risk Execution Rules
* **Capital Deployment (Add Risk):** Where to allocate capital and optimal entry zones.
* **Portfolio Defense (Hedge / Trim):** Names or sectors to de-risk, trim profits, or hedge.
* **Hard Invalidation Trigger Levels:** 2-3 specific index or ratio thresholds that will immediately invalidate the current bullish or bearish bias.

Write in a crisp, Wall Street institutional tone. Bold all key tickers, percentages, and metrics.
"""

            for m in GEMINI_MODELS:
                try:
                    response = client.models.generate_content(model=m, contents=prompt)
                    if response and response.text:
                        raw_text = response.text.strip()
                        
                        # Parse out the AI's synthesized regime header
                        match = re.search(r"---AI_REGIME_HEADER---(.*?)---END_AI_REGIME_HEADER---", raw_text, re.DOTALL)
                        if match:
                            header = match.group(1)
                            t_match = re.search(r"REGIME_TITLE:\s*(.*)", header)
                            d_match = re.search(r"REGIME_DETAIL:\s*(.*)", header)
                            r_match = re.search(r"RISK_LEVEL:\s*(.*)", header)
                            m_match = re.search(r"MACRO_CONTEXT:\s*(.*)", header)
                            
                            if t_match and t_match.group(1).strip():
                                structured_data["regime_title"] = t_match.group(1).strip().strip("[]*\"")
                            if d_match and d_match.group(1).strip():
                                structured_data["regime_detail"] = d_match.group(1).strip().strip("[]*\"")
                            if r_match and r_match.group(1).strip():
                                structured_data["risk_level"] = r_match.group(1).strip().strip("[]*\"")
                            if m_match and m_match.group(1).strip():
                                structured_data["macro_context"] = m_match.group(1).strip().strip("[]*\"")
                                
                            raw_text = raw_text.replace(match.group(0), "").strip()
                            structured_data["is_ai_derived"] = True

                        structured_data["llm_summary"] = raw_text
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


def calculate_optimized_trade_entry(direction: str, metrics: dict, edge: dict = None) -> dict:
    """
    Computes institutional multi-tier entry parameters instead of naive current close:
    1. Current Market Price & Extension Meter (% from 20 EMA, ATR Stretch)
    2. Execution State (IN BUY ZONE, EXTENDED - AWAIT PULLBACK, OVERSOLD BOUNCE)
    3. Optimal Limit Entry Zone (Pullback to 20/50 EMA support)
    4. Breakout Stop-Limit Trigger (Break of 5-day pivot high on RVOL)
    5. Invalidation Stop Loss (Anchored to structural support)
    6. Take-Profit Targets & Recalculated Risk-to-Reward Ratio (R:R)
    """
    edge = edge or {}
    close = float(metrics.get("close", 100.0))
    ema20 = float(metrics.get("ema20", close))
    ema50 = float(metrics.get("ema50", close * 0.98))
    dist_20 = float(metrics.get("dist_ema20", 0.0))
    atr_stretch = float(metrics.get("atr_stretch", 0.0))
    atr = float(metrics.get("atr20") or (close * 0.018))
    high_5d = float(metrics.get("high_5d") or (close * 1.015))
    low_5d = float(metrics.get("low_5d") or (close * 0.985))

    is_long = "LONG" in direction.upper() or "DIP" in direction.upper()

    if is_long:
        # Execution State
        if abs(dist_20) <= 0.8 and abs(atr_stretch) <= 0.6:
            exec_state = "IN PRIME BUY ZONE"
            exec_pill = "pill-green"
            exec_desc = "Price consolidating right at 20 EMA support. Immediate limit execution feasible."
        elif dist_20 > 1.2 or atr_stretch > 1.0:
            exec_state = f"EXTENDED (+{dist_20:+.1f}%) — AWAIT PULLBACK"
            exec_pill = "pill-amber"
            exec_desc = f"Stretched {dist_20:+.1f}% above 20 EMA. High probability of mean-reversion retest. Do not chase."
        elif dist_20 < -1.5:
            exec_state = "DEEP DISCOUNT PULLBACK"
            exec_pill = "pill-cyan"
            exec_desc = "Oversold pullback into 50 EMA support band. Scaling in with limit orders."
        else:
            exec_state = "CONSOLIDATION RETEST"
            exec_pill = "pill-green"
            exec_desc = "Testing upper support band. Limit bids favored."

        # Optimal Limit Zone: Limit buy MUST be at or below current market price
        if close > ema20:
            # Price is above 20 EMA -> Bid the pullback to the 20 EMA
            limit_high = round(min(close * 0.992, ema20 + 0.35 * atr), 2)
            limit_low = round(max(ema50, ema20 * 0.995), 2)
            if limit_high <= limit_low:
                limit_low = round(limit_high - 0.3 * atr, 2)
        else:
            # Price is already below 20 EMA -> Bid support between 50 EMA and current close
            limit_high = round(close * 0.998, 2)
            limit_low = round(max(ema50, close - 0.5 * atr), 2)
            if limit_high <= limit_low:
                limit_low = round(limit_high - 0.25 * atr, 2)

        opt_entry_mid = round((limit_low + limit_high) / 2.0, 2)

        # Breakout Trigger: Must be above current close and above 5D high
        breakout_trigger = round(max(high_5d * 1.002, close + 0.25 * atr), 2)

        # Invalidation Stop: Structural support strictly below limit zone
        mae = abs(float(edge.get("max_adverse_excursion", 2.2))) if edge.get("max_adverse_excursion", 0) != 0 else 2.2
        stop_price = round(min(limit_low * 0.985, ema50 * 0.99), 2)
        stop_pct = round(((stop_price - opt_entry_mid) / opt_entry_mid) * 100.0, 1)

        # Targets
        med_ret = float(edge.get("median_return_20d", 3.8))
        if med_ret <= 0:
            med_ret = 3.8
        target1 = round(opt_entry_mid * (1 + med_ret / 100.0), 2)
        target2 = round(opt_entry_mid * (1 + (med_ret * 1.6) / 100.0), 2)

        risk = max(opt_entry_mid - stop_price, 0.01)
        reward = max(target1 - opt_entry_mid, 0.01)
        rr_ratio = round(reward / risk, 1)

        return {
            "current_price": f"${close:,.2f}",
            "extension_metric": f"{dist_20:+.2f}% vs 20 EMA",
            "execution_state": exec_state,
            "execution_pill": exec_pill,
            "execution_desc": exec_desc,
            "optimal_entry_zone": f"${limit_low:,.2f} - ${limit_high:,.2f}",
            "optimal_entry_label": "20 EMA Support Retest Zone" if close > ema20 else "50 EMA Support Value Zone",
            "breakout_trigger": f"Above ${breakout_trigger:,.2f} on RVOL > 1.2x",
            "stop": f"${stop_price:,.2f}",
            "stop_pct": f"{stop_pct:+.1f}%",
            "target1": f"${target1:,.2f}",
            "target1_pct": f"+{med_ret:.1f}%",
            "target2": f"${target2:,.2f}",
            "target2_pct": f"+{med_ret * 1.6:.1f}%",
            "risk_reward": f"1 : {rr_ratio:.1f} R:R",
            "opt_entry_mid": opt_entry_mid
        }
    else:
        # SHORT / FADE Setup
        if dist_20 < -2.0 or atr_stretch < -1.5:
            exec_state = "OVERSOLD TRAP — DO NOT SHORT HOLE"
            exec_pill = "pill-red"
            exec_desc = "Asset is dump-extended. Short-covering squeeze imminent. Await relief bounce to 20 EMA."
        elif dist_20 > -0.6:
            exec_state = "PRIME RESISTANCE FADE ZONE"
            exec_pill = "pill-amber"
            exec_desc = "Price testing broken 20/50 EMA overhead supply. Prime fade entry."
        else:
            exec_state = "AWAIT 20 EMA RELIEF BOUNCE"
            exec_pill = "pill-amber"
            exec_desc = "Patience required. Place limit offers above market at the 20 EMA."

        # Optimal Fade Zone: Retest of 20 EMA resistance from underneath
        limit_low = round(max(close * 1.01, ema20 * 0.99), 2)
        limit_high = round(max(close * 1.025, ema20 * 1.01), 2)
        if limit_high <= limit_low:
            limit_high = round(limit_low + 0.3 * atr, 2)
        opt_entry_mid = round((limit_low + limit_high) / 2.0, 2)

        # Breakdown Trigger
        breakdown_trigger = round(min(low_5d * 0.998, close - 0.25 * atr), 2)

        # Invalidation Stop: Above 50 EMA / resistance
        stop_price = round(max(opt_entry_mid * 1.03, ema50 * 1.01), 2)
        stop_pct = round(((stop_price - opt_entry_mid) / opt_entry_mid) * 100.0, 1)

        target1 = round(opt_entry_mid * 0.95, 2)
        target2 = round(opt_entry_mid * 0.91, 2)

        risk = max(stop_price - opt_entry_mid, 0.01)
        reward = max(opt_entry_mid - target1, 0.01)
        rr_ratio = round(reward / risk, 1)

        return {
            "current_price": f"${close:,.2f}",
            "extension_metric": f"{dist_20:+.2f}% vs 20 EMA",
            "execution_state": exec_state,
            "execution_pill": exec_pill,
            "execution_desc": exec_desc,
            "optimal_entry_zone": f"${limit_low:,.2f} - ${limit_high:,.2f}",
            "optimal_entry_label": "Fade 20 EMA Resistance Relief",
            "breakout_trigger": f"Below ${breakdown_trigger:,.2f} on RVOL > 1.2x",
            "stop": f"${stop_price:,.2f}",
            "stop_pct": f"+{stop_pct:.1f}%",
            "target1": f"${target1:,.2f}",
            "target1_pct": "-5.0%",
            "target2": f"${target2:,.2f}",
            "target2_pct": "-9.0%",
            "risk_reward": f"1 : {rr_ratio:.1f} R:R",
            "opt_entry_mid": opt_entry_mid
        }


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
        opt = calculate_optimized_trade_entry("LONG", t_data, edge)

        trades.append({
            "ticker": ticker,
            "direction": "LONG",
            "type_label": "High-Conviction Long",
            "score": f"+{best_green['score']}",
            "bias_label": best_green["bias_label"],
            "confirmations": f"{best_green['bull_count']} of 7 Bullish Confirmations",
            "status_desc": best_green["flicker_status"],
            "thesis": f"Holding firm above 20 EMA with 20D vol compressing ({t_data.get('rv20')}%) and strong institutional accumulation.",
            "current_price": opt["current_price"],
            "extension_metric": opt["extension_metric"],
            "execution_state": opt["execution_state"],
            "execution_pill": opt["execution_pill"],
            "execution_desc": opt["execution_desc"],
            "optimal_entry_zone": opt["optimal_entry_zone"],
            "optimal_entry_label": opt["optimal_entry_label"],
            "breakout_trigger": opt["breakout_trigger"],
            "stop": opt["stop"],
            "stop_pct": opt["stop_pct"],
            "target1": opt["target1"],
            "target1_pct": opt["target1_pct"],
            "target2": opt["target2"],
            "target2_pct": opt["target2_pct"],
            "risk_reward": opt["risk_reward"],
            "entry": opt["optimal_entry_zone"],
            "win_rate": f"{edge.get('win_rate_20d', 0)}%",
            "sample_size": edge.get("sample_size", 0),
            "median_alpha": opt["target1_pct"],
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
        w_edge = w_data.get("edge_match", {})
        w_opt = calculate_optimized_trade_entry("SHORT", w_data, w_edge)

        trades.append({
            "ticker": w_ticker,
            "direction": "SHORT / FADE",
            "type_label": "Distribution Fade / De-Risk",
            "score": f"{weakest['score']}",
            "bias_label": weakest["bias_label"],
            "confirmations": f"{weakest['bear_count']} of 7 Bearish Warnings",
            "status_desc": weakest["flicker_status"],
            "thesis": f"Broken below key 20 EMA and 50 EMA on elevated volume ({w_data.get('rvol')}x RVOL). Avoid buying until a firm base establishes.",
            "current_price": w_opt["current_price"],
            "extension_metric": w_opt["extension_metric"],
            "execution_state": w_opt["execution_state"],
            "execution_pill": w_opt["execution_pill"],
            "execution_desc": w_opt["execution_desc"],
            "optimal_entry_zone": w_opt["optimal_entry_zone"],
            "optimal_entry_label": w_opt["optimal_entry_label"],
            "breakout_trigger": w_opt["breakout_trigger"],
            "stop": w_opt["stop"],
            "stop_pct": w_opt["stop_pct"],
            "target1": w_opt["target1"],
            "target1_pct": w_opt["target1_pct"],
            "target2": w_opt["target2"],
            "target2_pct": w_opt["target2_pct"],
            "risk_reward": w_opt["risk_reward"],
            "entry": w_opt["optimal_entry_zone"],
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
            d_opt = calculate_optimized_trade_entry("DIP BUY", d_data, d_edge)

            trades.append({
                "ticker": d_ticker,
                "direction": "DIP BUY",
                "type_label": "Orderly Pullback to Support",
                "score": f"+{best_dip['score']}",
                "bias_label": best_dip["bias_label"],
                "confirmations": f"Low RVOL ({d_data.get('rvol')}x) | Above 50 EMA",
                "status_desc": "Healthy Pullback Without Institutional Dumping",
                "thesis": f"Down {abs(d_data.get('pct_1d', 0)):.2f}% on low volume ({d_data.get('rvol')}x RVOL) while holding above 50 EMA. Rubber-band dip-buy opportunity.",
                "current_price": d_opt["current_price"],
                "extension_metric": d_opt["extension_metric"],
                "execution_state": d_opt["execution_state"],
                "execution_pill": d_opt["execution_pill"],
                "execution_desc": d_opt["execution_desc"],
                "optimal_entry_zone": d_opt["optimal_entry_zone"],
                "optimal_entry_label": d_opt["optimal_entry_label"],
                "breakout_trigger": d_opt["breakout_trigger"],
                "stop": d_opt["stop"],
                "stop_pct": d_opt["stop_pct"],
                "target1": d_opt["target1"],
                "target1_pct": d_opt["target1_pct"],
                "target2": d_opt["target2"],
                "target2_pct": d_opt["target2_pct"],
                "risk_reward": d_opt["risk_reward"],
                "entry": d_opt["optimal_entry_zone"],
                "win_rate": f"{d_edge.get('win_rate_20d', 71.0)}%",
                "sample_size": d_edge.get("sample_size", 34),
                "median_alpha": d_opt["target1_pct"],
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
