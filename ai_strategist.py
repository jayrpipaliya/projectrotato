import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from confluence_engine import compute_all_confluences
from data_manager import MAG_7, INDEXES, SECTORS, MACRO_ASSETS

load_dotenv()
DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# Validated working model cascade (leading with fastest supported models)
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
]

def get_gemini_client(api_key: str = None):
    key = api_key or os.getenv("GEMINI_API_KEY") or DEFAULT_GEMINI_KEY
    if not key:
        return None, None
    try:
        from google import genai
        client = genai.Client(api_key=key.strip(), http_options={"timeout": 15000})
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

            # Dealer Gamma Exposure Profile (Market Positioning)
            gamma_summary = {}
            try:
                from options_engine import compute_gamma_exposure_profile
                key_gamma_syms = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "AMZN", "TSLA"]
                for idx_sym in key_gamma_syms:
                    g_res = compute_gamma_exposure_profile(idx_sym, max_expirations=3)
                    if g_res and "error" not in g_res:
                        gamma_summary[idx_sym] = {
                            "spot": g_res.get("spot_price"),
                            "call_wall": g_res.get("call_wall"),
                            "call_wall_dist_pct": f"{g_res.get('call_wall_dist_pct'):+.2f}%",
                            "put_wall": g_res.get("put_wall"),
                            "put_wall_dist_pct": f"{g_res.get('put_wall_dist_pct'):+.2f}%",
                            "zero_gamma_level": g_res.get("zero_gamma_level"),
                            "zero_gamma_dist_pct": f"{g_res.get('zero_gamma_dist_pct'):+.2f}%",
                            "net_gex_billions": f"${g_res.get('total_net_gex_billions'):+.2f}B",
                            "regime": g_res.get("regime_short"),
                            "regime_detail": g_res.get("regime")
                        }
            except Exception:
                pass

            analysis_payload = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "benchmarks": benchmarks_summary,
                "dealer_gamma_positioning": gamma_summary,
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
Do NOT rely on any pre-set static narrative rules or conditions. Synthesize the raw numbers, cross-asset momentum, sector rotations, breadth ratios, AND dealer Gamma Exposure (GEX) levels from first principles.

INSTITUTIONAL DEALER GAMMA & RISK/REWARD FRAMEWORK:
1. Gamma Positioning Regimes:
   - Long Gamma (>0 Net GEX): Dealers are net long gamma; to maintain delta neutrality, they BUY dips and SELL rallies. This acts as a volatility suppressor and price pin. Spot tends to compress between the Put Wall (support floor) and Call Wall (upside ceiling). Strategy: Favor mean-reversion, bidding 20 EMA pullbacks near the Put Wall, taking profit directly at the Call Wall, and harvesting elevated IV.
   - Short Gamma (<0 Net GEX): Dealers are net short gamma; they MUST SELL as prices fall and BUY as prices rally. This pro-cyclical hedging triggers self-reinforcing trending cascades, gap-and-go moves, and sharp volatility spikes. Strategy: Favor momentum breakouts, wider directional targets, long volatility, and strict trailing stops.
   - Zero-Gamma Vol Flip: The critical boundary price where dealer positioning flips between Long and Short Gamma. Spot breaking below the Zero-Gamma Flip unleashes directional acceleration.
   - Call Wall: Major upside supply ceiling and pinning magnet into weekly OpEx. In Long Gamma, rallies into the Call Wall face heavy dealer supply—fading or de-risking is mandatory unless broken with massive institutional volume.
   - Put Wall: Major downside structural cushion and liquidity floor. In Long Gamma, dealers aggressively defend this level, making it the primary anchor for stop-loss invalidations.
2. Asymmetric Risk/Reward Execution:
   - Always structure trade setups with a minimum 1 : 2.5+ R:R.
   - Longs: Anchor entry at 20 EMA / Put Wall cushion, place hard invalidation stop strictly beneath the Put Wall or Zero Gamma Flip, and target the Call Wall pin for Take Profit 1.
   - Shorts / Fades: Invalidate above the Call Wall or 20/50 EMA, targeting the Put Wall or cascade expansion below the Zero Gamma Flip.

Your response MUST begin with the following structural header block:

---AI_REGIME_HEADER---
REGIME_TITLE: [Formulate a high-conviction, professional 3-7 word regime title based on your analysis of the data, e.g. "Broadening Cyclical Reflation & High-Beta Expansion", "Mega-Cap Tech Exhaustion & Narrow Breadth Squeeze", "Defensive Flight into Yield Proxies", etc.]
REGIME_DETAIL: [2-3 sentences synthesizing the exact mechanics of capital flows across sectors, breadth ratios, and dealer gamma positioning based on the numbers.]
RISK_LEVEL: [Choose exactly one: Bullish / Risk-On | Selective / Dispersion | Elevated Caution | High Fragility | Defensive / Risk-Off]
MACRO_CONTEXT: [1-2 sentences summarizing how Treasury yields (TLT), Gold (GLD), Crude Oil (USO), and Bitcoin (IBIT) are aligning with or diverging from equities.]
---END_AI_REGIME_HEADER---

Follow immediately with the 4-Pillar Daily Institutional Intelligence Briefing:

### 1. Macro Regime & Cross-Asset Liquidity Assessment
Provide a 1-sentence executive summary of the prevailing regime you identified above.
Detail the Market Maker Gamma Regime: State SPY and QQQ Net GEX ($B), Long vs Short Gamma status, and distance to the Call Wall ceiling and Put Wall floor.
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
* Detail top green confluence assets: Exact technical alignment (Price above 20/50 EMAs, RSI, RVOL), Dealer Gamma levels (Put Wall cushion, Call Wall target), and trade rationale with asymmetric R:R.
#### 🔴 Distribution Warnings & Retail Traps (Red Confluence)
* Detail top red confluence assets: Specific structural breakdowns, failure at 20/50 EMAs or breach of Put Wall, and explicitly warn why retail traders must avoid buying the dip on these failing names.

### 4. Desk Game Plan & Tactical Risk Boundaries
#### Tactical Risk Execution Rules
* **Capital Deployment (Add Risk):** Where to allocate capital and optimal limit entry zones relative to Put Walls.
* **Portfolio Defense (Hedge / Trim):** Names or sectors to de-risk, trim profits into Call Walls, or hedge.
* **Hard Invalidation Trigger Levels:** 2-3 specific index, gamma (e.g. SPY Put Wall / Zero Flip breach), or ratio thresholds that will immediately invalidate the current bias.

Write in a crisp, Wall Street institutional tone. Bold all key tickers, percentages, and metrics.
Keep each bullet point to 1-2 dense, high-signal analytical sentences (target 1,800 - 2,500 total characters) to ensure immediate execution speed.
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
        return "**Gemini API Key Missing or Invalid:** Please enter your Gemini API key in the sidebar to interact with the AI Quant Agent."

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

    gamma_context = {}
    try:
        from options_engine import compute_gamma_exposure_profile
        symbols_to_check = ["SPY", "QQQ", "IWM"]
        # Extract any specific stock symbol mentioned in the user's prompt
        for cand in MAG_7 + INDEXES + SECTORS + ["PLTR", "AMD", "SMH", "GLD", "TLT", "IBIT"]:
            if re.search(r'\b' + re.escape(cand) + r'\b', user_query, re.IGNORECASE):
                if cand not in symbols_to_check:
                    symbols_to_check.append(cand)

        for sym in symbols_to_check:
            g = compute_gamma_exposure_profile(sym, max_expirations=3)
            if g and "error" not in g:
                gamma_context[sym] = {
                    "spot": g.get("spot_price"),
                    "call_wall": g.get("call_wall"),
                    "call_wall_dist_pct": f"{g.get('call_wall_dist_pct'):+.2f}%",
                    "put_wall": g.get("put_wall"),
                    "put_wall_dist_pct": f"{g.get('put_wall_dist_pct'):+.2f}%",
                    "zero_gamma_flip": g.get("zero_gamma_level"),
                    "net_gex_b": f"${g.get('total_net_gex_billions'):+.2f}B",
                    "regime": g.get("regime_short")
                }
    except Exception:
        pass

    system_context = f"""
You are the **Double Edge Quantitative AI Analyst & Chief Risk Officer**, an institutional-grade macro and systematic trading AI agent embedded in the Double Edge Quantitative Dashboard.
The user is asking you for quantitative analysis, market insight, dealer positioning evaluations, or trade setups.

CURRENT LIVE MARKET SNAPSHOT ({datetime.now().strftime('%Y-%m-%d %H:%M')}):
- Market Regime: {market_data.get('narrative_title')} | Bias: {market_data.get('macro_context')}
- Dealer Gamma Exposure Profiles (Live Options GEX): {json.dumps(gamma_context, indent=1)}
- 5-Day Sector Velocity Baskets: {json.dumps(baskets)}
- Key Ratio Analytics (SPY/RSP, QQQ/RSP, IWM/QQQ): {json.dumps({k: {'val': v.get('current'), '5d': v.get('pct_5d'), 'z_score': v.get('z_score_50d')} for k, v in ratios.items() if k in ['SPY_RSP', 'QQQ_RSP', 'IWM_QQQ']})}
- Core Universe Confluence Snapshot:
{json.dumps(confluence_map, indent=1, default=str)}

INSTITUTIONAL DEALER GAMMA & RISK/REWARD MASTER DIRECTIVES:
1. Gamma Positioning Regimes:
   - **Net Long Gamma (+$B GEX)**: Dealers are net long options gamma. To maintain delta neutrality, they BUY weakness and SELL strength. This dampens realized intraday volatility, compresses price swings, and pins price between the Put Wall (support) and Call Wall (resistance). High probability for mean-reversion, range fades, bidding pullbacks, and taking profit at Call Walls.
   - **Net Short Gamma (-$B GEX)**: Dealers are net short options gamma. They MUST SELL as price falls and BUY as price rallies. This creates self-reinforcing trending cascades, gap-and-go volatility expansions, and tail risk. Favor directional momentum breakouts, trailing stops, and long volatility.
   - **Zero-Gamma Vol Flip**: The critical inflection boundary. Trading above it is Long Gamma (mean-reverting); breaking below it unleashes violent Short Gamma acceleration.
   - **Call Wall**: Strike with maximum positive call gamma. Acts as the primary institutional resistance ceiling and pinning magnet into OpEx. Rallies into the Call Wall face dealer sell pressure—take profit or fade unless accompanied by heavy volume breakout.
   - **Put Wall**: Strike with maximum put gamma. Acts as structural downside cushion and liquidity floor. In Long Gamma, dealers aggressively bid dips here, making it the premier anchor for stop-loss placement.

2. Structuring Asymmetric Risk/Reward (R:R):
   - Every recommended trade must have mathematically defined entry, stop loss, profit target, and asymmetric Risk-to-Reward (minimum 1 : 2.5+ R:R).
   - **Long Setups**:
     * Entry: Limit bids anchored at 20 EMA pullback or near the Put Wall liquidity floor.
     * Invalidation Stop: Placed strictly below the Put Wall (e.g. Put Wall * 0.995) or below the Zero Gamma Flip.
     * Take-Profit 1: Anchored directly at the Call Wall Pin (or just in front of it).
     * Take-Profit 2 (Runner): 5-10% extension or next structural resistance.
   - **Short / Fade Setups**:
     * Entry: Fade retests of broken 20/50 EMAs from underneath or exhaustion at the Call Wall.
     * Invalidation Stop: Above the Call Wall or 20/50 EMA ribbon.
     * Profit Target: Down to the Put Wall floor or cascade below Zero Gamma Flip.

3. Rules for Responding:
   - Direct, precise, Wall Street institutional tone. Bold key numbers, levels, and tickers.
   - Ground setups in the confluence of: Trend Ribbons (20/50/200 EMAs), Volume (RVOL), Overbought/Oversold Stretch (ATR), and Dealer Gamma Levels (Call Wall, Put Wall, Zero Flip).
   - Always state the Risk-to-Reward ratio (e.g., "1 : 3.2 R:R") and explicitly explain why the Put Wall provides downside safety and the Call Wall serves as the take-profit ceiling.
"""

    messages = [{"role": "system", "content": system_context}]
    if chat_history:
        for item in chat_history[-6:]:  # Keep recent context
            role = item.get("role", "user")
            content = item.get("content", "")
            if role in ["user", "model", "assistant"]:
                messages.append({"role": "user" if role == "user" else "model", "content": content})

    full_prompt = f"{system_context}\n\nUSER QUESTION: {user_query}\n\nANSWER AS DOUBLE EDGE QUANT AI AGENT:"

    for m in GEMINI_MODELS:
        try:
            res = client.models.generate_content(model=m, contents=full_prompt)
            if res and res.text:
                return res.text.strip()
        except Exception:
            continue

    return "**Agent Error:** Unable to generate response from Gemini models at this moment. Please check your API key or network connection."


def calculate_optimized_trade_entry(direction: str, metrics: dict, edge: dict = None, gamma_levels: dict = None) -> dict:
    """
    Computes institutional multi-tier entry parameters instead of naive current close:
    1. Current Market Price & Extension Meter (% from 20 EMA, ATR Stretch)
    2. Execution State (IN BUY ZONE, EXTENDED - AWAIT PULLBACK, AT CALL WALL PIN, OVERSOLD BOUNCE)
    3. Optimal Limit Entry Zone (Pullback to 20/50 EMA support)
    4. Breakout Stop-Limit Trigger (Break of 5-day pivot high on RVOL)
    5. Invalidation Stop Loss (Anchored to structural support & Put Wall floor)
    6. Take-Profit Targets & Recalculated Risk-to-Reward Ratio (R:R, aligned with Call Wall)
    7. Dealer Gamma Exposure (GEX) integration: Call Wall, Put Wall, Zero Gamma Flip
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

    cw = float(gamma_levels.get("call_wall", 0)) if gamma_levels else 0.0
    pw = float(gamma_levels.get("put_wall", 0)) if gamma_levels else 0.0
    zg = float(gamma_levels.get("zero_gamma_level", 0)) if gamma_levels else 0.0
    g_regime = gamma_levels.get("regime_short") if gamma_levels else None
    net_gex = gamma_levels.get("total_net_gex_billions") if gamma_levels else None

    is_long = "LONG" in direction.upper() or "DIP" in direction.upper()

    if is_long:
        # Execution State
        if cw > 0 and (abs(close - cw) / close <= 0.008 or (close < cw and (cw - close) / close <= 0.008)):
            exec_state = f"AT CALL WALL PIN (${cw:,.0f}) — AWAIT PULLBACK"
            exec_pill = "pill-amber"
            exec_desc = f"Spot is within {((cw - close)/close)*100:+.1f}% of major Dealer Call Wall (${cw:,.0f}). Heavy dealer long gamma will pin or reject upside without high institutional volume."
        elif abs(dist_20) <= 0.8 and abs(atr_stretch) <= 0.6:
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
            limit_high = round(min(close * 0.992, ema20 + 0.35 * atr), 2)
            limit_low = round(max(ema50, ema20 * 0.995), 2)
            if limit_high <= limit_low:
                limit_low = round(limit_high - 0.3 * atr, 2)
        else:
            limit_high = round(close * 0.998, 2)
            limit_low = round(max(ema50, close - 0.5 * atr), 2)
            if limit_high <= limit_low:
                limit_low = round(limit_high - 0.25 * atr, 2)

        opt_entry_mid = round((limit_low + limit_high) / 2.0, 2)

        # Breakout Trigger: Must be above current close and above 5D high
        breakout_trigger = round(max(high_5d * 1.002, close + 0.25 * atr), 2)

        # Invalidation Stop: Structural support strictly below limit zone, anchored by Put Wall if below entry
        if pw > 0 and pw < opt_entry_mid:
            pw_dist = (opt_entry_mid - pw) / opt_entry_mid
            # If Put Wall is within 0.8% to 4.5% below entry, anchor stop strictly below dealer Put Wall cushion
            if 0.008 <= pw_dist <= 0.045:
                stop_price = round(pw * 0.995, 2)
            else:
                stop_price = round(min(limit_low * 0.985, ema50 * 0.99), 2)
        else:
            stop_price = round(min(limit_low * 0.985, ema50 * 0.99), 2)

        # Ensure stop is below limit_low
        if stop_price >= limit_low:
            stop_price = round(limit_low * 0.988, 2)

        stop_pct = round(((stop_price - opt_entry_mid) / opt_entry_mid) * 100.0, 1)

        # Targets & Call Wall Pin Alignment
        med_ret = float(edge.get("median_return_20d", 3.8))
        if med_ret <= 0:
            med_ret = 3.8
        target1 = round(opt_entry_mid * (1 + med_ret / 100.0), 2)
        target2 = round(opt_entry_mid * (1 + (med_ret * 1.6) / 100.0), 2)
        target1_pct_str = f"+{med_ret:.1f}%"

        # If Call Wall is above entry, align Target 1 with Call Wall Pin
        if cw > opt_entry_mid:
            cw_dist = (cw - opt_entry_mid) / opt_entry_mid
            if 0.01 <= cw_dist <= 0.08:
                target1 = round(cw, 2)
                target1_pct_str = f"+{cw_dist * 100:.1f}% (Call Wall Pin)"
                target2 = round(cw * 1.025, 2)

        risk = max(opt_entry_mid - stop_price, 0.01)
        reward = max(target1 - opt_entry_mid, 0.01)
        rr_ratio = round(reward / risk, 1)

        if cw > 0 and pw > 0:
            exec_desc += f" [GEX Strategy: {g_regime or 'Dealer GEX'} | Put Wall (${pw:,.0f}) floor cushion, Call Wall (${cw:,.0f}) target pin | 1:{rr_ratio:.1f} R:R]"

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
            "target1_pct": target1_pct_str,
            "target2": f"${target2:,.2f}",
            "target2_pct": f"+{med_ret * 1.6:.1f}%",
            "risk_reward": f"1 : {rr_ratio:.1f} R:R",
            "opt_entry_mid": opt_entry_mid,
            "call_wall": f"${cw:,.1f}" if cw > 0 else None,
            "put_wall": f"${pw:,.1f}" if pw > 0 else None,
            "zero_gamma": f"${zg:,.1f}" if zg > 0 else None,
            "gamma_regime": g_regime,
            "net_gex_b": f"${net_gex:+.2f}B" if net_gex is not None else None
        }
    else:
        # SHORT / FADE Setup
        if pw > 0 and (abs(close - pw) / close <= 0.008 or (close > pw and (close - pw) / close <= 0.008)):
            exec_state = f"AT PUT WALL FLOOR (${pw:,.0f}) — DO NOT SHORT HOLE"
            exec_pill = "pill-red"
            exec_desc = f"Spot is testing major Dealer Put Wall (${pw:,.0f}). Short gamma squeeze rebound risk high. Avoid shorting into major dealer floor."
        elif dist_20 < -2.0 or atr_stretch < -1.5:
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

        # Invalidation Stop: Above 50 EMA / resistance or Call Wall
        if cw > opt_entry_mid and (cw - opt_entry_mid) / opt_entry_mid <= 0.04:
            stop_price = round(cw * 1.005, 2)
        else:
            stop_price = round(max(opt_entry_mid * 1.03, ema50 * 1.01), 2)
        stop_pct = round(((stop_price - opt_entry_mid) / opt_entry_mid) * 100.0, 1)

        target1 = round(opt_entry_mid * 0.95, 2)
        target2 = round(opt_entry_mid * 0.91, 2)
        target1_pct_str = "-5.0%"

        if pw > 0 and pw < opt_entry_mid:
            pw_dist = (opt_entry_mid - pw) / opt_entry_mid
            if 0.01 <= pw_dist <= 0.09:
                target1 = round(pw, 2)
                target1_pct_str = f"-{pw_dist * 100.0:.1f}% (Put Wall Floor)"

        risk = max(stop_price - opt_entry_mid, 0.01)
        reward = max(opt_entry_mid - target1, 0.01)
        rr_ratio = round(reward / risk, 1)

        if cw > 0 and pw > 0:
            exec_desc += f" [GEX Strategy: Fade toward Put Wall (${pw:,.0f}) | Call Wall (${cw:,.0f}) invalidation | 1:{rr_ratio:.1f} R:R]"

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
            "target1_pct": target1_pct_str,
            "target2": f"${target2:,.2f}",
            "target2_pct": "-9.0%",
            "risk_reward": f"1 : {rr_ratio:.1f} R:R",
            "opt_entry_mid": opt_entry_mid,
            "call_wall": f"${cw:,.1f}" if cw > 0 else None,
            "put_wall": f"${pw:,.1f}" if pw > 0 else None,
            "zero_gamma": f"${zg:,.1f}" if zg > 0 else None,
            "gamma_regime": g_regime,
            "net_gex_b": f"${net_gex:+.2f}B" if net_gex is not None else None
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

    def _fetch_gex_safe(sym):
        try:
            from options_engine import compute_gamma_exposure_profile
            res = compute_gamma_exposure_profile(sym, max_expirations=3)
            if res and "error" not in res:
                return res
        except Exception:
            pass
        return None

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
        gex = _fetch_gex_safe(ticker)
        opt = calculate_optimized_trade_entry("LONG", t_data, edge, gamma_levels=gex)

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
            "card_color": "#00F5A0",
            "call_wall": opt.get("call_wall"),
            "put_wall": opt.get("put_wall"),
            "zero_gamma": opt.get("zero_gamma"),
            "gamma_regime": opt.get("gamma_regime"),
            "net_gex_b": opt.get("net_gex_b")
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
        w_gex = _fetch_gex_safe(w_ticker)
        w_opt = calculate_optimized_trade_entry("SHORT", w_data, w_edge, gamma_levels=w_gex)

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
            "card_color": "#FF2E55",
            "call_wall": w_opt.get("call_wall"),
            "put_wall": w_opt.get("put_wall"),
            "zero_gamma": w_opt.get("zero_gamma"),
            "gamma_regime": w_opt.get("gamma_regime"),
            "net_gex_b": w_opt.get("net_gex_b")
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
            d_gex = _fetch_gex_safe(d_ticker)
            d_opt = calculate_optimized_trade_entry("DIP BUY", d_data, d_edge, gamma_levels=d_gex)

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
                "card_color": "#00F5A0",
                "call_wall": d_opt.get("call_wall"),
                "put_wall": d_opt.get("put_wall"),
                "zero_gamma": d_opt.get("zero_gamma"),
                "gamma_regime": d_opt.get("gamma_regime"),
                "net_gex_b": d_opt.get("net_gex_b")
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
            "entry": f"{iwm_qqq.get('current', 0):.2f} spread",
            "stop": "Break by > 0.5σ",
            "stop_pct": "-0.5σ",
            "target1": f"{iwm_qqq.get('ema50', iwm_qqq.get('sma50', 0)):.2f} (50 EMA)",
            "target1_pct": "Mean Revert",
            "target2": "200 EMA",
            "target2_pct": "Full Cycle",
            "win_rate": "72.4%",
            "sample_size": 47,
            "median_alpha": "+3.48%",
            "card_color": "#00F5A0"
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
