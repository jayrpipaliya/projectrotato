import json
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

from options_engine import fetch_term_structure, compute_realized_volatilities, compute_iv_rank_and_percentile
from data_manager import load_ticker_data, PROCESSED_DIR

DEFAULT_UNIVERSE = [
    # Benchmark Indexes & Macro ETFs (Apex Options Volume)
    'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'GDX', 'TLT', 'HYG', 'USO', 'IBIT', 'FXI', 'XBI', 'SMH',
    # Major Sector ETFs
    'XLF', 'XLE', 'XLK', 'XLI', 'XLY', 'XLP', 'XLU', 'XLV', 'XLB', 'XLC',
    # Mega-Cap Tech (Mag 7) & High-Beta Semis
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA',
    'AMD', 'AVGO', 'MU', 'INTC', 'QCOM', 'TSM', 'ARM',
    # High-Beta Crypto & Software Momentum
    'MSTR', 'COIN', 'MARA', 'PLTR', 'HOOD', 'NFLX', 'CRM', 'ORCL', 'SNOW',
    # High-Volume Blue Chips, Financials, Industrials & Energy
    'JPM', 'BAC', 'GS', 'BA', 'CAT', 'DIS', 'XOM', 'CVX', 'LLY'
]

OPPORTUNITIES_FILE = Path(PROCESSED_DIR) / "opportunities.json"
BASE_PROCESSED_FILE = Path(__file__).resolve().parent / "processed_data" / "opportunities.json"

def scan_volatility_opportunities(universe=None):
    """
    Scans the options universe across 3-Month (~90D), 6-Month (~180D), and LEAP (1Y+) horizons.
    Identifies contracts where Implied Volatility is cheap relative to:
    1. Realized Historical Volatility (IV < Clean HV30 or IV/HV < 1.0)
    2. 1-Year Normal Realized Volatility (IV < HV 1Y)
    3. 1-Year IV Percentile (Low Volatility Regime)
    """
    tickers = universe or DEFAULT_UNIVERSE
    opportunities = []
    scan_timestamp = int(time.time())
    scan_date_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    for ticker in tickers:
        try:
            df_hist = load_ticker_data(ticker)
            if df_hist.empty:
                continue
                
            hvs = compute_realized_volatilities(df_hist, ticker_symbol=ticker)
            clean_hv = float(hvs.get('HV_30_clean', hvs.get('HV_30', 20.0)))
            hv_1y = float(hvs.get('HV_252', hvs.get('HV_1Y', 20.0)))
            
            term_df, spot_price, next_earnings = fetch_term_structure(ticker, hvs_dict=hvs, hv_benchmark=clean_hv)
            if term_df is None or term_df.empty or spot_price is None:
                continue

            front_iv = float(term_df['ATM_IV'].iloc[0])
            front_dte = int(term_df['DTE'].iloc[0])
            iv_stats = compute_iv_rank_and_percentile(ticker, current_iv_pct=front_iv)
            front_iv_pct = float(iv_stats.get('iv_percentile', 50.0))
            front_iv_rank = float(iv_stats.get('iv_rank', 50.0))
            vol_regime = iv_stats.get('vol_regime', 'Neutral Volatility')

            for _, row in term_df.iterrows():
                dte = int(row['DTE'])
                exp = str(row['Expiration'])
                atm_iv = float(row['ATM_IV'])
                oi = int(row.get('Total_OI', 0) or 0)
                vol = int(row.get('Total_Volume', 0) or 0)
                spans_earnings = bool(row.get('Spans_Earnings', False))

                tenor = None
                horizon_code = None
                if 60 <= dte <= 125:
                    tenor = "3-Month (~90D)"
                    horizon_code = "3M"
                elif 140 <= dte <= 235:
                    tenor = "6-Month (~180D)"
                    horizon_code = "6M"
                elif dte >= 240:
                    tenor = "LEAP (1Y+)"
                    horizon_code = "LEAP"

                if not tenor:
                    continue

                # Liquidity Guardrail: eliminate illiquid strikes with zero or negligible open interest
                if oi < 50:
                    continue

                spread_clean = round(atm_iv - clean_hv, 2)
                spread_1y = round(atm_iv - hv_1y, 2)
                ratio_clean = round(atm_iv / max(clean_hv, 0.1), 2)
                ratio_1y = round(atm_iv / max(hv_1y, 0.1), 2)
                term_spread = round(atm_iv - front_iv, 1)

                contract_iv_stats = compute_iv_rank_and_percentile(ticker, current_iv_pct=atm_iv)
                contract_iv_pct = float(contract_iv_stats.get('iv_percentile', 50.0))
                contract_iv_rank = float(contract_iv_stats.get('iv_rank', 50.0))

                # Edge score: rewards low IV/HV ratio, negative term spread (backwardation), low contract IV percentile, and high liquidity
                discount_pts = max(0.0, clean_hv - atm_iv)
                inversion_pts = max(0.0, front_iv - atm_iv)
                edge_score = (
                    (1.15 - ratio_clean) * 45.0
                    + (discount_pts * 1.5)
                    + (inversion_pts * 1.0)
                    + ((100.0 - contract_iv_pct) * 0.20)
                    + min(np.log10(max(oi, 10.0)) * 2.0, 10.0)
                )

                # Strategic categorization
                is_discount = (ratio_clean < 1.0 or spread_clean < 0)
                is_low_vol = (contract_iv_pct <= 50.0 or front_iv_pct <= 50.0)
                is_inverted = (term_spread <= -1.5)

                if is_discount and is_low_vol:
                    opp_type = "double_edge"
                    type_badge = "Double Edge Prime"
                    edge_tier = "Prime Double Edge (IV < HV & Low 1Y Vol)"
                    tier_tone = "mint"
                    strategy = "Prime Vol Arbitrage: Outright LEAP or Diagonal (Cheap IV + Low 1Y Regime)"
                elif is_discount and is_inverted:
                    opp_type = "relative_discount"
                    type_badge = "Term & Realized Discount"
                    edge_tier = "Prime Inversion (IV < HV & Inverted)"
                    tier_tone = "mint"
                    strategy = "Calendar / Bullish Diagonal (Buy Cheap Back, Sell Spiked Front Week)"
                elif is_discount:
                    opp_type = "relative_discount"
                    type_badge = "Relative Discount (IV < HV)"
                    edge_tier = "Underpriced Vol (IV < HV)"
                    tier_tone = "mint"
                    strategy = "Long Variance / Calendar Long Leg (IV < Clean Realized HV)"
                elif is_low_vol:
                    opp_type = "low_vol"
                    type_badge = "Absolute Low Vol"
                    edge_tier = "Low Vol Percentile Regime"
                    tier_tone = "sky"
                    strategy = "Outright Long Option / LEAP Swing Accumulation (Low 1Y Regime)"
                elif is_inverted:
                    opp_type = "term_inversion"
                    type_badge = "Term Inversion"
                    edge_tier = "Inverted Curve Dislocation"
                    tier_tone = "amber"
                    strategy = "Volatility Calendar Spread (Harvest Spiked Front IV Decay)"
                else:
                    opp_type = "fair_value"
                    type_badge = "Fair Value Vol"
                    edge_tier = "Fair Value Vol"
                    tier_tone = "slate"
                    strategy = "Directional Spread / Covered Accumulation"

                opportunities.append({
                    "ticker": ticker,
                    "spot_price": round(spot_price, 2),
                    "horizon": tenor,
                    "horizon_code": horizon_code,
                    "expiration": exp,
                    "dte": dte,
                    "atm_iv": round(atm_iv, 1),
                    "front_iv": round(front_iv, 1),
                    "front_dte": front_dte,
                    "term_spread": term_spread,
                    "clean_hv": round(clean_hv, 1),
                    "hv_1y": round(hv_1y, 1),
                    "vrp_clean": spread_clean,
                    "vrp_1y": spread_1y,
                    "ratio_clean": ratio_clean,
                    "ratio_1y": ratio_1y,
                    "front_iv_percentile": round(front_iv_pct, 1),
                    "front_iv_rank": round(front_iv_rank, 1),
                    "contract_iv_percentile": round(contract_iv_pct, 1),
                    "contract_iv_rank": round(contract_iv_rank, 1),
                    "iv_percentile": round(front_iv_pct, 1),
                    "iv_rank": round(front_iv_rank, 1),
                    "vol_regime": vol_regime,
                    "total_oi": oi,
                    "total_vol": vol,
                    "spans_earnings": spans_earnings,
                    "edge_score": round(edge_score, 1),
                    "edge_tier": edge_tier,
                    "tier_tone": tier_tone,
                    "opportunity_type": opp_type,
                    "type_badge": type_badge,
                    "recommended_strategy": strategy
                })

        except Exception as e:
            print(f"[Opportunities Engine] Error scanning {ticker}: {e}")

    # Sort primarily by lowest ratio_clean (deepest volatility discount)
    opportunities.sort(key=lambda x: (x['ratio_clean'], -x['total_oi']))

    # Top Highlights across horizons and term structures
    top_leaps = [o for o in opportunities if o['horizon_code'] == 'LEAP'][:3]
    top_6m = [o for o in opportunities if o['horizon_code'] == '6M'][:3]
    top_3m = [o for o in opportunities if o['horizon_code'] == '3M'][:3]
    top_inversion = sorted([o for o in opportunities if o['term_spread'] < 0], key=lambda x: (x['term_spread'], x['ratio_clean']))[:3]

    relative_discount_count = sum(1 for o in opportunities if o['ratio_clean'] < 1.0)
    inverted_count = sum(1 for o in opportunities if o['term_spread'] < -1.0)
    low_vol_count = sum(1 for o in opportunities if o['contract_iv_percentile'] <= 50.0 or o['front_iv_percentile'] <= 50.0)
    double_edge_count = sum(1 for o in opportunities if o['ratio_clean'] < 1.0 and (o['contract_iv_percentile'] <= 50.0 or o['front_iv_percentile'] <= 50.0))

    # Actionable Trade Strip
    # 1. Prime Calendar Setup (3M or 6M tenor, deep term inversion, IV < HV)
    calendars = [
        o for o in opportunities 
        if o['horizon_code'] in ('3M', '6M') and o.get('term_spread', 0) <= -4.0 and o.get('ratio_clean', 1.0) < 1.0
    ]
    if not calendars:
        calendars = [o for o in opportunities if o.get('term_spread', 0) < 0]
    calendars.sort(key=lambda x: (x['ratio_clean'], x['term_spread']))
    prime_cal = calendars[0] if calendars else None

    # 2. Prime Long LEAP Value (1Y+ tenor, deepest discount to clean HV, liquid)
    leaps = [o for o in opportunities if o['horizon_code'] == 'LEAP' and o.get('ratio_clean', 1.0) < 1.0]
    if not leaps:
        leaps = [o for o in opportunities if o['horizon_code'] == 'LEAP']
    leaps.sort(key=lambda x: (x['ratio_clean'], -x['total_oi']))
    prime_leap = leaps[0] if leaps else None

    # 3. Double Edge Confluence (Directional Trend Confluence + Volatility Discount)
    ticker_biases = {}
    try:
        from confluence_engine import compute_confluence_score
        for t in tickers:
            try:
                conf = compute_confluence_score(t)
                ticker_biases[t] = conf
            except Exception:
                pass
    except Exception:
        pass

    confluence_candidates = []
    for o in opportunities:
        conf = ticker_biases.get(o['ticker'])
        if conf and conf.get('score', 0) >= 20 and o.get('ratio_clean', 1.0) <= 0.98:
            confluence_candidates.append({
                **o,
                'bias_score': conf.get('score', 0),
                'bias_label': conf.get('bias_label', 'Bullish')
            })
    if not confluence_candidates:
        for o in opportunities:
            conf = ticker_biases.get(o['ticker'])
            if conf and conf.get('score', 0) > 0 and o.get('ratio_clean', 1.0) < 1.0:
                confluence_candidates.append({
                    **o,
                    'bias_score': conf.get('score', 0),
                    'bias_label': conf.get('bias_label', 'Bullish')
                })
    confluence_candidates.sort(key=lambda x: (-x['bias_score'], x['ratio_clean']))
    prime_conf = confluence_candidates[0] if confluence_candidates else None

    # 4. Macro Term Structure State (Cross-asset curve backwardation vs contango)
    tickers_term = {}
    for o in opportunities:
        t = o['ticker']
        if t not in tickers_term:
            tickers_term[t] = {'front_iv': o['front_iv'], 'back_ivs': []}
        tickers_term[t]['back_ivs'].append(o['atm_iv'])

    inv_t_count = 0
    front_skews = []
    for t, t_info in tickers_term.items():
        if t_info['back_ivs']:
            avg_back = sum(t_info['back_ivs']) / len(t_info['back_ivs'])
            diff = t_info['front_iv'] - avg_back
            front_skews.append(diff)
            if diff > 1.0:
                inv_t_count += 1

    total_scanned_tickers = len(tickers_term)
    inversion_pct = round((inv_t_count / max(total_scanned_tickers, 1)) * 100)
    avg_front_skew = round(sum(front_skews) / max(len(front_skews), 1), 1)

    if inversion_pct >= 70:
        macro_label = "Deep Backwardation"
        macro_tone = "cherry"
    elif inversion_pct >= 50:
        macro_label = "Mild Backwardation"
        macro_tone = "amber"
    elif inversion_pct <= 30:
        macro_label = "Contango (Normal)"
        macro_tone = "mint"
    else:
        macro_label = "Flat / Mixed Curve"
        macro_tone = "slate"

    macro_term_state = {
        "state_label": macro_label,
        "tone": macro_tone,
        "inversion_pct": inversion_pct,
        "inverted_count": inv_t_count,
        "total_tickers": total_scanned_tickers,
        "avg_front_skew": avg_front_skew
    }

    # 5. Peak IV Crush Target (Most bloated front-month weekly cycle)
    crush_candidates = sorted(opportunities, key=lambda x: (x.get('front_iv', 0), -x.get('term_spread', 0)), reverse=True)
    top_crush = crush_candidates[0] if crush_candidates else None
    peak_crush = {
        "ticker": top_crush['ticker'] if top_crush else "—",
        "front_iv": top_crush['front_iv'] if top_crush else 0.0,
        "front_dte": top_crush.get('front_dte', 1) if top_crush else 1,
        "front_iv_percentile": top_crush.get('front_iv_percentile', 100.0) if top_crush else 100.0,
        "term_spread": top_crush['term_spread'] if top_crush else 0.0,
        "back_exp": top_crush['expiration'] if top_crush else "",
        "back_iv": top_crush['atm_iv'] if top_crush else 0.0
    } if top_crush else {}

    actionable_cards = {
        "prime_calendar": prime_cal,
        "prime_leap": prime_leap,
        "prime_confluence": prime_conf,
        "macro_term_state": macro_term_state,
        "peak_iv_crush": peak_crush
    }

    payload = {
        "scan_timestamp": scan_timestamp,
        "scan_date": scan_date_str,
        "universe_count": len(tickers),
        "total_opportunities": len(opportunities),
        "relative_discount_count": relative_discount_count,
        "inverted_count": inverted_count,
        "low_vol_count": low_vol_count,
        "double_edge_count": double_edge_count,
        "top_leaps": top_leaps,
        "top_6m": top_6m,
        "top_3m": top_3m,
        "top_inversion": top_inversion,
        "actionable_cards": actionable_cards,
        "opportunities": opportunities
    }

    # Save to disk across candidate cache paths
    candidates = [
        Path(PROCESSED_DIR) / "opportunities.json",
        Path(__file__).resolve().parent / "processed_data" / "opportunities.json",
        Path(__file__).resolve().parent / ".runtime" / "opportunities.json"
    ]
    for fpath in candidates:
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[Opportunities Engine] Saved {len(opportunities)} opportunities to {fpath}")
        except Exception as e:
            print(f"[Opportunities Engine] Failed saving to {fpath}: {e}")

    return payload

def get_opportunities_data(force_refresh=False):
    """
    Returns cached opportunities if recent (under 18 hours old) or performs scan.
    """
    if not force_refresh:
        candidates = [
            Path(PROCESSED_DIR) / "opportunities.json",
            Path(__file__).resolve().parent / "processed_data" / "opportunities.json",
            Path(__file__).resolve().parent / ".runtime" / "opportunities.json"
        ]
        best_data = None
        best_ts = 0
        for fpath in candidates:
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    ts = d.get('scan_timestamp', 0)
                    if ts > best_ts:
                        best_ts = ts
                        best_data = d
                except Exception:
                    pass
        if best_data and (time.time() - best_ts < 64800):
            return best_data
            
    return scan_volatility_opportunities()


if __name__ == "__main__":
    res = scan_volatility_opportunities()
    print("Done. Total opportunities:", res["total_opportunities"])
