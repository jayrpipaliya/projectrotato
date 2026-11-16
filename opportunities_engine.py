import json
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

from options_engine import fetch_term_structure, compute_realized_volatilities, compute_iv_rank_and_percentile
from data_manager import load_ticker_data, PROCESSED_DIR

DEFAULT_UNIVERSE = [
    'SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'SMH', 'GLD', 'TLT', 'XLF', 'XLE'
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
            iv_stats = compute_iv_rank_and_percentile(ticker, current_iv_pct=front_iv)
            iv_pct = float(iv_stats.get('iv_percentile', 50.0))
            iv_rank = float(iv_stats.get('iv_rank', 50.0))
            vol_regime = iv_stats.get('vol_regime', 'Neutral Volatility')

            for _, row in term_df.iterrows():
                dte = int(row['DTE'])
                exp = str(row['Expiration'])
                atm_iv = float(row['ATM_IV'])
                oi = int(row.get('Total_OI', 0) or 0)
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

                spread_clean = round(atm_iv - clean_hv, 2)
                spread_1y = round(atm_iv - hv_1y, 2)
                ratio_clean = round(atm_iv / max(clean_hv, 0.1), 2)
                ratio_1y = round(atm_iv / max(hv_1y, 0.1), 2)

                # Edge score: rewards low IV/HV ratio, low IV percentile, and high liquidity
                # Lower ratio = bigger discount = higher edge
                discount_pts = max(0.0, clean_hv - atm_iv)
                edge_score = (
                    (1.15 - ratio_clean) * 45.0
                    + (discount_pts * 1.5)
                    + ((100.0 - iv_pct) * 0.20)
                    + min(np.log10(max(oi, 10.0)) * 2.0, 10.0)
                )

                # Strategic categorization
                if ratio_clean <= 0.88 or spread_clean <= -3.0:
                    edge_tier = "Prime Discount (IV << HV)"
                    tier_tone = "mint"
                    strategy = "Long LEAP Call / Bullish Diagonal (Buy Deep Cheap Vega)"
                elif ratio_clean <= 0.98 or spread_clean < 0:
                    edge_tier = "Underpriced Vol (IV < HV)"
                    tier_tone = "mint"
                    strategy = "Calendar Spread (Long Leg) / Long Gamma"
                elif ratio_clean <= 1.05 and iv_pct <= 40:
                    edge_tier = "Low Vol Percentile Regime"
                    tier_tone = "sky"
                    strategy = "Low IV Long Option / Swing Accumulation"
                elif ratio_clean <= 1.08:
                    edge_tier = "Fair Value Vol"
                    tier_tone = "slate"
                    strategy = "Directional Spread"
                else:
                    edge_tier = "Elevated Premium"
                    tier_tone = "amber"
                    strategy = "Sell Leg Candidate"

                opportunities.append({
                    "ticker": ticker,
                    "spot_price": round(spot_price, 2),
                    "horizon": tenor,
                    "horizon_code": horizon_code,
                    "expiration": exp,
                    "dte": dte,
                    "atm_iv": round(atm_iv, 1),
                    "clean_hv": round(clean_hv, 1),
                    "hv_1y": round(hv_1y, 1),
                    "vrp_clean": spread_clean,
                    "vrp_1y": spread_1y,
                    "ratio_clean": ratio_clean,
                    "ratio_1y": ratio_1y,
                    "iv_percentile": round(iv_pct, 1),
                    "iv_rank": round(iv_rank, 1),
                    "vol_regime": vol_regime,
                    "total_oi": oi,
                    "spans_earnings": spans_earnings,
                    "edge_score": round(edge_score, 1),
                    "edge_tier": edge_tier,
                    "tier_tone": tier_tone,
                    "recommended_strategy": strategy
                })

        except Exception as e:
            print(f"[Opportunities Engine] Error scanning {ticker}: {e}")

    # Sort primarily by lowest ratio_clean (deepest volatility discount)
    opportunities.sort(key=lambda x: (x['ratio_clean'], -x['total_oi']))

    # Top Highlights across horizons
    top_leaps = [o for o in opportunities if o['horizon_code'] == 'LEAP'][:3]
    top_6m = [o for o in opportunities if o['horizon_code'] == '6M'][:3]
    top_3m = [o for o in opportunities if o['horizon_code'] == '3M'][:3]
    
    payload = {
        "scan_timestamp": scan_timestamp,
        "scan_date": scan_date_str,
        "universe_count": len(tickers),
        "total_opportunities": len(opportunities),
        "top_leaps": top_leaps,
        "top_6m": top_6m,
        "top_3m": top_3m,
        "opportunities": opportunities
    }

    # Save to disk
    for fpath in [OPPORTUNITIES_FILE, BASE_PROCESSED_FILE]:
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
        for fpath in [OPPORTUNITIES_FILE, BASE_PROCESSED_FILE]:
            if fpath.exists():
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if time.time() - data.get('scan_timestamp', 0) < 64800:
                        return data
                except Exception:
                    pass
            
    return scan_volatility_opportunities()

def render_opportunities_page():
    """
    Renders the Opportunities page inside Streamlit.
    """
    import streamlit as st
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
                padding: 1.25rem 1.75rem; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 1.5rem;">
        <h2 style="margin: 0; color: #f8fafc; font-size: 1.6rem; font-weight: 700;">
            Institutional Opportunities Desk
        </h2>
        <p style="margin: 0.35rem 0 0 0; color: #94a3b8; font-size: 0.95rem;">
            Systematic Long-Dated Volatility Mispricing Scanner: Identifies contracts where <b>Implied Volatility (IV) trades at a deep discount to Realized Volatility (IV &lt; Clean HV30)</b> across <b>3-Month (~90D)</b>, <b>6-Month (~180D)</b>, and <b>LEAP (1Y+)</b> horizons.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_btn, col_info = st.columns([1.5, 4.5])
    with col_btn:
        refresh_clicked = st.button("🔄 Refresh Live Options Scan", help="Runs real-time multi-threaded scan across all core underlyings.")
    
    with st.spinner("Loading volatility mispricings..."):
        data = get_opportunities_data(force_refresh=refresh_clicked)

    opps = data.get("opportunities", [])
    if not opps:
        st.warning("No volatility opportunities currently detected. Try refreshing.")
        return

    # Top KPI Metric Strip
    top_leap = data.get("top_leaps", [{}])[0] if data.get("top_leaps") else {}
    top_6m = data.get("top_6m", [{}])[0] if data.get("top_6m") else {}
    top_3m = data.get("top_3m", [{}])[0] if data.get("top_3m") else {}
    prime_count = sum(1 for o in opps if o.get("ratio_clean", 1.0) < 1.0)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Top LEAP Discount (1Y+)",
            f"{top_leap.get('ticker', '—')} {top_leap.get('expiration', '')}",
            f"{top_leap.get('ratio_clean', 1.0):.2f}x HV ({top_leap.get('vrp_clean', 0):+.1f} pp)" if top_leap else "—",
            delta_color="normal"
        )
    with k2:
        st.metric(
            "Top 6-Month Discount",
            f"{top_6m.get('ticker', '—')} {top_6m.get('expiration', '')}",
            f"{top_6m.get('ratio_clean', 1.0):.2f}x HV ({top_6m.get('vrp_clean', 0):+.1f} pp)" if top_6m else "—",
            delta_color="normal"
        )
    with k3:
        st.metric(
            "Top 3-Month Discount",
            f"{top_3m.get('ticker', '—')} {top_3m.get('expiration', '')}",
            f"{top_3m.get('ratio_clean', 1.0):.2f}x HV ({top_3m.get('vrp_clean', 0):+.1f} pp)" if top_3m else "—",
            delta_color="normal"
        )
    with k4:
        st.metric(
            "Discount Contracts (IV < HV)",
            f"{prime_count} / {len(opps)}",
            f"Scanned {data.get('universe_count', 15)} underlyings",
            delta_color="off"
        )

    st.markdown("---")

    # Filter Toolbar
    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 2.0])
    with f_col1:
        horizon_filter = st.selectbox(
            "Horizon Scope:",
            ["All Horizons", "3-Month (~90D)", "6-Month (~180D)", "LEAP (1Y+)"]
        )
    with f_col2:
        available_tickers = ["All Underlyings"] + sorted(list(set(o["ticker"] for o in opps)))
        ticker_filter = st.selectbox("Underlying Ticker:", available_tickers)
    with f_col3:
        sort_mode = st.selectbox(
            "Sort Opportunities by:",
            [
                "Deepest Vol Discount (Lowest IV/HV Ratio)",
                "Lowest 1-Year IV Percentile",
                "Highest Open Interest (Liquidity)"
            ]
        )

    # Filter data
    filtered = opps
    if horizon_filter != "All Horizons":
        code_map = {"3-Month (~90D)": "3M", "6-Month (~180D)": "6M", "LEAP (1Y+)": "LEAP"}
        target_code = code_map.get(horizon_filter)
        filtered = [o for o in filtered if o.get("horizon_code") == target_code]
    if ticker_filter != "All Underlyings":
        filtered = [o for o in filtered if o.get("ticker") == ticker_filter]

    if "Deepest Vol Discount" in sort_mode:
        filtered = sorted(filtered, key=lambda x: (x.get("ratio_clean", 1.0), x.get("vrp_clean", 0)))
    elif "Lowest 1-Year IV Percentile" in sort_mode:
        filtered = sorted(filtered, key=lambda x: (x.get("iv_percentile", 50.0), x.get("ratio_clean", 1.0)))
    elif "Highest Open Interest" in sort_mode:
        filtered = sorted(filtered, key=lambda x: -x.get("total_oi", 0))

    st.markdown(f"### Opportunities Radar ({len(filtered)} Contracts Matching Filters)")
    st.caption("Quantitative Edge: Contracts where Implied Volatility trades below Clean Realized Volatility ($IV/HV < 1.0x$) are mathematically underpriced, offering exceptional positive variance edge for Long LEAPS, Calendar Long Legs, and Bullish Diagonals.")

    # Render formatted table
    df_disp = pd.DataFrame(filtered)
    if not df_disp.empty:
        table_df = pd.DataFrame({
            "Ticker": df_disp["ticker"],
            "Spot": df_disp["spot_price"].apply(lambda s: f"${s:.2f}"),
            "Horizon": df_disp["horizon"],
            "Expiration": df_disp["expiration"] + " (" + df_disp["dte"].astype(str) + "d)",
            "ATM IV": df_disp["atm_iv"].apply(lambda v: f"{v:.1f}%"),
            "30D Clean HV": df_disp["clean_hv"].apply(lambda v: f"{v:.1f}%"),
            "VRP Spread": df_disp["vrp_clean"].apply(lambda v: f"{v:+.1f} pp"),
            "IV/HV Ratio": df_disp["ratio_clean"].apply(lambda r: f"{r:.2f}x"),
            "1Y IV Pct": df_disp["iv_percentile"].apply(lambda p: f"{p:.1f}%"),
            "Recommended Strategy": df_disp["recommended_strategy"],
            "Total OI": df_disp["total_oi"].apply(lambda oi: f"{oi:,.0f}")
        })
        st.dataframe(table_df, use_container_width=True, height=520)
    else:
        st.info("No contracts matched the selected filters.")

if __name__ == "__main__":
    res = scan_volatility_opportunities()
    print("Done. Total opportunities:", res["total_opportunities"])
