import sys
from data_manager import update_all_tickers, generate_synthetic_ratios, MAG_7, INDEXES, SECTORS

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("Starting Daily yfinance Market Update...")
    print("=" * 60)

    def progress(curr, total, ticker, res):
        print(f"[{curr}/{total}] {ticker:<6} -> {res.get('status')} (Added: {res.get('added_rows', 0)} rows, Last Date: {res.get('last_date', 'N/A')})")

    # Update Mag 7, Indexes, Sectors, and all other available CSVs
    results = update_all_tickers(progress_callback=progress)

    print("\n" + "=" * 60)
    updated = sum(1 for r in results if r.get("status") in ["updated", "created"])
    print(f"Daily Update Complete! {updated}/{len(results)} tickers updated.")
    print("=" * 60)

    print("\nRecording Gamma Exposure (GEX) Snapshots (7-Day Rolling Retention)...")
    try:
        from options_engine import compute_gamma_exposure_profile
        for b_sym in ["SPY", "QQQ", "IWM"]:
            try:
                prof = compute_gamma_exposure_profile(b_sym, force_refresh=True)
                if prof and "error" not in prof:
                    print(f"  [GEX Snapshot] {b_sym:<5} -> Spot ${prof.get('spot_price', 0):.2f} | Call Wall ${prof.get('call_wall', 0):.2f} | Put Wall ${prof.get('put_wall', 0):.2f} | Net GEX {prof.get('total_net_gex_billions', 0):+.2f}B ({prof.get('regime_short')})")
            except Exception as e:
                print(f"  [GEX Snapshot] {b_sym:<5} -> Error: {e}")
    except Exception as e:
        print(f"  [GEX Snapshot] Error loading options engine: {e}")
    print("=" * 60)
