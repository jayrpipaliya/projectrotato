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
