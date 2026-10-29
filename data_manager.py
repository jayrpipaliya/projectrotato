import os
import glob
import pandas as pd
import numpy as np
import yfinance as yf
try:
    cache_dir = "/tmp" if (os.path.exists("/tmp") or os.name != "nt") else os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass
    if hasattr(yf, "cache") and hasattr(yf.cache, "set_cache_location"):
        yf.cache.set_cache_location(cache_dir)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(cache_dir)
except Exception:
    pass
from datetime import datetime

RAW_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_csv")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
try:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
except Exception:
    PROCESSED_DIR = os.path.join("/tmp", "processed_data")
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
    except Exception:
        pass

MAG_7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
INDEXES = ["SPY", "QQQ", "DIA", "IWM", "RSP"]
SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "XBI", "SMH"]
MACRO_ASSETS = ["GLD", "USO", "TLT", "IBIT"]

# Fast in-memory cache for serverless environments (e.g. Vercel read-only containers)
_IN_MEMORY_TICKER_DATA: dict[str, pd.DataFrame] = {}

# Map ticker to filename pattern in raw_csv: <ticker_lower>_us_d.csv
def ticker_to_filename(ticker: str) -> str:
    return os.path.join(RAW_CSV_DIR, f"{ticker.lower()}_us_d.csv")

def filename_to_ticker(filepath: str) -> str:
    base = os.path.basename(filepath)
    return base.replace("_us_d.csv", "").upper()

def get_available_tickers() -> list[str]:
    files = glob.glob(os.path.join(RAW_CSV_DIR, "*_us_d.csv"))
    disk_tickers = [filename_to_ticker(f) for f in files]
    all_tickers = set(disk_tickers) | set(_IN_MEMORY_TICKER_DATA.keys())
    return sorted(list(all_tickers))

def load_ticker_data(ticker: str) -> pd.DataFrame:
    """Load daily OHLCV dataframe for a given ticker from in-memory cache, /tmp, or raw_csv."""
    ticker_up = ticker.upper()
    ticker_clean = ticker_up.replace('/', '_')
    if ticker_up in _IN_MEMORY_TICKER_DATA:
        df = _IN_MEMORY_TICKER_DATA[ticker_up].copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    if ticker_clean in _IN_MEMORY_TICKER_DATA:
        df = _IN_MEMORY_TICKER_DATA[ticker_clean].copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df

    # Check /tmp first if it exists
    tmp_path = os.path.join("/tmp", "raw_csv", f"{ticker.lower()}_us_d.csv")
    if os.path.exists(tmp_path):
        try:
            df = pd.read_csv(tmp_path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
        except Exception:
            pass

    filepath = ticker_to_filename(ticker)
    if not os.path.exists(filepath):
        # Check if it's a synthetic pair
        synthetic_path = os.path.join(PROCESSED_DIR, f"{ticker.lower().replace('/', '_')}.csv")
        if not os.path.exists(synthetic_path):
            bundle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data", f"{ticker.lower().replace('/', '_')}.csv")
            if os.path.exists(bundle_path):
                synthetic_path = bundle_path
        if os.path.exists(synthetic_path):
            try:
                df = pd.read_csv(synthetic_path)
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    return df.sort_values("Date").reset_index(drop=True)
            except Exception:
                pass
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            # Ensure numeric types
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def sync_tickers_batch(tickers: list[str] = None, force: bool = True) -> list[dict]:
    """
    Downloads incremental updates for all requested tickers in ONE single fast batch request (< 2 seconds).
    Updates in-memory DataFrames and writes to disk (or /tmp on serverless read-only platforms).
    """
    if tickers is None:
        tickers = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS

    results = []
    try:
        batch_df = yf.download(tickers, period="5d", progress=False, timeout=15)
        if batch_df.empty:
            return [{"ticker": t, "status": "empty"} for t in tickers]

        writable_dir = RAW_CSV_DIR
        try:
            test_path = os.path.join(RAW_CSV_DIR, ".write_test")
            with open(test_path, "w") as f:
                f.write("1")
            os.remove(test_path)
        except Exception:
            writable_dir = os.path.join("/tmp", "raw_csv")
            try:
                os.makedirs(writable_dir, exist_ok=True)
            except Exception:
                pass

        for t in tickers:
            try:
                sub = pd.DataFrame()
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if (col, t) in batch_df.columns:
                        sub[col] = batch_df[(col, t)]
                    elif (t, col) in batch_df.columns:
                        sub[col] = batch_df[(t, col)]
                    elif col in batch_df.columns and not isinstance(batch_df.columns, pd.MultiIndex):
                        sub[col] = batch_df[col]

                if sub.empty or "Close" not in sub.columns:
                    results.append({"ticker": t, "status": "up_to_date", "added_rows": 0})
                    continue

                sub = sub.dropna(subset=["Close"]).reset_index()
                if sub.empty:
                    results.append({"ticker": t, "status": "up_to_date", "added_rows": 0})
                    continue

                if "Date" not in sub.columns:
                    if "Datetime" in sub.columns:
                        sub["Date"] = sub["Datetime"]
                    elif "index" in sub.columns:
                        sub["Date"] = sub["index"]
                    elif len(sub.columns) > 0:
                        sub["Date"] = sub.index

                sub["Date"] = pd.to_datetime(sub["Date"])
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in sub.columns:
                        sub[col] = pd.to_numeric(sub[col], errors="coerce")

                existing = load_ticker_data(t)
                if existing.empty:
                    combined = sub
                else:
                    existing = existing.copy()
                    if "Date" in existing.columns:
                        existing["Date"] = pd.to_datetime(existing["Date"])
                        combined = pd.concat([existing[~existing["Date"].isin(sub["Date"])], sub], ignore_index=True)
                    else:
                        combined = sub

                combined = combined.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
                combined["Date"] = pd.to_datetime(combined["Date"])
                _IN_MEMORY_TICKER_DATA[t.upper()] = combined

                try:
                    target_file = os.path.join(writable_dir, f"{t.lower()}_us_d.csv")
                    csv_df = combined.copy()
                    csv_df["Date"] = csv_df["Date"].dt.strftime("%Y-%m-%d")
                    csv_df.to_csv(target_file, index=False)
                except Exception:
                    pass

                last_dt = combined["Date"].iloc[-1]
                last_dt_str = last_dt.strftime("%Y-%m-%d") if hasattr(last_dt, "strftime") else str(last_dt)[:10]

                results.append({
                    "ticker": t,
                    "status": "updated",
                    "added_rows": len(sub),
                    "last_date": last_dt_str
                })
            except Exception as ticker_err:
                results.append({"ticker": t, "status": "error", "error": str(ticker_err)})

    except Exception as e:
        results.append({"error": str(e)})

    # Recalculate synthetic ratios in-memory and on disk
    try:
        generate_synthetic_ratios()
    except Exception as e:
        results.append({"error": f"Synthetic ratio recalculation warning: {str(e)}"})

    return results

def update_single_ticker(ticker: str, force: bool = False) -> dict:
    """Fetch incremental daily updates from yfinance and append to CSV."""
    filepath = ticker_to_filename(ticker)
    if not os.path.exists(filepath):
        # Brand new download
        try:
            df_new = yf.download(ticker, period="max", progress=False)
            if df_new.empty:
                return {"ticker": ticker, "status": "empty", "added_rows": 0}
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = [c[0] for c in df_new.columns]
            df_new = df_new.reset_index()
            df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.strftime("%Y-%m-%d")
            cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df_new = df_new[[c for c in cols if c in df_new.columns]]
            df_new.to_csv(filepath, index=False)
            return {"ticker": ticker, "status": "created", "added_rows": len(df_new)}
        except Exception as e:
            return {"ticker": ticker, "status": "error", "error": str(e), "added_rows": 0}

    # Load existing to find last date
    df_existing = pd.read_csv(filepath)
    if df_existing.empty or "Date" not in df_existing.columns:
        last_date_str = "2000-01-01"
    else:
        last_date_str = str(df_existing["Date"].iloc[-1]).split()[0]

    last_date = pd.to_datetime(last_date_str).date()
    today = datetime.now().date()

    if not force and last_date >= today:
        return {"ticker": ticker, "status": "up_to_date", "added_rows": 0, "last_date": last_date_str}

    # Download from last_date forward (with lookback to handle splits & refresh recent bars)
    try:
        lookback_days = 5 if force else 2
        start_date = (pd.to_datetime(last_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        df_new = yf.download(ticker, start=start_date, progress=False)
        if df_new.empty:
            return {"ticker": ticker, "status": "up_to_date", "added_rows": 0, "last_date": last_date_str}

        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = [c[0] for c in df_new.columns]
        df_new = df_new.reset_index()
        df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.strftime("%Y-%m-%d")

        cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df_new = df_new[[c for c in cols if c in df_new.columns]]

        # Merge existing with new by replacing overlapping dates with fresh prices
        df_existing["Date"] = pd.to_datetime(df_existing["Date"]).dt.strftime("%Y-%m-%d")
        new_dates = set(df_new["Date"])
        df_kept = df_existing[~df_existing["Date"].isin(new_dates)]
        df_combined = pd.concat([df_kept, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df_combined.to_csv(filepath, index=False)

        new_last_date = str(df_combined["Date"].iloc[-1])
        added_count = len(df_combined) - len(df_existing)
        return {"ticker": ticker, "status": "updated", "added_rows": max(0, added_count), "last_date": new_last_date}
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e), "added_rows": 0}

def is_core_data_stale() -> bool:
    """Checks if SPY or core assets haven't been updated to the latest trading date."""
    spy_path = ticker_to_filename("SPY")
    if not os.path.exists(spy_path):
        return True
    try:
        df = pd.read_csv(spy_path)
        if df.empty or "Date" not in df.columns:
            return True
        last_date_str = str(df["Date"].iloc[-1]).split()[0]
        last_date = pd.to_datetime(last_date_str).date()
        today = datetime.now().date()
        # If weekday and last date is older than today, consider stale
        if last_date < today and today.weekday() < 5:
            return True
        return False
    except Exception:
        return True

def auto_sync_core_data(force: bool = False, progress_callback=None) -> list[dict]:
    """Automatically syncs core assets (Mag 7, Indexes, Sectors, Macro) if stale or forced."""
    if not force and not is_core_data_stale():
        return []
    core_tickers = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS
    return update_all_tickers(tickers=core_tickers, progress_callback=progress_callback)

def update_all_tickers(tickers: list[str] = None, progress_callback=None, force: bool = False) -> list[dict]:
    """Updates all specified tickers (or all available CSVs) incrementally."""
    if tickers is None:
        # Prioritize Mag 7, Indexes, Sectors, Macro Assets first, then any other files in raw_csv
        all_available = get_available_tickers()
        priority = MAG_7 + INDEXES + SECTORS + MACRO_ASSETS
        seen = set(priority)
        tickers = priority + [t for t in all_available if t not in seen]

    results = []
    total = len(tickers)
    for idx, ticker in enumerate(tickers):
        res = update_single_ticker(ticker, force=force)
        results.append(res)
        if progress_callback:
            progress_callback(idx + 1, total, ticker, res)

    # After updating tickers, compute synthetic ratios
    generate_synthetic_ratios()
    return results

def generate_synthetic_ratios():
    """Generates synthetic ratio series: SPY/RSP, QQQ/RSP, IWM/QQQ."""
    pairs = [
        ("SPY", "RSP", "SPY_RSP"),
        ("QQQ", "RSP", "QQQ_RSP"),
        ("IWM", "QQQ", "IWM_QQQ")
    ]
    for num, den, ratio_name in pairs:
        try:
            df_num = load_ticker_data(num)
            df_den = load_ticker_data(den)
            if df_num.empty or df_den.empty:
                continue

            df_num = df_num.copy()
            df_den = df_den.copy()
            df_num["Date"] = pd.to_datetime(df_num["Date"])
            df_den["Date"] = pd.to_datetime(df_den["Date"])

            merged = pd.merge(
                df_num[["Date", "Close", "Volume"]].rename(columns={"Close": f"Close_{num}", "Volume": f"Volume_{num}"}),
                df_den[["Date", "Close", "Volume"]].rename(columns={"Close": f"Close_{den}", "Volume": f"Volume_{den}"}),
                on="Date",
                how="inner"
            ).sort_values("Date")

            if merged.empty:
                continue

            merged["Close"] = merged[f"Close_{num}"] / merged[f"Close_{den}"]
            merged["Open"] = merged["Close"]
            merged["High"] = merged["Close"]
            merged["Low"] = merged["Close"]
            merged["Volume"] = (merged[f"Volume_{num}"] + merged[f"Volume_{den}"]) / 2

            ratio_df = merged[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            ratio_df["Date"] = pd.to_datetime(ratio_df["Date"])
            _IN_MEMORY_TICKER_DATA[ratio_name] = ratio_df
            _IN_MEMORY_TICKER_DATA[ratio_name.lower()] = ratio_df
            _IN_MEMORY_TICKER_DATA[ratio_name.replace("_", "/")] = ratio_df

            out_path = os.path.join(PROCESSED_DIR, f"{ratio_name.lower()}.csv")
            try:
                save_df = ratio_df.copy()
                save_df["Date"] = save_df["Date"].dt.strftime("%Y-%m-%d")
                save_df.to_csv(out_path, index=False)
            except Exception:
                try:
                    tmp_out = os.path.join("/tmp", "processed_data", f"{ratio_name.lower()}.csv")
                    os.makedirs(os.path.dirname(tmp_out), exist_ok=True)
                    save_df = ratio_df.copy()
                    save_df["Date"] = save_df["Date"].dt.strftime("%Y-%m-%d")
                    save_df.to_csv(tmp_out, index=False)
                except Exception:
                    pass
        except Exception:
            continue
