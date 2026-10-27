import os
import glob
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

RAW_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_csv")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)

MAG_7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
INDEXES = ["SPY", "QQQ", "DIA", "IWM", "RSP"]
SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "XBI", "SMH"]
MACRO_ASSETS = ["GLD", "USO", "TLT", "IBIT"]

# Map ticker to filename pattern in raw_csv: <ticker_lower>_us_d.csv
def ticker_to_filename(ticker: str) -> str:
    return os.path.join(RAW_CSV_DIR, f"{ticker.lower()}_us_d.csv")

def filename_to_ticker(filepath: str) -> str:
    base = os.path.basename(filepath)
    return base.replace("_us_d.csv", "").upper()

def get_available_tickers() -> list[str]:
    files = glob.glob(os.path.join(RAW_CSV_DIR, "*_us_d.csv"))
    return sorted([filename_to_ticker(f) for f in files])

def load_ticker_data(ticker: str) -> pd.DataFrame:
    """Load daily OHLCV dataframe for a given ticker from raw_csv."""
    filepath = ticker_to_filename(ticker)
    if not os.path.exists(filepath):
        # Check if it's a synthetic pair
        synthetic_path = os.path.join(PROCESSED_DIR, f"{ticker.lower().replace('/', '_')}.csv")
        if os.path.exists(synthetic_path):
            df = pd.read_csv(synthetic_path)
            df["Date"] = pd.to_datetime(df["Date"])
            return df.sort_values("Date").reset_index(drop=True)
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # Ensure numeric types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def update_single_ticker(ticker: str) -> dict:
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

    last_date = pd.to_datetime(last_date_str)
    today = datetime.now()

    if (today - last_date).days < 1:
        return {"ticker": ticker, "status": "up_to_date", "added_rows": 0, "last_date": last_date_str}

    # Download from last_date forward
    try:
        # Download with a small lookback to handle adjustments/splits
        start_date = (last_date - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        df_new = yf.download(ticker, start=start_date, progress=False)
        if df_new.empty:
            return {"ticker": ticker, "status": "up_to_date", "added_rows": 0, "last_date": last_date_str}

        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = [c[0] for c in df_new.columns]
        df_new = df_new.reset_index()
        df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.strftime("%Y-%m-%d")
        
        # Keep only dates strictly greater than last_date_str
        df_to_add = df_new[df_new["Date"] > last_date_str].copy()
        if df_to_add.empty:
            return {"ticker": ticker, "status": "up_to_date", "added_rows": 0, "last_date": last_date_str}

        cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df_to_add = df_to_add[[c for c in cols if c in df_to_add.columns]]

        # Append
        df_combined = pd.concat([df_existing, df_to_add], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["Date"]).sort_values("Date")
        df_combined.to_csv(filepath, index=False)

        new_last_date = str(df_combined["Date"].iloc[-1])
        return {"ticker": ticker, "status": "updated", "added_rows": len(df_to_add), "last_date": new_last_date}
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e), "added_rows": 0}

def update_all_tickers(tickers: list[str] = None, progress_callback=None) -> list[dict]:
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
        res = update_single_ticker(ticker)
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
        df_num = load_ticker_data(num)
        df_den = load_ticker_data(den)
        if df_num.empty or df_den.empty:
            continue

        merged = pd.merge(
            df_num[["Date", "Close", "Volume"]].rename(columns={"Close": f"Close_{num}", "Volume": f"Volume_{num}"}),
            df_den[["Date", "Close", "Volume"]].rename(columns={"Close": f"Close_{den}", "Volume": f"Volume_{den}"}),
            on="Date",
            how="inner"
        ).sort_values("Date")

        merged["Close"] = merged[f"Close_{num}"] / merged[f"Close_{den}"]
        merged["Open"] = merged["Close"]
        merged["High"] = merged["Close"]
        merged["Low"] = merged["Close"]
        merged["Volume"] = (merged[f"Volume_{num}"] + merged[f"Volume_{den}"]) / 2

        out_path = os.path.join(PROCESSED_DIR, f"{ratio_name.lower()}.csv")
        merged[["Date", "Open", "High", "Low", "Close", "Volume"]].to_csv(out_path, index=False)
