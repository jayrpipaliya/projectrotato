"""Reuse the original dashboard's market endpoints in the redesigned preview with durable cloud storage."""
import math
import os
import re
import sys
import shutil
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent
STAGED = (HERE / "legacy_dashboard" / "index.py").is_file()
ROOT = HERE if (STAGED or (HERE / "api" / "index.py").is_file()) else HERE.parents[1]
sys.path.insert(0, str(ROOT))

import data_manager
import durable_storage

# Keep generated ratios and gamma history in isolated runtime directory.
runtime = Path('/tmp/double-edge-preview') if os.environ.get('VERCEL') else HERE / '.runtime'
runtime.mkdir(parents=True, exist_ok=True)
raw_writable = Path('/tmp/raw_csv') if os.environ.get('VERCEL') else ROOT / 'raw_csv'
raw_writable.mkdir(parents=True, exist_ok=True)

for filename in ('gamma_history.json', 'spy_rsp.csv', 'qqq_rsp.csv', 'iwm_qqq.csv'):
    source, target = ROOT / 'processed_data' / filename, runtime / filename
    if source.is_file() and not target.exists():
        shutil.copyfile(source, target)
data_manager.PROCESSED_DIR = str(runtime)

# Restore persisted updates from durable cloud storage (Vercel Blob) on cold start
try:
    restored = durable_storage.sync_cloud_to_local(runtime, raw_writable)
    if restored > 0:
        print(f"[Double Edge] Restored {restored} persistent files from Vercel Blob.")
except Exception as e:
    print(f"[Double Edge] Cold-start cloud sync note: {e}")

if STAGED:
    from legacy_dashboard import index as legacy
else:
    from api import index as legacy
from data_manager import load_ticker_data

app = FastAPI(title="Double Edge Preview API")
READ_PATHS = {
    "/api/market-data", "/api/confluence", "/api/ratio/{ratio_name}",
    "/api/ticker/{ticker}", "/api/gamma/{ticker}", "/api/gamma/{ticker}/shift",
    "/api/gamma/{ticker}/heatmap", "/api/gamma/{ticker}/oi-history",
    "/api/options/{ticker}", "/api/options/{ticker}/smile",
    "/api/opportunities",
}
for route in legacy.app.routes:
    if route.path in READ_PATHS and getattr(route, "methods", None) == {"GET"}:
        app.add_api_route(route.path, route.endpoint, methods=["GET"])


@app.get("/api/preview/history/{ticker}")
def history(ticker: str):
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,14}", ticker):
        return JSONResponse({"error": "Invalid ticker."}, status_code=400)
    frame = load_ticker_data(ticker)
    if frame.empty:
        return JSONResponse({"error": "No OHLC data for this ticker."}, status_code=404)
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = []
    for date, *values in frame[columns].tail(520).itertuples(index=False, name=None):
        if all(math.isfinite(float(value)) for value in values):
            rows.append([date.strftime("%Y-%m-%d"), *map(float, values)])
    return {"ticker": ticker, "rows": rows}


def is_sync_request(request: Request):
    origin = request.headers.get("origin")
    expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
    return origin == expected and request.headers.get("x-preview-sync") == "1"


@app.post("/api/preview/sync-csv")
def sync_csv(request: Request):
    if not is_sync_request(request):
        return JSONResponse({"message": "Use the dashboard sync button."}, status_code=403)
    result = legacy.api_sync_csv()
    if isinstance(result, JSONResponse):
        return result
    return {"message": f"CSV cache reloaded for {result['loaded_count']} tickers. The hosted build reads its packaged CSVs."}


@app.post("/api/preview/sync-yfinance")
def sync_yfinance(request: Request):
    if not is_sync_request(request):
        return JSONResponse({"message": "Use the dashboard sync button."}, status_code=403)
    return run_yfinance()


# The local Node preview uses this original route name.
@app.post("/api/sync-data", include_in_schema=False)
def local_yfinance(request: Request):
    if request.client and request.client.host != "127.0.0.1" and not os.environ.get("VERCEL"):
        return JSONResponse({"message": "Local dashboard sync only."}, status_code=403)
    return run_yfinance()


def run_yfinance():
    result = legacy.api_sync_data({"scope": "core"})
    if isinstance(result, JSONResponse):
        return result

    # Regenerate synthetic ratios from fresh ticker data
    try:
        data_manager.generate_synthetic_ratios()
    except Exception as e:
        print(f"[Sync] Error generating synthetic ratios: {e}")

    legacy._CACHE.clear()

    # Persist updated files to durable cloud storage if configured
    persisted_count = 0
    if durable_storage.get_blob_token():
        try:
            for ratio in ('spy_rsp.csv', 'qqq_rsp.csv', 'iwm_qqq.csv'):
                f = runtime / ratio
                if f.is_file() and durable_storage.persist_file_to_cloud(f, ratio):
                    persisted_count += 1
            if raw_writable.is_dir():
                for p in raw_writable.glob("*_us_d.csv"):
                    if durable_storage.persist_file_to_cloud(p, p.name):
                        persisted_count += 1
            durable_storage.persist_gamma_history(runtime)
        except Exception as e:
            print(f"[Storage] Cloud persistence warning: {e}")

    storage_note = (
        f" Persisted {persisted_count} files to Vercel Blob."
        if persisted_count > 0
        else (" Persisted to Vercel Blob." if durable_storage.get_blob_token() else " Saved to runtime cache.")
    )
    msg = (
        f"Synced {result.get('updated_count', 0)} of {result.get('total', 0)} tickers "
        f"from yfinance (latest: {result.get('last_date', 'today')}).{storage_note}"
    )
    return {"message": msg, **result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
