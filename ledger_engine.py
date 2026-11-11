"""
Double Edge Quantitative Dashboard - Trade Ledger & Forward AI Auditor Engine
Handles persistent trade tracking, user parameter capture, baseline snapshotting,
mark-to-market calculations, and AI forward audits.
"""

import os
import json
import time
from datetime import datetime

# In-memory short-lived cache for ticker spot prices to keep UI blazing fast
_PRICE_CACHE = {}


def get_ledger_filepath() -> str:
    """Return persistent ledger path, with safe fallback for serverless/read-only environments."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    primary_dir = os.path.join(base_dir, "processed_data")
    os.makedirs(primary_dir, exist_ok=True)
    primary_file = os.path.join(primary_dir, "trade_ledger.json")

    # If writable, use primary file
    try:
        with open(primary_file, "a") as f:
            pass
        return primary_file
    except Exception:
        # Fallback to /tmp
        tmp_dir = os.path.join("/tmp", "dash_data")
        os.makedirs(tmp_dir, exist_ok=True)
        return os.path.join(tmp_dir, "trade_ledger.json")


def load_ledger() -> list[dict]:
    """Load all trade records from the persistent JSON ledger."""
    filepath = get_ledger_filepath()
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def save_ledger(trades: list[dict]) -> bool:
    """Persist trades list to JSON."""
    filepath = get_ledger_filepath()
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"[Ledger Engine] Error saving ledger: {e}")
        return False


def fetch_live_price(ticker: str) -> float:
    """Retrieve the latest spot close price for a ticker from local data or yfinance with 30s cache."""
    ticker_up = ticker.upper().strip()
    now = time.time()
    if ticker_up in _PRICE_CACHE:
        cached_px, ts = _PRICE_CACHE[ticker_up]
        if now - ts < 30 and cached_px > 0:
            return cached_px

    px = 0.0
    try:
        from data_manager import load_ticker_data
        df = load_ticker_data(ticker_up)
        if df is not None and not df.empty and "Close" in df.columns:
            px = float(df["Close"].iloc[-1])
    except Exception:
        pass

    if px <= 0:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker_up)
            fast_px = getattr(t, "fast_info", None)
            if fast_px and hasattr(fast_px, "last_price") and fast_px.last_price:
                px = float(fast_px.last_price)
            else:
                hist = t.history(period="2d")
                if not hist.empty and "Close" in hist.columns:
                    px = float(hist["Close"].iloc[-1])
        except Exception:
            pass

    if px > 0:
        _PRICE_CACHE[ticker_up] = (px, now)
    return px


def add_trade(
    ticker: str,
    direction: str,
    asset_type: str = "STOCK",  # "STOCK" or "OPTION"
    entry_price: float = 0.0,
    size: float = 1.0,  # Shares or Contracts
    stop_loss: float = 0.0,
    target_1: float = 0.0,
    target_2: float = 0.0,
    # Options specifics:
    contract_type: str = "CALL",  # "CALL" or "PUT"
    strike: float = None,
    expiration: str = None,
    underlying_entry_spot: float = None,
    # Context / Snapshot
    notes: str = "",
    ai_snapshot: dict = None
) -> dict:
    """
    Log a new trade with user execution parameters and a frozen AI snapshot.
    """
    trades = load_ledger()
    timestamp_id = int(time.time() * 1000) % 100000000
    ticker_clean = ticker.upper().strip()
    trade_id = f"TRD-{ticker_clean}-{timestamp_id}"

    is_option = asset_type.upper() == "OPTION"
    multiplier = 100 if is_option else 1
    position_val = round(float(entry_price) * float(size) * multiplier, 2)

    # Determine underlying spot at entry
    if underlying_entry_spot is None or underlying_entry_spot <= 0:
        if ai_snapshot and ai_snapshot.get("entry_price_spot"):
            try:
                underlying_entry_spot = float(ai_snapshot["entry_price_spot"])
            except Exception:
                underlying_entry_spot = 0.0

    if underlying_entry_spot is None or underlying_entry_spot <= 0:
        if not is_option:
            underlying_entry_spot = float(entry_price)
        else:
            underlying_entry_spot = fetch_live_price(ticker_clean)

    # Ensure ai_snapshot has entry_price_spot
    snap = ai_snapshot.copy() if ai_snapshot else {}
    if "entry_price_spot" not in snap or not snap["entry_price_spot"]:
        snap["entry_price_spot"] = underlying_entry_spot

    new_trade = {
        "id": trade_id,
        "ticker": ticker_clean,
        "direction": direction.upper(),
        "asset_type": "OPTION" if is_option else "STOCK",
        "entry_price": float(entry_price),
        "size": float(size),
        "position_value": position_val,
        "underlying_entry_spot": float(underlying_entry_spot) if underlying_entry_spot else 0.0,
        "stop_loss": float(stop_loss) if stop_loss else 0.0,
        "target_1": float(target_1) if target_1 else 0.0,
        "target_2": float(target_2) if target_2 else 0.0,
        "contract_type": contract_type.upper() if is_option else None,
        "strike": float(strike) if (is_option and strike is not None) else None,
        "expiration": str(expiration) if (is_option and expiration) else None,
        "notes": notes or "",
        "status": "ACTIVE",  # ACTIVE, TARGET_1_HIT, TARGET_2_HIT, STOPPED_OUT, CLOSED
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "closed_at": None,
        "exit_price": None,
        "realized_pnl": None,
        "realized_pct": None,
        "ai_snapshot": snap
    }

    trades.insert(0, new_trade)  # Most recent first
    save_ledger(trades)
    return new_trade


def update_trade(trade_id: str, updates: dict) -> dict:
    """Update fields on an existing trade (e.g. stop loss, notes, targets)."""
    trades = load_ledger()
    target_trade = None
    for t in trades:
        if t.get("id") == trade_id:
            for k, v in updates.items():
                if k != "id":  # immutable
                    t[k] = v
            target_trade = t
            break

    if target_trade:
        save_ledger(trades)
    return target_trade


def close_trade(trade_id: str, exit_price: float, outcome: str = "CLOSED", notes: str = None) -> dict:
    """
    Close an active trade and record realized PnL.
    Handles correct options pricing (Long Call or Long Put profit = exit_premium - entry_premium).
    """
    trades = load_ledger()
    target_trade = None
    for t in trades:
        if t.get("id") == trade_id:
            t["status"] = outcome
            t["exit_price"] = float(exit_price)
            t["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            if notes:
                t["notes"] = f"{t.get('notes', '')}\nClose Note: {notes}".strip()

            entry_px = float(t.get("entry_price", 0) or 0.0)
            size = float(t.get("size", 1.0))
            is_option = t.get("asset_type") == "OPTION"
            multiplier = 100 if is_option else 1
            direction = str(t.get("direction", "LONG")).upper()
            is_short = ("SHORT" in direction or "FADE" in direction) and not is_option

            if entry_px <= 0:
                pct = 0.0
                pnl = exit_price * size * multiplier
            else:
                if is_short:
                    # Short Equity position
                    pct = ((entry_px - exit_price) / entry_px) * 100
                    pnl = (entry_px - exit_price) * size * multiplier
                else:
                    # Long Equity OR Purchased Option (Call or Put)
                    pct = ((exit_price - entry_px) / entry_px) * 100
                    pnl = (exit_price - entry_px) * size * multiplier

            t["realized_pct"] = round(pct, 2)
            t["realized_pnl"] = round(pnl, 2)
            target_trade = t
            break

    if target_trade:
        save_ledger(trades)
    return target_trade


def delete_trade(trade_id: str) -> bool:
    """Remove a trade permanently from the ledger."""
    trades = load_ledger()
    initial_len = len(trades)
    trades = [t for t in trades if t.get("id") != trade_id]
    if len(trades) < initial_len:
        save_ledger(trades)
        return True
    return False


def compute_mark_to_market(trade: dict, live_spot: float = None) -> dict:
    """
    Calculate live PnL, progress toward Target 1 vs Stop Invalidation,
    and institutional desk trigger flags.
    Accurately accounts for directionality (Bullish vs Bearish) and Options mechanics.
    """
    ticker = trade.get("ticker", "")
    if live_spot is None or live_spot <= 0:
        live_spot = fetch_live_price(ticker)

    entry_px = float(trade.get("entry_price") or 0.0)
    stop_loss = float(trade.get("stop_loss") or 0.0)
    target_1 = float(trade.get("target_1") or 0.0)
    target_2 = float(trade.get("target_2") or 0.0)
    size = float(trade.get("size") or 1.0)
    is_option = trade.get("asset_type") == "OPTION"
    multiplier = 100 if is_option else 1

    direction = str(trade.get("direction", "LONG")).upper()
    is_short_label = "SHORT" in direction or "FADE" in direction or direction == "PUT"
    is_put = trade.get("contract_type") == "PUT"
    is_bearish = is_put or (is_short_label and not is_option)

    # Resolve underlying spot price at entry (CRITICAL FIX: never fallback to option premium entry_px)
    underlying_entry = float(
        trade.get("underlying_entry_spot") or
        trade.get("ai_snapshot", {}).get("entry_price_spot") or
        0.0
    )
    if underlying_entry <= 0:
        underlying_entry = live_spot if is_option else entry_px

    res = {
        "ticker": ticker,
        "live_spot": live_spot,
        "entry_price": entry_px,
        "underlying_entry_spot": underlying_entry,
        "unrealized_pct": 0.0,
        "unrealized_pnl": 0.0,
        "dist_to_stop_pct": 0.0,
        "dist_to_target1_pct": 0.0,
        "progress_pct": 0.0,
        "trigger_alert": None,
        "alert_type": "info"  # success, warning, error, info
    }

    if live_spot <= 0 or entry_px <= 0:
        return res

    # 1. Price Return & PnL
    if is_option:
        # Options estimate: delta-adjusted heuristic anchored on underlying spot difference
        # Delta: +0.45 for Calls, -0.45 for Puts
        delta = -0.45 if is_bearish else 0.45
        spot_diff = live_spot - underlying_entry
        est_contract_gain = spot_diff * delta
        est_current_opt_px = max(0.01, entry_px + est_contract_gain)
        unrealized_pct = ((est_current_opt_px - entry_px) / entry_px) * 100
        unrealized_pnl = (est_current_opt_px - entry_px) * size * 100
        res["est_option_px"] = round(est_current_opt_px, 2)
    else:
        # Stock
        if is_bearish:
            unrealized_pct = ((entry_px - live_spot) / entry_px) * 100
            unrealized_pnl = (entry_px - live_spot) * size
        else:
            unrealized_pct = ((live_spot - entry_px) / entry_px) * 100
            unrealized_pnl = (live_spot - entry_px) * size

    res["unrealized_pct"] = round(unrealized_pct, 2)
    res["unrealized_pnl"] = round(unrealized_pnl, 2)

    # 2. Distance to Stop & Target (based on underlying spot)
    ref_spot = live_spot
    if stop_loss > 0 and ref_spot > 0:
        if is_bearish:
            # Bearish setup: Stop is above spot, so buffer = stop_loss - ref_spot
            res["dist_to_stop_pct"] = round(((stop_loss - ref_spot) / ref_spot) * 100, 2)
        else:
            # Bullish setup: Stop is below spot, so buffer = ref_spot - stop_loss
            res["dist_to_stop_pct"] = round(((ref_spot - stop_loss) / ref_spot) * 100, 2)

    if target_1 > 0 and ref_spot > 0:
        if is_bearish:
            # Bearish setup: Target is below spot
            res["dist_to_target1_pct"] = round(((ref_spot - target_1) / ref_spot) * 100, 2)
        else:
            # Bullish setup: Target is above spot
            res["dist_to_target1_pct"] = round(((target_1 - ref_spot) / ref_spot) * 100, 2)

    # 3. Path Progress % towards Target 1
    bench_entry = underlying_entry if is_option else entry_px
    if target_1 > 0 and bench_entry > 0 and target_1 != bench_entry:
        if is_bearish:
            progress = ((bench_entry - ref_spot) / (bench_entry - target_1)) * 100
        else:
            progress = ((ref_spot - bench_entry) / (target_1 - bench_entry)) * 100
        res["progress_pct"] = max(-100.0, min(150.0, round(progress, 1)))

    # 4. Trigger Alerts (Direction-aware)
    if target_2 > 0 and ((not is_bearish and ref_spot >= target_2) or (is_bearish and ref_spot <= target_2)):
        res["trigger_alert"] = f"Target 2 Runner Hit (${ref_spot:,.2f})! Fully maximize gains or trail hard stop."
        res["alert_type"] = "success"
    elif target_1 > 0 and ((not is_bearish and ref_spot >= target_1) or (is_bearish and ref_spot <= target_1)):
        res["trigger_alert"] = f"Target 1 Reached (${ref_spot:,.2f})! Recommended: Harvest 50% profits & move stop to breakeven."
        res["alert_type"] = "success"
    elif stop_loss > 0 and ((not is_bearish and ref_spot <= stop_loss) or (is_bearish and ref_spot >= stop_loss)):
        res["trigger_alert"] = f"Invalidation Stop Breached (${ref_spot:,.2f})! Thesis broken—desk directive is to exit immediately."
        res["alert_type"] = "error"
    elif stop_loss > 0 and 0.0 <= res["dist_to_stop_pct"] < 1.0:
        res["trigger_alert"] = f"Critical Risk: Within {res['dist_to_stop_pct']:.1f}% of Invalidation Stop (${stop_loss:,.2f})."
        res["alert_type"] = "warning"
    elif target_1 > 0 and 0.0 <= res["dist_to_target1_pct"] < 1.0:
        res["trigger_alert"] = f"Approaching Target 1: Less than 1.0% away (${target_1:,.2f}). Prepare to lock in gains."
        res["alert_type"] = "info"

    return res


def get_ledger_stats() -> dict:
    """Calculate executive stats across all active and closed trades."""
    trades = load_ledger()
    active_trades = [t for t in trades if t.get("status") == "ACTIVE"]
    closed_trades = [t for t in trades if t.get("status") != "ACTIVE"]

    def _safe_float(v):
        try:
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    realized_pnl = sum([_safe_float(t.get("realized_pnl")) for t in closed_trades])
    winning_closed = [t for t in closed_trades if _safe_float(t.get("realized_pnl")) > 0]
    win_rate = round((len(winning_closed) / len(closed_trades) * 100), 1) if closed_trades else 0.0

    # Mark to market active
    open_unrealized_pnl = 0.0
    for t in active_trades:
        mtm = compute_mark_to_market(t)
        open_unrealized_pnl += mtm.get("unrealized_pnl", 0.0)

    return {
        "total_trades": len(trades),
        "active_trades_count": len(active_trades),
        "closed_trades_count": len(closed_trades),
        "win_rate_closed": win_rate,
        "realized_pnl": round(realized_pnl, 2),
        "open_unrealized_pnl": round(open_unrealized_pnl, 2),
        "active_trades": active_trades,
        "closed_trades": closed_trades
    }


def generate_ai_portfolio_audit(market_data: dict = None, gemini_api_key: str = None) -> str:
    """
    Run an institutional AI audit across all open active positions in the ledger.
    Compares live metrics to entry snapshot and generates tactical risk management directives.
    Includes latest dealer gamma profile (Call Wall / Put Wall shifts) for each position.
    """
    from ai_strategist import get_gemini_client, GEMINI_MODELS

    trades = load_ledger()
    active_trades = [t for t in trades if t.get("status") == "ACTIVE"]
    if not active_trades:
        return "No active positions found in the Trade Ledger to audit. Log an AI trade idea or add a manual setup to begin monitoring."

    client, key = get_gemini_client(gemini_api_key)
    if not client:
        return "Gemini API key is required to run the automated AI Portfolio & Progress Audit."

    # Build active trades snapshot with live MTM and latest gamma exposure
    audit_items = []
    for t in active_trades:
        mtm = compute_mark_to_market(t)
        snap = t.get("ai_snapshot", {})

        current_cw = None
        current_pw = None
        current_zg = None
        current_gex = None
        bench_sym = None
        bench_pw = None
        bench_zg = None
        bench_regime = None
        try:
            from options_engine import compute_gamma_exposure_profile
            from ai_strategist import get_asset_benchmark
            g = compute_gamma_exposure_profile(t.get("ticker"), max_expirations=3)
            if g and "error" not in g:
                current_cw = g.get("call_wall")
                current_pw = g.get("put_wall")
                current_zg = g.get("zero_gamma_level")
                current_gex = f"${g.get('total_net_gex_billions'):+.2f}B"

            bench_sym = get_asset_benchmark(t.get("ticker"))
            bg = compute_gamma_exposure_profile(bench_sym, max_expirations=3)
            if bg and "error" not in bg:
                bench_pw = bg.get("put_wall")
                bench_zg = bg.get("zero_gamma_level")
                bench_regime = bg.get("regime_short")
        except Exception:
            pass

        audit_items.append({
            "id": t.get("id"),
            "ticker": t.get("ticker"),
            "direction": t.get("direction"),
            "asset_type": t.get("asset_type"),
            "contract": f"{t.get('strike')} {t.get('contract_type')} Exp: {t.get('expiration')}" if t.get("asset_type") == "OPTION" else "STOCK",
            "entry_price": t.get("entry_price"),
            "entry_underlying_spot": mtm.get("underlying_entry_spot"),
            "live_spot": mtm.get("live_spot"),
            "unrealized_pnl_dollars": mtm.get("unrealized_pnl"),
            "unrealized_pct": mtm.get("unrealized_pct"),
            "stop_loss": t.get("stop_loss"),
            "dist_to_stop_pct": mtm.get("dist_to_stop_pct"),
            "target_1": t.get("target_1"),
            "target_2": t.get("target_2"),
            "progress_to_target_pct": mtm.get("progress_pct"),
            "alert": mtm.get("trigger_alert"),
            "entry_historical_win_rate": snap.get("win_rate"),
            "entry_sample_size": snap.get("sample_size"),
            "entry_confluence_score": snap.get("confluence_score"),
            "entry_call_wall": snap.get("call_wall"),
            "entry_put_wall": snap.get("put_wall"),
            "current_live_call_wall": current_cw,
            "current_live_put_wall": current_pw,
            "current_live_zero_gamma": current_zg,
            "current_live_net_gex": current_gex,
            "primary_benchmark": bench_sym,
            "benchmark_live_put_wall": bench_pw,
            "benchmark_live_zero_gamma": bench_zg,
            "benchmark_live_regime": bench_regime,
            "original_ai_thesis": snap.get("thesis")
        })

    prompt = f"""
You are the Chief Quantitative Risk Officer auditing open positions currently tracked in the Double Edge Quantitative Ledger.
Audit the following live active positions against their initial entry thesis, historical backtest odds, and current market/gamma levels:

ACTIVE TRADES DATA SNAPSHOT:
{json.dumps(audit_items, indent=2)}

AUDIT REPORT REQUIREMENTS:
1. Open with an Executive Summary Markdown Table reviewing each position:
   | Ticker | Position | Live P&L ($ / %) | Progress to Target | Stop Risk | Action Directive (HOLD / TRIM / TRAIL STOP / CUT) |
2. For each active trade, deliver a crisp 3-sentence institutional review:
   - Progress vs. Original AI Thesis & 20-Year Historical Odds.
   - Dual Gamma Evaluation: Compare Entry Call/Put Walls against Current Live Call/Put Walls. Confirm whether the primary benchmark (e.g. QQQ for tech equities like MSFT) is holding above its Put Wall / Zero-Gamma flip or triggering systematic de-risking.
   - Concrete Action Directive: Where to adjust the stop loss (e.g. breakeven, trailing 20 EMA), or whether to harvest profits now.
3. Strict Terminal Tone: Crisp, mathematical, visual tables, zero fluff, no memo headers or formal letters.
"""

    for m in GEMINI_MODELS:
        try:
            res = client.models.generate_content(model=m, contents=prompt)
            if res and res.text:
                return res.text.strip()
        except Exception:
            continue

    return "Audit generation failed due to API connection issues. Please try again in a moment."
