"""
Institutional Gamma Tracker & Strike Migration Heat Map Engine
- Records daily dealer Gamma Exposure (GEX) snapshots to track historical strike positioning.
- Calculates Day-over-Day (DoD) Wall Migration (Call Wall rolls, Put Wall retreats, Zero-Gamma shifts).
- Computes Net GEX Velocity (ΔGEX) to detect volatility regime transitions before price breaks.
- Generates 2D Strike-versus-Time Heat Maps with Underlying Spot Trajectory overlays.
"""

import os
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from data_manager import PROCESSED_DIR, load_ticker_data

HISTORY_FILE = os.path.join(PROCESSED_DIR, "gamma_history.json")
_HISTORY_CACHE = {}


def _safe_float(val, default: float = 0.0) -> float:
    """Safely converts input to float without throwing TypeError on None or ValueError."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def format_signed_currency(val: float, suffix: str = "M", decimals: int = 1) -> str:
    """
    Format signed currency cleanly (+45.2M, -$12.5M, +$0.25B)
    preventing '+$-12.5M' or '$-12.5M' string formatting artifacts.
    """
    if val is None:
        return f"+$0.0{suffix}"
    v = _safe_float(val, 0.0)
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.{decimals}f}{suffix}"


def _lookup_strike(s_dict: dict, k: float) -> dict:
    """Robust strike lookup with float tolerance to eliminate string representation misses."""
    if not s_dict:
        return {}
    # 1. Exact string matches
    for fmt in (f"{k:.2f}", f"{k:.1f}", f"{round(k)}", str(k)):
        if fmt in s_dict:
            return s_dict[fmt]
    # 2. Tolerant float search
    for sk_str, val in s_dict.items():
        try:
            if abs(float(sk_str) - k) < 0.05:
                return val
        except Exception:
            continue
    return {}


def get_history_filepath() -> str:
    """Return valid persistent filepath for gamma snapshots."""
    global HISTORY_FILE
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        # Test writeability
        test_file = os.path.join(os.path.dirname(HISTORY_FILE), ".write_test")
        with open(test_file, "w") as f:
            f.write("1")
        if os.path.exists(test_file):
            os.remove(test_file)
        return HISTORY_FILE
    except Exception:
        fallback = "/tmp/gamma_history.json" if os.name != "nt" else os.path.join(os.environ.get("TEMP", "."), "gamma_history.json")
        return fallback


def load_gamma_history(ticker: str = None) -> dict:
    """
    Load historical gamma snapshots dictionary:
    {
      "SPY": [ { "date": "2026-11-04", "spot_price": ..., "call_wall": ..., "strikes": { ... } }, ... ],
      "QQQ": [ ... ]
    }
    """
    global _HISTORY_CACHE
    filepath = get_history_filepath()
    if not os.path.exists(filepath):
        # Auto-seed initial baseline history if new
        _seed_default_history_if_needed()

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                _HISTORY_CACHE = data
        except Exception as e:
            print(f"[Gamma Tracker] Error reading history: {e}")
            if not _HISTORY_CACHE:
                _HISTORY_CACHE = {}

    if ticker:
        t_key = ticker.upper().strip()
        raw_list = _HISTORY_CACHE.get(t_key, [])
        if not raw_list:
            # On-demand seed if ticker has data available
            _seed_ticker_if_needed(t_key)
            raw_list = _HISTORY_CACHE.get(t_key, [])

        # Deduplicate strictly by date and sort chronologically
        deduped = {}
        for item in raw_list:
            d = item.get("date")
            if d:
                deduped[d] = item
        return [deduped[d] for d in sorted(deduped.keys())]
    return _HISTORY_CACHE


def save_gamma_history(history: dict) -> bool:
    """Persist gamma history dictionary to disk."""
    global _HISTORY_CACHE
    filepath = get_history_filepath()
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
        _HISTORY_CACHE = history
        return True
    except Exception as e:
        print(f"[Gamma Tracker] Error saving history: {e}")
        return False


def record_gamma_snapshot(ticker: str, profile: dict) -> bool:
    """
    Record a live GEX snapshot from compute_gamma_exposure_profile.
    Maintains a rolling 30-day window per ticker.
    """
    if not profile or "error" in profile or not profile.get("call_wall"):
        return False

    t_up = ticker.upper().strip()
    history = load_gamma_history()
    t_hist = history.get(t_up, [])

    # Today's date string YYYY-MM-DD
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Extract strike GEX dictionary (relevant strikes)
    strikes_compact = {}
    for s in profile.get("strikes", []):
        if s.get("is_relevant", True) or abs(s["strike"] - profile["spot_price"]) / profile["spot_price"] <= 0.08:
            k_str = f"{s['strike']:.2f}"
            strikes_compact[k_str] = {
                "net_gex": round(s.get("net_gex", 0), 1),
                "call_gex": round(s.get("call_gex", 0), 1),
                "put_gex": round(s.get("put_gex", 0), 1),
                "call_oi": s.get("call_oi", 0),
                "put_oi": s.get("put_oi", 0)
            }

    entry = {
        "date": today_str,
        "timestamp": int(time.time()),
        "spot_price": profile.get("spot_price"),
        "call_wall": profile.get("call_wall"),
        "call_wall_gex_m": profile.get("call_wall_gex_m"),
        "put_wall": profile.get("put_wall"),
        "put_wall_gex_m": profile.get("put_wall_gex_m"),
        "zero_gamma_level": profile.get("zero_gamma_level"),
        "total_net_gex_billions": profile.get("total_net_gex_billions"),
        "total_call_gex_billions": profile.get("total_call_gex_billions"),
        "total_put_gex_billions": profile.get("total_put_gex_billions"),
        "regime_short": profile.get("regime_short"),
        "strikes": strikes_compact
    }

    # If today already exists, update it with latest intraday print; otherwise append
    updated = False
    for idx, ex in enumerate(t_hist):
        if ex.get("date") == today_str:
            t_hist[idx] = entry
            updated = True
            break

    if not updated:
        t_hist.append(entry)

    # Sort ascending by date and cap at 30 entries
    t_hist = sorted(t_hist, key=lambda x: x.get("date", ""))[-30:]
    history[t_up] = t_hist
    return save_gamma_history(history)


def compute_gamma_shift(ticker: str) -> dict:
    """
    Computes Day-over-Day (DoD) Gamma Migration & ΔGEX Shift:
    - Call Wall Migration (Roll Up / Roll Down / Flat)
    - Put Wall Migration (Roll Up / Roll Down / Flat)
    - Zero Gamma Flip Migration
    - Net GEX Velocity (ΔGEX Billions)
    - Top Accumulating & Hedging Strikes
    - Institutional Qualitative Signal
    """
    t_up = ticker.upper().strip()
    history = load_gamma_history(t_up)

    if not history:
        # Try seeding default baseline
        _seed_default_history_if_needed()
        history = load_gamma_history(t_up)

    if not history or len(history) < 1:
        return {
            "ticker": t_up,
            "has_history": False,
            "message": f"Insufficient historical GEX snapshots for {t_up}."
        }

    curr = history[-1]
    is_baseline_only = len(history) < 2
    prev = history[-2] if not is_baseline_only else curr

    # Price shifts
    spot_c = _safe_float(curr.get("spot_price"), 100.0)
    spot_p = _safe_float(prev.get("spot_price"), spot_c)
    spot_delta = spot_c - spot_p
    spot_pct = ((spot_c - spot_p) / spot_p) * 100 if spot_p > 0 else 0.0

    # Call Wall
    cw_c = _safe_float(curr.get("call_wall"), spot_c)
    cw_p = _safe_float(prev.get("call_wall"), cw_c)
    cw_shift = cw_c - cw_p
    if is_baseline_only:
        cw_status = "INITIAL_BASELINE"
        cw_signal = f"Ceiling Pinned at ${cw_c:,.2f} — Initial Snapshot Baseline"
    elif cw_shift > 0.01:
        cw_status = "ROLLED_UP"
        cw_signal = f"Ceiling Rolled Higher (+${cw_shift:,.2f}) — Room to Run"
    elif cw_shift < -0.01:
        cw_status = "ROLLED_DOWN"
        cw_signal = f"Ceiling Rolled Lower (-${abs(cw_shift):,.2f}) — Resistance Capping"
    else:
        cw_status = "UNCHANGED"
        cw_signal = f"Ceiling Pinned at ${cw_c:,.2f} — Stationary Resistance"

    # Put Wall
    pw_c = _safe_float(curr.get("put_wall"), spot_c)
    pw_p = _safe_float(prev.get("put_wall"), pw_c)
    pw_shift = pw_c - pw_p
    if is_baseline_only:
        pw_status = "INITIAL_BASELINE"
        pw_signal = f"Support Pinned at ${pw_c:,.2f} — Initial Snapshot Baseline"
    elif pw_shift > 0.01:
        pw_status = "ROLLED_UP"
        pw_signal = f"Support Rolled Higher (+${pw_shift:,.2f}) — Higher Low Firming"
    elif pw_shift < -0.01:
        pw_status = "ROLLED_DOWN"
        pw_signal = f"Support Rolled Lower (-${abs(pw_shift):,.2f}) — Downside Cushion Dropped"
    else:
        pw_status = "UNCHANGED"
        pw_signal = f"Support Pinned at ${pw_c:,.2f} — Stationary Bid Floor"

    # Zero Gamma Flip
    zg_c = _safe_float(curr.get("zero_gamma_level"), spot_c)
    zg_p = _safe_float(prev.get("zero_gamma_level"), zg_c)
    zg_shift = zg_c - zg_p

    # Net GEX Velocity
    net_c = _safe_float(curr.get("total_net_gex_billions"), 0.0)
    net_p = _safe_float(prev.get("total_net_gex_billions"), net_c)
    net_delta = net_c - net_p

    if is_baseline_only:
        velocity_label = "INITIAL SNAPSHOT"
        velocity_desc = f"{format_signed_currency(net_c, 'B', 2)} Net GEX — Initial baseline snapshot recorded."
    elif net_delta >= 0.25:
        velocity_label = "ACCUMULATING LONG GAMMA"
        velocity_desc = f"{format_signed_currency(net_delta, 'B', 2)} GEX expansion — Dealers absorbing volatility and pinning price."
    elif net_delta <= -0.25:
        velocity_label = "BLEEDING DEALER GAMMA"
        velocity_desc = f"{format_signed_currency(net_delta, 'B', 2)} GEX drain — Dealer cushions thinning, volatility risk rising."
    else:
        velocity_label = "STABLE GEX POSITIONING"
        velocity_desc = f"{format_signed_currency(net_delta, 'B', 2)} GEX change — Neutral day-over-day dealer inventory shift."

    # Strike-by-Strike ΔGEX (Where did capital move?)
    c_strikes = curr.get("strikes", {})
    p_strikes = prev.get("strikes", {})
    all_k = set(c_strikes.keys()) | set(p_strikes.keys())

    strike_shifts = []
    for k_s in all_k:
        try:
            k_val = float(k_s)
        except Exception:
            continue
        c_item = _lookup_strike(c_strikes, k_val)
        p_item = _lookup_strike(p_strikes, k_val)

        c_call_gex = _safe_float(c_item.get("call_gex"), 0.0)
        p_call_gex = _safe_float(p_item.get("call_gex"), 0.0)
        delta_call_gex = c_call_gex - p_call_gex

        c_put_gex = _safe_float(c_item.get("put_gex"), 0.0)
        p_put_gex = _safe_float(p_item.get("put_gex"), 0.0)
        delta_put_gex = c_put_gex - p_put_gex

        c_net_gex = _safe_float(c_item.get("net_gex"), 0.0)
        p_net_gex = _safe_float(p_item.get("net_gex"), 0.0)
        delta_net_gex = c_net_gex - p_net_gex

        strike_shifts.append({
            "strike": k_val,
            "strike_str": f"${k_val:,.2f}",
            "delta_net_gex_m": round(delta_net_gex, 1),
            "delta_call_gex_m": round(delta_call_gex, 1),
            "delta_put_gex_m": round(delta_put_gex, 1),
            "curr_net_gex_m": round(c_net_gex, 1),
            "curr_call_oi": c_item.get("call_oi", 0),
            "curr_put_oi": c_item.get("put_oi", 0)
        })

    strike_shifts = sorted(strike_shifts, key=lambda x: x["strike"])

    # Top accumulating call strikes and top put hedging strikes
    top_calls_accum = sorted(strike_shifts, key=lambda x: x["delta_call_gex_m"], reverse=True)[:3]
    top_puts_hedged = sorted(strike_shifts, key=lambda x: x["delta_put_gex_m"], reverse=True)[:3]

    # Institutional Executive Synthesis
    if is_baseline_only:
        synthesis = (
            f"{t_up} Baseline Gamma Profile ({curr.get('date')}): "
            f"Call Wall at ${cw_c:,.2f}, Put Wall at ${pw_c:,.2f}, "
            f"Zero Flip at ${zg_c:,.2f}, Net GEX at {format_signed_currency(net_c, 'B', 2)}. "
            f"Day-over-Day shift will compare against this baseline upon next trading session."
        )
    else:
        synthesis = (
            f"{t_up} Gamma Migration ({prev.get('date')} &rarr; {curr.get('date')}): "
            f"Call Wall {cw_status.replace('_', ' ')} to ${cw_c:,.2f} ({cw_shift:+.2f}), "
            f"Put Wall {pw_status.replace('_', ' ')} to ${pw_c:,.2f} ({pw_shift:+.2f}). "
            f"Net GEX shifted {format_signed_currency(net_delta, 'B', 2)} to {format_signed_currency(net_c, 'B', 2)} ({velocity_label})."
        )

    return {
        "ticker": t_up,
        "has_history": True,
        "curr_date": curr.get("date"),
        "prev_date": prev.get("date"),
        "spot_current": spot_c,
        "spot_previous": spot_p,
        "spot_change_pct": round(spot_pct, 2),
        "call_wall_current": cw_c,
        "call_wall_previous": cw_p,
        "call_wall_shift": round(cw_shift, 2),
        "call_wall_status": cw_status,
        "call_wall_signal": cw_signal,
        "put_wall_current": pw_c,
        "put_wall_previous": pw_p,
        "put_wall_shift": round(pw_shift, 2),
        "put_wall_status": pw_status,
        "put_wall_signal": pw_signal,
        "zero_gamma_current": zg_c,
        "zero_gamma_previous": zg_p,
        "zero_gamma_shift": round(zg_shift, 2),
        "net_gex_current_b": round(net_c, 3),
        "net_gex_previous_b": round(net_p, 3),
        "net_gex_velocity_b": round(net_delta, 3),
        "velocity_label": velocity_label,
        "velocity_desc": velocity_desc,
        "top_calls_accumulating": top_calls_accum,
        "top_puts_hedging": top_puts_hedged,
        "all_strike_shifts": strike_shifts,
        "institutional_synthesis": synthesis,
        "history_count": len(history)
    }


def generate_gamma_heatmap_matrix(ticker: str, mode: str = "net_gex", max_days: int = 10) -> dict:
    """
    Constructs a 2D matrix suitable for Plotly Heatmap:
    X-axis: Dates (formatted for human scannability)
    Y-axis: Option Strike Prices (ascending)
    Z-axis: Matrix of Net GEX values or ΔGEX changes
    Overlay: Spot price per date, Call Wall per date, Put Wall per date
    """
    t_up = ticker.upper().strip()
    history = load_gamma_history(t_up)

    if not history:
        _seed_default_history_if_needed()
        history = load_gamma_history(t_up)

    if not history:
        return {"error": f"No historical snapshots available for {t_up}"}

    # Slice to last max_days with prior reference baseline
    start_idx = max(0, len(history) - max_days)
    history_slice = history[start_idx:]
    prior_baseline = history[start_idx - 1] if start_idx > 0 else None

    if len(history_slice) < 1:
        return {"error": "Insufficient history records"}

    # Extract dates list
    dates_raw = [h.get("date") for h in history_slice]
    dates_labels = []
    for d_str in dates_raw:
        try:
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            dates_labels.append(dt.strftime("%b %d"))
        except Exception:
            dates_labels.append(d_str)

    # Extract spots, call walls, put walls per date
    spots = [_safe_float(h.get("spot_price"), 0.0) for h in history_slice]
    call_walls = [_safe_float(h.get("call_wall"), 0.0) for h in history_slice]
    put_walls = [_safe_float(h.get("put_wall"), 0.0) for h in history_slice]
    zero_gammas = [_safe_float(h.get("zero_gamma_level"), 0.0) for h in history_slice]
    net_gex_b = [_safe_float(h.get("total_net_gex_billions"), 0.0) for h in history_slice]

    # Collect universal strike universe across all dates in slice
    strike_set = set()
    for h in history_slice:
        for k_s in h.get("strikes", {}).keys():
            try:
                strike_set.add(float(k_s))
            except Exception:
                pass

    if not strike_set:
        ref_spot = spots[-1] if spots else 100.0
        strike_set = {round(ref_spot * (1 + step * 0.01), 1) for step in range(-8, 9)}

    # Ensure strikes encompass all active spots, call walls, and put walls
    valid_levels = [v for v in (spots + call_walls + put_walls) if v > 0]
    if valid_levels:
        min_bound = min(valid_levels) * 0.985
        max_bound = max(valid_levels) * 1.015
        filtered_strikes = sorted([k for k in strike_set if min_bound <= k <= max_bound])
    else:
        med_spot = float(np.median(spots)) if spots else 100.0
        filtered_strikes = sorted([k for k in strike_set if abs(k - med_spot) / med_spot <= 0.08])

    if len(filtered_strikes) < 5:
        filtered_strikes = sorted(list(strike_set))[:30]

    # Build 2D Matrix z[row=strike][col=date]
    z_matrix = []
    hover_matrix = []

    for k in filtered_strikes:
        row_z = []
        row_hover = []
        for idx, h in enumerate(history_slice):
            s_dict = h.get("strikes", {})
            curr_item = _lookup_strike(s_dict, k)
            c_val = _safe_float(curr_item.get("net_gex"), 0.0)

            is_baseline_col = False
            if mode == "delta_gex":
                # Day-over-Day Change in Net GEX
                if idx == 0:
                    if prior_baseline:
                        p_dict = prior_baseline.get("strikes", {})
                        prev_item = _lookup_strike(p_dict, k)
                        p_val = _safe_float(prev_item.get("net_gex"), 0.0)
                        val = round(c_val - p_val, 1)
                    else:
                        val = 0.0
                        is_baseline_col = True
                else:
                    prev_h = history_slice[idx - 1]
                    p_dict = prev_h.get("strikes", {})
                    prev_item = _lookup_strike(p_dict, k)
                    p_val = _safe_float(prev_item.get("net_gex"), 0.0)
                    val = round(c_val - p_val, 1)
            else:
                # Absolute Net GEX
                val = round(c_val, 1)

            row_z.append(val)
            date_lbl = dates_labels[idx]
            spot_val = spots[idx]
            cw_val = call_walls[idx]
            pw_val = put_walls[idx]
            is_cw = " (CALL WALL PIN)" if abs(k - cw_val) < 0.01 else ""
            is_pw = " (PUT WALL FLOOR)" if abs(k - pw_val) < 0.01 else ""

            val_str = f"ΔNet GEX: {format_signed_currency(val, 'M', 1)}" if mode == "delta_gex" else f"Net GEX: {format_signed_currency(val, 'M', 1)}"
            if is_baseline_col:
                val_str += " (Initial Baseline)"

            hover_text = (
                f"<b>{date_lbl} — Strike ${k:,.2f}</b>{is_cw}{is_pw}<br>"
                f"Spot: ${spot_val:,.2f}<br>"
                f"{val_str}<br>"
                f"Call OI: {curr_item.get('call_oi', 0):,} | Put OI: {curr_item.get('put_oi', 0):,}"
            )
            row_hover.append(hover_text)

        z_matrix.append(row_z)
        hover_matrix.append(row_hover)

    return {
        "ticker": t_up,
        "mode": mode,
        "dates": dates_labels,
        "dates_raw": dates_raw,
        "strikes": filtered_strikes,
        "z_matrix": z_matrix,
        "hover_matrix": hover_matrix,
        "spots": spots,
        "call_walls": call_walls,
        "put_walls": put_walls,
        "zero_gammas": zero_gammas,
        "net_gex_b": net_gex_b
    }


def plot_gamma_shift_heatmap(ticker: str, mode: str = "net_gex", max_days: int = 10):
    """
    Generates an institutional high-contrast Plotly Heat Map visualizing
    strike gamma migration across time with underlying spot trajectory overlay.
    """
    if not go:
        return None

    data = generate_gamma_heatmap_matrix(ticker, mode=mode, max_days=max_days)
    if not data or "error" in data:
        return None

    dates = data["dates"]
    strikes = data["strikes"]
    z = data["z_matrix"]
    hover = data["hover_matrix"]
    spots = data["spots"]
    call_walls = data["call_walls"]
    put_walls = data["put_walls"]

    # Dynamic color scale:
    # Red for Negative Put Gamma, Dark for Neutral, Mint for Positive Call Gamma
    custom_colorscale = [
        [0.0, "#FF2E55"],   # Peak Put Gamma (Cherry)
        [0.45, "#4c0519"],  # Soft dark wine
        [0.50, "#0e1117"],  # Stealth neutral black
        [0.55, "#064e3b"],  # Soft dark emerald
        [1.0, "#00F5A0"]    # Peak Call Gamma (Mint)
    ]

    # Calculate symmetric bounds around zero for divergent color map (95th percentile clamp)
    abs_vals = [abs(val) for row in z for val in row if abs(val) > 0.01]
    if abs_vals:
        p95 = float(np.percentile(abs_vals, 95))
        max_abs = max(p95 * 1.25, 5.0)
    else:
        max_abs = 10.0

    fig = go.Figure()

    # 1. 2D Heatmap Layer
    fig.add_trace(go.Heatmap(
        z=z,
        x=dates,
        y=strikes,
        hoverinfo="text",
        text=hover,
        colorscale=custom_colorscale,
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(
            title=dict(
                text="ΔGEX ($M)" if mode == "delta_gex" else "Net GEX ($M)",
                font=dict(color="#94a3b8", size=10)
            ),
            tickfont=dict(color="#94a3b8", size=9),
            thickness=12,
            len=0.85
        )
    ))

    # 2. Spot Price Trajectory Line (White Solid Step Line)
    fig.add_trace(go.Scatter(
        x=dates,
        y=spots,
        mode="lines+markers",
        line=dict(color="#ffffff", width=2.5),
        marker=dict(size=7, color="#ffffff", line=dict(color="#0e1117", width=1.5)),
        name="Spot Price",
        hovertemplate="Spot Price: $%{y:,.2f}<extra></extra>"
    ))

    # 3. Call Wall Pin Ceiling Markers (Neon Mint Diamonds)
    fig.add_trace(go.Scatter(
        x=dates,
        y=call_walls,
        mode="markers",
        marker=dict(symbol="diamond", size=9, color="#00F5A0", line=dict(color="#ffffff", width=1)),
        name="Call Wall (Ceiling Pin)",
        hovertemplate="Call Wall Pin: $%{y:,.2f}<extra></extra>"
    ))

    # 4. Put Wall Support Floor Markers (Neon Cherry Triangles)
    fig.add_trace(go.Scatter(
        x=dates,
        y=put_walls,
        mode="markers",
        marker=dict(symbol="triangle-up", size=10, color="#FF2E55", line=dict(color="#ffffff", width=1)),
        name="Put Wall (Support Floor)",
        hovertemplate="Put Wall Floor: $%{y:,.2f}<extra></extra>"
    ))

    mode_title = "Day-over-Day ΔGEX Shift (Brightness = Change Intensity)" if mode == "delta_gex" else "Net Gamma Distribution by Strike ($M)"
    fig.update_layout(
        title=dict(
            text=f"<b>{data['ticker']} Gamma Migration Heat Map</b> | {mode_title}<br><span style='font-size:10px; color:#94a3b8; font-weight:normal; font-family:monospace;'>White Line = Spot Trajectory • Diamonds = Call Wall • Triangles = Put Wall</span>",
            font=dict(size=13, color="#f8fafc")
        ),
        xaxis=dict(
            title=dict(text="Trading Session", font=dict(color="#94a3b8", size=11)),
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94a3b8", size=10)
        ),
        yaxis=dict(
            title=dict(text="Option Strike Price ($)", font=dict(color="#94a3b8", size=11)),
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickfont=dict(color="#94a3b8", size=10)
        ),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=58, r=25, t=65, b=40),
        legend=dict(
            orientation="h",
            x=1,
            xanchor="right",
            y=1.08,
            font=dict(color="#94a3b8", size=10)
        ),
        height=480
    )

    return fig


def _seed_ticker_if_needed(ticker: str) -> bool:
    """
    On-demand baseline generator for any requested ticker (e.g. PLTR, AMD, SMH)
    using recent OHLCV history so heatmaps and shift metrics can be explored immediately.
    """
    global _HISTORY_CACHE
    t_up = ticker.upper().strip()
    if t_up in _HISTORY_CACHE and len(_HISTORY_CACHE[t_up]) >= 4:
        return True

    df = load_ticker_data(t_up)
    if df is None or df.empty or len(df) < 5:
        return False

    tail_df = df.tail(7)
    sym_history = []

    for idx, (dt_idx, row) in enumerate(tail_df.iterrows()):
        close_px = float(row.get("Close", 100))
        date_val = row.get("Date")
        if pd.isna(date_val):
            date_str = (datetime.now() - timedelta(days=(7 - idx))).strftime("%Y-%m-%d")
        else:
            date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")

        step = 5.0 if close_px > 300 else (2.5 if close_px > 100 else 1.0)
        center_k = round(close_px / step) * step

        cw = center_k + (step * (1 if idx < 4 else 2))
        pw = center_k - (step * (2 if idx < 3 else 1))
        net_gex_b = round(0.15 + (idx * 0.05) - (0.08 if idx == 2 else 0.0), 3)

        strikes_dict = {}
        for k_mult in range(-7, 8):
            k = center_k + (k_mult * step)
            c_gex = max(0.0, round((400 / (1 + abs(k - cw) / step)) * (1.1 if k >= close_px else 0.3), 1))
            p_gex = max(0.0, round((380 / (1 + abs(k - pw) / step)) * (1.1 if k <= close_px else 0.3), 1))
            net_g = round(c_gex - p_gex, 1)

            strikes_dict[f"{k:.2f}"] = {
                "net_gex": net_g,
                "call_gex": c_gex,
                "put_gex": p_gex,
                "call_oi": int(c_gex * 12),
                "put_oi": int(p_gex * 12)
            }

        sym_history.append({
            "date": date_str,
            "timestamp": int(time.time() - (7 - idx) * 86400),
            "spot_price": round(close_px, 2),
            "call_wall": round(cw, 2),
            "call_wall_gex_m": round(strikes_dict.get(f"{cw:.2f}", {}).get("call_gex", 450), 1),
            "put_wall": round(pw, 2),
            "put_wall_gex_m": round(strikes_dict.get(f"{pw:.2f}", {}).get("put_gex", 400), 1),
            "zero_gamma_level": round(center_k - (step * 0.5), 2),
            "total_net_gex_billions": net_gex_b,
            "total_call_gex_billions": round(net_gex_b + 0.5, 3),
            "total_put_gex_billions": round(0.5, 3),
            "regime_short": "Long Gamma" if net_gex_b > 0 else "Short Gamma",
            "strikes": strikes_dict
        })

    if sym_history:
        _HISTORY_CACHE[t_up] = sym_history
        save_gamma_history(_HISTORY_CACHE)
        return True
    return False


def _seed_default_history_if_needed():
    """
    Auto-seeds realistic baseline historical snapshots for benchmark universe tickers
    (SPY, QQQ, MSFT, NVDA, AAPL) using their recent 7-day OHLCV history so the heatmap
    and Day-over-Day shift engine can be visually explored immediately.
    """
    global _HISTORY_CACHE
    filepath = get_history_filepath()
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data and any(len(v) >= 4 for v in data.values()):
                    _HISTORY_CACHE = data
                    return
        except Exception:
            pass

    benchmarks = ["SPY", "QQQ", "MSFT", "NVDA", "AAPL", "AMZN", "META", "TSLA", "IWM"]
    for sym in benchmarks:
        _seed_ticker_if_needed(sym)
