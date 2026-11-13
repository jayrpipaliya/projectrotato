"""
Institutional Gamma Tracker & Strike Migration Heat Map Engine
- Records daily dealer Gamma Exposure (GEX) snapshots to track historical strike positioning.
- Calculates Day-over-Day (DoD) Wall Migration (Call Wall rolls, Put Wall retreats, Zero-Gamma shifts).
- Computes Net GEX Velocity (ΔGEX) to detect volatility regime transitions before price breaks.
- Generates 2D Strike-versus-Time Heat Maps with Underlying Spot Trajectory overlays.
"""

import os
import json
import math
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

def get_market_session_date(override_dt: datetime = None) -> str:
    """
    Returns the official US Equity Options Trading Session Date (YYYY-MM-DD)
    strictly anchored to America/New_York (US Eastern Time).

    Trading Session Alignment:
    - US Equity & Options trade 9:30 AM to 4:00 PM EST (indexes to 4:15 PM EST).
    - If current NY time is before 9:30 AM EST (overnight or pre-market):
      Options market has not opened yet; chains reflect the PREVIOUS business day's settlement.
      (e.g., at 2:30 AM EST Wednesday, the data is Tuesday's close).
      If Monday morning before 9:30 AM EST, rolls back to Friday.
    - If current NY time is Saturday or Sunday:
      Rolls back to Friday's settlement session.
    - If current NY time is >= 9:30 AM EST on a weekday:
      The session is today's date.
    - Cloud Server Guard: Immune to UTC clock drift (prevents 9:30 PM EST from becoming tomorrow UTC).
    """
    if override_dt:
        now_ny = override_dt
    else:
        try:
            from zoneinfo import ZoneInfo
            now_ny = datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            now_ny = datetime.utcnow() - timedelta(hours=5)

    weekday = now_ny.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday

    if weekday == 5:  # Saturday -> Friday
        session_dt = now_ny - timedelta(days=1)
    elif weekday == 6:  # Sunday -> Friday
        session_dt = now_ny - timedelta(days=2)
    elif now_ny.hour < 9 or (now_ny.hour == 9 and now_ny.minute < 30):
        # Before 9:30 AM EST on a weekday
        if weekday == 0:  # Monday morning -> Friday
            session_dt = now_ny - timedelta(days=3)
        else:
            session_dt = now_ny - timedelta(days=1)
    else:
        session_dt = now_ny

    return session_dt.strftime("%Y-%m-%d")


def _lookup_strike(s_dict: dict, k: float, interpolate: bool = True) -> dict:
    """
    Robust strike lookup with float tolerance and linear neighbor interpolation.
    Eliminates phantom ΔGEX spikes caused by strike grid resolution differences.
    """
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

    if not interpolate:
        return {}

    # 3. Dynamic linear interpolation between nearest neighboring strikes
    try:
        float_keys = []
        for sk in s_dict.keys():
            try:
                float_keys.append((float(sk), sk))
            except Exception:
                pass
        if len(float_keys) < 2:
            return {}
        float_keys.sort(key=lambda x: x[0])

        low_cand = [item for item in float_keys if item[0] < k]
        high_cand = [item for item in float_keys if item[0] > k]

        if low_cand and high_cand:
            k_low, key_low = low_cand[-1]
            k_high, key_high = high_cand[0]
            span = k_high - k_low
            # Only interpolate within a reasonable strike spread (<= 10.0 points)
            if 0 < span <= 10.0:
                weight = (k - k_low) / span
                low_val = s_dict[key_low]
                high_val = s_dict[key_high]

                net_l = _safe_float(low_val.get("net_gex"), 0.0)
                net_h = _safe_float(high_val.get("net_gex"), 0.0)
                call_l = _safe_float(low_val.get("call_gex"), 0.0)
                call_h = _safe_float(high_val.get("call_gex"), 0.0)
                put_l = _safe_float(low_val.get("put_gex"), 0.0)
                put_h = _safe_float(high_val.get("put_gex"), 0.0)
                coi_l = _safe_float(low_val.get("call_oi"), 0.0)
                coi_h = _safe_float(high_val.get("call_oi"), 0.0)
                poi_l = _safe_float(low_val.get("put_oi"), 0.0)
                poi_h = _safe_float(high_val.get("put_oi"), 0.0)

                return {
                    "net_gex": round(net_l + weight * (net_h - net_l), 1),
                    "call_gex": round(call_l + weight * (call_h - call_l), 1),
                    "put_gex": round(put_l + weight * (put_h - put_l), 1),
                    "call_oi": int(round(coi_l + weight * (coi_h - coi_l))),
                    "put_oi": int(round(poi_l + weight * (poi_h - poi_l))),
                    "interpolated": True
                }
    except Exception:
        pass

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

        # Deduplicate strictly by date, clamp any future-stamped dates from UTC drift, and sort
        max_valid_date = get_market_session_date()
        deduped = {}
        for item in raw_list:
            d = item.get("date")
            if d:
                # Clamp future-stamped dates to current valid market session date
                clean_d = min(d, max_valid_date) if d > max_valid_date else d
                item_copy = dict(item)
                item_copy["date"] = clean_d
                deduped[clean_d] = item_copy
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

    # Session date string YYYY-MM-DD anchored to US Eastern market calendar
    today_str = get_market_session_date()

    # Extract strike GEX dictionary (relevant strikes and active OI within +/- 18% moneyness)
    strikes_compact = {}
    spot_ref = _safe_float(profile.get("spot_price"), 100.0)
    for s in profile.get("strikes", []):
        k_val = _safe_float(s.get("strike"), 0.0)
        c_oi = int(_safe_float(s.get("call_oi"), 0))
        p_oi = int(_safe_float(s.get("put_oi"), 0))
        dist_ratio = abs(k_val - spot_ref) / spot_ref if spot_ref > 0 else 0.0

        if s.get("is_relevant", True) or dist_ratio <= 0.18 or (c_oi + p_oi >= 500):
            k_str = f"{k_val:.2f}"
            strikes_compact[k_str] = {
                "net_gex": round(s.get("net_gex", 0), 1),
                "call_gex": round(s.get("call_gex", 0), 1),
                "put_gex": round(s.get("put_gex", 0), 1),
                "call_oi": c_oi,
                "put_oi": p_oi
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
        "current_iv": profile.get("current_iv"),
        "iv_rank": profile.get("iv_rank"),
        "iv_percentile": profile.get("iv_percentile"),
        "vol_regime": profile.get("vol_regime"),
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
        "current_iv": curr.get("current_iv"),
        "iv_rank": curr.get("iv_rank"),
        "iv_percentile": curr.get("iv_percentile"),
        "vol_regime": curr.get("vol_regime"),
        "history_count": len(history)
    }


def compute_advanced_gamma_analytics(ticker: str) -> dict:
    """
    Computes institutional micro-structural gamma analytics beyond simple static walls:
    1. Local Pinning Density (Pin Index around spot: whether dealers are trapping price in a volatility black hole or if price is slippery)
    2. Air Pockets / Gamma Voids (low-GEX corridors above/below walls where moves accelerate violently with no dealer resistance)
    3. Top Overnight Accumulation & Hedging Vectors (exact strikes where fresh Call OI / Put Hedging entered overnight in the ΔGEX matrix)
    4. Gamma Skew Asymmetry & Convexity Slopes (ratio of upside to downside gamma resistance)
    5. Net GEX Magnitude & Regime Classification
    """
    shift = compute_gamma_shift(ticker)
    if not shift or not shift.get("has_history"):
        return {}

    t_up = ticker.upper().strip()
    history = load_gamma_history(t_up)
    if not history:
        return {}

    curr = history[-1]
    spot = _safe_float(curr.get("spot_price"), 100.0)
    cw = _safe_float(curr.get("call_wall"), spot)
    pw = _safe_float(curr.get("put_wall"), spot)
    zg = _safe_float(curr.get("zero_gamma_level"), spot)
    net_gex_b = _safe_float(curr.get("total_net_gex_billions"), 0.0)
    vel_b = _safe_float(shift.get("net_gex_velocity_b"), 0.0)

    strikes = curr.get("strikes", {})
    all_shifts = shift.get("all_strike_shifts", [])

    # 1. Local Pinning Density (ATM +/- 1.2% moneyness band)
    atm_band = 0.012
    atm_abs_gex = 0.0
    total_abs_gex = 0.0

    # 2. Asymmetric Skew (+2.5% above spot vs -2.5% below spot)
    gex_upside = 0.0
    gex_downside = 0.0

    for k_s, v in strikes.items():
        k = _safe_float(k_s, spot)
        net_g = _safe_float(v.get("net_gex"), 0.0)
        abs_g = abs(net_g)
        total_abs_gex += abs_g

        if abs(k - spot) / spot <= atm_band:
            atm_abs_gex += abs_g

        if 0 < (k - spot) / spot <= 0.025:
            gex_upside += net_g
        elif -0.025 <= (k - spot) / spot < 0:
            gex_downside += net_g

    pinning_ratio = round(atm_abs_gex / total_abs_gex, 2) if total_abs_gex > 0 else 0.0
    if pinning_ratio >= 0.40:
        pin_label = "HIGH_PIN_DENSITY"
        pin_desc = f"{int(pinning_ratio*100)}% of dealer inventory clustered within +/-1.2% of spot (${spot:,.1f}). Heavy volatility dampening; price behaves like a mean-reverting pin."
    elif pinning_ratio <= 0.20:
        pin_label = "DIFFUSE_UNPINNED"
        pin_desc = f"Only {int(pinning_ratio*100)}% of gamma near spot. Dealer pinning is weak; expect wider directional excursions."
    else:
        pin_label = "MODERATE_PIN"
        pin_desc = f"{int(pinning_ratio*100)}% gamma near spot. Moderate dealer constraint with standard mean-reversion tendencies."

    # 3. Air Pocket / Vacuum Detector
    call_strikes_above = [(float(k), _safe_float(v.get("call_gex"), 0.0)) for k, v in strikes.items() if float(k) >= spot]
    call_strikes_above.sort(key=lambda x: x[0])
    peak_call_gex = max([x[1] for x in call_strikes_above], default=1.0)

    upside_air_pocket = None
    for k, cg in call_strikes_above:
        if k > cw and cg < (peak_call_gex * 0.15):
            upside_air_pocket = k
            break
    if not upside_air_pocket and call_strikes_above:
        upside_air_pocket = call_strikes_above[-1][0]

    put_strikes_below = [(float(k), _safe_float(v.get("put_gex"), 0.0)) for k, v in strikes.items() if float(k) <= spot]
    put_strikes_below.sort(key=lambda x: x[0], reverse=True)
    peak_put_gex = max([x[1] for x in put_strikes_below], default=1.0)

    downside_air_pocket = None
    for k, pg in put_strikes_below:
        if k < pw and pg < (peak_put_gex * 0.15):
            downside_air_pocket = k
            break
    if not downside_air_pocket and put_strikes_below:
        downside_air_pocket = put_strikes_below[-1][0]

    # 4. Top Delta GEX Shifts (Overnight flow vectors)
    top_calls_accum = shift.get("top_calls_accumulating", [])[:2]
    top_puts_hedged = shift.get("top_puts_hedging", [])[:2]
    top_drained = sorted(all_shifts, key=lambda x: x.get("delta_net_gex_m", 0.0))[:2]

    # 5. Skew Asymmetry
    skew_ratio = round(gex_upside / abs(gex_downside), 2) if abs(gex_downside) > 0 else 1.0
    if skew_ratio >= 1.35:
        skew_label = "CALL_HEAVY_RESISTANCE"
        skew_desc = f"Call gamma outweighs put gamma by {skew_ratio:.1f}x. Rallies face heavy dealer supply, but downside has thinner cushioning."
    elif skew_ratio <= 0.65:
        skew_label = "PUT_HEAVY_SUPPORT"
        skew_desc = f"Put cushion outweighs overhead calls ({skew_ratio:.1f}x ratio). Dips are aggressively bought by dealers; upside room is cleaner."
    else:
        skew_label = "BALANCED_GAMMA_SLOPE"
        skew_desc = f"Dealer gamma is symmetrically distributed ({skew_ratio:.1f}x ratio) around spot."

    # 6. Absolute Net GEX Regime
    if net_gex_b >= 2.5:
        regime_desc = f"+${net_gex_b:.2f}B Deep Long Gamma — Heavy dealer volatility dampening (buy dips & fade rips)."
    elif net_gex_b >= 0.5:
        regime_desc = f"+${net_gex_b:.2f}B Long Gamma — Moderated range mean-reversion."
    elif net_gex_b <= -2.5:
        regime_desc = f"-${abs(net_gex_b):.2f}B Deep Short Gamma — Severe tail fragility & cascade risk."
    elif net_gex_b <= -0.5:
        regime_desc = f"-${abs(net_gex_b):.2f}B Short Gamma — Volatility expansion & directional momentum."
    else:
        regime_desc = f"{format_signed_currency(net_gex_b, 'B', 2)} Neutral Gamma — Minimal dealer delta-hedging constraint."

    return {
        "ticker": t_up,
        "spot": spot,
        "net_gex_billions": net_gex_b,
        "net_gex_regime_desc": regime_desc,
        "net_gex_velocity_b": format_signed_currency(vel_b, "B", 2),
        "velocity_label": shift.get("velocity_label"),
        "call_wall": cw,
        "call_wall_shift": f"{shift.get('call_wall_shift', 0.0):+.2f}",
        "call_wall_status": shift.get("call_wall_status"),
        "put_wall": pw,
        "put_wall_shift": f"{shift.get('put_wall_shift', 0.0):+.2f}",
        "put_wall_status": shift.get("put_wall_status"),
        "zero_gamma_level": zg,
        "pinning_index": {
            "ratio": pinning_ratio,
            "label": pin_label,
            "description": pin_desc
        },
        "gamma_voids_air_pockets": {
            "upside_vacuum_strike": upside_air_pocket,
            "upside_vacuum_desc": f"Above ${upside_air_pocket:,.1f}, dealer call resistance drops below 15% of peak. Breakouts past ${cw:,.1f} enter a gamma vacuum with explosive expansion risk.",
            "downside_vacuum_strike": downside_air_pocket,
            "downside_vacuum_desc": f"Below ${downside_air_pocket:,.1f}, put support floor vanishes. Cascades below ${pw:,.1f} enter a gamma vacuum."
        },
        "overnight_flow_vectors": {
            "top_call_accumulation": [f"${x['strike']:,.1f} (+{x['delta_call_gex_m']:,.1f}M)" for x in top_calls_accum],
            "top_put_hedging": [f"${x['strike']:,.1f} (+{x['delta_put_gex_m']:,.1f}M)" for x in top_puts_hedged],
            "top_gamma_drain": [f"${x['strike']:,.1f} ({x['delta_net_gex_m']:,.1f}M)" for x in top_drained]
        },
        "convexity_skew": {
            "ratio": skew_ratio,
            "label": skew_label,
            "description": skew_desc
        },
        "institutional_synthesis": shift.get("institutional_synthesis")
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


def get_historical_oi_migration(ticker: str, max_days: int = 5) -> dict:
    """
    Constructs a multi-day strike-by-strike Open Interest (OI) historical migration matrix:
    - Tracks date-by-date Call OI & Put OI across consecutive trading sessions.
    - Computes 1-Day (DoD) ΔOI and multi-day Net Accumulation.
    - Tags institutional structural roles (Call Wall, Put Wall, Zero Flip, ATM, and OTM Target Base).
    - Categorizes accumulation velocity (e.g. Heavy Base Building, Steady Accumulation, Unwinding).
    """
    t_up = ticker.upper().strip()
    history = load_gamma_history(t_up)
    if not history or len(history) < 1:
        return {
            "ticker": t_up,
            "has_data": False,
            "message": f"Insufficient historical GEX snapshots for {t_up}."
        }

    history_slice = history[-max_days:]
    dates_raw = [h.get("date", "") for h in history_slice]
    dates_labels = []
    for d_str in dates_raw:
        try:
            dates_labels.append(datetime.strptime(d_str, "%Y-%m-%d").strftime("%b %d"))
        except Exception:
            dates_labels.append(d_str)

    curr = history_slice[-1]
    spot = _safe_float(curr.get("spot_price"), 100.0)
    cw = _safe_float(curr.get("call_wall"), 0.0)
    pw = _safe_float(curr.get("put_wall"), 0.0)
    zg = _safe_float(curr.get("zero_gamma_level"), 0.0)

    # Gather union of all strikes present in the history slice and per-snapshot strike boundaries
    strike_set = set()
    snapshot_ranges = []
    for h in history_slice:
        s_keys = []
        for k_s in h.get("strikes", {}).keys():
            try:
                k_val = float(k_s)
                strike_set.add(k_val)
                s_keys.append(k_val)
            except Exception:
                continue
        if s_keys:
            snapshot_ranges.append((min(s_keys), max(s_keys)))
        else:
            snapshot_ranges.append((0.0, 0.0))

    sorted_strikes = sorted(list(strike_set))
    strike_rows = []

    for k in sorted_strikes:
        call_series = []
        put_series = []

        for idx, h in enumerate(history_slice):
            s_dict = h.get("strikes", {})
            min_k, max_k = snapshot_ranges[idx] if idx < len(snapshot_ranges) else (0.0, 0.0)

            # Check if strike was tracked in this session snapshot
            if min_k > 0 and (k < min_k - 0.05 or k > max_k + 0.05):
                # Outside the recorded grid on this date -> untracked
                call_series.append(None)
                put_series.append(None)
            else:
                item = _lookup_strike(s_dict, k, interpolate=False)
                c_val = int(_safe_float(item.get("call_oi"), 0)) if item else 0
                p_val = int(_safe_float(item.get("put_oi"), 0)) if item else 0
                call_series.append(c_val)
                put_series.append(p_val)

        valid_c = [v for v in call_series if v is not None]
        valid_p = [v for v in put_series if v is not None]

        c_latest = valid_c[-1] if valid_c else 0
        p_latest = valid_p[-1] if valid_p else 0

        # 1-Day change: compare latest to previous valid value
        if len(valid_c) >= 2 and call_series[-1] is not None and call_series[-2] is not None:
            c_1d = call_series[-1] - call_series[-2]
        elif len(valid_c) >= 2:
            c_1d = valid_c[-1] - valid_c[-2]
        else:
            c_1d = 0

        if len(valid_p) >= 2 and put_series[-1] is not None and put_series[-2] is not None:
            p_1d = put_series[-1] - put_series[-2]
        elif len(valid_p) >= 2:
            p_1d = valid_p[-1] - valid_p[-2]
        else:
            p_1d = 0

        # Total multi-day change: calculated from the FIRST tracked valid session
        c_baseline = valid_c[0] if valid_c else 0
        p_baseline = valid_p[0] if valid_p else 0

        c_tot = c_latest - c_baseline
        p_tot = p_latest - p_baseline

        c_pct = round((c_tot / c_baseline * 100.0), 1) if c_baseline > 0 else (100.0 if c_tot > 0 else 0.0)
        p_pct = round((p_tot / p_baseline * 100.0), 1) if p_baseline > 0 else (100.0 if p_tot > 0 else 0.0)

        # Dist vs spot
        dist_pct = round(((k - spot) / spot) * 100.0, 2)

        # Role tagging
        role = ""
        role_type = "normal"
        if cw > 0 and abs(k - cw) < 0.25:
            role = "Call Wall Pin"
            role_type = "call_wall"
        elif pw > 0 and abs(k - pw) < 0.25:
            role = "Put Wall Floor"
            role_type = "put_wall"
        elif zg > 0 and abs(k - zg) < 0.45:
            role = "Zero Vol Flip"
            role_type = "zero_flip"
        elif abs(k - spot) / spot < 0.0035:
            role = "ATM Straddle"
            role_type = "atm"
        elif k >= spot * 1.012 and c_latest >= 10000 and (c_tot >= 1500 or c_1d >= 1000):
            role = "🎯 Target Base"
            role_type = "target_base"

        # Pace classification
        if c_tot >= 5000 or c_1d >= 3000:
            call_pace = "🔥 Heavy Base Accumulation"
        elif c_tot >= 1000 or c_1d >= 800:
            call_pace = "🟢 Steady Accumulation"
        elif c_tot <= -3000 or c_1d <= -2000:
            call_pace = "🔴 Active Unwinding"
        elif c_tot <= -800:
            call_pace = "Muted Reduction"
        else:
            call_pace = "Stable"

        if p_tot >= 5000 or p_1d >= 3000:
            put_pace = "🛡️ Heavy Floor Hedging"
        elif p_tot >= 1000 or p_1d >= 800:
            put_pace = "🟢 Floor Accumulation"
        elif p_tot <= -3000 or p_1d <= -2000:
            put_pace = "🔴 Protective Unwinding"
        elif p_tot <= -800:
            put_pace = "Muted Reduction"
        else:
            put_pace = "Stable"

        # Relevant filter: walls, target base, ATM, within +/- 3.5% of spot, or > 12k OI
        is_rel = (role != "") or (abs(k - spot) / spot <= 0.035) or (max(c_latest, p_latest) >= 12000)

        # Get latest GEX metrics if available
        curr_item = _lookup_strike(curr.get("strikes", {}), k)
        net_gex = round(_safe_float(curr_item.get("net_gex"), 0.0), 1)
        call_gex = round(_safe_float(curr_item.get("call_gex"), 0.0), 1)
        put_gex = round(_safe_float(curr_item.get("put_gex"), 0.0), 1)

        strike_rows.append({
            "strike": k,
            "strike_str": f"${k:,.2f}",
            "dist_pct": dist_pct,
            "role": role,
            "role_type": role_type,
            "is_relevant": is_rel,
            "call_series": call_series,
            "put_series": put_series,
            "call_latest": c_latest,
            "put_latest": p_latest,
            "call_1d": c_1d,
            "put_1d": p_1d,
            "call_tot": c_tot,
            "put_tot": p_tot,
            "call_pct": c_pct,
            "put_pct": p_pct,
            "call_pace": call_pace,
            "put_pace": put_pace,
            "net_gex": net_gex,
            "call_gex": call_gex,
            "put_gex": put_gex
        })

    # Summary lists
    top_calls_accum = sorted(strike_rows, key=lambda x: x["call_tot"], reverse=True)[:5]
    top_puts_accum = sorted(strike_rows, key=lambda x: x["put_tot"], reverse=True)[:5]

    return {
        "ticker": t_up,
        "has_data": True,
        "dates_raw": dates_raw,
        "dates_labels": dates_labels,
        "num_days": len(dates_raw),
        "spot_price": spot,
        "call_wall": cw,
        "put_wall": pw,
        "zero_gamma": zg,
        "strikes": strike_rows,
        "total_strikes": len(strike_rows),
        "relevant_strikes": sum(1 for s in strike_rows if s["is_relevant"]),
        "top_calls_accum": top_calls_accum,
        "top_puts_accum": top_puts_accum
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

    # Institutional high-contrast multi-stop diverging color scale:
    # Tight central neutral deadband ensures subtle day-over-day changes illuminate
    # with soft emerald/wine tones, while major wall rolls peak in vivid neon mint/cherry.
    custom_colorscale = [
        [0.0, "#FF2E55"],    # -100% (Neon Cherry - Extreme Put Gamma / Negative Shift)
        [0.10, "#e11d48"],   # -80%
        [0.25, "#be123c"],   # -50%
        [0.38, "#881337"],   # -24%
        [0.44, "#4c0519"],   # -12%
        [0.47, "#2e0813"],   # -6% (subtle dark wine visible on small shifts)
        [0.49, "#180d15"],   # -2%
        [0.50, "#0b0f19"],   # 0.0 (True Stealth Neutral Background)
        [0.51, "#0d1b18"],   # +2%
        [0.53, "#063d2e"],   # +6% (subtle dark emerald visible on small shifts)
        [0.56, "#047857"],   # +12%
        [0.62, "#059669"],   # +24%
        [0.75, "#10b981"],   # +50%
        [0.90, "#34d399"],   # +80%
        [1.0, "#00F5A0"]     # +100% (Neon Mint - Extreme Call Gamma / Positive Shift)
    ]

    # Calculate symmetric bounds around zero using robust 92nd percentile clamp
    # Prevents isolated single-strike wall pins from compressing the rest of the matrix into darkness.
    abs_vals = [abs(val) for row in z for val in row if abs(val) > 0.05]
    if abs_vals:
        p92 = float(np.percentile(abs_vals, 92))
        max_abs = max(p92 * 1.15, 5.0)
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


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / 2.5066282746310002


def _bs_gamma(S: float, K: float, T: float = 0.06, sigma: float = 0.18) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def _seed_ticker_if_needed(ticker: str) -> bool:
    """
    On-demand baseline generator for any requested ticker (e.g. SPY, QQQ, PLTR, AMD, SMH)
    using recent OHLCV history with continuous Black-Scholes gamma profiles and dense strike
    resolution so heatmaps, wall migrations, and ΔGEX velocity can be explored immediately.
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

    # Step determination matching options market strike standards
    first_px = float(tail_df.iloc[0].get("Close", 100))
    if t_up in ["SPY", "QQQ", "IWM"]:
        step = 1.0
        k_radius = 35
    elif any(t_up == s for s in ["NVDA", "AAPL", "AMD", "PLTR", "AMZN"]):
        step = 1.0 if first_px < 200 else 2.5
        k_radius = 25
    else:
        step = 5.0 if first_px > 400 else (2.5 if first_px > 150 else 1.0)
        k_radius = 20

    # Determine contract volume scale based on asset capitalization
    # Uses realistic call-over-put institutional open interest ratio (~1.35x)
    # ensuring positive long gamma regimes match live engine scale (~$5.5B - $7.0B)
    if t_up in ["SPY", "QQQ"]:
        base_call_vol = 95000.0
        base_put_vol = 70000.0
    elif t_up in ["NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA"]:
        base_call_vol = 35000.0
        base_put_vol = 26000.0
    else:
        base_call_vol = 18000.0
        base_put_vol = 14000.0

    for idx, (dt_idx, row) in enumerate(tail_df.iterrows()):
        close_px = float(row.get("Close", 100))
        date_val = row.get("Date")
        if pd.isna(date_val):
            date_str = (datetime.now() - timedelta(days=(len(tail_df) - idx))).strftime("%Y-%m-%d")
        else:
            date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")

        # Dynamic Call & Put Wall placement aligned to psychological strikes
        wall_step = 5.0 if t_up in ["SPY", "QQQ"] else (step * 2 if step >= 2.5 else step * 5)
        cw = math.ceil((close_px + 1.0) / wall_step) * wall_step
        pw = math.floor((close_px - 1.0) / wall_step) * wall_step
        if cw - pw < wall_step:
            cw += wall_step

        min_k = int(round(close_px - (k_radius * step)))
        max_k = int(round(close_px + (k_radius * step)))

        strikes_dict = {}
        total_call_gex = 0.0
        total_put_gex = 0.0

        for k_int in range(min_k, max_k + 1, int(step) if step >= 1.0 else 1):
            k = float(k_int)
            gamma = _bs_gamma(close_px, k, T=0.06, sigma=0.18)

            dist_cw = abs(k - cw)
            dist_spot = abs(k - close_px)
            dist_pw = abs(k - pw)

            is_round_5 = (k % 5 == 0)
            is_round_10 = (k % 10 == 0)
            round_boost = 1.4 if is_round_10 else (1.2 if is_round_5 else 1.0)

            c_oi_base = base_call_vol * math.exp(-0.5 * (dist_cw / (step * 3.5))**2) * 2.2 + \
                        (base_call_vol * 0.45) * math.exp(-0.5 * (dist_spot / (step * 5.0))**2) + \
                        ((base_call_vol * 0.25) if k >= close_px else (base_call_vol * 0.06))
            c_oi = int(c_oi_base * round_boost)

            p_oi_base = base_put_vol * math.exp(-0.5 * (dist_pw / (step * 3.5))**2) * 2.2 + \
                        (base_put_vol * 0.45) * math.exp(-0.5 * (dist_spot / (step * 5.0))**2) + \
                        ((base_put_vol * 0.25) if k <= close_px else (base_put_vol * 0.06))
            p_oi = int(p_oi_base * round_boost)

            c_gex = round(close_px * c_oi * 100.0 * gamma * 0.01 * close_px / 1e6, 1)
            p_gex = round(close_px * p_oi * 100.0 * gamma * 0.01 * close_px / 1e6, 1)
            net_gex = round(c_gex - p_gex, 1)

            total_call_gex += c_gex
            total_put_gex += p_gex

            strikes_dict[f"{k:.2f}"] = {
                "net_gex": net_gex,
                "call_gex": c_gex,
                "put_gex": p_gex,
                "call_oi": c_oi,
                "put_oi": p_oi
            }

        tot_net_b = round((total_call_gex - total_put_gex) / 1000.0, 2)

        sym_history.append({
            "date": date_str,
            "timestamp": int(time.time() - (len(tail_df) - idx) * 86400),
            "spot_price": round(close_px, 2),
            "call_wall": round(cw, 2),
            "call_wall_gex_m": round(strikes_dict.get(f"{cw:.2f}", {}).get("call_gex", 3000.0), 1),
            "put_wall": round(pw, 2),
            "put_wall_gex_m": round(strikes_dict.get(f"{pw:.2f}", {}).get("put_gex", 2500.0), 1),
            "zero_gamma_level": round(close_px - step * 1.5, 2),
            "total_net_gex_billions": tot_net_b,
            "total_call_gex_billions": round(total_call_gex / 1000.0, 2),
            "total_put_gex_billions": round(total_put_gex / 1000.0, 2),
            "regime_short": "Long Gamma" if tot_net_b > 0 else "Short Gamma",
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
