"""
sol_1min_backtest.py — SOL/USDT backtest using Bitget 1-minute data.

Improvements over the 1H backtest (test_new_symbols.py):
  1. Activation detected to the exact minute within the activation 1H bar
  2. BE move triggered the exact minute price touches 1.618
  3. Outcome resolution uses real 1-min OHLCV — no next-bar-open approximation
  4. Stop checks see the real intrabar sequence of events

Bug fixed vs test_recent.py:
  - test_recent.py uses win R = 1.578 (WRONG — assumes stop at origin, risk=1×rng)
  - Correct: stop = origin - 0.5×rng  →  risk = 1.5×rng  →  win R = 1.618/1.5 - 0.04 = 1.039
  - All R totals in this file use the correct value

Walk-forward:
  - Same 6-month min train / 3-month step structure
  - Jan–Oct 2025 is highlighted as the unseen period (most recent data)
  - Final model trained on everything up to Jan 2025, tested on Jan–Oct 2025

Feature importances printed after training so you can see what the model is learning.
"""
from __future__ import annotations
import sys, io, warnings, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier

from test_new_symbols import add_confluence, FEATURES, CAT_FEATURES, P_THRESHOLD
from highlander import run_indicator, DIR_UP, DIR_DOWN

DATA   = Path("data")
M1_DIR = Path("bitget/SOLUSDT")

# ── Constants ────────────────────────────────────────────────────────────────
ATR_LEN         = 14
COST_R          = 0.04
WIN_R_GROSS     = 1.618 / 1.5          # = 1.07867  (reward/risk at f2618)
WIN_R           = WIN_R_GROSS - COST_R  # = 1.03867
BE_R            = -COST_R              # = -0.04
STOP_R          = -1.0 - COST_R       # = -1.04
MAX_ACT_H       = 50                   # hours before signal expires
MAX_TRADE_H     = 200                  # hours before trade times out
MIN_TRAIN_MONTHS= 6
STEP_MONTHS     = 3
cat_idx         = [FEATURES.index(c) for c in CAT_FEATURES]

FEATURES_EXT = FEATURES   # FEATURES now contains all 54 features
cat_idx_ext  = cat_idx

UNSEEN_START = pd.Timestamp("2025-01-01")


FILTER_DIST_ATR = 3.0   # Pine Script checkBreakRetestFilter distance
BLOCK_DIST_ATR  = 3.0   # Pine Script v4 opposing-level block distance


# ── Helpers ──────────────────────────────────────────────────────────────────
def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)


def session_of(h):
    if 0  <= h < 7:  return "Asia"
    if 7  <= h < 13: return "London"
    if 13 <= h < 17: return "NY-AM"
    if 17 <= h < 22: return "NY-PM"
    return "Off-hours"


# ── Load 1-minute data ───────────────────────────────────────────────────────
def load_1min():
    cache = DATA / "solusdt_1m_bitget.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True).astype(float)
        print(f"  1-min: {len(df):,} bars (cached)  "
              f"{df.index[0].date()} → {df.index[-1].date()}")
        return df

    print("  Loading 1-min Bitget data …", end="", flush=True)
    dfs = []
    for year_dir in sorted(M1_DIR.iterdir()):
        if not year_dir.is_dir(): continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir(): continue
            for day_file in sorted(month_dir.iterdir()):
                if day_file.suffix != ".csv": continue
                try:
                    d = pd.read_csv(day_file,
                                    names=["ts","open","high","low","close","volume"],
                                    header=0)
                    dfs.append(d)
                except Exception:
                    pass
        print(".", end="", flush=True)

    m1 = pd.concat(dfs, ignore_index=True)
    m1["time"] = pd.to_datetime(m1["ts"], unit="ms")
    m1 = (m1.drop_duplicates("time")
            .sort_values("time")
            .set_index("time")
            [["open","high","low","close","volume"]]
            .astype(float))
    m1.index.name = "time"
    m1.to_csv(cache)
    print(f"\n  {len(m1):,} bars  {m1.index[0].date()} → {m1.index[-1].date()}")
    return m1


# ── 1-minute outcome resolver ─────────────────────────────────────────────────
def resolve_outcome(m1_vals, m1_ts, start_idx, entry, stop, f1618, f2618, is_bull):
    """
    Scan 1-minute bars from start_idx.
    Returns (outcome_str, be_hit_bool, close_idx) or (None, False, end_idx) on timeout.
    close_idx is the 1-min bar index where the trade closed.
    """
    end_idx = min(start_idx + MAX_TRADE_H * 60, len(m1_vals))
    be_hit  = False
    cur_stop = stop

    for i in range(start_idx, end_idx):
        h = m1_vals[i, 1]
        l = m1_vals[i, 2]

        if is_bull:
            hit_tp   = h >= f2618
            hit_be   = h >= f1618
            hit_stop = l <= cur_stop

            if hit_tp:
                if hit_stop:
                    if i + 1 < len(m1_vals):
                        o_next = m1_vals[i+1, 0]
                        return ("win" if abs(o_next - f2618) < abs(o_next - cur_stop)
                                else ("be_stop" if be_hit else "full_stop")), be_hit, i
                    return "win", be_hit, i
                return "win", be_hit, i

            if hit_be and not be_hit:
                if hit_stop:
                    if i + 1 < len(m1_vals):
                        o_next = m1_vals[i+1, 0]
                        if o_next > entry:
                            be_hit   = True
                            cur_stop = entry
                            continue
                        else:
                            return "full_stop", False, i
                    return "full_stop", False, i
                be_hit   = True
                cur_stop = entry
                continue

            if hit_stop:
                return ("be_stop" if be_hit else "full_stop"), be_hit, i

        else:
            hit_tp   = l <= f2618
            hit_be   = l <= f1618
            hit_stop = h >= cur_stop

            if hit_tp:
                if hit_stop:
                    if i + 1 < len(m1_vals):
                        o_next = m1_vals[i+1, 0]
                        return ("win" if abs(o_next - f2618) < abs(o_next - cur_stop)
                                else ("be_stop" if be_hit else "full_stop")), be_hit, i
                    return "win", be_hit, i
                return "win", be_hit, i

            if hit_be and not be_hit:
                if hit_stop:
                    if i + 1 < len(m1_vals):
                        o_next = m1_vals[i+1, 0]
                        if o_next < entry:
                            be_hit   = True
                            cur_stop = entry
                            continue
                        else:
                            return "full_stop", False, i
                    return "full_stop", False, i
                be_hit   = True
                cur_stop = entry
                continue

            if hit_stop:
                return ("be_stop" if be_hit else "full_stop"), be_hit, i

    return None, be_hit, end_idx - 1


# ── Highlander filter helpers ─────────────────────────────────────────────────
def _hl_state_at(lhist: list, target_bar: int) -> dict | None:
    """Return level state dict at target_bar, or None if level didn't exist yet."""
    if not lhist or lhist[0]["bar"] > target_bar:
        return None
    state = -1; alive = True
    for ev in lhist:
        if ev["bar"] > target_bar:
            break
        e = ev["event"]
        if   e == "level_created":    state = 0
        elif e == "first_touch":      state = 3
        elif e == "origin_confirmed": state = 1
        elif e == "level_broken":     state = 2
        elif e == "level_deleted":    alive = False
    return {"state": state, "alive": alive,
            "direction": lhist[0]["direction"], "price": lhist[0]["price"]}


def _check_break_retest(is_bull: bool, ref_price: float, atr: float,
                        grouped: dict, bar_i: int) -> bool:
    """Replicates Pine Script checkBreakRetestFilter.
    Returns True (passes) if the closest level on the trade side is a broken
    opposite-direction level within FILTER_DIST_ATR ATRs."""
    if atr <= 0:
        return False
    best_dist = float("inf"); best_dir = None
    for lhist in grouped.values():
        s = _hl_state_at(lhist, bar_i)
        if s is None or not s["alive"] or s["state"] == 2:  # 2 = BROKEN_BSUT
            continue
        lp = s["price"]
        if is_bull and lp >= ref_price: continue   # must be below for bull
        if not is_bull and lp <= ref_price: continue  # must be above for bear
        d = abs(lp - ref_price)
        if d < best_dist:
            best_dist = d; best_dir = s["direction"]
    if best_dir is None:
        return False
    opposite = (best_dir == DIR_DOWN) if is_bull else (best_dir == DIR_UP)
    return opposite and (best_dist / atr) <= FILTER_DIST_ATR


def _has_opposing_level(is_bull: bool, ref_price: float, atr: float,
                        grouped: dict, bar_i: int) -> bool:
    """Replicates Pine Script v4 hasOpposingLevel.
    Returns True (blocked) if a confirmed opposing S/R level (green/red/orange)
    is within BLOCK_DIST_ATR ATRs on the wrong side."""
    if atr <= 0:
        return False
    for lhist in grouped.values():
        s = _hl_state_at(lhist, bar_i)
        if s is None or not s["alive"] or s["state"] == 2:
            continue
        lp = s["price"]
        # For short: blocking levels are below (support). For long: above (resistance).
        if is_bull and lp <= ref_price: continue
        if not is_bull and lp >= ref_price: continue
        is_confirmed_opposing = (
            (s["direction"] == DIR_DOWN and s["state"] == 1) if is_bull  # red above long
            else (s["direction"] == DIR_UP  and s["state"] == 1)          # green below short
        )
        is_orange = s["state"] == 3  # STATE_BREAK_TOUCHED
        if not (is_confirmed_opposing or is_orange):
            continue
        if (abs(lp - ref_price) / atr) <= BLOCK_DIST_ATR:
            return True
    return False


# ── Build trades with 1-min resolution ──────────────────────────────────────
def build_trades_1min(df_1h, df_1m, btc_df, use_pine_filter=False, use_opposing_filter=False):
    df = df_1h.copy()
    df["atr"]      = true_range(df).rolling(ATR_LEN).mean()
    df["atr_pct"]  = df["atr"] / df["close"]
    df["vol_avg"]  = df["volume"].rolling(20).mean()
    df["vol_ratio"]= df["volume"] / df["vol_avg"]

    htf = df[["close"]].resample("4h", label="left").last().dropna()
    htf["sma20"] = htf["close"].rolling(20).mean()
    htf["sma50"] = htf["close"].rolling(50).mean()
    htf["t"] = np.where(htf["sma20"]>htf["sma50"], 1,
               np.where(htf["sma20"]<htf["sma50"], -1, 0))
    idx = np.clip(htf.index.searchsorted(df.index, side="right")-1, 0, len(htf)-1)
    df["htf_label"] = np.where(htf["t"].values[idx]==1, "Bullish",
                      np.where(htf["t"].values[idx]==-1, "Bearish", "Range"))

    btc = btc_df.copy()
    btc["sma20"] = btc["close"].rolling(20).mean()
    btc["sma50"] = btc["close"].rolling(50).mean()
    btc["t"] = np.where(btc["sma20"]>btc["sma50"], 1,
               np.where(btc["sma20"]<btc["sma50"], -1, 0))
    bidx = np.clip(btc.index.searchsorted(df.index, side="right")-1, 0, len(btc)-1)
    df["btc_label"] = np.where(btc["t"].values[bidx]==1, "Up",
                      np.where(btc["t"].values[bidx]==-1, "Down", "Range"))

    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values

    # Build Highlander level history for Pine Script filters
    tick = max(float(df["close"].iloc[-1]) * 1e-5, 1e-6)
    _, hl_events = run_indicator(df[["open","high","low","close"]],
                                 min_range_ticks=3.0, tick_size=tick)
    hl_grouped: dict = {}
    for ev in hl_events:
        hl_grouped.setdefault(int(ev.level_id), []).append(ev.__dict__)

    # Convert 1-min to numpy int64 nanoseconds for fast searchsorted
    m1_ts   = df_1m.index.values.astype("datetime64[ns]").astype(np.int64)
    m1_vals = df_1m[["open","high","low","close"]].values

    records = []
    n_filtered_pine = 0
    n_filtered_opp  = 0
    for i in range(1, len(df)):
        bull = (C[i-1]<O[i-1]) and (C[i]>O[i]) and (C[i]>H[i-1])
        bear = (C[i-1]>O[i-1]) and (C[i]<O[i]) and (C[i]<L[i-1])

        for is_bull in ([True]*int(bull) + [False]*int(bear)):
            rng = H[i-1] - L[i-1]
            if rng <= 0: continue
            atr = float(df["atr"].iloc[i])
            if not np.isfinite(atr) or atr <= 0: continue

            origin = L[i-1] if is_bull else H[i-1]
            entry  = H[i-1] if is_bull else L[i-1]

            # ── Pine Script confluence filter (checkBreakRetestFilter) ──────
            if use_pine_filter:
                if not _check_break_retest(is_bull, entry, atr, hl_grouped, i):
                    n_filtered_pine += 1
                    continue

            # ── Pine Script v4 opposing level block ─────────────────────────
            if use_opposing_filter:
                if _has_opposing_level(is_bull, entry, atr, hl_grouped, i):
                    n_filtered_opp += 1
                    continue
            m_dir  = 1 if is_bull else -1
            stop   = origin - 0.5 * rng * m_dir
            f1618  = origin + 1.618 * rng * m_dir
            f2618  = origin + 2.618 * rng * m_dir
            risk   = abs(entry - stop)   # = 1.5 * rng always

            # ── Step 1: find activation 1H bar (fast, uses existing numpy arrays) ──
            act_bar = None
            for j in range(i+1, min(i+1+MAX_ACT_H, len(df))):
                if (is_bull and L[j] <= entry) or (not is_bull and H[j] >= entry):
                    act_bar = j
                    break
            if act_bar is None:
                continue

            # ── Step 2: find exact activation minute within that 1H bar ──
            act_1h_open_ns = np.int64(df.index[act_bar].value)
            act_1h_end_ns  = np.int64((df.index[act_bar] + pd.Timedelta(hours=1)).value)

            i_m_start = int(np.searchsorted(m1_ts, act_1h_open_ns, side="left"))
            i_m_end   = int(np.searchsorted(m1_ts, act_1h_end_ns,  side="left"))

            act_min_idx = None
            for j_m in range(i_m_start, i_m_end):
                h1m = m1_vals[j_m, 1]
                l1m = m1_vals[j_m, 2]
                if (is_bull and l1m <= entry) or (not is_bull and h1m >= entry):
                    act_min_idx = j_m
                    break

            # Fall back to 1H bar start if no 1-min data covers this hour
            if act_min_idx is None:
                if i_m_start < len(m1_ts):
                    act_min_idx = i_m_start
                else:
                    continue

            # ── Step 3: resolve outcome using 1-min data from activation minute ──
            outcome, be_hit, close_idx = resolve_outcome(
                m1_vals, m1_ts, act_min_idx,
                entry, stop, f1618, f2618, is_bull
            )
            if outcome is None:
                continue
            close_time = pd.Timestamp(m1_ts[min(close_idx, len(m1_ts)-1)], unit="ns")

            r = (WIN_R_GROSS - COST_R) if outcome == "win" \
                else (BE_R if outcome == "be_stop" else STOP_R)

            act_time    = pd.Timestamp(m1_ts[act_min_idx], unit="ns")
            act_1h_idx  = max(0, min(
                int(df.index.searchsorted(act_time.floor("h"), side="right")) - 1,
                len(df) - 1
            ))
            bars_to_act = max(1, act_bar - i)
            vr = float(df["vol_ratio"].iloc[i])

            # Engulf candle quality
            eng_rng        = H[i] - L[i]
            eng_body       = abs(C[i] - O[i])
            eng_upper_wick = H[i] - max(O[i], C[i])
            eng_lower_wick = min(O[i], C[i]) - L[i]
            _er = eng_rng if eng_rng > 0 else 1e-10
            body_pct       = round(eng_body       / _er, 3)
            upper_wick_pct = round(eng_upper_wick / _er, 3)
            lower_wick_pct = round(eng_lower_wick / _er, 3)
            engulf_ratio   = round(eng_rng / rng,   3) if rng > 0 else 1.0

            records.append({
                "engulf_time":        str(df.index[i]),
                "activation_time":    str(act_time),
                "close_time":         str(close_time),
                "direction":          "Long" if is_bull else "Short",
                "origin": origin, "entry": entry, "stop": stop,
                "fib_1618": f1618, "fib_2618": f2618, "rng": rng,
                "r_strategy_D_trail": round(r, 4),
                "outcome":            outcome,
                "rng_pct":            round(rng/entry*100, 3),
                "stop_pct":           round(risk/entry*100, 3),
                "tp1618_pct":         round(abs(f1618-entry)/entry*100, 3),
                "bars_to_activation": bars_to_act,
                "atr_pct_at_act":     round(float(df["atr_pct"].iloc[act_1h_idx]), 5),
                "atr_pct_at_eng":     round(float(df["atr_pct"].iloc[i]), 5),
                "htf_trend":          str(df["htf_label"].iloc[act_1h_idx]),
                "btc_trend":          str(df["btc_label"].iloc[act_1h_idx]),
                "session":            session_of(act_time.hour),
                "hour":               int(act_time.hour),
                "day":                act_time.day_name()[:3],
                "month":              int(act_time.month),
                "year":               int(act_time.year),
                "vol_ratio":          round(vr if np.isfinite(vr) else 1.0, 3),
                "rng_to_atr":         round(rng/atr, 3),
                "body_pct":           body_pct,
                "upper_wick_pct":     upper_wick_pct,
                "lower_wick_pct":     lower_wick_pct,
                "engulf_ratio":       engulf_ratio,
            })

    if use_pine_filter or use_opposing_filter:
        print(f"  Pine filter removed : {n_filtered_pine:,} engulfings (no broken level on trade side)")
        print(f"  Opposing filter removed: {n_filtered_opp:,} engulfings (opposing S/R too close)")
    ds = pd.DataFrame.from_records(records)
    ds["htf_aligned"] = (
        ((ds["direction"]=="Long")  & (ds["htf_trend"]=="Bullish")) |
        ((ds["direction"]=="Short") & (ds["htf_trend"]=="Bearish"))
    ).astype(int)
    ds["is_win_D"] = (ds["r_strategy_D_trail"] > 0).astype(int)
    return ds


# ── Entry-level features (orange lines / break levels on 1H + 4H) ────────────
def add_entry_level_features(ds, df_1h, df_4h):
    """
    For each engulf trade, check whether the entry price is near a Highlander
    level on the 1H or 4H timeframe at the time of the engulfing bar.

    Levels included: state=1 (origin confirmed), state=3 (first-touch / orange),
    state=2 (broken — flipped S/R).  Closest level in each TF is reported.
    """
    def _atr(df):
        h, l, pc = df["high"], df["low"], df["close"].shift(1)
        return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1).rolling(14).mean()

    def _state_at(hist, target_bar):
        if not hist or hist[0]["bar"] > target_bar: return None
        state = -1; confirmed = False; alive = True
        for ev in hist:
            if ev["bar"] > target_bar: break
            e = ev["event"]
            if   e == "level_created":     state = 0
            elif e == "first_touch":       state = 3
            elif e == "origin_confirmed":  state = 1
            elif e == "level_broken":      state = 2
            elif e == "level_deleted":     alive = False
            confirmed = ev["confirmed"]
        return {"state": state, "confirmed": confirmed, "alive": alive,
                "direction": hist[0]["direction"], "price": hist[0]["price"]}

    def _build_grouped(events):
        grouped = {}
        for ev in events:
            r = ev.__dict__
            lid = int(r["level_id"])
            grouped.setdefault(lid, []).append(r)
        return grouped

    df1 = df_1h[["open","high","low","close"]].copy().astype(float)
    df4 = df_4h[["open","high","low","close"]].copy().astype(float)
    df1["atr"] = _atr(df1)
    df4["atr"] = _atr(df4)

    tick_1h = max(float(df1["close"].iloc[-1]) * 1e-5, 1e-6)
    tick_4h = max(float(df4["close"].iloc[-1]) * 1e-5, 1e-6)

    print(f"  Highlander 1H ({len(df1):,} bars) …", end="", flush=True)
    _, events_1h = run_indicator(df1[["open","high","low","close"]], min_range_ticks=3.0, tick_size=tick_1h)
    print(f"  {len(events_1h):,} events")
    print(f"  Highlander 4H ({len(df4):,} bars) …", end="", flush=True)
    _, events_4h = run_indicator(df4[["open","high","low","close"]], min_range_ticks=3.0, tick_size=tick_4h)
    print(f"  {len(events_4h):,} events")

    grouped_1h = _build_grouped(events_1h)
    grouped_4h = _build_grouped(events_4h)
    bar_times_1h = df1.index.to_numpy()
    bar_times_4h = df4.index.to_numpy()

    def _nearest(grouped, eng_bar, df_tf, entry, is_bull):
        atr_v = float(df_tf["atr"].iloc[eng_bar])
        if not np.isfinite(atr_v) or atr_v <= 0:
            atr_v = abs(entry) * 0.01 or 1.0
        best = None; best_dist = float("inf")
        for lid, lhist in grouped.items():
            cb = next((h["bar"] for h in lhist if h["event"] == "level_created"), None)
            if cb is None or cb > eng_bar: continue
            db = next((h["bar"] for h in lhist if h["event"] == "level_deleted"), None)
            if db is not None and db <= eng_bar: continue
            s = _state_at(lhist, eng_bar)
            if s is None or s["state"] not in (1, 2, 3): continue
            dist = abs(s["price"] - entry)
            if dist < best_dist:
                best_dist = dist; best = s
        if best is None:
            return 0, 99.0, 0, 0, 0, 0
        dist_atr  = round(best_dist / atr_v, 3)
        at_level  = int(dist_atr <= 2.0)
        is_origin = int(best["state"] == 1)
        is_ft     = int(best["state"] == 3)
        is_broken = int(best["state"] == 2)
        dir_match = int((is_bull  and best["direction"] == DIR_UP) or
                        (not is_bull and best["direction"] == DIR_DOWN))
        return at_level, dist_atr, is_origin, is_ft, is_broken, dir_match

    ds = ds.copy()
    ds["engulf_time"] = pd.to_datetime(ds["engulf_time"])
    rows = []
    for _, row in ds.iterrows():
        eng_ts  = row["engulf_time"].to_numpy()
        is_bull = row["direction"] == "Long"
        entry   = float(row["entry"])
        origin  = float(row["origin"])

        eb1 = max(0, int(np.searchsorted(bar_times_1h, eng_ts, side="right")) - 1)
        eb4 = max(0, int(np.searchsorted(bar_times_4h, eng_ts, side="right")) - 1)
        eb1 = min(eb1, len(df1)-1)
        eb4 = min(eb4, len(df4)-1)

        at1, d1, o1, ft1, b1, dm1   = _nearest(grouped_1h, eb1, df1, entry,  is_bull)
        at4, d4, o4, ft4, b4, dm4   = _nearest(grouped_4h, eb4, df4, entry,  is_bull)
        _,  od1, oo1, oft1, ob1, odm1 = _nearest(grouped_1h, eb1, df1, origin, is_bull)
        _,  od4, oo4, oft4, ob4, odm4 = _nearest(grouped_4h, eb4, df4, origin, is_bull)

        rows.append({
            "entry_1h_at_level":  at1,  "entry_1h_dist_atr":  d1,
            "entry_1h_is_origin": o1,   "entry_1h_is_ft":     ft1,
            "entry_1h_is_broken": b1,   "entry_1h_dir_match": dm1,
            "entry_4h_at_level":  at4,  "entry_4h_dist_atr":  d4,
            "entry_4h_is_origin": o4,   "entry_4h_is_ft":     ft4,
            "entry_4h_is_broken": b4,   "entry_4h_dir_match": dm4,
            "origin_1h_dist_atr": od1,  "origin_1h_is_origin": oo1,
            "origin_1h_is_ft":    oft1, "origin_1h_is_broken": ob1,
            "origin_1h_dir_match":odm1,
            "origin_4h_dist_atr": od4,  "origin_4h_is_origin": oo4,
            "origin_4h_is_ft":    oft4, "origin_4h_is_broken": ob4,
            "origin_4h_dir_match":odm4,
        })

    return pd.concat([ds, pd.DataFrame(rows, index=ds.index)], axis=1)


# ── Walk-forward ─────────────────────────────────────────────────────────────
def walk_forward(ds, show_feature_importance=True, features=None, cat_features_idx=None, symbol="SOL/USDT", model_save_path=None):
    if features is None:         features        = FEATURES
    if cat_features_idx is None: cat_features_idx = cat_idx

    for col in ["support_dist_pct","support_dist_atr",
                "resistance_dist_pct","resistance_dist_atr",
                "entry_1h_dist_atr","entry_4h_dist_atr"]:
        if col in ds.columns:
            ds[col] = ds[col].fillna(99.0)
    ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)
    for c in CAT_FEATURES:
        ds[c] = ds[c].astype(str)
    ds["activation_time"] = pd.to_datetime(ds["activation_time"])

    t_start = ds["activation_time"].min()
    t_end   = ds["activation_time"].max()

    windows, ts = [], t_start + pd.DateOffset(months=MIN_TRAIN_MONTHS)
    while ts < t_end:
        te = min(ts + pd.DateOffset(months=STEP_MONTHS), t_end)
        windows.append((ts, te))
        ts = te

    chunks = []
    pos_w = tot_w = 0

    print(f"\n  {'Window':<30}  {'Train':>6}  {'Filt':>5}  {'WR':>6}  {'Net R':>8}  {'Note'}")
    print(f"  {'─'*70}")

    for ws, we in windows:
        train = ds[ds["activation_time"] < ws]
        test  = ds[(ds["activation_time"]>=ws)&(ds["activation_time"]<we)].copy().reset_index(drop=True)
        if len(train) < 100 or len(test) < 5:
            continue

        clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                                 cat_features=cat_features_idx, verbose=0,
                                 random_seed=42, l2_leaf_reg=5)
        clf.fit(train[features], train["is_win_D"])
        p = clf.predict_proba(test[features])[:,1]
        test["p_win_wf"] = p
        m = p >= P_THRESHOLD

        outcome_r = test["outcome"].map({"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R})
        net_r = outcome_r[m].sum() if m.sum() > 0 else 0.0
        wr    = test.loc[m,"is_win_D"].mean()*100 if m.sum() > 0 else 0.0
        sign  = "✅" if net_r > 0 else "❌"
        note  = "◀ UNSEEN" if ws >= UNSEEN_START else ""
        pos_w += int(net_r > 0)
        tot_w += 1

        print(f"  {sign}  {ws.date()}→{we.date()}   "
              f"train={len(train):5d}  filt={m.sum():4d}  "
              f"WR={wr:5.1f}%  net={net_r:+7.2f}R  {note}")
        chunks.append(test)

    if not chunks:
        print("  Not enough data")
        return

    all_oos  = pd.concat(chunks).sort_values("activation_time").reset_index(drop=True)
    m_all    = all_oos["p_win_wf"] >= P_THRESHOLD
    filtered = all_oos[m_all].copy()
    filtered["r_correct"] = filtered["outcome"].map(
        {"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R})

    # ── Apply execution constraints (one position per symbol at a time) ──────
    filtered["activation_time"] = pd.to_datetime(filtered["activation_time"])
    filtered["close_time"]      = pd.to_datetime(filtered["close_time"])
    sym_free_at = {}   # symbol → earliest time a new trade can open
    keep = []
    for _, row in filtered.iterrows():
        sym  = symbol
        free = sym_free_at.get(sym, pd.Timestamp.min)
        if row["activation_time"] >= free:
            keep.append(True)
            sym_free_at[sym] = row["close_time"]
        else:
            keep.append(False)
    constrained = filtered[keep].copy()
    n_skipped = len(filtered) - len(constrained)

    n  = len(constrained)
    w  = (constrained["outcome"]=="win").sum()
    be = (constrained["outcome"]=="be_stop").sum()
    fs = (constrained["outcome"]=="full_stop").sum()
    nr = constrained["r_correct"].sum()
    months_span = (constrained["activation_time"].max()-constrained["activation_time"].min()).days/30.44

    pos_r = constrained.loc[constrained["r_correct"]>0,"r_correct"].sum()
    neg_r = abs(constrained.loc[constrained["r_correct"]<0,"r_correct"].sum())
    pf    = pos_r/neg_r if neg_r > 0 else 999

    print(f"\n{'═'*65}")
    print(f"  {symbol}  1-min resolved  |  walk-forward OOS")
    print(f"{'═'*65}")
    print(f"  Date range      : {constrained['activation_time'].min().date()} → "
          f"{constrained['activation_time'].max().date()}")
    print(f"  Signals (filt.) : {n}  over {months_span:.0f} months  ({n/months_span:.1f}/month)")
    print(f"  Signals (filt.) : {len(filtered)}  →  {n} after execution constraints  ({n_skipped} skipped — symbol already active)")
    print(f"  Wins            : {w}  ({w/n*100:.1f}%)")
    print(f"  BE stops        : {be}  ({be/n*100:.1f}%)")
    print(f"  Full stops      : {fs}  ({fs/n*100:.1f}%)")
    print(f"  Profit factor   : {pf:.2f}")
    print(f"  Net R / month   : {nr/months_span:+.2f}R")
    print(f"  At $130/trade   : ${nr/months_span*130:,.0f}/month")
    print(f"  Positive windows: {pos_w}/{tot_w}")
    print(f"{'═'*65}")

    # ── Unseen 2025 section ──────────────────────────────────────────────────
    unseen = constrained[constrained["activation_time"] >= UNSEEN_START].copy()
    if len(unseen) > 0:
        un = len(unseen)
        uw = (unseen["outcome"]=="win").sum()
        ube= (unseen["outcome"]=="be_stop").sum()
        ufs= (unseen["outcome"]=="full_stop").sum()
        unr= unseen["r_correct"].sum()
        ump= (unseen["activation_time"].max()-unseen["activation_time"].min()).days/30.44
        upr= unseen.loc[unseen["r_correct"]>0,"r_correct"].sum()
        uneg=abs(unseen.loc[unseen["r_correct"]<0,"r_correct"].sum())
        upf = upr/uneg if uneg > 0 else 999

        print(f"\n{'═'*65}")
        print(f"  UNSEEN PERIOD (Jan 2025 onwards)  ◀ not seen during training")
        print(f"{'═'*65}")
        print(f"  Signals (filt.) : {un}  ({un/max(ump,0.1):.1f}/month)")
        print(f"  Wins            : {uw}  ({uw/un*100:.1f}%)")
        print(f"  BE stops        : {ube}  ({ube/un*100:.1f}%)")
        print(f"  Full stops      : {ufs}  ({ufs/un*100:.1f}%)")
        print(f"  Profit factor   : {upf:.2f}")
        print(f"  Net R / month   : {unr/max(ump,0.1):+.2f}R")
        print(f"  At $130/trade   : ${unr/max(ump,0.1)*130:,.0f}/month")
        print(f"{'═'*65}")

    # ── Drawdown + Monte Carlo ───────────────────────────────────────────────
    rs = constrained["r_correct"].values
    if len(rs) >= 5:
        # Actual max drawdown
        equity   = np.cumsum(rs)
        peak     = np.maximum.accumulate(equity)
        dd       = equity - peak
        max_dd   = dd.min()
        # Max consecutive losses
        streak = max_streak = 0
        for r in rs:
            streak = streak + 1 if r < 0 else 0
            max_streak = max(max_streak, streak)
        # Monte Carlo — shuffle trade order 2000 times
        rng      = np.random.default_rng(42)
        n_sims   = 2000
        sim_dds  = np.empty(n_sims)
        sim_fins = np.empty(n_sims)
        for k in range(n_sims):
            s    = rng.permutation(rs)
            eq   = np.cumsum(s)
            pk   = np.maximum.accumulate(eq)
            sim_dds[k]  = (eq - pk).min()
            sim_fins[k] = eq[-1]
        dd_median = np.percentile(sim_dds, 50)
        dd_95     = np.percentile(sim_dds, 5)   # worst 5% of sequences
        fin_95    = np.percentile(sim_fins, 5)   # 5th pct final R

        print(f"\n  ── Drawdown + Monte Carlo (2,000 simulations) ──")
        print(f"  Actual max drawdown      : {max_dd:.2f}R")
        print(f"  Max consecutive losses   : {max_streak}")
        print(f"  MC median max drawdown   : {dd_median:.2f}R")
        print(f"  MC worst 5% drawdown     : {dd_95:.2f}R  (95% of sequences stay above this)")
        print(f"  MC worst 5% final R      : {fin_95:.2f}R  (95% of sequences finish above this)")
        print(f"  ────────────────────────────────────────────────")

    # ── Feature importance ───────────────────────────────────────────────────
    if show_feature_importance:
        print(f"\n  Training final model for feature importance …")
        clf_final = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                                       cat_features=cat_features_idx, verbose=0,
                                       random_seed=42, l2_leaf_reg=5)
        clf_final.fit(ds[features], ds["is_win_D"])
        fi = pd.DataFrame({
            "feature":    features,
            "importance": clf_final.get_feature_importance()
        }).sort_values("importance", ascending=False)

        print(f"\n  Feature importance (what the model is learning):")
        print(f"  {'Feature':<30} {'Importance':>10}")
        print(f"  {'─'*42}")
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance"] / fi["importance"].max() * 20)
            print(f"  {row['feature']:<30} {row['importance']:>8.2f}%  {bar}")

    # Save final model (always, using last window's model if feature importance skipped)
    if not show_feature_importance:
        clf_final = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                                       cat_features=cat_features_idx, verbose=0,
                                       random_seed=42, l2_leaf_reg=5)
        clf_final.fit(ds[features], ds["is_win_D"])
    save_path = model_save_path or (DATA / "catboost_clf_solusdt_1h.cbm")
    clf_final.save_model(str(save_path))
    print(f"\n  Model saved → {save_path}")

    return constrained


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  SOL/USDT  1-Minute Resolution Backtest")
    print("=" * 65)

    print(f"\n  NOTE: Correct win R = {WIN_R:.4f}R  (stop at origin-0.5×rng, risk=1.5×rng)")
    print(f"  test_recent.py was using 1.578 (wrong — overstated wins by 52%)")

    # Load data
    df_1h  = pd.read_csv(DATA / "solusdt_1h.csv", index_col=0, parse_dates=True).astype(float)
    df_1h.index.name = "time"
    print(f"\n  1H data : {len(df_1h):,} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

    df_1m  = load_1min()
    btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)

    # Build trades
    print(f"\n  Building trades (1-min resolution) …")
    ds = build_trades_1min(df_1h, df_1m, btc_df)
    raw_wr = (ds["is_win_D"]==1).mean()*100
    print(f"  Trades: {len(ds):,}  Raw WR (no filter): {raw_wr:.1f}%")

    # Outcome breakdown (unfiltered)
    w_all  = (ds["outcome"]=="win").sum()
    be_all = (ds["outcome"]=="be_stop").sum()
    fs_all = (ds["outcome"]=="full_stop").sum()
    print(f"  Wins: {w_all} ({w_all/len(ds)*100:.1f}%)  "
          f"BE: {be_all} ({be_all/len(ds)*100:.1f}%)  "
          f"SL: {fs_all} ({fs_all/len(ds)*100:.1f}%)")

    # Add confluence (existing origin-level features)
    print(f"\n  Adding confluence features …")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ds = add_confluence(ds, df_1h)

    # Resample 1H → 4H for SOL
    df_4h_sol = (df_1h[["open","high","low","close","volume"]]
                 .resample("4h", label="left")
                 .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
                 .dropna())

    # Add orange-line / break-level features on 1H + 4H
    print(f"\n  Adding entry-level features (1H + 4H orange/break levels) …")
    ds = add_entry_level_features(ds, df_1h, df_4h_sol)

    # Walk-forward with extended features + feature importance
    print(f"\n  Walk-forward (extended features) …")
    walk_forward(ds, show_feature_importance=True,
                 features=FEATURES_EXT, cat_features_idx=cat_idx_ext)

    print("\n\nDone.")
