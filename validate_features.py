"""
validate_features.py

Checks that the live bot and backtest compute identical feature values
for the same engulfing signals.

For each signal the backtest finds in the last 60 days, we re-compute
all 54 features using the live bot's code path and compare them side by side.

Usage:
    python validate_features.py
"""
from __future__ import annotations
import sys, io, warnings, contextlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sol_1min_backtest import build_trades_1min, add_entry_level_features
from test_new_symbols import add_confluence, FEATURES, CAT_FEATURES
from highlander import run_indicator, DIR_UP, DIR_DOWN, STATE_ORIGIN, STATE_BREAK_TOUCHED, STATE_BROKEN_BSUT

DATA = Path("data")
ATR_LEN = 14

SYMBOLS = [
    ("BTC/USDT",  "btcusdt_1h.csv",  "btcusdt_1m.csv"),
    ("ETH/USDT",  "ethusdt_1h.csv",  "ethusdt_1m.csv"),
    ("SOL/USDT",  "solusdt_1h.csv",  "solusdt_1m_bitget.csv"),
    ("TRX/USDT",  "trxusdt_1h.csv",  "trxusdt_1m.csv"),
    ("LINK/USDT", "linkusdt_1h.csv", "linkusdt_1m.csv"),
    ("HBAR/USDT", "hbarusdt_1h.csv", "hbarusdt_1m.csv"),
    ("XRP/USDT",  "xrpusdt_1h.csv",  "xrpusdt_1m.csv"),
    ("DOGE/USDT", "dogeusdt_1h.csv", "dogeusdt_1m.csv"),
    ("LTC/USDT",  "ltcusdt_1h.csv",  "ltcusdt_1m.csv"),
    ("ADA/USDT",  "adausdt_1h.csv",  "adausdt_1m.csv"),
    ("XLM/USDT",  "xlmusdt_1h.csv",  "xlmusdt_1m.csv"),
    ("UNI/USDT",  "uniusdt_1h.csv",  "uniusdt_1m.csv"),
    ("APT/USDT",  "aptusdt_1h.csv",  "aptusdt_1m.csv"),
    ("NEAR/USDT", "nearusdt_1h.csv", "nearusdt_1m.csv"),
]


# ── Live bot helper functions (copied exactly from live_signals.py) ────────────

def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([abs(h - l), abs(h - pc), abs(l - pc)], axis=1).max(axis=1)

def session_of(h):
    if 0  <= h < 7:  return "Asia"
    if 7  <= h < 13: return "London"
    if 13 <= h < 17: return "NY-AM"
    if 17 <= h < 22: return "NY-PM"
    return "Off-hours"

def get_confluence_live(bar_idx, price, is_bull, levels, atr):
    above, below = [], []
    for lvl in levels:
        if lvl.created_bar > bar_idx: continue
        if lvl.deleted_bar is not None and lvl.deleted_bar <= bar_idx: continue
        if lvl.state == STATE_BROKEN_BSUT: continue
        diff = lvl.price - price
        info = dict(
            direction=lvl.dir, is_origin=(lvl.state == STATE_ORIGIN),
            confirmed=lvl.confirmed,
            dist_pct=abs(diff) / price * 100,
            dist_atr=abs(diff) / atr if atr > 0 else 99.0,
        )
        (above if diff > 0 else below).append(info)
    above.sort(key=lambda x: x["dist_pct"])
    below.sort(key=lambda x: x["dist_pct"])
    sup = (below[0] if below else None) if is_bull else (above[0] if above else None)
    res = (above[0] if above else None) if is_bull else (below[0] if below else None)
    g = lambda d, k, dfl: d[k] if d else dfl
    sup_dir = bool(sup) and ((is_bull and sup["direction"] == DIR_UP) or
                             (not is_bull and sup["direction"] == DIR_DOWN))
    sup_atr = g(sup, "dist_atr", 99.0)
    score = ((1 if sup_atr < 1 else 0) +
             (1 if g(sup, "is_origin", False) else 0) +
             (1 if g(sup, "confirmed", False) else 0) +
             (1 if sup_dir else 0) -
             (1 if res and g(res, "dist_atr", 99) < 1.5 and g(res, "is_origin", False) else 0))
    return {
        "support_present":      int(bool(sup)),
        "support_dist_pct":     round(g(sup, "dist_pct", 99.0), 3),
        "support_dist_atr":     round(sup_atr, 3),
        "support_is_origin":    int(g(sup, "is_origin", False)),
        "support_confirmed":    int(g(sup, "confirmed", False)),
        "support_dir_matches":  int(sup_dir),
        "resistance_present":   int(bool(res)),
        "resistance_dist_pct":  round(g(res, "dist_pct", 99.0), 3),
        "resistance_dist_atr":  round(g(res, "dist_atr", 99.0), 3),
        "resistance_is_origin": int(g(res, "is_origin", False)),
        "resistance_confirmed": int(g(res, "confirmed", False)),
        "confluence_score":     score,
    }

def get_entry_level_feat_live(entry, is_bull, atr, levels, i_cur, prefix):
    best = None; best_dist = float("inf")
    for lvl in levels:
        if lvl.created_bar > i_cur: continue
        if lvl.deleted_bar is not None and lvl.deleted_bar <= i_cur: continue
        if lvl.state not in (STATE_ORIGIN, STATE_BREAK_TOUCHED, STATE_BROKEN_BSUT): continue
        dist = abs(lvl.price - entry)
        if dist < best_dist:
            best_dist = dist; best = lvl
    if best is None:
        return {f"{prefix}_at_level": 0, f"{prefix}_dist_atr": 99.0,
                f"{prefix}_is_origin": 0, f"{prefix}_is_ft": 0,
                f"{prefix}_is_broken": 0, f"{prefix}_dir_match": 0}
    dist_atr = round(best_dist / atr, 3) if atr > 0 else 99.0
    return {
        f"{prefix}_at_level":  int(dist_atr <= 2.0),
        f"{prefix}_dist_atr":  dist_atr,
        f"{prefix}_is_origin": int(best.state == STATE_ORIGIN),
        f"{prefix}_is_ft":     int(best.state == STATE_BREAK_TOUCHED),
        f"{prefix}_is_broken": int(best.state == STATE_BROKEN_BSUT),
        f"{prefix}_dir_match": int((is_bull and best.dir == DIR_UP) or
                                   (not is_bull and best.dir == DIR_DOWN)),
    }


def compute_live_features(row, df_1h, btc_df):
    """Re-compute all live bot features for a backtest signal row."""
    eng_ts  = pd.to_datetime(row["engulf_time"])
    act_ts  = pd.to_datetime(row["activation_time"])
    is_bull = row["direction"] == "Long"
    entry   = float(row["entry"])
    origin  = float(row["origin"])

    # Find bar indices
    eng_bar = int(np.searchsorted(df_1h.index, eng_ts, side="right")) - 1
    act_bar = int(np.searchsorted(df_1h.index, act_ts, side="right")) - 1
    eng_bar = max(0, min(eng_bar, len(df_1h) - 1))
    act_bar = max(0, min(act_bar, len(df_1h) - 1))

    df_1h["atr"]      = true_range(df_1h).rolling(ATR_LEN).mean()
    df_1h["atr_pct"]  = df_1h["atr"] / df_1h["close"]
    df_1h["vol_avg20"] = df_1h["volume"].rolling(20).mean()
    df_1h["vol_ratio"] = df_1h["volume"] / df_1h["vol_avg20"]

    df_4h = (df_1h[["open","high","low","close","volume"]]
             .resample("4h", label="left")
             .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
             .dropna())
    df_4h["atr"] = true_range(df_4h).rolling(ATR_LEN).mean()

    # HTF trend (same SMA20/50 as live bot)
    sma20 = df_4h["close"].rolling(20).mean()
    sma50 = df_4h["close"].rolling(50).mean()
    htf_series = pd.Series(
        np.where(sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0)),
        index=df_4h.index)
    htf_pos = max(0, int(np.searchsorted(df_4h.index, act_ts, side="right")) - 1)
    htf_v = int(htf_series.iloc[htf_pos])
    htf_now = {1: "Bullish", -1: "Bearish", 0: "Range"}[htf_v]

    # BTC trend
    btc = btc_df.copy()
    btc["sma20"] = btc["close"].rolling(20).mean()
    btc["sma50"] = btc["close"].rolling(50).mean()
    btc_t = pd.Series(
        np.where(btc["sma20"] > btc["sma50"], 1, np.where(btc["sma20"] < btc["sma50"], -1, 0)),
        index=btc.index)
    btc_pos = max(0, int(np.searchsorted(btc.index, act_ts, side="right")) - 1)
    btc_v = int(btc_t.iloc[btc_pos])
    btc_now = {1: "Up", -1: "Down", 0: "Range"}[btc_v]

    atr_eng = float(df_1h["atr"].iloc[eng_bar])
    atr_act = float(df_1h["atr"].iloc[act_bar])
    atr_pct_act = float(df_1h["atr_pct"].iloc[act_bar])
    atr_pct_eng = float(df_1h["atr_pct"].iloc[eng_bar])
    vr = float(df_1h["vol_ratio"].iloc[eng_bar])
    # rng = previous bar's H-L (bar i-1, not the engulfing bar i)
    prev_bar = max(0, eng_bar - 1)
    rng = float(df_1h["high"].iloc[prev_bar] - df_1h["low"].iloc[prev_bar])
    risk = abs(entry - (origin - 0.5 * rng * (1 if is_bull else -1)))
    f1618 = origin + 1.618 * rng * (1 if is_bull else -1)
    direction = "Long" if is_bull else "Short"
    bars_elapsed = act_bar - eng_bar
    htf_aligned = int((direction == "Long" and htf_now == "Bullish") or
                      (direction == "Short" and htf_now == "Bearish"))

    # Highlander levels — 120-day trim, matching startup cache
    cutoff_120 = df_1h.index.max() - pd.Timedelta(days=120)
    df_1h_trim = df_1h[df_1h.index >= cutoff_120].copy()
    df_4h_trim = (df_1h_trim[["open","high","low","close","volume"]]
                  .resample("4h", label="left")
                  .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
                  .dropna())
    tick = max(float(df_1h_trim["close"].iloc[-1]) * 1e-5, 1e-6)
    levels_1h, _ = run_indicator(df_1h_trim[["open","high","low","close"]], min_range_ticks=3.0, tick_size=tick)
    tick4 = max(float(df_4h_trim["close"].iloc[-1]) * 1e-5, 1e-6)
    levels_4h, _ = run_indicator(df_4h_trim[["open","high","low","close"]], min_range_ticks=3.0, tick_size=tick4)

    ts_1h = df_1h_trim.index.to_numpy().astype("int64")
    ts_4h = df_4h_trim.index.to_numpy().astype("int64")

    def cache_bar(ts, ts_arr):
        idx = int(np.searchsorted(ts_arr, np.int64(ts.value), side="right")) - 1
        return max(0, min(idx, len(ts_arr) - 1))

    # Confluence at activation time
    act_price = float(df_1h["close"].iloc[act_bar])
    atr_now = atr_act
    ci_act = cache_bar(act_ts, ts_1h)
    conf = get_confluence_live(ci_act, act_price, is_bull, levels_1h, atr_now)

    # Entry/origin level features at engulf time
    ci_eng = cache_bar(eng_ts, ts_1h)
    ci4_eng = cache_bar(eng_ts, ts_4h)
    i4 = max(0, int(np.searchsorted(df_4h.index, eng_ts, side="right")) - 1)
    atr_4h = float(df_4h["atr"].iloc[i4]) if np.isfinite(df_4h["atr"].iloc[i4]) else atr_eng

    lf1h      = get_entry_level_feat_live(entry,  is_bull, atr_eng, levels_1h, ci_eng, "entry_1h")
    lf4h      = get_entry_level_feat_live(entry,  is_bull, atr_4h, levels_4h, ci4_eng, "entry_4h")
    lf_orig1h = get_entry_level_feat_live(origin, is_bull, atr_eng, levels_1h, ci_eng, "origin_1h")
    lf_orig4h = get_entry_level_feat_live(origin, is_bull, atr_4h, levels_4h, ci4_eng, "origin_4h")
    lf_orig1h.pop("origin_1h_at_level", None)
    lf_orig4h.pop("origin_4h_at_level", None)

    # Engulf candle quality
    hi = float(df_1h["high"].iloc[eng_bar])
    lo = float(df_1h["low"].iloc[eng_bar])
    op = float(df_1h["open"].iloc[eng_bar])
    cl = float(df_1h["close"].iloc[eng_bar])
    eng_rng = hi - lo
    _er = eng_rng if eng_rng > 0 else 1e-10
    body_pct       = round(abs(cl - op) / _er, 3)
    upper_wick_pct = round((hi - max(op, cl)) / _er, 3)
    lower_wick_pct = round((min(op, cl) - lo) / _er, 3)
    engulf_ratio   = round(eng_rng / rng, 3) if rng > 0 else 1.0

    feat = {
        "direction":          direction,
        "session":            session_of(act_ts.hour),
        "htf_trend":          htf_now,
        "btc_trend":          btc_now,
        "day":                act_ts.day_name()[:3],
        "bars_to_activation": bars_elapsed,
        "atr_pct_at_act":     round(atr_pct_act, 6),
        "atr_pct_at_eng":     round(atr_pct_eng, 6),
        "rng_pct":            round(rng / entry * 100, 4),
        "stop_pct":           round(risk / entry * 100, 4),
        "tp1618_pct":         round(abs(f1618 - entry) / entry * 100, 4),
        "rng_to_atr":         round(rng / atr_eng, 4) if atr_eng > 0 else 99.0,
        "vol_ratio":          round(vr if np.isfinite(vr) else 1.0, 4),
        "hour":               int(act_ts.hour),
        "month":              int(act_ts.month),
        "htf_aligned":        htf_aligned,
        **conf,
        "body_pct":           body_pct,
        "upper_wick_pct":     upper_wick_pct,
        "lower_wick_pct":     lower_wick_pct,
        "engulf_ratio":       engulf_ratio,
        **lf1h, **lf4h, **lf_orig1h, **lf_orig4h,
    }
    return feat


def compare_features(bt_row, live_feat):
    """Return list of (feature, backtest_val, live_val, match) tuples."""
    results = []
    for f in FEATURES:
        bv = bt_row.get(f)
        lv = live_feat.get(f)
        if bv is None or lv is None:
            results.append((f, bv, lv, "MISSING"))
            continue
        # Categorical: exact match
        if f in CAT_FEATURES:
            match = "OK" if str(bv) == str(lv) else "DIFF"
        else:
            # Numeric: allow small floating point tolerance
            try:
                match = "OK" if abs(float(bv) - float(lv)) < 0.01 else "DIFF"
            except Exception:
                match = "OK" if str(bv) == str(lv) else "DIFF"
        results.append((f, bv, lv, match))
    return results


if __name__ == "__main__":
    _btc_1h = pd.read_csv(DATA / "btcusdt_1h.csv", index_col=0, parse_dates=True).astype(float)
    btc_df = (_btc_1h[["open","high","low","close","volume"]]
              .resample("4h", label="left")
              .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
              .dropna())

    total_signals = 0
    total_diffs   = 0

    for symbol, csv_1h, csv_1m in SYMBOLS:
        path_1h = DATA / csv_1h
        path_1m = DATA / csv_1m
        slug    = symbol.replace("/","").lower()
        if not path_1h.exists() or not path_1m.exists():
            continue

        df_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True).astype(float)
        df_1h.index.name = "time"
        df_1m = pd.read_csv(path_1m, index_col=0, parse_dates=True).astype(float)
        df_1m.index.name = "time"

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                ds = build_trades_1min(df_1h, df_1m, btc_df)
            except Exception as e:
                print(f"  {symbol}: build_trades_1min failed: {e}")
                continue

        if ds is None or ds.empty:
            continue

        with contextlib.redirect_stdout(buf):
            ds = add_confluence(ds, df_1h)

        df_4h = (df_1h[["open","high","low","close","volume"]]
                 .resample("4h", label="left")
                 .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
                 .dropna())
        with contextlib.redirect_stdout(buf):
            ds = add_entry_level_features(ds, df_1h, df_4h)

        # Fill missing values same as backtest
        for col in ["support_dist_pct","support_dist_atr","resistance_dist_pct",
                    "resistance_dist_atr","entry_1h_dist_atr","entry_4h_dist_atr",
                    "origin_1h_dist_atr","origin_4h_dist_atr"]:
            if col in ds.columns:
                ds[col] = ds[col].fillna(99.0)
        if "vol_ratio" in ds.columns:
            ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)

        # Filter to last 30 days — same window as recent_stats.py
        ds["act_dt"] = pd.to_datetime(ds["activation_time"])
        cutoff = ds["act_dt"].max() - pd.Timedelta(days=30)
        recent = ds[ds["act_dt"] >= cutoff].copy()
        print(f"\n{symbol}  ({len(recent)} signals in last 30 days)")

        if recent.empty:
            print(f"  no recent signals")
            continue

        # Take up to 5 most recent signals
        for _, row in recent.tail(5).iterrows():
            total_signals += 1
            try:
                live_feat = compute_live_features(row, df_1h, btc_df)
            except Exception as e:
                print(f"  [{row['activation_time']}] ERROR computing live features: {e}")
                continue

            comparisons = compare_features(row, live_feat)
            diffs = [(f, bv, lv) for f, bv, lv, m in comparisons if m == "DIFF"]
            total_diffs += len(diffs)

            status = "✓ ALL MATCH" if not diffs else f"✗ {len(diffs)} DIFF(s)"
            print(f"  [{row['activation_time']}]  {row['direction']:<6}  entry={row['entry']:.4f}  {status}")

            if diffs:
                print(f"  {'Feature':<30} {'Backtest':>15} {'Live':>15}")
                print(f"  {'-'*62}")
                for f, bv, lv in diffs:
                    print(f"  {f:<30} {str(bv):>15} {str(lv):>15}")

    print(f"\n{'='*60}")
    print(f"  Checked {total_signals} signals across all symbols")
    print(f"  Total feature mismatches: {total_diffs}")
    if total_diffs == 0:
        print("  ✓ Live bot and backtest features are IDENTICAL")
    else:
        print("  ✗ Mismatches found — investigate above")
    print(f"{'='*60}")
