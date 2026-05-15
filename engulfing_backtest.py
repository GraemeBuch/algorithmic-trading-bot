"""
Backtest the 1v1w2w engulfing + fib-extension strategy.

Logic (mirroring Pine):
  1. Detect bullish/bearish engulfing candle.
       Bull: prev red, curr green, curr.close > prev.high
       Bear: prev green, curr red, curr.close < prev.low
  2. Define fib levels:
       fib0      = engulfing low (bull) / high (bear)        -- origin
       fib1.0    = engulfing high (bull) / low (bear)        -- entry on retest
       fib1.618  = origin + 1.618 * range (in trade dir)
       fib2.618, 3.618, 4.618  = same idea
       SL        = origin - 0.5*range (bull) / origin + 0.5*range (bear)
  3. Activation = price retests fib1.0 after engulfing bar
  4. Entry at fib1.0 (limit fill assumption)
  5. Walk forward: track which fib targets get hit and whether stop hits first
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR  = Path("data")
OUT_DIR.mkdir(exist_ok=True)
COST_R   = 0.04
ATR_LEN  = 14
MAX_BARS_TO_ACTIVATION = 50   # how long we wait for retest
MAX_BARS_AFTER_ENTRY   = 200  # how long the trade can run

# ─────── Engulfing detection ───────
def detect_engulfing(df):
    """Return DataFrame of engulfings with bar idx, direction, fib levels."""
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    bull = (c[:-1] < o[:-1])  # prev red
    cur_bull = (c[1:] > o[1:])  # curr green
    cur_above_prev_high = (c[1:] > h[:-1])
    bull_engulf = bull & cur_bull & cur_above_prev_high

    bear = (c[:-1] > o[:-1])  # prev green
    cur_bear = (c[1:] < o[1:])  # curr red
    cur_below_prev_low = (c[1:] < l[:-1])
    bear_engulf = bear & cur_bear & cur_below_prev_low

    rows = []
    n = len(df)
    for i in range(1, n):
        if bull_engulf[i-1]:
            origin = l[i-1]   # low of the previous (engulfed) red candle
            entry  = h[i-1]   # high of the previous (engulfed) red candle
            rng    = entry - origin
            if rng > 0:
                rows.append({"bar": i, "is_bull": True,
                             "origin": origin, "entry": entry, "rng": rng})
        if bear_engulf[i-1]:
            origin = h[i-1]   # high of the previous (engulfed) green candle
            entry  = l[i-1]   # low of the previous (engulfed) green candle
            rng    = origin - entry
            if rng > 0:
                rows.append({"bar": i, "is_bull": False,
                             "origin": origin, "entry": entry, "rng": rng})
    return pd.DataFrame(rows)


# ─────── Forward simulation ───────
def find_activation(H, L, engulf_bar, is_bull, entry_px, max_bars):
    """First bar after engulfing where price retests fib1.0 (engulfing extreme)."""
    n = len(H)
    end = min(engulf_bar + 1 + max_bars, n)
    for j in range(engulf_bar + 1, end):
        if is_bull and L[j] <= entry_px:
            return j
        if (not is_bull) and H[j] >= entry_px:
            return j
    return None

def simulate_fib(H, L, C, act_bar, is_bull, entry_px, stop_px, fib_targets, max_bars):
    """Walk forward post-activation. Track which fib targets hit & whether stop first.
    fib_targets is dict {name: price}.
    Returns dict with hit booleans, time-to-hit per target, stop hit flag, exit bar."""
    n = len(H)
    end = min(act_bar + 1 + max_bars, n)
    out = {f"hit_{name}": False for name in fib_targets}
    out.update({f"bars_to_{name}": None for name in fib_targets})
    out["stop_hit"] = False
    out["exit_bar"] = None
    out["max_favourable_excursion"] = 0.0  # in R-multiples to fib_-0.5 risk

    risk = abs(entry_px - stop_px)
    for j in range(act_bar + 1, end):
        hi, lo = H[j], L[j]

        # Check stop first
        if is_bull and lo <= stop_px:
            out["stop_hit"] = True
            out["exit_bar"] = j
            return out
        if (not is_bull) and hi >= stop_px:
            out["stop_hit"] = True
            out["exit_bar"] = j
            return out

        # Check each target (whichever first)
        for name, px in fib_targets.items():
            if out[f"hit_{name}"]: continue
            hit = (hi >= px) if is_bull else (lo <= px)
            if hit:
                out[f"hit_{name}"] = True
                out[f"bars_to_{name}"] = j - act_bar

        # Track favourable excursion
        if is_bull:
            mfe = (hi - entry_px) / risk
        else:
            mfe = (entry_px - lo) / risk
        if mfe > out["max_favourable_excursion"]:
            out["max_favourable_excursion"] = mfe

    out["exit_bar"] = end - 1
    return out


# ─────── Helpers ───────
def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def session_of(h):
    if 0 <= h < 7:    return "Asia"
    if 7 <= h < 13:   return "London"
    if 13 <= h < 17:  return "NY-AM"
    if 17 <= h < 22:  return "NY-PM"
    return "Off-hours"


# ─────── Pipeline per timeframe ───────
def run_tf(csv_path, tf_label, btc_path=None):
    print(f"\n{'═'*60}\nRunning engulfing backtest on {tf_label} ({csv_path})\n{'═'*60}")
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    cols = ["open","high","low","close"]
    has_volume = "volume" in df.columns
    if has_volume:
        df = df[cols + ["volume"]].astype(float)
    else:
        df = df[cols].astype(float)
    df["atr"]     = true_range(df).rolling(ATR_LEN).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    if has_volume:
        df["vol_avg20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_avg20"]

    H = df["high"].to_numpy(); L = df["low"].to_numpy()
    C = df["close"].to_numpy(); O = df["open"].to_numpy()
    print(f"  bars: {len(df):,}")

    # Detect engulfings
    eng = detect_engulfing(df)
    print(f"  engulfings detected: {len(eng):,}  "
          f"({(eng['is_bull']).sum()} bull / {(~eng['is_bull']).sum()} bear)")

    # HTF trend (resample to next-tier timeframe)
    if "1h" in str(csv_path).lower():
        htf_df = df[["close"]].resample("4h").last().dropna()
    else:
        htf_df = df[["close"]].resample("1D").last().dropna()
    htf_df["sma20"] = htf_df["close"].rolling(20).mean()
    htf_df["sma50"] = htf_df["close"].rolling(50).mean()
    htf_df["t"]     = np.where(htf_df["sma20"] > htf_df["sma50"], 1,
                       np.where(htf_df["sma20"] < htf_df["sma50"], -1, 0))
    df["htf_idx"] = htf_df.index.searchsorted(df.index, side="right") - 1
    df["htf_trend_n"] = np.where(df["htf_idx"] < 0, 0,
        htf_df["t"].to_numpy()[np.clip(df["htf_idx"], 0, len(htf_df)-1)])
    df["htf_label"] = np.where(df["htf_trend_n"] == 1, "Bullish",
                       np.where(df["htf_trend_n"] == -1, "Bearish", "Range"))

    # BTC context if available
    if btc_path:
        btc = pd.read_csv(btc_path); btc["time"] = pd.to_datetime(btc["time"])
        btc = btc.set_index("time").sort_index()
        btc["sma20"] = btc["close"].rolling(20).mean()
        btc["sma50"] = btc["close"].rolling(50).mean()
        btc["t"] = np.where(btc["sma20"] > btc["sma50"], 1,
                    np.where(btc["sma20"] < btc["sma50"], -1, 0))
        df["btc_idx"] = btc.index.searchsorted(df.index, side="right") - 1
        df["btc_trend_n"] = np.where(df["btc_idx"] < 0, 0,
            btc["t"].to_numpy()[np.clip(df["btc_idx"], 0, len(btc)-1)])
        df["btc_label"] = np.where(df["btc_trend_n"] == 1, "Up",
                           np.where(df["btc_trend_n"] == -1, "Down", "Range"))
    else:
        df["btc_label"] = "Unknown"

    # Build dataset
    records = []
    skipped = {"warmup": 0, "no_atr": 0, "no_activation": 0}
    for _, e in eng.iterrows():
        bar = int(e["bar"])
        if bar < ATR_LEN + 5: skipped["warmup"] += 1; continue
        atr = df["atr"].iloc[bar]
        if not np.isfinite(atr) or atr <= 0:
            skipped["no_atr"] += 1; continue

        is_bull = bool(e["is_bull"])
        origin  = float(e["origin"])
        entry   = float(e["entry"])
        rng     = float(e["rng"])

        # Find activation
        act_bar = find_activation(H, L, bar, is_bull, entry, MAX_BARS_TO_ACTIVATION)
        if act_bar is None:
            skipped["no_activation"] += 1; continue

        # Fib levels
        if is_bull:
            f1618 = origin + 1.618 * rng
            f2618 = origin + 2.618 * rng
            f3618 = origin + 3.618 * rng
            f4618 = origin + 4.618 * rng
            stop  = origin - 0.5  * rng
        else:
            f1618 = origin - 1.618 * rng
            f2618 = origin - 2.618 * rng
            f3618 = origin - 3.618 * rng
            f4618 = origin - 4.618 * rng
            stop  = origin + 0.5  * rng

        targets = {"f1618": f1618, "f2618": f2618, "f3618": f3618, "f4618": f4618}
        sim = simulate_fib(H, L, C, act_bar, is_bull, entry, stop, targets, MAX_BARS_AFTER_ENTRY)

        # Compute R outcomes for several exit strategies
        risk = abs(entry - stop)
        rng_ratio_to_risk = rng / risk    # always 1/1.5 = 0.667 by construction
        # R if exit at fib X
        r_at = {}
        for name, px in [("1618",f1618),("2618",f2618),("3618",f3618),("4618",f4618)]:
            if is_bull:
                r_at[name] = (px - entry) / risk
            else:
                r_at[name] = (entry - px) / risk

        # Outcome strategies:
        #  A) TP at 1.618 — full position there (R = 0.412)
        #  B) Hold to 2.618 (R = 1.08)
        #  C) Half at 1.618, half at 2.618 (avg R = 0.747)
        #  D) Trail: stop to BE at 1.618, hold to 2.618 ⇒ if stops out before 2.618, get +1.618 partial; else +2.618 full
        # Stop-first checks
        stop_first = sim["stop_hit"] and not sim["hit_f1618"]

        if stop_first:
            r_A = -1.0
            r_B = -1.0
            r_C = -1.0
            r_D = -1.0
        else:
            r_A = r_at["1618"] if sim["hit_f1618"] else -1.0
            r_B = r_at["2618"] if sim["hit_f2618"] else -1.0
            r_C = (r_at["1618"] + (r_at["2618"] if sim["hit_f2618"] else -1.0)) / 2
            # Strategy D: if hit 1.618 first, BE stop. If 2.618 hit, +2.618R. Else 0R.
            if sim["hit_f1618"]:
                if sim["hit_f2618"]:
                    r_D = r_at["2618"]
                else:
                    # stop is at BE after 1.618 — this is a 0R close (could also be small loss/win on dust)
                    r_D = 0.0
            else:
                r_D = -1.0

        ts = df.index[act_bar]
        records.append({
            "engulf_time":      df.index[bar],
            "activation_time":  ts,
            "is_bull":          is_bull,
            "direction":        "Long" if is_bull else "Short",
            "origin":           round(origin, 6),
            "entry":            round(entry, 6),
            "stop":             round(stop, 6),
            "fib_1618":         round(f1618, 6),
            "fib_2618":         round(f2618, 6),
            "fib_3618":         round(f3618, 6),
            "fib_4618":         round(f4618, 6),
            "rng":              round(rng, 6),
            "rng_pct":          round(rng / entry * 100, 3),  # engulfing range as %
            "stop_pct":         round(abs(entry - stop) / entry * 100, 3),
            "tp1618_pct":       round(abs(f1618 - entry) / entry * 100, 3),
            # Outcomes
            "stop_first":       int(stop_first),
            "hit_1618":         int(sim["hit_f1618"]),
            "hit_2618":         int(sim["hit_f2618"]),
            "hit_3618":         int(sim["hit_f3618"]),
            "hit_4618":         int(sim["hit_f4618"]),
            "max_R_excursion":  round(sim["max_favourable_excursion"], 3),
            "r_strategy_A_TP1618":   round(r_A - COST_R, 3),
            "r_strategy_B_TP2618":   round(r_B - COST_R, 3),
            "r_strategy_C_split":    round(r_C - COST_R, 3),
            "r_strategy_D_trail":    round(r_D - COST_R, 3),
            "bars_to_activation": act_bar - bar,
            # Features
            "atr_pct_at_act":   round(float(df["atr_pct"].iloc[act_bar]), 5),
            "atr_pct_at_eng":   round(float(df["atr_pct"].iloc[bar]), 5),
            "htf_trend":        str(df["htf_label"].iloc[act_bar]),
            "btc_trend":        str(df["btc_label"].iloc[act_bar]),
            "session":          session_of(ts.hour),
            "hour":             int(ts.hour),
            "day":              ts.day_name()[:3],
            "month":            int(ts.month),
            "year":             int(ts.year),
            "vol_ratio":        round(float(df["vol_ratio"].iloc[bar]), 3) if has_volume and np.isfinite(df["vol_ratio"].iloc[bar]) else np.nan,
            "rng_to_atr":       round(rng / atr, 3),  # engulfing strength relative to ATR
        })

    ds = pd.DataFrame.from_records(records)
    ds["htf_aligned"] = (((ds["direction"]=="Long")  & (ds["htf_trend"]=="Bullish")) |
                        ((ds["direction"]=="Short") & (ds["htf_trend"]=="Bearish"))).astype(int)
    out_path = OUT_DIR / f"engulfing_{tf_label}.csv"
    ds.to_csv(out_path, index=False)
    print(f"  trades: {len(ds):,}  (skipped: {skipped})")
    print(f"  saved → {out_path}")

    # ─────── SUMMARY ───────
    if len(ds) == 0: return ds

    print(f"\n— Outcomes (all {len(ds):,} trades) —")
    for col in ["r_strategy_A_TP1618", "r_strategy_B_TP2618", "r_strategy_C_split", "r_strategy_D_trail"]:
        wr = (ds[col] > 0).mean()
        avg = ds[col].mean()
        net = ds[col].sum()
        print(f"  {col:30s}  win {wr:5.1%}  avgR {avg:+.3f}  netR {net:+7.1f}")

    print(f"\n— Hit rates —")
    for fib in ["1618","2618","3618","4618"]:
        rate = ds[f"hit_{fib}"].mean()
        print(f"  reaches fib {fib}: {rate:.1%}")
    print(f"  stop hit first  : {ds['stop_first'].mean():.1%}")

    print(f"\n— Best filters by avg R (strategy C — half at 1618, half at 2618) —")
    for col in ["htf_trend", "session", "btc_trend", "htf_aligned"]:
        g = ds.groupby(col)["r_strategy_C_split"].agg(["count","mean"]).round(3)
        print(f"\n  by {col}:")
        print(g.to_string())

    # ATR / range buckets
    ds["atr_bin"] = pd.qcut(ds["atr_pct_at_act"], 4, labels=["Low","MidLo","MidHi","High"], duplicates="drop")
    ds["rng_to_atr_bin"] = pd.qcut(ds["rng_to_atr"], 4, labels=["Small","Med","Big","Huge"], duplicates="drop")
    print("\n  by ATR regime:")
    print(ds.groupby("atr_bin")["r_strategy_C_split"].agg(["count","mean"]).round(3).to_string())
    print("\n  by engulfing strength (rng/ATR):")
    print(ds.groupby("rng_to_atr_bin")["r_strategy_C_split"].agg(["count","mean"]).round(3).to_string())
    print("\n  triple-feature top combos (HTF × Session × ATR):")
    g3 = ds.groupby(["htf_aligned","session","atr_bin"])["r_strategy_C_split"].agg(["count","mean"]).round(3)
    g3 = g3[g3["count"] >= 30].sort_values("mean", ascending=False)
    print(g3.head(15).to_string())

    return ds


# ─────── Run both timeframes ───────
ds_1h = run_tf("data/sol_1h.csv", "1h", "data/btc_4h.csv")
ds_4h = run_tf("data/sol_4h.csv", "4h", "data/btc_4h.csv")

# Combined summary
print(f"\n{'═'*60}\nDATASETS BUILT\n{'═'*60}")
print(f"  1H trades: {len(ds_1h):,}  →  data/engulfing_1h.csv")
print(f"  4H trades: {len(ds_4h):,}  →  data/engulfing_4h.csv")
