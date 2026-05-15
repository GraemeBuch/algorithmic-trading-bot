"""
Do orange (STATE_BREAK_TOUCHED / first-touch) levels add value as support/resistance,
similar to origin levels?

Runs full backtest for SOL and SUI, then groups outcomes by what type of level
was nearest support at activation time.

Groups:
  none    — no level below entry (bull) / above entry (bear)
  yellow  — STATE_BREAK       (fresh, untested level)
  orange  — STATE_BREAK_TOUCHED (first touch — the one we're investigating)
  origin  — STATE_ORIGIN       (full 3-stage confirmation)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from highlander import (
    run_indicator, DIR_UP, DIR_DOWN,
    STATE_BREAK, STATE_BREAK_TOUCHED, STATE_ORIGIN, STATE_BROKEN_BSUT,
)

ATR_LEN         = 14
MAX_BARS_TO_ACT = 50
MAX_BARS_AFTER  = 200
COST_R          = 0.04

SYMBOLS = {
    "SOL/USDT": "data/sol_1h.csv",
    "SUI/USDT": "data/suiusdt_1h.csv",
}

STATE_LABEL = {
    STATE_BREAK:         "yellow (fresh)",
    STATE_BREAK_TOUCHED: "orange (1st touch)",
    STATE_ORIGIN:        "origin (confirmed)",
}


def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)


def load_1h(csv_path):
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df["atr"] = true_range(df).ewm(span=ATR_LEN, adjust=False).mean()
    return df


def run(symbol, csv_path):
    df = load_1h(csv_path)
    print(f"\n{'█'*60}")
    print(f"  {symbol}  —  {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"{'█'*60}")

    levels, _ = run_indicator(df[["open","high","low","close"]])

    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    T = df.index

    rows = []

    for i in range(1, len(df) - 1):
        bull = (C[i-1] < O[i-1]) and (C[i] > O[i]) and (C[i] > H[i-1])
        bear = (C[i-1] > O[i-1]) and (C[i] < O[i]) and (C[i] < L[i-1])
        if not (bull or bear):
            continue

        is_bull = bull
        origin  = L[i-1] if is_bull else H[i-1]
        entry   = H[i-1] if is_bull else L[i-1]
        rng     = H[i-1] - L[i-1]
        if rng <= 0:
            continue

        m     = 1 if is_bull else -1
        stop  = origin - 0.5 * rng * m
        be_lv = origin + 1.618 * rng * m
        tp    = origin + 2.618 * rng * m
        atr_i = float(df["atr"].iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        # Find activation
        act_bar = None
        for j in range(i + 1, min(i + 1 + MAX_BARS_TO_ACT, len(df))):
            if (is_bull and L[j] <= entry) or (not is_bull and H[j] >= entry):
                act_bar = j; break
        if act_bar is None:
            continue

        atr_act = float(df["atr"].iloc[act_bar])

        # Find outcome
        be_hit  = False
        outcome = None
        for j in range(act_bar + 1, min(act_bar + 1 + MAX_BARS_AFTER, len(df))):
            cur_stop = entry if be_hit else stop
            if is_bull:
                if not be_hit and H[j] >= be_lv: be_hit = True
                if L[j] <= cur_stop: outcome = "be_stop" if be_hit else "full_stop"; break
                if H[j] >= tp:       outcome = "win";  break
            else:
                if not be_hit and L[j] <= be_lv: be_hit = True
                if H[j] >= cur_stop: outcome = "be_stop" if be_hit else "full_stop"; break
                if L[j] <= tp:       outcome = "win";  break
        if outcome is None:
            continue

        r = (abs(tp - entry) / abs(entry - stop) - COST_R) if outcome == "win" \
            else (-COST_R if outcome == "be_stop" else -1.0 - COST_R)

        # Classify nearest support level by state at activation bar
        above, below = [], []
        for lvl in levels:
            if lvl.created_bar > act_bar:
                continue
            if lvl.deleted_bar is not None and lvl.deleted_bar <= act_bar:
                continue
            if lvl.state == STATE_BROKEN_BSUT:
                continue
            diff = lvl.price - entry
            info = dict(state=lvl.state, dist_atr=abs(diff)/atr_act if atr_act>0 else 99)
            (above if diff > 0 else below).append(info)

        above.sort(key=lambda x: x["dist_atr"])
        below.sort(key=lambda x: x["dist_atr"])

        sup = (below[0] if below else None) if is_bull else (above[0] if above else None)

        if sup is None:
            sup_group = "none"
        elif sup["state"] == STATE_ORIGIN:
            sup_group = "origin"
        elif sup["state"] == STATE_BREAK_TOUCHED:
            sup_group = "orange"
        elif sup["state"] == STATE_BREAK:
            sup_group = "yellow"
        else:
            sup_group = "other"

        rows.append({
            "outcome":   outcome,
            "r":         r,
            "sup_group": sup_group,
            "sup_atr":   sup["dist_atr"] if sup else None,
        })

    df_r = pd.DataFrame(rows)
    total = len(df_r)
    print(f"  Total activated trades: {total}")
    print()

    order = ["origin", "orange", "yellow", "none"]
    print(f"  {'Group':<22}  {'N':>5}  {'%total':>7}  {'WR':>7}  {'PF':>6}  {'Net R':>8}  {'R/trade':>8}")
    print(f"  {'─'*22}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*8}")

    for grp in order:
        sub = df_r[df_r["sup_group"] == grp]
        n   = len(sub)
        if n == 0:
            print(f"  {grp:<22}  {n:>5}  —")
            continue
        wr    = (sub["outcome"] == "win").mean() * 100
        pos_r = sub.loc[sub["r"] > 0, "r"].sum()
        neg_r = abs(sub.loc[sub["r"] < 0, "r"].sum())
        pf    = pos_r / neg_r if neg_r > 0 else float("inf")
        net_r = sub["r"].sum()
        rpt   = net_r / n
        pct   = n / total * 100
        print(f"  {grp:<22}  {n:>5}  {pct:>6.1f}%  {wr:>6.1f}%  {pf:>6.2f}  {net_r:>+8.1f}R  {rpt:>+8.3f}R")

    print()
    print("  Interpretation guide:")
    print("  orange WR / PF ≈ origin  → orange adds real value, track separately")
    print("  orange WR / PF ≈ yellow  → no extra value beyond 'some level is there'")
    print("  orange WR / PF ≈ none    → orange is noise, ignore it")


for sym, path in SYMBOLS.items():
    run(sym, path)

print("\n\nDone.")
