"""
sol_30m_1min_backtest.py — SOL/USDT backtest using 30-min candles for signals,
1-minute data for precise outcome resolution.

30-min bars are resampled from the cached 1-min data (genuine OHLCV).
HTF trend uses 4H SMA20/50.
"""
from __future__ import annotations
import sys, io, warnings, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier

from sol_1min_backtest import (
    resolve_outcome, true_range, session_of,
    WIN_R, BE_R, STOP_R, COST_R, WIN_R_GROSS,
    MAX_ACT_H, MAX_TRADE_H, ATR_LEN,
    _check_break_retest, _has_opposing_level,
)
from highlander import run_indicator, DIR_UP, DIR_DOWN
from test_new_symbols import add_confluence, P_THRESHOLD

DATA = Path("data")

UNSEEN_START     = pd.Timestamp("2025-01-01")
MIN_TRAIN_MONTHS = 6
STEP_MONTHS      = 3

FEATURES_30M = [
    "rng_pct", "stop_pct", "tp1618_pct", "rng_to_atr",
    "body_pct", "upper_wick_pct", "lower_wick_pct", "engulf_ratio",
    "vol_ratio", "atr_pct_at_eng", "atr_pct_at_act",
    "bars_to_activation",
    "htf_trend",
    "btc_trend",
    "support_present", "support_dist_pct", "support_dist_atr",
    "support_is_origin", "support_confirmed", "support_dir_matches",
    "resistance_present", "resistance_dist_pct", "resistance_dist_atr",
    "resistance_is_origin", "resistance_confirmed", "confluence_score",
    "session", "hour", "day", "month",
]
CAT_FEATURES_30M = ["htf_trend", "btc_trend", "session", "day"]


def build_30m_trades(df_30m: pd.DataFrame, df_1m: pd.DataFrame, btc_df: pd.DataFrame,
                     use_pine_filter: bool = True, use_opposing_filter: bool = True) -> pd.DataFrame:

    df = df_30m.copy()
    df["atr"]       = true_range(df).rolling(ATR_LEN).mean()
    df["atr_pct"]   = df["atr"] / df["close"]
    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg"]

    # ── HTF trend: 4H SMA20/50 ────────────────────────────────────────────────
    htf = df[["close"]].resample("4h", label="left").last().dropna()
    htf["sma20"] = htf["close"].rolling(20).mean()
    htf["sma50"] = htf["close"].rolling(50).mean()
    htf["t"] = np.where(htf["sma20"] > htf["sma50"],  1,
               np.where(htf["sma20"] < htf["sma50"], -1, 0))
    hidx = np.clip(htf.index.searchsorted(df.index, side="right") - 1, 0, len(htf) - 1)
    df["htf_label"] = np.where(htf["t"].values[hidx] ==  1, "Bullish",
                      np.where(htf["t"].values[hidx] == -1, "Bearish", "Range"))

    # ── BTC trend ─────────────────────────────────────────────────────────────
    btc = btc_df.copy()
    btc["sma20"] = btc["close"].rolling(20).mean()
    btc["sma50"] = btc["close"].rolling(50).mean()
    btc["t"] = np.where(btc["sma20"] > btc["sma50"],  1,
               np.where(btc["sma20"] < btc["sma50"], -1, 0))
    bidx = np.clip(btc.index.searchsorted(df.index, side="right") - 1, 0, len(btc) - 1)
    df["btc_label"] = np.where(btc["t"].values[bidx] ==  1, "Up",
                      np.where(btc["t"].values[bidx] == -1, "Down", "Range"))

    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values

    # ── Highlander on 30-min ──────────────────────────────────────────────────
    tick = max(float(df["close"].iloc[-1]) * 1e-5, 1e-6)
    _, hl_events = run_indicator(df[["open","high","low","close"]],
                                 min_range_ticks=3.0, tick_size=tick)
    hl_grouped: dict = {}
    for ev in hl_events:
        hl_grouped.setdefault(int(ev.level_id), []).append(ev.__dict__)

    # ── 1-min timestamps ──────────────────────────────────────────────────────
    m1_ts   = df_1m.index.values.astype("datetime64[ns]").astype(np.int64)
    m1_vals = df_1m[["open","high","low","close"]].values

    records = []
    n_filtered_pine = 0
    n_filtered_opp  = 0

    for i in range(1, len(df)):
        bull = (C[i-1] < O[i-1]) and (C[i] > O[i]) and (C[i] > H[i-1])
        bear = (C[i-1] > O[i-1]) and (C[i] < O[i]) and (C[i] < L[i-1])

        for is_bull in ([True] * int(bull) + [False] * int(bear)):
            rng = H[i-1] - L[i-1]
            if rng <= 0: continue
            atr = float(df["atr"].iloc[i])
            if not np.isfinite(atr) or atr <= 0: continue

            origin = L[i-1] if is_bull else H[i-1]
            entry  = H[i-1] if is_bull else L[i-1]
            m_dir  = 1 if is_bull else -1
            stop   = origin - 0.5   * rng * m_dir
            f1618  = origin + 1.618 * rng * m_dir
            f2618  = origin + 2.618 * rng * m_dir
            risk   = abs(entry - stop)

            if use_pine_filter:
                if not _check_break_retest(is_bull, entry, atr, hl_grouped, i):
                    n_filtered_pine += 1
                    continue
            if use_opposing_filter:
                if _has_opposing_level(is_bull, entry, atr, hl_grouped, i):
                    n_filtered_opp += 1
                    continue

            # ── Find activation bar (30-min) ──────────────────────────────────
            act_bar = None
            for j in range(i + 1, min(i + 1 + MAX_ACT_H * 2, len(df))):  # 2× since bars are 30m not 1H
                if (is_bull and L[j] <= entry) or (not is_bull and H[j] >= entry):
                    act_bar = j
                    break
            if act_bar is None:
                continue

            bars_to_act = act_bar - i

            # ── Find exact activation minute ──────────────────────────────────
            act_bar_open_ns = np.int64(df.index[act_bar].value)
            act_bar_end_ns  = np.int64((df.index[act_bar] + pd.Timedelta(minutes=30)).value)
            i_m_start = int(np.searchsorted(m1_ts, act_bar_open_ns, side="left"))
            i_m_end   = int(np.searchsorted(m1_ts, act_bar_end_ns,  side="left"))

            act_min_idx = None
            for j_m in range(i_m_start, i_m_end):
                h1m = m1_vals[j_m, 1]
                l1m = m1_vals[j_m, 2]
                if (is_bull and l1m <= entry) or (not is_bull and h1m >= entry):
                    act_min_idx = j_m
                    break
            if act_min_idx is None:
                if i_m_start < len(m1_ts):
                    act_min_idx = i_m_start
                else:
                    continue

            act_time = pd.Timestamp(m1_ts[act_min_idx], unit="ns")

            # ── Resolve outcome ───────────────────────────────────────────────
            outcome, be_hit = resolve_outcome(
                m1_vals, m1_ts, act_min_idx, entry, stop, f1618, f2618, is_bull
            )
            if outcome is None:
                continue

            r = (WIN_R_GROSS - COST_R) if outcome == "win" \
                else (-COST_R if outcome == "be_stop" else -1.0 - COST_R)

            vr         = float(df["vol_ratio"].iloc[i])
            eng_rng    = H[i] - L[i]
            _er        = eng_rng if eng_rng > 0 else 1e-10
            body_pct       = round(abs(C[i] - O[i])         / _er, 3)
            upper_wick_pct = round((H[i] - max(O[i],C[i]))  / _er, 3)
            lower_wick_pct = round((min(O[i],C[i]) - L[i])  / _er, 3)
            engulf_ratio   = round(eng_rng / rng, 3)

            records.append({
                "engulf_time":        str(df.index[i]),
                "activation_time":    str(act_time),
                "direction":          "Long" if is_bull else "Short",
                "origin": origin, "entry": entry, "stop": stop,
                "fib_1618": f1618, "fib_2618": f2618, "rng": rng,
                "r_strategy_D_trail": round(r, 4),
                "outcome":            outcome,
                "rng_pct":            round(rng / entry * 100, 3),
                "stop_pct":           round(risk / entry * 100, 3),
                "tp1618_pct":         round(abs(f1618 - entry) / entry * 100, 3),
                "bars_to_activation": bars_to_act,
                "atr_pct_at_act":     round(float(df["atr_pct"].iloc[act_bar]), 5),
                "atr_pct_at_eng":     round(float(df["atr_pct"].iloc[i]), 5),
                "htf_trend":          str(df["htf_label"].iloc[act_bar]),
                "btc_trend":          str(df["btc_label"].iloc[act_bar]),
                "session":            session_of(act_time.hour),
                "hour":               int(act_time.hour),
                "day":                act_time.day_name()[:3],
                "month":              int(act_time.month),
                "year":               int(act_time.year),
                "vol_ratio":          round(vr if np.isfinite(vr) else 1.0, 3),
                "rng_to_atr":         round(rng / atr, 3),
                "body_pct":           body_pct,
                "upper_wick_pct":     upper_wick_pct,
                "lower_wick_pct":     lower_wick_pct,
                "engulf_ratio":       engulf_ratio,
            })

    if use_pine_filter or use_opposing_filter:
        print(f"  Pine filter removed    : {n_filtered_pine:,} engulfings")
        print(f"  Opposing filter removed: {n_filtered_opp:,} engulfings")

    ds = pd.DataFrame.from_records(records)
    ds["htf_aligned"] = (
        ((ds["direction"] == "Long")  & (ds["htf_trend"] == "Bullish")) |
        ((ds["direction"] == "Short") & (ds["htf_trend"] == "Bearish"))
    ).astype(int)
    return ds


def walk_forward_30m(ds: pd.DataFrame):
    ds = ds.copy()
    ds["activation_time"] = pd.to_datetime(ds["activation_time"])
    ds = ds.sort_values("activation_time").reset_index(drop=True)

    cat_idx = [FEATURES_30M.index(c) for c in CAT_FEATURES_30M]
    start = ds["activation_time"].min()
    end   = ds["activation_time"].max()
    train_end = start + pd.DateOffset(months=MIN_TRAIN_MONTHS)

    oos_preds = pd.Series(np.nan, index=ds.index)

    while train_end <= end:
        test_end = train_end + pd.DateOffset(months=STEP_MONTHS)
        tr = ds[ds["activation_time"] <  train_end]
        te = ds[(ds["activation_time"] >= train_end) & (ds["activation_time"] < test_end)]
        if len(tr) < 20 or len(te) == 0:
            train_end = test_end
            continue

        X_tr = tr[FEATURES_30M].values.astype(object)
        y_tr = (tr["outcome"] == "win").astype(int).values
        X_te = te[FEATURES_30M].values.astype(object)

        clf = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                 verbose=False, random_seed=42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            clf.fit(X_tr, y_tr, cat_features=cat_idx)

        probs = clf.predict_proba(X_te)[:, 1]
        oos_preds.iloc[te.index] = probs
        train_end = test_end

    ds["p_win_wf"] = oos_preds
    ds = ds.dropna(subset=["p_win_wf"])
    m = ds["p_win_wf"] >= P_THRESHOLD

    filt_n = m.sum()
    filt_r = ds.loc[m, "r_strategy_D_trail"].sum()
    w  = (ds.loc[m, "outcome"] == "win").sum()
    be = (ds.loc[m, "outcome"] == "be_stop").sum()
    fs = (ds.loc[m, "outcome"] == "full_stop").sum()
    n  = filt_n
    span = (ds["activation_time"].max() - ds["activation_time"].min()).days / 30.44
    pos_r = ds.loc[m & (ds["r_strategy_D_trail"] > 0), "r_strategy_D_trail"].sum()
    neg_r = abs(ds.loc[m & (ds["r_strategy_D_trail"] < 0), "r_strategy_D_trail"].sum())
    pf    = pos_r / neg_r if neg_r > 0 else float("inf")

    print(f"\n{'═'*60}")
    print(f"  SOL/USDT  30-min signals + 1-min resolution  |  walk-forward OOS")
    print(f"{'═'*60}")
    print(f"  All signals (post-filter) : {len(ds):,}")
    print(f"  ML filtered               : {filt_n:,}  (P≥{P_THRESHOLD})")
    print(f"  Win: {w}  BE: {be}  Full stop: {fs}")
    if n > 0:
        print(f"  Win rate       : {w/n*100:.1f}%")
        print(f"  Full stop rate : {fs/n*100:.1f}%")
    print(f"  Total R        : {filt_r:.2f}R  over {span:.1f} months")
    print(f"  R/month        : {filt_r/span:.3f}R")
    print(f"  Profit factor  : {pf:.2f}")

    unseen = ds[m & (ds["activation_time"] >= UNSEEN_START)]
    if len(unseen) > 0:
        u_r = unseen["r_strategy_D_trail"].sum()
        u_m = (unseen["activation_time"].max() - unseen["activation_time"].min()).days / 30.44
        print(f"\n  ── Unseen period ({UNSEEN_START.date()} → present) ──")
        print(f"  Trades : {len(unseen)}  R: {u_r:.2f}  R/month: {u_r/max(u_m,0.1):.3f}")

    print(f"\n  {'Window':<22} {'N':>4} {'WR%':>6} {'R':>7} {'PF':>6}")
    print(f"  {'-'*22} {'-'*4} {'-'*6} {'-'*7} {'-'*6}")
    wsize = pd.DateOffset(months=6)
    t = ds["activation_time"].min()
    while t < ds["activation_time"].max():
        wm = m & (ds["activation_time"] >= t) & (ds["activation_time"] < t + wsize)
        wn = wm.sum()
        if wn > 0:
            wr  = (ds.loc[wm, "outcome"] == "win").sum() / wn * 100
            wr_ = ds.loc[wm, "r_strategy_D_trail"].sum()
            wp  = ds.loc[wm & (ds["r_strategy_D_trail"] > 0), "r_strategy_D_trail"].sum()
            wng = abs(ds.loc[wm & (ds["r_strategy_D_trail"] < 0), "r_strategy_D_trail"].sum())
            wpf = wp / wng if wng > 0 else float("inf")
            lbl = f"{t.date()} – {(t+wsize).date()}"
            print(f"  {lbl:<22} {wn:>4} {wr:>6.1f} {wr_:>7.2f} {wpf:>6.2f}")
        t += wsize
    print()


if __name__ == "__main__":
    # ── Load 1-min data and resample to 30-min ────────────────────────────────
    m1_path = None
    for candidate in ["solusdt_1m.csv", "solusdt_1m_bitget.csv"]:
        p = DATA / candidate
        if p.exists():
            m1_path = p
            break
    if m1_path is None:
        print("ERROR: no SOL 1-min cache found — run sol_1min_backtest.py first")
        sys.exit(1)

    df_1m = pd.read_csv(m1_path, index_col=0, parse_dates=True).astype(float)
    df_1m.index.name = "time"
    print(f"SOL 1m : {len(df_1m):,} bars  ({df_1m.index[0].date()} → {df_1m.index[-1].date()})")

    df_30m = (df_1m[["open","high","low","close","volume"]]
              .resample("30min", label="left")
              .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
              .dropna())
    print(f"SOL 30m: {len(df_30m):,} bars  ({df_30m.index[0].date()} → {df_30m.index[-1].date()})")

    btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)
    print(f"BTC 4H : {len(btc_df):,} bars loaded")

    # ── Build trades ──────────────────────────────────────────────────────────
    print(f"\nBuilding trades (30-min signals, 1-min resolution) …")
    ds = build_30m_trades(df_30m, df_1m, btc_df)
    if ds.empty:
        print("No trades found.")
        sys.exit(0)

    w  = (ds["outcome"] == "win").sum()
    be = (ds["outcome"] == "be_stop").sum()
    fs = (ds["outcome"] == "full_stop").sum()
    n  = len(ds)
    print(f"Trades: {n:,}  Raw WR: {w/n*100:.1f}%  BE: {be/n*100:.1f}%  SL: {fs/n*100:.1f}%")

    # ── Confluence ────────────────────────────────────────────────────────────
    print(f"\nAdding confluence features …")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ds = add_confluence(ds, df_30m)

    # ── Walk-forward ──────────────────────────────────────────────────────────
    print(f"\nWalk-forward …")
    walk_forward_30m(ds)

    print("Done.")
