"""
recent_stats.py — last-30-day OOS stats for all 14 live symbols.
Uses saved .cbm models (no retraining). Fast: ~5-15 min total.
"""
from __future__ import annotations
import sys, io, warnings, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from datetime import timedelta

from sol_1min_backtest import (
    build_trades_1min, add_entry_level_features, WIN_R, BE_R, STOP_R,
)
from test_new_symbols import add_confluence, FEATURES, CAT_FEATURES, P_THRESHOLD

DATA    = Path("data")
R_MAP   = {"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R}
DAYS_1M = 40   # 1-min window: enough buffer for trades started near the 30-day mark
DAYS_SH = 30   # stats window

SYMBOLS = [
    ("BTC/USDT",  "btcusdt_1h.csv",        "btcusdt_1m.csv"),
    ("ETH/USDT",  "ethusdt_1h.csv",        "ethusdt_1m.csv"),
    ("SOL/USDT",  "solusdt_1h.csv",        "solusdt_1m_bitget.csv"),
    ("TRX/USDT",  "trxusdt_1h.csv",        "trxusdt_1m.csv"),
    ("LINK/USDT", "linkusdt_1h.csv",       "linkusdt_1m.csv"),
    ("HBAR/USDT", "hbarusdt_1h.csv",       "hbarusdt_1m.csv"),
    ("XRP/USDT",  "xrpusdt_1h.csv",        "xrpusdt_1m.csv"),
    ("DOGE/USDT", "dogeusdt_1h.csv",       "dogeusdt_1m.csv"),
    ("LTC/USDT",  "ltcusdt_1h.csv",        "ltcusdt_1m.csv"),
    ("ADA/USDT",  "adausdt_1h.csv",        "adausdt_1m.csv"),
    ("XLM/USDT",  "xlmusdt_1h.csv",        "xlmusdt_1m.csv"),
    ("UNI/USDT",  "uniusdt_1h.csv",        "uniusdt_1m.csv"),
    ("APT/USDT",  "aptusdt_1h.csv",        "aptusdt_1m.csv"),
    ("NEAR/USDT", "nearusdt_1h.csv",       "nearusdt_1m.csv"),
]


def process_symbol(symbol, csv_1h, csv_1m, btc_df):
    slug = symbol.replace("/", "").lower()
    model_path = DATA / f"catboost_clf_{slug}_1h.cbm"

    if not model_path.exists():
        print(f"  {symbol}: no model file — skipping")
        return None

    # Load full 1H (needed for accurate rolling features)
    path_1h = DATA / csv_1h
    if not path_1h.exists():
        print(f"  {symbol}: missing 1H data — skipping")
        return None
    df_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True).astype(float)
    df_1h.index.name = "time"

    # Load full 1-min — needed for correct outcome resolution.
    path_1m = DATA / csv_1m
    if not path_1m.exists():
        print(f"  {symbol}: missing 1m data — skipping")
        return None
    df_1m = pd.read_csv(path_1m, index_col=0, parse_dates=True).astype(float)
    df_1m.index.name = "time"

    # Trim 1H to 120 days so run_indicator is fast (2,880 bars vs 64K).
    # 120 days gives enough warmup for SMA50 on 4H (~8 days needed).
    cutoff_1h = df_1h.index.max() - timedelta(days=120)
    df_1h = df_1h[df_1h.index >= cutoff_1h].copy()

    # Skip if 1-min data is more than 30 days stale
    data_age = (df_1h.index.max() - df_1m.index.max()).days
    if data_age > 30:
        print(f"  {symbol}: 1m data is {data_age} days old — skipping")
        return None

    print(f"  {symbol}: 1H={len(df_1h):,}  1m={len(df_1m):,}", flush=True)

    # Build trades — only signals within the trimmed 1H window
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            ds = build_trades_1min(df_1h, df_1m, btc_df)
        except KeyError:
            print(f"  {symbol}: no trades resolved in window")
            return None

    if ds is None or ds.empty:
        print(f"  {symbol}: no trades resolved")
        return None

    # Add confluence + entry-level features
    with contextlib.redirect_stdout(buf):
        ds = add_confluence(ds, df_1h)
    df_4h = (df_1h[["open","high","low","close","volume"]]
             .resample("4h", label="left")
             .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
             .dropna())
    with contextlib.redirect_stdout(buf):
        ds = add_entry_level_features(ds, df_1h, df_4h)

    # Ensure cat features are strings
    for c in CAT_FEATURES:
        if c in ds.columns:
            ds[c] = ds[c].astype(str)

    # Fill any missing features
    for col in ["support_dist_pct","support_dist_atr","resistance_dist_pct",
                "resistance_dist_atr","entry_1h_dist_atr","entry_4h_dist_atr"]:
        if col in ds.columns:
            ds[col] = ds[col].fillna(99.0)
    if "vol_ratio" in ds.columns:
        ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)

    # Load model and predict
    clf = CatBoostClassifier()
    clf.load_model(str(model_path))
    cat_idx = [FEATURES.index(c) for c in CAT_FEATURES if c in FEATURES]
    p = clf.predict_proba(ds[FEATURES])[:, 1]
    ds["p_win"] = p
    ds["symbol"] = symbol

    return ds[ds["p_win"] >= P_THRESHOLD].copy()


if __name__ == "__main__":
    btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)

    all_parts = []
    for symbol, csv_1h, csv_1m in SYMBOLS:
        result = process_symbol(symbol, csv_1h, csv_1m, btc_df)
        if result is not None and len(result):
            all_parts.append(result)

    if not all_parts:
        print("No trades found.")
        sys.exit(1)

    df = pd.concat(all_parts, ignore_index=True)
    df["act_dt"] = pd.to_datetime(df["activation_time"])
    df["R"]      = df["outcome"].map(R_MAP)

    cutoff = df["act_dt"].max() - timedelta(days=DAYS_SH)
    recent = df[df["act_dt"] >= cutoff].copy()

    print(f"\n{'='*72}")
    print(f"  LAST 30 DAYS  ({cutoff.date()} → {df['act_dt'].max().date()})")
    print(f"  All signals with ML score ≥ {P_THRESHOLD}")
    print(f"{'='*72}")
    print(f"  {'Symbol':<12} {'Trades':>6} {'Win%':>6} {'BE%':>6} {'Stop%':>6} {'Net R':>8} {'R/mo':>7}")
    print(f"  {'-'*60}")

    rows = []
    for sym, grp in recent.groupby("symbol"):
        n   = len(grp)
        w   = (grp["outcome"] == "win").sum()
        be  = (grp["outcome"] == "be_stop").sum()
        fs  = (grp["outcome"] == "full_stop").sum()
        r   = grp["R"].sum()
        rows.append((sym, n, w, be, fs, r))

    rows.sort(key=lambda x: x[5], reverse=True)
    tot_n = tot_w = tot_be = tot_fs = 0
    tot_r = 0.0

    for sym, n, w, be, fs, r in rows:
        rmo = r / DAYS_SH * 30
        print(f"  {sym:<12} {n:>6} {100*w/n:>5.0f}% {100*be/n:>5.0f}% "
              f"{100*fs/n:>5.0f}% {r:>+7.2f}R {rmo:>+6.2f}R")
        tot_n += n; tot_w += w; tot_be += be; tot_fs += fs; tot_r += r

    print(f"  {'-'*60}")
    if tot_n:
        rmo = tot_r / DAYS_SH * 30
        print(f"  {'TOTAL':<12} {tot_n:>6} {100*tot_w/tot_n:>5.0f}% "
              f"{100*tot_be/tot_n:>5.0f}% {100*tot_fs/tot_n:>5.0f}%"
              f" {tot_r:>+7.2f}R {rmo:>+6.2f}R")

    # Day-by-day breakdown
    print(f"\n  Daily R — last 30 days:")
    recent["date"] = recent["act_dt"].dt.date
    daily = recent.groupby("date")["R"].sum().reset_index()
    daily["cumR"] = daily["R"].cumsum()
    for _, row in daily.iterrows():
        bar = "█" * max(1, int(abs(row["R"]) * 4))
        color = "+" if row["R"] >= 0 else "-"
        print(f"  {row['date']}  {color}{abs(row['R']):>5.2f}R  cum={row['cumR']:>+6.2f}R  {bar}")

    print(f"\nTotal trades in window: {tot_n}  Net R: {tot_r:+.2f}R")
