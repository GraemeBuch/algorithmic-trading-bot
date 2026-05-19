"""
compare_live_backtest.py

Compares live trades from trade_journal.csv against what the backtest
would have taken over the same period.

Usage:
    python compare_live_backtest.py

Requires:
    - trade_journal.csv in the current directory (scp from server)
    - Up-to-date data CSVs (run download_new_symbols.py first)
    - Saved .cbm model files in data/
"""
from __future__ import annotations
import sys, io, warnings, contextlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from catboost import CatBoostClassifier

from sol_1min_backtest import build_trades_1min, add_entry_level_features, WIN_R, BE_R, STOP_R
from test_new_symbols import add_confluence, FEATURES, CAT_FEATURES, P_THRESHOLD

DATA         = Path("data")
JOURNAL_PATH = Path("trade_journal.csv")
R_MAP        = {"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R}

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


def run_backtest_signals(symbol, csv_1h, csv_1m, btc_df, start_dt, end_dt):
    """Return backtest signals for a symbol within the date window."""
    slug = symbol.replace("/", "").lower()
    model_path = DATA / f"catboost_clf_{slug}_1h.cbm"
    if not model_path.exists():
        return None

    path_1h = DATA / csv_1h
    path_1m = DATA / csv_1m
    if not path_1h.exists() or not path_1m.exists():
        return None

    df_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True).astype(float)
    df_1h.index.name = "time"
    df_1m = pd.read_csv(path_1m, index_col=0, parse_dates=True).astype(float)
    df_1m.index.name = "time"

    data_age = (df_1h.index.max() - df_1m.index.max()).days
    if data_age > 30:
        print(f"  {symbol}: 1m data is {data_age} days old — skipping")
        return None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            ds = build_trades_1min(df_1h, df_1m, btc_df)
        except Exception:
            return None

    if ds is None or ds.empty:
        return None

    with contextlib.redirect_stdout(buf):
        ds = add_confluence(ds, df_1h)
    df_4h = (df_1h[["open","high","low","close","volume"]]
             .resample("4h", label="left")
             .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
             .dropna())
    with contextlib.redirect_stdout(buf):
        ds = add_entry_level_features(ds, df_1h, df_4h)

    for c in CAT_FEATURES:
        if c in ds.columns:
            ds[c] = ds[c].astype(str)
    for col in ["support_dist_pct","support_dist_atr","resistance_dist_pct",
                "resistance_dist_atr","entry_1h_dist_atr","entry_4h_dist_atr"]:
        if col in ds.columns:
            ds[col] = ds[col].fillna(99.0)
    if "vol_ratio" in ds.columns:
        ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)

    clf = CatBoostClassifier()
    clf.load_model(str(model_path))
    cat_idx = [FEATURES.index(c) for c in CAT_FEATURES if c in FEATURES]
    p = clf.predict_proba(ds[FEATURES])[:, 1]
    ds["p_win"] = p
    ds["symbol"] = symbol

    ds = ds[ds["p_win"] >= P_THRESHOLD].copy()
    ds["act_dt"] = pd.to_datetime(ds["activation_time"])
    ds = ds[(ds["act_dt"] >= start_dt) & (ds["act_dt"] <= end_dt)]
    return ds if len(ds) else None


if __name__ == "__main__":
    if not JOURNAL_PATH.exists():
        print("trade_journal.csv not found. Copy it from the server first:")
        print("  scp root@178.104.81.50:/root/bot/trade_journal.csv .")
        sys.exit(1)

    journal = pd.read_csv(JOURNAL_PATH, parse_dates=["date_opened", "date_closed"])
    journal = journal[journal["outcome"] != "open"]  # only closed trades

    if journal.empty:
        print("No closed trades in journal yet.")
        sys.exit(0)

    start_dt = journal["date_opened"].min()
    end_dt   = journal["date_opened"].max()
    print(f"Journal window: {start_dt.date()} → {end_dt.date()}  ({len(journal)} closed trades)")

    # Run backtest for all symbols over the same window
    btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)
    print("Running backtest signals for same window…\n")

    bt_parts = []
    for symbol, csv_1h, csv_1m in SYMBOLS:
        result = run_backtest_signals(symbol, csv_1h, csv_1m, btc_df, start_dt, end_dt)
        if result is not None:
            bt_parts.append(result)
        else:
            print(f"  {symbol}: no signals in window")

    if not bt_parts:
        print("No backtest signals found in this window.")
        sys.exit(1)

    bt = pd.concat(bt_parts, ignore_index=True)
    bt["R"] = bt["outcome"].map(R_MAP)

    # Match live trades to backtest signals on symbol + direction + entry price (within 0.01%)
    print(f"\n{'='*70}")
    print(f"  SIGNAL MATCHING  (entry price tolerance: 0.01%)")
    print(f"{'='*70}")

    matched_live  = []
    matched_bt    = []
    live_only     = []

    for _, live in journal.iterrows():
        sym = live["symbol"]
        bt_sym = bt[bt["symbol"] == sym]
        if bt_sym.empty:
            live_only.append(live)
            continue
        tol = live["entry"] * 0.0001
        match = bt_sym[abs(bt_sym["entry"] - live["entry"]) <= tol]
        if match.empty:
            live_only.append(live)
        else:
            matched_live.append(live)
            matched_bt.append(match.iloc[0])

    bt_only_mask = ~bt.index.isin([m.name for m in matched_bt])
    bt_only = bt[bt_only_mask]

    print(f"\n  Matched (live = backtest signal) : {len(matched_live)}")
    print(f"  Live-only (bot took, backtest didn't): {len(live_only)}")
    print(f"  Backtest-only (backtest took, bot didn't): {len(bt_only)}")

    if matched_live:
        print(f"\n{'='*70}")
        print(f"  OUTCOME COMPARISON (matched signals only)")
        print(f"{'='*70}")
        print(f"  {'Symbol':<12} {'Dir':<6} {'Date':<12} {'Live':<10} {'Backtest':<10} {'Match'}")
        print(f"  {'-'*60}")
        agree = 0
        for live, bt_row in zip(matched_live, matched_bt):
            match_str = "✓" if live["outcome"] == bt_row["outcome"] else "✗"
            if live["outcome"] == bt_row["outcome"]:
                agree += 1
            print(f"  {live['symbol']:<12} {live['direction']:<6} "
                  f"{str(live['date_opened'].date()):<12} "
                  f"{live['outcome']:<10} {bt_row['outcome']:<10} {match_str}")
        pct = 100 * agree / len(matched_live) if matched_live else 0
        print(f"\n  Outcome agreement: {agree}/{len(matched_live)} ({pct:.0f}%)")

    if live_only:
        print(f"\n{'='*70}")
        print(f"  LIVE-ONLY TRADES (bot took these, backtest did not)")
        print(f"{'='*70}")
        for live in live_only:
            print(f"  {live['symbol']:<12} {live['direction']:<6} "
                  f"{str(live['date_opened'].date()):<12} entry={live['entry']}  outcome={live['outcome']}")

    if len(bt_only):
        print(f"\n{'='*70}")
        print(f"  BACKTEST-ONLY SIGNALS (backtest took, bot did not)")
        print(f"{'='*70}")
        for _, row in bt_only.iterrows():
            print(f"  {row['symbol']:<12} {row.get('direction','?'):<6} "
                  f"{str(row['act_dt'].date()):<12} entry={row['entry']:.4f}  outcome={row['outcome']}")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    live_r = journal["r_result"].sum()
    bt_r   = bt["R"].sum() if len(bt) else 0
    print(f"  Live net R  : {live_r:+.2f}R  ({len(journal)} trades)")
    print(f"  Backtest R  : {bt_r:+.2f}R  ({len(bt)} signals in window)")
    print(f"  Signal match: {len(matched_live)}/{len(journal)} live trades confirmed by backtest")
