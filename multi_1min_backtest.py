"""
multi_1min_backtest.py — 1-minute resolution backtest for all live bot symbols.

Uses locally cached 1-min data (data/{slug}_1m.csv).
Saves a model per symbol to data/catboost_clf_{slug}_1h.cbm — the same file
the live bot loads — so you can scp the updated models to the server after running.

SOL is handled separately by sol_1min_backtest.py.
AVAX is skipped (no 1-min data available).
"""
from __future__ import annotations
import sys, io, warnings, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path

from sol_1min_backtest import (
    build_trades_1min, add_entry_level_features,
    walk_forward, WIN_R, BE_R, STOP_R,
)
from test_new_symbols import add_confluence, FEATURES, CAT_FEATURES, P_THRESHOLD

DATA = Path("data")

# symbol, 1H csv, 1-min csv
SYMBOLS = [
    ("SUI/USDT",  "suiusdt_1h.csv",  "suiusdt_1m.csv"),
    ("TRX/USDT",  "trxusdt_1h.csv",  "trxusdt_1m.csv"),
    ("LINK/USDT", "linkusdt_1h.csv", "linkusdt_1m.csv"),
    ("HBAR/USDT", "hbarusdt_1h.csv", "hbarusdt_1m.csv"),
    ("TON/USDT",  "tonusdt_1h.csv",  "tonusdt_1m.csv"),
    ("XRP/USDT",  "xrpusdt_1h.csv",  "xrpusdt_1m.csv"),
    ("DOGE/USDT", "dogeusdt_1h.csv", "dogeusdt_1m.csv"),
    ("LTC/USDT",  "ltcusdt_1h.csv",  "ltcusdt_1m.csv"),
    ("ADA/USDT",  "adausdt_1h.csv",  "adausdt_1m.csv"),
    ("XLM/USDT",  "xlmusdt_1h.csv",  "xlmusdt_1m.csv"),
    ("UNI/USDT",  "uniusdt_1h.csv",  "uniusdt_1m.csv"),
    ("APT/USDT",  "aptusdt_1h.csv",  "aptusdt_1m.csv"),
]


def run_symbol(symbol: str, csv_1h: str, csv_1m: str, btc_df: pd.DataFrame):
    slug = symbol.replace("/", "").lower()
    print(f"\n{'█'*65}")
    print(f"  {symbol}  — 1-minute resolution backtest")
    print(f"{'█'*65}")

    # ── Load 1H data ──────────────────────────────────────────────────────────
    path_1h = DATA / csv_1h
    if not path_1h.exists():
        print(f"  MISSING: {path_1h} — skipping")
        return
    df_1h = pd.read_csv(path_1h, index_col=0, parse_dates=True).astype(float)
    df_1h.index.name = "time"
    print(f"  1H  : {len(df_1h):,} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

    # ── Load 1-min data ───────────────────────────────────────────────────────
    path_1m = DATA / csv_1m
    if not path_1m.exists():
        print(f"  MISSING: {path_1m} — skipping")
        return
    df_1m = pd.read_csv(path_1m, index_col=0, parse_dates=True).astype(float)
    df_1m.index.name = "time"
    print(f"  1m  : {len(df_1m):,} bars  ({df_1m.index[0].date()} → {df_1m.index[-1].date()})")

    # ── Build trades ──────────────────────────────────────────────────────────
    print(f"\n  Building trades (1-min resolution) …")
    ds = build_trades_1min(df_1h, df_1m, btc_df)
    if ds.empty:
        print(f"  No trades found — skipping")
        return
    w  = (ds["outcome"] == "win").sum()
    be = (ds["outcome"] == "be_stop").sum()
    fs = (ds["outcome"] == "full_stop").sum()
    n  = len(ds)
    print(f"  Trades: {n:,}  Raw WR: {w/n*100:.1f}%  BE: {be/n*100:.1f}%  SL: {fs/n*100:.1f}%")

    # ── Confluence ────────────────────────────────────────────────────────────
    print(f"\n  Adding confluence features …")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ds = add_confluence(ds, df_1h)

    # ── 4H resample ───────────────────────────────────────────────────────────
    df_4h = (df_1h[["open","high","low","close","volume"]]
             .resample("4h", label="left")
             .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
             .dropna())

    # ── Entry + origin level features ─────────────────────────────────────────
    print(f"\n  Adding entry-level features …")
    ds = add_entry_level_features(ds, df_1h, df_4h)

    # ── Walk-forward + save model ─────────────────────────────────────────────
    model_path = DATA / f"catboost_clf_{slug}_1h.cbm"
    print(f"\n  Walk-forward …")
    oos = walk_forward(
        ds,
        show_feature_importance=False,
        features=FEATURES,
        cat_features_idx=[FEATURES.index(c) for c in CAT_FEATURES],
        symbol=symbol,
        model_save_path=model_path,
    )
    if oos is not None and len(oos) > 0:
        oos = oos.copy()
        oos["symbol"] = symbol
        return oos
    return None


def portfolio_constraint(all_oos: pd.DataFrame, max_concurrent: int = 5):
    """
    Apply live bot execution constraints across all symbols combined:
      1. One position per symbol at a time
      2. Max concurrent open trades across all symbols
    Returns the filtered dataframe and prints a summary.
    """
    from sol_1min_backtest import WIN_R, BE_R, STOP_R

    df = all_oos.sort_values("activation_time").reset_index(drop=True)
    df["activation_time"] = pd.to_datetime(df["activation_time"])
    df["close_time"]      = pd.to_datetime(df["close_time"])

    open_trades = []   # list of (symbol, close_time)
    keep = []

    for _, row in df.iterrows():
        act  = row["activation_time"]
        sym  = row["symbol"]

        # Remove closed trades
        open_trades = [(s, ct) for s, ct in open_trades if ct > act]

        # Check constraints
        sym_active      = any(s == sym for s, _ in open_trades)
        slots_full      = len(open_trades) >= max_concurrent

        if sym_active or slots_full:
            keep.append(False)
        else:
            keep.append(True)
            open_trades.append((sym, row["close_time"]))

    constrained = df[keep].copy()
    n_skipped   = len(df) - len(constrained)

    constrained["r_correct"] = constrained["outcome"].map(
        {"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R})

    n  = len(constrained)
    if n == 0:
        print("  No trades after portfolio constraints")
        return constrained

    w   = (constrained["outcome"] == "win").sum()
    be  = (constrained["outcome"] == "be_stop").sum()
    fs  = (constrained["outcome"] == "full_stop").sum()
    nr  = constrained["r_correct"].sum()
    pos = constrained.loc[constrained["r_correct"] > 0, "r_correct"].sum()
    neg = abs(constrained.loc[constrained["r_correct"] < 0, "r_correct"].sum())
    pf  = pos / neg if neg > 0 else 999
    mspan = (constrained["activation_time"].max() -
             constrained["activation_time"].min()).days / 30.44

    print(f"\n{'█'*65}")
    print(f"  PORTFOLIO  —  all {len(all_oos['symbol'].unique())} symbols combined")
    print(f"  Constraints: 1 position/symbol  +  MAX_CONCURRENT={max_concurrent}")
    print(f"{'█'*65}")
    print(f"  Signals before constraints : {len(df)}")
    print(f"  Signals after constraints  : {n}  ({n_skipped} skipped)")
    print(f"  Date range  : {constrained['activation_time'].min().date()} → "
          f"{constrained['activation_time'].max().date()}")
    print(f"  Trades/month: {n/mspan:.1f}")
    print(f"  Win rate    : {w/n*100:.1f}%")
    print(f"  BE rate     : {be/n*100:.1f}%")
    print(f"  Full stop   : {fs/n*100:.1f}%")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Net R/month  : {nr/mspan:+.2f}R")
    print(f"  At $130/trade: ${nr/mspan*130:,.0f}/month")

    # Per-symbol breakdown
    print(f"\n  {'Symbol':<12} {'N':>5} {'WR%':>6} {'PF':>6} {'R/mo':>7}")
    print(f"  {'─'*42}")
    for sym in sorted(constrained['symbol'].unique()):
        s = constrained[constrained['symbol'] == sym]
        sn  = len(s)
        sw  = (s["outcome"] == "win").sum()
        sp  = s.loc[s["r_correct"] > 0, "r_correct"].sum()
        sng = abs(s.loc[s["r_correct"] < 0, "r_correct"].sum())
        spf = sp / sng if sng > 0 else 999
        sms = (s["activation_time"].max() - s["activation_time"].min()).days / 30.44
        srm = s["r_correct"].sum() / max(sms, 0.1)
        print(f"  {sym:<12} {sn:>5} {sw/sn*100:>5.1f}% {spf:>6.2f} {srm:>+6.2f}R")

    print(f"{'█'*65}")
    return constrained


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)
    print(f"BTC 4H: {len(btc_df):,} bars loaded")

    all_oos_parts = []
    for symbol, csv_1h, csv_1m in SYMBOLS:
        result = run_symbol(symbol, csv_1h, csv_1m, btc_df)
        if result is not None:
            all_oos_parts.append(result)

    if all_oos_parts:
        all_oos = pd.concat(all_oos_parts, ignore_index=True)
        portfolio_constraint(all_oos, max_concurrent=5)

    print("\n\nDone.")
