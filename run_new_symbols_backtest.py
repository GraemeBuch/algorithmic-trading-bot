"""
run_new_symbols_backtest.py — 1-min backtest for BTC, ETH, DOT, NEAR.
"""
from __future__ import annotations
import sys, io, warnings, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path

from multi_1min_backtest import run_symbol, portfolio_constraint
from sol_1min_backtest import WIN_R, BE_R, STOP_R

DATA = Path("data")

SYMBOLS = [
    ("BTC/USDT",  "btcusdt_1h.csv",  "btcusdt_1m.csv"),
    ("ETH/USDT",  "ethusdt_1h.csv",  "ethusdt_1m.csv"),
    ("DOT/USDT",  "dotusdt_1h.csv",  "dotusdt_1m.csv"),
    ("NEAR/USDT", "nearusdt_1h.csv", "nearusdt_1m.csv"),
]

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
