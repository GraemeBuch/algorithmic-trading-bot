"""
run_all_live_backtest.py — full walk-forward backtest for all 14 live symbols.

Saves updated .cbm model files to data/ and prints a combined portfolio summary.
Run time: ~30-60 minutes depending on hardware.
"""
from __future__ import annotations
import sys, io, warnings, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path

from multi_1min_backtest import run_symbol, portfolio_constraint

DATA = Path("data")

# All 14 live bot symbols — (symbol, 1H csv, 1-min csv)
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

if __name__ == "__main__":
    btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)
    print(f"BTC 4H: {len(btc_df):,} bars loaded")
    print(f"Running full walk-forward backtest for {len(SYMBOLS)} symbols …\n")

    all_oos_parts = []
    for symbol, csv_1h, csv_1m in SYMBOLS:
        result = run_symbol(symbol, csv_1h, csv_1m, btc_df)
        if result is not None:
            all_oos_parts.append(result)

    if all_oos_parts:
        all_oos = pd.concat(all_oos_parts, ignore_index=True)
        portfolio_constraint(all_oos, max_concurrent=5)
    else:
        print("No results — check data files.")

    print("\n\nDone.")
