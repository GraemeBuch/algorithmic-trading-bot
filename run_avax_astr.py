"""Retry AVAX and ASTR 1-min backtests."""
import sys
sys.path.insert(0, ".")
from pathlib import Path
from multi_1min_backtest import run_symbol, DATA
import pandas as pd

btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)
print(f"BTC 4H: {len(btc_df):,} bars loaded")

SYMBOLS = [
    ("AVAX/USDT", "avaxusdt_1h.csv", "AVAXUSDT",  "2020-09-01"),
    ("ASTR/USDT", "astrusdt_1h.csv", "ASTRUSDT",  "2022-01-01"),
]

for symbol, csv_1h, binance_sym, since_str in SYMBOLS:
    run_symbol(symbol, csv_1h, binance_sym, since_str, btc_df)

print("\n\nDone.")
