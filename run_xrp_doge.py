"""Run 1-min backtest for XRP and DOGE only (LINK/LTC already cached and done)."""
import sys
sys.path.insert(0, ".")
from pathlib import Path
from multi_1min_backtest import run_symbol, fetch_1min_binance, DATA
import pandas as pd

btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)
print(f"BTC 4H: {len(btc_df):,} bars loaded")

SYMBOLS = [
    ("XRP/USDT",  "xrpusdt_1h.csv",  "XRPUSDT",  "2019-01-01"),
    ("DOGE/USDT", "dogeusdt_1h.csv", "DOGEUSDT",  "2019-01-01"),
]

for symbol, csv_1h, binance_sym, since_str in SYMBOLS:
    run_symbol(symbol, csv_1h, binance_sym, since_str, btc_df)

print("\n\nDone.")
