"""
run_remaining_1min.py — 1-min backtest for all remaining symbols.
Already done: SOL, XRP, DOGE, LINK, LTC
"""
import sys
sys.path.insert(0, ".")
from pathlib import Path
from multi_1min_backtest import run_symbol, DATA
import pandas as pd

btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)
print(f"BTC 4H: {len(btc_df):,} bars loaded")

SYMBOLS = [
    ("SUI/USDT",  "suiusdt_1h.csv",  "SUIUSDT",  "2023-01-01"),
    ("AVAX/USDT", "avaxusdt_1h.csv", "AVAXUSDT", "2020-09-01"),
    ("ADA/USDT",  "adausdt_1h.csv",  "ADAUSDT",  "2019-01-01"),
    ("XLM/USDT",  "xlmusdt_1h.csv",  "XLMUSDT",  "2019-01-01"),
    ("HBAR/USDT", "hbarusdt_1h.csv", "HBARUSDT", "2019-09-01"),
    ("TRX/USDT",  "trxusdt_1h.csv",  "TRXUSDT",  "2019-01-01"),
    ("TON/USDT",  "tonusdt_1h.csv",  "TONUSDT",  "2023-01-01"),
    ("APT/USDT",  "aptusdt_1h.csv",  "APTUSDT",  "2022-10-01"),
    ("UNI/USDT",  "uniusdt_1h.csv",  "UNIUSDT",  "2020-09-01"),
    ("SHIB/USDT", "shibusdt_1h.csv", "SHIBUSDT", "2021-05-01"),
    ("PEPE/USDT", "pepeusdt_1h.csv", "PEPEUSDT", "2023-04-01"),
    ("ASTR/USDT", "astrusdt_1h.csv", "ASTR",     "2022-01-01"),
]

for symbol, csv_1h, binance_sym, since_str in SYMBOLS:
    run_symbol(symbol, csv_1h, binance_sym, since_str, btc_df)

print("\n\nAll done.")
