"""
Run FTM and APT through the full ML walk-forward pipeline
(same as test_new_symbols.py — build trades, confluence, walk-forward, train final model).
Data already cached from backtest_candidates.py.
"""
import sys
sys.path.insert(0, ".")
from test_new_symbols import (
    build_trades, add_confluence, walk_forward, train_final_model,
    fetch_binance_direct, COST_R
)
import pandas as pd
from pathlib import Path

DATA = Path("data")

CANDIDATES = [
    ("SHIB/USDT", "shibusdt_1h.csv",  "2020-10-01"),
    ("ASTR/USDT", "astrusdt_1h.csv",  "2022-01-01"),
    ("UNI/USDT",  "uniusdt_1h.csv",   "2020-09-01"),
]

# Load BTC 4H for macro trend features
btc_path = DATA / "btc_4h.csv"
print("Loading BTC 4H …")
btc_df = pd.read_csv(btc_path, parse_dates=["time"]).set_index("time").astype(float)
print(f"  {len(btc_df):,} bars ({btc_df.index[0].date()} → {btc_df.index[-1].date()})")

for symbol, csv_file, since in CANDIDATES:
    csv_path = DATA / csv_file

    print(f"\n{'█'*60}")
    print(f"  {symbol}  1H  PIPELINE")
    print(f"{'█'*60}")

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True).astype(float)
        df.index.name = "time"
        print(f"  Loaded {len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")
    else:
        print(f"  Fetching {symbol} …")
        df = fetch_binance_direct(symbol, "1h", since)
        if df is None:
            print(f"  Could not fetch {symbol}, skipping"); continue
        df.index.name = "time"
        df.to_csv(csv_path)

    print(f"\n  Step 1 — Building trades …")
    ds = build_trades(df, btc_df)
    raw_wr = (ds["is_win_D"]==1).mean()*100
    print(f"  Trades: {len(ds):,}   Raw WR (unfiltered): {raw_wr:.1f}%")

    print(f"\n  Step 2 — Adding confluence …")
    ds = add_confluence(ds, df)

    print(f"\n  Step 3 — Walk-forward ML …")
    filtered, pos_w, tot_w = walk_forward(ds, symbol)

    print(f"\n  Step 4 — Training final model …")
    train_final_model(ds, symbol)

print("\n\nDone.")
