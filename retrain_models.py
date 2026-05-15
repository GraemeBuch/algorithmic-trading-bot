"""
retrain_models.py — Retrain all 15 per-symbol CatBoost models on corrected data.

Corrections vs the old models:
  - Activation candle stop hit → full_stop (next-bar-open resolution), not ignored
  - No be_hit carry-forward from activation candle (f1618 touch on act bar is
    ambiguous — can't know if it happened before or after entry)
  - Outcome loop always starts with be_hit=False

Uses cached CSV data (no re-fetch needed).
Overwrites the existing .cbm files in data/.
"""
import sys, io, warnings, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from test_new_symbols import (
    build_trades, add_confluence, add_entry_level_features,
    walk_forward, train_final_model,
    FEATURES, CAT_FEATURES, P_THRESHOLD, COST_R,
)
import pandas as pd
from pathlib import Path

DATA = Path("data")

ALL_SYMBOLS = [
    ("SOL/USDT",  "solusdt_1h.csv",  "2021-01-01"),
    ("SUI/USDT",  "suiusdt_1h.csv",  "2023-05-01"),
    ("XRP/USDT",  "xrpusdt_1h.csv",  "2019-01-01"),
    ("DOGE/USDT", "dogeusdt_1h.csv", "2019-01-01"),
    ("LTC/USDT",  "ltcusdt_1h.csv",  "2019-01-01"),
    ("AVAX/USDT", "avaxusdt_1h.csv", "2021-01-01"),
    ("ADA/USDT",  "adausdt_1h.csv",  "2018-01-01"),
    ("XLM/USDT",  "xlmusdt_1h.csv",  "2018-01-01"),
    ("LINK/USDT", "linkusdt_1h.csv", "2019-01-01"),
    ("HBAR/USDT", "hbarusdt_1h.csv", "2019-01-01"),
    ("TRX/USDT",  "trxusdt_1h.csv",  "2019-01-01"),
    ("TON/USDT",  "tonusdt_1h.csv",  "2023-01-01"),
    ("FTM/USDT",  "ftmusdt_1h.csv",  "2021-01-01"),
    ("APT/USDT",  "aptusdt_1h.csv",  "2022-10-01"),
    ("UNI/USDT",  "uniusdt_1h.csv",  "2020-09-01"),
]

btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)
print(f"BTC 4H: {len(btc_df):,} bars loaded\n")

for symbol, csv_file, _ in ALL_SYMBOLS:
    csv_path = DATA / csv_file
    if not csv_path.exists():
        print(f"  MISSING CSV: {symbol} ({csv_file}) — skipping")
        continue

    print(f"\n{'█'*60}")
    print(f"  {symbol}")
    print(f"{'█'*60}")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).astype(float)
    df.index.name = "time"
    print(f"  Loaded {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ds = build_trades(df, btc_df)
    raw_wr = (ds["is_win_D"] == 1).mean() * 100
    print(f"  Trades: {len(ds):,}  Raw WR (no filter): {raw_wr:.1f}%")

    print(f"  Adding confluence …")
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        ds = add_confluence(ds, df)

    print(f"  Adding entry-level features …")
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        ds = add_entry_level_features(ds, df)

    print(f"  Walk-forward …")
    walk_forward(ds, symbol)

    print(f"\n  Training final model …")
    train_final_model(ds, symbol)

print("\n\n" + "═"*60)
print("  All models retrained on corrected data.")
print("═"*60)
