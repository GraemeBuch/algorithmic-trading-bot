"""
download_new_symbols.py — fetch 1H and 1-min data from Binance for new symbols.
Downloads: BTC, ETH, DOT, NEAR
"""
import time
import requests
import pandas as pd
from pathlib import Path

DATA = Path("data")

SYMBOLS = [
    ("BTCUSDT",  "btcusdt_1h.csv",   "btcusdt_1m.csv",  "2019-01-01"),
    ("ETHUSDT",  "ethusdt_1h.csv",   "ethusdt_1m.csv",  "2019-01-01"),
    ("DOTUSDT",  "dotusdt_1h.csv",   "dotusdt_1m.csv",  "2020-08-20"),
    ("NEARUSDT", "nearusdt_1h.csv",  "nearusdt_1m.csv", "2020-11-01"),
]


def fetch_binance(sym, interval, since_str, out_path):
    path = DATA / out_path
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  {sym} {interval}: cached ({len(df):,} bars, {df.index[0].date()} → {df.index[-1].date()})")
        return df

    since_ms = int(pd.Timestamp(since_str).timestamp() * 1000)
    bars = []
    print(f"  Fetching {sym} {interval} from Binance …", end="", flush=True)
    while True:
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={sym}&interval={interval}&startTime={since_ms}&limit=1000")
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            chunk = r.json()
        except Exception as e:
            print(f"\n  Error: {e}")
            time.sleep(5)
            continue
        if not chunk:
            break
        bars.extend(chunk)
        since_ms = chunk[-1][0] + 1
        if len(bars) % 100000 == 0:
            print(f"\n    {len(bars):,} bars …", end="", flush=True)
        else:
            print(".", end="", flush=True)
        if len(chunk) < 1000:
            break
        time.sleep(0.12)

    if not bars:
        print("  No data returned")
        return None

    df = pd.DataFrame(bars, columns=["time","open","high","low","close","volume",
                                     "close_time","qav","trades","tbbav","tbqav","ignore"])
    df = df[["time","open","high","low","close","volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    df.index.name = "time"
    df.to_csv(path)
    print(f"\n  Saved {len(df):,} bars → {path}")
    return df


if __name__ == "__main__":
    for sym, csv_1h, csv_1m, since in SYMBOLS:
        print(f"\n{'─'*55}")
        print(f"  {sym}")
        print(f"{'─'*55}")
        fetch_binance(sym, "1h", since, csv_1h)
        fetch_binance(sym, "1m", since, csv_1m)
    print("\nAll downloads complete.")
