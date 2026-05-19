"""
update_data.py — append missing bars to existing CSVs from Binance.
Updates 1H and 1-min data from where each file left off to now.
"""
import time, requests, pandas as pd
from pathlib import Path

DATA = Path("data")

SYMBOLS = [
    ("BTCUSDT",  "btcusdt_1h.csv",   "btcusdt_1m.csv"),
    ("ETHUSDT",  "ethusdt_1h.csv",   "ethusdt_1m.csv"),
    ("SOLUSDT",  "solusdt_1h.csv",   "solusdt_1m_bitget.csv"),
    ("TRXUSDT",  "trxusdt_1h.csv",   "trxusdt_1m.csv"),
    ("LINKUSDT", "linkusdt_1h.csv",  "linkusdt_1m.csv"),
    ("HBARUSDT", "hbarusdt_1h.csv",  "hbarusdt_1m.csv"),
    ("XRPUSDT",  "xrpusdt_1h.csv",  "xrpusdt_1m.csv"),
    ("DOGEUSDT", "dogeusdt_1h.csv",  "dogeusdt_1m.csv"),
    ("LTCUSDT",  "ltcusdt_1h.csv",   "ltcusdt_1m.csv"),
    ("ADAUSDT",  "adausdt_1h.csv",   "adausdt_1m.csv"),
    ("XLMUSDT",  "xlmusdt_1h.csv",   "xlmusdt_1m.csv"),
    ("UNIUSDT",  "uniusdt_1h.csv",   "uniusdt_1m.csv"),
    ("APTUSDT",  "aptusdt_1h.csv",   "aptusdt_1m.csv"),
    ("NEARUSDT", "nearusdt_1h.csv",  "nearusdt_1m.csv"),
]


def fetch_since(sym, interval, since_ms):
    bars = []
    while True:
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={sym}&interval={interval}&startTime={since_ms}&limit=1000")
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            chunk = r.json()
        except Exception as e:
            print(f"    Error: {e} — retrying")
            time.sleep(5)
            continue
        if not chunk:
            break
        bars.extend(chunk)
        since_ms = chunk[-1][0] + 1
        if len(chunk) < 1000:
            break
        time.sleep(0.12)
    return bars


def update(sym, csv_file, interval):
    path = DATA / csv_file
    if not path.exists():
        print(f"  {sym} {interval}: file missing — skipping")
        return

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    last_ts = df.index[-1]
    since_ms = int((last_ts + pd.Timedelta(minutes=1)).timestamp() * 1000)
    now_ms   = int(pd.Timestamp.utcnow().timestamp() * 1000)

    if since_ms >= now_ms:
        print(f"  {sym} {interval}: already up to date ({last_ts.date()})")
        return

    print(f"  {sym} {interval}: updating from {last_ts.date()} …", end="", flush=True)
    bars = fetch_since(sym, interval, since_ms)

    if not bars:
        print(" no new bars")
        return

    new = pd.DataFrame(bars, columns=["time","open","high","low","close","volume",
                                      "close_time","qav","trades","tbbav","tbqav","ignore"])
    new = new[["time","open","high","low","close","volume"]].astype(float)
    new["time"] = pd.to_datetime(new["time"], unit="ms")
    new = new.set_index("time")
    new.index.name = "time"

    # Drop the last bar (still forming) and any overlap
    new = new.iloc[:-1]
    new = new[new.index > last_ts]

    if new.empty:
        print(" no new complete bars")
        return

    combined = pd.concat([df, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined.to_csv(path)
    print(f" +{len(new):,} bars → now ends {combined.index[-1].date()}")


if __name__ == "__main__":
    for sym, csv_1h, csv_1m in SYMBOLS:
        print(f"\n{sym}")
        update(sym, csv_1h, "1h")
        update(sym, csv_1m, "1m")
    print("\nDone.")
