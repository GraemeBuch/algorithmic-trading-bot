"""
Quick backtest for candidate coins: NEAR, APTOS, SUI.
Fetches data from Binance, runs raw engulfing strategy (no ML filter),
and reports stats by year so we can see trend and trade volume.
"""
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
ATR_LEN = 14
COST_R = 0.04
MAX_BARS_TO_ACT = 50
MAX_BARS_AFTER_ENTRY = 200

CANDIDATES = [
    ("SHIB/USDT",   "SHIBUSDT",   "shibusdt_1h.csv",   "2020-10-01"),
    ("UNI/USDT",    "UNIUSDT",    "uniusdt_1h.csv",    "2020-09-01"),
    ("ASTR/USDT",   "ASTRUSDT",   "astrusdt_1h.csv",   "2022-01-01"),
    ("AAVE/USDT",   "AAVEUSDT",   "aaveusdt_1h.csv",   "2020-10-01"),
    ("MATIC/USDT",  "MATICUSDT",  "maticusdt_1h.csv",  "2019-04-01"),
    ("RENDER/USDT", "RENDERUSDT", "renderusdt_1h.csv", "2023-01-01"),
    ("KAS/USDT",    "KASUSDT",    "kasusdt_1h.csv",    "2022-06-01"),
]


def fetch_binance(sym_raw, since_str, csv_file):
    path = DATA / csv_file
    if path.exists():
        print(f"  {sym_raw}: using cached {csv_file}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df[["open","high","low","close","volume"]].astype(float).sort_index()

    since_ms = int(pd.Timestamp(since_str).timestamp() * 1000)
    bars = []
    print(f"  Fetching {sym_raw} 1h from Binance …", end="", flush=True)
    while True:
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={sym_raw}&interval=1h&startTime={since_ms}&limit=1000")
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            chunk = r.json()
        except Exception as e:
            print(f"\n  Error: {e}"); return None
        if not chunk: break
        bars.extend(chunk)
        since_ms = chunk[-1][0] + 1
        print(".", end="", flush=True)
        if len(chunk) < 1000: break
        time.sleep(0.15)

    if not bars:
        print("  No data"); return None

    df = pd.DataFrame(bars, columns=["time","open","high","low","close","volume",
                                     "close_time","qav","trades","tbbav","tbqav","ignore"])
    df = df[["time","open","high","low","close","volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.drop_duplicates("time").set_index("time").sort_index()
    df.to_csv(path)
    print(f"  {len(df):,} bars saved")
    return df


def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)


def backtest(df):
    df = df.copy()
    df["atr"] = true_range(df).rolling(ATR_LEN).mean()
    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    idx = df.index
    trades = []

    for i in range(1, len(df)):
        bull = (C[i-1] < O[i-1]) and (C[i] > O[i]) and (C[i] > H[i-1])
        bear = (C[i-1] > O[i-1]) and (C[i] < O[i]) and (C[i] < L[i-1])

        for is_bull in ([True]*int(bull) + [False]*int(bear)):
            rng = H[i-1] - L[i-1]
            if rng <= 0: continue
            atr = float(df["atr"].iloc[i])
            if not np.isfinite(atr) or atr <= 0: continue

            origin = L[i-1] if is_bull else H[i-1]
            entry  = H[i-1] if is_bull else L[i-1]
            m      = 1 if is_bull else -1
            stop   = origin - 0.5 * rng * m
            f1618  = origin + 1.618 * rng * m
            f2618  = origin + 2.618 * rng * m
            risk   = abs(entry - stop)

            act_bar = None
            for j in range(i+1, min(i+1+MAX_BARS_TO_ACT, len(df))):
                if (is_bull and L[j] <= entry) or (not is_bull and H[j] >= entry):
                    act_bar = j; break
            if act_bar is None: continue

            be_hit = False; outcome = None
            for j in range(act_bar+1, min(act_bar+1+MAX_BARS_AFTER_ENTRY, len(df))):
                cur_stop = entry if be_hit else stop
                has_next = (j + 1) < len(df)
                if is_bull:
                    hit_stop  = L[j] <= cur_stop
                    hit_f1618 = H[j] >= f1618
                    hit_f2618 = H[j] >= f2618
                    if not be_hit and hit_f1618: be_hit = True
                    if hit_f2618:
                        if hit_stop and has_next:
                            nxt = O[j+1]
                            outcome = "win" if abs(nxt-f2618) < abs(nxt-cur_stop) else "be_stop"
                        else:
                            outcome = "win"
                        break
                    if hit_stop:
                        if hit_f1618 and has_next:
                            outcome = "be_stop" if O[j+1] > stop else "full_stop"
                        else:
                            outcome = "be_stop" if be_hit else "full_stop"
                        break
                else:
                    hit_stop  = H[j] >= cur_stop
                    hit_f1618 = L[j] <= f1618
                    hit_f2618 = L[j] <= f2618
                    if not be_hit and hit_f1618: be_hit = True
                    if hit_f2618:
                        if hit_stop and has_next:
                            nxt = O[j+1]
                            outcome = "win" if abs(nxt-f2618) < abs(nxt-cur_stop) else "be_stop"
                        else:
                            outcome = "win"
                        break
                    if hit_stop:
                        if hit_f1618 and has_next:
                            outcome = "be_stop" if O[j+1] < stop else "full_stop"
                        else:
                            outcome = "be_stop" if be_hit else "full_stop"
                        break

            if outcome is None: continue

            r = (abs(f2618-entry)/risk - COST_R) if outcome == "win" \
                else (-COST_R if outcome == "be_stop" else -1.0 - COST_R)

            trades.append({
                "year":    idx[act_bar].year,
                "outcome": outcome,
                "r":       r,
            })

    return pd.DataFrame(trades)


def report(name, trades):
    if trades.empty:
        print(f"\n  {name}: no trades\n")
        return

    print(f"\n{'='*58}")
    print(f"  {name}")
    print(f"{'='*58}")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>6} {'BE%':>6} {'SL%':>6} {'TotalR':>8} {'Avg R':>8}")
    print(f"  {'-'*52}")

    years = sorted(trades["year"].unique())
    for y in years:
        t = trades[trades["year"] == y]
        n    = len(t)
        wins = (t["outcome"]=="win").sum()
        be   = (t["outcome"]=="be_stop").sum()
        sl   = (t["outcome"]=="full_stop").sum()
        tot  = t["r"].sum()
        avg  = t["r"].mean()
        print(f"  {y:<6} {n:>7,} {wins/n*100:>5.1f}% {be/n*100:>5.1f}% {sl/n*100:>5.1f}% {tot:>+8.1f}R {avg:>+7.4f}R")

    print(f"  {'-'*52}")
    n    = len(trades)
    wins = (trades["outcome"]=="win").sum()
    be   = (trades["outcome"]=="be_stop").sum()
    sl   = (trades["outcome"]=="full_stop").sum()
    tot  = trades["r"].sum()
    avg  = trades["r"].mean()
    print(f"  {'ALL':<6} {n:>7,} {wins/n*100:>5.1f}% {be/n*100:>5.1f}% {sl/n*100:>5.1f}% {tot:>+8.1f}R {avg:>+7.4f}R")
    print()


for name, sym, csv, since in CANDIDATES:
    print(f"\n{name}")
    df = fetch_binance(sym, since, csv)
    if df is None:
        continue
    trades = backtest(df)
    report(name, trades)
