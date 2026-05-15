"""
monthly_summary.py — per-symbol OOS stats for the last 30 days.
Reads data/oos_trade_log.csv produced by find_loss_streaks.py.
If the file doesn't exist, runs the full backtest first.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA = Path("data")
LOG  = DATA / "oos_trade_log.csv"

WIN_R  =  1.039
BE_R   = -0.040
STOP_R = -1.040
R_MAP  = {"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R}

def run_backtest_and_save():
    from multi_1min_backtest import run_symbol, portfolio_constraint
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
    btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)
    parts = []
    for sym, h1, m1 in SYMBOLS:
        print(f"  {sym} …", flush=True)
        r = run_symbol(sym, h1, m1, btc_df)
        if r is not None:
            parts.append(r)
    if not parts:
        print("No data.")
        sys.exit(1)
    all_oos = pd.concat(parts, ignore_index=True)
    constrained = portfolio_constraint(all_oos, max_concurrent=5)
    constrained["R"] = constrained["outcome"].map(R_MAP)
    constrained.to_csv(LOG, index=False)
    print(f"Saved {len(constrained):,} trades → {LOG}")
    return constrained


def analyse(df: pd.DataFrame, label: str, days: int):
    df = df.copy()
    df["act_dt"] = pd.to_datetime(df["activation_time"])
    cutoff = df["act_dt"].max() - timedelta(days=days)
    recent = df[df["act_dt"] >= cutoff].copy()

    if recent.empty:
        print(f"\nNo trades in last {days} days.")
        return

    print(f"\n{'='*70}")
    print(f"  {label}  —  last {days} days  ({cutoff.date()} → {df['act_dt'].max().date()})")
    print(f"{'='*70}")
    print(f"  {'Symbol':<12} {'Trades':>6} {'Win%':>6} {'BE%':>6} {'Stop%':>6} {'Net R':>8} {'R/month':>8}")
    print(f"  {'-'*64}")

    rows = []
    for sym, grp in recent.groupby("symbol"):
        n     = len(grp)
        wins  = (grp["outcome"] == "win").sum()
        be    = (grp["outcome"] == "be_stop").sum()
        fs    = (grp["outcome"] == "full_stop").sum()
        net_r = grp["R"].sum()
        r_mo  = net_r / days * 30
        rows.append((sym, n, wins, be, fs, net_r, r_mo))

    rows.sort(key=lambda x: x[6], reverse=True)
    total_n = total_w = total_be = total_fs = 0
    total_r = 0.0

    for sym, n, w, be, fs, net_r, r_mo in rows:
        print(f"  {sym:<12} {n:>6} {100*w/n:>5.0f}% {100*be/n:>5.0f}% {100*fs/n:>5.0f}%"
              f" {net_r:>+8.2f}R {r_mo:>+7.2f}R")
        total_n  += n
        total_w  += w
        total_be += be
        total_fs += fs
        total_r  += net_r

    print(f"  {'-'*64}")
    if total_n:
        r_mo_tot = total_r / days * 30
        print(f"  {'TOTAL':<12} {total_n:>6} {100*total_w/total_n:>5.0f}% "
              f"{100*total_be/total_n:>5.0f}% {100*total_fs/total_n:>5.0f}%"
              f" {total_r:>+8.2f}R {r_mo_tot:>+7.2f}R")

    # Day-by-day R for the period
    print(f"\n  Daily R  (last {days} days):")
    recent["date"] = recent["act_dt"].dt.date
    daily = recent.groupby("date")["R"].sum().reset_index()
    daily.columns = ["date", "R"]
    daily["cumR"] = daily["R"].cumsum()
    for _, row in daily.iterrows():
        bar = "█" * int(abs(row["R"]) * 3)
        sign = "+" if row["R"] >= 0 else "-"
        direction = " " if row["R"] >= 0 else "▼"
        print(f"  {row['date']}  {sign}{abs(row['R']):>5.2f}R  cumR={row['cumR']:>+6.2f}R  "
              f"{direction}{bar}")


if __name__ == "__main__":
    if LOG.exists():
        print(f"Reading existing log: {LOG}")
        df = pd.read_csv(LOG)
        df["R"] = df["outcome"].map(R_MAP)
    else:
        print("No saved log found — running full backtest …")
        df = run_backtest_and_save()

    analyse(df, "ALL 14 SYMBOLS — OOS WALK-FORWARD", days=30)
    analyse(df, "ALL 14 SYMBOLS — OOS WALK-FORWARD", days=90)
