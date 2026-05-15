"""
find_loss_streaks.py — run full OOS backtest for all 14 symbols,
save the combined trade log, then find the worst full-stop clusters.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path
from multi_1min_backtest import run_symbol, portfolio_constraint

DATA = Path("data")

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

WIN_R  =  1.039
BE_R   = -0.040
STOP_R = -1.040

R_MAP = {"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R}

if __name__ == "__main__":
    btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)

    all_parts = []
    for symbol, csv_1h, csv_1m in SYMBOLS:
        print(f"  {symbol} …", flush=True)
        result = run_symbol(symbol, csv_1h, csv_1m, btc_df)
        if result is not None:
            all_parts.append(result)

    if not all_parts:
        print("No results.")
        sys.exit(1)

    all_oos = pd.concat(all_parts, ignore_index=True)

    # Apply portfolio constraint (max 5 concurrent)
    constrained = portfolio_constraint(all_oos, max_concurrent=5)

    constrained = constrained.sort_values("activation_time").reset_index(drop=True)
    constrained["R"] = constrained["outcome"].map(R_MAP)
    constrained["date"] = pd.to_datetime(constrained["activation_time"]).dt.date

    # Save full log
    constrained.to_csv(DATA / "oos_trade_log.csv", index=False)
    print(f"\nSaved {len(constrained):,} trades → data/oos_trade_log.csv")

    # ── Find worst 10-trade windows by full-stop count ──────────────────────
    print("\n=== WORST 10-TRADE WINDOWS (most full stops) ===")
    worst = []
    for i in range(len(constrained) - 9):
        window = constrained.iloc[i:i+10]
        fs = (window["outcome"] == "full_stop").sum()
        wins = (window["outcome"] == "win").sum()
        be = (window["outcome"] == "be_stop").sum()
        total_r = window["R"].sum()
        worst.append({
            "start": window.iloc[0]["activation_time"],
            "end":   window.iloc[-1]["activation_time"],
            "full_stops": fs,
            "wins": wins,
            "be": be,
            "total_R": round(total_r, 2),
        })

    worst_df = pd.DataFrame(worst).sort_values("full_stops", ascending=False).head(20)
    for _, row in worst_df.iterrows():
        print(f"  {str(row['start'])[:16]} → {str(row['end'])[:16]}  "
              f"FS={row['full_stops']}  W={row['wins']}  BE={row['be']}  "
              f"R={row['total_R']:+.2f}")

    # ── Find worst 7-day periods by total R ─────────────────────────────────
    print("\n=== WORST 7-DAY ROLLING WINDOWS (by R) ===")
    constrained["act_dt"] = pd.to_datetime(constrained["activation_time"])
    constrained_sorted = constrained.set_index("act_dt").sort_index()

    weekly = []
    for i in range(len(constrained_sorted)):
        row = constrained_sorted.iloc[i]
        t_start = constrained_sorted.index[i]
        t_end   = t_start + pd.Timedelta(days=7)
        window  = constrained_sorted.loc[t_start:t_end]
        fs = (window["outcome"] == "full_stop").sum()
        wins = (window["outcome"] == "win").sum()
        total_r = window["R"].sum()
        weekly.append({
            "week_start": t_start.date(),
            "trades": len(window),
            "full_stops": fs,
            "wins": wins,
            "total_R": round(total_r, 2),
        })

    weekly_df = pd.DataFrame(weekly).sort_values("total_R").drop_duplicates("week_start").head(20)
    for _, row in weekly_df.iterrows():
        print(f"  {row['week_start']}  trades={row['trades']}  "
              f"FS={row['full_stops']}  W={row['wins']}  R={row['total_R']:+.2f}")
