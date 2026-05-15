"""
Run walk-forward on all symbols (existing + FTM/APT) and print a comparison table.
Uses cached CSV data — no network calls needed.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from test_new_symbols import build_trades, add_confluence, walk_forward, COST_R
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")

ALL_SYMBOLS = [
    ("SOL/USDT",  "solusdt_1h.csv"),
    ("SUI/USDT",  "suiusdt_1h.csv"),
    ("XRP/USDT",  "xrpusdt_1h.csv"),
    ("DOGE/USDT", "dogeusdt_1h.csv"),
    ("LTC/USDT",  "ltcusdt_1h.csv"),
    ("AVAX/USDT", "avaxusdt_1h.csv"),
    ("ADA/USDT",  "adausdt_1h.csv"),
    ("XLM/USDT",  "xlmusdt_1h.csv"),
    ("LINK/USDT", "linkusdt_1h.csv"),
    ("HBAR/USDT", "hbarusdt_1h.csv"),
    ("TRX/USDT",  "trxusdt_1h.csv"),
    ("TON/USDT",  "tonusdt_1h.csv"),
    # candidates
    ("FTM/USDT",  "ftmusdt_1h.csv"),
    ("APT/USDT",  "aptusdt_1h.csv"),
]

btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)

results = []

for symbol, csv_file in ALL_SYMBOLS:
    path = DATA / csv_file
    if not path.exists():
        print(f"  MISSING: {csv_file} — skipping")
        continue

    print(f"  {symbol} …", end="", flush=True)
    df = pd.read_csv(path, index_col=0, parse_dates=True).astype(float)
    df.index.name = "time"

    ds = build_trades(df, btc_df)
    ds = add_confluence(ds, df)

    # Suppress per-window output
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        filtered, pos_w, tot_w = walk_forward(ds, symbol)

    if filtered.empty:
        print(" not enough data")
        continue

    n   = len(filtered)
    w   = (filtered["outcome"] == "win").sum()
    be  = (filtered["outcome"] == "be_stop").sum()
    fs  = (filtered["outcome"] == "full_stop").sum()

    # Use actual R values (vary by trade range)
    tot_r  = filtered["r_strategy_D_trail"].sum()
    avg_r  = filtered["r_strategy_D_trail"].mean()

    months = (pd.to_datetime(filtered["activation_time"]).max() -
              pd.to_datetime(filtered["activation_time"]).min()).days / 30.44
    r_pm   = tot_r / months if months > 0 else 0
    sig_pm = n / months if months > 0 else 0

    pos_r  = filtered.loc[filtered["r_strategy_D_trail"] > 0, "r_strategy_D_trail"].sum()
    neg_r  = abs(filtered.loc[filtered["r_strategy_D_trail"] < 0, "r_strategy_D_trail"].sum())
    pf     = pos_r / neg_r if neg_r > 0 else 999.0

    results.append({
        "Symbol":    symbol,
        "Windows":   f"{pos_w}/{tot_w}",
        "Trades":    n,
        "Sig/mo":    round(sig_pm, 1),
        "Win%":      round(w / n * 100, 1),
        "BE%":       round(be / n * 100, 1),
        "SL%":       round(fs / n * 100, 1),
        "PF":        round(pf, 2),
        "R/mo":      round(r_pm, 1),
        "$130/mo":   int(r_pm * 130),
    })
    print(f"  Win={w/n*100:.1f}%  PF={pf:.2f}  {r_pm:+.1f}R/mo  {pos_w}/{tot_w} windows ✓")

# ── Print comparison table ───────────────────────────────────────────────────
res = pd.DataFrame(results).sort_values("R/mo", ascending=False).reset_index(drop=True)

print(f"\n\n{'═'*88}")
print(f"  {'Symbol':<12} {'Windows':>8} {'Trades':>7} {'Sig/mo':>7} {'Win%':>6} {'BE%':>5} {'SL%':>5} {'PF':>5} {'R/mo':>6} {'$/mo':>8}")
print(f"  {'─'*82}")
for _, r in res.iterrows():
    tag = " ◀ NEW" if r["Symbol"] in ("FTM/USDT","APT/USDT") else ""
    print(f"  {r['Symbol']:<12} {r['Windows']:>8} {r['Trades']:>7,} {r['Sig/mo']:>7.1f} "
          f"{r['Win%']:>5.1f}% {r['BE%']:>4.1f}% {r['SL%']:>4.1f}% "
          f"{r['PF']:>5.2f} {r['R/mo']:>+6.1f}R {r['$130/mo']:>7,}{tag}")
print(f"{'═'*88}\n")
