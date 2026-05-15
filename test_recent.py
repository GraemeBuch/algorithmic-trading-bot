"""
Apply trained ML models to Jan–Apr 2025 for FTM, APT, UNI
and compare against the existing 12 symbols for the same period.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from test_new_symbols import build_trades, add_confluence, FEATURES, CAT_FEATURES, P_THRESHOLD, COST_R
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data")

ALL_SYMBOLS = [
    # existing
    ("SOL/USDT",  "solusdt_1h.csv",  "catboost_clf_solusdt_1h.cbm"),
    ("SUI/USDT",  "suiusdt_1h.csv",  "catboost_clf_suiusdt_1h.cbm"),
    ("XRP/USDT",  "xrpusdt_1h.csv",  "catboost_clf_xrpusdt_1h.cbm"),
    ("DOGE/USDT", "dogeusdt_1h.csv", "catboost_clf_dogeusdt_1h.cbm"),
    ("LTC/USDT",  "ltcusdt_1h.csv",  "catboost_clf_ltcusdt_1h.cbm"),
    ("AVAX/USDT", "avaxusdt_1h.csv", "catboost_clf_avaxusdt_1h.cbm"),
    ("ADA/USDT",  "adausdt_1h.csv",  "catboost_clf_adausdt_1h.cbm"),
    ("XLM/USDT",  "xlmusdt_1h.csv",  "catboost_clf_xlmusdt_1h.cbm"),
    ("LINK/USDT", "linkusdt_1h.csv", "catboost_clf_linkusdt_1h.cbm"),
    ("HBAR/USDT", "hbarusdt_1h.csv", "catboost_clf_hbarusdt_1h.cbm"),
    ("TRX/USDT",  "trxusdt_1h.csv",  "catboost_clf_trxusdt_1h.cbm"),
    ("TON/USDT",  "tonusdt_1h.csv",  "catboost_clf_tonusdt_1h.cbm"),
    # new candidates
    ("FTM/USDT",  "ftmusdt_1h.csv",  "catboost_clf_ftmusdt_1h.cbm"),
    ("APT/USDT",  "aptusdt_1h.csv",  "catboost_clf_aptusdt_1h.cbm"),
    ("UNI/USDT",  "uniusdt_1h.csv",  "catboost_clf_uniusdt_1h.cbm"),
]

START = pd.Timestamp("2025-01-01")
END   = pd.Timestamp("2025-05-01")

btc_df = pd.read_csv(DATA / "btc_4h.csv", parse_dates=["time"]).set_index("time").astype(float)

results = []

for symbol, csv_file, model_file in ALL_SYMBOLS:
    csv_path   = DATA / csv_file
    model_path = DATA / model_file
    if not csv_path.exists() or not model_path.exists():
        print(f"  MISSING: {symbol}")
        continue

    print(f"  {symbol} …", end="", flush=True)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).astype(float)
    df.index.name = "time"

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ds = build_trades(df, btc_df)
        ds = add_confluence(ds, df)

    # Fill & type
    for col in ["support_dist_pct","support_dist_atr","resistance_dist_pct","resistance_dist_atr"]:
        ds[col] = ds[col].fillna(99.0)
    ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)
    for c in CAT_FEATURES:
        ds[c] = ds[c].astype(str)
    ds["activation_time"] = pd.to_datetime(ds["activation_time"])

    # Filter to Jan–Apr 2025
    mask = (ds["activation_time"] >= START) & (ds["activation_time"] < END)
    period = ds[mask].copy().reset_index(drop=True)

    if len(period) < 5:
        print(f"  too few trades ({len(period)})")
        continue

    # Load model and predict
    clf = CatBoostClassifier()
    clf.load_model(str(model_path))
    period["p_win"] = clf.predict_proba(period[FEATURES])[:, 1]

    filtered = period[period["p_win"] >= P_THRESHOLD]
    n = len(filtered)
    if n == 0:
        print(f"  0 signals after filter")
        continue

    w  = (filtered["outcome"] == "win").sum()
    be = (filtered["outcome"] == "be_stop").sum()
    fs = (filtered["outcome"] == "full_stop").sum()

    # R values
    r_vals = filtered["outcome"].map({"win": 1.578, "be_stop": -COST_R, "full_stop": -1.04})
    tot_r  = r_vals.sum()
    avg_r  = r_vals.mean()

    tag = " ◀ NEW" if symbol in ("FTM/USDT","APT/USDT","UNI/USDT") else ""
    print(f"  n={n}  Win={w/n*100:.0f}%  R={tot_r:+.1f}R{tag}")

    results.append({
        "Symbol":  symbol,
        "Signals": n,
        "Win%":    round(w/n*100, 1),
        "BE%":     round(be/n*100, 1),
        "SL%":     round(fs/n*100, 1),
        "Total R": round(tot_r, 1),
        "Avg R":   round(avg_r, 4),
        "New":     symbol in ("FTM/USDT","APT/USDT","UNI/USDT"),
    })

res = pd.DataFrame(results).sort_values("Total R", ascending=False).reset_index(drop=True)

print(f"\n\n{'═'*72}")
print(f"  Jan–Apr 2025  |  ML-filtered signals only")
print(f"{'═'*72}")
print(f"  {'Symbol':<12} {'Sigs':>5} {'Win%':>6} {'BE%':>5} {'SL%':>5} {'Total R':>8} {'Avg R':>8}")
print(f"  {'─'*64}")
for _, r in res.iterrows():
    tag = " ◀" if r["New"] else ""
    print(f"  {r['Symbol']:<12} {r['Signals']:>5} {r['Win%']:>5.1f}% {r['BE%']:>4.1f}% "
          f"{r['SL%']:>4.1f}% {r['Total R']:>+8.1f}R {r['Avg R']:>+8.4f}R{tag}")

old = res[~res["New"]]
new = res[res["New"]]
print(f"  {'─'*64}")
print(f"  {'Existing avg':<12} {old['Signals'].mean():>5.0f} {old['Win%'].mean():>5.1f}% "
      f"{old['BE%'].mean():>4.1f}% {old['SL%'].mean():>4.1f}% "
      f"{old['Total R'].mean():>+8.1f}R {old['Avg R'].mean():>+8.4f}R")
print(f"  {'New avg':<12} {new['Signals'].mean():>5.0f} {new['Win%'].mean():>5.1f}% "
      f"{new['BE%'].mean():>4.1f}% {new['SL%'].mean():>4.1f}% "
      f"{new['Total R'].mean():>+8.1f}R {new['Avg R'].mean():>+8.4f}R")
print(f"{'═'*72}\n")
