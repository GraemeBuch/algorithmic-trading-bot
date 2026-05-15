"""Find pockets of profitability in the OFT dataset using simple groupby analysis."""
import pandas as pd
import numpy as np

ds = pd.read_csv("data/oft_dataset.csv", parse_dates=["signal_time"])
print(f"Rows: {len(ds):,}")

# Bin numeric features into quartiles for groupby analysis
ds["atr_pct_bin"]    = pd.qcut(ds["atr_pct"], 4, labels=["Low","MidLo","MidHi","High"], duplicates="drop")
ds["vol_ratio_bin"]  = pd.qcut(ds["vol_ratio"].fillna(1), 4, labels=["Low","MidLo","MidHi","High"], duplicates="drop")
ds["origin_age_bin"] = pd.cut(ds["origin_age_bars"],
                               bins=[-1, 5, 20, 50, 200, 1e9],
                               labels=["VeryFresh","Fresh","Mid","Old","VeryOld"])

def show(group_cols, *, min_count=50):
    print(f"\n— By {' × '.join(group_cols)} (min count {min_count}) —")
    g = ds.groupby(group_cols).agg(
        n=("is_win","count"),
        win_rate=("is_win","mean"),
        avg_R=("r_after_cost","mean"),
    ).round(3)
    g = g[g["n"] >= min_count].sort_values("avg_R", ascending=False)
    print(g.head(20).to_string())

show(["htf_aligned"])
show(["session"])
show(["origin_age_bin"])
show(["atr_pct_bin"])
show(["vol_ratio_bin"])
show(["origin_confirmed_at_signal"])

# Pair-wise — find best 2-feature combos
show(["htf_aligned","session"])
show(["htf_aligned","atr_pct_bin"])
show(["htf_aligned","origin_age_bin"])
show(["origin_age_bin","atr_pct_bin"])
show(["session","atr_pct_bin"])
show(["origin_confirmed_at_signal","htf_aligned"])

# Three-way (only big buckets)
show(["htf_aligned","session","atr_pct_bin"], min_count=80)

print("\n— OVERALL stats —")
print(f"  Total signals : {len(ds):,}")
print(f"  Win rate      : {ds['is_win'].mean():.1%}")
print(f"  Avg R/trade   : {ds['r_after_cost'].mean():+.3f}R")
print(f"  Net total R   : {ds['r_after_cost'].sum():+.1f}R")
print(f"  At 1% risk    : {ds['r_after_cost'].sum()*0.01:+.1%} of starting capital (compounding ignored)")

# Simulated equity curve if you traded every signal
ds_sorted = ds.sort_values("signal_time").reset_index(drop=True)
ds_sorted["cum_R"] = ds_sorted["r_after_cost"].cumsum()
print(f"\n  Best cumulative R reached : {ds_sorted['cum_R'].max():+.1f}R")
print(f"  Worst drawdown from peak  : {(ds_sorted['cum_R'].cummax() - ds_sorted['cum_R']).max():+.1f}R")
