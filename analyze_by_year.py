"""
Year-by-year regime analysis — no ML, no data leakage.

Answers: "Does the engulfing strategy have edge in bear/chop years, or only in the
2025-2026 bull period the model was tested on?"

Uses the full enriched dataset and compares:
  - All signals unfiltered
  - Elite manual filter (HTF-aligned + matching confirmed Origin ≤2 ATR)
"""
import numpy as np
import pandas as pd

R_COL = "r_strategy_D_trail"


def max_dd(series):
    cum = series.cumsum().values
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def stats(df, label, *, min_n=0):
    if len(df) < min_n:
        return
    print(f"\n{'─'*64}")
    print(f"  {label}  (N={len(df):,}  period: "
          f"{df['activation_time'].min().date()} → {df['activation_time'].max().date()})")
    print(f"{'─'*64}")

    g = (df.groupby("year")
         .agg(
             n          = (R_COL, "count"),
             win_rate   = (R_COL, lambda x: (x > 0).mean()),
             avg_R      = (R_COL, "mean"),
             net_R      = (R_COL, "sum"),
         )
         .round(3))

    # add max drawdown per year
    g["max_dd"] = (
        df.groupby("year")[R_COL]
        .apply(max_dd)
        .round(2)
    )

    g["win_rate"] = g["win_rate"].map("{:.1%}".format)
    g["avg_R"]    = g["avg_R"].map("{:+.3f}".format)
    g["net_R"]    = g["net_R"].map("{:+.1f}R".format)
    g["max_dd"]   = g["max_dd"].map("{:+.1f}R".format)
    print(g.to_string())

    # overall
    all_R = df[R_COL]
    cum   = all_R.cumsum().values
    peak  = np.maximum.accumulate(cum)
    total_dd = float((cum - peak).min())
    print(f"\n  Overall: net {all_R.sum():+.1f}R  "
          f"avg {all_R.mean():+.3f}R  "
          f"win {(all_R > 0).mean():.1%}  "
          f"max_dd {total_dd:+.1f}R")


def run(csv_path, tf_label):
    print(f"\n{'═'*64}")
    print(f"  YEAR-BY-YEAR REGIME ANALYSIS — Engulfing {tf_label.upper()}")
    print(f"{'═'*64}")

    ds = pd.read_csv(csv_path, parse_dates=["engulf_time", "activation_time"])
    ds = ds.sort_values("activation_time").reset_index(drop=True)
    ds["year"] = ds["activation_time"].dt.year

    elite = (
        (ds["support_dir_matches"] == 1) &
        (ds["support_is_origin"]   == 1) &
        (ds["support_dist_atr"]    <= 2.0) &
        (ds["htf_aligned"]         == 1)
    )

    stats(ds,          "All signals (no filter)")
    stats(ds[elite],   "Elite filter: HTF-aligned + confirmed Origin ≤2 ATR", min_n=10)

    # All 4 exit strategies, raw avg R by year — helps see which regime suits each
    print("\n— Avg R by year across all 4 exit strategies (unfiltered) —")
    cols = ["r_strategy_A_TP1618", "r_strategy_B_TP2618",
            "r_strategy_C_split",  "r_strategy_D_trail"]
    pivot = ds.groupby("year")[cols].mean().round(3)
    pivot.columns = ["A_1618", "B_2618", "C_split", "D_trail"]
    print(pivot.to_string())

    # ATR regime split — key finding from the OFT analysis
    ds["atr_bin"] = pd.qcut(ds["atr_pct_at_act"], 4,
                             labels=["Low", "MidLo", "MidHi", "High"],
                             duplicates="drop")
    print("\n— Avg R by ATR regime (Strategy D, all years) —")
    g = ds.groupby("atr_bin")[R_COL].agg(["count", "mean"]).round(3)
    g.columns = ["n", "avg_R"]
    print(g.to_string())

    print("\n— Avg R: year × ATR regime (n shown, strategy D) —")
    cross = ds.groupby(["year", "atr_bin"])[R_COL].agg(["count", "mean"]).round(3)
    print(cross.to_string())


run("data/engulfing_1h_with_confluence.csv", "1h")
run("data/engulfing_4h_with_confluence.csv", "4h")
