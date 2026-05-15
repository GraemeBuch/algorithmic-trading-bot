"""
Walk-forward validation — expanding window, 6-month out-of-sample slices.

Answers: "Does the ML model hold up across different market regimes, or did it
only work because the test set happened to be a bull period?"

Method:
  - Minimum 12 months of training data before first test window
  - 6-month out-of-sample test slices (expanding: each window trains on ALL prior data)
  - Re-trains CatBoost from scratch for each slice — no leakage from future bars
  - Reports avgR / win rate at P≥0.50 for each window
  - Produces composite equity curve (stitched OOS windows)

Run: python walk_forward.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

P_THRESHOLD       = 0.50
MIN_TRAIN_MONTHS  = 12
STEP_MONTHS       = 6
R_COL             = "r_strategy_D_trail"

CAT_FEATURES = ["direction", "session", "htf_trend", "btc_trend", "day"]
NUM_FEATURES = [
    "bars_to_activation", "atr_pct_at_act", "atr_pct_at_eng",
    "rng_pct", "stop_pct", "tp1618_pct", "rng_to_atr",
    "vol_ratio", "hour", "month", "htf_aligned",
    "support_present", "support_dist_pct", "support_dist_atr",
    "support_is_origin", "support_confirmed", "support_dir_matches",
    "resistance_present", "resistance_dist_pct", "resistance_dist_atr",
    "resistance_is_origin", "resistance_confirmed", "confluence_score",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
cat_idx  = [FEATURES.index(c) for c in CAT_FEATURES]

# ─── Load & clean ────────────────────────────────────────────────────────────
ds = pd.read_csv("data/engulfing_1h_with_confluence.csv",
                 parse_dates=["engulf_time", "activation_time"])
ds = ds.sort_values("activation_time").reset_index(drop=True)

for col in ["support_dist_pct", "support_dist_atr",
            "resistance_dist_pct", "resistance_dist_atr"]:
    ds[col] = ds[col].fillna(99.0)
ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)
for c in CAT_FEATURES:
    ds[c] = ds[c].astype(str)

ds["is_win_D"] = (ds[R_COL] > 0).astype(int)

t_start = ds["activation_time"].min()
t_end   = ds["activation_time"].max()

print(f"Dataset: {len(ds):,} signals  "
      f"({t_start.date()} → {t_end.date()})")

# ─── Build window schedule ────────────────────────────────────────────────────
windows = []
ts = t_start + pd.DateOffset(months=MIN_TRAIN_MONTHS)
while ts < t_end:
    te = min(ts + pd.DateOffset(months=STEP_MONTHS), t_end)
    windows.append((ts, te))
    ts = te

print(f"Walk-forward windows: {len(windows)}  "
      f"(min train={MIN_TRAIN_MONTHS}m, step={STEP_MONTHS}m)\n")

# ─── Walk-forward loop ────────────────────────────────────────────────────────
results      = []
oos_chunks   = []

for i, (w_start, w_end) in enumerate(windows):
    train = ds[ds["activation_time"] <  w_start]
    test  = ds[(ds["activation_time"] >= w_start) &
               (ds["activation_time"] <  w_end)].copy().reset_index(drop=True)

    if len(train) < 100 or len(test) < 10:
        print(f"  Window {i+1:2d}: skipped (train={len(train)}, test={len(test)})")
        continue

    X_tr = train[FEATURES];  y_tr = train["is_win_D"]
    X_te = test[FEATURES];   y_te = test["is_win_D"]

    clf = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=5,
        cat_features=cat_idx,
        verbose=0, random_seed=42, l2_leaf_reg=5,
    )
    clf.fit(X_tr, y_tr)

    p_win = clf.predict_proba(X_te)[:, 1]
    test["p_win_wf"] = p_win

    m = p_win >= P_THRESHOLD
    baseline_avg = float(test[R_COL].mean())
    filtered_avg = float(test.loc[m, R_COL].mean()) if m.sum() > 0 else np.nan
    filtered_wr  = float(test.loc[m, "is_win_D"].mean()) if m.sum() > 0 else np.nan

    results.append({
        "window":       f"{w_start.date()} → {w_end.date()}",
        "train_n":      len(train),
        "test_n":       len(test),
        "filtered_n":   int(m.sum()),
        "baseline_avgR": round(baseline_avg, 3),
        "model_avgR":   round(filtered_avg, 3) if not np.isnan(filtered_avg) else None,
        "model_winR":   f"{filtered_wr:.1%}" if filtered_wr is not None and not np.isnan(filtered_wr) else None,
        "positive":     "YES" if not np.isnan(filtered_avg) and filtered_avg > 0 else "no",
    })

    print(f"  Window {i+1:2d}: {w_start.date()} → {w_end.date()}  "
          f"train={len(train):4d}  test={len(test):3d}  "
          f"filtered={m.sum():3d}  "
          f"baseline={baseline_avg:+.3f}  "
          f"model={filtered_avg:+.3f}" if not np.isnan(filtered_avg) else
          f"  Window {i+1:2d}: {w_start.date()} → {w_end.date()}  (no filtered signals)")

    oos_chunks.append(test)

# ─── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'═'*72}")
print(f"WALK-FORWARD SUMMARY  (P≥{P_THRESHOLD}, Strategy D, Engulfing 1H)")
print(f"{'═'*72}")
rdf = pd.DataFrame(results)
print(rdf.to_string(index=False))

wins = (rdf["positive"] == "YES").sum()
print(f"\nWindows positive: {wins}/{len(rdf)}")

# ─── Composite equity curve ───────────────────────────────────────────────────
all_oos = pd.concat(oos_chunks).sort_values("activation_time").reset_index(drop=True)
all_oos.to_csv("data/walk_forward_predictions.csv", index=False)

m_all = all_oos["p_win_wf"] >= P_THRESHOLD
print(f"\nComposite OOS — all windows stitched:")
print(f"  Total signals (all): {len(all_oos):,}")
print(f"  Filtered (P≥{P_THRESHOLD}):   {m_all.sum():,}")
print(f"  Baseline  avgR: {all_oos[R_COL].mean():+.3f}  netR: {all_oos[R_COL].sum():+.1f}")
if m_all.sum() > 0:
    print(f"  Filtered  avgR: {all_oos.loc[m_all, R_COL].mean():+.3f}  "
          f"netR: {all_oos.loc[m_all, R_COL].sum():+.1f}  "
          f"win: {all_oos.loc[m_all,'is_win_D'].mean():.1%}")

fig, ax = plt.subplots(figsize=(14, 6))
ts_all  = all_oos["activation_time"].values
ts_filt = all_oos.loc[m_all, "activation_time"].values

ax.plot(ts_all, all_oos[R_COL].cumsum().values,
        label="All signals (naive)", color="gray", lw=1.0)
if m_all.sum() > 0:
    ax.plot(ts_filt,
            all_oos.loc[m_all, R_COL].cumsum().values,
            label=f"Walk-forward model P≥{P_THRESHOLD}  (N={m_all.sum()})",
            color="C1", lw=1.8)

# Shade each window alternately so you can see the regime cuts
for j, (w_start, w_end) in enumerate(windows):
    if j % 2 == 0:
        ax.axvspan(w_start, w_end, alpha=0.04, color="C0")

ax.axhline(0, color="k", lw=0.5)
ax.set_title("Walk-forward equity curve — Engulfing 1H\n"
             "(each window re-trained on all prior data, zero leakage)")
ax.set_ylabel("Cumulative R")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/walk_forward_equity.png", dpi=130)
print("\nSaved → data/walk_forward_equity.png")
print("Saved → data/walk_forward_predictions.csv")
