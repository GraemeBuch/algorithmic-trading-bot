"""
Train a CatBoost model on the OFT (Origin First Touch) dataset.

Usage:
    pip install catboost pandas numpy matplotlib
    python train_catboost.py

What it does:
  1. Loads the dataset (oft_dataset.csv) produced by backtest_oft_v2.py
  2. Splits CHRONOLOGICALLY (last 20% as test) — never random shuffle on time-series
  3. Trains CatBoostClassifier to predict P(Win)
  4. Trains CatBoostRegressor to predict expected R-multiple
  5. Reports feature importance
  6. Threshold analysis: at each P(Win) cutoff, how does win rate / expectancy / # trades change?
  7. Plots equity curve "trade only when P(Win) > threshold" vs naive "trade all"

Run this on your own machine (sandbox here can't install catboost).
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import matplotlib.pyplot as plt


# ─── Load ──────────────────────────────────────────────
DATA = "data/oft_dataset.csv"  # adjust if needed
ds = pd.read_csv(DATA, parse_dates=["signal_time", "exit_time"])
ds = ds.sort_values("signal_time").reset_index(drop=True)
print(f"Loaded {len(ds):,} signals, {ds['signal_time'].min()} → {ds['signal_time'].max()}")

# ─── Features ──────────────────────────────────────────
CAT_FEATURES = [
    "direction", "day_of_week", "session", "htf_trend", "btc_trend",
]
NUM_FEATURES = [
    "atr_pct", "atr_value", "vol_ratio",
    "hour", "month",
    "htf_aligned",
    "level_age_bars", "origin_age_bars", "origin_confirmed_at_signal",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

# leave room for encoding NaN in vol_ratio
ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)
for c in CAT_FEATURES:
    ds[c] = ds[c].astype(str)

X = ds[FEATURES].copy()
y_class = ds["is_win"].astype(int)
y_reg   = ds["r_after_cost"].astype(float)

# ─── Chronological split (CRITICAL — no random!) ──────
split_idx = int(len(ds) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_class_train, y_class_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
y_reg_train,   y_reg_test   = y_reg.iloc[:split_idx],   y_reg.iloc[split_idx:]

print(f"Train: {len(X_train):,}  ({ds['signal_time'].iloc[0]} → {ds['signal_time'].iloc[split_idx-1]})")
print(f"Test : {len(X_test):,}  ({ds['signal_time'].iloc[split_idx]} → {ds['signal_time'].iloc[-1]})")

cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]

# ─── 1. Classifier — P(Win) ───────────────────────────
print("\n[1/2] Training CatBoostClassifier (P(Win))...")
clf = CatBoostClassifier(
    iterations=2000, learning_rate=0.03, depth=6,
    cat_features=cat_idx, eval_metric="AUC",
    early_stopping_rounds=100, verbose=200,
    random_seed=42, l2_leaf_reg=5,
)
clf.fit(X_train, y_class_train, eval_set=(X_test, y_class_test), use_best_model=True)

p_test = clf.predict_proba(X_test)[:, 1]

# ─── 2. Regressor — expected R ────────────────────────
print("\n[2/2] Training CatBoostRegressor (E[R])...")
reg = CatBoostRegressor(
    iterations=2000, learning_rate=0.03, depth=6,
    cat_features=cat_idx, loss_function="MAE",
    early_stopping_rounds=100, verbose=200,
    random_seed=42, l2_leaf_reg=5,
)
reg.fit(X_train, y_reg_train, eval_set=(X_test, y_reg_test), use_best_model=True)
r_pred_test = reg.predict(X_test)

# ─── Feature importances ─────────────────────────────
fi_clf = pd.DataFrame({
    "feature": FEATURES,
    "importance": clf.get_feature_importance()
}).sort_values("importance", ascending=False)
print("\n— Classifier feature importance —")
print(fi_clf.to_string(index=False))

# ─── Threshold analysis ───────────────────────────────
print("\n— Threshold analysis on TEST set —")
print("Threshold |  N trades | Win rate | Avg R  | Total R")
print("-" * 60)
for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    mask = p_test >= t
    n = mask.sum()
    if n == 0:
        continue
    wr = y_class_test[mask].mean()
    rmean = y_reg_test[mask].mean()
    tot = y_reg_test[mask].sum()
    print(f"  ≥ {t:.2f}  |  {n:5d}    | {wr:6.1%}  | {rmean:+.3f}R | {tot:+.1f}R")

print("\n— Baseline (no model, take everything) —")
print(f"  Total: N={len(y_class_test)}, win rate={y_class_test.mean():.1%}, "
      f"avg R={y_reg_test.mean():+.3f}R, total R={y_reg_test.sum():+.1f}R")

# ─── Plot equity curve comparison ────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
test_signals = ds.iloc[split_idx:].copy().reset_index(drop=True)
test_signals["p_win"] = p_test
test_signals["pred_R"] = r_pred_test

# Baseline
ax.plot(test_signals["signal_time"], y_reg_test.cumsum().values,
        label="Take all (baseline)", color="gray", lw=1.2)

# Several thresholds
for t, color in [(0.40, "C0"), (0.45, "C1"), (0.50, "C2"), (0.55, "C3")]:
    mask = test_signals["p_win"] >= t
    if mask.sum() < 5:
        continue
    cum = test_signals.loc[mask, "r_after_cost"].cumsum().values
    times = test_signals.loc[mask, "signal_time"]
    ax.plot(times, cum, label=f"P(Win) ≥ {t}  (N={mask.sum()})", color=color, lw=1.6)

ax.axhline(0, color="k", lw=0.5)
ax.set_title("Equity curve in R-multiples — Test set")
ax.set_ylabel("Cumulative R")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/equity_curve_thresholds.png", dpi=130)
print("\nSaved equity curve → data/equity_curve_thresholds.png")

# ─── Save models + predictions ────────────────────────
clf.save_model("data/catboost_clf.cbm")
reg.save_model("data/catboost_reg.cbm")
test_signals.to_csv("data/oft_test_predictions.csv", index=False)
print("Models saved → data/catboost_clf.cbm, data/catboost_reg.cbm")
print("Test predictions → data/oft_test_predictions.csv")

# ─── Suggested rule of thumb ──────────────────────────
print("\n" + "═" * 60)
print("HOW TO USE THIS")
print("═" * 60)
print("""
1. Find the threshold where avg R goes positive AND you still get
   enough trades (say >50/year) — that's your live filter.
2. In live trading: feature-engineer at signal time the same way,
   call clf.predict_proba(features)[:,1], take trade only if ≥ threshold.
3. Re-train every few months on the latest data — don't let model decay.
4. Always monitor live performance vs backtest. If divergence > 0.5R/trade
   over 50 trades, model is broken or markets shifted.
""")
