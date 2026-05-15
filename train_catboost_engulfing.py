"""
Train CatBoost on the enriched engulfing dataset (with Highlander confluence features).

Run on your own machine:
    pip install catboost pandas numpy matplotlib
    python train_catboost_engulfing.py 1h     # or 4h
    python train_catboost_engulfing.py both   # train both timeframes

What it does:
  1. Loads engulfing_{tf}_with_confluence.csv
  2. Chronological 80/20 split (no random shuffle on time-series)
  3. Trains CatBoostClassifier  → P(Win)
  4. Trains CatBoostRegressor   → expected R-multiple
  5. Compares model edge against:
        - Baseline (take all trades)
        - The 'break-retest' filter alone (no ML)
  6. Threshold analysis + equity curve plot
  7. Feature importance
  8. Saves models + test predictions
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor


DATA = Path("data")

# ─── Feature configuration ────────────────────────────────
CAT_FEATURES = [
    "direction", "session", "htf_trend", "btc_trend", "day",
]
NUM_FEATURES = [
    # Pattern features
    "bars_to_activation", "atr_pct_at_act", "atr_pct_at_eng",
    "rng_pct", "stop_pct", "tp1618_pct", "rng_to_atr",
    "vol_ratio",
    # Time features
    "hour", "month",
    # HTF
    "htf_aligned",
    # Confluence (the new gold)
    "support_present", "support_dist_pct", "support_dist_atr",
    "support_is_origin", "support_confirmed", "support_dir_matches",
    "resistance_present", "resistance_dist_pct", "resistance_dist_atr",
    "resistance_is_origin", "resistance_confirmed",
    "confluence_score",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

# Outcome target — Strategy D was best in backtest analysis
TARGET_R    = "r_strategy_D_trail"
TARGET_WIN  = "is_win_D"


def train_for_tf(tf):
    print(f"\n{'═'*68}\nCatBoost training — engulfing × Highlander confluence — {tf.upper()}\n{'═'*68}")

    # ─── Load + clean ───────────────────────────────────────
    csv_path = DATA / f"engulfing_{tf}_with_confluence.csv"
    if not csv_path.exists():
        print(f"  Missing {csv_path} — run engulfing_backtest.py + confluence_backtest.py first")
        return
    ds = pd.read_csv(csv_path, parse_dates=["engulf_time", "activation_time"])
    ds = ds.sort_values("activation_time").reset_index(drop=True)
    print(f"  loaded {len(ds):,} trades  ({ds['activation_time'].min()} → {ds['activation_time'].max()})")

    # Make missing-value handling explicit (CatBoost handles NaN but be deliberate)
    for col in ["support_dist_pct", "support_dist_atr", "resistance_dist_pct", "resistance_dist_atr"]:
        ds[col] = ds[col].fillna(99.0)         # "no level nearby" = 99
    ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)

    for c in CAT_FEATURES:
        ds[c] = ds[c].astype(str)

    ds[TARGET_WIN] = (ds[TARGET_R] > 0).astype(int)

    X = ds[FEATURES].copy()
    y_class = ds[TARGET_WIN]
    y_reg   = ds[TARGET_R]

    # ─── Chronological split ────────────────────────────────
    split_idx = int(len(ds) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_class_train, y_class_test = y_class.iloc[:split_idx], y_class.iloc[split_idx:]
    y_reg_train,   y_reg_test   = y_reg.iloc[:split_idx],   y_reg.iloc[split_idx:]
    test_signals = ds.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"  Train period: {ds['activation_time'].iloc[0]} → {ds['activation_time'].iloc[split_idx-1]}")
    print(f"  Test  period: {ds['activation_time'].iloc[split_idx]} → {ds['activation_time'].iloc[-1]}")

    cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]

    # ─── Classifier — P(Win) ───────────────────────────────
    print("\n[1/2] CatBoostClassifier (P(Win))...")
    clf = CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=6,
        cat_features=cat_idx, eval_metric="AUC",
        early_stopping_rounds=100, verbose=200,
        random_seed=42, l2_leaf_reg=5,
    )
    clf.fit(X_train, y_class_train, eval_set=(X_test, y_class_test), use_best_model=True)
    p_test = clf.predict_proba(X_test)[:, 1]
    test_signals["p_win"] = p_test

    # ─── Regressor — expected R ────────────────────────────
    print("\n[2/2] CatBoostRegressor (E[R])...")
    reg = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=6,
        cat_features=cat_idx, loss_function="MAE",
        early_stopping_rounds=100, verbose=200,
        random_seed=42, l2_leaf_reg=5,
    )
    reg.fit(X_train, y_reg_train, eval_set=(X_test, y_reg_test), use_best_model=True)
    test_signals["pred_R"] = reg.predict(X_test)

    # ─── Feature importances ───────────────────────────────
    fi = pd.DataFrame({"feature": FEATURES, "clf_imp": clf.get_feature_importance(),
                       "reg_imp": reg.get_feature_importance()})
    fi["combined"] = (fi["clf_imp"] + fi["reg_imp"]) / 2
    fi = fi.sort_values("combined", ascending=False)
    print("\n— Feature importance (top 15) —")
    print(fi.head(15).to_string(index=False))

    # ─── Threshold analysis vs baselines ───────────────────
    print("\n— Threshold analysis on TEST set —")
    print(f"{'Threshold':>10} | {'N':>6} | {'WinRate':>8} | {'AvgR':>8} | {'NetR':>8}")
    print("-" * 56)
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        mask = p_test >= t
        n = mask.sum()
        if n == 0: continue
        wr  = y_class_test[mask].mean()
        avg = y_reg_test[mask].mean()
        net = y_reg_test[mask].sum()
        print(f"   ≥ {t:.2f}  | {n:6d} | {wr:7.1%} | {avg:+7.3f} | {net:+7.1f}")

    # Compare against simple filter (break-retest) — align indices via .values
    test_filter = ((test_signals["support_dir_matches"] == 0) &
                   (test_signals["support_dist_atr"] <= 3.0)).values
    y_reg_test_arr = y_reg_test.values
    y_class_test_arr = y_class_test.values
    print(f"\n— Comparison —")
    print(f"  Take ALL test signals : N={len(y_reg_test):4d} avgR={y_reg_test_arr.mean():+.3f} netR={y_reg_test_arr.sum():+.1f}")
    if test_filter.sum() > 0:
        print(f"  Filter only (no ML)   : N={test_filter.sum():4d} avgR={y_reg_test_arr[test_filter].mean():+.3f} netR={y_reg_test_arr[test_filter].sum():+.1f}")
    # Best ML threshold for comparison
    for t in [0.45, 0.50]:
        m = p_test >= t
        if m.sum() > 0:
            print(f"  Model P ≥ {t:.2f}        : N={m.sum():4d} avgR={y_reg_test_arr[m].mean():+.3f} netR={y_reg_test_arr[m].sum():+.1f}")

    # ─── Equity curve ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    ts = test_signals["activation_time"].values
    ax.plot(ts, y_reg_test_arr.cumsum(), label="Take all (baseline)", color="gray", lw=1.0)
    if test_filter.sum() > 0:
        ax.plot(ts[test_filter], y_reg_test_arr[test_filter].cumsum(),
                label=f"Filter only (no ML, N={test_filter.sum()})", color="C0", lw=1.6)
    for t, color in [(0.45, "C5"), (0.50, "C1"), (0.55, "C2"), (0.60, "C3"), (0.65, "C4")]:
        m = p_test >= t
        if m.sum() < 5: continue
        ax.plot(ts[m], y_reg_test_arr[m].cumsum(),
                label=f"Model P≥{t:.2f}  (N={m.sum()})", color=color, lw=1.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Equity curve in R-multiples — {tf.upper()} test set")
    ax.set_ylabel("Cumulative R")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = DATA / f"engulfing_{tf}_equity_curve.png"
    plt.savefig(out_png, dpi=130)
    print(f"\n  Saved equity curve → {out_png}")

    # ─── Save artifacts ────────────────────────────────────
    clf_path = DATA / f"catboost_clf_engulfing_{tf}.cbm"
    reg_path = DATA / f"catboost_reg_engulfing_{tf}.cbm"
    pred_path = DATA / f"engulfing_{tf}_test_predictions.csv"
    clf.save_model(str(clf_path))
    reg.save_model(str(reg_path))
    test_signals.to_csv(pred_path, index=False)
    print(f"  Models   → {clf_path}, {reg_path}")
    print(f"  Test preds → {pred_path}")

    print(f"\n{'═'*68}")
    print(f"  Practical takeaway for {tf.upper()}:")
    print("    1. Find the threshold above which avgR > 0 AND N is large enough")
    print("       (target: at least 50 trades/year on test set).")
    print("    2. In live trading, replicate the same features at signal time")
    print("       and call clf.predict_proba(X_live)[:,1] — only trade if ≥ threshold.")
    print("    3. Re-train every 3-6 months as new data arrives.")
    print(f"{'═'*68}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "1h"
    if which == "both":
        train_for_tf("1h")
        train_for_tf("4h")
    else:
        train_for_tf(which)
