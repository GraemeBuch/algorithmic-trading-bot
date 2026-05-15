"""
Drawdown analysis on the out-of-sample test set (2025–2026).

Answers: "How bad does it get before recovering, and how many trades does
recovery take? What does monthly P&L look like?"

Uses saved test predictions — fully out-of-sample, zero leakage.
Compares: take-all baseline vs ML-filtered at P≥0.50.

Run: python analyze_drawdown.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_COL       = "r_strategy_D_trail"
P_THRESHOLD = 0.50


def equity_stats(r_series, label, times=None):
    r = r_series.values
    cum    = r.cumsum()
    peak   = np.maximum.accumulate(cum)
    dd     = cum - peak
    max_dd = dd.min()

    # Find worst drawdown period (by depth)
    trough_idx = int(np.argmin(dd))
    # walk back to find start of that drawdown
    start_idx = trough_idx
    while start_idx > 0 and dd[start_idx - 1] < 0:
        start_idx -= 1
    # walk forward to find recovery
    recover_idx = trough_idx
    while recover_idx < len(cum) - 1 and cum[recover_idx] < peak[trough_idx]:
        recover_idx += 1
    recovered = cum[recover_idx] >= peak[trough_idx]

    pf = (r[r > 0].sum() / abs(r[r <= 0].sum())) if (r <= 0).any() else np.inf

    print(f"\n{'─'*62}")
    print(f"  {label}  (N={len(r):,})")
    print(f"{'─'*62}")
    print(f"  Net R            : {cum[-1]:+.1f}R")
    print(f"  Avg R / trade    : {r.mean():+.4f}R")
    print(f"  Win rate         : {(r > 0).mean():.1%}")
    print(f"  Profit factor    : {pf:.2f}")
    print(f"  Max drawdown     : {max_dd:+.1f}R  "
          f"({trough_idx - start_idx} trades to trough)")
    if times is not None and len(times) > trough_idx:
        t0 = pd.Timestamp(times[start_idx]).date()
        t1 = pd.Timestamp(times[trough_idx]).date()
        t2 = pd.Timestamp(times[recover_idx]).date() if recovered else "not recovered"
        print(f"  Worst DD dates   : {t0} → trough {t1} → recovery {t2}")
    dur = recover_idx - start_idx if recovered else len(r) - start_idx
    print(f"  Worst DD duration: {dur} trades"
          + (" (not yet recovered at period end)" if not recovered else ""))
    print(f"  Longest losing streak: {_longest_losing(r)}")

    return cum, dd


def _longest_losing(r):
    best = cur = 0
    for x in r:
        if x <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# ─── Load ────────────────────────────────────────────────────────────────────
ds = pd.read_csv("data/engulfing_1h_test_predictions.csv",
                 parse_dates=["activation_time"])
ds = ds.sort_values("activation_time").reset_index(drop=True)

mask     = ds["p_win"] >= P_THRESHOLD
filtered = ds[mask].copy().reset_index(drop=True)

print("═" * 62)
print("DRAWDOWN ANALYSIS — Engulfing 1H Test Set (out-of-sample)")
print(f"Period: {ds['activation_time'].min().date()} → "
      f"{ds['activation_time'].max().date()}")
print("═" * 62)

cum_base, dd_base = equity_stats(
    ds[R_COL], "Baseline (all signals)",
    times=ds["activation_time"].values
)
cum_filt, dd_filt = equity_stats(
    filtered[R_COL], f"ML-filtered P≥{P_THRESHOLD}",
    times=filtered["activation_time"].values
)

# ─── Monthly P&L ─────────────────────────────────────────────────────────────
print(f"\n— Monthly P&L — ML-filtered (P≥{P_THRESHOLD}, Strategy D) —")
filtered["ym"] = filtered["activation_time"].dt.to_period("M")
monthly = (
    filtered.groupby("ym")[R_COL]
    .agg(trades="count", net_R="sum", avg_R="mean")
    .round(3)
)
monthly["net_R"] = monthly["net_R"].map("{:+.2f}R".format)
monthly["avg_R"] = monthly["avg_R"].map("{:+.3f}R".format)
print(monthly.to_string())

# ─── Percentile of outcomes ───────────────────────────────────────────────────
print(f"\n— Distribution of trade R (ML-filtered) —")
percs = [5, 10, 25, 50, 75, 90, 95]
vals  = np.percentile(filtered[R_COL], percs)
for p, v in zip(percs, vals):
    print(f"  p{p:3d}: {v:+.3f}R")

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 11))

# 1. Equity curves
ax = axes[0]
ax.plot(ds["activation_time"].values, cum_base,
        label="All signals (baseline)", color="gray", lw=1.0)
ax.plot(filtered["activation_time"].values, cum_filt,
        label=f"ML P≥{P_THRESHOLD}  (N={len(filtered)})", color="C1", lw=1.8)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Cumulative R")
ax.set_title("Out-of-sample equity curves (2025–2026)")
ax.legend(); ax.grid(alpha=0.3)

# 2. Drawdown of filtered
ax = axes[1]
ax.fill_between(filtered["activation_time"].values, dd_filt, 0,
                color="C1", alpha=0.5, label=f"Drawdown (ML P≥{P_THRESHOLD})")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Drawdown (R)")
ax.set_title(f"Drawdown profile — ML-filtered  (max = {dd_filt.min():+.1f}R)")
ax.legend(); ax.grid(alpha=0.3)

# 3. Monthly bar chart
ax = axes[2]
monthly_vals = (
    filtered.groupby("ym")[R_COL].sum().reset_index()
)
monthly_vals["ym_str"] = monthly_vals["ym"].astype(str)
colors = ["C2" if v >= 0 else "C3" for v in monthly_vals[R_COL]]
ax.bar(monthly_vals["ym_str"], monthly_vals[R_COL], color=colors)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Net R")
ax.set_title("Monthly P&L (ML-filtered)")
ax.tick_params(axis="x", rotation=45)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("data/drawdown_analysis.png", dpi=130)
print(f"\nSaved → data/drawdown_analysis.png")
