"""Stage 2: simulate Origin First Touch trades using pre-computed indicator state."""
import pickle
import numpy as np
import pandas as pd

DIR_UP = 1; DIR_DOWN = -1

ATR_LEN          = 14
STOP_ATR_MULT    = 1.5
TP_R_MULTIPLE    = 2.0
MAX_BARS_OPEN    = 96
COST_PER_TRADE_R = 0.04
MIN_BARS_BEFORE  = 24

OUT_PATH = "data/oft_dataset.csv"

# ─── Load data ───────────────────────────────────────────
print("Loading data...")
sol = pd.read_csv("data/sol_1h.csv")
sol["time"] = pd.to_datetime(sol["time"])
sol = sol.set_index("time").sort_index()
sol = sol[["open", "high", "low", "close", "volume"]].astype(float)

btc = pd.read_csv("data/btc_4h.csv")
btc["time"] = pd.to_datetime(btc["time"])
btc = btc.set_index("time").sort_index()

with open("data/highlander_state.pkl", "rb") as f:
    state = pickle.load(f)
levels  = state["levels"]
events  = state["events"]
level_by_id = {l["id"]: l for l in levels}

ev_df = pd.DataFrame(events)
oft   = ev_df[ev_df["event"] == "origin_first_touch"].copy()
print(f"  {len(oft):,} origin_first_touch events to simulate")

# ─── Indicators ──────────────────────────────────────────
def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

sol["atr14"]     = true_range(sol).rolling(ATR_LEN).mean()
sol["atr_pct"]   = sol["atr14"] / sol["close"]
sol["vol_avg20"] = sol["volume"].rolling(20).mean()
sol["vol_ratio"] = sol["volume"] / sol["vol_avg20"]

sol_4h = sol[["close"]].resample("4h").agg({"close": "last"}).dropna()
sol_4h["sma20"] = sol_4h["close"].rolling(20).mean()
sol_4h["sma50"] = sol_4h["close"].rolling(50).mean()
sol_4h["htf_trend"] = np.where(sol_4h["sma20"] > sol_4h["sma50"], 1,
                       np.where(sol_4h["sma20"] < sol_4h["sma50"], -1, 0))

btc["sma20"] = btc["close"].rolling(20).mean()
btc["sma50"] = btc["close"].rolling(50).mean()
btc["btc_trend"] = np.where(btc["sma20"] > btc["sma50"], 1,
                    np.where(btc["sma20"] < btc["sma50"], -1, 0))

# ─── Vectorized lookups for HTF/BTC ──────────────────────
def label_trend(v, pos="Bullish", neg="Bearish", zero="Range"):
    return np.where(v == 1, pos, np.where(v == -1, neg, zero))

# build aligned series of trend at any 1H timestamp by forward-fill
sol["htf_idx"] = sol_4h.index.searchsorted(sol.index, side="right") - 1
sol["htf_trend"] = np.where(sol["htf_idx"] < 0, 0,
    sol_4h["htf_trend"].to_numpy()[np.clip(sol["htf_idx"], 0, len(sol_4h)-1)])
sol["htf_label"] = label_trend(sol["htf_trend"], "Bullish", "Bearish", "Range")

sol["btc_idx"] = btc.index.searchsorted(sol.index, side="right") - 1
sol["btc_trend"] = np.where(sol["btc_idx"] < 0, 0,
    btc["btc_trend"].to_numpy()[np.clip(sol["btc_idx"], 0, len(btc)-1)])
sol["btc_label"] = label_trend(sol["btc_trend"], "Up", "Down", "Range")

# ─── Vectorized simulator ────────────────────────────────
sol_h_arr = sol["high"].to_numpy()
sol_l_arr = sol["low"].to_numpy()
sol_c_arr = sol["close"].to_numpy()
N = len(sol)

def simulate_vec(entry_bar, direction, entry_px, stop_px, tp_px):
    end = min(entry_bar + 1 + MAX_BARS_OPEN, N)
    if end <= entry_bar + 1:
        return None
    h_slice = sol_h_arr[entry_bar+1:end]
    l_slice = sol_l_arr[entry_bar+1:end]

    if direction == DIR_UP:
        stop_arr = l_slice <= stop_px
        tp_arr   = h_slice >= tp_px
    else:
        stop_arr = h_slice >= stop_px
        tp_arr   = l_slice <= tp_px

    stop_idx = np.argmax(stop_arr) if stop_arr.any() else len(stop_arr)
    tp_idx   = np.argmax(tp_arr)   if tp_arr.any()   else len(tp_arr)

    if stop_arr.any() or tp_arr.any():
        # Conservative: ambiguous bar (both hit same bar) → assume stop
        if stop_idx <= tp_idx:
            return {"exit_bar": entry_bar + 1 + stop_idx, "exit_price": stop_px,
                    "outcome": "Loss", "r": -1.0,
                    "bars_held": 1 + stop_idx,
                    "ambiguous": stop_arr[stop_idx] and tp_arr[stop_idx] if stop_idx < len(stop_arr) else False}
        else:
            return {"exit_bar": entry_bar + 1 + tp_idx, "exit_price": tp_px,
                    "outcome": "Win", "r": TP_R_MULTIPLE,
                    "bars_held": 1 + tp_idx, "ambiguous": False}

    # Timeout
    last_bar = end - 1
    last_close = sol_c_arr[last_bar]
    risk_per_unit = abs(entry_px - stop_px)
    r = (last_close - entry_px) / risk_per_unit if direction == DIR_UP else (entry_px - last_close) / risk_per_unit
    return {"exit_bar": last_bar, "exit_price": last_close,
            "outcome": "Win" if r > 0 else "Loss" if r < 0 else "Breakeven",
            "r": float(r), "bars_held": last_bar - entry_bar, "ambiguous": False}


# ─── Build dataset ───────────────────────────────────────
print("Simulating outcomes...")
records = []
skipped = {"early": 0, "no_atr": 0, "no_risk": 0}

atr_arr      = sol["atr14"].to_numpy()
atr_pct_arr  = sol["atr_pct"].to_numpy()
vol_ratio    = sol["vol_ratio"].to_numpy()
htf_label    = sol["htf_label"].to_numpy()
btc_label    = sol["btc_label"].to_numpy()
sol_index    = sol.index

for _, row in oft.iterrows():
    bar = int(row["bar"])
    if bar < MIN_BARS_BEFORE:
        skipped["early"] += 1; continue

    direction = int(row["direction"])
    level_px  = float(row["price"])
    atr       = atr_arr[bar]
    if not np.isfinite(atr) or atr <= 0:
        skipped["no_atr"] += 1; continue

    entry_px = sol_c_arr[bar]
    if direction == DIR_UP:
        stop_px = level_px - STOP_ATR_MULT * atr
        risk    = entry_px - stop_px
        tp_px   = entry_px + TP_R_MULTIPLE * risk
    else:
        stop_px = level_px + STOP_ATR_MULT * atr
        risk    = stop_px - entry_px
        tp_px   = entry_px - TP_R_MULTIPLE * risk

    if risk <= 0:
        skipped["no_risk"] += 1; continue

    sim = simulate_vec(bar, direction, entry_px, stop_px, tp_px)
    if sim is None:
        skipped["no_risk"] += 1; continue

    ts = sol_index[bar]
    direction_label = "Long" if direction == DIR_UP else "Short"
    htf = str(htf_label[bar])
    btc_t = str(btc_label[bar])
    htf_aligned = (
        (direction_label == "Long" and htf == "Bullish") or
        (direction_label == "Short" and htf == "Bearish")
    )

    h = ts.hour
    if 0 <= h < 7:    session = "Asia"
    elif 7 <= h < 13: session = "London"
    elif 13 <= h < 17:session = "NY-AM"
    elif 17 <= h < 22:session = "NY-PM"
    else:             session = "Off-hours"

    lid = int(row["level_id"])
    lvl = level_by_id[lid]
    sim_r_after_cost = sim["r"] - COST_PER_TRADE_R

    records.append({
        "level_id":     lid,
        "signal_time":  ts,
        "exit_time":    sol_index[sim["exit_bar"]],
        "bars_held":    int(sim["bars_held"]),
        "direction":    direction_label,
        "entry":        round(entry_px, 6),
        "stop":         round(stop_px, 6),
        "tp":           round(tp_px, 6),
        "exit_price":   round(sim["exit_price"], 6),
        "outcome":      sim["outcome"],
        "r_raw":        round(sim["r"], 3),
        "r_after_cost": round(sim_r_after_cost, 3),
        "is_win":       int(sim["outcome"] == "Win"),
        "atr_pct":      round(float(atr_pct_arr[bar]), 5),
        "atr_value":    round(float(atr), 4),
        "vol_ratio":    round(float(vol_ratio[bar]), 3) if np.isfinite(vol_ratio[bar]) else np.nan,
        "hour":         int(ts.hour),
        "day_of_week":  ts.day_name()[:3],
        "month":        int(ts.month),
        "year":         int(ts.year),
        "session":      session,
        "htf_trend":    htf,
        "htf_aligned":  int(htf_aligned),
        "btc_trend":    btc_t,
        "level_age_bars":  bar - int(lvl["created_bar"]),
        "origin_age_bars": bar - int(lvl.get("origin_bar") or bar),
        "origin_confirmed_at_signal": int(lvl["confirmed"]),
    })

ds = pd.DataFrame.from_records(records)
print(f"  signals processed: {len(ds):,}  (skipped: {skipped})")
ds.to_csv(OUT_PATH, index=False)
print(f"  saved → {OUT_PATH}")

# ─── Summary ────────────────────────────────────────────
print("\n" + "═" * 60)
print("BACKTEST SUMMARY — Origin First Touch / SOL 1H")
print("═" * 60)
print(f"Period:       {ds['signal_time'].min()} → {ds['signal_time'].max()}")
print(f"Total trades: {len(ds):,}")
wins = (ds["outcome"] == "Win").sum()
losses = (ds["outcome"] == "Loss").sum()
print(f"Wins:         {wins:,}  ({wins/len(ds):.1%})")
print(f"Losses:       {losses:,}  ({losses/len(ds):.1%})")
print(f"Other:        {len(ds) - wins - losses}  (timeout/breakeven)")
print(f"Avg R (raw)        : {ds['r_raw'].mean():+.3f}R")
print(f"Avg R (after cost) : {ds['r_after_cost'].mean():+.3f}R")
gp = ds[ds["r_raw"] > 0]["r_raw"].sum()
gl = abs(ds[ds["r_raw"] < 0]["r_raw"].sum())
print(f"Profit factor      : {gp/max(gl,1e-9):.2f}")

print("\n— By HTF alignment —")
print(ds.groupby("htf_aligned")["is_win"].agg(["count","mean"]).round(3))

print("\n— By HTF trend —")
print(ds.groupby("htf_trend")["is_win"].agg(["count","mean"]).round(3))

print("\n— By direction —")
print(ds.groupby("direction")["is_win"].agg(["count","mean"]).round(3))

print("\n— By session —")
print(ds.groupby("session")["is_win"].agg(["count","mean"]).round(3))

print("\n— By BTC trend —")
print(ds.groupby("btc_trend")["is_win"].agg(["count","mean"]).round(3))

print("\n— Avg R after cost by HTF align × direction —")
print(ds.groupby(["htf_aligned","direction"])["r_after_cost"].agg(["count","mean"]).round(3))
