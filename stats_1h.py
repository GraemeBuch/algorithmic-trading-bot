"""Quick 1H walk-forward stats for SOL and SUI."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from highlander import run_indicator, DIR_UP, DIR_DOWN

DATA             = Path("data")
ATR_LEN          = 14
COST_R           = 0.04
MAX_BARS_TO_ACT  = 50
MAX_BARS_AFTER   = 200
P_THRESHOLD      = 0.50
MIN_TRAIN_MONTHS = 6
STEP_MONTHS      = 3

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

SYMBOLS = {
    "SOL/USDT": "data/sol_1h.csv",
    "SUI/USDT": "data/suiusdt_1h.csv",
}


def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def session_of(h):
    if 0  <= h < 7:  return "Asia"
    if 7  <= h < 13: return "London"
    if 13 <= h < 17: return "NY-AM"
    return "NY-PM"


def load_1h(csv_path):
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df["atr"] = true_range(df).ewm(span=ATR_LEN, adjust=False).mean()
    df["session"] = df.index.hour.map(session_of)
    df["day"] = df.index.day_of_week.astype(str)
    return df


def htf_trend_series(df, htf_len=4):
    htf = df["close"].resample(f"{htf_len}h").last().dropna()
    sma20 = htf.rolling(20).mean()
    sma50 = htf.rolling(50).mean()
    trend = pd.Series("flat", index=htf.index)
    trend[sma20 > sma50] = "up"
    trend[sma20 < sma50] = "down"
    return trend.reindex(df.index, method="ffill")


def btc_trend_series(df):
    sma20 = df["close"].rolling(20).mean()
    sma50 = df["close"].rolling(50).mean()
    t = pd.Series("flat", index=df.index)
    t[sma20 > sma50] = "up"
    t[sma20 < sma50] = "down"
    return t


def backtest_trades(df, levels):
    htf_trend = htf_trend_series(df)
    btc_t     = btc_trend_series(df)

    rows = []
    for lv in levels:
        try:
            eng_i = df.index.get_loc(lv["eng_time"])
        except KeyError:
            continue
        direction = lv["direction"]
        origin = lv["origin"]
        entry  = lv["entry"]
        rng    = abs(entry - origin)
        stop   = entry - rng * 1.5 if direction == DIR_UP else entry + rng * 1.5
        be_lvl = entry + rng * 1.618 if direction == DIR_UP else entry - rng * 1.618
        tp     = entry + rng * 2.618 if direction == DIR_UP else entry - rng * 2.618
        actual_rr = abs(tp - entry) / abs(entry - stop)

        activated = False
        act_i = None
        for j in range(eng_i + 1, min(eng_i + 1 + MAX_BARS_TO_ACT, len(df))):
            bar = df.iloc[j]
            if direction == DIR_UP  and bar["low"]  <= entry:
                activated = True; act_i = j; break
            if direction == DIR_DOWN and bar["high"] >= entry:
                activated = True; act_i = j; break
        if not activated:
            continue

        act_bar = df.iloc[act_i]
        atr_act = act_bar["atr"]
        atr_eng = df.iloc[eng_i]["atr"]

        be_hit  = False
        outcome = None
        for j in range(act_i + 1, min(act_i + 1 + MAX_BARS_AFTER, len(df))):
            bar = df.iloc[j]
            cur_stop = entry if be_hit else stop
            if direction == DIR_UP:
                if not be_hit and bar["high"] >= be_lvl:
                    be_hit = True
                if bar["low"] <= cur_stop:
                    outcome = "be_stop" if be_hit else "full_stop"; break
                if bar["high"] >= tp:
                    outcome = "win"; break
            else:
                if not be_hit and bar["low"] <= be_lvl:
                    be_hit = True
                if bar["high"] >= cur_stop:
                    outcome = "be_stop" if be_hit else "full_stop"; break
                if bar["low"] <= tp:
                    outcome = "win"; break
        if outcome is None:
            continue

        r = (actual_rr - COST_R) if outcome == "win" else (-COST_R if outcome == "be_stop" else -1.0 - COST_R)

        ht = htf_trend.get(lv["eng_time"], "flat")
        bt = btc_t.get(lv["eng_time"], "flat")
        htf_aligned = int(
            (direction == DIR_UP and ht == "up") or
            (direction == DIR_DOWN and ht == "down")
        )
        vol_ratio = act_bar["volume"] / df["volume"].iloc[max(0, act_i - 20):act_i].mean() if act_i > 0 else 1.0

        rows.append({
            "eng_time": lv["eng_time"],
            "direction": "bull" if direction == DIR_UP else "bear",
            "session": df.at[lv["eng_time"], "session"] if lv["eng_time"] in df.index else "London",
            "htf_trend": ht,
            "btc_trend": bt,
            "day": df.index[eng_i].day_of_week,
            "bars_to_activation": act_i - eng_i,
            "atr_pct_at_act": atr_act / entry * 100,
            "atr_pct_at_eng": atr_eng / entry * 100,
            "rng_pct": rng / entry * 100,
            "stop_pct": abs(entry - stop) / entry * 100,
            "tp1618_pct": abs(tp - entry) / entry * 100,
            "rng_to_atr": rng / atr_eng if atr_eng > 0 else 1,
            "vol_ratio": vol_ratio,
            "hour": df.index[eng_i].hour,
            "month": df.index[eng_i].month,
            "htf_aligned": htf_aligned,
            "support_present": int(lv.get("support_present", False)),
            "support_dist_pct": lv.get("support_dist_pct", 0),
            "support_dist_atr": lv.get("support_dist_atr", 0),
            "support_is_origin": int(lv.get("support_is_origin", False)),
            "support_confirmed": int(lv.get("support_confirmed", False)),
            "support_dir_matches": int(lv.get("support_dir_matches", False)),
            "resistance_present": int(lv.get("resistance_present", False)),
            "resistance_dist_pct": lv.get("resistance_dist_pct", 0),
            "resistance_dist_atr": lv.get("resistance_dist_atr", 0),
            "resistance_is_origin": int(lv.get("resistance_is_origin", False)),
            "resistance_confirmed": int(lv.get("resistance_confirmed", False)),
            "confluence_score": lv.get("confluence_score", 0),
            "outcome": outcome,
            "r": r,
            "actual_rr": actual_rr,
        })
    return pd.DataFrame(rows)


def walk_forward(trades_df, label):
    trades_df = trades_df.copy()
    trades_df["eng_time"] = pd.to_datetime(trades_df["eng_time"])
    trades_df = trades_df.sort_values("eng_time").reset_index(drop=True)
    trades_df["day"] = trades_df["day"].astype(str)

    start  = trades_df["eng_time"].min().to_period("M")
    end    = trades_df["eng_time"].max().to_period("M")
    months = pd.period_range(start, end, freq="M")

    results = []
    pos_windows = 0
    total_windows = 0

    i = 0
    while i + MIN_TRAIN_MONTHS + STEP_MONTHS <= len(months):
        train_end  = months[i + MIN_TRAIN_MONTHS - 1].to_timestamp("M")
        test_start = months[i + MIN_TRAIN_MONTHS].to_timestamp()
        test_end   = months[i + MIN_TRAIN_MONTHS + STEP_MONTHS - 1].to_timestamp("M")

        train = trades_df[trades_df["eng_time"] <= train_end]
        test  = trades_df[(trades_df["eng_time"] >= test_start) & (trades_df["eng_time"] <= test_end)]

        if len(train) < 30 or len(test) < 10:
            i += STEP_MONTHS
            continue

        y_train = (train["outcome"] == "win").astype(int)
        model = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                   cat_features=cat_idx, verbose=0, random_seed=42)
        model.fit(train[FEATURES], y_train)
        probs = model.predict_proba(test[FEATURES])[:, 1]

        mask = probs >= P_THRESHOLD
        filt = test[mask].copy()
        filt["p_win"] = probs[mask]

        net_r = filt["r"].sum()
        n     = len(filt)
        wr    = (filt["outcome"] == "win").mean() * 100 if n > 0 else 0
        pos_windows += int(net_r > 0)
        total_windows += 1

        status = "✅" if net_r > 0 else "❌"
        print(f"  {test_start.strftime('%Y-%m-%d')}→{test_end.strftime('%Y-%m-%d')}:  N={n:3d}  WR={wr:.1f}%  net={net_r:+.1f}R  {status}")

        results.append(filt)
        i += STEP_MONTHS

    return pd.concat(results) if results else pd.DataFrame(), pos_windows, total_windows


for sym, csv_path in SYMBOLS.items():
    label = sym.split("/")[0]
    print(f"\n{'█'*60}")
    print(f"  {sym}  1H PIPELINE")
    print(f"{'█'*60}")

    df = load_1h(csv_path)
    print(f"  1H bars: {len(df):,}  ({df.index[0].date()} → {df.index[-1].date()})")

    levels = run_indicator(df)
    print(f"  Highlander levels: {len(levels):,}")

    trades = backtest_trades(df, levels)
    if len(trades) == 0:
        print("  No trades found"); continue

    raw_wr = (trades["outcome"] == "win").mean() * 100
    print(f"  Raw (no filter): win {raw_wr:.1f}%  trades {len(trades)}")

    print(f"\n  Walk-forward validation …")
    wf, pos_w, tot_w = walk_forward(trades, label)

    if len(wf) == 0:
        print("  No walk-forward results"); continue

    wins      = wf["outcome"] == "win"
    be_stops  = wf["outcome"] == "be_stop"
    full_stops = wf["outcome"] == "full_stop"
    n_total   = len(wf)
    n_wins    = wins.sum()
    n_be      = be_stops.sum()
    n_fs      = full_stops.sum()
    wr        = n_wins / n_total * 100

    pos_r  = wf.loc[wins, "r"].sum()
    neg_r  = abs(wf.loc[~wins, "r"].sum())
    pf     = pos_r / neg_r if neg_r > 0 else float("inf")
    net_r  = wf["r"].sum()

    start_dt    = pd.to_datetime(wf["eng_time"].min())
    end_dt      = pd.to_datetime(wf["eng_time"].max())
    n_months    = (end_dt - start_dt).days / 30.44
    per_month   = n_total / n_months
    r_per_month = net_r / n_months
    actual_rr   = wf["actual_rr"].mean()

    print(f"\n{'═'*60}")
    print(f"  FINAL RESULTS — {sym} 1H")
    print(f"{'═'*60}")
    print(f"  Signals/month    : {per_month:.1f}  ({n_total} total, {n_months:.0f} months)")
    print(f"  Win rate         : {wr:.1f}%")
    print(f"  Full stops       : {n_fs/n_total*100:.1f}%  ({n_fs})")
    print(f"  BE stops         : {n_be/n_total*100:.1f}%  ({n_be})")
    print(f"  Avg actual RR    : 1 : {actual_rr:.3f}")
    print(f"  Profit factor    : {pf:.2f}")
    print(f"  Net R / month    : {r_per_month:.1f}R  (actual)")
    print(f"  At $130 / trade  : ${r_per_month * 130:,.0f}/month")
    print(f"  At $195 / trade  : ${r_per_month * 195:,.0f}/month")
    print(f"  Positive windows : {pos_w}/{tot_w}")
    print(f"  Combined est     : see below")

print("\n\nAll done.")
