"""Scan last 3 days for SOL and SUI — engulfing + confluence + ML filter."""
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostClassifier
from highlander import run_indicator, DIR_UP, DIR_DOWN, STATE_BROKEN_BSUT, STATE_ORIGIN

P_THRESHOLD     = 0.50
ATR_LEN         = 14
MAX_BARS_TO_ACT = 50
COST_R          = 0.04
LOOKBACK_1H     = 600
LOOKBACK_4H     = 200
DAYS_BACK       = 3

SYMBOLS = {
    "SOL/USDT": "data/catboost_clf_engulfing_1h.cbm",
    "SUI/USDT": "data/catboost_clf_suiusdt_1h.cbm",
}

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


def fetch(exchange, symbol, tf, limit):
    raw = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df  = pd.DataFrame(raw, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
    return df.set_index("time").astype(float)


def session_of(h):
    if 0  <= h < 7:  return "Asia"
    if 7  <= h < 13: return "London"
    if 13 <= h < 17: return "NY-AM"
    if 17 <= h < 22: return "NY-PM"
    return "Off-hours"


def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)


def run_scan(symbol, model_path, df1h, df4h, df_btc4h):
    df = df1h.copy()
    df["atr"]     = true_range(df).ewm(span=ATR_LEN, adjust=False).mean()
    df["vol_avg"] = df["volume"].rolling(20).mean()

    # HTF trend (4H SMA20 vs SMA50)
    sma20_4h = df4h["close"].rolling(20).mean()
    sma50_4h = df4h["close"].rolling(50).mean()
    htf_raw  = pd.Series(
        np.where(sma20_4h > sma50_4h, "Bullish",
        np.where(sma20_4h < sma50_4h, "Bearish", "Range")),
        index=df4h.index
    )

    # BTC trend (4H SMA20 vs SMA50)
    sma20_btc = df_btc4h["close"].rolling(20).mean()
    sma50_btc = df_btc4h["close"].rolling(50).mean()
    btc_raw   = pd.Series(
        np.where(sma20_btc > sma50_btc, "Up",
        np.where(sma20_btc < sma50_btc, "Down", "Range")),
        index=df_btc4h.index
    )

    def lookup_trend(series, ts):
        pos = np.clip(series.index.searchsorted(ts, side="right") - 1, 0, len(series)-1)
        return series.iloc[pos]

    # Highlander levels
    levels, events = run_indicator(df[["open","high","low","close"]])

    # Detect engulfing candles
    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    T = df.index

    cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=DAYS_BACK))

    found_any = False
    rows = []

    for i in range(1, len(df) - 1):
        bull = (C[i-1] < O[i-1]) and (C[i] > O[i]) and (C[i] > H[i-1])
        bear = (C[i-1] > O[i-1]) and (C[i] < O[i]) and (C[i] < L[i-1])
        if not (bull or bear):
            continue

        is_bull = bull
        origin  = L[i-1] if is_bull else H[i-1]
        entry   = H[i-1] if is_bull else L[i-1]
        rng     = H[i-1] - L[i-1]
        if rng <= 0:
            continue

        m     = 1 if is_bull else -1
        stop  = origin - 0.5 * rng * m
        f1618 = origin + 1.618 * rng * m
        f2618 = origin + 2.618 * rng * m
        atr_i = float(df["atr"].iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        # Find activation within MAX_BARS_TO_ACT
        act_bar = None
        for j in range(i + 1, min(i + 1 + MAX_BARS_TO_ACT, len(df))):
            if (is_bull and L[j] <= entry) or (not is_bull and H[j] >= entry):
                act_bar = j
                break
        if act_bar is None:
            continue

        act_ts  = T[act_bar]
        atr_act = float(df["atr"].iloc[act_bar])
        vol_avg = float(df["vol_avg"].iloc[i]) if np.isfinite(df["vol_avg"].iloc[i]) else 1
        vol_r   = float(df["volume"].iloc[i]) / vol_avg if vol_avg > 0 else 1.0

        htf_str = lookup_trend(htf_raw, T[i])
        btc_str = lookup_trend(btc_raw, T[i])
        htf_aligned = int(
            (is_bull and htf_str == "Bullish") or
            (not is_bull and htf_str == "Bearish")
        )

        # Confluence
        above, below = [], []
        for lvl in levels:
            if lvl.created_bar > act_bar:
                continue
            if lvl.deleted_bar is not None and lvl.deleted_bar <= act_bar:
                continue
            if lvl.state == STATE_BROKEN_BSUT:
                continue
            diff = lvl.price - entry
            info = dict(
                direction=lvl.dir,
                is_origin=(lvl.state == STATE_ORIGIN),
                confirmed=lvl.confirmed,
                dist_pct=abs(diff) / entry * 100,
                dist_atr=abs(diff) / atr_act if atr_act > 0 else 99.0,
            )
            (above if diff > 0 else below).append(info)
        above.sort(key=lambda x: x["dist_pct"])
        below.sort(key=lambda x: x["dist_pct"])

        g   = lambda d, k, default: d[k] if d else default
        sup = (below[0] if below else None) if is_bull else (above[0] if above else None)
        res = (above[0] if above else None) if is_bull else (below[0] if below else None)
        sup_dir = bool(sup) and (
            (is_bull  and sup["direction"] == DIR_UP) or
            (not is_bull and sup["direction"] == DIR_DOWN)
        )
        sup_atr = g(sup, "dist_atr", 99.0)
        score   = (
            (1 if sup_atr < 1 else 0) +
            (1 if g(sup, "is_origin", False) else 0) +
            (1 if g(sup, "confirmed", False) else 0) +
            (1 if sup_dir else 0) -
            (1 if res and g(res,"dist_atr",99) < 1.5 and g(res,"is_origin",False) else 0)
        )

        rows.append({
            "eng_time":   T[i],
            "act_time":   act_ts,
            "direction":  "Long" if is_bull else "Short",
            "entry":      round(entry, 5),
            "stop":       round(stop,  5),
            "fib_1618":   round(f1618, 5),
            "fib_2618":   round(f2618, 5),
            "htf_trend":  htf_str,
            "btc_trend":  btc_str,
            "session":    session_of(T[i].hour),
            "day":        T[i].day_name()[:3],
            "hour":       T[i].hour,
            "month":      T[i].month,
            "htf_aligned":        htf_aligned,
            "bars_to_activation": act_bar - i,
            "atr_pct_at_act":     round(atr_act / entry * 100, 5),
            "atr_pct_at_eng":     round(atr_i   / entry * 100, 5),
            "rng_pct":            round(rng / entry * 100, 3),
            "stop_pct":           round(abs(entry-stop) / entry * 100, 3),
            "tp1618_pct":         round(abs(f1618-entry) / entry * 100, 3),
            "rng_to_atr":         round(rng / atr_i, 3),
            "vol_ratio":          round(vol_r if np.isfinite(vol_r) else 1.0, 3),
            "support_present":    int(bool(sup)),
            "support_dist_pct":   round(g(sup,"dist_pct",99.0), 3),
            "support_dist_atr":   round(sup_atr, 3),
            "support_is_origin":  int(g(sup,"is_origin",False)),
            "support_confirmed":  int(g(sup,"confirmed",False)),
            "support_dir_matches":int(sup_dir),
            "resistance_present": int(bool(res)),
            "resistance_dist_pct":round(g(res,"dist_pct",99.0), 3),
            "resistance_dist_atr":round(g(res,"dist_atr",99.0), 3),
            "resistance_is_origin":int(g(res,"is_origin",False)),
            "resistance_confirmed":int(g(res,"confirmed",False)),
            "confluence_score":   score,
        })

    if not rows:
        print(f"  No activated engulfments found in dataset")
        return

    ds = pd.DataFrame(rows)
    for c in CAT_FEATURES:
        ds[c] = ds[c].astype(str)

    # Apply ML model
    model = CatBoostClassifier()
    model.load_model(model_path)
    probs = model.predict_proba(ds[FEATURES])[:, 1]
    ds["p_win"] = probs

    # All engulfments in last 3 days
    recent = ds[ds["eng_time"] >= cutoff].copy()
    passed = recent[recent["p_win"] >= P_THRESHOLD]
    filtered = recent[recent["p_win"] < P_THRESHOLD]

    print(f"\n{'═'*60}")
    print(f"  {symbol}  — last {DAYS_BACK} days")
    print(f"  Cutoff: {cutoff.date()}  |  Now: {T[-1]}")
    print(f"{'═'*60}")
    print(f"  Total activated setups : {len(recent)}")
    print(f"  Passed ML filter (≥0.50): {len(passed)}")
    print(f"  Filtered out (<0.50)    : {len(filtered)}")

    if len(passed) > 0:
        print(f"\n  ✅ PASSED FILTER:")
        for _, r in passed.iterrows():
            actual_rr = abs(r["fib_2618"] - r["entry"]) / abs(r["entry"] - r["stop"])
            print(f"    {r['eng_time'].strftime('%Y-%m-%d %H:%M')}  {r['direction']:5s}  "
                  f"entry={r['entry']:.4f}  stop={r['stop']:.4f}  "
                  f"TP={r['fib_2618']:.4f}  p={r['p_win']:.2f}  "
                  f"HTF={r['htf_trend']}  RR=1:{actual_rr:.2f}")
    else:
        print(f"\n  No setups passed the filter in the last {DAYS_BACK} days.")

    if len(filtered) > 0:
        print(f"\n  ❌ FILTERED OUT:")
        for _, r in filtered.iterrows():
            print(f"    {r['eng_time'].strftime('%Y-%m-%d %H:%M')}  {r['direction']:5s}  "
                  f"entry={r['entry']:.4f}  p={r['p_win']:.2f}  HTF={r['htf_trend']}")


if __name__ == "__main__":
    print("Fetching data from Bybit …")
    ex = ccxt.bybit({"enableRateLimit": True})

    df_btc4h = fetch(ex, "BTC/USDT", "4h", LOOKBACK_4H)
    print(f"  BTC 4H: {len(df_btc4h)} bars to {df_btc4h.index[-1]}")

    for symbol, model_path in SYMBOLS.items():
        df1h = fetch(ex, symbol, "1h", LOOKBACK_1H)
        df4h = fetch(ex, symbol, "4h", LOOKBACK_4H)
        print(f"  {symbol} 1H: {len(df1h)} bars to {df1h.index[-1]}")
        run_scan(symbol, model_path, df1h, df4h, df_btc4h)

    print("\nDone.")
