"""
zec_pipeline.py — Full engulfing + ML backtest pipeline for ZEC/USDT 1H.

Steps:
  1. Fetch ZEC/USDT 1H from Binance (2019 → now)
  2. Engulfing backtest  → data/engulfing_zec_1h.csv
  3. Highlander confluence → data/engulfing_zec_1h_with_confluence.csv
  4. Train CatBoost      → data/catboost_clf_zec_1h.cbm
  5. Walk-forward validation (6-month OOS windows)
  6. Drawdown analysis on test set

Run: python zec_pipeline.py
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier
from highlander import run_indicator, DIR_UP, DIR_DOWN

DATA    = Path("data")
SYMBOL  = "SUI/USDT"
TF      = "1h"
SINCE   = "2023-05-01"

ATR_LEN              = 14
COST_R               = 0.04
MAX_BARS_TO_ACT      = 50
MAX_BARS_AFTER_ENTRY = 200
R_COL                = "r_strategy_D_trail"
P_THRESHOLD          = 0.50
MIN_TRAIN_MONTHS     = 6
STEP_MONTHS          = 3

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


# ── Helpers ───────────────────────────────────────────────────────────────────
def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def session_of(h):
    if 0 <= h < 7:   return "Asia"
    if 7 <= h < 13:  return "London"
    if 13 <= h < 17: return "NY-AM"
    if 17 <= h < 22: return "NY-PM"
    return "Off-hours"


# ── Step 1: Fetch data ────────────────────────────────────────────────────────
def fetch_binance_direct(symbol, tf, since_str):
    """Hit Binance klines endpoint directly — bypasses ccxt market loading (which geo-blocks)."""
    import requests
    sym = symbol.replace("/", "")
    since_ms = int(pd.Timestamp(since_str).timestamp() * 1000)
    bars = []
    print(f"  Fetching {symbol} {tf} via Binance direct …", end="", flush=True)
    while True:
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={sym}&interval={tf}&startTime={since_ms}&limit=1000")
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 451:
                print(f"\n  Binance still geo-blocked (451)")
                return None
            r.raise_for_status()
            chunk = r.json()
        except Exception as e:
            print(f"\n  Binance direct failed: {e!r:.80}")
            return None
        if not chunk:
            break
        bars.extend(chunk)
        since_ms = chunk[-1][0] + 1
        print(".", end="", flush=True)
        if len(chunk) < 1000:
            break
        time.sleep(0.2)
    if not bars:
        return None
    print(f"  {len(bars):,} bars")
    df = pd.DataFrame(bars, columns=["time","open","high","low","close","volume",
                                      "close_time","qav","trades","tbbav","tbqav","ignore"])
    df = df[["time","open","high","low","close","volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").set_index("time").sort_index()


def fetch_all(exchange, symbol, tf, since_str):
    since_ms = int(pd.Timestamp(since_str).timestamp() * 1000)

    # Test fetch — some exchanges restrict how far back you can go
    for attempt_since in [since_ms,
                          int(pd.Timestamp("2020-01-01").timestamp() * 1000),
                          int(pd.Timestamp("2021-01-01").timestamp() * 1000),
                          int(pd.Timestamp("2022-01-01").timestamp() * 1000)]:
        try:
            test = exchange.fetch_ohlcv(symbol, tf, since=attempt_since, limit=5)
            if test:
                since_ms = attempt_since
                if attempt_since != int(pd.Timestamp(since_str).timestamp() * 1000):
                    print(f"\n  Start date adjusted to {pd.Timestamp(attempt_since, unit='ms').date()}")
                break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Exchange returned no data for {symbol} {tf}")

    # Bybit paginates best going forwards with since; try up to 50 pages
    bars = []
    print(f"  Fetching {symbol} {tf} …", end="", flush=True)
    empty_streak = 0
    while True:
        try:
            chunk = exchange.fetch_ohlcv(symbol, tf, since=since_ms, limit=1000)
        except Exception:
            break
        if not chunk:
            empty_streak += 1
            if empty_streak >= 2:
                break
            time.sleep(1)
            continue
        empty_streak = 0
        # Deduplicate against already-fetched bars
        new_bars = [b for b in chunk if b[0] > (bars[-1][0] if bars else 0)]
        if not new_bars:
            break
        bars.extend(new_bars)
        since_ms = bars[-1][0] + 1
        print(".", end="", flush=True)
        if len(chunk) < 1000:
            break
        time.sleep(0.2)
    print(f"  {len(bars):,} bars")
    if not bars:
        raise RuntimeError(f"No bars fetched for {symbol}")
    df = pd.DataFrame(bars, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").set_index("time").sort_index().astype(float)


# ── Step 2: Engulfing backtest ────────────────────────────────────────────────
def detect_engulfing(df):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    rows = []
    for i in range(1, len(df)):
        bull = (c[i-1] < o[i-1]) and (c[i] > o[i]) and (c[i] > h[i-1])
        bear = (c[i-1] > o[i-1]) and (c[i] < o[i]) and (c[i] < l[i-1])
        if bull:
            rng = h[i-1] - l[i-1]
            if rng > 0:
                rows.append({"bar": i, "is_bull": True,  "origin": l[i-1], "entry": h[i-1], "rng": rng})
        if bear:
            rng = h[i-1] - l[i-1]
            if rng > 0:
                rows.append({"bar": i, "is_bull": False, "origin": h[i-1], "entry": l[i-1], "rng": rng})
    return pd.DataFrame(rows)


def find_activation(H, L, eng_bar, is_bull, entry, max_bars):
    end = min(eng_bar + 1 + max_bars, len(H))
    for j in range(eng_bar + 1, end):
        if is_bull  and L[j] <= entry: return j
        if not is_bull and H[j] >= entry: return j
    return None


def simulate_fib(H, L, act_bar, is_bull, entry, stop, fibs, max_bars):
    n   = len(H)
    end = min(act_bar + 1 + max_bars, n)
    out = {f"hit_{k}": False for k in fibs}
    out["stop_hit"] = False
    risk = abs(entry - stop)
    mfe  = 0.0
    for j in range(act_bar + 1, end):
        hi, lo = H[j], L[j]
        if is_bull  and lo <= stop: out["stop_hit"] = True; break
        if not is_bull and hi >= stop: out["stop_hit"] = True; break
        for k, px in fibs.items():
            if not out[f"hit_{k}"]:
                if (is_bull and hi >= px) or (not is_bull and lo <= px):
                    out[f"hit_{k}"] = True
        excursion = (hi - entry) / risk if is_bull else (entry - lo) / risk
        mfe = max(mfe, excursion)
    out["mfe"] = mfe
    return out


def run_engulfing_backtest(df, btc_df):
    print(f"\n{'═'*60}\nSTEP 2: Engulfing backtest\n{'═'*60}")
    df = df.copy()
    df["atr"]     = true_range(df).rolling(ATR_LEN).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    df["vol_avg"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg"]

    # HTF trend (resample to 4H)
    htf = df[["close"]].resample("4h").last().dropna()
    htf["sma20"] = htf["close"].rolling(20).mean()
    htf["sma50"] = htf["close"].rolling(50).mean()
    htf["t"] = np.where(htf["sma20"] > htf["sma50"], 1, np.where(htf["sma20"] < htf["sma50"], -1, 0))
    idx = np.clip(htf.index.searchsorted(df.index, side="right") - 1, 0, len(htf)-1)
    df["htf_label"] = np.where(htf["t"].values[idx] == 1, "Bullish",
                       np.where(htf["t"].values[idx] == -1, "Bearish", "Range"))

    # BTC trend
    btc = btc_df.copy()
    btc["sma20"] = btc["close"].rolling(20).mean()
    btc["sma50"] = btc["close"].rolling(50).mean()
    btc["t"] = np.where(btc["sma20"] > btc["sma50"], 1, np.where(btc["sma20"] < btc["sma50"], -1, 0))
    bidx = np.clip(btc.index.searchsorted(df.index, side="right") - 1, 0, len(btc)-1)
    df["btc_label"] = np.where(btc["t"].values[bidx] == 1, "Up",
                       np.where(btc["t"].values[bidx] == -1, "Down", "Range"))

    engs = detect_engulfing(df)
    print(f"  Engulfings: {len(engs):,}  ({engs['is_bull'].sum()} bull / {(~engs['is_bull']).sum()} bear)")

    H, L, C = df["high"].values, df["low"].values, df["close"].values
    records = []
    skipped = 0
    for _, e in engs.iterrows():
        bar = int(e["bar"])
        if bar < ATR_LEN + 5: skipped += 1; continue
        atr = float(df["atr"].iloc[bar])
        if not np.isfinite(atr) or atr <= 0: skipped += 1; continue

        is_bull = bool(e["is_bull"])
        origin, entry, rng = float(e["origin"]), float(e["entry"]), float(e["rng"])
        m    = 1 if is_bull else -1
        stop = origin - 0.5 * rng * m
        f1618 = origin + 1.618 * rng * m
        f2618 = origin + 2.618 * rng * m

        act_bar = find_activation(H, L, bar, is_bull, entry, MAX_BARS_TO_ACT)
        if act_bar is None: skipped += 1; continue

        fibs = {"f1618": f1618, "f2618": f2618}
        sim  = simulate_fib(H, L, act_bar, is_bull, entry, stop, fibs, MAX_BARS_AFTER_ENTRY)

        risk = abs(entry - stop)
        # Strategy D: trail to BE at 1.618, exit at 2.618
        if sim["stop_hit"] and not sim["hit_f1618"]:
            r_D = -1.0
        elif sim["hit_f1618"] and not sim["hit_f2618"]:
            r_D = 0.0
        else:
            r_D = 2.618 if sim["hit_f2618"] else sim["mfe"]

        ts  = df.index[act_bar]
        vr  = float(df["vol_ratio"].iloc[bar])
        records.append({
            "engulf_time":     str(df.index[bar]),
            "activation_time": str(ts),
            "direction":       "Long" if is_bull else "Short",
            "engulf_bar":      bar,
            "activation_bar":  act_bar,
            "origin": origin, "entry": entry, "stop": stop,
            "fib_1618": f1618, "fib_2618": f2618,
            "rng": rng,
            "rng_pct":     round(rng / entry * 100, 3),
            "stop_pct":    round(risk / entry * 100, 3),
            "tp1618_pct":  round(abs(f1618 - entry) / entry * 100, 3),
            "r_strategy_D_trail": round(r_D - COST_R, 3),
            "bars_to_activation": act_bar - bar,
            "atr_pct_at_act": round(float(df["atr_pct"].iloc[act_bar]), 5),
            "atr_pct_at_eng": round(float(df["atr_pct"].iloc[bar]), 5),
            "htf_trend":  str(df["htf_label"].iloc[act_bar]),
            "btc_trend":  str(df["btc_label"].iloc[act_bar]),
            "session":    session_of(ts.hour),
            "hour":       int(ts.hour),
            "day":        ts.day_name()[:3],
            "month":      int(ts.month),
            "year":       int(ts.year),
            "vol_ratio":  round(vr if np.isfinite(vr) else 1.0, 3),
            "rng_to_atr": round(rng / atr, 3),
        })

    ds = pd.DataFrame.from_records(records)
    ds["htf_aligned"] = (((ds["direction"]=="Long")  & (ds["htf_trend"]=="Bullish")) |
                         ((ds["direction"]=="Short") & (ds["htf_trend"]=="Bearish"))).astype(int)
    print(f"  Trades built: {len(ds):,}  (skipped {skipped})")
    for col in ["r_strategy_D_trail"]:
        wr  = (ds[col] > 0).mean()
        avg = ds[col].mean()
        net = ds[col].sum()
        print(f"  Raw (no filter): win {wr:.1%}  avgR {avg:+.3f}  netR {net:+.1f}")
    return ds


# ── Step 3: Confluence ────────────────────────────────────────────────────────
def add_confluence(ds, df_ohlc):
    print(f"\n{'═'*60}\nSTEP 3: Highlander confluence\n{'═'*60}")
    df = df_ohlc[["open","high","low","close"]].astype(float)
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    df = df.copy()
    df["atr"] = pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).rolling(14).mean()

    tick   = max(float(df["close"].iloc[-1]) * 1e-5, 1e-6)
    levels, events = run_indicator(df[["open","high","low","close"]], min_range_ticks=3.0, tick_size=tick)
    print(f"  Highlander levels: {len(levels):,}")

    # Build level history lookup — events is list[Event]
    records_ev = [e.__dict__ for e in events]
    grouped = {}
    for r in records_ev:
        lid = int(r["level_id"])
        grouped.setdefault(lid, []).append(r)
    meta = {}
    for lid, hist in grouped.items():
        meta[lid] = {
            "created_bar": next((h["bar"] for h in hist if h["event"]=="level_created"), None),
            "deleted_bar": next((h["bar"] for h in hist if h["event"]=="level_deleted"), None),
        }

    def state_at(hist, target_bar):
        if not hist or hist[0]["bar"] > target_bar: return None
        state = -1; confirmed = False; alive = True
        for ev in hist:
            if ev["bar"] > target_bar: break
            e = ev["event"]
            if e == "level_created":    state = 0
            elif e == "first_touch":    state = 3
            elif e == "origin_confirmed": state = 1
            elif e == "level_broken":   state = 2
            elif e == "level_deleted":  alive = False
            confirmed = ev["confirmed"]
        return {"state": state, "confirmed": confirmed, "alive": alive,
                "direction": hist[0]["direction"], "price": hist[0]["price"]}

    ds = ds.copy()
    ds["engulf_time"]     = pd.to_datetime(ds["engulf_time"])
    ds["activation_time"] = pd.to_datetime(ds["activation_time"])

    bar_times = df.index.to_numpy()
    rows = []
    for _, row in ds.iterrows():
        act_bar  = int(np.searchsorted(bar_times, row["activation_time"].to_numpy(), side="right") - 1)
        act_bar  = max(0, min(act_bar, len(df)-1))
        price    = float(df["close"].iloc[act_bar])
        atr_v    = float(df["atr"].iloc[act_bar]) if np.isfinite(df["atr"].iloc[act_bar]) else 1.0
        is_bull  = row["direction"] == "Long"

        above, below = [], []
        for lid, m in meta.items():
            if m["created_bar"] is None or m["created_bar"] > act_bar: continue
            if m["deleted_bar"] is not None and m["deleted_bar"] <= act_bar: continue
            s = state_at(grouped[lid], act_bar)
            if s is None or not s["alive"] or s["state"] == 2: continue
            diff = s["price"] - price
            info = {
                "direction": s["direction"], "state": s["state"],
                "confirmed": s["confirmed"], "is_origin": s["state"] == 1,
                "dist_pct": abs(diff) / price * 100,
                "dist_atr": abs(diff) / atr_v if atr_v > 0 else 99.0,
            }
            (above if diff > 0 else below).append(info)
        above.sort(key=lambda x: x["dist_pct"])
        below.sort(key=lambda x: x["dist_pct"])

        sup = (below[0] if below else None) if is_bull else (above[0] if above else None)
        res = (above[0] if above else None) if is_bull else (below[0] if below else None)
        g   = lambda d, k, default: d[k] if d else default

        sup_dir = bool(sup) and (
            (is_bull and sup["direction"] == DIR_UP) or
            (not is_bull and sup["direction"] == DIR_DOWN)
        )
        sup_atr = g(sup, "dist_atr", 99.0)
        score = (
            (1 if sup_atr < 1 else 0) +
            (1 if g(sup, "is_origin", False) else 0) +
            (1 if g(sup, "confirmed", False) else 0) +
            (1 if sup_dir else 0) -
            (1 if res and g(res, "dist_atr", 99) < 1.5 and g(res, "is_origin", False) else 0)
        )
        rows.append({
            "support_present":      int(bool(sup)),
            "support_dist_pct":     round(g(sup, "dist_pct", 99.0), 3),
            "support_dist_atr":     round(sup_atr, 3),
            "support_is_origin":    int(g(sup, "is_origin", False)),
            "support_confirmed":    int(g(sup, "confirmed", False)),
            "support_dir_matches":  int(sup_dir),
            "resistance_present":   int(bool(res)),
            "resistance_dist_pct":  round(g(res, "dist_pct", 99.0), 3),
            "resistance_dist_atr":  round(g(res, "dist_atr", 99.0), 3),
            "resistance_is_origin": int(g(res, "is_origin", False)),
            "resistance_confirmed": int(g(res, "confirmed", False)),
            "confluence_score":     score,
        })

    conf_df  = pd.DataFrame(rows, index=ds.index)
    enriched = pd.concat([ds, conf_df], axis=1)
    print(f"  Confluence added to {len(enriched):,} trades")
    return enriched


# ── Step 4: Train CatBoost ────────────────────────────────────────────────────
def train_model(ds):
    print(f"\n{'═'*60}\nSTEP 4: Train CatBoost\n{'═'*60}")
    ds = ds.copy()
    for col in ["support_dist_pct","support_dist_atr","resistance_dist_pct","resistance_dist_atr"]:
        ds[col] = ds[col].fillna(99.0)
    ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)
    for c in CAT_FEATURES:
        ds[c] = ds[c].astype(str)
    ds["is_win_D"] = (ds[R_COL] > 0).astype(int)

    split = int(len(ds) * 0.8)
    train, test = ds.iloc[:split], ds.iloc[split:]
    print(f"  Train: {len(train):,}  Test: {len(test):,}")

    clf = CatBoostClassifier(
        iterations=400, learning_rate=0.05, depth=5,
        cat_features=cat_idx, verbose=0, random_seed=42, l2_leaf_reg=5,
    )
    clf.fit(train[FEATURES], train["is_win_D"])

    p_win = clf.predict_proba(test[FEATURES])[:, 1]
    test  = test.copy()
    test["p_win"] = p_win
    m = p_win >= P_THRESHOLD

    print(f"\n  TEST SET results (P≥{P_THRESHOLD}):")
    print(f"  Baseline : win {(test[R_COL]>0).mean():.1%}  avgR {test[R_COL].mean():+.3f}  netR {test[R_COL].sum():+.1f}  N={len(test)}")
    if m.sum() > 0:
        print(f"  Filtered : win {(test.loc[m,R_COL]>0).mean():.1%}  avgR {test.loc[m,R_COL].mean():+.3f}  netR {test.loc[m,R_COL].sum():+.1f}  N={m.sum()}")

    sym_slug = SYMBOL.replace("/","").lower()
    clf.save_model(str(DATA / f"catboost_clf_{sym_slug}_1h.cbm"))
    test.to_csv(DATA / f"{sym_slug}_1h_test_predictions.csv", index=False)
    print(f"  Model saved → data/catboost_clf_{sym_slug}_1h.cbm")
    return clf, test


# ── Step 5: Walk-forward ──────────────────────────────────────────────────────
def walk_forward(ds):
    print(f"\n{'═'*60}\nSTEP 5: Walk-forward validation\n{'═'*60}")
    ds = ds.copy()
    for col in ["support_dist_pct","support_dist_atr","resistance_dist_pct","resistance_dist_atr"]:
        ds[col] = ds[col].fillna(99.0)
    ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)
    for c in CAT_FEATURES:
        ds[c] = ds[c].astype(str)
    ds["is_win_D"] = (ds[R_COL] > 0).astype(int)
    ds["activation_time"] = pd.to_datetime(ds["activation_time"])

    t_start, t_end = ds["activation_time"].min(), ds["activation_time"].max()
    windows, ts = [], t_start + pd.DateOffset(months=MIN_TRAIN_MONTHS)
    while ts < t_end:
        te = min(ts + pd.DateOffset(months=STEP_MONTHS), t_end)
        windows.append((ts, te)); ts = te

    results, chunks = [], []
    for i, (ws, we) in enumerate(windows):
        train = ds[ds["activation_time"] <  ws]
        test  = ds[(ds["activation_time"] >= ws) & (ds["activation_time"] < we)].copy().reset_index(drop=True)
        if len(train) < 100 or len(test) < 10: continue

        clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                                  cat_features=cat_idx, verbose=0, random_seed=42, l2_leaf_reg=5)
        clf.fit(train[FEATURES], train["is_win_D"])
        p   = clf.predict_proba(test[FEATURES])[:, 1]
        test["p_win_wf"] = p
        m   = p >= P_THRESHOLD

        base_avg = float(test[R_COL].mean())
        filt_avg = float(test.loc[m, R_COL].mean()) if m.sum() > 0 else np.nan
        filt_wr  = float(test.loc[m, "is_win_D"].mean()) if m.sum() > 0 else np.nan

        results.append({
            "window":        f"{ws.date()} → {we.date()}",
            "train_n":       len(train),
            "test_n":        len(test),
            "filtered_n":    int(m.sum()),
            "baseline_avgR": round(base_avg, 3),
            "model_avgR":    round(filt_avg, 3) if not np.isnan(filt_avg) else None,
            "model_wr":      f"{filt_wr:.1%}" if filt_wr and not np.isnan(filt_wr) else None,
            "positive":      "YES" if not np.isnan(filt_avg) and filt_avg > 0 else "no",
        })
        chunks.append(test)
        pos = "✅" if results[-1]["positive"] == "YES" else "❌"
        print(f"  {pos} {results[-1]['window']}  train={len(train):4d}  filt={m.sum():3d}  "
              f"baseline={base_avg:+.3f}  model={filt_avg:+.3f}" if not np.isnan(filt_avg)
              else f"  ❌ {results[-1]['window']}  (no filtered signals)")

    if not results:
        print(f"\n  Not enough data for walk-forward ({len(ds):,} trades, need more history)")
        print(f"  Raw strategy stats across all data:")
        print(f"  Baseline avgR: {ds[R_COL].mean():+.3f}  netR: {ds[R_COL].sum():+.1f}  win: {(ds[R_COL]>0).mean():.1%}  N={len(ds)}")
        return pd.DataFrame()
    rdf = pd.DataFrame(results)
    wins = (rdf["positive"] == "YES").sum()
    print(f"\n{'═'*60}")
    print(f"WALK-FORWARD SUMMARY  (P≥{P_THRESHOLD}, {SYMBOL} 1H)")
    print(f"{'═'*60}")
    print(rdf.to_string(index=False))
    print(f"\nWindows positive: {wins}/{len(rdf)}")

    if chunks:
        all_oos = pd.concat(chunks).sort_values("activation_time").reset_index(drop=True)
        m_all   = all_oos["p_win_wf"] >= P_THRESHOLD
        print(f"\nComposite OOS (all windows):")
        print(f"  Total signals: {len(all_oos):,}  |  Filtered: {m_all.sum():,}")
        print(f"  Baseline  avgR: {all_oos[R_COL].mean():+.3f}  netR: {all_oos[R_COL].sum():+.1f}")
        if m_all.sum() > 0:
            print(f"  Filtered  avgR: {all_oos.loc[m_all, R_COL].mean():+.3f}  "
                  f"netR: {all_oos.loc[m_all, R_COL].sum():+.1f}  "
                  f"win: {all_oos.loc[m_all,'is_win_D'].mean():.1%}")

        # Equity curve
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(all_oos["activation_time"].values, all_oos[R_COL].cumsum().values,
                label="All signals", color="gray", lw=1.0)
        if m_all.sum() > 0:
            ax.plot(all_oos.loc[m_all, "activation_time"].values,
                    all_oos.loc[m_all, R_COL].cumsum().values,
                    label=f"ML P≥{P_THRESHOLD}  (N={m_all.sum()})", color="C1", lw=1.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("ZEC/USDT 1H — Walk-forward equity curve (zero leakage)")
        ax.set_ylabel("Cumulative R"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        sym_slug = SYMBOL.replace("/","").lower()
        plt.savefig(DATA / f"{sym_slug}_walk_forward_equity.png", dpi=130)
        print(f"  Saved → data/{sym_slug}_walk_forward_equity.png")
    return rdf


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import ccxt
    # Try exchanges in order — check each actually has ZEC/USDT
    for ex_id in ("bybit", "kucoin", "okx", "gate"):
        try:
            exchange = getattr(ccxt, ex_id)({"enableRateLimit": True})
            markets  = exchange.load_markets()
            if SYMBOL in markets:
                print(f"  Exchange: {ex_id}")
                break
            print(f"  {ex_id}: no {SYMBOL}, trying next…")
        except Exception as e:
            print(f"  {ex_id} unavailable ({e!r:.60}), trying next…")
    else:
        raise RuntimeError(f"No exchange found with {SYMBOL} — check your connection")

    print(f"{'═'*60}\nZEC/USDT 1H — Full pipeline\n{'═'*60}")

    # Fetch ZEC
    sym_slug = SYMBOL.replace("/","").lower()
    zec_path = DATA / f"{sym_slug}_1h.csv"
    if zec_path.exists():
        print(f"  Using cached {zec_path}")
        zec = pd.read_csv(zec_path, parse_dates=["time"]).set_index("time").astype(float)
    else:
        # Try Binance direct first (full history, no geo-block on klines endpoint)
        zec = fetch_binance_direct(SYMBOL, TF, SINCE)
        if zec is None:
            zec = fetch_all(exchange, SYMBOL, TF, SINCE)
        zec.to_csv(zec_path)
        print(f"  Saved → {zec_path}")

    # Fetch/reuse BTC 4H
    btc_path = DATA / "btc_4h.csv"
    if btc_path.exists():
        btc = pd.read_csv(btc_path, parse_dates=["time"]).set_index("time").astype(float)
    else:
        btc = fetch_all(exchange, "BTC/USDT", "4h", SINCE)
        btc.to_csv(btc_path)

    print(f"  ZEC bars: {len(zec):,}  ({zec.index[0].date()} → {zec.index[-1].date()})")

    # Pipeline
    ds       = run_engulfing_backtest(zec, btc)
    ds.to_csv(DATA / f"engulfing_{sym_slug}_1h.csv", index=False)

    enriched = add_confluence(ds, zec)
    enriched.to_csv(DATA / f"engulfing_{sym_slug}_1h_with_confluence.csv", index=False)

    clf, test = train_model(enriched)
    walk_forward(enriched)

    print(f"\n{'═'*60}\nDONE\n{'═'*60}")
