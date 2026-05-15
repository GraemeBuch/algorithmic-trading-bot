"""
test_new_symbols.py — Full engulfing + ML walk-forward pipeline for new symbols.
Runs the identical pipeline used for SOL and SUI.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from highlander import run_indicator, DIR_UP, DIR_DOWN

DATA                 = Path("data")
ATR_LEN              = 14
COST_R               = 0.04
MAX_BARS_TO_ACT      = 50
MAX_BARS_AFTER_ENTRY = 200
P_THRESHOLD          = 0.50
MIN_TRAIN_MONTHS     = 6
STEP_MONTHS          = 3

SYMBOLS_TO_TEST = [
    ("SOL/USDT",  "2021-01-01"),
    ("SUI/USDT",  "2023-05-01"),
    ("XRP/USDT",  "2019-01-01"),
    ("DOGE/USDT", "2019-01-01"),
    ("LTC/USDT",  "2019-01-01"),
    ("AVAX/USDT", "2021-01-01"),
    ("ADA/USDT",  "2018-01-01"),
    ("XLM/USDT",  "2018-01-01"),
    ("LINK/USDT", "2019-01-01"),
    ("HBAR/USDT", "2019-01-01"),
    ("TRX/USDT",  "2019-01-01"),
    ("TON/USDT",  "2023-01-01"),
]

CAT_FEATURES = ["direction", "session", "htf_trend", "btc_trend", "day"]
NUM_FEATURES = [
    "bars_to_activation", "atr_pct_at_act", "atr_pct_at_eng",
    "rng_pct", "stop_pct", "tp1618_pct", "rng_to_atr",
    "vol_ratio", "hour", "month", "htf_aligned",
    "support_present", "support_dist_pct", "support_dist_atr",
    "support_is_origin", "support_confirmed", "support_dir_matches",
    "resistance_present", "resistance_dist_pct", "resistance_dist_atr",
    "resistance_is_origin", "resistance_confirmed", "confluence_score",
    # Engulf candle quality
    "body_pct", "upper_wick_pct", "lower_wick_pct", "engulf_ratio",
    # Entry price vs nearest Highlander level (1H + 4H)
    "entry_1h_at_level", "entry_1h_dist_atr", "entry_1h_is_origin",
    "entry_1h_is_ft",    "entry_1h_is_broken", "entry_1h_dir_match",
    "entry_4h_at_level", "entry_4h_dist_atr", "entry_4h_is_origin",
    "entry_4h_is_ft",    "entry_4h_is_broken", "entry_4h_dir_match",
    # Origin (bounce point) vs nearest Highlander level (1H + 4H)
    "origin_1h_dist_atr", "origin_1h_is_origin", "origin_1h_is_ft",
    "origin_1h_is_broken", "origin_1h_dir_match",
    "origin_4h_dist_atr", "origin_4h_is_origin", "origin_4h_is_ft",
    "origin_4h_is_broken", "origin_4h_dir_match",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
cat_idx  = [FEATURES.index(c) for c in CAT_FEATURES]


def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)


def session_of(h):
    if 0  <= h < 7:  return "Asia"
    if 7  <= h < 13: return "London"
    if 13 <= h < 17: return "NY-AM"
    if 17 <= h < 22: return "NY-PM"
    return "Off-hours"


def fetch_binance_direct(symbol, tf, since_str):
    import requests
    sym      = symbol.replace("/", "")
    since_ms = int(pd.Timestamp(since_str).timestamp() * 1000)
    bars     = []
    print(f"  Fetching {symbol} {tf} via Binance …", end="", flush=True)
    while True:
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={sym}&interval={tf}&startTime={since_ms}&limit=1000")
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 451:
                print(f"\n  Binance geo-blocked (451)"); return None
            r.raise_for_status()
            chunk = r.json()
        except Exception as e:
            print(f"\n  Binance failed: {e!r:.60}"); return None
        if not chunk: break
        bars.extend(chunk)
        since_ms = chunk[-1][0] + 1
        print(".", end="", flush=True)
        if len(chunk) < 1000: break
        time.sleep(0.15)
    if not bars: return None
    print(f"  {len(bars):,} bars")
    df = pd.DataFrame(bars, columns=["time","open","high","low","close","volume",
                                     "close_time","qav","trades","tbbav","tbqav","ignore"])
    df = df[["time","open","high","low","close","volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").set_index("time").sort_index()


def fetch_all(exchange, symbol, tf, since_str):
    since_ms = int(pd.Timestamp(since_str).timestamp() * 1000)
    bars = []
    print(f"  Fetching {symbol} {tf} …", end="", flush=True)
    empty_streak = 0
    while True:
        try:
            chunk = exchange.fetch_ohlcv(symbol, tf, since=since_ms, limit=1000)
        except Exception as e:
            print(f"\n  fetch error: {e!r:.60}")
            break
        if not chunk:
            empty_streak += 1
            if empty_streak >= 2: break
            time.sleep(1); continue
        empty_streak = 0
        new_bars = [b for b in chunk if b[0] > (bars[-1][0] if bars else 0)]
        if not new_bars: break
        bars.extend(new_bars)
        since_ms = bars[-1][0] + 1
        print(".", end="", flush=True)
        if len(chunk) < 1000: break
        time.sleep(0.25)
    print(f"  {len(bars):,} bars")
    if not bars:
        raise RuntimeError(f"No bars for {symbol}")
    df = pd.DataFrame(bars, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df.drop_duplicates("time").set_index("time").sort_index().astype(float)


def build_trades(df, btc_df):
    df = df.copy()
    df["atr"]       = true_range(df).rolling(ATR_LEN).mean()
    df["atr_pct"]   = df["atr"] / df["close"]
    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg"]

    htf = df[["close"]].resample("4h").last().dropna()
    htf["sma20"] = htf["close"].rolling(20).mean()
    htf["sma50"] = htf["close"].rolling(50).mean()
    htf["t"] = np.where(htf["sma20"]>htf["sma50"], 1, np.where(htf["sma20"]<htf["sma50"], -1, 0))
    idx = np.clip(htf.index.searchsorted(df.index, side="right")-1, 0, len(htf)-1)
    df["htf_label"] = np.where(htf["t"].values[idx]==1,"Bullish",
                      np.where(htf["t"].values[idx]==-1,"Bearish","Range"))

    btc = btc_df.copy()
    btc["sma20"] = btc["close"].rolling(20).mean()
    btc["sma50"] = btc["close"].rolling(50).mean()
    btc["t"] = np.where(btc["sma20"]>btc["sma50"],1,np.where(btc["sma20"]<btc["sma50"],-1,0))
    bidx = np.clip(btc.index.searchsorted(df.index, side="right")-1, 0, len(btc)-1)
    df["btc_label"] = np.where(btc["t"].values[bidx]==1,"Up",
                      np.where(btc["t"].values[bidx]==-1,"Down","Range"))

    O=df["open"].values; H=df["high"].values; L=df["low"].values; C=df["close"].values
    records = []
    for i in range(1, len(df)):
        bull = (C[i-1]<O[i-1]) and (C[i]>O[i]) and (C[i]>H[i-1])
        bear = (C[i-1]>O[i-1]) and (C[i]<O[i]) and (C[i]<L[i-1])
        for is_bull in ([True]*int(bull) + [False]*int(bear)):
            rng = H[i-1] - L[i-1]
            if rng <= 0: continue
            atr = float(df["atr"].iloc[i])
            if not np.isfinite(atr) or atr <= 0: continue
            origin = L[i-1] if is_bull else H[i-1]
            entry  = H[i-1] if is_bull else L[i-1]
            m      = 1 if is_bull else -1
            stop   = origin - 0.5*rng*m
            f1618  = origin + 1.618*rng*m
            f2618  = origin + 2.618*rng*m
            risk   = abs(entry-stop)

            # Activation
            act_bar = None
            for j in range(i+1, min(i+1+MAX_BARS_TO_ACT, len(df))):
                if (is_bull and L[j]<=entry) or (not is_bull and H[j]>=entry):
                    act_bar=j; break
            if act_bar is None: continue

            # Apply full ambiguous-bar logic to the activation candle itself.
            # Entry was touched (that's what makes it act_bar), but the same
            # candle may have also touched stop, f1618, and/or f2618.
            # Use next-bar-open to resolve, same as we do for subsequent bars.
            outcome   = None
            be_hit    = False
            has_next_act = (act_bar + 1) < len(df)
            if is_bull:
                act_hit_stop  = L[act_bar] <= stop
                act_hit_f1618 = H[act_bar] >= f1618
                act_hit_f2618 = H[act_bar] >= f2618
            else:
                act_hit_stop  = H[act_bar] >= stop
                act_hit_f1618 = L[act_bar] <= f1618
                act_hit_f2618 = L[act_bar] <= f2618

            if has_next_act:
                nxt = O[act_bar + 1]
                if act_hit_f2618:
                    # Touched TP on activation candle — use next-bar-open to resolve
                    if act_hit_stop:
                        outcome = "win" if abs(nxt - f2618) < abs(nxt - stop) else "full_stop"
                    else:
                        outcome = "win"
                elif act_hit_stop:
                    # Stop touched on activation candle (with or without f1618).
                    # Do NOT carry be_hit forward — we can't know if f1618 was
                    # touched before or after entry, so don't assume SL moved.
                    # Use next-bar-open: closer to stop → full_stop, else continue.
                    if abs(nxt - stop) < abs(nxt - entry):
                        outcome = "full_stop"
                # act_hit_f1618 only (no stop): no outcome change, no be_hit carry-forward.
                # The outcome loop starts fresh with be_hit=False.

            # Outcome loop — skipped if activation bar already resolved outcome
            if outcome is None:
                for j in range(act_bar+1, min(act_bar+1+MAX_BARS_AFTER_ENTRY, len(df))):
                    cur_stop = entry if be_hit else stop
                    has_next  = (j + 1) < len(df)
                    if is_bull:
                        hit_stop  = L[j] <= cur_stop
                        hit_f1618 = H[j] >= f1618
                        hit_f2618 = H[j] >= f2618
                        if not be_hit and hit_f1618: be_hit=True
                        if hit_f2618:
                            # All-4 ambiguous: same bar touches stop AND TP
                            if hit_stop and has_next:
                                nxt = O[j+1]
                                outcome = "win" if abs(nxt-f2618)<abs(nxt-cur_stop) else "be_stop"
                            else:
                                outcome = "win"
                            break
                        if hit_stop:
                            # Mid-3 ambiguous: same bar touches stop AND f1618 (be_hit just set)
                            if hit_f1618 and has_next:
                                nxt  = O[j+1]
                                outcome = "be_stop" if nxt > stop else "full_stop"
                            else:
                                outcome = "be_stop" if be_hit else "full_stop"
                            break
                    else:
                        hit_stop  = H[j] >= cur_stop
                        hit_f1618 = L[j] <= f1618
                        hit_f2618 = L[j] <= f2618
                        if not be_hit and hit_f1618: be_hit=True
                        if hit_f2618:
                            if hit_stop and has_next:
                                nxt = O[j+1]
                                outcome = "win" if abs(nxt-f2618)<abs(nxt-cur_stop) else "be_stop"
                            else:
                                outcome = "win"
                            break
                        if hit_stop:
                            if hit_f1618 and has_next:
                                nxt  = O[j+1]
                                outcome = "be_stop" if nxt < stop else "full_stop"
                            else:
                                outcome = "be_stop" if be_hit else "full_stop"
                            break
            if outcome is None: continue

            r_D = (abs(f2618-entry)/risk - COST_R) if outcome=="win" \
                  else (-COST_R if outcome=="be_stop" else -1.0-COST_R)

            ts  = df.index[act_bar]
            vr  = float(df["vol_ratio"].iloc[i])

            # Engulf candle quality (bar i)
            eng_rng        = H[i] - L[i]
            _er            = eng_rng if eng_rng > 0 else 1e-10
            eng_body       = abs(C[i] - O[i])
            eng_upper_wick = H[i] - max(O[i], C[i])
            eng_lower_wick = min(O[i], C[i]) - L[i]
            body_pct       = round(eng_body       / _er, 3)
            upper_wick_pct = round(eng_upper_wick / _er, 3)
            lower_wick_pct = round(eng_lower_wick / _er, 3)
            engulf_ratio   = round(eng_rng / rng,   3) if rng > 0 else 1.0

            records.append({
                "engulf_time":        str(df.index[i]),
                "activation_time":    str(ts),
                "direction":          "Long" if is_bull else "Short",
                "origin": origin, "entry": entry, "stop": stop,
                "fib_1618": f1618, "fib_2618": f2618, "rng": rng,
                "r_strategy_D_trail": round(r_D, 4),
                "outcome":            outcome,
                "rng_pct":            round(rng/entry*100, 3),
                "stop_pct":           round(risk/entry*100, 3),
                "tp1618_pct":         round(abs(f1618-entry)/entry*100, 3),
                "bars_to_activation": act_bar-i,
                "atr_pct_at_act":     round(float(df["atr_pct"].iloc[act_bar]), 5),
                "atr_pct_at_eng":     round(float(df["atr_pct"].iloc[i]), 5),
                "htf_trend":          str(df["htf_label"].iloc[act_bar]),
                "btc_trend":          str(df["btc_label"].iloc[act_bar]),
                "session":            session_of(ts.hour),
                "hour":               int(ts.hour),
                "day":                ts.day_name()[:3],
                "month":              int(ts.month),
                "year":               int(ts.year),
                "vol_ratio":          round(vr if np.isfinite(vr) else 1.0, 3),
                "rng_to_atr":         round(rng/atr, 3),
                "body_pct":           body_pct,
                "upper_wick_pct":     upper_wick_pct,
                "lower_wick_pct":     lower_wick_pct,
                "engulf_ratio":       engulf_ratio,
            })

    ds = pd.DataFrame.from_records(records)
    ds["htf_aligned"] = (((ds["direction"]=="Long") &(ds["htf_trend"]=="Bullish")) |
                         ((ds["direction"]=="Short")&(ds["htf_trend"]=="Bearish"))).astype(int)
    ds["is_win_D"] = (ds["r_strategy_D_trail"] > 0).astype(int)
    return ds


def add_confluence(ds, df_ohlc):
    df = df_ohlc[["open","high","low","close"]].copy().astype(float)
    h,l,pc = df["high"],df["low"],df["close"].shift(1)
    df["atr"] = pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).rolling(14).mean()
    tick = max(float(df["close"].iloc[-1])*1e-5, 1e-6)
    levels, events = run_indicator(df[["open","high","low","close"]], min_range_ticks=3.0, tick_size=tick)
    print(f"  Highlander levels: {len(levels):,}")

    records_ev = [e.__dict__ for e in events]
    grouped = {}
    for r in records_ev:
        lid = int(r["level_id"])
        grouped.setdefault(lid,[]).append(r)

    def state_at(hist, target_bar):
        if not hist or hist[0]["bar"] > target_bar: return None
        state=-1; confirmed=False; alive=True
        for ev in hist:
            if ev["bar"] > target_bar: break
            e = ev["event"]
            if e=="level_created":    state=0
            elif e=="first_touch":    state=3
            elif e=="origin_confirmed": state=1
            elif e=="level_broken":   state=2
            elif e=="level_deleted":  alive=False
            confirmed = ev["confirmed"]
        return {"state":state,"confirmed":confirmed,"alive":alive,
                "direction":hist[0]["direction"],"price":hist[0]["price"]}

    ds = ds.copy()
    ds["engulf_time"]     = pd.to_datetime(ds["engulf_time"])
    ds["activation_time"] = pd.to_datetime(ds["activation_time"])
    bar_times = df.index.to_numpy()
    rows = []
    for _, row in ds.iterrows():
        act_bar = int(np.searchsorted(bar_times, row["activation_time"].to_numpy(), side="right")-1)
        act_bar = max(0, min(act_bar, len(df)-1))
        price   = float(df["close"].iloc[act_bar])
        atr_v   = float(df["atr"].iloc[act_bar]) if np.isfinite(df["atr"].iloc[act_bar]) else 1.0
        is_bull = row["direction"]=="Long"
        above,below = [],[]
        for lid,lhist in grouped.items():
            cb = next((h["bar"] for h in lhist if h["event"]=="level_created"),None)
            if cb is None or cb>act_bar: continue
            db = next((h["bar"] for h in lhist if h["event"]=="level_deleted"),None)
            if db is not None and db<=act_bar: continue
            s = state_at(lhist, act_bar)
            if s is None or not s["alive"] or s["state"]==2: continue
            diff = s["price"]-price
            info = {"direction":s["direction"],"is_origin":s["state"]==1,
                    "confirmed":s["confirmed"],
                    "dist_pct":abs(diff)/price*100,
                    "dist_atr":abs(diff)/atr_v if atr_v>0 else 99.0}
            (above if diff>0 else below).append(info)
        above.sort(key=lambda x:x["dist_pct"])
        below.sort(key=lambda x:x["dist_pct"])
        sup = (below[0] if below else None) if is_bull else (above[0] if above else None)
        res = (above[0] if above else None) if is_bull else (below[0] if below else None)
        g = lambda d,k,dfl: d[k] if d else dfl
        sup_dir = bool(sup) and ((is_bull and sup["direction"]==DIR_UP) or
                                 (not is_bull and sup["direction"]==DIR_DOWN))
        sup_atr = g(sup,"dist_atr",99.0)
        score = ((1 if sup_atr<1 else 0) +
                 (1 if g(sup,"is_origin",False) else 0) +
                 (1 if g(sup,"confirmed",False) else 0) +
                 (1 if sup_dir else 0) -
                 (1 if res and g(res,"dist_atr",99)<1.5 and g(res,"is_origin",False) else 0))
        rows.append({
            "support_present":      int(bool(sup)),
            "support_dist_pct":     round(g(sup,"dist_pct",99.0),3),
            "support_dist_atr":     round(sup_atr,3),
            "support_is_origin":    int(g(sup,"is_origin",False)),
            "support_confirmed":    int(g(sup,"confirmed",False)),
            "support_dir_matches":  int(sup_dir),
            "resistance_present":   int(bool(res)),
            "resistance_dist_pct":  round(g(res,"dist_pct",99.0),3),
            "resistance_dist_atr":  round(g(res,"dist_atr",99.0),3),
            "resistance_is_origin": int(g(res,"is_origin",False)),
            "resistance_confirmed": int(g(res,"confirmed",False)),
            "confluence_score":     score,
        })
    return pd.concat([ds, pd.DataFrame(rows, index=ds.index)], axis=1)


def add_entry_level_features(ds, df_ohlc):
    """
    Check whether the entry price is near any Highlander level (origin, first-touch,
    or broken) on the 1H and 4H timeframe at the time of the engulfing bar.
    """
    def _atr(df):
        h, l, pc = df["high"], df["low"], df["close"].shift(1)
        return pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).rolling(14).mean()

    def _state_at(hist, target_bar):
        if not hist or hist[0]["bar"] > target_bar: return None
        state=-1; confirmed=False; alive=True
        for ev in hist:
            if ev["bar"] > target_bar: break
            e = ev["event"]
            if   e=="level_created":     state=0
            elif e=="first_touch":       state=3
            elif e=="origin_confirmed":  state=1
            elif e=="level_broken":      state=2
            elif e=="level_deleted":     alive=False
            confirmed = ev["confirmed"]
        return {"state":state,"confirmed":confirmed,"alive":alive,
                "direction":hist[0]["direction"],"price":hist[0]["price"]}

    def _build_grouped(events):
        g={}
        for ev in events:
            r=ev.__dict__; lid=int(r["level_id"])
            g.setdefault(lid,[]).append(r)
        return g

    df1 = df_ohlc[["open","high","low","close"]].copy().astype(float)
    df4 = (df_ohlc[["open","high","low","close","volume"]] if "volume" in df_ohlc.columns
           else df_ohlc[["open","high","low","close"]]).copy().astype(float)
    df4 = df1.resample("4h", label="left").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    df1["atr"] = _atr(df1)
    df4["atr"] = _atr(df4)

    tick1 = max(float(df1["close"].iloc[-1])*1e-5, 1e-6)
    tick4 = max(float(df4["close"].iloc[-1])*1e-5, 1e-6)
    _, ev1 = run_indicator(df1[["open","high","low","close"]], min_range_ticks=3.0, tick_size=tick1)
    _, ev4 = run_indicator(df4[["open","high","low","close"]], min_range_ticks=3.0, tick_size=tick4)
    g1 = _build_grouped(ev1)
    g4 = _build_grouped(ev4)
    bt1 = df1.index.to_numpy()
    bt4 = df4.index.to_numpy()

    def _nearest(grouped, eng_bar, df_tf, entry, is_bull):
        atr_v = float(df_tf["atr"].iloc[eng_bar])
        if not np.isfinite(atr_v) or atr_v<=0: atr_v = abs(entry)*0.01 or 1.0
        best=None; best_dist=float("inf")
        for lid,lhist in grouped.items():
            cb=next((h["bar"] for h in lhist if h["event"]=="level_created"),None)
            if cb is None or cb>eng_bar: continue
            db=next((h["bar"] for h in lhist if h["event"]=="level_deleted"),None)
            if db is not None and db<=eng_bar: continue
            s=_state_at(lhist,eng_bar)
            if s is None or s["state"] not in (1,2,3): continue
            dist=abs(s["price"]-entry)
            if dist<best_dist: best_dist=dist; best=s
        if best is None: return 0,99.0,0,0,0,0
        dist_atr=round(best_dist/atr_v,3)
        return (int(dist_atr<=2.0), dist_atr,
                int(best["state"]==1), int(best["state"]==3), int(best["state"]==2),
                int((is_bull and best["direction"]==DIR_UP) or
                    (not is_bull and best["direction"]==DIR_DOWN)))

    ds = ds.copy()
    ds["engulf_time"] = pd.to_datetime(ds["engulf_time"])
    rows=[]
    for _,row in ds.iterrows():
        eng_ts = row["engulf_time"].to_numpy()
        is_bull = row["direction"]=="Long"
        entry   = float(row["entry"])
        origin  = float(row["origin"])
        eb1 = min(max(0,int(np.searchsorted(bt1,eng_ts,side="right"))-1), len(df1)-1)
        eb4 = min(max(0,int(np.searchsorted(bt4,eng_ts,side="right"))-1), len(df4)-1)
        at1,d1,o1,ft1,b1,dm1       = _nearest(g1,eb1,df1,entry, is_bull)
        at4,d4,o4,ft4,b4,dm4       = _nearest(g4,eb4,df4,entry, is_bull)
        _,od1,oo1,oft1,ob1,odm1    = _nearest(g1,eb1,df1,origin,is_bull)
        _,od4,oo4,oft4,ob4,odm4    = _nearest(g4,eb4,df4,origin,is_bull)
        rows.append({
            "entry_1h_at_level":at1,  "entry_1h_dist_atr":d1,
            "entry_1h_is_origin":o1,  "entry_1h_is_ft":ft1,
            "entry_1h_is_broken":b1,  "entry_1h_dir_match":dm1,
            "entry_4h_at_level":at4,  "entry_4h_dist_atr":d4,
            "entry_4h_is_origin":o4,  "entry_4h_is_ft":ft4,
            "entry_4h_is_broken":b4,  "entry_4h_dir_match":dm4,
            "origin_1h_dist_atr":od1, "origin_1h_is_origin":oo1,
            "origin_1h_is_ft":oft1,   "origin_1h_is_broken":ob1,
            "origin_1h_dir_match":odm1,
            "origin_4h_dist_atr":od4, "origin_4h_is_origin":oo4,
            "origin_4h_is_ft":oft4,   "origin_4h_is_broken":ob4,
            "origin_4h_dir_match":odm4,
        })
    return pd.concat([ds, pd.DataFrame(rows, index=ds.index)], axis=1)


_DIST_COLS = ["support_dist_pct","support_dist_atr","resistance_dist_pct","resistance_dist_atr",
              "entry_1h_dist_atr","entry_4h_dist_atr",
              "origin_1h_dist_atr","origin_4h_dist_atr"]


def train_final_model(ds, symbol):
    """Train on full dataset and save model for live use."""
    slug = symbol.replace("/","").lower()
    ds = ds.copy()
    for col in _DIST_COLS:
        if col in ds.columns: ds[col] = ds[col].fillna(99.0)
    ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)
    for c in CAT_FEATURES:
        ds[c] = ds[c].astype(str)

    clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                             cat_features=cat_idx, verbose=0, random_seed=42, l2_leaf_reg=5)
    clf.fit(ds[FEATURES], ds["is_win_D"])
    model_path = DATA / f"catboost_clf_{slug}_1h.cbm"
    clf.save_model(str(model_path))
    print(f"  Model saved → {model_path}")
    return str(model_path)


def walk_forward(ds, symbol):
    for col in _DIST_COLS:
        if col in ds.columns: ds[col] = ds[col].fillna(99.0)
    ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)
    for c in CAT_FEATURES:
        ds[c] = ds[c].astype(str)
    ds["activation_time"] = pd.to_datetime(ds["activation_time"])

    t_start, t_end = ds["activation_time"].min(), ds["activation_time"].max()
    windows, ts = [], t_start + pd.DateOffset(months=MIN_TRAIN_MONTHS)
    while ts < t_end:
        te = min(ts + pd.DateOffset(months=STEP_MONTHS), t_end)
        windows.append((ts, te)); ts = te

    chunks = []; pos_w = 0; tot_w = 0
    for ws, we in windows:
        train = ds[ds["activation_time"] < ws]
        test  = ds[(ds["activation_time"]>=ws)&(ds["activation_time"]<we)].copy().reset_index(drop=True)
        if len(train)<100 or len(test)<10: continue
        clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                                 cat_features=cat_idx, verbose=0, random_seed=42, l2_leaf_reg=5)
        clf.fit(train[FEATURES], train["is_win_D"])
        p = clf.predict_proba(test[FEATURES])[:,1]
        test["p_win_wf"] = p
        m = p >= P_THRESHOLD
        net_r = test.loc[m,"r_strategy_D_trail"].sum() if m.sum()>0 else 0
        wr    = test.loc[m,"is_win_D"].mean()*100 if m.sum()>0 else 0
        pos   = "✅" if net_r>0 else "❌"
        pos_w += int(net_r>0); tot_w += 1
        print(f"  {pos}  {ws.date()}→{we.date()}:  train={len(train):4d}  filt={m.sum():3d}"
              f"  WR={wr:.1f}%  net={net_r:+.1f}R")
        chunks.append(test)

    if not chunks:
        print(f"  Not enough data for walk-forward")
        return pd.DataFrame(), 0, 0

    all_oos = pd.concat(chunks).sort_values("activation_time").reset_index(drop=True)
    m_all   = all_oos["p_win_wf"] >= P_THRESHOLD
    filtered = all_oos[m_all]

    # 3-outcome breakdown
    WIN_R  = 1.078667 - COST_R
    BE_R   = -COST_R
    STOP_R = -1.0 - COST_R
    outcome_map = {"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R}
    filtered = filtered.copy()
    filtered["r_correct"] = filtered["outcome"].map(outcome_map)

    n   = len(filtered)
    w   = (filtered["outcome"]=="win").sum()
    be  = (filtered["outcome"]=="be_stop").sum()
    fs  = (filtered["outcome"]=="full_stop").sum()
    nr  = filtered["r_correct"].sum()
    pos_r = filtered.loc[filtered["r_correct"]>0,"r_correct"].sum()
    neg_r = abs(filtered.loc[filtered["r_correct"]<0,"r_correct"].sum())
    pf    = pos_r/neg_r if neg_r>0 else 999
    months_span = (filtered["activation_time"].max()-filtered["activation_time"].min()).days/30.44

    print(f"\n{'═'*60}")
    print(f"  RESULTS  —  {symbol} 1H")
    print(f"{'═'*60}")
    print(f"  Date range     : {filtered['activation_time'].min().date()} → {filtered['activation_time'].max().date()}")
    print(f"  Signals total  : {n}  over {months_span:.0f} months  ({n/months_span:.1f}/month)")
    print(f"  Wins           : {w}  ({w/n*100:.1f}%)")
    print(f"  BE stops       : {be}  ({be/n*100:.1f}%)")
    print(f"  Full stops     : {fs}  ({fs/n*100:.1f}%)")
    print(f"  Profit factor  : {pf:.2f}")
    print(f"  Net R / month  : {nr/months_span:+.1f}R")
    print(f"  At $130/trade  : ${nr/months_span*130:,.0f}/month")
    print(f"  Positive windows: {pos_w}/{tot_w}")
    return filtered, pos_w, tot_w


if __name__ == "__main__":
    import ccxt

    # Use Bitget futures feed to match live bot
    exchange = None
    for ex_id in ("bitget",):
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
            ex.load_markets()
            print(f"  Exchange: {ex_id} (futures)")
            exchange = ex
            break
        except Exception as e:
            print(f"  {ex_id} failed: {e!r:.50}")
    if exchange is None:
        for ex_id in ("bybit","okx","gate"):
            try:
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
                ex.load_markets()
                print(f"  Exchange: {ex_id} (fallback)")
                exchange = ex
                break
            except Exception as e:
                print(f"  {ex_id} failed: {e!r:.50}")
    if exchange is None:
        raise RuntimeError("No exchange available")

    # BTC 4H for macro trend
    btc_path = DATA / "btc_4h.csv"
    if btc_path.exists():
        btc_df = pd.read_csv(btc_path, parse_dates=["time"]).set_index("time").astype(float)
        print(f"  BTC 4H: {len(btc_df):,} bars (cached)")
    else:
        btc_df = fetch_all(exchange, "BTC/USDT", "4h", "2018-01-01")
        btc_df.index.name = "time"
        btc_df.to_csv(btc_path)

    for symbol, since in SYMBOLS_TO_TEST:
        slug     = symbol.replace("/","").lower()
        csv_path = DATA / f"{slug}_1h.csv"

        print(f"\n{'█'*60}")
        print(f"  {symbol}  1H  PIPELINE")
        print(f"{'█'*60}")

        # Fetch or load (re-fetch if cached file is too short < 5000 bars)
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time").astype(float)
            if len(df) >= 5000:
                print(f"  Loaded {len(df):,} bars from cache  ({df.index[0].date()} → {df.index[-1].date()})")
            else:
                print(f"  Cached file too short ({len(df)} bars) — re-fetching …")
                csv_path.unlink()
                df = None

        if not csv_path.exists():
            df = fetch_binance_direct(symbol, "1h", since)
            if df is None:
                if symbol not in exchange.markets:
                    print(f"  {symbol} not on {exchange.id}, skipping"); continue
                df = fetch_all(exchange, symbol, "1h", since)
            df.index.name = "time"
            df.to_csv(csv_path)
            print(f"  Saved → {csv_path}")

        print(f"\n  Step 1 — Building trades …")
        ds = build_trades(df, btc_df)
        raw_wr = (ds["is_win_D"]==1).mean()*100
        print(f"  Trades: {len(ds):,}  Raw WR (no filter): {raw_wr:.1f}%")

        print(f"\n  Step 2 — Adding confluence …")
        ds = add_confluence(ds, df)

        print(f"\n  Step 3 — Walk-forward …")
        walk_forward(ds, symbol)

        print(f"\n  Step 4 — Training final model …")
        train_final_model(ds, symbol)

    print("\n\nDone.")
