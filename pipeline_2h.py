"""
pipeline_2h.py — Full engulfing + ML pipeline on 2H timeframe.

Loads existing 1H CSVs, resamples to 2H, runs full pipeline for SUI and SOL.
"""
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
R_COL            = "r_strategy_D_trail"
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
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def session_of(h):
    if 0  <= h < 7:  return "Asia"
    if 7  <= h < 13: return "London"
    if 13 <= h < 17: return "NY-AM"
    if 17 <= h < 22: return "NY-PM"
    return "Off-hours"

def load_1h(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")[["open","high","low","close","volume"]].astype(float)

def resample_2h(df1h):
    return df1h.resample("2h").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"),    close=("close","last"),
        volume=("volume","sum")
    ).dropna()

def detect_engulfing(df):
    O,H,L,C = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    recs = []
    for i in range(1, len(df)-1):
        bull = (C[i-1]<O[i-1]) and (C[i]>O[i]) and (C[i]>H[i-1])
        bear = (C[i-1]>O[i-1]) and (C[i]<O[i]) and (C[i]<L[i-1])
        if bull: recs.append({"bar":i,"is_bull":True, "origin":L[i-1],"entry":H[i-1],"rng":H[i-1]-L[i-1]})
        if bear: recs.append({"bar":i,"is_bull":False,"origin":H[i-1],"entry":L[i-1],"rng":H[i-1]-L[i-1]})
    return pd.DataFrame(recs)

def find_activation(H, L, bar, is_bull, entry, max_b):
    for j in range(bar+1, min(bar+1+max_b, len(H))):
        if (is_bull and L[j]<=entry) or (not is_bull and H[j]>=entry):
            return j
    return None

def simulate_fib(H, L, act_bar, is_bull, entry, stop, f1618, f2618, max_b):
    end = min(act_bar+1+max_b, len(H))
    risk = abs(entry-stop)
    hit1=False; hit2=False; stopped=False; mfe=0.0
    for j in range(act_bar+1, end):
        hi, lo = H[j], L[j]
        if (is_bull and lo<=stop) or (not is_bull and hi>=stop): stopped=True; break
        if (is_bull and hi>=f1618) or (not is_bull and lo<=f1618): hit1=True
        if (is_bull and hi>=f2618) or (not is_bull and lo<=f2618): hit2=True
        exc = (hi-entry)/risk if is_bull else (entry-lo)/risk
        mfe = max(mfe, exc)
    return stopped, hit1, hit2, mfe


def run_backtest(df2h, btc_df, label):
    print(f"\n{'═'*60}\n  {label} 2H — Engulfing backtest\n{'═'*60}")
    df = df2h.copy()
    df["atr"]       = true_range(df).rolling(ATR_LEN).mean()
    df["atr_pct"]   = df["atr"] / df["close"]
    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg"]

    # HTF: resample 2H to 8H (equivalent of 4H from 1H)
    htf = df[["close"]].resample("8h").last().dropna()
    htf["sma20"] = htf["close"].rolling(20).mean()
    htf["sma50"] = htf["close"].rolling(50).mean()
    htf["t"] = np.where(htf["sma20"]>htf["sma50"], 1, np.where(htf["sma20"]<htf["sma50"], -1, 0))
    idx = np.clip(htf.index.searchsorted(df.index, side="right")-1, 0, len(htf)-1)
    df["htf_label"] = np.where(htf["t"].values[idx]==1, "Bullish",
                      np.where(htf["t"].values[idx]==-1, "Bearish", "Range"))

    # BTC trend
    btc = btc_df.copy()
    btc["sma20"] = btc["close"].rolling(20).mean()
    btc["sma50"] = btc["close"].rolling(50).mean()
    btc["t"] = np.where(btc["sma20"]>btc["sma50"], 1, np.where(btc["sma20"]<btc["sma50"], -1, 0))
    bidx = np.clip(btc.index.searchsorted(df.index, side="right")-1, 0, len(btc)-1)
    df["btc_label"] = np.where(btc["t"].values[bidx]==1, "Up",
                      np.where(btc["t"].values[bidx]==-1, "Down", "Range"))

    engs = detect_engulfing(df)
    print(f"  Engulfings: {len(engs):,}  ({engs['is_bull'].sum()} bull / {(~engs['is_bull']).sum()} bear)")

    H, L = df["high"].values, df["low"].values
    records = []; skipped = 0
    for _, e in engs.iterrows():
        bar = int(e["bar"])
        if bar < ATR_LEN+5: skipped+=1; continue
        atr = float(df["atr"].iloc[bar])
        if not np.isfinite(atr) or atr<=0: skipped+=1; continue
        is_bull = bool(e["is_bull"])
        origin, entry, rng = float(e["origin"]), float(e["entry"]), float(e["rng"])
        if rng<=0: skipped+=1; continue
        m = 1 if is_bull else -1
        stop  = origin - 0.5*rng*m
        f1618 = origin + 1.618*rng*m
        f2618 = origin + 2.618*rng*m
        act_bar = find_activation(H, L, bar, is_bull, entry, MAX_BARS_TO_ACT)
        if act_bar is None: skipped+=1; continue
        stopped, hit1, hit2, mfe = simulate_fib(H, L, act_bar, is_bull, entry, stop, f1618, f2618, MAX_BARS_AFTER)
        if stopped and not hit1:   r_D = -1.0
        elif hit1 and not hit2:    r_D =  0.0
        else: r_D = 2.618 if hit2 else mfe
        ts = df.index[act_bar]
        vr = float(df["vol_ratio"].iloc[bar])
        records.append({
            "engulf_time":        str(df.index[bar]),
            "activation_time":    str(ts),
            "direction":          "Long" if is_bull else "Short",
            "engulf_bar":         bar, "activation_bar": act_bar,
            "origin": origin, "entry": entry, "stop": stop,
            "fib_1618": f1618, "fib_2618": f2618, "rng": rng,
            "rng_pct":            round(rng/entry*100, 3),
            "stop_pct":           round(abs(entry-stop)/entry*100, 3),
            "tp1618_pct":         round(abs(f1618-entry)/entry*100, 3),
            R_COL:                round(r_D-COST_R, 3),
            "bars_to_activation": act_bar-bar,
            "atr_pct_at_act":     round(float(df["atr_pct"].iloc[act_bar]), 5),
            "atr_pct_at_eng":     round(float(df["atr_pct"].iloc[bar]), 5),
            "htf_trend":          str(df["htf_label"].iloc[act_bar]),
            "btc_trend":          str(df["btc_label"].iloc[act_bar]),
            "session":            session_of(ts.hour),
            "hour":               int(ts.hour),
            "day":                ts.day_name()[:3],
            "month":              int(ts.month),
            "year":               int(ts.year),
            "vol_ratio":          round(vr if np.isfinite(vr) else 1.0, 3),
            "rng_to_atr":         round(rng/atr, 3),
        })

    ds = pd.DataFrame.from_records(records)
    ds["htf_aligned"] = (((ds["direction"]=="Long")  & (ds["htf_trend"]=="Bullish")) |
                         ((ds["direction"]=="Short") & (ds["htf_trend"]=="Bearish"))).astype(int)
    print(f"  Trades: {len(ds):,}  skipped {skipped}")
    wr = (ds[R_COL]>0).mean()
    print(f"  Raw (no filter): win {wr:.1%}  avgR {ds[R_COL].mean():+.3f}")
    return ds


def add_confluence(ds, df2h):
    print(f"\n  Adding Highlander confluence …")
    df = df2h[["open","high","low","close"]].astype(float).copy()
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    df["atr"] = pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).rolling(14).mean()
    tick = max(float(df["close"].iloc[-1])*1e-5, 1e-6)
    levels, events = run_indicator(df[["open","high","low","close"]], min_range_ticks=3.0, tick_size=tick)
    print(f"  Highlander levels: {len(levels):,}")

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
        state=-1; confirmed=False; alive=True
        for ev in hist:
            if ev["bar"] > target_bar: break
            e = ev["event"]
            if e=="level_created":      state=0
            elif e=="first_touch":      state=3
            elif e=="origin_confirmed": state=1
            elif e=="level_broken":     state=2
            elif e=="level_deleted":    alive=False
            confirmed = ev["confirmed"]
        return {"state":state,"confirmed":confirmed,"alive":alive,
                "direction":hist[0]["direction"],"price":hist[0]["price"]}

    ds = ds.copy()
    ds["activation_time"] = pd.to_datetime(ds["activation_time"])
    bar_times = df.index.to_numpy()
    rows = []
    for _, row in ds.iterrows():
        act_bar = int(np.searchsorted(bar_times, row["activation_time"].to_numpy(), side="right")-1)
        act_bar = max(0, min(act_bar, len(df)-1))
        price   = float(df["close"].iloc[act_bar])
        atr_v   = float(df["atr"].iloc[act_bar]) if np.isfinite(df["atr"].iloc[act_bar]) else 1.0
        is_bull = row["direction"]=="Long"
        above, below = [], []
        for lid, m in meta.items():
            if m["created_bar"] is None or m["created_bar"]>act_bar: continue
            if m["deleted_bar"] is not None and m["deleted_bar"]<=act_bar: continue
            s = state_at(grouped[lid], act_bar)
            if s is None or not s["alive"] or s["state"]==2: continue
            diff = s["price"]-price
            info = {"direction":s["direction"],"state":s["state"],"confirmed":s["confirmed"],
                    "is_origin":s["state"]==1,
                    "dist_pct":abs(diff)/price*100,
                    "dist_atr":abs(diff)/atr_v if atr_v>0 else 99.0}
            (above if diff>0 else below).append(info)
        above.sort(key=lambda x: x["dist_pct"])
        below.sort(key=lambda x: x["dist_pct"])
        sup = (below[0] if below else None) if is_bull else (above[0] if above else None)
        res = (above[0] if above else None) if is_bull else (below[0] if below else None)
        g   = lambda d, k, default: d[k] if d else default
        sup_dir = bool(sup) and ((is_bull and sup["direction"]==DIR_UP) or
                                  (not is_bull and sup["direction"]==DIR_DOWN))
        sup_atr = g(sup,"dist_atr",99.0)
        score = ((1 if sup_atr<1 else 0)+(1 if g(sup,"is_origin",False) else 0)+
                 (1 if g(sup,"confirmed",False) else 0)+(1 if sup_dir else 0)-
                 (1 if res and g(res,"dist_atr",99)<1.5 and g(res,"is_origin",False) else 0))
        rows.append({
            "support_present":int(bool(sup)),"support_dist_pct":round(g(sup,"dist_pct",99.0),3),
            "support_dist_atr":round(sup_atr,3),"support_is_origin":int(g(sup,"is_origin",False)),
            "support_confirmed":int(g(sup,"confirmed",False)),"support_dir_matches":int(sup_dir),
            "resistance_present":int(bool(res)),"resistance_dist_pct":round(g(res,"dist_pct",99.0),3),
            "resistance_dist_atr":round(g(res,"dist_atr",99.0),3),
            "resistance_is_origin":int(g(res,"is_origin",False)),
            "resistance_confirmed":int(g(res,"confirmed",False)),"confluence_score":score,
        })
    conf_df = pd.DataFrame(rows, index=ds.index)
    return pd.concat([ds, conf_df], axis=1)


def walk_forward(ds, sym_slug):
    print(f"\n  Walk-forward validation …")
    ds = ds.copy()
    for col in ["support_dist_pct","support_dist_atr","resistance_dist_pct","resistance_dist_atr"]:
        ds[col] = ds[col].fillna(99.0)
    ds["vol_ratio"] = ds["vol_ratio"].fillna(1.0)
    for c in CAT_FEATURES: ds[c] = ds[c].astype(str)
    ds["is_win_D"] = (ds[R_COL]>0).astype(int)
    ds["activation_time"] = pd.to_datetime(ds["activation_time"])

    t_start, t_end = ds["activation_time"].min(), ds["activation_time"].max()
    windows, ts = [], t_start+pd.DateOffset(months=MIN_TRAIN_MONTHS)
    while ts < t_end:
        te = min(ts+pd.DateOffset(months=STEP_MONTHS), t_end)
        windows.append((ts, te)); ts = te

    results, all_oos = [], []
    for ws, we in windows:
        train = ds[ds["activation_time"] <  ws]
        test  = ds[(ds["activation_time"]>=ws)&(ds["activation_time"]<we)].copy().reset_index(drop=True)
        if len(train)<100 or len(test)<10: continue
        clf = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                                  cat_features=cat_idx, verbose=0, random_seed=42, l2_leaf_reg=5)
        clf.fit(train[FEATURES], train["is_win_D"])
        p = clf.predict_proba(test[FEATURES])[:,1]
        test["p_win"] = p
        m = p>=P_THRESHOLD
        if m.sum()==0: continue
        filt = test.loc[m]
        actual_rr = (filt["fib_2618"]-filt["entry"]).abs()/(filt["entry"]-filt["stop"]).abs()
        wins      = filt[filt["is_win_D"]==1]
        full_stop = filt[filt[R_COL]<=-0.9]
        be_stop   = filt[(filt[R_COL]>-0.9)&(filt[R_COL]<0)]
        gp = len(wins)*actual_rr[filt["is_win_D"]==1].mean() if len(wins)>0 else 0
        gl = len(full_stop)*1.04+len(be_stop)*0.04
        net = gp-gl
        results.append({
            "window": f"{ws.date()}→{we.date()}",
            "n": m.sum(), "wr": filt["is_win_D"].mean(),
            "actual_rr": actual_rr[filt["is_win_D"]==1].mean() if len(wins)>0 else 0,
            "net_r": net, "positive": net>0,
        })
        all_oos.append(test)
        print(f"  {ws.date()}→{we.date()}:  N={m.sum():3d}  WR={filt['is_win_D'].mean():.1%}  net={net:+.1f}R  {'✅' if net>0 else '❌'}")

    if not results:
        print("  Not enough data for walk-forward"); return pd.DataFrame()

    all_oos_df = pd.concat(all_oos).reset_index(drop=True)
    slug = f"{sym_slug}_2h"
    all_oos_df.to_csv(DATA/f"{slug}_test_predictions.csv", index=False)

    # Train final model on all data
    clf_final = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=5,
                                    cat_features=cat_idx, verbose=0, random_seed=42, l2_leaf_reg=5)
    clf_final.fit(ds[FEATURES], ds["is_win_D"])
    clf_final.save_model(str(DATA/f"catboost_clf_{slug}.cbm"))
    print(f"  Model saved → data/catboost_clf_{slug}.cbm")

    res_df = pd.DataFrame(results)
    print(f"\n  Windows: {len(res_df)}  positive: {res_df['positive'].sum()}/{len(res_df)}")
    return all_oos_df


def report(sym, all_oos):
    print(f"\n{'═'*60}")
    print(f"  FINAL RESULTS — {sym} 2H")
    print(f"{'═'*60}")
    sig = all_oos[all_oos["p_win"]>=P_THRESHOLD].copy()
    if len(sig)==0: print("  No signals"); return
    sig["activation_time"] = pd.to_datetime(sig["activation_time"])
    months = sig["activation_time"].dt.to_period("M").nunique()
    wins      = sig[sig["is_win_D"]==1]
    full_stop = sig[sig[R_COL]<=-0.9]
    be_stop   = sig[(sig[R_COL]>-0.9)&(sig[R_COL]<0)]
    actual_rr = (wins["fib_2618"]-wins["entry"]).abs()/(wins["entry"]-wins["stop"]).abs()
    avg_rr    = actual_rr.mean() if len(wins)>0 else 0
    gp = len(wins)*avg_rr
    gl = len(full_stop)*1.04+len(be_stop)*0.04
    net = gp-gl
    pf  = gp/gl if gl>0 else float("inf")
    print(f"  Signals/month    : {len(sig)/months:.1f}  ({len(sig)} total, {months} months)")
    print(f"  Win rate         : {sig['is_win_D'].mean()*100:.1f}%")
    print(f"  Full stops       : {len(full_stop)/len(sig)*100:.1f}%  ({len(full_stop)})")
    print(f"  BE stops         : {len(be_stop)/len(sig)*100:.1f}%  ({len(be_stop)})")
    print(f"  Avg actual RR    : 1 : {avg_rr:.3f}")
    print(f"  Profit factor    : {pf:.2f}")
    print(f"  Net R / month    : {net/months:.1f}R  (actual)")
    print(f"  At $130 / trade  : ${net/months*130:.0f}/month")
    print(f"  At $195 / trade  : ${net/months*195:.0f}/month")


if __name__ == "__main__":
    btc_df = pd.read_csv(DATA/"btc_4h.csv", index_col=0, parse_dates=True)

    for sym, csv_path in SYMBOLS.items():
        print(f"\n\n{'█'*60}")
        print(f"  {sym}  2H PIPELINE")
        print(f"{'█'*60}")
        sym_slug = sym.replace("/","").lower()
        df1h = load_1h(csv_path)
        df2h = resample_2h(df1h)
        print(f"  2H bars: {len(df2h):,}  ({df2h.index[0].date()} → {df2h.index[-1].date()})")

        ds       = run_backtest(df2h, btc_df, sym)
        ds       = add_confluence(ds, df2h)
        all_oos  = walk_forward(ds, sym_slug)
        if len(all_oos): report(sym, all_oos)
