"""Check recent SOL engulfing — why isn't it in /pending?"""
import ccxt, pandas as pd, numpy as np
from catboost import CatBoostClassifier
from highlander import run_indicator, DIR_UP, DIR_DOWN, STATE_BROKEN_BSUT, STATE_ORIGIN

ATR_LEN = 14
CAT_FEATURES = ["direction","session","htf_trend","btc_trend","day"]
NUM_FEATURES = [
    "bars_to_activation","atr_pct_at_act","atr_pct_at_eng","rng_pct","stop_pct",
    "tp1618_pct","rng_to_atr","vol_ratio","hour","month","htf_aligned","support_present",
    "support_dist_pct","support_dist_atr","support_is_origin","support_confirmed",
    "support_dir_matches","resistance_present","resistance_dist_pct","resistance_dist_atr",
    "resistance_is_origin","resistance_confirmed","confluence_score",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
cat_idx  = [FEATURES.index(c) for c in CAT_FEATURES]

def session_of(h):
    if 0<=h<7:  return "Asia"
    if 7<=h<13: return "London"
    if 13<=h<17: return "NY-AM"
    if 17<=h<22: return "NY-PM"
    return "Off-hours"

ex = ccxt.bybit({"enableRateLimit": True})

df = pd.DataFrame(ex.fetch_ohlcv("SOL/USDT","1h",limit=600),
    columns=["time","open","high","low","close","volume"])
df["time"] = pd.to_datetime(df["time"],unit="ms",utc=True).dt.tz_localize(None)
df = df.set_index("time").astype(float)
h,l,pc = df["high"],df["low"],df["close"].shift(1)
df["atr"]     = pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).ewm(span=ATR_LEN,adjust=False).mean()
df["vol_avg"] = df["volume"].rolling(20).mean()

sym4h = pd.DataFrame(ex.fetch_ohlcv("SOL/USDT","4h",limit=200),
    columns=["time","open","high","low","close","volume"])
sym4h["time"] = pd.to_datetime(sym4h["time"],unit="ms",utc=True).dt.tz_localize(None)
sym4h = sym4h.set_index("time").astype(float)

btc4h = pd.DataFrame(ex.fetch_ohlcv("BTC/USDT","4h",limit=200),
    columns=["time","open","high","low","close","volume"])
btc4h["time"] = pd.to_datetime(btc4h["time"],unit="ms",utc=True).dt.tz_localize(None)
btc4h = btc4h.set_index("time").astype(float)

htf_s = pd.Series(
    np.where(sym4h["close"].rolling(20).mean()>sym4h["close"].rolling(50).mean(),"Bullish",
    np.where(sym4h["close"].rolling(20).mean()<sym4h["close"].rolling(50).mean(),"Bearish","Range")),
    index=sym4h.index)
btc_s = pd.Series(
    np.where(btc4h["close"].rolling(20).mean()>btc4h["close"].rolling(50).mean(),"Up",
    np.where(btc4h["close"].rolling(20).mean()<btc4h["close"].rolling(50).mean(),"Down","Range")),
    index=btc4h.index)

levels, _ = run_indicator(df[["open","high","low","close"]])

O=df["open"].values; H=df["high"].values; L=df["low"].values; C=df["close"].values; T=df.index
cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=8)

def lkp(s, ts):
    pos = np.clip(s.index.searchsorted(ts, side="right")-1, 0, len(s)-1)
    return s.iloc[pos]

print("SOL/USDT — engulfings in last 8 hours\n")
found = False
for i in range(max(0,len(df)-12), len(df)-1):
    bull = (C[i-1]<O[i-1]) and (C[i]>O[i]) and (C[i]>H[i-1])
    bear = (C[i-1]>O[i-1]) and (C[i]<O[i]) and (C[i]<L[i-1])
    if not (bull or bear): continue
    if T[i] < cutoff: continue
    found = True

    is_bull  = bull
    entry    = H[i-1] if is_bull else L[i-1]
    rng      = H[i-1]-L[i-1]
    stop     = entry - 1.5*rng if is_bull else entry + 1.5*rng
    f2618    = entry + abs(entry-stop)*(1.618/1.5) if is_bull else entry - abs(entry-stop)*(1.618/1.5)
    atr_i    = float(df["atr"].iloc[i])
    vol_r    = float(df["volume"].iloc[i]) / max(float(df["vol_avg"].iloc[i]),1e-9)
    direction = "Long" if is_bull else "Short"
    htf_str  = lkp(htf_s, T[i])
    btc_str  = lkp(btc_s, T[i])
    htf_aligned = int((is_bull and htf_str=="Bullish") or (not is_bull and htf_str=="Bearish"))

    # Confluence
    above, below = [], []
    for lv in levels:
        if lv.created_bar > i: continue
        if lv.deleted_bar is not None and lv.deleted_bar <= i: continue
        if lv.state == STATE_BROKEN_BSUT: continue
        diff = lv.price - entry
        info = dict(direction=lv.dir, is_origin=(lv.state==STATE_ORIGIN),
                    confirmed=lv.confirmed,
                    dist_pct=abs(diff)/entry*100,
                    dist_atr=abs(diff)/atr_i if atr_i>0 else 99.0)
        (above if diff>0 else below).append(info)
    above.sort(key=lambda x:x["dist_pct"])
    below.sort(key=lambda x:x["dist_pct"])
    g   = lambda d,k,dfl: d[k] if d else dfl
    sup = (below[0] if below else None) if is_bull else (above[0] if above else None)
    res = (above[0] if above else None) if is_bull else (below[0] if below else None)
    sup_dir = bool(sup) and ((is_bull and sup["direction"]==DIR_UP) or (not is_bull and sup["direction"]==DIR_DOWN))
    sup_atr = g(sup,"dist_atr",99.0)
    score   = (
        (1 if sup_atr<1 else 0) +
        (1 if g(sup,"is_origin",False) else 0) +
        (1 if g(sup,"confirmed",False) else 0) +
        (1 if sup_dir else 0) -
        (1 if res and g(res,"dist_atr",99)<1.5 and g(res,"is_origin",False) else 0)
    )

    # Activated?
    activated = False
    act_i = None
    for j in range(i+1, min(i+51, len(df))):
        if (is_bull and L[j]<=entry) or (not is_bull and H[j]>=entry):
            activated = True; act_i = j; break

    print(f"{'='*60}")
    print(f"  {T[i]}  {direction}")
    print(f"  Entry: {entry:.4f}   Stop: {stop:.4f}   TP: {f2618:.4f}")
    print(f"  HTF: {htf_str}   BTC: {btc_str}   Aligned: {bool(htf_aligned)}")
    print(f"  Range: {rng:.4f}   ATR: {atr_i:.4f}   Range/ATR: {rng/atr_i:.2f}")
    print()
    if sup:
        print(f"  Support level : {g(sup,'dist_pct',99):.3f}% away  origin={g(sup,'is_origin',False)}  confirmed={g(sup,'confirmed',False)}  dir_match={sup_dir}")
    else:
        print(f"  Support level : NONE below entry  ← no support confluence for this bull setup")
    if res:
        print(f"  Resistance    : {g(res,'dist_pct',99):.3f}% away  origin={g(res,'is_origin',False)}  confirmed={g(res,'confirmed',False)}")
    print(f"  Confluence score: {score}")
    print()

    if activated:
        print(f"  Already ACTIVATED at {T[act_i]}")
    else:
        print(f"  PENDING — price hasn't retested {entry:.4f} yet")

    # ML score
    row = {
        "direction": direction, "session": session_of(T[i].hour),
        "htf_trend": htf_str, "btc_trend": btc_str, "day": T[i].day_name()[:3],
        "bars_to_activation": 1, "atr_pct_at_act": atr_i/entry*100, "atr_pct_at_eng": atr_i/entry*100,
        "rng_pct": rng/entry*100, "stop_pct": abs(entry-stop)/entry*100,
        "tp1618_pct": abs(f2618-entry)/entry*100, "rng_to_atr": rng/atr_i if atr_i>0 else 1,
        "vol_ratio": vol_r, "hour": T[i].hour, "month": T[i].month, "htf_aligned": htf_aligned,
        "support_present": int(bool(sup)), "support_dist_pct": round(g(sup,"dist_pct",99.0),3),
        "support_dist_atr": round(sup_atr,3), "support_is_origin": int(g(sup,"is_origin",False)),
        "support_confirmed": int(g(sup,"confirmed",False)), "support_dir_matches": int(sup_dir),
        "resistance_present": int(bool(res)), "resistance_dist_pct": round(g(res,"dist_pct",99.0),3),
        "resistance_dist_atr": round(g(res,"dist_atr",99.0),3),
        "resistance_is_origin": int(g(res,"is_origin",False)),
        "resistance_confirmed": int(g(res,"confirmed",False)), "confluence_score": score,
    }
    X = pd.DataFrame([row])
    for c in CAT_FEATURES: X[c] = X[c].astype(str)
    model = CatBoostClassifier()
    model.load_model("data/catboost_clf_engulfing_1h.cbm")
    p = model.predict_proba(X[FEATURES])[0,1]
    if p >= 0.50:
        print(f"  ML score: {p:.3f}  → ✅ PASSES — bot SHOULD show this in /pending or /filtered")
    else:
        print(f"  ML score: {p:.3f}  → ❌ FILTERED OUT (score < 0.50) — appears in /filtered, not /pending")

if not found:
    print("No engulfings in last 8 hours — market has not produced a qualifying setup")
