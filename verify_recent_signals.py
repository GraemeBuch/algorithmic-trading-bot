"""
verify_recent_signals.py

Downloads fresh 1H data from Bitget and runs the exact same signal detection
and ML scoring as the backtest. Shows activated trades with outcomes (win/BE/stop)
resolved using 1H bar data.
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import ccxt
from pathlib import Path
from catboost import CatBoostClassifier

from test_new_symbols import add_confluence, FEATURES, CAT_FEATURES, P_THRESHOLD
from sol_1min_backtest import true_range, session_of, ATR_LEN, WIN_R, BE_R, STOP_R
from highlander import run_indicator, DIR_UP, DIR_DOWN
from live_signals import get_entry_level_feat, calc_htf_trend, fetch_ohlcv, _fsym

DATA     = Path("data")
LOOKBACK = 900   # 1H bars (~37 days)
DAYS     = 30

SYMBOLS = {
    "BTC/USDT":  "data/catboost_clf_btcusdt_1h.cbm",
    "ETH/USDT":  "data/catboost_clf_ethusdt_1h.cbm",
    "SOL/USDT":  "data/catboost_clf_solusdt_1h.cbm",
    "TRX/USDT":  "data/catboost_clf_trxusdt_1h.cbm",
    "LINK/USDT": "data/catboost_clf_linkusdt_1h.cbm",
    "HBAR/USDT": "data/catboost_clf_hbarusdt_1h.cbm",
    "XRP/USDT":  "data/catboost_clf_xrpusdt_1h.cbm",
    "DOGE/USDT": "data/catboost_clf_dogeusdt_1h.cbm",
    "LTC/USDT":  "data/catboost_clf_ltcusdt_1h.cbm",
    "ADA/USDT":  "data/catboost_clf_adausdt_1h.cbm",
    "XLM/USDT":  "data/catboost_clf_xlmusdt_1h.cbm",
    "UNI/USDT":  "data/catboost_clf_uniusdt_1h.cbm",
    "APT/USDT":  "data/catboost_clf_aptusdt_1h.cbm",
    "NEAR/USDT": "data/catboost_clf_nearusdt_1h.cbm",
}


def resolve_outcome_1h(H_arr, L_arr, start_idx, entry, stop, f1618, f2618, is_bull, max_bars=200):
    """Resolve trade outcome bar-by-bar on 1H data from start_idx."""
    be_hit   = False
    cur_stop = stop
    end      = min(start_idx + max_bars, len(H_arr))
    for j in range(start_idx, end):
        h, l = H_arr[j], L_arr[j]
        if is_bull:
            if h >= f2618:
                return "win", j
            if h >= f1618 and not be_hit:
                if l <= cur_stop:
                    return "full_stop", j  # stop and BE on same bar — stop wins
                be_hit, cur_stop = True, entry
            elif l <= cur_stop:
                return ("be_stop" if be_hit else "full_stop"), j
        else:
            if l <= f2618:
                return "win", j
            if l <= f1618 and not be_hit:
                if h >= cur_stop:
                    return "full_stop", j
                be_hit, cur_stop = True, entry
            elif h >= cur_stop:
                return ("be_stop" if be_hit else "full_stop"), j
    return None, end - 1  # still open / timed out


def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr"]     = true_range(df).rolling(ATR_LEN).mean()
    df["atr_pct"] = df["atr"] / df["close"]
    vol_ma        = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / vol_ma.replace(0, np.nan)
    df["hour"]    = df.index.hour
    df["month"]   = df.index.month
    df["day"]     = df.index.day_name()
    return df


def detect_signals(df1h, df4h, btc4h, clf, symbol, cutoff):
    df   = prep_df(df1h)
    df4h = prep_df(df4h)

    htf_trend = calc_htf_trend(df4h)
    btc_trend = calc_htf_trend(btc4h)

    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    ts_idx = df.index

    tick = max(float(df["close"].iloc[-1]) * 1e-5, 1e-6)
    levels, _ = run_indicator(df[["open","high","low","close"]],
                              min_range_ticks=3.0, tick_size=tick)

    tick4 = max(float(df4h["close"].iloc[-1]) * 1e-5, 1e-6)
    levels_4h, _ = run_indicator(df4h[["open","high","low","close"]],
                                 min_range_ticks=3.0, tick_size=tick4)

    df4h_idx = df4h.index.values.astype("datetime64[ns]")
    df4h_atr = df4h["atr"].values

    results = []
    for i in range(1, len(df)):
        t = ts_idx[i]
        if t < cutoff:
            continue

        bull = (C[i-1] < O[i-1]) and (C[i] > O[i]) and (C[i] > H[i-1])
        bear = (C[i-1] > O[i-1]) and (C[i] < O[i]) and (C[i] < L[i-1])

        for is_bull in ([True]*int(bull) + [False]*int(bear)):
            origin = L[i-1] if is_bull else H[i-1]
            entry  = H[i-1] if is_bull else L[i-1]
            rng    = H[i-1] - L[i-1]
            if rng <= 0: continue
            atr_v = float(df["atr"].iloc[i])
            if not np.isfinite(atr_v) or atr_v <= 0: continue

            m     = 1 if is_bull else -1
            stop  = origin - 0.5 * rng * m
            f1618 = origin + 1.618 * rng * m
            f2618 = origin + 2.618 * rng * m

            eng_rng        = H[i] - L[i]
            _er            = eng_rng if eng_rng > 0 else 1e-10
            body_pct       = round(abs(C[i] - O[i]) / _er, 3)
            upper_wick_pct = round((H[i] - max(O[i], C[i])) / _er, 3)
            lower_wick_pct = round((min(O[i], C[i]) - L[i]) / _er, 3)
            engulf_ratio   = round(eng_rng / rng, 3) if rng > 0 else 1.0
            vr = float(df["vol_ratio"].iloc[i])

            i_4h   = max(0, int(np.searchsorted(df4h_idx, t.to_numpy(), side="right")) - 1)
            i_4h   = min(i_4h, len(df4h) - 1)
            atr_4h = float(df4h_atr[i_4h]) if np.isfinite(df4h_atr[i_4h]) else atr_v

            lf1h      = get_entry_level_feat(entry,  is_bull, atr_v,  levels,    i,    "entry_1h")
            lf4h      = get_entry_level_feat(entry,  is_bull, atr_4h, levels_4h, i_4h, "entry_4h")
            lf_orig1h = get_entry_level_feat(origin, is_bull, atr_v,  levels,    i,    "origin_1h")
            lf_orig4h = get_entry_level_feat(origin, is_bull, atr_4h, levels_4h, i_4h, "origin_4h")
            lf_orig1h = {k: v for k, v in lf_orig1h.items() if k != "origin_1h_at_level"}
            lf_orig4h = {k: v for k, v in lf_orig4h.items() if k != "origin_4h_at_level"}

            htf_val = int(htf_trend.iloc[min(i, len(htf_trend)-1)])
            btc_val = int(btc_trend.iloc[min(np.searchsorted(btc_trend.index, t, side="right")-1,
                                              len(btc_trend)-1)])
            htf_str = {1:"Bullish", -1:"Bearish", 0:"Range"}.get(htf_val, "Range")
            btc_str = {1:"Bullish", -1:"Bearish", 0:"Range"}.get(btc_val, "Range")

            feat = {
                "direction":      "Long" if is_bull else "Short",
                "session":        session_of(t.hour),
                "htf_trend":      htf_str,
                "btc_trend":      btc_str,
                "day":            t.day_name()[:3],
                "bars_to_activation": 1,
                "atr_pct_at_act": float(df["atr_pct"].iloc[i]),
                "atr_pct_at_eng": float(df["atr_pct"].iloc[i]),
                "rng_pct":        rng / float(df["close"].iloc[i-1]) * 100,
                "stop_pct":       abs(entry - stop) / entry * 100,
                "tp1618_pct":     abs(f1618 - entry) / entry * 100,
                "rng_to_atr":     rng / atr_v,
                "vol_ratio":      vr if np.isfinite(vr) else 1.0,
                "hour":           t.hour,
                "month":          t.month,
                "htf_aligned":    int((is_bull and htf_val == 1) or (not is_bull and htf_val == -1)),
                "body_pct":       body_pct,
                "upper_wick_pct": upper_wick_pct,
                "lower_wick_pct": lower_wick_pct,
                "engulf_ratio":   engulf_ratio,
                **lf1h, **lf4h, **lf_orig1h, **lf_orig4h,
            }

            row_df = pd.DataFrame([{
                "engulf_time": str(t), "direction": feat["direction"],
                "entry": entry, "origin": origin, "stop": stop,
                "f1618": f1618, "f2618": f2618, "atr": atr_v,
                **feat
            }])
            try:
                row_df = add_confluence(row_df, df)
                for col in ["support_present","support_dist_pct","support_dist_atr",
                            "support_is_origin","support_confirmed","support_dir_matches",
                            "resistance_present","resistance_dist_pct","resistance_dist_atr",
                            "resistance_is_origin","resistance_confirmed","confluence_score"]:
                    feat[col] = float(row_df[col].iloc[0]) if col in row_df.columns else 0.0
            except Exception:
                for col in ["support_present","support_dist_pct","support_dist_atr",
                            "support_is_origin","support_confirmed","support_dir_matches",
                            "resistance_present","resistance_dist_pct","resistance_dist_atr",
                            "resistance_is_origin","resistance_confirmed","confluence_score"]:
                    feat[col] = 0.0

            X = pd.DataFrame([feat])[FEATURES]
            for c in CAT_FEATURES:
                X[c] = X[c].astype(str)
            p_win = float(clf.predict_proba(X)[0, 1])
            if p_win < P_THRESHOLD:
                continue

            # ── Activation check ──────────────────────────────────────────
            EXPIRY  = 60
            act_idx = None
            pending = False
            for j in range(i + 1, min(i + 1 + EXPIRY, len(df))):
                if (is_bull and L[j] <= entry) or (not is_bull and H[j] >= entry):
                    act_idx = j
                    break
            else:
                if i + 1 + EXPIRY > len(df):
                    pending = True

            if act_idx is None and not pending:
                continue  # expired without activation

            # ── Outcome resolution ────────────────────────────────────────
            if act_idx is not None:
                outcome, _ = resolve_outcome_1h(H, L, act_idx, entry, stop, f1618, f2618, is_bull)
                if outcome is None:
                    outcome = "open"
            else:
                outcome = "pending"

            results.append({
                "engulf_bar": t,
                "direction":  "Long" if is_bull else "Short",
                "entry":      round(entry, 6),
                "stop":       round(stop, 6),
                "f1618":      round(f1618, 6),
                "f2618":      round(f2618, 6),
                "p_win":      round(p_win, 3),
                "outcome":    outcome,
                "act_bar":    str(ts_idx[act_idx]) if act_idx is not None else "—",
            })
    return results


def print_symbol_summary(symbol, sigs):
    closed = [s for s in sigs if s["outcome"] in ("win", "be_stop", "full_stop")]
    open_  = [s for s in sigs if s["outcome"] in ("open", "pending")]

    n   = len(closed)
    w   = sum(1 for s in closed if s["outcome"] == "win")
    be  = sum(1 for s in closed if s["outcome"] == "be_stop")
    fs  = sum(1 for s in closed if s["outcome"] == "full_stop")
    r_map = {"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R}
    net_r = sum(r_map[s["outcome"]] for s in closed)

    print(f"\n{'─'*72}")
    print(f"  {symbol}")
    print(f"{'─'*72}")
    print(f"  {'Engulf Bar':<22} {'Dir':<6} {'Entry':<10} {'P(win)':<7} Outcome")
    print(f"  {'─'*60}")
    for s in sigs:
        icon = {"win":"✅","be_stop":"🔒","full_stop":"❌","open":"⏳","pending":"⏳"}.get(s["outcome"],"?")
        print(f"  {str(s['engulf_bar']):<22} {s['direction']:<6} {s['entry']:<10} "
              f"{s['p_win']:<7} {icon} {s['outcome']}")

    if n > 0:
        pos = sum(r_map[s["outcome"]] for s in closed if r_map[s["outcome"]] > 0)
        neg = abs(sum(r_map[s["outcome"]] for s in closed if r_map[s["outcome"]] < 0))
        pf  = pos / neg if neg > 0 else 999
        print(f"\n  Closed: {n}  |  Win: {w} ({w/n*100:.0f}%)  BE: {be} ({be/n*100:.0f}%)  "
              f"Full stop: {fs} ({fs/n*100:.0f}%)")
        print(f"  Net R: {net_r:+.2f}R  |  PF: {pf:.2f}  |  Open/pending: {len(open_)}")
    else:
        print(f"\n  No closed trades yet  |  Open/pending: {len(open_)}")

    return {"n": n, "w": w, "be": be, "fs": fs, "net_r": net_r, "open": len(open_)}


if __name__ == "__main__":
    print("Connecting to Bitget …")
    exchange = ccxt.bitget({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    btc4h = fetch_ohlcv(exchange, "BTC/USDT", "4h", 500)
    btc4h = prep_df(btc4h)

    cutoff = pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=DAYS)
    print(f"Period: {cutoff.date()} → today  ({DAYS} days)\n")

    totals = []
    for symbol, model_path in SYMBOLS.items():
        clf = CatBoostClassifier(verbose=0)
        clf.load_model(model_path)
        df1h = fetch_ohlcv(exchange, symbol, "1h", LOOKBACK)
        df4h = fetch_ohlcv(exchange, symbol, "4h", 500)
        sigs = detect_signals(df1h, df4h, btc4h, clf, symbol, cutoff)
        stats = print_symbol_summary(symbol, sigs)
        stats["symbol"] = symbol
        totals.append(stats)

    # ── Portfolio summary ─────────────────────────────────────────────────────
    print(f"\n\n{'█'*72}")
    print(f"  PORTFOLIO SUMMARY  —  last {DAYS} days  —  all {len(SYMBOLS)} symbols")
    print(f"{'█'*72}")
    print(f"  {'Symbol':<12} {'N':>4} {'Win%':>6} {'BE%':>6} {'Stop%':>7} {'Net R':>7} {'PF':>6}")
    print(f"  {'─'*55}")

    tot_n = tot_w = tot_be = tot_fs = 0
    tot_r = 0.0
    r_map = {"win": WIN_R, "be_stop": BE_R, "full_stop": STOP_R}

    for s in totals:
        n = s["n"]
        if n == 0:
            print(f"  {s['symbol']:<12} {'—':>4}")
            continue
        w, be, fs = s["w"], s["be"], s["fs"]
        nr = s["net_r"]
        pos = w * WIN_R
        neg = abs(be * BE_R + fs * STOP_R)
        pf  = pos / neg if neg > 0 else 999
        print(f"  {s['symbol']:<12} {n:>4} {w/n*100:>5.0f}% {be/n*100:>5.0f}% "
              f"{fs/n*100:>6.0f}% {nr:>+6.2f}R {pf:>6.2f}")
        tot_n += n; tot_w += w; tot_be += be; tot_fs += fs; tot_r += nr

    print(f"  {'─'*55}")
    if tot_n > 0:
        pos_tot = tot_w * WIN_R
        neg_tot = abs(tot_be * BE_R + tot_fs * STOP_R)
        pf_tot  = pos_tot / neg_tot if neg_tot > 0 else 999
        print(f"  {'TOTAL':<12} {tot_n:>4} {tot_w/tot_n*100:>5.0f}% {tot_be/tot_n*100:>5.0f}% "
              f"{tot_fs/tot_n*100:>6.0f}% {tot_r:>+6.2f}R {pf_tot:>6.2f}")

    print(f"{'█'*72}")
    print("\nDone.")
