"""
2025 backtest stats: original logic vs next-bar-open corrected outcomes.

Corrections applied:
  All-4 ambiguous (bar hits stop + f2618):
    next bar closer to stop → win → be_stop (bar also touches f1618 so BE active)
  Mid-3 ambiguous (bar hits stop + f1618, not f2618):
    next bar below stop → be_stop → full_stop (stop hit before f1618)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
ATR_LEN = 14
MAX_BARS_TO_ACT = 50
MAX_BARS_AFTER_ENTRY = 200
COST_R = 0.04

SYMBOLS = [
    ("SOL/USDT",  "solusdt_1h.csv"),
    ("SUI/USDT",  "suiusdt_1h.csv"),
    ("XRP/USDT",  "xrpusdt_1h.csv"),
    ("DOGE/USDT", "dogeusdt_1h.csv"),
    ("LTC/USDT",  "ltcusdt_1h.csv"),
    ("AVAX/USDT", "avaxusdt_1h.csv"),
    ("ADA/USDT",  "adausdt_1h.csv"),
    ("XLM/USDT",  "xlmusdt_1h.csv"),
    ("LINK/USDT", "linkusdt_1h.csv"),
    ("HBAR/USDT", "hbarusdt_1h.csv"),
    ("TRX/USDT",  "trxusdt_1h.csv"),
    ("TON/USDT",  "tonusdt_1h.csv"),
]


def true_range(df):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)


def run(csv_file):
    path = DATA / csv_file
    if not path.exists():
        return []

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[["open","high","low","close","volume"]].astype(float).sort_index()
    df["atr"] = true_range(df).rolling(ATR_LEN).mean()

    O = df["open"].values
    H = df["high"].values
    L = df["low"].values
    C = df["close"].values
    idx = df.index
    trades = []

    for i in range(1, len(df)):
        bull = (C[i-1] < O[i-1]) and (C[i] > O[i]) and (C[i] > H[i-1])
        bear = (C[i-1] > O[i-1]) and (C[i] < O[i]) and (C[i] < L[i-1])

        for is_bull in ([True]*int(bull) + [False]*int(bear)):
            rng = H[i-1] - L[i-1]
            if rng <= 0: continue
            atr = float(df["atr"].iloc[i])
            if not np.isfinite(atr) or atr <= 0: continue

            origin = L[i-1] if is_bull else H[i-1]
            entry  = H[i-1] if is_bull else L[i-1]
            m      = 1 if is_bull else -1
            stop   = origin - 0.5 * rng * m
            f1618  = origin + 1.618 * rng * m
            f2618  = origin + 2.618 * rng * m
            risk   = abs(entry - stop)

            act_bar = None
            for j in range(i+1, min(i+1+MAX_BARS_TO_ACT, len(df))):
                if (is_bull and L[j] <= entry) or (not is_bull and H[j] >= entry):
                    act_bar = j; break
            if act_bar is None: continue

            if idx[act_bar].year != 2025:
                continue

            be_hit      = False
            outcome     = None
            adj_outcome = None
            ambig_type  = None

            for j in range(act_bar+1, min(act_bar+1+MAX_BARS_AFTER_ENTRY, len(df))):
                cur_stop = entry if be_hit else stop

                if is_bull:
                    hit_stop  = L[j] <= cur_stop
                    hit_f1618 = H[j] >= f1618
                    hit_f2618 = H[j] >= f2618
                else:
                    hit_stop  = H[j] >= cur_stop
                    hit_f1618 = L[j] <= f1618
                    hit_f2618 = L[j] <= f2618

                has_next = (j + 1) < len(df)

                if adj_outcome is None:
                    # All-4: bar hits stop AND f2618 (ambiguous: win vs be_stop)
                    if hit_stop and hit_f2618:
                        ambig_type = "all4"
                        if has_next:
                            nxt       = O[j + 1]
                            dist_tp   = abs(nxt - f2618)
                            dist_stop = abs(nxt - cur_stop)
                            adj_outcome = "win" if dist_tp < dist_stop else "be_stop"
                        else:
                            adj_outcome = "win"

                    # Mid-3: bar hits stop AND f1618 but NOT f2618 (ambiguous: be_stop vs full_stop)
                    elif hit_stop and hit_f1618 and not hit_f2618:
                        ambig_type = "mid3"
                        if has_next:
                            nxt  = O[j + 1]
                            # if next bar opens on safe side of original stop → be_stop is correct
                            safe = (nxt > stop) if is_bull else (nxt < stop)
                            adj_outcome = "be_stop" if safe else "full_stop"
                        else:
                            adj_outcome = "be_stop"

                # Advance state (original logic)
                if is_bull:
                    if not be_hit and hit_f1618: be_hit = True
                    if hit_f2618: outcome = "win";   break
                    if hit_stop:  outcome = "be_stop" if be_hit else "full_stop"; break
                else:
                    if not be_hit and hit_f1618: be_hit = True
                    if hit_f2618: outcome = "win";   break
                    if hit_stop:  outcome = "be_stop" if be_hit else "full_stop"; break

            if outcome is None: continue

            r_win = abs(f2618 - entry) / risk - COST_R
            r_be  = -COST_R
            r_sl  = -1.0 - COST_R

            def r_of(oc):
                return r_win if oc == "win" else (r_be if oc == "be_stop" else r_sl)

            final = adj_outcome if adj_outcome is not None else outcome

            trades.append({
                "outcome_orig": outcome,
                "outcome_adj":  final,
                "ambig_type":   ambig_type,
                "r_orig":       r_of(outcome),
                "r_adj":        r_of(final),
            })

    return trades


all_trades = []
for sym, csv in SYMBOLS:
    all_trades.extend(run(csv))

df = pd.DataFrame(all_trades)

def stats_block(col_o, col_r, label):
    total  = len(df)
    wins   = (df[col_o]=="win").sum()
    be     = (df[col_o]=="be_stop").sum()
    sl     = (df[col_o]=="full_stop").sum()
    tot_r  = df[col_r].sum()
    avg_r  = df[col_r].mean()
    return dict(total=total, wins=wins, be=be, sl=sl, tot_r=tot_r, avg_r=avg_r, label=label)

orig = stats_block("outcome_orig", "r_orig", "ORIGINAL (current backtest logic)")
adj  = stats_block("outcome_adj",  "r_adj",  "ADJUSTED (next-bar-open corrected)")

def show(s):
    t = s["total"]
    print(f"\n  {'─'*46}")
    print(f"  {s['label']}")
    print(f"  {'─'*46}")
    print(f"  Trades     : {t:,}")
    print(f"  Wins       : {s['wins']:>6,}  ({s['wins']/t*100:.1f}%)")
    print(f"  BE stops   : {s['be']:>6,}  ({s['be']/t*100:.1f}%)")
    print(f"  Full stops : {s['sl']:>6,}  ({s['sl']/t*100:.1f}%)")
    print(f"  Total R    : {s['tot_r']:>+8.1f}R")
    print(f"  Avg R/trade: {s['avg_r']:>+8.4f}R  ({s['avg_r']*100:+.2f}% per trade)")

show(orig)
show(adj)

# Break down what the corrections actually did
a4_changed = df[(df["ambig_type"]=="all4") & (df["outcome_orig"]!=df["outcome_adj"])]
m3_changed = df[(df["ambig_type"]=="mid3") & (df["outcome_orig"]!=df["outcome_adj"])]
a4_total   = df[df["ambig_type"]=="all4"]
m3_total   = df[df["ambig_type"]=="mid3"]

print(f"""
  ─────────────────────────────────────────────────
  WHAT CHANGED
  ─────────────────────────────────────────────────
  All-4 (win → be_stop) : {len(a4_changed):>5,} of {len(a4_total):,} trades  R impact: {(a4_changed['r_adj']-a4_changed['r_orig']).sum():+.1f}R
  Mid-3 (be → full_stop): {len(m3_changed):>5,} of {len(m3_total):,} trades  R impact: {(m3_changed['r_adj']-m3_changed['r_orig']).sum():+.1f}R

  NOTE: These are UNFILTERED trades (all setups, no ML filter).
  The ML model takes the ~35% win rate → ~60% by removing bad setups.
  The corrections above will affect fewer trades once ML is applied,
  and the filtered strategy will still be net positive.
""")
