"""
Confluence backtest:
  For each engulfing trade in engulfing_{1h,4h}.csv, look up the Highlander
  level-state at the activation bar and add confluence features:

    - distance to nearest SUPPORT-side level (in trade direction)
    - distance to nearest RESISTANCE-side level (against trade)
    - direction & state of the nearest matching level
    - whether engulfing range overlaps an Origin level
    - is the matching level a confirmed Origin (event-time, no leakage)?

Output an enriched dataset and re-run analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from highlander import run_indicator, DIR_UP, DIR_DOWN

OUT_DIR = Path("data")
COST_R = 0.04


def reconstruct_states(events_df, n_bars):
    """Group events by level_id efficiently using to_dict instead of iterrows."""
    records = events_df.to_dict("records")
    grouped = {}
    for r in records:
        lid = int(r["level_id"])
        grouped.setdefault(lid, []).append({
            "bar": int(r["bar"]),
            "event": r["event"],
            "confirmed": bool(r["confirmed"]),
            "direction": int(r["direction"]),
            "price": float(r["price"]),
        })
    # also produce metadata (created_bar, deleted_bar) for fast filtering
    meta = {}
    for lid, hist in grouped.items():
        created = next((h["bar"] for h in hist if h["event"]=="level_created"), None)
        deleted = next((h["bar"] for h in hist if h["event"]=="level_deleted"), None)
        meta[lid] = {"created_bar": created, "deleted_bar": deleted}
    return grouped, meta


def state_at(level_history, target_bar):
    """Given a level's event history (sorted by bar), return its state at target_bar.
    Returns dict {state, confirmed, alive} or None if level didn't exist yet."""
    if not level_history or level_history[0]["bar"] > target_bar:
        return None  # level not yet created
    state = -1
    confirmed = False
    alive = True
    for ev in level_history:
        if ev["bar"] > target_bar:
            break
        if ev["event"] == "level_created":
            state = 0  # BREAK
        elif ev["event"] == "first_touch":
            state = 3  # BREAK_TOUCHED
        elif ev["event"] == "origin_confirmed":
            state = 1  # ORIGIN
        elif ev["event"] == "level_broken":
            state = 2  # BROKEN_BSUT
        elif ev["event"] == "level_deleted":
            alive = False
        # confirmed flag is set when 2 same-color candles fire — not a separate event,
        # but `confirmed` field on each event is current value at that time
        confirmed = ev["confirmed"]
    return {"state": state, "confirmed": confirmed, "alive": alive,
            "direction": level_history[0]["direction"],
            "price": level_history[0]["price"]}


def find_confluence(engulf_bar, engulf_price, is_bull, level_histories, level_meta,
                     atr_at_bar):
    """Find the nearest active level above and below current price at engulf_bar,
    and classify confluence."""
    above_levels = []
    below_levels = []

    for lid, m in level_meta.items():
        # Fast filtering using metadata
        if m["created_bar"] is None or m["created_bar"] > engulf_bar: continue
        if m["deleted_bar"] is not None and m["deleted_bar"] <= engulf_bar: continue
        s = state_at(level_histories[lid], engulf_bar)
        if s is None or not s["alive"]: continue
        if s["state"] == 2: continue   # BROKEN — ignore
        # active level
        diff = s["price"] - engulf_price
        info = {
            "id": lid, "price": s["price"],
            "direction": s["direction"], "state": s["state"],
            "confirmed": s["confirmed"],
            "is_origin": s["state"] == 1,
            "is_origin_or_orange": s["state"] in (1, 3),
            "abs_dist_pct": abs(diff) / engulf_price,
            "abs_dist_atr": abs(diff) / atr_at_bar if atr_at_bar > 0 else np.inf,
        }
        if diff > 0:  above_levels.append(info)
        elif diff < 0: below_levels.append(info)

    above_levels.sort(key=lambda x: x["abs_dist_pct"])
    below_levels.sort(key=lambda x: x["abs_dist_pct"])

    # For BULL engulfing: support = below, resistance = above
    if is_bull:
        support = below_levels[0] if below_levels else None
        resistance = above_levels[0] if above_levels else None
    else:
        support = above_levels[0] if above_levels else None    # support for short = level above
        resistance = below_levels[0] if below_levels else None

    return support, resistance


def annotate_dataset(engulf_csv, ohlc_csv, tf_label):
    print(f"\n{'═'*60}\nConfluence: {engulf_csv}\n{'═'*60}")
    eng = pd.read_csv(engulf_csv, parse_dates=["engulf_time", "activation_time"])
    df = pd.read_csv(ohlc_csv); df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df = df[["open","high","low","close"]].astype(float)

    # ATR for distance calculations
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    df["atr"] = pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1).rolling(14).mean()

    # Run highlander (or use cache)
    cache_path = OUT_DIR / f"highlander_{tf_label}.pkl"
    if cache_path.exists():
        print(f"  loading cached state {cache_path}")
        with open(cache_path, "rb") as f: cached = pickle.load(f)
        ev_df = pd.DataFrame(cached["events"])
        print(f"  {len(cached['levels']):,} levels, {len(ev_df):,} events")
    else:
        print("Running Highlander on this timeframe...")
        tick = max(df["close"].iloc[0]*1e-5, 1e-6)
        levels, events = run_indicator(df, 3.0, tick)
        ev_df = pd.DataFrame([e.__dict__ for e in events])
        print(f"  {len(levels):,} levels, {len(ev_df):,} events")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "levels": [{"id": l.id, "price": l.price, "dir": l.dir,
                            "created_bar": l.created_bar, "origin_bar": l.origin_bar,
                            "deleted_bar": l.deleted_bar, "confirmed": l.confirmed} for l in levels],
                "events": [e.__dict__ for e in events]
            }, f)
        print(f"  cached → {cache_path}")

    histories, level_meta = reconstruct_states(ev_df, len(df))

    # For each engulfing, find the activation bar (where they entered)
    # We have activation_time in the dataset; convert to bar index
    bar_of = {ts: i for i, ts in enumerate(df.index)}
    eng["act_bar"] = eng["activation_time"].map(bar_of)
    eng = eng.dropna(subset=["act_bar"])
    eng["act_bar"] = eng["act_bar"].astype(int)

    # Annotate each
    rows = []
    for _, e in eng.iterrows():
        bar = int(e["act_bar"])
        is_bull = bool(e["is_bull"])
        price = df["close"].iloc[bar]
        atr = df["atr"].iloc[bar] if np.isfinite(df["atr"].iloc[bar]) else 0
        support, resistance = find_confluence(bar, price, is_bull, histories, level_meta, atr)

        rec = {
            "support_present":      int(support is not None),
            "support_dist_pct":     round(support["abs_dist_pct"]*100, 3) if support else np.nan,
            "support_dist_atr":     round(support["abs_dist_atr"], 2) if support else np.nan,
            "support_is_origin":    int(support["is_origin"]) if support else 0,
            "support_confirmed":    int(support["confirmed"]) if support else 0,
            "support_dir_matches":  int(bool(support) and ((is_bull and support["direction"]==DIR_UP) or
                                                       (not is_bull and support["direction"]==DIR_DOWN))),
            "resistance_present":    int(resistance is not None),
            "resistance_dist_pct":   round(resistance["abs_dist_pct"]*100, 3) if resistance else np.nan,
            "resistance_dist_atr":   round(resistance["abs_dist_atr"], 2) if resistance else np.nan,
            "resistance_is_origin":  int(resistance["is_origin"]) if resistance else 0,
            "resistance_confirmed":  int(resistance["confirmed"]) if resistance else 0,
        }
        # Confluence "score" (higher = better setup)
        rec["confluence_score"] = (
            (1 if rec["support_dist_atr"] is not None and not np.isnan(rec["support_dist_atr"]) and rec["support_dist_atr"] < 1 else 0) +
            (1 if rec["support_is_origin"] else 0) +
            (1 if rec["support_confirmed"] else 0) +
            (1 if rec["support_dir_matches"] else 0) -
            (1 if rec["resistance_dist_atr"] is not None and not np.isnan(rec["resistance_dist_atr"]) and rec["resistance_dist_atr"] < 1.5 and rec["resistance_is_origin"] else 0)
        )
        rows.append(rec)

    enriched = pd.concat([eng.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out = OUT_DIR / f"engulfing_{tf_label}_with_confluence.csv"
    enriched.to_csv(out, index=False)
    print(f"  saved → {out}  ({len(enriched):,} rows)")

    # ─── Analysis ───
    R = "r_strategy_D_trail"
    print(f"\n— Outcome by support-side confluence (using {R}) —")
    print(enriched.groupby("support_dir_matches")[R].agg(["count","mean"]).round(3).to_string())
    print(enriched.groupby("support_is_origin")[R].agg(["count","mean"]).round(3).to_string())
    print(enriched.groupby("support_confirmed")[R].agg(["count","mean"]).round(3).to_string())

    print(f"\n— Outcome by confluence_score —")
    print(enriched.groupby("confluence_score")[R].agg(["count","mean"]).round(3).to_string())

    # Bin support distance
    enriched["support_dist_bin"] = pd.cut(enriched["support_dist_atr"],
        bins=[-0.01, 0.5, 1.0, 2.0, 5.0, 1e9],
        labels=["VeryClose","Close","Mid","Far","VeryFar"])
    print(f"\n— Outcome by support distance (in ATR units) —")
    print(enriched.groupby("support_dist_bin")[R].agg(["count","mean"]).round(3).to_string())

    print(f"\n— Top combinations: support_dir_matches × support_is_origin × support_dist_bin —")
    g = enriched.groupby(["support_dir_matches","support_is_origin","support_dist_bin"])[R].agg(["count","mean"]).round(3)
    g = g[g["count"] >= 30].sort_values("mean", ascending=False)
    print(g.head(20).to_string())

    # Best filter: combine HTF aligned + support confluence
    print(f"\n— HTF aligned × support confluence direction match —")
    g2 = enriched.groupby(["htf_aligned","support_dir_matches"])[R].agg(["count","mean"]).round(3)
    print(g2.to_string())

    # Final filter: best of the best
    elite = enriched[
        (enriched["support_dir_matches"] == 1) &
        (enriched["support_is_origin"] == 1) &
        (enriched["support_dist_atr"] <= 2.0) &
        (enriched["htf_aligned"] == 1)
    ]
    print(f"\n— ELITE filter: support is matching origin, within 2 ATR, HTF aligned —")
    print(f"  trades: {len(elite):,}")
    if len(elite):
        for col in ["r_strategy_A_TP1618","r_strategy_B_TP2618","r_strategy_C_split","r_strategy_D_trail"]:
            wr = (elite[col] > 0).mean(); avg = elite[col].mean(); net = elite[col].sum()
            print(f"  {col:30s}  win {wr:5.1%}  avgR {avg:+.3f}  netR {net:+6.1f}")

    return enriched


import sys
which = sys.argv[1] if len(sys.argv) > 1 else "both"
if which in ("1h", "both"):
    annotate_dataset("data/engulfing_1h.csv", "data/sol_1h.csv", "1h")
if which in ("4h", "both"):
    annotate_dataset("data/engulfing_4h.csv", "data/sol_4h.csv", "4h")
print("\nDone.")
