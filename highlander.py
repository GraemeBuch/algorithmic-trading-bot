"""
Python port of the Pine Script:
  Highlander Break Levels / Origin Levels v6.39

Implements the same state machine bar-by-bar:
  STATE_BREAK         (yellow)   – fresh level, untested
  STATE_BREAK_TOUCHED (orange)   – touched once after a gap, awaits second touch
  STATE_ORIGIN        (green/red) – touched twice with a gap, confirmed
  STATE_BROKEN_BSUT   (gray)     – closed through cleanly without touch
  DELETE                          – broken-then-retested → removed

Definitions kept consistent with the Pine code:
  Gap1: ≥1 no-touch bar between creation and first touch
  Gap2: ≥1 no-touch bar between first touch (orange) and rejection (origin promote)

Also produces an "events" table containing every relevant interaction so we
can build strategy backtests on top.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


# ─────────────────────────── State / Direction constants ─────────────────────────
STATE_BREAK = 0
STATE_ORIGIN = 1
STATE_BROKEN_BSUT = 2
STATE_BREAK_TOUCHED = 3

DIR_UP = 1
DIR_DOWN = -1

STATE_NAMES = {
    STATE_BREAK: "BREAK",
    STATE_ORIGIN: "ORIGIN",
    STATE_BROKEN_BSUT: "BROKEN",
    STATE_BREAK_TOUCHED: "BREAK_TOUCHED",
}


@dataclass
class Level:
    price: float
    dir: int                  # DIR_UP or DIR_DOWN
    state: int = STATE_BREAK
    confirmed: bool = False
    first_touch_bar: Optional[int] = None
    touch_seen: bool = False
    gap1_seen: bool = False
    gap2_seen: bool = False
    created_bar: int = 0
    origin_bar: Optional[int] = None  # bar at which it became ORIGIN
    # bar at which level was destroyed (broken+retested), or None if still alive
    deleted_bar: Optional[int] = None
    touches: list = field(default_factory=list)  # list of bar indices where touch happened


@dataclass
class Event:
    """A relevant interaction that downstream strategies will care about."""
    bar: int
    timestamp: pd.Timestamp
    event: str        # 'level_created' / 'first_touch' / 'origin_confirmed' / 'origin_touch' / 'break_body' / 'break_wick' / 'level_broken' / 'level_deleted'
    direction: int    # DIR_UP / DIR_DOWN
    price: float
    state_before: int
    state_after: int
    confirmed: bool
    touch_count: int  # number of touches on this level so far
    level_id: int     # stable identifier for the level


# ─────────────────────────── Engine ─────────────────────────
def run_indicator(df: pd.DataFrame,
                  min_range_ticks: float = 3.0,
                  tick_size: float = 0.0001) -> tuple[list[Level], list[Event]]:
    """
    Run the Highlander state machine over a DataFrame with columns
    ['open','high','low','close']. Index expected to be datetime.

    Returns (levels, events).
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    ts = df.index.to_numpy()

    n_bars = len(df)
    min_range = tick_size * min_range_ticks

    levels: list[Level] = []
    events: list[Event] = []

    next_id = 0

    for i in range(n_bars):
        # We need at least 3 bars of history (uses [1] and [2] offsets)
        if i < 2:
            continue

        is_red = c[i] < o[i]
        is_green = c[i] > o[i]
        was_red_prev = c[i-1] < o[i-1]
        was_green_prev = c[i-1] > o[i-1]
        was_red_2 = c[i-2] < o[i-2]
        was_green_2 = c[i-2] > o[i-2]

        # ── 1. Detect new candidate levels ────────────────────────
        # DOWN: first 2 reds (current and previous red, but NOT pre-previous)
        if is_red and was_red_prev and not was_red_2:
            lvl_down = h[i-1]            # high of the FIRST candle
            range1 = h[i-1] - l[i-1]
            if range1 >= min_range:
                lvl = Level(price=lvl_down, dir=DIR_DOWN,
                            state=STATE_BREAK, created_bar=i)
                lvl.id = next_id; next_id += 1
                levels.append(lvl)
                events.append(Event(
                    bar=i, timestamp=ts[i], event="level_created",
                    direction=DIR_DOWN, price=lvl_down,
                    state_before=-1, state_after=STATE_BREAK,
                    confirmed=False, touch_count=0, level_id=lvl.id,
                ))

        # UP: first 2 greens
        if is_green and was_green_prev and not was_green_2:
            lvl_up = l[i-1]              # low of the FIRST candle
            range1 = h[i-1] - l[i-1]
            if range1 >= min_range:
                lvl = Level(price=lvl_up, dir=DIR_UP,
                            state=STATE_BREAK, created_bar=i)
                lvl.id = next_id; next_id += 1
                levels.append(lvl)
                events.append(Event(
                    bar=i, timestamp=ts[i], event="level_created",
                    direction=DIR_UP, price=lvl_up,
                    state_before=-1, state_after=STATE_BREAK,
                    confirmed=False, touch_count=0, level_id=lvl.id,
                ))

        # ── 2. Per-bar update over all live levels ────────────────
        # iterate over a copy so we can flag deletes
        for lvl in list(levels):
            if lvl.deleted_bar is not None:
                continue

            price = lvl.price
            d = lvl.dir
            prev_state = lvl.state

            cl = c[i]
            hi = h[i]
            lo = l[i]

            touch = (hi >= price) and (lo <= price)
            if d == DIR_DOWN:
                reject = touch and (cl < price)
                beyond = cl > price
                bsut_retest = (lo <= price) if lvl.state == STATE_BROKEN_BSUT else False
            else:
                reject = touch and (cl > price)
                beyond = cl < price
                bsut_retest = (hi >= price) if lvl.state == STATE_BROKEN_BSUT else False

            # GAP1: between creation and first touch
            if lvl.state == STATE_BREAK and not lvl.touch_seen and not touch:
                lvl.gap1_seen = True

            # First valid touch (only once): requires GAP1
            first_valid_touch = (lvl.state == STATE_BREAK
                                 and not lvl.touch_seen
                                 and lvl.gap1_seen and touch)
            if first_valid_touch:
                lvl.touch_seen = True
                lvl.first_touch_bar = i
                lvl.state = STATE_BREAK_TOUCHED
                lvl.touches.append(i)
                events.append(Event(
                    bar=i, timestamp=ts[i], event="first_touch",
                    direction=d, price=price,
                    state_before=prev_state, state_after=lvl.state,
                    confirmed=lvl.confirmed, touch_count=len(lvl.touches),
                    level_id=lvl.id,
                ))

            # GAP2: after first touch (orange) before origin
            if lvl.state == STATE_BREAK_TOUCHED and lvl.touch_seen and not touch:
                lvl.gap2_seen = True

            # Rejection that promotes BREAK_TOUCHED -> ORIGIN
            can_reject = (lvl.state == STATE_BREAK_TOUCHED
                          and lvl.touch_seen and lvl.gap2_seen and reject)
            if can_reject:
                lvl.state = STATE_ORIGIN
                lvl.confirmed = False
                lvl.origin_bar = i
                lvl.touches.append(i)
                events.append(Event(
                    bar=i, timestamp=ts[i], event="origin_confirmed",
                    direction=d, price=price,
                    state_before=prev_state, state_after=lvl.state,
                    confirmed=False, touch_count=len(lvl.touches),
                    level_id=lvl.id,
                ))

            # Confirm ORIGIN after 2 same-colour candles
            if lvl.state == STATE_ORIGIN and not lvl.confirmed:
                if d == DIR_DOWN:
                    cond = (cl < o[i]) and (c[i-1] < o[i-1])
                else:
                    cond = (cl > o[i]) and (c[i-1] > o[i-1])
                if cond:
                    lvl.confirmed = True

            # ── Origin retests (downstream "Origin First Touch" / "Origin Touch")
            if lvl.state == STATE_ORIGIN and i > (lvl.origin_bar or 0):
                if touch:
                    # how many origin-era touches so far
                    origin_touches_so_far = sum(1 for tb in lvl.touches if tb >= (lvl.origin_bar or 0))
                    if origin_touches_so_far == 0 or lvl.touches[-1] != i:
                        lvl.touches.append(i)
                    origin_touches_after = sum(1 for tb in lvl.touches if tb > (lvl.origin_bar or 0))
                    event_name = "origin_first_touch" if origin_touches_after == 1 else "origin_touch"
                    events.append(Event(
                        bar=i, timestamp=ts[i], event=event_name,
                        direction=d, price=price,
                        state_before=prev_state, state_after=lvl.state,
                        confirmed=lvl.confirmed, touch_count=origin_touches_after,
                        level_id=lvl.id,
                    ))

            # ── Break-level interactions while still BREAK / orange:
            # break_body = close beyond level (with touch this bar)
            # break_wick = wick beyond but body holds inside
            if lvl.state in (STATE_BREAK, STATE_BREAK_TOUCHED):
                if d == DIR_DOWN:
                    body_through = touch and cl > price
                    wick_through = (hi > price) and (cl <= price)
                else:
                    body_through = touch and cl < price
                    wick_through = (lo < price) and (cl >= price)
                if body_through:
                    events.append(Event(
                        bar=i, timestamp=ts[i], event="break_body",
                        direction=d, price=price,
                        state_before=prev_state, state_after=lvl.state,
                        confirmed=lvl.confirmed, touch_count=len(lvl.touches),
                        level_id=lvl.id,
                    ))
                elif wick_through and touch:
                    events.append(Event(
                        bar=i, timestamp=ts[i], event="break_wick",
                        direction=d, price=price,
                        state_before=prev_state, state_after=lvl.state,
                        confirmed=lvl.confirmed, touch_count=len(lvl.touches),
                        level_id=lvl.id,
                    ))

            # NO-TOUCH HARD CLOSE → BSUT
            if d == DIR_DOWN:
                no_touch_close_beyond = beyond and lo > price
            else:
                no_touch_close_beyond = beyond and hi < price
            if (lvl.state in (STATE_BREAK, STATE_BREAK_TOUCHED, STATE_ORIGIN)
                    and no_touch_close_beyond):
                lvl.state = STATE_BROKEN_BSUT
                lvl.confirmed = False
                events.append(Event(
                    bar=i, timestamp=ts[i], event="level_broken",
                    direction=d, price=price,
                    state_before=prev_state, state_after=lvl.state,
                    confirmed=False, touch_count=len(lvl.touches),
                    level_id=lvl.id,
                ))

            # BSUT retest → DELETE
            if lvl.state == STATE_BROKEN_BSUT and bsut_retest:
                lvl.deleted_bar = i
                events.append(Event(
                    bar=i, timestamp=ts[i], event="level_deleted",
                    direction=d, price=price,
                    state_before=prev_state, state_after=-1,
                    confirmed=False, touch_count=len(lvl.touches),
                    level_id=lvl.id,
                ))

    return levels, events


# ─────────────────────── Pretty CLI / sanity check ─────────────
if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/sol_1h.csv"

    df = pd.read_csv(csv_path)
    # standardize columns
    if "time" in df.columns and "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")[["open", "high", "low", "close", "volume"]]
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")[["open", "high", "low", "close"]]
    else:
        raise ValueError("Expected a 'time' column.")

    # tick size — for crypto we use a small absolute value
    tick = max(df["close"].iloc[0] * 1e-5, 1e-6)
    levels, events = run_indicator(df, min_range_ticks=3.0, tick_size=tick)

    ev_df = pd.DataFrame([e.__dict__ for e in events])
    print(f"Bars processed   : {len(df):,}")
    print(f"Levels created   : {len(levels):,}")
    print(f"Events generated : {len(events):,}")
    print()
    print("Event counts by type:")
    print(ev_df["event"].value_counts().to_string())
