#!/usr/bin/env python3
"""
live_signals.py — Multi-symbol 1H engulfing signal scanner.

Monitors SOL/USDT and SUI/USDT simultaneously.
Tracks full trade lifecycle: pending → activated → won / stopped out.

Discord commands:
    /pending   — setups waiting for retest
    /active    — trades currently open
    /results   — recent won/stopped trades
    /overview  — everything for all symbols
    /status    — alias for /overview
"""
from __future__ import annotations
import os
import time
import json
import pathlib
import requests
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from catboost import CatBoostClassifier
from highlander import (run_indicator, DIR_UP, DIR_DOWN,
                        STATE_ORIGIN, STATE_BREAK_TOUCHED, STATE_BROKEN_BSUT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
EXCHANGE_ID     = "bitget"
SYMBOL_BTC      = "BTC/USDT"
P_THRESHOLD     = 0.50
ATR_LEN         = 14
MAX_BARS_TO_ACT = 50
LOOKBACK_1H     = 990    # ~41 days of 1H bars; startup cache provides full-history levels separately
LOOKBACK_4H     = 200
MAX_CLOSED_HIST = 20
MAX_CANCEL_HIST   = 10
MAX_FILTERED_HIST = 20
JOURNAL_PATH    = Path(__file__).parent / "trade_journal.csv"

# ── Execution config ──────────────────────────────────────────────────────────
BITGET_API_KEY    = os.getenv("BITGET_API_KEY",    "")
BITGET_SECRET     = os.getenv("BITGET_SECRET",     "")
BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
AUTO_TRADE        = bool(BITGET_API_KEY and BITGET_SECRET and BITGET_PASSPHRASE)
ACCOUNT_RISK_PCT  = 0.01    # 1.0% of free balance risked per trade
LEVERAGE          = 10      # fixed leverage applied to all positions
MAX_MARGIN_PCT    = 0.20    # max margin per trade as fraction of free balance
MAX_CONCURRENT    = 5       # max simultaneously open executed trades
LIMIT_FILL_TIMEOUT = 120    # seconds to wait for limit entry order to fill

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

# Full 1H CSV paths — used at startup to pre-compute Highlander levels from full history,
# matching the data the ML model was trained on.
CSV_1H_PATHS = {
    "BTC/USDT":  "data/btcusdt_1h.csv",
    "ETH/USDT":  "data/ethusdt_1h.csv",
    "SOL/USDT":  "data/solusdt_1h.csv",
    "TRX/USDT":  "data/trxusdt_1h.csv",
    "LINK/USDT": "data/linkusdt_1h.csv",
    "HBAR/USDT": "data/hbarusdt_1h.csv",
    "XRP/USDT":  "data/xrpusdt_1h.csv",
    "DOGE/USDT": "data/dogeusdt_1h.csv",
    "LTC/USDT":  "data/ltcusdt_1h.csv",
    "ADA/USDT":  "data/adausdt_1h.csv",
    "XLM/USDT":  "data/xlmusdt_1h.csv",
    "UNI/USDT":  "data/uniusdt_1h.csv",
    "APT/USDT":  "data/aptusdt_1h.csv",
    "NEAR/USDT": "data/nearusdt_1h.csv",
}

# Startup level cache — populated once at bot launch, used in every scan_symbol call.
_startup_cache: dict = {}   # symbol -> {levels_1h, ts_1h, levels_4h, ts_4h}

CAT_FEATURES = ["direction", "session", "htf_trend", "btc_trend", "day"]
NUM_FEATURES = [
    "bars_to_activation", "atr_pct_at_act", "atr_pct_at_eng",
    "rng_pct", "stop_pct", "tp1618_pct", "rng_to_atr",
    "vol_ratio", "hour", "month", "htf_aligned",
    "support_present", "support_dist_pct", "support_dist_atr",
    "support_is_origin", "support_confirmed", "support_dir_matches",
    "resistance_present", "resistance_dist_pct", "resistance_dist_atr",
    "resistance_is_origin", "resistance_confirmed", "confluence_score",
    "body_pct", "upper_wick_pct", "lower_wick_pct", "engulf_ratio",
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


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class Pending:
    eng_time:         pd.Timestamp
    is_bull:          bool
    origin:           float
    entry:            float
    rng:              float
    stop:             float
    f1618:            float
    f2618:            float
    atr_at_eng:       float
    atr_pct_at_eng:   float
    vol_ratio_at_eng: float
    # Engulf candle quality (stored at detection time)
    body_pct:         float = 0.5
    upper_wick_pct:   float = 0.25
    lower_wick_pct:   float = 0.25
    engulf_ratio:     float = 1.0
    # Entry price vs nearest Highlander level — 1H
    entry_1h_at_level:  int   = 0
    entry_1h_dist_atr:  float = 99.0
    entry_1h_is_origin: int   = 0
    entry_1h_is_ft:     int   = 0
    entry_1h_is_broken: int   = 0
    entry_1h_dir_match: int   = 0
    # Entry price vs nearest Highlander level — 4H
    entry_4h_at_level:  int   = 0
    entry_4h_dist_atr:  float = 99.0
    entry_4h_is_origin: int   = 0
    entry_4h_is_ft:     int   = 0
    entry_4h_is_broken: int   = 0
    entry_4h_dir_match: int   = 0
    # Origin (bounce point) vs nearest Highlander level — 1H
    origin_1h_dist_atr:  float = 99.0
    origin_1h_is_origin: int   = 0
    origin_1h_is_ft:     int   = 0
    origin_1h_is_broken: int   = 0
    origin_1h_dir_match: int   = 0
    # Origin (bounce point) vs nearest Highlander level — 4H
    origin_4h_dist_atr:  float = 99.0
    origin_4h_is_origin: int   = 0
    origin_4h_is_ft:     int   = 0
    origin_4h_is_broken: int   = 0
    origin_4h_dir_match: int   = 0


@dataclass
class ActiveTrade:
    symbol:         str
    direction:      str
    entry:          float
    stop:           float
    f1618:          float
    f2618:          float
    risk:           float
    p_win:          float
    eng_time:       pd.Timestamp
    activated_time: pd.Timestamp
    session:        str
    htf:            str
    btc:            str
    be_hit:         bool          = False
    be_sl_set:      bool          = False  # True once SL has been successfully moved to entry
    be_hit_time:    Optional[pd.Timestamp] = None  # timestamp of 1m bar that triggered BE
    sl_move_attempts: int         = 0      # give up after 3 failed attempts
    live_activated: bool          = False  # activated via forming bar — skip outcome check on same bar
    activation_bar: Optional[pd.Timestamp] = None  # ts_forming at activation — skip until this bar closes
    # Execution tracking — populated when AUTO_TRADE is on
    order_id:       Optional[str] = None
    sl_order_id:    Optional[str] = None
    tp_order_id:    Optional[str] = None
    qty:            Optional[float] = None


@dataclass
class ClosedTrade:
    symbol:         str
    direction:      str
    entry:          float
    stop:           float
    f2618:          float
    risk:           float
    p_win:          float
    eng_time:       pd.Timestamp
    activated_time: pd.Timestamp
    closed_time:    pd.Timestamp
    outcome:        str    # "won" | "stopped"
    pnl_r:          float


# ── Trade journal ─────────────────────────────────────────────────────────────
_JOURNAL_COLS = [
    "date_opened", "date_closed", "symbol", "direction",
    "entry", "stop", "tp", "exit_price",
    "outcome", "pnl_usdt", "r_result", "p_win",
]

def _journal_init():
    if not JOURNAL_PATH.exists():
        import csv
        with open(JOURNAL_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_JOURNAL_COLS).writeheader()

def _journal_open(trade: "ActiveTrade"):
    import csv
    _journal_init()
    row = {
        "date_opened":  pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M"),
        "date_closed":  "",
        "symbol":       trade.symbol,
        "direction":    trade.direction,
        "entry":        round(trade.entry, 8),
        "stop":         round(trade.stop,  8),
        "tp":           round(trade.f2618, 8),
        "exit_price":   "",
        "outcome":      "open",
        "pnl_usdt":     "",
        "r_result":     "",
        "p_win":        round(trade.p_win, 3),
    }
    with open(JOURNAL_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_JOURNAL_COLS).writerow(row)

def _journal_close(trade: "ActiveTrade", outcome: str, exit_price: float, pnl_usdt: float, r_result: float):
    import csv, tempfile, shutil
    _journal_init()
    rows = []
    updated = False
    with open(JOURNAL_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (not updated and row["symbol"] == trade.symbol
                    and row["direction"] == trade.direction
                    and row["outcome"] == "open"
                    and abs(float(row["entry"]) - trade.entry) < 1e-9):
                row["date_closed"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M")
                row["exit_price"]  = round(exit_price, 8)
                row["outcome"]     = outcome
                row["pnl_usdt"]    = round(pnl_usdt, 4)
                row["r_result"]    = round(r_result, 4)
                updated = True
            rows.append(row)
    tmp = JOURNAL_PATH.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_JOURNAL_COLS)
        w.writeheader()
        w.writerows(rows)
    shutil.move(str(tmp), str(JOURNAL_PATH))


# ── Helpers ───────────────────────────────────────────────────────────────────
def session_of(h: int) -> str:
    if 0 <= h < 7:    return "Asia"
    if 7 <= h < 13:   return "London"
    if 13 <= h < 17:  return "NY-AM"
    if 17 <= h < 22:  return "NY-PM"
    return "Off-hours"


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    return pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def fetch_ohlcv(exchange, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    # Use futures symbol format for Bitget swap exchange
    fetch_sym = _fsym(symbol) if EXCHANGE_ID == "bitget" else symbol
    PAGE      = 1000   # Bitget max bars per request

    if limit <= PAGE:
        raw = exchange.fetch_ohlcv(fetch_sym, tf, limit=limit)
        pages = [raw]
    else:
        # Paginate backwards: fetch most-recent PAGE bars, then older pages
        pages = []
        since = None
        remaining = limit
        while remaining > 0:
            n   = min(remaining, PAGE)
            raw = exchange.fetch_ohlcv(fetch_sym, tf, limit=n,
                                       params={"endTime": since} if since else {})
            if not raw:
                break
            # Fetch next page ending just before the earliest bar we got
            since     = raw[0][0] - 1   # 1 ms before earliest timestamp
            remaining -= len(raw)
            pages.insert(0, raw)        # prepend so result stays chronological
            if len(raw) < n:
                break                   # exchange returned fewer than asked — no more data

    combined = [bar for page in pages for bar in page]
    # Deduplicate and sort by timestamp
    seen = set()
    unique = []
    for bar in combined:
        if bar[0] not in seen:
            seen.add(bar[0])
            unique.append(bar)
    unique.sort(key=lambda b: b[0])
    # Keep only the most-recent `limit` bars
    unique = unique[-limit:]

    df = pd.DataFrame(unique, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
    return df.set_index("time").astype(float)


def fetch_binance_bars(symbol: str, tf: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV bars from Binance public REST API (no auth needed).
    Used for signal detection so bar data matches the training corpus.
    Trade execution stays on Bitget.
    """
    bn_sym = symbol.replace("/", "").split(":")[0]  # "BTC/USDT" -> "BTCUSDT"
    bars: list = []
    end_ms: int | None = None
    remaining = limit
    while remaining > 0:
        n = min(remaining, 1000)
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={bn_sym}&interval={tf}&limit={n}"
               + (f"&endTime={end_ms}" if end_ms else ""))
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            chunk = r.json()
        except Exception as exc:
            print(f"  Binance fetch error ({bn_sym} {tf}): {exc} — retrying")
            time.sleep(5)
            continue
        if not chunk:
            break
        bars = chunk + bars
        end_ms = int(chunk[0][0]) - 1
        remaining -= len(chunk)
        if len(chunk) < n:
            break
        time.sleep(0.1)
    df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume",
                                      "ct", "qv", "n", "tbbv", "tbqv", "ig"])
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms")
    df = df.drop_duplicates("time").sort_values("time").tail(limit)
    return df.set_index("time").astype(float)


def calc_htf_trend(df4h: pd.DataFrame) -> pd.Series:
    sma20 = df4h["close"].rolling(20).mean()
    sma50 = df4h["close"].rolling(50).mean()
    return pd.Series(
        np.where(sma20 > sma50, 1, np.where(sma20 < sma50, -1, 0)),
        index=df4h.index,
    )


def lookup(series: pd.Series, ts: pd.Timestamp) -> int:
    pos = np.clip(series.index.searchsorted(ts, side="right") - 1, 0, len(series) - 1)
    return int(series.iloc[pos])


def precompute_startup_levels() -> None:
    """Load 1H CSVs (trimmed to 120 days) and run run_indicator once per symbol at startup.
    Uses the same 120-day window as the backtest so live level-proximity features
    match backtest features exactly — preventing the live bot taking 3x more signals.
    Takes ~10-20s for 14 symbols; runs once only.
    """
    global _startup_cache
    print("\nPre-computing Highlander levels from 120-day 1H history …")
    for symbol, csv_path in CSV_1H_PATHS.items():
        p = Path(csv_path)
        if not p.exists():
            print(f"  {symbol}: CSV missing — will use API-only levels")
            continue
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True).astype(float)
            df.index.name = "time"
            cutoff = df.index.max() - pd.Timedelta(days=120)
            df = df[df.index >= cutoff].copy()
            tick1 = max(float(df["close"].iloc[-1]) * 1e-5, 1e-6)
            lvl1, _ = run_indicator(df[["open","high","low","close"]],
                                    min_range_ticks=3.0, tick_size=tick1)
            ts1 = df.index.to_numpy().astype("int64")
            df4 = df.resample("4h", label="left").agg(
                {"open":"first","high":"max","low":"min","close":"last"}).dropna()
            tick4 = max(float(df4["close"].iloc[-1]) * 1e-5, 1e-6)
            lvl4, _ = run_indicator(df4[["open","high","low","close"]],
                                    min_range_ticks=3.0, tick_size=tick4)
            ts4 = df4.index.to_numpy().astype("int64")
            _startup_cache[symbol] = {"levels_1h": lvl1, "ts_1h": ts1,
                                      "levels_4h": lvl4, "ts_4h": ts4}
            print(f"  {symbol}: {len(df):,} bars  {len(lvl1)} 1H levels  {len(lvl4)} 4H levels")
        except Exception as exc:
            print(f"  {symbol}: failed ({exc}) — will use API-only levels")
    print(f"Startup cache ready for {len(_startup_cache)}/{len(CSV_1H_PATHS)} symbols.\n")


def _cache_bar(ts: pd.Timestamp, ts_int64: np.ndarray) -> int:
    """Convert a live timestamp to the matching bar index in the cached full-history array."""
    idx = int(np.searchsorted(ts_int64, np.int64(ts.value), side="right")) - 1
    return max(0, min(idx, len(ts_int64) - 1))


def get_confluence(bar_idx: int, price: float, is_bull: bool,
                   levels, atr: float) -> dict:
    above, below = [], []
    for lvl in levels:
        if lvl.created_bar > bar_idx:
            continue
        if lvl.deleted_bar is not None and lvl.deleted_bar <= bar_idx:
            continue
        if lvl.state == STATE_BROKEN_BSUT:
            continue
        diff = lvl.price - price
        info = dict(
            direction=lvl.dir,
            is_origin=(lvl.state == STATE_ORIGIN),
            confirmed=lvl.confirmed,
            dist_pct=abs(diff) / price * 100,
            dist_atr=abs(diff) / atr if atr > 0 else 99.0,
        )
        (above if diff > 0 else below).append(info)

    above.sort(key=lambda x: x["dist_pct"])
    below.sort(key=lambda x: x["dist_pct"])

    sup = (below[0] if below else None) if is_bull else (above[0] if above else None)
    res = (above[0] if above else None) if is_bull else (below[0] if below else None)

    g = lambda d, k, default: d[k] if d else default
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
    return {
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
    }


def get_entry_level_feat(entry: float, is_bull: bool, atr: float,
                         levels, i_cur: int, prefix: str) -> dict:
    """Find nearest Highlander level (origin/first-touch/broken) to entry price."""
    best = None; best_dist = float("inf")
    for lvl in levels:
        if lvl.created_bar > i_cur: continue
        if lvl.deleted_bar is not None and lvl.deleted_bar <= i_cur: continue
        if lvl.state not in (STATE_ORIGIN, STATE_BREAK_TOUCHED, STATE_BROKEN_BSUT): continue
        dist = abs(lvl.price - entry)
        if dist < best_dist:
            best_dist = dist; best = lvl
    if best is None:
        return {f"{prefix}_at_level": 0, f"{prefix}_dist_atr": 99.0,
                f"{prefix}_is_origin": 0, f"{prefix}_is_ft": 0,
                f"{prefix}_is_broken": 0, f"{prefix}_dir_match": 0}
    dist_atr = round(best_dist / atr, 3) if atr > 0 else 99.0
    return {
        f"{prefix}_at_level":  int(dist_atr <= 2.0),
        f"{prefix}_dist_atr":  dist_atr,
        f"{prefix}_is_origin": int(best.state == STATE_ORIGIN),
        f"{prefix}_is_ft":     int(best.state == STATE_BREAK_TOUCHED),
        f"{prefix}_is_broken": int(best.state == STATE_BROKEN_BSUT),
        f"{prefix}_dir_match": int((is_bull  and best.dir == DIR_UP) or
                                   (not is_bull and best.dir == DIR_DOWN)),
    }


def _ts(v) -> str:
    return str(v) if isinstance(v, pd.Timestamp) else v


# ── Discord webhook helpers ───────────────────────────────────────────────────
def send_discord(content: str = "", *, embeds: list | None = None) -> None:
    if not DISCORD_WEBHOOK:
        return
    payload: dict = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds
    try:
        import requests
        r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as exc:
        print(f"  Discord send failed: {exc!r}")


# ── Execution helpers ─────────────────────────────────────────────────────────
def _make_tex():
    """Create authenticated Bitget futures exchange instance."""
    import ccxt
    return ccxt.bitget({
        "apiKey":          BITGET_API_KEY,
        "secret":          BITGET_SECRET,
        "password":        BITGET_PASSPHRASE,
        "enableRateLimit": True,
        "options":         {"defaultType": "swap"},
    })


def _fsym(symbol: str) -> str:
    """'SOL/USDT' → 'SOL/USDT:USDT'  (perpetual futures symbol)."""
    return symbol.split("/")[0] + "/USDT:USDT"


def _free_balance(tex) -> float:
    bal = tex.fetch_balance({"type": "swap"})
    return float(bal.get("USDT", {}).get("free", 0.0))


def _open_trade_count(sym_states: dict) -> int:
    """Count trades across all symbols that have been executed (have an order_id)."""
    return sum(
        1 for st in sym_states.values()
        for t in st["active_trades"]
        if t.order_id is not None
    )


def _exec_open(tex, trade: ActiveTrade, sym_states: dict) -> bool:
    """
    Open a futures position for `trade`.
    Returns True on success, False if skipped (concurrent trade cap).
    Sets trade.order_id / sl_order_id / tp_order_id / qty in-place.
    """
    if _open_trade_count(sym_states) >= MAX_CONCURRENT:
        send_discord(content=(
            f"⚠️ **{trade.symbol}** {trade.direction} signal — "
            f"concurrent cap ({MAX_CONCURRENT}) reached, enter manually if desired."
        ))
        return False

    # Don't open a second trade on the same symbol
    for t in sym_states[trade.symbol]["active_trades"]:
        if t.order_id is not None and t.eng_time != trade.eng_time:
            send_discord(content=(
                f"⚠️ **{trade.symbol}** {trade.direction} signal skipped — "
                f"already have an active trade on this symbol."
            ))
            return False

    fsym       = _fsym(trade.symbol)
    is_long    = (trade.direction == "Long")
    entry_side = "buy"  if is_long else "sell"
    close_side = "sell" if is_long else "buy"

    balance = _free_balance(tex)
    risk_amt = balance * ACCOUNT_RISK_PCT
    risk_per_unit = abs(trade.entry - trade.stop)
    if risk_per_unit <= 0:
        print(f"  exec_open: zero risk range for {trade.symbol}, skip")
        return False

    lev = LEVERAGE  # fixed — no dynamic scaling

    # Risk-based qty, capped so margin never exceeds MAX_MARGIN_PCT of balance
    raw_qty = risk_amt / risk_per_unit
    max_qty = (balance * MAX_MARGIN_PCT * lev) / trade.entry
    qty     = float(tex.amount_to_precision(fsym, min(raw_qty, max_qty)))
    if qty <= 0:
        print(f"  exec_open: qty rounded to 0 for {trade.symbol}, skip")
        return False

    try:
        tex.set_leverage(lev, fsym, params={"marginCoin": "USDT"})
    except Exception as e:
        print(f"  set_leverage warning {trade.symbol}: {e}")
        send_discord(content=(
            f"⚠️ **Leverage not set** — {trade.direction} {trade.symbol}  "
            f"Tried `{lev}x`, Bitget may be using its default.  "
            f"Set leverage manually to `{lev}x` on Bitget.  Error: `{e}`"
        ))

    notional = qty * trade.entry

    # Limit entry at exact entry price — matches backtest assumption
    entry_price = float(tex.price_to_precision(fsym, trade.entry))
    order = tex.create_order(fsym, "limit", entry_side, qty, entry_price, params={})
    trade.order_id = order["id"]
    trade.qty      = qty

    # Wait up to LIMIT_FILL_TIMEOUT seconds for the limit to fill.
    # Price should already be at/past entry when activation was detected,
    # so a fill within a few seconds is expected.
    filled = False
    elapsed = 0
    while elapsed < LIMIT_FILL_TIMEOUT:
        time.sleep(5)
        elapsed += 5
        try:
            status = tex.fetch_order(trade.order_id, fsym)
            if status.get("status") in ("closed", "filled"):
                filled = True
                break
        except Exception:
            pass

    if not filled:
        # Price bounced above entry before our limit filled — cancel and drop trade
        try:
            tex.cancel_order(trade.order_id, fsym)
        except Exception:
            pass
        trade.order_id = None
        # Remove this trade from active_trades so it doesn't occupy a slot
        sym_states[trade.symbol]["active_trades"] = [
            t for t in sym_states[trade.symbol]["active_trades"]
            if t.eng_time != trade.eng_time
        ]
        print(f"  ⏳ Limit not filled {trade.symbol} @ {entry_price} — cancelled, signal dropped")
        send_discord(content=(
            f"⏳ **Limit not filled** — {trade.direction} {trade.symbol}  "
            f"Entry `{entry_price}` not reached within {LIMIT_FILL_TIMEOUT}s — cancelled."
        ))
        return False

    sl_price = float(tex.price_to_precision(fsym, trade.stop))
    tp_price = float(tex.price_to_precision(fsym, trade.f2618))

    # Stop loss TPSL order (pos_loss plan — correct for one-way mode)
    try:
        sl = tex.create_order(fsym, "market", close_side, qty, params={
            "stopLossPrice": sl_price,
        })
        trade.sl_order_id = sl["id"]
    except Exception as e:
        print(f"  SL order failed {trade.symbol}: {e} — closing position immediately")
        send_discord(content=(
            f"🚨 **SL FAILED** — {trade.direction} {trade.symbol}  "
            f"Could not place SL at `{sl_price}` — closing position at market now.  Error: `{e}`"
        ))
        try:
            tex.create_order(fsym, "market", close_side, qty, params={"reduceOnly": True})
            print(f"  Emergency close executed for {trade.symbol} — no SL could be placed")
            send_discord(content=(
                f"🏁 **Emergency close** — {trade.direction} {trade.symbol}  "
                f"Position closed at market (SL placement failed)."
            ))
        except Exception as e2:
            print(f"  EMERGENCY CLOSE FAILED {trade.symbol}: {e2} — manual action required!")
            send_discord(content=(
                f"🚨🚨 **EMERGENCY CLOSE FAILED** — {trade.symbol}  "
                f"Position is OPEN with NO stop loss — manual action required on Bitget!  Error: `{e2}`"
            ))
        return False

    # Take profit limit order
    try:
        tp = tex.create_order(fsym, "limit", close_side, qty, tp_price, params={
            "reduceOnly": True,
        })
        trade.tp_order_id = tp["id"]
    except Exception as e:
        print(f"  TP order warning {trade.symbol}: {e}")

    margin = notional / lev
    _journal_open(trade)
    print(f"  ✅ OPENED {trade.direction} {trade.symbol}  qty={qty}  entry={entry_price}  "
          f"SL={sl_price}  TP={tp_price}  margin=~${margin:.2f}  "
          f"risk=${risk_amt:.2f} ({ACCOUNT_RISK_PCT*100:.1f}%)  lev={lev}x")
    send_discord(content=(
        f"✅ **EXECUTED** {trade.direction} {trade.symbol}  "
        f"entry `{entry_price}` (limit)  qty `{qty}`  SL `{sl_price}`  TP `{tp_price}`  "
        f"margin ~`${margin:.2f}`  risk `{ACCOUNT_RISK_PCT*100:.1f}%`  lev `{lev}x`"
    ))
    return True


def _exec_move_sl(tex, trade: ActiveTrade) -> None:
    """Move stop loss to entry (break-even) after 1.618 is touched."""
    fsym       = _fsym(trade.symbol)
    is_long    = (trade.direction == "Long")
    close_side = "sell" if is_long else "buy"
    be_price   = float(tex.price_to_precision(fsym, trade.entry))

    # Cancel old SL (TPSL plan order)
    if trade.sl_order_id:
        try:
            tex.cancel_order(trade.sl_order_id, fsym, params={"trigger": True, "planType": "pos_loss"})
        except Exception as e:
            print(f"  cancel SL warning {trade.symbol}: {e}")

    # New SL at entry (TPSL pos_loss order)
    try:
        sl = tex.create_order(fsym, "market", close_side, trade.qty, params={
            "stopLossPrice": be_price,
        })
        trade.sl_order_id = sl["id"]
        trade.be_sl_set   = True
        print(f"  🔒 BE stop set {be_price} for {trade.symbol} (entry {trade.entry:.6g})")
        send_discord(content=(
            f"🔒 **BE stop set** — {trade.direction} {trade.symbol}  "
            f"SL moved to `{be_price}` (entry `{trade.entry:.6g}`)"
        ))
    except Exception as e:
        # Error 40917 (long) / 40916 (short): stop price on wrong side of mark price.
        # This means price has already retraced to entry — close at market as a
        # normal BE exit, no alarm needed.
        err_str = str(e)
        is_price_at_entry = "40917" in err_str or "40916" in err_str or "45122" in err_str or "45121" in err_str
        # 43023 = position already closed (e.g. TP limit order already filled) — not an error
        if "43023" in err_str:
            trade.be_sl_set = True
            print(f"  {trade.symbol} position already closed (TP filled) — SL move skipped")
            return
        if is_price_at_entry:
            print(f"  BE close {trade.symbol}: price at entry, closing at market")
            try:
                tex.create_order(fsym, "market", close_side, trade.qty, params={"reduceOnly": True})
                trade.be_sl_set = True
                print(f"  🏁 BE market close executed for {trade.symbol}")
                send_discord(content=(
                    f"🔒 **BE close** — {trade.direction} {trade.symbol}  "
                    f"Price returned to entry `{be_price}` before SL could move — closed at market."
                ))
            except Exception as e2:
                print(f"  BE close failed {trade.symbol}: {e2}")
                send_discord(content=(
                    f"🚨 **EMERGENCY CLOSE FAILED** — {trade.symbol}  "
                    f"Manual intervention required on Bitget!  Error: `{e2}`"
                ))
        else:
            print(f"  move SL error {trade.symbol}: {e} — attempting emergency market close")
            send_discord(content=(
                f"🚨 **SL MOVE FAILED** — {trade.direction} {trade.symbol}  "
                f"Could not move SL to entry `{be_price}`.  "
                f"Closing position at market.  Error: `{e}`"
            ))
            try:
                tex.create_order(fsym, "market", close_side, trade.qty, params={"reduceOnly": True})
                trade.be_sl_set = True
                print(f"  🏁 Emergency close executed for {trade.symbol}")
            except Exception as e2:
                print(f"  emergency close failed {trade.symbol}: {e2}")
                send_discord(content=(
                    f"🚨 **EMERGENCY CLOSE FAILED** — {trade.symbol}  "
                    f"Manual intervention required on Bitget!  Error: `{e2}`"
                ))


def _exec_close(tex, trade: ActiveTrade, reason: str) -> None:
    """Market-close position and cancel remaining SL/TP orders."""
    if not trade.order_id:
        return  # was never executed
    fsym       = _fsym(trade.symbol)
    is_long    = (trade.direction == "Long")
    close_side = "sell" if is_long else "buy"

    if trade.tp_order_id:
        try:
            tex.cancel_order(trade.tp_order_id, fsym)
        except Exception as e:
            print(f"  cancel TP warning {trade.symbol}: {e}")
    if trade.sl_order_id:
        try:
            tex.cancel_order(trade.sl_order_id, fsym, params={"trigger": True, "planType": "pos_loss"})
        except Exception as e:
            print(f"  cancel SL warning {trade.symbol}: {e}")

    if trade.qty and trade.qty > 0:
        try:
            tex.create_order(fsym, "market", close_side, trade.qty, params={
                "reduceOnly": True,
            })
            print(f"  🏁 CLOSED {trade.symbol} ({reason})")
        except Exception as e:
            if "22002" in str(e):
                print(f"  🏁 CLOSED {trade.symbol} ({reason})  [exchange-closed]")
            else:
                print(f"  close position warning {trade.symbol}: {e}")


def _process_execution(tex, r: dict, sym_states: dict) -> None:
    """Called after every scan_symbol result to open/modify/close positions."""
    # 1. Open new trades
    for trade in r.get("newly_active", []):
        try:
            _exec_open(tex, trade, sym_states)
        except Exception as e:
            print(f"  exec_open error {trade.symbol}: {e}")
            send_discord(content=f"⚠️ Execution error opening {trade.symbol}: `{e}`")

    # 2. Move SL to entry when BE level (1.618) is first touched.
    # Retries up to 3 times then gives up to avoid spam.
    MAX_SL_ATTEMPTS = 3
    for trade in r.get("active_trades", []):
        if trade.be_hit and not trade.be_sl_set and trade.order_id:
            if trade.sl_move_attempts >= MAX_SL_ATTEMPTS:
                continue
            trade.sl_move_attempts += 1
            try:
                _exec_move_sl(tex, trade)
            except Exception as e:
                print(f"  exec_move_sl error {trade.symbol}: {e}")
                send_discord(content=f"⚠️ Error moving SL for {trade.symbol}: `{e}`")
            if not trade.be_sl_set and trade.sl_move_attempts >= MAX_SL_ATTEMPTS:
                trade.be_sl_set = True  # give up — manual intervention needed
                send_discord(content=(
                    f"⚠️ **{trade.symbol}** SL move failed {MAX_SL_ATTEMPTS}x — "
                    f"giving up. Move SL to entry manually on Bitget."
                ))

    # 3. Close positions that the bot detected as won or stopped
    for ct in r.get("newly_closed", []):
        # Find the matching ActiveTrade (now removed from active_trades) by eng_time
        # We stored it in sym_states before update — look it up via newly_closed fields
        # The trade object was in active_trades before scan; pass its info via r
        for trade in r.get("_closed_trade_objects", []):
            if trade.eng_time == ct.eng_time and trade.symbol == ct.symbol:
                try:
                    _exec_close(tex, trade, ct.outcome)
                except Exception as e:
                    print(f"  exec_close error {ct.symbol}: {e}")


def _wh_pending_embed(s: Pending, bars_left: int, symbol: str, cur_price: float = 0.0) -> dict:
    label = symbol.split("/")[0]
    emoji = "📈" if s.is_bull else "📉"
    dir_  = "Long" if s.is_bull else "Short"
    if cur_price > 0:
        pct_away = (s.entry - cur_price) / cur_price * 100 if s.is_bull else (cur_price - s.entry) / cur_price * 100
        dist_str = f"{pct_away:+.2f}% away"
    else:
        dist_str = "—"
    return {
        "title":  f"{emoji} New pending setup — {dir_}  {label}/USDT 1H",
        "color":  0x5588FF if s.is_bull else 0xFF8855,
        "fields": [
            {"name": "Entry  (Fib 1.0)",  "value": f"`{s.entry:.5f}`",   "inline": True},
            {"name": "Stop   (Fib −0.5)", "value": f"`{s.stop:.5f}`",    "inline": True},
            {"name": "Expires in",        "value": f"{bars_left} bars",  "inline": True},
            {"name": "Distance",          "value": dist_str,             "inline": True},
        ],
        "footer": {"text": f"Engulfing bar: {s.eng_time}"},
    }


def _wh_signal_embed(t: ActiveTrade) -> dict:
    label    = t.symbol.split("/")[0]
    emoji    = "🟢" if t.direction == "Long" else "🔴"
    risk_pct = t.risk / t.entry * 100
    return {
        "title":  f"{emoji} {t.direction.upper()} ACTIVATED  {label}/USDT 1H  |  P(win) = {t.p_win:.3f}",
        "color":  0x00CC44 if t.direction == "Long" else 0xFF3333,
        "fields": [
            {"name": "Entry  (Fib 1.0)",          "value": f"`{t.entry:.5f}`",                        "inline": True},
            {"name": "Stop   (Fib −0.5)",          "value": f"`{t.stop:.5f}`",                         "inline": True},
            {"name": "Risk",                        "value": f"`{t.risk:.5f}  ({risk_pct:.2f}%)`",     "inline": True},
            {"name": "BE level  (Fib 1.618)",       "value": f"`{t.f1618:.5f}`",                       "inline": True},
            {"name": "Take Profit  (Fib 2.618)",    "value": f"`{t.f2618:.5f}`",                       "inline": True},
            {"name": "Session",                     "value": t.session,                                 "inline": True},
            {"name": f"{label} HTF",                "value": t.htf,                                     "inline": True},
            {"name": "BTC Trend",                   "value": t.btc,                                     "inline": True},
            {"name": "HTF Aligned",                 "value": "✅ Yes" if (
                (t.direction == "Long" and t.htf == "Bullish") or
                (t.direction == "Short" and t.htf == "Bearish")) else "❌ No",                          "inline": True},
        ],
        "footer": {"text": f"Engulfing: {t.eng_time} UTC  |  Activated: {t.activated_time} UTC"},
    }


def _wh_closed_embed(c: ClosedTrade) -> dict:
    label = c.symbol.split("/")[0]
    won   = (c.outcome == "won")
    emoji = "✅" if won else "❌"
    color = 0x00CC44 if won else 0xFF3333
    title = f"{emoji} {'WON' if won else 'STOPPED'}  {c.direction}  {label}/USDT 1H"
    return {
        "title":  title,
        "color":  color,
        "fields": [
            {"name": "Entry",      "value": f"`{c.entry:.5f}`",                    "inline": True},
            {"name": "Exit level", "value": f"`{c.f2618:.5f}`" if won else f"`{c.stop:.5f}`", "inline": True},
            {"name": "P&L",        "value": f"`{c.pnl_r:+.2f}R`",                 "inline": True},
        ],
        "footer": {"text": f"Activated: {c.activated_time} UTC  |  Closed: {c.closed_time} UTC"},
    }


def _wh_reminder_embed(t: ActiveTrade, remaining: int) -> dict:
    label   = t.symbol.split("/")[0]
    emoji   = "🟢" if t.direction == "Long" else "🔴"
    return {
        "title":  f"🔔 REMINDER ({remaining} left) — {emoji} {t.direction.upper()} {label}/USDT",
        "color":  0xFF9900,
        "fields": [
            {"name": "Entry",       "value": f"`{t.entry:.5f}`",  "inline": True},
            {"name": "Stop",        "value": f"`{t.stop:.5f}`",   "inline": True},
            {"name": "TP 2.618",    "value": f"`{t.f2618:.5f}`",  "inline": True},
            {"name": "BE level",    "value": f"`{t.f1618:.5f}`",  "inline": True},
            {"name": "P(win)",      "value": f"`{t.p_win:.3f}`",  "inline": True},
        ],
        "footer": {"text": f"Activated: {t.activated_time} UTC  |  Move SL to entry once {t.f1618:.5f} is touched"},
    }


def _wh_filtered_embed(s: Pending, symbol: str, direction: str, p_win: float, threshold: float) -> dict:
    label = symbol.split("/")[0]
    emoji = "⛔"
    return {
        "title":  f"{emoji} Filtered — {direction}  {label}/USDT 1H  |  P(win) = {p_win:.3f}",
        "color":  0xAAAAAA,
        "fields": [
            {"name": "Entry",       "value": f"`{s.entry:.5f}`",   "inline": True},
            {"name": "Stop",        "value": f"`{s.stop:.5f}`",    "inline": True},
            {"name": "TP 2.618",    "value": f"`{s.f2618:.5f}`",   "inline": True},
            {"name": "P(win)",      "value": f"`{p_win:.3f}`  (threshold {threshold})",  "inline": True},
        ],
        "footer": {"text": f"Engulfing: {s.eng_time}  |  Entry retested but ML score too low — skip this trade"},
    }


def _wh_cancelled_embed(s: Pending, symbol: str) -> dict:
    label = symbol.split("/")[0]
    emoji = "⏰"
    dir_  = "Long" if s.is_bull else "Short"
    return {
        "title":  f"{emoji} Pending expired — {dir_}  {label}/USDT 1H",
        "color":  0x555555,
        "fields": [
            {"name": "Entry", "value": f"`{s.entry:.5f}`", "inline": True},
            {"name": "Stop",  "value": f"`{s.stop:.5f}`",  "inline": True},
        ],
        "footer": {"text": f"Engulfing bar: {s.eng_time}  |  Expired after {MAX_BARS_TO_ACT} bars"},
    }


# ── Per-symbol scanner ────────────────────────────────────────────────────────
def scan_symbol(
    symbol: str,
    clf: CatBoostClassifier,
    exchange,
    btc_trend: pd.Series,
    sym_state: dict,
) -> dict:
    df1h_raw = fetch_binance_bars(symbol, "1h", LOOKBACK_1H)
    # Grab live intrabar high/low before dropping the forming bar.
    # Used only for pending-setup activation so we enter the moment price
    # touches the entry level — not an hour later at bar close.
    ts_forming = df1h_raw.index[-1]   # timestamp of the still-open bar
    live_hi    = float(df1h_raw["high"].iloc[-1])
    live_lo    = float(df1h_raw["low"].iloc[-1])
    df1h = df1h_raw.iloc[:-1]
    df4h = fetch_binance_bars(symbol, "4h", LOOKBACK_4H).iloc[:-1]

    # Fetch recent 1m bars for active trade monitoring.
    # Using closed 1m bars avoids the forming 1H bar accumulation problem
    # (1H high/low grows throughout the hour — a wick to f1618 at minute 5
    # keeps live_hi above f1618 for the remaining 55 minutes).
    df1m_mon = None
    if sym_state.get("active_trades"):
        try:
            df1m_raw = fetch_binance_bars(symbol, "1m", 5)
            df1m_mon = df1m_raw.iloc[:-1]  # drop forming 1m bar
        except Exception:
            pass

    df1h["atr"]       = true_range(df1h).rolling(ATR_LEN).mean()
    df1h["atr_pct"]   = df1h["atr"] / df1h["close"]
    df1h["vol_avg20"] = df1h["volume"].rolling(20).mean()
    df1h["vol_ratio"] = df1h["volume"] / df1h["vol_avg20"]

    htf_trend = calc_htf_trend(df4h)
    tick      = max(float(df1h["close"].iloc[-1]) * 1e-5, 1e-6)
    levels, _ = run_indicator(
        df1h[["open", "high", "low", "close"]],
        min_range_ticks=3.0, tick_size=tick,
    )
    tick4h       = max(float(df4h["close"].iloc[-1]) * 1e-5, 1e-6)
    levels_4h, _ = run_indicator(
        df4h[["open", "high", "low", "close"]],
        min_range_ticks=3.0, tick_size=tick4h,
    )
    df4h_idx  = df4h.index.to_numpy()
    df4h_atr  = true_range(df4h).rolling(ATR_LEN).mean().values  # numpy array

    O      = df1h["open"].to_numpy()
    H      = df1h["high"].to_numpy()
    L      = df1h["low"].to_numpy()
    C      = df1h["close"].to_numpy()
    ts_idx = df1h.index
    ts     = ts_idx[-1]

    pending           = sym_state["pending"]
    active_trades     = sym_state["active_trades"]
    closed_trades     = sym_state["closed_trades"]
    cancelled_pending = sym_state["cancelled_pending"]
    filtered_setups   = sym_state["filtered_setups"]
    notified_signals  = sym_state["notified_signals"]
    notified_pending  = sym_state["notified_pending"]
    notified_closed   = sym_state["notified_closed"]
    notified_filtered = sym_state.get("notified_filtered", set())

    htf_v_now = lookup(htf_trend, ts)
    btc_v_now = lookup(btc_trend, ts)
    htf_now   = {1: "Bullish", -1: "Bearish", 0: "Range"}[htf_v_now]
    btc_now   = {1: "Up",      -1: "Down",    0: "Range"}[btc_v_now]

    i_cur  = len(df1h) - 1
    hi_cur = H[i_cur]
    lo_cur = L[i_cur]

    # ── 0. Collect active origin levels ──────────────────────────────────────
    cur_price = float(df1h["close"].iloc[i_cur])
    atr_cur   = float(df1h["atr"].iloc[i_cur])
    o_sup, o_res = [], []
    for lvl in levels:
        if lvl.created_bar > i_cur:
            continue
        if lvl.deleted_bar is not None and lvl.deleted_bar <= i_cur:
            continue
        if lvl.state != STATE_ORIGIN:
            continue
        diff = lvl.price - cur_price
        info = {
            "price":     round(float(lvl.price), 5),
            "dist_pct":  round(abs(diff) / cur_price * 100, 3),
            "dist_atr":  round(abs(diff) / atr_cur, 3) if atr_cur > 0 else 99.0,
            "confirmed": bool(lvl.confirmed),
            "dir":       "Up" if lvl.dir == DIR_UP else "Down",
        }
        (o_res if diff > 0 else o_sup).append(info)
    o_sup.sort(key=lambda x: x["dist_pct"])
    o_res.sort(key=lambda x: x["dist_pct"])

    # ── 1. Check active trades for TP / stop ──────────────────────────────────
    still_active          = []
    newly_closed          = []
    closed_trade_objects  = []   # ActiveTrade refs for execution layer to call _exec_close
    for trade in active_trades:
        is_long = (trade.direction == "Long")

        # 1m bar monitoring only — never use 1H bar hi/lo for trade outcome checks.
        # The 1H bar can't tell order of events within the bar, causing false
        # triggers. 1m bars polled every 60s are sufficient and match the backtest.
        act_end = (trade.activation_bar + pd.Timedelta(hours=1)
                   if trade.activation_bar is not None else pd.Timestamp.min)
        valid_1m = (
            df1m_mon[df1m_mon.index > act_end]
            if (df1m_mon is not None and len(df1m_mon) > 0)
            else pd.DataFrame()
        )
        m_hi = float(valid_1m["high"].max()) if len(valid_1m) > 0 else None
        m_lo = float(valid_1m["low"].min())  if len(valid_1m) > 0 else None

        was_be_hit = trade.be_hit
        if not trade.be_hit:
            if m_hi is not None and ((is_long and m_hi >= trade.f1618) or (not is_long and m_lo <= trade.f1618)):
                trade.be_hit = True
                trade.be_hit_time = valid_1m.index[-1]

        be_just_triggered = not was_be_hit and trade.be_hit

        # Skip the activation bar — its H/L includes pre-activation price action.
        if trade.live_activated:
            if trade.activation_bar is not None and ts <= trade.activation_bar:
                still_active.append(trade)
                continue
            trade.live_activated = False

        cur_stop = trade.entry if trade.be_hit else trade.stop

        # After BE: only use 1m bars that closed AFTER the BE-triggering bar.
        # The bar that touched f1618 may also wick to entry in the same minute.
        if trade.be_hit and trade.be_hit_time is not None and len(valid_1m) > 0:
            post_be  = valid_1m[valid_1m.index > trade.be_hit_time]
            m_hi_chk = float(post_be["high"].max()) if len(post_be) > 0 else None
            m_lo_chk = float(post_be["low"].min())  if len(post_be) > 0 else None
        else:
            m_hi_chk = m_hi
            m_lo_chk = m_lo

        if be_just_triggered:
            # Skip stop check this cycle — Bitget's SL order handles it if
            # price is genuinely at entry.
            stopped = won = False
        else:
            stopped = (is_long  and m_lo_chk is not None and m_lo_chk <= cur_stop) or \
                      (not is_long and m_hi_chk is not None and m_hi_chk >= cur_stop)
            won     = (is_long  and m_hi_chk is not None and m_hi_chk >= trade.f2618) or \
                      (not is_long and m_lo_chk is not None and m_lo_chk <= trade.f2618)

        if won:
            pnl = abs(trade.f2618 - trade.entry) / trade.risk
            ct = ClosedTrade(
                symbol=trade.symbol, direction=trade.direction,
                entry=trade.entry, stop=trade.stop, f2618=trade.f2618,
                risk=trade.risk, p_win=trade.p_win,
                eng_time=trade.eng_time, activated_time=trade.activated_time,
                closed_time=ts, outcome="won", pnl_r=round(pnl, 2),
            )
            _journal_close(trade, "won", trade.f2618,
                           pnl_usdt=trade.risk * pnl, r_result=round(pnl, 4))
            newly_closed.append(ct)
            closed_trade_objects.append(trade)
        elif stopped:
            outcome = "be_stop" if trade.be_hit else "stopped"
            pnl_r   = -0.04    if trade.be_hit else -1.0
            exit_px = trade.entry if trade.be_hit else cur_stop
            ct = ClosedTrade(
                symbol=trade.symbol, direction=trade.direction,
                entry=trade.entry, stop=cur_stop, f2618=trade.f2618,
                risk=trade.risk, p_win=trade.p_win,
                eng_time=trade.eng_time, activated_time=trade.activated_time,
                closed_time=ts, outcome=outcome, pnl_r=pnl_r,
            )
            _journal_close(trade, outcome, exit_px,
                           pnl_usdt=trade.risk * pnl_r, r_result=pnl_r)
            newly_closed.append(ct)
            closed_trade_objects.append(trade)
        else:
            still_active.append(trade)

    for ct, trade_obj in zip(newly_closed, closed_trade_objects):
        closed_trades.append(ct)
        key = (ct.symbol, ct.eng_time, ct.outcome)
        if key not in notified_closed and trade_obj.order_id is not None:
            send_discord(embeds=[_wh_closed_embed(ct)])
            notified_closed.add(key)
    closed_trades = closed_trades[-MAX_CLOSED_HIST:]
    active_trades = still_active

    # ── 2. Detect new engulfings ──────────────────────────────────────────────
    known     = {p.eng_time for p in pending} | notified_signals | notified_filtered
    scan_from = max(1, len(df1h) - MAX_BARS_TO_ACT - 2)

    for i in range(scan_from, len(df1h)):
        t = ts_idx[i]
        if t in known:
            continue
        bull_eng = (C[i-1] < O[i-1]) and (C[i] > O[i]) and (C[i] > H[i-1])
        bear_eng = (C[i-1] > O[i-1]) and (C[i] < O[i]) and (C[i] < L[i-1])

        for is_bull in ([True] * int(bull_eng) + [False] * int(bear_eng)):
            origin = L[i-1] if is_bull else H[i-1]
            entry  = H[i-1] if is_bull else L[i-1]
            rng    = H[i-1] - L[i-1]
            if rng <= 0:
                continue
            atr_v = float(df1h["atr"].iloc[i])
            if not np.isfinite(atr_v) or atr_v <= 0:
                continue
            m  = 1 if is_bull else -1
            vr = float(df1h["vol_ratio"].iloc[i])
            already_hit = (
                any(
                    (is_bull and L[j] <= entry) or (not is_bull and H[j] >= entry)
                    for j in range(i + 1, len(df1h))
                ) or
                (is_bull and live_lo <= entry) or
                (not is_bull and live_hi >= entry)
            )
            if already_hit:
                continue

            # Engulf candle quality
            eng_rng        = H[i] - L[i]
            _er            = eng_rng if eng_rng > 0 else 1e-10
            body_pct       = round(abs(C[i] - O[i])              / _er, 3)
            upper_wick_pct = round((H[i] - max(O[i], C[i]))      / _er, 3)
            lower_wick_pct = round((min(O[i], C[i]) - L[i])      / _er, 3)
            engulf_ratio   = round(eng_rng / rng, 3) if rng > 0 else 1.0

            # Entry-level features at engulf bar time
            i_4h   = max(0, int(np.searchsorted(df4h_idx, t.to_numpy(), side="right")) - 1)
            i_4h   = min(i_4h, len(df4h) - 1)
            atr_4h = float(df4h_atr[i_4h]) if np.isfinite(df4h_atr[i_4h]) else atr_v
            _sc = _startup_cache.get(symbol)
            if _sc:
                _ei1 = _cache_bar(t, _sc["ts_1h"])
                _ei4 = _cache_bar(t, _sc["ts_4h"])
                lf1h      = get_entry_level_feat(entry,  is_bull, atr_v,  _sc["levels_1h"], _ei1, "entry_1h")
                lf4h      = get_entry_level_feat(entry,  is_bull, atr_4h, _sc["levels_4h"], _ei4, "entry_4h")
                lf_orig1h = get_entry_level_feat(origin, is_bull, atr_v,  _sc["levels_1h"], _ei1, "origin_1h")
                lf_orig4h = get_entry_level_feat(origin, is_bull, atr_4h, _sc["levels_4h"], _ei4, "origin_4h")
            else:
                lf1h      = get_entry_level_feat(entry,  is_bull, atr_v,  levels,    i,    "entry_1h")
                lf4h      = get_entry_level_feat(entry,  is_bull, atr_4h, levels_4h, i_4h, "entry_4h")
                lf_orig1h = get_entry_level_feat(origin, is_bull, atr_v,  levels,    i,    "origin_1h")
                lf_orig4h = get_entry_level_feat(origin, is_bull, atr_4h, levels_4h, i_4h, "origin_4h")
            # origin_* don't have an _at_level flag in NUM_FEATURES — drop it
            lf_orig1h = {k: v for k, v in lf_orig1h.items() if k != "origin_1h_at_level"}
            lf_orig4h = {k: v for k, v in lf_orig4h.items() if k != "origin_4h_at_level"}

            new_p = Pending(
                eng_time=t, is_bull=is_bull, origin=origin,
                entry=entry, rng=rng,
                stop=origin - 0.5   * rng * m,
                f1618=origin + 1.618 * rng * m,
                f2618=origin + 2.618 * rng * m,
                atr_at_eng=atr_v,
                atr_pct_at_eng=float(df1h["atr_pct"].iloc[i]),
                vol_ratio_at_eng=vr if np.isfinite(vr) else 1.0,
                body_pct=body_pct, upper_wick_pct=upper_wick_pct,
                lower_wick_pct=lower_wick_pct, engulf_ratio=engulf_ratio,
                **lf1h, **lf4h, **lf_orig1h, **lf_orig4h,
            )
            pending.append(new_p)
            if t not in notified_pending:
                send_discord(embeds=[_wh_pending_embed(new_p, MAX_BARS_TO_ACT, symbol, cur_price)])
                notified_pending.add(t)

    # ── 3. Check retests on current bar ──────────────────────────────────────
    atr_now  = float(df1h["atr"].iloc[i_cur])
    apct_now = float(df1h["atr_pct"].iloc[i_cur])

    surviving    = []
    newly_active = []
    retest_rows  = []
    newly_cancelled = []

    for s in pending:
        bars_elapsed = int(round((ts - s.eng_time).total_seconds() / 3600))
        if bars_elapsed > MAX_BARS_TO_ACT:
            newly_cancelled.append(s)
            cancelled_pending.append({
                "symbol":    symbol,
                "direction": "Long" if s.is_bull else "Short",
                "eng_time":  str(s.eng_time),
                "entry":     s.entry,
                "expired_at": str(ts),
            })
            continue

        # Same-bar guard: don't activate from a bar that IS the engulfing bar.
        # Closed bar: skip if ts == eng_time.
        # Live bar: skip if ts_forming == eng_time (forming bar is still the
        #           engulfing bar itself, which happens when the bot scans on
        #           the exact same hour the engulfing formed).
        closed_touched = (ts != s.eng_time) and (
            (s.is_bull  and lo_cur  <= s.entry) or
            (not s.is_bull and hi_cur  >= s.entry)
        )
        live_touched = (ts_forming != s.eng_time) and (
            (s.is_bull  and live_lo <= s.entry) or
            (not s.is_bull and live_hi >= s.entry)
        )
        if not (closed_touched or live_touched):
            surviving.append(s)
            continue

        if not np.isfinite(atr_now) or atr_now <= 0:
            surviving.append(s)
            continue

        direction = "Long" if s.is_bull else "Short"
        htf_align = int(
            (direction == "Long"  and htf_now == "Bullish") or
            (direction == "Short" and htf_now == "Bearish")
        )
        _sc = _startup_cache.get(symbol)
        if _sc:
            _ci = _cache_bar(ts, _sc["ts_1h"])
            conf = get_confluence(_ci, float(df1h["close"].iloc[i_cur]), s.is_bull, _sc["levels_1h"], atr_now)
        else:
            conf = get_confluence(i_cur, float(df1h["close"].iloc[i_cur]), s.is_bull, levels, atr_now)
        risk = abs(s.entry - s.stop)
        vr   = float(df1h["vol_ratio"].iloc[i_cur])

        feat = {
            "direction":          direction,
            "session":            session_of(ts.hour),
            "htf_trend":          htf_now,
            "btc_trend":          btc_now,
            "day":                ts.day_name()[:3],
            "bars_to_activation": bars_elapsed,
            "atr_pct_at_act":     round(apct_now, 6),
            "atr_pct_at_eng":     round(s.atr_pct_at_eng, 6),
            "rng_pct":            round(s.rng   / s.entry * 100, 4),
            "stop_pct":           round(risk     / s.entry * 100, 4),
            "tp1618_pct":         round(abs(s.f1618 - s.entry) / s.entry * 100, 4),
            "rng_to_atr":         round(s.rng / s.atr_at_eng, 4),
            "vol_ratio":          round(vr if np.isfinite(vr) else 1.0, 4),
            "hour":               int(ts.hour),
            "month":              int(ts.month),
            "htf_aligned":        htf_align,
            **conf,
            # Engulf candle quality (stored at detection time)
            "body_pct":           s.body_pct,
            "upper_wick_pct":     s.upper_wick_pct,
            "lower_wick_pct":     s.lower_wick_pct,
            "engulf_ratio":       s.engulf_ratio,
            # Entry-level proximity (stored at detection time)
            "entry_1h_at_level":  s.entry_1h_at_level,
            "entry_1h_dist_atr":  s.entry_1h_dist_atr,
            "entry_1h_is_origin": s.entry_1h_is_origin,
            "entry_1h_is_ft":     s.entry_1h_is_ft,
            "entry_1h_is_broken": s.entry_1h_is_broken,
            "entry_1h_dir_match": s.entry_1h_dir_match,
            "entry_4h_at_level":  s.entry_4h_at_level,
            "entry_4h_dist_atr":  s.entry_4h_dist_atr,
            "entry_4h_is_origin": s.entry_4h_is_origin,
            "entry_4h_is_ft":     s.entry_4h_is_ft,
            "entry_4h_is_broken": s.entry_4h_is_broken,
            "entry_4h_dir_match": s.entry_4h_dir_match,
            # Origin proximity (stored at detection time)
            "origin_1h_dist_atr":  s.origin_1h_dist_atr,
            "origin_1h_is_origin": s.origin_1h_is_origin,
            "origin_1h_is_ft":     s.origin_1h_is_ft,
            "origin_1h_is_broken": s.origin_1h_is_broken,
            "origin_1h_dir_match": s.origin_1h_dir_match,
            "origin_4h_dist_atr":  s.origin_4h_dist_atr,
            "origin_4h_is_origin": s.origin_4h_is_origin,
            "origin_4h_is_ft":     s.origin_4h_is_ft,
            "origin_4h_is_broken": s.origin_4h_is_broken,
            "origin_4h_dir_match": s.origin_4h_dir_match,
        }

        X = pd.DataFrame([feat])[FEATURES]
        for c in CAT_FEATURES:
            X[c] = X[c].astype(str)
        p_win = float(clf.predict_proba(X)[0, 1])

        if p_win >= P_THRESHOLD:
            is_live = (live_touched and not closed_touched)
            trade = ActiveTrade(
                symbol=symbol, direction=direction,
                entry=s.entry, stop=s.stop, f1618=s.f1618, f2618=s.f2618,
                risk=risk, p_win=p_win,
                eng_time=s.eng_time, activated_time=ts,
                session=feat["session"], htf=htf_now, btc=btc_now,
                live_activated=is_live,
                activation_bar=ts_forming if is_live else ts,
            )
            newly_active.append(trade)
            if s.eng_time not in notified_signals:
                send_discord(embeds=[_wh_signal_embed(trade)])
                notified_signals.add(s.eng_time)
            retest_rows.append((direction, s.eng_time, s.entry, s.stop, p_win, "SIGNAL"))
        else:
            retest_rows.append((direction, s.eng_time, s.entry, s.stop, p_win, "filtered"))
            if s.eng_time not in notified_filtered:
                send_discord(embeds=[_wh_filtered_embed(s, symbol, direction, p_win, P_THRESHOLD)])
                notified_filtered.add(s.eng_time)
            filtered_setups.append({
                "symbol":    symbol,
                "direction": direction,
                "eng_time":  str(s.eng_time),
                "entry":     s.entry,
                "stop":      s.stop,
                "f2618":     s.f2618,
                "p_win":     round(p_win, 3),
                "filtered_at": str(ts),
            })

    cancelled_pending = cancelled_pending[-MAX_CANCEL_HIST:]
    filtered_setups   = filtered_setups[-MAX_FILTERED_HIST:]
    active_trades.extend(newly_active)
    pending = surviving

    return {
        "pending":            pending,
        "active_trades":      active_trades,
        "closed_trades":      closed_trades,
        "cancelled_pending":  cancelled_pending,
        "filtered_setups":    filtered_setups,
        "notified_signals":   notified_signals,
        "notified_pending":   notified_pending,
        "notified_closed":    notified_closed,
        "notified_filtered":  notified_filtered,
        "df1h":               df1h,
        "htf_now":            htf_now,
        "btc_now":            btc_now,
        "ts":                 ts,
        "retest_rows":        retest_rows,
        "newly_active":          newly_active,
        "newly_cancelled":       newly_cancelled,
        "newly_closed":          newly_closed,
        "_closed_trade_objects": closed_trade_objects,
        "origin_support":     o_sup[:5],
        "origin_resistance":  o_res[:5],
    }


# ── State serialisation helpers ───────────────────────────────────────────────
def _trade_to_dict(t: ActiveTrade) -> dict:
    return {
        "symbol": t.symbol, "direction": t.direction,
        "entry": t.entry, "stop": t.stop, "f1618": t.f1618, "f2618": t.f2618,
        "risk": t.risk, "p_win": t.p_win, "be_hit": t.be_hit, "be_sl_set": t.be_sl_set,
        "be_hit_time": _ts(t.be_hit_time) if t.be_hit_time is not None else None,
        "sl_move_attempts": t.sl_move_attempts,
        "live_activated": t.live_activated,
        "eng_time": _ts(t.eng_time), "activated_time": _ts(t.activated_time),
        "session": t.session, "htf": t.htf, "btc": t.btc,
        "order_id": t.order_id, "sl_order_id": t.sl_order_id,
        "tp_order_id": t.tp_order_id, "qty": t.qty,
        "activation_bar": _ts(t.activation_bar) if t.activation_bar is not None else None,
    }


def _closed_to_dict(c: ClosedTrade) -> dict:
    return {
        "symbol": c.symbol, "direction": c.direction,
        "entry": c.entry, "stop": c.stop, "f2618": c.f2618,
        "risk": c.risk, "p_win": c.p_win, "outcome": c.outcome, "pnl_r": c.pnl_r,
        "eng_time": _ts(c.eng_time), "activated_time": _ts(c.activated_time),
        "closed_time": _ts(c.closed_time),
    }


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    import ccxt
    exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True, "options": {"defaultType": "swap"}})

    clfs: dict[str, CatBoostClassifier] = {}
    for sym, model_path in SYMBOLS.items():
        clf = CatBoostClassifier()
        clf.load_model(model_path)
        clfs[sym] = clf
        print(f"Model loaded  : {model_path}  ({sym})")

    precompute_startup_levels()   # ~2-3 min — builds full Highlander level cache

    print(f"Monitoring    : {', '.join(SYMBOLS.keys())} 1H on {EXCHANGE_ID}")
    print(f"Signal filter : P >= {P_THRESHOLD}")
    print("=" * 70)

    sym_states: dict[str, dict] = {
        sym: {
            "pending":           [],
            "active_trades":     [],
            "closed_trades":     [],
            "cancelled_pending": [],
            "filtered_setups":   [],
            "notified_signals":  set(),
            "notified_pending":  set(),
            "notified_closed":   set(),
            "notified_filtered": set(),
        }
        for sym in SYMBOLS
    }
    last_bar:          Optional[pd.Timestamp]  = None
    cached_btc_trend:  Optional[pd.Series]     = None
    reminder_queue:    list                    = []

    tex = None
    if AUTO_TRADE:
        try:
            tex = _make_tex()
            bal = _free_balance(tex)
            print(f"Auto-trade     : ENABLED  |  Balance: ${bal:.2f} USDT  |  "
                  f"Risk: {ACCOUNT_RISK_PCT*100:.1f}%  |  Leverage: {LEVERAGE}x  |  "
                  f"Max concurrent: {MAX_CONCURRENT}")
        except Exception as e:
            print(f"Auto-trade DISABLED — could not connect to Bitget: {e}")
            tex = None
    else:
        print("Auto-trade     : DISABLED (set BITGET_API_KEY / SECRET / PASSPHRASE to enable)")

    while True:
        try:
            # Lightweight probe — fetch 3 bars, then only drop the last if it's
            # still the forming (current) bar.  Using limit=2 and always dropping
            # iloc[-1] fails right after a bar closes because KuCoin sometimes
            # hasn't opened the next forming bar yet, returning [prev, just_closed]
            # — iloc[:-1] then gives prev == last_bar → stuck in sleep loop.
            first_sym   = next(iter(SYMBOLS))
            probe       = fetch_ohlcv(exchange, first_sym, "1h", 3)
            now_hour    = pd.Timestamp(datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)).floor("h")
            if probe.index[-1] >= now_hour:   # last bar is still forming → drop it
                probe = probe.iloc[:-1]
            current_bar = probe.index[-1]

            if current_bar == last_bar:
                # No new bar yet — but still poll symbols that have pending
                # setups so we catch live activations within the forming bar.
                # (Full scan only fires at bar close; this closes the gap.)
                syms_with_pending = [s for s in SYMBOLS if sym_states[s]["pending"] or sym_states[s]["active_trades"]]
                if syms_with_pending and cached_btc_trend is not None:
                    print(f"  ↻ intrabar poll: {', '.join(s.split('/')[0] for s in syms_with_pending)}")
                    for sym in syms_with_pending:
                        try:
                            r = scan_symbol(sym, clfs[sym], exchange, cached_btc_trend, sym_states[sym])
                            sym_states[sym].update({k: r[k] for k in (
                                "pending", "active_trades", "closed_trades",
                                "cancelled_pending", "filtered_setups",
                                "notified_signals", "notified_pending", "notified_closed",
                                "notified_filtered",
                            )})
                            if tex:
                                _process_execution(tex, r, sym_states)
                            for t in r.get("newly_active", []):
                                reminder_queue.append({"trade": t, "remaining": 5})
                        except Exception as e:
                            print(f"  intrabar poll error {sym}: {e}")
                # Fire pending reminders before sleeping
                still_pending = []
                for rem in reminder_queue:
                    send_discord(embeds=[_wh_reminder_embed(rem["trade"], rem["remaining"])])
                    rem["remaining"] -= 1
                    if rem["remaining"] > 0:
                        still_pending.append(rem)
                reminder_queue = still_pending
                time.sleep(30)
                continue

            # New bar — now fetch full data
            b4h              = fetch_binance_bars(SYMBOL_BTC, "4h", LOOKBACK_4H).iloc[:-1]
            btc_trend        = calc_htf_trend(b4h)
            cached_btc_trend = btc_trend

            results: dict[str, dict] = {}
            for sym in SYMBOLS:
                r = scan_symbol(sym, clfs[sym], exchange, btc_trend, sym_states[sym])
                sym_states[sym].update({k: r[k] for k in (
                    "pending", "active_trades", "closed_trades", "cancelled_pending",
                    "filtered_setups", "notified_signals", "notified_pending", "notified_closed",
                    "notified_filtered",
                )})
                if tex:
                    _process_execution(tex, r, sym_states)
                for t in r.get("newly_active", []):
                    reminder_queue.append({"trade": t, "remaining": 5})
                results[sym] = r

            ts      = results[first_sym]["ts"]
            # Anchor last_bar to the timestamp the scan actually used, not the
            # probe's view.  KuCoin sometimes delivers the just-closed bar as
            # "forming" in the large fetch but "closed" in the 3-bar probe, so
            # setting last_bar = current_bar (probe) would skip the real scan
            # of that bar.  Using ts keeps the bot in sync with the data.
            last_bar = ts
            btc_now = results[first_sym]["btc_now"]

            # ── Save state ────────────────────────────────────────────────────
            state: dict = {
                "last_scan": _ts(ts),
                "btc_trend": btc_now,
                "session":   session_of(ts.hour),
                "symbols":   {},
            }
            for sym, r in results.items():
                label   = sym.split("/")[0]
                st      = sym_states[sym]
                state["symbols"][sym] = {
                    "label":     label,
                    "price":     float(r["df1h"]["close"].iloc[-1]),
                    "htf_trend": r["htf_now"],
                    "pending": [
                        {
                            "direction": "Long" if s.is_bull else "Short",
                            "eng_time":  str(s.eng_time),
                            "entry":     s.entry, "stop": s.stop,
                            "f1618":     s.f1618, "f2618": s.f2618,
                            "bars_left": MAX_BARS_TO_ACT - int(round(
                                (ts - s.eng_time).total_seconds() / 3600)),
                        }
                        for s in sorted(st["pending"], key=lambda x: x.eng_time)
                    ],
                    "active_trades":      [_trade_to_dict(t) for t in st["active_trades"]],
                    "closed_trades":      [_closed_to_dict(c) for c in st["closed_trades"]],
                    "cancelled_pending":  st["cancelled_pending"],
                    "filtered_setups":    st["filtered_setups"],
                    "origin_support":     r.get("origin_support", []),
                    "origin_resistance":  r.get("origin_resistance", []),
                }
            pathlib.Path("data/scanner_state.json").write_text(json.dumps(state, indent=2))

            # ── Print summary ─────────────────────────────────────────────────
            print(f"\n{'═' * 70}")
            parts = [f"  {ts}"]
            for sym, r in results.items():
                label = sym.split("/")[0]
                parts.append(f"{label} ${float(r['df1h']['close'].iloc[-1]):.4f}  HTF: {r['htf_now']}")
            parts.append(f"BTC: {btc_now}  |  Session: {session_of(ts.hour)}")
            print("  |  ".join(parts))
            print(f"{'═' * 70}")

            for sym, r in results.items():
                label             = sym.split("/")[0]
                st                = sym_states[sym]
                pending           = st["pending"]
                active_trades     = st["active_trades"]
                closed_trades     = st["closed_trades"]
                cancelled_pending = st["cancelled_pending"]
                retest_rows       = r["retest_rows"]

                print(f"\n── {label}/USDT {'─' * (60 - len(label))}")

                # Pending
                print(f"\n  PENDING ({len(pending)})")
                if pending:
                    print(f"  {'Dir':<6}  {'Engulf Bar':<22}  {'Entry':>9}  {'Stop':>9}  {'TP 2.618':>10}  {'Expires':>10}")
                    print(f"  {'─'*6}  {'─'*22}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*10}")
                    for s in sorted(pending, key=lambda x: x.eng_time):
                        bl = MAX_BARS_TO_ACT - int(round((ts - s.eng_time).total_seconds() / 3600))
                        d  = "Long" if s.is_bull else "Short"
                        print(f"  {d:<6}  {str(s.eng_time):<22}  {s.entry:>9.4f}  {s.stop:>9.4f}  {s.f2618:>10.4f}  {bl:>7} bars")
                else:
                    print("  none")

                # Active trades
                print(f"\n  ACTIVE TRADES ({len(active_trades)})")
                if active_trades:
                    print(f"  {'Dir':<6}  {'Activated':<22}  {'Entry':>9}  {'Stop':>9}  {'TP 2.618':>10}  {'P(win)':>7}")
                    print(f"  {'─'*6}  {'─'*22}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*7}")
                    for t in active_trades:
                        print(f"  {t.direction:<6}  {str(t.activated_time):<22}  {t.entry:>9.4f}  {t.stop:>9.4f}  {t.f2618:>10.4f}  {t.p_win:>7.3f}")
                else:
                    print("  none")

                # Retests
                print(f"\n  RETESTS THIS BAR")
                if retest_rows:
                    print(f"  {'Dir':<6}  {'Engulf Bar':<22}  {'Entry':>9}  {'Stop':>9}  {'P(win)':>7}  Result")
                    print(f"  {'─'*6}  {'─'*22}  {'─'*9}  {'─'*9}  {'─'*7}  {'─'*8}")
                    for (d, et, en, st_, p, res) in retest_rows:
                        flag = "✅ SIGNAL" if res == "SIGNAL" else "filtered"
                        print(f"  {d:<6}  {str(et):<22}  {en:>9.4f}  {st_:>9.4f}  {p:>7.3f}  {flag}")
                else:
                    print("  none this bar")

                # Recent closed
                recent = closed_trades[-5:][::-1]
                print(f"\n  RECENT RESULTS ({len(closed_trades)} total)")
                if recent:
                    print(f"  {'Outcome':<8}  {'Dir':<6}  {'Closed':<22}  {'Entry':>9}  {'P&L':>7}")
                    print(f"  {'─'*8}  {'─'*6}  {'─'*22}  {'─'*9}  {'─'*7}")
                    for c in recent:
                        tag = "✅ WON  " if c.outcome == "won" else "❌ STOP "
                        print(f"  {tag}  {c.direction:<6}  {str(c.closed_time):<22}  {c.entry:>9.4f}  {c.pnl_r:>+7.2f}R")
                else:
                    print("  none yet")

                # Recently cancelled
                recent_cancel = cancelled_pending[-3:][::-1]
                print(f"\n  RECENTLY CANCELLED ({len(cancelled_pending)} total)")
                if recent_cancel:
                    for c in recent_cancel:
                        print(f"  {c['direction']:<6}  Entry {c['entry']:.4f}  eng: {c['eng_time']}  expired: {c['expired_at']}")
                else:
                    print("  none")

                # Recently filtered
                filtered_setups = st["filtered_setups"]
                recent_filtered = filtered_setups[-5:][::-1]
                print(f"\n  RECENTLY FILTERED ({len(filtered_setups)} total)")
                if recent_filtered:
                    print(f"  {'Dir':<6}  {'Engulf Bar':<22}  {'Entry':>9}  {'P(win)':>7}  {'Filtered At'}")
                    print(f"  {'─'*6}  {'─'*22}  {'─'*9}  {'─'*7}  {'─'*20}")
                    for f in recent_filtered:
                        print(f"  {f['direction']:<6}  {f['eng_time']:<22}  {f['entry']:>9.4f}  {f['p_win']:>7.3f}  {f['filtered_at']}")
                else:
                    print("  none")

            print()

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:
            print(f"  Error: {exc!r} — retrying in 60s")
            time.sleep(60)


# ── Discord slash-command bot ─────────────────────────────────────────────────
def start_discord_bot() -> None:
    BOT_TOKEN = os.getenv("BOT_TOKEN", "")
    if not BOT_TOKEN:
        return
    try:
        import discord
        from discord import app_commands
    except ImportError:
        print("discord.py not installed — bot disabled.  Run: pip install discord.py")
        return

    intents = discord.Intents.default()
    client  = discord.Client(intents=intents)
    tree    = app_commands.CommandTree(client)

    def _load_state() -> dict | None:
        p = pathlib.Path("data/scanner_state.json")
        return json.loads(p.read_text()) if p.exists() else None

    def _pending_embed(state: dict, sym: str) -> discord.Embed:
        sd    = state.get("symbols", {}).get(sym, {})
        label = sd.get("label", sym.split("/")[0])
        rows  = sd.get("pending", [])
        embed = discord.Embed(
            title=f"📋  Pending Setups  ({len(rows)})  —  {label}/USDT 1H",
            color=0x5588FF,
        )
        embed.set_footer(text=(
            f"Last scan: {state.get('last_scan','?')}  |  "
            f"{label} ${sd.get('price',0):.4f}  |  "
            f"HTF: {sd.get('htf_trend','?')}  |  "
            f"BTC: {state.get('btc_trend','?')}  |  "
            f"Session: {state.get('session','?')}"
        ))
        if not rows:
            embed.description = "_No pending setups._"
            return embed
        for s in rows:
            emoji = "📈" if s["direction"] == "Long" else "📉"
            embed.add_field(
                name=f"{emoji} {s['direction']}  |  Entry `{s['entry']:.5f}`",
                value=(
                    f"Stop `{s['stop']:.5f}`  |  TP `{s['f2618']:.5f}`\n"
                    f"Expires in **{s['bars_left']} bars**  |  Engulf: `{s['eng_time']}`"
                ),
                inline=False,
            )
        return embed

    def _active_embed(state: dict, sym: str) -> discord.Embed:
        sd     = state.get("symbols", {}).get(sym, {})
        label  = sd.get("label", sym.split("/")[0])
        trades = sd.get("active_trades", [])
        price  = sd.get("price", 0)
        embed  = discord.Embed(
            title=f"⚡  Active Trades  ({len(trades)})  —  {label}/USDT 1H",
            color=0xFFAA00,
        )
        embed.set_footer(text=f"{label} ${price:.4f}  |  Last scan: {state.get('last_scan','?')}")
        if not trades:
            embed.description = "_No active trades._"
            return embed
        for t in trades:
            emoji     = "🟢" if t["direction"] == "Long" else "🔴"
            is_long   = t["direction"] == "Long"
            unreal_pct = (price - t["entry"]) / t["risk"] if is_long else (t["entry"] - price) / t["risk"]
            sign      = "+" if unreal_pct >= 0 else ""
            be_hit    = t.get("be_hit", False)
            cur_stop  = t["entry"] if be_hit else t["stop"]
            be_label  = "🔒 BE" if be_hit else "original"
            embed.add_field(
                name=f"{emoji} {t['direction']}  |  Entry `{t['entry']:.5f}`  |  P(win) {t['p_win']:.2f}",
                value=(
                    f"Stop `{cur_stop:.5f}` ({be_label})  |  TP `{t['f2618']:.5f}`\n"
                    f"Unrealised: **{sign}{unreal_pct:.2f}R**  |  HTF: {t['htf']}  |  Activated: `{t['activated_time']}`"
                ),
                inline=False,
            )
        return embed

    def _results_embed(state: dict, sym: str) -> discord.Embed:
        sd     = state.get("symbols", {}).get(sym, {})
        label  = sd.get("label", sym.split("/")[0])
        closed = sd.get("closed_trades", [])
        total_r = sum(c["pnl_r"] for c in closed)
        wins    = sum(1 for c in closed if c["outcome"] == "won")
        embed   = discord.Embed(
            title=f"📊  Results  ({len(closed)} trades)  —  {label}/USDT 1H",
            color=0x00CC44 if total_r >= 0 else 0xFF3333,
        )
        embed.set_footer(text=(
            f"Win rate: {wins}/{len(closed)}  |  "
            f"Total P&L: {total_r:+.2f}R  |  "
            f"Last scan: {state.get('last_scan','?')}"
        ))
        if not closed:
            embed.description = "_No closed trades this session._"
            return embed
        for c in reversed(closed[-10:]):
            emoji = "✅" if c["outcome"] == "won" else "❌"
            embed.add_field(
                name=f"{emoji} {c['direction']}  |  {c['pnl_r']:+.2f}R  |  {c['closed_time']}",
                value=f"Entry `{c['entry']:.5f}`  |  P(win) {c['p_win']:.2f}",
                inline=False,
            )
        return embed

    def _filtered_embed(state: dict, sym: str) -> discord.Embed:
        sd       = state.get("symbols", {}).get(sym, {})
        label    = sd.get("label", sym.split("/")[0])
        filtered = sd.get("filtered_setups", [])
        embed    = discord.Embed(
            title=f"🚫  Filtered Setups  ({len(filtered)})  —  {label}/USDT 1H",
            color=0xFF8800,
        )
        embed.set_footer(text=f"Retested but model score < {P_THRESHOLD}  |  Last scan: {state.get('last_scan','?')}")
        if not filtered:
            embed.description = "_No filtered setups this session._"
            return embed
        for f in reversed(filtered[-10:]):
            emoji = "📈" if f["direction"] == "Long" else "📉"
            embed.add_field(
                name=f"{emoji} {f['direction']}  |  Entry `{f['entry']:.5f}`  |  P(win) `{f['p_win']:.3f}`",
                value=f"Stop `{f['stop']:.5f}`  |  TP `{f['f2618']:.5f}`  |  Engulf: `{f['eng_time']}`\nFiltered at: `{f['filtered_at']}`",
                inline=False,
            )
        return embed

    def _cancelled_embed(state: dict, sym: str) -> discord.Embed:
        sd        = state.get("symbols", {}).get(sym, {})
        label     = sd.get("label", sym.split("/")[0])
        cancelled = sd.get("cancelled_pending", [])
        embed     = discord.Embed(
            title=f"⏰  Cancelled Setups  ({len(cancelled)})  —  {label}/USDT 1H",
            color=0x555555,
        )
        if not cancelled:
            embed.description = "_No expired setups._"
            return embed
        for c in reversed(cancelled[-5:]):
            embed.add_field(
                name=f"{c['direction']}  |  Entry `{c['entry']:.5f}`",
                value=f"Engulf: `{c['eng_time']}`  |  Expired: `{c['expired_at']}`",
                inline=False,
            )
        return embed

    def _levels_embed(state: dict, sym: str) -> discord.Embed:
        sd    = state.get("symbols", {}).get(sym, {})
        label = sd.get("label", sym.split("/")[0])
        price = sd.get("price", 0)
        o_sup = sd.get("origin_support", [])
        o_res = sd.get("origin_resistance", [])
        embed = discord.Embed(
            title=f"🎯  Origin Levels  —  {label}/USDT  ${price:.4f}",
            color=0x9900FF,
        )
        embed.set_footer(text=(
            f"HTF: {sd.get('htf_trend','?')}  |  BTC: {state.get('btc_trend','?')}  |  "
            f"Scan: {state.get('last_scan','?')}"
        ))
        if o_res:
            lines = []
            for lv in o_res[:5]:
                tick = "✅" if lv["confirmed"] else "🔲"
                lines.append(
                    f"{tick} `{lv['price']:.5f}`  —  {lv['dist_pct']:.2f}% / {lv['dist_atr']:.1f}ATR above"
                )
            embed.add_field(name=f"🔴 Origin Resistance ({len(o_res)})", value="\n".join(lines), inline=False)
        else:
            embed.add_field(name="🔴 Origin Resistance", value="_none within lookback_", inline=False)

        if o_sup:
            lines = []
            for lv in o_sup[:5]:
                tick = "✅" if lv["confirmed"] else "🔲"
                lines.append(
                    f"{tick} `{lv['price']:.5f}`  —  {lv['dist_pct']:.2f}% / {lv['dist_atr']:.1f}ATR below"
                )
            embed.add_field(name=f"🟢 Origin Support ({len(o_sup)})", value="\n".join(lines), inline=False)
        else:
            embed.add_field(name="🟢 Origin Support", value="_none within lookback_", inline=False)

        return embed

    def _overview_embed(state: dict, sym: str) -> discord.Embed:
        sd       = state.get("symbols", {}).get(sym, {})
        label    = sd.get("label", sym.split("/")[0])
        price    = sd.get("price", 0)
        pend     = sd.get("pending", [])
        active   = sd.get("active_trades", [])
        closed   = sd.get("closed_trades", [])
        filtered = sd.get("filtered_setups", [])
        o_sup    = sd.get("origin_support", [])
        o_res    = sd.get("origin_resistance", [])
        total_r  = sum(c["pnl_r"] for c in closed)
        wins     = sum(1 for c in closed if c["outcome"] == "won")
        embed    = discord.Embed(
            title=f"📡  {label}/USDT 1H  —  ${price:.4f}",
            color=0x5588FF,
        )
        embed.set_footer(text=(
            f"HTF: {sd.get('htf_trend','?')}  |  BTC: {state.get('btc_trend','?')}  |  "
            f"Session: {state.get('session','?')}  |  Scan: {state.get('last_scan','?')}"
        ))
        # Pending
        if pend:
            lines = []
            for s in pend[:5]:
                e = "📈" if s["direction"] == "Long" else "📉"
                lines.append(f"{e} {s['direction']} `{s['entry']:.5f}` — {s['bars_left']}b left")
            embed.add_field(name=f"📋 Pending ({len(pend)})", value="\n".join(lines), inline=False)
        else:
            embed.add_field(name="📋 Pending (0)", value="_none_", inline=False)

        # Active
        if active:
            lines = []
            for t in active[:5]:
                e = "🟢" if t["direction"] == "Long" else "🔴"
                is_long = t["direction"] == "Long"
                unr = (price - t["entry"]) / t["risk"] if is_long else (t["entry"] - price) / t["risk"]
                lines.append(f"{e} {t['direction']} `{t['entry']:.5f}` — {unr:+.2f}R unreal")
            embed.add_field(name=f"⚡ Active ({len(active)})", value="\n".join(lines), inline=False)
        else:
            embed.add_field(name="⚡ Active (0)", value="_none_", inline=False)

        # Recent results
        if closed:
            lines = []
            for c in reversed(closed[-5:]):
                e = "✅" if c["outcome"] == "won" else "❌"
                lines.append(f"{e} {c['direction']} `{c['entry']:.5f}` — {c['pnl_r']:+.2f}R")
            embed.add_field(
                name=f"📊 Results ({wins}W/{len(closed)-wins}L, {total_r:+.2f}R total)",
                value="\n".join(lines), inline=False,
            )
        else:
            embed.add_field(name="📊 Results", value="_none yet_", inline=False)

        # Filtered
        if filtered:
            lines = []
            for f in reversed(filtered[-5:]):
                e = "📈" if f["direction"] == "Long" else "📉"
                lines.append(f"{e} {f['direction']} `{f['entry']:.5f}` — P(win) {f['p_win']:.3f}")
            embed.add_field(name=f"🚫 Filtered ({len(filtered)})", value="\n".join(lines), inline=False)
        else:
            embed.add_field(name="🚫 Filtered (0)", value="_none_", inline=False)

        # Origin levels
        res_line = f"🔴 `{o_res[0]['price']:.5f}`  {o_res[0]['dist_pct']:.2f}% above" if o_res else "🔴 none"
        sup_line = f"🟢 `{o_sup[0]['price']:.5f}`  {o_sup[0]['dist_pct']:.2f}% below" if o_sup else "🟢 none"
        embed.add_field(
            name="🎯 Nearest Origin Levels",
            value=f"{res_line}\n{sup_line}\n_/levels for full list_",
            inline=False,
        )

        return embed

    @client.event
    async def on_ready():
        import asyncio
        print(f"Discord bot online: {client.user}  ({len(client.guilds)} server(s))")
        for guild in client.guilds:
            try:
                tree.copy_global_to(guild=guild)
                cmds = await asyncio.wait_for(tree.sync(guild=guild), timeout=60)
                print(f"  Slash commands registered: {[c.name for c in cmds]}  → {guild.name}")
            except asyncio.TimeoutError:
                print(f"  Sync timed out for {guild.name} — commands will still work shortly")
            except Exception as e:
                print(f"  Sync error: {e!r}")

    async def _respond(interaction: discord.Interaction, embed_fn, *args):
        try:
            await interaction.response.defer()
        except (discord.errors.NotFound, discord.errors.InteractionResponded):
            return  # interaction token expired (>3s) or already responded
        state = _load_state()
        if not state:
            await interaction.followup.send("⚠️ No scanner state — is `live_signals.py` running?")
            return
        syms   = list(state.get("symbols", {}).keys()) or list(SYMBOLS.keys())
        embeds = [embed_fn(state, sym, *args) for sym in syms]
        await interaction.followup.send(embeds=embeds[:10])

    @tree.command(name="overview", description="Full dashboard — pending, active trades, results for all symbols")
    async def cmd_overview(interaction: discord.Interaction):
        await _respond(interaction, _overview_embed)

    @tree.command(name="status", description="Full dashboard (alias for /overview)")
    async def cmd_status(interaction: discord.Interaction):
        await _respond(interaction, _overview_embed)

    @tree.command(name="pending", description="Pending setups waiting for retest")
    async def cmd_pending(interaction: discord.Interaction):
        await _respond(interaction, _pending_embed)

    @tree.command(name="active", description="Currently active (open) trades")
    async def cmd_active(interaction: discord.Interaction):
        await _respond(interaction, _active_embed)

    @tree.command(name="results", description="Recent won / stopped trades")
    async def cmd_results(interaction: discord.Interaction):
        await _respond(interaction, _results_embed)

    @tree.command(name="filtered", description="Setups that were retested but rejected by the model")
    async def cmd_filtered(interaction: discord.Interaction):
        await _respond(interaction, _filtered_embed)

    @tree.command(name="cancelled", description="Pending setups that expired without activating")
    async def cmd_cancelled(interaction: discord.Interaction):
        await _respond(interaction, _cancelled_embed)

    @tree.command(name="levels", description="Active origin support and resistance levels")
    async def cmd_levels(interaction: discord.Interaction):
        await _respond(interaction, _levels_embed)

    import threading
    threading.Thread(target=client.run, args=(BOT_TOKEN,), daemon=True).start()


if __name__ == "__main__":
    try:
        import ccxt  # noqa: F401
    except ImportError:
        print("ccxt not installed.  Run:  pip install ccxt")
        raise SystemExit(1)
    start_discord_bot()
    main()
