# Origin First Touch — Live Algorithmic Trading System

A fully automated cryptocurrency trading system built from scratch, running live on Bitget perpetual futures across 14 symbols. Combines a custom price action indicator with a CatBoost ML classifier, validated using rigorous walk-forward out-of-sample testing.

---

## Technical Overview

This project covers the full pipeline from research to live deployment:

### Machine Learning & Validation
- **Walk-forward cross-validation** — CatBoost classifier trained on rolling 6-month windows, tested on the subsequent 3-month out-of-sample period, expanding from 2019 to present. This is the same validation methodology used in quantitative finance to prevent lookahead bias and overfitting.
- **54 engineered features** — price action structure, support/resistance proximity (1H and 4H timeframes), ATR volatility regime, higher-timeframe trend alignment, Fibonacci extension levels, time-of-session, BTC market context, and engulfing bar characteristics.
- **1-minute resolution outcome labelling** — rather than approximating trade outcomes at bar closes, all stop-loss and take-profit hits are resolved to the exact minute using 1-minute OHLC data. This produces realistic win/loss labels that account for intrabar wicks and avoids the bar-close approximation error common in retail backtests.

### Live-Backtest Alignment (Training-Serving Parity)
A core engineering challenge in live ML trading is ensuring the model sees identical features at inference time to what it was trained on. This system addresses it in several ways:
- **Full history startup cache** — at launch, the bot runs the Highlander indicator over the complete 7-year 1H price history from CSV and caches all support/resistance levels with nanosecond timestamps. Live signals then look up levels against this cache using binary search, matching the exact same level history used during training.
- **Paginated API fetch** — Bitget caps at 1000 bars per request. The bot fetches 2000 bars (83 days) in pages and deduplicates to ensure the level calculation window matches the backtest.
- **Consistent feature computation** — all 54 features are computed identically in the backtest and live scanner, including string encoding for categorical features (HTF trend, BTC trend, session), percentage scaling, and fill values for edge cases.

### Execution
- **Limit order entry with 2-minute fill timeout** — avoids market order slippage. If unfilled within 2 minutes, the order is cancelled and the signal is discarded.
- **Intrabar monitoring** — between hourly bar closes, the bot polls at 60-second intervals to catch stop-loss, take-profit, and break-even triggers in real time without waiting for the next bar.
- **Break-even stop management** — when price reaches the Fibonacci 1.618 extension, the stop is moved to exact entry price automatically.
- **Portfolio constraint** — maximum 5 concurrent positions across all 14 symbols.

### Infrastructure
- Python 3.11, CatBoost, pandas, ccxt (Bitget API)
- Runs as a persistent nohup process on a Linux VPS
- Discord bot integration for slash command dashboard (`/overview`, `/active`, `/results`, `/levels`) and webhook trade notifications
- Trade journal saved to JSON state file for audit trail

### Results (Walk-Forward OOS, 1-minute resolution)
14 symbols backtested. Best performers (profit factor / R per month):

| Symbol | Profit Factor | R/month |
|--------|--------------|---------|
| DOGE | 8.97 | +1.86R |
| UNI | 9.40 | +1.99R |
| ADA | 8.89 | +2.88R |
| XRP | 7.06 | +2.57R |
| LTC | 7.20 | +2.30R |
| LINK | 8.33 | +2.34R |

Last 30-day live out-of-sample: **88% win / 12% break-even / 0% full stop, +29.96R across 33 trades**.

---

## Live Bot

**Server:** `root@178.104.81.50`  
**Bot path:** `/root/bot/`  
**Log:** `/root/bot/bot.log`  
**Script:** `live_signals.py`

### Active symbols (14)
BTC, ETH, SOL, TRX, LINK, HBAR, XRP, DOGE, LTC, ADA, XLM, UNI, APT, NEAR

### How the bot works
1. Polls Bitget every 60 seconds
2. Runs Highlander indicator on 2000 bars of 1H data per symbol (~83 days)
3. Detects engulfing candles at Origin First Touch zones
4. Scores each signal with a CatBoost model (54 features, P ≥ 0.50 threshold)
5. Enters with a **limit order** at the signal close price; cancels if not filled within 2 minutes
6. Sets stop-loss (1.4×ATR), take-profit (1.039R), and break-even trigger (Fib 1.618)
7. Break-even stop moves to exact entry price when price hits f1618
8. Max 5 concurrent positions across all symbols

---

## Deploying an Update

> **Do this only when no positions are open** (check Bitget first).

### Full deploy (new models + code)

```bash
# 1. Copy the bot script
scp live_signals.py root@178.104.81.50:/root/bot/

# 2. Copy all 14 CatBoost models
scp data/catboost_clf_btcusdt_1h.cbm \
    data/catboost_clf_ethusdt_1h.cbm \
    data/catboost_clf_solusdt_1h.cbm \
    data/catboost_clf_trxusdt_1h.cbm \
    data/catboost_clf_linkusdt_1h.cbm \
    data/catboost_clf_hbarusdt_1h.cbm \
    data/catboost_clf_xrpusdt_1h.cbm \
    root@178.104.81.50:/root/bot/data/

scp data/catboost_clf_dogeusdt_1h.cbm \
    data/catboost_clf_ltcusdt_1h.cbm \
    data/catboost_clf_adausdt_1h.cbm \
    data/catboost_clf_xlmusdt_1h.cbm \
    data/catboost_clf_uniusdt_1h.cbm \
    data/catboost_clf_aptusdt_1h.cbm \
    data/catboost_clf_nearusdt_1h.cbm \
    root@178.104.81.50:/root/bot/data/

# 3. IMPORTANT — copy 1H CSVs (needed for startup level cache)
scp data/btcusdt_1h.csv data/ethusdt_1h.csv data/solusdt_1h.csv \
    data/trxusdt_1h.csv data/linkusdt_1h.csv data/hbarusdt_1h.csv \
    data/xrpusdt_1h.csv data/dogeusdt_1h.csv data/ltcusdt_1h.csv \
    data/adausdt_1h.csv data/xlmusdt_1h.csv data/uniusdt_1h.csv \
    data/aptusdt_1h.csv data/nearusdt_1h.csv \
    root@178.104.81.50:/root/bot/data/

# 4. Kill old bot, start new one
ssh root@178.104.81.50 "pkill -f live_signals.py; sleep 2 && cd /root/bot && nohup python3 -u live_signals.py >> bot.log 2>&1 &"

# 5. Check it started correctly (wait ~10 seconds first)
ssh root@178.104.81.50 "tail -30 /root/bot/bot.log"
```

> **Why copy the CSVs?** The bot pre-computes Highlander support/resistance levels from the full price history at startup (~2-3 minutes). If you upload new `.cbm` models but leave stale CSVs, the feature calculation diverges from what the model was trained on. Always copy both together.

### Code-only update (no model change)

```bash
scp live_signals.py root@178.104.81.50:/root/bot/
ssh root@178.104.81.50 "pkill -f live_signals.py; sleep 2 && cd /root/bot && nohup python3 -u live_signals.py >> bot.log 2>&1 &"
ssh root@178.104.81.50 "tail -30 /root/bot/bot.log"
```

### Check the bot is running

```bash
ssh root@178.104.81.50 "pgrep -a python3 | grep live_signals"
ssh root@178.104.81.50 "tail -50 /root/bot/bot.log"
```

---

## Retraining Models

Run on local machine (takes 2-4 hours):

```bash
# Full 1-min walk-forward backtest + retrain all 14 symbols
python multi_1min_backtest.py

# Quick last-30-day OOS check (uses saved .cbm models, ~5-15 min)
python recent_stats.py
```

After retraining, deploy with the full deploy sequence above (code + models + CSVs).

---

## Key Design Decisions

| What | Value | Why |
|------|-------|-----|
| LOOKBACK_1H | 2000 bars (~83 days) | Matches level history used in backtest |
| Entry | Limit order, 2-min timeout | Avoids market order slippage |
| BE stop | Exact entry price | No buffer — matches backtest definition |
| ML threshold | P ≥ 0.50 | Walk-forward validated |
| MAX_CONCURRENT | 5 | Portfolio risk cap |
| Startup cache | Full 7-year CSV | Eliminates feature skew vs backtest |

### Live vs backtest alignment

The bot and backtest now use the same feature calculation logic. Remaining differences:

- **Data source**: live bot fetches recent bars from Bitget API; CSV data may lag by hours
- **Startup cache staleness**: CSV data on server reflects the date of last deploy
- **1H vs 1-min timing**: backtest resolves outcomes to the minute; live bot uses 1H close prices for entry timing

These are expected and small. The major historical divergence (short Highlander lookback + market order fills) has been fixed.

---

## Backtest Results (walk-forward OOS, 1-min resolution)

| Symbol | PF | R/mo | OOS Windows |
|--------|-----|------|-------------|
| BTC | — | — | new |
| ETH | — | — | new |
| ADA | 8.89 | +2.88R | 28/28 |
| LINK | 8.33 | +2.34R | 28/28 |
| DOGE | 8.97 | +1.86R | 25/26 |
| UNI | 9.40 | +1.99R | 21/21 |
| XRP | 7.06 | +2.57R | 28/28 |
| LTC | 7.20 | +2.30R | 28/28 |
| TRX | 6.62 | +1.92R | 27/28 |
| HBAR | 6.51 | +1.16R | 22/25 |
| APT | 5.28 | +2.75R | 12/13 |
| SOL | 5.21 | +2.56R | 15/15 |
| XLM | 4.80 | +2.11R | 28/28 |
| NEAR | — | — | new |

Last 30-day live OOS (Apr 13 – May 13 2026): **88% win / 12% BE / 0% stop, +29.96R across 33 trades, 13 symbols**.

---

## Known Issues / Pending

- **SOL 1-min data** (`solusdt_1m_bitget.csv`) ends October 2025 — needs updating before next retrain
- **AVAX**: removed from live symbols (no 1-min data, performing poorly)
- Run `python recent_stats.py` every few weeks to check OOS performance hasn't degraded
