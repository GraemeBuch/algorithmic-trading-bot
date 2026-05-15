#!/usr/bin/env python3
"""
discord_status.py — post the current scanner state to Discord on demand.

Usage:
    python discord_status.py          # posts pending setups + latest signal
    python discord_status.py signal   # latest signal only
    python discord_status.py pending  # pending setups only

Reads data/scanner_state.json written by live_signals.py each bar.
"""
import json, sys, os, pathlib

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
STATE   = pathlib.Path("data/scanner_state.json")

mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"


def post(embeds: list) -> None:
    if not WEBHOOK:
        print("DISCORD_WEBHOOK not set in .env")
        sys.exit(1)
    r = requests.post(WEBHOOK, json={"embeds": embeds}, timeout=10)
    r.raise_for_status()
    print(f"Sent {len(embeds)} embed(s) — HTTP {r.status_code}")


def pending_embed(state: dict) -> dict:
    rows = state.get("pending", [])
    if not rows:
        desc = "_No pending setups right now._"
    else:
        lines = []
        for s in rows:
            emoji = "📈" if s["direction"] == "Long" else "📉"
            lines.append(
                f"{emoji} **{s['direction']}**  |  Entry `{s['entry']:.5f}`"
                f"  |  Stop `{s['stop']:.5f}`  |  Expires in {s['bars_left']} bars\n"
                f"　 TP `{s['f2618']:.5f}`  |  Engulf bar: `{s['eng_time']}`"
            )
        desc = "\n\n".join(lines)

    return {
        "title":       f"📋  Pending Setups  ({len(rows)})  —  SOL/USDT 1H",
        "description": desc,
        "color":       0x5588FF,
        "footer":      {"text": f"Last scan: {state.get('last_scan', '?')}  |  "
                                f"SOL ${state.get('sol_price', 0):.4f}  |  "
                                f"HTF: {state.get('htf_trend', '?')}  |  "
                                f"BTC: {state.get('btc_trend', '?')}  |  "
                                f"Session: {state.get('session', '?')}"},
    }


def signal_embed(sig: dict | None) -> dict:
    if not sig:
        return {
            "title":       "🔍  Latest Signal  —  SOL/USDT 1H",
            "description": "_No signal fired yet this session._",
            "color":       0x888888,
        }
    direction = sig["direction"]
    color     = 0x00CC44 if direction == "Long" else 0xFF3333
    emoji     = "🟢" if direction == "Long" else "🔴"
    risk      = float(sig["risk"])
    entry     = float(sig["entry"])
    risk_pct  = risk / entry * 100
    return {
        "title":  f"{emoji} {direction.upper()}  SOL/USDT 1H  |  P(win) = {float(sig['p_win']):.3f}",
        "color":  color,
        "fields": [
            {"name": "Entry  (Fib 1.0)",       "value": f"`{float(sig['entry']):.5f}`",  "inline": True},
            {"name": "Stop   (Fib −0.5)",       "value": f"`{float(sig['stop']):.5f}`",   "inline": True},
            {"name": "Risk",                    "value": f"`{risk:.5f}  ({risk_pct:.2f}%)`", "inline": True},
            {"name": "BE level  (Fib 1.618)",   "value": f"`{float(sig['f1618']):.5f}`", "inline": True},
            {"name": "Take Profit  (Fib 2.618)","value": f"`{float(sig['f2618']):.5f}`", "inline": True},
            {"name": "Session",                 "value": sig.get("session", "?"),         "inline": True},
            {"name": "SOL HTF",                 "value": sig.get("htf", "?"),             "inline": True},
            {"name": "BTC Trend",               "value": sig.get("btc", "?"),             "inline": True},
            {"name": "HTF Aligned",             "value": "✅ Yes" if sig.get("htf_align") else "❌ No", "inline": True},
            {"name": "Confluence",              "value": str(sig.get("conf_score", "?")), "inline": True},
            {"name": "Engulfing bar",           "value": str(sig.get("eng_time", "?")),   "inline": True},
            {"name": "Bars to activation",      "value": str(sig.get("bars_ago", "?")),   "inline": True},
        ],
        "footer": {"text": f"Activated: {sig.get('ts', '?')}"},
    }


if not STATE.exists():
    print("No state file found — is live_signals.py running?")
    sys.exit(1)

state = json.loads(STATE.read_text())
embeds = []

if mode in ("all", "pending"):
    embeds.append(pending_embed(state))
if mode in ("all", "signal"):
    embeds.append(signal_embed(state.get("latest_signal")))

post(embeds)
