#!/usr/bin/env python3
"""
discord_bot.py — Highlander signal bot (slash commands).

Commands:
    /status    — pending setups + latest signal
    /signal    — latest signal only
    /pending   — pending setups only

Reads data/scanner_state.json written by live_signals.py each bar.

Setup:
    pip install discord.py
    Add BOT_TOKEN=... to .env
    python discord_bot.py
"""
import json, os, pathlib
import discord
from discord import app_commands

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
STATE     = pathlib.Path("data/scanner_state.json")


def load_state() -> dict | None:
    if not STATE.exists():
        return None
    return json.loads(STATE.read_text())


def make_pending_embed(state: dict) -> discord.Embed:
    rows = state.get("pending", [])
    embed = discord.Embed(
        title=f"📋  Pending Setups  ({len(rows)})  —  SOL/USDT 1H",
        color=0x5588FF,
    )
    embed.set_footer(text=(
        f"Last scan: {state.get('last_scan', '?')}  |  "
        f"SOL ${state.get('sol_price', 0):.4f}  |  "
        f"HTF: {state.get('htf_trend', '?')}  |  "
        f"BTC: {state.get('btc_trend', '?')}  |  "
        f"Session: {state.get('session', '?')}"
    ))
    if not rows:
        embed.description = "_No pending setups right now._"
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


def make_signal_embed(sig: dict | None) -> discord.Embed:
    if not sig:
        return discord.Embed(
            title="🔍  Latest Signal  —  SOL/USDT 1H",
            description="_No signal fired yet this session._",
            color=0x888888,
        )
    direction = sig["direction"]
    color     = 0x00CC44 if direction == "Long" else 0xFF3333
    emoji     = "🟢" if direction == "Long" else "🔴"
    entry     = float(sig["entry"])
    risk      = float(sig["risk"])

    embed = discord.Embed(
        title=f"{emoji} {direction.upper()}  SOL/USDT 1H  |  P(win) = {float(sig['p_win']):.3f}",
        color=color,
    )
    embed.add_field(name="Entry  (Fib 1.0)",        value=f"`{entry:.5f}`",                        inline=True)
    embed.add_field(name="Stop   (Fib −0.5)",        value=f"`{float(sig['stop']):.5f}`",            inline=True)
    embed.add_field(name="Risk",                     value=f"`{risk:.5f}  ({risk/entry*100:.2f}%)`", inline=True)
    embed.add_field(name="BE level  (Fib 1.618)",    value=f"`{float(sig['f1618']):.5f}`",           inline=True)
    embed.add_field(name="Take Profit  (Fib 2.618)", value=f"`{float(sig['f2618']):.5f}`",           inline=True)
    embed.add_field(name="Session",                  value=sig.get("session", "?"),                  inline=True)
    embed.add_field(name="SOL HTF",                  value=sig.get("htf", "?"),                      inline=True)
    embed.add_field(name="BTC Trend",                value=sig.get("btc", "?"),                      inline=True)
    embed.add_field(name="HTF Aligned",              value="✅ Yes" if sig.get("htf_align") else "❌ No", inline=True)
    embed.add_field(name="Confluence",               value=str(sig.get("conf_score", "?")),          inline=True)
    embed.add_field(name="Engulfing bar",            value=str(sig.get("eng_time", "?")),            inline=True)
    embed.add_field(name="Bars to activation",       value=str(sig.get("bars_ago", "?")),            inline=True)
    embed.set_footer(text=f"Activated: {sig.get('ts', '?')}")
    return embed


# ── Bot setup ─────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
client  = discord.Client(intents=intents)
tree    = app_commands.CommandTree(client)


@client.event
async def on_ready():
    for guild in client.guilds:
        await tree.sync(guild=guild)
        print(f"  Synced to: {guild.name} ({guild.id})")
    print(f"Bot online: {client.user}  —  slash commands synced")


@tree.command(name="status", description="Pending setups + latest signal")
async def cmd_status(interaction: discord.Interaction):
    state = load_state()
    if not state:
        await interaction.response.send_message("⚠️ No scanner state — is `live_signals.py` running?")
        return
    await interaction.response.send_message(
        embeds=[make_pending_embed(state), make_signal_embed(state.get("latest_signal"))]
    )


@tree.command(name="signal", description="Latest signal only")
async def cmd_signal(interaction: discord.Interaction):
    state = load_state()
    if not state:
        await interaction.response.send_message("⚠️ No scanner state — is `live_signals.py` running?")
        return
    await interaction.response.send_message(embed=make_signal_embed(state.get("latest_signal")))


@tree.command(name="pending", description="All pending setups")
async def cmd_pending(interaction: discord.Interaction):
    state = load_state()
    if not state:
        await interaction.response.send_message("⚠️ No scanner state — is `live_signals.py` running?")
        return
    await interaction.response.send_message(embed=make_pending_embed(state))


if __name__ == "__main__":
    if not BOT_TOKEN:
        print("BOT_TOKEN not set — add it to .env")
        raise SystemExit(1)
    client.run(BOT_TOKEN)
