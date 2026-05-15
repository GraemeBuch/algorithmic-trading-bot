#!/usr/bin/env python3
"""
Position Size Calculator
Usage: python position_calculator.py
"""

def calculate(account, risk_pct, entry, stop, tp=None, leverage=3):
    risk_dollar    = account * (risk_pct / 100)
    stop_dist      = abs(entry - stop)
    stop_dist_pct  = stop_dist / entry * 100
    position_size  = risk_dollar / stop_dist
    units          = position_size / entry
    margin_needed  = position_size / leverage

    print(f"\n{'═' * 45}")
    print(f"  ACCOUNT         ${account:>10,.2f}")
    print(f"  Risk            {risk_pct}%  =  ${risk_dollar:,.2f}")
    print(f"{'─' * 45}")
    print(f"  Entry           ${entry:>10.5f}")
    print(f"  Stop            ${stop:>10.5f}")
    print(f"  Stop distance   {stop_dist_pct:.3f}%  (${stop_dist:.5f})")
    print(f"{'─' * 45}")
    print(f"  Position size   ${position_size:>10,.2f}")
    print(f"  Units to buy    {units:>13,.2f}")
    print(f"  Leverage        {leverage}x")
    print(f"  Margin used     ${margin_needed:>10,.2f}  ({margin_needed/account*100:.1f}% of account)")
    print(f"  Margin free     ${account - margin_needed:>10,.2f}")

    if tp:
        tp_dist   = abs(tp - entry)
        profit    = tp_dist * units
        rr        = tp_dist / stop_dist
        print(f"{'─' * 45}")
        print(f"  Take Profit     ${tp:>10.5f}")
        print(f"  Profit if TP    ${profit:>10,.2f}  ({profit/account*100:.2f}% of account)")
        print(f"  Risk:Reward     1 : {rr:.2f}")

    print(f"  Max loss        ${risk_dollar:>10,.2f}  ({risk_pct}% of account)")
    print(f"{'═' * 45}\n")


def main():
    print("\n╔══════════════════════════════════════════╗")
    print("║      Highlander Position Calculator      ║")
    print("╚══════════════════════════════════════════╝")

    # Account
    acc_input = input("\nAccount size [$13000]: ").strip()
    account = float(acc_input) if acc_input else 13000.0

    # Risk
    risk_input = input("Risk % per trade [1]: ").strip()
    risk_pct = float(risk_input) if risk_input else 1.0

    # Leverage
    lev_input = input("Leverage [3]: ").strip()
    leverage = float(lev_input) if lev_input else 3.0

    while True:
        print("\n─── New Trade ───────────────────────────────")
        sym = input("Symbol (e.g. SUI, SOL) or q to quit: ").strip().upper()
        if sym == "Q":
            break

        entry = float(input(f"Entry price: "))
        stop  = float(input(f"Stop price:  "))

        tp_input = input(f"TP price (optional, press Enter to skip): ").strip()
        tp = float(tp_input) if tp_input else None

        calculate(account, risk_pct, entry, stop, tp=tp, leverage=leverage)

        # Update account for compounding
        update = input("Update account balance? (press Enter to skip): ").strip()
        if update:
            account = float(update)
            print(f"  Account updated to ${account:,.2f}")


if __name__ == "__main__":
    main()
