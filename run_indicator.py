"""Stage 1: run indicator on SOL 1H, save levels + events as pickle for stage 2."""
import pickle
import pandas as pd
from highlander import run_indicator

print("Loading SOL 1H data...")
sol = pd.read_csv("data/sol_1h.csv")
sol["time"] = pd.to_datetime(sol["time"])
sol = sol.set_index("time").sort_index()
sol = sol[["open", "high", "low", "close", "volume"]].astype(float)
print(f"  {len(sol):,} bars")

tick = max(sol["close"].iloc[0] * 1e-5, 1e-6)
print("Running indicator state machine...")
levels, events = run_indicator(sol[["open", "high", "low", "close"]],
                               min_range_ticks=3.0, tick_size=tick)
print(f"  {len(levels):,} levels, {len(events):,} events")

# Light-weight pickle: just core fields we need downstream
levels_dump = [{"id": l.id, "price": l.price, "dir": l.dir,
                "created_bar": l.created_bar, "origin_bar": l.origin_bar,
                "deleted_bar": l.deleted_bar, "confirmed": l.confirmed}
               for l in levels]
events_dump = [e.__dict__ for e in events]

with open("data/highlander_state.pkl", "wb") as f:
    pickle.dump({"levels": levels_dump, "events": events_dump}, f)

print("Saved → data/highlander_state.pkl")
