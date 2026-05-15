import pandas as pd
import sqlite3

df = pd.read_csv("trade_journal.csv")
conn = sqlite3.connect("trades.db")
df.to_sql("trade_journal", conn, if_exists="replace", index=False)
conn.close()