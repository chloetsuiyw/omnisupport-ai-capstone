from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
total = 0
for f in sorted(DATA_DIR.glob("support_records_part_*.parquet")):
    df = pd.read_parquet(f, columns=["ticket_id"])
    print(f.name, len(df))
    total += len(df)

print("TOTAL:", total)
