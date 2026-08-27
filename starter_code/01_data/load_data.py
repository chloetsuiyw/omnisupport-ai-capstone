from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def iter_support_shards(columns=None):
    """Yield each primary Parquet shard as a pandas DataFrame.

    Pass `columns=[...]` during exploration to avoid reading unused columns.
    """
    for path in sorted(DATA_DIR.glob("support_records_part_*.parquet")):
        yield pd.read_parquet(path, columns=columns)


def load_preview():
    return pd.read_csv(DATA_DIR / "dataset_preview.csv")


if __name__ == "__main__":
    preview = load_preview()
    print(preview.shape)
    print(preview.head())
