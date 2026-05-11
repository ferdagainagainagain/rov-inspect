#!/usr/bin/env python3
"""Inspect a Deep Trekker combined_log.csv before running the pipeline.

Use the output to fill COLUMN_MAP_DEFAULT in src/rov_inspect/sync.py.

Usage:
    python scripts/explore_logs.py "/path/to/VIDEO 1/data/combined_log.csv"
"""
import sys
from pathlib import Path
import pandas as pd


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    df = pd.read_csv(path)

    print(f"File:    {path}")
    print(f"Shape:   {df.shape[0]} rows × {df.shape[1]} cols\n")

    print("Columns:")
    for c in df.columns:
        sample = df[c].iloc[0] if len(df) > 0 else None
        print(f"  {c!r:35s}  dtype={str(df[c].dtype):8s}  first={sample!r}")

    print("\nHead (first 5 rows):")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.head())

    print("\nNumeric describe:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.describe())


if __name__ == "__main__":
    main()
