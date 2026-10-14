"""Merge monthly parsed prices into one clean, continuous hourly series.

Usage:
    python src/etl.py --zone GR

Reads   data/raw/prices_<zone>_*.parquet
Writes  data/processed/prices_<zone>.parquet
        columns: timestamp_utc, timestamp_local, price_eur_mwh
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_raw(zone: str) -> pd.DataFrame:
    files = sorted(RAW_DIR.glob(f"prices_{zone}_*.parquet"))
    if not files:
        raise SystemExit(f"No raw files for zone {zone} — run src/fetch_entsoe.py first.")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.drop_duplicates("timestamp_utc", keep="last").sort_values("timestamp_utc")
    return df.reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete hourly grid so gaps become explicit NaN rows."""
    df = df.set_index("timestamp_utc")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full_index)
    missing = int(df["price_eur_mwh"].isna().sum())
    if missing:
        print(f"Warning: {missing} missing hours kept as NaN (interpolate later if needed)")
    df.index.name = "timestamp_utc"
    return df.reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zone", default="GR")
    args = parser.parse_args(argv)

    df = clean(load_raw(args.zone))
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert("Europe/Athens")
    df = df[["timestamp_utc", "timestamp_local", "price_eur_mwh"]]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"prices_{args.zone}.parquet"
    df.to_parquet(out, index=False)

    price = df["price_eur_mwh"]
    print(f"Saved {out}")
    print(f"  rows      : {len(df)}")
    print(f"  period    : {df['timestamp_utc'].min()} -> {df['timestamp_utc'].max()}")
    print(f"  missing   : {int(price.isna().sum())}")
    print(f"  price EUR/MWh: min {price.min():.2f} | mean {price.mean():.2f} | max {price.max():.2f}")
    print(f"  negative-price hours: {int((price < 0).sum())}")


if __name__ == "__main__":
    main()
