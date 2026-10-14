"""Build the modelling dataset: calendar + lag features.

Usage:
    python src/features.py --zone GR

Reads   data/processed/prices_<zone>.parquet
Writes  data/processed/features_<zone>.parquet

Rows with missing data (warm-up period, missing hours) are dropped at the end.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")

LAGS = (24, 48, 168)        # hours: 1 day, 2 days, 1 week
ROLL_WINDOWS = (24, 168)    # hours


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    local = out["timestamp_local"]

    # Calendar features (local time), raw + cyclic encodings
    out["hour"] = local.dt.hour
    out["dayofweek"] = local.dt.dayofweek
    out["month"] = local.dt.month
    out["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    # Lag features
    for lag in LAGS:
        out[f"lag_{lag}h"] = out["price_eur_mwh"].shift(lag)

    # Rolling means, shifted 24 h to avoid leakage
    for win in ROLL_WINDOWS:
        out[f"roll_mean_{win}h"] = out["price_eur_mwh"].shift(24).rolling(win).mean()

    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zone", default="GR")
    args = parser.parse_args(argv)

    df = pd.read_parquet(PROCESSED_DIR / f"prices_{args.zone}.parquet")
    feat = build_features(df)

    before = len(feat)
    feat = feat.dropna().reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"features_{args.zone}.parquet"
    feat.to_parquet(out, index=False)

    feature_cols = [c for c in feat.columns
                    if c not in ("timestamp_utc", "timestamp_local", "price_eur_mwh")]
    print(f"Saved {out}")
    print(f"  rows kept: {len(feat)} of {before} (dropped warm-up / missing rows)")
    print(f"  features : {feature_cols}")


if __name__ == "__main__":
    main()
