"""Download day-ahead electricity prices from the ENTSO-E Transparency Platform.

Downloads hourly day-ahead prices (document type A44) for one or more bidding
zones and a date range, using the `entsoe-py` client, and stores them as
monthly Parquet files under data/raw/.

Usage:
    python src/fetch_entsoe.py --start 2023-01-01 --end 2026-10-01 --zone GR

The ENTSO-E API token is read from the ENTSOE_API_TOKEN environment variable
(see .env.example). Existing monthly files are skipped, so the download can be
re-run safely to resume an interrupted range.

Outputs:
    data/raw/prices_<ZONE>_<YYYYMM>.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

try:
    from entsoe import EntsoePandasClient
except ImportError:  # pragma: no cover
    sys.exit("entsoe-py is required. Install it with: pip install -r requirements.txt")

DEFAULT_OUT_DIR = Path("data/raw")


def month_chunks(start: date, end: date):
    """Yield (chunk_start, chunk_end) date pairs covering [start, end] month by month."""
    cur = start
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        yield cur, min(end, nxt - timedelta(days=1))
        cur = nxt


def fetch(client: EntsoePandasClient, zone: str, start: date, end: date, out_dir: Path) -> None:
    """Download day-ahead prices month by month and save one Parquet file per month."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk_start, chunk_end in month_chunks(start, end):
        fname = out_dir / f"prices_{zone}_{chunk_start:%Y%m}.parquet"
        if fname.exists():
            print(f"[skip] {fname.name} (already downloaded)")
            continue

        ts_start = pd.Timestamp(chunk_start, tz="UTC")
        ts_end = pd.Timestamp(chunk_end + timedelta(days=1), tz="UTC")

        series = client.query_day_ahead_prices(zone, start=ts_start, end=ts_end)
        frame = pd.Series(series.values, index=series.index, name="price_eur_mwh").to_frame()
        frame.index.name = "timestamp_utc"
        frame.to_parquet(fname)

        print(
            f"[ok]   {fname.name}: {len(frame)} hours "
            f"({frame.index.min():%Y-%m-%d} -> {frame.index.max():%Y-%m-%d})"
        )


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="end date (YYYY-MM-DD)")
    ap.add_argument("--zone", default="GR", help="bidding zone code (e.g. GR) or EIC code")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output directory for Parquet files")
    ap.add_argument("--token", default=None, help="API token (default: ENTSOE_API_TOKEN env var)")
    args = ap.parse_args()

    token = args.token or os.getenv("ENTSOE_API_TOKEN")
    if not token:
        sys.exit("No API token found. Set ENTSOE_API_TOKEN in .env or pass --token.")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        sys.exit("--end must not be before --start.")

    client = EntsoePandasClient(api_key=token)
    fetch(client, args.zone, start, end, Path(args.out))


if __name__ == "__main__":
    main()
