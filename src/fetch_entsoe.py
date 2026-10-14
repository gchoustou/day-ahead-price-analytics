"""Download day-ahead electricity prices from the ENTSO-E Transparency Platform.

The Transparency Platform REST API (https://web-api.tp.entsoe.eu) exposes
market data as XML. This script downloads day-ahead prices (document type
A44) for a bidding zone and a date range, stores the raw XML, and parses it
into monthly Parquet files.

Usage
-----
    python src/fetch_entsoe.py --start 2023-01-01 --end 2026-10-01 --zone GR

The API token is read from the ENTSOE_API_TOKEN environment variable
(copy .env.example to .env and paste your token), or can be passed with
--token.

Output
------
    data/raw/xml/<zone>_<yyyymm>.xml           raw API response
    data/raw/prices_<zone>_<yyyymm>.parquet    parsed hourly prices
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv optional at runtime
    pass

API_URL = "https://web-api.tp.entsoe.eu/api"
DOC_TYPE_DAY_AHEAD = "A44"

# EIC codes of bidding zones (Area Codes on the Transparency Platform)
ZONES = {
    "GR": "10YGR-HTSO-----Y",
    "IT_NORD": "10Y1001A1001A73I",
    "FR": "10YFR-RTE------C",
    "DE_LU": "10Y1001A1001A82H",
    "ES": "10YES-REE------0",
}

RAW_DIR = Path("data/raw")
XML_DIR = RAW_DIR / "xml"


def _local(tag: str) -> str:
    """Return the local name of an XML tag ({namespace}name -> name)."""
    return tag.rsplit("}", 1)[-1]


def _children(parent: ET.Element, name: str) -> list[ET.Element]:
    """Children of *parent* whose local tag name equals *name*."""
    return [c for c in parent if _local(c.tag) == name]


def _child(parent: ET.Element, name: str) -> ET.Element | None:
    found = _children(parent, name)
    return found[0] if found else None


def _text(parent: ET.Element, name: str) -> str | None:
    elem = _child(parent, name)
    return elem.text if elem is not None else None


def month_chunks(start: date, end: date):
    """Yield (first, last) day pairs covering [start, end], month by month."""
    cur = start
    while cur <= end:
        nxt = (
            date(cur.year + 1, 1, 1)
            if cur.month == 12
            else date(cur.year, cur.month + 1, 1)
        )
        yield cur, min(end, nxt - timedelta(days=1))
        cur = nxt


def fetch_month(token: str, zone_eic: str, start: date, end: date) -> bytes:
    """GET one month of day-ahead prices (validated XML bytes) from the API."""
    start_utc = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_utc = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
    params = {
        "securityToken": token,
        "documentType": DOC_TYPE_DAY_AHEAD,
        "in_Domain": zone_eic,
        "out_Domain": zone_eic,
        "periodStart": start_utc.strftime("%Y%m%d%H%M"),
        "periodEnd": end_utc.strftime("%Y%m%d%H%M"),
    }
    resp = requests.get(API_URL, params=params, timeout=90)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    if _local(root.tag) != "Publication_MarketDocument":
        reason = _text(root, "text") or root.tag
        raise RuntimeError(f"ENTSO-E API returned '{_local(root.tag)}': {reason}")
    return resp.content


def parse_prices(xml_bytes: bytes) -> pd.DataFrame:
    """Parse an A44 Publication_MarketDocument into an hourly DataFrame."""
    root = ET.fromstring(xml_bytes)
    rows: list[tuple[datetime, float]] = []
    for series in _children(root, "TimeSeries"):
        for period in _children(series, "Period"):
            interval = _child(period, "timeInterval")
            start_txt = _text(interval, "start") if interval is not None else None
            if not start_txt:
                continue
            period_start = datetime.fromisoformat(start_txt.replace("Z", "+00:00"))
            for point in _children(period, "Point"):
                position = int(_text(point, "position"))
                amount = float(_text(point, "price.amount"))
                rows.append((period_start + timedelta(hours=position - 1), amount))

    df = pd.DataFrame(rows, columns=["timestamp_utc", "price_eur_mwh"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.drop_duplicates("timestamp_utc", keep="last").sort_values("timestamp_utc")
    return df.reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", required=True, type=date.fromisoformat,
                        help="first day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=date.fromisoformat,
                        help="last day, YYYY-MM-DD")
    parser.add_argument("--zone", default="GR", choices=sorted(ZONES),
                        help="bidding zone code")
    parser.add_argument("--token", default=None,
                        help="API token (default: ENTSOE_API_TOKEN env var)")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="pause between API calls, seconds")
    args = parser.parse_args(argv)

    token = args.token or os.getenv("ENTSOE_API_TOKEN")
    if not token:
        sys.exit("ERROR: no API token. Set ENTSOE_API_TOKEN in .env or pass --token.")

    zone_eic = ZONES[args.zone]
    XML_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for start, end in month_chunks(args.start, args.end):
        tag = f"{args.zone}_{start:%Y%m}"
        parquet_path = RAW_DIR / f"prices_{tag}.parquet"
        if parquet_path.exists():
            print(f"[skip] {tag}: already downloaded")
            continue
        xml_bytes = fetch_month(token, zone_eic, start, end)
        (XML_DIR / f"{tag}.xml").write_bytes(xml_bytes)
        df = parse_prices(xml_bytes)
        df.to_parquet(parquet_path, index=False)
        total += len(df)
        print(f"[ok] {tag}: {len(df)} hourly points ({start} -> {end})")
        time.sleep(args.sleep)

    print(f"Done. {total} new rows written under {RAW_DIR}.")


if __name__ == "__main__":
    main()
