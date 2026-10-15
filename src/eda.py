"""First-pass exploratory data analysis of day-ahead prices (GR).

Usage:
    python src/eda.py --zone GR

Reads   data/processed/prices_<zone>.parquet
Writes  figures/eda_overview.png, figures/eda_negative_prices.png
Prints  yearly statistics and key patterns.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("figures")

NAVY = "#1F3B57"
ORANGE = "#D2691E"

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load(zone: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"prices_{zone}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found — run src/etl.py first.")
    return pd.read_parquet(path).set_index("timestamp_utc")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zone", default="GR")
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(exist_ok=True)
    df = load(args.zone)
    local = df["timestamp_local"]
    price = df["price_eur_mwh"]

    print(f"Rows: {len(df)}  |  {df.index.min()}  ->  {df.index.max()}")

    # ------------------------------------------------------------- statistics
    yearly = df.groupby(local.dt.year)["price_eur_mwh"].agg(
        mean="mean", median="median", min="min", max="max", hours="count"
    )
    yearly["pct_negative"] = (
        df.assign(year=local.dt.year)
        .groupby("year")["price_eur_mwh"]
        .apply(lambda s: round(100 * (s < 0).mean(), 2))
    )
    print("\n=== Yearly statistics (local time) ===")
    print(yearly.round(2).to_string())

    neg = df[price < 0]
    print(f"\nNegative prices: {len(neg)} hours ({100 * len(neg) / len(df):.1f}%)")
    print(f"Deepest low : {price.min():.1f} EUR/MWh on "
          f"{price.idxmin().strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"Highest peak: {price.max():.1f} EUR/MWh on "
          f"{price.idxmax().strftime('%Y-%m-%d %H:%M %Z')}")

    hp = df.groupby(local.dt.hour)["price_eur_mwh"].mean()
    print("\nMean price by hour of day (local):")
    print(hp.round(1).to_string())

    # ------------------------------------------------------------ overview fig
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Day-ahead electricity prices — {args.zone} bidding zone "
        f"({df.index.min():%b %Y} – {df.index.max():%b %Y})",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.bar(hp.index, hp.values, color=NAVY)
    ax.set_title("Mean price by hour of day (local)")
    ax.set_xlabel("hour")
    ax.set_ylabel("EUR/MWh")
    ax.set_xticks(range(0, 24, 2))

    ax = axes[0, 1]
    dp = df.groupby(local.dt.dayofweek)["price_eur_mwh"].mean()
    ax.bar(range(7), dp.values, color=NAVY)
    ax.set_xticks(range(7), DAYS)
    ax.set_title("Mean price by day of week")
    ax.set_ylabel("EUR/MWh")

    ax = axes[1, 0]
    monthly = price.resample("ME").mean()
    ax.plot(monthly.index, monthly.values, color=NAVY, lw=1.5)
    ax.set_title("Monthly mean price")
    ax.set_ylabel("EUR/MWh")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    ax.hist(price, bins=80, color=NAVY, alpha=0.85)
    ax.axvline(0, color=ORANGE, lw=1.2, ls="--")
    ax.set_title("Price distribution (hourly)")
    ax.set_xlabel("EUR/MWh")
    ax.set_ylabel("hours")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out1 = FIG_DIR / "eda_overview.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out1}")

    # -------------------------------------------------- negative prices figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Negative day-ahead prices", fontsize=13)

    ax = axes[0]
    ny = neg.groupby(neg["timestamp_local"].dt.year).size()
    ax.bar(ny.index.astype(str), ny.values, color=ORANGE)
    ax.set_title("count by year")
    ax.set_ylabel("hours")

    ax = axes[1]
    nh = (neg.groupby(neg["timestamp_local"].dt.hour).size()
          .reindex(range(24), fill_value=0))
    ax.bar(nh.index, nh.values, color=ORANGE)
    ax.set_title("count by hour of day (local)")
    ax.set_xlabel("hour")
    ax.set_xticks(range(0, 24, 2))

    ax = axes[2]
    nm = (neg.groupby(neg["timestamp_local"].dt.month).size()
          .reindex(range(1, 13), fill_value=0))
    ax.bar(nm.index, nm.values, color=ORANGE)
    ax.set_title("count by month")
    ax.set_xlabel("month")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = FIG_DIR / "eda_negative_prices.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
