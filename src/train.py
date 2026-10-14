"""Train and evaluate day-ahead price forecasting models.

Usage:
    python src/train.py --zone GR

Reads   data/processed/features_<zone>.parquet
Writes  figures/predictions_<zone>.png
        figures/feature_importance_<zone>.png
        figures/metrics_<zone>.csv

Models: naive 24 h persistence | Ridge | Gradient boosting
Split : chronological — the last 20% of hours form the test set
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")

TARGET = "price_eur_mwh"
NON_FEATURES = ["timestamp_utc", "timestamp_local", TARGET]


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zone", default="GR")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args(argv)

    df = pd.read_parquet(PROCESSED_DIR / f"features_{args.zone}.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]

    split = int(len(df) * (1 - args.test_frac))
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    preds: dict[str, np.ndarray] = {}
    # 1) Naive baseline: yesterday's same-hour price
    preds["Naive (lag 24 h)"] = test["lag_24h"].to_numpy()
    # 2) Ridge regression
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    preds["Ridge"] = ridge.predict(X_test)
    # 3) Gradient boosting
    gbr = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=0,
    ).fit(X_train, y_train)
    preds["Gradient boosting"] = gbr.predict(X_test)

    table = pd.DataFrame({name: metrics(y_test, p) for name, p in preds.items()}).T
    table = table.sort_values("MAE")
    print(table.round(2).to_string())

    FIGURES_DIR.mkdir(exist_ok=True)
    table.round(3).to_csv(FIGURES_DIR / f"metrics_{args.zone}.csv")

    # Predictions plot — last two weeks of the test set
    window = 24 * 14
    t = test["timestamp_local"].iloc[-window:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, y_test.iloc[-window:], label="Actual", color="black", lw=1.6)
    for name, y_pred in preds.items():
        ax.plot(t, y_pred[-window:], label=name, lw=1.0, alpha=0.85)
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Day-ahead prices ({args.zone}) — predictions on last two test weeks")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"predictions_{args.zone}.png", dpi=150)
    print(f"Saved {FIGURES_DIR / ('predictions_' + args.zone + '.png')}")

    # Feature importances (gradient boosting, top 15)
    imp = pd.Series(gbr.feature_importances_, index=feature_cols).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#1F3B57")
    ax.set_title("Gradient boosting — feature importance (top 15)")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"feature_importance_{args.zone}.png", dpi=150)
    print(f"Saved {FIGURES_DIR / ('feature_importance_' + args.zone + '.png')}")


if __name__ == "__main__":
    main()
