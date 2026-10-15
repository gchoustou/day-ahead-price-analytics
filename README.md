# Day-Ahead Electricity Price Analytics — Greek Bidding Zone

Analysis and forecasting of day-ahead electricity prices from the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu), for the
Greek bidding zone (GR).

## Overview

- **Data**: hourly day-ahead prices (document type `A44`), downloaded with the
  [`entsoe-py`](https://github.com/EnergieID/entsoe-py) client.
- **Pipeline**: monthly API downloads (Parquet) -> merge & clean -> EDA ->
  feature engineering -> forecasting models (naive baseline vs Ridge vs
  Gradient Boosting) -> evaluation and plots.
- **Purpose**: end-to-end portfolio project — data ingestion from a public API,
  storage in Parquet, analysis with pandas, modelling with scikit-learn, and
  version control with Git/GitHub. Non-commercial.

## Repository structure

```
day-ahead-price-analytics/
├── data/                  # local data (git-ignored)
│   ├── raw/               # monthly downloaded price files (Parquet)
│   └── processed/         # merged, cleaned price series
├── notebooks/
│   └── 01_eda.ipynb       # exploratory data analysis
├── src/
│   ├── fetch_entsoe.py    # download day-ahead prices via entsoe-py
│   ├── etl.py             # merge + clean -> single Parquet
│   ├── features.py        # calendar + lag features
│   └── train.py           # model training + evaluation
├── figures/               # generated plots
├── requirements.txt
└── .env.example
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then paste your API token into .env
```

A free API token is provided by ENTSO-E on request (Transparency Platform ->
Web API).

## Usage

```bash
# 1) Download day-ahead prices (Greek zone) for a period
python src/fetch_entsoe.py --start 2023-01-01 --end 2026-10-01 --zone GR

# 2) Merge + clean into one hourly series
python src/etl.py --zone GR

# 3) Build the modelling dataset (calendar + lag features)
python src/features.py --zone GR

# 4) Train + evaluate models
python src/train.py --zone GR
```

## Results

_(to be filled in as the analysis progresses — metrics table, predictions plot,
key insights on hourly / weekly / seasonal patterns and negative prices.)_

## Notes

- All timestamps kept in UTC, with a local (Europe/Athens) index alongside.
- Missing hours are made explicit during the ETL step (no silent gaps).
- From 1 Oct 2025 ENTSO-E publishes day-ahead prices at 15-minute resolution
  (SDAC 15-min MTU); `etl.py` aggregates to hourly means for one consistent
  series across the whole history.
