# Dataset Extraction

## Overview

This project does not use a static dataset file. Instead, historical financial
data is downloaded dynamically from **Yahoo Finance** using the `yfinance` library.
The asset universe, extraction parameters, and output paths are all defined in
`src/config/settings.py`.

## Asset Universe

| Asset     | Type                        |
|-----------|-----------------------------|
| SPY       | US Large Cap Equity ETF     |
| QQQ       | US Technology Equity ETF    |
| BTC-USD   | Bitcoin / Cryptocurrency    |
| XLE       | US Energy Sector ETF        |
| GRID      | Clean Energy ETF            |
| FLKR      | Frontier Markets ETF        |
| GLD       | Gold ETF                    |
| EWW       | Mexico Equity ETF           |
| EWZ       | Brazil Equity ETF           |

Asset metadata is stored in:

```text
data/raw/asset_details.csv
```

Expected columns:

| Column   | Description               |
|----------|---------------------------|
| `asset`  | Yahoo Finance ticker      |
| `domain` | Asset category or sector  |

## Extraction Configuration

| Parameter              | Value                        |
|------------------------|------------------------------|
| Default start date     | `2015-01-01`                 |
| Default end date       | Latest available (None)      |
| yfinance interval      | `1d` (daily, resampled monthly) |
| Price column           | `adjusted_close`             |

## Extraction Process

Data extraction is handled by `src/data/extract_data.py` and triggered via:

```bash
python scripts/01_extract_data.py
```

Or through the full pipeline:

```bash
python scripts/05_run_full_pipeline.py --amount 10000 --start-date 2015-01-01
```

The extraction flow:yfinance API
│
▼
Download daily adjusted close prices for all assets
│
▼
Resample to monthly frequency (last trading day of month)
│
▼
data/raw/monthly_prices.csv

## Output File

```text
data/raw/monthly_prices.csv
```

This file contains monthly adjusted close prices with assets as columns
and dates as the index.

## Dataset Building

After extraction, the modeling dataset is built by `src/data/build_dataset.py`:

```bash
python scripts/02_build_dataset.py
```

This step computes all financial features from the raw monthly prices
and stores the result in:

```text
data/processed/model_dataset.csv
```

## Model Dataset Structure

| Column                       | Type    | Description                                       |
|------------------------------|---------|---------------------------------------------------|
| `date`                       | date    | Month-end date                                    |
| `asset`                      | str     | Asset ticker                                      |
| `return_1m`                  | float   | 1-month return                                    |
| `return_3m`                  | float   | 3-month return                                    |
| `return_6m`                  | float   | 6-month return                                    |
| `return_12m`                 | float   | 12-month return                                   |
| `volatility_3m`              | float   | Rolling 3-month volatility                        |
| `volatility_6m`              | float   | Rolling 6-month volatility                        |
| `drawdown_3m`                | float   | Rolling 3-month max drawdown                      |
| `drawdown_6m`                | float   | Rolling 6-month max drawdown                      |
| `sharpe_3m`                  | float   | Rolling 3-month Sharpe ratio                      |
| `sharpe_6m`                  | float   | Rolling 6-month Sharpe ratio                      |
| `relative_strength_3m`       | float   | Return relative to SPY over 3 months              |
| `relative_strength_6m`       | float   | Return relative to SPY over 6 months              |
| `target_outperform_next_month` | int  | 1 if asset outperforms SPY next month, else 0     |

## Benchmark Asset

The classification target is defined relative to the benchmark:

```python
BENCHMARK_ASSET = "SPY"
```

An asset is labeled `1` (outperform) if its next-month return exceeds SPY's
next-month return, and `0` otherwise.

## Regression Target

The regression target `y_regression` is defined as the next month's `return_1m`
for each asset. This target is **computed in memory** during training and is
**not persisted** in `model_dataset.csv`.
