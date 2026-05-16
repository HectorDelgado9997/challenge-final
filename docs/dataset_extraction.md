# Dataset Extraction

## Objective

The objective of the dataset extraction stage is to build a reproducible financial dataset using historical price data downloaded from `yfinance`.

The output of this stage is a monthly price dataset that will later be transformed into a machine learning dataset.

## Data Source

The main data source is:

```text
yfinance

The project downloads historical daily adjusted close prices and converts them into monthly prices using the last available trading price of each month.

Asset Universe

The asset universe is defined in:

data/raw/asset_details.csv

The file contains two columns:

asset
domain

Example:

asset,domain
SPY,US Equity ETF
QQQ,US Equity ETF
BTC-USD,Crypto
XLE,Sector ETF
GRID,Thematic ETF
FLKR,International ETF
GLD,Commodity ETF
EWW,Mexico ETF
EWZ,Brazil ETF
Design Decision

The original project requirement refers to an input containing an attribute and a domain. In this implementation:

asset  = financial ticker
domain = financial asset category

This design keeps the project aligned with the requirement while making the financial dataset explicit and easy to validate.

Extraction Process

The extraction process is implemented in:

src/data/extract_data.py

The executable script is:

scripts/01_extract_data.py

The process follows these steps:

1. Load data/raw/asset_details.csv
2. Validate asset and domain columns
3. Extract the list of tickers
4. Download daily adjusted close prices from yfinance
5. Convert daily prices to monthly prices
6. Save the monthly price dataset
Output File

The output file is:

data/raw/monthly_prices.csv

It contains:

date
asset
adjusted_close
Monthly Price Construction

Monthly prices are calculated using the last available adjusted close price for each month.

This avoids creating artificial prices and reflects the most recent available monthly valuation for each asset.

Data Validation

Validation is implemented in:

src/data/validate_data.py

The validation process checks:

File existence
Required columns
Empty datasets
Null assets
Duplicated assets
Invalid date ranges
Invalid asset lists
Default Configuration

The default extraction start date is defined in:

src/config/settings.py

Default value:

2015-01-01

The default end date is:

None

This allows yfinance to download data up to the latest available date.

How to Run

From the project root:

python scripts/01_extract_data.py
Expected Result

The script should generate:

data/raw/monthly_prices.csv

To validate the result:

python -c "import pandas as pd; df = pd.read_csv('data/raw/monthly_prices.csv'); print(df.head()); print(df.shape)"

SPY is used as the benchmark asset.
BTC is represented as BTC-USD because this is the valid yfinance ticker.
Only historical price data is used.
No external macroeconomic variables are included.
The project is designed for local execution.

