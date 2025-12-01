# Data Requirements

Please download the following datasets and place them in this directory (`data/`).

## 1. Aggregate Time Series (FRED)

**Source**: [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/)

Please download the following series (quarterly frequency, 1960-present):

1.  **Real Gross Domestic Product (GDPC1)**
    - Link: https://fred.stlouisfed.org/series/GDPC1
    - File name: `GDPC1.csv`
2.  **Personal Consumption Expenditures: Chain-type Price Index (PCEPI)**
    - Link: https://fred.stlouisfed.org/series/PCEPI
    - File name: `PCEPI.csv`
3.  **Effective Federal Funds Rate (FEDFUNDS)**
    - Link: https://fred.stlouisfed.org/series/FEDFUNDS
    - Note: Convert to quarterly average if downloading monthly.
    - File name: `FEDFUNDS.csv`

## 2. Distributional Moments (SCF)

**Source**: [Survey of Consumer Finances (SCF)](https://www.federalreserve.gov/econres/scfindex.htm)

We need the following moments (calculated from SCF microdata):

1.  **Liquid Wealth Share**: Share of total wealth held in liquid assets.
2.  **Gini Coefficient (Liquid Wealth)**
3.  **Gini Coefficient (Illiquid Wealth)**
4.  **Hand-to-Mouth Share**: Fraction of households with near-zero liquid wealth.

_If you have a pre-processed file with these moments, please save it as `moments.json` or `moments.csv`._

## Expected Directory Structure

```
data/
  GDPC1.csv
  PCEPI.csv
  FEDFUNDS.csv
  moments.json (optional, or we will use calibrated values)
```
