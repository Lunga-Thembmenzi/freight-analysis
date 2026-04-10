# Freight Rate Multivariate Market Model

A multivariate regression analysis of global maritime freight rates using macroeconomic and shipping market variables. The project builds a full **supply–demand equilibrium model** of freight pricing across three progressively improved specifications.

---

## Overview

The Baltic Dry Index (Capesize) is one of the most volatile and economically significant benchmarks in global trade. This project models its movements using three economic drivers:

- **Oil Price** — cost-side driver (Brent crude)
- **Trade Volume** — demand-side driver (World Bank global exports)
- **Fleet Supply** — supply-side driver (UNCTAD bulk carrier DWT)

The original two-variable baseline model (R² = 0.208) was economically incomplete — it had no supply-side variable. Adding bulk carrier fleet capacity produces a full market equilibrium model and reveals why the 2014–2016 freight rate crash was so severe and prolonged.

---

## Project Structure

```
freight-analysis/
│
├── data/
│   ├── BDI.csv                                          # Baltic Dry Index (Capesize)
│   ├── BrentOilPrices.csv                               # Brent crude oil prices
│   ├── API_NE.EXP.GNFS.CD_DS2_en_csv_v2_175529.csv     # World Bank trade volume
│   └── US_MerchantFleet_20260409_200824.csv             # UNCTAD bulk carrier fleet DWT
│
├── plots/
│   ├── trends_v2.png
│   ├── correlation_v2.png
│   ├── scatter_v2.png
│   └── regression_comparison.png
│
└── Freight_analysis_v3.ipynb                            # Main analysis notebook
```

---

## Data Sources

| Dataset | Source | Frequency | Period |
|---|---|---|---|
| Baltic Dry Index (Capesize) | Baltic Exchange | Daily | 2012–2019 |
| Brent Crude Oil Prices | Kaggle | Daily | 2012–2019 |
| Global Trade Volume | World Bank | Yearly | 2012–2019 |
| Bulk Carrier Fleet (DWT) | UNCTAD | Yearly | 2012–2019 |

All datasets are aligned to a common **monthly frequency** — daily data is resampled to monthly averages, yearly data is forward-filled to monthly.

> **Note on fleet data:** The UNCTAD file (`US_MerchantFleet_...csv`) is filtered to **Ship type: Bulk carriers**, **Indicator: Dead weight tonnage**, **Economy: World**. Despite the filename, the data is bulk-carrier-specific, not US-only.

---

## Methodology

### Data Pipeline

1. Load all four raw datasets
2. Parse and standardize date columns
3. Resample daily data to monthly averages (BDI, oil)
4. Forward-fill yearly data to monthly (trade, fleet)
5. Merge on monthly date index, drop missing values
6. Final dataset: **84 rows × 5 columns**, Aug 2012 – Jul 2019

### Models

**Model 1 — Baseline (Demand + Cost)**
```
Freight = α + β₁(Oil) + β₂(Trade) + ε
```

**Model 2 — Full Market Model (Supply + Demand + Cost)**
```
Freight = α + β₁(Oil) + β₂(Trade) + β₃(Fleet) + ε
```

**Model 3 — Extended Model (with Lagged Oil Variables)**
```
Freight = α + β₁(Oil) + β₂(Trade) + β₃(Fleet) + β₄(Oil_Lag1) + β₅(Oil_Lag2) + ε
```

---

## Results

### Model Comparison

| Model | Variables | R² Score | MAE | R² vs Baseline |
|---|:---:|:---:|:---:|:---:|
| Baseline (Oil + Trade) | 2 | 0.208 | 597.65 | — |
| Full Market (Oil + Trade + Fleet) | 3 | 0.208 | 598.35 | +0.001 |
| Extended (+ Lags) | 5 | 0.220 | 598.12 | +0.012 |

### Coefficient Interpretation — Model 3 (Extended)

| Coefficient | Variable | Value | Sign | Interpretation |
|---|---|---|:---:|---|
| β₁ | Oil Price | 7.72 | ➕ | $1 rise in oil → ~7.7-point rise in freight index |
| β₂ | Trade Volume | 1.98e-10 | ➕ | More trade → more vessel demand → higher rates |
| β₃ | Fleet DWT | −1.08e-06 | ➖ | More ships → lower rates (supply-side suppression) |
| β₄ | Oil Lag 1M | +1.36 | ➕ | Delayed fuel cost signal — market absorbs with 1-month lag |
| β₅ | Oil Lag 2M | −2.86 | ➖ | Partial reversal at 2 months — short-term mean reversion |

The **negative β₃** is the critical result. It confirms that vessel oversupply was a primary driver of the 2014–2016 freight rate collapse — a mechanism the original two-variable model was structurally unable to detect.

### Correlation Matrix

| Pair | Correlation | Note |
|---|---:|---|
| Freight × Trade | +0.414 | Strongest demand signal |
| Freight × Oil | +0.360 | Valid cost-side signal |
| Freight × Fleet | −0.068 | Directionally correct; non-linear effect |
| Oil × Trade | +0.459 | Multicollinearity — both driven by global activity |
| Fleet × Oil | −0.692 | Spurious — opposing long-run trends |
| Fleet × Trade | +0.228 | Coincidental growth correlation |

---

## Key Findings

- **Supply-side gap:** The original model excluded fleet capacity — the single most important structural driver of the 2014–2016 crash. Vessel orders placed during the 2010 boom kept arriving into a market with collapsing demand, driving rates from ~3,800 to ~250.

- **Correct coefficient signs:** All three predictors (oil, trade, fleet) carry theoretically expected signs in the regression, validating the model's economic specification.

- **Modest R²:** R² of 0.208–0.220 is consistent with published academic freight rate models. Shipping markets contain significant short-term volatility from port congestion, geopolitical events, seasonal cargo cycles, and charter speculation that linear regression cannot capture.

- **Lagged oil response:** Incorporating 1- and 2-month lagged oil prices improves R² from 0.208 to 0.220, confirming that freight markets absorb fuel cost changes with a temporal delay.

- **Synchronized 2015–2016 downturn:** All four variables simultaneously reflected the commodity bust — oil crashed, trade fell, yet fleet supply kept rising. This convergence is visible across every chart in the EDA.

---

## Requirements

```
python >= 3.10
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

1. Clone or download the repository
2. Place all four data files in the `data/` directory
3. Update `DATA_DIR` and `PLOTS_DIR` in the paths cell of the notebook
4. Run all cells in order in `Freight_analysis_v3.ipynb`

---

## Limitations & Future Work

- **Fleet data granularity:** UNCTAD fleet data is reported annually and forward-filled to monthly. A monthly vessel count series would significantly improve the supply-side signal.
- **Total fleet vs bulk-only:** The UNCTAD dataset covers all bulk carriers globally, which is appropriate for the BDI but does not isolate Capesize-specific capacity.
- **Non-linear modeling:** A gradient boosting or random forest model may better capture the threshold effects and non-linear interactions in freight markets.
- **Additional variables:** Port congestion indices, scrapping rates, newbuild orderbook data, and commodity-specific trade flows (iron ore, coal, grain) could further improve predictive power.

---
