# A-Share ML Return Prediction

A machine learning framework for weekly A-share return prediction and portfolio backtesting.

This is an academic research project focused on factor modeling, rolling validation, and simulated backtest evaluation. It is not investment advice, a live trading system, or evidence of personal fund management.

## Project Overview

This repository presents a National University of Singapore course project on weekly A-share return prediction and factor-based strategy evaluation. It compares Ridge Regression, Elastic Net, and MLP models on six ChiNext stocks, with `399006.SZ` used as the benchmark reference.

The project is best read as a modeling and validation workflow: building factor features, training predictive models, generating weekly ranking signals, and evaluating results through both predictive and portfolio-level metrics.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Analysis | pandas, NumPy |
| Modeling | scikit-learn, PyTorch |
| Workflow | Jupyter Notebook, modular Python scripts |
| Market Data | AKShare / JQData-style data access, local CSV inputs |
| Visualization | matplotlib |

## Key Features

- Built a weekly factor dataset for six ChiNext stocks.
- Used valuation, quality, growth, fund flow, momentum, volatility, liquidity, RSI, and MACD-style signals.
- Compared Ridge Regression, Elastic Net, and MLP model pipelines.
- Applied chronological splitting and rolling-origin validation to reduce look-ahead bias.
- Evaluated model outputs through NAV, Sharpe, drawdown, hit rate, RankIC, turnover, and benchmark-relative indicators.
- Kept input CSV files in the repository so the notebook can be reviewed without relying on external market-data APIs.
- Added a lightweight `src/` package for reusable data loading, feature preparation, walk-forward prediction, backtesting, and evaluation utilities.

## Methodology

1. Prepare weekly stock-level factor features and next-week return labels.
2. Load the `399006.SZ` benchmark return series.
3. Train Ridge Regression with cross-validation and generate weekly ranking signals.
4. Train Elastic Net with a grid over alpha and l1 ratio, using validation performance for model selection.
5. Train an MLP model and save predictions for the same evaluation framework.
6. Standardize all model outputs for post-cost walk-forward comparison.
7. Compare strategy behavior against benchmark movement and risk indicators.

## Results / Metrics

The executed notebook contains the following evidence:

- Ridge, Elastic Net, and MLP model code paths are present.
- The unified evaluation loads 109 out-of-sample weekly periods for each model.
- The evaluation confirms six tickers and 654 standardized prediction rows per model.
- The MLP section reports best validation RankIC around 0.4582 and saves a test prediction file with 271 rows.
- The final comparison table includes hit rate, mean RankIC, max drawdown, turnover events, and benchmark-relative fields.
- Resume-facing summary metrics include Hit Rate 54.2% and RankIC 0.06 under the weekly simulated backtest setting.

Some annualized return outputs in the notebook are very high because they come from a small, concentrated weekly sample. For interviews, this project should be discussed as a controlled factor-modeling and backtesting workflow rather than proof of investable alpha.

## How to Run

Use Python 3.9 to 3.11 when possible.

```powershell
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook DSA5205_project2_Teams12_1115.ipynb
```

Command-line execution:

```bash
jupyter nbconvert --execute --to notebook --inplace DSA5205_project2_Teams12_1115.ipynb --ExecutePreprocessor.timeout=1200
```

Lightweight script pipeline:

```powershell
python -m src.run_research_pipeline --models ridge,elasticnet --min-train-weeks 52 --output-dir outputs/src_pipeline
```

The script pipeline is a reusable engineering refactor of the core workflow. The notebook remains the complete research record for the original course analysis and MLP training path.

## Repository Structure

```text
.
|-- DSA5205_project2_Teams12_1115.ipynb
|-- src/
|   |-- data_loader.py
|   |-- feature_engineering.py
|   |-- models.py
|   |-- backtest.py
|   |-- evaluation.py
|   `-- run_research_pipeline.py
|-- factors_6stocks_weekly_20251107_013757.csv
|-- bm_399006_weekly.csv
|-- MLP_data.csv
|-- requirements.txt
`-- README.md
```

## Limitations

- The sample is small and concentrated, so performance metrics should not be generalized.
- Backtest results depend on factor construction, transaction-cost assumptions, rebalance rules, and market regime.
- The project does not represent live trading, client assets, investment advice, or fundraising activity.
- The strongest interview angle is modeling discipline: feature construction, validation design, benchmark comparison, and risk-aware interpretation.
