# A-Share ML Return Prediction and Strategy Backtesting

This repository contains a National University of Singapore course project on weekly A-share return prediction and factor strategy backtesting. It compares Ridge, Elastic Net, and MLP models on six ChiNext stocks, with `399006.SZ` used as the benchmark reference. The project is positioned as a modeling and validation exercise, not as a live trading strategy.

## What This Project Shows

- Built a weekly factor dataset for six ChiNext stocks with valuation, quality, growth, flow, momentum, volatility, liquidity, RSI, and MACD-style signals.
- Used `399006.SZ` weekly returns as the benchmark reference for strategy evaluation.
- Compared Ridge, Elastic Net, and MLPRegressor / neural-network pipelines.
- Applied chronological splitting, rolling-origin validation, and post-cost weekly rebalancing logic to reduce look-ahead bias.
- Exported model-specific prediction files and a unified evaluation workflow for NAV, Sharpe, drawdown, hit rate, RankIC, turnover, and benchmark comparison.

## Data and Inputs

The repository keeps the main inputs beside the notebook so the analysis can run even when external data APIs are unavailable.

```text
.
|-- DSA5205_project2_Teams12_1115.ipynb
|-- factors_6stocks_weekly_20251107_013757.csv
|-- bm_399006_weekly.csv
|-- MLP_data.csv
|-- requirements.txt
`-- README.md
```

## Modeling Workflow

1. Prepare weekly stock-level factors and next-week return labels.
2. Load or build the `399006.SZ` benchmark return series.
3. Train Ridge with cross-validation and generate weekly ranking signals.
4. Train Elastic Net with a grid over alpha and l1 ratio, using validation Sharpe for model selection.
5. Train an MLP model and save predictions for the same evaluation framework.
6. Standardize all model outputs for post-cost walk-forward comparison.

## Verified Notebook Outputs

The executed notebook contains the following evidence:

- Ridge, Elastic Net, and MLP model code paths are present in the notebook.
- The unified evaluation loads 109 out-of-sample weekly periods for each model.
- The unified evaluation confirms six tickers and 654 standardized prediction rows per model.
- The MLP section reports best validation RankIC around 0.4582 and a saved test prediction file with 271 rows.
- The final comparison table includes hit rate, mean RankIC, max drawdown, turnover events, and benchmark-relative fields.

Some annualized return outputs in the notebook are very high because they are generated from a small, concentrated weekly sample. For resume or interview use, this project should be discussed as a controlled factor-modeling and backtesting workflow, not as proof of investable alpha.

## Reproduce

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

## Scope and Limits

This is an academic quant-finance project. It demonstrates factor construction, model comparison, validation discipline, and backtest reporting. It should not be described as personal fund management, live trading, fundraising, or investment advice.

## Stack

Python, pandas, NumPy, scikit-learn, PyTorch, matplotlib, AKShare/JQData-style data access, and Jupyter Notebook.
