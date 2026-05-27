from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_estimator(model_name: str) -> Pipeline:
    name = model_name.lower()
    if name == "ridge":
        estimator = RidgeCV(alphas=np.logspace(-4, 3, 20))
    elif name in {"elasticnet", "elastic_net"}:
        estimator = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 2, 16),
            max_iter=20000,
            cv=3,
        )
    elif name == "mlp":
        estimator = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=800,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def walk_forward_predict(
    frame: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    label_col: str = "label_next_week_ret",
    min_train_weeks: int = 52,
) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(frame["date"].dropna().unique()))
    rows: list[pd.DataFrame] = []

    if len(dates) <= min_train_weeks + 1:
        raise ValueError("Not enough weekly observations for the requested walk-forward window.")

    for idx in range(min_train_weeks, len(dates)):
        test_date = dates[idx]
        train = frame[frame["date"] < test_date]
        test = frame[frame["date"] == test_date]
        if train.empty or test.empty:
            continue

        estimator = build_estimator(model_name)
        estimator.fit(train[feature_cols], train[label_col])

        pred = test[["date", "stock_code", label_col]].copy()
        pred["model"] = model_name.lower()
        pred["score_pred_next_week"] = estimator.predict(test[feature_cols])
        rows.append(pred)

    if not rows:
        raise ValueError("Walk-forward prediction produced no rows.")
    return pd.concat(rows, ignore_index=True)

