from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WEEKS_PER_YEAR


def rank_ic_by_week(
    predictions: pd.DataFrame,
    score_col: str = "score_pred_next_week",
    label_col: str = "label_next_week_ret",
) -> pd.Series:
    values = {}
    for date, group in predictions.groupby("date"):
        group = group.dropna(subset=[score_col, label_col])
        if group[score_col].nunique() < 2 or group[label_col].nunique() < 2:
            continue
        values[pd.to_datetime(date)] = group[score_col].corr(group[label_col], method="spearman")
    return pd.Series(values, name="rank_ic").sort_index()


def hit_rate(
    predictions: pd.DataFrame,
    score_col: str = "score_pred_next_week",
    label_col: str = "label_next_week_ret",
) -> float:
    valid = predictions.dropna(subset=[score_col, label_col])
    if valid.empty:
        return float("nan")
    return float((np.sign(valid[score_col]) == np.sign(valid[label_col])).mean())


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    nav = (1.0 + r).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min())


def performance_summary(weekly_returns: pd.DataFrame, ret_col: str = "ret_net") -> dict[str, float]:
    r = weekly_returns[ret_col].dropna()
    if r.empty:
        return {"weeks": 0}

    ann_return = (1.0 + r).prod() ** (WEEKS_PER_YEAR / len(r)) - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    sharpe = np.nan if ann_vol == 0 else ann_return / ann_vol
    return {
        "weeks": int(len(r)),
        "annual_return": float(ann_return),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown(r),
        "mean_turnover": float(weekly_returns.get("turnover", pd.Series(dtype=float)).mean()),
    }
