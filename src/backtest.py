from __future__ import annotations

import pandas as pd


def long_top_k_returns(
    predictions: pd.DataFrame,
    top_k: int = 3,
    cost_bps: float = 10.0,
    score_col: str = "score_pred_next_week",
    label_col: str = "label_next_week_ret",
) -> pd.DataFrame:
    cost = cost_bps / 10000.0
    pred = predictions.dropna(subset=["date", "stock_code", score_col, label_col]).copy()
    pred["date"] = pd.to_datetime(pred["date"])

    weekly_rows = []
    previous_holdings: set[str] = set()

    for date, group in pred.sort_values(["date", score_col], ascending=[True, False]).groupby("date"):
        selected = group.head(top_k).copy()
        holdings = set(selected["stock_code"].astype(str))
        turnover = len(holdings.symmetric_difference(previous_holdings)) / max(top_k, 1)
        gross_ret = selected[label_col].mean()
        net_ret = gross_ret - cost * turnover
        weekly_rows.append(
            {
                "date": date,
                "ret_gross": float(gross_ret),
                "ret_net": float(net_ret),
                "turnover": float(turnover),
                "holdings": ",".join(sorted(holdings)),
            }
        )
        previous_holdings = holdings

    return pd.DataFrame(weekly_rows)

