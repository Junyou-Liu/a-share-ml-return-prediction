from __future__ import annotations

import pandas as pd

from .config import FACTOR_COLUMNS


def available_factor_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in FACTOR_COLUMNS if col in df.columns]


def build_model_frame(df: pd.DataFrame, label_col: str = "label_next_week_ret") -> pd.DataFrame:
    features = available_factor_columns(df)
    required = ["date", "stock_code", label_col, *features]
    if not features:
        raise ValueError("No configured factor columns were found in the dataset.")

    frame = df[required].copy()
    for col in [label_col, *features]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "stock_code", label_col])
    return frame.sort_values(["date", "stock_code"]).reset_index(drop=True)

