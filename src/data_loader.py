from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CODE_CANDIDATES, DATE_CANDIDATES, PRICE_CANDIDATES


def _first_existing(columns: list[str], candidates: tuple[str, ...], label: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(f"Missing {label} column. Expected one of: {', '.join(candidates)}")


def normalize_stock_code(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^\d]", "", regex=True)
    return cleaned.str.zfill(6)


def load_factor_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = _first_existing(list(df.columns), DATE_CANDIDATES, "date")
    code_col = _first_existing(list(df.columns), CODE_CANDIDATES, "stock code")

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    if code_col != "stock_code":
        df = df.rename(columns={code_col: "stock_code"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_code"] = normalize_stock_code(df["stock_code"])
    df = df.dropna(subset=["date", "stock_code"]).sort_values(["stock_code", "date"])
    return df.reset_index(drop=True)


def add_forward_return_label(df: pd.DataFrame, label_col: str = "label_next_week_ret") -> pd.DataFrame:
    if label_col in df.columns:
        return df.copy()

    price_col = next((col for col in PRICE_CANDIDATES if col in df.columns), None)
    if price_col is None:
        raise ValueError(
            f"Cannot build {label_col}: no existing label and no price column in {PRICE_CANDIDATES}."
        )

    out = df.sort_values(["stock_code", "date"]).copy()
    out["ret_week"] = out.groupby("stock_code")[price_col].pct_change()
    out[label_col] = out.groupby("stock_code")["ret_week"].shift(-1)
    return out


def load_benchmark_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Benchmark file must contain a date column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

