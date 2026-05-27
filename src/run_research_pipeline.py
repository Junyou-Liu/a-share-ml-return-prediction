from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import long_top_k_returns
from .config import FACTOR_FILE
from .data_loader import add_forward_return_label, load_factor_data
from .evaluation import hit_rate, performance_summary, rank_ic_by_week
from .feature_engineering import available_factor_columns, build_model_frame
from .models import walk_forward_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a walk-forward A-share return prediction workflow.")
    parser.add_argument("--data", default=str(FACTOR_FILE), help="Path to the weekly factor CSV.")
    parser.add_argument("--models", default="ridge,elasticnet", help="Comma-separated models: ridge, elasticnet, mlp.")
    parser.add_argument("--min-train-weeks", type=int, default=52)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output-dir", default="outputs/src_pipeline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_factor_data(args.data)
    labeled = add_forward_return_label(raw)
    frame = build_model_frame(labeled)
    features = available_factor_columns(frame)

    summaries = {}
    for model_name in [name.strip().lower() for name in args.models.split(",") if name.strip()]:
        preds = walk_forward_predict(
            frame,
            feature_cols=features,
            model_name=model_name,
            min_train_weeks=args.min_train_weeks,
        )
        weekly = long_top_k_returns(preds, top_k=args.top_k, cost_bps=args.cost_bps)
        rank_ic = rank_ic_by_week(preds)

        preds.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)
        weekly.to_csv(output_dir / f"{model_name}_weekly_returns.csv", index=False)
        rank_ic.to_csv(output_dir / f"{model_name}_rank_ic.csv", header=True)

        summary = performance_summary(weekly)
        summary["hit_rate"] = hit_rate(preds)
        summary["mean_rank_ic"] = float(rank_ic.mean()) if not rank_ic.empty else None
        summary["n_predictions"] = int(len(preds))
        summaries[model_name] = summary

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(pd.DataFrame(summaries).T.round(4).to_string())
    print(f"\nOutputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

