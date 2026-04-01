from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.data_pipeline import TARGET_COLUMN, TIMESTAMP_COLUMN, load_dataset, summarize_dataset
from src.evaluation import evaluate_holdout, evaluate_walk_forward
from src.features import build_feature_frame
from src.modeling import get_candidate_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Seoul bike demand forecasting models.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output artifacts folder")
    parser.add_argument("--holdout-hours", type=int, default=24 * 30, help="Size of holdout window in hours")
    parser.add_argument("--cv-splits", type=int, default=4, help="Number of walk-forward folds")
    parser.add_argument("--cv-test-hours", type=int, default=24 * 7, help="Hours per CV test fold")
    return parser.parse_args()


def _choose_champion(walk_forward_results: pd.DataFrame) -> str:
    summary = (
        walk_forward_results.groupby("model", as_index=False)[["mae", "rmse", "smape"]]
        .mean()
        .sort_values("rmse")
    )
    candidates_only = summary[~summary["model"].str.startswith("baseline_")]
    if candidates_only.empty:
        raise ValueError("No trainable candidate models found in walk-forward results")
    return str(candidates_only.iloc[0]["model"])


def _save_diagnostics(
    artifacts_dir: Path,
    timestamps: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> None:
    diagnostics = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": y_true,
            "predicted": y_pred,
        }
    )
    diagnostics["error"] = diagnostics["predicted"] - diagnostics["actual"]
    diagnostics.to_csv(artifacts_dir / "holdout_predictions.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(diagnostics["timestamp"], diagnostics["actual"], label="actual", linewidth=1.5)
    plt.plot(diagnostics["timestamp"], diagnostics["predicted"], label="predicted", linewidth=1.5)
    plt.title("Holdout Forecast: Actual vs Predicted")
    plt.xlabel("Timestamp")
    plt.ylabel("Rented Bike Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "holdout_actual_vs_predicted.png", dpi=140)
    plt.close()

    diagnostics["hour"] = pd.to_datetime(diagnostics["timestamp"]).dt.hour
    error_by_hour = (
        diagnostics.groupby("hour", as_index=False)
        .agg(error=("error", lambda values: values.abs().mean()))
        .sort_values("hour")
    )

    plt.figure(figsize=(10, 4))
    plt.bar(error_by_hour["hour"], error_by_hour["error"], color="#2f6fdf")
    plt.title("Mean Absolute Error by Hour (Holdout)")
    plt.xlabel("Hour")
    plt.ylabel("Mean Absolute Error")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "holdout_error_by_hour.png", dpi=140)
    plt.close()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(args.data_path)
    summary = summarize_dataset(raw_df)

    feature_set = build_feature_frame(raw_df)
    frame = feature_set.frame
    feature_columns = feature_set.feature_columns

    X = frame[feature_columns]
    y = frame[TARGET_COLUMN]
    timestamps = frame[TIMESTAMP_COLUMN]

    walk_forward_results = evaluate_walk_forward(
        X,
        y,
        models=get_candidate_models(),
        n_splits=args.cv_splits,
        test_size=args.cv_test_hours,
    )
    walk_forward_results.to_csv(artifacts_dir / "walk_forward_metrics.csv", index=False)

    holdout_size = int(args.holdout_hours)
    if holdout_size <= 0 or holdout_size >= len(frame):
        raise ValueError("holdout-hours must be positive and smaller than the prepared dataset length")

    split_idx = len(frame) - holdout_size
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    ts_test = timestamps.iloc[split_idx:]

    holdout_results = evaluate_holdout(
        X_train,
        y_train,
        X_test,
        y_test,
        models=get_candidate_models(),
    )
    holdout_results.to_csv(artifacts_dir / "holdout_metrics.csv", index=False)

    champion = _choose_champion(walk_forward_results)
    champion_model = get_candidate_models()[champion]
    champion_model.fit(X_train, y_train)
    holdout_pred = pd.Series(champion_model.predict(X_test), index=y_test.index)

    _save_diagnostics(artifacts_dir, ts_test, y_test, holdout_pred)

    champion_model.fit(X, y)
    joblib.dump(champion_model, artifacts_dir / "champion_model.joblib")

    frame.to_csv(artifacts_dir / "prepared_feature_frame.csv", index=False)

    metadata = {
        "target_column": TARGET_COLUMN,
        "timestamp_column": TIMESTAMP_COLUMN,
        "feature_columns": feature_columns,
        "champion_model": champion,
        "holdout_hours": holdout_size,
        "dataset_summary": {
            "row_count": summary.row_count,
            "start_timestamp": summary.start_timestamp,
            "end_timestamp": summary.end_timestamp,
            "target_mean": summary.target_mean,
            "target_std": summary.target_std,
        },
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Champion model: {champion}")
    print(f"Artifacts saved to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
