from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_pipeline import TARGET_COLUMN, TIMESTAMP_COLUMN, load_dataset
from src.features import build_feature_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recursive forecast simulation for the final horizon window.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifact directory from training")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours")
    return parser.parse_args()


def _build_recursive_row(frame: pd.DataFrame, idx: int, history: list[float]) -> pd.Series:
    row = frame.iloc[idx].copy()

    row["lag_1"] = history[-1]
    row["lag_2"] = history[-2]
    row["lag_3"] = history[-3]
    row["lag_24"] = history[-24]
    row["lag_168"] = history[-168]

    row["roll_mean_3"] = float(np.mean(history[-3:]))
    row["roll_mean_24"] = float(np.mean(history[-24:]))
    row["roll_mean_168"] = float(np.mean(history[-168:]))

    return row


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    horizon = int(args.horizon)

    metadata_path = artifacts_dir / "metadata.json"
    model_path = artifacts_dir / "champion_model.joblib"

    if not metadata_path.exists() or not model_path.exists():
        raise FileNotFoundError("Artifacts not found. Run train.py first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]

    model = joblib.load(model_path)

    raw_df = load_dataset(args.data_path)
    frame = build_feature_frame(raw_df).frame

    if horizon <= 0 or horizon >= len(frame):
        raise ValueError("horizon must be positive and smaller than prepared dataset length")

    start_idx = len(frame) - horizon
    history = frame.iloc[:start_idx][TARGET_COLUMN].tolist()

    if len(history) < 168:
        raise ValueError("Not enough history for recursive lags. Need at least 168 rows before forecast start.")

    predictions = []
    timestamps = []
    actuals = []

    for idx in range(start_idx, len(frame)):
        row = _build_recursive_row(frame, idx, history)
        pred = float(model.predict(row[feature_columns].to_frame().T)[0])
        history.append(pred)

        predictions.append(pred)
        timestamps.append(row[TIMESTAMP_COLUMN])
        actuals.append(float(row[TARGET_COLUMN]))

    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual": actuals,
            "predicted_recursive": predictions,
        }
    )
    result["absolute_error"] = (result["actual"] - result["predicted_recursive"]).abs()

    result.to_csv(artifacts_dir / "recursive_forecast_last_window.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(result["timestamp"], result["actual"], label="actual", linewidth=1.5)
    plt.plot(result["timestamp"], result["predicted_recursive"], label="predicted_recursive", linewidth=1.5)
    plt.title(f"Recursive Forecast Simulation (Last {horizon} Hours)")
    plt.xlabel("Timestamp")
    plt.ylabel("Rented Bike Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_dir / "recursive_forecast_last_window.png", dpi=140)
    plt.close()

    mae = float(result["absolute_error"].mean())
    rmse = float(np.sqrt(np.mean((result["actual"] - result["predicted_recursive"]) ** 2)))

    print("Forecast simulation complete")
    print(f"Horizon: {horizon} hours")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Saved: {(artifacts_dir / 'recursive_forecast_last_window.csv').resolve()}")


if __name__ == "__main__":
    main()
