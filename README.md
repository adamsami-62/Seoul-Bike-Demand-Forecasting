# Seoul Bike Demand Forecasting

An end-to-end machine learning pipeline for predicting hourly bike rental demand in Seoul using weather conditions, time features, and historical demand signals.

## What This Project Does

The goal is to predict how many bikes will be rented in a given hour based on the current weather, time of day, and recent rental history. Accurate hourly forecasts help operations teams plan bike availability and staffing more efficiently.

The pipeline covers data cleaning, feature engineering, multi-model training, time-aware validation, and a recursive 24-hour forecast simulation, producing reproducible artifacts at each stage.

## Dataset

8,760 hourly observations from December 2017 to November 2018, sourced from the Seoul Open Data Portal. Dataset obtained from [here](https://www.kaggle.com/datasets/sadamumer/seoul-bike-hourly-rentals-and-weather-20172018). Each row contains the rental count for that hour alongside weather readings including temperature, humidity, wind speed, rainfall, snowfall, and solar radiation.

## Approach

**Validation design**

This project uses walk-forward cross-validation across 4 folds, with a final 30-day holdout window kept completely unseen until the end. All splits are strictly chronological.

**Baselines**

Two naive baselines are included: previous hour and previous day same hour.

**Feature engineering**

Features are grouped into four categories:
- Time features: day of week, month, cyclical sine/cosine encodings for hour, day, and month
- Lag features: rental counts from 1, 2, 3, 24, and 168 hours ago
- Rolling means: 3-hour, 24-hour, and 168-hour rolling averages (shift-before-roll to prevent leakage)
- Interaction features: temperature x humidity, temperature x wind, rainfall x humidity

**Model comparison**

Three model families are compared: Ridge Regression, Random Forest, and HistGradientBoostingRegressor. The champion is selected by lowest average RMSE across walk-forward folds, excluding baselines.

## Results

Holdout window performance (final 30 unseen days):

| Model | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| baseline_prev_hour | 173.87 | 278.81 | 27.70 |
| baseline_prev_day | 283.19 | 486.90 | 64.23 |
| ridge | 157.10 | 226.96 | 44.18 |
| random_forest | 60.44 | 96.73 | 22.90 |
| hist_gradient_boosting | 54.90 | 77.59 | 31.00 |

The champion model (HistGradientBoosting) reduces MAE by 68% compared to the best naive baseline and achieves an R² of 0.97 on the holdout window, explaining 97% of the variance in hourly demand across the final 30 unseen days.

## Visualizations

Holdout window: actual vs predicted demand across the final 30 days.

![Holdout actual vs predicted](artifacts/holdout_actual_vs_predicted.png)

Recursive 24-hour forecast simulation on the final day of data.

![Recursive forecast](artifacts/recursive_forecast_last_window.png)

Mean absolute error broken down by hour of day.

![Error by hour](artifacts/holdout_error_by_hour.png)

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate:

```bash
python train.py --data-path "Seoul-bicycle-rental-dataset.csv" --artifacts-dir artifacts
```

Run 24-hour recursive forecast simulation:

```bash
python forecast.py --data-path "Seoul-bicycle-rental-dataset.csv" --artifacts-dir artifacts --horizon 24
```

## Repository Structure
```
├── src/
│   ├── data_pipeline.py   # Loading, column normalization, schema validation
│   ├── features.py        # Lag, rolling, cyclical, and interaction features
│   ├── modeling.py        # Candidate model definitions
│   └── evaluation.py      # MAE/RMSE/sMAPE, walk-forward, holdout evaluation
├── artifacts/
│   ├── holdout_actual_vs_predicted.png
│   ├── holdout_error_by_hour.png
│   ├── recursive_forecast_last_window.png
│   ├── holdout_metrics.csv
│   ├── walk_forward_metrics.csv
│   └── recursive_forecast_last_window.csv
├── train.py               # Full training and evaluation script
├── forecast.py            # Recursive 24-hour forecast simulation
├── requirements.txt
└── Seoul-bicycle-rental-dataset.csv
```

## Tools

Python, Pandas, NumPy, Scikit-learn, Matplotlib
