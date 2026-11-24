
import numpy as np
import pandas as pd


def calculate_mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return np.mean(np.abs(actual - predicted))


def calculate_rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_wape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    denominator = np.sum(np.abs(actual))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(actual - predicted))
        / denominator
    ) * 100


def calculate_rmsse(train_actual, valid_actual, predicted):
    train_actual = np.asarray(train_actual, dtype=float)
    valid_actual = np.asarray(valid_actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(train_actual) < 2:
        return np.nan

    denominator = np.mean(
        np.diff(train_actual) ** 2
    )

    if denominator == 0:
        return np.nan

    numerator = np.mean(
        (valid_actual - predicted) ** 2
    )

    return np.sqrt(numerator / denominator)


def evaluate_forecast(actual, predicted):
    return {
        "MAE": calculate_mae(actual, predicted),
        "RMSE": calculate_rmse(actual, predicted),
        "WAPE": calculate_wape(actual, predicted)
    }


if __name__ == "__main__":

    from src.data.prepare import load_m5
    from src.models.baselines import (
        time_based_split,
        naive_forecast,
        seasonal_naive_forecast
    )

    print("Loading M5 dataset...")

    df = load_m5(
        raw_dir="data/raw",
        max_series=100
    )

    train_df, valid_df = time_based_split(
        df,
        horizon=28
    )

    actual = valid_df["sales"].to_numpy()

    print("\nEvaluating Naive Forecast...")

    naive_predictions = naive_forecast(
        train_df,
        valid_df
    )

    naive_metrics = evaluate_forecast(
        actual,
        naive_predictions
    )

    print("Naive Forecast Results:")
    print(naive_metrics)

    print("\nEvaluating Seasonal Naive Forecast...")

    seasonal_predictions = seasonal_naive_forecast(
        train_df,
        valid_df
    )

    seasonal_metrics = evaluate_forecast(
        actual,
        seasonal_predictions
    )

    print("Seasonal Naive Results:")
    print(seasonal_metrics)

    print("\nEvaluation completed!")
