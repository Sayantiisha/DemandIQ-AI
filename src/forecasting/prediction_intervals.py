
import numpy as np
import pandas as pd

from pathlib import Path


def calculate_prediction_intervals(
    predictions,
    lower_quantile=0.05,
    upper_quantile=0.95
):
    """
    Estimate prediction intervals using historical
    forecast residuals.

    predictions must contain:
        id, date, sales, prediction
    """

    df = predictions.copy()

    df["residual"] = (
        df["sales"] - df["prediction"]
    )

    # Estimate residual distribution separately
    # for each product-store series.
    residual_stats = (
        df.groupby("id")["residual"]
        .quantile([
            lower_quantile,
            upper_quantile
        ])
        .unstack()
    )

    residual_stats.columns = [
        "lower_residual",
        "upper_residual"
    ]

    df = df.join(
        residual_stats,
        on="id"
    )

    # Calculate lower and upper demand estimates.
    df["lower_forecast"] = (
        df["prediction"] + df["lower_residual"]
    ).clip(lower=0)

    df["upper_forecast"] = (
        df["prediction"] + df["upper_residual"]
    ).clip(lower=0)

    # Ensure lower <= point <= upper
    df["lower_forecast"] = np.minimum(
        df["lower_forecast"],
        df["prediction"]
    )

    df["upper_forecast"] = np.maximum(
        df["upper_forecast"],
        df["prediction"]
    )

    return df


def evaluate_interval_coverage(df):

    actual = df["sales"].to_numpy()

    lower = df["lower_forecast"].to_numpy()
    upper = df["upper_forecast"].to_numpy()

    coverage = np.mean(
        (actual >= lower) & (actual <= upper)
    ) * 100

    average_width = np.mean(upper - lower)

    return {
        "coverage_percent": coverage,
        "average_interval_width": average_width
    }


def main():

    input_path = Path(
        "outputs/lightgbm_predictions.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            "Run LightGBM forecasting first."
        )

    df = pd.read_csv(input_path)

    print("Calculating prediction intervals...")

    result = calculate_prediction_intervals(df)

    metrics = evaluate_interval_coverage(result)

    print("\n========== UNCERTAINTY RESULTS ==========")

    print(
        "Interval coverage:",
        round(metrics["coverage_percent"], 2),
        "%"
    )

    print(
        "Average interval width:",
        round(metrics["average_interval_width"], 2)
    )

    Path("outputs").mkdir(exist_ok=True)

    result.to_csv(
        "outputs/prediction_intervals.csv",
        index=False
    )

    print("\nSaved: outputs/prediction_intervals.csv")

    print("\nPrediction interval calculation completed!")


if __name__ == "__main__":
    main()
