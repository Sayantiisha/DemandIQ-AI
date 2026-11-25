
import pandas as pd
import numpy as np

from pathlib import Path


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    denominator = np.abs(actual).sum()

    wape = (
        np.abs(actual - predicted).sum()
        / denominator * 100
        if denominator > 0 else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape
    }


def evaluate_file(file_path, model_name):

    df = pd.read_csv(file_path)

    required_columns = ["id", "date", "sales", "prediction"]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{model_name} file is missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    actual = df["sales"].to_numpy()
    predicted = df["prediction"].to_numpy()

    metrics = calculate_metrics(actual, predicted)

    metrics["Model"] = model_name
    metrics["Series"] = df["id"].nunique()
    metrics["Forecast_Days"] = df.groupby("id").size().min()

    return metrics, df[["id", "date", "sales"]]


def compare_models():

    print("Starting model comparison...")

    output_dir = Path("outputs")

    model_files = {
        "LightGBM": output_dir / "lightgbm_predictions.csv",
        "Prophet": output_dir / "prophet_predictions.csv"
    }

    results = []
    reference = None

    for model_name, file_path in model_files.items():

        if not file_path.exists():
            print(f"Skipping {model_name}: prediction file not found")
            continue

        print(f"Evaluating {model_name}...")

        metrics, evaluation_data = evaluate_file(
            file_path,
            model_name
        )

        if reference is None:
            reference = evaluation_data
        else:
            if not reference.equals(evaluation_data):
                raise ValueError(
                    f"{model_name} predictions do not match "
                    "the reference model's IDs, dates, or actual sales."
                )

        results.append(metrics)

    if not results:
        print("No prediction files available.")
        return

    comparison = pd.DataFrame(results)

    comparison = comparison[
        [
            "Model",
            "MAE",
            "RMSE",
            "WAPE",
            "Series",
            "Forecast_Days"
        ]
    ]

    comparison = comparison.sort_values("WAPE")

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    print("\nComparison saved:")
    print("outputs/model_comparison.csv")


if __name__ == "__main__":
    compare_models()
