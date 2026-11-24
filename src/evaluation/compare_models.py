
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

    actual = df["sales"].to_numpy()
    predicted = df["prediction"].to_numpy()

    metrics = calculate_metrics(actual, predicted)

    metrics["Model"] = model_name

    return metrics


def compare_models():

    print("Starting model comparison...")

    output_dir = Path("outputs")

    model_files = {
        "LightGBM": output_dir / "lightgbm_predictions.csv",
        "Prophet": output_dir / "prophet_predictions.csv"
    }

    results = []

    for model_name, file_path in model_files.items():

        if not file_path.exists():
            print(f"Skipping {model_name}: prediction file not found")
            continue

        print(f"Evaluating {model_name}...")

        results.append(
            evaluate_file(file_path, model_name)
        )

    if not results:
        print("No prediction files available.")
        return

    comparison = pd.DataFrame(results)

    comparison = comparison[
        ["Model", "MAE", "RMSE", "WAPE"]
    ]

    comparison = comparison.sort_values(
        "WAPE"
    )

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    output_dir.mkdir(exist_ok=True)

    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    print("\nComparison saved:")
    print("outputs/model_comparison.csv")


if __name__ == "__main__":
    compare_models()
