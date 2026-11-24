
import pandas as pd
from pathlib import Path


def reconcile_forecasts(
    forecast_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame
):
    """
    Aggregate bottom-level product-store forecasts
    into store, department, category and state totals.

    forecast_df columns:
        id, date, prediction

    hierarchy_df columns:
        id, store_id, state_id, cat_id, dept_id
    """

    print("Starting hierarchical reconciliation...")

    # Attach hierarchy information
    df = forecast_df.merge(
        hierarchy_df[
            [
                "id",
                "store_id",
                "state_id",
                "cat_id",
                "dept_id"
            ]
        ].drop_duplicates("id"),
        on="id",
        how="left"
    )

    # Bottom-level product-store forecasts
    bottom_level = (
        df.groupby(
            ["id", "date"],
            as_index=False
        )["prediction"].sum()
    )

    # Store-level forecasts
    store_level = (
        df.groupby(
            ["state_id", "store_id", "date"],
            as_index=False
        )["prediction"].sum()
    )

    # Department-level forecasts
    department_level = (
        df.groupby(
            ["state_id", "store_id", "dept_id", "date"],
            as_index=False
        )["prediction"].sum()
    )

    # Category-level forecasts
    category_level = (
        df.groupby(
            ["state_id", "store_id", "cat_id", "date"],
            as_index=False
        )["prediction"].sum()
    )

    # State-level forecasts
    state_level = (
        df.groupby(
            ["state_id", "date"],
            as_index=False
        )["prediction"].sum()
    )

    # Overall business-level forecasts
    total_level = (
        df.groupby(
            ["date"],
            as_index=False
        )["prediction"].sum()
    )

    print("Bottom-level rows:", len(bottom_level))
    print("Store-level rows:", len(store_level))
    print("Department-level rows:", len(department_level))
    print("Category-level rows:", len(category_level))
    print("State-level rows:", len(state_level))

    return {
        "bottom": bottom_level,
        "store": store_level,
        "department": department_level,
        "category": category_level,
        "state": state_level,
        "total": total_level
    }


if __name__ == "__main__":

    prediction_path = Path(
        "outputs/lightgbm_predictions.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            "Run LightGBM forecasting first."
        )

    from src.data.prepare import load_m5

    # Load hierarchy information from the M5 data
    print("Loading hierarchy information...")

    raw = load_m5(
        raw_dir="data/raw",
        max_series=100
    )

    hierarchy = raw[
        [
            "id",
            "store_id",
            "state_id",
            "cat_id",
            "dept_id"
        ]
    ].drop_duplicates("id")

    forecasts = pd.read_csv(prediction_path)
    forecasts["date"] = pd.to_datetime(
        forecasts["date"]
    )

    reconciled = reconcile_forecasts(
        forecasts,
        hierarchy
    )

    Path("outputs").mkdir(exist_ok=True)

    for level, result in reconciled.items():

        result.to_csv(
            f"outputs/forecast_{level}.csv",
            index=False
        )

        print(
            f"Saved {level} forecasts."
        )

    print("\nHierarchical reconciliation completed!")
