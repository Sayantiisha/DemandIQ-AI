
import pandas as pd
from pathlib import Path

from src.data.prepare import load_m5, save_panel


def preprocess_data():

    print("\nStarting DemandIQ AI preprocessing...")

    # 1. Load and merge M5 datasets
    print("\nLoading M5 datasets...")

    df = load_m5(
        raw_dir="data/raw",
        max_series=100
    )

    print("Dataset loaded:", df.shape)

    # 2. Remove duplicate records
    print("\nChecking duplicate records...")

    duplicates = df.duplicated(
        subset=["id", "date"]
    ).sum()

    print("Duplicate records:", duplicates)

    df = df.drop_duplicates(
        subset=["id", "date"]
    )

    # 3. Handle missing sales values
    print("\nHandling missing sales...")

    df["sales"] = pd.to_numeric(
        df["sales"], errors="coerce"
    )

    df["sales"] = df["sales"].fillna(0)

    # 4. Convert data types
    print("\nConverting data types...")

    df["date"] = pd.to_datetime(df["date"])

    df["sales"] = df["sales"].astype("float32")

    # 5. Check missing prices
    print("\nChecking missing prices...")

    if "sell_price" in df.columns:
        print(
            "Missing prices:",
            df["sell_price"].isna().sum()
        )

    # 6. Sort records
    df = df.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    # 7. Validate the dataset
    print("\nFinal validation...")

    print("Rows:", len(df))
    print("Products:", df["item_id"].nunique())
    print("Stores:", df["store_id"].nunique())
    print("States:", df["state_id"].nunique())

    print(
        "Date range:",
        df["date"].min(),
        "to",
        df["date"].max()
    )

    print(
        "Total sales:",
        df["sales"].sum()
    )

    # 8. Save processed dataset
    output_path = save_panel(
        df,
        "data/processed/m5_cleaned.csv"
    )

    print("\nProcessed dataset saved:", output_path)

    print("\nPreprocessing completed successfully!")

    return df


if __name__ == "__main__":
    preprocess_data()
