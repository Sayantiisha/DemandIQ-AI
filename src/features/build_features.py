
import pandas as pd
import numpy as np
from pathlib import Path


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    print("Starting feature engineering...")

    df = df.copy()

    # Ensure chronological order
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    # Calendar features
    print("Creating calendar features...")

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )

    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(int)

    # Lag features
    print("Creating lag features...")

    grouped_sales = df.groupby("id")["sales"]

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grouped_sales.shift(lag)

    # Rolling statistics
    print("Creating rolling features...")

    for window in [7, 14, 28]:

        df[f"rolling_mean_{window}"] = (
            df.groupby("id")["sales"]
            .transform(
                lambda x: x.shift(1).rolling(window).mean()
            )
        )

        df[f"rolling_std_{window}"] = (
            df.groupby("id")["sales"]
            .transform(
                lambda x: x.shift(1).rolling(window).std()
            )
        )

    # Price features
    print("Creating price features...")

    if "sell_price" in df.columns:

        df["sell_price"] = pd.to_numeric(
            df["sell_price"], errors="coerce"
        )

        df["price_change"] = (
            df.groupby("id")["sell_price"]
            .pct_change()
        )

    # Holiday and event features
    print("Creating event features...")

    if "event_name_1" in df.columns:

        df["has_event"] = (
            df["event_name_1"].notna()
        ).astype(int)

    if "event_name_2" in df.columns:

        df["has_event_2"] = (
            df["event_name_2"].notna()
        ).astype(int)

    # SNAP features
    for col in [
        "snap_CA",
        "snap_TX",
        "snap_WI"
    ]:

        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Save feature dataset
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "m5_features.csv"

    df.to_csv(file_path, index=False)

    print("\nFeature engineering completed!")
    print("Final dataset shape:", df.shape)
    print("Saved:", file_path)

    return df


if __name__ == "__main__":

    from src.data.prepare import load_m5

    df = load_m5(
        raw_dir="data/raw",
        max_series=100
    )

    featured_df = build_features(df)

    print("\nAvailable features:")
    print(featured_df.columns.tolist())
