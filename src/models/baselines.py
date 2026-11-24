
import pandas as pd
import numpy as np


def time_based_split(
    df: pd.DataFrame,
    horizon: int = 28
):
    """
    Split each product-store series chronologically.
    The last `horizon` days are reserved for validation.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train_parts = []
    valid_parts = []

    for series_id, group in df.groupby("id"):

        group = group.sort_values("date")

        if len(group) <= horizon:
            continue

        train_parts.append(group.iloc[:-horizon])
        valid_parts.append(group.iloc[-horizon:])

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    valid_df = pd.concat(
        valid_parts,
        ignore_index=True
    )

    return train_df, valid_df


def naive_forecast(train_df, valid_df):

    """
    Predict future sales using the most recent
    observed sales value.
    """

    last_sales = (
        train_df.sort_values("date")
        .groupby("id")["sales"]
        .last()
    )

    predictions = valid_df["id"].map(last_sales)

    return predictions.fillna(0).to_numpy()


def seasonal_naive_forecast(train_df, valid_df):

    """
    Predict sales using the value from the same
    weekday one week earlier.
    """

    history = train_df[
        ["id", "date", "sales"]
    ].copy()

    history["date"] = (
        pd.to_datetime(history["date"])
    )

    history = history.set_index(
        ["id", "date"]
    )["sales"]

    predictions = []

    for row in valid_df.itertuples():

        previous_date = (
            row.date - pd.Timedelta(days=7)
        )

        key = (row.id, previous_date)

        prediction = history.get(key, np.nan)

        predictions.append(prediction)

    return np.nan_to_num(
        predictions,
        nan=0.0
    )


if __name__ == "__main__":

    from src.data.prepare import load_m5

    print("Loading M5 data...")

    df = load_m5(
        raw_dir="data/raw",
        max_series=100
    )

    print("Creating train-validation split...")

    train_df, valid_df = time_based_split(
        df,
        horizon=28
    )

    print("Training rows:", len(train_df))
    print("Validation rows:", len(valid_df))

    print("\nGenerating naive predictions...")

    naive_predictions = naive_forecast(
        train_df,
        valid_df
    )

    print("Naive predictions:", len(naive_predictions))

    print("\nGenerating seasonal naive predictions...")

    seasonal_predictions = seasonal_naive_forecast(
        train_df,
        valid_df
    )

    print(
        "Seasonal predictions:",
        len(seasonal_predictions)
    )

    print("\nBaseline forecasting completed!")
