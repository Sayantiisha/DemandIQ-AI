import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.prepare import load_m5


# ==================================================
# 1. CONFIGURATION
# ==================================================

MAX_SERIES = 100
FORECAST_HORIZON = 28

OUTPUT_DIR = Path("outputs")
ARTIFACT_DIR = Path("artifacts")

OUTPUT_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

def train_prophet():

    print("\n========== PROPHET MODEL TRAINING ==========")

    df = load_m5(max_series=MAX_SERIES)

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)

    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    all_predictions = []

    # ==================================================
    # 3. TRAIN ONE PROPHET MODEL PER SERIES
    # ==================================================

    for index, (series_id, group) in enumerate(df.groupby("id"), start=1):

        print(f"Training series {index}/{df['id'].nunique()}: {series_id}")

        group = group.sort_values("date").reset_index(drop=True)

        prophet_df = group[["date", "sales"]].rename(
            columns={
                "date": "ds",
                "sales": "y"
            }
        )

        # Last 28 days are reserved for validation
        train_data = prophet_df.iloc[:-FORECAST_HORIZON].copy()
        test_data = prophet_df.iloc[-FORECAST_HORIZON:].copy()

        # Create Prophet model
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.80
        )

        # Train
        model.fit(train_data)

        # Forecast next 28 days
        future = model.make_future_dataframe(
            periods=FORECAST_HORIZON,
            freq="D"
        )

        forecast = model.predict(future)

        # Keep only validation-period predictions
        forecast = forecast.tail(FORECAST_HORIZON)

        predictions = pd.DataFrame({
            "id": series_id,
            "date": test_data["ds"].values,
            "sales": test_data["y"].values,
            "prediction": np.maximum(
                forecast["yhat"].values, 0
            )
        })

        all_predictions.append(predictions)

    # ==================================================
    # 4. COMBINE PREDICTIONS
    # ==================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions_df["date"] = pd.to_datetime(
        predictions_df["date"]
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "prophet_predictions.csv",
        index=False
    )

    # ==================================================
    # 5. EVALUATION METRICS
    # ==================================================

    actual = predictions_df["sales"].values
    predicted = predictions_df["prediction"].values

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    total_actual = np.sum(np.abs(actual))

    wape = (
        np.sum(np.abs(actual - predicted)) / total_actual * 100
        if total_actual != 0
        else 0
    )

    metrics = pd.DataFrame([{
        "model": "Prophet",
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "series": predictions_df["id"].nunique(),
        "forecast_days": FORECAST_HORIZON
    }])

    metrics.to_csv(
        OUTPUT_DIR / "prophet_metrics.csv",
        index=False
    )

    # ==================================================
    # 6. SAVE MODEL METADATA
    # ==================================================

    joblib.dump(
        {
            "model": "Prophet",
            "series": predictions_df["id"].nunique(),
            "forecast_horizon": FORECAST_HORIZON
        },
        ARTIFACT_DIR / "prophet_model_metadata.pkl"
    )

    # ==================================================
    # 7. DISPLAY RESULTS
    # ==================================================

    print("\n========== PROPHET RESULTS ==========")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"WAPE: {wape:.4f}%")
    print(f"Series: {predictions_df['id'].nunique()}")
    print(f"Forecast days: {FORECAST_HORIZON}")

    print("\nPredictions saved:")
    print(OUTPUT_DIR / "prophet_predictions.csv")

    print("\nMetrics saved:")
    print(OUTPUT_DIR / "prophet_metrics.csv")


if __name__ == "__main__":
    train_prophet()
