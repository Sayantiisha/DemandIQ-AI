
# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# import joblib

# from pathlib import Path
# from src.data.prepare import load_m5


# # ==================================================
# # 1. CONFIGURATION
# # ==================================================

# FEATURES = [
#     "day",
#     "month",
#     "dayofweek",
#     "weekofyear",
#     "is_weekend",
#     "lag_1",
#     "lag_7",
#     "lag_14",
#     "lag_28",
#     "rolling_mean_7",
#     "rolling_mean_28",
#     "rolling_std_7",
#     "sell_price",
#     "price_change",
#     "has_event",
#     "has_event_2",
#     "snap_CA",
#     "snap_TX",
#     "snap_WI",
# ]

# MAX_SERIES = 100
# FORECAST_DAYS = 28
# RANDOM_STATE = 42


# # ==================================================
# # 2. FEATURE ENGINEERING
# # ==================================================

# def create_training_features(df):

#     df = df.copy()

#     df["date"] = pd.to_datetime(df["date"])

#     df = df.sort_values(
#         ["id", "date"]
#     ).reset_index(drop=True)

#     # Calendar features
#     df["day"] = df["date"].dt.day
#     df["month"] = df["date"].dt.month
#     df["dayofweek"] = df["date"].dt.day_of_week

#     df["weekofyear"] = (
#         df["date"].dt.isocalendar().week.astype(int)
#     )

#     df["is_weekend"] = (
#         df["dayofweek"] >= 5
#     ).astype(int)

#     # Lag features
#     for lag in [1, 7, 14, 28]:

#         df[f"lag_{lag}"] = (
#             df.groupby("id")["sales"].shift(lag)
#         )

#     # Rolling mean features
#     for window in [7, 28]:

#         df[f"rolling_mean_{window}"] = (
#             df.groupby("id")["sales"]
#             .transform(
#                 lambda x: x.shift(1).rolling(
#                     window,
#                     min_periods=window
#                 ).mean()
#             )
#         )

#     # Rolling standard deviation
#     df["rolling_std_7"] = (
#         df.groupby("id")["sales"]
#         .transform(
#             lambda x: x.shift(1).rolling(
#                 7,
#                 min_periods=7
#             ).std(ddof=0)
#         )
#     )

#     # Price features
#     if "sell_price" not in df.columns:
#         df["sell_price"] = np.nan

#     df["price_change"] = (
#         df.groupby("id")["sell_price"]
#         .pct_change(fill_method=None)
#     )

#     # Event features
#     for col in ["event_name_1", "event_name_2"]:

#         if col not in df.columns:
#             df[col] = np.nan

#     df["has_event"] = (
#         df["event_name_1"].notna().astype(int)
#     )

#     df["has_event_2"] = (
#         df["event_name_2"].notna().astype(int)
#     )

#     # SNAP features
#     for col in ["snap_CA", "snap_TX", "snap_WI"]:

#         if col not in df.columns:
#             df[col] = 0

#         df[col] = (
#             pd.to_numeric(df[col], errors="coerce")
#             .fillna(0)
#             .astype(int)
#         )

#     return df


# # ==================================================
# # 3. PREPARE FEATURES
# # ==================================================

# def prepare_features(df):

#     df = df.copy()

#     df[FEATURES] = (
#         df[FEATURES]
#         .replace([np.inf, -np.inf], np.nan)
#         .fillna(0)
#     )

#     return df


# # ==================================================
# # 4. RECURSIVE FORECASTING
# # ==================================================

# def recursive_forecast(model, train_df, future_df):

#     train_df = train_df.copy()
#     future_df = future_df.copy()

#     train_df["date"] = pd.to_datetime(train_df["date"])
#     future_df["date"] = pd.to_datetime(future_df["date"])

#     # Historical sales
#     histories = {}

#     for series_id, group in train_df.groupby("id"):

#         group = group.sort_values("date")

#         histories[series_id] = (
#             group["sales"].astype(float).tolist()
#         )

#     # Last known prices
#     last_prices = {}

#     for series_id, group in train_df.groupby("id"):

#         prices = (
#             group.sort_values("date")["sell_price"]
#             .dropna()
#         )

#         last_prices[series_id] = (
#             float(prices.iloc[-1])
#             if not prices.empty
#             else np.nan
#         )

#     predictions = []

#     future_df = future_df.sort_values(
#         ["date", "id"]
#     )

#     # Forecast one day at a time
#     for date, daily_rows in future_df.groupby(
#         "date",
#         sort=True
#     ):

#         for row in daily_rows.itertuples(index=False):

#             series_id = row.id

#             sales_history = histories.get(
#                 series_id,
#                 []
#             )

#             if len(sales_history) < 28:

#                 prediction = 0.0

#             else:

#                 values = np.asarray(
#                     sales_history,
#                     dtype=float
#                 )

#                 current_price = getattr(
#                     row,
#                     "sell_price",
#                     np.nan
#                 )

#                 previous_price = last_prices.get(
#                     series_id,
#                     np.nan
#                 )

#                 # Price change
#                 if (
#                     pd.isna(current_price)
#                     or pd.isna(previous_price)
#                     or previous_price == 0
#                 ):

#                     price_change = 0.0

#                 else:

#                     price_change = (
#                         float(current_price)
#                         / float(previous_price)
#                     ) - 1

#                 # SNAP values
#                 snap_values = {}

#                 for col in ["snap_CA", "snap_TX", "snap_WI"]:

#                     value = getattr(row, col, 0)

#                     snap_values[col] = (
#                         0 if pd.isna(value)
#                         else int(value)
#                     )

#                 # Forecast features
#                 features = {

#                     "day": date.day,

#                     "month": date.month,

#                     "dayofweek": date.day_of_week,

#                     "weekofyear": int(
#                         date.isocalendar().week
#                     ),

#                     "is_weekend": int(
#                         date.day_of_week >= 5
#                     ),

#                     "lag_1": values[-1],

#                     "lag_7": values[-7],

#                     "lag_14": values[-14],

#                     "lag_28": values[-28],

#                     "rolling_mean_7": values[-7:].mean(),

#                     "rolling_mean_28": values[-28:].mean(),

#                     "rolling_std_7": values[-7:].std(ddof=0),

#                     "sell_price": (
#                         0.0
#                         if pd.isna(current_price)
#                         else float(current_price)
#                     ),

#                     "price_change": price_change,

#                     "has_event": int(
#                         pd.notna(
#                             getattr(row, "event_name_1", None)
#                         )
#                     ),

#                     "has_event_2": int(
#                         pd.notna(
#                             getattr(row, "event_name_2", None)
#                         )
#                     ),

#                     **snap_values,

#                 }

#                 X = pd.DataFrame(
#                     [features],
#                     columns=FEATURES
#                 )

#                 X = prepare_features(X)

#                 prediction = max(
#                     0.0,
#                     float(model.predict(X)[0])
#                 )

#             # Update history with prediction
#             histories.setdefault(
#                 series_id,
#                 []
#             ).append(prediction)

#             # Update last known price
#             if not pd.isna(
#                 getattr(row, "sell_price", np.nan)
#             ):

#                 last_prices[series_id] = float(
#                     row.sell_price
#                 )

#             predictions.append({

#                 "id": series_id,

#                 "date": date,

#                 "prediction": prediction,

#             })

#     return pd.DataFrame(predictions)


# # ==================================================
# # 5. TRAIN LIGHTGBM MODEL
# # ==================================================

# def train_model():

#     print("\nLoading M5 dataset...")

#     df = load_m5(
#         raw_dir="data/raw",
#         max_series=MAX_SERIES
#     )

#     df["date"] = pd.to_datetime(df["date"])

#     df["sales"] = pd.to_numeric(
#         df["sales"],
#         errors="coerce"
#     ).fillna(0)

#     df = df.sort_values(
#         ["id", "date"]
#     ).reset_index(drop=True)

#     print("\nDataset shape:", df.shape)

#     print(
#         "Number of series:",
#         df["id"].nunique()
#     )

#     # Validation split
#     dates = sorted(
#         df["date"].unique()
#     )

#     if len(dates) <= FORECAST_DAYS + 28:

#         raise ValueError(
#             "Not enough dates for training and validation."
#         )

#     cutoff = dates[-FORECAST_DAYS - 1]

#     print("\nTraining cutoff:", cutoff)

#     train_raw = df[
#         df["date"] <= cutoff
#     ].copy()

#     valid_raw = df[
#         df["date"] > cutoff
#     ].copy()

#     print(
#         "Training rows:",
#         len(train_raw)
#     )

#     print(
#         "Validation rows:",
#         len(valid_raw)
#     )

#     # Build training features
#     print("\nBuilding training features...")

#     train_features = create_training_features(
#         train_raw
#     )

#     # Remove initial rows without sufficient lag history
#     train_features = train_features.dropna(
#         subset=[
#             "lag_1",
#             "lag_7",
#             "lag_14",
#             "lag_28",
#             "rolling_mean_7",
#             "rolling_mean_28",
#             "rolling_std_7",
#         ]
#     )

#     train_features = prepare_features(
#         train_features
#     )

#     X_train = train_features[FEATURES]

#     y_train = train_features["sales"]

#     print(
#         "Training feature rows:",
#         len(X_train)
#     )

#     if X_train.empty:

#         raise ValueError(
#             "No training rows available after feature engineering."
#         )

#     # Train model
#     print("\nTraining LightGBM model...")

#     model = lgb.LGBMRegressor(

#         objective="regression",

#         n_estimators=500,

#         learning_rate=0.05,

#         num_leaves=31,

#         max_depth=-1,

#         subsample=0.8,

#         colsample_bytree=0.8,

#         random_state=RANDOM_STATE,

#         n_jobs=-1,

#         verbosity=-1,

#     )

#     model.fit(
#         X_train,
#         y_train
#     )

#     print("\nModel training completed!")

#     # Recursive forecasting
#     print(
#         "\nGenerating",
#         FORECAST_DAYS,
#         "day forecasts..."
#     )

#     predictions = recursive_forecast(
#         model,
#         train_raw,
#         valid_raw
#     )

#     # Merge actual and predicted values
#     results = valid_raw[
#         ["id", "date", "sales"]
#     ].merge(

#         predictions,

#         on=["id", "date"],

#         how="inner",

#         validate="one_to_one"

#     )

#     if len(results) != len(valid_raw):

#         raise ValueError(
#             "Prediction count does not match validation rows."
#         )

#     # Evaluation
#     actual = results[
#         "sales"
#     ].to_numpy(dtype=float)

#     predicted = results[
#         "prediction"
#     ].to_numpy(dtype=float)

#     mae = np.mean(
#         np.abs(actual - predicted)
#     )

#     rmse = np.sqrt(
#         np.mean(
#             (actual - predicted) ** 2
#         )
#     )

#     denominator = np.abs(actual).sum()

#     wape = (

#         np.abs(actual - predicted).sum()
#         / denominator * 100

#         if denominator > 0

#         else np.nan

#     )

#     print(
#         "\n========== LIGHTGBM RESULTS =========="
#     )

#     print("MAE:", round(mae, 4))

#     print("RMSE:", round(rmse, 4))

#     print("WAPE:", round(wape, 4), "%")

#     print(
#         "Prediction rows:",
#         len(results)
#     )

#     # Save model and results
#     Path("artifacts").mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     Path("outputs").mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     joblib.dump(
#         model,
#         "artifacts/lightgbm_model.pkl"
#     )

#     results.to_csv(
#         "outputs/lightgbm_predictions.csv",
#         index=False
#     )

#     metrics = pd.DataFrame([{

#         "model": "LightGBM",

#         "MAE": mae,

#         "RMSE": rmse,

#         "WAPE": wape,

#         "series": df["id"].nunique(),

#         "forecast_days": FORECAST_DAYS,

#     }])

#     metrics.to_csv(
#         "outputs/lightgbm_metrics.csv",
#         index=False
#     )

#     print(
#         "\nModel saved: artifacts/lightgbm_model.pkl"
#     )

#     print(
#         "Predictions saved: outputs/lightgbm_predictions.csv"
#     )

#     print(
#         "Metrics saved: outputs/lightgbm_metrics.csv"
#     )

#     print(
#         "\nLightGBM training completed successfully!"
#     )


# # ==================================================
# # 6. MAIN
# # ==================================================

# if __name__ == "__main__":

#     train_model()



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
