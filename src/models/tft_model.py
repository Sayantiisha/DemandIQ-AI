
import pandas as pd
import numpy as np
import torch

from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE


def train_tft():

    print("Loading M5 data...")

    from src.data.prepare import load_m5

    df = load_m5(
        raw_dir="data/raw",
        max_series=10
    )

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    # Numeric time index
    df["time_idx"] = (
        df["date"] - df["date"].min()
    ).dt.days

    # Static categorical identifier
    df["series_id"] = df["id"].astype(str)

    # Known calendar features
    df["day_of_week"] = (
        df["date"].dt.dayofweek.astype(str)
    )

    df["month"] = (
        df["date"].dt.month.astype(str)
    )

    df["weekend"] = (
        df["date"].dt.dayofweek >= 5
    ).astype(str)

    # Fill optional known future features
    for col in ["snap_CA", "snap_TX", "snap_WI"]:
        if col not in df.columns:
            df[col] = 0

        df[col] = df[col].fillna(0).astype(float)

    # Price is included as an observed historical feature.
    if "sell_price" not in df.columns:
        df["sell_price"] = np.nan

    df["sell_price"] = (
        df.groupby("series_id")["sell_price"]
        .transform(lambda x: x.ffill().bfill())
        .fillna(0)
    )

    # Ensure complete target values
    df["sales"] = df["sales"].fillna(0).astype(float)

    max_prediction_length = 28
    max_encoder_length = 56

    training_cutoff = (
        df["time_idx"].max() - max_prediction_length
    )

    training_data = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_prediction_length=max_prediction_length,
        static_categoricals=["series_id"],
        time_varying_known_categoricals=[
            "day_of_week",
            "month",
            "weekend"
        ],
        time_varying_known_reals=[
            "time_idx",
            "snap_CA",
            "snap_TX",
            "snap_WI"
        ],
        time_varying_unknown_reals=[
            "sales",
            "sell_price"
        ],
        target_normalizer=GroupNormalizer(
            groups=["series_id"],
            transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False
    )

    validation_data = TimeSeriesDataSet.from_dataset(
        training_data,
        df,
        predict=True,
        stop_randomization=True
    )

    train_loader = training_data.to_dataloader(
        train=True,
        batch_size=32,
        num_workers=0
    )

    val_loader = validation_data.to_dataloader(
        train=False,
        batch_size=32,
        num_workers=0
    )

    print("Creating Temporal Fusion Transformer...")

    model = TemporalFusionTransformer.from_dataset(
        training_data,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=MAE(),
        optimizer="Adam",
        reduce_on_plateau_patience=3
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        enable_model_summary=True
    )

    print("Starting TFT training...")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print("TFT training completed!")

    # Save trained model checkpoint
    Path("artifacts").mkdir(exist_ok=True)

    trainer.save_checkpoint(
        "artifacts/tft_model.ckpt"
    )

    print("Saved: artifacts/tft_model.ckpt")

    # Generate validation predictions
    predictions = model.predict(
        val_loader,
        return_y=True,
        trainer_kwargs={
            "accelerator": "auto",
            "devices": 1
        }
    )

    prediction_values = predictions.output.detach().cpu().numpy()
    actual_values = predictions.y[0].detach().cpu().numpy()

    mae = np.mean(
        np.abs(actual_values - prediction_values)
    )

    rmse = np.sqrt(
        np.mean((actual_values - prediction_values) ** 2)
    )

    print("\n========== TFT RESULTS ==========")
    print("MAE:", round(float(mae), 4))
    print("RMSE:", round(float(rmse), 4))

    print("\nTFT forecasting completed!")


if __name__ == "__main__":
    train_tft()
