
import pandas as pd
import pytest

from src.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_wape
)

from src.optimization.inventory import optimize_order_quantity


# -------------------------------
# 1. Forecasting metrics tests
# -------------------------------

def test_mae():

    actual = [100, 200, 300]
    predicted = [110, 190, 310]

    result = calculate_mae(actual, predicted)

    assert result == pytest.approx(10.0)


def test_rmse():

    actual = [100, 200, 300]
    predicted = [110, 190, 310]

    result = calculate_rmse(actual, predicted)

    assert result == pytest.approx(10.0)


def test_wape():

    actual = [100, 200, 300]
    predicted = [110, 190, 310]

    result = calculate_wape(actual, predicted)

    assert result == pytest.approx(
        (30 / 600) * 100
    )


# -------------------------------
# 2. Inventory optimization test
# -------------------------------

def test_inventory_optimization():

    result = optimize_order_quantity(
        demand=100,
        current_inventory=30,
        incoming_stock=10,
        unit_cost=1.0,
        holding_cost=0.5,
        shortage_cost=10.0
    )

    assert result["recommended_order"] >= 0

    assert (
        result["recommended_order"]
        + result["current_inventory"]
        + result["incoming_stock"]
        + result["expected_shortage"]
        - result["expected_leftover"]
    ) == pytest.approx(result["demand"])


# -------------------------------
# 3. Prediction data validation
# -------------------------------

def test_prediction_file():

    path = "outputs/lightgbm_predictions.csv"

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        pytest.skip("LightGBM predictions have not been generated.")

    required_columns = {
        "id",
        "date",
        "sales",
        "prediction"
    }

    assert required_columns.issubset(df.columns)

    assert not df.empty

    assert pd.to_numeric(
        df["prediction"],
        errors="coerce"
    ).notna().all()

    assert (df["prediction"] >= 0).all()


# -------------------------------
# 4. Inventory output validation
# -------------------------------

def test_inventory_output():

    path = "outputs/inventory_recommendations.csv"

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        pytest.skip("Inventory recommendations not generated.")

    required_columns = {
        "id",
        "demand",
        "recommended_order",
        "expected_shortage"
    }

    assert required_columns.issubset(df.columns)

    assert not df.empty

    assert (df["recommended_order"] >= 0).all()
