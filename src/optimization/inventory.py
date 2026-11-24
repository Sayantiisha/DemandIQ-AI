
import pandas as pd
from pathlib import Path
from ortools.linear_solver import pywraplp


def optimize_order_quantity(
    demand,
    current_inventory,
    incoming_stock,
    unit_cost,
    holding_cost,
    shortage_cost,
    max_order_quantity=10000
):
    """
    Optimize order quantity for one planning period.

    All costs are illustrative assumptions and should
    be replaced with real business data when available.
    """

    solver = pywraplp.Solver.CreateSolver("CBC")

    if solver is None:
        raise RuntimeError("OR-Tools CBC solver is unavailable.")

    # Integer decision variables
    order_qty = solver.IntVar(
        0,
        max_order_quantity,
        "order_quantity"
    )

    shortage = solver.NumVar(
        0,
        solver.infinity(),
        "shortage"
    )

    leftover = solver.NumVar(
        0,
        solver.infinity(),
        "leftover_stock"
    )

    available_stock = (
        current_inventory + incoming_stock + order_qty
    )

    # Inventory balance
    solver.Add(
        available_stock + shortage - leftover == demand
    )

    # Minimize purchasing, holding and shortage costs
    total_cost = (
        unit_cost * order_qty
        + holding_cost * leftover
        + shortage_cost * shortage
    )

    solver.Minimize(total_cost)

    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Could not find an optimal order quantity.")

    return {
        "demand": demand,
        "current_inventory": current_inventory,
        "incoming_stock": incoming_stock,
        "recommended_order": int(round(order_qty.solution_value())),
        "expected_shortage": round(shortage.solution_value(), 2),
        "expected_leftover": round(leftover.solution_value(), 2),
        "estimated_total_cost": round(solver.Objective().Value(), 2)
    }


def generate_recommendations():

    prediction_file = Path(
        "outputs/lightgbm_predictions.csv"
    )

    if not prediction_file.exists():
        raise FileNotFoundError(
            "Run LightGBM forecasting first."
        )

    predictions = pd.read_csv(prediction_file)

    predictions["date"] = pd.to_datetime(
        predictions["date"]
    )

    # Plan for the first 7 forecast days.
    forecast_start = predictions["date"].min()

    planning_data = predictions[
        predictions["date"] < (
            forecast_start + pd.Timedelta(days=7)
        )
    ]

    # Total seven-day predicted demand per product-store series.
    demand_by_series = (
        planning_data.groupby("id")["prediction"]
        .sum()
        .reset_index(name="forecast_demand")
    )

    recommendations = []

    for row in demand_by_series.itertuples():

        # Illustrative inventory and cost assumptions.
        # M5 does not provide actual on-hand inventory
        # or procurement and holding costs.
        result = optimize_order_quantity(
            demand=max(0, float(row.forecast_demand)),
            current_inventory=20,
            incoming_stock=0,
            unit_cost=1.0,
            holding_cost=0.5,
            shortage_cost=10.0,
            max_order_quantity=10000
        )

        result["id"] = row.id

        recommendations.append(result)

    result_df = pd.DataFrame(recommendations)

    Path("outputs").mkdir(exist_ok=True)

    result_df.to_csv(
        "outputs/inventory_recommendations.csv",
        index=False
    )

    print("\n========== INVENTORY RECOMMENDATIONS ==========")

    print(
        result_df.head(10).to_string(index=False)
    )

    print("\nRecommendations saved:")
    print("outputs/inventory_recommendations.csv")


if __name__ == "__main__":
    generate_recommendations()
