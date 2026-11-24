
import pandas as pd


def validate_totals():

    bottom = pd.read_csv(
        "outputs/forecast_bottom.csv"
    )

    total = pd.read_csv(
        "outputs/forecast_total.csv"
    )

    calculated = (
        bottom.groupby("date")["prediction"]
        .sum()
        .sort_index()
    )

    stored = (
        total.set_index("date")["prediction"]
        .sort_index()
    )

    difference = (calculated - stored).abs()

    print("Maximum reconciliation difference:")
    print(difference.max())

    if difference.max() < 1e-6:
        print("PASS: Forecast totals are consistent.")
    else:
        print("FAIL: Forecast totals are inconsistent.")


if __name__ == "__main__":
    validate_totals()
