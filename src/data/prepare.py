
import pandas as pd
from pathlib import Path


def load_m5(raw_dir="data/raw", max_series=None):

    raw_dir = Path(raw_dir)

    sales_path = raw_dir / "sales_train_validation.csv"
    calendar_path = raw_dir / "calendar.csv"
    prices_path = raw_dir / "sell_prices.csv"

    # Check required files
    for path in [sales_path, calendar_path, prices_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {path}"
            )

    print("Loading M5 sales data...")

    sales = pd.read_csv(sales_path)

    # Select a subset of product-store series for development
    if max_series is not None:
        sales = sales.head(max_series).copy()

    print("Selected sales records:", len(sales))

    # Identify daily sales columns
    day_columns = [
        col for col in sales.columns
        if col.startswith("d_")
    ]

    # Convert wide sales data into long format
    sales_long = sales.melt(
        id_vars=[
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id"
        ],
        value_vars=day_columns,
        var_name="d",
        value_name="sales"
    )

    print("Sales data converted to long format.")

    # Load calendar
    calendar = pd.read_csv(calendar_path)

    # Load selling prices
    prices = pd.read_csv(prices_path)

    # Merge calendar information
    df = sales_long.merge(
        calendar,
        on="d",
        how="left"
    )

    # Merge weekly selling prices
    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Sort records
    df = df.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    print("M5 dataset loaded successfully.")
    print("Final dataset shape:", df.shape)

    return df


def save_panel(df, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("Dataset saved to:", output_path)

    return output_path
