
import pandas as pd
import os

# Dataset directory
DATA_PATH = "data/raw"

# File paths
sales_path = os.path.join(
    DATA_PATH, "sales_train_validation.csv"
)

calendar_path = os.path.join(
    DATA_PATH, "calendar.csv"
)

prices_path = os.path.join(
    DATA_PATH, "sell_prices.csv"
)


def inspect_sales():
    print("\n========== SALES DATA ==========")

    # Read a sample to avoid loading the entire dataset
    sales = pd.read_csv(sales_path, nrows=1000)

    print("\nShape of sample:", sales.shape)

    print("\nFirst five rows:")
    print(sales.head())

    print("\nColumn names:")
    print(sales.columns.tolist())

    print("\nProduct hierarchy:")
    for col in ["state_id", "store_id", "cat_id",
                "dept_id", "item_id"]:
        print(col, ":", sales[col].nunique())

    print("\nMissing values:")
    print(sales.isnull().sum().sum())


def inspect_calendar():
    print("\n========== CALENDAR DATA ==========")

    calendar = pd.read_csv(calendar_path)

    print("\nShape:", calendar.shape)

    print("\nFirst five rows:")
    print(calendar.head())

    print("\nColumns:")
    print(calendar.columns.tolist())

    print("\nMissing values:")
    print(calendar.isnull().sum())


def inspect_prices():
    print("\n========== SELL PRICES DATA ==========")

    prices = pd.read_csv(prices_path, nrows=10000)

    print("\nShape of sample:", prices.shape)

    print("\nFirst five rows:")
    print(prices.head())

    print("\nMissing values:")
    print(prices.isnull().sum())

    print("\nPrice statistics:")
    print(prices["sell_price"].describe())


if __name__ == "__main__":

    inspect_sales()
    inspect_calendar()
    inspect_prices()

    print("\nEDA completed successfully!")
