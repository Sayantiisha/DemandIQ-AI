
import pandas as pd
import plotly.express as px

from pathlib import Path


def visualize_results():

    file_path = Path(
        "outputs/model_comparison.csv"
    )

    if not file_path.exists():
        print("Run compare_models.py first.")
        return

    df = pd.read_csv(file_path)

    fig = px.bar(
        df,
        x="Model",
        y="WAPE",
        color="Model",
        title="DemandIQ AI - Model Performance Comparison",
        labels={
            "WAPE": "Weighted Absolute Percentage Error (%)",
            "Model": "Forecasting Model"
        },
        text_auto=".2f"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        yaxis_title="WAPE (%)"
    )

    fig.show()

    fig.write_html(
        "outputs/model_comparison.html"
    )

    print(
        "Chart saved: outputs/model_comparison.html"
    )


if __name__ == "__main__":
    visualize_results()
