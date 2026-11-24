
from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


OUTPUT_DIR = Path("outputs")

app = Dash(__name__)
app.title = "DemandIQ AI"


def load_csv(filename):
    path = OUTPUT_DIR / filename

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


forecasts = load_csv("lightgbm_predictions.csv")
inventory = load_csv("inventory_recommendations.csv")
comparison = load_csv("model_comparison.csv")


def metric_card(title, value):
    return html.Div(
        [
            html.H4(title),
            html.H2(value)
        ],
        style={
            "backgroundColor": "#ffffff",
            "padding": "20px",
            "borderRadius": "10px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
            "flex": "1",
            "minWidth": "180px"
        }
    )


app.layout = html.Div(
    style={
        "backgroundColor": "#f4f6fa",
        "minHeight": "100vh",
        "padding": "25px",
        "fontFamily": "Arial"
    },
    children=[

        html.H1(
            "DemandIQ AI",
            style={"color": "#172554"}
        ),

        html.P(
            "Retail Demand Forecasting & Inventory Optimization"
        ),

        html.Hr(),

        html.Div(
            id="kpi-cards",
            style={
                "display": "flex",
                "gap": "15px",
                "flexWrap": "wrap"
            }
        ),

        html.Br(),

        html.Div(
            [
                html.Label("Select Product / Store Series"),

                dcc.Dropdown(
                    id="series-dropdown",
                    options=[],
                    placeholder="Select a series"
                )
            ],
            style={"maxWidth": "500px"}
        ),

        html.Br(),

        html.Div(
            [
                html.H3("Demand Forecast"),
                dcc.Graph(id="forecast-chart")
            ],
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px"
            }
        ),

        html.Br(),

        html.Div(
            [
                html.H3("Model Performance"),
                dcc.Graph(id="model-chart")
            ],
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px"
            }
        ),

        html.Br(),

        html.Div(
            [
                html.H3("Inventory Recommendations"),
                html.Div(id="inventory-table")
            ],
            style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px"
            }
        )
    ]
)


@app.callback(
    Output("kpi-cards", "children"),
    Input("series-dropdown", "value")
)
def update_kpis(selected_series):

    total_series = (
        forecasts["id"].nunique()
        if "id" in forecasts.columns else 0
    )

    total_forecast = (
        round(forecasts["prediction"].sum(), 2)
        if "prediction" in forecasts.columns else 0
    )

    total_orders = (
        int(inventory["recommended_order"].sum())
        if "recommended_order" in inventory.columns else 0
    )

    return [
        metric_card("Forecast Series", str(total_series)),
        metric_card("Total Predicted Demand", str(total_forecast)),
        metric_card("Recommended Order Units", str(total_orders))
    ]


@app.callback(
    Output("series-dropdown", "options"),
    Input("kpi-cards", "children")
)
def update_dropdown(_):

    if forecasts.empty or "id" not in forecasts.columns:
        return []

    return [
        {"label": series, "value": series}
        for series in sorted(forecasts["id"].astype(str).unique())
    ]


@app.callback(
    Output("forecast-chart", "figure"),
    Input("series-dropdown", "value")
)
def update_forecast(selected_series):

    if forecasts.empty:
        return px.line(title="Forecast data not found")

    data = forecasts.copy()

    if selected_series:
        data = data[
            data["id"].astype(str) == str(selected_series)
        ]

    data["date"] = pd.to_datetime(data["date"])

    fig = px.line(
        data,
        x="date",
        y="prediction",
        title="Predicted Daily Demand",
        markers=True
    )

    if "sales" in data.columns:
        fig.add_scatter(
            x=data["date"],
            y=data["sales"],
            mode="lines",
            name="Actual Sales"
        )

    return fig


@app.callback(
    Output("model-chart", "figure"),
    Input("series-dropdown", "value")
)
def update_model_chart(selected_series):

    if comparison.empty:
        return px.bar(title="Model comparison data not found")

    if not {"model", "WAPE"}.issubset(comparison.columns):
        return px.bar(title="Expected model/WAPE columns not found")

    return px.bar(
        comparison,
        x="model",
        y="WAPE",
        title="Model Comparison – WAPE",
        text_auto=".2f"
    )


@app.callback(
    Output("inventory-table", "children"),
    Input("series-dropdown", "value")
)
def update_inventory(selected_series):

    if inventory.empty:
        return html.P(
            "Inventory recommendations not found. Run inventory optimization first."
        )

    data = inventory.copy()

    if selected_series:
        data = data[
            data["id"].astype(str) == str(selected_series)
        ]

    columns = [
        "id",
        "demand",
        "current_inventory",
        "recommended_order",
        "expected_shortage",
        "expected_leftover"
    ]

    available_columns = [
        col for col in columns if col in data.columns
    ]

    return html.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th(col.replace("_", " ").title())
                    for col in available_columns
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(row[col])
                    for col in available_columns
                ])
                for _, row in data.head(20).iterrows()
            ])
        ],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "textAlign": "left"
        }
    )


if __name__ == "__main__":
    app.run(debug=True)

