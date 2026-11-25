from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output


# ==========================================================
# CONFIGURATION
# ==========================================================

OUTPUT_DIR = Path("outputs")

app = Dash(
    __name__,
    title="DemandIQ AI",
    suppress_callback_exceptions=True
)

server = app.server
app.title = "DemandIQ AI"


# ==========================================================
# DATA LOADING
# ==========================================================

def load_csv(filename):

    path = OUTPUT_DIR / filename

    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as error:
            print(f"Error loading {filename}: {error}")

    return pd.DataFrame()


forecasts = load_csv("lightgbm_predictions.csv")
inventory = load_csv("inventory_recommendations.csv")
comparison = load_csv("model_comparison.csv")


# ==========================================================
# COLORS
# ==========================================================

COLORS = {
    "primary": "#2563EB",
    "dark": "#172554",
    "background": "#F1F5F9",
    "white": "#FFFFFF",
    "text": "#334155",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "green": "#16A34A",
    "orange": "#F59E0B",
}


# ==========================================================
# REUSABLE COMPONENTS
# ==========================================================

def metric_card(title, value, subtitle, color):

    return html.Div(
        [
            html.Div(
                title,
                className="metric-title"
            ),

            html.H2(
                value,
                className="metric-value",
                style={"color": color}
            ),

            html.P(
                subtitle,
                className="metric-subtitle"
            )
        ],
        className="metric-card"
    )


def chart_card(title, graph):

    return html.Div(
        [
            html.H3(title, className="section-title"),
            graph
        ],
        className="dashboard-card"
    )


def empty_figure(message):

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": COLORS["muted"]}
    )

    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis={"visible": False},
        yaxis={"visible": False}
    )

    return fig


# ==========================================================
# DATA HELPERS
# ==========================================================

def get_total_series():

    if "id" not in forecasts.columns:
        return 0

    return forecasts["id"].nunique()


def get_total_forecast():

    if "prediction" not in forecasts.columns:
        return 0

    return round(
        pd.to_numeric(
            forecasts["prediction"],
            errors="coerce"
        ).sum(),
        2
    )


def get_total_orders():

    if "recommended_order" not in inventory.columns:
        return 0

    return int(
        pd.to_numeric(
            inventory["recommended_order"],
            errors="coerce"
        ).fillna(0).sum()
    )


# ==========================================================
# APP LAYOUT
# ==========================================================

app.layout = html.Div(

    className="app-container",

    children=[

        # HEADER
        html.Div(
            [

                html.Div(
                    [
                        html.H1(
                            "DemandIQ AI",
                            className="app-title"
                        ),

                        html.P(
                            "Retail Demand Forecasting & Inventory Optimization",
                            className="app-subtitle"
                        )
                    ]
                ),

                html.Div(
                    [
                        html.Span(
                            "●",
                            style={"color": COLORS["green"]}
                        ),

                        html.Span(
                            " System Dashboard",
                            className="status-text"
                        )
                    ],
                    className="system-status"
                )

            ],
            className="header"
        ),

        # KPI SECTION
        html.Div(
            id="kpi-cards",
            className="kpi-grid"
        ),

        # FILTER SECTION
        html.Div(
            [

                html.Label(
                    "Select Product / Store Series",
                    className="filter-label"
                ),

                dcc.Dropdown(
                    id="series-dropdown",
                    options=[],
                    placeholder="Select a product / store series",
                    clearable=True,
                    className="series-dropdown"
                )

            ],
            className="filter-card"
        ),

        # FORECAST SECTION
        html.Div(
            [
                chart_card(
                    "Demand Forecast",
                    dcc.Graph(
                        id="forecast-chart",
                        config={
                            "responsive": True,
                            "displayModeBar": False
                        },
                        style={"width": "100%"}
                    )
                )
            ],
            className="dashboard-grid"
        ),

        # MODEL PERFORMANCE
        html.Div(
            [
                chart_card(
                    "Model Performance",
                    dcc.Graph(
                        id="model-chart",
                        config={
                            "responsive": True,
                            "displayModeBar": False
                        },
                        style={"width": "100%"}
                    )
                )
            ],
            className="dashboard-grid"
        ),

        # INVENTORY SECTION
        html.Div(
            [

                html.H3(
                    "Inventory Recommendations",
                    className="section-title"
                ),

                html.Div(
                    id="inventory-table",
                    className="table-container"
                )

            ],
            className="dashboard-card"
        ),

        # FOOTER
        html.Footer(
            "DemandIQ AI | Intelligent Retail Forecasting & Inventory Optimization",
            className="footer"
        )

    ]
)


# ==========================================================
# KPI CALLBACK
# ==========================================================

@app.callback(
    Output("kpi-cards", "children"),
    Input("series-dropdown", "value")
)
def update_kpis(selected_series):

    forecast_data = forecasts.copy()
    inventory_data = inventory.copy()

    # Filter selected product/store
    if selected_series:
        forecast_data = forecast_data[
            forecast_data["id"].astype(str) == str(selected_series)
        ]

        inventory_data = inventory_data[
            inventory_data["id"].astype(str) == str(selected_series)
        ]

    # 1. Total forecast series
    total_series = forecast_data["id"].nunique()

    # 2. Total predicted demand
    total_forecast = pd.to_numeric(
        forecast_data["prediction"],
        errors="coerce"
    ).fillna(0).sum()

    # 3. Total recommended orders
    total_orders = pd.to_numeric(
        inventory_data["recommended_order"],
        errors="coerce"
    ).fillna(0).sum()

    return [

        metric_card(
            "Forecast Series",
            f"{total_series:,}",
            "Number of product/store series",
            COLORS["primary"]
        ),

        metric_card(
            "Predicted Demand",
            f"{total_forecast:,.2f}",
            "Total predicted units",
            COLORS["green"]
        ),

        metric_card(
            "Recommended Orders",
            f"{total_orders:,.0f}",
            "Units to replenish",
            COLORS["orange"]
        )

    ]






# @app.callback(
#     Output("kpi-cards", "children"),
#     Input("series-dropdown", "value")
# )
# def update_kpis(selected_series):

#     forecast_data = forecasts.copy()
#     inventory_data = inventory.copy()

#     if selected_series:

#         if "id" in forecast_data.columns:
#             forecast_data = forecast_data[
#                 forecast_data["id"].astype(str) == str(selected_series)
#             ]

#         if "id" in inventory_data.columns:
#             inventory_data = inventory_data[
#                 inventory_data["id"].astype(str) == str(selected_series)
#             ]

#     total_series = (
#         forecast_data["id"].nunique()
#         if "id" in forecast_data.columns else 0
#     )

#     total_forecast = (
#         pd.to_numeric(
#             forecast_data["prediction"],
#             errors="coerce"
#         ).sum()
#         if "prediction" in forecast_data.columns else 0
#     )

#     total_orders = (
#         pd.to_numeric(
#             inventory_data["recommended_order"],
#             errors="coerce"
#         ).fillna(0).sum()
#         if "recommended_order" in inventory_data.columns else 0
#     )

#     return [

#         metric_card(
#             "Forecast Series",
#             f"{total_series:,}",
#             "Selected product/store series",
#             COLORS["primary"]
#         ),

#         metric_card(
#             "Predicted Demand",
#             f"{total_forecast:,.0f}",
#             "Total forecasted units",
#             COLORS["green"]
#         ),

#         metric_card(
#             "Recommended Orders",
#             f"{total_orders:,.0f}",
#             "Units to replenish",
#             COLORS["orange"]
#         )

#     ]


# ==========================================================
# DROPDOWN CALLBACK
# ==========================================================

@app.callback(
    Output("series-dropdown", "options"),
    Input("series-dropdown", "id")
)
def update_dropdown(_):

    if forecasts.empty or "id" not in forecasts.columns:
        return []

    series_list = sorted(
        forecasts["id"].dropna().astype(str).unique()
    )

    return [
        {
            "label": series,
            "value": series
        }
        for series in series_list
    ]


# ==========================================================
# FORECAST CHART CALLBACK
# ==========================================================

@app.callback(
    Output("forecast-chart", "figure"),
    Input("series-dropdown", "value")
)
def update_forecast(selected_series):

    if forecasts.empty:
        return empty_figure("Forecast data not found")

    if not {"date", "prediction"}.issubset(forecasts.columns):
        return empty_figure("Required forecast columns are missing")

    data = forecasts.copy()

    if selected_series and "id" in data.columns:

        data = data[
            data["id"].astype(str) == str(selected_series)
        ]

    if data.empty:
        return empty_figure("No forecast data available")

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["prediction"] = pd.to_numeric(
        data["prediction"],
        errors="coerce"
    )

    data = data.dropna(subset=["date", "prediction"])

    fig = px.line(
        data,
        x="date",
        y="prediction",
        markers=True,
        title="Predicted Daily Demand",
        labels={
            "date": "Date",
            "prediction": "Demand Units"
        }
    )

    fig.update_traces(
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=6)
    )

    if "sales" in data.columns:

        data["sales"] = pd.to_numeric(
            data["sales"],
            errors="coerce"
        )

        fig.add_scatter(
            x=data["date"],
            y=data["sales"],
            mode="lines",
            name="Actual Sales",
            line=dict(
                color=COLORS["orange"],
                width=2,
                dash="dot"
            )
        )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=400,
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    return fig


# ==========================================================
# MODEL PERFORMANCE CALLBACK
# ==========================================================

@app.callback(
    Output("model-chart", "figure"),
    Input("series-dropdown", "value")
)
def update_model_chart(selected_series):

    if comparison.empty:
        return empty_figure("Model comparison data not found")

    comparison.columns = comparison.columns.str.strip().str.lower()

    if not {"model", "wape"}.issubset(comparison.columns):
        return empty_figure("Expected model/WAPE columns are missing")

    
    fig = px.bar(
    comparison,
    x="model",
    y="wape",
    title="Model Comparison – WAPE",
    text_auto=".2f",
    labels={
        "model": "Model",
        "wape": "WAPE (%)"
    }
    )

    fig.update_traces(
        marker_color=COLORS["primary"],
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        height=350,
        margin=dict(l=20, r=20, t=50, b=30)
    )

    return fig


# ==========================================================
# INVENTORY TABLE CALLBACK
# ==========================================================

@app.callback(
    Output("inventory-table", "children"),
    Input("series-dropdown", "value")
)
def update_inventory(selected_series):

    if inventory.empty:
        return html.P(
            "Inventory recommendations not found. Run inventory optimization first.",
            className="empty-message"
        )

    data = inventory.copy()

    if selected_series and "id" in data.columns:

        data = data[
            data["id"].astype(str) == str(selected_series)
        ]

    if data.empty:
        return html.P(
            "No inventory recommendations available.",
            className="empty-message"
        )

    columns = [
        "id",
        "demand",
        "current_inventory",
        "recommended_order",
        "expected_shortage",
        "expected_leftover"
    ]

    available_columns = [
        col for col in columns
        if col in data.columns
    ]

    header = html.Thead(
        html.Tr([
            html.Th(col.replace("_", " ").title())
            for col in available_columns
        ])
    )

    body = html.Tbody([
        html.Tr([
            html.Td(
                f"{row[col]:,.2f}"
                if isinstance(row[col], (int, float))
                else str(row[col])
            )
            for col in available_columns
        ])
        for _, row in data.head(20).iterrows()
    ])

    return html.Div(
        html.Table(
            [header, body],
            className="inventory-table"
        ),
        className="table-scroll"
    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        use_reloader=False,
        port=8050
    )
