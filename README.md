# DemandIQ AI
## Retail Demand Forecasting & Inventory Optimization System

DemandIQ AI is an end-to-end machine learning project that forecasts retail product demand across multiple stores and supports inventory ordering decisions.

### Project Overview

The system combines historical sales data, feature engineering, machine learning forecasting, hierarchical aggregation, prediction intervals, and mathematical optimization to support retail inventory planning.

### Key Features

- Retail demand forecasting using LightGBM and Prophet.
- Baseline forecasting models for comparison.
- Time-series feature engineering with lag and rolling statistics.
- Forecast evaluation using MAE, RMSE, and WAPE.
- Hierarchical forecast aggregation.
- Prediction interval estimation.
- Inventory order optimization using Google OR-Tools.
- Interactive analytics dashboard using Plotly Dash.

### Dataset

**M5 Forecasting – Accuracy**

Source: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Required files:
- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

Download the dataset from Kaggle and place the files in `data/raw/`.

The dataset is not included in this repository.

### Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Forecasting | LightGBM, Prophet |
| Deep Learning Extension | PyTorch Forecasting |
| Optimization | Google OR-Tools |
| Visualization | Plotly, Dash |
| Testing | Pytest |
| Version Control | Git, GitHub |

### Project Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DemandIQ_AI
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the M5 dataset and place the required CSV files in `data/raw/`.

### Run the Project

Run the forecasting pipeline:

```bash
python -m src.pipeline --max-series 100 --horizon 28
```

Run inventory optimization:

```bash
python -m src.optimization.inventory
```

Start the dashboard:

```bash
python -m src.dashboard.app
```

Open `http://127.0.0.1:8050` in your browser.

### Testing

Run automated tests:

```bash
pytest -v
```

### Important Limitations

- Initial development uses a subset of the M5 dataset.
- Inventory levels, incoming stock, and cost values are illustrative assumptions because these are not supplied by M5.
- Forecast reconciliation currently uses bottom-up aggregation.
- Prediction intervals require independent calibration before their coverage can be treated as reliable.
- Advanced TFT and Darts integrations require further implementation and validation.

### Project Status

Portfolio development project. Model performance, deployment readiness, and end-to-end functionality should be verified using the actual execution results before production use.