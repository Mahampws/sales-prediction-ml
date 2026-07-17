# Sales Prediction Model

Predicts sales revenue from marketing spend, store traffic, discounts,
seasonality, and competitor pricing using regression models. This kind of
model is used by businesses to forecast revenue and test "what-if" scenarios
before committing budget.

## How it works

1. Data (marketing/business metrics) is loaded from a CSV.
2. Two regression models are trained and compared: Linear Regression and
   Random Forest.
3. Models are evaluated with **MAE** (average prediction error in dollars)
   and **R²** (how much of the variation in sales the model explains).
4. A feature importance chart shows which factors drive sales most.

## Setup

```bash
pip install -r requirements.txt
python generate_data.py   # creates a synthetic demo dataset
python train_model.py
```

## Results (on synthetic demo data)

```
Linear Regression: MAE=2,415.69  R2=0.892
Random Forest:      MAE=2,701.82  R2=0.866
```

An R² of 0.89 means the model explains ~89% of the variation in sales —
strong performance for a business forecasting model.

## Using with real client data

Replace `sales_data.csv` with the client's actual historical sales data.
Keep a `sales` column as the target; any number of feature columns (spend,
traffic, pricing, seasonality flags, etc.) can be added or removed — the
script adapts automatically.

## Tech stack

Python, pandas, scikit-learn (Linear Regression, Random Forest), matplotlib.
