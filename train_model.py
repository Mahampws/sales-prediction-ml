"""
Sales Prediction Model
-----------------------
Trains a regression model to predict sales based on marketing spend, store
traffic, discounts, seasonality, and competitor pricing. This kind of model
helps businesses forecast revenue and evaluate "what-if" scenarios (e.g.
"what happens to sales if we increase the discount by 10%?").

Usage:
    python generate_data.py      # creates sales_data.csv (skip if using real data)
    python train_model.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def load_data(path="sales_data.csv"):
    return pd.read_csv(path)


def train_and_evaluate(df):
    X = df.drop(columns=["sales"])
    y = df["sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"model": model, "mae": mae, "r2": r2, "preds": preds}
        print(f"{name}: MAE={mae:,.2f}  R2={r2:.3f}")

    return results, X_test, y_test


def plot_feature_importance(model, feature_names, out="feature_importance.png"):
    importances = model.feature_importances_
    order = importances.argsort()[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in order], importances[order])
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(out)
    print(f"Saved feature importance chart -> {out}")


def main():
    df = load_data()
    results, X_test, y_test = train_and_evaluate(df)

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\nBest model: {best_name} (R2={results[best_name]['r2']:.3f})")

    rf_model = results["Random Forest"]["model"]
    plot_feature_importance(rf_model, list(X_test.columns))


if __name__ == "__main__":
    main()
