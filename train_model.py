
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import (
    make_features,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)

DATA_PATH = "data/custom_food_delivery_dataset.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = make_features(df)
    y = df[TARGET]

    # Keep the same test set for a fair model comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Fill missing numeric values with the median.
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Fill missing categories and convert them into numeric columns.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    models = {
        "Dummy Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    best_model = None
    best_mae = float("inf")
    best_name = None

    for name, regressor in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
        })

        print(f"MAE: {mae:.2f} minutes")
        print(f"RMSE: {rmse:.2f} minutes")
        print(f"R2 Score: {r2:.4f}")

        # Select the model with the lowest MAE.
        if mae < best_mae:
            best_mae = mae
            best_model = pipeline
            best_name = name

    results_df = pd.DataFrame(results).sort_values("MAE")

    results_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    model_path = os.path.join(MODEL_DIR, "delivery_time_model.joblib")
    joblib.dump(best_model, model_path)

    # Plot actual versus predicted delivery times.
    predictions = best_model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.4)
    plt.xlabel("Actual Delivery Time (minutes)")
    plt.ylabel("Predicted Delivery Time (minutes)")
    plt.title(f"Actual vs Predicted - {best_name}")

    lower = min(y_test.min(), predictions.min())
    upper = max(y_test.max(), predictions.max())
    plt.plot([lower, upper], [lower, upper], linestyle="--")

    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, "actual_vs_predicted.png")
    plt.savefig(plot_path)
    plt.close()

    print("\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))

    print(f"\nBest model: {best_name}")
    print(f"Best MAE: {best_mae:.2f} minutes")
    print(f"Saved model: {model_path}")
    print(f"Saved comparison: {results_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
