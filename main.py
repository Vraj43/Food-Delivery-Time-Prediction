
import pandas as pd
from pathlib import Path

# Dataset location
DATA_PATH = Path("data") / "custom_food_delivery_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n===== FIRST FIVE ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

target = "Time_taken (min)"

if target in df.columns:
    print("\n===== DELIVERY TIME STATISTICS =====")
    print(df[target].describe())
else:
    print(f"\nTarget column '{target}' was not found.")

from src.features import make_features, FEATURES, TARGET

X = make_features(df)
y = df[TARGET]

print("\n===== FEATURE PREPARATION =====")
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("\nPrepared features:")
print(X.head())
print("\nMissing values in prepared features:")
print(X.isnull().sum())
