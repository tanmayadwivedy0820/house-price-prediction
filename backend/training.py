"""
training.py
-----------
Production training script for the California Housing price model.
Extracted from the exploratory notebook: keeps only the path that
builds, evaluates, and SAVES the final model. All EDA / plotting is
intentionally dropped — that belongs in the notebook, not in production.

Run from the backend/ directory:
    python training.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)


def train_model():
    # --- Config from environment (single source of truth) ---
    load_dotenv()
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
    DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR", "dataset") / os.getenv("DATASET_NAME", "housing.csv")
    MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR", "model_dir") / os.getenv("MODEL_NAME", "house_price_model.joblib")
    TARGET_COL = os.getenv("TARGET_COL", "median_house_value")
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- Separate features and target ---
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- Identify column types (data-driven, not hardcoded) ---
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Numerical: {numerical_features}")
    print(f"Categorical: {categorical_features}")

    # --- Preprocessing pipeline (baked into the model so the API sends raw values) ---
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # --- Final model: tuned HistGradientBoostingRegressor (params from notebook GridSearchCV) ---
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            l2_regularization=0.1,
            learning_rate=0.1,
            max_depth=6,
            max_leaf_nodes=63,
            min_samples_leaf=20,
        )),
    ])

    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- Evaluation ---
    for split_name, Xs, ys in [("TRAIN", X_train, y_train), ("TEST", X_test, y_test)]:
        pred = model.predict(Xs)
        rmse = root_mean_squared_error(ys, pred)
        mae = mean_absolute_error(ys, pred)
        r2 = r2_score(ys, pred)
        print(f"=== {split_name} === RMSE: {rmse:,.2f} | MAE: {mae:,.2f} | R2: {r2:.4f}")

    # --- Save the FULL pipeline (preprocessing + model together) ---
    dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
