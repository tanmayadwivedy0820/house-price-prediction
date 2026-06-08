"""
main.py — FastAPI service for the House Price model.

Endpoints:
  GET  /health         — health check
  POST /predict        — predict median_house_value from 9 inputs
  GET  /dataset-stats  — EDA summary (rows, target stats, feature distributions)
  GET  /model-info     — model config + permutation feature importances

Run:  uvicorn main:app --reload
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import load
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance

load_dotenv()
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR", "dataset") / os.getenv("DATASET_NAME", "housing.csv")
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR", "model_dir") / os.getenv("MODEL_NAME", "house_price_model.joblib")
TARGET_COL = os.getenv("TARGET_COL", "median_house_value")

model = load(MODEL_PATH)

app = FastAPI(title="House Price Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_info_cache = None


class HouseInput(BaseModel):
    longitude: float = Field(..., json_schema_extra={"example": -122.23})
    latitude: float = Field(..., json_schema_extra={"example": 37.88})
    housing_median_age: float = Field(..., json_schema_extra={"example": 41})
    total_rooms: float = Field(..., json_schema_extra={"example": 880})
    total_bedrooms: Optional[float] = Field(None, json_schema_extra={"example": 129})
    population: float = Field(..., json_schema_extra={"example": 322})
    households: float = Field(..., json_schema_extra={"example": 126})
    median_income: float = Field(..., json_schema_extra={"example": 8.3252})
    ocean_proximity: str = Field(..., json_schema_extra={"example": "NEAR BAY"})


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: HouseInput):
    row = pd.DataFrame([payload.model_dump()])
    try:
        prediction = float(model.predict(row)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
    return {
        "predicted_value": round(prediction, 2),
        "currency": "USD",
        "note": "Estimated median house value for the given block group.",
    }


@app.get("/dataset-stats")
def dataset_stats():
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Dataset not found on server.")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    target = df[TARGET_COL]
    target_summary = {
        "min": float(target.min()), "max": float(target.max()),
        "mean": round(float(target.mean()), 2), "median": float(target.median()),
    }
    t_binned = target.value_counts(bins=20, sort=False)
    target_hist = {
        "bin_edges": [round(float(i.left), 1) for i in t_binned.index] + [round(float(t_binned.index[-1].right), 1)],
        "counts": [int(c) for c in t_binned.values],
    }

    describe = df[numeric_cols].describe().round(3)
    numeric_summary = {col: describe[col].to_dict() for col in numeric_cols}

    categorical_summary = {
        col: {str(k): int(v) for k, v in df[col].value_counts().items()}
        for col in cat_cols
    }

    return {
        "n_rows": int(df.shape[0]),
        "n_features": len(feature_cols),
        "numeric_features": numeric_cols,
        "categorical_features": cat_cols,
        "target_summary": target_summary,
        "target_histogram": target_hist,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }


@app.get("/model-info")
def model_info():
    global _model_info_cache
    if _model_info_cache is not None:
        return _model_info_cache

    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=[TARGET_COL])

    sample = df.sample(n=min(2000, len(df)), random_state=42)
    Xs = sample.drop(columns=[TARGET_COL])
    ys = sample[TARGET_COL]

    result = permutation_importance(
        model, Xs, ys, n_repeats=5, random_state=42, scoring="r2", n_jobs=1
    )
    ranked = sorted(
        zip(X.columns.tolist(), result.importances_mean),
        key=lambda p: p[1], reverse=True,
    )

    regressor = model.named_steps["model"]
    _model_info_cache = {
        "model_type": type(regressor).__name__,
        "n_features": X.shape[1],
        "importance_method": "permutation (R2 drop, 5 repeats, 2000-row sample)",
        "feature_importances": [
            {"feature": name, "importance": round(float(imp), 5)}
            for name, imp in ranked
        ],
    }
    return _model_info_cache
