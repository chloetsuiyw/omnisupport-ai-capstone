"""Regression starter for a business-relevant continuous target.
TODO: choose features, preprocessing, baseline, models, metrics, and error analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor

from data_audit import load_full_dataset, clean_and_prepare
from feature_engineering import build_features, get_feature_columns

def load_model_ready_data():
    df = load_full_dataset()
    df = clean_and_prepare(df)
    df = build_features(df)
    X = df[get_feature_columns()]
    y = df["resolution_time_hours"]
    return X, y

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_baseline(y_train, y_val):
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    preds = dummy.predict(np.zeros((len(y_val), 1)))
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return {"mae": mae, "rmse": rmse}

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["str", "object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ], remainder="passthrough")
    return preprocessor, categorical_cols, numeric_cols

def train_rf_model(X_train, y_train, X_val, y_val):
    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return pipeline, {"mae": mae, "rmse": rmse}

def get_feature_importances(pipeline, X_train):
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    cat_cols = preprocessor.transformers_[0][2]
    ohe = preprocessor.named_transformers_["cat"]
    ohe_feature_names = ohe.get_feature_names_out(cat_cols)

    remainder_cols = [c for c in X_train.columns if c not in cat_cols]
    all_feature_names = list(ohe_feature_names) + remainder_cols

    importances = pd.Series(model.feature_importances_, index=all_feature_names)
    return importances.sort_values(ascending=False)

def segment_error_analysis(pipeline, X_val, y_val, segment_col):
    preds = pipeline.predict(X_val)
    errors = pd.DataFrame({
        "segment": X_val[segment_col].values,
        "actual": y_val.values,
        "predicted": preds,
        "abs_error": np.abs(y_val.values - preds),
    })
    summary = errors.groupby("segment")["abs_error"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    return summary

def tune_max_depth(X_train, y_train, X_val, y_val, depths=[6, 12, 20, None]):
    results = []
    preprocessor, _, _ = build_preprocessor(X_train)
    for depth in depths:
        model = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=42, n_jobs=-1)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        results.append({"max_depth": depth, "mae": mae, "rmse": rmse})
        print(f"max_depth={depth}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    return pd.DataFrame(results)

def train_regression_model(df):
    raise NotImplementedError("Student task")

if __name__ == "__main__":
    X, y = load_model_ready_data()
    print("Feature matrix shape:", X.shape)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))

    baseline_scores = train_baseline(y_train, y_val)
    print("\n--- Baseline (predict mean) ---")
    print(baseline_scores)

    print("\n--- Random Forest ---")
    rf_pipeline, rf_scores = train_rf_model(X_train, y_train, X_val, y_val)
    print(rf_scores)

    print("\n--- Feature Importances (top 15) ---")
    importances = get_feature_importances(rf_pipeline, X_train)
    print(importances.head(15))

    print("\n--- Error by issue_category ---")
    print(segment_error_analysis(rf_pipeline, X_val, y_val, "issue_category"))

    print("\n--- Error by priority ---")
    print(segment_error_analysis(rf_pipeline, X_val, y_val, "priority"))

    print("\n--- Tuning: max_depth comparison ---")
    tuning_results = tune_max_depth(X_train, y_train, X_val, y_val)
    print(tuning_results)