"""Classical classification starter.

TODO: define prediction-time features, split strategy, preprocessing pipeline, baselines, model comparison, tuning, threshold analysis, and evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, f1_score, classification_report
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from data_audit import load_full_dataset, clean_and_prepare
from feature_engineering import build_features, get_feature_columns

def load_model_ready_data():
    df = load_full_dataset()
    df = clean_and_prepare(df)
    df = build_features(df)
    X = df[get_feature_columns()]
    y = df["escalated"]
    return X, y

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_baseline(y_train, y_val):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    preds = dummy.predict(np.zeros((len(y_val), 1)))
    f1 = f1_score(y_val, preds)
    return {"f1": f1, "predicted_positive_rate": preds.mean()}

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["str", "object"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ], remainder="passthrough")
    return preprocessor

def train_rf_classifier(X_train, y_train, X_val, y_val):
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_val)[:, 1]
    preds = pipeline.predict(X_val)
    f1 = f1_score(y_val, preds)
    pr_auc = average_precision_score(y_val, probs)
    return pipeline, {"f1": f1, "pr_auc": pr_auc}

def threshold_table(pipeline, X_val, y_val, thresholds=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
    probs = pipeline.predict_proba(X_val)[:, 1]
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        pct_flagged = preds.mean()
        recall = (preds[y_val == 1] == 1).mean()  # % of true escalations caught
        precision = (y_val[preds == 1] == 1).mean() if preds.sum() > 0 else np.nan
        rows.append({
            "threshold": t,
            "pct_flagged": round(pct_flagged, 4),
            "escalations_caught_pct": round(recall, 4),
            "precision": round(precision, 4),
        })
    return pd.DataFrame(rows)

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
    df = pd.DataFrame({
        "segment": X_val[segment_col].values,
        "actual": y_val.values,
        "predicted": preds,
    })
    rows = []
    for seg, group in df.groupby("segment"):
        f1 = f1_score(group["actual"], group["predicted"]) if group["actual"].sum() > 0 else np.nan
        rows.append({"segment": seg, "f1": f1, "count": len(group), "escalation_rate": group["actual"].mean()})
    return pd.DataFrame(rows).sort_values("f1", ascending=False)

from sklearn.model_selection import cross_val_score, StratifiedKFold

def run_cross_validation(X, y, cv_folds=5):
    """K-fold cross-validation for the classification model, addressing the
    ml_evaluation_requirements.md need for CV on at least one classical ML comparison."""
    preprocessor = build_preprocessor(X)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced")),
    ])
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    return {"fold_scores": scores.tolist(), "mean_f1": scores.mean(), "std_f1": scores.std()}

def build_pipeline():
    """Build the classifier component of the pipeline (preprocessing is added
    once X is available, since column types must be inspected first)."""
    return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced")


def evaluate_classifier(model, X_test, y_test):
    """Evaluate a trained (pipeline) classifier on held-out test data."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "f1": f1_score(y_test, preds),
        "pr_auc": average_precision_score(y_test, probs),
    }

def demographic_subgroup_analysis(pipeline, X_val, y_val, demographic_col="customer_region"):
    preds = pipeline.predict(X_val)
    df = pd.DataFrame({
        "segment": X_val[demographic_col].values,
        "actual": y_val.values,
        "predicted": preds,
    })
    rows = []
    for seg, group in df.groupby("segment"):
        if group["actual"].sum() == 0:
            continue
        f1 = f1_score(group["actual"], group["predicted"])
        rows.append({"segment": seg, "f1": f1, "count": len(group), "escalation_rate": group["actual"].mean()})
    return pd.DataFrame(rows).sort_values("f1", ascending=False)

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    X, y = load_model_ready_data()
    print("Feature matrix shape:", X.shape)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))
    print("Train escalation rate:", y_train.mean().round(4))
    print("Val escalation rate:", y_val.mean().round(4))

    baseline_scores = train_baseline(y_train, y_val)
    print("\n--- Baseline (predict majority class) ---")
    print(baseline_scores)

    print("\n--- Random Forest Classifier ---")
    rf_pipeline, rf_scores = train_rf_classifier(X_train, y_train, X_val, y_val)
    print(rf_scores)

    probs = rf_pipeline.predict_proba(X_val)[:, 1]
    print("\nProbability distribution:")
    print(pd.Series(probs).describe())
    print("\n--- Threshold Table ---")
    print(threshold_table(rf_pipeline, X_val, y_val))
    print("\n--- Feature Importances (top 15) ---")
    importances = get_feature_importances(rf_pipeline, X_train)
    print(importances.head(15))
    print("\n--- F1 by issue_category ---")
    print(segment_error_analysis(rf_pipeline, X_val, y_val, "issue_category"))
    print("\n--- F1 by priority ---")
    print(segment_error_analysis(rf_pipeline, X_val, y_val, "priority"))

    print("\n--- 5-Fold Cross-Validation ---")
    cv_results = run_cross_validation(X, y, cv_folds=5)
    print(cv_results)

    print("\n--- Demographic Subgroup Analysis (customer_region) ---")
    print(demographic_subgroup_analysis(rf_pipeline, X_val, y_val, "customer_region"))

    print("\n--- Final Test-Set Evaluation (using build_pipeline/evaluate_classifier) ---")
    test_eval = evaluate_classifier(rf_pipeline, X_test, y_test)
    print(test_eval)