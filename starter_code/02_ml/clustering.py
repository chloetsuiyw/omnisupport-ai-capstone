"""Unsupervised learning starter.
TODO: define a meaningful segmentation unit, preprocess features, compare clustering choices, and validate interpretation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_audit import load_full_dataset, clean_and_prepare
from feature_engineering import build_features

def build_customer_features(df):
    """Aggregate ticket-level data into one row per customer."""
    agg = df.groupby("customer_id").agg(
        total_tickets=("ticket_id", "count"),
        avg_order_value=("order_value_capped", "mean"),
        avg_delivery_delay=("delivery_delay_days", "mean"),
        escalation_rate=("escalated", "mean"),
        avg_resolution_time=("resolution_time_hours", "mean"),
        avg_tenure_months=("customer_tenure_months", "mean"),
        refund_rate=("refund_requested", "mean"),
    ).reset_index()
    return agg

def prepare_for_clustering(customer_df):
    feature_cols = ["total_tickets", "avg_order_value", "avg_delivery_delay",
                     "escalation_rate", "avg_resolution_time", "avg_tenure_months", "refund_rate"]
    X = customer_df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, feature_cols

def compare_k_values(X_scaled, k_values=[2, 3, 4, 5, 6]):
    results = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels, sample_size=20000, random_state=42)
        results.append({"k": k, "inertia": kmeans.inertia_, "silhouette": sil_score})
        print(f"k={k}: inertia={kmeans.inertia_:.1f}, silhouette={sil_score:.4f}")
    return pd.DataFrame(results)

def build_final_clusters(customer_df, X_scaled, k=4):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    customer_df = customer_df.copy()
    customer_df["cluster"] = labels
    return customer_df

def profile_clusters(customer_df, feature_cols):
    profile = customer_df.groupby("cluster")[feature_cols].mean()
    profile["count"] = customer_df.groupby("cluster").size()
    return profile

def build_segments(df):
    raise NotImplementedError("Student task")

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = load_full_dataset()
    df = clean_and_prepare(df)
    df = build_features(df)

    customer_df = build_customer_features(df)
    print("Number of unique customers:", customer_df.shape[0])
    print(customer_df.describe())

    X_scaled, feature_cols = prepare_for_clustering(customer_df)
    print("\n--- Comparing k values ---")
    k_results = compare_k_values(X_scaled)

    print("\n--- Final clustering (k=4) ---")
    customer_df = build_final_clusters(customer_df, X_scaled, k=4)
    profile = profile_clusters(customer_df, feature_cols)
    print(profile)

    long_tail_customers = df[df["resolution_status_after_7d"] == "open_after_7d"]["customer_id"].unique()
    cluster3_customers = customer_df[customer_df["cluster"] == 3]["customer_id"]
    overlap = cluster3_customers.isin(long_tail_customers).mean()
    print(f"\nCluster 3 overlap with open_after_7d customers: {overlap:.2%}")