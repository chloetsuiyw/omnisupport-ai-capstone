"""Feature engineering — builds a leakage-safe feature set for both the
resolution_time_hours regression and escalated classification targets.
"""

import pandas as pd
from data_audit import load_full_dataset, clean_and_prepare

# Columns known only after ticket creation — excluded per Phase 1 leakage audit.
LEAKAGE_COLUMNS = [
    "resolution_status_after_7d",
    "final_agent_disposition",
    "refund_amount",
    "csat_score",
    "customer_sentiment",
]

def build_features(df):
    df = df.copy()

    # Timestamp-derived features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ticket_hour"] = df["timestamp"].dt.hour
    df["ticket_day_of_week"] = df["timestamp"].dt.dayofweek
    df["ticket_month"] = df["timestamp"].dt.month

    # Text-length features (cheap, useful signal ahead of full NLP work in Phase 8)
    df["issue_title_length"] = df["issue_title"].astype(str).str.len()
    df["issue_description_length"] = df["issue_description"].astype(str).str.len()

    # Fill structural/expected missingness
    df["customer_region"] = df["customer_region"].replace("", "Unknown")

    return df

def get_feature_columns():
    """Columns safe to use for BOTH regression and classification targets."""
    return [
        "support_channel", "customer_region", "customer_age_band", "preferred_language",
        "customer_tenure_months", "product_category", "order_value_capped",
        "order_value_was_capped", "delivery_delay_days", "previous_ticket_count",
        "issue_category", "priority", "attachment_available",
        "accessibility_support_flag", "ticket_hour", "ticket_day_of_week",
        "ticket_month", "issue_title_length", "issue_description_length",
    ]

if __name__ == "__main__":
    df = load_full_dataset()
    df = clean_and_prepare(df)
    df = build_features(df)
    print(df.shape)
    print(df[get_feature_columns()].head())