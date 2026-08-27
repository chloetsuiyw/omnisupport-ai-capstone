"""Start your data-quality audit here.

TODO: quantify missingness, duplicates, invalid categories, outliers, and possible leakage.
"""

from pathlib import Path
import pandas as pd

def load_full_dataset():
    data_dir = Path("data/raw")
    dfs = [pd.read_parquet(f) for f in sorted(data_dir.glob("support_records_part_*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    return df

def check_missingness(df):
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    result = pd.DataFrame({"missing_count": missing_counts, "missing_pct": missing_pct})
    result = result[result["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    return result

def check_blank_strings(df):
    text_cols = df.select_dtypes(include="str").columns.tolist()
    results = {}
    for col in text_cols:
        blank_count = (df[col].astype(str).str.strip() == "").sum()
        if blank_count > 0:
            results[col] = blank_count
    return pd.Series(results, name="blank_count").sort_values(ascending=False)

def check_image_id_consistency(df):
    blank_image = (df["image_id"].astype(str).str.strip() == "")
    return pd.crosstab(df["attachment_available"], blank_image)

def check_image_linkage(df):
    linked = df[df["image_id"].astype(str).str.strip() != ""]
    return linked["image_id"].nunique(), len(linked)

def check_duplicates(df):
    full_dupes = df.duplicated().sum()
    id_dupes = df["ticket_id"].duplicated().sum()
    return {"full_row_duplicates": int(full_dupes), "ticket_id_duplicates": int(id_dupes)}

def check_category_values(df, columns):
    results = {}
    for col in columns:
        results[col] = df[col].value_counts(dropna=False)
    return results

def clean_categories(df):
    df = df.copy()
    df["support_channel"] = df["support_channel"].replace({"Web Chat": "web_chat"})
    df["customer_region"] = df["customer_region"].replace({"london": "London"})
    df["product_category"] = df["product_category"].replace({"Electronics": "electronics"})
    return df

def check_outliers(df, columns):
    results = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        results[col] = {
            "q1": q1, "q3": q3, "lower_bound": lower, "upper_bound": upper,
            "outlier_count": len(outliers),
            "outlier_pct": round(len(outliers) / len(df) * 100, 2),
            "min": df[col].min(), "max": df[col].max()
        }
    return pd.DataFrame(results).T

def check_target_balance(df, col):
    counts = df[col].value_counts()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})

def clean_and_prepare(df):
    """Apply confirmed cleaning steps: fix category casing, drop exact duplicates,
    and cap order_value at the 99th percentile (keeping a flag) for modeling."""
    df = clean_categories(df)
    df = df.drop_duplicates()

    cap_value = df["order_value"].quantile(0.99)
    df["order_value_was_capped"] = (df["order_value"] > cap_value).astype(int)
    df["order_value_capped"] = df["order_value"].clip(upper=cap_value)

    return df

def check_long_tail_resolution(df, threshold=300):
    long_tail = df[df["resolution_time_hours"] > threshold]
    print(f"Rows with resolution_time_hours > {threshold}: {len(long_tail)}")
    print("\nescalated breakdown:")
    print(long_tail["escalated"].value_counts(normalize=True))
    print("\nresolution_status_after_7d breakdown:")
    print(long_tail["resolution_status_after_7d"].value_counts(normalize=True))

def run_audit(df):
    """Run the full Phase 2 data-quality audit and return a single report dict."""
    report = {
        "shape": df.shape,
        "missingness_nan": check_missingness(df),
        "missingness_blank": check_blank_strings(df),
        "image_id_consistency": check_image_id_consistency(df),
        "image_id_linkage": check_image_linkage(df),
        "duplicates": check_duplicates(df),
        "categories": check_category_values(df, [
            "support_channel", "customer_region", "customer_age_band",
            "preferred_language", "product_category", "issue_category",
            "priority", "customer_sentiment", "resolution_status_after_7d",
            "final_agent_disposition"
        ]),
        "outliers": check_outliers(df, ["order_value", "delivery_delay_days", "resolution_time_hours"]),
        "target_balance": check_target_balance(df, "escalated"),
    }
    return report

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = load_full_dataset()
    print("Shape:", df.shape)

    print("\n--- Full audit report ---")
    report = run_audit(df)
    for key, value in report.items():
        print(f"\n{key}:")
        print(value)

    print("\n--- Cleaned dataset preview ---")
    df_clean = clean_and_prepare(df)
    print("Cleaned shape:", df_clean.shape)
    print(df_clean[["order_value", "order_value_capped", "order_value_was_capped"]].describe())
    print("\n--- Long-tail resolution time investigation ---")
    check_long_tail_resolution(df_clean, threshold=300)