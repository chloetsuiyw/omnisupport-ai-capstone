"""EDA visualisations for Phase 2 — resolution time, escalation rate, order value capping."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from data_audit import load_full_dataset, clean_and_prepare

OUTPUT_DIR = Path("outputs/01_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_resolution_time_distribution(df, save_path=OUTPUT_DIR / "resolution_time_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["resolution_time_hours"], bins=100)
    axes[0].set_title("Resolution Time (hours) - raw")
    axes[0].set_xlabel("Hours")

    axes[1].hist(df["resolution_time_hours"], bins=100)
    axes[1].set_yscale("log")
    axes[1].set_title("Resolution Time (hours) - log y-scale")
    axes[1].set_xlabel("Hours")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

def plot_escalation_rate_by_category(df, save_path=OUTPUT_DIR / "escalation_rate_by_category.png"):
    rate = df.groupby("issue_category")["escalated"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    rate.plot(kind="bar", ax=ax)
    ax.set_title("Escalation Rate by Issue Category")
    ax.set_ylabel("Escalation Rate")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

def plot_order_value_capping(df, save_path=OUTPUT_DIR / "order_value_capping.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(df["order_value"])
    axes[0].set_title("order_value - before capping")

    axes[1].boxplot(df["order_value_capped"])
    axes[1].set_title("order_value - after capping (99th pct)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    df = load_full_dataset()
    df_clean = clean_and_prepare(df)

    plot_resolution_time_distribution(df_clean)
    plot_escalation_rate_by_category(df_clean)
    plot_order_value_capping(df_clean)