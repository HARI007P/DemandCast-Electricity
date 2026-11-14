"""
DemandCast - Exploratory Data Analysis (EDA) Module
Analyzes hourly, daily, weekly, and monthly electricity consumption patterns,
renewable generation interactions (Solar & Wind), and exports publication-ready
visualizations and summary statistics for the web dashboard.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DemandCast.EDA")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_demand_data.parquet")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")
METRICS_DIR = os.path.join(BASE_DIR, "artifacts", "metrics")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "lines.linewidth": 1.8
})


def load_data():
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_name"] = df["timestamp"].dt.day_name()
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["month_name"] = df["timestamp"].dt.strftime("%b")
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def plot_overall_timeseries(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df["timestamp"], df["demand_mw"], color="#1e40af", alpha=0.85, label="Electricity Demand (MW)")
    axes[0].set_ylabel("Demand (MW)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(df["timestamp"], df["solar_mw"], color="#f59e0b", alpha=0.85, label="Solar Generation (MW)")
    axes[1].set_ylabel("Solar (MW)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(df["timestamp"], df["wind_mw"], color="#10b981", alpha=0.85, label="Wind Generation (MW)")
    axes[2].set_ylabel("Wind (MW)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    axes[3].plot(df["timestamp"], df["total_gen_mw"], color="#6366f1", alpha=0.85, label="Total Generation (MW)")
    axes[3].set_ylabel("Generation (MW)")
    axes[3].set_xlabel("Date")
    axes[3].legend(loc="upper left")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    fig.suptitle("DemandCast: India Grid Electricity Consumption & Renewables (2024 - 2025)", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "overall_timeseries.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_hourly_profile(df: pd.DataFrame):
    hourly_stats = df.groupby(["hour", "is_weekend"])["demand_mw"].agg(["mean", "std", "median"]).reset_index()
    weekday_data = hourly_stats[hourly_stats["is_weekend"] == 0]
    weekend_data = hourly_stats[hourly_stats["is_weekend"] == 1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(weekday_data["hour"], weekday_data["mean"], color="#2563eb", marker="o", label="Weekday (Mon-Fri) Mean")
    ax.fill_between(weekday_data["hour"],
                    weekday_data["mean"] - weekday_data["std"],
                    weekday_data["mean"] + weekday_data["std"],
                    color="#2563eb", alpha=0.15, label="Weekday ±1 Std Dev")

    ax.plot(weekend_data["hour"], weekend_data["mean"], color="#dc2626", marker="s", linestyle="--", label="Weekend (Sat-Sun) Mean")
    ax.fill_between(weekend_data["hour"],
                    weekend_data["mean"] - weekend_data["std"],
                    weekend_data["mean"] + weekend_data["std"],
                    color="#dc2626", alpha=0.15, label="Weekend ±1 Std Dev")

    ax.set_title("Hourly Electricity Demand Profile (Diurnal Cycle)")
    ax.set_xlabel("Hour of Day (00:00 - 23:00 IST)")
    ax.set_ylabel("Demand (MW)")
    ax.set_xticks(range(0, 24))
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Highlight peak load hour
    peak_hr = weekday_data.loc[weekday_data["mean"].idxmax()]["hour"]
    peak_val = weekday_data["mean"].max()
    ax.annotate(f"Peak Demand: {peak_val:.0f} MW @ {int(peak_hr):02d}:00",
                xy=(peak_hr, peak_val),
                xytext=(peak_hr - 5, peak_val + 7000),
                arrowprops=dict(facecolor="#1e3a8a", shrink=0.08, width=1.5, headwidth=7),
                fontweight="bold", color="#1e3a8a")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "hourly_demand_profile.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_weekly_profile(df: pd.DataFrame):
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_stats = df.groupby("day_name")["demand_mw"].agg(["mean", "median", "std"]).reindex(day_order)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(day_stats.index, day_stats["mean"], yerr=day_stats["std"],
                  capsize=4, color=["#3b82f6" if d not in ["Saturday", "Sunday"] else "#ef4444" for d in day_order],
                  alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.set_title("Weekly Electricity Demand Variation across Days")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean Demand (MW)")
    ax.set_ylim(bottom=df["demand_mw"].min() * 0.9)
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h / 2, f"{h:.0f} MW",
                ha="center", va="center", color="white", fontweight="bold", rotation=90)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "weekly_demand_profile.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_monthly_trend(df: pd.DataFrame):
    monthly_stats = df.groupby("year_month")["demand_mw"].agg(["mean", "min", "max", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(monthly_stats))
    ax.plot(x, monthly_stats["mean"], marker="o", color="#059669", linewidth=2.2, label="Monthly Mean Demand")
    ax.fill_between(x, monthly_stats["min"], monthly_stats["max"], color="#10b981", alpha=0.2, label="Min-Max Range")

    ax.set_xticks(x)
    ax.set_xticklabels(monthly_stats["year_month"], rotation=45, ha="right")
    ax.set_title("Monthly Electricity Demand Trend & Seasonal Fluctuation")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "monthly_seasonal_trend.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_correlations(df: pd.DataFrame):
    cols = ["demand_mw", "solar_mw", "wind_mw", "total_gen_mw"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    labels = ["Demand", "Solar", "Wind", "Total Gen"]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color="white" if abs(val) > 0.5 else "black", fontweight="bold")

    ax.set_title("Correlation Matrix: Demand vs Renewable Generation", pad=20)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)


def export_eda_metrics(df: pd.DataFrame):
    stats = {
        "dataset_name": "Indian National Electricity Demand, Solar and Wind Dataset",
        "source": "Grid-India / Mendeley Data (y58jknpgs8)",
        "period": {
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
            "total_hours": int(len(df)),
            "total_days": float(len(df) / 24.0)
        },
        "demand_mw": {
            "mean": float(df["demand_mw"].mean()),
            "std": float(df["demand_mw"].std()),
            "min": float(df["demand_mw"].min()),
            "p25": float(df["demand_mw"].quantile(0.25)),
            "median": float(df["demand_mw"].median()),
            "p75": float(df["demand_mw"].quantile(0.75)),
            "max": float(df["demand_mw"].max())
        },
        "solar_mw": {
            "mean": float(df["solar_mw"].mean()),
            "max": float(df["solar_mw"].max()),
            "peak_hour": int(df.groupby("hour")["solar_mw"].mean().idxmax())
        },
        "wind_mw": {
            "mean": float(df["wind_mw"].mean()),
            "max": float(df["wind_mw"].max())
        },
        "diurnal_peaks": {
            "peak_hour": int(df.groupby("hour")["demand_mw"].mean().idxmax()),
            "peak_demand_avg": float(df.groupby("hour")["demand_mw"].mean().max()),
            "trough_hour": int(df.groupby("hour")["demand_mw"].mean().idxmin()),
            "trough_demand_avg": float(df.groupby("hour")["demand_mw"].mean().min())
        },
        "weekday_vs_weekend": {
            "weekday_avg": float(df[df["is_weekend"] == 0]["demand_mw"].mean()),
            "weekend_avg": float(df[df["is_weekend"] == 1]["demand_mw"].mean()),
            "weekend_drop_mw": float(df[df["is_weekend"] == 0]["demand_mw"].mean() - df[df["is_weekend"] == 1]["demand_mw"].mean()),
            "weekend_drop_percent": float(
                (df[df["is_weekend"] == 0]["demand_mw"].mean() - df[df["is_weekend"] == 1]["demand_mw"].mean())
                / df[df["is_weekend"] == 0]["demand_mw"].mean() * 100.0
            )
        },
        "correlations": {
            "demand_vs_solar": float(df["demand_mw"].corr(df["solar_mw"])),
            "demand_vs_wind": float(df["wind_mw"].corr(df["demand_mw"])),
            "demand_vs_total_gen": float(df["demand_mw"].corr(df["total_gen_mw"]))
        }
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    out_file = os.path.join(METRICS_DIR, "eda_summary.json")
    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved EDA summary metrics: %s", out_file)
    return stats


def run_eda():
    df = load_data()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    plot_overall_timeseries(df)
    plot_hourly_profile(df)
    plot_weekly_profile(df)
    plot_monthly_trend(df)
    plot_correlations(df)
    metrics = export_eda_metrics(df)
    logger.info("EDA completed successfully.")
    return metrics


if __name__ == "__main__":
    run_eda()
