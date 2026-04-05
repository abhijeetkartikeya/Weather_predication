"""
Visualise actual vs. predicted values for each weather variable.

Produces:
  - One PNG per variable (4 total)
  - One combined 2x2 comparison plot
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import PREDICTIONS_CSV, PLOTS_DIR, TARGET_VARIABLES
from feature_engineering import _ALIAS

# ─── Display settings ──────────────────────────────────────────────────────
_UNITS = {
    "temp": "Temperature (deg C)",
    "cloud": "Cloud Cover (%)",
    "wind": "Wind Speed (m/s)",
    "ghi": "Shortwave Radiation (W/m2)",
}


def _style_ax(ax, title: str):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)


def plot_results(pred_df: pd.DataFrame | None = None):
    """Create individual and combined comparison plots."""
    if pred_df is None:
        pred_df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["time"])

    aliases = [_ALIAS[v] for v in TARGET_VARIABLES]

    # ── Individual plots ────────────────────────────────────────────────
    for alias in aliases:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(pred_df["time"], pred_df[f"actual_{alias}"],
                color="green", linewidth=0.8, label="Actual")
        ax.plot(pred_df["time"], pred_df[f"predicted_{alias}"],
                color="orange", linewidth=0.8, linestyle="--",
                label="Predicted")
        _style_ax(ax, f"Actual vs Predicted — {_UNITS[alias]}")
        ax.set_ylabel(_UNITS[alias])
        fig.tight_layout()
        path = f"{PLOTS_DIR}/{alias}_actual_vs_predicted.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")

    # ── Combined 2x2 plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    for ax, alias in zip(axes.flatten(), aliases):
        ax.plot(pred_df["time"], pred_df[f"actual_{alias}"],
                color="green", linewidth=0.7, label="Actual")
        ax.plot(pred_df["time"], pred_df[f"predicted_{alias}"],
                color="orange", linewidth=0.7, linestyle="--",
                label="Predicted")
        _style_ax(ax, _UNITS[alias])

    fig.suptitle("Weather Forecast — Actual vs Predicted (Recursive, No Leakage)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    combined_path = f"{PLOTS_DIR}/combined_comparison.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {combined_path}")


if __name__ == "__main__":
    plot_results()
