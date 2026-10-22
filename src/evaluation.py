"""
Evaluation Module
Computes regression metrics (MSE, RMSE, MAE, R2, Adjusted R2),
performs residual diagnostics, and generates publication-quality plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_features: int) -> Dict[str, float]:
    """
    Computes MSE, RMSE, MAE, R2, Adjusted R2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Adjusted R2
    if n > p_features + 1:
        adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / (n - p_features - 1))
    else:
        adj_r2 = r2

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Adjusted_R2": float(adj_r2)
    }


def compare_models(
    baseline_metrics: Dict[str, float],
    fs_metrics: Dict[str, float]
) -> pd.DataFrame:
    """
    Creates comparison DataFrame between Baseline and Feature Selected Model.
    """
    comparison = pd.DataFrame([
        {
            "Metric": "R2 (Coefficient of Determination)",
            "Baseline Model": f"{baseline_metrics['R2']:.4f}",
            "Feature Selected Model": f"{fs_metrics['R2']:.4f}",
            "Difference (FS - Base)": f"{(fs_metrics['R2'] - baseline_metrics['R2']):+.4f}"
        },
        {
            "Metric": "Adjusted R2",
            "Baseline Model": f"{baseline_metrics['Adjusted_R2']:.4f}",
            "Feature Selected Model": f"{fs_metrics['Adjusted_R2']:.4f}",
            "Difference (FS - Base)": f"{(fs_metrics['Adjusted_R2'] - baseline_metrics['Adjusted_R2']):+.4f}"
        },
        {
            "Metric": "RMSE (Root Mean Squared Error)",
            "Baseline Model": f"{baseline_metrics['RMSE']:,.2f}",
            "Feature Selected Model": f"{fs_metrics['RMSE']:,.2f}",
            "Difference (FS - Base)": f"{(fs_metrics['RMSE'] - baseline_metrics['RMSE']):+,.2f}"
        },
        {
            "Metric": "MAE (Mean Absolute Error)",
            "Baseline Model": f"{baseline_metrics['MAE']:,.2f}",
            "Feature Selected Model": f"{fs_metrics['MAE']:,.2f}",
            "Difference (FS - Base)": f"{(fs_metrics['MAE'] - baseline_metrics['MAE']):+,.2f}"
        },
        {
            "Metric": "MSE (Mean Squared Error)",
            "Baseline Model": f"{baseline_metrics['MSE']:,.2e}",
            "Feature Selected Model": f"{fs_metrics['MSE']:,.2e}",
            "Difference (FS - Base)": f"{(fs_metrics['MSE'] - baseline_metrics['MSE']):+,.2e}"
        }
    ])
    return comparison


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    pi_lower: Optional[np.ndarray] = None,
    pi_upper: Optional[np.ndarray] = None,
    title: str = "Actual vs. Predicted with Intervals",
    save_path: Optional[str] = None,
    sample_size: int = 100
) -> plt.Figure:
    """
    Generates Actual vs Predicted plot with Prediction Interval (PI) and Confidence Interval (CI).
    Samples a subset if dataset is large for clear visualization.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    n = len(y_true)
    if n > sample_size:
        idx = np.random.RandomState(42).choice(n, sample_size, replace=False)
        # Sort by predicted value for clean ribbon plotting
        sort_order = np.argsort(y_pred[idx])
        plot_idx = idx[sort_order]
    else:
        plot_idx = np.argsort(y_pred)

    y_t_sub = y_true[plot_idx]
    y_p_sub = y_pred[plot_idx]
    x_axis = np.arange(len(plot_idx))

    # Plot Prediction Interval ribbon
    if pi_lower is not None and pi_upper is not None:
        ax.fill_between(
            x_axis,
            pi_lower[plot_idx],
            pi_upper[plot_idx],
            color="#ffe0b2",
            alpha=0.6,
            label="95% Prediction Interval (PI)"
        )

    # Plot Confidence Interval ribbon
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            x_axis,
            ci_lower[plot_idx],
            ci_upper[plot_idx],
            color="#bbdefb",
            alpha=0.7,
            label="95% Confidence Interval (CI)"
        )

    # Line for Predicted and Points for Actual
    ax.plot(x_axis, y_p_sub, color="#1976d2", lw=2, label="Predicted Value (ŷ)")
    ax.scatter(x_axis, y_t_sub, color="#d32f2f", s=30, alpha=0.8, edgecolors="k", lw=0.5, label="Actual Value (y)")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Sorted Test Samples", fontsize=11)
    ax.set_ylabel("Target Value", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str = "Model",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generates Residual Diagnostics:
    1. Residuals vs. Fitted Values (Homoscedasticity)
    2. Residual Histogram & KDE (Normality)
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Residuals vs. Fitted
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0], alpha=0.5, color="#1565c0", edgecolor="none")
    axes[0].axhline(0, color="red", linestyle="--", lw=1.8)
    axes[0].set_title(f"{title_prefix}: Residuals vs. Fitted Values", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Fitted Values (ŷ)", fontsize=11)
    axes[0].set_ylabel("Residuals (y - ŷ)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Residual Distribution (Normality)
    sns.histplot(residuals, kde=True, ax=axes[1], color="#2e7d32", bins=30)
    axes[1].axvline(0, color="red", linestyle="--", lw=1.8)
    axes[1].set_title(f"{title_prefix}: Residual Distribution (Normality Check)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residuals", fontsize=11)
    axes[1].set_ylabel("Density / Count", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
