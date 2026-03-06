"""
Evaluation utilities for AgroOpt model assessment.

All functions accept numpy arrays and return either metrics dicts or
matplotlib Axes objects, so they work both standalone and inside
multi-panel notebook figures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : ground-truth target values (hg/ha)
    y_pred : model predictions (hg/ha)

    Returns
    -------
    dict with keys: rmse, mae, r2, mape
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    # MAPE — guard against zero targets (not present in this dataset but safe)
    mask = y_true > 0
    mape = float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ---------------------------------------------------------------------------
# Prediction scatter (actual vs predicted)
# ---------------------------------------------------------------------------

def plot_predictions_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ax: plt.Axes | None = None,
    sample_size: int = 15_000,
) -> plt.Axes:
    """
    Scatter plot of actual vs predicted values.

    A diagonal line (perfect prediction) is drawn for reference.
    Points are sampled if the dataset is large.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    rng = np.random.default_rng(42)
    if len(y_true) > sample_size:
        idx = rng.choice(len(y_true), size=sample_size, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred

    metrics = compute_metrics(y_true, y_pred)      # full set for metrics
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())

    ax.scatter(yt, yp, alpha=0.08, s=4, color="steelblue")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual yield (hg/ha)")
    ax.set_ylabel("Predicted yield (hg/ha)")
    ax.set_title(
        f"{model_name}\n"
        f"R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:,.0f}  "
        f"MAE={metrics['mae']:,.0f}",
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.legend(fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Residuals analysis
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    sample_size: int = 15_000,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Two-panel residuals analysis:
      Left  — residuals vs predicted values (checks heteroscedasticity)
      Right — residuals distribution (checks normality of errors)
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_left, ax_right = axes

    rng = np.random.default_rng(42)
    if len(y_true) > sample_size:
        idx = rng.choice(len(y_true), size=sample_size, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred

    residuals = yt - yp

    # Residuals vs predicted
    ax_left.scatter(yp, residuals, alpha=0.08, s=4, color="steelblue")
    ax_left.axhline(0, color="red", ls="--", lw=1.5)
    ax_left.set_xlabel("Predicted (hg/ha)")
    ax_left.set_ylabel("Residual (actual − predicted)")
    ax_left.set_title(f"{model_name} — Residuals vs Predicted", fontweight="bold")
    ax_left.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K")
    )
    ax_left.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K")
    )

    # Residuals histogram
    ax_right.hist(residuals, bins=80, color="steelblue", alpha=0.85, edgecolor="white")
    ax_right.axvline(0, color="red", ls="--", lw=1.5)
    ax_right.set_xlabel("Residual (hg/ha)")
    ax_right.set_ylabel("Count")
    ax_right.set_title(
        f"{model_name} — Residual Distribution\n"
        f"mean={residuals.mean():,.0f}  std={residuals.std():,.0f}",
        fontweight="bold",
    )
    ax_right.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K")
    )
    return ax_left, ax_right


# ---------------------------------------------------------------------------
# Feature importance (tree models)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    pipeline: Pipeline,
    feature_cols: list[str],
    model_name: str,
    top_n: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Horizontal bar chart of feature importances for tree-based models.

    Extracts ``feature_importances_`` from the last step of a sklearn Pipeline.
    Raises ValueError for models that do not expose feature importances (e.g. Ridge).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Extract the actual estimator from the pipeline
    estimator = pipeline.steps[-1][1]
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError(
            f"Model '{model_name}' does not expose feature_importances_. "
            "Use a tree-based model (RF, XGBoost, LightGBM, HGB)."
        )

    importances = pd.Series(
        estimator.feature_importances_, index=feature_cols
    ).sort_values(ascending=False).head(top_n)

    colors = [
        "#e74c3c" if v > importances.quantile(0.75) else
        "#f39c12" if v > importances.quantile(0.5)  else
        "#95a5a6"
        for v in importances.values
    ]
    importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Impurity / Gain)")
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances", fontweight="bold")

    for i, (_name, val) in enumerate(importances.items()):
        ax.text(val + importances.max() * 0.005, i, f"{val:.4f}",
                va="center", fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Ridge coefficient importance (linear model equivalent)
# ---------------------------------------------------------------------------

def plot_ridge_coefficients(
    pipeline: Pipeline,
    feature_cols: list[str],
    model_name: str,
    top_n: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Horizontal bar chart of Ridge coefficient magnitudes.

    After StandardScaler normalisation all features share the same scale,
    so |coef| is a valid proxy for feature importance in a linear model.
    Bars are coloured by sign: green = positive effect, red = negative.

    Raises ValueError if the pipeline's last step has no ``coef_`` attribute.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    estimator = pipeline.steps[-1][1]
    if not hasattr(estimator, "coef_"):
        raise ValueError(
            f"Model '{model_name}' does not expose coef_. "
            "Use a linear model (Ridge, Lasso, ElasticNet)."
        )

    coefs = pd.Series(estimator.coef_, index=feature_cols)
    top = coefs.abs().sort_values(ascending=False).head(top_n)
    top_coefs = coefs[top.index]

    colors = ["#27ae60" if v > 0 else "#e74c3c" for v in top_coefs.values]
    top_coefs.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Ridge Coefficient (standardised scale)")
    ax.set_title(
        f"{model_name} — Top {top_n} Feature Coefficients\n"
        "Green = positive effect on yield, Red = negative",
        fontweight="bold",
    )
    for i, (_name, val) in enumerate(top_coefs.items()):
        x = val + np.sign(val) * top_coefs.abs().max() * 0.01
        ax.text(x, i, f"{val:+.1f}", va="center", fontsize=8,
                ha="left" if val >= 0 else "right")
    return ax


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Convert training results dict into a tidy comparison DataFrame.

    Parameters
    ----------
    results : dict returned by ``train_all_models()``
              keys = model names, values = dicts with 'metrics', 'cv_rmse_mean', etc.

    Returns
    -------
    DataFrame sorted by test R² descending.
    """
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Model":            name,
            "Test R²":          round(m["r2"],   4),
            "Test RMSE":        round(m["rmse"],  1),
            "Test MAE":         round(m["mae"],   1),
            "Test MAPE (%)":    round(m["mape"],  2),
            "CV RMSE mean":     round(res["cv_rmse_mean"], 1),
            "CV RMSE std":      round(res["cv_rmse_std"],  1),
            "Train time (s)":   round(res["train_time_s"], 1),
        })
    return (
        pd.DataFrame(rows)
        .set_index("Model")
        .sort_values("Test R²", ascending=False)
    )
