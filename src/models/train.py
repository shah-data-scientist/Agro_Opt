"""
Phase 5 — Model Training for AgroOpt.

Strategy
--------
Five regression models are trained on the 36-feature matrix produced by
Phase 3 (features_dataset.csv) and compared using a held-out test set and
5-fold cross-validation.  All experiments are tracked with MLflow.

Models
------
  1. Ridge Regression          — linear baseline; L2 regularisation handles
                                 collinearity (requires StandardScaler)
  2. Random Forest             — ensemble of decision trees; immune to
                                 collinearity; built-in feature importance
  3. Hist Gradient Boosting    — sklearn's histogram-based GBM; fast on large
                                 datasets (replaces slow GradientBoostingRegressor)
  4. XGBoost                   — extreme gradient boosting; strong baseline
  5. LightGBM                  — light gradient boosting; fastest at scale

Data split
----------
  80% train / 20% test,  random_state=42  (stratification not needed for regression)

Cross-validation
----------------
  5-fold CV on a 100 000-row subsample of the training set (RMSE scoring).
  Full training uses the entire training set.

MLflow tracking
---------------
  Tracking URI : mlflow/mlruns/   (local file store)
  Experiment   : agro-opt-yield-prediction
  Per run logs : params, test metrics (R², RMSE, MAE, MAPE), CV RMSE, train time

Output artefacts
----------------
  models/best_model.pkl       — best-performing sklearn Pipeline
  models/scaler.pkl           — StandardScaler fitted on full training data
  models/feature_names.json   — ordered list of feature column names
  models/model_results.json   — comparison table (all 5 models)
  mlflow/mlruns/              — MLflow experiment artefacts
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.models.evaluate import compute_metrics
from src.utils.config import settings
from src.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Model registry — name → {model, scale, params}
# ---------------------------------------------------------------------------

def _build_registry() -> dict[str, dict]:
    """
    Build the model registry using the project random seed from settings.

    Keeping this as a function (rather than a module-level constant) ensures
    ``settings.random_seed`` is evaluated after the config singleton is ready.
    """
    seed = settings.random_seed
    return {
        "ridge": {
            "model": Ridge(alpha=10.0),
            "scale": True,   # Ridge requires StandardScaler
            "params": {"model": "Ridge", "alpha": 10.0},
        },
        "random_forest": {
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=seed,
            ),
            "scale": False,
            "params": {
                "model": "RandomForest",
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_leaf": 5,
            },
        },
        "hist_gradient_boosting": {
            "model": HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_leaf=20,
                random_state=seed,
            ),
            "scale": False,
            "params": {
                "model": "HistGradientBoosting",
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_depth": 8,
            },
        },
        "xgboost": {
            "model": XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=seed,
                verbosity=0,
            ),
            "scale": False,
            "params": {
                "model": "XGBoost",
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
            },
        },
        "lightgbm": {
            "model": LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=seed,
                verbose=-1,
            ),
            "scale": False,
            "params": {
                "model": "LightGBM",
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 8,
                "num_leaves": 63,
            },
        },
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(
    feat_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load features_dataset.csv and return (X, y, feature_cols).

    Parameters
    ----------
    feat_path : path override.  Defaults to settings.paths.features.

    Returns
    -------
    X           : (n_samples, n_features) float64 array
    y           : (n_samples,) float64 array — yield_hg_ha
    feature_cols: ordered list of feature column names
    """
    feat_path = Path(feat_path) if feat_path else settings.paths.features
    logger.info(f"Loading features from {feat_path} …")
    feat = pd.read_csv(feat_path)

    target_col   = settings.data.target_column
    feature_cols = [c for c in feat.columns if c != target_col]

    X = feat[feature_cols].values.astype(np.float64)
    y = feat[target_col].values.astype(np.float64)
    logger.info(f"  X: {X.shape} | y: {y.shape}")
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

def _cv_rmse(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    sample_size: int = 100_000,
) -> tuple[float, float]:
    """
    5-fold cross-validated RMSE on a random subsample.

    Using a subsample (default 100 K) keeps CV tractable on the full 533 K
    training set while still producing reliable estimates.

    Returns
    -------
    (mean_rmse, std_rmse)
    """
    if len(X) > sample_size:
        rng = np.random.default_rng(settings.random_seed)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        Xc, yc = X[idx], y[idx]
    else:
        Xc, yc = X, y

    scores = cross_val_score(
        pipeline, Xc, yc,
        cv=n_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    rmse_scores = np.sqrt(-scores)
    return float(rmse_scores.mean()), float(rmse_scores.std())


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    registry: dict,
) -> dict:
    """
    Train one model, evaluate on the held-out test set, and log to MLflow.

    Parameters
    ----------
    model_name   : key into ``registry``
    X_train / y_train : training split
    X_test  / y_test  : test split
    feature_cols : feature names (for logging)
    registry     : model registry dict (from ``_build_registry()``)

    Returns
    -------
    dict with keys:
        model_name, pipeline, metrics, cv_rmse_mean, cv_rmse_std, train_time_s
    """
    cfg = registry[model_name]

    # Build sklearn Pipeline
    steps: list = []
    if cfg["scale"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", cfg["model"]))
    pipeline = Pipeline(steps)

    # Cross-validation on training subsample
    logger.info(f"  [{model_name}] Cross-validating (5-fold, 100K sample) …")
    cv_mean, cv_std = _cv_rmse(pipeline, X_train, y_train)
    logger.info(f"  [{model_name}] CV RMSE: {cv_mean:,.1f} ± {cv_std:,.1f}")

    # Full training
    logger.info(f"  [{model_name}] Training on full training set ({len(X_train):,} rows) …")
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info(f"  [{model_name}] Trained in {elapsed:.1f}s")

    # Test evaluation
    y_pred   = pipeline.predict(X_test)
    metrics  = compute_metrics(y_test, y_pred)
    logger.info(
        f"  [{model_name}] Test — R²={metrics['r2']:.4f} | "
        f"RMSE={metrics['rmse']:,.0f} | MAE={metrics['mae']:,.0f} | "
        f"MAPE={metrics['mape']:.2f}%"
    )

    # MLflow logging
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(cfg["params"])
        mlflow.log_metrics({
            "test_r2":       metrics["r2"],
            "test_rmse":     metrics["rmse"],
            "test_mae":      metrics["mae"],
            "test_mape":     metrics["mape"],
            "cv_rmse_mean":  cv_mean,
            "cv_rmse_std":   cv_std,
            "train_time_s":  elapsed,
        })
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return {
        "model_name":    model_name,
        "pipeline":      pipeline,
        "y_pred":        y_pred,
        "metrics":       metrics,
        "cv_rmse_mean":  cv_mean,
        "cv_rmse_std":   cv_std,
        "train_time_s":  elapsed,
    }


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_all_models(
    feat_path: str | Path | None = None,
    save: bool = True,
) -> dict:
    """
    Train all 5 models, select the best by test R², and persist artefacts.

    Parameters
    ----------
    feat_path : override path to features_dataset.csv
    save      : write best_model.pkl, scaler.pkl, feature_names.json,
                model_results.json to models/

    Returns
    -------
    results : dict[model_name -> train_model() output dict]
              Also contains key ``"_split"`` with X_train, y_train, X_test, y_test.
    """
    settings.paths.ensure_dirs()

    # Setup MLflow — use file:/// URI so Windows paths are accepted
    mlflow_dir = Path(settings.paths.mlflow_tracking_uri)
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlflow_dir.as_uri())   # e.g. file:///C:/…/mlruns
    mlflow.set_experiment(settings.paths.mlflow_experiment_name)
    logger.info(
        f"MLflow tracking → {settings.paths.mlflow_tracking_uri} | "
        f"experiment: {settings.paths.mlflow_experiment_name}"
    )

    # Load data
    X, y, feature_cols = load_features(feat_path)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.training.test_size,
        random_state=settings.random_seed,
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Build model registry
    registry = _build_registry()

    # Train all models
    results: dict = {}
    for model_name in registry:
        logger.info(f"=== Training: {model_name} ===")
        results[model_name] = train_model(
            model_name, X_train, y_train, X_test, y_test,
            feature_cols, registry,
        )

    # Identify best model by test R²
    best_name   = max(results, key=lambda k: results[k]["metrics"]["r2"])
    best_result = results[best_name]
    logger.info(
        f"Best model: {best_name} "
        f"(test R²={best_result['metrics']['r2']:.4f})"
    )

    # Persist artefacts
    if save:
        # Best model pipeline
        best_path = settings.paths.best_model
        with open(best_path, "wb") as f:
            pickle.dump(best_result["pipeline"], f)
        logger.info(f"Best model saved → {best_path}")

        # Standalone scaler (fitted on full training data — used by the API)
        scaler = StandardScaler().fit(X_train)
        with open(settings.paths.scaler, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved → {settings.paths.scaler}")

        # Feature names
        with open(settings.paths.feature_names, "w") as f:
            json.dump(feature_cols, f, indent=2)
        logger.info(f"Feature names saved → {settings.paths.feature_names}")

        # Model comparison JSON
        summary = {
            "best_model": best_name,
            "train_rows": int(len(X_train)),
            "test_rows":  int(len(X_test)),
            "models": {
                name: {
                    "test_r2":       res["metrics"]["r2"],
                    "test_rmse":     res["metrics"]["rmse"],
                    "test_mae":      res["metrics"]["mae"],
                    "test_mape":     res["metrics"]["mape"],
                    "cv_rmse_mean":  res["cv_rmse_mean"],
                    "cv_rmse_std":   res["cv_rmse_std"],
                    "train_time_s":  res["train_time_s"],
                }
                for name, res in results.items()
            },
        }
        summary_path = settings.paths.models_dir / "model_results.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Model comparison saved → {summary_path}")

    # Attach split data for downstream notebook use
    results["_split"] = {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "feature_cols": feature_cols,
    }

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    results = train_all_models(save=True)
    split   = results.pop("_split")

    from src.models.evaluate import build_comparison_table
    table = build_comparison_table(results)

    print("\n" + "=" * 75)
    print("MODEL TRAINING COMPLETE")
    print("=" * 75)
    print(f"Train rows : {len(split['X_train']):,}")
    print(f"Test rows  : {len(split['X_test']):,}")
    print()
    print("Model comparison (sorted by Test R²):")
    print(table.to_string())
