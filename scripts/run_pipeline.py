"""
AgroOpt — End-to-End MLOps Pipeline
=====================================
Deliverable 3: Complete MLOps Pipeline

Orchestrates the full ML lifecycle in a single command:

  Stage 1 — Data Preprocessing
    Load and merge the synthetic crop-yield dataset (crop_yield.csv)
    with FAO reference data (yield, rainfall, temperature, pesticides).
    Output: data/processed/merged_dataset.csv

  Stage 2 — Feature Engineering
    Derive 36 features from the merged dataset: domain-knowledge
    transformations (stress indices, GDD proxy, interaction terms)
    and one-hot encoding of categorical variables.
    Output: data/processed/features_dataset.csv

  Stage 3 — Model Training + Experiment Tracking
    Train five regression models (Ridge, RandomForest, HGB, XGBoost,
    LightGBM) with 5-fold cross-validation. All parameters and metrics
    (R², RMSE, MAE, MAPE, CV-RMSE) are logged to MLflow.
    Output: models/best_model.pkl, models/feature_names.json,
            models/model_results.json, mlflow/mlruns/

  Stage 4 — Model Evaluation Summary
    Load the saved model_results.json and print a ranked comparison
    table of all five models.

Usage
-----
    python run_pipeline.py                  # run all stages
    python run_pipeline.py --from-stage 3  # resume at training
    python run_pipeline.py --skip-merge    # skip if merged CSV exists

Environment
-----------
    Activate the project virtual environment first:
        .venv/Scripts/activate   (Windows)
        source .venv/bin/activate (Linux/macOS)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger

from src.utils.config import settings
from src.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def stage_banner(n: int, title: str) -> None:
    logger.info("=" * 60)
    logger.info(f"  STAGE {n}: {title}")
    logger.info("=" * 60)


def stage_1_preprocess(skip_if_exists: bool = False) -> None:
    """Merge raw datasets → data/processed/merged_dataset.csv."""
    stage_banner(1, "Data Preprocessing & Merge")

    merged_path: Path = settings.paths.merged
    if skip_if_exists and merged_path.exists():
        logger.info(f"Skipping merge — {merged_path} already exists.")
        return

    from src.data.merge_datasets import merge_datasets
    df = merge_datasets()
    logger.info(f"Merged dataset: {df.shape[0]:,} rows × {df.shape[1]} cols → {merged_path}")


def stage_2_features(skip_if_exists: bool = False) -> None:
    """Engineer 36 features → data/processed/features_dataset.csv."""
    stage_banner(2, "Feature Engineering")

    features_path: Path = settings.paths.features
    if skip_if_exists and features_path.exists():
        logger.info(f"Skipping feature engineering — {features_path} already exists.")
        return

    from src.features.feature_engineering import build_features
    import pandas as pd
    merged = pd.read_csv(settings.paths.merged)
    df_feat = build_features(merged)
    df_feat.to_csv(features_path, index=False)
    logger.info(f"Feature dataset: {df_feat.shape[0]:,} rows × {df_feat.shape[1]} cols → {features_path}")


def stage_3_train() -> None:
    """Train all models, log to MLflow, save best pipeline."""
    stage_banner(3, "Model Training + MLflow Experiment Tracking")

    from src.models.train import run_training
    results = run_training()

    best = max(results, key=lambda r: r["r2"])
    logger.info(f"Best model: {best['model']} — R²={best['r2']:.4f}, RMSE={best['rmse']:,.0f} hg/ha")


def stage_4_evaluate() -> None:
    """Print ranked model comparison from saved model_results.json."""
    stage_banner(4, "Model Evaluation Summary")

    results_path = settings.paths.models_dir / "model_results.json"
    if not results_path.exists():
        logger.warning(f"{results_path} not found — run stage 3 first.")
        return

    with results_path.open() as fh:
        results: list[dict] = json.load(fh)

    ranked = sorted(results, key=lambda r: r["r2"], reverse=True)
    header = f"{'Rank':<5}{'Model':<25}{'R²':>8}{'RMSE':>14}{'MAE':>12}{'Train s':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for i, r in enumerate(ranked, 1):
        logger.info(
            f"{i:<5}{r['model']:<25}{r['r2']:>8.4f}"
            f"{r['rmse']:>14,.0f}{r['mae']:>12,.0f}{r['train_time_s']:>10.1f}"
        )
    logger.info(f"Best model saved to: {settings.paths.best_model}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgroOpt end-to-end MLOps pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        metavar="N",
        help="Start execution from stage N (default: 1 = all stages).",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip stage 1 if merged_dataset.csv already exists.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip stage 2 if features_dataset.csv already exists.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    t0 = time.perf_counter()
    logger.info("AgroOpt MLOps Pipeline starting …")
    logger.info(f"Project root: {settings.paths.models_dir.parent}")

    try:
        if args.from_stage <= 1:
            stage_1_preprocess(skip_if_exists=args.skip_merge)
        if args.from_stage <= 2:
            stage_2_features(skip_if_exists=args.skip_features)
        if args.from_stage <= 3:
            stage_3_train()
        if args.from_stage <= 4:
            stage_4_evaluate()
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise

    elapsed = time.perf_counter() - t0
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
