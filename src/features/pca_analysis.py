"""
Phase 4 — PCA Analysis for AgroOpt.

Strategy
--------
Starting from features_dataset.csv (666 494 rows × 37 cols), this module:

  1. Loads the feature matrix X (36 features) and target y (yield_hg_ha).
  2. Scales X with StandardScaler — PCA is variance-sensitive and requires
     all features to be on the same scale before computing eigenvectors.
  3. Fits a full PCA (all 36 components) to capture the complete variance
     structure of the feature space.
  4. Reports how many principal components are needed to explain
     90 / 95 / 99 % of the total variance.
  5. Extracts and saves the loadings matrix (features × components), which
     shows the contribution of each original feature to each PC.
  6. Saves a JSON sidecar with key PCA metrics for downstream reference.

Key findings anticipated
------------------------
  • rainfall_mm and rainfall_anomaly are perfectly collinear (fao_rainfall_mm
    is a 2013 constant = 715 mm), so they load identically on PC1.
    PCA surfaces this as a single effective variance dimension.
  • Similarly, temperature_celsius and temp_anomaly are perfectly collinear
    (fao_avg_temp is a 2013 constant = 16.44 °C).
  • The effective rank of the feature matrix is therefore lower than 36;
    PCA quantifies this by showing rapid cumulative variance growth.

Output
------
  data/processed/pca_loadings.csv   — loadings matrix (36 features × 36 PCs)
  data/processed/pca_results.json   — key metrics (n_components_*, EVR, top loadings)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.config import settings
from src.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Core PCA pipeline
# ---------------------------------------------------------------------------

def run_pca(
    feat_path: str | Path | None = None,
    save: bool = True,
) -> dict:
    """
    Load the feature matrix, scale, and fit a full PCA.

    Parameters
    ----------
    feat_path : path to features_dataset.csv.
                Defaults to settings.paths.features.
    save      : if True, persist loadings CSV and JSON metadata to
                data/processed/.

    Returns
    -------
    result : dict with the following keys:

        ``pca``                      — fitted sklearn PCA (all components)
        ``scaler``                   — fitted StandardScaler
        ``X_scaled``                 — scaled feature matrix (numpy array)
        ``feature_cols``             — list of feature column names
        ``explained_variance_ratio`` — per-component EVR (numpy array)
        ``cumulative_evr``           — cumulative EVR (numpy array)
        ``n_components_90``          — PCs needed to explain 90% variance
        ``n_components_95``          — PCs needed to explain 95% variance
        ``n_components_99``          — PCs needed to explain 99% variance
        ``loadings_df``              — DataFrame (features × PCs)
    """
    settings.paths.ensure_dirs()

    feat_path = Path(feat_path) if feat_path else settings.paths.features
    logger.info(f"Loading feature dataset from {feat_path} …")
    feat = pd.read_csv(feat_path)
    logger.info(f"  Loaded: {feat.shape[0]:,} rows × {feat.shape[1]} cols")

    target_col = settings.data.target_column
    feature_cols = [c for c in feat.columns if c != target_col]

    X = feat[feature_cols].values
    logger.info(f"  Feature matrix X: {X.shape}")

    # ------------------------------------------------------------------
    # 1. Scale features (mandatory before PCA — removes scale bias)
    # ------------------------------------------------------------------
    logger.info("Scaling features with StandardScaler …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------
    # 2. Fit full PCA (retain all components for analysis)
    # ------------------------------------------------------------------
    n_full = X_scaled.shape[1]
    logger.info(f"Fitting PCA with {n_full} components …")
    pca = PCA(n_components=n_full, random_state=settings.random_seed)
    pca.fit(X_scaled)

    evr     = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    logger.info(f"  PC1 explains {evr[0]*100:.2f}% of variance")
    logger.info(f"  PC2 explains {evr[1]*100:.2f}% of variance")

    def _n_for(threshold: float) -> int:
        """Smallest number of PCs needed to reach ``threshold`` cumulative EVR."""
        return int(np.searchsorted(cum_evr, threshold) + 1)

    n90 = _n_for(0.90)
    n95 = _n_for(0.95)
    n99 = _n_for(0.99)

    logger.info(f"  PCs for ≥90% variance: {n90}")
    logger.info(f"  PCs for ≥95% variance: {n95}")
    logger.info(f"  PCs for ≥99% variance: {n99}")

    # ------------------------------------------------------------------
    # 3. Loadings matrix  (features × components)
    #    pca.components_ shape is (n_components, n_features)
    #    We transpose so rows = features, cols = PCs — easier to read
    # ------------------------------------------------------------------
    pc_names = [f"PC{i + 1}" for i in range(n_full)]
    loadings_df = pd.DataFrame(
        pca.components_.T,       # (n_features, n_components)
        index=feature_cols,
        columns=pc_names,
    )

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    if save:
        load_path = settings.paths.processed_dir / "pca_loadings.csv"
        loadings_df.round(6).to_csv(load_path)
        logger.info(f"Loadings saved → {load_path}")

        meta = {
            "n_features": int(n_full),
            "feature_cols": feature_cols,
            "explained_variance_ratio": [round(v, 6) for v in evr.tolist()],
            "cumulative_evr":           [round(v, 6) for v in cum_evr.tolist()],
            "n_components_90": int(n90),
            "n_components_95": int(n95),
            "n_components_99": int(n99),
            "pc1_top_features": loadings_df["PC1"].abs().nlargest(5).index.tolist(),
            "pc2_top_features": loadings_df["PC2"].abs().nlargest(5).index.tolist(),
        }
        meta_path = settings.paths.processed_dir / "pca_results.json"
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metadata saved → {meta_path}")

    return {
        "pca":                      pca,
        "scaler":                   scaler,
        "X_scaled":                 X_scaled,
        "feature_cols":             feature_cols,
        "explained_variance_ratio": evr,
        "cumulative_evr":           cum_evr,
        "n_components_90":          n90,
        "n_components_95":          n95,
        "n_components_99":          n99,
        "loadings_df":              loadings_df,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    result = run_pca(save=True)

    evr     = result["explained_variance_ratio"]
    cum_evr = result["cumulative_evr"]
    loads   = result["loadings_df"]

    print("\n" + "=" * 65)
    print("PCA ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"Total features      : {result['pca'].n_features_in_}")
    print(f"Components for ≥90% : {result['n_components_90']}")
    print(f"Components for ≥95% : {result['n_components_95']}")
    print(f"Components for ≥99% : {result['n_components_99']}")
    print()
    print("Top 12 components by individual explained variance:")
    for i in range(min(12, len(evr))):
        bar = "█" * int(evr[i] * 100)
        print(
            f"  PC{i+1:2d}  {evr[i]*100:5.2f}%  "
            f"(cumulative {cum_evr[i]*100:5.1f}%)  {bar}"
        )
    print()
    print("PC1 — top 8 features by |loading|:")
    print(loads["PC1"].abs().nlargest(8).round(4).to_string())
    print()
    print("PC2 — top 8 features by |loading|:")
    print(loads["PC2"].abs().nlargest(8).round(4).to_string())
