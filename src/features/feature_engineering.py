"""
Phase 3 — Feature Engineering for AgroOpt.

Strategy
--------
Starting from merged_dataset.csv (666 494 rows × 15 cols), this module:

  1. Drops zero-variance columns (year = 2013 constant, fao_pesticides_tonnes).
  2. Derives four FAO-reference features that contextualise local conditions
     against the 2013 USA national baseline:
       • rainfall_anomaly   — local vs national average rainfall
       • temp_anomaly       — local vs national average temperature
     (fao_rainfall_mm and fao_avg_temp are then dropped; fao_yield_hg_ha is
      kept as a crop-level benchmark alongside the crop dummies.)
  3. Engineers agronomic interaction features:
       • rainfall_x_fertilizer, rainfall_x_irrigation, temp_x_irrigation
       • agro_intensity  (0 = no practice, 1 = one, 2 = both)
  4. Engineers climate indices:
       • heat_moisture_ratio  — temperature / (rainfall / 100)
       • aridity_index        — rainfall / temperature
       • harvest_rainfall_rate — rainfall / days_to_harvest
  5. Engineers crop-biology features (domain knowledge):
       • water_stress   — |local rainfall - crop optimal| / crop optimal
       • heat_stress    — degrees above crop's upper thermal limit (≥ 0)
       • gdd_proxy      — growing degree days × growing period
  6. Adds an ordinal soil quality score (Sandy=1 … Loam=5).
  7. One-hot encodes: crop, region, soil_type, weather_condition.

Output
------
  data/processed/features_dataset.csv   (36 features + 1 target)
  data/processed/features_dataset.json  (sidecar metadata)

Target variable: yield_hg_ha  (NOT modified — no log transform at this stage)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.config import settings
from src.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Domain-knowledge constants (crop biology)
# ---------------------------------------------------------------------------
#
# All constants are sourced from peer-reviewed agronomy literature and
# international crop research organisations.  Every value can be verified
# against the cited reference before use in a production model.
# ---------------------------------------------------------------------------

# Optimal annual rainfall (mm) per crop
# Used in: water_stress = |rainfall_mm - optimal| / optimal
# Sources:
#   Wheat   500 mm — FAO Irrigation & Drainage Paper No. 56 (Allen et al., 1998);
#                    CIMMYT Wheat Agronomy Guidelines
#   Maize   650 mm — FAO Plant Production Paper 33 (Doorenbos & Kassam, 1979)
#   Soybean 600 mm — USDA NRCS Agronomy Technical Note No. 1
#   Rice   1200 mm — IRRI Rice Knowledge Bank — Water Management section
CROP_OPTIMAL_RAINFALL_MM: dict[str, float] = {
    "Maize":   650.0,
    "Rice":   1200.0,
    "Soybean": 600.0,
    "Wheat":   500.0,
}

# Upper thermal limit (°C) — temperature above which heat stress occurs
# Used in: heat_stress = max(0, temperature_celsius - T_max)
# Sources:
#   Wheat   25 °C — CIMMYT Wheat Physiology Manual;
#                   Wardlaw & Wrigley (1994) Aust. J. Plant Physiol. 21(6)
#   Maize   32 °C — FAO Plant Production Paper 33; Tollenaar et al. (2004)
#   Soybean 32 °C — USDA-ARS Soybean Physiology; Board & Kahlon (2011)
#   Rice    35 °C — IRRI heat tolerance research; Jagadish et al. (2007)
CROP_MAX_TEMP_C: dict[str, float] = {
    "Maize":   32.0,   # tropical/subtropical cereal
    "Rice":    35.0,   # warm-season paddy crop
    "Soybean": 32.0,   # warm-season legume
    "Wheat":   25.0,   # cool-season cereal
}

# Base temperature (°C) for growing degree day (GDD) calculation
# Used in: gdd_proxy = max(0, temperature_celsius - T_base) * days_to_harvest
# Sources:
#   Wheat    0 °C — McMaster & Wilhelm (1997) Agric. Forest Meteorol. 87; USDA NRCS
#   Soybean  6 °C — USDA GDD tables for soybean phenology
#   Maize   10 °C — Cross & Zuber (1972) Agron. J.; USDA / FAO standard GDD base
#   Rice    10 °C — Yin et al. (1996) Field Crops Res. 48; IRRI GDD reference
CROP_BASE_TEMP_C: dict[str, float] = {
    "Maize":   10.0,
    "Rice":    10.0,
    "Soybean":  6.0,
    "Wheat":    0.0,   # very cold-tolerant
}

# Soil quality score (1 = poor, 5 = excellent)
# Ordinal ranking based on water-retention capacity and cation exchange capacity (CEC)
# Sources: FAO Guidelines for Soil Description (4th ed., 2006);
#          Brady & Weil "The Nature and Properties of Soils" (15th ed.)
#   Loam   5 — optimal sand/silt/clay balance; best drainage + nutrient retention
#   Silt   4 — high fertility, excellent moisture retention, slight compaction risk
#   Clay   3 — high CEC (nutrients retained), but poor drainage / waterlogging risk
#   Peaty  3 — high organic matter and N, may be acidic (pH < 5.5)
#   Chalky 2 — alkaline (pH > 7.5), limits Fe/Mn/Zn micronutrient availability
#   Sandy  1 — fast-draining, low CEC, poor water and nutrient retention
SOIL_QUALITY_SCORE: dict[str, int] = {
    "Loam":   5,
    "Silt":   4,
    "Clay":   3,
    "Peaty":  3,
    "Chalky": 2,
    "Sandy":  1,
}

# Columns that are constants in the year=2013 snapshot → no predictive value
ZERO_VARIANCE_COLS: list[str] = ["year", "fao_pesticides_tonnes"]

# FAO reference cols used to derive anomaly features, then dropped
FAO_CONSUMED_COLS: list[str] = ["fao_rainfall_mm", "fao_avg_temp"]


# ---------------------------------------------------------------------------
# Core transformation
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all features from the merged dataset.

    The returned DataFrame contains all engineered features **plus** the
    target column ``yield_hg_ha``.  The caller is responsible for
    separating X and y before model training.

    Parameters
    ----------
    df : merged dataset (output of merge_datasets())

    Returns
    -------
    feat : DataFrame, shape (n_rows, n_features + 1)
    """
    feat = df.copy()

    # ------------------------------------------------------------------
    # 1. Drop zero-variance constants
    # ------------------------------------------------------------------
    cols_to_drop = [c for c in ZERO_VARIANCE_COLS if c in feat.columns]
    feat = feat.drop(columns=cols_to_drop)
    logger.info(f"Dropped zero-variance columns: {cols_to_drop}")

    # ------------------------------------------------------------------
    # 2. Boolean → integer (required before numeric operations)
    # ------------------------------------------------------------------
    for col in ["fertilizer_used", "irrigation_used"]:
        if col in feat.columns:
            feat[col] = feat[col].astype(int)

    # ------------------------------------------------------------------
    # 3. FAO reference-derived features
    #    rainfall_anomaly : how much local rainfall deviates from
    #                       the 2013 USA national average (715 mm)
    #    temp_anomaly     : how much local temperature deviates from
    #                       the 2013 USA national average (16.44 °C)
    # ------------------------------------------------------------------
    logger.info("Building FAO reference features …")
    feat["rainfall_anomaly"] = (
        feat["rainfall_mm"] - feat["fao_rainfall_mm"]
    ).round(3)
    feat["temp_anomaly"] = (
        feat["temperature_celsius"] - feat["fao_avg_temp"]
    ).round(3)

    # Drop consumed FAO constants (fao_yield_hg_ha kept as crop benchmark)
    consumed = [c for c in FAO_CONSUMED_COLS if c in feat.columns]
    feat = feat.drop(columns=consumed)
    logger.info(f"Dropped FAO reference constants (used for anomalies): {consumed}")

    # ------------------------------------------------------------------
    # 4. Agronomic practice interactions
    # ------------------------------------------------------------------
    logger.info("Building agronomic interaction features …")
    feat["rainfall_x_fertilizer"] = (
        feat["rainfall_mm"] * feat["fertilizer_used"]
    ).round(3)
    feat["rainfall_x_irrigation"] = (
        feat["rainfall_mm"] * feat["irrigation_used"]
    ).round(3)
    feat["temp_x_irrigation"] = (
        feat["temperature_celsius"] * feat["irrigation_used"]
    ).round(3)
    # Combined management intensity: 0 = none, 1 = one practice, 2 = both
    feat["agro_intensity"] = feat["fertilizer_used"] + feat["irrigation_used"]

    # ------------------------------------------------------------------
    # 5. Climate indices
    # ------------------------------------------------------------------
    logger.info("Building climate index features …")

    # Heat-moisture ratio: high → hot and dry (more stress)
    # Units: °C per 100 mm of rainfall
    feat["heat_moisture_ratio"] = (
        feat["temperature_celsius"] / (feat["rainfall_mm"] / 100.0)
    ).round(4)

    # Aridity index: high → cool and wet (favourable in general)
    # Units: mm per °C
    feat["aridity_index"] = (
        feat["rainfall_mm"] / feat["temperature_celsius"]
    ).round(4)

    # Average rainfall received per growing day
    feat["harvest_rainfall_rate"] = (
        feat["rainfall_mm"] / feat["days_to_harvest"]
    ).round(4)

    # ------------------------------------------------------------------
    # 6. Crop-biology features  (domain knowledge)
    # ------------------------------------------------------------------
    logger.info("Building crop biology features …")

    # Water stress: relative deviation from the crop's optimal rainfall need
    # 0 = perfect rainfall match; higher = more stressed (deficit or excess)
    opt_rain = feat["crop"].map(CROP_OPTIMAL_RAINFALL_MM)
    feat["water_stress"] = (
        (feat["rainfall_mm"] - opt_rain).abs() / opt_rain
    ).round(4)

    # Heat stress: degrees of temperature above the crop's upper thermal limit
    # 0 when temperature is within the safe range; positive = excess heat
    max_temp = feat["crop"].map(CROP_MAX_TEMP_C)
    feat["heat_stress"] = (
        feat["temperature_celsius"] - max_temp
    ).clip(lower=0.0).round(3)

    # Growing Degree Days proxy: accumulated thermal units × growing period
    # Captures the total heat energy available during the season
    base_temp = feat["crop"].map(CROP_BASE_TEMP_C)
    feat["gdd_proxy"] = (
        (feat["temperature_celsius"] - base_temp).clip(lower=0.0)
        * feat["days_to_harvest"]
    ).round(1)

    # ------------------------------------------------------------------
    # 7. Soil quality score (ordinal domain knowledge)
    # ------------------------------------------------------------------
    feat["soil_quality_score"] = feat["soil_type"].map(SOIL_QUALITY_SCORE)
    logger.info("Added soil_quality_score (Sandy=1 … Loam=5)")

    # ------------------------------------------------------------------
    # 8. One-hot encoding of categorical features
    # ------------------------------------------------------------------
    cat_cols = ["crop", "region", "soil_type", "weather_condition"]
    feat = pd.get_dummies(feat, columns=cat_cols, drop_first=False, dtype=int)
    logger.info(f"One-hot encoded: {cat_cols}")

    n_features = feat.shape[1] - 1  # exclude target
    logger.info(
        f"Feature matrix ready: {n_features} features + target "
        f"({feat.shape[0]:,} rows)"
    )
    return feat


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def engineer_features(save: bool = True) -> pd.DataFrame:
    """
    Load merged dataset, build features, and optionally persist.

    Parameters
    ----------
    save : write features_dataset.csv + sidecar JSON to data/processed/

    Returns
    -------
    feat : engineered feature DataFrame (includes target yield_hg_ha)
    """
    settings.paths.ensure_dirs()

    # Load merged dataset
    merged_path = settings.paths.merged
    logger.info(f"Loading merged dataset from {merged_path} …")
    df = pd.read_csv(merged_path)
    logger.info(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Build features
    feat = build_features(df)

    # Separate features vs target for summary logging
    target_col = settings.data.target_column  # "yield_hg_ha"
    feature_cols = [c for c in feat.columns if c != target_col]
    logger.info(f"  Features ({len(feature_cols)}): {feature_cols}")

    # Correlation summary vs target
    numeric_feat = feat[feature_cols].select_dtypes(include="number")
    corr = numeric_feat.corrwith(feat[target_col]).abs().sort_values(ascending=False)
    logger.info(f"  Top 10 |correlation| with {target_col}:\n{corr.head(10).round(4).to_string()}")

    if save:
        out_path = settings.paths.features
        feat.to_csv(out_path, index=False)
        logger.info(f"Saved → {out_path}")

        meta = {
            "rows": len(feat),
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "target": target_col,
            "dropped_zero_variance": ZERO_VARIANCE_COLS,
            "fao_consumed": FAO_CONSUMED_COLS,
            "domain_constants": {
                "CROP_OPTIMAL_RAINFALL_MM": CROP_OPTIMAL_RAINFALL_MM,
                "CROP_MAX_TEMP_C": CROP_MAX_TEMP_C,
                "CROP_BASE_TEMP_C": CROP_BASE_TEMP_C,
                "SOIL_QUALITY_SCORE": SOIL_QUALITY_SCORE,
            },
        }
        meta_path = Path(str(out_path)).with_suffix(".json")
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metadata → {meta_path}")

    return feat


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    feat = engineer_features(save=True)

    target_col = settings.data.target_column
    feature_cols = [c for c in feat.columns if c != target_col]

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 65)
    print(f"Shape     : {feat.shape}")
    print(f"Features  : {len(feature_cols)}")
    print(f"Target    : {target_col}")
    print()
    print("All features:")
    for i, col in enumerate(feature_cols, 1):
        dtype = str(feat[col].dtype)
        n_unique = feat[col].nunique()
        print(f"  {i:2d}. {col:<35s}  dtype={dtype:<8s}  unique={n_unique:,}")
    print()
    print(f"Null count: {feat.isnull().sum().sum()}")
    print()

    numeric = feat[feature_cols].select_dtypes(include="number")
    corr = numeric.corrwith(feat[target_col]).abs().sort_values(ascending=False)
    print(f"Top 15 features by |Pearson r| with {target_col}:")
    for col, r in corr.head(15).items():
        bar = "█" * int(r * 30)
        print(f"  {col:<35s}  r={r:.4f}  {bar}")
