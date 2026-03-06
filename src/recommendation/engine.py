"""
Phase 6 — Recommendation Engine for AgroOpt.

Given a farm's local conditions (climate, soil, management practices),
this module:

  1. Reconstructs the exact 36-feature vector expected by the trained Ridge
     pipeline (mirroring feature_engineering.py logic, using no data-derived
     statistics — only domain constants and the provided inputs).
  2. Predicts crop yield (hg/ha) for each of the four crops.
  3. Returns a ranked recommendation list.
  4. Optionally optimises management inputs (fertilizer, irrigation) to
     maximise predicted yield for a specified crop.

Public API
----------
  load_assets()                        → assets dict
  predict_yield(conditions, crop, assets) → float (hg/ha)
  recommend_crop(conditions, assets)   → list[dict] ranked by yield
  optimize_conditions(conditions, crop, assets) → dict

Inputs (FarmConditions)
-----------------------
  rainfall_mm          : float  — observed seasonal rainfall (mm)
  temperature_celsius  : float  — mean growing-season temperature (°C)
  days_to_harvest      : int    — length of growing season (days)
  region               : str    — one of East / North / South / West
  soil_type            : str    — one of Chalky / Clay / Loam / Peaty / Sandy / Silt
  weather_condition    : str    — one of Cloudy / Rainy / Sunny
  fertilizer_used      : bool   — whether fertilizer is applied (default True)
  irrigation_used      : bool   — whether irrigation is used (default True)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import joblib
from loguru import logger

from src.utils.config import settings

# ---------------------------------------------------------------------------
# Valid categorical options (must match training data)
# ---------------------------------------------------------------------------
CROPS: list[str] = ["Maize", "Rice", "Soybean", "Wheat"]
REGIONS: list[str] = ["East", "North", "South", "West"]
SOIL_TYPES: list[str] = ["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"]
WEATHER_CONDITIONS: list[str] = ["Cloudy", "Rainy", "Sunny"]

# ---------------------------------------------------------------------------
# Domain-knowledge constants (mirrors feature_engineering.py)
# ---------------------------------------------------------------------------
CROP_OPTIMAL_RAINFALL_MM: dict[str, float] = {
    "Maize":   650.0,
    "Rice":   1200.0,
    "Soybean": 600.0,
    "Wheat":   500.0,
}
CROP_MAX_TEMP_C: dict[str, float] = {
    "Maize":   32.0,
    "Rice":    35.0,
    "Soybean": 32.0,
    "Wheat":   25.0,
}
CROP_BASE_TEMP_C: dict[str, float] = {
    "Maize":   10.0,
    "Rice":    10.0,
    "Soybean":  6.0,
    "Wheat":    0.0,
}
SOIL_QUALITY_SCORE: dict[str, int] = {
    "Loam":   5,
    "Silt":   4,
    "Clay":   3,
    "Peaty":  3,
    "Chalky": 2,
    "Sandy":  1,
}


# ---------------------------------------------------------------------------
# Farm conditions input model
# ---------------------------------------------------------------------------

@dataclass
class FarmConditions:
    """
    Observed conditions for a single farm / growing season.

    Parameters
    ----------
    rainfall_mm          : Seasonal rainfall in millimetres.
    temperature_celsius  : Mean growing-season temperature in °C.
    days_to_harvest      : Duration of the growing season in days.
    region               : Geographic region (East / North / South / West).
    soil_type            : Soil classification.
    weather_condition    : Dominant weather pattern (Cloudy / Rainy / Sunny).
    fertilizer_used      : True if fertilizer is applied.
    irrigation_used      : True if supplemental irrigation is used.
    """

    rainfall_mm: float
    temperature_celsius: float
    days_to_harvest: int
    region: Literal["East", "North", "South", "West"]
    soil_type: Literal["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"]
    weather_condition: Literal["Cloudy", "Rainy", "Sunny"]
    fertilizer_used: bool = True
    irrigation_used: bool = True

    def validate(self) -> None:
        """Raise ValueError for any out-of-range or invalid field."""
        if self.rainfall_mm < 0:
            raise ValueError(f"rainfall_mm must be ≥ 0, got {self.rainfall_mm}")
        if not (-10 <= self.temperature_celsius <= 50):
            raise ValueError(
                f"temperature_celsius must be in [-10, 50], got {self.temperature_celsius}"
            )
        if self.days_to_harvest <= 0:
            raise ValueError(f"days_to_harvest must be > 0, got {self.days_to_harvest}")
        if self.region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}, got {self.region!r}")
        if self.soil_type not in SOIL_TYPES:
            raise ValueError(f"soil_type must be one of {SOIL_TYPES}, got {self.soil_type!r}")
        if self.weather_condition not in WEATHER_CONDITIONS:
            raise ValueError(
                f"weather_condition must be one of {WEATHER_CONDITIONS}, "
                f"got {self.weather_condition!r}"
            )


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------

def load_assets() -> dict:
    """
    Load all artefacts needed by the recommendation engine.

    Returns
    -------
    dict with keys:
      - ``model``        : fitted sklearn Pipeline (StandardScaler + Ridge)
      - ``feature_cols`` : list[str] of the 36 feature names (in order)
      - ``fao_refs``     : dict with FAO 2013 USA reference constants
                           {crop: fao_yield_hg_ha, "fao_rainfall_mm": …, "fao_avg_temp": …}
    """
    # Load trained pipeline
    model_path = settings.paths.best_model
    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Load feature column names
    feat_path = settings.paths.feature_names
    with feat_path.open() as f:
        feature_cols: list[str] = json.load(f)
    logger.info(f"Loaded {len(feature_cols)} feature names")

    # Load FAO reference constants from merged dataset (per-crop constants)
    fao_refs = _load_fao_refs()

    return {"model": model, "feature_cols": feature_cols, "fao_refs": fao_refs}


def _load_fao_refs() -> dict:
    """
    Load per-crop FAO 2013 USA reference values.

    Tries ``models/fao_refs.json`` first (lightweight; suitable for Docker
    where ``data/`` is not present).  Falls back to reading the full merged
    dataset CSV when the JSON does not exist (development / notebook use).

    Returns
    -------
    dict with keys:
      - ``fao_yield_hg_ha`` : {crop: float}
      - ``fao_rainfall_mm`` : float  (USA national average, 2013)
      - ``fao_avg_temp``    : float  (USA national average, 2013)
    """
    json_path = settings.paths.models_dir / "fao_refs.json"
    if json_path.exists():
        logger.info(f"Loading FAO refs from {json_path}")
        with json_path.open() as f:
            refs = json.load(f)
        fao_yield: dict = refs["fao_yield_hg_ha"]
        fao_rainfall_mm: float = float(refs["fao_rainfall_mm"])
        fao_avg_temp: float = float(refs["fao_avg_temp"])
    else:
        merged_path = settings.paths.merged
        # Read only the FAO columns — avoids loading the full 666 K-row file
        cols = ["crop", "fao_yield_hg_ha", "fao_rainfall_mm", "fao_avg_temp"]
        df = pd.read_csv(merged_path, usecols=cols)
        fao_yield = (
            df.groupby("crop")["fao_yield_hg_ha"]
            .first()
            .to_dict()
        )
        fao_rainfall_mm = float(df["fao_rainfall_mm"].iloc[0])
        fao_avg_temp = float(df["fao_avg_temp"].iloc[0])

    logger.info(
        f"FAO 2013 USA refs — rainfall: {fao_rainfall_mm:.1f} mm, "
        f"avg_temp: {fao_avg_temp:.2f} °C"
    )
    for crop, y in fao_yield.items():
        logger.debug(f"  fao_yield_hg_ha [{crop}]: {y:,.0f} hg/ha")

    return {
        "fao_yield_hg_ha": fao_yield,
        "fao_rainfall_mm": fao_rainfall_mm,
        "fao_avg_temp": fao_avg_temp,
    }


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

def build_feature_vector(
    conditions: FarmConditions,
    crop: str,
    fao_refs: dict,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Reconstruct the 36-element feature vector for a (conditions, crop) pair.

    The logic mirrors ``feature_engineering.build_features()`` exactly:
    all derived features are computed from the input values and domain
    constants only — no data-derived statistics are needed, so there is
    no risk of data leakage at inference time.

    Parameters
    ----------
    conditions  : FarmConditions
    crop        : one of CROPS
    fao_refs    : output of _load_fao_refs()
    feature_cols: ordered list of 36 feature names from feature_names.json

    Returns
    -------
    np.ndarray of shape (1, 36), dtype float64
    """
    r = conditions.rainfall_mm
    t = conditions.temperature_celsius
    d = conditions.days_to_harvest
    fert = int(conditions.fertilizer_used)
    irri = int(conditions.irrigation_used)

    fao_rain = fao_refs["fao_rainfall_mm"]
    fao_temp = fao_refs["fao_avg_temp"]
    fao_yield = fao_refs["fao_yield_hg_ha"][crop]

    # --- Derived scalars ---
    rainfall_anomaly = round(r - fao_rain, 3)
    temp_anomaly = round(t - fao_temp, 3)

    rainfall_x_fertilizer = round(r * fert, 3)
    rainfall_x_irrigation = round(r * irri, 3)
    temp_x_irrigation = round(t * irri, 3)
    agro_intensity = fert + irri

    heat_moisture_ratio = round(t / (r / 100.0), 4) if r != 0 else 0.0
    aridity_index = round(r / t, 4) if t != 0 else 0.0
    harvest_rainfall_rate = round(r / d, 4) if d != 0 else 0.0

    opt_rain = CROP_OPTIMAL_RAINFALL_MM[crop]
    water_stress = round(abs(r - opt_rain) / opt_rain, 4)

    t_max = CROP_MAX_TEMP_C[crop]
    heat_stress = round(max(0.0, t - t_max), 3)

    t_base = CROP_BASE_TEMP_C[crop]
    gdd_proxy = round(max(0.0, t - t_base) * d, 1)

    soil_quality_score = SOIL_QUALITY_SCORE[conditions.soil_type]

    # Build scalar features dict
    scalars: dict[str, float] = {
        "rainfall_mm":           r,
        "temperature_celsius":   t,
        "fertilizer_used":       float(fert),
        "irrigation_used":       float(irri),
        "days_to_harvest":       float(d),
        "fao_yield_hg_ha":       fao_yield,
        "rainfall_anomaly":      rainfall_anomaly,
        "temp_anomaly":          temp_anomaly,
        "rainfall_x_fertilizer": rainfall_x_fertilizer,
        "rainfall_x_irrigation": rainfall_x_irrigation,
        "temp_x_irrigation":     temp_x_irrigation,
        "agro_intensity":        float(agro_intensity),
        "heat_moisture_ratio":   heat_moisture_ratio,
        "aridity_index":         aridity_index,
        "harvest_rainfall_rate": harvest_rainfall_rate,
        "water_stress":          water_stress,
        "heat_stress":           heat_stress,
        "gdd_proxy":             gdd_proxy,
        "soil_quality_score":    float(soil_quality_score),
    }

    # One-hot encodings
    for c in CROPS:
        scalars[f"crop_{c}"] = 1.0 if crop == c else 0.0
    for reg in REGIONS:
        scalars[f"region_{reg}"] = 1.0 if conditions.region == reg else 0.0
    for st in SOIL_TYPES:
        scalars[f"soil_type_{st}"] = 1.0 if conditions.soil_type == st else 0.0
    for wc in WEATHER_CONDITIONS:
        scalars[f"weather_condition_{wc}"] = 1.0 if conditions.weather_condition == wc else 0.0

    # Assemble in exact column order
    row = np.array([scalars[col] for col in feature_cols], dtype=np.float64)
    return row.reshape(1, -1)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_yield(
    conditions: FarmConditions,
    crop: str,
    assets: dict,
) -> float:
    """
    Predict yield (hg/ha) for the given farm conditions and crop.

    Parameters
    ----------
    conditions : FarmConditions
    crop       : one of CROPS
    assets     : output of load_assets()

    Returns
    -------
    float — predicted yield in hg/ha
    """
    if crop not in CROPS:
        raise ValueError(f"crop must be one of {CROPS}, got {crop!r}")

    X = build_feature_vector(
        conditions, crop, assets["fao_refs"], assets["feature_cols"]
    )
    pred: float = float(assets["model"].predict(X)[0])
    return max(0.0, pred)  # yield cannot be negative


# ---------------------------------------------------------------------------
# Crop recommendation
# ---------------------------------------------------------------------------

def recommend_crop(
    conditions: FarmConditions,
    assets: dict,
) -> list[dict]:
    """
    Rank all four crops by predicted yield for the given farm conditions.

    Parameters
    ----------
    conditions : FarmConditions (validated before call)
    assets     : output of load_assets()

    Returns
    -------
    list of dicts, sorted by predicted yield (highest first):
      [{"rank": 1, "crop": "Maize", "predicted_yield_hg_ha": 45321.4,
        "predicted_yield_t_ha": 4.53, "water_stress": 0.12, "heat_stress": 0.0}, …]
    """
    conditions.validate()

    results = []
    for crop in CROPS:
        y_hg_ha = predict_yield(conditions, crop, assets)
        ws = round(
            abs(conditions.rainfall_mm - CROP_OPTIMAL_RAINFALL_MM[crop])
            / CROP_OPTIMAL_RAINFALL_MM[crop],
            4,
        )
        hs = round(max(0.0, conditions.temperature_celsius - CROP_MAX_TEMP_C[crop]), 3)
        results.append(
            {
                "crop":                   crop,
                "predicted_yield_hg_ha":  round(y_hg_ha, 1),
                "predicted_yield_t_ha":   round(y_hg_ha / 10_000, 3),
                "water_stress":           ws,
                "heat_stress":            hs,
                "fao_benchmark_hg_ha":    assets["fao_refs"]["fao_yield_hg_ha"][crop],
            }
        )

    results.sort(key=lambda d: d["predicted_yield_hg_ha"], reverse=True)
    for i, row in enumerate(results, 1):
        row["rank"] = i

    return results


# ---------------------------------------------------------------------------
# Condition optimisation
# ---------------------------------------------------------------------------

def optimize_conditions(
    conditions: FarmConditions,
    crop: str,
    assets: dict,
) -> dict:
    """
    Grid-search over management inputs to find the combination that
    maximises predicted yield for a given crop and location.

    Searched inputs
    ---------------
    - fertilizer_used  : {True, False}
    - irrigation_used  : {True, False}
    - days_to_harvest  : 60 → 200, step 10

    The climate/soil/region inputs are treated as fixed (they reflect
    observed conditions the farmer cannot change).

    Parameters
    ----------
    conditions : FarmConditions — baseline farm conditions
    crop       : target crop
    assets     : output of load_assets()

    Returns
    -------
    dict with keys:
      - ``best_conditions`` : FarmConditions with optimal management inputs
      - ``best_yield_hg_ha``: float
      - ``best_yield_t_ha`` : float
      - ``baseline_yield_hg_ha``  : float  (original conditions)
      - ``yield_gain_hg_ha``      : float  (improvement from optimisation)
      - ``grid_results``          : list[dict] all grid combinations sorted by yield
    """
    if crop not in CROPS:
        raise ValueError(f"crop must be one of {CROPS}, got {crop!r}")
    conditions.validate()

    baseline_yield = predict_yield(conditions, crop, assets)
    days_range = list(range(60, 210, 10))

    grid_results = []
    for fert in (True, False):
        for irri in (True, False):
            for days in days_range:
                candidate = FarmConditions(
                    rainfall_mm=conditions.rainfall_mm,
                    temperature_celsius=conditions.temperature_celsius,
                    days_to_harvest=days,
                    region=conditions.region,
                    soil_type=conditions.soil_type,
                    weather_condition=conditions.weather_condition,
                    fertilizer_used=fert,
                    irrigation_used=irri,
                )
                y = predict_yield(candidate, crop, assets)
                grid_results.append(
                    {
                        "fertilizer_used": fert,
                        "irrigation_used": irri,
                        "days_to_harvest": days,
                        "predicted_yield_hg_ha": round(y, 1),
                        "predicted_yield_t_ha": round(y / 10_000, 3),
                    }
                )

    grid_results.sort(key=lambda d: d["predicted_yield_hg_ha"], reverse=True)
    best = grid_results[0]

    best_conditions = FarmConditions(
        rainfall_mm=conditions.rainfall_mm,
        temperature_celsius=conditions.temperature_celsius,
        days_to_harvest=best["days_to_harvest"],
        region=conditions.region,
        soil_type=conditions.soil_type,
        weather_condition=conditions.weather_condition,
        fertilizer_used=best["fertilizer_used"],
        irrigation_used=best["irrigation_used"],
    )

    return {
        "crop":                  crop,
        "best_conditions":       best_conditions,
        "best_yield_hg_ha":      best["predicted_yield_hg_ha"],
        "best_yield_t_ha":       best["predicted_yield_t_ha"],
        "baseline_yield_hg_ha":  round(baseline_yield, 1),
        "baseline_yield_t_ha":   round(baseline_yield / 10_000, 3),
        "yield_gain_hg_ha":      round(best["predicted_yield_hg_ha"] - baseline_yield, 1),
        "yield_gain_pct":        round(
            (best["predicted_yield_hg_ha"] - baseline_yield) / baseline_yield * 100, 2
        ) if baseline_yield > 0 else 0.0,
        "grid_results":          grid_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging()
    assets = load_assets()

    # Example: midwest farm — warm, moderate rainfall, loam soil
    farm = FarmConditions(
        rainfall_mm=650.0,
        temperature_celsius=22.0,
        days_to_harvest=120,
        region="East",
        soil_type="Loam",
        weather_condition="Sunny",
        fertilizer_used=True,
        irrigation_used=True,
    )

    print("\n" + "=" * 65)
    print("CROP RECOMMENDATION")
    print("=" * 65)
    recs = recommend_crop(farm, assets)
    for r in recs:
        print(
            f"  #{r['rank']} {r['crop']:<10s} "
            f"{r['predicted_yield_t_ha']:.3f} t/ha  "
            f"(water_stress={r['water_stress']:.2f}, "
            f"heat_stress={r['heat_stress']:.1f}°C)"
        )

    print("\n" + "=" * 65)
    print(f"CONDITION OPTIMISATION — {recs[0]['crop']}")
    print("=" * 65)
    opt = optimize_conditions(farm, recs[0]["crop"], assets)
    bc = opt["best_conditions"]
    print(f"  Baseline yield  : {opt['baseline_yield_t_ha']:.3f} t/ha")
    print(f"  Optimised yield : {opt['best_yield_t_ha']:.3f} t/ha")
    print(f"  Gain            : +{opt['yield_gain_pct']:.1f}%")
    print(f"  Best config     : fertilizer={bc.fertilizer_used}, "
          f"irrigation={bc.irrigation_used}, days={bc.days_to_harvest}")
