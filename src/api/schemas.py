"""
Pydantic request / response schemas for the AgroOpt API.

All schemas use strict field validation and clear docstrings so
that the auto-generated OpenAPI docs are self-explanatory.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared request body
# ---------------------------------------------------------------------------

class FarmConditionsRequest(BaseModel):
    """Observed conditions for a single farm / growing season."""

    rainfall_mm: float = Field(
        ...,
        ge=0,
        le=5000,
        description="Seasonal rainfall in millimetres (0–5000).",
        examples=[650.0],
    )
    temperature_celsius: float = Field(
        ...,
        ge=-10,
        le=50,
        description="Mean growing-season temperature in °C (−10 to 50).",
        examples=[22.0],
    )
    days_to_harvest: int = Field(
        ...,
        ge=1,
        le=365,
        description="Duration of the growing season in days (1–365).",
        examples=[120],
    )
    region: Literal["East", "North", "South", "West"] = Field(
        ...,
        description="Geographic region of the farm.",
        examples=["East"],
    )
    soil_type: Literal["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"] = Field(
        ...,
        description="Soil classification.",
        examples=["Loam"],
    )
    weather_condition: Literal["Cloudy", "Rainy", "Sunny"] = Field(
        ...,
        description="Dominant weather pattern during the growing season.",
        examples=["Sunny"],
    )
    fertilizer_used: bool = Field(
        True,
        description="Whether fertilizer is applied.",
    )
    irrigation_used: bool = Field(
        True,
        description="Whether supplemental irrigation is used.",
    )


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for a single-crop yield prediction."""

    conditions: FarmConditionsRequest
    crop: Literal["Maize", "Rice", "Soybean", "Wheat"] = Field(
        ...,
        description="Target crop to predict yield for.",
        examples=["Maize"],
    )


class PredictResponse(BaseModel):
    """Predicted yield for one (conditions, crop) pair."""

    crop: str
    predicted_yield_hg_ha: float = Field(
        ..., description="Predicted yield in hectogram per hectare."
    )
    predicted_yield_t_ha: float = Field(
        ..., description="Predicted yield in tonnes per hectare."
    )


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Request body for crop ranking."""

    conditions: FarmConditionsRequest


class CropRanking(BaseModel):
    """Single entry in the crop recommendation list."""

    rank: int
    crop: str
    predicted_yield_hg_ha: float
    predicted_yield_t_ha: float
    water_stress: float = Field(..., description="0 = no stress, >0 = deviation from optimal rainfall.")
    heat_stress: float = Field(..., description="°C above crop's maximum temperature threshold.")
    fao_benchmark_hg_ha: float = Field(..., description="FAO 2013 USA reference yield (hg/ha).")


class RecommendResponse(BaseModel):
    """Ranked list of all four crops for the given conditions."""

    rankings: list[CropRanking]


# ---------------------------------------------------------------------------
# /optimize
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    """Request body for management input optimisation."""

    conditions: FarmConditionsRequest
    crop: Literal["Maize", "Rice", "Soybean", "Wheat"] = Field(
        ...,
        description="Crop to optimise management inputs for.",
        examples=["Maize"],
    )


class OptimizedConditions(BaseModel):
    """The management inputs that maximise predicted yield."""

    fertilizer_used: bool
    irrigation_used: bool
    days_to_harvest: int


class OptimizeResponse(BaseModel):
    """Result of the management input grid search."""

    crop: str
    best_conditions: OptimizedConditions
    best_yield_hg_ha: float
    best_yield_t_ha: float
    baseline_yield_hg_ha: float
    baseline_yield_t_ha: float
    yield_gain_hg_ha: float
    yield_gain_pct: float


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """API health status."""

    status: str = "ok"
    model: str
    n_features: int
