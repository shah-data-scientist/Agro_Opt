"""
Phase 7 — FastAPI application for AgroOpt.

Endpoints
---------
GET  /health     — liveness probe; confirms model is loaded
POST /predict    — predict yield for one (conditions, crop) pair
POST /recommend  — rank all four crops by predicted yield
POST /optimize   — grid-search management inputs for a target crop

The trained Ridge pipeline and FAO reference constants are loaded once
at startup via the lifespan context manager and stored in ``app.state``
so every request reuses the same in-memory assets.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status
from loguru import logger

from src.recommendation.engine import (
    FarmConditions,
    load_assets,
    predict_yield,
    recommend_crop,
    optimize_conditions,
)
from src.api.schemas import (
    FarmConditionsRequest,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    CropRanking,
    OptimizeRequest,
    OptimizeResponse,
    OptimizedConditions,
    HealthResponse,
)
from src.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_farm_conditions(req: FarmConditionsRequest) -> FarmConditions:
    """Convert a Pydantic request model to the engine dataclass."""
    return FarmConditions(
        rainfall_mm=req.rainfall_mm,
        temperature_celsius=req.temperature_celsius,
        days_to_harvest=req.days_to_harvest,
        region=req.region,
        soil_type=req.soil_type,
        weather_condition=req.weather_condition,
        fertilizer_used=req.fertilizer_used,
        irrigation_used=req.irrigation_used,
    )


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load all ML assets into app.state at startup; release at shutdown."""
    setup_logging()
    logger.info("AgroOpt API starting — loading model assets …")
    app.state.assets = load_assets()
    model_name = type(app.state.assets["model"].named_steps.get("ridge",
                 app.state.assets["model"])).__name__
    n_features = len(app.state.assets["feature_cols"])
    logger.info(f"Model ready: {model_name}, {n_features} features")
    app.state.model_name = model_name
    app.state.n_features = n_features
    yield
    logger.info("AgroOpt API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgroOpt API",
    description=(
        "Data-driven crop yield prediction and recommendation API. "
        "Powered by a Ridge regression pipeline trained on 666 K synthetic "
        "USA farm records (R² = 0.913)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["Health"],
)
def health() -> HealthResponse:
    """Return API status and confirm the model is loaded."""
    return HealthResponse(
        status="ok",
        model=app.state.model_name,
        n_features=app.state.n_features,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict yield for a single crop",
    tags=["Prediction"],
)
def predict(body: PredictRequest) -> PredictResponse:
    """
    Predict the yield (hg/ha and t/ha) for a specified crop given the
    farm's local conditions.

    The Ridge pipeline is applied to the 36-feature vector reconstructed
    from the request inputs using the same feature engineering logic as
    the training phase.
    """
    try:
        conditions = _to_farm_conditions(body.conditions)
        conditions.validate()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    y_hg_ha = predict_yield(conditions, body.crop, app.state.assets)
    return PredictResponse(
        crop=body.crop,
        predicted_yield_hg_ha=round(y_hg_ha, 1),
        predicted_yield_t_ha=round(y_hg_ha / 10_000, 3),
    )


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Rank all crops by predicted yield",
    tags=["Recommendation"],
)
def recommend(body: RecommendRequest) -> RecommendResponse:
    """
    Evaluate all four crops (Maize, Rice, Soybean, Wheat) for the given
    farm conditions and return them ranked by predicted yield, highest first.

    Each entry also includes agro-stress indicators (water_stress,
    heat_stress) and the FAO 2013 USA benchmark yield for comparison.
    """
    try:
        conditions = _to_farm_conditions(body.conditions)
        rankings_raw = recommend_crop(conditions, app.state.assets)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    rankings = [CropRanking(**row) for row in rankings_raw]
    return RecommendResponse(rankings=rankings)


@app.post(
    "/optimize",
    response_model=OptimizeResponse,
    summary="Optimise management inputs for a target crop",
    tags=["Optimisation"],
)
def optimize(body: OptimizeRequest) -> OptimizeResponse:
    """
    Grid-search over management inputs (fertilizer, irrigation,
    days_to_harvest 60–200 step 10) to find the combination that
    maximises predicted yield for the specified crop.

    Climate, soil, and region inputs are treated as fixed — they reflect
    observed conditions the farmer cannot change.
    """
    try:
        conditions = _to_farm_conditions(body.conditions)
        result = optimize_conditions(conditions, body.crop, app.state.assets)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    bc = result["best_conditions"]
    return OptimizeResponse(
        crop=result["crop"],
        best_conditions=OptimizedConditions(
            fertilizer_used=bc.fertilizer_used,
            irrigation_used=bc.irrigation_used,
            days_to_harvest=bc.days_to_harvest,
        ),
        best_yield_hg_ha=result["best_yield_hg_ha"],
        best_yield_t_ha=result["best_yield_t_ha"],
        baseline_yield_hg_ha=result["baseline_yield_hg_ha"],
        baseline_yield_t_ha=result["baseline_yield_t_ha"],
        yield_gain_hg_ha=result["yield_gain_hg_ha"],
        yield_gain_pct=result["yield_gain_pct"],
    )
