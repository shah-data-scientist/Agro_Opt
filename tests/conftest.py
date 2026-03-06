"""
Shared pytest fixtures for the AgroOpt test suite.

All fixtures use ``scope="session"`` so the model is trained only once
per test run. The model asset fixture builds a minimal in-memory Ridge
pipeline so tests run without needing the gitignored pkl artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# ---------------------------------------------------------------------------
# Feature names & FAO refs (loaded from committed JSON files)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def feature_cols() -> list[str]:
    """Ordered list of 36 feature names matching the training feature set."""
    with (MODELS_DIR / "feature_names.json").open() as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def fao_refs() -> dict:
    """FAO 2013 USA reference constants (per-crop yield + national averages)."""
    with (MODELS_DIR / "fao_refs.json").open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Minimal in-memory model asset (avoids loading gitignored pkl files)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def assets(feature_cols: list[str], fao_refs: dict) -> dict:
    """
    Minimal sklearn Ridge pipeline trained on synthetic random data.

    Predicts the same target shape as the real model so all engine
    functions can be called without loading the pkl artefact.
    """
    n_features = len(feature_cols)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((500, n_features))
    # Centre yield around 50 000 hg/ha (≈ 5 t/ha), realistic range
    y = rng.standard_normal(500) * 10_000 + 50_000
    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    model.fit(X, y)
    return {"model": model, "feature_cols": feature_cols, "fao_refs": fao_refs}


# ---------------------------------------------------------------------------
# Default FarmConditions (representative USA mid-west scenario)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def conditions():
    """A valid FarmConditions instance used across engine tests."""
    from src.recommendation.engine import FarmConditions
    return FarmConditions(
        rainfall_mm=650.0,
        temperature_celsius=22.0,
        days_to_harvest=120,
        region="East",
        soil_type="Loam",
        weather_condition="Sunny",
        fertilizer_used=True,
        irrigation_used=True,
    )


# ---------------------------------------------------------------------------
# FastAPI TestClient with mocked load_assets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_client(assets: dict):
    """
    FastAPI TestClient with load_assets patched to return the in-memory
    fixture.  This allows the lifespan handler to run without pkl files.
    """
    from fastapi.testclient import TestClient
    from src.api.app import app

    with patch("src.api.app.load_assets", return_value=assets):
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# Reusable valid request payload
# ---------------------------------------------------------------------------

VALID_CONDITIONS: dict = {
    "rainfall_mm": 650.0,
    "temperature_celsius": 22.0,
    "days_to_harvest": 120,
    "region": "East",
    "soil_type": "Loam",
    "weather_condition": "Sunny",
    "fertilizer_used": True,
    "irrigation_used": True,
}
