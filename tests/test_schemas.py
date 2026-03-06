"""
Tests for src/api/schemas.py — Pydantic request/response validation.

Covers field constraints (ge/le bounds, Literal enums) on
FarmConditionsRequest and PredictRequest.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import FarmConditionsRequest, PredictRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID: dict = {
    "rainfall_mm": 650.0,
    "temperature_celsius": 22.0,
    "days_to_harvest": 120,
    "region": "East",
    "soil_type": "Loam",
    "weather_condition": "Sunny",
    "fertilizer_used": True,
    "irrigation_used": True,
}


def _cond(**overrides) -> dict:
    return {**VALID, **overrides}


# ---------------------------------------------------------------------------
# FarmConditionsRequest — valid input
# ---------------------------------------------------------------------------

class TestFarmConditionsRequestValid:
    def test_all_fields_accepted(self):
        req = FarmConditionsRequest(**VALID)
        assert req.rainfall_mm == 650.0
        assert req.region == "East"

    def test_boundary_rainfall_zero(self):
        req = FarmConditionsRequest(**_cond(rainfall_mm=0))
        assert req.rainfall_mm == 0.0

    def test_boundary_rainfall_max(self):
        req = FarmConditionsRequest(**_cond(rainfall_mm=5000))
        assert req.rainfall_mm == 5000.0

    def test_boundary_temperature_min(self):
        req = FarmConditionsRequest(**_cond(temperature_celsius=-10))
        assert req.temperature_celsius == -10.0

    def test_boundary_temperature_max(self):
        req = FarmConditionsRequest(**_cond(temperature_celsius=50))
        assert req.temperature_celsius == 50.0

    def test_boundary_days_min(self):
        req = FarmConditionsRequest(**_cond(days_to_harvest=1))
        assert req.days_to_harvest == 1

    def test_boundary_days_max(self):
        req = FarmConditionsRequest(**_cond(days_to_harvest=365))
        assert req.days_to_harvest == 365

    @pytest.mark.parametrize("region", ["East", "North", "South", "West"])
    def test_all_regions_accepted(self, region):
        req = FarmConditionsRequest(**_cond(region=region))
        assert req.region == region

    @pytest.mark.parametrize("soil", ["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"])
    def test_all_soil_types_accepted(self, soil):
        req = FarmConditionsRequest(**_cond(soil_type=soil))
        assert req.soil_type == soil

    @pytest.mark.parametrize("weather", ["Cloudy", "Rainy", "Sunny"])
    def test_all_weather_conditions_accepted(self, weather):
        req = FarmConditionsRequest(**_cond(weather_condition=weather))
        assert req.weather_condition == weather


# ---------------------------------------------------------------------------
# FarmConditionsRequest — invalid input raises ValidationError
# ---------------------------------------------------------------------------

class TestFarmConditionsRequestInvalid:
    def test_rainfall_below_zero(self):
        with pytest.raises(ValidationError):
            FarmConditionsRequest(**_cond(rainfall_mm=-1))

    def test_rainfall_above_max(self):
        with pytest.raises(ValidationError):
            FarmConditionsRequest(**_cond(rainfall_mm=5001))

    def test_temperature_too_low(self):
        with pytest.raises(ValidationError):
            FarmConditionsRequest(**_cond(temperature_celsius=-11))

    def test_temperature_too_high(self):
        with pytest.raises(ValidationError):
            FarmConditionsRequest(**_cond(temperature_celsius=51))

    def test_days_zero(self):
        with pytest.raises(ValidationError):
            FarmConditionsRequest(**_cond(days_to_harvest=0))

    def test_days_above_max(self):
        with pytest.raises(ValidationError):
            FarmConditionsRequest(**_cond(days_to_harvest=366))

    def test_invalid_region(self):
        with pytest.raises(ValidationError):
            FarmConditionsRequest(**_cond(region="Central"))

    def test_invalid_soil_type(self):
        with pytest.raises(ValidationError):
            FarmConditionsRequest(**_cond(soil_type="Gravel"))

    def test_invalid_weather_condition(self):
        with pytest.raises(ValidationError):
            FarmConditionsRequest(**_cond(weather_condition="Windy"))


# ---------------------------------------------------------------------------
# PredictRequest — crop Literal
# ---------------------------------------------------------------------------

class TestPredictRequest:
    @pytest.mark.parametrize("crop", ["Maize", "Rice", "Soybean", "Wheat"])
    def test_valid_crops(self, crop):
        req = PredictRequest(conditions=VALID, crop=crop)
        assert req.crop == crop

    def test_invalid_crop_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(conditions=VALID, crop="Barley")

    def test_missing_crop_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(conditions=VALID)
