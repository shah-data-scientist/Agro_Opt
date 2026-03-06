"""
Tests for src/recommendation/engine.py — FarmConditions validation,
feature vector construction, prediction, recommendation, and optimisation.

All tests use the in-memory ``assets`` fixture from conftest.py so no
pkl artefacts are required.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.recommendation.engine import (
    FarmConditions,
    CROPS,
    build_feature_vector,
    predict_yield,
    recommend_crop,
    optimize_conditions,
)


# ---------------------------------------------------------------------------
# FarmConditions.validate()
# ---------------------------------------------------------------------------

class TestFarmConditionsValidate:
    def test_valid_conditions_pass(self, conditions):
        conditions.validate()  # should not raise

    def test_negative_rainfall_raises(self):
        fc = FarmConditions(
            rainfall_mm=-1,
            temperature_celsius=20,
            days_to_harvest=100,
            region="East",
            soil_type="Loam",
            weather_condition="Sunny",
        )
        with pytest.raises(ValueError, match="rainfall_mm"):
            fc.validate()

    def test_temperature_too_low_raises(self):
        fc = FarmConditions(
            rainfall_mm=500,
            temperature_celsius=-15,
            days_to_harvest=100,
            region="East",
            soil_type="Loam",
            weather_condition="Sunny",
        )
        with pytest.raises(ValueError, match="temperature_celsius"):
            fc.validate()

    def test_temperature_too_high_raises(self):
        fc = FarmConditions(
            rainfall_mm=500,
            temperature_celsius=55,
            days_to_harvest=100,
            region="East",
            soil_type="Loam",
            weather_condition="Sunny",
        )
        with pytest.raises(ValueError, match="temperature_celsius"):
            fc.validate()

    def test_zero_days_to_harvest_raises(self):
        fc = FarmConditions(
            rainfall_mm=500,
            temperature_celsius=20,
            days_to_harvest=0,
            region="East",
            soil_type="Loam",
            weather_condition="Sunny",
        )
        with pytest.raises(ValueError, match="days_to_harvest"):
            fc.validate()


# ---------------------------------------------------------------------------
# build_feature_vector()
# ---------------------------------------------------------------------------

class TestBuildFeatureVector:
    def test_output_shape(self, conditions, fao_refs, feature_cols):
        vec = build_feature_vector(conditions, "Maize", fao_refs, feature_cols)
        assert vec.shape == (1, 36)

    def test_output_dtype_float64(self, conditions, fao_refs, feature_cols):
        vec = build_feature_vector(conditions, "Maize", fao_refs, feature_cols)
        assert vec.dtype == np.float64

    def test_no_nans(self, conditions, fao_refs, feature_cols):
        for crop in CROPS:
            vec = build_feature_vector(conditions, crop, fao_refs, feature_cols)
            assert not np.isnan(vec).any(), f"NaN found for crop={crop}"

    @pytest.mark.parametrize("crop", CROPS)
    def test_all_crops_produce_vector(self, conditions, fao_refs, feature_cols, crop):
        vec = build_feature_vector(conditions, crop, fao_refs, feature_cols)
        assert vec.shape == (1, 36)

    def test_vectors_differ_across_crops(self, conditions, fao_refs, feature_cols):
        """Different crops should produce different feature vectors."""
        vectors = [
            build_feature_vector(conditions, c, fao_refs, feature_cols)
            for c in CROPS
        ]
        for i in range(len(vectors) - 1):
            assert not np.allclose(vectors[i], vectors[i + 1])

    def test_fertilizer_flag_affects_vector(self, fao_refs, feature_cols):
        fc_on = FarmConditions(
            rainfall_mm=600, temperature_celsius=20, days_to_harvest=120,
            region="East", soil_type="Loam", weather_condition="Sunny",
            fertilizer_used=True, irrigation_used=False,
        )
        fc_off = FarmConditions(
            rainfall_mm=600, temperature_celsius=20, days_to_harvest=120,
            region="East", soil_type="Loam", weather_condition="Sunny",
            fertilizer_used=False, irrigation_used=False,
        )
        v_on = build_feature_vector(fc_on, "Maize", fao_refs, feature_cols)
        v_off = build_feature_vector(fc_off, "Maize", fao_refs, feature_cols)
        assert not np.allclose(v_on, v_off)


# ---------------------------------------------------------------------------
# predict_yield()
# ---------------------------------------------------------------------------

class TestPredictYield:
    @pytest.mark.parametrize("crop", CROPS)
    def test_returns_float(self, conditions, crop, assets):
        result = predict_yield(conditions, crop, assets)
        assert isinstance(result, float)

    @pytest.mark.parametrize("crop", CROPS)
    def test_returns_positive_value(self, conditions, crop, assets):
        result = predict_yield(conditions, crop, assets)
        assert result > 0, f"Yield should be positive, got {result} for {crop}"

    def test_high_rainfall_increases_yield(self, fao_refs, feature_cols, assets):
        """Higher rainfall should generally increase predicted yield."""
        fc_low = FarmConditions(
            rainfall_mm=100, temperature_celsius=22, days_to_harvest=120,
            region="East", soil_type="Loam", weather_condition="Sunny",
        )
        fc_high = FarmConditions(
            rainfall_mm=2000, temperature_celsius=22, days_to_harvest=120,
            region="East", soil_type="Loam", weather_condition="Sunny",
        )
        y_low = predict_yield(fc_low, "Maize", assets)
        y_high = predict_yield(fc_high, "Maize", assets)
        # Rainfall is the dominant feature (r=0.764) — high should beat low
        assert y_high > y_low


# ---------------------------------------------------------------------------
# recommend_crop()
# ---------------------------------------------------------------------------

class TestRecommendCrop:
    def test_returns_four_rankings(self, conditions, assets):
        rankings = recommend_crop(conditions, assets)
        assert len(rankings) == 4

    def test_rankings_contain_all_crops(self, conditions, assets):
        rankings = recommend_crop(conditions, assets)
        crops_returned = {r["crop"] for r in rankings}
        assert crops_returned == set(CROPS)

    def test_rankings_sorted_descending(self, conditions, assets):
        rankings = recommend_crop(conditions, assets)
        yields = [r["predicted_yield_hg_ha"] for r in rankings]
        assert yields == sorted(yields, reverse=True)

    def test_rank_field_sequential(self, conditions, assets):
        rankings = recommend_crop(conditions, assets)
        ranks = [r["rank"] for r in rankings]
        assert ranks == list(range(1, 5))

    def test_stress_fields_present(self, conditions, assets):
        rankings = recommend_crop(conditions, assets)
        for row in rankings:
            assert "water_stress" in row
            assert "heat_stress" in row
            assert row["water_stress"] >= 0
            assert row["heat_stress"] >= 0

    def test_fao_benchmark_present(self, conditions, assets):
        rankings = recommend_crop(conditions, assets)
        for row in rankings:
            assert "fao_benchmark_hg_ha" in row
            assert row["fao_benchmark_hg_ha"] > 0


# ---------------------------------------------------------------------------
# optimize_conditions()
# ---------------------------------------------------------------------------

class TestOptimizeConditions:
    def test_returns_expected_keys(self, conditions, assets):
        result = optimize_conditions(conditions, "Maize", assets)
        for key in ("crop", "best_conditions", "best_yield_hg_ha",
                    "baseline_yield_hg_ha", "yield_gain_hg_ha", "yield_gain_pct"):
            assert key in result

    def test_best_yield_gte_baseline(self, conditions, assets):
        result = optimize_conditions(conditions, "Maize", assets)
        assert result["best_yield_hg_ha"] >= result["baseline_yield_hg_ha"]

    def test_yield_gain_non_negative(self, conditions, assets):
        result = optimize_conditions(conditions, "Maize", assets)
        assert result["yield_gain_hg_ha"] >= 0
        assert result["yield_gain_pct"] >= 0

    @pytest.mark.parametrize("crop", CROPS)
    def test_all_crops_optimised(self, conditions, crop, assets):
        result = optimize_conditions(conditions, crop, assets)
        assert result["crop"] == crop
        assert result["best_yield_hg_ha"] > 0

    def test_best_conditions_has_bool_management(self, conditions, assets):
        result = optimize_conditions(conditions, "Wheat", assets)
        bc = result["best_conditions"]
        assert isinstance(bc.fertilizer_used, bool)
        assert isinstance(bc.irrigation_used, bool)
        assert isinstance(bc.days_to_harvest, int)
