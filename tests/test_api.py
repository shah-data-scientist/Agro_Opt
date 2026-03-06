"""
Tests for src/api/app.py — FastAPI endpoint integration tests.

Uses the ``test_client`` fixture from conftest.py which patches
``load_assets`` with an in-memory model so no pkl files are required.
"""

from __future__ import annotations

import pytest
from tests.conftest import VALID_CONDITIONS


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200

    def test_response_shape(self, test_client):
        data = test_client.get("/health").json()
        assert "status" in data
        assert "model" in data
        assert "n_features" in data

    def test_status_ok(self, test_client):
        data = test_client.get("/health").json()
        assert data["status"] == "ok"

    def test_n_features_36(self, test_client):
        data = test_client.get("/health").json()
        assert data["n_features"] == 36


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def _payload(self, crop="Maize", **cond_overrides):
        return {
            "conditions": {**VALID_CONDITIONS, **cond_overrides},
            "crop": crop,
        }

    def test_status_200(self, test_client):
        resp = test_client.post("/predict", json=self._payload())
        assert resp.status_code == 200

    def test_response_contains_crop(self, test_client):
        data = test_client.post("/predict", json=self._payload("Rice")).json()
        assert data["crop"] == "Rice"

    def test_response_contains_yield_fields(self, test_client):
        data = test_client.post("/predict", json=self._payload()).json()
        assert "predicted_yield_hg_ha" in data
        assert "predicted_yield_t_ha" in data

    def test_yields_are_positive(self, test_client):
        data = test_client.post("/predict", json=self._payload()).json()
        assert data["predicted_yield_hg_ha"] > 0
        assert data["predicted_yield_t_ha"] > 0

    def test_t_ha_equals_hg_ha_divided_by_10000(self, test_client):
        data = test_client.post("/predict", json=self._payload()).json()
        assert abs(data["predicted_yield_t_ha"] - data["predicted_yield_hg_ha"] / 10_000) < 1e-3

    @pytest.mark.parametrize("crop", ["Maize", "Rice", "Soybean", "Wheat"])
    def test_all_crops_return_200(self, test_client, crop):
        resp = test_client.post("/predict", json=self._payload(crop))
        assert resp.status_code == 200

    def test_invalid_crop_returns_422(self, test_client):
        resp = test_client.post("/predict", json=self._payload("Barley"))
        assert resp.status_code == 422

    def test_missing_conditions_returns_422(self, test_client):
        resp = test_client.post("/predict", json={"crop": "Maize"})
        assert resp.status_code == 422

    def test_negative_rainfall_returns_422(self, test_client):
        resp = test_client.post("/predict", json=self._payload(rainfall_mm=-1))
        assert resp.status_code == 422

    def test_invalid_region_returns_422(self, test_client):
        resp = test_client.post("/predict", json=self._payload(region="Central"))
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------

class TestRecommend:
    def _payload(self, **cond_overrides):
        return {"conditions": {**VALID_CONDITIONS, **cond_overrides}}

    def test_status_200(self, test_client):
        resp = test_client.post("/recommend", json=self._payload())
        assert resp.status_code == 200

    def test_returns_four_rankings(self, test_client):
        data = test_client.post("/recommend", json=self._payload()).json()
        assert len(data["rankings"]) == 4

    def test_rankings_sorted_descending(self, test_client):
        rankings = test_client.post("/recommend", json=self._payload()).json()["rankings"]
        yields = [r["predicted_yield_hg_ha"] for r in rankings]
        assert yields == sorted(yields, reverse=True)

    def test_rank_field_sequential(self, test_client):
        rankings = test_client.post("/recommend", json=self._payload()).json()["rankings"]
        assert [r["rank"] for r in rankings] == [1, 2, 3, 4]

    def test_all_four_crops_present(self, test_client):
        rankings = test_client.post("/recommend", json=self._payload()).json()["rankings"]
        assert {r["crop"] for r in rankings} == {"Maize", "Rice", "Soybean", "Wheat"}

    def test_stress_fields_non_negative(self, test_client):
        rankings = test_client.post("/recommend", json=self._payload()).json()["rankings"]
        for r in rankings:
            assert r["water_stress"] >= 0
            assert r["heat_stress"] >= 0

    def test_missing_conditions_returns_422(self, test_client):
        resp = test_client.post("/recommend", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /optimize
# ---------------------------------------------------------------------------

class TestOptimize:
    def _payload(self, crop="Maize", **cond_overrides):
        return {
            "conditions": {**VALID_CONDITIONS, **cond_overrides},
            "crop": crop,
        }

    def test_status_200(self, test_client):
        resp = test_client.post("/optimize", json=self._payload())
        assert resp.status_code == 200

    def test_response_crop_matches_request(self, test_client):
        data = test_client.post("/optimize", json=self._payload("Wheat")).json()
        assert data["crop"] == "Wheat"

    def test_best_yield_gte_baseline(self, test_client):
        data = test_client.post("/optimize", json=self._payload()).json()
        assert data["best_yield_hg_ha"] >= data["baseline_yield_hg_ha"]

    def test_yield_gain_non_negative(self, test_client):
        data = test_client.post("/optimize", json=self._payload()).json()
        assert data["yield_gain_hg_ha"] >= 0
        assert data["yield_gain_pct"] >= 0

    def test_best_conditions_fields_present(self, test_client):
        data = test_client.post("/optimize", json=self._payload()).json()
        bc = data["best_conditions"]
        assert "fertilizer_used" in bc
        assert "irrigation_used" in bc
        assert "days_to_harvest" in bc

    @pytest.mark.parametrize("crop", ["Maize", "Rice", "Soybean", "Wheat"])
    def test_all_crops_return_200(self, test_client, crop):
        resp = test_client.post("/optimize", json=self._payload(crop))
        assert resp.status_code == 200

    def test_invalid_crop_returns_422(self, test_client):
        resp = test_client.post("/optimize", json=self._payload("Barley"))
        assert resp.status_code == 422
