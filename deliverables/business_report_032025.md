# AgroOpt — Business Summary Report

**Project:** AgroOpt — Agricultural Yield Prediction & Optimization Platform
**Author:** Shah Shahul
**Date:** March 2025
**Programme:** OpenClassrooms — Machine Learning Engineer (Bac+5)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Business Context & Problem Statement](#2-business-context--problem-statement)
3. [Solution Overview](#3-solution-overview)
4. [Data & Methodology](#4-data--methodology)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Training & Experiment Tracking](#6-model-training--experiment-tracking)
7. [API & Deployment Architecture](#7-api--deployment-architecture)
8. [CI/CD Pipeline](#8-cicd-pipeline)
9. [Results & Business Value](#9-results--business-value)
10. [Recommendations & Next Steps](#10-recommendations--next-steps)

---

## 1. Executive Summary

AgroOpt is an end-to-end MLOps platform that predicts crop yields, recommends optimal crops for given farm conditions, and suggests management improvements to maximise agricultural output. The platform is trained on 666,494 labelled farm records covering four major US crops (Maize, Rice, Soybean, Wheat) anchored to FAO 2013 reference statistics.

The best model — a regularised Ridge regression pipeline — achieves **R² = 0.913** and **RMSE = 4,989 hg/ha** (~0.5 t/ha) on a held-out test set of 133,299 samples. The entire platform is containerised with Docker, continuously tested via GitHub Actions, and served through a FastAPI REST API with a Streamlit frontend.

**Key outcomes:**

| Metric | Value |
|--------|-------|
| Training rows | 533,195 |
| Test rows | 133,299 |
| Best model R² | 0.913 |
| Best model RMSE | 4,989 hg/ha (~0.5 t/ha) |
| API endpoints | 4 (/health, /predict, /recommend, /optimize) |
| Automated tests | 106 passing |
| CI/CD | GitHub Actions (lint + test + docker-build + docker-publish) |

---

## 2. Business Context & Problem Statement

### Context

Agriculture accounts for roughly 70% of global freshwater use and feeds over 8 billion people. In the United States alone, crop production represents a $200 billion industry where small improvements in yield prediction and management decisions translate directly into economic gains and reduced environmental impact.

Farmers and agronomists face three recurring decisions:

1. **What to plant?** — Which crop will perform best given local soil, climate, and historical rainfall?
2. **How much will I yield?** — What is a realistic yield forecast given current conditions?
3. **What can I change?** — Which management levers (fertilizer, irrigation, harvest timing) will most improve the outcome?

These decisions are today made largely from experience or simple lookup tables. Machine learning offers the opportunity to systematise and personalise these predictions at scale.

### Problem Statement

Build a production-grade ML system that:
- Ingests farm conditions (rainfall, temperature, soil type, region, management practices)
- Predicts yield in hg/ha for any of the four main US crops
- Recommends the best-suited crop for the given conditions
- Optimises discrete management variables (fertilizer, irrigation, days to harvest) to maximise predicted yield
- Exposes all capabilities through a REST API with a visual frontend
- Is reproducible, tested, version-controlled, and continuously deployed

---

## 3. Solution Overview

AgroOpt is structured as a four-layer platform:

```
Raw Data (CSV)
     |
     v
[Stage 1] Data Preprocessing  --> data/processed/merged_dataset.csv
     |
     v
[Stage 2] Feature Engineering --> data/processed/features_dataset.csv
     |
     v
[Stage 3] Model Training       --> models/best_model.pkl  +  mlflow/mlruns/
     |
     v
[Stage 4] Serving (API + UI)   --> FastAPI :8000  +  Streamlit :8501
```

The full pipeline is orchestrated by `run_pipeline.py` and can be resumed from any stage using `--from-stage N`.

---

## 4. Data & Methodology

### Data Sources

| Source | Description | Rows | Key Fields |
|--------|-------------|------|------------|
| `crop_yield.csv` | Synthetic US farm records | ~666K | crop, region, soil_type, rainfall, temp, fertilizer, irrigation, harvest_days, yield |
| FAO `yield.csv` | Historical USA crop yield (hg/ha) | — | Reference benchmark per crop |
| FAO `rainfall.csv` | USA annual rainfall (mm) | — | National average: 715 mm |
| FAO `temp.csv` | USA average temperature (52 states aggregated) | — | National average: 16.44°C |
| FAO `pesticides.csv` | USA pesticide use (tonnes) | — | Excluded (zero variance after merge) |

### Crop Scope

Four crops are in scope, selected based on FAO data availability for the USA:

| Crop | FAO Benchmark Yield (hg/ha) |
|------|-----------------------------|
| Maize | 99,256 |
| Rice | 86,232 |
| Wheat | 31,673 |
| Soybean | 29,615 |

Barley and Cotton were excluded — no matching FAO USA counterparts, which would have introduced systematic NaN values.

### Data Quality

After merging and validation:

- **666,494 rows × 15 columns**
- **0 null values, 0 duplicates**
- FAO rainfall for 2003 was imputed with the median (year absent from rainfall.csv)
- Target variable: `yield_hg_ha` (hectogram per hectare)

---

## 5. Feature Engineering

36 features are derived from the 15 merged columns:

### Domain Features

| Feature | Formula / Description |
|---------|----------------------|
| `water_stress` | `max(0, 1 - rainfall_mm / fao_rainfall_mm)` |
| `heat_stress` | `max(0, temperature_celsius - 35) / 15` |
| `gdd_proxy` | `max(0, temperature_celsius - 10) * days_to_harvest` |
| `aridity_index` | `temperature_celsius / (rainfall_mm + 1)` |
| `harvest_rainfall_rate` | `rainfall_mm / days_to_harvest` |
| `soil_quality_score` | Ordinal score (Loam=5, Silt=4, Clay=3, Peaty=2, Sandy=1, Chalky=0) |
| `rainfall_anomaly` | `rainfall_mm - fao_rainfall_mm` |
| `temp_anomaly` | `temperature_celsius - fao_avg_temp` |

### Interaction Terms

`rainfall_x_fertilizer`, `rainfall_x_irrigation`, `temp_x_irrigation`, `gdd_x_irrigation`, `fao_yield_x_rainfall`, `fao_yield_x_temp`

### One-Hot Encoding

- `crop` → 4 binary columns (Maize, Rice, Soybean, Wheat)
- `region` → 4 binary columns (East, North, South, West)
- `soil_type` → 6 binary columns
- `weather_condition` → 3 binary columns (Cloudy, Rainy, Sunny)

### Top Features by Correlation with Yield

| Feature | |Pearson r| |
|---------|-------------|
| `rainfall_mm` | 0.764 |
| `rainfall_anomaly` | 0.764 |
| `rainfall_x_fertilizer` | 0.668 |
| `harvest_rainfall_rate` | 0.647 |
| `rainfall_x_irrigation` | 0.594 |

Rainfall is the dominant driver — consistent with US agricultural literature.

### Dimensionality Analysis (PCA)

A PCA on the 36-feature space shows:
- Effective rank: 27 (9 zero-eigenvalue components from collinear pairs and one-hot group constraints)
- 90% of variance explained by 17 principal components
- Tree-based models use all 36 features; linear baseline uses StandardScaler + Ridge

---

## 6. Model Training & Experiment Tracking

### Approach

Five regression models were trained with 5-fold cross-validation on 533,195 samples. All runs are logged to MLflow (local tracking server at `mlflow/mlruns/`).

### Results

| Rank | Model | R² | RMSE (hg/ha) | MAE (hg/ha) | MAPE (%) | Train Time |
|------|-------|----|--------------|--------------|---------:|-----------:|
| 1 | **Ridge** | **0.9130** | **4,989** | **3,976** | 12.06 | 1.4s |
| 2 | Hist Gradient Boosting | 0.9128 | 4,995 | 3,981 | 12.15 | 17.5s |
| 3 | LightGBM | 0.9128 | 4,996 | 3,981 | 12.13 | 7.6s |
| 4 | XGBoost | 0.9127 | 4,998 | 3,983 | 12.14 | 12.1s |
| 5 | Random Forest | 0.9099 | 5,080 | 4,048 | 12.30 | 231s |

### Key Finding

All five models converge to R² ≈ 0.91, confirming that the synthetic dataset has predominantly **linear relationships**. Ridge wins because:
- The target is largely a linear combination of rainfall and management flags
- No complex non-linear interactions exist in the synthetic data
- Ridge avoids the variance overhead of tree ensembles on a linear problem

The deployed model is the **Ridge + StandardScaler pipeline** (`models/best_model.pkl`).

### MLflow Tracking

Each run logs: model name, hyperparameters, R², RMSE, MAE, MAPE, CV-RMSE mean/std, and training time. The experiment is stored under `mlflow/mlruns/` and can be explored at `http://localhost:5000` after running `mlflow ui`.

---

## 7. API & Deployment Architecture

### FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model status, version, feature count |
| POST | `/predict` | Yield prediction for a specific crop |
| POST | `/recommend` | Ranked list of all 4 crops for given conditions |
| POST | `/optimize` | Best management settings for a target crop |

### Request Example — `/predict`

```json
{
  "conditions": {
    "rainfall_mm": 650,
    "temperature_celsius": 22,
    "days_to_harvest": 120,
    "region": "East",
    "soil_type": "Loam",
    "weather_condition": "Sunny",
    "fertilizer_used": true,
    "irrigation_used": true
  },
  "crop": "Maize"
}
```

### Response Example — `/predict`

```json
{
  "crop": "Maize",
  "predicted_yield_hg_ha": 98412,
  "predicted_yield_t_ha": 9.84
}
```

### Docker Architecture

```
docker-compose.yml
  ├── api      (Dockerfile.api)      → python:3.11-slim, port 8000
  │            models/ mounted as read-only volume
  └── frontend (Dockerfile.frontend) → python:3.11-slim, port 8501
               API_BASE_URL=http://api:8000
```

Both services run as non-root `appuser`, with restart policies and health checks. The API image does not include xgboost/lightgbm — only scikit-learn is needed for the deployed Ridge model, keeping the image lean.

---

## 8. CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

Triggered on every push and pull request to `main`:

| Job | Tool | What it checks |
|-----|------|----------------|
| `lint` | flake8 | PEP8, unused imports, undefined names |
| `test` | pytest + coverage | 106 tests across schemas, engine, API |
| `docker-build` | docker/build-push-action | Both images build without errors |

Tests run entirely in-memory (no pkl files required): a minimal Ridge pipeline is trained on synthetic data as a pytest fixture, and `load_assets` is patched via `unittest.mock`.

### Continuous Deployment (`.github/workflows/docker-publish.yml`)

Triggered on push to `main` or version tags (`v*.*.*`):

- Authenticates to `ghcr.io` via `GITHUB_TOKEN`
- Builds and pushes `agro-opt-api` and `agro-opt-frontend` with BuildKit layer caching (`type=gha`)

---

## 9. Results & Business Value

### Model Performance in Business Terms

An RMSE of 4,989 hg/ha corresponds to **~0.5 t/ha**. In context:

- Maize average yield: ~9.9 t/ha → prediction error ≈ 5%
- Wheat average yield: ~3.2 t/ha → prediction error ≈ 16%
- Soybean average yield: ~3.0 t/ha → prediction error ≈ 17%

The model is most accurate for high-yield crops (Maize, Rice) where absolute errors are smallest relative to yield level.

### Actionable Outputs

The `/recommend` endpoint lets a farmer enter their current conditions and immediately see which crop is projected to yield highest — removing guesswork from planting decisions.

The `/optimize` endpoint performs a grid search over:
- Fertilizer use (yes/no)
- Irrigation use (yes/no)
- Days to harvest (90, 100, 110, 120, 130, 140, 150, 160)

It returns the configuration that maximises predicted yield alongside the estimated yield gain over the current baseline — a direct input for farm management planning.

### MLOps Value

| Capability | Benefit |
|------------|---------|
| MLflow tracking | Full reproducibility — every model run is versioned with parameters and metrics |
| Docker containerisation | Consistent environment from dev to production; one-command deployment |
| CI/CD pipelines | Every code change is automatically linted, tested, and built |
| REST API | Model predictions accessible to any downstream system (dashboards, mobile apps, ERP) |
| 106 automated tests | Confidence in correctness of data validation, feature engineering, and API behaviour |

---

## 10. Recommendations & Next Steps

### Short-term

1. **Real farm data integration** — The synthetic dataset enforces linear relationships. Replacing it with real USDA/NASS historical records would likely improve tree model performance over Ridge and capture non-linear soil-climate interactions.

2. **Pesticide feature** — Currently excluded due to zero variance in the synthetic data. With real data, pesticide usage is a meaningful predictor and should be reintroduced.

3. **Confidence intervals** — Replace point predictions with prediction intervals (e.g., quantile regression or conformal prediction) to give farmers a risk range alongside the central estimate.

### Medium-term

4. **Regional expansion** — Extend beyond East/North/South/West to county-level granularity using USDA climate zones.

5. **Time-series forecasting** — Incorporate seasonal weather forecasts (e.g., NOAA 90-day outlooks) as input features to make predictions forward-looking rather than condition-conditioned.

6. **Model retraining pipeline** — Automate periodic retraining as new crop season data becomes available, with automated drift detection and model promotion gating.

### Long-term

7. **Satellite imagery features** — Integrate NDVI (Normalized Difference Vegetation Index) from Sentinel-2 or Landsat as additional predictors for soil health and crop stress.

8. **Economic optimisation** — Extend the `/optimize` endpoint to account for commodity prices and input costs, shifting the objective from maximum yield to maximum profit.

---

## Appendix — Project Structure

```
Agro_Opt/
├── src/
│   ├── data/               # Data loading, preprocessing, merging
│   ├── features/           # Feature engineering (36 features)
│   ├── models/             # Training, evaluation, prediction
│   ├── recommendation/     # FarmConditions engine (predict/recommend/optimize)
│   ├── api/                # FastAPI app, schemas, endpoints
│   └── frontend/           # Streamlit dashboard
├── models/
│   ├── best_model.pkl       # Deployed Ridge pipeline
│   ├── feature_names.json   # 36 ordered feature names
│   └── fao_refs.json        # FAO 2013 USA reference constants
├── tests/
│   ├── conftest.py          # Session-scoped fixtures (in-memory model)
│   ├── test_schemas.py      # 30 Pydantic validation tests
│   ├── test_engine.py       # 40 engine unit tests
│   └── test_api.py          # 36 API integration tests
├── .github/workflows/
│   ├── ci.yml               # Lint + test + docker-build
│   └── docker-publish.yml   # Push to ghcr.io on main/tags
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── run_pipeline.py          # End-to-end 4-stage MLOps pipeline
└── deliverables/
    ├── Shah_Shahul_1_Screenshot_032025.png   # MLflow experiment dashboard
    └── Shah_Shahul_5_Report_032025.md        # This document
```

---

*Generated as part of the AgroOpt MLOps project — OpenClassrooms Machine Learning Engineer programme.*
