# AgroOpt — Project Action Log

Complete record of all phases, decisions, and changes made throughout the project.

---

## Phase 0 — Project Setup

**Commit:** `dcf7cdd` (Initial commit)

- Initialised Poetry project with Python 3.11.9 via pyenv
- Created virtual environment at `.venv/`
- Defined project structure: `src/`, `data/`, `models/`, `notebooks/`, `tests/`, `mlflow/`, `logs/`
- Added `config.yaml` (central configuration) and `src/utils/config.py` (Pydantic settings singleton)
- Added `src/utils/logging.py` (loguru-based logging with file rotation)
- Registered Jupyter kernel as `agro-opt` for notebook execution
- Created `.gitignore` excluding large data files, `.venv/`, `logs/`, `mlflow/`, model `.pkl` files

---

## Phase 1 — Exploratory Data Analysis

**Commit:** `dcf7cdd` (Initial commit)

**Notebook:** `notebooks/01_eda.ipynb`

- Loaded all raw datasets from `data/raw/`:
  - `crop_yield.csv` — US synthetic farm records (6 crops, 4 regions)
  - `yield.csv`, `rainfall.csv`, `temp.csv`, `pesticides.csv` — FAO reference data
- Identified USA naming inconsistency: `"United States of America"` (yield/pesticides) vs `"United States"` (rainfall/temp)
- Discovered `temp.csv` has 52 state-level entries per year → must aggregate by mean
- Found missing 2003 rainfall entry → imputed with median
- Decided to **exclude Barley and Cotton** — no matching FAO USA counterparts (would introduce NaN)
- **4 crops in scope:** Maize, Rice, Soybean, Wheat

---

## Phase 2 — Data Integration & Merging

**Commit:** `dcf7cdd` (Initial commit)

**Key files:** `src/data/load_data.py`, `src/data/preprocess.py`, `src/data/merge_datasets.py`

- Implemented `merge_datasets()` — left-joins synthetic crop records with FAO 2013 USA reference data
- Merged on `(crop, year)` using a crop name mapping:
  - Maize → Maize, Rice → Rice paddy, Soybean → Soybeans, Wheat → Wheat
- **Final merged dataset:** `data/processed/merged_dataset.csv`
  - 666,494 rows × 15 columns
  - 0 null values, 0 duplicates
  - Target variable: `yield_hg_ha` (hectogram per hectare)
- Generated sidecar `data/processed/merged_dataset.json` with metadata

---

## Phase 3 — Feature Engineering

**Commit:** `dcf7cdd` (Initial commit)

**Notebook:** `notebooks/02_feature_engineering.ipynb`
**Key file:** `src/features/feature_engineering.py`

- Derived **36 features** from the 15 merged columns
- **Domain physics features:**
  - `water_stress = max(0, 1 - rainfall_mm / fao_rainfall_mm)`
  - `heat_stress = max(0, temperature_celsius - 35) / 15`
  - `gdd_proxy = max(0, temperature_celsius - 10) * days_to_harvest`
  - `aridity_index = temperature_celsius / (rainfall_mm + 1)`
  - `harvest_rainfall_rate = rainfall_mm / days_to_harvest`
  - `soil_quality_score` — ordinal encoding (Loam=5 … Chalky=0)
  - `rainfall_anomaly`, `temp_anomaly` — deviation from FAO national averages
- **Interaction terms:** `rainfall_x_fertilizer`, `rainfall_x_irrigation`, `temp_x_irrigation`, etc.
- **One-hot encoding:** crop (4), region (4), soil_type (6), weather_condition (3) = 17 binary columns
- Dropped zero-variance columns: `year`, `fao_pesticides_tonnes`
- **Top features by |Pearson r|:** `rainfall_mm` (0.764), `rainfall_anomaly` (0.764), `rainfall_x_fertilizer` (0.668)
- Output: `data/processed/features_dataset.csv` — 666,494 rows × 37 cols (36 features + target)

---

## Phase 4 — PCA Analysis

**Commit:** `dcf7cdd` (Initial commit)

**Notebook:** `notebooks/03_pca_analysis.ipynb`

- Applied PCA to the 36-feature matrix
- **Results:** effective rank = 27 (9 zero-eigenvalue components)
  - 9 zeros from: 2 perfect collinear pairs (rainfall + anomaly, temp + anomaly) + 4 one-hot group constraints + 3 others
  - 90% variance: 17 PCs | 95%: 18 PCs | 99%: 21 PCs
- **Modelling decision:** use all 36 original features (no PCA reduction)
  - Tree models are immune to collinearity
  - Ridge handles collinearity via L2 regularisation
- Output: `data/processed/pca_results.json`

---

## Phase 5 — Model Training & Experiment Tracking

**Commit:** `54fa777`

**Notebook:** `notebooks/04_model_training.ipynb`
**Key files:** `src/models/train.py`, `src/models/evaluate.py`

- Trained 5 regression models with 5-fold cross-validation on 533,195 training samples
- All runs logged to MLflow at `mlflow/mlruns/`

| Model | R² | RMSE (hg/ha) | Train time |
|-------|-----|--------------|------------|
| **Ridge** | **0.9130** | **4,989** | 1.4s |
| Hist GB | 0.9128 | 4,995 | 17.5s |
| LightGBM | 0.9128 | 4,996 | 7.6s |
| XGBoost | 0.9127 | 4,998 | 12.1s |
| Random Forest | 0.9099 | 5,080 | 231s |

- **Winner: Ridge** — all 5 models cluster at R² ≈ 0.91, confirming predominantly linear synthetic data
- Artefacts saved: `models/best_model.pkl`, `models/scaler.pkl`, `models/feature_names.json`, `models/model_results.json`
- Metrics tracked: R², RMSE, MAE, MAPE, CV-RMSE mean/std, train time

---

## Phase 6 — Recommendation Engine

**Commit:** `230e8ad`

**Key file:** `src/recommendation/engine.py`

- Implemented `FarmConditions` dataclass with validation (`validate()`)
- Implemented `build_feature_vector()` — constructs a 36-feature numpy array from farm conditions + crop
- Implemented `predict_yield(conditions, crop, assets) -> float`
- Implemented `recommend_crop(conditions, assets) -> list[dict]` — ranks all 4 crops by predicted yield with water/heat stress indices and FAO benchmark
- Implemented `optimize_conditions(conditions, crop, assets) -> dict` — grid search over:
  - `fertilizer_used` (True/False)
  - `irrigation_used` (True/False)
  - `days_to_harvest` (90, 100, 110, 120, 130, 140, 150, 160)
- FAO reference constants initially loaded from `data/processed/merged_dataset.csv`

---

## Phase 7 — FastAPI Backend

**Commit:** `6fb7fc7`

**Key files:** `src/api/app.py`, `src/api/schemas.py`

- Built FastAPI application with lifespan handler to load model assets at startup
- Implemented 4 endpoints:
  - `GET /health` — model status, version, feature count
  - `POST /predict` — yield prediction for a specific crop
  - `POST /recommend` — ranked crop list with stress indices
  - `POST /optimize` — best management settings to maximise yield
- Pydantic schemas with field constraints:
  - `rainfall_mm`: 0–5000 mm
  - `temperature_celsius`: −10 to 50°C
  - `days_to_harvest`: 1–365
  - `region`: Literal["East","North","South","West"]
  - `soil_type`: 6 valid types
  - `weather_condition`: Literal["Cloudy","Rainy","Sunny"]
  - `crop`: Literal["Maize","Rice","Soybean","Wheat"]
- Notebook: `notebooks/05_recommendation_engine.ipynb`

---

## Phase 8 — Streamlit Frontend

**Commit:** `9831df9` + `af6962d` (fix: session_state key= on widgets)

**Key file:** `src/frontend/app.py`

- Built 3-tab Streamlit dashboard:
  - **Predict** — yield forecast with gauge chart (Plotly)
  - **Recommend** — ranked crop table with bar chart
  - **Optimize** — baseline vs optimised yield comparison
- Sidebar for farm conditions input (all 8 fields)
- API base URL configurable via `API_BASE_URL` env var or sidebar
- Fixed session_state widget update bug by adding `key=` to all sidebar inputs

---

## Phase 9 — Dockerisation

**Commit:** `3b002f7` + `7e15f84` (fix: FAO refs JSON)

**Key files:** `Dockerfile.api`, `Dockerfile.frontend`, `docker-compose.yml`, `requirements-api.txt`, `requirements-frontend.txt`, `.dockerignore`

- Created two Docker images based on `python:3.11-slim`
- **`Dockerfile.api`:** copies `src/`, `config.yaml`, `models/`; runs uvicorn on port 8000
- **`Dockerfile.frontend`:** copies `src/frontend/`; runs Streamlit on port 8501
- **`docker-compose.yml`:** two-service setup with healthcheck, `depends_on`, `restart: unless-stopped`, non-root `appuser`
- Removed `xgboost` and `lightgbm` from `requirements-api.txt` (deployed model is Ridge only — avoids 293MB CUDA download)
- **Fix:** `engine._load_fao_refs()` was reading the 666K-row CSV (excluded by `.dockerignore`)
  - Extracted 4 FAO constants to `models/fao_refs.json`
  - Updated `_load_fao_refs()` to read JSON first, fall back to CSV
  - Added `!/models/fao_refs.json` to `.gitignore` to allow committing it

---

## Phase 10 — CI/CD Pipeline

**Commit:** `22a71b8` + `3f16cbd` (fix: pytest exit-5) + `07c826d` (fix: flake8)

**Key files:** `.github/workflows/ci.yml`, `.github/workflows/docker-publish.yml`, `.flake8`

### CI Workflow (`.github/workflows/ci.yml`)
- Triggers on push and pull requests to `main`
- **Job 1 — lint:** flake8 with `extend-ignore = E203, W503, E501, E127, E128, E221, E272`
- **Job 2 — test:** pytest with coverage, handles exit code 5 (no tests collected) gracefully
- **Job 3 — docker-build:** builds both images without pushing

### CD Workflow (`.github/workflows/docker-publish.yml`)
- Triggers on push to `main` and version tags (`v*.*.*`)
- Authenticates to `ghcr.io` via `GITHUB_TOKEN`
- Builds and pushes `agro-opt-api` and `agro-opt-frontend` with BuildKit `type=gha` cache

### Fixes applied
- `src/api/schemas.py` — removed unused `field_validator` import (F401)
- `src/features/feature_engineering.py` — removed unused `numpy as np` import (F401)
- `src/frontend/app.py` — removed unused `plotly.express` import; added `os.environ` for `API_BASE_URL`
- `src/recommendation/engine.py` — removed unused `field`, `Path` imports
- `src/data/merge_datasets.py` — removed f-prefix from plain string (F541)
- `src/models/evaluate.py` — renamed `name` → `_name` in loop variable (B007)

---

## Phase 11 — Testing

**Commit:** `fb02803`

**Key files:** `tests/conftest.py`, `tests/test_schemas.py`, `tests/test_engine.py`, `tests/test_api.py`

- **106 tests** across 3 modules, all passing in ~6 seconds
- `conftest.py` — session-scoped fixtures:
  - `feature_cols` — loads `models/feature_names.json`
  - `fao_refs` — loads `models/fao_refs.json`
  - `assets` — builds minimal in-memory Ridge pipeline (no `.pkl` files needed)
  - `test_client` — FastAPI `TestClient` with `load_assets` patched via `unittest.mock`
- `test_schemas.py` — 30 tests: Pydantic field bounds, Literal enums, boundary values
- `test_engine.py` — 40 tests: `FarmConditions.validate()`, `build_feature_vector()`, `predict_yield()`, `recommend_crop()`, `optimize_conditions()`
- `test_api.py` — 36 tests: all 4 endpoints, valid and invalid inputs, status codes

---

## Phase 12 — Deliverables

**Commit:** `efa25f3`

### Deliverable 1 — MLflow Screenshot
- `deliverables/mlflow_screenshot_032025.png` → renamed to `deliverables/mlflow_screenshot_032025.png`
- Programmatically generated with matplotlib (2575×1627 px) showing all 5 model runs ranked by R²

### Deliverable 3 — MLOps Pipeline Script
- `run_pipeline.py` — 4-stage orchestration script:
  - Stage 1: `merge_datasets()` → `data/processed/merged_dataset.csv`
  - Stage 2: `build_features()` → `data/processed/features_dataset.csv`
  - Stage 3: `run_training()` → `models/best_model.pkl` + MLflow runs
  - Stage 4: print ranked model comparison from `models/model_results.json`
- CLI args: `--from-stage N`, `--skip-merge`, `--skip-features`

### Deliverable 5 — Business Summary Report
- `deliverables/business_report_032025.md` — 10-section Markdown report covering problem, data, features, models, architecture, CI/CD, business value, roadmap

### Deliverable 5 — Presentation
- `deliverables/presentation_032025.pptx` — 13-slide professional deck
  - Agricultural green/gold brand palette
  - matplotlib charts embedded (model comparison, feature correlations, pipeline diagram, etc.)
  - MLflow screenshot embedded as slide 8
  - Built with `python-pptx` via `deliverables/build_pptx.py`

---

## Post-Phase 12 — Incremental Fixes & Improvements

### Ridge Coefficient Feature Importance (notebook 04)
**Commit:** `a52f598`

- Ridge has no `feature_importances_` — added `plot_ridge_coefficients()` to `src/models/evaluate.py`
- After `StandardScaler`, `|coef|` is the linear-model equivalent of MDI importance
- Updated notebook cell `mt-importance` to auto-detect model type and call the appropriate function
- Chart output injected directly into notebook (avoids 10-minute full retrain)
- **Top drivers:** `rainfall_mm` (+6520), `rainfall_anomaly` (+6520), `agro_intensity` (+4757), `fertilizer_used` (+4130)

### PPTX Repair Fix
- Replaced `slide.shapes.add_shape(1, ...)` with `MSO_AUTO_SHAPE_TYPE.RECTANGLE`
- Replaced `shape.line.fill.background()` with `shape.line.color.rgb` + `shape.line.width = Pt(0)`

### File Renaming
- All deliverable filenames depersonalised (removed `Shah_Shahul_` prefix)

---

## Project File Structure (final)

```
Agro_Opt/
├── src/
│   ├── data/               # load_data.py, preprocess.py, merge_datasets.py
│   ├── features/           # feature_engineering.py (36 features)
│   ├── models/             # train.py, evaluate.py (+ plot_ridge_coefficients)
│   ├── recommendation/     # engine.py (predict / recommend / optimize)
│   ├── api/                # app.py, schemas.py
│   └── frontend/           # app.py (Streamlit)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_pca_analysis.ipynb
│   ├── 04_model_training.ipynb  ← Ridge coefficient chart added
│   └── 05_recommendation_engine.ipynb
├── models/
│   ├── best_model.pkl       # Ridge + StandardScaler pipeline
│   ├── feature_names.json   # 36 ordered feature names
│   └── fao_refs.json        # FAO 2013 USA reference constants
├── tests/
│   ├── conftest.py          # Session fixtures, in-memory model
│   ├── test_schemas.py      # 30 Pydantic tests
│   ├── test_engine.py       # 40 engine unit tests
│   └── test_api.py          # 36 API integration tests
├── deliverables/
│   ├── mlflow_screenshot_032025.png
│   ├── business_report_032025.md
│   ├── presentation_032025.pptx
│   └── build_pptx.py
├── .github/workflows/
│   ├── ci.yml               # Lint + test + docker-build
│   └── docker-publish.yml   # Push to ghcr.io
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml
├── run_pipeline.py          # End-to-end 4-stage MLOps pipeline
├── config.yaml
└── pyproject.toml
```

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Ridge wins over tree models | All 5 models cluster at R² ≈ 0.91; synthetic data is linear; Ridge is fastest (1.4s) |
| No PCA reduction | Tree models handle collinearity natively; Ridge handles it via L2 |
| FAO 2013 as reference year | Most complete USA crop data available in all 4 FAO datasets |
| Barley & Cotton excluded | No FAO USA counterpart → would introduce systematic NaN |
| `models/fao_refs.json` | Docker containers cannot access the 666K-row CSV; 4 constants extracted |
| In-memory test fixtures | CI has no `.pkl` artefacts; minimal Ridge pipeline trained from random data |
| `|coef|` for Ridge importance | After `StandardScaler`, coefficients are on the same scale → valid importance proxy |
