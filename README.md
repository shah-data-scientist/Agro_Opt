# AgroOpt — Agricultural Crop Yield Prediction & Recommendation System

> A production-grade machine learning system that predicts crop yields and recommends
> the best crop for given environmental conditions.

---

## System Architecture

```
┌──────────────┐      HTTP      ┌──────────────────┐
│  Streamlit   │ ◄────────────► │  FastAPI Backend  │
│  Frontend    │                │  /predict         │
│  (Port 8501) │                │  /recommend       │
└──────────────┘                └────────┬─────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   ML Models          │
                              │   (joblib artifacts) │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   MLflow Tracking    │
                              │   (experiment logs)  │
                              └─────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker & Docker Compose (optional)

### Install with Poetry

```bash
# Clone
git clone <repo-url>
cd agro-opt

# Install all dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Run the full pipeline

```bash
# Run end-to-end (merge → features → train)
poetry run python scripts/run_pipeline.py

# Or run individual stages:
poetry run python -m src.data.merge_datasets
poetry run python -m src.features.feature_engineering
poetry run python -m src.models.train

# Start the API
poetry run uvicorn src.api.main:app --reload --port 8000

# Start Streamlit (new terminal)
poetry run streamlit run src/frontend/app.py
```

### Docker

```bash
docker compose up --build
```

### MLflow UI

```bash
poetry run mlflow ui --backend-store-uri mlflow/mlruns
# Open http://localhost:5000
```

---

## Repository Structure

```
agro-opt/
├── config.yaml                   # Central configuration (all pipeline settings)
├── pyproject.toml                # Poetry dependencies
├── docker-compose.yml            # Local dev/demo orchestration
├── Dockerfile.api                # FastAPI container
├── Dockerfile.frontend           # Streamlit container
│
├── scripts/
│   └── run_pipeline.py           # End-to-end MLOps orchestration
│
├── src/
│   ├── api/
│   │   ├── app.py                # FastAPI routes (/health, /predict, /recommend, /optimize)
│   │   ├── main.py               # Uvicorn entrypoint
│   │   └── schemas.py            # Pydantic request/response models
│   ├── data/
│   │   ├── load_data.py          # Typed loaders for all raw datasets
│   │   ├── merge_datasets.py     # FAO + synthetic data integration
│   │   └── preprocess.py        # Cleaning & validation
│   ├── features/
│   │   ├── feature_engineering.py  # 36-feature derivation
│   │   └── pca_analysis.py      # Dimensionality reduction
│   ├── frontend/
│   │   └── app.py               # Streamlit UI (3 tabs: Predict / Recommend / Optimize)
│   ├── models/
│   │   ├── train.py             # 5-model training with MLflow logging
│   │   └── evaluate.py          # Metrics & plots
│   ├── recommendation/
│   │   └── engine.py            # Yield prediction + grid-search optimisation
│   └── utils/
│       ├── config.py             # Settings singleton (config.yaml → Pydantic)
│       └── logging.py           # Loguru setup
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_pca_analysis.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_recommendation_engine.ipynb
│
├── tests/
│   ├── conftest.py               # pytest fixtures
│   ├── test_api.py
│   ├── test_engine.py
│   └── test_schemas.py
│
├── Data/
│   ├── raw/                      # Source CSVs (crop_yield, yield, rainfall, temp, pesticides)
│   └── processed/                # Merged & feature datasets (git-ignored)
│
├── models/                       # Trained artefacts (best_model.pkl, feature_names.json)
├── reports/figures/              # All output plots (EDA, features, PCA, model evaluation)
├── deliverables/                 # Business report, schema diagram, presentation script
└── .github/workflows/            # CI (lint + test + build) and Docker publish
```

---

## Datasets

| Dataset | Source | Key Variables |
|---|---|---|
| Agriculture Crop Yield | `data/raw/crop_yield.csv` | crop, country, year, yield |
| FAO Yield | `data/raw/yield.csv` | Area, Item, Year, Value (hg/ha) |
| Pesticides | `data/raw/pesticides.csv` | Area, Year, Value (tonnes) |
| Rainfall | `data/raw/rainfall.csv` | country_name, year, avg_precipitation_mm |
| Temperature | `data/raw/temp.csv` | country, year, avg_temp |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/predict` | Predict yield for a given crop + conditions |
| POST | `/recommend` | Recommend best crop(s) for given conditions |

---

## ML Models Evaluated

- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

Best model selected by cross-validated RMSE.

---

## MLOps

All training runs are tracked with **MLflow**:
- Parameters logged per run
- Metrics: RMSE, MAE, R²
- Model artifacts saved and registered

---

## Testing

```bash
poetry run pytest tests/ -v --cov=src
```
