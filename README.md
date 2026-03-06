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
# 1. Merge raw datasets
poetry run python -m src.data.merge_datasets

# 2. Feature engineering
poetry run python -m src.features.feature_engineering

# 3. Train models (logged in MLflow)
poetry run python -m src.models.train_model

# 4. Start the API
poetry run uvicorn src.api.main:app --reload --port 8000

# 5. Start Streamlit (new terminal)
poetry run streamlit run app/streamlit_app.py
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
├── config.yaml                   # Central configuration
├── pyproject.toml                # Poetry dependencies
├── requirements.txt              # pip-compatible deps
│
├── data/
│   ├── raw/                      # Original datasets (tracked by git)
│   └── processed/                # Generated artefacts (git-ignored)
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py          # Dataset loaders
│   │   ├── merge_datasets.py     # Dataset integration
│   │   └── preprocess.py        # Cleaning & imputation
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train_model.py        # MLflow-tracked training
│   │   ├── predict.py            # Inference helpers
│   │   └── recommend.py         # Simulation-based recommender
│   ├── evaluation/
│   │   └── evaluate_model.py
│   ├── api/
│   │   ├── main.py               # FastAPI application
│   │   ├── routes.py             # Endpoints
│   │   └── schemas.py           # Pydantic request/response models
│   └── utils/
│       ├── config.py             # Settings singleton
│       └── logging.py           # Loguru setup
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
│
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.streamlit
│
├── .github/workflows/ci_cd.yml
└── mlflow/
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
