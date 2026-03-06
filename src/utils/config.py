"""
Configuration management for AgroOpt.

Loads config.yaml and exposes a typed Settings object.
Environment variables override YAML values where applicable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger
from pydantic import field_validator
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Project root — two levels up from this file (src/utils/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Typed settings (Pydantic v2)
# ---------------------------------------------------------------------------


class PathsConfig(BaseSettings):
    """File and directory paths."""

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    external_dir: Path = PROJECT_ROOT / "data" / "external"

    dataset_1: Path = PROJECT_ROOT / "data" / "raw" / "crop_yield.csv"
    dataset_2_yield: Path = PROJECT_ROOT / "data" / "raw" / "yield.csv"
    dataset_2_pesticides: Path = PROJECT_ROOT / "data" / "raw" / "pesticides.csv"
    dataset_2_rainfall: Path = PROJECT_ROOT / "data" / "raw" / "rainfall.csv"
    dataset_2_temp: Path = PROJECT_ROOT / "data" / "raw" / "temp.csv"

    merged: Path = PROJECT_ROOT / "data" / "processed" / "merged_dataset.csv"
    features: Path = PROJECT_ROOT / "data" / "processed" / "features_dataset.csv"

    models_dir: Path = PROJECT_ROOT / "models"
    best_model: Path = PROJECT_ROOT / "models" / "best_model.pkl"
    feature_names: Path = PROJECT_ROOT / "models" / "feature_names.json"
    scaler: Path = PROJECT_ROOT / "models" / "scaler.pkl"
    label_encoder: Path = PROJECT_ROOT / "models" / "label_encoder.pkl"

    mlflow_tracking_uri: str = str(PROJECT_ROOT / "mlflow" / "mlruns")
    mlflow_experiment_name: str = "agro-opt-yield-prediction"

    model_config = {"arbitrary_types_allowed": True}

    def ensure_dirs(self) -> None:
        """Create all required directories if they do not exist."""
        for attr in [
            "raw_dir",
            "processed_dir",
            "external_dir",
            "models_dir",
        ]:
            getattr(self, attr).mkdir(parents=True, exist_ok=True)
        logger.info("Directory structure verified.")


class DataConfig(BaseSettings):
    target_column: str = "yield_hg_ha"
    crop_column: str = "crop"
    year_column: str = "year"
    drop_threshold: float = 0.8

    model_config = {"arbitrary_types_allowed": True}


class TrainingConfig(BaseSettings):
    test_size: float = 0.2
    val_size: float = 0.1
    cv_folds: int = 5
    random_seed: int = 42

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("test_size", "val_size")
    @classmethod
    def must_be_fraction(cls, v: float) -> float:
        assert 0 < v < 1, "Must be between 0 and 1"
        return v


class APIConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"

    # Override via env vars: API_HOST, API_PORT …
    model_config = {"env_prefix": "API_", "arbitrary_types_allowed": True}


class StreamlitConfig(BaseSettings):
    api_base_url: str = "http://localhost:8000"
    page_title: str = "AgroOpt — Crop Yield & Recommendation"
    page_icon: str = "🌾"

    model_config = {"arbitrary_types_allowed": True}


class Settings:
    """
    Top-level settings object assembled from config.yaml.

    Usage
    -----
    >>> from src.utils.config import settings
    >>> settings.paths.raw_dir
    PosixPath('/…/data/raw')
    """

    def __init__(self) -> None:
        raw = _load_yaml(CONFIG_PATH)
        self.project_name: str = raw.get("project", {}).get("name", "agro-opt")
        self.random_seed: int = raw.get("project", {}).get("random_seed", 42)

        self.paths = PathsConfig()
        self.data = DataConfig()
        self.training = TrainingConfig(random_seed=self.random_seed)
        self.api = APIConfig()
        self.streamlit = StreamlitConfig()

        # Honour YAML overrides for API URL (useful in Docker)
        streamlit_url = (
            raw.get("streamlit", {}).get("api_base_url")
            or os.environ.get("API_BASE_URL")
        )
        if streamlit_url:
            self.streamlit = StreamlitConfig(api_base_url=streamlit_url)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Settings project={self.project_name}>"


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()
