"""
Data loading utilities for AgroOpt.

Provides typed loaders for every raw dataset and a convenience
function that loads all datasets at once.

Dataset inventory
-----------------
crop_yield   : 1 000 000 rows — synthetic agronomic dataset
               (Region, Soil_Type, Crop, Rainfall_mm, Temperature_Celsius,
                Fertilizer_Used, Irrigation_Used, Weather_Condition,
                Days_to_Harvest, Yield_tons_per_hectare)

yield_fao    : 56 717 rows — FAO yield values in hg/ha per country/crop/year
pesticides   :  4 349 rows — total pesticide use (tonnes) per country/year
rainfall     :  6 727 rows — average annual rainfall (mm) per country/year
temp         : 71 311 rows — average temperature (°C) per country/year
yield_df     : 28 242 rows — pre-merged FAO dataset (yield+rain+pest+temp)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger

from src.utils.config import settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV, trying UTF-8 then Latin-1 encoding."""
    try:
        df = pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", **kwargs)
        logger.warning(f"Used latin-1 encoding for {path.name}")
    logger.info(f"Loaded {path.name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_crop_yield() -> pd.DataFrame:
    """
    Load the Agriculture Crop Yield dataset (synthetic, 1 M rows).

    Columns
    -------
    Region, Soil_Type, Crop, Rainfall_mm, Temperature_Celsius,
    Fertilizer_Used, Irrigation_Used, Weather_Condition,
    Days_to_Harvest, Yield_tons_per_hectare

    Notes
    -----
    Yield is expressed in **tons per hectare** (t/ha).
    No country or year dimension — regional-level synthetic data.
    """
    path = settings.paths.dataset_1
    df = _read(path)

    # Normalise column names to snake_case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Cast booleans (may arrive as strings)
    for col in ["fertilizer_used", "irrigation_used"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False})

    return df


def load_yield_fao() -> pd.DataFrame:
    """
    Load the FAO crop yield dataset (56 717 rows).

    Columns kept: Area, Item, Year, Value (hg/ha)
    Renamed to: country, crop, year, yield_hg_ha
    """
    path = settings.paths.dataset_2_yield
    df = _read(path)

    # Keep only the Yield element (already the only Element in this file)
    df = df.rename(columns={"Area": "country", "Item": "crop", "Year": "year", "Value": "yield_hg_ha"})
    df = df[["country", "crop", "year", "yield_hg_ha"]].copy()
    df["yield_hg_ha"] = pd.to_numeric(df["yield_hg_ha"], errors="coerce")
    df = df.dropna(subset=["yield_hg_ha"])

    return df


def load_pesticides() -> pd.DataFrame:
    """
    Load pesticide use dataset (4 349 rows).

    Columns kept: Area, Year, Value
    Renamed to: country, year, pesticides_tonnes
    """
    path = settings.paths.dataset_2_pesticides
    df = _read(path)

    df = df.rename(columns={"Area": "country", "Year": "year", "Value": "pesticides_tonnes"})
    df = df[["country", "year", "pesticides_tonnes"]].copy()
    df["pesticides_tonnes"] = pd.to_numeric(df["pesticides_tonnes"], errors="coerce")

    return df


def load_rainfall() -> pd.DataFrame:
    """
    Load average rainfall dataset (6 727 rows).

    Columns: Area, Year, average_rain_fall_mm_per_year
    Renamed to: country, year, rainfall_mm
    """
    path = settings.paths.dataset_2_rainfall
    df = _read(path)

    df = df.rename(
        columns={
            " Area": "country",
            "Area": "country",
            "Year": "year",
            "average_rain_fall_mm_per_year": "rainfall_mm",
        }
    )
    # Strip leading spaces from column names (some files have ' Area')
    df.columns = df.columns.str.strip()
    if "Area" in df.columns:
        df = df.rename(columns={"Area": "country"})

    df = df[["country", "year", "rainfall_mm"]].copy()
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    return df


def load_temp() -> pd.DataFrame:
    """
    Load average temperature dataset (71 311 rows).

    Columns: year, country, avg_temp
    Kept as-is.
    """
    path = settings.paths.dataset_2_temp
    df = _read(path)

    df = df.rename(columns={"year": "year", "country": "country", "avg_temp": "avg_temp"})
    df = df[["country", "year", "avg_temp"]].copy()
    df["avg_temp"] = pd.to_numeric(df["avg_temp"], errors="coerce")

    return df


def load_yield_df() -> pd.DataFrame:
    """
    Load the pre-merged FAO dataset (28 242 rows).

    Columns: Area, Item, Year, hg/ha_yield, average_rain_fall_mm_per_year,
             pesticides_tonnes, avg_temp

    Renamed to: country, crop, year, yield_hg_ha, rainfall_mm,
                pesticides_tonnes, avg_temp
    """
    path = settings.paths.raw_dir / "yield_df.csv"
    df = _read(path)

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.rename(
        columns={
            "Area": "country",
            "Item": "crop",
            "Year": "year",
            "hg/ha_yield": "yield_hg_ha",
            "average_rain_fall_mm_per_year": "rainfall_mm",
        }
    )

    numeric_cols = ["yield_hg_ha", "rainfall_mm", "pesticides_tonnes", "avg_temp"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_all() -> Dict[str, pd.DataFrame]:
    """
    Load all raw datasets and return them in a named dictionary.

    Returns
    -------
    dict with keys:
        crop_yield, yield_fao, pesticides, rainfall, temp, yield_df
    """
    return {
        "crop_yield": load_crop_yield(),
        "yield_fao": load_yield_fao(),
        "pesticides": load_pesticides(),
        "rainfall": load_rainfall(),
        "temp": load_temp(),
        "yield_df": load_yield_df(),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.logging import setup_logging

    setup_logging()
    datasets = load_all()
    for name, df in datasets.items():
        print(f"{name:12s} → {df.shape[0]:>8,} rows × {df.shape[1]:>2} cols | "
              f"nulls: {df.isnull().sum().sum():>6,}")
