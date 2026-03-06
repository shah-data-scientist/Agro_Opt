"""
Preprocessing transformations for AgroOpt.

All functions are pure (no side-effects) and return a new DataFrame.
Every decision that removes or modifies data is logged.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# crop_yield cleaning
# ---------------------------------------------------------------------------

def clean_crop_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the synthetic US crop-yield dataset.

    Transformations
    ---------------
    1. Drop 231 rows where yield_tons_per_hectare < 0  (data artefact).
    2. Normalise column names to snake_case (already done in loader).

    Parameters
    ----------
    df : raw crop_yield DataFrame from load_crop_yield()

    Returns
    -------
    Cleaned DataFrame.
    """
    n_before = len(df)

    mask_neg = df["yield_tons_per_hectare"] < 0
    n_neg = mask_neg.sum()
    if n_neg:
        logger.warning(
            f"crop_yield: dropping {n_neg} rows with negative yield "
            f"({n_neg/n_before*100:.3f}% of dataset)"
        )
        df = df[~mask_neg].copy()

    logger.info(
        f"crop_yield cleaned: {len(df):,} rows retained "
        f"(removed {n_before - len(df):,})"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# USA FAO data extraction — yearly, per crop
# ---------------------------------------------------------------------------

# Country names used for USA differ across FAO raw files:
#   yield.csv      → "United States of America"
#   pesticides.csv → "United States of America"
#   rainfall.csv   → "United States"  (short form — the original mismatch)
#   temp.csv       → "United States"  (short form, 52 state entries per year)
USA_IN_YIELD = "United States of America"
USA_IN_PEST  = "United States of America"
USA_IN_RAIN  = "United States"
USA_IN_TEMP  = "United States"

# Crop name mapping  FAO name → crop_yield name
FAO_TO_CY_CROP: dict[str, str] = {
    "Maize":         "Maize",
    "Rice, paddy":   "Rice",
    "Soybeans":      "Soybean",
    "Wheat":         "Wheat",
    # FAO crops with no crop_yield counterpart (excluded from enrichment):
    # "Potatoes", "Sorghum", "Sweet potatoes", "Cassava", "Yams", ...
}


def build_usa_fao_lookup(
    raw_yield_path: str,
    raw_pest_path: str,
    raw_rain_path: str,
    raw_temp_path: str,
    year_start: int = 1990,
    year_end: int = 2013,
) -> pd.DataFrame:
    """
    Extract USA rows from the four raw FAO files, keeping the year dimension.

    Each file uses a different country name for USA — handled explicitly.

    Steps
    -----
    1. yield.csv      → filter "United States of America", 1990-2013.
    2. pesticides.csv → filter "United States of America".
    3. rainfall.csv   → filter "United States" (short form).
    4. temp.csv       → filter "United States", aggregate 52 state rows
                        to one national mean per year.
    5. Join all four on year (country fixed to USA so join key = year only).
    6. Filter to the 4 crops that have a crop_yield counterpart.
    7. Map FAO crop names → crop_yield crop names.
    8. Return one row per (crop, year) — yearly values, no aggregation.

    Returns
    -------
    DataFrame with columns:
        crop                  (crop_yield name, e.g. "Maize")
        year                  (1990–2013)
        fao_yield_hg_ha       (USA FAO yield for that crop-year, hg/ha)
        fao_rainfall_mm       (USA annual rainfall for that year, mm)
        fao_pesticides_tonnes (USA total pesticides for that year, tonnes)
        fao_avg_temp          (USA annual temperature for that year, °C)
    """
    # ------------------------------------------------------------------ yield
    y = pd.read_csv(raw_yield_path)
    y = y.rename(columns={
        "Area": "country", "Item": "fao_crop",
        "Year": "year", "Value": "fao_yield_hg_ha"
    })
    y = y[["country", "fao_crop", "year", "fao_yield_hg_ha"]]
    y = y[(y["country"] == USA_IN_YIELD) & y["year"].between(year_start, year_end)]
    y["fao_yield_hg_ha"] = pd.to_numeric(y["fao_yield_hg_ha"], errors="coerce")
    logger.info(f"USA yield rows ({year_start}-{year_end}): {len(y)}")

    # --------------------------------------------------------------- pesticides
    p = pd.read_csv(raw_pest_path)
    p = p.rename(columns={"Area": "country", "Year": "year", "Value": "fao_pesticides_tonnes"})
    p = p[["country", "year", "fao_pesticides_tonnes"]]
    p = p[(p["country"] == USA_IN_PEST) & p["year"].between(year_start, year_end)]
    p["fao_pesticides_tonnes"] = pd.to_numeric(p["fao_pesticides_tonnes"], errors="coerce")
    # Drop country col — join key is year only (country is fixed to USA)
    p = p[["year", "fao_pesticides_tonnes"]]
    logger.info(f"USA pesticides rows ({year_start}-{year_end}): {len(p)}")

    # ----------------------------------------------------------------- rainfall
    r = pd.read_csv(raw_rain_path)
    r.columns = r.columns.str.strip()
    r = r.rename(columns={
        "Area": "country", "Year": "year",
        "average_rain_fall_mm_per_year": "fao_rainfall_mm"
    })
    r = r[["country", "year", "fao_rainfall_mm"]]
    r = r[(r["country"] == USA_IN_RAIN) & r["year"].between(year_start, year_end)]
    r["fao_rainfall_mm"] = pd.to_numeric(r["fao_rainfall_mm"], errors="coerce")
    n_null_rain = r["fao_rainfall_mm"].isnull().sum()
    if n_null_rain:
        med = r["fao_rainfall_mm"].median()
        r["fao_rainfall_mm"] = r["fao_rainfall_mm"].fillna(med)
        logger.warning(f"USA rainfall: imputed {n_null_rain} nulls with median {med:.1f} mm")
    r = r[["year", "fao_rainfall_mm"]]
    logger.info(f"USA rainfall rows ({year_start}-{year_end}): {len(r)}")

    # ---------------------------------------------------------------- temperature
    # temp.csv has 52 state-level rows per year for USA → aggregate to national mean
    t = pd.read_csv(raw_temp_path)
    t["avg_temp"] = pd.to_numeric(t["avg_temp"], errors="coerce")
    t = t[(t["country"] == USA_IN_TEMP) & t["year"].between(year_start, year_end)]
    t_annual = (
        t.groupby("year")["avg_temp"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"avg_temp": "fao_avg_temp"})
    )
    n_null_temp = t_annual["fao_avg_temp"].isnull().sum()
    if n_null_temp:
        med = t_annual["fao_avg_temp"].median()
        t_annual["fao_avg_temp"] = t_annual["fao_avg_temp"].fillna(med)
        logger.warning(f"USA temp: imputed {n_null_temp} nulls with median {med:.1f} °C")
    logger.info(
        f"USA temp: {len(t)} state rows → {len(t_annual)} annual means "
        f"(range {t_annual['fao_avg_temp'].min():.1f}–{t_annual['fao_avg_temp'].max():.1f} °C)"
    )

    # --------------------------------- join all tables on year (country = USA fixed)
    # y has (fao_crop, year, fao_yield_hg_ha) — keep fao_crop to identify crops
    merged = (
        y
        .merge(p,        on="year", how="left")
        .merge(r,        on="year", how="left")
        .merge(t_annual, on="year", how="left")
    )
    logger.info(
        f"USA FAO joined (by year): {len(merged)} rows "
        f"across {merged['fao_crop'].nunique()} crops × "
        f"{merged['year'].nunique()} years"
    )

    # Post-join null check: rainfall may be missing for entire years
    # (e.g. 2003 absent from rainfall.csv) — impute with column median
    n_null_rain_post = merged["fao_rainfall_mm"].isnull().sum()
    if n_null_rain_post:
        med_rain = merged["fao_rainfall_mm"].median()
        merged["fao_rainfall_mm"] = merged["fao_rainfall_mm"].fillna(med_rain)
        logger.warning(
            f"fao_rainfall_mm: imputed {n_null_rain_post} post-join nulls "
            f"(year missing from rainfall.csv) with median {med_rain:.1f} mm"
        )

    # --------------------------------- filter to crops that have a crop_yield match
    lookup = merged[merged["fao_crop"].isin(FAO_TO_CY_CROP)].copy()
    lookup["crop"] = lookup["fao_crop"].map(FAO_TO_CY_CROP)
    lookup = lookup.drop(columns=["fao_crop", "country"])

    for col in ["fao_yield_hg_ha", "fao_rainfall_mm", "fao_pesticides_tonnes", "fao_avg_temp"]:
        lookup[col] = lookup[col].round(2)

    # Column order: crop, year first
    lookup = lookup[["crop", "year", "fao_yield_hg_ha", "fao_rainfall_mm",
                      "fao_pesticides_tonnes", "fao_avg_temp"]]

    logger.info(
        f"USA FAO lookup: {lookup['crop'].nunique()} crops × "
        f"{lookup['year'].nunique()} years = {len(lookup)} rows"
    )
    logger.info(f"\n{lookup.to_string(index=False)}")

    return lookup
