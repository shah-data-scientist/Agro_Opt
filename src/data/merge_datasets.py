"""
Phase 2 — Data Integration for AgroOpt.

Strategy
--------
The two source datasets belong to different contexts:

  crop_yield.csv  — 1 000 000 synthetic US agronomic observations
                    (Region: East/West/North/South = US Census regions)
                    Columns: region, soil_type, crop, rainfall_mm,
                             temperature_celsius, fertilizer_used,
                             irrigation_used, weather_condition,
                             days_to_harvest, yield_tons_per_hectare

  FAO raw files   — Real-world US yield data extracted from the FAO
                    family (yield.csv + pesticides.csv + rainfall.csv
                    + temp.csv), filtered to United States of America,
                    1990–2013.
                    Note: yield_df.csv is NOT used directly because it
                    excluded USA due to a country-name mismatch bug.

Merge logic
-----------
1. Build USA FAO lookup for year 2013 only (most recent reference year).
2. Map FAO crop names → crop_yield crop names.
3. Filter crop_yield to only crops that have FAO USA data
   (Maize, Rice, Soybean, Wheat). Barley and Cotton are excluded
   because they have no FAO USA counterpart.
4. INNER JOIN on 'crop' — zero NaN values guaranteed.
   The year column is set to 2013 for all rows (FAO reference year).
5. Convert yield_tons_per_hectare → yield_hg_ha (× 10 000) to match
   the FAO unit (hg/ha).  yield_tons_per_hectare is then dropped.

Output columns (all retained)
------------------------------
From crop_yield  : crop, year, region, soil_type, weather_condition,
                   rainfall_mm, temperature_celsius, fertilizer_used,
                   irrigation_used, days_to_harvest
Target variable  : yield_hg_ha   (= yield_tons_per_hectare × 10 000)
From FAO USA     : fao_yield_hg_ha, fao_rainfall_mm,
                   fao_pesticides_tonnes, fao_avg_temp
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.load_data import load_crop_yield
from src.data.preprocess import build_usa_fao_lookup, clean_crop_yield
from src.utils.config import settings
from src.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_datasets(save: bool = True) -> pd.DataFrame:
    """
    Build the merged dataset and optionally save to data/processed/.

    Parameters
    ----------
    save : persist merged_dataset.csv to disk.

    Returns
    -------
    merged : pd.DataFrame with all columns from both sources.
    """
    settings.paths.ensure_dirs()

    # ------------------------------------------------------------------
    # Step 1 — Load and clean crop_yield
    # ------------------------------------------------------------------
    logger.info("Step 1 — Loading and cleaning crop_yield …")
    cy = load_crop_yield()
    cy = clean_crop_yield(cy)
    logger.info(f"  crop_yield clean: {len(cy):,} rows × {cy.shape[1]} cols")

    # ------------------------------------------------------------------
    # Step 2 — Build USA FAO lookup for year 2013 (most recent reference)
    # ------------------------------------------------------------------
    FAO_REFERENCE_YEAR = 2013
    logger.info(f"Step 2 — Building USA FAO lookup for year {FAO_REFERENCE_YEAR} …")
    fao_lookup = build_usa_fao_lookup(
        raw_yield_path=str(settings.paths.dataset_2_yield),
        raw_pest_path=str(settings.paths.dataset_2_pesticides),
        raw_rain_path=str(settings.paths.dataset_2_rainfall),
        raw_temp_path=str(settings.paths.dataset_2_temp),
        year_start=FAO_REFERENCE_YEAR,
        year_end=FAO_REFERENCE_YEAR,
    )
    logger.info(
        f"  FAO lookup (year {FAO_REFERENCE_YEAR}): "
        f"{len(fao_lookup)} crops — {sorted(fao_lookup['crop'].tolist())}"
    )

    # ------------------------------------------------------------------
    # Step 3 — Filter crop_yield to crops with FAO USA coverage
    # ------------------------------------------------------------------
    CROPS_WITH_FAO = set(fao_lookup["crop"].unique())
    n_before_filter = len(cy)
    cy = cy[cy["crop"].isin(CROPS_WITH_FAO)].copy().reset_index(drop=True)
    n_dropped = n_before_filter - len(cy)
    logger.warning(
        f"  Removed {n_dropped:,} rows for crops without FAO USA data "
        f"(Barley, Cotton). {len(cy):,} rows retained → crops: {sorted(CROPS_WITH_FAO)}"
    )

    # ------------------------------------------------------------------
    # Step 4 — INNER JOIN on crop (FAO reference year = 2013 for all rows)
    # ------------------------------------------------------------------
    logger.info(f"Step 4 — Merging on 'crop' (inner join, FAO ref year = {FAO_REFERENCE_YEAR}) …")
    merged = cy.merge(fao_lookup, on="crop", how="inner")
    logger.info(
        f"  Merged rows: {len(merged):,} | "
        f"Crops: {sorted(merged['crop'].unique())} | "
        f"FAO reference year: {merged['year'].unique()[0]}"
    )

    # ------------------------------------------------------------------
    # Step 5 — Unit conversion: t/ha → hg/ha  (1 t/ha = 10 000 hg/ha)
    # ------------------------------------------------------------------
    logger.info("Step 5 — Converting yield_tons_per_hectare → yield_hg_ha (× 10 000) …")
    merged["yield_hg_ha"] = (merged["yield_tons_per_hectare"] * 10_000).round(2)
    merged = merged.drop(columns=["yield_tons_per_hectare"])
    logger.info(
        f"  yield_hg_ha range: "
        f"{merged['yield_hg_ha'].min():,.0f} – {merged['yield_hg_ha'].max():,.0f} hg/ha"
    )

    # ------------------------------------------------------------------
    # Step 6 — Column order
    # ------------------------------------------------------------------
    col_order = [
        # Crop identifier
        "crop",
        # FAO reference year (constant = 2013 for all rows)
        "year",
        # US regional context
        "region",
        "soil_type",
        "weather_condition",
        # Agronomic observations (from synthetic US data)
        "rainfall_mm",
        "temperature_celsius",
        "fertilizer_used",
        "irrigation_used",
        "days_to_harvest",
        # Target variable (hg/ha — same unit as fao_yield_hg_ha)
        "yield_hg_ha",
        # FAO USA enrichment columns (actual yearly values, 1990-2013)
        "fao_yield_hg_ha",
        "fao_rainfall_mm",
        "fao_pesticides_tonnes",
        "fao_avg_temp",
    ]
    merged = merged[col_order]

    logger.info(
        f"Merged dataset: {len(merged):,} rows × {merged.shape[1]} cols"
    )
    logger.info(f"Columns: {list(merged.columns)}")
    logger.info(f"\n{merged.describe(include='all').T.to_string()}")

    # ------------------------------------------------------------------
    # Step 7 — Null audit
    # ------------------------------------------------------------------
    null_report = merged.isnull().sum()
    null_report = null_report[null_report > 0]
    if len(null_report):
        logger.warning("Null audit — unexpected nulls found:")
        for col, n in null_report.items():
            logger.warning(f"  {col}: {n:,} nulls ({n/len(merged)*100:.1f}%)")
    else:
        logger.info("Null audit: no nulls. ✓")

    # ------------------------------------------------------------------
    # Step 8 — Save
    # ------------------------------------------------------------------
    if save:
        out_path = settings.paths.merged
        merged.to_csv(out_path, index=False)
        logger.info(f"Saved → {out_path}")

        # Save merge metadata as a sidecar JSON
        crops_in_merged = sorted(merged["crop"].unique().tolist())
        years_in_merged = sorted(int(y) for y in merged["year"].unique())
        meta = {
            "rows": len(merged),
            "cols": list(merged.columns),
            "crops": crops_in_merged,
            "crops_excluded": ["Barley", "Cotton"],
            "exclusion_reason": "No FAO USA yield data available — would create NaN values",
            "years": years_in_merged,
            "fao_source": "United States of America, 1990–2013 (yearly per crop)",
            "target": "yield_hg_ha",
            "yield_unit": "hg/ha",
            "total_nulls": int(merged.isnull().sum().sum()),
            "null_counts": {k: int(v) for k, v in merged.isnull().sum().items()},
        }
        meta_path = Path(str(out_path)).with_suffix(".json")
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metadata → {meta_path}")

    return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()
    df = merge_datasets(save=True)

    print("\n" + "=" * 65)
    print("MERGE COMPLETE")
    print("=" * 65)
    print(f"Shape      : {df.shape}")
    print(f"Columns    : {list(df.columns)}")
    print(f"Target     : yield_hg_ha  (hg/ha)")
    print()
    print("Null counts per column:")
    nulls = df.isnull().sum()
    for col, n in nulls.items():
        bar = "▓" * int(n / len(df) * 30)
        print(f"  {col:<28s} {n:>7,}  {bar}")
    print()
    print("Year range:", df["year"].min(), "–", df["year"].max())
    print()
    print("Crops in merged dataset:")
    print(df.groupby("crop")[["yield_hg_ha", "fao_yield_hg_ha"]].agg(
        ["count", "mean"]
    ).round(0).to_string())
    print()
    print("Sample rows:")
    print(df.sample(5, random_state=42).to_string(index=False))
