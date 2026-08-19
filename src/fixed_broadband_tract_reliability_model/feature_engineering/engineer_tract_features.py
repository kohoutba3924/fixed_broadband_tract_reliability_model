import operator
from functools import reduce
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from shapely import wkb

WAREHOUSE_PATH = Path("../ml_feature_pipeline/dbt_project/warehouse.duckdb")
OUTPUT_PATH = Path("data/processed/feature_engineering/tract_features.parquet")
DIM_TABLE = "dim_tract"


# ---------------------------------------------------------
# Load dim_tract from DuckDB
# ---------------------------------------------------------


def load_dim_tract() -> pl.DataFrame:
    con = duckdb.connect(str(WAREHOUSE_PATH))
    df = con.execute(f"SELECT * FROM {DIM_TABLE}").pl()
    con.close()
    return df


# ---------------------------------------------------------
# Geometry decoding: area, perimeter, compactness
# ---------------------------------------------------------


def decode_geometry(df: pl.DataFrame) -> pl.DataFrame:
    def compute_metrics(wkb_bytes):
        geom = wkb.loads(wkb_bytes)
        area = geom.area
        perimeter = geom.length
        compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else None
        return area, perimeter, compactness

    metrics = df.select("geometry_wkb").to_series().map_elements(compute_metrics)

    areas = [m[0] for m in metrics]
    perims = [m[1] for m in metrics]
    compacts = [m[2] for m in metrics]

    return df.with_columns(
        [
            pl.Series("tract_area", areas),
            pl.Series("tract_perimeter", perims),
            pl.Series("tract_compactness", compacts),
        ]
    )


# ---------------------------------------------------------
# ACS ratio engineering
# ---------------------------------------------------------


def engineer_acs_features(df: pl.DataFrame) -> pl.DataFrame:
    # Create all intermediate totals
    df = df.with_columns(
        [
            # Age structure totals
            (
                pl.col("male_65_66")
                + pl.col("male_67_69")
                + pl.col("male_70_74")
                + pl.col("male_75_79")
                + pl.col("male_80_84")
                + pl.col("male_85_plus")
                + pl.col("female_65_66")
                + pl.col("female_67_69")
                + pl.col("female_70_74")
                + pl.col("female_75_79")
                + pl.col("female_80_84")
                + pl.col("female_85_plus")
            ).alias("population_65_plus"),
            (pl.col("male_under_5") + pl.col("female_under_5")).alias(
                "population_under_5"
            ),
            # Disability total
            (
                pl.col("disability_under_18")
                + pl.col("disability_18_64")
                + pl.col("disability_65_plus")
            ).alias("disability_total"),
            # Education total
            (
                pl.col("edu_high_school")
                + pl.col("edu_bachelors")
                + pl.col("edu_masters")
                + pl.col("edu_professional")
                + pl.col("edu_doctorate")
            ).alias("edu_total"),
            # Language totals
            (
                pl.col("limited_english_5_17")
                + pl.col("limited_english_18_64")
                + pl.col("limited_english_65_plus")
            ).alias("limited_english_total"),
            (
                pl.col("no_english_5_17")
                + pl.col("no_english_18_64")
                + pl.col("no_english_65_plus")
            ).alias("no_english_total"),
        ]
    )

    # Compute ratios using the totals created above
    df = df.with_columns(
        [
            # Sex ratios
            (pl.col("population_male") / pl.col("population_total")).alias("pct_male"),
            (pl.col("population_female") / pl.col("population_total")).alias(
                "pct_female"
            ),
            (pl.col("population_male") / pl.col("population_female")).alias(
                "sex_ratio"
            ),
            # Age structure ratios
            (pl.col("population_65_plus") / pl.col("population_total")).alias(
                "pct_65_plus"
            ),
            (pl.col("population_under_5") / pl.col("population_total")).alias(
                "pct_under_5"
            ),
            # Disability ratio
            (pl.col("disability_total") / pl.col("population_total")).alias(
                "pct_disability"
            ),
            # Education ratios
            (pl.col("edu_high_school") / pl.col("edu_total")).alias("pct_high_school"),
            (pl.col("edu_bachelors") / pl.col("edu_total")).alias("pct_bachelors"),
            (
                (
                    pl.col("edu_masters")
                    + pl.col("edu_professional")
                    + pl.col("edu_doctorate")
                )
                / pl.col("edu_total")
            ).alias("pct_grad_degree"),
            # Poverty
            (pl.col("poverty_below") / pl.col("poverty_universe")).alias("pct_poverty"),
            # Unemployment
            (pl.col("unemployment_civilian") / pl.col("labor_force_civilian")).alias(
                "pct_unemployment"
            ),
            # Housing occupancy
            (pl.col("housing_occupied") / pl.col("housing_units")).alias(
                "pct_housing_occupied"
            ),
            (pl.col("housing_vacant") / pl.col("housing_units")).alias(
                "pct_housing_vacant"
            ),
            # Race / ethnicity
            (pl.col("race_white") / pl.col("race_universe")).alias("pct_white"),
            (pl.col("race_black") / pl.col("race_universe")).alias("pct_black"),
            (pl.col("race_asian") / pl.col("race_universe")).alias("pct_asian"),
            (pl.col("race_hispanic") / pl.col("race_universe")).alias("pct_hispanic"),
            (pl.col("race_american_indian") / pl.col("race_universe")).alias(
                "pct_american_indian"
            ),
            (pl.col("race_pacific_islander") / pl.col("race_universe")).alias(
                "pct_pacific_islander"
            ),
            (pl.col("race_other") / pl.col("race_universe")).alias("pct_race_other"),
            # Language isolation
            (pl.col("limited_english_total") / pl.col("language_universe")).alias(
                "pct_limited_english"
            ),
            (pl.col("no_english_total") / pl.col("language_universe")).alias(
                "pct_no_english"
            ),
            # Vehicle availability
            (pl.col("vehicle_none") / pl.col("vehicle_universe")).alias(
                "pct_vehicle_none"
            ),
            # Housing structure
            (pl.col("housing_1_unit") / pl.col("housing_structure_universe")).alias(
                "pct_single_family"
            ),
            (
                (
                    pl.col("housing_2_unit")
                    + pl.col("housing_3_4_unit")
                    + pl.col("housing_5_9_unit")
                )
                / pl.col("housing_structure_universe")
            ).alias("pct_small_multi_family"),
            (
                (
                    pl.col("housing_10_19_unit")
                    + pl.col("housing_20_49_unit")
                    + pl.col("housing_50_plus_unit")
                )
                / pl.col("housing_structure_universe")
            ).alias("pct_large_multi_family"),
            (
                pl.col("housing_mobile_home") / pl.col("housing_structure_universe")
            ).alias("pct_mobile_home"),
        ]
    )

    return df


# -----------------------------------------------------------------------
# Normalize ACS fields for empty tracts, represent non-residential areas
# -----------------------------------------------------------------------


def normalize_empty_tracts(df: pl.DataFrame) -> pl.DataFrame:

    empty_mask = pl.col("tract_code").str.starts_with("99")

    # Median fields to zero
    median_fields = [
        "median_age",
        "median_household_income",
        "median_home_value",
        "median_gross_rent",
    ]

    # Ratio fields (pct_*)
    ratio_fields = [col for col in df.columns if col.startswith("pct_")]

    # Additional derived fields needing normalization
    special_fields = ["sex_ratio"]

    df = df.with_columns(
        [
            # Median fields → 0
            pl.when(empty_mask).then(0).otherwise(pl.col(col)).alias(col)
            for col in median_fields
        ]
        + [
            # Ratio fields → 0
            pl.when(empty_mask).then(0).otherwise(pl.col(col)).alias(col)
            for col in ratio_fields
        ]
        + [
            # Special derived fields → 0
            pl.when(empty_mask).then(0).otherwise(pl.col(col)).alias(col)
            for col in special_fields
        ]
    )

    return df


# ---------------------------------------------------------
# Sanitization metadata
# ---------------------------------------------------------


def add_sanitization_metadata(df: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = [col for col in df.columns if col not in ["tract", "geometry_wkb"]]

    df = df.with_columns(
        [pl.col(col).is_null().alias(f"{col}_was_null") for col in numeric_cols]
    )

    null_sum_expr = reduce(
        operator.add, [pl.col(f"{col}_was_null") for col in numeric_cols]
    )

    df = df.with_columns(null_sum_expr.alias("tract_num_sanitized_fields"))

    df = df.with_columns(
        [
            (pl.col("tract_num_sanitized_fields") > 0).alias("tract_is_sanitized"),
            pl.when(pl.col("tract_num_sanitized_fields") == 0)
            .then(pl.lit("none"))
            .when(pl.col("tract_num_sanitized_fields") <= 10)
            .then(pl.lit("partial"))
            .when(pl.col("tract_num_sanitized_fields") <= 25)
            .then(pl.lit("moderate"))
            .otherwise(pl.lit("high"))
            .alias("tract_sanitization_level"),
        ]
    )

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():
    print("[1/7] Loading dim_tract from DuckDB...")
    df = load_dim_tract()

    print("[2/7] Decoding geometry (area, perimeter, compactness)...")
    df = decode_geometry(df)

    print("[3/7] Engineering ACS ratio features...")
    df = engineer_acs_features(df)

    print("[4/7] Normalizing ACS summary fields for non-residential tracts...")
    df = normalize_empty_tracts(df)

    print("[5/7] Dropping static fields and data types inconsistent with modeling...")
    df = df.drop(
        [
            "state_fips",
            "county_fips",
            "tract_code",
            "tract_bucket",
            "geometry_wkb",
        ]
    )

    print("[6/7] Adding sanitization metadata...")
    df = add_sanitization_metadata(df)

    print("[7/7] Writing output parquet...")
    df.write_parquet(OUTPUT_PATH)

    print("dim_tract feature engineering complete.")


if __name__ == "__main__":
    main()
