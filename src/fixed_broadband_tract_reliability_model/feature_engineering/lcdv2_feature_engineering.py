import operator
from functools import reduce
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

WAREHOUSE_PATH = Path("../ml_feature_pipeline/dbt_project/warehouse.duckdb")
OUTPUT_PATH = Path("data/processed/lcdv2_tract_quarter_features.parquet")
LCDV2_TABLE = "fact_lcdv2_tract_hourly"


# ---------------------------------------------------------
# Load LCDv2 from DuckDB
# ---------------------------------------------------------


def load_lcdv2() -> pl.DataFrame:
    con = duckdb.connect(str(WAREHOUSE_PATH))
    df = con.execute(f"SELECT * FROM {LCDV2_TABLE}").pl()
    con.close()
    return df


# ---------------------------------------------------------
# Add year + quarter fields
# ---------------------------------------------------------


def add_time_fields(df: pl.DataFrame) -> pl.DataFrame:
    quarter_expr = pl.col("floored_timestamp").dt.quarter()

    return df.with_columns(
        [
            pl.col("floored_timestamp").dt.year().alias("year"),
            pl.when(quarter_expr == 1)
            .then(pl.lit("Q1"))
            .when(quarter_expr == 2)
            .then(pl.lit("Q2"))
            .when(quarter_expr == 3)
            .then(pl.lit("Q3"))
            .otherwise(pl.lit("Q4"))
            .alias("quarter"),
        ]
    )


# ---------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------


def pct(expr: pl.Expr, prefix: str):
    return [
        expr.quantile(0.10).alias(f"{prefix}_p10"),
        expr.quantile(0.25).alias(f"{prefix}_p25"),
        expr.quantile(0.75).alias(f"{prefix}_p75"),
        expr.quantile(0.90).alias(f"{prefix}_p90"),
    ]


# ---------------------------------------------------------
# Circular wind direction features
# ---------------------------------------------------------


def circular_direction_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("weighted_wind_direction") * np.pi / 180).alias("wind_dir_rad")
    )

    df = df.with_columns(
        [
            pl.col("wind_dir_rad").cos().alias("wind_dir_cos"),
            pl.col("wind_dir_rad").sin().alias("wind_dir_sin"),
        ]
    )

    grouped = df.group_by(["tract", "year", "quarter"]).agg(
        [
            pl.col("wind_dir_cos").mean().alias("mean_cos"),
            pl.col("wind_dir_sin").mean().alias("mean_sin"),
        ]
    )

    grouped = grouped.with_columns(
        np.sqrt(pl.col("mean_cos") ** 2 + pl.col("mean_sin") ** 2).alias(
            "wind_dir_resultant_length"
        )
    )

    grouped = grouped.with_columns(
        (1 - pl.col("wind_dir_resultant_length")).alias("wind_dir_circular_variance")
    )

    return grouped.select(
        [
            "tract",
            "year",
            "quarter",
            "wind_dir_resultant_length",
            "wind_dir_circular_variance",
        ]
    )


# ---------------------------------------------------------
# Hourly-level extreme-condition booleans
# ---------------------------------------------------------


def add_extreme_condition_booleans(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            # Temperature
            pl.col("weighted_dry_bulb_temp").lt(32).alias("temp_below_freezing"),
            pl.col("weighted_dry_bulb_temp").gt(90).alias("temp_above_90"),
            # Wind gust tiers
            pl.col("weighted_wind_gust_speed").gt(20).alias("gust_gt_20"),
            pl.col("weighted_wind_gust_speed").gt(30).alias("gust_gt_30"),
            pl.col("weighted_wind_gust_speed").gt(40).alias("gust_gt_40"),
            pl.col("weighted_wind_gust_speed").gt(50).alias("gust_gt_50"),
            # Precip buckets
            pl.col("weighted_precipitation").gt(0.01).alias("precip_gt_0_01"),
            pl.col("weighted_precipitation").gt(0.10).alias("precip_gt_0_10"),
            pl.col("weighted_precipitation").gt(0.25).alias("precip_gt_0_25"),
            pl.col("weighted_precipitation").gt(0.50).alias("precip_gt_0_50"),
            # Visibility
            pl.col("weighted_visibility").lt(1).alias("visibility_lt_1"),
            # Pressure
            pl.col("weighted_station_pressure").lt(29.0).alias("low_pressure"),
        ]
    )


# ---------------------------------------------------------
# Compound weather stress booleans
# ---------------------------------------------------------


def add_compound_booleans(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                pl.col("weighted_precipitation").gt(0)
                & pl.col("weighted_wind_speed").gt(15)
            ).alias("wet_windy"),
            (
                pl.col("weighted_dry_bulb_temp").lt(32)
                & pl.col("weighted_wind_speed").gt(15)
            ).alias("cold_windy"),
            (
                pl.col("weighted_dry_bulb_temp").gt(85)
                & pl.col("weighted_relative_humidity").gt(80)
            ).alias("hot_humid"),
        ]
    )


# ---------------------------------------------------------
# Storm event booleans
# ---------------------------------------------------------


def add_storm_booleans(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                pl.col("weighted_wind_gust_speed").gt(30)
                & pl.col("weighted_precipitation").gt(0.10)
                & pl.col("weighted_visibility").lt(2)
                & pl.col("weighted_station_pressure").lt(29.2)
            ).alias("storm_event")
        ]
    )


# ---------------------------------------------------------
# Run-length encoding helper for clusters
# ---------------------------------------------------------


def count_clusters(boolean_series: pl.Series, min_length: int = 3) -> int:
    arr = boolean_series.to_numpy()
    count = 0
    current = 0
    for val in arr:
        if val:
            current += 1
        else:
            if current >= min_length:
                count += 1
            current = 0
    if current >= min_length:
        count += 1
    return count


# ---------------------------------------------------------
# Sustained high-wind periods + severe clusters
# ---------------------------------------------------------


def compute_cluster_features(df: pl.DataFrame) -> pl.DataFrame:
    results = []

    for keys, group in df.group_by(["tract", "year", "quarter"], maintain_order=True):
        tract, year, quarter = keys

        sustained = count_clusters(group["weighted_wind_speed"].gt(20), min_length=3)
        severe_clusters = count_clusters(group["storm_event"], min_length=3)

        results.append(
            {
                "tract": tract,
                "year": year,
                "quarter": quarter,
                "sustained_high_wind_periods": sustained,
                "severe_weather_clusters": severe_clusters,
            }
        )

    return pl.DataFrame(results)


# ---------------------------------------------------------
# Core tract-quarter aggregation (including counts)
# ---------------------------------------------------------


def aggregate_tract_quarter(df: pl.DataFrame) -> pl.DataFrame:
    numeric_vars = [
        "weighted_dry_bulb_temp",
        "weighted_wet_bulb_temp",
        "weighted_dew_point_temp",
        "weighted_relative_humidity",
        "weighted_wind_speed",
        "weighted_wind_gust_speed",
        "weighted_precipitation",
        "weighted_visibility",
        "weighted_station_pressure",
        "weighted_barometric_pressure",
    ]

    boolean_vars = [
        "temp_below_freezing",
        "temp_above_90",
        "gust_gt_20",
        "gust_gt_30",
        "gust_gt_40",
        "gust_gt_50",
        "precip_gt_0_01",
        "precip_gt_0_10",
        "precip_gt_0_25",
        "precip_gt_0_50",
        "visibility_lt_1",
        "low_pressure",
        "wet_windy",
        "cold_windy",
        "hot_humid",
        "storm_event",
    ]

    agg_exprs = []

    # Numeric summaries
    for var in numeric_vars:
        agg_exprs.extend(
            [
                pl.col(var).mean().alias(f"{var}_mean"),
                pl.col(var).median().alias(f"{var}_median"),
                pl.col(var).std().alias(f"{var}_std"),
                pl.col(var).min().alias(f"{var}_min"),
                pl.col(var).max().alias(f"{var}_max"),
                (pl.col(var).max() - pl.col(var).min()).alias(f"{var}_range"),
                (pl.col(var).std() / pl.col(var).mean()).alias(f"{var}_cv"),
            ]
        )
        agg_exprs.extend(pct(pl.col(var), var))

    # Boolean → count
    for var in boolean_vars:
        agg_exprs.append(pl.col(var).sum().alias(f"{var}_hours"))

    # Coverage
    agg_exprs.append(pl.len().alias("observed_hours"))

    grouped = df.group_by(["tract", "year", "quarter"]).agg(agg_exprs)

    # Expected hours per quarter (approx)
    grouped = grouped.with_columns(pl.lit(24 * 90).alias("expected_hours"))

    grouped = grouped.with_columns(
        (pl.col("observed_hours") / pl.col("expected_hours")).alias("coverage_ratio")
    )

    return grouped


# ---------------------------------------------------------
# Sanitization metadata
# ---------------------------------------------------------


def add_sanitization_metadata(df: pl.DataFrame) -> pl.DataFrame:
    sanitized_cols = [
        col
        for col in df.columns
        if col.endswith("_mean")
        or col.endswith("_median")
        or col.endswith("_std")
        or col.endswith("_min")
        or col.endswith("_max")
        or col.endswith("_range")
        or col.endswith("_cv")
        or col.endswith("_p10")
        or col.endswith("_p25")
        or col.endswith("_p75")
        or col.endswith("_p90")
    ]

    df = df.with_columns(
        [pl.col(col).is_null().alias(f"{col}_was_null") for col in sanitized_cols]
    )

    null_sum_expr = reduce(
        operator.add, [pl.col(f"{col}_was_null") for col in sanitized_cols]
    )

    df = df.with_columns(null_sum_expr.alias("num_sanitized_fields"))

    df = df.with_columns(
        [
            (pl.col("num_sanitized_fields") > 0).alias("is_sanitized"),
            pl.when(pl.col("num_sanitized_fields") == 0)
            .then(pl.lit("none"))
            .when(pl.col("num_sanitized_fields") <= 5)
            .then(pl.lit("partial"))
            .when(pl.col("num_sanitized_fields") <= 12)
            .then(pl.lit("moderate"))
            .otherwise(pl.lit("high"))
            .alias("sanitization_level"),
        ]
    )

    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():
    print("[1/8] Loading LCDv2 from DuckDB...")
    df = load_lcdv2()

    print("[2/8] Adding time fields...")
    df = add_time_fields(df)

    print("[3/8] Computing hourly booleans (extreme, compound, storm)...")
    df = add_extreme_condition_booleans(df)
    df = add_compound_booleans(df)
    df = add_storm_booleans(df)

    print("[4/8] Computing circular wind direction features...")
    wind_dir = circular_direction_features(df)

    print("[5/8] Aggregating tract-quarter features...")
    grouped = aggregate_tract_quarter(df)

    print("[6/8] Computing cluster features (sustained wind + severe weather)...")
    clusters = compute_cluster_features(df)

    print("[7/8] Joining feature sets and adding sanitization metadata...")
    final = grouped.join(wind_dir, on=["tract", "year", "quarter"], how="left").join(
        clusters, on=["tract", "year", "quarter"], how="left"
    )
    final = add_sanitization_metadata(final)

    print("[8/8] Writing output parquet...")
    final.write_parquet(OUTPUT_PATH)

    print("LCDv2 feature engineering complete.")


if __name__ == "__main__":
    main()
