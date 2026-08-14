import operator
from functools import reduce
from pathlib import Path

import numpy as np
import polars as pl

INPUT_PATH = Path("data/processed/ookla_fixed_broadband_service_multi_quarter.parquet")
OUTPUT_PATH = Path("data/processed/ookla_tract_quarter_target_features.parquet")


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------


def load_ookla():
    return pl.read_parquet(INPUT_PATH)


# ---------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------


def compute_percentiles(expr: pl.Expr, prefix: str):
    return [
        expr.quantile(0.10).alias(f"{prefix}_p10"),
        expr.quantile(0.25).alias(f"{prefix}_p25"),
        expr.quantile(0.75).alias(f"{prefix}_p75"),
        expr.quantile(0.90).alias(f"{prefix}_p90"),
    ]


# ---------------------------------------------------------
# Aggregation: tile → tract‑quarter
# ---------------------------------------------------------


def aggregate_tract_quarter(df: pl.DataFrame) -> pl.DataFrame:
    grouped = df.group_by(["tract", "year", "quarter"]).agg(
        [
            # Central tendency
            pl.col("avg_download_mbps").mean().alias("download_mean"),
            pl.col("avg_download_mbps").median().alias("download_median"),
            pl.col("avg_upload_mbps").mean().alias("upload_mean"),
            pl.col("avg_upload_mbps").median().alias("upload_median"),
            pl.col("avg_latency_ms").mean().alias("latency_mean"),
            pl.col("avg_latency_ms").median().alias("latency_median"),
            # Distribution shape
            pl.col("avg_download_mbps").std().alias("download_std"),
            pl.col("avg_download_mbps").min().alias("download_min"),
            pl.col("avg_download_mbps").max().alias("download_max"),
            pl.col("avg_upload_mbps").std().alias("upload_std"),
            pl.col("avg_upload_mbps").min().alias("upload_min"),
            pl.col("avg_upload_mbps").max().alias("upload_max"),
            pl.col("avg_latency_ms").std().alias("latency_std"),
            pl.col("avg_latency_ms").min().alias("latency_min"),
            pl.col("avg_latency_ms").max().alias("latency_max"),
            # Sampling confidence
            pl.col("tests_count").sum().alias("tests_total"),
            pl.col("devices_count").sum().alias("devices_total"),
            pl.len().alias("tile_count"),
        ]
        + compute_percentiles(pl.col("avg_download_mbps"), "download")
        + compute_percentiles(pl.col("avg_upload_mbps"), "upload")
        + compute_percentiles(pl.col("avg_latency_ms"), "latency")
    )

    return grouped


# ---------------------------------------------------------
# Derived features: heterogeneity + sampling density
# ---------------------------------------------------------


def add_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            # Ranges
            (pl.col("download_max") - pl.col("download_min")).alias("download_range"),
            (pl.col("upload_max") - pl.col("upload_min")).alias("upload_range"),
            (pl.col("latency_max") - pl.col("latency_min")).alias("latency_range"),
            # Coefficients of variation
            (pl.col("download_std") / pl.col("download_mean")).alias("download_cv"),
            (pl.col("upload_std") / pl.col("upload_mean")).alias("upload_cv"),
            (pl.col("latency_std") / pl.col("latency_mean")).alias("latency_cv"),
            # Sampling density
            (pl.col("tests_total") / pl.col("tile_count")).alias("tests_per_tile"),
            (pl.col("devices_total") / pl.col("tile_count")).alias("devices_per_tile"),
        ]
    )


# ---------------------------------------------------------
# Threshold flags
# ---------------------------------------------------------


def add_threshold_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("download_p25") < 25).alias("download_lt_25_flag"),
            (pl.col("download_p25") < 100).alias("download_lt_100_flag"),
            (pl.col("upload_p25") < 3).alias("upload_lt_3_flag"),
            (pl.col("upload_p25") < 20).alias("upload_lt_20_flag"),
            (pl.col("latency_p75") > 100).alias("latency_gt_100_flag"),
            (pl.col("latency_p75") > 150).alias("latency_gt_150_flag"),
            (pl.col("latency_p90") > 200).alias("latency_p90_gt_200_flag"),
        ]
    )


# ---------------------------------------------------------
# Sanitization: replace nulls with safe defaults
# ---------------------------------------------------------


def sanitize_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            # CVs
            pl.col("download_cv").fill_null(0),
            pl.col("upload_cv").fill_null(0),
            pl.col("latency_cv").fill_null(0),
            # Ranges
            pl.col("download_range").fill_null(0),
            pl.col("upload_range").fill_null(0),
            pl.col("latency_range").fill_null(0),
            # Percentiles → median
            pl.col("download_p10").fill_null(pl.col("download_median")),
            pl.col("download_p25").fill_null(pl.col("download_median")),
            pl.col("download_p75").fill_null(pl.col("download_median")),
            pl.col("download_p90").fill_null(pl.col("download_median")),
            pl.col("upload_p10").fill_null(pl.col("upload_median")),
            pl.col("upload_p25").fill_null(pl.col("upload_median")),
            pl.col("upload_p75").fill_null(pl.col("upload_median")),
            pl.col("upload_p90").fill_null(pl.col("upload_median")),
            pl.col("latency_p10").fill_null(pl.col("latency_median")),
            pl.col("latency_p25").fill_null(pl.col("latency_median")),
            pl.col("latency_p75").fill_null(pl.col("latency_median")),
            pl.col("latency_p90").fill_null(pl.col("latency_median")),
            # Flags
            pl.col("download_lt_25_flag").fill_null(False),
            pl.col("download_lt_100_flag").fill_null(False),
            pl.col("upload_lt_3_flag").fill_null(False),
            pl.col("upload_lt_20_flag").fill_null(False),
            pl.col("latency_gt_100_flag").fill_null(False),
            pl.col("latency_gt_150_flag").fill_null(False),
            pl.col("latency_p90_gt_200_flag").fill_null(False),
            # Confidence
            pl.col("tests_per_tile").fill_null(0),
            pl.col("devices_per_tile").fill_null(0),
            pl.col("tests_total").fill_null(0),
        ]
    )


# ---------------------------------------------------------
# Sanitization metadata
# ---------------------------------------------------------


def add_sanitization_metadata(df: pl.DataFrame) -> pl.DataFrame:
    sanitized_cols = [
        "download_cv",
        "upload_cv",
        "latency_cv",
        "download_range",
        "upload_range",
        "latency_range",
        "download_p10",
        "download_p25",
        "download_p75",
        "download_p90",
        "upload_p10",
        "upload_p25",
        "upload_p75",
        "upload_p90",
        "latency_p10",
        "latency_p25",
        "latency_p75",
        "latency_p90",
        "download_lt_25_flag",
        "download_lt_100_flag",
        "upload_lt_3_flag",
        "upload_lt_20_flag",
        "latency_gt_100_flag",
        "latency_gt_150_flag",
        "latency_p90_gt_200_flag",
        "tests_per_tile",
        "devices_per_tile",
        "tests_total",
    ]

    # Create *_was_null columns row-wise
    df = df.with_columns(
        [pl.col(col).is_null().alias(f"{col}_was_null") for col in sanitized_cols]
    )

    # Build expression to sum all *_was_null columns
    null_sum_expr = reduce(
        operator.add, [pl.col(f"{col}_was_null") for col in sanitized_cols]
    )

    # Add num_sanitized_fields
    df = df.with_columns(null_sum_expr.alias("num_sanitized_fields"))

    # Add metadata that depends on num_sanitized_fields
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
# Normalization helpers
# ---------------------------------------------------------


def norm_speed(x):
    return np.clip(x / 200, 0, 1)


def norm_inverse_latency(x):
    return np.clip(1 - (x / 200), 0, 1)


def norm_confidence(tests_total):
    return np.clip(np.log(tests_total + 1) / np.log(500), 0, 1)


# ---------------------------------------------------------
# Reliability Index
# ---------------------------------------------------------


def compute_reliability_index(df: pl.DataFrame) -> pl.DataFrame:

    def calc(row):
        # Component Score (0–1)
        comp = (
            norm_speed(row["download_median"])
            + norm_speed(row["download_p25"])
            + norm_speed(row["download_p10"])
            + norm_speed(row["upload_median"])
            + norm_speed(row["upload_p25"])
            + norm_inverse_latency(row["latency_median"])
            + norm_inverse_latency(row["latency_p75"])
            + norm_inverse_latency(row["latency_p90"])
        ) / 8.0

        # Penalty Score (0–1)
        penalty_raw = (
            (row["download_lt_25_flag"]) ** 2
            + (row["download_lt_100_flag"]) ** 1.5
            + (row["upload_lt_3_flag"]) ** 2
            + (row["upload_lt_20_flag"]) ** 1.5
            + (row["latency_gt_100_flag"]) ** 1.5
            + (row["latency_gt_150_flag"]) ** 2
            + (row["latency_p90_gt_200_flag"])
            + (row["download_cv"]) ** 2
            + (row["upload_cv"]) ** 2
            + (row["latency_cv"]) ** 2
            + (row["download_range"] / 100) ** 2
            + (row["latency_range"] / 100) ** 2
        )

        MAX_PENALTY = 12.0
        penalty = np.clip(penalty_raw / MAX_PENALTY, 0, 1)

        # Confidence Score (0–1)
        conf = norm_confidence(row["tests_total"])

        # Final Score (0–100)
        raw = comp - penalty + conf
        return np.clip(100 * raw, 0, 100)

    return df.with_columns(
        pl.struct(df.columns)
        .map_elements(lambda row: calc(row))
        .alias("reliability_index")
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():
    df = load_ookla()
    grouped = aggregate_tract_quarter(df)
    derived = add_derived_features(grouped)
    thresholds = add_threshold_features(derived)
    sanitized = sanitize_features(thresholds)
    metadata = add_sanitization_metadata(sanitized)
    final = compute_reliability_index(metadata)
    final.write_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    main()
