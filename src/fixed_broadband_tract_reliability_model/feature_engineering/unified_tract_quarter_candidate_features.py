from pathlib import Path

import polars as pl

LCDV2_PATH = Path(
    "data/processed/feature_engineering/lcdv2_tract_quarter_features.parquet"
)
TRACT_PATH = Path("data/processed/feature_engineering/tract_features.parquet")
OOKLA_PATH = Path(
    "data/processed/feature_engineering/ookla_tract_quarter_target_features.parquet"
)

OUTPUT_PATH = Path("data/processed/feature_engineering/tract_quarter_features.parquet")


def load_lcdv2() -> pl.DataFrame:
    return pl.read_parquet(LCDV2_PATH)


def load_dim_tract() -> pl.DataFrame:
    return pl.read_parquet(TRACT_PATH)


def load_ookla_target() -> pl.DataFrame:
    # Only keep keys + reliability_index
    df = pl.read_parquet(OOKLA_PATH)
    return df.select(
        [
            "tract",
            "year",
            "quarter",
            "reliability_index",
            "tests_total",
            "devices_total",
            "tile_count",
            "tests_per_tile",
            "devices_per_tile",
        ]
    )


def assemble_unified_dataset(
    lcdv2: pl.DataFrame, dim_tract: pl.DataFrame, ookla: pl.DataFrame
) -> pl.DataFrame:

    # Join LCDv2 + Ookla on tract-year-quarter
    df = lcdv2.join(ookla, on=["tract", "year", "quarter"], how="left")

    # Join dim_tract on tract only
    df = df.join(dim_tract, on="tract", how="left")

    # Keep only tracts with non-zero population
    df = df.filter(pl.col("population_total") > 0)

    # Keep only rows with a valid reliability_index
    df = df.filter(pl.col("reliability_index").is_not_null())

    return df


def main():
    print("[1/4] Loading LCDv2 tract-quarter features...")
    lcdv2 = load_lcdv2()

    print("[2/4] Loading dim_tract features...")
    dim_tract = load_dim_tract()

    print("[3/4] Loading Ookla target features...")
    ookla = load_ookla_target()

    print("[4/4] Assembling unified tract-quarter dataset...")
    df = assemble_unified_dataset(lcdv2, dim_tract, ookla)

    print("Writing output parquet...")
    df.write_parquet(OUTPUT_PATH)

    print("Unified tract-quarter feature matrix complete.")


if __name__ == "__main__":
    main()
