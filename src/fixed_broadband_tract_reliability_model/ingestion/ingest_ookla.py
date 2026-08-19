import re
from pathlib import Path

import duckdb
import geopandas as gpd
import polars as pl
from shapely.geometry import Point

RAW_DIR = Path("data/raw")
TRACT_DB_PATH = Path("../ml_feature_pipeline/dbt_project/warehouse.duckdb")
OUTPUT_PATH = Path(
    "data/processed/ingestion/ookla_fixed_broadband_service_multi_quarter.parquet"
)

WI_BOUNDS = {
    "lat_min": 42.49,
    "lat_max": 47.08,
    "lon_min": -92.89,
    "lon_max": -86.82,
}


def list_ookla_files() -> list[Path]:
    return sorted(RAW_DIR.glob("*.parquet"))


def parse_year_quarter_from_filename(path: Path) -> tuple[int, str]:
    # Expect filenames like: 2024-07-01_performance_fixed_tiles.parquet
    m = re.match(r"(\d{4})-(\d{2})-\d{2}_", path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {path.name}")

    year = int(m.group(1))
    month = int(m.group(2))

    if month == 1:
        quarter = "Q1"
    elif month == 4:
        quarter = "Q2"
    elif month == 7:
        quarter = "Q3"
    elif month == 10:
        quarter = "Q4"
    else:
        raise ValueError(f"Unexpected quarter month {month} in {path.name}")

    return year, quarter


def load_dim_tract_geometries() -> gpd.GeoDataFrame:
    con = duckdb.connect(str(TRACT_DB_PATH))
    df = con.query("SELECT tract, geometry_wkb FROM dim_tract").df()
    con.close()

    df["geometry_wkb"] = df["geometry_wkb"].apply(bytes)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkb(df["geometry_wkb"]),
        crs="EPSG:4326",
    )
    return gdf


def filter_wisconsin(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("tile_y") >= WI_BOUNDS["lat_min"])
        & (pl.col("tile_y") <= WI_BOUNDS["lat_max"])
        & (pl.col("tile_x") >= WI_BOUNDS["lon_min"])
        & (pl.col("tile_x") <= WI_BOUNDS["lon_max"])
    )


def normalize_and_spatial_join(
    df: pl.DataFrame, gdf_tracts: gpd.GeoDataFrame
) -> pl.DataFrame:
    # Rename tile_x/tile_y BEFORE join
    df = df.rename(
        {
            "tile_x": "centroid_lon",
            "tile_y": "centroid_lat",
            "tile": "tile_polygon",
        }
    )

    # Convert Polars → Pandas → GeoDataFrame with point geometry
    points = gpd.GeoDataFrame(
        df.to_pandas(),
        geometry=[Point(xy) for xy in zip(df["centroid_lon"], df["centroid_lat"])],
        crs="EPSG:4326",
    )

    # Spatial join: assign each tile to a tract polygon
    joined = gpd.sjoin(points, gdf_tracts, how="inner", predicate="within")

    # Drop geometry columns
    joined_no_geom = joined.drop(columns=["geometry", "geometry_wkb"])

    # Convert back to Polars
    df_polars = pl.from_pandas(joined_no_geom)

    # Apply transformations + renames + drop raw fields
    df_polars = df_polars.with_columns(
        [
            # Unit conversions
            (pl.col("avg_d_kbps") / 1000).alias("avg_download_mbps"),
            (pl.col("avg_u_kbps") / 1000).alias("avg_upload_mbps"),
            # Renames (no unit change)
            pl.col("avg_lat_ms").alias("avg_latency_ms"),
            pl.col("avg_lat_down_ms").alias("avg_latency_download_ms"),
            pl.col("avg_lat_up_ms").alias("avg_latency_upload_ms"),
            pl.col("tests").alias("tests_count"),
            pl.col("devices").alias("devices_count"),
        ]
    ).drop(
        [
            # Drop raw fields that were transformed or renamed
            "avg_d_kbps",
            "avg_u_kbps",
            "avg_lat_ms",
            "avg_lat_down_ms",
            "avg_lat_up_ms",
            "tests",
            "devices",
        ]
    )

    return df_polars


def main():
    files = list_ookla_files()
    total = len(files)

    if total == 0:
        raise RuntimeError(f"No Ookla parquet files found in {RAW_DIR}")

    print(f"Found {total} Ookla quarterly parquet files to ingest.\n")

    gdf_tracts = load_dim_tract_geometries()

    dfs: list[pl.DataFrame] = []

    for idx, path in enumerate(files, start=1):
        year, quarter = parse_year_quarter_from_filename(path)
        print(f"[{idx}/{total}] Processing {path.name} ({year} {quarter})...")

        # Load parquet
        df = pl.read_parquet(path)

        # Filter to Wisconsin tiles
        df = filter_wisconsin(df)

        # Normalize + spatial join
        df_norm = normalize_and_spatial_join(df, gdf_tracts)

        # Add year + quarter
        df_norm = df_norm.with_columns(
            [
                pl.lit(year).alias("year"),
                pl.lit(quarter).alias("quarter"),
            ]
        )

        dfs.append(df_norm)

        percent = (idx / total) * 100
        print(f"    → Completed {percent:5.1f}%\n")

    print("Unioning all quarterly datasets...")
    df_all = pl.concat(dfs, how="vertical")

    print(f"Writing unified multi-quarter parquet to {OUTPUT_PATH} ...")
    df_all.write_parquet(OUTPUT_PATH)

    print("Done. Multi-quarter Ookla ingestion complete.")


if __name__ == "__main__":
    main()
