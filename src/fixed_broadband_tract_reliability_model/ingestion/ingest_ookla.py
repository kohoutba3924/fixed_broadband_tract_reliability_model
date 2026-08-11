from pathlib import Path

import duckdb
import geopandas as gpd
import polars as pl
from shapely.geometry import Point

OOKLA_INPUT_PATH = Path("data/raw/2024-10-01_performance_fixed_tiles.parquet")
TRACT_PATH = Path("../ml_feature_pipeline/dbt_project/warehouse.duckdb")
OUTPUT_PATH = Path("data/processed/ookla_fixed_broadband_service.parquet")


WI_BOUNDS = {
    "lat_min": 42.49,
    "lat_max": 47.08,
    "lon_min": -92.89,
    "lon_max": -86.82,
}


def load_ookla_lazy():
    return pl.scan_parquet(OOKLA_INPUT_PATH)


def filter_wisconsin(df_lazy):
    return df_lazy.filter(
        (pl.col("tile_y") >= WI_BOUNDS["lat_min"])
        & (pl.col("tile_y") <= WI_BOUNDS["lat_max"])
        & (pl.col("tile_x") >= WI_BOUNDS["lon_min"])
        & (pl.col("tile_x") <= WI_BOUNDS["lon_max"])
    )


def load_dim_tract_geometries():
    con = duckdb.connect(str(TRACT_PATH))
    df = con.query("SELECT tract, geometry_wkb FROM dim_tract").df()
    con.close()

    df["geometry_wkb"] = df["geometry_wkb"].apply(bytes)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkb(df["geometry_wkb"]),
        crs="EPSG:4326",
    )

    return gdf


def spatial_join(df_filtered: pl.DataFrame, gdf_tracts: gpd.GeoDataFrame):
    points = gpd.GeoDataFrame(
        df_filtered.to_pandas(),
        geometry=[
            Point(xy) for xy in zip(df_filtered["tile_x"], df_filtered["tile_y"])
        ],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(points, gdf_tracts, how="inner", predicate="within")

    # Drop both geometries: point geometry + tract geometry_wkb
    joined_no_geom = joined.drop(columns=["geometry", "geometry_wkb"])

    return pl.from_pandas(joined_no_geom)


def main():
    df_lazy = load_ookla_lazy()
    df_filtered = filter_wisconsin(df_lazy).collect()
    gdf_tracts = load_dim_tract_geometries()
    df_joined = spatial_join(df_filtered, gdf_tracts)

    df_joined.write_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    main()
