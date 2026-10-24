"""
Population-weighted centroid for every PC4 area, built from CBS 100 m population grid cells.

Inputs:  data/raw/cbs/cbs_vk100_2023.gpkg, pc4_stats.csv, cbs_pc4_2023_v2.gpkg
Outputs: data/processed/pc4_population_weighted_centroids.{csv,gpkg}
         data/processed/pc4_stats_with_population_weighted_centroids.{csv,gpkg}
         outputs/figures/pc4_centroids/  (visual check of one random PC4)
"""
import random
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import clean_pc4, find_column, header, read_csv_auto, save_csv, save_figure
from mobility_transition.settings import RANDOM_SEED, TARGET_CRS

POP_COL = "aantal_inwoners"
PC4_STATS_CANDIDATES = ["Postal code", "postcode", "pc4", "PC4", "Postcode", "postcode4", "Postcode4", "VertPC"]
PC4_GEOM_CANDIDATES = [
    "postcode", "pc4", "PC4", "Postcode", "postcode4", "Postcode4", "Postal code", "statcode", "gebiedscode", "code",
]


def to_target_crs(gdf, label):
    if gdf.crs is None:
        warnings.warn(f"{label} has no CRS; assuming {TARGET_CRS}. Check results if they look spatially wrong.")
        gdf = gdf.set_crs(TARGET_CRS)
    return gdf.to_crs(TARGET_CRS)


def population_weighted_centroids(joined):
    """x = sum(x_i * pop_i) / sum(pop_i), same for y, one row per PC4."""
    df = joined.assign(wx=joined["x"] * joined[POP_COL], wy=joined["y"] * joined[POP_COL])
    c = df.groupby("pc4_clean").agg(
        wx=("wx", "sum"), wy=("wy", "sum"), population_represented=(POP_COL, "sum"), n_vk_cells=(POP_COL, "size"),
    ).reset_index()
    c["x_weighted"] = c["wx"] / c["population_represented"]
    c["y_weighted"] = c["wy"] / c["population_represented"]
    c["method"] = "population_weighted_centroid"  # only populated cells are used, so no unweighted fallback is needed
    c["centroid_id"] = 1
    return c[["pc4_clean", "x_weighted", "y_weighted", "population_represented", "n_vk_cells", "method", "centroid_id"]]


def plot_random_pc4(vk, pc4, centroids_gdf, available_pc4s, out_dir):
    """Population heatmap plus weighted centroid for one random PC4, as a sanity check."""
    pc4_code = random.Random(RANDOM_SEED).choice(sorted(available_pc4s))
    print(f"PC4 selected for visual check: {pc4_code}")

    boundary = pc4[pc4["pc4_clean"] == pc4_code]
    centroid = centroids_gdf[centroids_gdf["pc4_clean"] == pc4_code]
    cells = gpd.sjoin(vk, boundary[["geometry"]], how="inner", predicate="intersects").drop(columns="index_right")
    cells = gpd.clip(cells, boundary)

    fig, ax = plt.subplots(figsize=(8, 8))
    cells.plot(ax=ax, column=POP_COL, cmap="YlOrRd", legend=True, edgecolor="grey", linewidth=0.1, alpha=0.85)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)
    centroid.plot(ax=ax, color="blue", markersize=140, marker="X", label="Population-weighted centroid")
    ax.set_title(f"Population distribution and weighted centroid in PC4 {pc4_code}", fontsize=13)
    ax.set_axis_off()
    ax.legend()

    out_path = out_dir / f"population_heatmap_weighted_centroid_PC4_{pc4_code}.png"
    save_figure(fig, out_path)
    print(f"Saved: {out_path}")


def main():
    paths.require(paths.VK100_GPKG, paths.PC4_STATS_CSV, paths.PC4_POLYGONS_2023_GPKG)

    header("LOADING VK100 POPULATION GRID")
    vk = gpd.read_file(paths.VK100_GPKG)
    print(f"Rows loaded: {len(vk):,}")
    if POP_COL not in vk.columns:
        raise ValueError(f"Column '{POP_COL}' not found in VK100 data.")
    vk[POP_COL] = pd.to_numeric(vk[POP_COL], errors="coerce")
    vk = to_target_crs(vk.loc[vk[POP_COL] > 0, [POP_COL, "geometry"]].copy(), "VK100 grid")
    print(f"Populated cells retained: {len(vk):,}")

    header("LOADING PC4 STATS AND POLYGONS")
    pc4_stats = read_csv_auto(paths.PC4_STATS_CSV)
    stats_col = find_column(pc4_stats, PC4_STATS_CANDIDATES, "PC4 column in stats CSV")
    pc4_stats["pc4_clean"] = clean_pc4(pc4_stats[stats_col])

    pc4_geom = gpd.read_file(paths.PC4_POLYGONS_2023_GPKG)
    geom_col = find_column(pc4_geom, PC4_GEOM_CANDIDATES, "PC4 column in geometry file")
    pc4_geom["pc4_clean"] = clean_pc4(pc4_geom[geom_col])
    pc4_geom = to_target_crs(pc4_geom, "PC4 polygons")

    pc4 = pc4_geom.merge(pc4_stats.drop(columns=[stats_col], errors="ignore"), on="pc4_clean", how="left")
    print(f"PC4 polygons: {len(pc4):,} (stats column: {stats_col}, geometry column: {geom_col})")
    if "Number of inhabitants" in pc4.columns:
        print(f"PC4s with matched stats: {pc4['Number of inhabitants'].notna().sum():,}")

    header("ASSIGNING GRID POPULATION TO PC4")
    # Each 100 m cell's population is placed at the cell's centroid
    points = vk.copy()
    points["geometry"] = points.geometry.centroid
    points["x"], points["y"] = points.geometry.x, points.geometry.y

    joined = gpd.sjoin(points, pc4[["pc4_clean", "geometry"]], how="inner", predicate="within")
    print(f"Grid cells joined to a PC4: {len(joined):,}")
    if joined.empty:
        raise ValueError("Spatial join found no matches; the CRS is probably wrong or the PC4 geometry was misread.")

    centroids = population_weighted_centroids(joined)
    centroids_gdf = gpd.GeoDataFrame(
        centroids, geometry=gpd.points_from_xy(centroids["x_weighted"], centroids["y_weighted"]), crs=TARGET_CRS
    )

    header("EXPORTING")
    save_csv(centroids_gdf.assign(geometry_wkt=centroids_gdf.geometry.to_wkt()).drop(columns="geometry"), paths.PC4_CENTROIDS_CSV)
    centroids_gdf.to_file(paths.PC4_CENTROIDS_GPKG, driver="GPKG")

    pc4_with_centroids = pc4.merge(centroids, on="pc4_clean", how="left")
    save_csv(
        pc4_with_centroids.assign(geometry_wkt=pc4_with_centroids.geometry.to_wkt()).drop(columns="geometry"),
        paths.PC4_WITH_CENTROIDS_CSV,
    )
    pc4_with_centroids.to_file(paths.PC4_WITH_CENTROIDS_GPKG, driver="GPKG")
    for f in [paths.PC4_CENTROIDS_CSV, paths.PC4_CENTROIDS_GPKG, paths.PC4_WITH_CENTROIDS_CSV, paths.PC4_WITH_CENTROIDS_GPKG]:
        print(f"Saved: {f}")

    plot_random_pc4(vk, pc4, centroids_gdf, joined["pc4_clean"].dropna().unique(), paths.step_dir(paths.FIGURES, "pc4_centroids"))

    header("SUMMARY")
    print(f"PC4 polygons: {len(pc4):,}")
    print(f"PC4s with weighted centroids: {len(centroids_gdf):,}")
    print(f"Populated grid cells used: {len(joined):,}")


if __name__ == "__main__":
    main()
