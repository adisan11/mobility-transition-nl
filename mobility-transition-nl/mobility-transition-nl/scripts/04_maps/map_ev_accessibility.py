"""
National PC4 maps of EV charging accessibility (percentile index and quintile classes).

Inputs:  data/processed/pc4_ev_accessibility_enriched.csv  (from 03_analysis/ev_accessibility_inequality.py)
         data/raw/cbs/cbs_pc4_2024_v1.gpkg
Outputs: outputs/maps/ev_accessibility/
"""
import geopandas as gpd
import matplotlib.pyplot as plt

from mobility_transition import paths
from mobility_transition.io import clean_pc4, find_column, header, read_csv_auto, save_figure
from mobility_transition.settings import PC4_CANDIDATES

ACCESS_COL = "ev_access_capacity_decay_strict_10km"
PERCENTILE_COL = f"{ACCESS_COL}_percentile_0_100"
QUINTILE_COL = f"{ACCESS_COL}_quintile_class"

OUT = paths.step_dir(paths.MAPS, "ev_accessibility")


def main():
    paths.require(paths.EV_ACCESSIBILITY_ENRICHED_CSV, paths.PC4_POLYGONS_2024_GPKG)

    header("CREATING NL PC4 EV ACCESSIBILITY MAPS")
    ev = read_csv_auto(paths.EV_ACCESSIBILITY_ENRICHED_CSV)
    ev["pc4_clean"] = clean_pc4(ev["pc4_clean"])

    polygons = gpd.read_file(paths.PC4_POLYGONS_2024_GPKG)
    if polygons.empty:
        raise ValueError("PC4 polygon file contains no features.")
    polygons.columns = polygons.columns.str.strip()
    polygons["pc4_clean"] = clean_pc4(polygons[find_column(polygons, PC4_CANDIDATES, "PC4 column in polygons")])
    print("Warning: polygon file has no CRS defined." if polygons.crs is None else f"Polygon CRS: {polygons.crs}")

    map_gdf = polygons.merge(ev[["pc4_clean", ACCESS_COL, PERCENTILE_COL, QUINTILE_COL]], on="pc4_clean", how="left")
    map_gdf.to_file(OUT / "nl_pc4_ev_accessibility_joined.gpkg", layer="pc4_ev_accessibility", driver="GPKG")

    for column, title, file_name, categorical in [
        (PERCENTILE_COL, "EV charging accessibility percentile-normalised index by PC4", "nl_pc4_ev_accessibility_percentile_map.png", False),
        (QUINTILE_COL, "EV charging accessibility quintile classes by PC4", "nl_pc4_ev_accessibility_quintile_map.png", True),
    ]:
        fig, ax = plt.subplots(figsize=(10, 12))
        map_gdf.plot(column=column, ax=ax, legend=True, linewidth=0, categorical=categorical, missing_kwds={"label": "No data"})
        ax.set_title(title)
        ax.set_axis_off()
        save_figure(fig, OUT / file_name)

    print(f"Maps saved to: {OUT}")


if __name__ == "__main__":
    main()
