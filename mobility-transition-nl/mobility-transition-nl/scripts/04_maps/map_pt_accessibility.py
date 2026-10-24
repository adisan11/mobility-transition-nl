"""
National PC4 maps of BTM and train accessibility (percentile-normalised; continuous and quantile-classed).
Missing PC4s are grey; zero accessibility is kept as a real zero.

Inputs:  data/processed/pc4_pt_accessibility_distance_decay.csv, data/raw/cbs/cbs_pc4_2023_v2.gpkg
Outputs: outputs/maps/pt_accessibility/
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import clean_pc4, find_column, header, read_csv_auto, save_csv, save_figure
from mobility_transition.settings import TARGET_CRS

OUT = paths.step_dir(paths.MAPS, "pt_accessibility")
RAW_COLS = {"btm": "pt_access_decay_btm", "train": "pt_access_decay_train"}
MAP_COLS = {"btm": "btm_access_percentile_0_100", "train": "train_access_percentile_0_100"}  # alt: *_access_log_norm_0_100
TITLES = {"btm": "BTM", "train": "Train"}
CMAP = "viridis"
MISSING = {"color": "lightgrey", "label": "Missing"}
PC4_CANDIDATES = ["pc4_clean", "pc4", "PC4", "postcode", "Postcode", "Postal code", "postcode4", "Postcode4",
                  "statcode", "gebiedscode", "code", "VertPC"]


def scale_to_100(values):
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo) * 100 if pd.notna(lo) and pd.notna(hi) and hi > lo else np.nan


def add_transforms(gdf, raw_col, prefix):
    """Log, log-normalised, percentile, winsorised (P95) and quintile-class versions of the index."""
    gdf[raw_col] = pd.to_numeric(gdf[raw_col], errors="coerce")
    gdf[f"{prefix}_access_log"] = np.log1p(gdf[raw_col])
    gdf[f"{prefix}_access_log_norm_0_100"] = scale_to_100(gdf[f"{prefix}_access_log"])
    gdf[f"{prefix}_access_percentile_0_100"] = gdf[raw_col].rank(method="average", pct=True) * 100
    gdf[f"{prefix}_access_winsorised"] = gdf[raw_col].clip(upper=gdf[raw_col].quantile(0.95))
    gdf[f"{prefix}_access_winsor_norm_0_100"] = scale_to_100(gdf[f"{prefix}_access_winsorised"])
    try:
        gdf[f"{prefix}_access_quantile_5"] = pd.qcut(gdf[raw_col], q=5, labels=["Very low", "Low", "Medium", "High", "Very high"], duplicates="drop")
    except ValueError:
        gdf[f"{prefix}_access_quantile_5"] = "No class"
    return gdf


def summary_row(series, name):
    v = pd.to_numeric(series, errors="coerce").dropna()
    stats = ["n_zero", "percent_zero", "mean", "median", "p25", "p75", "p90", "maximum"]
    if v.empty:
        return {"measure": name, "n_pc4": 0, **{k: np.nan for k in stats}}
    return {"measure": name, "n_pc4": len(v), "n_zero": int((v == 0).sum()), "percent_zero": 100 * (v == 0).mean(),
            "mean": v.mean(), "median": v.median(), "p25": v.quantile(0.25), "p75": v.quantile(0.75),
            "p90": v.quantile(0.90), "maximum": v.max()}


def draw(ax, gdf, column, title, quantiles=False):
    kwargs = dict(ax=ax, column=column, cmap=CMAP, linewidth=0, legend=True, missing_kwds=MISSING)
    if quantiles:
        try:
            gdf.plot(scheme="Quantiles", k=7, **kwargs)
        except ImportError:
            print("mapclassify is not installed; falling back to a continuous map.")
            gdf.plot(**kwargs)
    else:
        gdf.plot(**kwargs)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_axis_off()


def single_map(gdf, column, title, file_name, quantiles=False):
    fig, ax = plt.subplots(figsize=(10, 12))
    draw(ax, gdf, column, title, quantiles)
    save_figure(fig, OUT / file_name)
    print(f"Saved: {OUT / file_name}")


def main():
    paths.require(paths.PT_ACCESSIBILITY_CSV, paths.PC4_POLYGONS_2023_GPKG)
    header("PC4 PT ACCESSIBILITY MAPS")

    pc4 = gpd.read_file(paths.PC4_POLYGONS_2023_GPKG)
    pc4["pc4_clean"] = clean_pc4(pc4[find_column(pc4, PC4_CANDIDATES, "PC4 column in polygons")])
    if pc4.crs is None:
        print(f"PC4 geometry has no CRS; assuming {TARGET_CRS}.")
        pc4 = pc4.set_crs(TARGET_CRS)
    pc4 = pc4.to_crs(TARGET_CRS)

    access = read_csv_auto(paths.PT_ACCESSIBILITY_CSV)
    access["pc4_clean"] = clean_pc4(access[find_column(access, PC4_CANDIDATES, "PC4 column in accessibility file")])
    missing = [c for c in RAW_COLS.values() if c not in access.columns]
    if missing:
        raise ValueError(f"Missing accessibility columns: {missing}")
    access[list(RAW_COLS.values())] = access[list(RAW_COLS.values())].apply(pd.to_numeric, errors="coerce")

    pc4_map = pc4.merge(access[["pc4_clean", *RAW_COLS.values()]].drop_duplicates("pc4_clean"), on="pc4_clean", how="left")
    for prefix, col in RAW_COLS.items():
        print(f"PC4 polygons with {TITLES[prefix]} value: {pc4_map[col].notna().sum():,}; missing: {pc4_map[col].isna().sum():,}")
        pc4_map = add_transforms(pc4_map, col, prefix)

    summary = pd.DataFrame([summary_row(pc4_map[col], f"{TITLES[p]} accessibility") for p, col in RAW_COLS.items()])
    save_csv(summary, OUT / "pc4_btm_train_accessibility_map_summary.csv")
    print(f"\n{summary.to_string(index=False)}")
    pc4_map.to_file(OUT / "pc4_btm_train_accessibility_mapped.gpkg", driver="GPKG")

    subtitle = "Percentile-normalised distance-decay index"
    for prefix in RAW_COLS:
        single_map(pc4_map, MAP_COLS[prefix], f"{TITLES[prefix]} accessibility by PC4\n{subtitle}",
                   f"map_pc4_{prefix}_accessibility_percentile_nl.png")
        single_map(pc4_map, MAP_COLS[prefix], f"{TITLES[prefix]} accessibility by PC4\n{subtitle}, quantile classes",
                   f"map_pc4_{prefix}_accessibility_quantiles_nl.png", quantiles=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 11))
    for ax, prefix in zip(axes, RAW_COLS):
        draw(ax, pc4_map, MAP_COLS[prefix], f"{TITLES[prefix]} accessibility by PC4\n{subtitle}")
        ax.title.set_fontsize(13)
    save_figure(fig, OUT / "map_pc4_btm_train_accessibility_percentile_side_by_side.png")
    print(f"Saved side-by-side map to {OUT}")


if __name__ == "__main__":
    main()
