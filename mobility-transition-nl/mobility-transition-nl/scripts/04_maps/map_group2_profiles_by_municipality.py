"""
Group 2 mobility profiles by municipality (WoGem). Raw FactorP-weighted shares are kept; maps show each profile's
percentile rank across municipalities (relative position, not an absolute share). The dominant-profile map uses
raw shares, since percentiles are ranked separately per profile and aren't comparable across profiles.

Inputs:  data/processed/ODiN2022_2023_mobility_transition_respondent_level.csv, data/raw/cbs/BestuurlijkeGebieden_2023.gpkg
Outputs: outputs/maps/group2_municipality_profiles/
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import find_column, header, read_csv_auto, save_csv, save_figure
from mobility_transition.odin import clean_opid
from mobility_transition.profiles import KEYS, SHARE_COLS, area_profile_shares, national_summary, plot_dominant_map
from mobility_transition.settings import MOBILITY_GROUPS, TARGET_CRS

OUT = paths.step_dir(paths.MAPS, "group2_municipality_profiles")
MUNI, WEIGHT, GROUP, AREA = "WoGem", "FactorP", "mobility_group_2", "municipality_code_clean"
CMAP = "viridis"
EDGE = {"linewidth": 0.2, "edgecolor": "white"}
MISSING = {"color": "lightgrey", "label": "No classified ODiN respondents"}
# Thesis labels for titles/legends; the data keep the original group values
DISPLAY = {"Car users": "Car-dependent, low PT use", "Multimodal users": "Multimodal car and PT user", "PT users": "PT user",
           "Infrequent car and PT users": "Low car, low PT use"}
FILE_KEY = {g: k.replace("car_pt", "car_and_pt") for g, k in KEYS.items()}
CODE_CANDIDATES = ["municipality_code_clean", "GM_CODE", "gm_code", "GM_CODE_", "gemeentecode", "Gemeentecode", "GEMEENTECODE",
                   "gemeente_code", "Gemeente_code", "statcode", "STATCODE", "code", "Code", "WoGem", "wogem", "GM2023", "GM2024", "GM2025", "GM2026"]
NAME_CANDIDATES = ["GM_NAAM", "gm_naam", "gemeentenaam", "Gemeentenaam", "GEMEENTENAAM", "gemeente", "Gemeente", "naam", "Naam", "name", "Name"]
RANK_LABELS = ["Very low relative share", "Low relative share", "Middle relative share", "High relative share", "Very high relative share"]


def clean_municipality_code(series):
    """Four-digit municipality code: 363, '0363' and 'GM0363' all become '0363'."""
    return series.astype("string").str.strip().str.extract(r"(\d{1,4})$", expand=False).str.zfill(4)


def percentile_col(group):
    return f"{KEYS[group]}_share_percentile_0_100"


def add_relative_transforms(data, share_col, prefix):
    """Percentile rank (0-100), five relative classes and seven quantile classes of one profile's share."""
    data[share_col] = pd.to_numeric(data[share_col], errors="coerce")
    pct = f"{prefix}_share_percentile_0_100"
    data[pct] = data[share_col].rank(method="average", pct=True) * 100
    data[f"{prefix}_relative_rank_5"] = pd.cut(data[pct], bins=[0, 20, 40, 60, 80, 100], labels=RANK_LABELS, include_lowest=True)
    try:
        data[f"{prefix}_share_quantile_7"] = pd.qcut(data[share_col], q=7, labels=False, duplicates="drop").astype("Int64") + 1
    except ValueError:
        data[f"{prefix}_share_quantile_7"] = pd.Series(pd.NA, index=data.index, dtype="Int64")
    return data


def draw_percentile(ax, gdf, group, legend_label, shrink):
    gdf.plot(ax=ax, column=percentile_col(group), cmap=CMAP, vmin=0, vmax=100, legend=True,
             legend_kwds={"label": legend_label, "shrink": shrink}, missing_kwds=MISSING, **EDGE)
    ax.set_axis_off()


def make_maps(gdf):
    for g in MOBILITY_GROUPS:
        fig, ax = plt.subplots(figsize=(10, 12))
        draw_percentile(ax, gdf, g, "Percentile rank of mobility profile share", 0.7)
        ax.set_title(f"{DISPLAY[g]} by municipality\nPercentile rank of mobility profile share", fontsize=14, pad=20)
        save_figure(fig, OUT / f"map_municipality_{FILE_KEY[g]}_percentile_rank.png")

        fig, ax = plt.subplots(figsize=(10, 12))
        try:
            gdf.plot(ax=ax, column=percentile_col(g), cmap=CMAP, scheme="Quantiles", k=7, legend=True, missing_kwds=MISSING, **EDGE)
            ax.set_axis_off()
        except ImportError:
            print("mapclassify is not installed; falling back to the continuous percentile map.")
            draw_percentile(ax, gdf, g, "Percentile rank of mobility profile share", 0.7)
        ax.set_title(f"{DISPLAY[g]} by municipality\nRelative weighted profile share, quantile classes", fontsize=14, pad=20)
        save_figure(fig, OUT / f"map_municipality_{FILE_KEY[g]}_percentile_quantiles.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 18))
    for ax, g in zip(axes.flatten(), MOBILITY_GROUPS):
        gdf.plot(ax=ax, column=percentile_col(g), cmap=CMAP, vmin=0, vmax=100, legend=True,
                 legend_kwds={"label": "Percentile rank", "shrink": 0.65},
                 missing_kwds={"color": "lightgrey", "label": "No classified respondents"}, **EDGE)
        ax.set_title(DISPLAY[g], fontsize=14, pad=12)
        ax.set_axis_off()
    fig.suptitle("Relative mobility profile distribution by municipality\nPercentile rank of mobility profile share", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, OUT / "_municipality_all_profiles_percentile_side_by_side.png")

    plot_dominant_map(gdf, "Dominant mobility profile by municipality\nBased on raw FactorP-weighted profile shares",
                      OUT / "map_group_2_municipality_dominant_mobility_profile.png", labels=DISPLAY, edge=True)


def main():
    paths.require(paths.RESPONDENT_LEVEL_CSV, paths.MUNICIPALITY_GPKG)
    header("GROUP 2 MOBILITY PROFILE DISTRIBUTION BY MUNICIPALITY", 80)

    respondents = read_csv_auto(paths.RESPONDENT_LEVEL_CSV)
    missing = [c for c in ["OPID", MUNI, WEIGHT, GROUP] if c not in respondents.columns]
    if missing:
        raise ValueError(f"Respondent-level ODiN dataset is missing required columns: {missing}")

    respondents["OPID"] = clean_opid(respondents["OPID"])
    n_missing = int(respondents["OPID"].isna().sum())
    dup_rows = int(respondents.loc[respondents["OPID"].notna(), "OPID"].duplicated(keep=False).sum())
    print(f"Respondent rows: {len(respondents):,}; unique OPIDs: {respondents['OPID'].nunique(dropna=True):,}; "
          f"missing: {n_missing:,}; duplicated: {dup_rows:,}")
    if n_missing:
        raise ValueError("Missing OPIDs in the respondent-level file.")
    if dup_rows:
        examples = respondents.loc[respondents["OPID"].duplicated(keep=False), ["OPID", MUNI, WEIGHT, GROUP]].head(30)
        raise ValueError(f"Duplicate OPIDs in the respondent-level file; run create_respondent_level_odin.py first.\n\n{examples.to_string(index=False)}")

    respondents[AREA] = clean_municipality_code(respondents[MUNI])
    respondents[WEIGHT] = pd.to_numeric(respondents[WEIGHT], errors="coerce")
    respondents[GROUP] = respondents[GROUP].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})

    valid = {"profile": respondents[GROUP].isin(MOBILITY_GROUPS), "municipality": respondents[AREA].notna(),
             "weight": respondents[WEIGHT].notna() & respondents[WEIGHT].gt(0)}
    analysis = respondents.loc[valid["profile"] & valid["municipality"] & valid["weight"], ["OPID", AREA, WEIGHT, GROUP]].copy()
    print(f"Group 2 municipality mapping sample: {len(analysis):,}")
    if analysis.empty:
        raise ValueError("No respondents remain in the Group 2 municipality mapping sample.")
    profile_counts = analysis[GROUP].value_counts().reindex(MOBILITY_GROUPS, fill_value=0)
    print(f"\n{profile_counts.to_string()}")

    profiles, deviation = area_profile_shares(analysis, AREA)
    for g in MOBILITY_GROUPS:
        profiles = add_relative_transforms(profiles, SHARE_COLS[g], KEYS[g])
        print(f"{g}: raw share {profiles[SHARE_COLS[g]].min():.3f}%-{profiles[SHARE_COLS[g]].max():.3f}%, "
              f"percentile {profiles[percentile_col(g)].min():.3f}-{profiles[percentile_col(g)].max():.3f}")
    # Dominant profile and HHI come from raw shares (computed in area_profile_shares); move them after the transforms as before
    profiles = profiles[[c for c in profiles.columns if c not in ("dominant_profile_share_pct", "dominant_mobility_profile", "mobility_profile_hhi")]
                        + ["dominant_profile_share_pct", "dominant_mobility_profile", "mobility_profile_hhi"]]
    save_csv(profiles, OUT / "municipality_group_2_weighted_mobility_profile_distribution.csv")

    summary = pd.DataFrame(national_summary(analysis, profiles, "municipality", "municipalities"))
    summary["minimum_profile_percentile_rank"] = [profiles[percentile_col(g)].min() for g in MOBILITY_GROUPS]
    summary["maximum_profile_percentile_rank"] = [profiles[percentile_col(g)].max() for g in MOBILITY_GROUPS]
    save_csv(summary, OUT / "group_2_municipality_mobility_profile_summary.csv")
    print(f"\n{summary.round(3).to_string(index=False)}")

    header("MUNICIPALITY GEOMETRY", 80)
    munis = gpd.read_file(paths.MUNICIPALITY_GPKG)
    code_col = find_column(munis, CODE_CANDIDATES, "municipality code column in geometry")
    name_col = find_column(munis, NAME_CANDIDATES, "municipality name column", required=False)
    print(f"Code column: {code_col}" + (f"; name column: {name_col}" if name_col else ""))
    munis[AREA] = clean_municipality_code(munis[code_col])
    munis = (munis.set_crs(TARGET_CRS) if munis.crs is None else munis).to_crs(TARGET_CRS)

    if munis[AREA].isna().any():
        raise ValueError("Missing municipality codes in the geometry.")
    if munis[AREA].duplicated().any():
        examples = munis.loc[munis[AREA].duplicated(keep=False), [code_col, AREA]].head(30)
        raise ValueError(f"Duplicate municipality codes in the geometry; stopped to avoid duplicating values.\n\n{examples.to_string(index=False)}")

    odin_codes, geom_codes = set(analysis[AREA].dropna()), set(munis[AREA].dropna())
    unmatched_odin, unmatched_geom = sorted(odin_codes - geom_codes), sorted(geom_codes - odin_codes)
    print(f"ODiN codes: {len(odin_codes):,}; geometry codes: {len(geom_codes):,}; ODiN codes not in geometry: {len(unmatched_odin):,}; "
          f"geometry codes not in ODiN: {len(unmatched_geom):,}")
    if unmatched_odin:
        print(f"First unmatched ODiN codes: {unmatched_odin[:30]}")

    muni_map = munis.merge(profiles, on=AREA, how="left", validate="one_to_one")
    mapped = int(muni_map["weighted_classified_population"].notna().sum())
    print(f"Municipalities: {len(muni_map):,}; with classified respondents: {mapped:,}")
    print("Note: a percentile of 90 means a higher profile share than ~90% of mapped municipalities, not that 90% belong to the profile.")
    muni_map.to_file(OUT / "municipality_group_2_weighted_mobility_profile_distribution.gpkg", driver="GPKG")

    audit = pd.DataFrame(
        [("Total unique respondent rows", len(respondents)), ("Unique OPIDs in respondent file", respondents["OPID"].nunique()),
         ("Duplicate OPID rows in respondent file", dup_rows),
         ("Respondents with valid Group 2 mobility profile", int(valid["profile"].sum())),
         ("Respondents with valid WoGem municipality", int(valid["municipality"].sum())),
         ("Respondents with valid positive FactorP", int(valid["weight"].sum())),
         ("Final unique OPID municipality mapping sample", len(analysis)), ("Municipalities with classified respondents", len(profiles)),
         ("Total FactorP weight in mapping sample", analysis[WEIGHT].sum()),
         ("ODiN municipality codes absent from geometry", len(unmatched_odin)), ("Geometry municipality codes absent from ODiN", len(unmatched_geom)),
         ("Mapped municipality polygons", mapped), ("Unmapped municipality polygons", len(muni_map) - mapped),
         ("Maximum absolute profile share sum deviation from 100", deviation)]
        + [(f"Unique respondents - {DISPLAY[g]}", int(profile_counts.get(g, 0))) for g in MOBILITY_GROUPS],
        columns=["metric", "value"],
    )
    save_csv(audit, OUT / "group_2_municipality_mapping_audit.csv")

    make_maps(muni_map)
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
