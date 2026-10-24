"""
FactorP-weighted share of each Group 2 mobility profile per residential PC4, with single-profile, combined and
dominant-profile maps. Each respondent counts once; PC4s without classified respondents stay missing (grey), not 0.

Inputs:  data/processed/ODiN2022_2023_mobility_transition_master_with_mobility_groups.csv, data/raw/cbs/cbs_pc4_2023_v2.gpkg
Outputs: data/processed/group2_pc4_unique_opid_respondent_sample.csv (used by 03_analysis/pc4_sample_support_diagnostics.py)
         outputs/maps/group2_pc4_profiles/
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import clean_pc4, find_column, header, read_csv_auto, save_csv, save_figure
from mobility_transition.odin import clean_opid
from mobility_transition.profiles import KEYS, SHARE_COLS, area_profile_shares, national_summary, plot_dominant_map
from mobility_transition.settings import MOBILITY_GROUPS, TARGET_CRS

OUT = paths.step_dir(paths.MAPS, "group2_pc4_profiles")
PC4_ODIN, WEIGHT, GROUP = "WoPC", "FactorP", "mobility_group_2"
CMAP = "viridis"
MAP_FILES = {g: f"map_pc4_{k.replace('car_pt', 'car_and_pt')}_weighted_share.png" for g, k in KEYS.items()}
GEOM_PC4_CANDIDATES = ["pc4_clean", "pc4", "PC4", "postcode", "Postcode", "Postal code", "postcode4", "Postcode4", "statcode", "gebiedscode", "code"]


def unique_respondents(odin):
    """One row per OPID; stops if PC4, FactorP or profile differ within an OPID."""
    source = odin.loc[odin["OPID"].notna(), ["OPID", "pc4_clean", WEIGHT, GROUP]]
    counts = source.groupby("OPID", sort=False).agg(**{f"n_{c}": (c, "nunique") for c in ["pc4_clean", WEIGHT, GROUP]})
    conflicts = {label: int(counts[f"n_{c}"].gt(1).sum()) for label, c in [("residential PC4", "pc4_clean"), ("FactorP", WEIGHT), ("Group 2 profile", GROUP)]}
    for label, n in conflicts.items():
        print(f"OPIDs with conflicting {label}: {n:,}")
    bad = counts.index[(counts > 1).any(axis=1)]
    if len(bad):
        examples = source[source["OPID"].isin(bad)].sort_values("OPID").head(50)
        raise ValueError("Conflicting respondent-level values within OPIDs; keeping the first row could assign the wrong PC4, "
                         f"FactorP or profile.\n{conflicts}\n\n{examples.to_string(index=False)}")

    first = lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA
    respondents = source.groupby("OPID", as_index=False, sort=False).agg(pc4_clean=("pc4_clean", first), **{WEIGHT: (WEIGHT, first), GROUP: (GROUP, first)})
    return respondents, conflicts


def draw_share(ax, gdf, column, legend_label, missing_label, shrink):
    gdf.plot(ax=ax, column=column, cmap=CMAP, vmin=0, vmax=100, linewidth=0, legend=True,
             legend_kwds={"label": legend_label, "shrink": shrink}, missing_kwds={"color": "lightgrey", "label": missing_label})
    ax.set_axis_off()


def make_maps(pc4_map):
    for g in MOBILITY_GROUPS:
        fig, ax = plt.subplots(figsize=(10, 12))
        draw_share(ax, pc4_map, SHARE_COLS[g], "Weighted share within PC4 (%)", "No classified ODiN respondents", 0.7)
        ax.set_title(f"{g} by PC4\nFactorP-weighted share of unique respondents", fontsize=14, pad=20)
        save_figure(fig, OUT / MAP_FILES[g])

    fig, axes = plt.subplots(2, 2, figsize=(16, 18))
    for ax, g in zip(axes.flatten(), MOBILITY_GROUPS):
        draw_share(ax, pc4_map, SHARE_COLS[g], "Weighted share (%)", "No classified respondents", 0.65)
        ax.set_title(g, fontsize=14, pad=12)
    fig.suptitle("Mobility profile distribution by residential PC4\nFactorP-weighted shares of unique respondents", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, OUT / "map_group_2_pc4_all_mobility_profiles.png")

    plot_dominant_map(pc4_map, "Dominant mobility profile by residential PC4\nBased on FactorP-weighted unique respondents",
                      OUT / "map_group_2_pc4_dominant_mobility_profile.png")


def main():
    paths.require(paths.MASTER_WITH_GROUPS_CSV, paths.PC4_POLYGONS_2023_GPKG)
    header("GROUP 2 MOBILITY PROFILE DISTRIBUTION BY PC4", 80)

    odin = read_csv_auto(paths.MASTER_WITH_GROUPS_CSV)
    missing = [c for c in ["OPID", PC4_ODIN, WEIGHT, GROUP] if c not in odin.columns]
    if missing:
        raise ValueError(f"ODiN file is missing required columns: {missing}")
    odin["OPID"] = clean_opid(odin["OPID"])
    odin["pc4_clean"] = clean_pc4(odin[PC4_ODIN])
    odin[WEIGHT] = pd.to_numeric(odin[WEIGHT], errors="coerce")
    odin[GROUP] = odin[GROUP].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})

    n_raw, n_missing_opid = len(odin), int(odin["OPID"].isna().sum())
    n_unique = odin["OPID"].nunique(dropna=True)
    dup_rows = int(odin.loc[odin["OPID"].notna(), "OPID"].duplicated(keep=False).sum())
    repeated = odin["OPID"].notna().sum() - n_unique
    print(f"Raw rows: {n_raw:,}; missing OPID: {n_missing_opid:,}; unique OPIDs: {n_unique:,}; repeated rows collapsed: {repeated:,}")

    respondents, conflicts = unique_respondents(odin)
    if respondents["OPID"].duplicated().any() or len(respondents) != n_unique:
        raise ValueError("Respondent collapse did not produce exactly one row per unique OPID.")

    valid = {"profile": respondents[GROUP].isin(MOBILITY_GROUPS), "pc4": respondents["pc4_clean"].notna(),
             "weight": respondents[WEIGHT].notna() & respondents[WEIGHT].gt(0)}
    analysis = respondents.loc[valid["profile"] & valid["pc4"] & valid["weight"], ["OPID", "pc4_clean", WEIGHT, GROUP]].copy()
    print(f"Group 2 PC4 mapping sample: {len(analysis):,} of {len(respondents):,} respondents")
    if analysis.empty:
        raise ValueError("No respondents remain in the Group 2 PC4 mapping sample.")
    save_csv(analysis, paths.GROUP2_PC4_RESPONDENTS_CSV)

    profile_counts = analysis[GROUP].value_counts().reindex(MOBILITY_GROUPS, fill_value=0)
    print(f"\n{profile_counts.to_string()}")

    pc4_profiles, deviation = area_profile_shares(analysis, "pc4_clean")
    save_csv(pc4_profiles, OUT / "pc4_group_2_weighted_mobility_profile_distribution.csv")

    total_w = analysis[WEIGHT].sum()
    summary = pd.DataFrame(national_summary(analysis, pc4_profiles, "pc4", "pc4"))
    save_csv(summary, OUT / "group_2_pc4_mobility_profile_summary.csv")
    print(f"\n{summary.round(3).to_string(index=False)}")

    audit = pd.DataFrame(
        [("Raw ODiN rows", n_raw), ("Rows with missing OPID", n_missing_opid), ("Unique nonmissing OPIDs", n_unique),
         ("Rows belonging to duplicated OPIDs", dup_rows), ("Repeated nonmissing OPID rows removed", repeated)]
        + [(f"OPIDs with conflicting {k}", v) for k, v in conflicts.items()]
        + [("Unique respondent rows after OPID collapse", len(respondents)),
           ("Respondents with valid Group 2 mobility profile", int(valid["profile"].sum())),
           ("Respondents with valid residential PC4", int(valid["pc4"].sum())), ("Respondents with valid positive FactorP", int(valid["weight"].sum())),
           ("Final unique OPID Group 2 PC4 mapping sample", len(analysis)), ("Unique OPIDs in final mapping sample", analysis["OPID"].nunique()),
           ("PC4s with classified respondents", len(pc4_profiles)), ("Total FactorP weight in mapping sample", total_w),
           ("Maximum absolute profile share sum deviation from 100", deviation)]
        + [(f"Unique respondents - {g}", int(profile_counts.get(g, 0))) for g in MOBILITY_GROUPS],
        columns=["metric", "value"],
    )
    save_csv(audit, OUT / "group_2_pc4_mapping_audit.csv")

    pc4 = gpd.read_file(paths.PC4_POLYGONS_2023_GPKG)
    pc4["pc4_clean"] = clean_pc4(pc4[find_column(pc4, GEOM_PC4_CANDIDATES, "PC4 column in geometry")])
    pc4 = (pc4.set_crs(TARGET_CRS) if pc4.crs is None else pc4).to_crs(TARGET_CRS)
    if pc4["pc4_clean"].duplicated().any():
        raise ValueError("Duplicate PC4 identifiers in the geometry file; stopped to avoid duplicating mapped values.")

    pc4_map = pc4.merge(pc4_profiles, on="pc4_clean", how="left", validate="one_to_one")
    print(f"\nPC4 polygons: {len(pc4_map):,}; with classified respondents: {pc4_map['weighted_classified_population'].notna().sum():,}")
    pc4_map.to_file(OUT / "pc4_group_2_weighted_mobility_profile_distribution.gpkg", driver="GPKG")
    make_maps(pc4_map)
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
