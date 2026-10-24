"""
Attaches the threshold mobility groups to every ODiN trip row (left join on OPID, rows and OPIDs never change).
Group labels are renamed to the thesis labels; mobility_threshold_sample flags respondents in the threshold sample.

Inputs:  data/processed/ODiN2022_2023_merged_common_columns_only.csv, threshold_analysis_sample_with_groups.csv
Outputs: data/processed/ODiN2022_2023_mobility_transition_master_with_mobility_groups.csv
         outputs/tables/threshold_groups/07_mobility_group_merge_audit.csv
"""
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import header, read_csv_auto, save_csv
from mobility_transition.odin import clean_opid
from mobility_transition.settings import MOBILITY_GROUPS, THRESHOLD_GROUP_RENAME

GROUP_COLS = ["mobility_group_1", "mobility_group_2"]


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns:\n{missing}")


def main():
    paths.require(paths.ODIN_MERGED_CSV, paths.THRESHOLD_GROUPS_CSV)
    header("MERGING MOBILITY GROUPS INTO ODiN", 78)

    odin = read_csv_auto(paths.ODIN_MERGED_CSV)
    groups = read_csv_auto(paths.THRESHOLD_GROUPS_CSV)
    require_columns(odin, ["OPID"], "ODiN file")
    require_columns(groups, ["OPID", "mobility_group_main", "mobility_group_sensitivity"], "Threshold classification file")
    odin["OPID"], groups["OPID"] = clean_opid(odin["OPID"]), clean_opid(groups["OPID"])

    header("OPID DIAGNOSTICS", 78)
    for label, df in [("ODiN", odin), ("Classification", groups)]:
        ids = df["OPID"]
        print(f"{label}: {len(df):,} rows, {ids.nunique(dropna=True):,} unique OPIDs, {int(ids.isna().sum()):,} missing, "
              f"{int(ids.dropna().duplicated(keep=False).sum()):,} rows with duplicated OPID")

    if groups["OPID"].isna().any():
        print("WARNING: classification rows with missing OPID are removed.")
        groups = groups[groups["OPID"].notna()].copy()
    if groups["OPID"].duplicated().any():
        examples = groups.loc[groups["OPID"].duplicated(keep=False), ["OPID", "mobility_group_main", "mobility_group_sensitivity"]].sort_values("OPID").head(20)
        raise ValueError(f"The classification file contains duplicate OPIDs; merge stopped.\n\n{examples.to_string(index=False)}")

    lookup = groups[["OPID", "mobility_group_main", "mobility_group_sensitivity"]].rename(
        columns={"mobility_group_main": "mobility_group_1", "mobility_group_sensitivity": "mobility_group_2"})
    for col in GROUP_COLS:
        lookup[col] = lookup[col].replace(THRESHOLD_GROUP_RENAME)
        unexpected = sorted(set(lookup[col].dropna()) - set(MOBILITY_GROUPS))
        if unexpected:
            print(f"WARNING: unexpected {col} labels: {unexpected}")

    existing = [c for c in GROUP_COLS + ["mobility_threshold_sample"] if c in odin.columns]
    if existing:
        print(f"Replacing existing columns: {existing}")
        odin = odin.drop(columns=existing)

    n_rows, n_opids = len(odin), odin["OPID"].nunique(dropna=True)
    merged = odin.merge(lookup, on="OPID", how="left", validate="many_to_one", indicator="_merge")
    if len(merged) != n_rows or merged["OPID"].nunique(dropna=True) != n_opids:
        raise ValueError("Row or OPID count changed during the merge.")
    matched = merged["_merge"].eq("both")
    merged["mobility_threshold_sample"] = matched.astype(int)

    matched_opids = merged.loc[matched, "OPID"].nunique(dropna=True)
    unmatched_opids = merged.loc[~matched, "OPID"].nunique(dropna=True)
    row_rate = 100 * matched.sum() / len(merged) if len(merged) else np.nan
    resp_rate = 100 * matched_opids / n_opids if n_opids else np.nan
    print(f"\nMatched rows: {int(matched.sum()):,} ({row_rate:.2f}%); matched respondents: {matched_opids:,} ({resp_rate:.2f}%)")

    respondents = merged[["OPID", *GROUP_COLS]].drop_duplicates(subset=["OPID"])
    distributions = {col: respondents[col].dropna().value_counts().reindex(MOBILITY_GROUPS, fill_value=0) for col in GROUP_COLS}
    for col, dist in distributions.items():
        print(f"\n{col} respondent distribution:\n{dist.to_string()}")
    crosswalk = pd.crosstab(respondents["mobility_group_1"], respondents["mobility_group_2"], dropna=False).reindex(
        index=MOBILITY_GROUPS, columns=MOBILITY_GROUPS, fill_value=0)
    print(f"\nGroup 1 vs Group 2 crosswalk:\n{crosswalk.to_string()}")

    audit = pd.DataFrame(
        [("ODiN rows before merge", n_rows), ("ODiN rows after merge", len(merged)), ("ODiN unique nonmissing OPIDs", n_opids),
         ("Classification rows", len(lookup)), ("Classification unique OPIDs", lookup["OPID"].nunique(dropna=True)),
         ("Matched ODiN rows", int(matched.sum())), ("Unmatched ODiN rows", int((~matched).sum())),
         ("Row classification match rate percent", row_rate), ("Matched respondents", matched_opids),
         ("Unmatched respondents", unmatched_opids), ("Respondent classification match rate percent", resp_rate)]
        + [(f"Group {i} respondents - {g}", int(distributions[col].get(g, 0))) for i, col in enumerate(GROUP_COLS, 1) for g in MOBILITY_GROUPS],
        columns=["metric", "value"],
    )

    save_csv(merged.drop(columns="_merge"), paths.MASTER_WITH_GROUPS_CSV)
    save_csv(audit, paths.THRESHOLD_TABLES / "07_mobility_group_merge_audit.csv")
    print(f"\nSaved: {paths.MASTER_WITH_GROUPS_CSV}\nNew columns: {GROUP_COLS + ['mobility_threshold_sample']}")


if __name__ == "__main__":
    main()
