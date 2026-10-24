"""
Adds respondent/household variables and PC4 sustainability-accessibility measures (direct train, 10 km EV
charging, P85 BTM feeder-to-train, urbanity) to the Group 2 respondent file.

The respondent file is an immutable spine: every merge is a left join and the script stops if the row count,
OPID set or Group 2 distribution changes. Zero accessibility stays zero; unmatched PC4s stay missing.

Inputs:  data/processed/ODiN2022_2023_mobility_transition_respondent_level.csv (spine)
         data/processed/ODiN2022_2023_pt.csv (lookup for missing respondent variables only)
         data/processed/pc4_pt_accessibility_distance_decay.csv, pc4_ev_accessibility.csv,
         pc4_btm_feeder_train_accessibility.csv, data/raw/cbs/pc4_stats.csv
Outputs: data/processed/group2_respondents_with_sustainability_accessibility.csv
         outputs/tables/group2_merge/ (audits, scaling, PC4 lookup, missingness)
"""
import warnings

import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import clean_pc4, detect_csv_format, header, read_csv_auto, save_csv
from mobility_transition.settings import MOBILITY_GROUPS

TABLES = paths.step_dir(paths.TABLES, "group2_merge")

ID, GROUP, RES_PC4 = "OPID", "mobility_group_2", "WoPC"
TRAIN_COL = "pt_access_decay_train"
EV_COL = "ev_access_capacity_decay_strict_10km"
FEEDER_COL = "train_feeder_p85_access_decay"
URBANITY = "urbanity"

REQUIRED_VARS = ["FactorP", "Leeftijd", "Geslacht", "HHGestInkG", "HHSam", "BetWerk", "Jaar", "HHAuto", "BrandstofPa1", "WoPC"]
# Exact ODiN names first; aliases only if the exact field is absent
ALIASES = {
    "FactorP": ["FactorP", "factorP", "factor_p", "WegfactorP", "weegfactorP", "PersonWeight", "person_weight"],
    "HHSam": ["HHSam", "HHSamenstelling"],
    "Jaar": ["Jaar", "Year", "year", "JAAR"],
}
PC4_CANDIDATES = ["pc4_clean", "PC4", "pc4", "postcode4", "Postcode4", "Postal code", "postcode", "Postcode", "VertPC"]


# ---------------- helpers ----------------

def clean_identifier(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def first_existing(columns, candidates, required=True):
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns:\n{candidates}\n\nAvailable columns:\n{list(columns)}")
    return None


def first_nonmissing(series):
    valid = series.dropna()
    return np.nan if valid.empty else valid.iloc[0]


def mode_or_first(series):
    valid = series.dropna()
    if valid.empty:
        return np.nan
    modes = valid.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else valid.iloc[0]


def standardise_log1p(data, raw, out, label, positive_only=False):
    """
    log1p, then z-score across unique PC4s.
    positive_only: scale among PC4s with access only; zero-access PC4s get 0 and need a separate presence dummy.
    """
    values = pd.to_numeric(data[raw], errors="coerce")
    if (values.dropna() < 0).any():
        raise ValueError(f"{raw} contains negative accessibility values.")
    data[raw] = values
    logs = np.log1p(values)
    mask = values.gt(0) if positive_only else values.notna()
    mean, sd = logs[mask].mean(), logs[mask].std(ddof=0)
    if not mask.any() or not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Cannot standardise {raw}; usable PC4 values have no variance.")

    if positive_only:
        data[out] = np.nan
        data.loc[values.eq(0), out] = 0.0
        data.loc[mask, out] = (logs[mask] - mean) / sd
        transformation = "log1p; standardised among positive-access PC4s; zero-access PC4s assigned intensity 0 and modelled with presence indicator"
    else:
        data[out] = (logs - mean) / sd
        transformation = "log1p; standardised across all nonmissing unique PC4s"

    return data, {
        "measure": label, "raw_column": raw, "model_column": out, "transformation": transformation,
        "pc4_log_mean": mean, "pc4_log_sd": sd, "n_pc4_nonmissing": int(values.notna().sum()),
        "n_pc4_zero": int(values.eq(0).sum()), "n_pc4_positive": int(values.gt(0).sum()),
    }


def assert_spine_preserved(before, after, stage):
    if len(before) != len(after):
        raise ValueError(f"Respondent row count changed at '{stage}': {len(before):,} -> {len(after):,}")
    b, a = set(before[ID].dropna().astype(str)), set(after[ID].dropna().astype(str))
    if b != a:
        raise ValueError(f"OPID set changed at '{stage}'.\nLost: {sorted(b - a)[:20]}\nAdded: {sorted(a - b)[:20]}")


def load_pc4_table(path, label, dedupe="warn"):
    df = read_csv_auto(path)
    df["pc4_clean"] = clean_pc4(df[first_existing(df.columns.tolist(), PC4_CANDIDATES + ["WoPC"])])
    df = df.dropna(subset=["pc4_clean"]).copy()
    n_dup = int(df["pc4_clean"].duplicated().sum())
    if n_dup:
        if dedupe == "error":
            raise ValueError(f"The {label} file must have one row per PC4; duplicates: {n_dup:,}")
        warnings.warn(f"{label} contains {n_dup:,} duplicate PC4 rows; keeping the first per PC4.")
        df = df.drop_duplicates("pc4_clean", keep="first").copy()
    return df


# ---------------- respondent variables ----------------

def build_person_lookup(target_opids):
    """One row per OPID from the leg-level ODiN file (first non-missing value; modal home PC4)."""
    encoding, sep = detect_csv_format(paths.ODIN_TRIPS_CSV)
    columns = pd.read_csv(paths.ODIN_TRIPS_CSV, encoding=encoding, sep=sep, nrows=0).columns.astype(str).str.strip().tolist()
    resolved = {ID: first_existing(columns, [ID, "opid"])}
    for var in REQUIRED_VARS:
        actual = first_existing(columns, ALIASES.get(var, [var]), required=False)
        if actual is not None:
            resolved[var] = actual

    source = read_csv_auto(paths.ODIN_TRIPS_CSV, usecols=list(dict.fromkeys(resolved.values())))
    source = source.rename(columns={a: s for s, a in resolved.items() if a != s})
    source[ID] = clean_identifier(source[ID])
    source = source[source[ID].notna() & source[ID].isin(target_opids)].copy()
    print(f"ODiN source rows belonging to Group 2 respondents: {len(source):,}")

    numeric = [c for c in REQUIRED_VARS if c in source.columns]
    source[numeric] = source[numeric].apply(pd.to_numeric, errors="coerce")

    conflicts = []
    for var in numeric:
        n = source.groupby(ID, sort=False)[var].nunique(dropna=True)
        conflicts.append({"source": "odin_pt_profile_source", "variable": var,
                          "respondents_with_multiple_nonmissing_values": int(n.gt(1).sum()),
                          "maximum_distinct_values_within_respondent": int(n.max()) if len(n) else 0})

    agg = {c: first_nonmissing for c in numeric if c != RES_PC4}
    person = source.groupby(ID, as_index=False, sort=False).agg(agg) if agg else source[[ID]].drop_duplicates(ID)
    if RES_PC4 in source.columns:
        person = person.merge(source.groupby(ID, sort=False)[RES_PC4].agg(mode_or_first).reset_index(),
                              on=ID, how="outer", validate="one_to_one")

    variable_audit = pd.DataFrame([{
        "source": "odin_pt_profile_source", "standard_variable": var, "source_column_used": resolved.get(var, "NOT FOUND"),
        "present_in_source": int(var in resolved), "n_person_nonmissing": int(person[var].notna().sum()) if var in person.columns else 0,
    } for var in REQUIRED_VARS])
    return person, variable_audit, pd.DataFrame(conflicts)


def fill_from_lookup(spine, lookup):
    """Fill only values missing in the spine; never adds or removes respondents."""
    before = spine.copy()
    cols = [c for c in REQUIRED_VARS if c in lookup.columns]
    temp = {c: f"__src_{c}" for c in cols}
    spine = spine.merge(lookup[[ID] + cols].rename(columns=temp), on=ID, how="left", validate="one_to_one")

    rows = []
    for var in REQUIRED_VARS:
        if var not in temp:
            n = int(spine[var].isna().sum()) if var in spine.columns else len(spine)
            rows.append({"source": "odin_pt_profile_source", "variable": var, "n_missing_before": n, "n_filled_from_source": 0, "n_missing_after": n})
            continue
        if var not in spine.columns:
            spine[var] = np.nan
        missing = spine[var].isna()
        fillable = missing & spine[temp[var]].notna()
        spine.loc[fillable, var] = spine.loc[fillable, temp[var]]
        rows.append({"source": "odin_pt_profile_source", "variable": var, "n_missing_before": int(missing.sum()),
                     "n_filled_from_source": int(fillable.sum()), "n_missing_after": int(spine[var].isna().sum())})
        spine = spine.drop(columns=temp[var])

    assert_spine_preserved(before, spine, "fill respondent variables from ODiN source")
    return spine, pd.DataFrame(rows)


# ---------------- PC4 lookups ----------------

def build_train_lookup():
    pt = load_pc4_table(paths.PT_ACCESSIBILITY_CSV, "PT accessibility")
    if TRAIN_COL not in pt.columns:
        if "decay_access_rail" not in pt.columns:
            raise ValueError(f"The PT accessibility file has neither '{TRAIN_COL}' nor 'decay_access_rail'.")
        pt[TRAIN_COL] = pt["decay_access_rail"]
    pt[TRAIN_COL] = pd.to_numeric(pt[TRAIN_COL], errors="coerce")
    pt[TRAIN_COL] = pt[TRAIN_COL].where(pt[TRAIN_COL] >= 0)
    pt["has_direct_train_access"] = np.where(pt[TRAIN_COL].isna(), np.nan, pt[TRAIN_COL].gt(0).astype(float))
    pt, scaling = standardise_log1p(pt, TRAIN_COL, "train_positive_intensity_z", "Direct train accessibility intensity", positive_only=True)
    return pt[["pc4_clean", TRAIN_COL, "has_direct_train_access", "train_positive_intensity_z"]].copy(), scaling


def build_measure_lookup(path, col, z_col, label, dedupe="warn"):
    df = load_pc4_table(path, label, dedupe)
    if col not in df.columns:
        raise ValueError(f"{label} file is missing '{col}'.\nAvailable columns:\n{df.columns.tolist()}")
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].where(df[col] >= 0)
    df, scaling = standardise_log1p(df, col, z_col, label)
    return df[["pc4_clean", col, z_col]].copy(), scaling


def build_urbanity_lookup():
    stats = read_csv_auto(paths.PC4_STATS_CSV)
    if URBANITY not in stats.columns:
        raise ValueError(f"PC4 statistics file is missing '{URBANITY}'.")
    stats["pc4_clean"] = clean_pc4(stats[first_existing(stats.columns.tolist(), PC4_CANDIDATES)])
    stats[URBANITY] = pd.to_numeric(stats[URBANITY], errors="coerce")
    stats = stats.dropna(subset=["pc4_clean"]).copy()
    stats[URBANITY] = stats[URBANITY].where(stats[URBANITY].isin([1, 2, 3, 4, 5]))
    if stats["pc4_clean"].duplicated().any():
        return stats.groupby("pc4_clean", as_index=False)[URBANITY].agg(mode_or_first)
    return stats[["pc4_clean", URBANITY]].copy()


def merge_flagged(respondents, lookup, flag, stage):
    before = respondents.copy()
    out = respondents.merge(lookup, on="pc4_clean", how="left", validate="many_to_one", indicator="__merge")
    out[flag] = out["__merge"].eq("both").astype(int)
    out = out.drop(columns="__merge")
    assert_spine_preserved(before, out, stage)
    return out


# ---------------- main ----------------

def main():
    paths.require(paths.RESPONDENT_LEVEL_CSV, paths.ODIN_TRIPS_CSV, paths.PT_ACCESSIBILITY_CSV,
                  paths.EV_ACCESSIBILITY_CSV, paths.FEEDER_ACCESSIBILITY_CSV, paths.PC4_STATS_CSV)
    header("GROUP 2 RESPONDENT MERGE WITH PC4 SUSTAINABILITY ACCESSIBILITY", 84)

    respondents = read_csv_auto(paths.RESPONDENT_LEVEL_CSV)
    missing = [c for c in [ID, GROUP] if c not in respondents.columns]
    if missing:
        raise ValueError(f"Respondent file is missing: {missing}")
    respondents[ID] = clean_identifier(respondents[ID])
    if respondents[ID].isna().any():
        raise ValueError("The respondent spine contains missing OPIDs.")
    if respondents[ID].duplicated().any():
        examples = respondents.loc[respondents[ID].duplicated(keep=False), [ID, GROUP]].sort_values(ID).head(20)
        raise ValueError(f"The respondent spine has duplicate OPIDs; it must be one row per respondent.\n\n{examples.to_string(index=False)}")

    spine = respondents[[ID, GROUP]].copy()
    n_spine = len(respondents)
    initial_dist = respondents[GROUP].value_counts(dropna=False).reindex(MOBILITY_GROUPS)
    print(f"\nRespondent spine: {n_spine:,}\n{initial_dist.to_string()}")

    # Respondent/household variables: fill gaps from the ODiN source used to build the groups
    lookup, variable_audit, conflicts = build_person_lookup(set(respondents[ID].dropna().astype("string")))
    respondents, fill_audit = fill_from_lookup(respondents, lookup)
    for c in [c for c in REQUIRED_VARS if c in respondents.columns]:
        respondents[c] = pd.to_numeric(respondents[c], errors="coerce")

    # Income decile -> quintile (same rule as the master builder); only fills missing quintiles
    derived = np.ceil(respondents["HHGestInkG"].where(respondents["HHGestInkG"].between(1, 10)) / 2).astype("Int64")
    if "income_quintile" not in respondents.columns:
        respondents["income_quintile"] = derived
    else:
        respondents["income_quintile"] = pd.to_numeric(respondents["income_quintile"], errors="coerce")
        gap = respondents["income_quintile"].isna()
        respondents.loc[gap, "income_quintile"] = derived.loc[gap]

    if RES_PC4 not in respondents.columns:
        raise ValueError(f"'{RES_PC4}' is absent from both the respondent file and the ODiN source; no residential PC4.")
    respondents["pc4_clean"] = clean_pc4(respondents[RES_PC4])

    train, train_scaling = build_train_lookup()
    ev, ev_scaling = build_measure_lookup(paths.EV_ACCESSIBILITY_CSV, EV_COL, "ev_access_z", "Strict EV charging accessibility within 10 km")
    feeder, feeder_scaling = build_measure_lookup(paths.FEEDER_ACCESSIBILITY_CSV, FEEDER_COL, "train_feeder_p85_access_z",
                                                  "P85 BTM feeder-to-train accessibility", dedupe="error")
    urbanity = build_urbanity_lookup()
    pc4_lookup = (train.merge(ev, on="pc4_clean", how="outer", validate="one_to_one")
                  .merge(feeder, on="pc4_clean", how="outer", validate="one_to_one")
                  .merge(urbanity, on="pc4_clean", how="outer", validate="one_to_one"))

    # PC4 source files are authoritative: drop any older versions of these columns
    spatial = [TRAIN_COL, "has_direct_train_access", "train_positive_intensity_z", EV_COL, "ev_access_z", FEEDER_COL, "train_feeder_p85_access_z", URBANITY]
    respondents = respondents.drop(columns=[c for c in spatial if c in respondents.columns])
    for table, flag, stage in [(train, "matched_train_accessibility", "train accessibility merge"),
                               (ev, "matched_ev_accessibility", "EV accessibility merge"),
                               (feeder, "matched_p85_feeder_accessibility", "P85 feeder accessibility merge"),
                               (urbanity, "matched_pc4_urbanity", "PC4 urbanity merge")]:
        respondents = merge_flagged(respondents, table, flag, stage)

    assert_spine_preserved(spine, respondents, "final merged output")
    final_dist = respondents[GROUP].value_counts(dropna=False).reindex(MOBILITY_GROUPS)
    if not initial_dist.fillna(0).equals(final_dist.fillna(0)):
        raise ValueError("Mobility Group 2 distribution changed during the merge.")

    flags = ["matched_train_accessibility", "matched_ev_accessibility", "matched_p85_feeder_accessibility", "matched_pc4_urbanity"]
    audit = pd.DataFrame(
        [("Mobility-group respondent spine rows", n_spine), ("Final merged respondent rows", len(respondents)),
         ("Respondents lost during merge", n_spine - len(respondents)),
         ("Unique OPIDs in respondent spine", spine[ID].nunique(dropna=True)), ("Unique OPIDs in final output", respondents[ID].nunique(dropna=True)),
         ("Respondents with valid WoPC-derived PC4", int(respondents["pc4_clean"].notna().sum())),
         ("Respondents matched to direct train accessibility", int(respondents[flags[0]].sum())),
         ("Respondents matched to 10 km EV accessibility", int(respondents[flags[1]].sum())),
         ("Respondents matched to P85 feeder accessibility", int(respondents[flags[2]].sum())),
         ("Respondents matched to PC4 urbanity", int(respondents[flags[3]].sum())),
         ("Respondents with zero direct train accessibility", int(respondents[TRAIN_COL].eq(0).sum())),
         ("Respondents missing direct train accessibility", int(respondents[TRAIN_COL].isna().sum())),
         ("Respondents missing 10 km EV accessibility", int(respondents[EV_COL].isna().sum())),
         ("Respondents missing P85 feeder accessibility", int(respondents[FEEDER_COL].isna().sum())),
         ("EV accessibility file used", str(paths.EV_ACCESSIBILITY_CSV))]
        + [row for g in MOBILITY_GROUPS for row in [(f"Group 2 respondents before merge - {g}", int(initial_dist.get(g, 0))),
                                                    (f"Group 2 respondents after merge - {g}", int(final_dist.get(g, 0)))]],
        columns=["metric", "value"],
    )

    regression_vars = [ID, GROUP, "FactorP", "Leeftijd", "Geslacht", "HHGestInkG", "income_quintile", "HHSam", "BetWerk", "Jaar",
                       "HHAuto", "BrandstofPa1", "WoPC", "pc4_clean", *spatial]
    n = len(respondents)
    missingness = pd.DataFrame([
        {"variable": c, "present_in_output": int(c in respondents.columns), "n_total": n,
         "n_missing": int(respondents[c].isna().sum()) if c in respondents.columns else n,
         "percent_missing": (100 * respondents[c].isna().sum() / n if n else np.nan) if c in respondents.columns else 100.0}
        for c in regression_vars
    ])

    by_pc4 = respondents.groupby("pc4_clean", dropna=False).agg(n_respondents=(ID, "size"), **{f: (f, "max") for f in flags}).reset_index()
    unmatched = by_pc4[by_pc4["pc4_clean"].isna() | np.logical_or.reduce([by_pc4[f].eq(0) for f in flags])].sort_values("n_respondents", ascending=False)

    save_csv(respondents, paths.GROUP2_ANALYSIS_CSV)
    save_csv(audit, TABLES / "01_group2_direct_source_merge_audit.csv")
    save_csv(pd.concat([variable_audit, fill_audit], ignore_index=True, sort=False), TABLES / "02_group2_odin_source_variable_audit.csv")
    save_csv(conflicts if not conflicts.empty else pd.DataFrame(columns=["source", "variable", "respondents_with_multiple_nonmissing_values",
                                                                          "maximum_distinct_values_within_respondent"]),
             TABLES / "03_group2_odin_person_level_conflicts.csv")
    save_csv(pd.DataFrame([train_scaling, ev_scaling, feeder_scaling]), TABLES / "04_sustainability_accessibility_scaling_information.csv")
    save_csv(pc4_lookup, TABLES / "05_pc4_sustainability_accessibility_lookup.csv")
    save_csv(missingness, TABLES / "06_group2_regression_variable_missingness.csv")
    save_csv(unmatched, TABLES / "07_group2_unmatched_pc4.csv")

    header("GROUP 2 MERGE COMPLETE", 84)
    print(f"Respondents: {n_spine:,} -> {len(respondents):,} (lost: {n_spine - len(respondents):,})")
    print(f"\nGroup 2 distribution after merges:\n{final_dist.to_string()}")
    print(f"\nPC4 matches: " + ", ".join(f"{f.replace('matched_', '')} {int(respondents[f].sum()):,}" for f in flags))
    print(f"\nSaved: {paths.GROUP2_ANALYSIS_CSV}\nAudits: {TABLES}")


if __name__ == "__main__":
    main()
