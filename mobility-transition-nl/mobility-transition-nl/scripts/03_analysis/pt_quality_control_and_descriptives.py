"""
Quality control of the master dataset and first PT descriptives: integrity checks, raw code distributions,
PC4 accessibility summaries, PT use by accessibility quintile and accessibility by urbanity.

Inputs:  data/processed/ODiN2022_2023_mobility_transition_master.csv, pc4_pt_accessibility_distance_decay.csv,
         mobility_transition_unmatched_pc4.csv (optional)
Outputs: outputs/tables/pt_descriptives/, outputs/figures/pt_descriptives/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import clean_pc4, header, read_csv_auto, save_csv, save_figure
from mobility_transition.stats import weighted_mean, weighted_share

TABLES = paths.step_dir(paths.TABLES, "pt_descriptives")
FIGURES = paths.step_dir(paths.FIGURES, "pt_descriptives")

WEIGHT_CANDIDATES = ["person_weight", "FactorP", "factorP", "WegfactorP"]
KEY_CODE_VARIABLES = [
    "FqBTM", "btm_frequency_ordered", "FqTrein", "train_frequency_ordered", "FqAutoB", "Geslacht", "Leeftijd",
    "HHGestInkG", "income_quintile", "OPRijbewijsAu", "HHAuto", "BrandstofPa1", "urbanity",
]
# Common survey non-response codes; flagged so they can be set to missing before modelling
POSSIBLE_SPECIAL_CODES = {-99999, -99998, -99997, -9999, -999, -99, -9, -8, -7, -1, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999}
ACCESS_COLS = {"BTM": "pt_access_decay_btm", "Train": "pt_access_decay_train"}


def first_existing(columns, candidates):
    lookup = {c.lower(): c for c in columns}
    return next((lookup[c.lower()] for c in candidates if c.lower() in lookup), None)


def weighted_frequency_table(df, variable, weight_col):
    subset = df[[variable, weight_col]].apply(pd.to_numeric, errors="coerce").dropna()
    subset = subset[subset[weight_col] > 0]
    result = pd.concat(
        [subset.groupby(variable).size().rename("n_unweighted"), subset.groupby(variable)[weight_col].sum().rename("weighted_population")],
        axis=1,
    ).reset_index()
    result["unweighted_percent"] = result["n_unweighted"] / result["n_unweighted"].sum() * 100
    result["weighted_percent"] = result["weighted_population"] / result["weighted_population"].sum() * 100
    return result


def accessibility_summary(series, name):
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    row = {"measure": name, "n_pc4": len(valid), "n_missing": int(values.isna().sum())}
    if valid.empty:
        return {**row, **{k: np.nan for k in ["n_zero", "percent_zero", "mean", "std", "minimum", "p10", "p25", "median", "p75", "p90", "maximum"]}}
    return {
        **row,
        "n_zero": int(valid.eq(0).sum()), "percent_zero": 100 * valid.eq(0).mean(), "mean": valid.mean(), "std": valid.std(),
        "minimum": valid.min(), "p10": valid.quantile(0.10), "p25": valid.quantile(0.25), "median": valid.median(),
        "p75": valid.quantile(0.75), "p90": valid.quantile(0.90), "maximum": valid.max(),
    }


def grouped_use_summary(df, quintile_col, frequency_col, weekly_col, weight_col):
    missing = sorted({quintile_col, frequency_col, weekly_col, weight_col} - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for grouped PT-use summary: {missing}")

    quintiles = pd.to_numeric(df[quintile_col], errors="coerce")
    rows = []
    for q in sorted(quintiles.dropna().unique()):
        g = df[quintiles.eq(q)]
        w = pd.to_numeric(g[weight_col], errors="coerce")
        rows.append({
            "accessibility_quintile": int(q),
            "n_unweighted": len(g),
            "weighted_population": w.where(w > 0).sum(),
            "unweighted_mean_frequency": pd.to_numeric(g[frequency_col], errors="coerce").mean(),
            "weighted_mean_frequency": weighted_mean(g[frequency_col], w),
            "unweighted_weekly_or_more_percent": 100 * pd.to_numeric(g[weekly_col], errors="coerce").mean(),
            "weighted_weekly_or_more_percent": 100 * weighted_share(g[weekly_col], w),
        })
    return pd.DataFrame(rows)


def line_plot(data, y_col, title, xlabel, ylabel, file_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data["accessibility_quintile"], data[y_col], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(data["accessibility_quintile"].dropna().unique()))
    ax.grid(True, alpha=0.3)
    save_figure(fig, FIGURES / file_name)


def boxplot_by_urbanity(data, value_col, title, ylabel, file_name):
    valid = data[["urbanity", value_col]].apply(pd.to_numeric, errors="coerce").dropna()
    groups = sorted(valid["urbanity"].unique())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [valid.loc[valid["urbanity"].eq(g), value_col].values for g in groups],
        tick_labels=[str(int(g)) if float(g).is_integer() else str(g) for g in groups], showfliers=False,
    )
    ax.set_title(title)
    ax.set_xlabel("Urbanity category (1 = most urban, 5 = least urban)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FIGURES / file_name)


def integrity_audit(master):
    rows = [("Master rows", len(master))]
    if "respondent_id" in master.columns:
        rows += [("Unique respondent_id", master["respondent_id"].nunique(dropna=True)),
                 ("Duplicate respondent_id", master["respondent_id"].duplicated().sum())]
    if "OPID" in master.columns:
        rows.append(("Unique OPID", master["OPID"].nunique(dropna=True)))
    if "n_odin_source_rows" in master.columns:
        src = pd.to_numeric(master["n_odin_source_rows"], errors="coerce")
        rows += [("Respondents with one source row", int(src.eq(1).sum())),
                 ("Respondents with more than one source row", int(src.gt(1).sum())),
                 ("Maximum source rows for one respondent", src.max())]
    if "pc4_clean" in master.columns:
        rows.append(("Respondents with valid PC4", master["pc4_clean"].notna().sum()))
    for flag, label in [("matched_pt_accessibility", "Respondents matched to PT accessibility"),
                        ("matched_pc4_urbanity", "Respondents matched to urbanity")]:
        if flag in master.columns:
            rows.append((label, master[flag].eq(1).sum()))
    for col in ACCESS_COLS.values():
        if col in master.columns:
            v = pd.to_numeric(master[col], errors="coerce")
            rows += [(f"Negative values in {col}", int(v.lt(0).sum())), (f"Zero values in {col}", int(v.eq(0).sum())),
                     (f"Missing values in {col}", int(v.isna().sum()))]
    return pd.DataFrame(rows, columns=["metric", "value"])


def code_distributions(master):
    rows = []
    for variable in [v for v in KEY_CODE_VARIABLES if v in master.columns]:
        for value, count in master[variable].astype("string").fillna("<MISSING>").value_counts(dropna=False).items():
            num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            rows.append({
                "variable": variable, "value": value, "n": int(count), "percent": 100 * count / len(master),
                "possible_special_missing_code": bool(pd.notna(num) and (num in POSSIBLE_SPECIAL_CODES or num < 0)),
            })
    return pd.DataFrame(rows)


def main():
    paths.require(paths.MASTER_CSV, paths.PT_ACCESSIBILITY_CSV)
    header("QUALITY CONTROL AND FIRST PT DESCRIPTIVE ANALYSIS", 72)

    master = read_csv_auto(paths.MASTER_CSV)
    pc4 = read_csv_auto(paths.PT_ACCESSIBILITY_CSV)

    weight_col = first_existing(master.columns, WEIGHT_CANDIDATES)
    if weight_col is None:
        raise ValueError("No person-level survey-weight variable was found.")
    print(f"Using survey weight: {weight_col}")

    pc4_col = first_existing(pc4.columns, ["pc4_clean", "PC4", "pc4", "VertPC", "postcode4"])
    if pc4_col is None:
        raise ValueError("No PC4 column was found in the PT file.")
    pc4["pc4_clean"] = clean_pc4(pc4[pc4_col])
    pc4 = pc4.drop_duplicates("pc4_clean")

    integrity = integrity_audit(master)
    save_csv(integrity, TABLES / "01_master_integrity_summary.csv")
    save_csv(code_distributions(master), TABLES / "02_variable_code_distributions.csv")

    for col in ACCESS_COLS.values():
        if col not in pc4.columns:
            raise ValueError(f"{col} was not found in the PC4 PT file.")
    access_summary = pd.DataFrame([accessibility_summary(pc4[col], f"{label} accessibility") for label, col in ACCESS_COLS.items()])
    save_csv(access_summary, TABLES / "03_pc4_accessibility_summary.csv")

    freq_tables = []
    for variable, outcome in [("btm_frequency_ordered", "BTM"), ("train_frequency_ordered", "Train")]:
        if variable in master.columns:
            freq_tables.append(weighted_frequency_table(master, variable, weight_col).assign(outcome=outcome))
    if freq_tables:
        freq = pd.concat(freq_tables, ignore_index=True)
        save_csv(freq[["outcome"] + [c for c in freq.columns if c != "outcome"]], TABLES / "04_pt_use_frequency_distribution.csv")

    summaries = {}
    for label, prefix, file_name in [("BTM", "btm", "05_btm_use_by_btm_accessibility_quintile.csv"),
                                     ("Train", "train", "06_train_use_by_train_accessibility_quintile.csv")]:
        summaries[label] = grouped_use_summary(
            master, f"pt_access_decay_{prefix}_quintile_pc4", f"{prefix}_frequency_ordered", f"{prefix}_weekly_or_more", weight_col
        )
        save_csv(summaries[label], TABLES / file_name)

    # One row per PC4 so areas with many respondents aren't counted repeatedly
    if "urbanity" in master.columns:
        pc4_context = master[["pc4_clean", "urbanity", *ACCESS_COLS.values()]].dropna(subset=["pc4_clean"]).drop_duplicates("pc4_clean")
        rows = []
        for urbanity, g in pc4_context.groupby("urbanity"):
            for label, col in ACCESS_COLS.items():
                v = pd.to_numeric(g[col], errors="coerce").dropna()
                rows.append({"urbanity": urbanity, "accessibility_measure": label, "n_pc4": len(v), "mean": v.mean(), "median": v.median(),
                             "p25": v.quantile(0.25), "p75": v.quantile(0.75), "percent_zero": 100 * v.eq(0).mean()})
        save_csv(pd.DataFrame(rows), TABLES / "07_accessibility_by_urbanity.csv")

    sample_rows = [("All respondent records", len(master))]
    for flag, label in [("matched_pt_accessibility", "Matched to PT accessibility"), ("sample_btm_unadjusted", "BTM unadjusted sample"),
                        ("sample_btm_adjusted_core", "BTM current core-adjusted sample"), ("sample_train_unadjusted", "Train unadjusted sample"),
                        ("sample_train_adjusted_core", "Train current core-adjusted sample")]:
        if flag in master.columns:
            sample_rows.append((label, int(pd.to_numeric(master[flag], errors="coerce").eq(1).sum())))
    save_csv(pd.DataFrame(sample_rows, columns=["stage", "n"]), TABLES / "08_analysis_sample_flow.csv")

    if paths.MASTER_UNMATCHED_PC4_CSV.exists():
        save_csv(read_csv_auto(paths.MASTER_UNMATCHED_PC4_CSV), TABLES / "09_unmatched_pc4_summary.csv")

    for label, prefix in [("BTM", "btm"), ("Train", "train")]:
        s = summaries[label]
        line_plot(s, "weighted_mean_frequency", f"Weighted mean {label}-use frequency by {label} accessibility quintile",
                  f"{label} accessibility quintile", "Mean ordered frequency (1 = least, 5 = most frequent)",
                  f"{prefix}_mean_frequency_by_accessibility_quintile.png")
        line_plot(s, "weighted_weekly_or_more_percent", f"Weekly-or-more {label} use by {label} accessibility quintile",
                  f"{label} accessibility quintile", "Weighted percentage", f"{prefix}_weekly_share_by_accessibility_quintile.png")

    if "urbanity" in master.columns:
        pc4_plot = (master[["pc4_clean", "urbanity", "log1p_pt_access_decay_btm", "log1p_pt_access_decay_train"]]
                    .dropna(subset=["pc4_clean"]).drop_duplicates("pc4_clean"))
        for label, prefix in [("BTM", "btm"), ("Train", "train")]:
            boxplot_by_urbanity(pc4_plot, f"log1p_pt_access_decay_{prefix}", f"{label} accessibility by urbanity",
                                f"Log(1 + {label} accessibility)", f"{prefix}_accessibility_by_urbanity.png")

    header("QUALITY CONTROL COMPLETE", 72)
    print(f"\nMaster integrity:\n{integrity.to_string(index=False)}")
    print(f"\nPC4 accessibility summary:\n{access_summary.to_string(index=False)}")
    for label, s in summaries.items():
        print(f"\n{label} use by accessibility quintile:\n{s.to_string(index=False)}")
    print("""
Interpretation checks:
1. If every n_odin_source_rows equals 1, the annualised ODiN file was already respondent-level.
2. Check 02_variable_code_distributions.csv: special non-response codes must be set to missing before regression.
3. The adjusted-sample flags only test non-missingness; rebuild them once valid categories are confirmed.
4. Zero accessibility is a real zero; unmatched PC4s stay missing.""")
    print(f"\nTables: {TABLES}\nFigures: {FIGURES}")


if __name__ == "__main__":
    main()
