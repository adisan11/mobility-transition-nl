"""
Classifies adult ODiN respondents into four mobility groups from their stated car, BTM and train frequencies,
under two thresholds:
  Group 1 (main):        low PT = BTM and train both "few times/year" or less (codes 4-5)
  Group 2 (sensitivity): low PT = BTM and train both "few times/month" or less (codes 3-5)
Regular car use = FqAutoB codes 1-2; car-light = codes 3-5.

Input:   data/processed/ODiN2022_2023_pt.csv
Outputs: data/processed/threshold_analysis_sample_with_groups.csv
         outputs/tables/threshold_groups/, outputs/figures/threshold_groups/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import detect_csv_format, header, save_csv, save_figure
from mobility_transition.settings import FREQUENCY_CODES, FREQUENCY_LABELS

TABLES = paths.step_dir(paths.TABLES, "threshold_groups")  # == paths.THRESHOLD_TABLES
FIGURES = paths.step_dir(paths.FIGURES, "threshold_groups")

MINIMUM_AGE, MAXIMUM_AGE = 18, 96
LONG_FREQUENCY_LABELS = {1: "Daily or almost daily", 2: "Several times per week", 3: "A few times per month",
                         4: "A few times per year", 5: "Never or almost never"}
MODE_COLUMNS = {"Car": "FqAutoB", "BTM": "FqBTM", "Train": "FqTrein"}
REQUIRED = ["OPID", "Leeftijd", "FqAutoB", "FqBTM", "FqTrein"]
OPTIONAL = ["Jaar", "FactorP", "HHAuto", "BrandstofPa1", "Geslacht", "Opleiding", "HHGestInkG", "Sted", "HHPers", "BetWerk", "WoPC", "VertPC"]
GROUP_ORDER = ["Car-dependent, low PT use", "Multimodal regular car + PT", "Car-light PT user", "Car-light low-PT user"]
SPECIFICATIONS = {
    "Main": {"column": "mobility_group_main", "low_pt": [4, 5], "some_pt": [1, 2, 3]},
    "Sensitivity": {"column": "mobility_group_sensitivity", "low_pt": [3, 4, 5], "some_pt": [1, 2]},
}


def clean_filename(text):
    clean = text.lower()
    for old, new in [("+", "plus"), ("&", "and"), ("/", "_"), ("\\", "_"), (" ", "_"), (",", ""), ("-", "_")]:
        clean = clean.replace(old, new)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def read_selected_columns(path):
    encoding, sep = detect_csv_format(path)
    print(f"Reading: {path.name} (encoding={encoding}, sep={sep!r})")
    original = pd.read_csv(path, encoding=encoding, sep=sep, nrows=0).columns
    lookup = {str(c).strip(): c for c in original}
    missing = [c for c in REQUIRED if c not in lookup]
    if missing:
        raise ValueError(f"The ODiN file is missing required variables:\n{missing}")
    data = pd.read_csv(path, encoding=encoding, sep=sep, low_memory=False,
                       usecols=[lookup[c] for c in REQUIRED + OPTIONAL if c in lookup])
    data.columns = data.columns.astype(str).str.strip()
    return data


def prepare_person_sample(raw):
    """Collapse trip rows to one row per OPID (first non-missing value) and keep valid adult complete cases."""
    raw = raw.copy()
    raw["OPID"] = raw["OPID"].astype("string").str.strip()
    n_raw, n_unique = len(raw), raw["OPID"].nunique(dropna=True)
    raw = raw[raw["OPID"].notna() & raw["OPID"].ne("")]

    check_cols = ["Leeftijd", "FqAutoB", "FqBTM", "FqTrein", "HHAuto", "BrandstofPa1", "FactorP", "Jaar", "Geslacht",
                  "Opleiding", "HHGestInkG", "Sted", "HHPers", "BetWerk", "WoPC", "VertPC"]
    inconsistency = [(f"OPIDs with multiple nonmissing values for {c}", int(raw.groupby("OPID", sort=False)[c].nunique(dropna=True).gt(1).sum()))
                     for c in check_cols if c in raw.columns]

    person = raw.groupby("OPID", as_index=False, sort=False).first()
    numeric = [c for c in check_cols if c in person.columns]
    person[numeric] = person[numeric].apply(pd.to_numeric, errors="coerce")

    valid = {
        f"Valid age {MINIMUM_AGE}-{MAXIMUM_AGE}": person["Leeftijd"].between(MINIMUM_AGE, MAXIMUM_AGE),
        "Valid FqAutoB codes 1-5": person["FqAutoB"].isin(FREQUENCY_CODES),
        "Valid FqBTM codes 1-5": person["FqBTM"].isin(FREQUENCY_CODES),
        "Valid FqTrein codes 1-5": person["FqTrein"].isin(FREQUENCY_CODES),
    }
    mask = np.logical_and.reduce(list(valid.values()))
    sample = person.loc[mask].copy()

    audit = ([("Raw displacement rows", n_raw), ("Unique nonmissing OPIDs in raw data", n_unique),
              ("Person-level records after OPID collapse", len(person))]
             + [(k, int(v.sum())) for k, v in valid.items()]
             + [("Final adult complete-case threshold sample", len(sample))] + inconsistency)
    return sample, pd.DataFrame(audit, columns=["metric", "value"])


def classify(data, column, low_pt_codes, some_pt_codes):
    regular_car, car_light = data["FqAutoB"].isin([1, 2]), data["FqAutoB"].isin([3, 4, 5])
    low_btm, low_train = data["FqBTM"].isin(low_pt_codes), data["FqTrein"].isin(low_pt_codes)
    some_btm, some_train = data["FqBTM"].isin(some_pt_codes), data["FqTrein"].isin(some_pt_codes)
    low_pt, some_pt = low_btm & low_train, some_btm | some_train

    data[column] = np.select(
        [regular_car & low_pt, regular_car & some_pt, car_light & some_pt, car_light & low_pt], GROUP_ORDER, default="Unclassified")
    for suffix, flag in [("regular_car", regular_car), ("car_light", car_light), ("low_pt", low_pt), ("some_pt", some_pt),
                         ("low_btm", low_btm), ("low_train", low_train), ("meaningful_btm", some_btm), ("meaningful_train", some_train)]:
        data[f"{column}_{suffix}"] = flag.astype(int)
    return data


def valid_weights(data):
    return data["FactorP"].where(data["FactorP"].gt(0)) if "FactorP" in data.columns else pd.Series(np.nan, index=data.index)


def group_summary(data, column, spec):
    w = valid_weights(data)
    total_w = w.sum() if w.notna().any() else np.nan
    rows = []
    for group in GROUP_ORDER + ["Unclassified"]:
        mask = data[column].eq(group)
        weighted = total_w > 0 if pd.notna(total_w) else False
        rows.append({
            "specification": spec, "group": group, "n_unweighted": int(mask.sum()),
            "share_unweighted_percent": 100 * mask.sum() / len(data) if len(data) else np.nan,
            "weighted_observations": w[mask].sum() if weighted else np.nan,
            "share_weighted_percent": 100 * w[mask].sum() / total_w if weighted else np.nan,
        })
    return pd.DataFrame(rows)


def frequency_distribution(data, column, spec):
    w = valid_weights(data)
    has_w = w.notna().any()
    rows = []
    for group in GROUP_ORDER + ["Unclassified"]:
        g = data[data[column].eq(group)]
        gw = w.loc[g.index]
        total = gw.sum() if has_w else np.nan
        for mode, mode_col in MODE_COLUMNS.items():
            for code in FREQUENCY_CODES:
                mask = g[mode_col].eq(code)
                weighted = has_w and total > 0
                obs = gw.loc[g.index[mask]].sum() if weighted else np.nan
                rows.append({
                    "specification": spec, "group": group, "mode": mode, "mode_column": mode_col, "frequency_code": code,
                    "frequency_label": LONG_FREQUENCY_LABELS[code], "n_unweighted": int(mask.sum()),
                    "share_unweighted_percent": 100 * mask.sum() / len(g) if len(g) else np.nan,
                    "weighted_observations": obs, "share_weighted_percent": 100 * obs / total if weighted else np.nan,
                })
    return pd.DataFrame(rows)


def save_crosswalks(data):
    pd.crosstab(data["mobility_group_main"], data["mobility_group_sensitivity"], dropna=False).to_csv(
        TABLES / "05_main_vs_sensitivity_crosswalk_unweighted.csv", encoding="utf-8-sig")
    if "FactorP" in data.columns and data["FactorP"].gt(0).any():
        pd.pivot_table(data.assign(_w=valid_weights(data)), index="mobility_group_main", columns="mobility_group_sensitivity",
                       values="_w", aggfunc="sum", fill_value=0).to_csv(TABLES / "06_main_vs_sensitivity_crosswalk_weighted.csv", encoding="utf-8-sig")


# ---------------- plots ----------------

def share_metric(table, use_weighted=True):
    weighted = use_weighted and table["share_weighted_percent"].notna().any()
    return ("share_weighted_percent", "Weighted share within group (%)") if weighted else ("share_unweighted_percent", "Unweighted share within group (%)")


def plot_group_shares(summary, value_col, file_name, ylabel):
    pivot = summary[summary["group"].isin(GROUP_ORDER)].pivot(index="group", columns="specification", values=value_col)
    pivot = pivot.reindex(GROUP_ORDER).reindex(columns=list(SPECIFICATIONS))
    ax = pivot.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_title("Mobility group shares: main vs sensitivity specification")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Specification")
    save_figure(ax.figure, FIGURES / file_name)


def plot_frequency_profile(freq, spec, group, out_dir):
    metric, ylabel = share_metric(freq)
    g = freq[freq["specification"].eq(spec) & freq["group"].eq(group)]
    if g.empty:
        return
    pivot = g.pivot(index="mode", columns="frequency_code", values=metric).reindex(list(MODE_COLUMNS)).reindex(columns=FREQUENCY_CODES)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom, x = np.zeros(len(pivot)), np.arange(len(pivot))
    for code in FREQUENCY_CODES:
        values = pivot[code].fillna(0).to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=FREQUENCY_LABELS[code])
        bottom += values
    ax.set_title(f"{group}\n{spec} specification")
    ax.set_xlabel("Mode")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Reported frequency", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, out_dir / f"{clean_filename(spec)}_{clean_filename(group)}_frequency_profile.png")


def plot_mode_comparison(freq, group, mode, out_dir):
    metric, ylabel = share_metric(freq)
    d = freq[freq["group"].eq(group) & freq["mode"].eq(mode) & freq["specification"].isin(list(SPECIFICATIONS))]
    if d.empty:
        return
    pivot = d.pivot(index="frequency_code", columns="specification", values=metric).reindex(FREQUENCY_CODES).reindex(columns=list(SPECIFICATIONS))
    ax = pivot.plot(kind="bar", figsize=(9, 5.5), width=0.8)
    ax.set_title(f"{mode} frequency: {group}")
    ax.set_xlabel("Reported frequency")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([FREQUENCY_LABELS[c] for c in FREQUENCY_CODES], rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Specification")
    save_figure(ax.figure, out_dir / f"{clean_filename(group)}_{clean_filename(mode)}_main_vs_sensitivity.png")


def main():
    paths.require(paths.ODIN_TRIPS_CSV)
    header("THRESHOLD-BASED MOBILITY GROUP ANALYSIS", 78)

    sample, audit = prepare_person_sample(read_selected_columns(paths.ODIN_TRIPS_CSV))
    if sample.empty:
        raise ValueError("No respondents remain in the adult complete-case sample.")
    for spec in SPECIFICATIONS.values():
        sample = classify(sample, spec["column"], spec["low_pt"], spec["some_pt"])

    save_csv(audit, TABLES / "01_threshold_sample_audit.csv")
    save_csv(sample, paths.THRESHOLD_GROUPS_CSV)

    summary = pd.concat([group_summary(sample, s["column"], name) for name, s in SPECIFICATIONS.items()], ignore_index=True)
    freq = pd.concat([frequency_distribution(sample, s["column"], name) for name, s in SPECIFICATIONS.items()], ignore_index=True)
    save_csv(summary, paths.THRESHOLD_GROUP_SUMMARY_CSV)
    save_csv(freq, paths.THRESHOLD_FREQUENCY_CSV)
    save_crosswalks(sample)

    plot_group_shares(summary, "share_unweighted_percent", "group_share_main_vs_sensitivity_unweighted.png", "Unweighted share of respondents (%)")
    if summary["share_weighted_percent"].notna().any():
        plot_group_shares(summary, "share_weighted_percent", "group_share_main_vs_sensitivity_weighted.png", "Weighted share of respondents (%)")
    for spec in SPECIFICATIONS:
        out = paths.step_dir(FIGURES, f"{spec.lower()}_group_frequency_profiles")
        for group in GROUP_ORDER:
            plot_frequency_profile(freq, spec, group, out)
    comparison_dir = paths.step_dir(FIGURES, "main_vs_sensitivity_double_bar_frequency_profiles")
    for group in GROUP_ORDER:
        for mode in MODE_COLUMNS:
            plot_mode_comparison(freq, group, mode, comparison_dir)

    header("GROUP SUMMARY", 78)
    cols = ["specification", "group", "n_unweighted", "share_unweighted_percent", "weighted_observations", "share_weighted_percent"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nGroups: {paths.THRESHOLD_GROUPS_CSV}\nTables: {TABLES}\nFigures: {FIGURES}")


if __name__ == "__main__":
    main()
