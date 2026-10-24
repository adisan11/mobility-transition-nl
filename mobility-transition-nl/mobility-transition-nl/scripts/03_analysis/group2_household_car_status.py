"""
Household car status (ICEV / EV / no car) within each Group 2 mobility profile, as stacked bars.

Input:   data/processed/ODiN2022_2023_mobility_transition_respondent_level.csv
Outputs: outputs/tables/group2_car_status/, outputs/figures/group2_car_status/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import header, read_csv_auto, save_csv, save_figure
from mobility_transition.odin import clean_opid, household_car_status
from mobility_transition.settings import MOBILITY_GROUPS

TABLES = paths.step_dir(paths.TABLES, "group2_car_status")
FIGURES = paths.step_dir(paths.FIGURES, "group2_car_status")

GROUP, WEIGHT, CARS, FUEL, ID = "mobility_group_2", "FactorP", "HHAuto", "BrandstofPa1", "OPID"
STATUSES = ["ICEV", "EV", "No car"]
COLOURS = {"ICEV": "#7A7A7A", "EV": "#2E8B57", "No car": "#4C78A8"}
FUEL_LABELS = {1: "Gasoline", 2: "Diesel", 3: "LPG", 4: "Electric", 5: "Other", 6: "Unknown",
               7: "Not applicable; no passenger car in household name"}
MIN_LABEL_SHARE = 2.0  # segments below this % get no label


def pct(part, whole):
    return 100 * part / whole if whole > 0 else np.nan


def summarise(data, groups):
    rows = []
    for group in groups:
        g = data[data[GROUP].eq(group)]
        if g.empty:
            continue
        classified = g[g["household_car_status"].isin(STATUSES)]
        unclassified = g[g["household_car_status"].eq("Unclassified")]
        n_all, w_all = len(g), g[WEIGHT].sum()
        n_cls, w_cls = len(classified), classified[WEIGHT].sum()
        for status in STATUSES:
            s = classified[classified["household_car_status"].eq(status)]
            rows.append({
                "mobility_group_2": group, "household_car_status": status,
                "n_all_profile_respondents": n_all, "weighted_all_profile_respondents": w_all,
                "n_classified_respondents": n_cls, "weighted_classified_population": w_cls,
                "n_unclassified_respondents": len(unclassified), "weighted_unclassified_population": unclassified[WEIGHT].sum(),
                "unclassified_share_unweighted_percent": pct(len(unclassified), n_all),
                "unclassified_share_weighted_percent": pct(unclassified[WEIGHT].sum(), w_all),
                "n_status": len(s), "weighted_status_population": s[WEIGHT].sum(),
                "share_among_classified_unweighted_percent": pct(len(s), n_cls),
                "share_among_classified_weighted_percent": pct(s[WEIGHT].sum(), w_cls),
                "share_of_all_profile_unweighted_percent": pct(len(s), n_all),
                "share_of_all_profile_weighted_percent": pct(s[WEIGHT].sum(), w_all),
            })
    return pd.DataFrame(rows)


def classification_audit(data):
    has_group = data[GROUP].notna()
    return pd.DataFrame(
        [("Rows loaded", len(data)), ("Unique nonmissing OPIDs", data[ID].nunique(dropna=True)), ("Rows with missing OPID", int(data[ID].isna().sum())),
         ("Rows with nonmissing mobility_group_2", int(has_group.sum())),
         ("Rows classifiable as ICEV, EV or No car", int((has_group & data["household_car_status"].isin(STATUSES)).sum())),
         ("Rows unclassified for household car status", int((has_group & data["household_car_status"].eq("Unclassified")).sum())),
         ("Rows with HHAuto == 0", int(data[CARS].eq(0).sum())), ("Rows with HHAuto >= 1", int(data[CARS].ge(1).sum())),
         ("Rows with BrandstofPa1 1-3", int(data[FUEL].isin([1, 2, 3]).sum()))]
        + [(f"Rows with BrandstofPa1 == {code}", int(data[FUEL].eq(code).sum())) for code in [4, 5, 6, 7]],
        columns=["metric", "value"],
    )


def unclassified_diagnostics(data, groups):
    u = data[data[GROUP].notna() & data["household_car_status"].eq("Unclassified")]
    if u.empty:
        return pd.DataFrame(columns=[GROUP, CARS, FUEL, "fuel_label", "n_respondents", "weighted_population"])
    d = u.groupby([GROUP, CARS, FUEL, "fuel_label"], dropna=False, observed=False).agg(
        n_respondents=(ID, "size"), weighted_population=(WEIGHT, "sum")).reset_index()
    d[GROUP] = pd.Categorical(d[GROUP], categories=groups, ordered=True)
    d = d.sort_values([GROUP, "weighted_population"], ascending=[True, False])
    d[GROUP] = d[GROUP].astype("string")
    return d


def plot_stacked(summary, groups, share_col, file_name, ylabel, title):
    pivot = summary[summary["mobility_group_2"].isin(groups)].pivot(
        index="mobility_group_2", columns="household_car_status", values=share_col).reindex(index=groups, columns=STATUSES)
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    x, bottom = np.arange(len(pivot)), np.zeros(len(pivot))
    for status in STATUSES:
        values = pivot[status].fillna(0).to_numpy(dtype=float)
        bars = ax.bar(x, values, bottom=bottom, label=status, color=COLOURS[status], width=0.72)
        for bar, v, b in zip(bars, values, bottom):
            if v >= MIN_LABEL_SHARE:
                ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
        bottom += values
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Household car status", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, FIGURES / file_name)


def main():
    paths.require(paths.RESPONDENT_LEVEL_CSV)
    header("GROUP 2 MOBILITY PROFILE BY HOUSEHOLD CAR STATUS", 78)

    data = read_csv_auto(paths.RESPONDENT_LEVEL_CSV)
    missing = [c for c in [ID, GROUP, WEIGHT, CARS, FUEL] if c not in data.columns]
    if missing:
        raise ValueError(f"The respondent-level file is missing required columns:\n{missing}")

    data[ID] = clean_opid(data[ID])
    data[[WEIGHT, CARS, FUEL]] = data[[WEIGHT, CARS, FUEL]].apply(pd.to_numeric, errors="coerce")
    data[GROUP] = data[GROUP].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})

    if data[ID].isna().any():
        print(f"WARNING: removing {int(data[ID].isna().sum()):,} rows with missing OPID.")
        data = data[data[ID].notna()].copy()
    if data[ID].duplicated().any():
        examples = data.loc[data[ID].duplicated(keep=False), [ID, GROUP, CARS, FUEL, WEIGHT]].sort_values(ID).head(20)
        raise ValueError(f"Duplicate OPIDs in the respondent file; stopped to avoid double counting.\n\n{examples.to_string(index=False)}")
    bad_weight = data[WEIGHT].isna() | data[WEIGHT].le(0)
    if bad_weight.any():
        print(f"WARNING: removing {int(bad_weight.sum()):,} respondents with missing or non-positive {WEIGHT}.")
        data = data[~bad_weight].copy()
    if data.empty:
        raise ValueError("No respondents remain after OPID and weight checks.")

    observed = data[GROUP].dropna().unique().tolist()
    unexpected = sorted(set(observed) - set(MOBILITY_GROUPS))
    if unexpected:
        print(f"WARNING: unexpected mobility_group_2 labels: {unexpected}")
    groups = [g for g in MOBILITY_GROUPS if g in observed] + unexpected
    if not groups:
        raise ValueError("No nonmissing mobility_group_2 profiles were found.")

    data["household_car_status"] = household_car_status(data[CARS], data[FUEL])
    data["owns_household_car"] = np.where(data[CARS].notna(), data[CARS].ge(1).astype(int), np.nan)
    data["fuel_label"] = data[FUEL].map(FUEL_LABELS).fillna("Missing or unexpected fuel code")

    summary = summarise(data, groups)
    if summary.empty:
        raise ValueError("No Group 2 profile rows were available for the car-status summary.")
    metrics = ["n_status", "weighted_status_population", "share_among_classified_unweighted_percent",
               "share_among_classified_weighted_percent", "share_of_all_profile_unweighted_percent", "share_of_all_profile_weighted_percent"]
    wide = summary.pivot(index="mobility_group_2", columns="household_car_status", values=metrics).reindex(index=groups)
    wide.columns = [f"{m}__{s.lower().replace(' ', '_')}" for m, s in wide.columns]

    save_csv(data[[ID, GROUP, WEIGHT, CARS, FUEL, "fuel_label", "owns_household_car", "household_car_status"]],
             TABLES / "01_respondent_household_car_status_classification.csv")
    save_csv(classification_audit(data), TABLES / "02_household_car_status_classification_audit.csv")
    save_csv(summary, TABLES / "03_group_2_household_car_status_summary_long.csv")
    save_csv(wide.reset_index(), TABLES / "04_group_2_household_car_status_summary_wide.csv")
    save_csv(unclassified_diagnostics(data, groups), TABLES / "05_unclassified_household_car_status_diagnostics.csv")

    plot_stacked(summary, groups, "share_among_classified_weighted_percent", "group_2_household_car_status_weighted_stacked_bar.png",
                 "Weighted share within mobility profile (%)", "Household car status by Mobility profile")
    plot_stacked(summary, groups, "share_among_classified_unweighted_percent", "group_2_household_car_status_unweighted_stacked_bar.png",
                 "Share of respondents within mobility profile (%)",
                 "Household car status by Group 2 mobility profile\nUnweighted shares among ICEV, EV and no-car classifications")

    header("WEIGHTED HOUSEHOLD CAR STATUS SHARES BY GROUP 2 PROFILE", 78)
    print(summary.pivot(index="mobility_group_2", columns="household_car_status", values="share_among_classified_weighted_percent")
          .reindex(index=groups, columns=STATUSES).round(2).to_string())
    header("UNCLASSIFIED SHARE BY GROUP 2 PROFILE", 78)
    cols = ["mobility_group_2", "n_all_profile_respondents", "n_classified_respondents", "n_unclassified_respondents",
            "unclassified_share_unweighted_percent", "unclassified_share_weighted_percent"]
    print(summary[cols].drop_duplicates("mobility_group_2").set_index("mobility_group_2").reindex(groups).round(3).to_string())
    print(f"\nTables: {TABLES}\nFigures: {FIGURES}")


if __name__ == "__main__":
    main()
