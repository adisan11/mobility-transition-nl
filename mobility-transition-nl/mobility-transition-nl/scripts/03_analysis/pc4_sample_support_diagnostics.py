"""
How well supported are the PC4 mobility-profile shares? Per PC4: respondent count, effective sample size
(Kish: (sum w)^2 / sum w^2), largest single-respondent weight share, profile concentration, and a
leave-one-respondent-out test of whether the dominant profile changes.

Inputs:  data/processed/group2_pc4_unique_opid_respondent_sample.csv (from 04_maps/map_group2_profiles_by_pc4.py),
         data/raw/cbs/cbs_pc4_2023_v2.gpkg
Outputs: outputs/tables/pc4_sample_support/, outputs/figures/pc4_sample_support/, outputs/maps/pc4_sample_support/
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import clean_pc4, find_column, header, read_csv_auto, save_csv, save_figure
from mobility_transition.settings import MOBILITY_GROUPS, TARGET_CRS

TABLES = paths.step_dir(paths.TABLES, "pc4_sample_support")
FIGURES = paths.step_dir(paths.FIGURES, "pc4_sample_support")
MAPS = paths.step_dir(paths.MAPS, "pc4_sample_support")

WEIGHT, GROUP = "FactorP", "mobility_group_2"
SHORT = {"Car users": "car", "Multimodal users": "multimodal", "PT users": "pt", "Infrequent car and PT users": "infrequent_car_pt"}
RAW_N_THRESHOLDS = [1, 2, 3, 5, 10, 15, 20, 30, 50]
EFFECTIVE_N_THRESHOLDS = [1, 2, 3, 5, 10, 15, 20, 30]
SIZE_BINS, SIZE_LABELS = [0, 1, 4, 9, 19, 49, np.inf], ["1", "2-4", "5-9", "10-19", "20-49", "50+"]
GEOM_PC4_CANDIDATES = ["pc4_clean", "pc4", "PC4", "postcode", "Postcode", "Postal code", "postcode4", "Postcode4", "statcode", "gebiedscode", "code"]


def pc4_diagnostics(respondents):
    rows = []
    for pc4, g in respondents.groupby("pc4_clean", sort=False):
        w = g[WEIGHT].to_numpy(dtype=float)
        n, total, sum_sq = len(g), w.sum(), np.square(w).sum()
        effective_n = total ** 2 / sum_sq if sum_sq > 0 else np.nan
        by_profile = g.groupby(GROUP)[WEIGHT].sum()
        weights = {p: float(by_profile.get(p, 0.0)) for p in MOBILITY_GROUPS}
        shares = {p: 100 * weights[p] / total for p in MOBILITY_GROUPS}
        ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        dominant, top, second = ranked[0][0], ranked[0][1], ranked[1][1]

        row = {
            "pc4_clean": pc4, "n_classified_respondents": n, "weighted_classified_population": total, "sum_weight_squared": sum_sq,
            "effective_sample_size": effective_n, "effective_n_to_raw_n_ratio": effective_n / n if n else np.nan,
            "maximum_respondent_weight": w.max(), "maximum_respondent_weight_share_pct": 100 * w.max() / total if total > 0 else np.nan,
            "factorp_cv": w.std(ddof=0) / w.mean() if w.mean() > 0 else np.nan, "n_profiles_present": int(g[GROUP].nunique()),
            "dominant_mobility_profile": dominant, "dominant_profile_share_pct": top, "second_profile_share_pct": second,
            "dominant_profile_margin_pct": top - second, "mobility_profile_hhi": sum((s / 100) ** 2 for s in shares.values()),
            "any_profile_100_pct": int(np.isclose(top, 100)), "dominant_share_ge_90_pct": int(top >= 90), "dominant_share_ge_80_pct": int(top >= 80),
        }
        for p in MOBILITY_GROUPS:
            row[f"weighted_{SHORT[p]}"] = weights[p]
            row[f"share_{SHORT[p]}_pct"] = shares[p]
        rows.append(row)
    d = pd.DataFrame(rows)
    d["sample_size_bin"] = pd.cut(d["n_classified_respondents"], bins=SIZE_BINS, labels=SIZE_LABELS, include_lowest=True)
    return d


def threshold_coverage(d, column, thresholds, label, extra):
    total_pc4, total_n, total_w = len(d), d["n_classified_respondents"].sum(), d["weighted_classified_population"].sum()
    rows = []
    for t in thresholds:
        r = d[d[column].ge(t)]
        k = len(r)
        row = {label: t, "n_pc4_retained": k, "share_pc4_retained_pct": 100 * k / total_pc4,
               "n_respondents_retained": r["n_classified_respondents"].sum(),
               "share_respondents_retained_pct": 100 * r["n_classified_respondents"].sum() / total_n,
               "weighted_population_retained": r["weighted_classified_population"].sum(),
               "share_weighted_population_retained_pct": 100 * r["weighted_classified_population"].sum() / total_w,
               "share_pc4_with_100_pct_profile": 100 * r["any_profile_100_pct"].mean() if k else np.nan}
        for name, (col, fn) in extra.items():
            row[name] = (getattr(r[col], fn)() * (100 if name.startswith("share") else 1)) if k else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_out(respondents):
    """For each PC4, remove each respondent in turn and check whether the dominant weighted profile changes."""
    rows = []
    for pc4, g in respondents.groupby("pc4_clean", sort=False):
        w, profiles, n = g[WEIGHT].to_numpy(dtype=float), g[GROUP].to_numpy(), len(g)
        total = w.sum()
        base_w = np.array([w[profiles == p].sum() for p in MOBILITY_GROUPS])
        base_shares = 100 * base_w / total
        base_dom = int(np.argmax(base_shares))
        row = {"pc4_clean": pc4, "n_classified_respondents": n, "baseline_dominant_profile": MOBILITY_GROUPS[base_dom]}
        if n == 1:
            rows.append({**row, "n_valid_leave_one_out_tests": 0, "n_dominant_profile_changes": np.nan, "dominant_profile_change_rate_pct": np.nan,
                         "maximum_absolute_share_shift_pct": np.nan, "mean_maximum_share_shift_pct": np.nan, "dominant_profile_stable_all_loo": np.nan})
            continue

        changes, shifts = 0, []
        for i in range(n):
            loo_total = total - w[i]
            if loo_total <= 0:
                continue
            loo_w = base_w.copy()
            loo_w[MOBILITY_GROUPS.index(profiles[i])] -= w[i]
            loo_shares = 100 * loo_w / loo_total
            changes += int(np.argmax(loo_shares) != base_dom)
            shifts.append(np.abs(loo_shares - base_shares).max())
        k = len(shifts)
        rows.append({**row, "n_valid_leave_one_out_tests": k, "n_dominant_profile_changes": changes,
                     "dominant_profile_change_rate_pct": 100 * changes / k if k else np.nan,
                     "maximum_absolute_share_shift_pct": max(shifts) if shifts else np.nan,
                     "mean_maximum_share_shift_pct": np.mean(shifts) if shifts else np.nan,
                     "dominant_profile_stable_all_loo": int(changes == 0)})
    return pd.DataFrame(rows)


def bar(x_labels, values, title, ylabel, file_name, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(values))
    ax.bar(x, values)
    ax.set_title(title)
    ax.set_xlabel("Unweighted respondents per PC4")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FIGURES / file_name)


def figures(d, bin_diag, loo_bins):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(d["n_classified_respondents"], bins=50)
    ax.set_title("Distribution of classified ODiN respondents per PC4")
    ax.set_xlabel("Unweighted number of respondents")
    ax.set_ylabel("Number of PC4 areas")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FIGURES / "distribution_pc4_unweighted_respondent_count.png")

    sorted_n = np.sort(d["n_classified_respondents"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_n, np.arange(1, len(sorted_n) + 1) / len(sorted_n) * 100)
    ax.set_title("Cumulative distribution of PC4 respondent sample size")
    ax.set_xlabel("Unweighted respondents per PC4")
    ax.set_ylabel("Cumulative share of PC4 areas (%)")
    ax.grid(True, alpha=0.3)
    save_figure(fig, FIGURES / "cumulative_pc4_sample_size_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(d["n_classified_respondents"], d["effective_sample_size"], alpha=0.4)
    top = d["n_classified_respondents"].max()
    ax.plot([0, top], [0, top], linestyle="--")
    ax.set_title("Raw and weight-adjusted PC4 sample size")
    ax.set_xlabel("Unweighted respondent count")
    ax.set_ylabel("Effective sample size")
    ax.grid(True, alpha=0.3)
    save_figure(fig, FIGURES / "raw_sample_size_vs_effective_sample_size.png")

    bar(bin_diag["sample_size_bin"], bin_diag["share_pc4_with_100_pct_profile"], "PC4 areas with a 100% mobility-profile share",
        "Share of PC4 areas (%)", "extreme_profile_shares_by_pc4_sample_size.png")

    data, labels = [], []
    for b in SIZE_LABELS:
        v = d.loc[d["sample_size_bin"].astype("string").eq(b), "dominant_profile_share_pct"].dropna().to_numpy()
        if len(v):
            data.append(v)
            labels.append(b)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title("Dominant mobility-profile share by PC4 sample size")
    ax.set_xlabel("Unweighted respondents per PC4")
    ax.set_ylabel("Dominant weighted profile share (%)")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FIGURES / "dominant_profile_share_by_pc4_sample_size.png")

    loo = loo_bins[loo_bins["n_pc4"].gt(0)]
    bar(loo["sample_size_bin"], loo["share_dominant_profile_stable_pct"], "Dominant mobility-profile stability after removing one respondent",
        "PC4 areas stable in every leave-one-out test (%)", "leave_one_out_instability_by_pc4_sample_size.png", ylim=(0, 100))


def maps(pc4_map):
    specs = [
        ("n_classified_respondents", "Unweighted classified ODiN respondents by PC4", {}, "No classified respondents", "map_pc4_unweighted_respondent_count.png"),
        ("effective_sample_size", "Effective respondent sample size by PC4", {}, "No classified respondents", "map_pc4_effective_sample_size.png"),
        ("maximum_respondent_weight_share_pct", "Largest single respondent contribution to PC4 weight\nShare of total FactorP weight",
         {"vmin": 0, "vmax": 100}, "No classified respondents", "map_pc4_largest_respondent_weight_share.png"),
        ("dominant_profile_change_rate_pct", "Dominant mobility-profile instability by PC4\nShare of leave-one-respondent-out tests changing the dominant profile",
         {"vmin": 0, "vmax": 100}, "Not testable / no respondents", "map_pc4_leave_one_out_dominant_profile_instability.png"),
    ]
    for column, title, scale, missing_label, file_name in specs:
        fig, ax = plt.subplots(figsize=(10, 12))
        pc4_map.plot(ax=ax, column=column, cmap="viridis", linewidth=0, legend=True, missing_kwds={"color": "lightgrey", "label": missing_label}, **scale)
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_axis_off()
        save_figure(fig, MAPS / file_name)


def main():
    paths.require(paths.GROUP2_PC4_RESPONDENTS_CSV, paths.PC4_POLYGONS_2023_GPKG)
    header("PC4 MOBILITY PROFILE SAMPLE SUPPORT DIAGNOSTICS", 80)

    respondents = read_csv_auto(paths.GROUP2_PC4_RESPONDENTS_CSV)
    missing = [c for c in ["OPID", "pc4_clean", WEIGHT, GROUP] if c not in respondents.columns]
    if missing:
        raise ValueError(f"Respondent file is missing required columns:\n{missing}")
    respondents["pc4_clean"] = clean_pc4(respondents["pc4_clean"])
    respondents[WEIGHT] = pd.to_numeric(respondents[WEIGHT], errors="coerce")
    respondents = respondents[respondents["pc4_clean"].notna() & respondents[WEIGHT].gt(0) & respondents[GROUP].isin(MOBILITY_GROUPS)].copy()
    if respondents["OPID"].duplicated().any():
        raise ValueError("Duplicate OPIDs found in the respondent sample.")
    print(f"Unique respondents: {len(respondents):,}; PC4 areas: {respondents['pc4_clean'].nunique():,}")

    d = pc4_diagnostics(respondents)
    pct = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    header("PC4 RESPONDENT COUNT DISTRIBUTION", 80)
    print(d["n_classified_respondents"].describe(percentiles=pct).to_string())
    print("\nEffective sample size distribution:\n" + d["effective_sample_size"].describe(percentiles=pct).to_string())

    raw_cov = threshold_coverage(d, "n_classified_respondents", RAW_N_THRESHOLDS, "minimum_unweighted_n", {
        "share_pc4_dominant_ge_90_pct": ("dominant_share_ge_90_pct", "mean"),
        "median_effective_sample_size": ("effective_sample_size", "median"),
        "median_maximum_weight_share_pct": ("maximum_respondent_weight_share_pct", "median")})
    eff_cov = threshold_coverage(d, "effective_sample_size", EFFECTIVE_N_THRESHOLDS, "minimum_effective_sample_size", {
        "median_raw_n": ("n_classified_respondents", "median"),
        "median_maximum_weight_share_pct": ("maximum_respondent_weight_share_pct", "median")})
    save_csv(raw_cov, TABLES / "02_pc4_sample_threshold_coverage.csv")
    save_csv(eff_cov, TABLES / "04_pc4_effective_sample_size_threshold_coverage.csv")
    header("RAW SAMPLE SIZE THRESHOLD COVERAGE", 80)
    print(raw_cov.round(2).to_string(index=False))

    bin_diag = d.groupby("sample_size_bin", observed=False).agg(
        n_pc4=("pc4_clean", "size"), total_respondents=("n_classified_respondents", "sum"), median_raw_n=("n_classified_respondents", "median"),
        median_effective_n=("effective_sample_size", "median"), mean_effective_n=("effective_sample_size", "mean"),
        median_effective_n_ratio=("effective_n_to_raw_n_ratio", "median"), median_max_weight_share_pct=("maximum_respondent_weight_share_pct", "median"),
        mean_dominant_share_pct=("dominant_profile_share_pct", "mean"), median_dominant_share_pct=("dominant_profile_share_pct", "median"),
        mean_dominant_margin_pct=("dominant_profile_margin_pct", "mean"), share_pc4_with_100_pct_profile=("any_profile_100_pct", "mean"),
        share_pc4_dominant_ge_90_pct=("dominant_share_ge_90_pct", "mean"), mean_n_profiles_present=("n_profiles_present", "mean"),
    ).reset_index()
    bin_diag[["share_pc4_with_100_pct_profile", "share_pc4_dominant_ge_90_pct"]] *= 100
    save_csv(bin_diag, TABLES / "03_pc4_sample_size_bin_diagnostics.csv")
    header("SAMPLE SIZE BIN DIAGNOSTICS", 80)
    print(bin_diag.round(2).to_string(index=False))

    header("LEAVE-ONE-RESPONDENT-OUT STABILITY", 80)
    loo = leave_one_out(respondents)
    save_csv(loo, TABLES / "05_pc4_leave_one_out_stability.csv")
    d = d.merge(loo.drop(columns=["n_classified_respondents", "baseline_dominant_profile"]), on="pc4_clean", how="left", validate="one_to_one")
    loo_bins = d[d["n_classified_respondents"].ge(2)].groupby("sample_size_bin", observed=False).agg(
        n_pc4=("pc4_clean", "size"), share_dominant_profile_stable_pct=("dominant_profile_stable_all_loo", "mean"),
        mean_dominant_change_rate_pct=("dominant_profile_change_rate_pct", "mean"),
        median_dominant_change_rate_pct=("dominant_profile_change_rate_pct", "median"),
        median_max_share_shift_pct=("maximum_absolute_share_shift_pct", "median"),
        p90_max_share_shift_pct=("maximum_absolute_share_shift_pct", lambda x: x.quantile(0.90)),
    ).reset_index()
    loo_bins["share_dominant_profile_stable_pct"] *= 100
    print(loo_bins.round(2).to_string(index=False))
    save_csv(d, TABLES / "01_pc4_sample_support_diagnostics.csv")

    figures(d, bin_diag, loo_bins)

    pc4 = gpd.read_file(paths.PC4_POLYGONS_2023_GPKG)
    pc4["pc4_clean"] = clean_pc4(pc4[find_column(pc4, GEOM_PC4_CANDIDATES, "PC4 column in geometry")])
    pc4 = (pc4.set_crs(TARGET_CRS) if pc4.crs is None else pc4).to_crs(TARGET_CRS)
    if pc4["pc4_clean"].duplicated().any():
        raise ValueError("Duplicate PC4 identifiers detected in geometry.")
    pc4_map = pc4.merge(d, on="pc4_clean", how="left", validate="one_to_one")
    pc4_map.to_file(MAPS / "pc4_sample_support_diagnostics.gpkg", driver="GPKG")
    maps(pc4_map)

    header("KEY DIAGNOSTIC SUMMARY", 80)
    n = d["n_classified_respondents"]
    for t in [5, 10, 20]:
        print(f"PC4s with fewer than {t} respondents: {int(n.lt(t).sum()):,} ({100 * n.lt(t).mean():.2f}%)")
    print(f"Single-respondent PC4s: {int(n.eq(1).sum()):,} ({100 * n.eq(1).mean():.2f}%)")
    print(f"PC4s with one profile at 100%: {int(d['any_profile_100_pct'].sum()):,} ({100 * d['any_profile_100_pct'].mean():.2f}%)")
    testable = d["dominant_profile_stable_all_loo"].notna()
    print(f"Leave-one-out testable PC4s: {int(testable.sum()):,}; dominant profile stable in every test: "
          f"{100 * d.loc[testable, 'dominant_profile_stable_all_loo'].mean():.2f}%")
    print(f"\nTables: {TABLES}\nFigures: {FIGURES}\nMaps: {MAPS}")


if __name__ == "__main__":
    main()
