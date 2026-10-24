"""
Distribution of PC4 EV charging accessibility: percentile/log/quintile versions, population-weighted
Lorenz curve and Gini, and the relationship with CBS urbanity.

Inputs:  data/processed/pc4_ev_accessibility.csv, data/raw/cbs/pc4_stats.csv
Outputs: data/processed/pc4_ev_accessibility_enriched.csv   (used by 04_maps/map_ev_accessibility.py)
         outputs/tables/ev_accessibility_inequality/, outputs/figures/ev_accessibility_inequality/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import clean_pc4, find_column, header, read_csv_auto, save_csv, save_figure
from mobility_transition.settings import PC4_CANDIDATES

# Main index: strict public/access-unspecified, capacity-weighted, 10 km catchment
ACCESS_COL = "ev_access_capacity_decay_strict_10km"
POPULATION_COL = "population_represented"
URBANITY_COL = "urbanity"

PERCENTILE_COL = f"{ACCESS_COL}_percentile_0_100"
LOG_COL = f"log_{ACCESS_COL}"
QUINTILE_COL = f"{ACCESS_COL}_quintile_class"

EXCLUDE_ZERO_ACCESS_FOR_URBANITY = True
EXCLUDE_ZERO_ACCESS_FOR_LORENZ = False
LORENZ_MARKERS = (50, 70, 80)  # population percentiles annotated on the curve

TABLES = paths.step_dir(paths.TABLES, "ev_accessibility_inequality")
FIGURES = paths.step_dir(paths.FIGURES, "ev_accessibility_inequality")


def weighted_lorenz_curve(df, value_col, weight_col, exclude_zeros=False):
    """
    Population-weighted Lorenz curve, sorted from lowest to highest accessibility.
    x = cumulative population share, y = cumulative share of supply (accessibility * population).
    """
    data = df[[value_col, weight_col]].apply(pd.to_numeric, errors="coerce").dropna()
    data = data[(data[weight_col] > 0) & (data[value_col] >= 0)].copy()
    if exclude_zeros:
        data = data[data[value_col] > 0].copy()
    if data.empty:
        raise ValueError("No valid rows available for Lorenz curve.")

    data = data.sort_values(value_col).reset_index(drop=True)
    data["accessibility_supply"] = data[value_col] * data[weight_col]
    total_supply = data["accessibility_supply"].sum()
    if total_supply <= 0:
        raise ValueError("Total accessibility supply is zero. Cannot create Lorenz curve.")

    data["cum_population_share"] = data[weight_col].cumsum() / data[weight_col].sum()
    data["cum_supply_share"] = data["accessibility_supply"].cumsum() / total_supply
    return pd.concat(
        [pd.DataFrame({"cum_population_share": [0.0], "cum_supply_share": [0.0]}),
         data[["cum_population_share", "cum_supply_share"]]],
        ignore_index=True,
    )


def gini_from_lorenz(lorenz):
    return 1 - 2 * np.trapezoid(lorenz["cum_supply_share"].values, lorenz["cum_population_share"].values)


def supply_share_at(lorenz, population_share):
    return np.interp(population_share, lorenz["cum_population_share"], lorenz["cum_supply_share"])


def load_and_enrich():
    header("LOADING EV ACCESSIBILITY")
    ev = read_csv_auto(paths.EV_ACCESSIBILITY_CSV)
    ev["pc4_clean"] = clean_pc4(ev[find_column(ev, PC4_CANDIDATES, "PC4 column")])

    for col, hint in [(ACCESS_COL, ""), (POPULATION_COL, " It comes from the population-weighted centroid step.")]:
        if col not in ev.columns:
            raise ValueError(f"Column '{col}' not found in {paths.EV_ACCESSIBILITY_CSV.name}.{hint}")

    ev[ACCESS_COL] = pd.to_numeric(ev[ACCESS_COL], errors="coerce")
    ev[POPULATION_COL] = pd.to_numeric(ev[POPULATION_COL], errors="coerce")
    ev = ev.dropna(subset=["pc4_clean", ACCESS_COL]).copy()
    print(f"PC4s: {len(ev):,}\n{ev[ACCESS_COL].describe()}")

    ev[PERCENTILE_COL] = ev[ACCESS_COL].rank(method="average", pct=True) * 100
    ev[LOG_COL] = np.log1p(ev[ACCESS_COL])
    try:
        ev[QUINTILE_COL] = pd.qcut(ev[ACCESS_COL], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    except ValueError:  # too many ties to form 5 bins
        ev[QUINTILE_COL] = np.nan

    save_csv(ev, paths.EV_ACCESSIBILITY_ENRICHED_CSV)
    print(f"Saved: {paths.EV_ACCESSIBILITY_ENRICHED_CSV}")
    return ev


def lorenz_analysis(ev):
    header("LORENZ CURVE AND GINI")
    lorenz = weighted_lorenz_curve(ev, ACCESS_COL, POPULATION_COL, EXCLUDE_ZERO_ACCESS_FOR_LORENZ)
    gini = gini_from_lorenz(lorenz)
    shares = {p: supply_share_at(lorenz, p / 100) for p in LORENZ_MARKERS}

    print(f"Gini coefficient: {gini:.4f}")
    for p, s in shares.items():
        print(f"Bottom {p}% of population receives {s * 100:.2f}% of accessibility supply.")

    save_csv(lorenz, TABLES / "lorenz_curve_data_ev_accessibility.csv")
    summary = {
        "accessibility_column": ACCESS_COL,
        "population_column": POPULATION_COL,
        "exclude_zero_accessibility": EXCLUDE_ZERO_ACCESS_FOR_LORENZ,
        "gini_coefficient": gini,
        **{f"bottom_{p}_population_supply_share": s for p, s in shares.items()},
        **{f"bottom_{p}_population_supply_percent": s * 100 for p, s in shares.items()},
    }
    save_csv(pd.DataFrame([summary]), TABLES / "gini_summary_ev_accessibility.csv")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(lorenz["cum_population_share"], lorenz["cum_supply_share"], linewidth=2, label=f"Lorenz curve (Gini = {gini:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Line of equality")
    for p, s in shares.items():
        x = p / 100
        ax.scatter(x, s, s=45, zorder=3)
        ax.plot([x, x], [0, s], linestyle=":", linewidth=1, alpha=0.7)
        ax.plot([0, x], [s, s], linestyle=":", linewidth=1, alpha=0.7)
        ax.annotate(f"{p}% of population\nreceives {s * 100:.1f}% of supply", xy=(x, s), xytext=(8, 8),
                    textcoords="offset points", fontsize=9)
    ax.set_title("Lorenz curve of EV charging accessibility")
    ax.set_xlabel("Cumulative share of population")
    ax.set_ylabel("Cumulative share of accessibility supply")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, FIGURES / "lorenz_curve_ev_accessibility.png")
    return gini


def urbanity_analysis(ev):
    header("URBANITY ANALYSIS")
    stats = read_csv_auto(paths.PC4_STATS_CSV)
    stats["pc4_clean"] = clean_pc4(stats[find_column(stats, PC4_CANDIDATES, "PC4 column in pc4_stats")])
    if URBANITY_COL not in stats.columns:
        raise ValueError(f"Could not find '{URBANITY_COL}' in pc4_stats. Available columns:\n{list(stats.columns)}")

    stats[URBANITY_COL] = pd.to_numeric(stats[URBANITY_COL], errors="coerce")
    n_before = len(stats)
    stats = stats[stats[URBANITY_COL].isin([1, 2, 3, 4, 5])].copy()
    print(f"Removed invalid urbanity rows: {n_before - len(stats):,}; retained: {len(stats):,}")

    merged = stats[["pc4_clean", URBANITY_COL]].merge(
        ev[["pc4_clean", ACCESS_COL, PERCENTILE_COL, LOG_COL, QUINTILE_COL]], on="pc4_clean", how="inner"
    ).dropna(subset=[URBANITY_COL, ACCESS_COL])

    if EXCLUDE_ZERO_ACCESS_FOR_URBANITY:
        n_before = len(merged)
        merged = merged[merged[ACCESS_COL] > 0].copy()
        print(f"Removed PC4s with zero EV accessibility: {n_before - len(merged):,}; retained: {len(merged):,}")
    save_csv(merged, TABLES / "pc4_urbanity_ev_accessibility_merged.csv")

    corr = pd.DataFrame([
        {
            "relationship": f"urbanity vs {label}",
            "method": method.capitalize(),
            "correlation": merged[URBANITY_COL].corr(merged[col], method=method),
            "n": len(merged),
        }
        for label, col in [("raw EV accessibility", ACCESS_COL), ("log EV accessibility", LOG_COL),
                           ("percentile-normalised EV accessibility", PERCENTILE_COL)]
        for method in ["pearson", "spearman"]
    ])
    save_csv(corr, TABLES / "urbanity_ev_accessibility_correlation.csv")
    print(f"\n{corr}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged[URBANITY_COL], merged[LOG_COL], alpha=0.4, s=20)
    ax.set_title("Urbanity vs log EV charging accessibility")
    ax.set_xlabel("Urbanity category")
    ax.set_ylabel(f"log(1 + {ACCESS_COL})")
    ax.grid(True, alpha=0.3)
    save_figure(fig, FIGURES / "urbanity_vs_ev_accessibility_scatter.png")

    categories = sorted(merged[URBANITY_COL].dropna().unique())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [merged.loc[merged[URBANITY_COL] == c, ACCESS_COL].values for c in categories],
        tick_labels=categories, showfliers=False, whis=(0, 100),  # whiskers span the actual min and max
    )
    ax.set_title("EV charging accessibility by urbanity category")
    ax.set_xlabel("Urbanity category")
    ax.set_ylabel(ACCESS_COL)
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FIGURES / "urbanity_vs_ev_accessibility_boxplot.png")


def main():
    paths.require(paths.EV_ACCESSIBILITY_CSV, paths.PC4_STATS_CSV)
    ev = load_and_enrich()
    gini = lorenz_analysis(ev)
    urbanity_analysis(ev)

    header("DONE")
    print(f"Index: {ACCESS_COL}\nGini: {gini:.4f}\nTables: {TABLES}\nFigures: {FIGURES}")


if __name__ == "__main__":
    main()
