"""
Group 1 vs Group 2 comparison figures: group shares, and for each mobility group the stated car/BTM/train
frequency distribution as paired stacked bars.

Inputs:  outputs/tables/threshold_groups/03_group_summary_main_and_sensitivity.csv,
         04_frequency_distribution_by_group_mode.csv (from build_threshold_mobility_groups.py)
Outputs: outputs/figures/threshold_groups/paired_group_1_group_2_mobility_profiles/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import header, read_csv_auto, save_figure
from mobility_transition.settings import FREQUENCY_CODES, FREQUENCY_LABELS, MOBILITY_GROUPS, THRESHOLD_GROUP_RENAME

OUT = paths.step_dir(paths.FIGURES, "threshold_groups/paired_group_1_group_2_mobility_profiles")
SPEC_RENAME = {"Main": "Group 1", "Sensitivity": "Group 2"}
SPECS = list(SPEC_RENAME.values())
MODES = ["Car", "BTM", "Train"]
TICK_LABELS = ["Car\nusers", "Multimodal\nusers", "PT\nusers", "Infrequent car\nand PT users"]


def clean_filename(text):
    clean = str(text).lower()
    for old, new in [("+", "plus"), ("&", "and"), ("/", "_"), ("\\", "_"), (" ", "_"), (",", ""), ("-", "_")]:
        clean = clean.replace(old, new)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean.strip("_")


def relabel(data):
    data = data.assign(specification_display=data["specification"].map(SPEC_RENAME), group_display=data["group"].map(THRESHOLD_GROUP_RENAME))
    return data[data["specification_display"].notna() & data["group_display"].notna()].copy()


def metric_for(data, weighted_label, unweighted_label):
    if "share_weighted_percent" in data.columns and data["share_weighted_percent"].notna().any():
        return "share_weighted_percent", weighted_label
    return "share_unweighted_percent", unweighted_label


def plot_group_shares(summary):
    metric, ylabel = metric_for(summary, "Weighted share of respondents (%)", "Share of respondents (%)")
    pivot = summary.pivot(index="group_display", columns="specification_display", values=metric).reindex(MOBILITY_GROUPS).reindex(columns=SPECS)

    x, width = np.arange(len(pivot)), 0.36
    values = {s: pivot[s].fillna(0).to_numpy(dtype=float) for s in SPECS}
    offsets = {"Group 1": -width / 2, "Group 2": width / 2}
    top = max(v.max() for v in values.values())

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for spec in SPECS:
        ax.bar(x + offsets[spec], values[spec], width=width, label=spec)
        for pos, v in zip(x + offsets[spec], values[spec]):
            ax.text(pos, v + top * 0.015, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Mobility group shares")
    ax.set_xlabel("Mobility group")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(TICK_LABELS)
    ax.set_ylim(0, top * 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Grouping")
    save_figure(fig, OUT / "mobility_group_shares_group_1_vs_group_2.png")


def plot_paired_profile(freq, group):
    """For each mode, side-by-side stacked bars of the frequency distribution under Group 1 and Group 2."""
    g = freq[freq["group_display"].eq(group)]
    if g.empty:
        print(f"No frequency data found for: {group}")
        return
    metric, ylabel = metric_for(g, "Weighted share within group (%)", "Share within group (%)")
    index = pd.MultiIndex.from_product([MODES, SPECS, FREQUENCY_CODES], names=["mode", "specification_display", "frequency_code"])
    profile = g.set_index(["mode", "specification_display", "frequency_code"])[metric].reindex(index).fillna(0)

    centres, width = np.arange(len(MODES), dtype=float), 0.32
    positions = {"Group 1": centres - width / 2, "Group 2": centres + width / 2}
    bottoms = {s: np.zeros(len(MODES)) for s in SPECS}
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"][:5]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for code, colour in zip(FREQUENCY_CODES, colours):
        for spec in SPECS:
            values = np.array([profile.loc[(mode, spec, code)] for mode in MODES], dtype=float)
            ax.bar(positions[spec], values, width=width, bottom=bottoms[spec], color=colour,
                   label=FREQUENCY_LABELS[code] if spec == "Group 1" else None)
            bottoms[spec] += values

    ticks = [p for i in range(len(MODES)) for p in (positions["Group 1"][i], positions["Group 2"][i])]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{mode}\n{spec}" for mode in MODES for spec in SPECS])
    ax.set_title(group)
    ax.set_xlabel("Mode and grouping")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Reported frequency", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, OUT / f"{clean_filename(group)}_group_1_group_2_mobility_profile.png")


def main():
    paths.require(paths.THRESHOLD_GROUP_SUMMARY_CSV, paths.THRESHOLD_FREQUENCY_CSV)
    header("PAIRED GROUP 1 AND GROUP 2 MOBILITY PROFILE PLOTS", 78)

    summary = relabel(read_csv_auto(paths.THRESHOLD_GROUP_SUMMARY_CSV))
    freq = relabel(read_csv_auto(paths.THRESHOLD_FREQUENCY_CSV))
    freq["frequency_code"] = pd.to_numeric(freq["frequency_code"], errors="coerce")

    plot_group_shares(summary)
    for group in MOBILITY_GROUPS:
        print(f"Creating paired profile: {group}")
        plot_paired_profile(freq, group)
    print(f"\nFigures: {OUT}")


if __name__ == "__main__":
    main()
