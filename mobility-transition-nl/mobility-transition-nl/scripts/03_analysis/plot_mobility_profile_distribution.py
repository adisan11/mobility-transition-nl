"""
Bar chart of the four mobility profiles (weighted share, with respondent counts).

NOTE: the values below are typed in rather than read from the pipeline. They should match the Group 2 summary in
outputs/tables/threshold_groups/03_group_summary_main_and_sensitivity.csv; update them if the grouping changes.

Output: outputs/figures/threshold_groups/mobility_profile_weighted_share.{png,pdf}
"""
import matplotlib.pyplot as plt
import pandas as pd

from mobility_transition import paths

OUT = paths.step_dir(paths.FIGURES, "threshold_groups")

DATA = pd.DataFrame({
    "Mobility profile": ["Car-dependent,\nlow PT use", "Multimodal car\nand PT user", "PT user", "Low car,\nlow PT use"],
    "Respondents": [64_405, 5_880, 12_052, 28_439],
    "Percentage share": [58.1, 4.7, 10.4, 26.8],
})


def main():
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar(DATA["Mobility profile"], DATA["Percentage share"], width=0.65)
    for bar, share, n in zip(bars, DATA["Percentage share"], DATA["Respondents"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{share:.1f}%\nN = {n:,}", ha="center", va="bottom", fontsize=11)

    ax.set_title("Distribution of mobility profiles", fontsize=18, pad=18)
    ax.set_ylabel("Percentage share (%)", fontsize=13)
    ax.set_xlabel("Mobility profile", fontsize=13)
    ax.set_ylim(0, DATA["Percentage share"].max() + 12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "mobility_profile_weighted_share.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "mobility_profile_weighted_share.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {OUT}")


if __name__ == "__main__":
    main()
