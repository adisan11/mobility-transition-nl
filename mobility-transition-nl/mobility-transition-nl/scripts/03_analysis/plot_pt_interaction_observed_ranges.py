"""
Re-plots the accessibility x urbanity interaction predictions from the PT ordinal models, showing each urbanity
curve only within that group's observed 5th-95th percentile accessibility range (no extrapolation). Models are not refit.

Inputs:  outputs/tables/pt_ordinal_models/07_accessibility_urbanity_interaction_predictions.csv
         data/processed/ODiN2022_2023_mobility_transition_master.csv (observed ranges only)
Outputs: outputs/tables/pt_ordinal_models/, outputs/figures/pt_ordinal_models/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import header, read_csv_auto, save_csv, save_figure
from mobility_transition.settings import URBANITY_LABELS

TABLES = paths.step_dir(paths.TABLES, "pt_ordinal_models")
FIGURES = paths.step_dir(paths.FIGURES, "pt_ordinal_models")

ACCESS_COLUMNS = {"BTM": "pt_access_decay_btm", "Train": "pt_access_decay_train"}
LOWER_Q, UPPER_Q = 0.05, 0.95


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label}:\n{missing}\n\nAvailable columns:\n{list(df.columns)}")


def observed_ranges(master):
    """5th-95th percentile of log accessibility per urbanity group, over unique PC4s (accessibility is a PC4 property)."""
    master["urbanity_group"] = pd.to_numeric(master["urbanity"], errors="coerce").map(URBANITY_LABELS)
    rows = []
    for outcome, col in ACCESS_COLUMNS.items():
        log_col = f"log1p_{col}"
        master[log_col] = np.log1p(pd.to_numeric(master[col], errors="coerce"))
        keep = ["pc4_clean", "urbanity_group", log_col] if "pc4_clean" in master.columns else ["urbanity_group", log_col]
        base = master[keep].dropna()
        if "pc4_clean" in base.columns:
            base = base.drop_duplicates(["pc4_clean", "urbanity_group"])
        for group, g in base.groupby("urbanity_group"):
            lo, hi = g[log_col].quantile(LOWER_Q), g[log_col].quantile(UPPER_Q)
            rows.append({
                "outcome": outcome, "urbanity_group": group,
                "lower_log1p_accessibility": lo, "upper_log1p_accessibility": hi,
                "lower_raw_accessibility": np.expm1(lo), "upper_raw_accessibility": np.expm1(hi),
                "n_spatial_units_or_rows": len(g),
            })
    return pd.DataFrame(rows)


def plot(data, outcome, file_name):
    fig, ax = plt.subplots(figsize=(9, 6))
    for group in URBANITY_LABELS.values():
        g = data[data["urbanity_group"].eq(group)].sort_values("log1p_accessibility")
        if not g.empty:
            ax.plot(g["log1p_accessibility"], 100 * g["predicted_weekly_or_more"], linewidth=2, label=group.split(" ", 1)[1])
    ax.set_title(f"{outcome} accessibility and weekly-or-more use by urbanity\n"
                 "Predictions shown only within each urbanity group's observed 5th–95th percentile range")
    ax.set_xlabel("Log(1 + accessibility index)")
    ax.set_ylabel("Adjusted probability of weekly-or-more use (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Urbanity", fontsize=8)
    save_figure(fig, FIGURES / file_name)
    print(f"Saved: {FIGURES / file_name}")


def main():
    paths.require(paths.PT_ORDINAL_INTERACTION_PREDICTIONS_CSV, paths.MASTER_CSV)
    header("INTERACTION PLOTS WITHIN OBSERVED RANGES")

    pred = read_csv_auto(paths.PT_ORDINAL_INTERACTION_PREDICTIONS_CSV)
    require_columns(pred, ["outcome", "model", "urbanity_group", "log1p_accessibility", "raw_accessibility", "predicted_weekly_or_more"],
                    "interaction prediction file")
    master = read_csv_auto(paths.MASTER_CSV)
    require_columns(master, ["urbanity", *ACCESS_COLUMNS.values()], "master dataset")

    ranges = observed_ranges(master)
    save_csv(ranges, TABLES / "interaction_plot_observed_accessibility_ranges.csv")
    print(f"\nObserved accessibility ranges:\n{ranges.to_string(index=False)}")

    for outcome in ACCESS_COLUMNS:
        merged = pred[pred["outcome"].eq(outcome)].merge(
            ranges[ranges["outcome"].eq(outcome)], on=["outcome", "urbanity_group"], how="left", validate="many_to_one")
        filtered = merged[merged["log1p_accessibility"].between(
            merged["lower_log1p_accessibility"], merged["upper_log1p_accessibility"], inclusive="both")]
        prefix = outcome.lower()
        save_csv(filtered, TABLES / f"{prefix}_interaction_predictions_observed_urbanity_ranges.csv")
        plot(filtered, outcome, f"{prefix}_interaction_observed_urbanity_ranges.png")


if __name__ == "__main__":
    main()
