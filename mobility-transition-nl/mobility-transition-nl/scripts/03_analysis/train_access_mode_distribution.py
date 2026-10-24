"""
Mode used immediately before the first train leg, across all ODiN journeys containing a train leg.

Input:   data/processed/ODiN2022_2023_pt.csv (leg level)
Outputs: outputs/tables/train_access_mode_distribution/, outputs/figures/train_access_mode_distribution/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import detect_csv_format, header, read_csv_auto, save_csv, save_figure
from mobility_transition.settings import RVM_CAR, RVM_CYCLE, RVM_FEEDER_PT, RVM_LABELS, RVM_TRAIN, RVM_WALK
from mobility_transition.stats import equalise_year_weights

TABLES = paths.step_dir(paths.TABLES, "train_access_mode_distribution")
FIGURES = paths.step_dir(paths.FIGURES, "train_access_mode_distribution")

PERSON, JOURNEY, YEAR, LEG_NR, MODE, ROLE, WEIGHT = "OPID", "VerplID", "Jaar", "RitNr", "Rvm", "RvmRol", "FactorV"
EQUALISE_YEAR_WEIGHTS = True  # each pooled survey year contributes the same total weight


def clean_identifier(series):
    return (series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}))


def round_to_100(percentages, decimals=1):
    """Display rounding that still sums to exactly 100 (largest-remainder method); exact values are untouched."""
    values = pd.to_numeric(percentages, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(values) == 0 or values.sum() <= 0:
        return pd.Series(np.zeros(len(values)), index=percentages.index, dtype=float)

    scaled = 100.0 * values / values.sum() * 10 ** decimals
    units = np.floor(scaled).astype(int)
    remaining = int(round(100.0 * 10 ** decimals)) - int(units.sum())
    remainders = scaled - units
    if remaining > 0:
        units[np.argsort(-remainders, kind="stable")[:remaining]] += 1
    elif remaining < 0:
        units[np.argsort(remainders, kind="stable")[:abs(remaining)]] -= 1
    return pd.Series(units / 10 ** decimals, index=percentages.index, dtype=float)


def weighted_distribution(data, category):
    w = pd.to_numeric(data["analysis_weight"], errors="coerce")
    data = data[w.notna() & np.isfinite(w) & w.gt(0)]
    if data.empty:
        raise ValueError("No positively weighted train journeys remain.")

    table = data.groupby(category, dropna=False).agg(
        n_train_journeys=(JOURNEY, "size"), n_unique_respondents=(PERSON, "nunique"), weighted_train_journeys=("analysis_weight", "sum")
    ).reset_index()
    table["weighted_percent"] = 100.0 * table["weighted_train_journeys"] / table["weighted_train_journeys"].sum()
    table["unweighted_percent"] = 100.0 * table["n_train_journeys"] / table["n_train_journeys"].sum()
    table["weighted_percent_display"] = round_to_100(table["weighted_percent"])
    table["unweighted_percent_display"] = round_to_100(table["unweighted_percent"])

    if not np.isclose(table["weighted_percent"].sum(), 100.0, atol=1e-10) or not np.isclose(table["weighted_percent_display"].sum(), 100.0, atol=1e-10):
        raise AssertionError("Weighted percentages do not sum to 100%.")
    return table.sort_values("weighted_percent", ascending=False)


def detailed_access_mode(mode, role):
    if pd.isna(mode):
        return "Train as first recorded leg"
    mode = int(mode)
    if mode == RVM_CAR:
        if pd.isna(role):
            return "Passenger car"
        return {1: "Car driver", 2: "Car passenger"}.get(int(role), f"Passenger car, role code {int(role)}")
    return RVM_LABELS.get(mode, f"Other mode: Rvm {mode}") if mode != RVM_TRAIN else f"Other mode: Rvm {mode}"


def broad_access_mode(mode):
    if pd.isna(mode):
        return "Train as first recorded leg"
    mode = int(mode)
    if mode == RVM_WALK:
        return "Walking"
    if mode in RVM_CYCLE:
        return "Bicycle"
    if mode in RVM_FEEDER_PT:
        return "Bus, tram or metro"
    return "Passenger car" if mode == RVM_CAR else "Other mode"


def load_legs():
    encoding, sep = detect_csv_format(paths.ODIN_TRIPS_CSV)
    columns = pd.read_csv(paths.ODIN_TRIPS_CSV, encoding=encoding, sep=sep, nrows=0).columns.str.strip()
    required = [PERSON, JOURNEY, LEG_NR, MODE, WEIGHT]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Required columns are missing:\n{missing}")

    legs = read_csv_auto(paths.ODIN_TRIPS_CSV, usecols=required + [c for c in [YEAR, ROLE] if c in columns])
    print(f"Leg rows loaded: {len(legs):,}")
    legs[PERSON], legs[JOURNEY] = clean_identifier(legs[PERSON]), clean_identifier(legs[JOURNEY])
    numeric = [LEG_NR, MODE, WEIGHT] + [c for c in [YEAR, ROLE] if c in legs.columns]
    legs[numeric] = legs[numeric].apply(pd.to_numeric, errors="coerce")
    legs = legs.dropna(subset=[PERSON, JOURNEY, LEG_NR, MODE])
    legs = legs[legs[LEG_NR].between(1, 30)].copy()
    legs[[LEG_NR, MODE]] = legs[[LEG_NR, MODE]].astype(int)
    return legs


def main():
    paths.require(paths.ODIN_TRIPS_CSV)
    legs = load_legs()

    # Year is part of the journey key so 2022 and 2023 IDs can't be merged
    key = ([YEAR] if YEAR in legs.columns else []) + [PERSON, JOURNEY]
    legs = legs.sort_values(key + [LEG_NR]).drop_duplicates(subset=key + [LEG_NR], keep="first")

    records = []
    for _, journey in legs.groupby(key, sort=False):
        journey = journey.sort_values(LEG_NR)
        trains = np.flatnonzero(journey[MODE].eq(RVM_TRAIN).to_numpy())
        if len(trains) == 0:
            continue
        first = int(trains[0])
        train_leg = journey.iloc[first]
        if first == 0:
            mode = role = np.nan
        else:
            prev = journey.iloc[first - 1]
            mode, role = prev[MODE], prev[ROLE] if ROLE in journey.columns else np.nan
        weights = pd.to_numeric(journey[WEIGHT], errors="coerce").dropna()
        record = {
            PERSON: train_leg[PERSON], JOURNEY: train_leg[JOURNEY], "first_train_leg_number": int(train_leg[LEG_NR]),
            "immediate_access_rvm": mode, "immediate_access_rvmrol": role,
            "detailed_access_mode": detailed_access_mode(mode, role), "broad_access_mode": broad_access_mode(mode),
            WEIGHT: weights.iloc[0] if not weights.empty else np.nan,
        }
        if YEAR in journey.columns:
            record[YEAR] = train_leg[YEAR]
        records.append(record)

    journeys = pd.DataFrame(records)
    if journeys.empty:
        raise ValueError("No journeys containing a train leg were found.")
    print(f"Train journeys found: {len(journeys):,}")

    journeys["analysis_weight"] = pd.to_numeric(journeys[WEIGHT], errors="coerce")
    journeys = journeys[journeys["analysis_weight"].gt(0)].copy()
    if EQUALISE_YEAR_WEIGHTS and YEAR in journeys.columns and journeys[YEAR].nunique(dropna=True) > 1:
        journeys["analysis_weight"], factors = equalise_year_weights(journeys["analysis_weight"], journeys[YEAR])
        for year, f in factors.items():
            print(f"Year-weight equalisation factor {int(year)}: {f:.6f}")

    distribution = weighted_distribution(journeys, "broad_access_mode")
    save_csv(distribution, TABLES / "train_access_mode_distribution.csv")
    save_csv(journeys, TABLES / "all_train_journeys_access_mode.csv")

    plot = distribution.sort_values("weighted_percent")
    pos = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(pos, plot["weighted_percent_display"])
    ax.set_yticks(pos)
    ax.set_yticklabels(plot["broad_access_mode"])
    for p, v in zip(pos, plot["weighted_percent_display"]):
        ax.text(v + 0.3, p, f"{v:.1f}%", va="center")
    ax.set_title("Modes used immediately before the first train leg")
    ax.set_xlabel("Weighted share of train journeys (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(plot["weighted_percent_display"].max() * 1.15, 10))
    ax.grid(True, axis="x", alpha=0.3)
    save_figure(fig, FIGURES / "train_access_mode_distribution.png")

    header("TRAIN ACCESS-MODE DISTRIBUTION", 72)
    print(distribution[["broad_access_mode", "n_train_journeys", "n_unique_respondents", "weighted_percent", "weighted_percent_display"]].to_string(index=False))
    print(f"\nWeighted percentage sum (exact): {distribution['weighted_percent'].sum():.12f}%")
    print(f"Weighted percentage sum (displayed): {distribution['weighted_percent_display'].sum():.1f}%")
    print(f"\nTables: {TABLES}\nFigures: {FIGURES}")


if __name__ == "__main__":
    main()
