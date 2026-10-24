"""
How available and how reached is the train, by urbanity?
  Part A: share of PC4s with a station in the rail catchment, and positive train accessibility, by urbanity.
  Part B: from ODiN legs, the access chain before the first train leg of home-origin journeys, by urbanity.

Inputs:  data/processed/pc4_pt_accessibility_distance_decay.csv, data/raw/cbs/pc4_stats.csv,
         data/processed/ODiN2022_2023_pt.csv (leg level)
Outputs: outputs/tables/train_access_urbanity/, outputs/figures/train_access_urbanity/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import clean_pc4, detect_csv_format, find_column, header, read_csv_auto, save_csv, save_figure
from mobility_transition.settings import (
    ODIN_OWN_HOME, RVM_CAR, RVM_CYCLE, RVM_FEEDER_PT, RVM_LABELS, RVM_TRAIN, RVM_WALK,
)
from mobility_transition.stats import equalise_year_weights, weighted_mean, weighted_percentile

TABLES = paths.step_dir(paths.TABLES, "train_access_urbanity")
FIGURES = paths.step_dir(paths.FIGURES, "train_access_urbanity")

TRAIN_ACCESS_COL = "pt_access_decay_train"
URBANITY = "urbanity"
URBANITY_LABELS = {1: "Very highly urban", 2: "Highly urban", 3: "Moderately urban", 4: "Slightly urban", 5: "Non-urban"}
PC4_CANDIDATES = ["pc4_clean", "PC4", "pc4", "postcode4", "Postcode4", "VertPC", "Postal code"]

# ODiN columns
OPID, JOURNEY, YEAR, LEG_ID, LEG_NR = "OPID", "VerplID", "Jaar", "RitID", "RitNr"
DEP_LOC, HOME_PC4, MODE, ROLE, DIST_HM, WEIGHT = "VertLoc", "VertPC", "Rvm", "RvmRol", "AfstR", "FactorV"
DURATION_CANDIDATES = ["RReisduur", "RitReisduur", "RReisduurMin"]
CAR_ROLE_LABELS = {1: "Car driver", 2: "Car passenger", 3: "Car role unknown"}

EQUALISE_YEAR_WEIGHTS = True
MAX_ACCESS_DISTANCE_M = 100_000
MAX_ACCESS_TIME_MIN = 240
MIN_RESPONDENTS = 50  # below this, categories are flagged as unreliable
CHAIN_ORDER = ["Direct walking", "Direct bicycle", "Feeder public transport", "Passenger car",
               "Other or mixed access chain", "Train as first recorded leg", "Pre-train mode missing"]


def clean_identifier(series):
    return (series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}))


def reliability(n):
    return np.where(n >= MIN_RESPONDENTS, f"At least {MIN_RESPONDENTS} unique respondents",
                    f"Fewer than {MIN_RESPONDENTS} unique respondents: interpret cautiously")


# ---------------- access-mode classification ----------------

def immediate_mode_label(mode, role):
    if pd.isna(mode):
        return "Train as first recorded leg"
    mode = int(mode)
    if mode == RVM_CAR:
        return CAR_ROLE_LABELS.get(int(role), f"Passenger car, RvmRol {int(role)}") if pd.notna(role) else "Passenger car, role unavailable"
    return RVM_LABELS.get(mode, f"Rvm code {mode}")


def broad_immediate_mode(mode):
    if pd.isna(mode):
        return "Train as first recorded leg"
    mode = int(mode)
    if mode == RVM_WALK:
        return "Walking"
    if mode in RVM_CYCLE:
        return "Bicycle"
    if mode in RVM_FEEDER_PT:
        return "Feeder public transport"
    return "Passenger car" if mode == RVM_CAR else "Other mode"


def classify_chain(pre_train):
    """Broad category of all legs before the first train leg (PT feeder beats car beats other)."""
    if pre_train.empty:
        return "Train as first recorded leg"
    modes = set(pd.to_numeric(pre_train[MODE], errors="coerce").dropna().astype(int))
    if not modes:
        return "Pre-train mode missing"
    if modes == {RVM_WALK}:
        return "Direct walking"
    if modes <= (RVM_CYCLE | {RVM_WALK}) and modes & RVM_CYCLE:
        return "Direct bicycle"
    if modes & RVM_FEEDER_PT:
        return "Feeder public transport"
    return "Passenger car" if RVM_CAR in modes else "Other or mixed access chain"


def chain_text(pre_train):
    labels = [immediate_mode_label(leg.get(MODE), leg.get(ROLE)) for _, leg in pre_train.iterrows()]
    return " -> ".join(labels + ["Train"])


# ---------------- Part A ----------------

def pc4_train_access(pc4_stats):
    header("PART A: PC4 TRAIN ACCESSIBILITY BY URBANITY", 76)
    access = read_csv_auto(paths.PT_ACCESSIBILITY_CSV)
    if TRAIN_ACCESS_COL not in access.columns:
        raise ValueError(f"'{TRAIN_ACCESS_COL}' is not in the accessibility file.")
    access["pc4_clean"] = clean_pc4(access[find_column(access, PC4_CANDIDATES, "PC4 column")])
    access[TRAIN_ACCESS_COL] = pd.to_numeric(access[TRAIN_ACCESS_COL], errors="coerce")

    pc4 = access[["pc4_clean", TRAIN_ACCESS_COL]].drop_duplicates("pc4_clean").merge(
        pc4_stats[["pc4_clean", URBANITY]].drop_duplicates("pc4_clean"), on="pc4_clean", how="left", validate="one_to_one")
    pc4 = pc4[pc4[URBANITY].isin(URBANITY_LABELS)].copy()
    pc4["has_train_access"] = pc4[TRAIN_ACCESS_COL] > 0

    presence = pc4.groupby(URBANITY).agg(
        n_pc4=("pc4_clean", "size"),
        n_pc4_with_train_access=("has_train_access", "sum"),
        n_pc4_without_train_access=("has_train_access", lambda v: (~v).sum()),
        percent_with_train_access=("has_train_access", lambda v: 100 * v.mean()),
        percent_without_train_access=("has_train_access", lambda v: 100 * (~v).mean()),
    ).reset_index()
    presence["urbanity_label"] = presence[URBANITY].map(URBANITY_LABELS)
    save_csv(presence, TABLES / "train_access_presence_by_urbanity.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(presence[URBANITY], presence["percent_with_train_access"])
    for _, row in presence.iterrows():
        ax.text(row[URBANITY], row["percent_with_train_access"] + 1.5, f"{row['percent_with_train_access']:.1f}%", ha="center", va="bottom")
    ax.set_title("PC4 areas with a railway station inside the active-access catchment")
    ax.set_xlabel("Urbanity category, 1 = most urban and 5 = non-urban")
    ax.set_ylabel("PC4 areas with positive train accessibility (%)")
    ax.set_xticks(list(URBANITY_LABELS))
    ax.set_xticklabels(URBANITY_LABELS.values(), rotation=25, ha="right")
    ax.set_ylim(0, min(105, presence["percent_with_train_access"].max() + 12))
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FIGURES / "figure_train_access_presence_by_urbanity.png")

    # Intensity among PC4s that do have access
    positive = pc4[pc4[TRAIN_ACCESS_COL] > 0].copy()
    positive_table = pd.DataFrame([
        {"urbanity": int(u), "urbanity_label": URBANITY_LABELS.get(int(u), str(u)), "n_positive_access_pc4": len(v),
         "mean_positive_train_access": v.mean(), "median_positive_train_access": v.median(),
         "p25_positive_train_access": v.quantile(0.25), "p75_positive_train_access": v.quantile(0.75),
         "p90_positive_train_access": v.quantile(0.90)}
        for u, g in positive.groupby(URBANITY) for v in [g[TRAIN_ACCESS_COL].dropna()]
    ])
    save_csv(positive_table, TABLES / "positive_train_accessibility_by_urbanity.csv")

    positive["log1p_positive_train_access"] = np.log1p(positive[TRAIN_ACCESS_COL])
    levels = sorted(positive[URBANITY].dropna().unique())
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot([positive.loc[positive[URBANITY].eq(u), "log1p_positive_train_access"].dropna().to_numpy() for u in levels],
               tick_labels=[URBANITY_LABELS[int(u)] for u in levels], showfliers=False)
    ax.set_title("Train accessibility among PC4 areas with positive access")
    ax.set_xlabel("Urbanity category")
    ax.set_ylabel("Log(1 + train accessibility)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FIGURES / "figure_positive_train_accessibility_by_urbanity.png")
    return presence, positive_table


# ---------------- Part B ----------------

def load_legs(pc4_stats):
    header("PART B: HOW RESIDENTS REACH TRAIN STATIONS BY URBANITY", 76)
    encoding, sep = detect_csv_format(paths.ODIN_TRIPS_CSV)
    columns = pd.read_csv(paths.ODIN_TRIPS_CSV, encoding=encoding, sep=sep, nrows=0).columns.str.strip().tolist()
    duration_col = find_column(pd.DataFrame(columns=columns), DURATION_CANDIDATES, "duration", required=False)

    required = [OPID, JOURNEY, LEG_NR, DEP_LOC, HOME_PC4, MODE, DIST_HM, WEIGHT]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Required ODiN leg-level columns are missing:\n{missing}")
    requested = [OPID, JOURNEY, YEAR, LEG_ID, LEG_NR, DEP_LOC, HOME_PC4, MODE, ROLE, DIST_HM, WEIGHT] + ([duration_col] if duration_col else [])
    legs = read_csv_auto(paths.ODIN_TRIPS_CSV, usecols=[c for c in requested if c in columns])
    print(f"ODiN leg rows loaded: {len(legs):,}")
    print(f"Using leg-duration column: {duration_col}" if duration_col else
          "WARNING: no leg-duration column; access-time summaries will be unavailable.")

    legs[OPID], legs[JOURNEY] = clean_identifier(legs[OPID]), clean_identifier(legs[JOURNEY])
    legs["pc4_clean"] = clean_pc4(legs[HOME_PC4])
    numeric = [LEG_NR, DEP_LOC, MODE, DIST_HM, WEIGHT] + [c for c in [YEAR, ROLE, duration_col] if c and c in legs.columns]
    legs[numeric] = legs[numeric].apply(pd.to_numeric, errors="coerce")
    legs = legs.dropna(subset=[OPID, JOURNEY, LEG_NR, DEP_LOC, MODE])
    legs = legs[legs[LEG_NR].between(1, 30)].copy()
    legs[[LEG_NR, MODE]] = legs[[LEG_NR, MODE]].astype(int)

    n_loaded = len(legs)
    legs = legs.merge(pc4_stats[["pc4_clean", URBANITY]].drop_duplicates("pc4_clean"), on="pc4_clean", how="left", validate="many_to_one")
    home = legs[legs[DEP_LOC].eq(ODIN_OWN_HOME) & legs[URBANITY].isin(URBANITY_LABELS)]
    home = home.sort_values([OPID, JOURNEY, LEG_NR] + ([LEG_ID] if LEG_ID in home.columns else []))
    n_before = len(home)
    home = home.drop_duplicates(subset=[OPID, JOURNEY, LEG_NR], keep="first")
    return home, duration_col, {"loaded": n_loaded, "home": n_before, "duplicates": n_before - len(home)}


def extract_access_chains(home, duration_col):
    """One record per home-origin journey containing a train leg, describing the legs before the first train leg."""
    records = []
    for (opid, journey_id), journey in home.groupby([OPID, JOURNEY], sort=False):
        journey = journey.sort_values(LEG_NR)
        is_train = journey[MODE].eq(RVM_TRAIN).to_numpy()
        if not is_train.any():
            continue
        first = np.flatnonzero(is_train)[0]
        train_leg, pre = journey.iloc[first], journey.iloc[:first]

        if pre.empty:
            mode = role = dist = time = np.nan
            total_dist = total_time = np.nan  # train as first leg: access unobserved, not zero
        else:
            last = pre.iloc[-1]
            mode = last[MODE]
            role = last.get(ROLE, np.nan) if ROLE in pre.columns else np.nan
            dist = pd.to_numeric(pd.Series([last[DIST_HM]]), errors="coerce").iloc[0] * 100  # hm -> m
            time = pd.to_numeric(pd.Series([last[duration_col]]), errors="coerce").iloc[0] if duration_col else np.nan
            total_dist = pd.to_numeric(pre[DIST_HM], errors="coerce").clip(lower=0).sum(min_count=1) * 100
            total_time = pd.to_numeric(pre[duration_col], errors="coerce").clip(lower=0).sum(min_count=1) if duration_col else np.nan

        weights = journey[WEIGHT].dropna()
        urbanity = int(train_leg[URBANITY])
        record = {
            OPID: opid, JOURNEY: journey_id,
            YEAR: journey[YEAR].dropna().iloc[0] if YEAR in journey.columns and journey[YEAR].notna().any() else np.nan,
            "pc4_clean": train_leg["pc4_clean"], URBANITY: urbanity, "urbanity_label": URBANITY_LABELS[urbanity],
            "first_train_leg_number": int(train_leg[LEG_NR]), "n_legs_before_train": len(pre),
            "immediate_pre_train_rvm": mode, "immediate_pre_train_rvmrol": role,
            "immediate_pre_train_mode": immediate_mode_label(mode, role), "broad_immediate_mode": broad_immediate_mode(mode),
            "access_chain_category": classify_chain(pre), "access_chain": chain_text(pre),
            "immediate_leg_distance_m": dist, "total_access_distance_m": total_dist,
            "total_access_distance_km": total_dist / 1000 if pd.notna(total_dist) else np.nan,
            "immediate_leg_time_min": time, "total_access_time_min": total_time,
            WEIGHT: weights.iloc[0] if not weights.empty else np.nan,
        }
        if LEG_ID in journey.columns:
            record["first_train_leg_id"] = train_leg[LEG_ID]
        records.append(record)

    journeys = pd.DataFrame(records)
    if journeys.empty:
        raise ValueError("No home-origin journeys containing a train leg were found for urbanity 1-5.")
    print(f"Home-origin train journeys extracted: {len(journeys):,}")
    return journeys


def category_table(data, category):
    g = data.groupby([URBANITY, category], dropna=False).agg(
        n_journeys=(JOURNEY, "size"), n_unique_respondents=(OPID, "nunique"), weighted_journeys=("analysis_weight", "sum")).reset_index()
    g["weighted_percent"] = 100 * g["weighted_journeys"] / g.groupby(URBANITY)["weighted_journeys"].transform("sum")
    g["reliability_flag"] = reliability(g["n_unique_respondents"])
    g["urbanity_label"] = g[URBANITY].map(URBANITY_LABELS)
    return g


def distance_time_summary(journeys, duration_col):
    pct = [0.25, 0.50, 0.75, 0.85, 0.90]
    pct_names = ["p25", "median", "p75", "p85", "p90"]
    rows = []
    for group_cols, label in [([URBANITY], "All home-origin train journeys"), ([URBANITY, "broad_immediate_mode"], "By broad immediate mode"),
                              ([URBANITY, "access_chain_category"], "By full access-chain category")]:
        for keys, g in journeys.groupby(group_cols, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = {"grouping": label, "n_journeys": len(g), "n_unique_respondents": g[OPID].nunique(),
                   "weighted_journeys": pd.to_numeric(g["analysis_weight"], errors="coerce").sum(), **dict(zip(group_cols, keys))}

            d = g[g["valid_access_distance"]]
            row["n_valid_distance_journeys"] = len(d)
            row["weighted_mean_access_distance_m"] = weighted_mean(d["total_access_distance_m"], d["analysis_weight"])
            row.update({f"weighted_{n}_access_distance_m": v for n, v in
                        zip(pct_names, weighted_percentile(d["total_access_distance_m"], d["analysis_weight"], pct))})

            if duration_col:
                t = g[g["valid_access_time"]]
                row["n_valid_time_journeys"] = len(t)
                row["weighted_mean_access_time_min"] = weighted_mean(t["total_access_time_min"], t["analysis_weight"])
                row.update({f"weighted_{n}_access_time_min": v for n, v in
                            zip(pct_names, weighted_percentile(t["total_access_time_min"], t["analysis_weight"], pct))})
            else:
                row["n_valid_time_journeys"] = 0
            row["reliability_flag"] = reliability(row["n_unique_respondents"])
            rows.append(row)
    out = pd.DataFrame(rows)
    out["urbanity_label"] = out[URBANITY].map(URBANITY_LABELS)
    return out


def plot_chain_stacked(chain_table):
    plot = chain_table.pivot(index=URBANITY, columns="access_chain_category", values="weighted_percent").fillna(0).reindex(list(URBANITY_LABELS))
    order = [c for c in CHAIN_ORDER if c in plot.columns]
    plot = plot[order + [c for c in plot.columns if c not in order]]
    plot.index = [URBANITY_LABELS[int(v)] for v in plot.index]

    fig, ax = plt.subplots(figsize=(12, 7))
    plot.plot(kind="bar", stacked=True, ax=ax, width=0.75)
    ax.set_title("Access-chain category before the first train leg by urbanity")
    ax.set_xlabel("Urbanity level")
    ax.set_ylabel("Weighted share of home-origin train journeys (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(plot.index, rotation=25, ha="right")
    ax.legend(title="Access-chain category", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FIGURES / "figure_train_access_chain_category_all_urbanity_stacked.png")


def main():
    paths.require(paths.PT_ACCESSIBILITY_CSV, paths.PC4_STATS_CSV, paths.ODIN_TRIPS_CSV)

    pc4_stats = read_csv_auto(paths.PC4_STATS_CSV)
    if URBANITY not in pc4_stats.columns:
        raise ValueError(f"'{URBANITY}' is not in pc4_stats.csv.")
    pc4_stats["pc4_clean"] = clean_pc4(pc4_stats[find_column(pc4_stats, PC4_CANDIDATES, "PC4 column")])
    pc4_stats[URBANITY] = pd.to_numeric(pc4_stats[URBANITY], errors="coerce")

    presence, positive_table = pc4_train_access(pc4_stats)

    home, duration_col, counts = load_legs(pc4_stats)
    journeys = extract_access_chains(home, duration_col)

    journeys["analysis_weight"] = pd.to_numeric(journeys[WEIGHT], errors="coerce")
    if EQUALISE_YEAR_WEIGHTS and YEAR in journeys.columns and journeys[YEAR].notna().any():
        journeys["analysis_weight"], factors = equalise_year_weights(journeys["analysis_weight"], journeys[YEAR])
        for year, f in factors.items():
            print(f"Year-weight equalisation factor {year}: {f:.6f}")
    journeys["valid_access_distance"] = journeys["total_access_distance_m"].between(0, MAX_ACCESS_DISTANCE_M)
    journeys["valid_access_time"] = journeys["total_access_time_min"].between(0, MAX_ACCESS_TIME_MIN) if duration_col else False
    save_csv(journeys, TABLES / "home_origin_train_access_journeys_all_urbanity.csv")

    immediate = category_table(journeys, "immediate_pre_train_mode").assign(table_role="Supplementary detailed immediate-mode breakdown")
    chain = category_table(journeys, "access_chain_category").assign(table_role="Main descriptive broad access-chain table")
    save_csv(immediate, TABLES / "supplementary_train_immediate_access_mode_table_all_urbanity.csv")
    save_csv(chain, TABLES / "main_train_access_chain_category_table_all_urbanity.csv")

    raw_codes = journeys.groupby([URBANITY, "immediate_pre_train_rvm", "immediate_pre_train_rvmrol", "immediate_pre_train_mode"], dropna=False).agg(
        n_journeys=(JOURNEY, "size"), n_unique_respondents=(OPID, "nunique"), weighted_journeys=("analysis_weight", "sum")).reset_index()
    save_csv(raw_codes, TABLES / "supplementary_rvm_codes_immediately_before_train_all_urbanity.csv")
    save_csv(distance_time_summary(journeys, duration_col), TABLES / "train_access_distance_time_summary_all_urbanity.csv")
    plot_chain_stacked(chain)

    audit = pd.DataFrame(
        [("ODiN leg rows loaded", counts["loaded"]), ("Own-home-origin leg rows in urbanity 1 to 5", counts["home"]),
         ("Duplicate leg-number rows removed", counts["duplicates"]), ("Extracted home-origin train journeys", len(journeys)),
         ("Unique respondents in extracted train journeys", journeys[OPID].nunique())]
        + [(f"Urbanity {u} train journeys", int(journeys[URBANITY].eq(u).sum())) for u in URBANITY_LABELS]
        + [("Journeys with valid access distance", int(journeys["valid_access_distance"].sum())),
           ("Journeys with valid access time", int(pd.Series(journeys["valid_access_time"]).sum()))],
        columns=["metric", "value"],
    )
    save_csv(audit, TABLES / "train_access_journey_sample_audit_all_urbanity.csv")

    header("ANALYSIS COMPLETE", 76)
    for title, table in [("Train-access presence by urbanity", presence), ("Positive train accessibility by urbanity", positive_table),
                         ("Main broad access-chain table", chain), ("Sample audit", audit)]:
        print(f"\n{title}:\n{table.to_string(index=False)}")
    print(f"""
Interpretation:
- Zero train accessibility means no station inside the rail catchment; the positive-access figure excludes those PC4s.
- Journeys start at home; access covers every leg before the first train leg.
- When the train is the first recorded leg, access distance and time are missing, not zero.
- Categories with fewer than {MIN_RESPONDENTS} unique respondents are flagged.""")
    print(f"\nTables: {TABLES}\nFigures: {FIGURES}")


if __name__ == "__main__":
    main()
