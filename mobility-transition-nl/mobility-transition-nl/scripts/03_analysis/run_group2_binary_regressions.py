"""
Four weighted, PC4-clustered binary logits on the Group 2 respondents, with direct train, EV charging and P85
feeder-to-train accessibility plus socioeconomic controls:
  A  Car users vs PT users
  B  Car users vs Multimodal users
  C  Car users vs Infrequent car and PT users
  D  ICEV vs EV households, within Car users
Outcome 1 is always the car-dependent / ICEV category.

PC4 fixed effects are deliberately not used: all accessibility measures are constant within PC4 and would be absorbed.

Input:   data/processed/group2_respondents_with_sustainability_accessibility.csv
Outputs: outputs/tables/group2_regressions/ (incl. thesis-style tables as CSV and LaTeX),
         outputs/figures/group2_regressions/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import header, read_csv_auto, save_csv, save_figure
from mobility_transition.models import CONVERGENCE_KEYS, PROBABILITY_FLOOR, coefficient_stats, create_design_matrix, fit_clustered_logit
from mobility_transition.odin import clean_opid, household_car_status
from mobility_transition.settings import URBANITY_LABELS
from mobility_transition.stats import weighted_mean, weighted_share

TABLES = paths.step_dir(paths.TABLES, "group2_regressions")
FIGURES = paths.step_dir(paths.FIGURES, "group2_regressions")

ID, GROUP, WEIGHT, PC4 = "OPID", "mobility_group_2", "FactorP", "pc4_clean"
CARS, FUEL = "HHAuto", "BrandstofPa1"
TRAIN_PRESENT, TRAIN_INTENSITY, EV_Z, FEEDER_Z = "has_direct_train_access", "train_positive_intensity_z", "ev_access_z", "train_feeder_p85_access_z"
RAW_TRAIN, RAW_EV, RAW_FEEDER = "pt_access_decay_train", "ev_access_capacity_decay_strict_10km", "train_feeder_p85_access_decay"
MINIMUM_AGE, MAXIMUM_AGE = 18, 96
MAX_ITER, GTOL = 1500, 1e-8

COMPARISONS = {
    "A_Car_users_vs_PT_users": {"code": "A", "target": "Car users", "comparator": "PT users", "outcome": "outcome_car_vs_pt",
                                "comparator_group": "PT users", "figure": "figure_A_car_users_vs_pt_users_odds_ratios.png"},
    "B_Car_users_vs_Multimodal_users": {"code": "B", "target": "Car users", "comparator": "Multimodal users", "outcome": "outcome_car_vs_multimodal",
                                        "comparator_group": "Multimodal users", "figure": "figure_B_car_users_vs_multimodal_users_odds_ratios.png"},
    "C_Car_users_vs_Infrequent_car_and_PT_users": {"code": "C", "target": "Car users", "comparator": "Infrequent car and PT users",
                                                   "outcome": "outcome_car_vs_infrequent", "comparator_group": "Infrequent car and PT users",
                                                   "figure": "figure_C_car_users_vs_infrequent_users_odds_ratios.png"},
    "D_ICEV_vs_EV_within_Car_users": {"code": "D", "target": "ICEV Car users", "comparator": "EV Car users",
                                      "outcome": "outcome_icev_vs_ev_within_car_users", "comparator_group": None,
                                      "figure": "figure_D_icev_vs_ev_within_car_users_odds_ratios.png"},
}

HOUSEHOLD = {1: "One-person household", 2: "Couple", 3: "Couple + child(ren)", 4: "Couple + child(ren) + other(s)", 5: "Pair + other(s)",
             6: "Single-parent household + child(ren)", 7: "Single-parent household + child(ren) + other(s)", 8: "Other household"}
CATEGORIES = {
    "income_group": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "age_group": ["18-29", "30-44", "45-64", "65+"],
    "gender_group": ["Male", "Female"],
    "household_composition_group": list(HOUSEHOLD.values()),
    "paid_work_group": ["No paid work", "Paid work"],
    "urbanity_group": list(URBANITY_LABELS.values()),
    "year_group": ["2022", "2023"],
}
REFS = {"income_group": "Q3", "age_group": "30-44", "gender_group": "Male", "household_composition_group": "One-person household",
        "paid_work_group": "No paid work", "year_group": "2022", "urbanity_group": "3 Moderately urban"}
TERM_PREFIX = {col: f'C({col}, Treatment(reference="{ref}"))[T.' for col, ref in REFS.items()}
FORMULA = (f"1 + {TRAIN_PRESENT} + {TRAIN_INTENSITY} + {EV_Z} + {FEEDER_Z}"
           + "".join(f' + C({col}, Treatment(reference="{ref}"))' for col, ref in REFS.items()))
MODEL_COLUMNS = [TRAIN_PRESENT, TRAIN_INTENSITY, EV_Z, FEEDER_Z, *REFS]

ACCESS_LABELS = {TRAIN_PRESENT: "Direct train access present", TRAIN_INTENSITY: "Train accessibility intensity, where present",
                 EV_Z: "EV-charging accessibility, 10 km", FEEDER_Z: "P85 BTM feeder-to-train accessibility"}
LABEL_PREFIX = {"income_group": "Income: ", "age_group": "Age: ", "gender_group": "Gender: ", "household_composition_group": "Household composition: ",
                "paid_work_group": "Paid work: ", "year_group": "Year: ", "urbanity_group": "Urbanity: "}

# Rows for the thesis-style socioeconomic table: (term, label)
SOCIOECONOMIC_TERMS = (
    [(TERM_PREFIX["income_group"] + f"{q}]", f"Income {q}") for q in ["Q1", "Q2", "Q4", "Q5"]]
    + [(TERM_PREFIX["age_group"] + f"{a}]", f"Age {a}") for a in ["18-29", "45-64", "65+"]]
    + [(TERM_PREFIX["gender_group"] + "Female]", "Female")]
    + [(TERM_PREFIX["household_composition_group"] + f"{h}]", h) for h in list(HOUSEHOLD.values())[1:]]
    + [(TERM_PREFIX["paid_work_group"] + "Paid work]", "Paid work")]
    + [(TERM_PREFIX["urbanity_group"] + f"{u}]", u.split(" ", 1)[1]) for u in URBANITY_LABELS.values() if not u.startswith("3")]
)
INFRASTRUCTURE_TERMS = list(ACCESS_LABELS.items())


def significance_stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ---------------- preparation ----------------

def prepare_respondents(data):
    data = data.copy()
    numeric = [WEIGHT, "Leeftijd", "Geslacht", "income_quintile", "HHSam", "BetWerk", "urbanity", "Jaar", CARS, FUEL,
               TRAIN_PRESENT, TRAIN_INTENSITY, EV_Z, FEEDER_Z, RAW_TRAIN, RAW_EV, RAW_FEEDER]
    present = [c for c in numeric if c in data.columns]
    data[present] = data[present].apply(pd.to_numeric, errors="coerce")
    data[ID] = clean_opid(data[ID])
    data[PC4] = data[PC4].astype("string").str.strip()
    if data[ID].isna().any():
        raise ValueError("The merged respondent file contains missing OPIDs.")
    if data[ID].duplicated().any():
        raise ValueError("The merged respondent file contains duplicate OPIDs; one row per respondent is expected.")
    data[WEIGHT] = data[WEIGHT].where(data[WEIGHT] > 0)

    data["household_car_status"] = household_car_status(data[CARS], data[FUEL])

    car_user = data[GROUP].eq("Car users")
    for spec in COMPARISONS.values():
        if spec["comparator_group"]:
            data[spec["outcome"]] = np.where(car_user, 1, np.where(data[GROUP].eq(spec["comparator_group"]), 0, np.nan))
        else:  # ICEV vs EV, Car users only
            status = data["household_car_status"]
            data[spec["outcome"]] = np.where(car_user & status.eq("ICEV"), 1, np.where(car_user & status.eq("EV"), 0, np.nan))

    data["income_group"] = data["income_quintile"].map({i: f"Q{i}" for i in range(1, 6)})
    data["age_group"] = pd.cut(data["Leeftijd"].where(data["Leeftijd"].between(MINIMUM_AGE, MAXIMUM_AGE)),
                               bins=[MINIMUM_AGE - 0.001, 29, 44, 64, MAXIMUM_AGE], labels=CATEGORIES["age_group"], include_lowest=True)
    data["gender_group"] = data["Geslacht"].map({1: "Male", 2: "Female"})
    data["household_composition_group"] = data["HHSam"].map(HOUSEHOLD)  # code 9 (unknown) -> missing
    # BetWerk: 0 = no paid work, 1-3 = paid work, 4/5 = missing
    data["has_paid_work"] = np.select([data["BetWerk"].eq(0), data["BetWerk"].isin([1, 2, 3])], [0.0, 1.0], default=np.nan)
    data["paid_work_group"] = data["has_paid_work"].map({0.0: "No paid work", 1.0: "Paid work"})
    data["urbanity_group"] = data["urbanity"].map(URBANITY_LABELS)
    data["year_group"] = data["Jaar"].round().astype("Int64").astype("string")
    for col, cats in CATEGORIES.items():
        data[col] = pd.Categorical(data[col], categories=cats)
    return data


def model_sample(data, outcome):
    required = [outcome, WEIGHT, PC4, *MODEL_COLUMNS]
    sample = data.loc[data[outcome].notna()]
    sample = sample.loc[sample[required].notna().all(axis=1)].copy()
    sample[outcome] = sample[outcome].astype(int)
    if sample[outcome].nunique() != 2:
        raise ValueError(f"{outcome} does not contain both outcome groups after complete-case filtering.")
    sample["model_weight"] = sample[WEIGHT] / sample[WEIGHT].mean()
    return sample


# ---------------- models ----------------

def fit_model(sample, name, spec):
    design, _, columns = create_design_matrix(FORMULA, sample)
    y = sample[spec["outcome"]].to_numpy(dtype=float)
    w = sample["model_weight"].to_numpy(dtype=float)
    print(f"\nFitting {name}: N={len(sample):,}, clusters={sample[PC4].nunique():,}, predictors={design.shape[1]}")
    fitted = fit_clustered_logit(design.to_numpy(dtype=float), y, w, sample[PC4].astype("string"), MAX_ITER, GTOL)

    rate = np.clip(np.average(y, weights=w), PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    null_ll = np.sum(w * (y * np.log(rate) + (1 - y) * np.log(1 - rate)))  # intercept-only, same sample and weights
    fitted.update(names=columns, null_log_likelihood=null_ll, mcfadden_pseudo_r2=1 - fitted["log_likelihood"] / null_ll,
                  event_share_weighted=weighted_mean(sample[spec["outcome"]], sample[WEIGHT]))
    return fitted


def tidy(fitted, name, spec):
    base = {"model": name, "model_code": spec["code"]}
    labels = {"outcome_1": spec["target"], "outcome_0": spec["comparator"]}
    rows = []
    for term, est, se in zip(fitted["names"], fitted["beta"], fitted["standard_errors"]):
        z, p, lo, hi = coefficient_stats(est, se)
        odds = np.exp(est)
        if term == "Intercept":
            interpretation = "Model intercept; not interpreted substantively."
        elif pd.isna(p) or p >= 0.05:
            interpretation = "No statistically significant association at the 5% level."
        else:
            interpretation = (f"Associated with {abs((odds - 1) * 100):.1f}% {'higher' if odds > 1 else 'lower'} odds of belonging to "
                              f"{spec['target']} rather than {spec['comparator']}, holding the other model variables constant.")
        rows.append({**base, **labels, "term": term, "coefficient": est, "cluster_robust_se": se, "z": z, "p_value": p,
                     "significance": significance_stars(p), "odds_ratio": odds, "ci95_lower_odds_ratio": np.exp(lo),
                     "ci95_upper_odds_ratio": np.exp(hi), "interpretation": interpretation})

    fit_row = {**base, **labels, "comparison_group": spec["comparator"], "formula": FORMULA, "n_unweighted": fitted["n"],
               "n_pc4_clusters": fitted["n_clusters"],
               **{k: fitted[k] for k in ["n_parameters", "event_share_unweighted", "event_share_weighted", "log_likelihood",
                                         "null_log_likelihood", "mcfadden_pseudo_r2", "aic", "bic"]},
               **{k: fitted[k] for k in CONVERGENCE_KEYS}}
    return rows, fit_row, {**base, **{k: fitted[k] for k in CONVERGENCE_KEYS}}


def sample_audits_and_profiles(respondents):
    sample_rows, profile_rows, category_rows = [], [], []
    for name, spec in COMPARISONS.items():
        out = spec["outcome"]
        binary = respondents[respondents[out].notna()]
        sample = model_sample(respondents, out)
        base = {"model": name, "model_code": spec["code"]}
        for stage, n in [("Binary comparison before complete-case filtering", len(binary)),
                         (f"Outcome 1 before complete-case filtering: {spec['target']}", int(binary[out].eq(1).sum())),
                         (f"Outcome 0 before complete-case filtering: {spec['comparator']}", int(binary[out].eq(0).sum())),
                         ("Complete-case regression sample", len(sample)),
                         (f"Outcome 1 in complete-case sample: {spec['target']}", int(sample[out].eq(1).sum())),
                         (f"Outcome 0 in complete-case sample: {spec['comparator']}", int(sample[out].eq(0).sum()))]:
            sample_rows.append({**base, "stage": stage, "n": n})

        for value, label in [(1, spec["target"]), (0, spec["comparator"])]:
            g = sample[sample[out].eq(value)]
            w = g[WEIGHT]
            profile_rows.append({
                **base, "group": label, "n_unweighted": len(g), "weighted_observations": w.sum(),
                "share_direct_train_access": weighted_mean(g[TRAIN_PRESENT], w),
                "mean_train_positive_intensity_z": weighted_mean(g[TRAIN_INTENSITY], w),
                "mean_ev_access_z": weighted_mean(g[EV_Z], w), "mean_train_feeder_p85_access_z": weighted_mean(g[FEEDER_Z], w),
                "mean_income_quintile": weighted_mean(g["income_quintile"], w), "mean_age": weighted_mean(g["Leeftijd"], w),
                "female_share": weighted_share(g["gender_group"].astype("string").eq("Female"), w),
                "paid_work_share": weighted_share(g["has_paid_work"], w), "mean_urbanity_code": weighted_mean(g["urbanity"], w),
            })
            for variable, col in [("Household composition", "household_composition_group"), ("Paid work", "paid_work_group")]:
                values = g[col].astype("string")
                for cat in g[col].cat.categories:
                    category_rows.append({**base, "group": label, "variable": variable, "category": cat,
                                          "n_unweighted": int(values.eq(cat).sum()), "weighted_share": weighted_share(values.eq(cat), w)})

    save_csv(pd.DataFrame(sample_rows), TABLES / "02_comparison_sample_audit.csv")
    save_csv(pd.DataFrame(profile_rows), TABLES / "03_weighted_target_comparator_profiles.csv")
    save_csv(pd.DataFrame(category_rows), TABLES / "04_household_composition_and_work_distributions.csv")


def clean_term_label(term):
    if term in ACCESS_LABELS:
        return ACCESS_LABELS[term]
    for col, prefix in TERM_PREFIX.items():
        term = term.replace(prefix, LABEL_PREFIX[col])
    return term.rstrip("]")


def plot_odds_ratios(coefficients, name, spec):
    ci = ["odds_ratio", "ci95_lower_odds_ratio", "ci95_upper_odds_ratio"]
    d = coefficients[coefficients["model"].eq(name) & ~coefficients["term"].eq("Intercept")]
    d = d[np.isfinite(d[ci]).all(axis=1)].copy()
    if d.empty:
        print(f"No valid coefficients to plot for {name}.")
        return
    d["label"] = d["term"].map(clean_term_label)
    d = d.sort_values("odds_ratio")
    pos = np.arange(len(d))
    odds, lo, hi = (d[c].to_numpy(dtype=float) for c in ci)

    fig, ax = plt.subplots(figsize=(11, max(8, 0.36 * len(d) + 2.5)))
    ax.errorbar(odds, pos, xerr=np.vstack([odds - lo, hi - odds]), fmt="o", capsize=3)
    ax.axvline(1, linestyle="--", linewidth=1)
    ax.set_yticks(pos)
    ax.set_yticklabels(d["label"].tolist())
    ax.set_xscale("log")
    ax.set_title(f"Odds of belonging to {spec['target']}\nReference outcome: {spec['comparator']}")
    ax.set_xlabel("Odds ratio, logarithmic scale")
    ax.grid(True, axis="x", alpha=0.3)
    save_figure(fig, FIGURES / spec["figure"])


def run_models(respondents):
    coef_rows, fit_rows, conv_rows, samples = [], [], [], []
    for name, spec in COMPARISONS.items():
        sample = model_sample(respondents, spec["outcome"])
        sample["comparison_model"], sample["comparison_model_code"] = name, spec["code"]
        sample["comparison_group_label"] = np.where(sample[spec["outcome"]].eq(1), spec["target"], spec["comparator"])
        samples.append(sample)
        rows, fit_row, conv_row = tidy(fit_model(sample, name, spec), name, spec)
        coef_rows.extend(rows)
        fit_rows.append(fit_row)
        conv_rows.append(conv_row)

    coefficients, fit_stats = pd.DataFrame(coef_rows), pd.DataFrame(fit_rows)
    save_csv(coefficients, TABLES / "06_four_model_coefficients.csv")
    save_csv(fit_stats, TABLES / "05_four_model_fit_statistics.csv")
    save_csv(pd.DataFrame(conv_rows), TABLES / "07_four_model_convergence.csv")

    combined = pd.concat(samples, ignore_index=True)
    keep = [ID, PC4, GROUP, "household_car_status", "comparison_model", "comparison_model_code", "comparison_group_label", WEIGHT,
            "income_quintile", "income_group", "Leeftijd", "age_group", "Geslacht", "gender_group", "HHSam", "household_composition_group",
            "BetWerk", "paid_work_group", "urbanity", "urbanity_group", "year_group", RAW_TRAIN, RAW_EV, RAW_FEEDER,
            TRAIN_PRESENT, TRAIN_INTENSITY, EV_Z, FEEDER_Z]
    save_csv(combined[[c for c in keep if c in combined.columns]], TABLES / "08_four_comparison_analysis_respondents.csv")

    for name, spec in COMPARISONS.items():
        plot_odds_ratios(coefficients, name, spec)
    return fit_stats, coefficients


# ---------------- thesis-style tables ----------------

def save_table(table, stem):
    save_csv(table, TABLES / f"{stem}.csv")
    (TABLES / f"{stem}.tex").write_text(table.to_latex(index=False, escape=True), encoding="utf-8")


def overview_table(fit_stats):
    o = fit_stats.sort_values("model_code")
    table = pd.DataFrame({
        "Model": o["model_code"], "Comparison group": o["outcome_0"], "Outcome coded 1": o["outcome_1"],
        "N": o["n_unweighted"].astype(int), "PC4 clusters": o["n_pc4_clusters"].astype(int),
        "Weighted target share (%)": (100 * o["event_share_weighted"]).round(1),
        "McFadden pseudo-R2": o["mcfadden_pseudo_r2"].round(3), "Converged": o["optimizer_success"].map({True: "Yes", False: "No"}),
    })
    save_table(table, "Table_7_like_overview_binary_comparison_models")
    return table


def odds_ratio_table(coefficients, terms, stem, include_ci):
    rows = []
    for term, label in terms:
        row = {"Predictor": label}
        for code in ["A", "B", "C", "D"]:
            m = coefficients[coefficients["model_code"].eq(code) & coefficients["term"].eq(term)]
            if m.empty:
                row[f"Model {code}"] = "NA"
                continue
            r = m.iloc[0]
            ci = f" [{r['ci95_lower_odds_ratio']:.2f}, {r['ci95_upper_odds_ratio']:.2f}]" if include_ci else ""
            row[f"Model {code}"] = f"{r['odds_ratio']:.2f}{ci}{r['significance']}"
        rows.append(row)
    table = pd.DataFrame(rows)
    save_table(table, stem)
    return table


GUIDE = f"""GROUP 2 BINARY LOGISTIC REGRESSIONS — INTERPRETATION GUIDE
{'=' * 78}

OUTCOME CODING
--------------
Model A:
    1 = Car users
    0 = PT users

Model B:
    1 = Car users
    0 = Multimodal users

Model C:
    1 = Car users
    0 = Infrequent car and PT users

Model D:
    Restricted to mobility_group_2 == 'Car users'.
    1 = ICEV household car status
    0 = EV household car status

    ICEV = HHAuto >= 1 and BrandstofPa1 in {{1, 2, 3}}
    EV   = HHAuto >= 1 and BrandstofPa1 == 4

    No-car and unclassified fuel combinations are excluded from Model D.

SUSTAINABILITY-ENABLING ACCESSIBILITY VARIABLES
----------------------------------------------
1. Direct train access present
       has_direct_train_access
       1 if pt_access_decay_train > 0; 0 if the raw index equals zero.

2. Train accessibility intensity, where present
       train_positive_intensity_z
       log1p direct-train accessibility intensity, standardised among positive-access
       PC4s. Zero-access PC4s are assigned zero and represented separately by the
       direct-train presence indicator.

3. EV-charging accessibility, 10 km
       ev_access_z
       log1p ev_access_capacity_decay_strict_10km, standardised across unique PC4s.

4. P85 BTM feeder-to-train accessibility
       train_feeder_p85_access_z
       log1p train_feeder_p85_access_decay, standardised across unique PC4s.

SOCIOECONOMIC CONTROLS AND REFERENCES
-------------------------------------
Income:
    Reference = Q3

Age:
    Reference = 30-44

Gender:
    Reference = Male

Household composition (HHSam):
    Reference = One-person household (code 1)
    Code 9 unknown is treated as missing.

Paid work:
    Reference = No paid work
    BetWerk 0 = no paid work
    BetWerk 1, 2 or 3 = paid work
    BetWerk 4 and 5 = missing

Urbanity:
    Reference = Moderately urban

Year:
    Reference = 2022

PC4 FIXED-EFFECTS IDENTIFICATION NOTE
-------------------------------------
PC4 fixed effects are NOT included. The four accessibility predictors are merged at
PC4 level and are therefore constant for all respondents within a given PC4. A full
set of PC4 fixed effects would absorb all between-PC4 accessibility variation, making
the accessibility coefficients unidentified/perfectly collinear with the PC4 dummies.

The models instead use PC4-clustered robust standard errors and control for urbanity,
year, and the listed respondent socioeconomic characteristics. This is the comparable
structure to the earlier weighted PC4-clustered binary comparison models.

INTERPRETATION
--------------
Odds ratio above 1:
    Higher odds of belonging to the outcome coded 1 rather than the comparison group.

Odds ratio below 1:
    Lower odds of belonging to the outcome coded 1; equivalently, relatively greater
    odds of belonging to the comparison group.

The accessibility odds ratios for the three standardised intensity measures represent
a one-spatial-standard-deviation increase in the relevant log-transformed PC4 index.

Significance levels:
    *   p < 0.05
    **  p < 0.01
    *** p < 0.001

These models estimate associations, not causal effects or observed transitions.
"""


def main():
    paths.require(paths.GROUP2_ANALYSIS_CSV)
    header("GROUP 2 BINARY LOGISTIC REGRESSIONS — SUSTAINABILITY ACCESSIBILITY", 78)

    raw = read_csv_auto(paths.GROUP2_ANALYSIS_CSV)
    required = [ID, GROUP, WEIGHT, PC4, CARS, FUEL, "Leeftijd", "Geslacht", "income_quintile", "HHSam", "BetWerk", "urbanity", "Jaar",
                TRAIN_PRESENT, TRAIN_INTENSITY, EV_Z, FEEDER_Z]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Merged Group 2 respondent-accessibility file is missing required columns:\n{missing}")

    respondents = prepare_respondents(raw)
    sample_audits_and_profiles(respondents)
    fit_stats, coefficients = run_models(respondents)

    overview = overview_table(fit_stats)
    socioeconomic = odds_ratio_table(coefficients, SOCIOECONOMIC_TERMS, "Table_9_like_selected_socioeconomic_predictors", include_ci=False)
    infrastructure = odds_ratio_table(coefficients, INFRASTRUCTURE_TERMS, "Table_infrastructure_sustainability_access_predictors", include_ci=True)

    car_user = respondents[GROUP].eq("Car users")
    status = respondents["household_car_status"]
    hhsam = pd.to_numeric(raw["HHSam"], errors="coerce")
    audit = pd.DataFrame(
        [("Respondent rows loaded", len(raw)), ("Unique nonmissing OPIDs", raw[ID].nunique(dropna=True)),
         ("Rows with nonmissing mobility_group_2", int(raw[GROUP].notna().sum()))]
        + [(g, int(raw[GROUP].eq(g).sum())) for g in ["Car users", "PT users", "Multimodal users", "Infrequent car and PT users"]]
        + [("Car users classified as household ICEV", int((car_user & status.eq("ICEV")).sum())),
           ("Car users classified as household EV", int((car_user & status.eq("EV")).sum())),
           ("Car users excluded from Model D due to No car/Unclassified status", int((car_user & ~status.isin(["ICEV", "EV"])).sum())),
           ("Rows with HHSam code 9 unknown", int(hhsam.eq(9).sum())), ("Rows with HHSam codes 1-8", int(hhsam.isin(range(1, 9)).sum())),
           ("PC4 fixed effects included", "No - PC4-level accessibility predictors are constant within PC4 and would be absorbed"),
           ("PC4 cluster-robust standard errors", "Yes"), ("Year fixed effects", "Yes"), ("Urbanity categorical controls", "Yes")],
        columns=["metric", "value"],
    )
    save_csv(audit, TABLES / "01_input_and_variable_audit.csv")
    (TABLES / "09_model_interpretation_and_fixed_effects_note.txt").write_text(GUIDE, encoding="utf-8")

    for title, table in [("TABLE 7-LIKE MODEL OVERVIEW", overview), ("SOCIOECONOMIC ODDS-RATIO TABLE", socioeconomic),
                         ("SUSTAINABILITY INFRASTRUCTURE ODDS-RATIO TABLE", infrastructure)]:
        header(title, 78)
        print(table.to_string(index=False))
    print(f"\nTables: {TABLES}\nFigures: {FIGURES}")


if __name__ == "__main__":
    main()
