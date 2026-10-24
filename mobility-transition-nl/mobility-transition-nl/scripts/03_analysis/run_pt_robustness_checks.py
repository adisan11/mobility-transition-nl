"""
Robustness checks for the PT ordinal models:
  A. practical proportional-odds check (accessibility OR across cumulative cut-offs)
  B. binary weekly-or-more nested models
  C. train catchment sensitivity: 3.5 km vs 4 km (continuous and two-part), if the 4 km index exists

Inputs:  data/processed/ODiN2022_2023_mobility_transition_master.csv
         data/processed/pc4_pt_accessibility_distance_decay_train_4000m.csv (optional, for C)
Outputs: outputs/tables/pt_robustness/, outputs/figures/pt_robustness/
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from mobility_transition import paths
from mobility_transition.io import clean_pc4, find_column, header, read_csv_auto, save_csv, save_figure
from mobility_transition.models import (
    CONVERGENCE_KEYS, coefficient_stats, create_design_matrix, fit_clustered_logit, predict_probability, rebuild_design_matrix,
)
from mobility_transition.settings import PC4_CANDIDATES, URBANITY_LABELS

TABLES = paths.step_dir(paths.TABLES, "pt_robustness")
FIGURES = paths.step_dir(paths.FIGURES, "pt_robustness")

MINIMUM_AGE, MAXIMUM_AGE = 18, 96
ORDERED_CATEGORIES = [1, 2, 3, 4, 5]
WEEKLY_OR_MORE = [4, 5]  # ordered frequency categories that mean weekly or more
MAX_ITER, GTOL = 1000, 1e-8
RUN_BINARY_INTERACTION_MODELS = True
RUN_4000M_CHECKS = True

REFS = {"income": "Q3", "age": "30-44", "gender": "Male", "licence": "Has licence", "car": "1 car", "urbanity": "3 Moderately urban"}
CORE_CONTROLS = (
    f'C(income_group, Treatment(reference="{REFS["income"]}"))'
    f' + C(age_group, Treatment(reference="{REFS["age"]}"))'
    f' + C(gender_group, Treatment(reference="{REFS["gender"]}"))'
    f' + C(licence_group, Treatment(reference="{REFS["licence"]}"))'
    f' + C(car_group, Treatment(reference="{REFS["car"]}"))'
)
URBANITY_TERM = f'C(urbanity_group, Treatment(reference="{REFS["urbanity"]}"))'
CATEGORIES = {
    "age_group": ["18-29", "30-44", "45-64", "65+"],
    "gender_group": ["Male", "Female"],
    "income_group": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "licence_group": ["No licence", "Has licence"],
    "car_group": ["No car", "1 car", "2+ cars"],
    "urbanity_group": list(URBANITY_LABELS.values()),
}
CATCHMENT_LABEL = "Train weekly-or-more catchment sensitivity"


# ---------------- data ----------------

def merge_4000m_accessibility(master):
    if not RUN_4000M_CHECKS or not paths.PT_ACCESSIBILITY_TRAIN4000_CSV.exists():
        print("\n4 km train-accessibility file not found; the 3.5 vs 4 km comparison is skipped."
              "\n  -> create it with: python scripts/02_accessibility/build_pt_accessibility.py --rail-catchment 4000")
        return master, False

    t4 = read_csv_auto(paths.PT_ACCESSIBILITY_TRAIN4000_CSV)
    if "pt_access_decay_train" not in t4.columns:
        raise ValueError("The 4 km accessibility file does not contain 'pt_access_decay_train'.")
    t4["pc4_clean"] = clean_pc4(t4[find_column(t4, PC4_CANDIDATES, "PC4 column")])
    lookup = t4[["pc4_clean", "pt_access_decay_train"]].drop_duplicates("pc4_clean").rename(
        columns={"pt_access_decay_train": "pt_access_decay_train_4000m"})

    master = master.copy()
    master["pc4_clean"] = clean_pc4(master["pc4_clean"])
    master = master.merge(lookup, on="pc4_clean", how="left", validate="many_to_one")
    print(f"\nRespondents matched to 4 km train accessibility: {master['pt_access_decay_train_4000m'].notna().sum():,}")
    return master, True


def prepare_analysis_data(master):
    data = master.copy()
    required = [
        "respondent_id", "pc4_clean", "person_weight", "btm_frequency_ordered", "train_frequency_ordered",
        "pt_access_decay_btm", "pt_access_decay_train", "Geslacht", "Leeftijd", "income_quintile", "OPRijbewijsAu", "HHAuto", "urbanity",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"The master dataset is missing required columns:\n{missing}")

    numeric = required[2:] + (["pt_access_decay_train_4000m"] if "pt_access_decay_train_4000m" in data.columns else [])
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")

    data["btm_outcome"] = data["btm_frequency_ordered"].where(data["btm_frequency_ordered"].isin(ORDERED_CATEGORIES))
    data["train_outcome"] = data["train_frequency_ordered"].where(data["train_frequency_ordered"].isin(ORDERED_CATEGORIES))
    data["age_valid"] = data["Leeftijd"].where(data["Leeftijd"].between(MINIMUM_AGE, MAXIMUM_AGE))
    data["age_group"] = pd.cut(data["age_valid"], bins=[MINIMUM_AGE - 0.001, 29, 44, 64, MAXIMUM_AGE],
                               labels=CATEGORIES["age_group"], include_lowest=True)
    data["gender_group"] = data["Geslacht"].map({1: "Male", 2: "Female"})
    data["income_group"] = data["income_quintile"].map({i: f"Q{i}" for i in range(1, 6)})
    data["licence_group"] = data["OPRijbewijsAu"].map({0: "No licence", 1: "Has licence"})
    cars = data["HHAuto"].where(data["HHAuto"].between(0, 10))
    data["car_group"] = np.select([cars.eq(0), cars.eq(1), cars.ge(2)], ["No car", "1 car", "2+ cars"], default=None)
    data["urbanity_group"] = data["urbanity"].map(URBANITY_LABELS)
    for col, cats in CATEGORIES.items():
        data[col] = pd.Categorical(data[col], categories=cats)

    data["valid_person_weight"] = data["person_weight"].where(data["person_weight"] > 0)
    data["pc4_cluster"] = clean_pc4(data["pc4_clean"])

    # log1p accessibility, standardised across unique PC4s
    scale = {}
    specs = [("pt_access_decay_btm", "btm_log_access", "btm_access_z"), ("pt_access_decay_train", "train_log_access", "train_access_z")]
    two_part = [("pt_access_decay_train", "train_log_access", "has_train_access_3500m", "train_positive_intensity_3500m_z")]
    if "pt_access_decay_train_4000m" in data.columns:
        specs.append(("pt_access_decay_train_4000m", "train_4000m_log_access", "train_4000m_access_z"))
        two_part.append(("pt_access_decay_train_4000m", "train_4000m_log_access", "has_train_access_4000m", "train_positive_intensity_4000m_z"))

    for raw, log_col, z_col in specs:
        data[raw] = data[raw].where(data[raw] >= 0)
        data[log_col] = np.log1p(data[raw])
        pc4 = data[["pc4_cluster", log_col]].dropna().drop_duplicates("pc4_cluster")[log_col]
        mean, sd = pc4.mean(), pc4.std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Cannot standardise {log_col}.")
        data[z_col] = (data[log_col] - mean) / sd
        scale[z_col] = {"raw_column": raw, "log_column": log_col, "pc4_log_mean": mean, "pc4_log_std": sd}

    # Two-part train measure: any direct access, plus intensity standardised among PC4s with access
    for raw, log_col, presence, intensity in two_part:
        data[presence] = np.where(data[raw].notna(), data[raw].gt(0).astype(float), np.nan)
        pos = data.loc[data[raw] > 0, ["pc4_cluster", log_col]].dropna().drop_duplicates("pc4_cluster")[log_col]
        mean, sd = pos.mean(), pos.std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Cannot standardise positive values for {raw}.")
        data[intensity] = 0.0
        positive = data[raw].gt(0)
        data.loc[positive, intensity] = (data.loc[positive, log_col] - mean) / sd
        data.loc[data[raw].isna(), intensity] = np.nan
        scale[intensity] = {"raw_column": raw, "log_column": log_col, "positive_pc4_log_mean": mean, "positive_pc4_log_std": sd}

    return data, scale


def build_common_sample(data, outcome_col, access_cols):
    required = [outcome_col, "valid_person_weight", "pc4_cluster", "income_group", "age_group", "gender_group",
                "licence_group", "car_group", "urbanity_group", *access_cols]
    sample = data.loc[data[required].notna().all(axis=1)].copy()
    sample[outcome_col] = sample[outcome_col].astype(int)
    sample["model_weight"] = sample["valid_person_weight"] / sample["valid_person_weight"].mean()
    return sample


def nested_formulas(access):
    formulas = {
        "M1_access_only": f"1 + {access}",
        "M2_conversion_factors": f"1 + {access} + {CORE_CONTROLS}",
        "M3_add_urbanity": f"1 + {access} + {CORE_CONTROLS} + {URBANITY_TERM}",
    }
    if RUN_BINARY_INTERACTION_MODELS:
        formulas["M4_access_x_urbanity"] = f"1 + {access} * {URBANITY_TERM} + {CORE_CONTROLS}"
    return formulas


def two_part_formula(presence, intensity):
    return f"1 + {presence} + {intensity} + {CORE_CONTROLS} + {URBANITY_TERM}"


# ---------------- model helpers ----------------

def fit(X, y, sample, outcome, model):
    fitted = fit_clustered_logit(X, y, sample["model_weight"], sample["pc4_cluster"], MAX_ITER, GTOL, label=f"{outcome} — {model}")
    fitted.update(outcome=outcome, model=model)
    return fitted


def fit_null(y, sample, outcome):
    return fit(np.ones((len(y), 1)), y, sample, outcome, "Null_intercept_only")


def tidy(fitted, names, population, null_ll):
    rows = []
    for term, est, se in zip(names, fitted["beta"], fitted["standard_errors"]):
        z, p, lo, hi = coefficient_stats(est, se)
        rows.append({
            "outcome": fitted["outcome"], "model": fitted["model"], "term": term, "coefficient": est, "cluster_robust_se": se,
            "z": z, "p_value": p, "ci95_lower_coefficient": lo, "ci95_upper_coefficient": hi,
            "odds_ratio": np.exp(est), "ci95_lower_odds_ratio": np.exp(lo), "ci95_upper_odds_ratio": np.exp(hi),
        })
    fit_row = {
        "outcome": fitted["outcome"], "model": fitted["model"], "n_unweighted": fitted["n"], "n_clusters": fitted["n_clusters"],
        **{k: fitted[k] for k in ["n_parameters", "event_share_unweighted", "log_likelihood", "aic", "bic"]},
        **{k: fitted[k] for k in CONVERGENCE_KEYS},
        "weighted_population": population,
        "mcfadden_pseudo_r2": 1 - fitted["log_likelihood"] / null_ll,
    }
    return rows, fit_row


def normalised_weights(sample):
    w = sample["model_weight"].to_numpy(dtype=float)
    return w / w.sum()


def adjusted_curve(fitted, sample, design_info, columns, access_z, scale, n_grid=60):
    """Average predicted probability when everyone's accessibility is set to each value (PC4 5th-95th percentile)."""
    s = scale[access_z]
    pc4 = sample[["pc4_cluster", s["log_column"]]].dropna().drop_duplicates("pc4_cluster")[s["log_column"]]
    w = normalised_weights(sample)
    rows = []
    for log_value in np.linspace(pc4.quantile(0.05), pc4.quantile(0.95), n_grid):
        scenario = sample.copy()
        scenario[access_z] = (log_value - s["pc4_log_mean"]) / s["pc4_log_std"]
        prob = predict_probability(fitted["beta"], rebuild_design_matrix(design_info, columns, scenario))
        rows.append({"outcome": fitted["outcome"], "model": fitted["model"], "log1p_accessibility": log_value,
                     "raw_accessibility": np.expm1(log_value), "adjusted_probability": np.sum(w * prob)})
    return pd.DataFrame(rows)


def two_part_scenarios(fitted, sample, design_info, columns, raw, log_col, presence, intensity, scale):
    """Predictions at zero access and at percentiles of positive access."""
    positive = sample.loc[sample[raw] > 0, ["pc4_cluster", log_col]].dropna().drop_duplicates("pc4_cluster")[log_col]
    s = scale[intensity]
    w = normalised_weights(sample)
    rows = []
    for name, pct in {"Zero direct access": None, "Positive access P25": 0.25, "Positive access median": 0.50,
                      "Positive access P75": 0.75, "Positive access P90": 0.90}.items():
        scenario = sample.copy()
        if pct is None:
            log_value = raw_value = 0.0
            scenario[presence], scenario[intensity] = 0.0, 0.0
        else:
            log_value = positive.quantile(pct)
            raw_value = np.expm1(log_value)
            scenario[presence] = 1.0
            scenario[intensity] = (log_value - s["positive_pc4_log_mean"]) / s["positive_pc4_log_std"]
        prob = predict_probability(fitted["beta"], rebuild_design_matrix(design_info, columns, scenario))
        rows.append({"outcome": fitted["outcome"], "model": fitted["model"], "scenario": name, "raw_accessibility": raw_value,
                     "log1p_accessibility": log_value, "adjusted_probability": np.sum(w * prob)})
    return pd.DataFrame(rows)


# ---------------- plots ----------------

def plot_po_cutoffs(table, outcome, file_name):
    table = table.sort_values("threshold")
    x = np.arange(len(table))
    est, lo, hi = (table[c].to_numpy() for c in ["odds_ratio", "ci95_lower_odds_ratio", "ci95_upper_odds_ratio"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x, est, yerr=np.vstack([est - lo, hi - est]), fmt="o", capsize=4)
    ax.axhline(1, linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(table["threshold_label"], rotation=15, ha="right")
    ax.set_title(f"{outcome}: accessibility odds ratio across cumulative cut-offs")
    ax.set_xlabel("Binary cumulative outcome")
    ax.set_ylabel("Accessibility odds ratio")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FIGURES / file_name)


def plot_curves(table, title, ylabel, file_name, by_model=False):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    groups = table.groupby("model") if by_model else [(None, table)]
    for name, g in groups:
        g = g.sort_values("log1p_accessibility")
        ax.plot(g["log1p_accessibility"], 100 * g["adjusted_probability"], linewidth=2, label=name)
    ax.set_title(title)
    ax.set_xlabel("Log(1 + train accessibility index)" if by_model else "Log(1 + accessibility index)")
    ax.set_ylabel(ylabel)
    if by_model:
        ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, FIGURES / file_name)


# ---------------- analyses ----------------

PO_THRESHOLDS = {2: "Outcome ≥ 2", 3: "Outcome ≥ 3", 4: "Weekly-or-more (≥ 4)", 5: "Highest-frequency category (= 5)"}


def proportional_odds_check(outcome, sample, outcome_col, access_z):
    """Same model at each cumulative cut-off; a stable accessibility OR supports proportional odds."""
    design, _, columns = create_design_matrix(f"1 + {access_z} + {CORE_CONTROLS} + {URBANITY_TERM}", sample)
    X = design.to_numpy(dtype=float)
    i = columns.index(access_z)
    rows = []
    for threshold, label in PO_THRESHOLDS.items():
        y = sample[outcome_col].ge(threshold).astype(int).to_numpy()
        fitted = fit(X, y, sample, outcome, f"PO_check_threshold_{threshold}")
        est, se = fitted["beta"][i], fitted["standard_errors"][i]
        z = est / se
        p = 2 * (1 - norm.cdf(abs(z)))
        rows.append({
            "outcome": outcome, "threshold": threshold, "threshold_label": label, "event_share_unweighted": y.mean(),
            "accessibility_term": access_z, "coefficient": est, "cluster_robust_se": se, "z": z, "p_value": p,
            "odds_ratio": np.exp(est), "ci95_lower_odds_ratio": np.exp(est - 1.96 * se), "ci95_upper_odds_ratio": np.exp(est + 1.96 * se),
            "optimizer_success": fitted["optimizer_success"], "gradient_infinity_norm": fitted["gradient_infinity_norm"],
        })
    po = pd.DataFrame(rows)

    coef = po["coefficient"]
    summary = {
        "outcome": outcome, "n_cutoffs": len(po),
        "all_coefficients_positive": bool((coef > 0).all()),
        "all_accessibility_terms_p_lt_0_05": bool((po["p_value"] < 0.05).all()),
        "minimum_accessibility_coefficient": coef.min(), "maximum_accessibility_coefficient": coef.max(),
        "coefficient_range": coef.max() - coef.min(), "coefficient_mean": coef.mean(), "coefficient_sd": coef.std(ddof=0),
        "coefficient_cv_absolute": coef.std(ddof=0) / abs(coef.mean()) if coef.mean() != 0 else np.nan,
        "minimum_accessibility_or": po["odds_ratio"].min(), "maximum_accessibility_or": po["odds_ratio"].max(),
        "interpretation_note": "This is a practical coefficient-stability diagnostic, not a formal survey-weighted clustered Brant test.",
    }
    plot_po_cutoffs(po, outcome, f"{outcome.lower()}_proportional_odds_accessibility_or_by_cutoff.png")
    return rows, summary


def weekly_models(outcome, sample, outcome_col, access_z, scale, out):
    label = f"{outcome} weekly-or-more"
    y = sample[outcome_col].isin(WEEKLY_OR_MORE).astype(int).to_numpy()
    null_ll = fit_null(y, sample, label)["log_likelihood"]
    population = sample["valid_person_weight"].sum()

    stored = {}
    for model, formula in nested_formulas(access_z).items():
        design, info, columns = create_design_matrix(formula, sample)
        fitted = fit(design.to_numpy(dtype=float), y, sample, label, model)
        rows, fit_row = tidy(fitted, columns, population, null_ll)
        out["coefficients"].extend(rows)
        out["fit"].append(fit_row)
        out["convergence"].append({"outcome": label, "model": model, **{k: fitted[k] for k in CONVERGENCE_KEYS}})
        stored[model] = (fitted, info, columns)

    fitted, info, columns = stored["M3_add_urbanity"]
    curve = adjusted_curve(fitted, sample, info, columns, access_z, scale)
    out["predictions"].append(curve)
    plot_curves(curve, f"{outcome} weekly-or-more use: binary-logit sensitivity",
                "Adjusted probability of weekly-or-more use (%)", f"{outcome.lower()}_binary_weekly_adjusted_probability.png")


def catchment_comparison(data, scale):
    access_cols = ["train_access_z", "train_4000m_access_z", "has_train_access_3500m", "train_positive_intensity_3500m_z",
                   "has_train_access_4000m", "train_positive_intensity_4000m_z"]
    sample = build_common_sample(data, "train_outcome", access_cols)
    y = sample["train_outcome"].isin(WEEKLY_OR_MORE).astype(int).to_numpy()
    null_ll = fit_null(y, sample, CATCHMENT_LABEL)["log_likelihood"]
    population = sample["valid_person_weight"].sum()

    specs = {
        "Continuous 3.5 km": {"formula": f"1 + train_access_z + {CORE_CONTROLS} + {URBANITY_TERM}", "terms": ["train_access_z"]},
        "Continuous 4.0 km": {"formula": f"1 + train_4000m_access_z + {CORE_CONTROLS} + {URBANITY_TERM}", "terms": ["train_4000m_access_z"]},
        "Two-part 3.5 km": {"two_part": ("pt_access_decay_train", "train_log_access", "has_train_access_3500m", "train_positive_intensity_3500m_z")},
        "Two-part 4.0 km": {"two_part": ("pt_access_decay_train_4000m", "train_4000m_log_access", "has_train_access_4000m", "train_positive_intensity_4000m_z")},
    }
    comparison_rows, prediction_frames = [], []
    for model, spec in specs.items():
        if "two_part" in spec:
            raw, log_col, presence, intensity = spec["two_part"]
            formula, terms = two_part_formula(presence, intensity), [presence, intensity]
        else:
            formula, terms = spec["formula"], spec["terms"]

        design, info, columns = create_design_matrix(formula, sample)
        fitted = fit(design.to_numpy(dtype=float), y, sample, CATCHMENT_LABEL, model)
        rows, fit_row = tidy(fitted, columns, population, null_ll)
        for row in rows:
            if row["term"] in terms:
                comparison_rows.append({**row, **{k: fit_row[k] for k in ["n_unweighted", "n_clusters", "aic", "bic", "mcfadden_pseudo_r2", "optimizer_success"]}})

        if "two_part" in spec:
            prediction_frames.append(two_part_scenarios(fitted, sample, info, columns, raw, log_col, presence, intensity, scale))
        else:
            prediction_frames.append(adjusted_curve(fitted, sample, info, columns, terms[0], scale))
    return comparison_rows, prediction_frames


def main():
    paths.require(paths.MASTER_CSV)
    header("PUBLIC-TRANSPORT ROBUSTNESS CHECKS", 78)

    master, has_4000m = merge_4000m_accessibility(read_csv_auto(paths.MASTER_CSV))
    data, scale = prepare_analysis_data(master)

    out = {"coefficients": [], "fit": [], "convergence": [], "predictions": []}
    audit_rows, po_rows, po_summaries = [], [], []
    for outcome, outcome_col, access_z in [("BTM", "btm_outcome", "btm_access_z"), ("Train", "train_outcome", "train_access_z")]:
        sample = build_common_sample(data, outcome_col, [access_z])
        audit_rows.append({"outcome": outcome, "analysis": "Common robustness sample", "n_unweighted": len(sample),
                           "n_clusters": sample["pc4_cluster"].nunique(), "weighted_population": sample["valid_person_weight"].sum()})
        rows, summary = proportional_odds_check(outcome, sample, outcome_col, access_z)
        po_rows.extend(rows)
        po_summaries.append(summary)
        weekly_models(outcome, sample, outcome_col, access_z, scale, out)

    comparison_rows, catchment_frames = catchment_comparison(data, scale) if has_4000m else ([], [])

    po_terms, coefficients = pd.DataFrame(po_rows), pd.DataFrame(out["coefficients"])
    save_csv(pd.DataFrame(audit_rows), TABLES / "01_robustness_sample_audit.csv")
    save_csv(po_terms, TABLES / "02_proportional_odds_accessibility_terms.csv")
    save_csv(pd.DataFrame(po_summaries), TABLES / "03_proportional_odds_diagnostic_summary.csv")
    save_csv(pd.DataFrame(out["fit"]), TABLES / "04_binary_weekly_model_fit.csv")
    save_csv(coefficients, TABLES / "05_binary_weekly_coefficients.csv")
    save_csv(pd.concat(out["predictions"], ignore_index=True), TABLES / "06_binary_weekly_predictions.csv")
    save_csv(pd.DataFrame(out["convergence"]), TABLES / "07_binary_weekly_convergence.csv")

    if comparison_rows:
        save_csv(pd.DataFrame(comparison_rows), TABLES / "08_train_3500m_vs_4000m_binary_comparison.csv")
    if catchment_frames:
        preds = pd.concat(catchment_frames, ignore_index=True, sort=False)
        save_csv(preds, TABLES / "09_train_3500m_vs_4000m_predictions.csv")
        continuous = preds[preds["model"].str.startswith("Continuous")]
        if not continuous.empty:
            plot_curves(continuous, "Train weekly-or-more use under 3.5 km and 4 km catchments",
                        "Adjusted probability of weekly-or-more train use (%)", "train_3500m_vs_4000m_weekly_probability.png", by_model=True)

    header("ROBUSTNESS CHECKS COMPLETE", 78)
    ci = ["odds_ratio", "ci95_lower_odds_ratio", "ci95_upper_odds_ratio"]
    print("\nProportional-odds diagnostic:")
    print(po_terms[["outcome", "threshold_label", "coefficient", "cluster_robust_se", "p_value", *ci]].to_string(index=False))
    print("\nDiagnostic summary:\n" + pd.DataFrame(po_summaries).to_string(index=False))
    print("\nWeekly-or-more accessibility terms:")
    access_terms = coefficients[coefficients["term"].str.contains("access", case=False, na=False)]
    print(access_terms[["outcome", "model", "term", "coefficient", "cluster_robust_se", "p_value", *ci]].to_string(index=False))
    if comparison_rows:
        print("\nTrain catchment sensitivity terms:")
        print(pd.DataFrame(comparison_rows)[["model", "term", "coefficient", "cluster_robust_se", "p_value", *ci, "aic", "bic", "mcfadden_pseudo_r2"]].to_string(index=False))
    print("""
Interpretation:
- The cumulative binary models check whether the accessibility association changes across outcome cut-offs;
  they are not a formal survey-weighted, PC4-clustered Brant test.
- The weekly-or-more models check whether conclusions hold under a simple binary outcome.
- For 3.5 vs 4 km, compare direction, effect size, predicted probabilities and whether conclusions change.
- Check 07_binary_weekly_convergence.csv before interpreting estimates.""")
    print(f"\nTables: {TABLES}\nFigures: {FIGURES}")


if __name__ == "__main__":
    main()
