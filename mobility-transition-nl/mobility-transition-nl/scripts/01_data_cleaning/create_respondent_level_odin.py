"""
Collapses the ODiN trip file (with mobility groups) to one row per respondent, keeping only columns that are
constant within each OPID. Stops if a key variable (municipality, weight, mobility groups) varies within an OPID.

Input:   data/processed/ODiN2022_2023_mobility_transition_master_with_mobility_groups.csv
Outputs: data/processed/ODiN2022_2023_mobility_transition_respondent_level.csv
         outputs/tables/respondent_level/ (column and respondent audits)
"""
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import header, read_csv_auto, save_csv

input_csv = paths.MASTER_WITH_GROUPS_CSV
output_csv = paths.RESPONDENT_LEVEL_CSV
AUDIT_DIR = paths.step_dir(paths.TABLES, "respondent_level")
column_audit_csv = AUDIT_DIR / "column_audit.csv"
respondent_audit_csv = AUDIT_DIR / "respondent_audit.csv"

OPID_COLUMN = "OPID"
# Needed for the municipality mapping later; the script stops if any vary within an OPID.
REQUIRED_RESPONDENT_COLUMNS = ["WoGem", "FactorP", "mobility_group_1", "mobility_group_2", "mobility_threshold_sample"]
REQUIRED = [OPID_COLUMN, *REQUIRED_RESPONDENT_COLUMNS]
PROGRESS_EVERY_N_COLUMNS = 25
CONFLICT_COL = "n_opids_with_more_than_one_nonmissing_value"


def clean_opid(series):
    # Turn float-read IDs like 12345.0 back into "12345"
    numeric = pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.strip()
    is_int = numeric.notna() & np.isclose(numeric, np.floor(numeric))
    cleaned.loc[is_int] = numeric.loc[is_int].astype("Int64").astype("string")
    return cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def validate_columns(data, required_columns, file_label):
    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns:\n{missing}\n\n"
            f"Available columns:\n{data.columns.tolist()}"
        )


def main():
    # Load
    header("CREATING RESPONDENT-LEVEL ODIN DATASET", 80)
    paths.require(input_csv)
    odin = read_csv_auto(input_csv)
    print(f"Rows: {len(odin):,}  Columns: {len(odin.columns):,}")
    validate_columns(odin, REQUIRED, "Merged ODiN mobility transition master")

    # Clean OPID
    odin[OPID_COLUMN] = clean_opid(odin[OPID_COLUMN])
    n_raw_rows = len(odin)
    n_missing_opid_rows = int(odin[OPID_COLUMN].isna().sum())

    header("RAW OPID DIAGNOSTICS", 80)
    print(f"Raw ODiN rows:                  {n_raw_rows:,}")
    print(f"Rows with missing OPID:         {n_missing_opid_rows:,}")
    if n_missing_opid_rows > 0:
        print("\nRows with missing OPID will be excluded because they cannot be assigned to a respondent.")

    odin = odin.loc[odin[OPID_COLUMN].notna()].copy()
    n_unique_opids = odin[OPID_COLUMN].nunique(dropna=True)
    duplicate_rows_to_remove = len(odin) - n_unique_opids

    print(f"Rows with nonmissing OPID:      {len(odin):,}")
    print(f"Unique OPIDs:                   {n_unique_opids:,}")
    print(f"Repeated OPID rows to collapse: {duplicate_rows_to_remove:,}")

    # Find columns that are constant within each OPID
    header("IDENTIFYING RESPONDENT-CONSTANT COLUMNS", 80)
    candidate_columns = [c for c in odin.columns if c != OPID_COLUMN]
    grouped = odin.groupby(OPID_COLUMN, sort=False, observed=True)

    constant_columns, varying_columns, column_audit_rows = [], [], []
    n_candidate_columns = len(candidate_columns)

    for i, column in enumerate(candidate_columns, start=1):
        unique_counts = grouped[column].nunique(dropna=True)
        n_conflicts = int(unique_counts.gt(1).sum())
        is_constant = n_conflicts == 0
        (constant_columns if is_constant else varying_columns).append(column)

        column_audit_rows.append({
            "column": column,
            "kept_in_respondent_file": int(is_constant),
            CONFLICT_COL: n_conflicts,
            "maximum_unique_nonmissing_values_within_opid": int(unique_counts.max() if len(unique_counts) > 0 else 0),
        })

        if i % PROGRESS_EVERY_N_COLUMNS == 0 or i == n_candidate_columns:
            print(f"Checked {i:,} / {n_candidate_columns:,} columns")

    # Required variables must not vary within OPID
    header("REQUIRED RESPONDENT VARIABLE CHECK", 80)
    audit_lookup = {row["column"]: row for row in column_audit_rows}
    required_conflicts = []

    for column in REQUIRED_RESPONDENT_COLUMNS:
        conflict_count = audit_lookup[column][CONFLICT_COL]
        print(f"{column:<32} conflicting OPIDs: {conflict_count:,}")
        if conflict_count > 0:
            required_conflicts.append(column)

    if required_conflicts:
        conflict_examples = []
        for column in required_conflicts:
            unique_counts = grouped[column].nunique(dropna=True)
            example_opids = unique_counts.loc[unique_counts.gt(1)].head(10).index
            examples = odin.loc[odin[OPID_COLUMN].isin(example_opids), [OPID_COLUMN, column]].copy()
            examples["conflicting_column"] = column
            examples = examples.rename(columns={column: "observed_value"})
            conflict_examples.append(examples[["conflicting_column", OPID_COLUMN, "observed_value"]])

        conflict_examples = pd.concat(conflict_examples, ignore_index=True)
        raise ValueError(
            "One or more required respondent-level variables vary within OPID.\n\n"
            "The respondent file has NOT been created because taking an arbitrary first row could assign "
            "the wrong municipality, weight, or mobility group.\n\n"
            f"Conflicting required columns:\n{required_conflicts}\n\n"
            f"Example conflicting values:\n{conflict_examples.head(50).to_string(index=False)}"
        )

    print("\nAll required respondent variables are constant within OPID.")

    # Collapse to one row per OPID
    header("COLLAPSING TO ONE ROW PER OPID", 80)
    respondents = grouped[constant_columns].first().reset_index()

    if respondents[OPID_COLUMN].duplicated().any():
        raise ValueError("Duplicate OPIDs remain after respondent-level collapse.")
    if len(respondents) != n_unique_opids:
        raise ValueError(
            "Respondent row count does not equal the number of unique OPIDs.\n"
            f"Unique OPIDs:    {n_unique_opids:,}\nRespondent rows: {len(respondents):,}"
        )

    print(f"Respondent-level rows created: {len(respondents):,}")
    print(f"Columns retained:              {len(respondents.columns):,}")
    print(f"Columns removed because they vary within OPID: {len(varying_columns):,}")
    print("\nConfirmed: each OPID appears exactly once.")

    # Audits
    column_audit = pd.DataFrame(column_audit_rows).sort_values(
        ["kept_in_respondent_file", CONFLICT_COL, "column"], ascending=[False, False, True]
    )
    save_csv(column_audit, column_audit_csv)

    respondent_audit = pd.DataFrame(
        [
            ("Raw ODiN rows", n_raw_rows),
            ("Rows with missing OPID", n_missing_opid_rows),
            ("Rows with nonmissing OPID", len(odin)),
            ("Unique OPIDs", n_unique_opids),
            ("Repeated OPID rows collapsed", duplicate_rows_to_remove),
            ("Respondent-level output rows", len(respondents)),
            ("Input columns", len(odin.columns)),
            ("Respondent-level output columns", len(respondents.columns)),
            ("Columns retained as respondent-constant", len(constant_columns)),
            ("Columns removed because they vary within OPID", len(varying_columns)),
        ],
        columns=["metric", "value"],
    )
    save_csv(respondent_audit, respondent_audit_csv)

    # Save
    header("SAVING OUTPUTS", 80)
    save_csv(respondents, output_csv)
    print(f"\nRespondent-level ODiN file saved to:\n{output_csv}")
    print(f"\nColumn audit saved to:\n{column_audit_csv}")
    print(f"\nRespondent-level audit saved to:\n{respondent_audit_csv}")

    validate_columns(respondents, REQUIRED, "Respondent-level ODiN output")
    print(f"\nRequired variables retained:\n{REQUIRED}")
    print("\nDone.")


if __name__ == "__main__":
    main()
