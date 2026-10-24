"""
Stacks the yearly ODiN files, keeping only columns present in every year, and tags each row with ODiN_year.

Inputs:  data/raw/odin/ODiN<year>_*.csv (see RAW_FILES)
Output:  data/processed/ODiN<years>_merged_common_columns_only.csv
         (default 2022+2023 -> ODiN2022_2023_merged_common_columns_only.csv, used downstream)
"""
import argparse

import pandas as pd

from mobility_transition import paths
from mobility_transition.io import header, read_csv_auto, save_csv

RAW_FILES = {
    2022: paths.ODIN_DIR / "ODiN2022_Databestand.csv",
    2023: paths.ODIN_DIR / "ODiN2023_Databestand.csv",
    2024: paths.ODIN_DIR / "ODiN2024_DANS_Databestand_v2.0.csv",
}


def read_year(path, year):
    df = pd.read_excel(path) if path.suffix.lower() in [".xlsx", ".xls"] else read_csv_auto(path)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    df["ODiN_year"] = year
    print(f"{year}: {len(df):,} rows, {len(df.columns):,} columns")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023], choices=sorted(RAW_FILES))
    years = sorted(parser.parse_args().years)

    paths.require(*[RAW_FILES[y] for y in years])
    header(f"MERGING ODiN {', '.join(map(str, years))}")
    frames = [read_year(RAW_FILES[y], y) for y in years]

    # Columns shared by all years, in the order of the first file
    common = set(frames[0].columns).intersection(*[f.columns for f in frames[1:]])
    common_columns = [c for c in frames[0].columns if c in common]
    merged = pd.concat([f[common_columns] for f in frames], ignore_index=True)

    output = paths.PROCESSED / f"ODiN{'_'.join(map(str, years))}_merged_common_columns_only.csv"
    save_csv(merged, output)
    print(f"\nCommon columns: {len(common_columns):,}\nTotal rows: {len(merged):,}\nSaved: {output}")


if __name__ == "__main__":
    main()
