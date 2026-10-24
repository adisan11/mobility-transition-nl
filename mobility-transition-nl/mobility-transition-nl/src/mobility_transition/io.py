"""Shared reading, cleaning and saving helpers."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
SEPARATORS = [",", ";", "\t", "|"]


def header(title, width=70):
    print("\n" + "=" * width + f"\n{title}\n" + "=" * width)


def detect_csv_format(path):
    """Encoding and separator from the first 100 kB, without loading the file."""
    last_error = None
    for encoding in ENCODINGS:
        try:
            with open(path, "r", encoding=encoding) as f:
                sample = f.read(100_000)
            try:
                separator = csv.Sniffer().sniff(sample, delimiters=SEPARATORS).delimiter
            except csv.Error:
                first_line = sample.splitlines()[0] if sample else ""
                separator = max(SEPARATORS, key=first_line.count)
            return encoding, separator
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def read_csv_auto(path, usecols=None, verbose=True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")
    encoding, separator = detect_csv_format(path)
    if verbose:
        print(f"\nReading: {path.name} (encoding={encoding}, sep={separator!r})")
    data = pd.read_csv(path, encoding=encoding, sep=separator, usecols=usecols, low_memory=False)
    data.columns = data.columns.astype(str).str.strip()
    return data


def save_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig, path, dpi=300):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def clean_pc4(series):
    """Four-digit PC4 string (handles 1234.0, 'PC1234', padding); invalid values become <NA>."""
    return series.astype("string").str.strip().str.extract(r"(\d{4})", expand=False).astype("string")


def find_column(df, candidates, label, required=True):
    found = next((c for c in candidates if c in df.columns), None)
    if found is None and required:
        raise ValueError(f"Could not find {label}. Tried: {candidates}\nAvailable columns: {list(df.columns)}")
    return found
