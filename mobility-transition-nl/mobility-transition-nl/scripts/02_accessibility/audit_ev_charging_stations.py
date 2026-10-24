"""
Audit of the filtered OSM charging stations written by build_ev_accessibility.py:
access classes, operational status, capacity/socket/fallback supply sources, duplicates.

Input:   data/processed/osm_ev_charging_stations.gpkg
Outputs: outputs/tables/ev_station_audit/
"""
import numpy as np
import pandas as pd
import geopandas as gpd

from mobility_transition import paths
from mobility_transition.io import save_csv

gpkg_path = paths.EV_STATIONS_GPKG
output_dir = paths.step_dir(paths.TABLES, "ev_station_audit")

LAYER_NAME = None  # None = auto-pick the most likely charging layer

# Optional benchmark. These count charging points, not OSM station locations, so they aren't directly comparable.
OFFICIAL_REGULAR_CHARGING_POINTS = None
OFFICIAL_FAST_CHARGING_POINTS = None
OFFICIAL_BENCHMARK_DATE = None

layers_csv = output_dir / "01_gpkg_layers.csv"
column_inventory_csv = output_dir / "02_column_inventory.csv"
access_distribution_csv = output_dir / "03_access_class_distribution.csv"
status_distribution_csv = output_dir / "04_operational_status_distribution.csv"
supply_source_csv = output_dir / "05_supply_source_distribution.csv"
access_supply_cross_tab_csv = output_dir / "06_access_by_supply_source.csv"
attribute_presence_csv = output_dir / "07_capacity_socket_information_presence.csv"
station_summary_csv = output_dir / "08_station_summary_for_thesis.csv"
possible_duplicates_csv = output_dir / "09_possible_duplicate_stations.csv"
sample_rows_csv = output_dir / "10_station_sample_rows.csv"
summary_text = output_dir / "11_station_summary_for_thesis.txt"

CAPACITY, SOCKETS, FALLBACK = "OSM capacity tag", "Summed socket counts", "Fallback value of 1"

# ============================================================
# HELPERS
# ============================================================


def save(df, path):
    save_csv(df, path)


def normalise_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def first_matching_column(columns, exact_candidates, keyword_groups=None):
    """Exact (case-insensitive) name match first, then the first column containing all keywords of a group."""
    lower_map = {str(c).lower(): c for c in columns}
    for candidate in exact_candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for group in keyword_groups or []:
        for column in columns:
            if all(k in str(column).lower() for k in group):
                return column
    return None


def numeric_from_messy(series):
    """Numbers from values like 4, "4", "4;6" (first value) or "capacity=4". Doesn't infer counts from yes/no."""
    numeric = pd.to_numeric(series, errors="coerce")
    remaining = numeric.isna() & series.notna()
    if remaining.any():
        extracted = series[remaining].astype("string").str.extract(r"([-+]?\d*\.?\d+)", expand=False)
        numeric.loc[remaining] = pd.to_numeric(extracted, errors="coerce")
    return numeric


def classify_access(raw_value):
    """Broad access category from either a precomputed class or a raw OSM access tag."""
    value = normalise_text(raw_value)

    if value in {"", "none", "nan", "unknown", "unspecified", "not_recorded"}:
        return "Access unspecified"
    if value in {"public", "yes", "permissive", "designated", "open", "unrestricted",
                 "public_or_unspecified", "strict_public", "principal"}:
        return "Public"
    conditional_terms = [
        "customer", "customers", "destination", "employee", "employees", "member", "members", "permit",
        "residents", "hotel", "guest", "conditional", "semi_public", "semipublic",
    ]
    if any(term in value for term in conditional_terms):
        return "Conditionally accessible"
    if value in {"private", "no", "restricted", "owners"} or "private" in value:
        return "Private or inaccessible"
    return f"Other: {value}"


def classify_supply_source(raw_value):
    value = normalise_text(raw_value)
    if value == "":
        return pd.NA
    if "fallback" in value or value in {"default", "minimum", "one", "1"}:
        return FALLBACK
    if "socket" in value:
        return SOCKETS
    if "capacity" in value:
        return CAPACITY
    return f"Other: {value}"


def safe_share(numerator, denominator):
    return np.nan if denominator == 0 else numerator / denominator


def example_values(series, maximum=5):
    return " | ".join(series.dropna().astype("string").drop_duplicates().head(maximum).tolist())


def list_gpkg_layers(path):
    try:
        import pyogrio
        return [str(row[0]) for row in pyogrio.list_layers(path)]
    except Exception:
        try:
            import fiona
            return list(fiona.listlayers(path))
        except Exception as exc:
            raise RuntimeError("Could not list GeoPackage layers. Install pyogrio or fiona.") from exc


def distribution(series, label):
    out = series.value_counts(dropna=False).rename_axis(label).reset_index(name="n_stations")
    out["percent_of_stations"] = 100 * out["n_stations"] / len(series)
    return out


def main():
    # ============================================================
    # OPEN GEOPACKAGE AND PICK A LAYER
    # ============================================================

    paths.require(gpkg_path)

    layers = list_gpkg_layers(gpkg_path)
    if not layers:
        raise ValueError("The GeoPackage contains no readable layers.")

    layer_rows = []
    for layer in layers:
        try:
            sample = gpd.read_file(gpkg_path, layer=layer, rows=5)
            layer_rows.append({
                "layer": layer,
                "readable": True,
                "geometry_column": sample.geometry.name if hasattr(sample, "geometry") else pd.NA,
                "crs": str(sample.crs),
                "sample_columns": " | ".join(map(str, sample.columns)),
            })
        except Exception as exc:
            layer_rows.append({"layer": layer, "readable": False, "geometry_column": pd.NA, "crs": pd.NA, "sample_columns": str(exc)})
    save(pd.DataFrame(layer_rows), layers_csv)

    if LAYER_NAME is not None:
        if LAYER_NAME not in layers:
            raise ValueError(f"Layer '{LAYER_NAME}' was not found.\nAvailable layers: {layers}")
        selected_layer = LAYER_NAME
    else:
        charging_layers = [l for l in layers if any(t in l.lower() for t in ["charg", "ev", "station"])]
        selected_layer = charging_layers[0] if charging_layers else layers[0]

    print("=" * 78 + "\nEV CHARGING-STATION GEOPACKAGE AUDIT\n" + "=" * 78)
    print(f"\nGeoPackage: {gpkg_path}\nAvailable layers: {layers}\nSelected layer: {selected_layer}")

    stations = gpd.read_file(gpkg_path, layer=selected_layer)
    stations.columns = stations.columns.str.strip()
    geom_name = stations.geometry.name

    print(f"Features loaded: {len(stations):,}\nCRS: {stations.crs}\n\nColumns:\n{stations.columns.tolist()}")

    # ============================================================
    # COLUMN INVENTORY
    # ============================================================

    inventory_rows = []
    for column in stations.columns:
        col = stations[column]
        examples = (
            str(stations.geometry.geom_type.value_counts(dropna=False).to_dict()) if column == geom_name else example_values(col)
        )
        inventory_rows.append({
            "column": column,
            "dtype": str(col.dtype),
            "n_nonmissing": int(col.notna().sum()),
            "percent_nonmissing": 100 * col.notna().mean(),
            "n_unique_nonmissing": int(col.nunique(dropna=True)),
            "example_values": examples,
        })
    save(pd.DataFrame(inventory_rows), column_inventory_csv)

    # ============================================================
    # DETECT ANALYTICAL COLUMNS
    # ============================================================

    columns = stations.columns.tolist()


    def find(exact, keywords):
        return first_matching_column(columns, exact, keywords)


    station_id_col = find(["station_id", "osm_id", "osm_node_id", "node_id", "id", "@id"], [("station", "id"), ("osm", "id")])
    access_class_col = find(
        ["access_class", "access_category", "access_group", "public_access_class", "network_class"],
        [("access", "class"), ("access", "category")],
    )
    access_raw_col = find(["access", "access_raw", "osm_access"], [("access",)])
    status_col = find(
        ["operational_status", "station_status", "status", "operational"],
        [("operational", "status"), ("station", "status")],
    )
    supply_col = find(
        ["estimated_supply", "station_supply", "charging_supply", "simultaneous_supply", "supply", "capacity_estimated"],
        [("estimated", "supply"), ("simultaneous", "supply"), ("station", "supply")],
    )
    supply_source_col = find(
        ["supply_source", "capacity_source", "supply_estimation_source", "estimated_supply_source"],
        [("supply", "source"), ("capacity", "source")],
    )
    capacity_col = find(["capacity", "capacity_raw", "osm_capacity", "charging_capacity"], [("capacity",)])
    socket_total_col = find(
        ["socket_count_sum", "socket_count_total", "total_socket_count", "total_sockets", "sockets_total", "socket_supply"],
        [("socket", "total"), ("socket", "sum"), ("socket", "count")],
    )
    fast_supply_col = find(
        ["fast_supply", "fast_capacity", "estimated_fast_supply", "fast_charging_supply"],
        [("fast", "supply"), ("fast", "capacity")],
    )
    max_power_col = find(
        ["max_power_kw", "maximum_power_kw", "max_output_kw", "charging_output_kw"],
        [("max", "power"), ("output", "kw")],
    )

    print("\nDetected analytical columns:")
    for label, col in [
        ("Station ID", station_id_col), ("Access class", access_class_col), ("Raw access", access_raw_col),
        ("Operational status", status_col), ("Estimated supply", supply_col), ("Supply source", supply_source_col),
        ("Capacity", capacity_col), ("Socket total", socket_total_col), ("Fast supply", fast_supply_col),
        ("Maximum power", max_power_col),
    ]:
        print(f"{label}: {col}")

    # ============================================================
    # ACCESS AND STATUS DISTRIBUTIONS
    # ============================================================

    access_source = access_class_col or access_raw_col
    stations["_access_class_standardised"] = (
        stations[access_source].map(classify_access) if access_source is not None else "Access field unavailable"
    )
    save(distribution(stations["_access_class_standardised"], "access_class"), access_distribution_csv)

    if status_col is not None:
        status_distribution = distribution(stations[status_col].astype("string").fillna("<missing>"), "operational_status")
    else:
        status_distribution = pd.DataFrame([{
            "operational_status": "No operational-status column found", "n_stations": len(stations), "percent_of_stations": 100.0,
        }])
    save(status_distribution, status_distribution_csv)

    # ============================================================
    # CAPACITY / SOCKET / FALLBACK SUPPLY
    # ============================================================

    empty = pd.Series(np.nan, index=stations.index)
    capacity_values = numeric_from_messy(stations[capacity_col]) if capacity_col is not None else empty.copy()
    socket_values = numeric_from_messy(stations[socket_total_col]) if socket_total_col is not None else empty.copy()

    # No pre-summed socket column: sum every numeric socket-related column instead
    socket_component_columns = []
    if socket_total_col is None:
        socket_component_columns = [
            c for c in columns
            if "socket" in str(c).lower() and c not in (geom_name, supply_source_col)
            and numeric_from_messy(stations[c]).notna().any()
        ]
        if socket_component_columns:
            socket_values = pd.concat(
                [numeric_from_messy(stations[c]).rename(c) for c in socket_component_columns], axis=1
            ).sum(axis=1, min_count=1)

    stations["_capacity_numeric"] = capacity_values
    stations["_socket_total_numeric"] = socket_values
    has_capacity = stations["_capacity_numeric"].gt(0)
    has_sockets = stations["_socket_total_numeric"].gt(0)

    if supply_source_col is not None:
        stations["_supply_source_standardised"] = stations[supply_source_col].map(classify_supply_source)
    else:
        stations["_supply_source_standardised"] = np.select(
            [has_capacity, ~has_capacity & has_sockets], [CAPACITY, SOCKETS], default=FALLBACK
        )

    if supply_col is not None:
        stations["_estimated_supply_numeric"] = numeric_from_messy(stations[supply_col])
    else:
        stations["_estimated_supply_numeric"] = np.where(
            has_capacity, stations["_capacity_numeric"], np.where(has_sockets, stations["_socket_total_numeric"], 1.0)
        )

    source = stations["_supply_source_standardised"]
    is_fallback = source.eq(FALLBACK)
    stations.loc[is_fallback & stations["_estimated_supply_numeric"].isna(), "_estimated_supply_numeric"] = 1.0

    save(distribution(source.astype("string").fillna("<missing>"), "supply_source"), supply_source_csv)
    save(pd.crosstab(stations["_access_class_standardised"], source, margins=True).reset_index(), access_supply_cross_tab_csv)

    presence_masks = [
        ("Valid OSM capacity tag", has_capacity),
        ("No capacity, but positive socket count", ~has_capacity & has_sockets),
        ("Neither capacity nor socket count; fallback 1", ~has_capacity & ~has_sockets),
    ]
    save(
        pd.DataFrame([
            {"information_category": label, "n_stations": int(mask.sum()), "percent_of_stations": 100 * mask.mean()}
            for label, mask in presence_masks
        ]),
        attribute_presence_csv,
    )

    # Fast supply: explicit field, else whole station when max power >= 50 kW
    if fast_supply_col is not None:
        stations["_fast_supply_numeric"] = numeric_from_messy(stations[fast_supply_col])
    elif max_power_col is not None:
        stations["_fast_supply_numeric"] = pd.Series(
            np.where(numeric_from_messy(stations[max_power_col]).ge(50), stations["_estimated_supply_numeric"], 0.0),
            index=stations.index,
        )
    else:
        stations["_fast_supply_numeric"] = empty.copy()

    # ============================================================
    # DUPLICATE CHECKS
    # ============================================================

    duplicate_frames = []
    if station_id_col is not None:
        ids = stations[station_id_col]
        duplicated_id = stations[ids.notna() & ids.duplicated(keep=False)].copy()
        if not duplicated_id.empty:
            duplicated_id["_duplicate_reason"] = "Duplicate station ID"
            duplicate_frames.append(duplicated_id)

    if stations.crs is not None:
        projected = stations.to_crs("EPSG:28992") if stations.crs.is_geographic else stations
        coordinate_key = projected.geometry.x.round(1).astype("string") + "_" + projected.geometry.y.round(1).astype("string")
        duplicated_coords = stations[coordinate_key.duplicated(keep=False)].copy()
        if not duplicated_coords.empty:
            duplicated_coords["_duplicate_reason"] = "Same rounded coordinates"
            duplicate_frames.append(duplicated_coords)

    possible_duplicates = (
        pd.concat(duplicate_frames, ignore_index=True).drop_duplicates()
        if duplicate_frames else pd.DataFrame(columns=["_duplicate_reason"])
    )
    save(possible_duplicates, possible_duplicates_csv)

    # ============================================================
    # THESIS SUMMARY
    # ============================================================

    access = stations["_access_class_standardised"]
    principal_mask = access.isin(["Public", "Access unspecified"])
    conditional_mask = access.eq("Conditionally accessible")
    private_mask = access.eq("Private or inaccessible")

    supply = stations["_estimated_supply_numeric"]
    fast = stations["_fast_supply_numeric"]
    total_supply = supply.sum(min_count=1)
    principal_supply = supply[principal_mask].sum(min_count=1)
    capacity_n, socket_n, fallback_n = (int(source.eq(s).sum()) for s in (CAPACITY, SOCKETS, FALLBACK))

    summary_rows = [
        ("Selected GeoPackage layer", selected_layer, ""),
        ("Total station features", len(stations), "Counts OSM station features, not individual charging points."),
        ("Unique station IDs", stations[station_id_col].nunique(dropna=True) if station_id_col is not None else np.nan,
         f"ID column: {station_id_col}"),
        ("Public or access-unspecified stations", int(principal_mask.sum()), "Principal network used for the strict accessibility measure."),
        ("Conditionally accessible stations", int(conditional_mask.sum()), "Customer, destination, employee, member or permit access."),
        ("Private or inaccessible stations", int(private_mask.sum()), "Normally excluded from both principal and inclusive networks."),
        ("Stations using OSM capacity tag", capacity_n, ""),
        ("Stations using summed socket counts", socket_n, ""),
        ("Stations using fallback supply of 1", fallback_n, ""),
        ("Fallback share of all stations", is_fallback.mean(), "Proportion rather than percentage."),
        ("Estimated total simultaneous supply", total_supply, "Sum of hierarchical capacity/socket/fallback station supply."),
        ("Estimated simultaneous supply in principal network", principal_supply, "Public plus access-unspecified stations."),
        ("Median estimated station supply", supply.median(), ""),
        ("Mean estimated station supply", supply.mean(), ""),
        ("Stations with positive fast-charging supply", int(fast.gt(0).sum()), "Available only when a fast-supply or power field was detected."),
        ("Estimated total fast-charging supply", fast.sum(min_count=1), ""),
        ("Missing or empty geometries", int((stations.geometry.isna() | stations.geometry.is_empty).sum()), ""),
        ("Possible duplicate rows", len(possible_duplicates), "Duplicate IDs and/or identical rounded coordinates."),
    ]
    if OFFICIAL_REGULAR_CHARGING_POINTS is not None:
        summary_rows.append((
            "Official regular charging-point benchmark", OFFICIAL_REGULAR_CHARGING_POINTS,
            f"Benchmark date: {OFFICIAL_BENCHMARK_DATE}. Charging points are not directly comparable with OSM station features.",
        ))
    if OFFICIAL_FAST_CHARGING_POINTS is not None:
        summary_rows.append(("Official fast charging-point benchmark", OFFICIAL_FAST_CHARGING_POINTS, f"Benchmark date: {OFFICIAL_BENCHMARK_DATE}."))

    station_summary = pd.DataFrame(summary_rows, columns=["measure", "value", "notes"])
    save(station_summary, station_summary_csv)

    sample_columns = [
        c for c in [
            station_id_col, access_class_col, access_raw_col, status_col, capacity_col, socket_total_col, supply_col,
            supply_source_col, fast_supply_col, max_power_col, "_access_class_standardised", "_capacity_numeric",
            "_socket_total_numeric", "_estimated_supply_numeric", "_supply_source_standardised", "_fast_supply_numeric", geom_name,
        ]
        if c is not None and c in stations.columns
    ]
    save(stations[sample_columns].head(30), sample_rows_csv)

    n = len(stations)


    def supply_line(label, value):
        return f"{label}: {value:,.0f}" if pd.notna(value) else f"{label}: unavailable"


    summary_lines = [
        "EV CHARGING-STATION GEOPACKAGE SUMMARY", "=" * 78, "",
        f"GeoPackage: {gpkg_path}",
        f"Layer: {selected_layer}",
        f"Total station features: {n:,}",
        f"Public or access-unspecified stations: {int(principal_mask.sum()):,}",
        f"Conditionally accessible stations: {int(conditional_mask.sum()):,}",
        f"Private or inaccessible stations: {int(private_mask.sum()):,}",
        "",
        "SUPPLY INFORMATION",
        f"OSM capacity tag used: {capacity_n:,} ({100 * safe_share(capacity_n, n):.1f}%)",
        f"Socket counts used: {socket_n:,} ({100 * safe_share(socket_n, n):.1f}%)",
        f"Fallback supply of 1 used: {fallback_n:,} ({100 * safe_share(fallback_n, n):.1f}%)",
        supply_line("Estimated total simultaneous charging supply", total_supply),
        supply_line("Estimated simultaneous supply in the principal network", principal_supply),
        "",
        "INTERPRETATION",
        "The station-feature total should not be compared directly with official charging-point totals. One OSM feature "
        "represents a location and may contain several simultaneously usable points.",
        "Use 02_column_inventory.csv to verify which original fields were detected and 10_station_sample_rows.csv to "
        "inspect example records.",
    ]
    summary_text.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    print(station_summary.to_string(index=False))
    print("\nSocket component columns used when no pre-summed socket field existed:")
    print(socket_component_columns if socket_component_columns else "None")
    print(f"\nOutputs saved in:\n{output_dir}")
    print("\nRead 11_station_summary_for_thesis.txt first.\nDone.")


if __name__ == "__main__":
    main()
