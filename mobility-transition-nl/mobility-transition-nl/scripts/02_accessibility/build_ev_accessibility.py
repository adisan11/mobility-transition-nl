"""
PC4 public EV charging accessibility (5/10/15 km catchments, exponential distance decay) from OSM charging stations.

Inputs:  data/raw/osm/netherlands-*.osm.pbf, data/processed/pc4_population_weighted_centroids.gpkg
Outputs: data/processed/osm_ev_charging_stations.gpkg
         data/processed/pc4_ev_accessibility.{csv,gpkg}   (main index: ev_access_capacity_decay_strict_10km)
         data/processed/pc4_ev_station_links_debug.csv    (optional, large)
"""
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from mobility_transition import paths
from mobility_transition.io import clean_pc4, find_column, header, save_csv
from mobility_transition.settings import TARGET_CRS

try:
    import osmium
except ImportError as exc:
    raise ImportError("The 'osmium' package is required to read the .osm.pbf file: pip install osmium") from exc
try:
    from tqdm import tqdm
except ImportError as exc:
    raise ImportError("The 'tqdm' package is required for the progress indicator: pip install tqdm") from exc

# 10 km is the main specification; 5 and 15 km are sensitivity checks.
CATCHMENTS_M = [5000, 10000, 15000]
MAIN_CATCHMENT_M = 10000
BETA = 1.5  # same decay as the PT index: exp(-beta * distance / catchment)
EXPORT_DEBUG_LINKS = True  # full PC4-station link table; can be large

if MAIN_CATCHMENT_M not in CATCHMENTS_M:
    raise ValueError("MAIN_CATCHMENT_M must be included in CATCHMENTS_M.")

STATION_NEAREST_COLS = ["station_id", "access_class", "estimated_supply", "estimated_fast_supply", "geometry"]

strict_public_values = {"", "yes", "public", "permissive", "designated"}
conditional_access_values = {"customers", "customer", "destination", "permit", "members", "employees"}
excluded_access_values = {"private", "no"}
excluded_status_values = {
    "broken", "closed", "disused", "out_of_order", "out of order", "planned", "construction",
    "under_construction", "under construction", "temporarily_closed", "temporarily closed",
}
# DC rapid/fast-charging socket types
fast_socket_types = {"ccs", "ccs_combo_1", "ccs_combo_2", "chademo", "tesla_supercharger", "nacs", "gb_t_dc"}

PC4_GEOM_CANDIDATES = ["pc4_clean", "pc4", "PC4", "postcode", "Postcode", "postal_code", "Postal code", "VertPC"]

# =================================================
# HELPERS
# =================================================


def normalise_text(value):
    return "" if value is None else str(value).strip().lower()


def first_number(value):
    """First number in a tag, e.g. '4;6' -> 4.0, '22 kW' -> 22.0."""
    if value is None:
        return np.nan
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value))
    return float(match.group(0).replace(",", ".")) if match else np.nan


def parse_power_kw(value):
    """'22 kW' -> 22, '50000 W' -> 50, '150' -> 150 (unitless values assumed kW)."""
    if value is None:
        return np.nan
    text = normalise_text(value)
    number = first_number(text)
    if pd.isna(number):
        return np.nan
    if "mw" in text:
        return number * 1000
    if "kw" in text:
        return number
    if re.search(r"(^|[^k])w($|[^a-z])", text):
        return number / 1000
    return number


def socket_count_from_tags(tags):
    """Sums socket:<type>=N tags, ignoring :output/:voltage/:current attributes."""
    total = 0.0
    for key, value in tags.items():
        key_lower = key.lower()
        if not key_lower.startswith("socket:") or key_lower.endswith((":output", ":voltage", ":current")):
            continue
        count = first_number(value)
        if pd.notna(count) and count > 0:
            total += count
    return total


def fast_socket_count_from_tags(tags):
    total = 0.0
    for socket_type in fast_socket_types:
        count = first_number(tags.get(f"socket:{socket_type}"))
        if pd.notna(count) and count > 0:
            total += count
    return total


def maximum_recorded_output_kw(tags):
    outputs = []
    for key, value in tags.items():
        key_lower = key.lower()
        if key_lower.endswith(":output") or key_lower in {"output", "charging_station:output", "max_power", "power"}:
            parsed = parse_power_kw(value)
            if pd.notna(parsed):
                outputs.append(parsed)
    return max(outputs) if outputs else np.nan


def classify_access(tags):
    """public_or_unspecified / conditional / private_or_no, based on access and vehicle tags."""
    values = [normalise_text(tags.get(k)) for k in ("access", "vehicle", "motor_vehicle", "motorcar")]
    values = [v for v in values if v]
    if any(v in excluded_access_values for v in values):
        return "private_or_no"
    if any(v in conditional_access_values for v in values):
        return "conditional"
    return "public_or_unspecified"


def is_operational(tags):
    if any(normalise_text(tags.get(k)) in excluded_status_values for k in ("operational_status", "status", "state")):
        return False
    return normalise_text(tags.get("disused")) != "yes" and normalise_text(tags.get("abandoned")) != "yes"


def supports_motorcars(tags):
    """Rejects stations closed to motor vehicles or likely bicycle-only."""
    motor_tags = [normalise_text(tags.get(k)) for k in ("motorcar", "motor_vehicle", "vehicle")]
    if "no" in motor_tags:
        return False

    positive = {"yes", "designated", "permissive", "destination", "customers", "customer"}
    has_positive_motor_tag = any(v in positive for v in motor_tags)
    likely_bicycle_only = normalise_text(tags.get("bicycle")) in {"yes", "designated"} and not has_positive_motor_tag
    return not likely_bicycle_only


def build_station_record(osm_type, osm_id, tags, geometry):
    if tags.get("amenity") != "charging_station" or not is_operational(tags) or not supports_motorcars(tags):
        return None

    capacity_tag = first_number(tags.get("capacity"))
    socket_count = socket_count_from_tags(tags)
    fast_socket_count = fast_socket_count_from_tags(tags)
    max_output_kw = maximum_recorded_output_kw(tags)

    # Supply: capacity tag > socket count > assume 1
    if pd.notna(capacity_tag) and capacity_tag > 0:
        estimated_supply, supply_source = capacity_tag, "capacity"
    elif socket_count > 0:
        estimated_supply, supply_source = socket_count, "socket_count"
    else:
        estimated_supply, supply_source = 1.0, "fallback_1"

    # Fast supply: fast sockets, else whole station if >= 50 kW
    if fast_socket_count > 0:
        estimated_fast_supply = fast_socket_count
    elif pd.notna(max_output_kw) and max_output_kw >= 50:
        estimated_fast_supply = estimated_supply
    else:
        estimated_fast_supply = 0.0

    return {
        "station_id": f"{osm_type}/{osm_id}",
        "osm_type": osm_type,
        "osm_id": str(osm_id),
        **{k: tags.get(k) for k in ("name", "operator", "network", "brand", "access")},
        "access_class": classify_access(tags),
        "capacity_tag": capacity_tag,
        "socket_count": socket_count,
        "estimated_supply": estimated_supply,
        "supply_source": supply_source,
        "fast_socket_count": fast_socket_count,
        "estimated_fast_supply": estimated_fast_supply,
        "max_output_kw": max_output_kw,
        **{k: tags.get(k) for k in ("fee", "opening_hours", "authentication")},
        "geometry": geometry,
    }


class ChargingStationHandler(osmium.SimpleHandler):
    """
    Streams OSM nodes tagged amenity=charging_station.
    Progress shows nodes processed, not a percentage (a total would need a second full scan).
    """

    def __init__(self, progress_bar, progress_batch=100_000):
        super().__init__()
        self.records = []
        self.n_all_nodes_processed = 0
        self.n_charging_nodes_seen = 0
        self.progress_bar = progress_bar
        self.progress_batch = progress_batch
        self._last_progress_update = 0

    def _flush_progress(self, minimum):
        pending = self.n_all_nodes_processed - self._last_progress_update
        if pending >= minimum and pending > 0:
            self.progress_bar.update(pending)
            self._last_progress_update = self.n_all_nodes_processed

    def finish_progress(self):
        self._flush_progress(1)

    def node(self, node):
        self.n_all_nodes_processed += 1
        self._flush_progress(self.progress_batch)

        # Check the tag directly so only charging stations get converted to a dict
        if node.tags.get("amenity") != "charging_station":
            return

        self.n_charging_nodes_seen += 1
        tags = dict(node.tags)
        try:
            if not node.location.valid():
                return
            geometry = Point(node.location.lon, node.location.lat)
        except Exception:
            return

        record = build_station_record("node", node.id, tags, geometry)
        if record is not None:
            self.records.append(record)


def nearest_station(pc4, station_subset, distance_col, id_col):
    joined = gpd.sjoin_nearest(pc4[["pc4_clean", "geometry"]], station_subset[STATION_NEAREST_COLS], how="left", distance_col=distance_col)
    return (
        joined.sort_values(distance_col).drop_duplicates("pc4_clean")
        [["pc4_clean", "station_id", distance_col]].rename(columns={"station_id": id_col})
    )


def extract_stations():
    header("EXTRACTING OSM EV CHARGING STATIONS")
    print(f"Reading: {paths.OSM_PBF.name} (node-only scan, no way-location index)")

    progress_bar = tqdm(total=None, desc="Scanning OSM nodes", unit="node", unit_scale=True, dynamic_ncols=True, mininterval=0.5)
    handler = ChargingStationHandler(progress_bar=progress_bar)
    try:
        handler.apply_file(str(paths.OSM_PBF), locations=False)
    finally:
        handler.finish_progress()
        progress_bar.close()

    if not handler.records:
        raise ValueError("No usable amenity=charging_station features were extracted. Check that the PBF covers the Netherlands.")

    stations = gpd.GeoDataFrame(handler.records, geometry="geometry", crs="EPSG:4326")
    print(f"\nAll OSM nodes processed: {handler.n_all_nodes_processed:,}")
    print(f"Raw charging-station nodes encountered: {handler.n_charging_nodes_seen:,}")
    print(f"Operational motorcar station nodes retained: {len(stations):,}")

    # Drop private/no-access; conditional stations stay for the inclusive sensitivity measure
    stations = stations[stations["access_class"] != "private_or_no"].copy()
    stations = stations.to_crs(TARGET_CRS).drop_duplicates(subset=["station_id"], keep="first").copy()

    print("\nStations retained by access class:")
    print(stations["access_class"].value_counts(dropna=False))
    print("\nSupply-source completeness:")
    print(stations["supply_source"].value_counts(dropna=False))
    print("\nEstimated station supply summary:")
    print(stations["estimated_supply"].describe())

    stations.to_file(paths.EV_STATIONS_GPKG, layer="charging_stations", driver="GPKG")
    print(f"\nSaved: {paths.EV_STATIONS_GPKG}")
    return stations


def load_centroids():
    header("LOADING PC4 POPULATION-WEIGHTED CENTROIDS")
    pc4 = gpd.read_file(paths.PC4_CENTROIDS_GPKG)
    if pc4.empty:
        raise ValueError("The PC4 centroid file contains no features.")
    if pc4.crs is None:
        raise ValueError("The PC4 centroid file has no CRS; assign one before running.")

    pc4_col = find_column(pc4, PC4_GEOM_CANDIDATES, "PC4 column in centroid file")
    pc4["pc4_clean"] = clean_pc4(pc4[pc4_col])
    pc4 = pc4.dropna(subset=["pc4_clean", "geometry"]).to_crs(TARGET_CRS)

    # Fall back to a representative point if the layer contains polygons
    non_point = ~pc4.geometry.geom_type.isin(["Point", "MultiPoint"])
    if non_point.any():
        pc4.loc[non_point, "geometry"] = pc4.loc[non_point, "geometry"].representative_point()

    pc4 = pc4.drop_duplicates(subset=["pc4_clean"], keep="first").copy()
    pc4["centroid_x_rd"] = pc4.geometry.x
    pc4["centroid_y_rd"] = pc4.geometry.y
    print(f"PC4 centroids loaded: {len(pc4):,} (column: {pc4_col}, CRS: {pc4.crs})")
    return pc4


def link_stations(pc4, stations):
    """All PC4-station pairs within the largest catchment, with straight-line distance."""
    header("FINDING CHARGING STATIONS WITHIN CATCHMENTS")
    max_catchment_m = max(CATCHMENTS_M)

    buffers = stations.copy()
    buffers["geometry_station"] = buffers.geometry
    buffers["geometry"] = buffers.geometry.buffer(max_catchment_m)

    nearby = gpd.sjoin(
        pc4[["pc4_clean", "centroid_x_rd", "centroid_y_rd", "geometry"]],
        buffers[[
            "station_id", "name", "operator", "access_class", "estimated_supply", "estimated_fast_supply",
            "supply_source", "max_output_kw", "geometry_station", "geometry",
        ]],
        how="left", predicate="within",
    ).dropna(subset=["station_id"]).copy()

    nearby["distance_m"] = nearby.geometry.distance(nearby["geometry_station"])
    nearby = nearby[nearby["distance_m"] <= max_catchment_m].copy()
    print(f"PC4-station links within {max_catchment_m:,} m: {len(nearby):,}")
    print(f"PC4s with at least one linked station: {nearby['pc4_clean'].nunique():,}")
    return nearby


def accessibility_tables(nearby, all_pc4):
    header("CALCULATING DISTANCE-DECAY ACCESSIBILITY")
    networks = {
        "strict": nearby["access_class"] == "public_or_unspecified",
        "inclusive": nearby["access_class"].isin(["public_or_unspecified", "conditional"]),
    }
    tables = []
    for network_name, network_mask in networks.items():
        for catchment in CATCHMENTS_M:
            subset = nearby[network_mask & (nearby["distance_m"] <= catchment)].copy()
            suffix = f"{network_name}_{catchment // 1000}km"
            metric_cols = [
                f"n_stations_{suffix}", f"capacity_{suffix}", f"fast_capacity_{suffix}",
                f"ev_access_station_decay_{suffix}", f"ev_access_capacity_decay_{suffix}", f"ev_access_fast_decay_{suffix}",
            ]

            if subset.empty:
                empty = all_pc4.copy()
                for i, col in enumerate(metric_cols):
                    empty[col] = 0 if i == 0 else 0.0
                tables.append(empty)
                continue

            subset["decay_weight"] = np.exp(-BETA * (subset["distance_m"] / catchment))
            subset["capacity_decay_contribution"] = subset["estimated_supply"] * subset["decay_weight"]
            subset["fast_decay_contribution"] = subset["estimated_fast_supply"] * subset["decay_weight"]

            sources = [
                ("station_id", "nunique"), ("estimated_supply", "sum"), ("estimated_fast_supply", "sum"),
                ("decay_weight", "sum"), ("capacity_decay_contribution", "sum"), ("fast_decay_contribution", "sum"),
            ]
            tables.append(subset.groupby("pc4_clean", as_index=False).agg(**dict(zip(metric_cols, sources))))
            print(f"{network_name.title()} network, {catchment // 1000} km: {subset['pc4_clean'].nunique():,} PC4s with access")
    return tables


def quality_table(nearby, all_pc4, catchment):
    """Linked strict stations and the share relying on the fallback supply of 1."""
    km = catchment // 1000
    n_col = f"n_linked_stations_quality_strict_{km}km"
    fallback_col = f"share_linked_stations_supply_fallback_strict_{km}km"
    subset = nearby[(nearby["access_class"] == "public_or_unspecified") & (nearby["distance_m"] <= catchment)].copy()

    if subset.empty:
        out = all_pc4.copy()
        out[n_col] = 0
        out[fallback_col] = 0.0
        return out

    subset["fallback_supply"] = (subset["supply_source"] == "fallback_1").astype(int)
    out = subset.groupby("pc4_clean", as_index=False).agg(**{n_col: ("station_id", "nunique"), fallback_col: ("fallback_supply", "mean")})
    out = all_pc4.merge(out, on="pc4_clean", how="left")
    out[[n_col, fallback_col]] = out[[n_col, fallback_col]].fillna(0)
    return out


def main():
    paths.require(paths.OSM_PBF, paths.PC4_CENTROIDS_GPKG)
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)

    stations = extract_stations()
    pc4 = load_centroids()

    header("CALCULATING NEAREST-STATION DISTANCES")
    strict_stations = stations[stations["access_class"] == "public_or_unspecified"].copy()
    if strict_stations.empty:
        raise ValueError("No strict public/unspecified-access stations remain after filtering.")
    inclusive_nearest = nearest_station(pc4, stations, "nearest_inclusive_station_m", "nearest_inclusive_station_id")
    strict_nearest = nearest_station(pc4, strict_stations, "nearest_strict_public_station_m", "nearest_strict_public_station_id")

    nearby = link_stations(pc4, stations)
    all_pc4 = pc4[["pc4_clean"]].copy()

    # Centroid attributes (incl. population_represented, needed downstream) are carried through
    result = pc4.merge(inclusive_nearest, on="pc4_clean", how="left").merge(strict_nearest, on="pc4_clean", how="left")
    for table in accessibility_tables(nearby, all_pc4):
        result = result.merge(table, on="pc4_clean", how="left")

    access_cols = [c for c in result.columns if c.startswith(("n_stations_", "capacity_", "fast_capacity_", "ev_access_"))]
    result[access_cols] = result[access_cols].fillna(0)

    # Log, 0-100 and quintile versions of the strict capacity index for every catchment
    for catchment in CATCHMENTS_M:
        index_col = f"ev_access_capacity_decay_strict_{catchment // 1000}km"
        if index_col not in result.columns:
            raise ValueError(f"Expected accessibility column was not created: {index_col}")
        result[f"log1p_{index_col}"] = np.log1p(result[index_col].clip(lower=0))
        maximum = result[index_col].max()
        result[f"{index_col}_norm_0_100"] = result[index_col] / maximum * 100 if pd.notna(maximum) and maximum > 0 else 0.0
        try:
            result[f"{index_col}_quintile"] = pd.qcut(result[index_col], q=5, labels=False, duplicates="drop") + 1
        except ValueError:
            result[f"{index_col}_quintile"] = np.nan

    for catchment in CATCHMENTS_M:
        result = result.merge(quality_table(nearby, all_pc4, catchment), on="pc4_clean", how="left")

    # Generic aliases used downstream point to the main catchment
    main_km = MAIN_CATCHMENT_M // 1000
    main_index_col = f"ev_access_capacity_decay_strict_{main_km}km"
    result["n_linked_stations_quality"] = result[f"n_linked_stations_quality_strict_{main_km}km"]
    result["share_linked_stations_supply_fallback"] = result[f"share_linked_stations_supply_fallback_strict_{main_km}km"]

    header("EXPORTING RESULTS")
    result_gdf = gpd.GeoDataFrame(result, geometry="geometry", crs=TARGET_CRS)
    save_csv(result_gdf.drop(columns="geometry"), paths.EV_ACCESSIBILITY_CSV)
    result_gdf.to_file(paths.EV_ACCESSIBILITY_GPKG, layer="pc4_ev_accessibility", driver="GPKG")
    print(f"Saved: {paths.EV_ACCESSIBILITY_CSV}\nSaved: {paths.EV_ACCESSIBILITY_GPKG}")

    if EXPORT_DEBUG_LINKS:
        links = nearby.drop(columns="geometry", errors="ignore").copy()
        links["station_geometry_wkt"] = links["geometry_station"].to_wkt()
        save_csv(links.drop(columns="geometry_station", errors="ignore"), paths.EV_STATION_LINKS_CSV)
        print(f"Saved: {paths.EV_STATION_LINKS_CSV}")

    header("SUMMARY")
    print(f"PC4 centroids analysed: {len(result_gdf):,}")
    print(f"Charging-station nodes retained: {len(stations):,}")
    print(f"Strict public/unspecified stations: {len(strict_stations):,}")
    print(f"Conditional-access stations: {(stations['access_class'] == 'conditional').sum():,}")
    print(
        f"\nMain index: {main_index_col}\nEstimated simultaneously usable charging supply within {main_km} km, weighted by "
        f"exp(-{BETA} * distance / {MAIN_CATCHMENT_M}), using strict public or access-unspecified stations."
    )
    print(result[main_index_col].describe())
    print(f"\nPC4s with zero strict-public {main_km} km accessibility: {(result[main_index_col] == 0).sum():,}")
    for catchment in CATCHMENTS_M:
        print(f"\n{catchment // 1000} km:")
        print(result[f"ev_access_capacity_decay_strict_{catchment // 1000}km"].describe())
    print(f"\nCatchments: {[c // 1000 for c in CATCHMENTS_M]} km; distance-decay beta: {BETA}")


if __name__ == "__main__":
    main()
