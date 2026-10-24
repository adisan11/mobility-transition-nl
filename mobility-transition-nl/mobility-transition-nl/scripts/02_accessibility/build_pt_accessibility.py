"""
PC4 public-transport accessibility: GTFS weekly departures at stops within mode-specific catchments of each
population-weighted centroid, with exponential distance decay. Produces BTM (bus+tram+metro, matches ODiN FqBTM),
train (matches FqTrein) and total indices.

Inputs:  data/raw/gtfs/ (loose .txt files or one .zip), data/processed/pc4_population_weighted_centroids.csv
Outputs: data/processed/pc4_pt_accessibility_distance_decay.{csv,gpkg}, pc4_pt_stop_links_debug.csv
         With --rail-catchment 4000: the *_train_4000m sensitivity versions instead.
"""
import argparse
import zipfile
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd

from mobility_transition import paths
from mobility_transition.io import clean_pc4, find_column, header, read_csv_auto, save_csv
from mobility_transition.settings import (
    GTFS_ROUTE_TYPE_TO_MODE, PT_BETA, PT_CATCHMENTS_M, RAIL_CATCHMENT_SENSITIVITY_M, TARGET_CRS,
)

ANALYSIS_START_DATE = None  # "YYYY-MM-DD"; None = first Monday in calendar_dates.txt
STOP_TIMES_CHUNKSIZE = 2_000_000
MODES = ["bus", "tram", "metro", "rail"]
GTFS_REQUIRED = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt", "calendar_dates.txt"]


# ---------------- GTFS reading ----------------

def find_gtfs_source(folder):
    """Loose .txt files in the folder, otherwise the largest .zip in it."""
    if all((folder / f).exists() for f in GTFS_REQUIRED):
        print(f"Using loose GTFS files in: {folder}")
        return {"type": "folder", "path": folder, "zip": None}

    zips = list(folder.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No GTFS .txt files or .zip found in: {folder}\nContents: {[p.name for p in folder.iterdir()]}")
    zip_path = max(zips, key=lambda p: p.stat().st_size)
    print(f"Using GTFS zip: {zip_path.name}")
    return {"type": "zip", "path": zip_path, "zip": zipfile.ZipFile(zip_path)}


def read_gtfs(source, filename, chunksize=None, **kwargs):
    handle = source["path"] / filename if source["type"] == "folder" else source["zip"].open(filename)
    return pd.read_csv(handle, low_memory=False, chunksize=chunksize, **kwargs)


def first_monday(dates):
    d, max_date = dates.min(), dates.max()
    while d.weekday() != 0:
        d += timedelta(days=1)
    if d > max_date:
        raise ValueError("No Monday inside the calendar_dates.txt date range.")
    return d


# ---------------- pipeline steps ----------------

def load_active_trips(source):
    """Trips for bus/tram/metro/rail, with how many days each runs in the representative week."""
    header("GTFS ROUTES, TRIPS AND CALENDAR")
    routes = read_gtfs(source, "routes.txt", usecols=["route_id", "route_type"], dtype={"route_id": str})
    routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce").astype("Int64")
    routes["mode"] = routes["route_type"].map(GTFS_ROUTE_TYPE_TO_MODE)
    print(routes.groupby(["route_type", "mode"], dropna=False).size().reset_index(name="n_routes").to_string(index=False))
    routes = routes[routes["mode"].isin(MODES)]

    trips = read_gtfs(source, "trips.txt", usecols=["route_id", "service_id", "trip_id"], dtype=str)
    trips = trips.merge(routes[["route_id", "mode"]], on="route_id", how="inner")
    print(f"\nTrips (excluding ferry/unknown): {len(trips):,}\n{trips['mode'].value_counts().to_string()}")

    cal = read_gtfs(source, "calendar_dates.txt", dtype={"service_id": str})
    cal["date_dt"] = cal["date"].apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date())
    cal["exception_type"] = pd.to_numeric(cal["exception_type"], errors="coerce").fillna(1).astype(int)
    print(f"\ncalendar_dates range: {cal['date_dt'].min()} to {cal['date_dt'].max()}")

    start = datetime.strptime(ANALYSIS_START_DATE, "%Y-%m-%d").date() if ANALYSIS_START_DATE else first_monday(cal["date_dt"])
    week = [start + timedelta(days=i) for i in range(7)]
    print(f"Representative week: {week[0]} to {week[-1]}")

    # exception_type 1 = service runs on that date
    active = cal[cal["date_dt"].isin(week) & cal["exception_type"].eq(1)][["service_id", "date_dt"]].drop_duplicates()
    if active.empty:
        raise ValueError("No active services in the representative week; set ANALYSIS_START_DATE inside the feed's range.")
    days = active.groupby("service_id", as_index=False).size().rename(columns={"size": "active_days_in_week"})

    trips_active = trips.merge(days, on="service_id", how="inner")
    print(f"Trips active in week: {len(trips_active):,}")
    return trips_active[["trip_id", "mode", "active_days_in_week"]], week


def count_stop_arrivals(source, trip_weights):
    """Weekly arrivals per stop and mode, streamed through stop_times.txt in chunks."""
    header("COUNTING WEEKLY ARRIVALS (stop_times.txt)")
    parts = []
    chunks = read_gtfs(source, "stop_times.txt", chunksize=STOP_TIMES_CHUNKSIZE, usecols=["trip_id", "stop_id"], dtype=str)
    for i, chunk in enumerate(chunks, start=1):
        chunk = chunk.merge(trip_weights, on="trip_id", how="inner")
        if not chunk.empty:
            parts.append(chunk.groupby(["stop_id", "mode"], as_index=False)["active_days_in_week"].sum()
                         .rename(columns={"active_days_in_week": "weekly_arrivals"}))
        print(f"Processed chunk {i}")
    if not parts:
        raise ValueError("No stop arrivals counted. Check calendar_dates and the selected week.")
    return pd.concat(parts, ignore_index=True).groupby(["stop_id", "mode"], as_index=False)["weekly_arrivals"].sum()


def build_canonical_stops(source, stop_arrivals, catchments):
    """Platforms are merged into their parent station so a station counts once per mode."""
    header("CANONICAL STOPS")
    stops = read_gtfs(source, "stops.txt", dtype=str)
    stops.columns = stops.columns.str.strip()
    for col in ["stop_id", "stop_name", "stop_lat", "stop_lon"]:
        if col not in stops.columns:
            raise ValueError(f"Required column '{col}' not found in stops.txt")
    for col in ["parent_station", "location_type"]:
        if col not in stops.columns:
            stops[col] = np.nan

    stops = stops[["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station", "location_type"]].copy()
    stops[["stop_lat", "stop_lon"]] = stops[["stop_lat", "stop_lon"]].apply(pd.to_numeric, errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
    has_parent = stops["parent_station"].notna() & (stops["parent_station"].astype(str).str.strip() != "")
    stops["canonical_stop_id"] = np.where(has_parent, stops["parent_station"], stops["stop_id"])

    arrivals = stop_arrivals.merge(stops[["stop_id", "canonical_stop_id"]], on="stop_id", how="left")
    arrivals["canonical_stop_id"] = arrivals["canonical_stop_id"].fillna(arrivals["stop_id"])
    canonical = arrivals.groupby(["canonical_stop_id", "mode"], as_index=False)["weekly_arrivals"].sum()

    # Coordinates: the parent-station row if present, else the mean of its platforms
    own = stops.drop_duplicates("stop_id")[["stop_id", "stop_name", "stop_lat", "stop_lon"]].rename(
        columns={"stop_id": "canonical_stop_id", "stop_name": "canonical_stop_name"})
    child_mean = stops.groupby("canonical_stop_id", as_index=False).agg(
        stop_lat_fallback=("stop_lat", "mean"), stop_lon_fallback=("stop_lon", "mean"),
        canonical_stop_name_fallback=("stop_name", "first"))
    canonical = canonical.merge(own, on="canonical_stop_id", how="left").merge(child_mean, on="canonical_stop_id", how="left")
    for col in ["stop_lat", "stop_lon", "canonical_stop_name"]:
        canonical[col] = canonical[col].fillna(canonical[f"{col}_fallback"])

    canonical = canonical.dropna(subset=["stop_lat", "stop_lon"]).copy()
    canonical["catchment_m"] = canonical["mode"].map(catchments)
    canonical["beta"] = canonical["mode"].map(PT_BETA)
    canonical = canonical.dropna(subset=["catchment_m", "beta"])

    stops_gdf = gpd.GeoDataFrame(
        canonical, geometry=gpd.points_from_xy(canonical["stop_lon"], canonical["stop_lat"]), crs="EPSG:4326"
    ).to_crs(TARGET_CRS)
    print(f"Canonical stop-mode records: {len(stops_gdf):,}\n{stops_gdf['mode'].value_counts().to_string()}")
    rail = stops_gdf[stops_gdf["mode"] == "rail"]
    print(f"\nRail stations: {len(rail):,}")
    if not rail.empty:
        print(rail[["canonical_stop_id", "canonical_stop_name", "weekly_arrivals"]].nlargest(10, "weekly_arrivals").to_string(index=False))
    return stops_gdf


def load_centroids():
    header("PC4 POPULATION-WEIGHTED CENTROIDS")
    pc4 = read_csv_auto(paths.PC4_CENTROIDS_CSV)
    pc4_col = find_column(pc4, ["pc4_clean", "pc4", "PC4", "Postal code", "postcode", "Postcode", "VertPC"], "PC4 column")
    x_col = find_column(pc4, ["x_weighted", "x", "X", "centroid_x"], "x column")
    y_col = find_column(pc4, ["y_weighted", "y", "Y", "centroid_y"], "y column")

    pc4["pc4_clean"] = clean_pc4(pc4[pc4_col])
    pc4[[x_col, y_col]] = pc4[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
    pc4 = pc4.dropna(subset=["pc4_clean", x_col, y_col]).copy()
    gdf = gpd.GeoDataFrame(pc4, geometry=gpd.points_from_xy(pc4[x_col], pc4[y_col]), crs=TARGET_CRS)
    print(f"Centroids: {len(gdf):,}")
    return gdf, x_col, y_col


def link_pc4_to_stops(pc4_gdf, stops_gdf, x_col, y_col):
    """All PC4-stop pairs within each stop's mode-specific catchment."""
    header("LINKING PC4 CENTROIDS TO STOPS")
    buffers = stops_gdf.copy()
    buffers["geometry_stop"] = buffers.geometry
    buffers["geometry"] = buffers.geometry.buffer(buffers["catchment_m"])

    nearby = gpd.sjoin(
        pc4_gdf[["pc4_clean", x_col, y_col, "geometry"]],
        buffers[["canonical_stop_id", "canonical_stop_name", "mode", "weekly_arrivals", "catchment_m", "beta", "geometry_stop", "geometry"]],
        how="left", predicate="within",
    ).dropna(subset=["canonical_stop_id"]).copy()
    nearby["distance_m"] = nearby.geometry.distance(nearby["geometry_stop"])
    nearby = nearby[nearby["distance_m"] <= nearby["catchment_m"]].copy()

    print(f"PC4-stop links: {len(nearby):,}")
    print("PC4s with access by mode:\n" + nearby.groupby("mode")["pc4_clean"].nunique().sort_values(ascending=False).to_string())
    if nearby[nearby["mode"] == "rail"].empty:
        print("WARNING: no rail links found; check rail stop coordinates or parent_station aggregation.")
    return nearby


def summarise(nearby, key):
    return nearby.groupby(["pc4_clean", key], as_index=False).agg(
        n_stops=("canonical_stop_id", "nunique"), weekly_arrivals=("weekly_arrivals", "sum"),
        decay_access=("decay_weighted_arrivals", "sum"), mean_distance_m=("distance_m", "mean"), min_distance_m=("distance_m", "min"),
    )


def to_wide(df, key):
    wide = df.pivot(index="pc4_clean", columns=key)
    wide.columns = [f"{metric}_{group}" for metric, group in wide.columns]
    return wide.reset_index()


def add_normalised_index_and_quintile(df, col):
    maximum = df[col].max()
    df[f"{col}_norm_0_100"] = df[col] / maximum * 100 if pd.notna(maximum) and maximum > 0 else 0.0
    try:  # labels=False avoids failure when ties drop bin edges
        df[f"{col}_quintile"] = (pd.qcut(df[col], q=5, labels=False, duplicates="drop") + 1).astype("Int64")
        n_groups = df[f"{col}_quintile"].nunique(dropna=True)
        if n_groups < 5:
            print(f"WARNING: {col} produced {n_groups} quantile groups instead of 5 because of ties.")
    except ValueError as exc:
        print(f"WARNING: Could not create quintiles for {col}: {exc}")
        df[f"{col}_quintile"] = pd.Series(pd.NA, index=df.index, dtype="Int64")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rail-catchment", type=int, default=PT_CATCHMENTS_M["rail"],
                        help=f"Rail catchment in metres (main: {PT_CATCHMENTS_M['rail']}, sensitivity: {RAIL_CATCHMENT_SENSITIVITY_M})")
    args = parser.parse_args()

    catchments = {**PT_CATCHMENTS_M, "rail": args.rail_catchment}
    sensitivity = args.rail_catchment != PT_CATCHMENTS_M["rail"]
    if sensitivity and args.rail_catchment != RAIL_CATCHMENT_SENSITIVITY_M:
        raise ValueError(f"Only the {RAIL_CATCHMENT_SENSITIVITY_M} m sensitivity has registered output paths.")
    out_csv, out_gpkg, out_links = (
        (paths.PT_ACCESSIBILITY_TRAIN4000_CSV, paths.PT_ACCESSIBILITY_TRAIN4000_GPKG, paths.PT_STOP_LINKS_TRAIN4000_CSV) if sensitivity
        else (paths.PT_ACCESSIBILITY_CSV, paths.PT_ACCESSIBILITY_GPKG, paths.PT_STOP_LINKS_CSV)
    )
    if sensitivity:
        header(f"TRAIN CATCHMENT SENSITIVITY: RAIL CATCHMENT = {args.rail_catchment:,} m")

    paths.require(paths.GTFS_DIR, paths.PC4_CENTROIDS_CSV)
    source = find_gtfs_source(paths.GTFS_DIR)

    trip_weights, week = load_active_trips(source)
    stops_gdf = build_canonical_stops(source, count_stop_arrivals(source, trip_weights), catchments)
    pc4_gdf, x_col, y_col = load_centroids()
    nearby = link_pc4_to_stops(pc4_gdf, stops_gdf, x_col, y_col)

    header("DISTANCE-DECAY ACCESSIBILITY")
    nearby["decay_weight"] = np.exp(-nearby["beta"] * (nearby["distance_m"] / nearby["catchment_m"]))
    nearby["decay_weighted_arrivals"] = nearby["weekly_arrivals"] * nearby["decay_weight"]
    # ODiN asks about bus/tram/metro together (FqBTM) and train separately (FqTrein), so group to match.
    # Grouping from the links counts a stop serving several BTM modes only once in n_stops_btm.
    nearby["pt_group"] = np.where(nearby["mode"].eq("rail"), "train", "btm")

    mode_wide = to_wide(summarise(nearby, "mode"), "mode")
    group_wide = to_wide(summarise(nearby, "pt_group"), "pt_group").rename(
        columns={"decay_access_btm": "pt_access_decay_btm", "decay_access_train": "pt_access_decay_train"})
    overall = nearby.groupby("pc4_clean", as_index=False).agg(
        n_pt_stops_total=("canonical_stop_id", "nunique"), weekly_arrivals_total=("weekly_arrivals", "sum"),
        pt_access_decay_total=("decay_weighted_arrivals", "sum"),
        mean_distance_to_accessible_stop_m=("distance_m", "mean"), min_distance_to_accessible_stop_m=("distance_m", "min"),
    )

    # Every PC4 is kept; PC4s without stops get zero access
    result = (pc4_gdf.drop(columns="geometry").merge(overall, on="pc4_clean", how="left")
              .merge(mode_wide, on="pc4_clean", how="left").merge(group_wide, on="pc4_clean", how="left"))
    zero_cols = [c for c in result.columns if c.startswith(("n_stops_", "weekly_arrivals_", "decay_access_", "pt_access_decay_"))
                 or c in ["n_pt_stops_total", "weekly_arrivals_total", "pt_access_decay_total"]]
    result[zero_cols] = result[zero_cols].fillna(0)
    for col in ["pt_access_decay_total", "pt_access_decay_btm", "pt_access_decay_train"]:
        add_normalised_index_and_quintile(result, col)

    header("EXPORTING")
    save_csv(result, out_csv)
    gpd.GeoDataFrame(result, geometry=gpd.points_from_xy(result[x_col], result[y_col]), crs=TARGET_CRS).to_file(out_gpkg, driver="GPKG")
    links = nearby.drop(columns="geometry", errors="ignore").copy()
    links["geometry_stop_wkt"] = links["geometry_stop"].to_wkt()
    save_csv(links.drop(columns="geometry_stop"), out_links)
    for f in [out_csv, out_gpkg, out_links]:
        print(f"Saved: {f}")

    header("SUMMARY")
    print(f"Representative week: {week[0]} to {week[-1]}")
    print(f"PC4 centroids: {len(pc4_gdf):,}; with any PT stop: {(result['n_pt_stops_total'] > 0).sum():,}; "
          f"without: {(result['n_pt_stops_total'] == 0).sum():,}")
    for label, n_col, idx in [("BTM", "n_stops_btm", "pt_access_decay_btm"), ("Train", "n_stops_train", "pt_access_decay_train")]:
        print(f"{label}: {(result[n_col] > 0).sum():,} PC4s with access, mean {result[idx].mean():,.2f}, median {result[idx].median():,.2f}")
    print("Catchments: " + ", ".join(f"{m} {c} m" for m, c in catchments.items()))


if __name__ == "__main__":
    main()
