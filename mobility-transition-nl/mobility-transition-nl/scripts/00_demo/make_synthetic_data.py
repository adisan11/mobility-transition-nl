"""
Generates a small, entirely synthetic version of every raw input (plus placeholders for the few inputs whose
build scripts are not in this repository yet), so the full pipeline can be run and tested without the licensed
ODiN microdata or the large OSM/GTFS/CBS downloads.

Numbers produced from this data are meaningless; it only exercises the code paths.
Run it through `python run_pipeline.py --demo`, which points MOBILITY_DATA_DIR / MOBILITY_OUTPUT_DIR at demo/.
"""
import geopandas as gpd
import numpy as np
import osmium
import pandas as pd
from shapely.geometry import box

from mobility_transition import paths
from mobility_transition.io import save_csv

SEED = 7
N_COLS, N_ROWS, CELL = 10, 6, 2000  # 60 PC4 areas of 2 x 2 km
X0, Y0 = 140000, 450000  # RD New, around Utrecht
N_RESPONDENTS_PER_YEAR = 2500


def pc4_grid():
    codes, polys = [], []
    for i in range(N_COLS * N_ROWS):
        cx, cy = X0 + (i % N_COLS) * CELL, Y0 + (i // N_COLS) * CELL
        codes.append(str(3500 + i))
        polys.append(box(cx, cy, cx + CELL, cy + CELL))
    return gpd.GeoDataFrame({"postcode": codes}, geometry=polys, crs="EPSG:28992")


def write_cbs(rng, grid):
    out = paths.RAW / "cbs"
    out.mkdir(parents=True, exist_ok=True)
    grid.to_file(paths.PC4_POLYGONS_2023_GPKG)
    grid.to_file(paths.PC4_POLYGONS_2024_GPKG)

    # 100 m population cells
    xs, ys = np.meshgrid(np.arange(X0, X0 + N_COLS * CELL, 100), np.arange(Y0, Y0 + N_ROWS * CELL, 100))
    cells = [box(x, y, x + 100, y + 100) for x, y in zip(xs.ravel(), ys.ravel())]
    gpd.GeoDataFrame({"aantal_inwoners": rng.integers(-5, 40, len(cells))}, geometry=cells, crs="EPSG:28992").to_file(paths.VK100_GPKG)

    codes = grid["postcode"].astype(int)
    save_csv(pd.DataFrame({"Postal code": codes, "Number of inhabitants": rng.integers(500, 9000, len(codes)),
                           "urbanity": rng.integers(1, 6, len(codes))}), paths.PC4_STATS_CSV)

    munis = grid.assign(gm=codes % 12 + 1).dissolve("gm").reset_index()
    munis["statcode"] = ["GM%04d" % g for g in munis["gm"]]
    munis[["statcode", "geometry"]].to_file(paths.MUNICIPALITY_GPKG)


def to_lonlat(x, y):
    p = gpd.GeoSeries(gpd.points_from_xy(x, y), crs="EPSG:28992").to_crs("EPSG:4326")
    return p.x.to_numpy(), p.y.to_numpy()


def write_gtfs(rng):
    out = paths.GTFS_DIR
    out.mkdir(parents=True, exist_ok=True)
    n = 40
    lon, lat = to_lonlat(rng.uniform(X0, X0 + N_COLS * CELL, n), rng.uniform(Y0, Y0 + N_ROWS * CELL, n))
    pd.DataFrame({"stop_id": [f"s{i}" for i in range(n)], "stop_name": [f"Stop {i}" for i in range(n)],
                  "stop_lat": lat, "stop_lon": lon, "parent_station": ""}).to_csv(out / "stops.txt", index=False)
    route_types = [3] * 6 + [0, 1, 2, 2]  # bus, tram, metro, rail
    pd.DataFrame({"route_id": [f"r{i}" for i in range(len(route_types))], "route_type": route_types}).to_csv(out / "routes.txt", index=False)

    trips, stop_times = [], []
    for t in range(200):
        r = t % len(route_types)
        trips.append({"route_id": f"r{r}", "service_id": "wk" if t % 3 else "we", "trip_id": f"t{t}"})
        for s in rng.choice(n, size=4, replace=False):
            stop_times.append({"trip_id": f"t{t}", "stop_id": f"s{s}"})
    pd.DataFrame(trips).to_csv(out / "trips.txt", index=False)
    pd.DataFrame(stop_times).to_csv(out / "stop_times.txt", index=False)
    dates = pd.date_range("2026-06-08", "2026-06-21")
    pd.DataFrame({"service_id": ["wk" if d.weekday() < 5 else "we" for d in dates], "date": dates.strftime("%Y%m%d"),
                  "exception_type": 1}).to_csv(out / "calendar_dates.txt", index=False)


def write_osm(rng):
    paths.OSM_PBF.parent.mkdir(parents=True, exist_ok=True)
    paths.OSM_PBF.unlink(missing_ok=True)
    n = 120
    lon, lat = to_lonlat(rng.uniform(X0 - 5000, X0 + N_COLS * CELL + 5000, n), rng.uniform(Y0 - 5000, Y0 + N_ROWS * CELL + 5000, n))
    access = ["", "yes", "public", "customers", "private"]
    writer = osmium.SimpleWriter(str(paths.OSM_PBF))
    try:
        for i in range(n):
            tags = {"amenity": "charging_station", "access": access[i % len(access)]}
            if i % 3 == 0:
                tags["capacity"] = str(rng.integers(1, 8))
            elif i % 3 == 1:
                tags["socket:type2"] = str(rng.integers(1, 4))
            if i % 7 == 0:
                tags["socket:ccs"], tags["socket:ccs:output"] = "2", "150 kW"
            tags = {k: v for k, v in tags.items() if v}
            writer.add_node(osmium.osm.mutable.Node(id=i + 1, location=(float(lon[i]), float(lat[i])), tags=tags))
        writer.add_node(osmium.osm.mutable.Node(id=n + 1, location=(5.1, 52.1), tags={"amenity": "bench"}))
    finally:
        writer.close()


def write_odin(rng, grid):
    paths.ODIN_DIR.mkdir(parents=True, exist_ok=True)
    pcs = grid["postcode"].tolist()
    rows = []
    for year in [2022, 2023]:
        for i in range(N_RESPONDENTS_PER_YEAR):
            pc = rng.choice(pcs)
            person = {
                "OPID": f"{year}{i:05d}", "Jaar": year, "Leeftijd": rng.integers(10, 90), "FqAutoB": rng.integers(1, 7),
                "FqBTM": rng.integers(1, 6), "FqTrein": rng.integers(1, 6), "FactorP": rng.uniform(0.5, 2), "HHAuto": rng.integers(0, 3),
                "BrandstofPa1": rng.choice([1, 2, 3, 4, 7]), "Geslacht": rng.integers(1, 3), "HHGestInkG": rng.integers(1, 11),
                "HHSam": rng.integers(1, 10), "BetWerk": rng.integers(0, 6), "HHPers": rng.integers(1, 6), "WoPC": pc,
                "WoGem": int(pc) % 12 + 1, "VertLoc": 1, "OPRijbewijsAu": rng.integers(0, 2),
            }
            for journey in range(1, rng.integers(2, 5)):
                n_legs = rng.integers(1, 4)
                modes = list(rng.choice([1, 3, 8, 9], n_legs - 1)) + [2] if rng.random() < 0.3 else list(rng.choice([1, 3, 4, 7, 8, 9], n_legs))
                for leg, mode in enumerate(modes, start=1):
                    rows.append({**person, "VerplID": journey, "RitID": leg, "RitNr": leg, "VertPC": pc, "AankPC": rng.choice(pcs),
                                 "Rvm": mode, "RvmRol": 1, "AfstR": rng.integers(1, 300), "RReisduur": rng.integers(1, 60),
                                 "FactorV": rng.uniform(0.5, 2)})
    odin = pd.DataFrame(rows)
    for year in [2022, 2023]:
        odin[odin["Jaar"].eq(year)].to_csv(paths.ODIN_DIR / f"ODiN{year}_Databestand.csv", index=False, sep=";")
    return odin


def write_placeholders(rng, grid, odin):
    """Inputs whose build scripts are not in the repo yet; real versions come from those scripts."""
    pcs = grid["postcode"].tolist()
    save_csv(odin, paths.ODIN_TRIPS_CSV)
    save_csv(pd.DataFrame({"pc4_clean": pcs, "train_feeder_p85_access_decay": rng.gamma(1, 2, len(pcs))}), paths.FEEDER_ACCESSIBILITY_CSV)

    # Respondent-level master with the columns the PT descriptive/robustness scripts use
    m = odin.drop_duplicates("OPID").rename(columns={"FactorP": "person_weight"}).copy()
    m["respondent_id"], m["pc4_clean"] = m["OPID"], m["WoPC"]
    access = pd.DataFrame({"pc4_clean": pcs, "pt_access_decay_btm": rng.gamma(1, 3, len(pcs)) * rng.integers(0, 2, len(pcs)),
                           "pt_access_decay_train": rng.gamma(1, 3, len(pcs)) * rng.integers(0, 2, len(pcs))})
    m = m.merge(access, on="pc4_clean")
    m["urbanity"] = rng.integers(1, 6, len(m))
    m["income_quintile"] = np.ceil(m["HHGestInkG"] / 2).astype(int)
    for p, fq in [("btm", "FqBTM"), ("train", "FqTrein")]:
        m[f"{p}_frequency_ordered"] = 6 - m[fq]
        m[f"{p}_weekly_or_more"] = m[fq].between(1, 2).astype(int)
        m[f"pt_access_decay_{p}_quintile_pc4"] = pd.qcut(m[f"pt_access_decay_{p}"].rank(method="first"), 5, labels=False) + 1
        m[f"log1p_pt_access_decay_{p}"] = np.log1p(m[f"pt_access_decay_{p}"])
    save_csv(m, paths.MASTER_CSV)


def main():
    rng = np.random.default_rng(SEED)
    grid = pc4_grid()
    write_cbs(rng, grid)
    write_gtfs(rng)
    write_osm(rng)
    odin = write_odin(rng, grid)
    write_placeholders(rng, grid, odin)
    print(f"Synthetic demo data written to {paths.DATA_DIR}")


if __name__ == "__main__":
    main()
