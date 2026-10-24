"""
Single registry of every input and output file.

Scripts never hard-code paths: they import them from here, so a file written by one step is guaranteed to be
the file read by the next. Data can live outside the repo by setting MOBILITY_DATA_DIR / MOBILITY_OUTPUT_DIR.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MOBILITY_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("MOBILITY_OUTPUT_DIR", ROOT / "outputs"))

RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
TABLES = OUTPUT_DIR / "tables"
FIGURES = OUTPUT_DIR / "figures"
MAPS = OUTPUT_DIR / "maps"

# ---- Raw inputs (download manually, see README) ----
MUNICIPALITY_GPKG = RAW / "cbs" / "BestuurlijkeGebieden_2023.gpkg"
VK100_GPKG = RAW / "cbs" / "cbs_vk100_2023.gpkg"
PC4_STATS_CSV = RAW / "cbs" / "pc4_stats.csv"
PC4_POLYGONS_2023_GPKG = RAW / "cbs" / "cbs_pc4_2023_v2.gpkg"
PC4_POLYGONS_2024_GPKG = RAW / "cbs" / "cbs_pc4_2024_v1.gpkg"
OSM_PBF = RAW / "osm" / "netherlands-260614.osm.pbf"
GTFS_DIR = RAW / "gtfs"
ODIN_DIR = RAW / "odin"

# ---- Processed data (created by scripts) ----
PC4_CENTROIDS_CSV = PROCESSED / "pc4_population_weighted_centroids.csv"
PC4_CENTROIDS_GPKG = PROCESSED / "pc4_population_weighted_centroids.gpkg"
PC4_WITH_CENTROIDS_CSV = PROCESSED / "pc4_stats_with_population_weighted_centroids.csv"
PC4_WITH_CENTROIDS_GPKG = PROCESSED / "pc4_stats_with_population_weighted_centroids.gpkg"

# ODiN pipeline (threshold mobility groups)
ODIN_MERGED_CSV = PROCESSED / "ODiN2022_2023_merged_common_columns_only.csv"
THRESHOLD_GROUPS_CSV = PROCESSED / "threshold_analysis_sample_with_groups.csv"
THRESHOLD_TABLES = TABLES / "threshold_groups"
THRESHOLD_GROUP_SUMMARY_CSV = THRESHOLD_TABLES / "03_group_summary_main_and_sensitivity.csv"
THRESHOLD_FREQUENCY_CSV = THRESHOLD_TABLES / "04_frequency_distribution_by_group_mode.csv"
MASTER_WITH_GROUPS_CSV = PROCESSED / "ODiN2022_2023_mobility_transition_master_with_mobility_groups.csv"
RESPONDENT_LEVEL_CSV = PROCESSED / "ODiN2022_2023_mobility_transition_respondent_level.csv"
GROUP2_ANALYSIS_CSV = PROCESSED / "group2_respondents_with_sustainability_accessibility.csv"
GROUP2_PC4_RESPONDENTS_CSV = PROCESSED / "group2_pc4_unique_opid_respondent_sample.csv"

# Master respondent dataset (one row per ODiN respondent)
MASTER_CSV = PROCESSED / "ODiN2022_2023_mobility_transition_master.csv"
MASTER_UNMATCHED_PC4_CSV = PROCESSED / "mobility_transition_unmatched_pc4.csv"
ODIN_TRIPS_CSV = PROCESSED / "ODiN2022_2023_pt.csv"  # trip/leg-level ODiN 2022-2023

FEEDER_ACCESSIBILITY_CSV = PROCESSED / "pc4_btm_feeder_train_accessibility.csv"

EV_STATIONS_GPKG = PROCESSED / "osm_ev_charging_stations.gpkg"
EV_ACCESSIBILITY_CSV = PROCESSED / "pc4_ev_accessibility.csv"
EV_ACCESSIBILITY_GPKG = PROCESSED / "pc4_ev_accessibility.gpkg"
EV_STATION_LINKS_CSV = PROCESSED / "pc4_ev_station_links_debug.csv"
EV_ACCESSIBILITY_ENRICHED_CSV = PROCESSED / "pc4_ev_accessibility_enriched.csv"

PT_ACCESSIBILITY_CSV = PROCESSED / "pc4_pt_accessibility_distance_decay.csv"
PT_ACCESSIBILITY_GPKG = PROCESSED / "pc4_pt_accessibility_distance_decay.gpkg"
PT_STOP_LINKS_CSV = PROCESSED / "pc4_pt_stop_links_debug.csv"
PT_ACCESSIBILITY_TRAIN4000_CSV = PROCESSED / "pc4_pt_accessibility_distance_decay_train_4000m.csv"
PT_ACCESSIBILITY_TRAIN4000_GPKG = PROCESSED / "pc4_pt_accessibility_distance_decay_train_4000m.gpkg"
PT_STOP_LINKS_TRAIN4000_CSV = PROCESSED / "pc4_pt_stop_links_debug_train_4000m.csv"

# Model outputs read by later scripts
PT_ORDINAL_INTERACTION_PREDICTIONS_CSV = TABLES / "pt_ordinal_models" / "07_accessibility_urbanity_interaction_predictions.csv"

# Which script creates each processed file, so missing inputs give a useful error
PRODUCED_BY = {
    PC4_CENTROIDS_CSV: "scripts/01_data_cleaning/build_pc4_population_centroids.py",
    PC4_CENTROIDS_GPKG: "scripts/01_data_cleaning/build_pc4_population_centroids.py",
    PC4_WITH_CENTROIDS_CSV: "scripts/01_data_cleaning/build_pc4_population_centroids.py",
    PC4_WITH_CENTROIDS_GPKG: "scripts/01_data_cleaning/build_pc4_population_centroids.py",
    EV_STATIONS_GPKG: "scripts/02_accessibility/build_ev_accessibility.py",
    EV_ACCESSIBILITY_CSV: "scripts/02_accessibility/build_ev_accessibility.py",
    EV_ACCESSIBILITY_GPKG: "scripts/02_accessibility/build_ev_accessibility.py",
    ODIN_MERGED_CSV: "scripts/01_data_cleaning/merge_odin_years.py",
    THRESHOLD_GROUPS_CSV: "scripts/01_data_cleaning/build_threshold_mobility_groups.py",
    THRESHOLD_GROUP_SUMMARY_CSV: "scripts/01_data_cleaning/build_threshold_mobility_groups.py",
    THRESHOLD_FREQUENCY_CSV: "scripts/01_data_cleaning/build_threshold_mobility_groups.py",
    MASTER_WITH_GROUPS_CSV: "scripts/01_data_cleaning/merge_threshold_groups_into_odin.py",
    RESPONDENT_LEVEL_CSV: "scripts/01_data_cleaning/create_respondent_level_odin.py",
    GROUP2_ANALYSIS_CSV: "scripts/01_data_cleaning/merge_group2_accessibility.py",
    GROUP2_PC4_RESPONDENTS_CSV: "scripts/04_maps/map_group2_profiles_by_pc4.py",
    FEEDER_ACCESSIBILITY_CSV: "scripts/02_accessibility/build_feeder_train_accessibility.py (not yet added)",
    MASTER_CSV: "scripts/01_data_cleaning/build_master_dataset.py",
    MASTER_UNMATCHED_PC4_CSV: "scripts/01_data_cleaning/build_master_dataset.py",
    PT_ACCESSIBILITY_CSV: "scripts/02_accessibility/build_pt_accessibility.py",
    PT_ACCESSIBILITY_GPKG: "scripts/02_accessibility/build_pt_accessibility.py",
    PT_ACCESSIBILITY_TRAIN4000_CSV: "scripts/02_accessibility/build_pt_accessibility.py --rail-catchment 4000",
    PT_ORDINAL_INTERACTION_PREDICTIONS_CSV: "scripts/03_analysis/run_pt_ordinal_models.py",
    EV_ACCESSIBILITY_ENRICHED_CSV: "scripts/03_analysis/ev_accessibility_inequality.py",
}


def require(*files):
    """Stop early if inputs are missing, saying which script (or download) provides each one."""
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        lines = [f"  {f}\n    -> run {PRODUCED_BY[f]}" if f in PRODUCED_BY else f"  {f}\n    -> raw input, see README"
                 for f in missing]
        raise FileNotFoundError("Missing input files:\n" + "\n".join(lines))


def step_dir(base, name):
    """Output subfolder for one script, created on first use."""
    folder = Path(base) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder
