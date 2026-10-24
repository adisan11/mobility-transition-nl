"""Project-wide analytical settings."""

TARGET_CRS = "EPSG:28992"  # RD New, metres; used for all distance work
RANDOM_SEED = 42

PC4_CANDIDATES = [
    "pc4_clean", "Postal code", "postcode", "Postcode", "pc4", "PC4", "postcode4", "Postcode4", "VertPC",
]

# PT accessibility (straight-line catchments from PC4 population-weighted centroids)
PT_CATCHMENTS_M = {"bus": 1500, "tram": 1000, "metro": 1500, "rail": 3500}
RAIL_CATCHMENT_SENSITIVITY_M = 4000
PT_BETA = {"bus": 1.5, "tram": 1.5, "metro": 1.5, "rail": 1.5}  # decay = exp(-beta * distance / catchment)
GTFS_ROUTE_TYPE_TO_MODE = {0: "tram", 1: "metro", 2: "rail", 3: "bus"}  # ferry (4) excluded

URBANITY_LABELS = {
    1: "1 Very highly urban", 2: "2 Highly urban", 3: "3 Moderately urban", 4: "4 Slightly urban", 5: "5 Non-urban",
}

# ODiN leg mode (Rvm) codes, as used in the train access-chain analysis
RVM_CAR, RVM_TRAIN, RVM_WALK = 1, 2, 9
RVM_FEEDER_PT = {3, 4, 5}  # bus, tram, metro
RVM_CYCLE = {6, 7, 8}  # speed pedelec, e-bike, bicycle
RVM_LABELS = {1: "Passenger car", 2: "Train", 3: "Bus", 4: "Tram", 5: "Metro", 6: "Speed pedelec",
              7: "Electric bicycle", 8: "Bicycle", 9: "Walking"}
ODIN_OWN_HOME = 1  # VertLoc code: trip starts at own home

# Threshold mobility groups. Group 1 = main specification, Group 2 = sensitivity (stricter PT threshold).
FREQUENCY_CODES = [1, 2, 3, 4, 5]  # 1 = daily ... 5 = never (FqAutoB, FqBTM, FqTrein)
FREQUENCY_LABELS = {1: "Daily/almost daily", 2: "Several times/week", 3: "Few times/month", 4: "Few times/year", 5: "Never/almost never"}
THRESHOLD_GROUP_RENAME = {
    "Car-dependent, low PT use": "Car users",
    "Multimodal regular car + PT": "Multimodal users",
    "Car-light PT user": "PT users",
    "Car-light low-PT user": "Infrequent car and PT users",
}
MOBILITY_GROUPS = list(THRESHOLD_GROUP_RENAME.values())
