# Mobility transition in the Netherlands: PT and EV accessibility at PC4 level

[!\[demo-pipeline](https://github.com/adisan11/mobility-transition-nl/actions/workflows/demo-pipeline.yml/badge.svg)](https://github.com/YOUR-USERNAME/mobility-transition-nl/actions/workflows/demo-pipeline.yml)

Code for an MSc thesis on structural car dependence and the availability of low-carbon alternatives in the
Netherlands, using the ODiN 2022–2023 travel survey. It builds population-weighted PC4 centroids, public-transport and
EV-charging accessibility indices, threshold-based mobility profiles, and weighted PC4-clustered regression models.

## Quick start (no data needed)

```bash
python -m venv .venv \&\& source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
pip install -r requirements-lock.txt \&\& pip install -e .
python run\_pipeline.py --demo
```

`--demo` generates a small synthetic version of every input in `demo/` and runs the full pipeline on it
(about a minute). The numbers are meaningless; the run shows that every step executes and hands its outputs to the
next. The same check runs on GitHub Actions on every push.

## Reproducibility

* **One entry point.** `python run\_pipeline.py` runs all steps in dependency order; `--stage spatial groups analysis maps`
runs a subset, `--list` shows the steps, `--keep-going` continues past failures.
* **One path registry.** No script contains a file path. `src/mobility\_transition/paths.py` defines every input and
output once, so each step reads exactly what the previous step wrote; a missing input stops the script and names the
step that creates it.
* **One parameter registry.** Catchments, decay parameters, CRS, ODiN codes and the random seed live in
`src/mobility\_transition/settings.py`.
* **Run logs.** Every run writes `logs/run\_<timestamp>.json` with the git commit (and whether the working tree was
clean), Python and package versions, data locations, each step's status and runtime, and SHA-256 hashes of every
processed dataset. Two runs on the same inputs give identical CSV hashes (GeoPackages embed a write timestamp, so
their hashes differ).
* **Pinned environment.** `requirements-lock.txt` has the exact tested versions; `requirements.txt` the minimums.

## Repository layout

```
├── run\_pipeline.py            runs every step in order and writes a run log
├── src/mobility\_transition/   shared code
│   ├── paths.py               every input/output file, defined once
│   ├── settings.py            parameters (catchments, decay, CRS, codes, seed)
│   ├── io.py, stats.py        reading/saving, weighted statistics
│   ├── models.py              weighted logit with PC4-clustered standard errors
│   └── odin.py, profiles.py   ODiN and mobility-profile helpers
├── scripts/
│   ├── 00\_demo/               synthetic data generator
│   ├── 01\_data\_cleaning/      raw data -> analytical datasets
│   ├── 02\_accessibility/      PT and EV accessibility indices
│   ├── 03\_analysis/           descriptives, regressions, diagnostics
│   └── 04\_maps/               map production
├── data/raw/, data/processed/ real inputs and generated datasets (not tracked)
└── outputs/                   tables/, figures/, maps/ (not tracked)
```

Data can live outside the repository: set `MOBILITY\_DATA\_DIR` and `MOBILITY\_OUTPUT\_DIR`.

## Raw data

Place these files in `data/raw/` (names as set in `paths.py`):

|File|Folder|Source|
|-|-|-|
|`cbs\_vk100\_2023.gpkg`|`cbs/`|CBS 100 m population grid (Vierkantstatistieken)|
|`pc4\_stats.csv`|`cbs/`|CBS statistics per 4-digit postcode, incl. urbanity|
|`cbs\_pc4\_2023\_v2.gpkg`, `cbs\_pc4\_2024\_v1.gpkg`|`cbs/`|CBS PC4 boundaries|
|`BestuurlijkeGebieden\_2023.gpkg`|`cbs/`|Municipal boundaries (PDOK)|
|`netherlands-260614.osm.pbf`|`osm/`|Geofabrik OpenStreetMap extract|
|GTFS feed|`gtfs/`|Dutch national GTFS|
|`ODiN2022\_Databestand.csv`, `ODiN2023\_Databestand.csv`|`odin/`|CBS / DANS (licensed microdata)|

**ODiN is licensed microdata. Never commit it or any respondent-level file derived from it.** `.gitignore` excludes
everything under `data/`, `outputs/`, `demo/` and `logs/`.

## Project status

The repository covers the spatial indices, the threshold mobility-group analysis, and the PT/EV descriptive and
robustness analyses. Not yet included:

* the preparation of the leg-level file `ODiN2022\_2023\_pt.csv`;
* the BTM feeder-to-train accessibility index (`pc4\_btm\_feeder\_train\_accessibility.csv`);
* the respondent master dataset used by the PT descriptive and robustness scripts;
* the PT ordinal models (the interaction-plot step is skipped until they are added);
* the OD routing and PT conversion-friction analyses.

In `--demo` mode the first three are replaced by synthetic placeholders so the downstream steps still run.

## Run order

`run\_pipeline.py` runs these in order. Scripts can also be run one at a time from the repository root; each stops with a
clear message naming the step to run first if an input is missing.

**Spatial inputs**

|#|Script|Creates|
|-|-|-|
|1|`01\_data\_cleaning/build\_pc4\_population\_centroids.py`|`pc4\_population\_weighted\_centroids.\*`|
|2|`02\_accessibility/build\_pt\_accessibility.py`|`pc4\_pt\_accessibility\_distance\_decay.\*`|
|3|`02\_accessibility/build\_pt\_accessibility.py --rail-catchment 4000`|`\*\_train\_4000m` sensitivity index|
|4|`02\_accessibility/build\_ev\_accessibility.py`|`osm\_ev\_charging\_stations.gpkg`, `pc4\_ev\_accessibility.\*`|
|5|`02\_accessibility/audit\_ev\_charging\_stations.py`|`outputs/tables/ev\_station\_audit/`|
|6|*feeder-to-train accessibility (script pending)*|`pc4\_btm\_feeder\_train\_accessibility.csv`|

**Threshold mobility groups (Group 1 = main, Group 2 = sensitivity)**

|#|Script|Creates|
|-|-|-|
|7|`01\_data\_cleaning/merge\_odin\_years.py`|`ODiN2022\_2023\_merged\_common\_columns\_only.csv`|
|8|`01\_data\_cleaning/build\_threshold\_mobility\_groups.py`|`threshold\_analysis\_sample\_with\_groups.csv`, group summaries|
|9|`01\_data\_cleaning/merge\_threshold\_groups\_into\_odin.py`|`...\_master\_with\_mobility\_groups.csv`|
|10|`01\_data\_cleaning/create\_respondent\_level\_odin.py`|`...\_respondent\_level.csv`|
|11|`01\_data\_cleaning/merge\_group2\_accessibility.py`|`group2\_respondents\_with\_sustainability\_accessibility.csv`|
|12|`03\_analysis/run\_group2\_binary\_regressions.py`|models A-D, thesis tables (CSV + LaTeX)|
|13|`03\_analysis/group2\_household\_car\_status.py`|ICEV / EV / no-car shares by profile|
|14|`03\_analysis/plot\_threshold\_group\_profiles.py`|Group 1 vs Group 2 figures|
|15|`03\_analysis/plot\_mobility\_profile\_distribution.py`|profile distribution bar chart (hard-coded values)|
|16|`04\_maps/map\_group2\_profiles\_by\_pc4.py`|PC4 profile maps, `group2\_pc4\_unique\_opid\_respondent\_sample.csv`|
|17|`03\_analysis/pc4\_sample\_support\_diagnostics.py`|PC4 sample-size / stability diagnostics|
|18|`04\_maps/map\_group2\_profiles\_by\_municipality.py`|municipality percentile maps|

**PT and EV analyses**

|#|Script|Creates|
|-|-|-|
|19|`03\_analysis/pt\_quality\_control\_and\_descriptives.py`|`outputs/\*/pt\_descriptives/`|
|20|*PT ordinal models (script pending)*|`outputs/tables/pt\_ordinal\_models/`|
|21|`03\_analysis/plot\_pt\_interaction\_observed\_ranges.py`|observed-range interaction figures|
|22|`03\_analysis/run\_pt\_robustness\_checks.py`|`outputs/\*/pt\_robustness/`|
|23|`03\_analysis/train\_access\_by\_urbanity.py`|`outputs/\*/train\_access\_urbanity/`|
|24|`03\_analysis/train\_access\_mode\_distribution.py`|`outputs/\*/train\_access\_mode\_distribution/`|
|25|`03\_analysis/ev\_accessibility\_inequality.py`|`pc4\_ev\_accessibility\_enriched.csv`, Lorenz/Gini|
|26|`04\_maps/map\_pt\_accessibility.py`|`outputs/maps/pt\_accessibility/`|
|27|`04\_maps/map\_ev\_accessibility.py`|`outputs/maps/ev\_accessibility/`|



