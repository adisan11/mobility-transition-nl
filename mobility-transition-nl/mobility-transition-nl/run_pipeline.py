"""
Runs the pipeline in dependency order and writes a run log (logs/run_<timestamp>.json) with the git commit,
Python and package versions, data locations, each step's status and duration, and SHA-256 hashes of every
processed dataset produced.

    python run_pipeline.py                      # everything, on the real data in data/
    python run_pipeline.py --stage spatial      # one or more stages
    python run_pipeline.py --demo               # synthetic data in demo/ (no ODiN, OSM or GTFS needed)
    python run_pipeline.py --list               # show the steps
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

# (stage, script, extra args, skip reason if an upstream script is not in the repo yet)
STEPS = [
    ("spatial", "01_data_cleaning/build_pc4_population_centroids.py", [], None),
    ("spatial", "02_accessibility/build_pt_accessibility.py", [], None),
    ("spatial", "02_accessibility/build_pt_accessibility.py", ["--rail-catchment", "4000"], None),
    ("spatial", "02_accessibility/build_ev_accessibility.py", [], None),
    ("spatial", "02_accessibility/audit_ev_charging_stations.py", [], None),
    ("groups", "01_data_cleaning/merge_odin_years.py", [], None),
    ("groups", "01_data_cleaning/build_threshold_mobility_groups.py", [], None),
    ("groups", "01_data_cleaning/merge_threshold_groups_into_odin.py", [], None),
    ("groups", "01_data_cleaning/create_respondent_level_odin.py", [], None),
    ("groups", "01_data_cleaning/merge_group2_accessibility.py", [], None),
    ("analysis", "03_analysis/run_group2_binary_regressions.py", [], None),
    ("analysis", "03_analysis/group2_household_car_status.py", [], None),
    ("analysis", "03_analysis/plot_threshold_group_profiles.py", [], None),
    ("analysis", "03_analysis/plot_mobility_profile_distribution.py", [], None),
    ("analysis", "03_analysis/pt_quality_control_and_descriptives.py", [], None),
    ("analysis", "03_analysis/run_pt_robustness_checks.py", [], None),
    ("analysis", "03_analysis/plot_pt_interaction_observed_ranges.py", [], "needs PT ordinal model output (script not yet in repo)"),
    ("analysis", "03_analysis/train_access_by_urbanity.py", [], None),
    ("analysis", "03_analysis/train_access_mode_distribution.py", [], None),
    ("analysis", "03_analysis/ev_accessibility_inequality.py", [], None),
    ("maps", "04_maps/map_pt_accessibility.py", [], None),
    ("maps", "04_maps/map_ev_accessibility.py", [], None),
    ("maps", "04_maps/map_group2_profiles_by_pc4.py", [], None),
    ("maps", "03_analysis/pc4_sample_support_diagnostics.py", [], None),
    ("maps", "04_maps/map_group2_profiles_by_municipality.py", [], None),
]
STAGES = ["spatial", "groups", "analysis", "maps"]
PACKAGES = ["numpy", "pandas", "geopandas", "shapely", "pyogrio", "matplotlib", "scipy", "patsy", "osmium", "tqdm", "mapclassify", "jinja2"]
HASH_LIMIT_BYTES = 500 * 1024 ** 2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        return f"{commit}{' (uncommitted changes)' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return None


def versions():
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", nargs="+", choices=["all", *STAGES], default=["all"])
    parser.add_argument("--demo", action="store_true", help="generate and use synthetic data in demo/")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed step")
    parser.add_argument("--list", action="store_true", help="list steps and exit")
    args = parser.parse_args()

    stages = STAGES if "all" in args.stage else [s for s in STAGES if s in args.stage]
    steps = [s for s in STEPS if s[0] in stages]
    if args.list:
        for stage, script, extra, skip in steps:
            print(f"{stage:<9} {script} {' '.join(extra)}" + (f"   [skipped: {skip}]" if skip else ""))
        return

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    if args.demo:
        env["MOBILITY_DATA_DIR"] = str(ROOT / "demo" / "data")
        env["MOBILITY_OUTPUT_DIR"] = str(ROOT / "demo" / "outputs")
        steps = [("demo", "00_demo/make_synthetic_data.py", [], None)] + steps

    started = datetime.now()
    log = {
        "started": started.isoformat(timespec="seconds"), "git_commit": git_commit(), "versions": versions(),
        "demo": args.demo, "data_dir": env.get("MOBILITY_DATA_DIR", str(ROOT / "data")),
        "output_dir": env.get("MOBILITY_OUTPUT_DIR", str(ROOT / "outputs")), "steps": [],
    }

    failed = False
    for stage, script, extra, skip in steps:
        name = f"{script} {' '.join(extra)}".strip()
        if skip:
            print(f"\n--- SKIP {name}: {skip}")
            log["steps"].append({"stage": stage, "step": name, "status": "skipped", "reason": skip})
            continue
        print(f"\n=== [{stage}] {name}")
        t0 = time.time()
        result = subprocess.run([sys.executable, str(SCRIPTS / script), *extra], cwd=ROOT, env=env)
        status = "ok" if result.returncode == 0 else "failed"
        log["steps"].append({"stage": stage, "step": name, "status": status, "seconds": round(time.time() - t0, 1)})
        if status == "failed":
            failed = True
            print(f"!!! {name} failed (exit {result.returncode})")
            if not args.keep_going:
                break

    processed = Path(log["data_dir"]) / "processed"
    log["processed_files"] = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256(p) if p.stat().st_size <= HASH_LIMIT_BYTES else None}
        for p in sorted(processed.glob("*")) if p.is_file() and p.name != ".gitkeep"
    } if processed.exists() else {}
    log["finished"] = datetime.now().isoformat(timespec="seconds")

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"run_{started:%Y%m%d_%H%M%S}{'_demo' if args.demo else ''}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    counts = {s: sum(r["status"] == s for r in log["steps"]) for s in ["ok", "failed", "skipped"]}
    print(f"\nSteps: {counts['ok']} ok, {counts['failed']} failed, {counts['skipped']} skipped. Log: {log_path.relative_to(ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
