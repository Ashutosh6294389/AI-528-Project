import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT / "storage"
RUNS_ROOT = STORAGE_ROOT / "runs"
ACTIVE_RUN_FILE = STORAGE_ROOT / "active_run.json"


def build_run_paths(run_id: str) -> dict:
    run_root = RUNS_ROOT / run_id
    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "bronze": str(run_root / "delta_tables" / "bronze" / "youtube_raw"),
        "silver": str(run_root / "delta_tables" / "silver" / "youtube_enriched"),
        "gold": {
            "latest_snapshot": str(run_root / "delta_tables" / "gold" / "latest_snapshot"),
            "category_summary": str(run_root / "delta_tables" / "gold" / "category_summary"),
            "views_timeseries": str(run_root / "delta_tables" / "gold" / "views_timeseries"),
            "region_timeseries": str(run_root / "delta_tables" / "gold" / "region_timeseries"),
            "channel_leaderboard": str(run_root / "delta_tables" / "gold" / "channel_leaderboard"),
        },
        "checkpoint": str(run_root / "checkpoints" / "medallion"),
    }


def create_new_active_run(delete_older_runs: bool = True) -> dict:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = build_run_paths(run_id)

    Path(paths["bronze"]).mkdir(parents=True, exist_ok=True)
    Path(paths["silver"]).mkdir(parents=True, exist_ok=True)
    for gold_path in paths["gold"].values():
        Path(gold_path).mkdir(parents=True, exist_ok=True)
    Path(paths["checkpoint"]).mkdir(parents=True, exist_ok=True)

    ACTIVE_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_RUN_FILE.write_text(json.dumps(paths, indent=2))

    if delete_older_runs:
        cleanup_old_runs(run_id)

    return paths


def cleanup_old_runs(active_run_id: str) -> None:
    if not RUNS_ROOT.exists():
        return

    for child in RUNS_ROOT.iterdir():
        if child.is_dir() and child.name != active_run_id:
            shutil.rmtree(child, ignore_errors=True)


def get_active_run_paths() -> dict | None:
    if ACTIVE_RUN_FILE.exists():
        return json.loads(ACTIVE_RUN_FILE.read_text())

    if not RUNS_ROOT.exists():
        return None

    run_dirs = sorted([path for path in RUNS_ROOT.iterdir() if path.is_dir()])
    if not run_dirs:
        return None

    latest_run = run_dirs[-1].name
    paths = build_run_paths(latest_run)
    ACTIVE_RUN_FILE.write_text(json.dumps(paths, indent=2))
    return paths
