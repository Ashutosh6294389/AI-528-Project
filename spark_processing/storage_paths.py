from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT / "storage"
DELTA_ROOT = STORAGE_ROOT / "delta_tables"
CHECKPOINT_ROOT = STORAGE_ROOT / "checkpoints_medallion"


def get_medallion_paths() -> dict:
    bronze_path = DELTA_ROOT / "bronze" / "youtube_raw"
    silver_path = DELTA_ROOT / "silver" / "youtube_enriched"
    gold_root = DELTA_ROOT / "gold"

    paths = {
        "bronze": str(bronze_path),
        "silver": str(silver_path),
        "gold": {
            "latest_snapshot": str(gold_root / "latest_snapshot"),
            "category_summary": str(gold_root / "category_summary"),
            "views_timeseries": str(gold_root / "views_timeseries"),
            "region_timeseries": str(gold_root / "region_timeseries"),
            "channel_leaderboard": str(gold_root / "channel_leaderboard"),
            "duration_distribution": str(gold_root / "duration_distribution"),
            "subscriber_tier_distribution": str(gold_root / "subscriber_tier_distribution"),
            "tag_usage_frequency": str(gold_root / "tag_usage_frequency"),
            "trending_rank_distribution": str(gold_root / "trending_rank_distribution"),
        },
        "checkpoint": str(CHECKPOINT_ROOT),
    }
    return paths


def ensure_medallion_paths() -> dict:
    paths = get_medallion_paths()

    Path(paths["bronze"]).mkdir(parents=True, exist_ok=True)
    Path(paths["silver"]).mkdir(parents=True, exist_ok=True)
    for gold_path in paths["gold"].values():
        Path(gold_path).mkdir(parents=True, exist_ok=True)
    Path(paths["checkpoint"]).mkdir(parents=True, exist_ok=True)

    return paths
