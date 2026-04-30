"""Resolve where each medallion layer lives.

This used to return file-system paths to Delta tables. Storage now lives
in MongoDB, so every layer is identified by a collection name. Streaming
checkpoints are the one piece that has to stay on disk (Spark requirement),
so `checkpoint` is still a file-system path.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT / "storage"
CHECKPOINT_ROOT = STORAGE_ROOT / "checkpoints_medallion"


# Collection names — kept stable so docs / queries / dashboards can refer
# to them by name. Override the URI / DB at runtime via env vars on the
# mongo_io module.
BRONZE_COLLECTION = "bronze_raw"
SILVER_COLLECTION = "silver_enriched"
GOLD_COLLECTIONS = {
    "latest_snapshot": "gold_latest_snapshot",
    "category_summary": "gold_category_summary",
    "views_timeseries": "gold_views_timeseries",
    "region_timeseries": "gold_region_timeseries",
    "channel_leaderboard": "gold_channel_leaderboard",
    "duration_distribution": "gold_duration_distribution",
    "subscriber_tier_distribution": "gold_subscriber_tier_distribution",
    "tag_usage_frequency": "gold_tag_usage_frequency",
    "trending_rank_distribution": "gold_trending_rank_distribution",
}


def get_medallion_paths() -> dict:
    """Return the paths/collections every layer of the pipeline uses.

    Keys are kept identical to the previous Delta layout so consumers
    don't have to learn a new shape — the values now refer to MongoDB
    collections instead of disk paths.
    """
    return {
        "bronze": BRONZE_COLLECTION,
        "silver": SILVER_COLLECTION,
        "gold": dict(GOLD_COLLECTIONS),
        "checkpoint": str(CHECKPOINT_ROOT),
    }


def ensure_medallion_paths() -> dict:
    """Create the streaming-checkpoint directory and ensure all MongoDB
    collections + their indexes exist. Idempotent — safe to call on every
    pipeline start.
    """
    paths = get_medallion_paths()
    Path(paths["checkpoint"]).mkdir(parents=True, exist_ok=True)

    # Trigger collection + index creation in MongoDB. Imported lazily so
    # this module can be imported in environments without pymongo (e.g.
    # documentation builds).
    try:
        from analytics.mongo_io import setup_indexes
        setup_indexes(verbose=True)
    except Exception as exc:  # pragma: no cover — best-effort
        print(f"[storage_paths] index setup skipped: {exc}")

    return paths
