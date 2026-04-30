"""Resolve where each medallion layer lives.

Current architecture is hybrid:

- Bronze lives in MongoDB so the project still demonstrates a NoSQL /
  distributed data-store layer.
- Silver and Gold live as local Delta tables so Spark + Streamlit reads stay
  fast for the dashboard.
- Streaming checkpoints stay on disk (Spark requirement).
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT / "storage"
CHECKPOINT_ROOT = STORAGE_ROOT / "checkpoints_medallion"
DELTA_ROOT = STORAGE_ROOT / "delta_tables"
SILVER_ROOT = DELTA_ROOT / "silver"
GOLD_ROOT = DELTA_ROOT / "gold"


# Bronze collection name — kept stable so docs / demos can refer to it.
BRONZE_COLLECTION = "bronze_raw"
SILVER_PATH = str(SILVER_ROOT / "youtube_enriched")
GOLD_COLLECTIONS = {
    "latest_snapshot": str(GOLD_ROOT / "latest_snapshot"),
    "category_summary": str(GOLD_ROOT / "category_summary"),
    "views_timeseries": str(GOLD_ROOT / "views_timeseries"),
    "region_timeseries": str(GOLD_ROOT / "region_timeseries"),
    "channel_leaderboard": str(GOLD_ROOT / "channel_leaderboard"),
    "duration_distribution": str(GOLD_ROOT / "duration_distribution"),
    "subscriber_tier_distribution": str(GOLD_ROOT / "subscriber_tier_distribution"),
    "tag_usage_frequency": str(GOLD_ROOT / "tag_usage_frequency"),
    "trending_rank_distribution": str(GOLD_ROOT / "trending_rank_distribution"),
}


def get_medallion_paths() -> dict:
    """Return the storage target for each medallion layer."""
    return {
        "bronze": BRONZE_COLLECTION,
        "silver": SILVER_PATH,
        "gold": dict(GOLD_COLLECTIONS),
        "checkpoint": str(CHECKPOINT_ROOT),
    }


def ensure_medallion_paths() -> dict:
    """Create local Delta roots and ensure Bronze Mongo indexes exist."""
    paths = get_medallion_paths()
    Path(paths["checkpoint"]).mkdir(parents=True, exist_ok=True)
    Path(paths["silver"]).parent.mkdir(parents=True, exist_ok=True)
    for gold_path in paths["gold"].values():
        Path(gold_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        from analytics.mongo_io import setup_indexes
        setup_indexes(verbose=True)
    except Exception as exc:  # pragma: no cover — best-effort
        print(f"[storage_paths] index setup skipped: {exc}")

    return paths
