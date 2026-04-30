"""Centralised MongoDB I/O for the YouTube analytics pipeline.

This module is the single point of contact between Spark and MongoDB.
In the current hybrid architecture Bronze still uses MongoDB, while
Silver/Gold are local Delta tables.

NoSQL advantages we deliberately exploit:
  * Schema flexibility — Silver records keep `tags_array` as a native BSON
    array (no flattening / explode-at-write step). Channel metadata stays
    nested.
  * TTL on Bronze — raw documents auto-expire after a configurable
    retention window. Delta cannot do that natively.
"""
from __future__ import annotations

import os
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession

# ---------------------------------------------------------------------------
# Connection settings (env-var overrides keep credentials out of source)
# ---------------------------------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "youtube_analytics")
BRONZE_MONGO_URI = os.environ.get("BRONZE_MONGO_URI", "mongodb://localhost:27017")
BRONZE_MONGO_DB = os.environ.get("BRONZE_MONGO_DB", MONGO_DB)

# Spark MongoDB connector v10+ — declared via spark.jars.packages in the
# session builders. This constant is exposed so callers can include it in
# their SparkSession config without copy-pasting the version string.
MONGO_SPARK_PACKAGE = "org.mongodb.spark:mongo-spark-connector_2.12:10.2.1"

# How long Bronze documents live before MongoDB auto-deletes them.
BRONZE_TTL_DAYS = int(os.environ.get("BRONZE_TTL_DAYS", "30"))


# ---------------------------------------------------------------------------
# Spark read / write helpers
# ---------------------------------------------------------------------------
def mongo_read(spark: SparkSession, collection: str) -> DataFrame:
    """Load an entire MongoDB collection as a Spark DataFrame.

    The Spark MongoDB connector pushes simple filters / projections down
    when subsequent operations support it, so callers should still chain
    `.filter(...)` and `.select(...)` rather than reading everything into
    memory.
    """
    return (
        spark.read.format("mongodb")
        .option("connection.uri", MONGO_URI)
        .option("database", MONGO_DB)
        .option("collection", collection)
        .load()
    )


def mongo_write(
    df: DataFrame,
    collection: str,
    mode: str = "append",
    *,
    connection_uri: str | None = None,
    database: str | None = None,
) -> None:
    """Write a Spark DataFrame to a MongoDB collection.

    `mode='overwrite'` uses the connector's atomic rename behaviour, so
    dashboard readers never observe a partially-rebuilt collection.
    """
    (
        df.write.format("mongodb")
        .mode(mode)
        .option("connection.uri", connection_uri or MONGO_URI)
        .option("database", database or MONGO_DB)
        .option("collection", collection)
        .save()
    )


def mongo_collection_has_data(collection: str) -> bool:
    """Cheap existence/emptiness probe used by the dashboard's optional
    Gold loaders.  Uses pymongo so we don't pay Spark startup cost just to
    answer 'is there anything to read here?'.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        return False
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        return client[MONGO_DB][collection].estimated_document_count() > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Index / TTL setup (run once, idempotent)
# ---------------------------------------------------------------------------
def _create_index_safe(coll, keys: Iterable, **kwargs) -> None:
    """Create an index, ignoring 'already exists' errors so this is safe
    to call on every pipeline start."""
    try:
        coll.create_index(list(keys), **kwargs)
    except Exception as exc:  # pragma: no cover — best-effort
        msg = str(exc).lower()
        if "already exists" in msg or "indexoptionsconflict" in msg:
            return
        print(f"[mongo_io] index create skipped on {coll.name}: {exc}")


def setup_indexes(verbose: bool = True) -> None:
    """Create the Bronze indexes (and TTL). Safe to call repeatedly."""
    from pymongo import ASCENDING, DESCENDING, MongoClient

    client = MongoClient(BRONZE_MONGO_URI)
    db = client[BRONZE_MONGO_DB]

    # ---- Bronze ----------------------------------------------------------
    bronze = db["bronze_raw"]
    _create_index_safe(bronze, [("collection_batch_id", ASCENDING)])
    _create_index_safe(bronze, [("video_id", ASCENDING), ("trending_region", ASCENDING)])
    # TTL: auto-delete Bronze documents older than BRONZE_TTL_DAYS days. The
    # field must be a BSON Date — `ingestion_timestamp` is set by the
    # streaming job via current_timestamp().
    _create_index_safe(
        bronze,
        [("ingestion_timestamp", ASCENDING)],
        expireAfterSeconds=BRONZE_TTL_DAYS * 24 * 3600,
        name="bronze_ttl",
    )

    if verbose:
        print(f"[mongo_io] bronze indexes ready on {BRONZE_MONGO_URI}/{BRONZE_MONGO_DB}")


def attach_mongo_jars(builder):
    """Append the MongoDB connector to a SparkSession.builder. Use this in
    every Spark builder in the project so we don't drift on the version
    string."""
    return builder.config("spark.jars.packages", MONGO_SPARK_PACKAGE)
