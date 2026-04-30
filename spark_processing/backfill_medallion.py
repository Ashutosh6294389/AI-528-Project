"""One-shot backfill: read an existing local Delta dump (if you still have
one from the previous architecture) and rebuild Bronze / Silver / Gold in
MongoDB. Run this once if you want to migrate historical data into Mongo;
otherwise just start the streaming job and Mongo will fill from Kafka.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.mongo_io import MONGO_SPARK_PACKAGE, mongo_write
from runtime_config import local_delta_jars_csv, resolve_spark_master
from storage_paths import ensure_medallion_paths
from transformations import build_silver_df, refresh_gold_tables


# Optional source for the one-shot migration. If this folder doesn't
# exist, we just refresh Gold from whatever Silver already has.
SOURCE_DELTA_PATH = os.environ.get(
    "BACKFILL_SOURCE_DELTA",
    "storage/delta_tables/silver/youtube_enriched",
)
MEDALLION_PATHS = ensure_medallion_paths()
BRONZE_COLLECTION = MEDALLION_PATHS["bronze"]
SILVER_COLLECTION = MEDALLION_PATHS["silver"]
GOLD_COLLECTIONS = MEDALLION_PATHS["gold"]


def _build_spark() -> SparkSession:
    builder = SparkSession.builder.appName("YouTubeMedallionBackfill")
    spark_master = resolve_spark_master("local[4]")
    if spark_master:
        builder = builder.master(spark_master)

    builder = (
        builder
        .config("spark.jars.packages", MONGO_SPARK_PACKAGE)
        .config("spark.jars", local_delta_jars_csv(Path(__file__).resolve().parents[1]))
        # Heap headroom — backfill scans the whole Silver in one go, much
        # bigger than any streaming micro-batch.
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.maxResultSize", "4g")
        # Smaller Mongo write/read partitions so each Spark task only has
        # to deserialize a manageable BSON cursor.
        .config("spark.mongodb.read.partitionerOptions.partitionSizeMB", "16")
        .config("spark.mongodb.write.maxBatchSize", "256")
        # Fewer shuffle partitions — the data isn't that big, and lots of
        # tiny partitions waste task scheduling time.
        .config("spark.sql.shuffle.partitions", "16")
    )
    if Path(SOURCE_DELTA_PATH).exists():
        builder = (
            builder
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
    else:
        builder = (
            builder
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
    return builder.getOrCreate()


SKIP_GOLD = os.environ.get("SKIP_GOLD", "").lower() in ("1", "true", "yes")


def main() -> None:
    spark = _build_spark()

    if Path(SOURCE_DELTA_PATH).exists():
        print(f"[backfill] migrating from local Delta at {SOURCE_DELTA_PATH}")
        source_df = spark.read.format("delta").load(SOURCE_DELTA_PATH)
        # Repartition before writing so each Mongo write task is small.
        # 16 partitions ≈ <50 MB each on the typical Silver schema.
        source_df = source_df.repartition(16)
        bronze_df = (
            source_df.withColumn("ingestion_timestamp", current_timestamp())
            .withColumn("source_system", lit("delta.backfill"))
            .withColumn("file_name", lit(SOURCE_DELTA_PATH))
        )
        print("[backfill] writing Bronze ...")
        mongo_write(bronze_df, BRONZE_COLLECTION, mode="append")

        print("[backfill] writing Silver ...")
        silver_df = build_silver_df(bronze_df)
        silver_df.write.format("delta").mode("append").save(SILVER_COLLECTION)
        print("[backfill] Bronze + Silver written.")
    else:
        print(
            f"[backfill] no Delta source at {SOURCE_DELTA_PATH} — "
            "refreshing Gold from whatever Silver currently holds."
        )

    if SKIP_GOLD:
        print(
            "[backfill] SKIP_GOLD=1 — skipping Gold refresh. "
            "The streaming pipeline (or a manual run with SKIP_GOLD unset) "
            "will rebuild Gold next time it executes refresh_gold_tables."
        )
    else:
        print("[backfill] refreshing Gold ...")
        refresh_gold_tables(spark, SILVER_COLLECTION, GOLD_COLLECTIONS)

    print("Backfill complete.")


if __name__ == "__main__":
    main()
