import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json, lit
from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Make sibling packages importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analytics.mongo_io import BRONZE_MONGO_DB, BRONZE_MONGO_URI, MONGO_SPARK_PACKAGE, mongo_write
from runtime_config import (
    KAFKA_SPARK_PACKAGES,
    KAFKA_TOPIC,
    build_kafka_spark_options,
    local_delta_jars_csv,
    local_kafka_jars_csv,
    resolve_spark_master,
    use_local_kafka_jars,
)
from storage_paths import ensure_medallion_paths
from transformations import build_silver_df, refresh_gold_tables


TOPIC = KAFKA_TOPIC
MEDALLION_PATHS = ensure_medallion_paths()
BRONZE_COLLECTION = MEDALLION_PATHS["bronze"]
SILVER_COLLECTION = MEDALLION_PATHS["silver"]
GOLD_COLLECTIONS = MEDALLION_PATHS["gold"]
CHECKPOINT_PATH = MEDALLION_PATHS["checkpoint"]

builder = SparkSession.builder.appName("YouTubeMedallionStreaming")
spark_master = resolve_spark_master("local[*]")
if spark_master:
    builder = builder.master(spark_master)

package_list = [MONGO_SPARK_PACKAGE]
if not use_local_kafka_jars():
    package_list.append(KAFKA_SPARK_PACKAGES)

builder = builder.config("spark.jars.packages", ",".join(package_list))

local_jars = [local_delta_jars_csv(PROJECT_ROOT)]
if use_local_kafka_jars():
    local_jars.append(local_kafka_jars_csv(PROJECT_ROOT))
local_jars = [value for value in local_jars if value]
if local_jars:
    builder = builder.config("spark.jars", ",".join(local_jars))

spark = (
    builder
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    # Heap sizing — Gold refresh scans the entire Silver collection, so
    # the driver/executor needs enough RAM to hold a partition of BSON
    # documents while the connector decodes them.
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.maxResultSize", "2g")
    # Smaller Mongo read partitions = smaller per-task working set when
    # the connector deserializes a cursor batch.
    .config("spark.mongodb.read.partitionerOptions.partitionSizeMB", "32")
    .getOrCreate()
)

# How often Gold gets rebuilt — on every Nth micro-batch instead of every
# batch. Default 5 ≈ 3-4 minutes between Gold refreshes at the streaming
# cadence, which keeps the dashboard fresh without a full Silver scan
# every 45 seconds.
GOLD_REFRESH_EVERY_N_BATCHES = int(os.environ.get("GOLD_REFRESH_EVERY", "5"))

spark.sparkContext.setLogLevel("WARN")


schema = StructType([
    StructField("collection_batch_id", StringType(), True),
    StructField("collected_at", StringType(), True),
    StructField("surface", StringType(), True),
    StructField("trending_region", StringType(), True),
    StructField("trending_category_id", StringType(), True),
    StructField("trending_page", IntegerType(), True),
    StructField("trending_rank", IntegerType(), True),
    StructField("video_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("description", StringType(), True),
    StructField("published_at", StringType(), True),
    StructField("category_id", StringType(), True),
    StructField("category_name", StringType(), True),
    StructField("tags", StringType(), True),
    StructField("default_language", StringType(), True),
    StructField("thumbnail_url", StringType(), True),
    StructField("view_count", IntegerType(), True),
    StructField("like_count", IntegerType(), True),
    StructField("comment_count", IntegerType(), True),
    StructField("favorite_count", IntegerType(), True),
    StructField("duration_iso", StringType(), True),
    StructField("definition", StringType(), True),
    StructField("caption", BooleanType(), True),
    StructField("licensed_content", BooleanType(), True),
    StructField("content_rating", StringType(), True),
    StructField("projection", StringType(), True),
    StructField("channel_id", StringType(), True),
    StructField("channel_title", StringType(), True),
    StructField("channel_subscriber_count", IntegerType(), True),
    StructField("channel_view_count", IntegerType(), True),
    StructField("channel_video_count", IntegerType(), True),
    StructField("channel_country", StringType(), True),
])


def process_batch(batch_df, batch_id: int) -> None:
    if batch_df.isEmpty():
        return

    parsed_batch = (
        batch_df.withColumn("ingestion_timestamp", current_timestamp())
        .withColumn("source_system", lit("kafka.youtube-data"))
        .withColumn("file_name", lit(TOPIC))
        .persist()
    )

    # Bronze: raw documents land in MongoDB exactly as they came off Kafka,
    # with our ingestion metadata. The TTL index on `ingestion_timestamp`
    # auto-expires old records (configurable via BRONZE_TTL_DAYS).
    mongo_write(
        parsed_batch,
        BRONZE_COLLECTION,
        mode="append",
        connection_uri=BRONZE_MONGO_URI,
        database=BRONZE_MONGO_DB,
    )

    # Silver: enriched + typed documents — `tags_array` is stored as a
    # local Delta table for fast dashboard queries.
    silver_batch = build_silver_df(parsed_batch)
    silver_batch.write.format("delta").mode("append").save(SILVER_COLLECTION)

    # Gold: nine aggregated collections rebuilt atomically from Silver.
    # Skipped on most batches — Gold scanning the full Silver every 45 s
    # is wasteful as Silver grows. Refresh every Nth batch instead.
    if (batch_id % GOLD_REFRESH_EVERY_N_BATCHES) == 0:
        refresh_gold_tables(spark, SILVER_COLLECTION, GOLD_COLLECTIONS)
        gold_msg = "gold refreshed"
    else:
        gold_msg = (
            f"gold skipped ({batch_id % GOLD_REFRESH_EVERY_N_BATCHES}"
            f"/{GOLD_REFRESH_EVERY_N_BATCHES} until next refresh)"
        )

    parsed_batch.unpersist()
    print(f"Processed batch {batch_id}: bronze appended, silver appended, {gold_msg}.")


kafka_reader = spark.readStream.format("kafka")
for option_name, option_value in build_kafka_spark_options(TOPIC).items():
    kafka_reader = kafka_reader.option(option_name, option_value)
kafka_df = kafka_reader.load()

parsed_stream = (
    kafka_df.selectExpr("CAST(value AS STRING) as json_value")
    .select(from_json(col("json_value"), schema).alias("data"))
    .select("data.*")
)

query = (
    parsed_stream.writeStream.foreachBatch(process_batch)
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .start()
)

print(
    "Spark streaming started. Writing Bronze to MongoDB at "
    f"{BRONZE_MONGO_URI} / {BRONZE_MONGO_DB}; "
    f"Silver/Gold to local Delta under {PROJECT_ROOT / 'storage' / 'delta_tables'}"
)
query.awaitTermination()
