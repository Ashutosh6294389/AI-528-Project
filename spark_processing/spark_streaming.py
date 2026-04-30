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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.mongo_io import MONGO_SPARK_PACKAGE, mongo_write
from storage_paths import ensure_medallion_paths
from transformations import build_silver_df, refresh_gold_tables


KAFKA_BROKER = "localhost:9092"
TOPIC = "youtube-data"
MEDALLION_PATHS = ensure_medallion_paths()
BRONZE_COLLECTION = MEDALLION_PATHS["bronze"]
SILVER_COLLECTION = MEDALLION_PATHS["silver"]
GOLD_COLLECTIONS = MEDALLION_PATHS["gold"]
CHECKPOINT_PATH = MEDALLION_PATHS["checkpoint"]


spark = (
    SparkSession.builder.appName("YouTubeMedallionStreaming")
    .master("local[*]")
    # The MongoDB connector is fetched via Maven on first run; the Kafka
    # JARs are still loaded from the local jars/ folder.
    .config("spark.jars.packages", MONGO_SPARK_PACKAGE)
    .config(
        "spark.jars",
        "jars/spark-sql-kafka-0-10_2.12-3.4.1.jar,"
        "jars/spark-token-provider-kafka-0-10_2.12-3.4.1.jar,"
        "jars/kafka-clients-3.4.1.jar,"
        "jars/commons-pool2-2.11.1.jar",
    )
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
    mongo_write(parsed_batch, BRONZE_COLLECTION, mode="append")

    # Silver: enriched + typed documents — `tags_array` is stored as a
    # native BSON array, no flattening at write time.
    silver_batch = build_silver_df(parsed_batch)
    mongo_write(silver_batch, SILVER_COLLECTION, mode="append")

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


kafka_df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKER)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

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
    "Spark streaming started. Writing Bronze/Silver/Gold to MongoDB at "
    f"{__import__('analytics.mongo_io', fromlist=['MONGO_URI']).MONGO_URI} / "
    f"{__import__('analytics.mongo_io', fromlist=['MONGO_DB']).MONGO_DB}"
)
query.awaitTermination()
