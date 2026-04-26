from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, from_json, lit
from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from storage_paths import ensure_medallion_paths
from transformations import build_silver_df, refresh_gold_tables


KAFKA_BROKER = "localhost:9092"
TOPIC = "youtube-data"
MEDALLION_PATHS = ensure_medallion_paths()
BRONZE_PATH = MEDALLION_PATHS["bronze"]
SILVER_PATH = MEDALLION_PATHS["silver"]
GOLD_PATHS = MEDALLION_PATHS["gold"]
CHECKPOINT_PATH = MEDALLION_PATHS["checkpoint"]


spark = (
    SparkSession.builder.appName("YouTubeMedallionStreaming")
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
    .config(
        "spark.jars",
        "jars/delta-core_2.12-2.4.0.jar,"
        "jars/delta-storage-2.4.0.jar,"
        "jars/spark-sql-kafka-0-10_2.12-3.4.1.jar,"
        "jars/spark-token-provider-kafka-0-10_2.12-3.4.1.jar,"
        "jars/kafka-clients-3.4.1.jar,"
        "jars/commons-pool2-2.11.1.jar",
    )
    .getOrCreate()
)

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

    parsed_batch.write.format("delta").mode("append").save(BRONZE_PATH)

    silver_batch = build_silver_df(parsed_batch)
    silver_batch.write.format("delta").mode("append").option("mergeSchema", "true").save(SILVER_PATH)

    refresh_gold_tables(spark, SILVER_PATH, GOLD_PATHS)

    parsed_batch.unpersist()
    print(f"Processed batch {batch_id}: bronze appended, silver refreshed, gold refreshed.")


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

print("Spark streaming started. Writing to persistent Bronze/Silver/Gold Delta tables...")
query.awaitTermination()
