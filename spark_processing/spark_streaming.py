from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    BooleanType,
)

KAFKA_BROKER = "localhost:9092"
TOPIC = "youtube-data"
DELTA_PATH = "storage/delta_tables/youtube_enriched"
CHECKPOINT_PATH = "storage/checkpoints_enriched"

spark = (
    SparkSession.builder.appName("YouTubeSparkStreaming")
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
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

kafka_df = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKER)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

parsed_df = (
    kafka_df.selectExpr("CAST(value AS STRING) as json_value")
    .select(from_json(col("json_value"), schema).alias("data"))
    .select("data.*")
)

query = (
    parsed_df.writeStream.format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .start(DELTA_PATH)
)

print("Spark streaming started. Writing enriched YouTube data to Delta Lake...")
query.awaitTermination()
