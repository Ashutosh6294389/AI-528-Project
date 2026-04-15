from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)


KAFKA_BROKER = "localhost:9092"
TOPIC = "youtube-data"
DELTA_PATH = "storage/delta_tables/youtube"
CHECKPOINT_PATH = "storage/checkpoints"

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
    StructField("timestamp", StringType(), True),
    StructField("surface", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("rank", IntegerType(), True),
    StructField("region", StringType(), True),
    StructField("category", StringType(), True),
    StructField("category_id", StringType(), True),
    StructField("video_id", StringType(), True),
    StructField("channel_id", StringType(), True),
    StructField("channel_title", StringType(), True),
    StructField("title", StringType(), True),
    StructField("views", IntegerType(), True),
    StructField("likes", IntegerType(), True),
    StructField("comments", IntegerType(), True),
    StructField("publish_time", StringType(), True),
    StructField("engagements", IntegerType(), True),
    StructField("engagement_rate", DoubleType(), True),
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
