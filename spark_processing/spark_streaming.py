from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import *

# -------------------------------
# Spark Session (FIXED)
# -------------------------------
spark = SparkSession.builder \
    .appName("YouTubeAnalytics") \
    .master("local[*]") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.jars", "jars/delta-core_2.12-2.4.0.jar,jars/spark-sql-kafka-0-10_2.12-3.4.1.jar") \
    .config("spark.jars", "jars/delta-core_2.12-2.4.0.jar,jars/spark-sql-kafka-0-10_2.12-3.4.1.jar,jars/kafka-clients-3.4.1.jar")\
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.hadoop.security.authentication", "simple") \
    .config("spark.hadoop.security.authorization", "false") \
    .config("spark.driver.extraJavaOptions", "--add-opens java.base/javax.security.auth=ALL-UNNAMED") \
    .config("spark.executor.extraJavaOptions", "--add-opens java.base/javax.security.auth=ALL-UNNAMED") \
    .config(
        "spark.jars",
        "jars/delta-core_2.12-2.4.0.jar,"
        "jars/delta-storage-2.4.0.jar,"
        "jars/spark-sql-kafka-0-10_2.12-3.4.1.jar,"
        "jars/spark-token-provider-kafka-0-10_2.12-3.4.1.jar,"
        "jars/kafka-clients-3.4.1.jar,"
        "jars/commons-pool2-2.11.1.jar"
    )\
    .getOrCreate()
 
# -------------------------------
# Schema
# -------------------------------
schema = StructType([
    StructField("title", StringType()),
    StructField("category", StringType()),
    StructField("views", StringType()),
    StructField("likes", StringType()),
    StructField("comments", StringType())
])

# -------------------------------
# Read Kafka Stream
# -------------------------------
raw_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "youtube-data") \
    .load()

parsed_df = raw_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# -------------------------------
# Clean Data
# -------------------------------
clean_df = parsed_df \
    .withColumn("views", col("views").cast("int")) \
    .withColumn("likes", col("likes").cast("int")) \
    .withColumn("comments", col("comments").cast("int"))

# -------------------------------
# Write to Delta Lake
# -------------------------------
query = clean_df.writeStream \
    .format("delta") \
    .option("path", "storage/delta_tables/youtube") \
    .option("checkpointLocation", "storage/checkpoints") \
    .outputMode("append") \
    .start()

query.awaitTermination()