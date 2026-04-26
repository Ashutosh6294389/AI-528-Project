from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit

from storage_paths import ensure_medallion_paths
from transformations import build_silver_df, refresh_gold_tables


SOURCE_PATH = "storage/delta_tables/youtube_enriched"
MEDALLION_PATHS = ensure_medallion_paths()
BRONZE_PATH = MEDALLION_PATHS["bronze"]
SILVER_PATH = MEDALLION_PATHS["silver"]
GOLD_PATHS = MEDALLION_PATHS["gold"]


spark = (
    SparkSession.builder.appName("YouTubeMedallionBackfill")
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


def main():
    source_df = spark.read.format("delta").load(SOURCE_PATH)
    bronze_df = (
        source_df.withColumn("ingestion_timestamp", current_timestamp())
        .withColumn("source_system", lit("delta.backfill"))
        .withColumn("file_name", lit("storage/delta_tables/youtube_enriched"))
    )

    bronze_df.write.format("delta").mode("append").option("mergeSchema", "true").save(BRONZE_PATH)

    silver_df = build_silver_df(bronze_df)
    silver_df.write.format("delta").mode("append").option("mergeSchema", "true").save(SILVER_PATH)

    refresh_gold_tables(spark, SILVER_PATH, GOLD_PATHS)
    print("Backfill complete: Bronze appended, Silver appended, Gold refreshed.")


if __name__ == "__main__":
    main()
