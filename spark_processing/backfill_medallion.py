from pyspark.sql import SparkSession

from storage_paths import create_new_active_run
from transformations import build_silver_df, refresh_gold_tables


SOURCE_PATH = "storage/delta_tables/youtube_enriched"
RUN_PATHS = create_new_active_run(delete_older_runs=True)
BRONZE_PATH = RUN_PATHS["bronze"]
SILVER_PATH = RUN_PATHS["silver"]
GOLD_PATHS = RUN_PATHS["gold"]


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

    source_df.write.format("delta").mode("overwrite").save(BRONZE_PATH)

    silver_df = build_silver_df(source_df)
    silver_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(SILVER_PATH)

    refresh_gold_tables(spark, SILVER_PATH, GOLD_PATHS)
    print(f"Backfill complete for run {RUN_PATHS['run_id']}: Bronze, Silver, and Gold layers are ready.")


if __name__ == "__main__":
    main()
