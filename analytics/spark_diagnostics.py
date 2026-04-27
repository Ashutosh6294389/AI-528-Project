from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def _load_filtered_silver(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    sdf = spark.read.format("delta").load(silver_path)

    if category_name:
        sdf = sdf.filter(F.col("category_name") == category_name)

    if trending_region:
        sdf = sdf.filter(F.col("trending_region") == trending_region)

    window_days = {
        "Last 24 Hours": 1,
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "All Available": None,
    }
    days = window_days[history_window]
    if days is not None and "collected_at_ts" in sdf.columns:
        sdf = sdf.filter(
            F.col("collected_at_ts") >= F.expr(f"current_timestamp() - INTERVAL {days} DAYS")
        )

    return sdf


def build_velocity_shift_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
) -> DataFrame:
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    return (
        sdf.groupBy("time_bucket")
        .agg(
            F.avg("velocity").alias("avg_velocity"),
            F.avg("engagement_rate").alias("avg_engagement_rate"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
        .orderBy("time_bucket")
    )


def build_persistence_distribution_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
) -> DataFrame:
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    per_video = (
        sdf.groupBy("video_id", "trending_region")
        .agg(
            F.countDistinct("collection_batch_id").alias("batches_appeared"),
            F.max("view_count").alias("peak_views"),
        )
    )

    return (
        per_video.withColumn(
            "persistence_bucket",
            F.when(F.col("batches_appeared") == 1, F.lit("1 batch"))
            .when(F.col("batches_appeared") <= 3, F.lit("2-3 batches"))
            .when(F.col("batches_appeared") <= 6, F.lit("4-6 batches"))
            .otherwise(F.lit("7+ batches")),
        )
        .groupBy("persistence_bucket")
        .agg(
            F.count("*").alias("video_count"),
            F.avg("peak_views").alias("avg_peak_views"),
        )
    )


def build_duration_engagement_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    return (
        sdf.groupBy("duration_bucket")
        .agg(
            F.avg("engagement_rate").alias("avg_engagement_rate"),
            F.avg("view_count").alias("avg_views"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
    )


def build_tag_effectiveness_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
    top_n: int = 30,
) -> DataFrame:
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    return (
        sdf.withColumn("tag", F.explode("tags_array"))
        .filter(F.col("tag").isNotNull() & (F.trim(F.col("tag")) != ""))
        .groupBy("tag")
        .agg(
            F.countDistinct("video_id").alias("videos_using_tag"),
            F.avg("engagement_rate").alias("avg_engagement_rate"),
            F.avg("view_count").alias("avg_views"),
        )
        .orderBy(F.desc("videos_using_tag"))
        .limit(top_n)
    )
