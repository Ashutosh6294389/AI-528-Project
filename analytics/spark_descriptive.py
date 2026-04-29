"""Spark-based descriptive aggregations.

These mirror the pandas-based builders in business_analysis.py that the
descriptive tab actually renders, but every aggregation runs as a Spark
query against the Silver Delta layer. Only the small aggregated result
is converted to pandas (at the dashboard render boundary) so that
Streamlit/Altair can serialize it.
"""
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
    days = window_days.get(history_window)
    if days is not None and "collected_at_ts" in sdf.columns:
        sdf = sdf.filter(
            F.col("collected_at_ts") >= F.expr(f"current_timestamp() - INTERVAL {days} DAYS")
        )

    return sdf


def build_spark_category_summary(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Per-category totals + averages, ordered by total_views desc."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window, category_name, trending_region
    )
    return (
        sdf.groupBy("category_name")
        .agg(
            F.countDistinct("video_id").alias("videos"),
            F.sum("view_count").alias("total_views"),
            F.sum("like_count").alias("total_likes"),
            F.sum("comment_count").alias("total_comments"),
            F.avg("engagement_rate").alias("avg_engagement_rate"),
        )
        .orderBy(F.desc("total_views"))
    )


def build_spark_views_timeseries(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Per (time_bucket, category_name): total_views and total_engagements."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window, category_name, trending_region
    )
    return (
        sdf.groupBy("time_bucket", "category_name")
        .agg(
            F.sum("view_count").alias("total_views"),
            F.sum("engagements").alias("total_engagements"),
        )
        .orderBy("time_bucket", "category_name")
    )


def build_spark_duration_distribution(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Per (trending_region, category_name, duration_bucket): video count
    plus avg views and avg engagement rate inside each bucket."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window, category_name, trending_region
    )
    return (
        sdf.filter(F.col("duration_bucket").isNotNull())
        .groupBy("trending_region", "category_name", "duration_bucket")
        .agg(
            F.countDistinct("video_id").alias("video_count"),
            F.avg("view_count").alias("avg_views_in_bucket"),
            F.avg("engagement_rate").alias("avg_er_in_bucket"),
        )
    )


def build_spark_subscriber_tier_distribution(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Per (trending_region, subscriber_tier): video count and share of
    that region's trending videos."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window, category_name, trending_region
    )

    sdf = sdf.withColumn(
        "subscriber_tier",
        F.when(F.col("channel_subscriber_count") < 100_000, F.lit("Small (<100K)"))
        .when(F.col("channel_subscriber_count") < 1_000_000, F.lit("Mid (100K-1M)"))
        .when(F.col("channel_subscriber_count") < 10_000_000, F.lit("Large (1M-10M)"))
        .otherwise(F.lit("Mega (10M+)")),
    )

    per_video = sdf.dropDuplicates(["video_id", "trending_region", "subscriber_tier"])
    region_tier = (
        per_video.groupBy("trending_region", "subscriber_tier")
        .agg(F.countDistinct("video_id").alias("video_count"))
    )
    region_totals = (
        per_video.groupBy("trending_region")
        .agg(F.countDistinct("video_id").alias("region_total"))
    )

    return (
        region_tier.join(region_totals, on="trending_region", how="inner")
        .withColumn("pct", F.col("video_count") / F.col("region_total") * F.lit(100.0))
        .orderBy("trending_region", "subscriber_tier")
    )


def build_spark_tag_usage_frequency(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
    top_n: int = 30,
) -> DataFrame:
    """Top-N tags by # of distinct trending videos using them, with their
    primary category and region."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window, category_name, trending_region
    )

    exploded = (
        sdf.withColumn("tag", F.explode("tags_array"))
        .filter(F.col("tag").isNotNull() & (F.trim(F.col("tag")) != ""))
    )

    top_tags = (
        exploded.groupBy("tag")
        .agg(F.countDistinct("video_id").alias("videos_using_tag"))
        .orderBy(F.desc("videos_using_tag"))
        .limit(top_n)
    )

    # Attach the dominant category and region for each top tag (whichever
    # combination shows up most often). This matches the columns the chart
    # uses for color/tooltip.
    tag_combo_counts = (
        exploded.groupBy("tag", "category_name", "trending_region")
        .agg(F.count("*").alias("combo_count"))
    )
    from pyspark.sql.window import Window
    window = Window.partitionBy("tag").orderBy(F.desc("combo_count"))
    dominant = (
        tag_combo_counts.withColumn("rn", F.row_number().over(window))
        .filter(F.col("rn") == 1)
        .select("tag", "category_name", "trending_region")
    )

    return top_tags.join(dominant, on="tag", how="left").orderBy(
        F.desc("videos_using_tag")
    )


def build_spark_trending_rank_distribution(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Latest-snapshot category share of currently trending slots.
    Picks each video's most recent appearance in the window, then
    counts videos per (category, region)."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window, category_name, trending_region
    )
    sdf = sdf.filter(F.col("trending_rank").isNotNull())

    # Latest row per (video_id, trending_region) by collected_at_ts.
    from pyspark.sql.window import Window
    window = (
        Window.partitionBy("video_id", "trending_region")
        .orderBy(F.desc("collected_at_ts"))
    )
    latest = (
        sdf.withColumn("rn", F.row_number().over(window))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    region_counts = (
        latest.groupBy("trending_region", "category_name")
        .agg(F.countDistinct("video_id").alias("video_count"))
    )
    region_totals = (
        latest.groupBy("trending_region")
        .agg(F.countDistinct("video_id").alias("region_total"))
    )

    return (
        region_counts.join(region_totals, on="trending_region", how="inner")
        .withColumn(
            "pct_of_trending",
            F.col("video_count") / F.col("region_total") * F.lit(100.0),
        )
        .orderBy(F.desc("video_count"))
    )
