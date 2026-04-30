from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    array_remove,
    avg,
    col,
    count,
    countDistinct,
    current_timestamp,
    desc,
    explode,
    expr,
    length,
    lit,
    regexp_extract,
    regexp_replace,
    row_number,
    size,
    split,
    sum as spark_sum,
    to_timestamp,
    trim,
    upper,
    when,
)
from pyspark.sql.window import Window


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_silver_df(df: DataFrame) -> DataFrame:
    silver_df = (
        df.withColumn("collected_at_ts", to_timestamp("collected_at"))
        .withColumn("published_at_ts", to_timestamp("published_at"))
        .withColumn(
            "ingestion_timestamp",
            when(col("ingestion_timestamp").isNotNull(), col("ingestion_timestamp").cast("timestamp"))
            .otherwise(current_timestamp()),
        )
        .withColumn("source_system", when(col("source_system").isNull(), lit("unknown")).otherwise(col("source_system")))
        .withColumn("file_name", when(col("file_name").isNull(), lit("stream")).otherwise(col("file_name")))
        .withColumn("video_id", trim(col("video_id")))
        .withColumn("title", trim(col("title")))
        .withColumn("channel_id", trim(col("channel_id")))
        .withColumn("channel_title", trim(col("channel_title")))
        .withColumn("trending_region", upper_trimmed("trending_region"))
        .withColumn("category_name", trim(col("category_name")))
        .withColumn("view_count", col("view_count").cast("int"))
        .withColumn("like_count", col("like_count").cast("int"))
        .withColumn("comment_count", col("comment_count").cast("int"))
        .withColumn("favorite_count", col("favorite_count").cast("int"))
        .withColumn("trending_page", col("trending_page").cast("int"))
        .withColumn("trending_rank", col("trending_rank").cast("int"))
        .withColumn("channel_subscriber_count", col("channel_subscriber_count").cast("int"))
        .withColumn("channel_view_count", col("channel_view_count").cast("int"))
        .withColumn("channel_video_count", col("channel_video_count").cast("int"))
        .withColumn("engagements", col("like_count") + col("comment_count"))
        .withColumn(
            "like_rate",
            when(col("view_count") > 0, col("like_count") / col("view_count")).otherwise(lit(0.0)),
        )
        .withColumn(
            "comment_rate",
            when(col("view_count") > 0, col("comment_count") / col("view_count")).otherwise(lit(0.0)),
        )
        .withColumn(
            "engagement_rate",
            when(col("view_count") > 0, (col("like_count") + col("comment_count")) / col("view_count")).otherwise(lit(0.0)),
        )
        .withColumn(
            "video_age_hours",
            when(
                col("published_at_ts").isNotNull() & col("collected_at_ts").isNotNull(),
                (col("collected_at_ts").cast("long") - col("published_at_ts").cast("long")) / lit(3600.0),
            ).otherwise(lit(0.0)),
        )
        .withColumn("video_age_hours", when(col("video_age_hours") <= 0, lit(0.01)).otherwise(col("video_age_hours")))
        .withColumn(
            "velocity",
            when(col("video_age_hours") > 0, col("view_count") / col("video_age_hours")).otherwise(lit(0.0)),
        )
        .withColumn("title_word_count", size(split(trim(col("title")), r"\s+")))
        .withColumn("title_has_question", col("title").contains("?"))
        .withColumn("title_has_number", col("title").rlike(r"\d"))
        .withColumn(
            "title_caps_ratio",
            when(
                length(regexp_replace(col("title"), "[^A-Za-z]", "")) > 0,
                length(regexp_replace(col("title"), "[^A-Z]", "")) / length(regexp_replace(col("title"), "[^A-Za-z]", "")),
            ).otherwise(lit(0.0)),
        )
        .withColumn("tags_array", expr("filter(from_json(tags, 'array<string>'), x -> x is not null)"))
        .withColumn("tag_count", when(col("tags_array").isNotNull(), size(col("tags_array"))).otherwise(lit(0)))
        .withColumn("duration_seconds", duration_to_seconds_expr("duration_iso"))
        .withColumn(
            "duration_bucket",
            when(col("duration_seconds") < 240, lit("Short"))
            .when(col("duration_seconds") < 900, lit("Medium"))
            .otherwise(lit("Long")),
        )
        .withColumn("publish_day", expr("date_format(published_at_ts, 'EEEE')"))
        .withColumn("publish_hour", expr("hour(published_at_ts)"))
        .withColumn("time_bucket", expr("to_timestamp(from_unixtime(floor(unix_timestamp(collected_at_ts) / 300) * 300))"))
        .where(col("video_id").isNotNull() & (col("video_id") != ""))
        .where(col("collected_at_ts").isNotNull())
    )

    dedupe_window = Window.partitionBy(
        "collection_batch_id",
        "trending_region",
        "video_id",
    ).orderBy(col("ingestion_timestamp").desc())

    return (
        silver_df.withColumn("rn", row_number().over(dedupe_window))
        .where(col("rn") == 1)
        .drop("rn")
    )


def build_gold_latest_snapshot(silver_df: DataFrame) -> DataFrame:
    window = Window.partitionBy("video_id", "trending_region").orderBy(col("collected_at_ts").desc(), col("collection_batch_id").desc())
    return (
        silver_df.withColumn("rn", row_number().over(window))
        .where(col("rn") == 1)
        .drop("rn")
    )


def build_gold_category_summary(latest_snapshot_df: DataFrame) -> DataFrame:
    """Aggregated at (category_name, trending_region) grain so the
    dashboard can filter by either / both without falling back to Silver.
    Stores sums + sample_size alongside averages so the dashboard can
    weight-average across regions when no region filter is selected."""
    return (
        latest_snapshot_df.groupBy("category_name", "trending_region")
        .agg(
            countDistinct("video_id").alias("videos"),
            spark_sum("view_count").alias("total_views"),
            spark_sum("like_count").alias("total_likes"),
            spark_sum("comment_count").alias("total_comments"),
            spark_sum("engagement_rate").alias("sum_engagement_rate"),
            spark_sum("like_rate").alias("sum_like_rate"),
            count(lit(1)).alias("sample_size"),
            avg("engagement_rate").alias("avg_engagement_rate"),
            avg("like_rate").alias("avg_like_rate"),
        )
    )


def build_gold_views_timeseries(silver_df: DataFrame) -> DataFrame:
    """Aggregated at (category_name, trending_region, time_bucket) — adds
    region so the dashboard can filter directly without re-scanning Silver."""
    return (
        silver_df.groupBy("category_name", "trending_region", "time_bucket")
        .agg(
            spark_sum("view_count").alias("total_views"),
            spark_sum("engagements").alias("total_engagements"),
        )
    )


def build_gold_region_timeseries(silver_df: DataFrame) -> DataFrame:
    return (
        silver_df.groupBy("trending_region", "time_bucket")
        .agg(
            spark_sum("view_count").alias("total_views"),
            avg("engagement_rate").alias("avg_engagement_rate"),
        )
    )


def build_gold_channel_leaderboard(latest_snapshot_df: DataFrame) -> DataFrame:
    return (
        latest_snapshot_df.groupBy("channel_title")
        .agg(
            countDistinct("video_id").alias("videos"),
            spark_sum("view_count").alias("total_views"),
            spark_sum("engagements").alias("total_engagements"),
            avg("engagement_rate").alias("avg_engagement_rate"),
            avg("channel_subscriber_count").alias("avg_subscribers"),
        )
    )


def build_gold_duration_distribution(silver_df: DataFrame) -> DataFrame:
    """Aggregated at (trending_region, category_name, duration_bucket).
    Dashboard reads this small table whole and lets Altair filter/colour
    by category. Stores sum + count so any cross-category rollup can
    recompute averages correctly."""
    return (
        silver_df.filter(col("duration_bucket").isNotNull())
        .groupBy("trending_region", "category_name", "duration_bucket")
        .agg(
            countDistinct("video_id").alias("video_count"),
            avg("view_count").alias("avg_views_in_bucket"),
            avg("engagement_rate").alias("avg_er_in_bucket"),
            spark_sum("view_count").alias("sum_views_in_bucket"),
            spark_sum("engagements").alias("sum_engagements_in_bucket"),
        )
    )


def build_gold_subscriber_tier_distribution(silver_df: DataFrame) -> DataFrame:
    """Aggregated at (trending_region, category_name, subscriber_tier).
    Dashboard recomputes the within-region share at render time so any
    filter combination still works."""
    tiered = silver_df.withColumn(
        "subscriber_tier",
        when(col("channel_subscriber_count") < 100_000, lit("Small (<100K)"))
        .when(col("channel_subscriber_count") < 1_000_000, lit("Mid (100K-1M)"))
        .when(col("channel_subscriber_count") < 10_000_000, lit("Large (1M-10M)"))
        .otherwise(lit("Mega (10M+)")),
    )
    per_video = tiered.dropDuplicates(
        ["video_id", "trending_region", "category_name", "subscriber_tier"]
    )
    return (
        per_video.groupBy("trending_region", "category_name", "subscriber_tier")
        .agg(countDistinct("video_id").alias("video_count"))
    )


def build_gold_tag_usage_frequency(silver_df: DataFrame) -> DataFrame:
    """Aggregated at (tag, trending_region, category_name). Dashboard
    filters and re-ranks at render time."""
    exploded = (
        silver_df.withColumn("tag", explode("tags_array"))
        .filter(col("tag").isNotNull() & (trim(col("tag")) != ""))
    )
    return (
        exploded.groupBy("tag", "trending_region", "category_name")
        .agg(countDistinct("video_id").alias("videos_using_tag"))
    )


def build_gold_trending_rank_distribution(silver_df: DataFrame) -> DataFrame:
    """Latest-snapshot category share at (trending_region, category_name).
    Distinct videos currently trending in each region/category, plus the
    region total so the dashboard can divide for `pct_of_trending`."""
    silver = silver_df.filter(col("trending_rank").isNotNull())
    window = Window.partitionBy("video_id", "trending_region").orderBy(desc("collected_at_ts"))
    latest = (
        silver.withColumn("__rn", row_number().over(window))
        .where(col("__rn") == 1)
        .drop("__rn")
    )

    region_counts = (
        latest.groupBy("trending_region", "category_name")
        .agg(countDistinct("video_id").alias("video_count"))
    )
    region_totals = (
        latest.groupBy("trending_region")
        .agg(countDistinct("video_id").alias("region_total"))
    )
    return (
        region_counts.join(region_totals, on="trending_region", how="inner")
        .withColumn(
            "pct_of_trending",
            col("video_count") / col("region_total") * lit(100.0),
        )
    )


def refresh_gold_tables(spark: SparkSession, silver_path: str, gold_collections: dict[str, str]) -> None:
    """Rebuild every Gold Delta table from the local Silver Delta table.

    Gold is overwritten in full each refresh so the dashboard always sees a
    consistent business-ready snapshot derived from the latest Silver history.
    """
    from pyspark import StorageLevel

    silver_df = spark.read.format("delta").load(silver_path).persist(StorageLevel.DISK_ONLY)
    try:
        latest_snapshot_df = build_gold_latest_snapshot(silver_df).persist(
            StorageLevel.DISK_ONLY
        )
        try:
            latest_snapshot_df.write.format("delta").mode("overwrite").save(gold_collections["latest_snapshot"])
            build_gold_category_summary(latest_snapshot_df).write.format("delta").mode("overwrite").save(gold_collections["category_summary"])
            build_gold_channel_leaderboard(latest_snapshot_df).write.format("delta").mode("overwrite").save(gold_collections["channel_leaderboard"])

            build_gold_views_timeseries(silver_df).write.format("delta").mode("overwrite").save(gold_collections["views_timeseries"])
            build_gold_region_timeseries(silver_df).write.format("delta").mode("overwrite").save(gold_collections["region_timeseries"])
            build_gold_duration_distribution(silver_df).write.format("delta").mode("overwrite").save(gold_collections["duration_distribution"])
            build_gold_subscriber_tier_distribution(silver_df).write.format("delta").mode("overwrite").save(gold_collections["subscriber_tier_distribution"])
            build_gold_tag_usage_frequency(silver_df).write.format("delta").mode("overwrite").save(gold_collections["tag_usage_frequency"])
            build_gold_trending_rank_distribution(silver_df).write.format("delta").mode("overwrite").save(gold_collections["trending_rank_distribution"])
        finally:
            try:
                latest_snapshot_df.unpersist()
            except Exception:
                # SparkContext may already be torn down on failure paths.
                pass
    finally:
        try:
            silver_df.unpersist()
        except Exception:
            pass


def duration_to_seconds_expr(column_name: str):
    return (
        when(col(column_name).isNull(), lit(0))
        .otherwise(
            coalesce_int(regexp_extract(col(column_name), r"(\d+)H", 1)) * lit(3600)
            + coalesce_int(regexp_extract(col(column_name), r"(\d+)M", 1)) * lit(60)
            + coalesce_int(regexp_extract(col(column_name), r"(\d+)S", 1))
        )
    )


def coalesce_int(column):
    return when(column == "", lit(0)).otherwise(column.cast("int"))


def upper_trimmed(column_name: str):
    return upper(trim(col(column_name)))
