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
    """Load filtered Silver rows from the local Delta table."""
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


def build_engagement_shift_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
) -> DataFrame:
    """Avg engagement_rate over time for the chosen category, optionally
    sliced by trending_region so the user can see whether engagement is
    rising or falling per region."""
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    return (
        sdf.groupBy("time_bucket", "trending_region")
        .agg(
            F.avg("engagement_rate").alias("avg_engagement_rate"),
            F.avg("view_count").alias("avg_views"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
        .orderBy("time_bucket", "trending_region")
    )


def build_channel_mix_shift_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
) -> DataFrame:
    """Share of trending videos by channel-size tier over time for the
    chosen category. Helps answer 'are big or small creators driving
    this category right now?'."""
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    sdf = sdf.withColumn(
        "subscriber_tier",
        F.when(F.col("channel_subscriber_count") < 100_000, F.lit("Small (<100K)"))
        .when(F.col("channel_subscriber_count") < 1_000_000, F.lit("Mid (100K-1M)"))
        .when(F.col("channel_subscriber_count") < 10_000_000, F.lit("Large (1M-10M)"))
        .otherwise(F.lit("Mega (10M+)")),
    )

    per_bucket = (
        sdf.groupBy("time_bucket", "subscriber_tier")
        .agg(F.countDistinct("video_id").alias("videos"))
    )

    bucket_totals = (
        sdf.groupBy("time_bucket")
        .agg(F.countDistinct("video_id").alias("total_videos"))
    )

    return (
        per_bucket.join(bucket_totals, on="time_bucket", how="inner")
        .withColumn("share", F.col("videos") / F.col("total_videos"))
        .orderBy("time_bucket", "subscriber_tier")
    )


def build_duration_velocity_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Avg velocity by duration_bucket, scoped to the user's selection.
    Answers: 'Which duration bucket has the strongest velocity?'."""
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
            F.avg("velocity").alias("avg_velocity"),
            F.avg("view_count").alias("avg_views"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
    )


def build_duration_category_overindex_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
    top_n_categories: int = 8,
) -> DataFrame:
    """For each duration_bucket, returns the share of each category along
    with that category's overall share and a 'lift' = bucket_share /
    overall_share. Lift > 1 means the category is over-indexing inside
    that duration bucket. Answers: 'Which categories are over-indexing
    in each duration bucket?'."""
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        # Note: for over-indexing analysis we typically want the full
        # population of categories — so we deliberately ignore the
        # category_name filter here. trending_region is still applied so
        # users can scope the analysis to a region.
        category_name=None,
        trending_region=trending_region,
    )

    per_video = sdf.dropDuplicates(["video_id", "duration_bucket", "category_name"])

    bucket_category = (
        per_video.groupBy("duration_bucket", "category_name")
        .agg(F.countDistinct("video_id").alias("videos_in_bucket_cat"))
    )
    bucket_totals = (
        per_video.groupBy("duration_bucket")
        .agg(F.countDistinct("video_id").alias("videos_in_bucket"))
    )
    category_totals = (
        per_video.groupBy("category_name")
        .agg(F.countDistinct("video_id").alias("videos_in_cat"))
    )
    grand_total = per_video.agg(
        F.countDistinct("video_id").alias("total_videos")
    ).collect()[0]["total_videos"] or 1

    # Keep only the top_n categories overall, so the heatmap stays readable.
    top_categories = [
        row["category_name"]
        for row in category_totals.orderBy(F.desc("videos_in_cat"))
        .limit(top_n_categories)
        .collect()
        if row["category_name"] is not None
    ]

    joined = (
        bucket_category.join(bucket_totals, on="duration_bucket", how="inner")
        .join(category_totals, on="category_name", how="inner")
        .filter(F.col("category_name").isin(top_categories))
        .withColumn(
            "bucket_share",
            F.col("videos_in_bucket_cat") / F.col("videos_in_bucket"),
        )
        .withColumn(
            "overall_share",
            F.col("videos_in_cat") / F.lit(grand_total),
        )
        .withColumn(
            "lift",
            F.when(
                F.col("overall_share") > 0,
                F.col("bucket_share") / F.col("overall_share"),
            ).otherwise(F.lit(None)),
        )
    )

    return joined.orderBy("duration_bucket", F.desc("lift"))


def build_duration_region_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """duration_bucket × trending_region engagement-rate heatmap. Answers:
    'Does duration interact with region differently?'."""
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        # Always keep all regions for this diagnostic — the whole point is
        # to compare regions side by side.
        trending_region=None,
    )

    return (
        sdf.groupBy("duration_bucket", "trending_region")
        .agg(
            F.avg("engagement_rate").alias("avg_engagement_rate"),
            F.avg("view_count").alias("avg_views"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
        .orderBy("trending_region", "duration_bucket")
    )


def _add_subscriber_tier_column(sdf: DataFrame) -> DataFrame:
    """Bucket channel_subscriber_count into the same tier labels used
    elsewhere in the dashboard so all subscriber diagnostics line up."""
    return sdf.withColumn(
        "subscriber_tier",
        F.when(F.col("channel_subscriber_count") < 100_000, F.lit("Small (<100K)"))
        .when(F.col("channel_subscriber_count") < 1_000_000, F.lit("Mid (100K-1M)"))
        .when(F.col("channel_subscriber_count") < 10_000_000, F.lit("Large (1M-10M)"))
        .otherwise(F.lit("Mega (10M+)")),
    )


def build_subscriber_views_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Avg views per subscriber tier. Answers: 'Average views by
    subscriber tier'."""
    sdf = _add_subscriber_tier_column(
        _load_filtered_silver(
            spark,
            silver_path,
            history_window,
            category_name=category_name,
            trending_region=trending_region,
        )
    )

    return (
        sdf.groupBy("subscriber_tier")
        .agg(
            F.avg("view_count").alias("avg_views"),
            F.avg("like_count").alias("avg_likes"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
    )


def build_subscriber_engagement_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Avg engagement_rate per subscriber tier. Answers: 'Average
    engagement by subscriber tier'."""
    sdf = _add_subscriber_tier_column(
        _load_filtered_silver(
            spark,
            silver_path,
            history_window,
            category_name=category_name,
            trending_region=trending_region,
        )
    )

    return (
        sdf.groupBy("subscriber_tier")
        .agg(
            F.avg("engagement_rate").alias("avg_engagement_rate"),
            F.avg("view_count").alias("avg_views"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
    )


def build_subscriber_persistence_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Average number of distinct collection batches each video persists
    in trending, grouped by subscriber tier. Answers: 'Persistence by
    subscriber tier'."""
    sdf = _add_subscriber_tier_column(
        _load_filtered_silver(
            spark,
            silver_path,
            history_window,
            category_name=category_name,
            trending_region=trending_region,
        )
    )

    per_video = (
        sdf.groupBy("video_id", "subscriber_tier")
        .agg(
            F.countDistinct("collection_batch_id").alias("batches_appeared"),
            F.max("view_count").alias("peak_views"),
        )
    )

    return (
        per_video.groupBy("subscriber_tier")
        .agg(
            F.avg("batches_appeared").alias("avg_batches"),
            F.expr("percentile_approx(batches_appeared, 0.5)").alias("median_batches"),
            F.avg("peak_views").alias("avg_peak_views"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
    )


def build_subscriber_region_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
) -> DataFrame:
    """Subscriber-tier x region share of trending videos. Answers:
    'Region-specific creator size dependence'."""
    sdf = _add_subscriber_tier_column(
        _load_filtered_silver(
            spark,
            silver_path,
            history_window,
            category_name=category_name,
            # Always keep all regions — comparison is the point.
            trending_region=None,
        )
    )

    per_video = sdf.dropDuplicates(["video_id", "trending_region", "subscriber_tier"])

    region_tier = (
        per_video.groupBy("trending_region", "subscriber_tier")
        .agg(F.countDistinct("video_id").alias("videos"))
    )
    region_totals = (
        per_video.groupBy("trending_region")
        .agg(F.countDistinct("video_id").alias("region_total"))
    )

    return (
        region_tier.join(region_totals, on="trending_region", how="inner")
        .withColumn(
            "tier_share",
            F.col("videos") / F.col("region_total"),
        )
        .orderBy("trending_region", "subscriber_tier")
    )


def build_rank_new_vs_persisting_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
) -> DataFrame:
    """For the chosen category, count distinct videos per time_bucket
    split into 'new entries' (first time in trending) vs 'persisting'
    (appeared in an earlier bucket). Answers: 'Is the category gaining
    many new entries or just persisting longer?'."""
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    # Earliest time_bucket per video inside this category/region scope.
    first_seen = (
        sdf.groupBy("video_id")
        .agg(F.min("time_bucket").alias("first_time_bucket"))
    )

    annotated = (
        sdf.join(first_seen, on="video_id", how="inner")
        .withColumn(
            "entry_status",
            F.when(
                F.col("time_bucket") == F.col("first_time_bucket"),
                F.lit("New entry"),
            ).otherwise(F.lit("Persisting")),
        )
        .dropDuplicates(["video_id", "time_bucket", "entry_status"])
    )

    return (
        annotated.groupBy("time_bucket", "entry_status")
        .agg(F.countDistinct("video_id").alias("videos"))
        .orderBy("time_bucket", "entry_status")
    )


def build_top_rank_channel_concentration_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
    top_rank_threshold: int = 10,
    top_n_channels: int = 12,
) -> DataFrame:
    """For the chosen category, count how many top-ranked trending slots
    each channel holds. Answers: 'Are top ranks concentrated in a few
    channels?'."""
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    top_slots = sdf.filter(
        F.col("trending_rank").isNotNull()
        & (F.col("trending_rank") <= F.lit(top_rank_threshold))
    )

    return (
        top_slots.groupBy("channel_title")
        .agg(
            F.count("*").alias("top_rank_slots"),
            F.countDistinct("video_id").alias("unique_videos"),
            F.avg("trending_rank").alias("avg_rank"),
            F.avg("view_count").alias("avg_views"),
        )
        .filter(F.col("channel_title").isNotNull())
        .orderBy(F.desc("top_rank_slots"))
        .limit(top_n_channels)
    )


def build_velocity_vs_rank_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
    max_videos: int = 400,
) -> DataFrame:
    """Per-video avg velocity vs best (lowest) trending_rank within the
    chosen category. Answers: 'Are rank improvements driven by high
    velocity?'."""
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    return (
        sdf.filter(F.col("trending_rank").isNotNull())
        .groupBy("video_id", "channel_title")
        .agg(
            F.avg("velocity").alias("avg_velocity"),
            F.min("trending_rank").alias("best_rank"),
            F.avg("view_count").alias("avg_views"),
            F.avg("engagement_rate").alias("avg_engagement_rate"),
        )
        .orderBy(F.asc("best_rank"))
        .limit(max_videos)
    )


def build_category_share_drivers_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
) -> DataFrame:
    """Per time_bucket: unique-video count vs avg views per video for the
    chosen category. Answers: 'Is the category's share rising because of
    more videos or stronger individual videos?'."""
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
            F.countDistinct("video_id").alias("unique_videos"),
            F.avg("view_count").alias("avg_views_per_video"),
            F.sum("view_count").alias("total_views"),
            F.avg("engagement_rate").alias("avg_engagement_rate"),
        )
        .orderBy("time_bucket")
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


def build_high_engagement_tags_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
    min_videos: int = 5,
    top_n: int = 25,
) -> DataFrame:
    """Top tags ranked by avg engagement_rate (with a minimum frequency
    threshold so single-use noise tags don't dominate). Answers:
    'Which tags are associated with higher engagement?'."""
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
        .filter(F.col("videos_using_tag") >= F.lit(min_videos))
        .orderBy(F.desc("avg_engagement_rate"))
        .limit(top_n)
    )


def build_tag_region_concentration_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    min_videos: int = 8,
    top_n: int = 25,
) -> DataFrame:
    """For each tag, returns the share of usage across trending_region and
    a 'region_concentration' metric (max region share). Tags with high
    concentration are region-specific; tags with low concentration are
    more global. Answers: 'Which tags are region-specific vs global?'."""
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        # Keep all regions — the comparison only makes sense across regions.
        trending_region=None,
    )

    exploded = (
        sdf.withColumn("tag", F.explode("tags_array"))
        .filter(F.col("tag").isNotNull() & (F.trim(F.col("tag")) != ""))
        .select("tag", "trending_region", "video_id")
        .dropDuplicates(["tag", "trending_region", "video_id"])
    )

    per_tag_region = (
        exploded.groupBy("tag", "trending_region")
        .agg(F.countDistinct("video_id").alias("videos_in_region"))
    )
    per_tag_total = (
        exploded.groupBy("tag")
        .agg(F.countDistinct("video_id").alias("videos_total"))
        .filter(F.col("videos_total") >= F.lit(min_videos))
    )

    joined = (
        per_tag_region.join(per_tag_total, on="tag", how="inner")
        .withColumn(
            "region_share",
            F.col("videos_in_region") / F.col("videos_total"),
        )
    )

    # Compute concentration (max region share) per tag and pick top_n tags
    # by total volume so the chart is readable.
    concentration = (
        joined.groupBy("tag", "videos_total")
        .agg(F.max("region_share").alias("region_concentration"))
    )
    top_tags = [
        row["tag"]
        for row in concentration.orderBy(F.desc("videos_total"))
        .limit(top_n)
        .collect()
    ]

    return (
        joined.filter(F.col("tag").isin(top_tags))
        .join(
            concentration.select("tag", "region_concentration"),
            on="tag",
            how="inner",
        )
        .orderBy(F.desc("region_concentration"), "tag", "trending_region")
    )


def build_tag_density_performance_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Bucket each video by how many tags it carries and report average
    engagement / views / sample size per bucket. Answers: 'Are tag-heavy
    videos outperforming tag-light videos?'."""
    sdf = _load_filtered_silver(
        spark,
        silver_path,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )

    per_video = (
        sdf.dropDuplicates(["video_id"])
        .withColumn(
            "tag_count",
            F.when(F.col("tags_array").isNull(), F.lit(0))
            .otherwise(F.size(F.col("tags_array"))),
        )
        .withColumn(
            "tag_density_bucket",
            F.when(F.col("tag_count") == 0, F.lit("0 tags"))
            .when(F.col("tag_count") <= 5, F.lit("1-5 tags"))
            .when(F.col("tag_count") <= 10, F.lit("6-10 tags"))
            .when(F.col("tag_count") <= 20, F.lit("11-20 tags"))
            .otherwise(F.lit("21+ tags")),
        )
    )

    return (
        per_video.groupBy("tag_density_bucket")
        .agg(
            F.avg("engagement_rate").alias("avg_engagement_rate"),
            F.avg("view_count").alias("avg_views"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
    )


# ===========================================================================
# v2 diagnostics: mechanical decompositions of the descriptive trend
# ===========================================================================
# Each pair below decomposes one descriptive chart's trend into the levers
# that produced it (volume vs strength, persistence vs participation, etc.)
# rather than reporting parallel facts. See proposal in the dashboard
# discussion for the rationale per chart.

# ----- 1. Views Trend Over Time by Category --------------------------------
def build_views_volume_strength_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
) -> DataFrame:
    """Decomposes total_views over time into (unique_videos × avg_views_per_video)
    so the user can see whether growth is volume-led or strength-led."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return (
        sdf.groupBy("time_bucket")
        .agg(
            F.countDistinct("video_id").alias("unique_videos"),
            F.avg("view_count").alias("avg_views_per_video"),
            F.sum("view_count").alias("total_views"),
        )
        .orderBy("time_bucket")
    )


def build_views_new_vs_carryover_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
) -> DataFrame:
    """For each time_bucket, splits videos into 'New entry' (first time
    seen in trending in the window) vs 'Carry-over' (seen earlier).
    Tells the user whether the trend is fed by fresh content or by
    existing videos persisting."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    first_seen = (
        sdf.groupBy("video_id")
        .agg(F.min("time_bucket").alias("first_time_bucket"))
    )
    annotated = (
        sdf.join(first_seen, on="video_id", how="inner")
        .withColumn(
            "entry_status",
            F.when(
                F.col("time_bucket") == F.col("first_time_bucket"),
                F.lit("New entry"),
            ).otherwise(F.lit("Carry-over")),
        )
        .dropDuplicates(["video_id", "time_bucket", "entry_status"])
    )
    return (
        annotated.groupBy("time_bucket", "entry_status")
        .agg(F.countDistinct("video_id").alias("videos"))
        .orderBy("time_bucket", "entry_status")
    )


# ----- 2. Video Duration Distribution --------------------------------------
def build_duration_slot_footprint_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Per duration_bucket: distinct videos that hit trending (volume lever)
    and avg batches each one persists (stickiness lever). Decomposes
    bucket dominance into volume vs persistence."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    sdf = sdf.filter(F.col("duration_bucket").isNotNull())
    per_video = (
        sdf.groupBy("video_id", "duration_bucket")
        .agg(F.countDistinct("collection_batch_id").alias("batches"))
    )
    return (
        per_video.groupBy("duration_bucket")
        .agg(
            F.countDistinct("video_id").alias("distinct_videos"),
            F.avg("batches").alias("avg_batches_per_video"),
            (
                F.countDistinct("video_id") * F.avg("batches")
            ).alias("slot_footprint"),
        )
    )


def build_duration_audience_response_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Per duration_bucket: avg engagement_rate and avg views per video.
    Tells the user whether bucket dominance reflects audience preference
    or just publishing volume."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    sdf = sdf.filter(F.col("duration_bucket").isNotNull())
    return (
        sdf.groupBy("duration_bucket")
        .agg(
            F.avg("engagement_rate").alias("avg_engagement_rate"),
            F.avg("view_count").alias("avg_views_per_video"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
    )


# ----- 3. Channel Subscriber Size Distribution -----------------------------
def build_tier_effort_reward_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Per subscriber_tier: distinct channels participating × avg trending
    slots per channel. Decomposes tier dominance into broad participation
    vs concentrated dominance."""
    sdf = _add_subscriber_tier_column(
        _load_filtered_silver(
            spark, silver_path, history_window,
            category_name=category_name, trending_region=trending_region,
        )
    )
    sdf = sdf.filter(F.col("channel_id").isNotNull())
    per_channel = (
        sdf.groupBy("subscriber_tier", "channel_id")
        .agg(F.count("*").alias("slot_observations"))
    )
    return (
        per_channel.groupBy("subscriber_tier")
        .agg(
            F.countDistinct("channel_id").alias("distinct_channels"),
            F.avg("slot_observations").alias("avg_slots_per_channel"),
            F.sum("slot_observations").alias("total_slot_observations"),
        )
    )


def build_tier_persistence_engagement_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
) -> DataFrame:
    """Per subscriber_tier: avg batches each video persists × avg
    engagement rate. Decomposes whether a tier holds slots through
    stickiness or audience response."""
    sdf = _add_subscriber_tier_column(
        _load_filtered_silver(
            spark, silver_path, history_window,
            category_name=category_name, trending_region=trending_region,
        )
    )
    per_video = (
        sdf.groupBy("video_id", "subscriber_tier")
        .agg(
            F.countDistinct("collection_batch_id").alias("batches"),
            F.avg("engagement_rate").alias("video_er"),
        )
    )
    return (
        per_video.groupBy("subscriber_tier")
        .agg(
            F.avg("batches").alias("avg_batches_per_video"),
            F.avg("video_er").alias("avg_engagement_rate"),
            F.countDistinct("video_id").alias("unique_videos"),
        )
    )


# ----- 4. Top Tag Usage Frequency ------------------------------------------
def build_tag_adoption_intensity_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
    top_n: int = 15,
) -> DataFrame:
    """For each top tag: distinct channels using it × avg videos per
    channel. Decomposes a tag's frequency into broad adoption vs
    per-channel intensity."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    exploded = (
        sdf.withColumn("tag", F.explode("tags_array"))
        .filter(F.col("tag").isNotNull() & (F.trim(F.col("tag")) != ""))
        .filter(F.col("channel_id").isNotNull())
    )
    top_tags_df = (
        exploded.groupBy("tag")
        .agg(F.countDistinct("video_id").alias("videos_using_tag"))
        .orderBy(F.desc("videos_using_tag"))
        .limit(top_n)
    )
    top_list = [r["tag"] for r in top_tags_df.collect()]
    if not top_list:
        return top_tags_df  # empty schema, dashboard will detect via .empty

    filtered = exploded.filter(F.col("tag").isin(top_list))
    return (
        filtered.groupBy("tag")
        .agg(
            F.countDistinct("channel_id").alias("distinct_channels"),
            F.countDistinct("video_id").alias("videos_using_tag"),
        )
        .withColumn(
            "videos_per_channel",
            F.col("videos_using_tag") / F.col("distinct_channels"),
        )
        .orderBy(F.desc("videos_using_tag"))
    )


def build_tag_cooccurrence_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str | None = None,
    trending_region: str | None = None,
    top_n: int = 12,
) -> DataFrame:
    """For the most-used tag in scope, returns the other tags that ride
    along with it most often. Reveals whether the dominance is part of
    a multi-tag template (one tag dominates co-occurrence) or whether
    the tag is genuinely versatile (co-occurrences spread out)."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    exploded = (
        sdf.withColumn("tag", F.explode("tags_array"))
        .filter(F.col("tag").isNotNull() & (F.trim(F.col("tag")) != ""))
        .select("video_id", "tag")
        .distinct()
    )
    top1 = (
        exploded.groupBy("tag")
        .agg(F.countDistinct("video_id").alias("videos"))
        .orderBy(F.desc("videos"))
        .limit(1)
        .collect()
    )
    if not top1:
        return exploded.limit(0).select("tag").withColumn("co_videos", F.lit(0)).withColumn("primary_tag", F.lit(""))

    primary = top1[0]["tag"]
    primary_videos = top1[0]["videos"] or 1

    videos_with_primary = (
        exploded.filter(F.col("tag") == primary).select("video_id").distinct()
    )
    co_tags = (
        exploded.join(videos_with_primary, on="video_id", how="inner")
        .filter(F.col("tag") != primary)
        .groupBy("tag")
        .agg(F.countDistinct("video_id").alias("co_videos"))
        .withColumn("co_share", F.col("co_videos") / F.lit(primary_videos))
        .withColumn("primary_tag", F.lit(primary))
        .orderBy(F.desc("co_videos"))
        .limit(top_n)
    )
    return co_tags


# ----- 5. Trending Rank Distribution by Category ---------------------------
def build_rank_slot_turnover_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
) -> DataFrame:
    """Per (region within scope): distinct videos that ever held a slot,
    total slot-batch observations, and turnover_rate = distinct_videos /
    slot_observations. High turnover = vibrant pipeline; low = a few
    persistent videos hogging the category."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    sdf = sdf.filter(F.col("trending_rank").isNotNull())
    return (
        sdf.groupBy("trending_region")
        .agg(
            F.countDistinct("video_id").alias("distinct_videos"),
            F.count("*").alias("slot_observations"),
        )
        .withColumn(
            "turnover_rate",
            F.when(
                F.col("slot_observations") > 0,
                F.col("distinct_videos") / F.col("slot_observations"),
            ).otherwise(F.lit(0)),
        )
        .orderBy(F.desc("turnover_rate"))
    )


def build_rank_channel_concentration_v2_diagnostic(
    spark: SparkSession,
    silver_path: str,
    history_window: str,
    category_name: str,
    trending_region: str | None = None,
    top_n: int = 12,
) -> DataFrame:
    """For the selected category: each channel's share of all slot
    observations, top_n. Lets the user judge whether a category is
    held up by a few dominant channels (high top-3 share) or broadly
    distributed."""
    sdf = _load_filtered_silver(
        spark, silver_path, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    sdf = sdf.filter(
        F.col("trending_rank").isNotNull() & F.col("channel_title").isNotNull()
    )
    per_channel = (
        sdf.groupBy("channel_title")
        .agg(F.count("*").alias("slot_observations"))
    )
    total = per_channel.agg(F.sum("slot_observations").alias("t")).collect()[0]["t"] or 1
    return (
        per_channel.withColumn(
            "share_pct",
            F.col("slot_observations") / F.lit(total) * F.lit(100.0),
        )
        .orderBy(F.desc("slot_observations"))
        .limit(top_n)
    )
