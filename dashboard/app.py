import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.sql.types import TimestampType

from spark_processing.storage_paths import get_medallion_paths
from analytics.business_analysis import (
    prepare_dashboard_df,
    build_category_summary,
    build_top_videos,
    build_diagnostic_table,
    build_forecast,
    build_recommendations,
    build_views_timeseries,
    build_region_timeseries,
    build_publish_hour_heatmap,
    build_category_share_over_time,
    build_channel_leaderboard,
    build_bubble_dataset,
    build_outlier_videos,
    build_category_growth,
    build_comments_vs_views,
    build_engagement_heatmap,
    build_duration_distribution,
    build_subscriber_tier_distribution,
    build_tag_usage_frequency,
    build_hd_sd_distribution,
    build_caption_rate,
    build_view_velocity,
    build_trending_persistence,
    build_rank_movement,
    build_engagement_vs_views_correlation,
    build_title_characteristics,
    build_tag_count_vs_engagement,
    build_duration_vs_engagement,
    build_channel_size_vs_reach,
    build_regional_preference_divergence,
    build_recency_bias,
    build_weekend_weekday_behavior,
    build_trending_rank_distribution,
    build_trending_entry_probability,
    build_view_count_forecast_v2,
    build_trending_duration_prediction,
    build_peak_rank_forecast,
    build_category_share_forecast,
    build_optimal_posting_window,
    build_trending_gap_opportunity,
    build_creator_partnership_recommendations,
    build_format_prescriptions,
    build_campaign_timing_alerts,
    build_regional_expansion_recommendations,
)

MEDALLION_PATHS = get_medallion_paths()
SILVER_DELTA_PATH = MEDALLION_PATHS["silver"]
GOLD_LATEST_SNAPSHOT_PATH = MEDALLION_PATHS["gold"]["latest_snapshot"]
GOLD_CATEGORY_SUMMARY_PATH = MEDALLION_PATHS["gold"]["category_summary"]
GOLD_VIEWS_TIMESERIES_PATH = MEDALLION_PATHS["gold"]["views_timeseries"]
GOLD_REGION_TIMESERIES_PATH = MEDALLION_PATHS["gold"]["region_timeseries"]
GOLD_CHANNEL_LEADERBOARD_PATH = MEDALLION_PATHS["gold"]["channel_leaderboard"]
MAX_DASHBOARD_ROWS = 50000
DEFAULT_HISTORY_WINDOW_INDEX = 1

st.set_page_config(page_title="Business-Ready YouTube Analytics", layout="wide")
st.title("YouTube Business Analytics Dashboard")
st.caption("Descriptive, diagnostic, predictive, and prescriptive analytics for trending YouTube content")

def _safe_pandas_from_spark(sdf):
    timestamp_columns = [
        field.name for field in sdf.schema.fields if isinstance(field.dataType, TimestampType)
    ]
    for column_name in timestamp_columns:
        sdf = sdf.withColumn(column_name, col(column_name).cast("string"))
    return sdf.toPandas()


@st.cache_resource(show_spinner=False)
def get_spark():
    return (
        SparkSession.builder.appName("YouTubeAnalyticsDashboard")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
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


@st.cache_data(ttl=60, show_spinner=False)
def load_optional_delta(path: str, max_rows: int | None = None):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()

        spark = get_spark()
        sdf = spark.read.format("delta").load(path)

        if sdf.limit(1).count() == 0:
            return pd.DataFrame()

        if max_rows is not None:
            sdf = sdf.limit(max_rows)

        return _safe_pandas_from_spark(sdf)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_silver_filtered(selected_category: str, selected_region: str, history_window: str):
    try:
        if not os.path.exists(SILVER_DELTA_PATH):
            return pd.DataFrame(), False

        spark = get_spark()
        sdf = spark.read.format("delta").load(SILVER_DELTA_PATH)

        if selected_category != "All":
            sdf = sdf.filter(col("category_name") == selected_category)

        if selected_region != "All":
            sdf = sdf.filter(col("trending_region") == selected_region)

        window_days = {
            "Last 24 Hours": 1,
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "All Available": None,
        }
        days = window_days[history_window]
        if days is not None and "collected_at_ts" in sdf.columns:
            sdf = sdf.filter(col("collected_at_ts") >= expr(f"current_timestamp() - INTERVAL {days} DAYS"))

        if sdf.limit(1).count() == 0:
            return pd.DataFrame(), False

        dashboard_columns = [
            "collection_batch_id",
            "collected_at",
            "collected_at_ts",
            "surface",
            "trending_region",
            "trending_category_id",
            "trending_page",
            "trending_rank",
            "video_id",
            "title",
            "description",
            "published_at",
            "published_at_ts",
            "category_id",
            "category_name",
            "tags",
            "default_language",
            "thumbnail_url",
            "view_count",
            "like_count",
            "comment_count",
            "favorite_count",
            "duration_iso",
            "definition",
            "caption",
            "licensed_content",
            "content_rating",
            "projection",
            "channel_id",
            "channel_title",
            "channel_subscriber_count",
            "channel_view_count",
            "channel_video_count",
            "channel_country",
            "engagements",
            "like_rate",
            "comment_rate",
            "engagement_rate",
            "video_age_hours",
            "velocity",
            "title_word_count",
            "title_has_question",
            "title_has_number",
            "title_caps_ratio",
            "tags_array",
            "tag_count",
            "duration_seconds",
            "duration_bucket",
            "publish_day",
            "publish_hour",
            "time_bucket",
        ]
        available_columns = [name for name in dashboard_columns if name in sdf.columns]
        sdf = sdf.select(*available_columns)
        if "collected_at_ts" in sdf.columns:
            sdf = sdf.orderBy(col("collected_at_ts").desc())

        truncated = False
        if sdf.limit(MAX_DASHBOARD_ROWS + 1).count() > MAX_DASHBOARD_ROWS:
            sdf = sdf.limit(MAX_DASHBOARD_ROWS)
            truncated = True

        return _safe_pandas_from_spark(sdf), truncated
    except Exception as exc:
        st.error(f"Could not load Delta data: {exc}")
        return pd.DataFrame(), False


@st.cache_data(ttl=60, show_spinner=False)
def load_filter_source():
    if not gold_latest_snapshot_df.empty:
        return gold_latest_snapshot_df

    try:
        if not os.path.exists(SILVER_DELTA_PATH):
            return pd.DataFrame()

        spark = get_spark()
        sdf = (
            spark.read.format("delta").load(SILVER_DELTA_PATH)
            .select("category_name", "trending_region")
            .dropna(subset=["category_name", "trending_region"])
            .distinct()
            .orderBy("category_name", "trending_region")
        )
        return _safe_pandas_from_spark(sdf)
    except Exception:
        return pd.DataFrame()


gold_latest_snapshot_df = load_optional_delta(GOLD_LATEST_SNAPSHOT_PATH)
gold_category_summary_df = load_optional_delta(GOLD_CATEGORY_SUMMARY_PATH)
gold_views_timeseries_df = load_optional_delta(GOLD_VIEWS_TIMESERIES_PATH)
gold_region_timeseries_df = load_optional_delta(GOLD_REGION_TIMESERIES_PATH)
gold_channel_leaderboard_df = load_optional_delta(GOLD_CHANNEL_LEADERBOARD_PATH)

st.sidebar.header("Filters")

filter_source_df = load_filter_source()

if filter_source_df.empty:
    st.warning("No data available. Run the producer and Spark streaming first.")
    st.stop()

category_options = ["All"] + sorted(filter_source_df["category_name"].dropna().astype(str).unique().tolist())
region_options = ["All"] + sorted(filter_source_df["trending_region"].dropna().astype(str).unique().tolist())

selected_category = st.sidebar.selectbox("Category", category_options)
selected_region = st.sidebar.selectbox("Region", region_options)
history_window = st.sidebar.selectbox(
    "Silver History Window",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Available"],
    index=DEFAULT_HISTORY_WINDOW_INDEX,
    help="Gold tables still use full history. This window limits only detailed Silver-level computations.",
)

raw_df, silver_truncated = load_silver_filtered(selected_category, selected_region, history_window)
df = prepare_dashboard_df(raw_df)

if silver_truncated:
    st.info(
        f"Detailed Silver analysis is capped to the most recent {MAX_DASHBOARD_ROWS:,} rows for dashboard stability. "
        "Gold summary tables still use the full Silver history."
    )

if df.empty and filter_source_df.empty:
    st.warning("No data available. Run the producer and Spark streaming first.")
    st.stop()

if df.empty:
    st.warning("No Silver data matched the selected filters/window. Gold summary tables may still populate some charts.")
    st.stop()

use_gold_category_summary = (
    not gold_category_summary_df.empty and selected_region == "All" and selected_category == "All"
)
summary_df = gold_category_summary_df if use_gold_category_summary else build_category_summary(df)
top_videos_df = build_top_videos(df)
diagnostic_df = build_diagnostic_table(df)
forecast_df = build_forecast(df)
recommendations_df = build_recommendations(df)

use_gold_views_ts = (
    not gold_views_timeseries_df.empty and selected_region == "All" and selected_category == "All"
)
views_ts_df = gold_views_timeseries_df if use_gold_views_ts else build_views_timeseries(df)

if not gold_region_timeseries_df.empty and selected_category == "All":
    region_ts_df = gold_region_timeseries_df.copy()
    if selected_region != "All":
        region_ts_df = region_ts_df[region_ts_df["trending_region"] == selected_region]
else:
    region_ts_df = build_region_timeseries(df)

publish_heatmap_df = build_publish_hour_heatmap(df)
category_share_df = build_category_share_over_time(df)
use_gold_channel_board = (
    not gold_channel_leaderboard_df.empty and selected_region == "All" and selected_category == "All"
)
channel_board_df = gold_channel_leaderboard_df if use_gold_channel_board else build_channel_leaderboard(df)
bubble_df = build_bubble_dataset(df)
outlier_df = build_outlier_videos(df)
growth_df = build_category_growth(df)
comments_vs_views_df = build_comments_vs_views(df)

engagement_heatmap_df = build_engagement_heatmap(df)
duration_dist_df = build_duration_distribution(df)
subscriber_tier_df = build_subscriber_tier_distribution(df)
tag_usage_df = build_tag_usage_frequency(df)
hd_sd_df = build_hd_sd_distribution(df)
caption_rate_df = build_caption_rate(df)

velocity_df = build_view_velocity(df)
persistence_df = build_trending_persistence(df)
rank_movement_df = build_rank_movement(df)
corr_df, corr_scatter_df = build_engagement_vs_views_correlation(df)
title_char_df = build_title_characteristics(df)
tag_count_df = build_tag_count_vs_engagement(df)
duration_eng_df = build_duration_vs_engagement(df)
channel_size_df = build_channel_size_vs_reach(df)
regional_div_df = build_regional_preference_divergence(df)
recency_bias_df = build_recency_bias(df)
weekend_weekday_df = build_weekend_weekday_behavior(df)

trending_rank_dist_df = build_trending_rank_distribution(df)
entry_probability_df = build_trending_entry_probability(df)
video_forecast_df = build_view_count_forecast_v2(df)
duration_prediction_df = build_trending_duration_prediction(df)
peak_rank_forecast_df = build_peak_rank_forecast(df)
category_share_forecast_df = build_category_share_forecast(df)
optimal_posting_df = build_optimal_posting_window(df)
gap_opportunity_df = build_trending_gap_opportunity(df)
creator_partnership_df = build_creator_partnership_recommendations(df)
format_prescriptions_df = build_format_prescriptions(df)
campaign_alerts_df = build_campaign_timing_alerts(df)
regional_expansion_df = build_regional_expansion_recommendations(df)


if not gold_latest_snapshot_df.empty:
    latest_records = gold_latest_snapshot_df.copy()
    if selected_category != "All":
        latest_records = latest_records[latest_records["category_name"] == selected_category]
    if selected_region != "All":
        latest_records = latest_records[latest_records["trending_region"] == selected_region]
else:
    latest_records = df.sort_values("collected_at").drop_duplicates(subset=["video_id", "trending_region"], keep="last")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Unique Videos", f"{latest_records['video_id'].nunique():,}")
k2.metric("Total Views", f"{int(latest_records['view_count'].sum()):,}")

avg_eng = pd.to_numeric(latest_records["engagement_rate"], errors="coerce")
avg_eng = avg_eng.replace([float("inf"), -float("inf")], pd.NA).dropna()
avg_eng_value = 0 if avg_eng.empty else avg_eng.mean()
k3.metric("Avg Engagement Rate", f"{avg_eng_value * 100:.2f}%")

k4.metric("Tracked Categories", f"{latest_records['category_name'].nunique():,}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
)

with tab1:
    st.subheader("1. What is happening?")
    # st.markdown("Analytical question: Which categories, channels, and videos are driving the most business value?")

    # category_chart = (
    #     alt.Chart(summary_df)
    #     .mark_bar()
    #     .encode(
    #         x=alt.X("total_views:Q", title="Total Views"),
    #         y=alt.Y("category_name:N", sort="-x", title="Category"),
    #         tooltip=[
    #             "category_name",
    #             "videos",
    #             "total_views",
    #             "total_likes",
    #             "total_comments",
    #             alt.Tooltip("avg_engagement_rate:Q", format=".2%"),
    #         ],
    #         color=alt.Color("avg_engagement_rate:Q", title="Avg Engagement Rate"),
    #     )
    #     .properties(height=350)
    # )
    # st.altair_chart(category_chart, use_container_width=True)

    st.subheader("Views Trend Over Time by Category")
    st.markdown("Analytical question: Which categories are sustaining momentum over time?")

    if views_ts_df.empty:
        st.info("Not enough timestamped data yet for time-series trend analysis.")
    else:
        views_line = (
            alt.Chart(views_ts_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("total_views:Q", title="Total Views"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    alt.Tooltip("category_name:N", title="Category"),
                    alt.Tooltip("time_bucket:T", title="Time"),
                    alt.Tooltip("total_views:Q", title="Total Views", format=","),
                    alt.Tooltip("total_engagements:Q", title="Total Engagements", format=","),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(views_line, use_container_width=True)

    # st.subheader("Category Share of Total Views Over Time")
    # st.markdown("Analytical question: How is category dominance changing over time?")

    # if category_share_df.empty:
    #     st.info("Category share trend needs time-based data.")
    # else:
    #     area_chart = (
    #         alt.Chart(category_share_df)
    #         .mark_area()
    #         .encode(
    #             x=alt.X("time_bucket:T", title="Time"),
    #             y=alt.Y("view_share:Q", stack="normalize", title="Share of Views"),
    #             color=alt.Color("category_name:N", title="Category"),
    #             tooltip=[
    #                 alt.Tooltip("category_name:N", title="Category"),
    #                 alt.Tooltip("time_bucket:T", title="Time"),
    #                 alt.Tooltip("total_views:Q", title="Total Views", format=","),
    #                 alt.Tooltip("view_share:Q", title="View Share", format=".2%"),
    #             ],
    #         )
    #         .properties(height=350)
    #     )
    #     st.altair_chart(area_chart, use_container_width=True)
    
    st.subheader("Engagement Rate Heatmap (Category x Region)")
    st.markdown("Analytical question: Which category-region combinations show the strongest engagement?")

    if engagement_heatmap_df.empty:
        st.info("No engagement heatmap data available.")
    else:
        er_heatmap = (
            alt.Chart(engagement_heatmap_df)
            .mark_rect()
            .encode(
                x=alt.X("trending_region:N", title="Region"),
                y=alt.Y("category_name:N", title="Category"),
                color=alt.Color("avg_engagement_rate:Q", title="Avg Engagement Rate"),
                tooltip=[
                    "trending_region",
                    "category_name",
                    alt.Tooltip("avg_engagement_rate:Q", format=".2%"),
                    alt.Tooltip("avg_like_rate:Q", format=".2%"),
                    alt.Tooltip("avg_comment_rate:Q", format=".2%"),
                    "sample_size",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(er_heatmap, use_container_width=True)

    st.subheader("Video Duration Distribution")
    st.markdown("Analytical question: Which duration buckets are most common and which perform better?")

    if duration_dist_df.empty:
        st.info("No duration distribution data available.")
    else:
        duration_chart = (
            alt.Chart(duration_dist_df)
            .mark_bar()
            .encode(
                x=alt.X("duration_bucket:N", title="Duration Bucket"),
                y=alt.Y("video_count:Q", title="Video Count"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "trending_region",
                    "category_name",
                    "duration_bucket",
                    "video_count",
                    alt.Tooltip("avg_views_in_bucket:Q", format=","),
                    alt.Tooltip("avg_er_in_bucket:Q", format=".2%"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(duration_chart, use_container_width=True)

    st.subheader("Channel Subscriber Size Distribution")
    st.markdown("Analytical question: Are trending videos dominated by mega channels or smaller creators?")

    if subscriber_tier_df.empty:
        st.info("No subscriber tier data available.")
    else:
        subscriber_chart = (
            alt.Chart(subscriber_tier_df)
            .mark_bar()
            .encode(
                x=alt.X("trending_region:N", title="Region"),
                y=alt.Y("video_count:Q", stack="normalize", title="Share of Trending Videos"),
                color=alt.Color("subscriber_tier:N", title="Subscriber Tier"),
                tooltip=[
                    "trending_region",
                    "subscriber_tier",
                    "video_count",
                    alt.Tooltip("pct:Q", format=".1f"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(subscriber_chart, use_container_width=True)

    st.subheader("Top Tag Usage Frequency")
    st.markdown("Analytical question: Which tags appear most often in trending videos?")

    if tag_usage_df.empty:
        st.info("No tag usage data available.")
    else:
        tag_chart = (
            alt.Chart(tag_usage_df)
            .mark_bar()
            .encode(
                x=alt.X("videos_using_tag:Q", title="Videos Using Tag"),
                y=alt.Y("tag:N", sort="-x", title="Tag"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "tag",
                    "category_name",
                    "trending_region",
                    "videos_using_tag",
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(tag_chart, use_container_width=True)


    st.subheader("HD vs SD Distribution")
    st.markdown("Analytical question: What quality standard dominates trending content?")

    if hd_sd_df.empty:
        st.info("No HD/SD distribution data available.")
    else:
        hd_sd_chart = (
            alt.Chart(hd_sd_df)
            .mark_bar()
            .encode(
                x=alt.X("trending_region:N", title="Region"),
                y=alt.Y("video_count:Q", stack="normalize", title="Share of Videos"),
                color=alt.Color("definition:N", title="Definition"),
                tooltip=[
                    "trending_region",
                    "category_name",
                    "definition",
                    "video_count",
                    alt.Tooltip("pct:Q", format=".1f"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(hd_sd_chart, use_container_width=True)

    
    st.subheader("Caption / Subtitle Availability Rate")
    st.markdown("Analytical question: Which categories and regions have better accessibility coverage?")

    if caption_rate_df.empty:
        st.info("No caption rate data available.")
    else:
        caption_chart = (
            alt.Chart(caption_rate_df)
            .mark_bar()
            .encode(
                x=alt.X("caption_rate_pct:Q", title="Caption Rate (%)"),
                y=alt.Y("category_name:N", sort="-x", title="Category"),
                color=alt.Color("trending_region:N", title="Region"),
                tooltip=[
                    "trending_region",
                    "category_name",
                    "captioned_videos",
                    "total_unique_videos",
                    alt.Tooltip("caption_rate_pct:Q", format=".1f"),
                ],
            )
            .properties(height=450)
        )
        st.altair_chart(caption_chart, use_container_width=True)


    st.subheader("Trending Rank Distribution by Category")
    st.markdown("Mode 1: Latest snapshot only. Out of the current trending slots, how many belong to each category?")

    if trending_rank_dist_df.empty:
        st.info("No latest snapshot distribution data available.")
    else:
        trending_dist_chart = (
            alt.Chart(trending_rank_dist_df)
            .mark_bar()
            .encode(
                x=alt.X("video_count:Q", title="Current Trending Video Count"),
                y=alt.Y("category_name:N", sort="-x", title="Category"),
                color=alt.Color("trending_region:N", title="Region"),
                tooltip=[
                    "trending_region",
                    "category_name",
                    "video_count",
                    alt.Tooltip("pct_of_trending:Q", title="% of Trending", format=".2f"),
                ],
            )
            .properties(height=450)
        )
        st.altair_chart(trending_dist_chart, use_container_width=True)


    st.subheader("Top Channels Leaderboard")
    st.markdown("Analytical question: Which channels are consistently generating reach and engagement?")

    if channel_board_df.empty:
        st.info("Channel leaderboard needs channel-level data.")
    else:
        channel_chart = (
            alt.Chart(channel_board_df)
            .mark_bar()
            .encode(
                x=alt.X("total_views:Q", title="Total Views"),
                y=alt.Y("channel_title:N", sort="-x", title="Channel"),
                color=alt.Color("avg_engagement_rate:Q", title="Avg Engagement Rate"),
                tooltip=[
                    "channel_title",
                    "videos",
                    "total_views",
                    "total_engagements",
                    alt.Tooltip("avg_engagement_rate:Q", format=".2%"),
                    "avg_subscribers",
                ],
            )
            .properties(height=450)
        )
        st.altair_chart(channel_chart, use_container_width=True)

    st.subheader("Top Trending Videos Leaderboard")
    st.markdown("Analytical question: Which videos are leading the current trending batch right now?")

    if top_videos_df.empty:
        st.info("No latest-batch leaderboard data available.")
    else:
        st.dataframe(
            top_videos_df.rename(
                columns={
                    "category_name": "Category",
                    "trending_region": "Region",
                    "trending_rank": "YouTube Rank",
                    "computed_rank": "Views Rank",
                    "view_count": "Views",
                    "like_count": "Likes",
                    "comment_count": "Comments",
                    "like_rate": "Like Rate",
                    "comment_rate": "Comment Rate",
                    "channel_title": "Channel",
                    "video_id": "Video ID",
                    "title": "Title",
                }
            ),
            use_container_width=True,
        )


with tab2:
    st.subheader("2. Why is it happening?")
    st.markdown("Diagnostic analytics explains the drivers behind trending behavior, content momentum, audience response, and market differences.")

    st.subheader("View Velocity")
    st.markdown("Mode 1: Latest snapshot. Which videos are gaining views fastest relative to age?")
    if velocity_df.empty:
        st.info("No velocity data available.")
    else:
        velocity_chart = (
            alt.Chart(velocity_df.head(30))
            .mark_bar()
            .encode(
                x=alt.X("views_per_hour:Q", title="Views per Hour"),
                y=alt.Y("title:N", sort="-x", title="Video"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "title",
                    "category_name",
                    "trending_region",
                    "view_count",
                    "video_age_hours",
                    "views_per_hour",
                    "category_avg_velocity",
                    "relative_velocity",
                    "trending_rank",
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(velocity_chart, use_container_width=True)

    st.subheader("Trending Persistence Tracking")
    st.markdown("Mode 3: Full time-series. Which videos stay in trending for the longest duration?")
    if persistence_df.empty:
        st.info("No persistence data available.")
    else:
        persistence_chart = (
            alt.Chart(persistence_df.head(30))
            .mark_bar()
            .encode(
                x=alt.X("batches_appeared:Q", title="Batches Appeared"),
                y=alt.Y("title:N", sort="-x", title="Video"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "title",
                    "channel_title",
                    "category_name",
                    "trending_region",
                    "batches_appeared",
                    "estimated_hours_trending",
                    "best_rank_achieved",
                    "peak_view_count",
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(persistence_chart, use_container_width=True)

    st.subheader("Rank Movement Analysis")
    st.markdown("Mode 3: Full time-series. How is trending rank changing over consecutive batches?")
    if rank_movement_df.empty:
        st.info("No rank movement data available.")
    else:
        rank_chart = (
            alt.Chart(rank_movement_df.head(500))
            .mark_line(point=True)
            .encode(
                x=alt.X("collected_at:T", title="Collected Time"),
                y=alt.Y("trending_rank:Q", title="Trending Rank", sort="descending"),
                color=alt.Color("title:N", title="Video"),
                tooltip=[
                    "title",
                    "category_name",
                    "trending_region",
                    "collected_at",
                    "trending_rank",
                    "prev_rank",
                    "rank_improvement",
                    "views_gained_this_batch",
                ],
            )
            .properties(height=450)
        )
        st.altair_chart(rank_chart, use_container_width=True)

    st.subheader("Engagement vs View Count Correlation")
    st.markdown("Latest snapshot scatter plus category-region correlation scores.")
    if corr_scatter_df.empty:
        st.info("No correlation data available.")
    else:
        corr_chart = (
            alt.Chart(corr_scatter_df)
            .mark_circle(opacity=0.7)
            .encode(
                x=alt.X("view_count:Q", title="View Count"),
                y=alt.Y("engagement_rate:Q", title="Engagement Rate"),
                color=alt.Color("category_name:N", title="Category"),
                size=alt.Size("channel_subscriber_count:Q", title="Subscriber Count"),
                tooltip=[
                    "title",
                    "category_name",
                    "trending_region",
                    "view_count",
                    "engagement_rate",
                    "channel_subscriber_count",
                ],
            )
            .properties(height=450)
        )
        st.altair_chart(corr_chart, use_container_width=True)

        if not corr_df.empty:
            st.dataframe(corr_df, use_container_width=True)

    st.subheader("Title Characteristics vs Performance")
    st.markdown("Mode 2: Deduplicated historical. Which title patterns are associated with stronger performance?")
    if title_char_df.empty:
        st.info("No title-characteristics data available.")
    else:
        title_chart = (
            alt.Chart(title_char_df)
            .mark_bar()
            .encode(
                x=alt.X("avg_er:Q", title="Average Engagement Rate"),
                y=alt.Y("title_length:N", title="Title Length"),
                color=alt.Color("caps_level:N", title="Caps Level"),
                column=alt.Column("title_has_question:N", title="Has Question"),
                tooltip=[
                    "category_name",
                    "trending_region",
                    "title_has_question",
                    "title_has_number",
                    "caps_level",
                    "title_length",
                    "avg_velocity",
                    "avg_er",
                    "sample_size",
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(title_chart, use_container_width=True)

    st.subheader("Tag Count vs Engagement")
    st.markdown("Mode 2: Deduplicated historical. Does having more tags relate to stronger engagement?")
    if tag_count_df.empty:
        st.info("No tag-count data available.")
    else:
        tag_chart = (
            alt.Chart(tag_count_df)
            .mark_bar()
            .encode(
                x=alt.X("tag_count_bucket:N", title="Tag Count Bucket"),
                y=alt.Y("avg_er:Q", title="Average Engagement Rate"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "category_name",
                    "tag_count_bucket",
                    "avg_er",
                    "avg_views",
                    "n_videos",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(tag_chart, use_container_width=True)

    st.subheader("Duration vs Engagement Relationship")
    st.markdown("Mode 2: Deduplicated historical. Which duration buckets perform best?")
    if duration_eng_df.empty:
        st.info("No duration-engagement data available.")
    else:
        duration_chart = (
            alt.Chart(duration_eng_df)
            .mark_bar()
            .encode(
                x=alt.X("duration_bucket:N", title="Duration Bucket"),
                y=alt.Y("avg_er:Q", title="Average Engagement Rate"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "category_name",
                    "trending_region",
                    "duration_bucket",
                    "avg_like_rate",
                    "avg_comment_rate",
                    "avg_er",
                    "avg_views",
                    "median_views",
                    "n_videos",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(duration_chart, use_container_width=True)

    st.subheader("Channel Size vs Trending Reach")
    st.markdown("Mode 2: Deduplicated historical. How does creator size relate to trending reach?")
    if channel_size_df.empty:
        st.info("No channel-size data available.")
    else:
        channel_size_chart = (
            alt.Chart(channel_size_df)
            .mark_bar()
            .encode(
                x=alt.X("subscriber_tier:N", title="Subscriber Tier"),
                y=alt.Y("avg_views:Q", title="Average Views"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "trending_region",
                    "category_name",
                    "subscriber_tier",
                    "avg_trending_rank",
                    "best_rank_achieved",
                    "avg_views",
                    "unique_videos",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(channel_size_chart, use_container_width=True)

    st.subheader("Regional Content Preference Divergence")
    st.markdown("Latest snapshot. Which regions over-index or under-index on certain categories versus the global mix?")
    if regional_div_df.empty:
        st.info("No regional divergence data available.")
    else:
        divergence_chart = (
            alt.Chart(regional_div_df)
            .mark_bar()
            .encode(
                x=alt.X("divergence_score:Q", title="Divergence Score"),
                y=alt.Y("category_name:N", title="Category"),
                color=alt.Color("trending_region:N", title="Region"),
                tooltip=[
                    "trending_region",
                    "category_name",
                    "regional_pct",
                    "global_pct",
                    "divergence_score",
                ],
            )
            .properties(height=450)
        )
        st.altair_chart(divergence_chart, use_container_width=True)

    st.subheader("Recency Bias Detection")
    st.markdown("Mode 2: Deduplicated historical. How old are videos when they first enter trending?")
    if recency_bias_df.empty:
        st.info("No recency-bias data available.")
    else:
        recency_chart = (
            alt.Chart(recency_bias_df)
            .mark_bar()
            .encode(
                x=alt.X("age_at_trending_entry:N", title="Age at Trending Entry"),
                y=alt.Y("video_count:Q", title="Video Count"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "category_name",
                    "trending_region",
                    "age_at_trending_entry",
                    "video_count",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(recency_chart, use_container_width=True)

    st.subheader("Weekend vs Weekday Trending Behavior")
    st.markdown("Mode 3: Full time-series. Are engagement and views different on weekends vs weekdays?")
    if weekend_weekday_df.empty:
        st.info("No weekend-weekday data available.")
    else:
        weekend_chart = (
            alt.Chart(weekend_weekday_df)
            .mark_bar()
            .encode(
                x=alt.X("day_type:N", title="Day Type"),
                y=alt.Y("avg_er:Q", title="Average Engagement Rate"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "category_name",
                    "trending_region",
                    "day_type",
                    "avg_view_count",
                    "avg_er",
                    "unique_videos_seen",
                    "total_observations",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(weekend_chart, use_container_width=True)


with tab3:
    st.subheader("3. What will happen?")
    st.markdown("Predictive analytics estimates which videos are most likely to keep trending, how their views may evolve, how long they may remain visible, and how category share may shift next.")

    st.subheader("Trending Entry Probability Model")
    st.markdown("Mode 3 training with latest-snapshot inference. This scores the current batch using next-batch survival as a practical probability proxy.")

    if entry_probability_df.empty:
        st.info("Not enough historical batches yet to score next-batch trending probability.")
    else:
        entry_chart = (
            alt.Chart(entry_probability_df)
            .mark_bar()
            .encode(
                x=alt.X("predicted_probability:Q", title="Predicted Next-Batch Probability"),
                y=alt.Y("title:N", sort="-x", title="Video"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "title",
                    "channel_title",
                    "category_name",
                    "trending_region",
                    "trending_rank",
                    "view_count",
                    "velocity",
                    alt.Tooltip("engagement_rate:Q", title="Engagement Rate", format=".2%"),
                    alt.Tooltip("predicted_probability:Q", title="Probability", format=".2%"),
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(entry_chart, use_container_width=True)

    st.subheader("View Count Forecasting")
    st.markdown("Mode 3 per-video forecasting. The graph projects future view counts for currently trending videos with a simple confidence band.")

    if video_forecast_df.empty:
        st.info("View forecasting needs at least 4 observations for a currently trending video.")
    else:
        actual_line = (
            alt.Chart(video_forecast_df[video_forecast_df["series"] == "Actual"])
            .mark_line(point=True)
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("forecast_views:Q", title="View Count"),
                color=alt.Color("title:N", title="Video"),
                tooltip=[
                    "title",
                    "category_name",
                    "trending_region",
                    "time_bucket",
                    alt.Tooltip("forecast_views:Q", title="Views", format=","),
                    "series",
                ],
            )
        )
        forecast_line = (
            alt.Chart(video_forecast_df[video_forecast_df["series"] == "Forecast"])
            .mark_line(point=True, strokeDash=[6, 4])
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("forecast_views:Q", title="Forecast Views"),
                color=alt.Color("title:N", legend=None),
                tooltip=[
                    "title",
                    "category_name",
                    "trending_region",
                    "time_bucket",
                    alt.Tooltip("forecast_views:Q", title="Forecast Views", format=","),
                    alt.Tooltip("lower_bound:Q", title="Lower Bound", format=","),
                    alt.Tooltip("upper_bound:Q", title="Upper Bound", format=","),
                    "series",
                ],
            )
        )
        confidence_band = (
            alt.Chart(video_forecast_df[video_forecast_df["series"] == "Forecast"])
            .mark_area(opacity=0.15)
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("lower_bound:Q", title="Lower Bound"),
                y2="upper_bound:Q",
                color=alt.Color("title:N", legend=None),
            )
        )
        st.altair_chart((confidence_band + actual_line + forecast_line).properties(height=450), use_container_width=True)

    st.subheader("Trending Duration Prediction")
    st.markdown("Historical episodes are used to estimate total trending lifespan and remaining time for the videos in the current batch.")

    if duration_prediction_df.empty:
        st.info("Trending duration prediction needs more accumulated history before it becomes stable.")
    else:
        duration_prediction_chart = (
            alt.Chart(duration_prediction_df)
            .mark_bar()
            .encode(
                x=alt.X("predicted_remaining_hours:Q", title="Predicted Remaining Hours"),
                y=alt.Y("title:N", sort="-x", title="Video"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "title",
                    "channel_title",
                    "category_name",
                    "trending_region",
                    "current_rank",
                    alt.Tooltip("current_hours:Q", title="Observed Hours", format=".2f"),
                    alt.Tooltip("predicted_total_hours:Q", title="Predicted Total Hours", format=".2f"),
                    alt.Tooltip("predicted_remaining_hours:Q", title="Predicted Remaining Hours", format=".2f"),
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(duration_prediction_chart, use_container_width=True)

    st.subheader("Peak Rank and Category Forecasting")
    st.markdown("These charts estimate the best future rank a current video may reach and the next category-share trajectory using the accumulated time-series history.")

    if peak_rank_forecast_df.empty:
        st.info("Peak-rank forecasting needs more historical trajectories first.")
    else:
        peak_rank_chart = (
            alt.Chart(peak_rank_forecast_df)
            .mark_bar()
            .encode(
                x=alt.X("expected_rank_gain:Q", title="Expected Rank Improvement"),
                y=alt.Y("title:N", sort="-x", title="Video"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "title",
                    "category_name",
                    "trending_region",
                    "current_rank",
                    alt.Tooltip("predicted_peak_rank:Q", title="Predicted Peak Rank", format=".1f"),
                    alt.Tooltip("expected_rank_gain:Q", title="Expected Rank Gain", format=".1f"),
                    alt.Tooltip("avg_rank_delta:Q", title="Avg Rank Delta", format=".2f"),
                    alt.Tooltip("current_velocity:Q", title="Current Velocity", format=".2f"),
                ],
            )
            .properties(height=450)
        )
        st.altair_chart(peak_rank_chart, use_container_width=True)

    if category_share_forecast_df.empty:
        st.info("Category-share forecasting needs at least 2 time buckets per category.")
    else:
        category_share_chart = (
            alt.Chart(category_share_forecast_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("forecast_share:Q", title="Forecast Category Share"),
                color=alt.Color("category_name:N", title="Category"),
                strokeDash=alt.StrokeDash("series:N", title="Series"),
                tooltip=[
                    "category_name",
                    "series",
                    "time_bucket",
                    alt.Tooltip("forecast_share:Q", title="Category Share", format=".2%"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(category_share_chart, use_container_width=True)

with tab4:
    st.subheader("4. What should be done?")
    st.markdown("Prescriptive analytics converts the historical and current signals into recommended actions: when to post, where gaps exist, who to partner with, what content format to favor, when to intervene, and which regions to expand into.")

    st.subheader("Optimal Posting Window")
    st.markdown("Mode 2: Deduplicated historical. This ranks day-hour slots by average peak views with sample-size awareness.")
    if optimal_posting_df.empty:
        st.info("No posting-window recommendation data available.")
    else:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        posting_heatmap_df = optimal_posting_df.copy()
        posting_heatmap_df["publish_day"] = pd.Categorical(posting_heatmap_df["publish_day"], categories=day_order, ordered=True)

        posting_heatmap = (
            alt.Chart(posting_heatmap_df)
            .mark_rect()
            .encode(
                x=alt.X("publish_hour_utc:O", title="Publish Hour (UTC)"),
                y=alt.Y("publish_day:O", title="Publish Day"),
                color=alt.Color("avg_peak_views:Q", title="Avg Peak Views"),
                tooltip=[
                    "category_name",
                    "trending_region",
                    "publish_day",
                    "publish_hour_utc",
                    alt.Tooltip("avg_peak_views:Q", title="Avg Peak Views", format=","),
                    "sample_size",
                    "slot_rank",
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(posting_heatmap, use_container_width=True)

        best_slots = optimal_posting_df[optimal_posting_df["slot_rank"] == 1].copy()
        if not best_slots.empty:
            st.dataframe(best_slots, use_container_width=True)

    st.subheader("Trending Gap Opportunity Detector")
    st.markdown("Latest snapshot versus 7-day baseline. Categories with the biggest deficit are immediate opportunity spaces.")
    if gap_opportunity_df.empty:
        st.info("No trending-gap opportunity data available.")
    else:
        gap_chart = (
            alt.Chart(gap_opportunity_df)
            .mark_bar()
            .encode(
                x=alt.X("gap_z_score:Q", title="Gap Z-Score"),
                y=alt.Y("category_name:N", sort="-x", title="Category"),
                color=alt.Color("status:N", title="Status"),
                tooltip=[
                    "category_name",
                    "trending_region",
                    "current_count",
                    "avg_count",
                    alt.Tooltip("gap_z_score:Q", title="Gap Z-Score", format=".2f"),
                    "status",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(gap_chart, use_container_width=True)

    st.subheader("Creator Partnership Recommendation Engine")
    st.markdown("Mode 2: Deduplicated historical. This surfaces channels with repeat trending success and strong partnership fit.")
    if creator_partnership_df.empty:
        st.info("No creator partnership data available.")
    else:
        partner_chart = (
            alt.Chart(creator_partnership_df.head(25))
            .mark_bar()
            .encode(
                x=alt.X("partnership_score:Q", title="Partnership Score"),
                y=alt.Y("channel_title:N", sort="-x", title="Channel"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "channel_title",
                    "trending_region",
                    "category_name",
                    "trending_video_count",
                    alt.Tooltip("avg_er:Q", title="Avg Engagement Rate", format=".2%"),
                    alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                    alt.Tooltip("partnership_score:Q", title="Partnership Score", format=".2f"),
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(partner_chart, use_container_width=True)

    st.subheader("Format Prescriptions")
    st.markdown("Mode 2: Deduplicated historical. These patterns recommend what kind of packaging performs best by category.")
    if format_prescriptions_df.empty:
        st.info("No format-prescription data available.")
    else:
        format_chart = (
            alt.Chart(format_prescriptions_df.head(40))
            .mark_bar()
            .encode(
                x=alt.X("prescription_score:Q", title="Prescription Score"),
                y=alt.Y("feature_value:N", sort="-x", title="Recommended Feature"),
                color=alt.Color("feature_type:N", title="Feature Type"),
                tooltip=[
                    "category_name",
                    "feature_type",
                    "feature_value",
                    alt.Tooltip("avg_er:Q", title="Avg Engagement Rate", format=".2%"),
                    alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                    "sample_size",
                    alt.Tooltip("prescription_score:Q", title="Prescription Score", format=".2f"),
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(format_chart, use_container_width=True)

    st.subheader("Campaign Timing Alerts")
    st.markdown("Mode 3: Full time-series. This flags videos that should be boosted now, watched closely, or are losing momentum.")
    if campaign_alerts_df.empty:
        st.info("Campaign timing alerts need at least 3 consecutive observations per video.")
    else:
        alert_chart = (
            alt.Chart(campaign_alerts_df.head(30))
            .mark_bar()
            .encode(
                x=alt.X("total_rank_gain:Q", title="3-Batch Rank Gain"),
                y=alt.Y("title:N", sort="-x", title="Video"),
                color=alt.Color("alert_type:N", title="Alert Type"),
                tooltip=[
                    "title",
                    "category_name",
                    "trending_region",
                    "current_rank",
                    alt.Tooltip("avg_rank_delta:Q", title="Avg Rank Delta", format=".2f"),
                    "total_rank_gain",
                    alt.Tooltip("latest_views:Q", title="Latest Views", format=","),
                    "alert_type",
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(alert_chart, use_container_width=True)

    st.subheader("Regional Expansion Recommendations")
    st.markdown("Mode 2: Deduplicated historical. This estimates which target regions are the strongest next expansion markets.")
    if regional_expansion_df.empty:
        st.info("No regional expansion recommendation data available.")
    else:
        expansion_chart = (
            alt.Chart(regional_expansion_df.head(40))
            .mark_circle(size=180)
            .encode(
                x=alt.X("source_region:N", title="Source Region"),
                y=alt.Y("target_region:N", title="Target Region"),
                size=alt.Size("expansion_probability:Q", title="Expansion Probability"),
                color=alt.Color("category_name:N", title="Category"),
                tooltip=[
                    "category_name",
                    "source_region",
                    "target_region",
                    "shared_videos",
                    "source_videos",
                    alt.Tooltip("expansion_probability:Q", title="Expansion Probability", format=".2%"),
                ],
            )
            .properties(height=500)
        )
        st.altair_chart(expansion_chart, use_container_width=True)

    st.subheader("Decision Summary")
    if not summary_df.empty:
        best_views_category = summary_df.iloc[0]["category_name"]
        best_engagement_category = summary_df.sort_values("avg_engagement_rate", ascending=False).iloc[0]["category_name"]
        st.write(f"- Best category to scale for reach: **{best_views_category}**")
        st.write(f"- Best category to scale for engagement quality: **{best_engagement_category}**")
        if not gap_opportunity_df.empty:
            top_gap = gap_opportunity_df.iloc[0]
            st.write(
                f"- Biggest current whitespace: **{top_gap['category_name']}** in **{top_gap['trending_region']}** "
                f"(gap z-score {top_gap['gap_z_score']:.2f})"
            )
        if not optimal_posting_df.empty:
            top_slot = optimal_posting_df.sort_values(["slot_rank", "avg_peak_views"]).iloc[0]
            st.write(
                f"- Best observed posting slot in current filter: **{top_slot['publish_day']} at {int(top_slot['publish_hour_utc']):02d}:00 UTC**"
            )

with st.expander("Raw Data"):
    st.dataframe(df, use_container_width=True)

import time

time.sleep(10)
st.rerun()
