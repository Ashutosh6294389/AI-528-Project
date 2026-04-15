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
)

DELTA_PATH = "storage/delta_tables/youtube"

st.set_page_config(page_title="Business-Ready YouTube Analytics", layout="wide")
st.title("YouTube Business Analytics Dashboard")
st.caption("Descriptive, diagnostic, predictive, and prescriptive analytics for trending YouTube content")


@st.cache_resource(show_spinner=False)
def get_spark():
    return (
        SparkSession.builder.appName("YouTubeAnalyticsDashboard")
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


@st.cache_data(ttl=60, show_spinner=False)
def load_data():
    try:
        if not os.path.exists(DELTA_PATH):
            return pd.DataFrame()

        spark = get_spark()
        sdf = spark.read.format("delta").load(DELTA_PATH)

        if sdf.rdd.isEmpty():
            return pd.DataFrame()

        return sdf.toPandas()
    except Exception as exc:
        st.error(f"Could not load Delta data: {exc}")
        return pd.DataFrame()


raw_df = load_data()
df = prepare_dashboard_df(raw_df)

if df.empty:
    st.warning("No data available. Run the producer and Spark streaming first.")
    st.stop()

st.sidebar.header("Filters")

category_options = ["All"] + sorted(df["category"].dropna().astype(str).unique().tolist())
region_options = ["All"] + sorted(df["region"].dropna().astype(str).unique().tolist())

selected_category = st.sidebar.selectbox("Category", category_options)
selected_region = st.sidebar.selectbox("Region", region_options)

if selected_category != "All":
    df = df[df["category"] == selected_category]

if selected_region != "All":
    df = df[df["region"] == selected_region]

if df.empty:
    st.warning("No data matched the selected filters.")
    st.stop()

summary_df = build_category_summary(df)
top_videos_df = build_top_videos(df)
diagnostic_df = build_diagnostic_table(df)
forecast_df = build_forecast(df)
recommendations_df = build_recommendations(df)

views_ts_df = build_views_timeseries(df)
region_ts_df = build_region_timeseries(df)
publish_heatmap_df = build_publish_hour_heatmap(df)
category_share_df = build_category_share_over_time(df)
channel_board_df = build_channel_leaderboard(df)
bubble_df = build_bubble_dataset(df)
outlier_df = build_outlier_videos(df)
growth_df = build_category_growth(df)

latest_records = df.sort_values("fetched_at").drop_duplicates(subset=["video_id"], keep="last")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Unique Videos", f"{latest_records['video_id'].nunique():,}")
k2.metric("Total Views", f"{int(latest_records['views'].sum()):,}")
avg_eng = pd.to_numeric(latest_records["engagement_rate"], errors="coerce")
avg_eng = avg_eng.replace([float("inf"), -float("inf")], pd.NA).dropna()
avg_eng_value = 0 if avg_eng.empty else avg_eng.mean()

k3.metric("Avg Engagement Rate", f"{avg_eng_value * 100:.2f}%")

k4.metric("Tracked Categories", f"{latest_records['category'].nunique():,}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
)

with tab1:
    st.subheader("1. What is happening?")
    st.markdown("Analytical question: Which categories, channels, and videos are driving the most business value?")

    category_chart = (
        alt.Chart(summary_df)
        .mark_bar()
        .encode(
            x=alt.X("total_views:Q", title="Total Views"),
            y=alt.Y("category:N", sort="-x", title="Category"),
            tooltip=[
                "category",
                "videos",
                "total_views",
                "total_likes",
                "total_comments",
                alt.Tooltip("avg_engagement_rate:Q", format=".2%"),
            ],
            color=alt.Color("avg_engagement_rate:Q", title="Avg Engagement Rate"),
        )
        .properties(height=350)
    )
    st.altair_chart(category_chart, width="stretch")

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
                color=alt.Color("category:N", title="Category"),
                tooltip=[
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("time_bucket:T", title="Time"),
                    alt.Tooltip("total_views:Q", title="Total Views", format=","),
                    alt.Tooltip("total_engagements:Q", title="Total Engagements", format=","),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(views_line, width="stretch")


    st.subheader("Category Share of Total Views Over Time")
    st.markdown("Analytical question: How is category dominance changing over time?")

    if category_share_df.empty:
        st.info("Category share trend needs time-based data.")
    else:
        area_chart = (
            alt.Chart(category_share_df)
            .mark_area()
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("view_share:Q", stack="normalize", title="Share of Views"),
                color=alt.Color("category:N", title="Category"),
                tooltip=[
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("time_bucket:T", title="Time"),
                    alt.Tooltip("total_views:Q", title="Total Views", format=","),
                    alt.Tooltip("view_share:Q", title="View Share", format=".2%"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(area_chart, width="stretch")



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
                ],
            )
            .properties(height=450)
        )
        st.altair_chart(channel_chart, width="stretch")

    st.subheader("Top Videos")
    st.dataframe(top_videos_df, width="stretch")

with tab2:
    st.subheader("2. Why is it happening?")
    st.markdown("Analytical question: Which categories are efficient, and which videos are underperforming versus their category baseline?")

    scatter = (
        alt.Chart(summary_df)
        .mark_circle(size=180)
        .encode(
            x=alt.X("total_views:Q", title="Total Views"),
            y=alt.Y("avg_engagement_rate:Q", title="Average Engagement Rate"),
            tooltip=[
                "category",
                "videos",
                "total_views",
                alt.Tooltip("avg_engagement_rate:Q", format=".2%"),
            ],
            color=alt.Color("category:N", legend=None),
        )
        .properties(height=350)
    )
    st.altair_chart(scatter, width="stretch")

    st.subheader("Regional Engagement Trend Over Time")
    st.markdown("Analytical question: Which markets are improving or weakening over time?")

    if region_ts_df.empty:
        st.info("Region-based time trend requires enriched records with fetched timestamps.")
    else:
        region_line = (
            alt.Chart(region_ts_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("avg_engagement_rate:Q", title="Avg Engagement Rate"),
                color=alt.Color("region:N", title="Region"),
                tooltip=[
                    alt.Tooltip("region:N", title="Region"),
                    alt.Tooltip("time_bucket:T", title="Time"),
                    alt.Tooltip("total_views:Q", title="Total Views", format=","),
                    alt.Tooltip("avg_engagement_rate:Q", title="Avg Engagement Rate", format=".2%"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(region_line, width="stretch")


    st.subheader("Best Publishing Windows Heatmap")
    st.markdown("Analytical question: When are videos most likely to perform well?")

    if publish_heatmap_df.empty:
        st.info("Publishing-time analysis requires `published_at` data.")
    else:
        heatmap = (
            alt.Chart(publish_heatmap_df)
            .mark_rect()
            .encode(
                x=alt.X("publish_hour:O", title="Hour of Day"),
                y=alt.Y("publish_day:O", title="Day of Week"),
                color=alt.Color("avg_views:Q", title="Avg Views"),
                tooltip=[
                    "publish_day",
                    "publish_hour",
                    "videos",
                    "avg_views",
                    alt.Tooltip("avg_engagement_rate:Q", format=".2%"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(heatmap, width="stretch")

    st.subheader("Views vs Engagement Bubble Chart")
    st.markdown("Analytical question: Which videos combine high reach, strong engagement, and active discussion?")

    if bubble_df.empty:
        st.info("Bubble chart needs latest per-video records.")
    else:
        bubble_chart = (
            alt.Chart(bubble_df)
            .mark_circle(opacity=0.7)
            .encode(
                x=alt.X("views:Q", title="Views"),
                y=alt.Y("engagement_rate:Q", title="Engagement Rate"),
                size=alt.Size("bubble_size:Q", title="Comments"),
                color=alt.Color("category:N", title="Category"),
                tooltip=[
                    "title",
                    "channel_title",
                    "category",
                    "region",
                    "views",
                    alt.Tooltip("engagement_rate:Q", format=".2%"),
                    "bubble_size",
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(bubble_chart, width="stretch")

    st.subheader("Underperforming Videos")
    st.dataframe(diagnostic_df, width="stretch")

    st.subheader("Viral Outlier Videos")
    st.markdown("Analytical question: Which videos are outperforming the rest of the market?")

    if outlier_df.empty:
        st.info("No outlier candidates found yet.")
    else:
        st.dataframe(outlier_df, width="stretch")

with tab3:
    st.subheader("3. What will happen?")
    st.markdown("Analytical question: Based on recent snapshots, which categories are likely to keep growing?")

    if forecast_df.empty:
        st.info("Forecast needs at least 2 time snapshots per category. It becomes useful after rerunning the enriched producer.")
    else:
        if pd.api.types.is_datetime64_any_dtype(forecast_df["time_bucket"]):
            forecast_x = alt.X("time_bucket:T", title="Time")
        else:
            forecast_x = alt.X("time_bucket:N", title="Step")

        forecast_chart = (
            alt.Chart(forecast_df)
            .mark_line(point=True)
            .encode(
                x=forecast_x,
                y=alt.Y("total_views:Q", title="Views"),
                color=alt.Color("category:N", title="Category"),
                strokeDash="series:N",
                tooltip=["category", "series", "time_bucket", "total_views"],
            )
            .properties(height=380)
        )
        st.altair_chart(forecast_chart, width="stretch")

    st.subheader("Category Growth Rate Over Time")
    st.markdown("Analytical question: Which categories are accelerating fastest?")

    if growth_df.empty:
        st.info("Growth-rate chart needs multiple time snapshots.")
    else:
        growth_chart = (
            alt.Chart(growth_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("growth_rate:Q", title="Growth Rate"),
                color=alt.Color("category:N", title="Category"),
                tooltip=[
                    alt.Tooltip("category:N", title="Category"),
                    alt.Tooltip("time_bucket:T", title="Time"),
                    alt.Tooltip("total_views:Q", title="Total Views", format=","),
                    alt.Tooltip("growth_rate:Q", title="Growth Rate", format=".2%"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(growth_chart, width="stretch")


with tab4:
    st.subheader("4. What should be done?")
    st.markdown("Analytical question: Which actions should a content or marketing team take next?")

    st.dataframe(recommendations_df, width="stretch")

    st.subheader("Business Summary")
    if not summary_df.empty:
        best_views_category = summary_df.iloc[0]["category"]
        best_engagement_category = summary_df.sort_values(
            "avg_engagement_rate", ascending=False
        ).iloc[0]["category"]

        st.write(f"- Best-performing category by views: **{best_views_category}**")
        st.write(f"- Strongest average engagement: **{best_engagement_category}**")
        st.write("- Use this section in your presentation as the prescriptive analytics part.")

with st.expander("Raw Data"):
    st.dataframe(df, width="stretch")



import time

time.sleep(30)
st.rerun()

