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

latest_records = df.sort_values("fetched_at").drop_duplicates(subset=["video_id"], keep="last")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Unique Videos", f"{latest_records['video_id'].nunique():,}")
k2.metric("Total Views", f"{int(latest_records['views'].sum()):,}")
k3.metric("Avg Engagement Rate", f"{latest_records['engagement_rate'].mean() * 100:.2f}%")
k4.metric("Tracked Categories", f"{latest_records['category'].nunique():,}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
)

with tab1:
    st.subheader("1. What is happening?")
    st.markdown("Analytical question: Which categories and videos are driving the most business value?")

    category_chart = (
        alt.Chart(summary_df)
        .mark_bar()
        .encode(
            x=alt.X("total_views:Q", title="Total Views"),
            y=alt.Y("category:N", sort="-x", title="Category"),
            tooltip=["category", "videos", "total_views", "total_likes", "total_comments", alt.Tooltip("avg_engagement_rate:Q", format=".2%")],
            color=alt.Color("avg_engagement_rate:Q", title="Avg Engagement Rate"),
        )
        .properties(height=350)
    )
    st.altair_chart(category_chart, use_container_width=True)

    st.subheader("Top Videos")
    st.dataframe(top_videos_df, use_container_width=True)

with tab2:
    st.subheader("2. Why is it happening?")
    st.markdown("Analytical question: Which categories are efficient, and which videos are underperforming versus their category baseline?")

    scatter = (
        alt.Chart(summary_df)
        .mark_circle(size=180)
        .encode(
            x=alt.X("total_views:Q", title="Total Views"),
            y=alt.Y("avg_engagement_rate:Q", title="Average Engagement Rate"),
            tooltip=["category", "videos", "total_views", alt.Tooltip("avg_engagement_rate:Q", format=".2%")],
            color=alt.Color("category:N", legend=None),
        )
        .properties(height=350)
    )
    st.altair_chart(scatter, use_container_width=True)

    st.subheader("Underperforming Videos")
    st.dataframe(diagnostic_df, use_container_width=True)

with tab3:
    st.subheader("3. What will happen?")
    st.markdown("Analytical question: Based on recent snapshots, which categories are likely to keep growing?")

    if forecast_df.empty:
        st.info("Forecast needs at least 2 time snapshots per category. It becomes useful after rerunning the enriched producer.")
    else:
        forecast_chart = (
            alt.Chart(forecast_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_bucket:T", title="Time") if pd.api.types.is_datetime64_any_dtype(forecast_df["time_bucket"]) else alt.X("time_bucket:N", title="Step"),
                y=alt.Y("total_views:Q", title="Views"),
                color="category:N",
                strokeDash="series:N",
                tooltip=["category", "series", "time_bucket", "total_views"],
            )
            .properties(height=380)
        )
        st.altair_chart(forecast_chart, use_container_width=True)

with tab4:
    st.subheader("4. What should be done?")
    st.markdown("Analytical question: Which actions should a content or marketing team take next?")

    st.dataframe(recommendations_df, use_container_width=True)

    st.subheader("Business Summary")
    st.write(
        f"- Best-performing category by views: **{summary_df.iloc[0]['category']}**"
    )
    st.write(
        f"- Strongest average engagement: **{summary_df.sort_values('avg_engagement_rate', ascending=False).iloc[0]['category']}**"
    )
    st.write(
        "- Use this section in your presentation as the prescriptive analytics part."
    )

with st.expander("Raw Data"):
    st.dataframe(df, use_container_width=True)
