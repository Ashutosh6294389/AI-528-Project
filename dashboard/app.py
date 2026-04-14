category_map = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "18": "Short Movies",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism"
}

import streamlit as st
import os

import pandas as pd
import time
from pyspark.sql import SparkSession

st.set_page_config(page_title="YouTube Analytics", layout="wide")

st.title("📊 YouTube Analytics Dashboard")


DELTA_PATH = "storage/delta_tables/youtube"

# -------------------------------
# Load Data from Delta Lake
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_spark():
    return SparkSession.builder \
        .appName("YouTubeAnalyticsDashboard") \
        .master("local[*]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
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

def load_data():
    try:
        spark = get_spark()
        if not os.path.exists(DELTA_PATH):
            return pd.DataFrame()
        sdf = spark.read.format("delta").load(DELTA_PATH)
        if sdf.rdd.isEmpty():
            return pd.DataFrame()
        return sdf.toPandas()
    except Exception as e:
        st.error(f"Error loading Delta table: {e}")
        return pd.DataFrame()

df = load_data()

# -------------------------------
# If no data
# -------------------------------
if df.empty:
    st.warning("⚠️ No data available yet. Please run Spark streaming.")
else:
    for col in ["views", "likes", "comments"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Engagement Ratio
    df["engagement"] = df["likes"] / df["views"]

    # -------------------------------
    # Sidebar
    # -------------------------------
    st.sidebar.header("Filters")
    # Map all categories to readable names for the filter
    sidebar_categories = []
    for cat in df["category"].dropna().unique():
        cat_str = str(cat).strip()
        cat_str = category_map.get(cat_str, cat_str)
        if cat_str.lower() not in ["nan", "none", ""] and cat_str not in sidebar_categories:
            sidebar_categories.append(cat_str)
    selected_category = st.sidebar.selectbox("Select Category", ["All"] + sidebar_categories)

    if selected_category != "All":
        mapped_cats = df["category"].apply(lambda x: category_map.get(str(x).strip(), str(x).strip()))
        df = df[mapped_cats == selected_category]

    # -------------------------------
    # Metrics
    # -------------------------------
    st.subheader("📌 Key Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Views", int(df["views"].sum()))
    col2.metric("Total Likes", int(df["likes"].sum()))
    col3.metric("Total Comments", int(df["comments"].sum()))

    # -------------------------------
    # Charts (FIXED)
    # -------------------------------


    st.subheader("📊 Views per Category")

    # Convert all categories to string and map codes to names for all graphs
    categories = list(df["category"])
    views = list(df["views"])
    likes = list(df["likes"])
    comments = list(df["comments"])
    engagements = list(df["engagement"])

    # Views per Category
    cleaned = []
    for cat, v in zip(categories, views):
        cat_str = str(cat).strip()
        cat_str = category_map.get(cat_str, cat_str)
        if cat_str.lower() not in ["nan", "none", ""]:
            cleaned.append((cat_str, v))
    from collections import defaultdict
    agg = defaultdict(int)
    for cat, v in cleaned:
        try:
            v_int = int(v)
        except:
            v_int = 0
        agg[cat] += v_int
    sorted_items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    import pandas as pd
    chart_df = pd.DataFrame(sorted_items, columns=["category", "views"])
    chart_df.set_index("category", inplace=True)
    st.bar_chart(chart_df)

    # Likes per Category
    cleaned_likes = []
    for cat, l in zip(categories, likes):
        cat_str = str(cat).strip()
        cat_str = category_map.get(cat_str, cat_str)
        if cat_str.lower() not in ["nan", "none", ""]:
            cleaned_likes.append((cat_str, l))
    agg_likes = defaultdict(int)
    for cat, l in cleaned_likes:
        try:
            l_int = int(l)
        except:
            l_int = 0
        agg_likes[cat] += l_int
    sorted_likes = sorted(agg_likes.items(), key=lambda x: x[1], reverse=True)
    likes_df = pd.DataFrame(sorted_likes, columns=["category", "likes"])
    likes_df.set_index("category", inplace=True)
    # Example: st.bar_chart(likes_df)  # Uncomment to show likes per category

    # Comments per Category
    cleaned_comments = []
    for cat, c in zip(categories, comments):
        cat_str = str(cat).strip()
        cat_str = category_map.get(cat_str, cat_str)
        if cat_str.lower() not in ["nan", "none", ""]:
            cleaned_comments.append((cat_str, c))
    agg_comments = defaultdict(int)
    for cat, c in cleaned_comments:
        try:
            c_int = int(c)
        except:
            c_int = 0
        agg_comments[cat] += c_int
    sorted_comments = sorted(agg_comments.items(), key=lambda x: x[1], reverse=True)
    comments_df = pd.DataFrame(sorted_comments, columns=["category", "comments"])
    comments_df.set_index("category", inplace=True)
    # Example: st.bar_chart(comments_df)  # Uncomment to show comments per category


    st.subheader("❤️ Likes vs Views (Sampled)")
    # Map category for sampled data as well
    sample_df = df.sample(min(200, len(df))).copy()
    sample_df["category"] = sample_df["category"].apply(lambda x: category_map.get(str(x).strip(), str(x).strip()))
    st.scatter_chart(sample_df[["views", "likes"]])


    st.subheader("💬 Comments Spike Detection")
    # Map category for spike detection if needed (not used in chart, but for consistency)
    df["category"] = df["category"].apply(lambda x: category_map.get(str(x).strip(), str(x).strip()))
    df["comment_change"] = df["comments"].diff().fillna(0)
    df["spike"] = df["comment_change"].apply(lambda x: x if x > 1000 else 0)
    st.bar_chart(df["spike"].tail(200))



    st.subheader("🔥 Engagement by Category")
    # Aggregate engagement by mapped category name (pure Python)
    cleaned_eng = []
    for cat, eng in zip(categories, engagements):
        cat_str = str(cat).strip()
        cat_str = category_map.get(cat_str, cat_str)
        if cat_str.lower() not in ["nan", "none", ""]:
            try:
                eng_val = float(eng)
            except:
                eng_val = 0.0
            cleaned_eng.append((cat_str, eng_val))
    eng_sum = defaultdict(float)
    eng_count = defaultdict(int)
    for cat, eng in cleaned_eng:
        eng_sum[cat] += eng
        eng_count[cat] += 1
    eng_avg = {cat: (eng_sum[cat] / eng_count[cat]) if eng_count[cat] > 0 else 0.0 for cat in eng_sum}
    sorted_eng = sorted(eng_avg.items(), key=lambda x: x[1], reverse=True)
    eng_df = pd.DataFrame(sorted_eng, columns=["category", "engagement"])
    eng_df.set_index("category", inplace=True)
    st.bar_chart(eng_df)

    # -------------------------------
    # Raw Data
    # -------------------------------
    st.subheader("📄 Raw Data")
    st.dataframe(df)

# -------------------------------
# Auto Refresh
# -------------------------------
time.sleep(10)
st.rerun()