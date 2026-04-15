import pandas as pd
from sklearn.linear_model import LinearRegression


def prepare_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    defaults = {
        "timestamp": None,
        "surface": "mostPopular",
        "page_number": 1,
        "rank": None,
        "region": "IN",
        "category": "Other",
        "category_id": None,
        "video_id": None,
        "channel_id": None,
        "channel_title": "Unknown",
        "title": "",
        "views": 0,
        "likes": 0,
        "comments": 0,
        "publish_time": None,
        "engagements": 0,
        "engagement_rate": 0.0,
        "like_rate": 0.0,
        "comment_rate": 0.0,
    }

    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value

    numeric_cols = [
        "views",
        "likes",
        "comments",
        "engagements",
        "engagement_rate",
        "like_rate",
        "comment_rate",
        "page_number",
        "rank",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["views"] = df["views"].fillna(0)
    df["likes"] = df["likes"].fillna(0)
    df["comments"] = df["comments"].fillna(0)
    df["engagements"] = df["engagements"].fillna(df["likes"] + df["comments"])

    safe_views = df["views"].replace(0, pd.NA)

    df["like_rate"] = (df["likes"] / safe_views).fillna(0)
    df["comment_rate"] = (df["comments"] / safe_views).fillna(0)
    df["engagement_rate"] = (df["engagements"] / safe_views).fillna(0)

    df["like_rate"] = df["like_rate"].replace([float("inf"), -float("inf")], 0)
    df["comment_rate"] = df["comment_rate"].replace([float("inf"), -float("inf")], 0)
    df["engagement_rate"] = df["engagement_rate"].replace([float("inf"), -float("inf")], 0)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)

    df["category"] = df["category"].fillna("Other").astype(str).str.strip()
    df["surface"] = df["surface"].fillna("mostPopular").astype(str)

    if df["video_id"].isna().all():
        df["video_id"] = (
            df["title"]
            .fillna("unknown")
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )

    return df


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = build_latest_snapshot_df(df)
    work["category"] = work["category"].fillna("Other").astype(str).str.strip()

    summary = (
        work.groupby("category", dropna=False)
        .agg(
            videos=("video_id", "nunique"),
            total_views=("views", "sum"),
            total_likes=("likes", "sum"),
            total_comments=("comments", "sum"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_like_rate=("like_rate", "mean"),
        )
        .reset_index()
        .sort_values(["total_views", "avg_engagement_rate"], ascending=[False, False])
    )

    return summary



def build_top_videos(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    top = (
        df.sort_values(["timestamp", "views", "engagement_rate"], ascending=[True, False, False])
        .drop_duplicates(subset=["video_id", "region"], keep="last")
        .sort_values(["views", "engagement_rate"], ascending=[False, False])
        .loc[:, ["title", "channel_title", "category", "region", "views", "likes", "comments", "engagement_rate", "rank"]]
        .head(top_n)
    )
    return top


def build_diagnostic_table(df: pd.DataFrame) -> pd.DataFrame:
    category_baseline = (
        df.groupby("category")
        .agg(
            category_avg_views=("views", "mean"),
            category_avg_engagement_rate=("engagement_rate", "mean"),
        )
        .reset_index()
    )

    latest = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["video_id", "region"], keep="last")
        .merge(category_baseline, on="category", how="left")
    )

    latest["engagement_gap"] = latest["engagement_rate"] - latest["category_avg_engagement_rate"]
    latest["view_gap"] = latest["views"] - latest["category_avg_views"]

    diagnostic = latest.loc[
        :,
        [
            "title",
            "channel_title",
            "category",
            "region",
            "views",
            "engagement_rate",
            "category_avg_engagement_rate",
            "engagement_gap",
            "view_gap",
            "rank",
        ],
    ].sort_values(["engagement_gap", "view_gap"], ascending=[True, True])

    return diagnostic.head(20)


def build_forecast(df: pd.DataFrame, top_n_categories: int = 5, horizon: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()

    if work["timestamp"].notna().any():
        work["time_bucket"] = work["timestamp"].dt.floor("5min")
    else:
        work = work.reset_index(drop=True)
        work["time_bucket"] = work.index.astype(str)

    top_categories = (
        work.groupby("category")["views"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_categories)
        .index.tolist()
    )

    work = work[work["category"].isin(top_categories)]

    ts = (
        work.groupby(["category", "time_bucket"], as_index=False)
        .agg(total_views=("views", "sum"))
        .sort_values(["category", "time_bucket"])
    )

    forecast_rows = []

    for category, group in ts.groupby("category"):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        group["step"] = range(len(group))
        X_train = group[["step"]]
        y_train = group["total_views"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        for _, row in group.iterrows():
            forecast_rows.append(
                {
                    "category": category,
                    "time_bucket": row["time_bucket"],
                    "total_views": row["total_views"],
                    "series": "Actual",
                }
            )

        last_step = int(group["step"].max())
        last_time = group["time_bucket"].iloc[-1]

        for i in range(1, horizon + 1):
            future_step = pd.DataFrame({"step": [last_step + i]})
            pred = max(0, float(model.predict(future_step)[0]))

            if isinstance(last_time, pd.Timestamp):
                future_bucket = last_time + pd.Timedelta(minutes=5 * i)
            else:
                future_bucket = f"Forecast {i}"

            forecast_rows.append(
                {
                    "category": category,
                    "time_bucket": future_bucket,
                    "total_views": pred,
                    "series": "Forecast",
                }
            )

    return pd.DataFrame(forecast_rows)


def build_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    summary = build_category_summary(df)
    if summary.empty:
        return pd.DataFrame(columns=["priority", "recommendation", "why"])

    high_views = summary["total_views"].median()
    high_eng = summary["avg_engagement_rate"].median()

    actions = []

    for _, row in summary.iterrows():
        category = row["category"]

        if row["total_views"] >= high_views and row["avg_engagement_rate"] < high_eng:
            actions.append(
                {
                    "priority": "High",
                    "recommendation": f"Improve CTA and thumbnail strategy for {category}",
                    "why": "This category gets strong reach but converts weakly on engagement.",
                }
            )

        if row["total_views"] < high_views and row["avg_engagement_rate"] >= high_eng:
            actions.append(
                {
                    "priority": "Medium",
                    "recommendation": f"Promote more {category} content",
                    "why": "This category engages well and may scale efficiently with more distribution.",
                }
            )

        if row["avg_like_rate"] >= summary["avg_like_rate"].quantile(0.75):
            actions.append(
                {
                    "priority": "Medium",
                    "recommendation": f"Reuse winning creative patterns from {category}",
                    "why": "The category shows above-average like efficiency and is likely resonating with viewers.",
                }
            )

    if not actions:
        actions.append(
            {
                "priority": "Medium",
                "recommendation": "Keep collecting more snapshots before making major content decisions",
                "why": "The current sample is too balanced to generate a strong recommendation confidently.",
            }
        )

    return pd.DataFrame(actions).drop_duplicates()


def build_views_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work = work.dropna(subset=["timestamp"])
    work["time_bucket"] = work["timestamp"].dt.floor("5min")

    ts = (
        work.groupby(["category", "time_bucket"], as_index=False)
        .agg(
            total_views=("views", "sum"),
            total_engagements=("engagements", "sum"),
        )
        .sort_values("time_bucket")
    )
    return ts


def build_region_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns or "region" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work = work.dropna(subset=["timestamp"])
    work["time_bucket"] = work["timestamp"].dt.floor("5min")

    ts = (
        work.groupby(["region", "time_bucket"], as_index=False)
        .agg(
            total_views=("views", "sum"),
            avg_engagement_rate=("engagement_rate", "mean"),
        )
        .sort_values("time_bucket")
    )
    return ts


def build_publish_hour_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "publish_time" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["publish_time"] = pd.to_datetime(work["publish_time"], errors="coerce", utc=True)
    work = work.dropna(subset=["publish_time"])

    work["publish_day"] = work["publish_time"].dt.day_name()
    work["publish_hour"] = work["publish_time"].dt.hour

    day_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
    work["publish_day"] = pd.Categorical(work["publish_day"], categories=day_order, ordered=True)

    heatmap = (
        work.groupby(["publish_day", "publish_hour"], as_index=False)
        .agg(
            avg_views=("views", "mean"),
            avg_engagement_rate=("engagement_rate", "mean"),
            videos=("video_id", "nunique"),
        )
    )
    return heatmap


def build_category_share_over_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work = work.dropna(subset=["timestamp"])
    work["time_bucket"] = work["timestamp"].dt.floor("5min")

    grouped = (
        work.groupby(["category", "time_bucket"], as_index=False)
        .agg(total_views=("views", "sum"))
    )

    totals = (
        grouped.groupby("time_bucket", as_index=False)
        .agg(bucket_views=("total_views", "sum"))
    )

    merged = grouped.merge(totals, on="time_bucket", how="left")
    merged["view_share"] = merged["total_views"] / merged["bucket_views"]
    return merged


def build_channel_leaderboard(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    if df.empty or "channel_title" not in df.columns:
        return pd.DataFrame()

    board = (
        df.groupby("channel_title", as_index=False)
        .agg(
            videos=("video_id", "nunique"),
            total_views=("views", "sum"),
            total_engagements=("engagements", "sum"),
            avg_engagement_rate=("engagement_rate", "mean"),
        )
        .sort_values(["total_views", "avg_engagement_rate"], ascending=[False, False])
        .head(top_n)
    )
    return board


def build_bubble_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    latest = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["video_id", "region"], keep="last")
        .copy()
    )

    latest["bubble_size"] = latest["comments"].clip(lower=1)

    return latest[
        [
            "title",
            "channel_title",
            "category",
            "region",
            "views",
            "engagement_rate",
            "bubble_size",
        ]
    ]


def build_outlier_videos(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    latest = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["video_id", "region"], keep="last")
        .copy()
    )

    latest["views"] = pd.to_numeric(latest["views"], errors="coerce")
    latest["engagement_rate"] = pd.to_numeric(latest["engagement_rate"], errors="coerce")
    latest["comments"] = pd.to_numeric(latest["comments"], errors="coerce")

    latest = latest.dropna(subset=["views", "engagement_rate"])
    latest = latest[
        latest["views"].replace([float("inf"), -float("inf")], pd.NA).notna()
        & latest["engagement_rate"].replace([float("inf"), -float("inf")], pd.NA).notna()
    ].copy()

    if latest.empty:
        return pd.DataFrame()

    views_std = latest["views"].std(ddof=0)
    eng_std = latest["engagement_rate"].std(ddof=0)

    if pd.isna(views_std) or views_std == 0:
        latest["views_zscore"] = 0.0
    else:
        latest["views_zscore"] = (latest["views"] - latest["views"].mean()) / views_std

    if pd.isna(eng_std) or eng_std == 0:
        latest["eng_rate_zscore"] = 0.0
    else:
        latest["eng_rate_zscore"] = (
            (latest["engagement_rate"] - latest["engagement_rate"].mean()) / eng_std
        )

    latest["viral_score"] = (
        latest["views"].rank(pct=True) + latest["engagement_rate"].rank(pct=True)
    )

    return latest.sort_values("viral_score", ascending=False).head(top_n)[
        [
            "title",
            "channel_title",
            "category",
            "region",
            "views",
            "engagement_rate",
            "comments",
            "viral_score",
        ]
    ]


def build_category_growth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    work = work.dropna(subset=["timestamp"])
    work["time_bucket"] = work["timestamp"].dt.floor("5min")

    ts = (
        work.groupby(["category", "time_bucket"], as_index=False)
        .agg(total_views=("views", "sum"))
        .sort_values(["category", "time_bucket"])
    )

    ts["previous_views"] = ts.groupby("category")["total_views"].shift(1)
    ts["growth_rate"] = (
        (ts["total_views"] - ts["previous_views"]) / ts["previous_views"]
    ).replace([float("inf"), -float("inf")], 0)

    return ts.dropna(subset=["growth_rate"])


def build_comments_vs_views(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    latest = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["video_id", "region"], keep="last")
        .copy()
    )

    latest = latest[
        [
            "title",
            "category",
            "region",
            "channel_title",
            "views",
            "comments",
            "likes",
            "rank",
        ]
    ].dropna(subset=["views", "comments"])

    return latest

def build_latest_snapshot_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    latest = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["video_id", "region"], keep="last")
        .copy()
    )
    return latest

