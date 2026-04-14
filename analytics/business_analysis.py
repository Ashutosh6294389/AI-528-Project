import pandas as pd
from sklearn.linear_model import LinearRegression


def prepare_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    defaults = {
        "video_id": None,
        "channel_title": "Unknown",
        "published_at": None,
        "fetched_at": None,
        "region": "IN",
        "category": "Other",
        "views": 0,
        "likes": 0,
        "comments": 0,
        "engagements": 0,
        "like_rate": 0.0,
        "comment_rate": 0.0,
        "engagement_rate": 0.0,
    }

    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value

    for col in ["views", "likes", "comments", "engagements"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "engagements" not in df.columns or (df["engagements"] == 0).all():
        df["engagements"] = df["likes"] + df["comments"]

    df["engagement_rate"] = df["engagements"].div(df["views"]).fillna(0)
    df["like_rate"] = df["likes"].div(df["views"]).fillna(0)
    df["comment_rate"] = df["comments"].div(df["views"]).fillna(0)

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce", utc=True)

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
    summary = (
        df.groupby("category", dropna=False)
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
        df.sort_values(["views", "engagement_rate"], ascending=[False, False])
        .drop_duplicates(subset=["video_id"])
        .loc[:, ["title", "channel_title", "category", "region", "views", "likes", "comments", "engagement_rate"]]
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
        df.sort_values("fetched_at")
        .drop_duplicates(subset=["video_id"], keep="last")
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
        ],
    ].sort_values(["engagement_gap", "view_gap"], ascending=[True, True])

    return diagnostic.head(20)


def build_forecast(df: pd.DataFrame, top_n_categories: int = 5, horizon: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()

    if work["fetched_at"].notna().any():
        work["time_bucket"] = work["fetched_at"].dt.floor("h")
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
        model = LinearRegression()
        model.fit(group[["step"]], group["total_views"])

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
            pred = max(0, float(model.predict([[last_step + i]])[0]))
            if isinstance(last_time, pd.Timestamp):
                future_bucket = last_time + pd.Timedelta(hours=i)
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
