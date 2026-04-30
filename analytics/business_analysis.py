import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


def parse_duration_seconds(duration_iso: str) -> int:
    if not duration_iso or pd.isna(duration_iso):
        return 0

    pattern = re.compile(
        r"PT"
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+)S)?"
    )
    match = pattern.fullmatch(str(duration_iso))
    if not match:
        return 0

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def duration_bucket(seconds: int) -> str:
    if seconds < 240:
        return "Short"
    if seconds < 900:
        return "Medium"
    return "Long"


def build_latest_snapshot_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    return (
        df.sort_values("collected_at")
        .drop_duplicates(subset=["video_id", "trending_region"], keep="last")
        .copy()
    )


def prepare_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    defaults = {
        "collection_batch_id": None,
        "collected_at": None,
        "surface": "mostPopular",
        "trending_region": "IN",
        "trending_category_id": None,
        "trending_page": 1,
        "trending_rank": None,
        "video_id": None,
        "title": "",
        "description": "",
        "published_at": None,
        "category_id": None,
        "category_name": "Other",
        "tags": "[]",
        "default_language": None,
        "thumbnail_url": None,
        "view_count": 0,
        "like_count": 0,
        "comment_count": 0,
        "favorite_count": 0,
        "duration_iso": None,
        "definition": None,
        "caption": False,
        "licensed_content": False,
        "content_rating": None,
        "projection": None,
        "channel_id": None,
        "channel_title": "Unknown",
        "channel_subscriber_count": 0,
        "channel_view_count": 0,
        "channel_video_count": 0,
        "channel_country": None,
    }

    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value

    numeric_cols = [
        "trending_page",
        "trending_rank",
        "view_count",
        "like_count",
        "comment_count",
        "favorite_count",
        "channel_subscriber_count",
        "channel_view_count",
        "channel_video_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["collected_at"] = pd.to_datetime(df["collected_at"], errors="coerce", utc=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    df["engagements"] = df["like_count"] + df["comment_count"]
    # Use np.nan (a float64 value) instead of pd.NA — pd.NA forces the
    # Series to object dtype, which then trips pandas 2.2's deprecation
    # warning on the .fillna(0) below.
    safe_views = df["view_count"].replace(0, np.nan)

    df["like_rate"] = (df["like_count"] / safe_views).fillna(0)
    df["comment_rate"] = (df["comment_count"] / safe_views).fillna(0)
    df["engagement_rate"] = (df["engagements"] / safe_views).fillna(0)

    df["like_rate"] = df["like_rate"].replace([np.inf, -np.inf], 0)
    df["comment_rate"] = df["comment_rate"].replace([np.inf, -np.inf], 0)
    df["engagement_rate"] = df["engagement_rate"].replace([np.inf, -np.inf], 0)

    now_ts = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
    df["video_age_hours"] = (
        (now_ts - df["published_at"]).dt.total_seconds() / 3600
    ).fillna(0).clip(lower=0.01)

    df["velocity"] = (df["view_count"] / df["video_age_hours"]).fillna(0)
    df["title_word_count"] = df["title"].fillna("").astype(str).str.split().str.len()
    df["title_has_question"] = df["title"].fillna("").astype(str).str.contains(r"\?")
    df["title_has_number"] = df["title"].fillna("").astype(str).str.contains(r"\d")
    df["title_caps_ratio"] = df["title"].fillna("").astype(str).apply(
        lambda text: (
            sum(1 for c in text if c.isupper()) / max(sum(1 for c in text if c.isalpha()), 1)
        )
    )
    df["tag_count"] = df["tags"].fillna("[]").astype(str).apply(
        lambda value: 0 if value in ("[]", "", "nan", "None") else value.count(",") + 1
    )

    df["duration_seconds"] = df["duration_iso"].apply(parse_duration_seconds)
    df["duration_bucket"] = df["duration_seconds"].apply(duration_bucket)

    df["category_name"] = df["category_name"].fillna("Other").astype(str).str.strip()
    df["surface"] = df["surface"].fillna("mostPopular").astype(str)
    df["channel_title"] = df["channel_title"].fillna("Unknown").astype(str)

    return df


def build_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = build_latest_snapshot_df(df)

    summary = (
        work.groupby("category_name", dropna=False)
        .agg(
            videos=("video_id", "nunique"),
            total_views=("view_count", "sum"),
            total_likes=("like_count", "sum"),
            total_comments=("comment_count", "sum"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_like_rate=("like_rate", "mean"),
        )
        .reset_index()
        .sort_values(["total_views", "avg_engagement_rate"], ascending=[False, False])
    )
    return summary


def build_top_videos(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    latest_batch_time = df["collected_at"].max()
    latest_batch = df[df["collected_at"] == latest_batch_time].copy()

    if latest_batch.empty:
        return pd.DataFrame()

    latest_batch = latest_batch.sort_values("view_count", ascending=False).copy()
    latest_batch["computed_rank"] = range(1, len(latest_batch) + 1)

    leaderboard = latest_batch[
        [
            "video_id",
            "title",
            "channel_title",
            "category_name",
            "trending_region",
            "trending_rank",
            "computed_rank",
            "view_count",
            "like_count",
            "comment_count",
            "like_rate",
            "comment_rate",
        ]
    ].head(top_n)

    return leaderboard


def build_diagnostic_table(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df.groupby("category_name")
        .agg(
            category_avg_views=("view_count", "mean"),
            category_avg_engagement_rate=("engagement_rate", "mean"),
        )
        .reset_index()
    )

    latest = (
        build_latest_snapshot_df(df)
        .merge(baseline, on="category_name", how="left")
    )

    latest["engagement_gap"] = latest["engagement_rate"] - latest["category_avg_engagement_rate"]
    latest["view_gap"] = latest["view_count"] - latest["category_avg_views"]

    return latest[
        [
            "title",
            "channel_title",
            "category_name",
            "trending_region",
            "view_count",
            "engagement_rate",
            "category_avg_engagement_rate",
            "engagement_gap",
            "view_gap",
            "trending_rank",
        ]
    ].sort_values(["engagement_gap", "view_gap"], ascending=[True, True]).head(20)


def build_forecast(df: pd.DataFrame, top_n_categories: int = 5, horizon: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work = work.dropna(subset=["collected_at"])
    work["time_bucket"] = work["collected_at"].dt.floor("5min")

    top_categories = (
        work.groupby("category_name")["view_count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_categories)
        .index.tolist()
    )

    work = work[work["category_name"].isin(top_categories)]

    ts = (
        work.groupby(["category_name", "time_bucket"], as_index=False)
        .agg(total_views=("view_count", "sum"))
        .sort_values(["category_name", "time_bucket"])
    )

    rows = []

    for category, group in ts.groupby("category_name"):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue

        group["step"] = range(len(group))
        model = LinearRegression()
        model.fit(group[["step"]], group["total_views"])

        for _, row in group.iterrows():
            rows.append(
                {
                    "category_name": category,
                    "time_bucket": row["time_bucket"],
                    "total_views": row["total_views"],
                    "series": "Actual",
                }
            )

        last_step = int(group["step"].max())
        last_time = group["time_bucket"].iloc[-1]

        for i in range(1, horizon + 1):
            pred = max(0, float(model.predict(pd.DataFrame({"step": [last_step + i]}))[0]))
            rows.append(
                {
                    "category_name": category,
                    "time_bucket": last_time + pd.Timedelta(minutes=5 * i),
                    "total_views": pred,
                    "series": "Forecast",
                }
            )

    return pd.DataFrame(rows)


def build_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    summary = build_category_summary(df)
    if summary.empty:
        return pd.DataFrame(columns=["priority", "recommendation", "why"])

    high_views = summary["total_views"].median()
    high_eng = summary["avg_engagement_rate"].median()

    actions = []

    for _, row in summary.iterrows():
        category = row["category_name"]

        if row["total_views"] >= high_views and row["avg_engagement_rate"] < high_eng:
            actions.append(
                {
                    "priority": "High",
                    "recommendation": f"Improve CTA and thumbnails for {category}",
                    "why": "Strong reach but weaker engagement suggests better packaging could lift performance.",
                }
            )

        if row["total_views"] < high_views and row["avg_engagement_rate"] >= high_eng:
            actions.append(
                {
                    "priority": "Medium",
                    "recommendation": f"Promote more {category} content",
                    "why": "This category engages efficiently and may scale well with more distribution.",
                }
            )

    if not actions:
        actions.append(
            {
                "priority": "Medium",
                "recommendation": "Continue collecting more data before making major content decisions",
                "why": "Current category behavior is still too balanced for stronger prescriptive confidence.",
            }
        )

    return pd.DataFrame(actions).drop_duplicates()


def build_views_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["collected_at"]).copy()
    if work.empty:
        return pd.DataFrame()

    work["time_bucket"] = work["collected_at"].dt.floor("5min")
    return (
        work.groupby(["category_name", "time_bucket"], as_index=False)
        .agg(
            total_views=("view_count", "sum"),
            total_engagements=("engagements", "sum"),
        )
        .sort_values("time_bucket")
    )


def build_region_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["collected_at"]).copy()
    if work.empty:
        return pd.DataFrame()

    work["time_bucket"] = work["collected_at"].dt.floor("5min")
    return (
        work.groupby(["trending_region", "time_bucket"], as_index=False)
        .agg(
            total_views=("view_count", "sum"),
            avg_engagement_rate=("engagement_rate", "mean"),
        )
        .sort_values("time_bucket")
    )


def build_publish_hour_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    work = build_latest_snapshot_df(df)
    work = work.dropna(subset=["published_at"]).copy()
    if work.empty:
        return pd.DataFrame()

    work["publish_day"] = work["published_at"].dt.day_name()
    work["publish_hour"] = work["published_at"].dt.hour

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    work["publish_day"] = pd.Categorical(work["publish_day"], categories=day_order, ordered=True)

    heatmap = (
        work.groupby(["publish_day", "publish_hour"], as_index=False, observed=True)
        .agg(
            avg_views=("view_count", "mean"),
            avg_engagement_rate=("engagement_rate", "mean"),
            videos=("video_id", "nunique"),
        )
    )

    heatmap["publish_day"] = pd.Categorical(
        heatmap["publish_day"], categories=day_order, ordered=True
    )
    return heatmap.sort_values(["publish_day", "publish_hour"]).reset_index(drop=True)


def build_category_share_over_time(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["collected_at"]).copy()
    if work.empty:
        return pd.DataFrame()

    work["time_bucket"] = work["collected_at"].dt.floor("5min")

    grouped = (
        work.groupby(["category_name", "time_bucket"], as_index=False)
        .agg(total_views=("view_count", "sum"))
    )

    totals = grouped.groupby("time_bucket", as_index=False).agg(bucket_views=("total_views", "sum"))
    merged = grouped.merge(totals, on="time_bucket", how="left")
    merged["view_share"] = merged["total_views"] / merged["bucket_views"]
    return merged


def build_channel_leaderboard(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    return (
        latest.groupby("channel_title", as_index=False)
        .agg(
            videos=("video_id", "nunique"),
            total_views=("view_count", "sum"),
            total_engagements=("engagements", "sum"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_subscribers=("channel_subscriber_count", "mean"),
        )
        .sort_values(["total_views", "avg_engagement_rate"], ascending=[False, False])
        .head(top_n)
    )


def build_bubble_dataset(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    latest["bubble_size"] = latest["comment_count"].clip(lower=1)
    return latest[
        [
            "title",
            "channel_title",
            "category_name",
            "trending_region",
            "view_count",
            "engagement_rate",
            "bubble_size",
        ]
    ]


def build_outlier_videos(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    latest = latest.dropna(subset=["view_count", "engagement_rate"]).copy()
    latest["viral_score"] = (
        latest["view_count"].rank(pct=True) + latest["engagement_rate"].rank(pct=True)
    )

    return latest.sort_values("viral_score", ascending=False).head(top_n)[
        [
            "title",
            "channel_title",
            "category_name",
            "trending_region",
            "view_count",
            "engagement_rate",
            "comment_count",
            "viral_score",
        ]
    ]


def build_category_growth(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["collected_at"]).copy()
    if work.empty:
        return pd.DataFrame()

    work["time_bucket"] = work["collected_at"].dt.floor("5min")

    ts = (
        work.groupby(["category_name", "time_bucket"], as_index=False)
        .agg(total_views=("view_count", "sum"))
        .sort_values(["category_name", "time_bucket"])
    )

    ts["previous_views"] = ts.groupby("category_name")["total_views"].shift(1)
    ts["growth_rate"] = (
        (ts["total_views"] - ts["previous_views"]) / ts["previous_views"]
    ).replace([float("inf"), -float("inf")], 0)

    return ts.dropna(subset=["growth_rate"])


def build_comments_vs_views(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    return latest[
        [
            "title",
            "category_name",
            "trending_region",
            "channel_title",
            "view_count",
            "comment_count",
            "like_count",
            "trending_rank",
        ]
    ].dropna(subset=["view_count", "comment_count"])


def build_engagement_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    heatmap = (
        latest.groupby(["trending_region", "category_name"], as_index=False)
        .agg(
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_like_rate=("like_rate", "mean"),
            avg_comment_rate=("comment_rate", "mean"),
            sample_size=("video_id", "nunique"),
        )
    )
    return heatmap


def build_duration_distribution(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    duration_df = (
        latest.groupby(["trending_region", "category_name", "duration_bucket"], as_index=False)
        .agg(
            video_count=("video_id", "nunique"),
            avg_views_in_bucket=("view_count", "mean"),
            avg_er_in_bucket=("engagement_rate", "mean"),
        )
    )
    return duration_df


def build_subscriber_tier_distribution(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    def tier(subs):
        if subs > 10_000_000:
            return "Mega (>10M)"
        if subs > 1_000_000:
            return "Large (1M-10M)"
        if subs > 100_000:
            return "Mid (100K-1M)"
        return "Small (<100K)"

    latest = latest.copy()
    latest["subscriber_tier"] = latest["channel_subscriber_count"].apply(tier)

    tier_df = (
        latest.groupby(["trending_region", "subscriber_tier"], as_index=False)
        .agg(video_count=("video_id", "nunique"))
    )

    totals = (
        tier_df.groupby("trending_region", as_index=False)
        .agg(region_total=("video_count", "sum"))
    )

    tier_df = tier_df.merge(totals, on="trending_region", how="left")
    tier_df["pct"] = (tier_df["video_count"] / tier_df["region_total"]) * 100
    return tier_df


def build_tag_usage_frequency(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    work = latest.copy()

    def parse_tags(tag_str):
        if pd.isna(tag_str) or not str(tag_str).strip():
            return []
        text = str(tag_str).strip()
        if text in ("[]", "nan", "None"):
            return []
        text = text.strip("[]")
        if not text:
            return []
        parts = [p.strip().strip("'").strip('"').lower() for p in text.split(",")]
        return [p for p in parts if p]

    work["tag_list"] = work["tags"].apply(parse_tags)
    exploded = work.explode("tag_list")
    exploded = exploded.dropna(subset=["tag_list"])
    exploded = exploded[exploded["tag_list"] != ""]

    if exploded.empty:
        return pd.DataFrame()

    tag_df = (
        exploded.groupby(["category_name", "trending_region", "tag_list"], as_index=False)
        .agg(videos_using_tag=("video_id", "nunique"))
        .sort_values(["videos_using_tag"], ascending=False)
        .rename(columns={"tag_list": "tag"})
    )

    return tag_df.head(top_n)


def build_hd_sd_distribution(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    dist = (
        latest.groupby(["trending_region", "category_name", "definition"], as_index=False)
        .agg(video_count=("video_id", "nunique"))
    )

    totals = (
        dist.groupby(["trending_region", "category_name"], as_index=False)
        .agg(total_count=("video_count", "sum"))
    )

    dist = dist.merge(totals, on=["trending_region", "category_name"], how="left")
    dist["pct"] = (dist["video_count"] / dist["total_count"]) * 100
    return dist


def build_caption_rate(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    cap_df = (
        latest.groupby(["trending_region", "category_name"], as_index=False)
        .agg(
            captioned_videos=("caption", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
            total_unique_videos=("video_id", "nunique"),
        )
    )

    cap_df["caption_rate_pct"] = (
        cap_df["captioned_videos"] / cap_df["total_unique_videos"] * 100
    ).fillna(0)

    return cap_df


def build_latest_snapshot_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    latest_batch = df["collection_batch_id"].dropna().max()
    if pd.isna(latest_batch):
        return pd.DataFrame()

    return df[df["collection_batch_id"] == latest_batch].copy()


def build_view_velocity(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_only(df)
    if latest.empty:
        return pd.DataFrame()

    latest = latest.copy()
    latest["views_per_hour"] = (latest["view_count"] / latest["video_age_hours"].replace(0, pd.NA)).fillna(0)

    cat_avg = (
        latest.groupby(["category_name", "trending_region"], as_index=False)
        .agg(category_avg_velocity=("views_per_hour", "mean"))
    )

    latest = latest.merge(cat_avg, on=["category_name", "trending_region"], how="left")
    latest["relative_velocity"] = (
        latest["views_per_hour"] / latest["category_avg_velocity"].replace(0, pd.NA)
    ).fillna(0)

    return latest[
        [
            "video_id",
            "title",
            "category_name",
            "trending_region",
            "view_count",
            "video_age_hours",
            "views_per_hour",
            "category_avg_velocity",
            "relative_velocity",
            "trending_rank",
        ]
    ].sort_values("views_per_hour", ascending=False)


def build_trending_persistence(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    persistence = (
        df.groupby(
            ["video_id", "title", "channel_title", "category_name", "trending_region"],
            as_index=False
        )
        .agg(
            batches_appeared=("collection_batch_id", "nunique"),
            first_seen=("collected_at", "min"),
            last_seen=("collected_at", "max"),
            best_rank_achieved=("trending_rank", "min"),
            peak_view_count=("view_count", "max"),
        )
    )

    persistence["estimated_hours_trending"] = persistence["batches_appeared"] * 0.0125
    return persistence.sort_values("batches_appeared", ascending=False)


def build_rank_movement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.sort_values(["video_id", "trending_region", "collected_at"]).copy()
    work["prev_rank"] = work.groupby(["video_id", "trending_region"])["trending_rank"].shift(1)
    work["rank_improvement"] = work["prev_rank"] - work["trending_rank"]
    work["views_gained_this_batch"] = (
        work["view_count"] - work.groupby(["video_id", "trending_region"])["view_count"].shift(1)
    )

    return work[
        [
            "video_id",
            "title",
            "category_name",
            "trending_region",
            "collected_at",
            "trending_rank",
            "prev_rank",
            "rank_improvement",
            "views_gained_this_batch",
        ]
    ].dropna(subset=["prev_rank"])


def build_engagement_vs_views_correlation(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_only(df)
    if latest.empty:
        return pd.DataFrame(), pd.DataFrame()

    latest = latest[latest["view_count"] > 1000].copy()

    corr_rows = []
    for (category, region), group in latest.groupby(["category_name", "trending_region"]):
        if len(group) < 2:
            continue

        corr_er = group["view_count"].corr(group["engagement_rate"])
        corr_likes = group["view_count"].corr(group["like_count"])

        corr_rows.append(
            {
                "category_name": category,
                "trending_region": region,
                "corr_views_vs_er": corr_er,
                "corr_views_vs_likes": corr_likes,
                "n": len(group),
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    scatter_df = latest[
        [
            "video_id",
            "title",
            "category_name",
            "trending_region",
            "view_count",
            "engagement_rate",
            "channel_subscriber_count",
        ]
    ]
    return corr_df, scatter_df


def build_title_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    work = latest.copy()
    work["caps_level"] = work["title_caps_ratio"].apply(lambda x: "high_caps" if x > 0.3 else "normal_caps")

    def title_len_bucket(n):
        if n <= 5:
            return "short"
        if n <= 10:
            return "medium"
        return "long"

    work["title_length"] = work["title_word_count"].apply(title_len_bucket)

    return (
        work.groupby(
            [
                "category_name",
                "trending_region",
                "title_has_question",
                "title_has_number",
                "caps_level",
                "title_length",
            ],
            as_index=False
        )
        .agg(
            avg_velocity=("velocity", "mean"),
            avg_er=("engagement_rate", "mean"),
            sample_size=("video_id", "nunique"),
        )
        .query("sample_size >= 10")
    )


def build_tag_count_vs_engagement(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    work = latest.copy()

    def bucket(tag_count):
        if tag_count == 0:
            return "no_tags"
        if tag_count <= 5:
            return "1-5"
        if tag_count <= 15:
            return "6-15"
        if tag_count <= 30:
            return "16-30"
        return "30+"

    work["tag_count_bucket"] = work["tag_count"].apply(bucket)

    return (
        work.groupby(["category_name", "tag_count_bucket"], as_index=False)
        .agg(
            avg_er=("engagement_rate", "mean"),
            avg_views=("view_count", "mean"),
            n_videos=("video_id", "nunique"),
        )
    )


def build_duration_vs_engagement(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    return (
        latest.groupby(["category_name", "trending_region", "duration_bucket"], as_index=False)
        .agg(
            avg_like_rate=("like_rate", "mean"),
            avg_comment_rate=("comment_rate", "mean"),
            avg_er=("engagement_rate", "mean"),
            avg_views=("view_count", "mean"),
            median_views=("view_count", "median"),
            n_videos=("video_id", "nunique"),
        )
    )


def build_channel_size_vs_reach(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    work = latest.copy()

    def subscriber_tier(subs):
        if subs > 10_000_000:
            return "Mega (>10M)"
        if subs > 1_000_000:
            return "Large (1M-10M)"
        if subs > 100_000:
            return "Mid (100K-1M)"
        return "Small (<100K)"

    work["subscriber_tier"] = work["channel_subscriber_count"].apply(subscriber_tier)

    return (
        work.groupby(["trending_region", "category_name", "subscriber_tier"], as_index=False)
        .agg(
            avg_trending_rank=("trending_rank", "mean"),
            best_rank_achieved=("trending_rank", "min"),
            avg_views=("view_count", "mean"),
            unique_videos=("video_id", "nunique"),
        )
    )


def build_regional_preference_divergence(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_only(df)
    if latest.empty:
        return pd.DataFrame()

    global_avg = (
        latest.groupby("category_name", as_index=False)
        .agg(video_count=("video_id", "count"))
    )
    global_total = global_avg["video_count"].sum()
    global_avg["global_pct"] = (global_avg["video_count"] * 100.0 / global_total)

    regional = (
        latest.groupby(["trending_region", "category_name"], as_index=False)
        .agg(video_count=("video_id", "count"))
    )

    regional_totals = (
        regional.groupby("trending_region", as_index=False)
        .agg(region_total=("video_count", "sum"))
    )

    regional = regional.merge(regional_totals, on="trending_region", how="left")
    regional["regional_pct"] = regional["video_count"] * 100.0 / regional["region_total"]

    merged = regional.merge(global_avg[["category_name", "global_pct"]], on="category_name", how="left")
    merged["divergence_score"] = merged["regional_pct"] - merged["global_pct"]

    return merged.sort_values("divergence_score", ascending=False)


def build_recency_bias(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    first_entry = (
        df.groupby(["video_id", "category_name", "trending_region"], as_index=False)
        .agg(first_age_hours=("video_age_hours", "min"))
    )

    def bucket(hours):
        if hours < 6:
            return "0-6h"
        if hours < 24:
            return "6-24h"
        if hours < 72:
            return "1-3 days"
        if hours < 168:
            return "3-7 days"
        return ">7 days"

    first_entry["age_at_trending_entry"] = first_entry["first_age_hours"].apply(bucket)

    return (
        first_entry.groupby(["category_name", "trending_region", "age_at_trending_entry"], as_index=False)
        .agg(video_count=("video_id", "nunique"))
    )


def build_weekend_weekday_behavior(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["day_type"] = work["collected_at"].dt.dayofweek.apply(lambda d: "Weekend" if d >= 5 else "Weekday")

    return (
        work.groupby(["category_name", "trending_region", "day_type"], as_index=False)
        .agg(
            avg_view_count=("view_count", "mean"),
            avg_er=("engagement_rate", "mean"),
            unique_videos_seen=("video_id", "nunique"),
            total_observations=("video_id", "count"),
        )
    )

def build_trending_rank_distribution(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_only(df)
    if latest.empty:
        return pd.DataFrame()

    dist = (
        latest.groupby(["trending_region", "category_name"], as_index=False)
        .agg(video_count=("video_id", "count"))
    )

    totals = (
        dist.groupby("trending_region", as_index=False)
        .agg(region_total=("video_count", "sum"))
    )

    dist = dist.merge(totals, on="trending_region", how="left")
    dist["pct_of_trending"] = (dist["video_count"] * 100.0 / dist["region_total"]).round(2)

    return dist.sort_values(["trending_region", "video_count"], ascending=[True, False])


def build_trending_entry_probability(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.sort_values(["trending_region", "collected_at", "video_id"]).copy()
    work["batch_order"] = work.groupby("trending_region")["collected_at"].rank(method="dense").astype(int)
    work["title_has_question_int"] = work["title_has_question"].astype(int)
    work["title_has_number_int"] = work["title_has_number"].astype(int)

    batch_presence = (
        work[["video_id", "trending_region", "batch_order"]]
        .drop_duplicates()
        .assign(in_next_batch=1)
    )
    batch_presence["batch_order"] = batch_presence["batch_order"] - 1

    train = work.merge(
        batch_presence,
        on=["video_id", "trending_region", "batch_order"],
        how="left",
    )
    train["in_next_batch"] = train["in_next_batch"].fillna(0).astype(int)

    feature_cols = [
        "view_count",
        "velocity",
        "video_age_hours",
        "like_rate",
        "comment_rate",
        "tag_count",
        "title_word_count",
        "title_has_question_int",
        "title_has_number_int",
        "channel_subscriber_count",
    ]

    latest = build_latest_snapshot_only(train)
    if latest.empty:
        return pd.DataFrame()

    latest_batch_id = latest["collection_batch_id"].iloc[0]
    historical = train[train["collection_batch_id"] != latest_batch_id].dropna(subset=feature_cols).copy()

    if historical.empty or historical["in_next_batch"].nunique() < 2:
        fallback = latest.copy()
        fallback["predicted_probability"] = (
            0.35 * fallback["engagement_rate"].rank(pct=True) +
            0.35 * fallback["velocity"].rank(pct=True) +
            0.30 * (1 - fallback["trending_rank"].rank(pct=True))
        ).fillna(0)
        return fallback[
            [
                "video_id",
                "title",
                "channel_title",
                "category_name",
                "trending_region",
                "trending_rank",
                "view_count",
                "velocity",
                "engagement_rate",
                "predicted_probability",
            ]
        ].sort_values("predicted_probability", ascending=False).head(top_n)

    model = LogisticRegression(max_iter=1000)
    model.fit(historical[feature_cols], historical["in_next_batch"])

    score_df = latest.dropna(subset=feature_cols).copy()
    if score_df.empty:
        return pd.DataFrame()

    score_df["predicted_probability"] = model.predict_proba(score_df[feature_cols])[:, 1]
    return score_df[
        [
            "video_id",
            "title",
            "channel_title",
            "category_name",
            "trending_region",
            "trending_rank",
            "view_count",
            "velocity",
            "engagement_rate",
            "predicted_probability",
        ]
    ].sort_values("predicted_probability", ascending=False).head(top_n)


def build_view_count_forecast_v2(df: pd.DataFrame, horizon: int = 4, top_n_videos: int = 8) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.dropna(subset=["collected_at"]).sort_values(["video_id", "trending_region", "collected_at"]).copy()
    latest = build_latest_snapshot_only(work)
    if latest.empty:
        return pd.DataFrame()

    candidates = latest[["video_id", "trending_region"]].drop_duplicates()
    histories = work.merge(candidates, on=["video_id", "trending_region"], how="inner")

    counts = (
        histories.groupby(["video_id", "trending_region"], as_index=False)
        .agg(points=("collected_at", "count"), last_views=("view_count", "max"))
    )
    counts = counts[counts["points"] >= 4].sort_values(["points", "last_views"], ascending=[False, False]).head(top_n_videos)
    if counts.empty:
        return pd.DataFrame()

    selected = histories.merge(counts[["video_id", "trending_region"]], on=["video_id", "trending_region"], how="inner")
    rows = []

    for (video_id, region), group in selected.groupby(["video_id", "trending_region"]):
        group = group.sort_values("collected_at").reset_index(drop=True)
        group["step"] = np.arange(len(group))

        model = LinearRegression()
        model.fit(group[["step"]], group["view_count"])
        fitted = model.predict(group[["step"]])
        residual_std = float(np.std(group["view_count"] - fitted)) if len(group) > 1 else 0.0

        for _, row in group.iterrows():
            rows.append(
                {
                    "video_id": video_id,
                    "trending_region": region,
                    "title": row["title"],
                    "category_name": row["category_name"],
                    "time_bucket": row["collected_at"],
                    "forecast_views": row["view_count"],
                    "lower_bound": max(0.0, row["view_count"] - residual_std),
                    "upper_bound": row["view_count"] + residual_std,
                    "series": "Actual",
                }
            )

        last_step = int(group["step"].max())
        last_time = group["collected_at"].iloc[-1]
        delta = group["collected_at"].diff().median()
        if pd.isna(delta) or delta <= pd.Timedelta(0):
            delta = pd.Timedelta(minutes=1)

        for i in range(1, horizon + 1):
            pred = float(model.predict(pd.DataFrame({"step": [last_step + i]}))[0])
            pred = max(0.0, pred)
            rows.append(
                {
                    "video_id": video_id,
                    "trending_region": region,
                    "title": group["title"].iloc[-1],
                    "category_name": group["category_name"].iloc[-1],
                    "time_bucket": last_time + (delta * i),
                    "forecast_views": pred,
                    "lower_bound": max(0.0, pred - residual_std),
                    "upper_bound": pred + residual_std,
                    "series": "Forecast",
                }
            )

    return pd.DataFrame(rows)


def build_trending_duration_prediction(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.sort_values(["video_id", "trending_region", "collected_at"]).copy()
    episode_df = (
        work.groupby(["video_id", "trending_region"], as_index=False)
        .agg(
            title=("title", "last"),
            category_name=("category_name", "last"),
            channel_title=("channel_title", "last"),
            age_at_entry=("video_age_hours", "min"),
            initial_velocity=("velocity", "first"),
            initial_er=("engagement_rate", "first"),
            subscriber_count=("channel_subscriber_count", "first"),
            total_hours_trending=("collection_batch_id", lambda x: x.nunique() * 0.0125),
        )
    )

    latest = build_latest_snapshot_only(work)
    if latest.empty or len(episode_df) < 5:
        return pd.DataFrame()

    feature_cols = ["age_at_entry", "initial_velocity", "initial_er", "subscriber_count"]
    train = episode_df.dropna(subset=feature_cols + ["total_hours_trending"]).copy()
    if len(train) < 5:
        return pd.DataFrame()

    model = LinearRegression()
    model.fit(train[feature_cols], train["total_hours_trending"])

    current = (
        work.merge(latest[["video_id", "trending_region"]].drop_duplicates(), on=["video_id", "trending_region"], how="inner")
        .groupby(["video_id", "trending_region"], as_index=False)
        .agg(
            title=("title", "last"),
            category_name=("category_name", "last"),
            channel_title=("channel_title", "last"),
            current_rank=("trending_rank", "last"),
            current_hours=("collection_batch_id", lambda x: x.nunique() * 0.0125),
            age_at_entry=("video_age_hours", "min"),
            initial_velocity=("velocity", "first"),
            initial_er=("engagement_rate", "first"),
            subscriber_count=("channel_subscriber_count", "first"),
        )
    )

    current["predicted_total_hours"] = model.predict(current[feature_cols]).clip(min=0)
    current["predicted_remaining_hours"] = (current["predicted_total_hours"] - current["current_hours"]).clip(lower=0)

    return current[
        [
            "video_id",
            "title",
            "channel_title",
            "category_name",
            "trending_region",
            "current_rank",
            "current_hours",
            "predicted_total_hours",
            "predicted_remaining_hours",
        ]
    ].sort_values("predicted_remaining_hours", ascending=False).head(top_n)


def build_peak_rank_forecast(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.sort_values(["video_id", "trending_region", "collected_at"]).copy()
    work["prev_rank"] = work.groupby(["video_id", "trending_region"])["trending_rank"].shift(1)
    work["rank_delta"] = (work["prev_rank"] - work["trending_rank"]).fillna(0)

    episode_df = (
        work.groupby(["video_id", "trending_region"], as_index=False)
        .agg(
            title=("title", "last"),
            category_name=("category_name", "last"),
            current_rank=("trending_rank", "last"),
            avg_rank_delta=("rank_delta", "mean"),
            current_velocity=("velocity", "last"),
            current_er=("engagement_rate", "last"),
            best_rank_achieved=("trending_rank", "min"),
            observations=("collection_batch_id", "nunique"),
        )
    )

    feature_cols = ["current_rank", "avg_rank_delta", "current_velocity", "current_er", "observations"]
    train = episode_df.dropna(subset=feature_cols + ["best_rank_achieved"]).copy()
    if len(train) < 5:
        return pd.DataFrame()

    model = LinearRegression()
    model.fit(train[feature_cols], train["best_rank_achieved"])

    latest = build_latest_snapshot_only(work)
    current = episode_df.merge(
        latest[["video_id", "trending_region"]].drop_duplicates(),
        on=["video_id", "trending_region"],
        how="inner",
    )
    if current.empty:
        return pd.DataFrame()

    current["predicted_peak_rank"] = model.predict(current[feature_cols]).clip(min=1)
    current["expected_rank_gain"] = (current["current_rank"] - current["predicted_peak_rank"]).clip(lower=0)

    return current[
        [
            "video_id",
            "title",
            "category_name",
            "trending_region",
            "current_rank",
            "predicted_peak_rank",
            "expected_rank_gain",
            "avg_rank_delta",
            "current_velocity",
        ]
    ].sort_values("expected_rank_gain", ascending=False).head(top_n)


def build_category_share_forecast(df: pd.DataFrame, horizon: int = 4, top_n_categories: int = 6) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.dropna(subset=["collected_at"]).copy()
    work["time_bucket"] = work["collected_at"].dt.floor("5min")

    grouped = (
        work.groupby(["category_name", "time_bucket"], as_index=False)
        .agg(total_views=("view_count", "sum"))
    )
    if grouped.empty:
        return pd.DataFrame()

    totals = grouped.groupby("time_bucket", as_index=False).agg(bucket_total=("total_views", "sum"))
    grouped = grouped.merge(totals, on="time_bucket", how="left")
    grouped["view_share"] = grouped["total_views"] / grouped["bucket_total"].replace(0, pd.NA)
    grouped["view_share"] = grouped["view_share"].fillna(0)

    top_categories = (
        grouped.groupby("category_name")["total_views"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n_categories)
        .index.tolist()
    )
    grouped = grouped[grouped["category_name"].isin(top_categories)].copy()

    rows = []
    for category, group in grouped.groupby("category_name"):
        group = group.sort_values("time_bucket").reset_index(drop=True)
        if len(group) < 2:
            continue

        group["step"] = np.arange(len(group))
        model = LinearRegression()
        model.fit(group[["step"]], group["view_share"])

        for _, row in group.iterrows():
            rows.append(
                {
                    "category_name": category,
                    "time_bucket": row["time_bucket"],
                    "forecast_share": row["view_share"],
                    "series": "Actual",
                }
            )

        last_step = int(group["step"].max())
        last_time = group["time_bucket"].iloc[-1]
        for i in range(1, horizon + 1):
            pred = float(model.predict(pd.DataFrame({"step": [last_step + i]}))[0])
            rows.append(
                {
                    "category_name": category,
                    "time_bucket": last_time + pd.Timedelta(minutes=5 * i),
                    "forecast_share": float(np.clip(pred, 0, 1)),
                    "series": "Forecast",
                }
            )

    return pd.DataFrame(rows)


def build_optimal_posting_window(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.dropna(subset=["published_at"]).copy()
    if work.empty:
        return pd.DataFrame()

    video_peaks = (
        work.groupby(["video_id", "category_name", "trending_region"], as_index=False)
        .agg(
            publish_hour_utc=("published_at", lambda x: x.iloc[0].hour if len(x) else None),
            publish_day=("published_at", lambda x: x.iloc[0].day_name() if len(x) else None),
            peak_views=("view_count", "max"),
        )
    )

    result = (
        video_peaks.groupby(["category_name", "trending_region", "publish_hour_utc", "publish_day"], as_index=False)
        .agg(
            avg_peak_views=("peak_views", "mean"),
            sample_size=("video_id", "nunique"),
        )
    )
    result["eligible_score"] = np.where(result["sample_size"] >= 5, result["avg_peak_views"], 0)
    result["slot_rank"] = (
        result.groupby(["category_name", "trending_region"])["eligible_score"]
        .rank(method="dense", ascending=False)
    )
    return result.sort_values(["category_name", "trending_region", "slot_rank", "avg_peak_views"])


def build_trending_gap_opportunity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    latest = build_latest_snapshot_only(df)
    if latest.empty:
        return pd.DataFrame()

    current = (
        latest.groupby(["category_name", "trending_region"], as_index=False)
        .agg(current_count=("video_id", "count"))
    )

    work = df.dropna(subset=["collected_at"]).copy()
    cutoff = work["collected_at"].max() - pd.Timedelta(days=7)
    work = work[work["collected_at"] >= cutoff]
    if work.empty:
        return pd.DataFrame()

    work["dt"] = work["collected_at"].dt.date
    historical = (
        work.groupby(["category_name", "trending_region", "dt"], as_index=False)
        .agg(daily_count=("video_id", "nunique"))
    )
    historical = (
        historical.groupby(["category_name", "trending_region"], as_index=False)
        .agg(
            avg_count=("daily_count", "mean"),
            stddev_count=("daily_count", lambda x: x.std(ddof=0)),
        )
    )

    merged = current.merge(historical, on=["category_name", "trending_region"], how="inner")
    merged["stddev_count"] = merged["stddev_count"].replace(0, np.nan)
    merged["gap_z_score"] = ((merged["avg_count"] - merged["current_count"]) / merged["stddev_count"]).replace([np.inf, -np.inf], 0).fillna(0)
    merged["status"] = np.where(merged["gap_z_score"] > 1.0, "OPPORTUNITY", "NORMAL")
    return merged.sort_values("gap_z_score", ascending=False)


def build_creator_partnership_recommendations(df: pd.DataFrame, min_trending_videos: int = 2) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    profile = (
        latest.groupby(["channel_id", "channel_title", "trending_region", "category_name"], as_index=False)
        .agg(
            trending_video_count=("video_id", "nunique"),
            avg_er=("engagement_rate", "mean"),
            avg_views=("view_count", "mean"),
            subscriber_count=("channel_subscriber_count", "max"),
        )
    )
    profile = profile[profile["trending_video_count"] >= min_trending_videos].copy()
    if profile.empty:
        return pd.DataFrame()

    def pct_rank(series):
        return series.rank(pct=True, method="average")

    profile["partnership_score"] = (
        0.4 * profile.groupby("category_name")["avg_er"].transform(pct_rank) +
        0.4 * profile.groupby("category_name")["trending_video_count"].transform(pct_rank) +
        0.2 * profile.groupby("category_name")["avg_views"].transform(pct_rank)
    )

    return profile.sort_values("partnership_score", ascending=False)


def build_format_prescriptions(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    work = latest.copy()
    work["has_question"] = np.where(work["title_has_question"], "Question Title", "No Question")
    work["has_number"] = np.where(work["title_has_number"], "Number in Title", "No Number")
    work["caps_style"] = np.where(work["title_caps_ratio"] > 0.3, "High Caps", "Normal Caps")
    work["tag_band"] = pd.cut(
        work["tag_count"],
        bins=[-1, 0, 5, 15, 1000],
        labels=["0 tags", "1-5 tags", "6-15 tags", "16+ tags"],
    )

    rows = []
    feature_sets = [
        ("duration_bucket", "Duration"),
        ("has_question", "Question Titles"),
        ("has_number", "Numbered Titles"),
        ("caps_style", "Caps Style"),
        ("tag_band", "Tag Density"),
    ]

    for feature_col, feature_name in feature_sets:
        grouped = (
            work.groupby(["category_name", feature_col], as_index=False, observed=True)
            .agg(
                avg_er=("engagement_rate", "mean"),
                avg_views=("view_count", "mean"),
                sample_size=("video_id", "nunique"),
            )
        )
        grouped = grouped[grouped["sample_size"] >= 5]
        if grouped.empty:
            continue
        grouped["feature_type"] = feature_name
        grouped = grouped.rename(columns={feature_col: "feature_value"})
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["prescription_score"] = (
        0.6 * result.groupby(["category_name", "feature_type"])["avg_er"].transform(lambda s: s.rank(pct=True)) +
        0.4 * result.groupby(["category_name", "feature_type"])["avg_views"].transform(lambda s: s.rank(pct=True))
    )
    return result.sort_values("prescription_score", ascending=False)


def build_campaign_timing_alerts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.sort_values(["video_id", "trending_region", "collected_at"]).copy()
    work["prev_rank"] = work.groupby(["video_id", "trending_region"])["trending_rank"].shift(1)
    work["rank_delta"] = work["prev_rank"] - work["trending_rank"]

    recent = work.groupby(["video_id", "trending_region"]).tail(3).copy()
    counts = recent.groupby(["video_id", "trending_region"])["rank_delta"].count().reset_index(name="points")
    recent = recent.merge(counts, on=["video_id", "trending_region"], how="left")
    recent = recent[recent["points"] == 3]
    if recent.empty:
        return pd.DataFrame()

    summary = (
        recent.groupby(["video_id", "trending_region"], as_index=False)
        .agg(
            title=("title", "last"),
            category_name=("category_name", "last"),
            current_rank=("trending_rank", "last"),
            avg_rank_delta=("rank_delta", "mean"),
            total_rank_gain=("rank_delta", "sum"),
            latest_views=("view_count", "last"),
        )
    )

    summary["alert_type"] = np.select(
        [
            summary["total_rank_gain"] >= 6,
            summary["total_rank_gain"] <= -6,
        ],
        [
            "Boost Now",
            "Losing Momentum",
        ],
        default="Monitor",
    )
    return summary.sort_values(["alert_type", "total_rank_gain"], ascending=[True, False])


def build_regional_expansion_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    latest = build_latest_snapshot_df(df)
    if latest.empty:
        return pd.DataFrame()

    base = latest[["video_id", "category_name", "trending_region"]].drop_duplicates()
    region_pairs = base.merge(base, on=["video_id", "category_name"], how="inner")
    region_pairs = region_pairs[region_pairs["trending_region_x"] != region_pairs["trending_region_y"]]

    pair_counts = (
        region_pairs.groupby(["category_name", "trending_region_x", "trending_region_y"], as_index=False)
        .agg(shared_videos=("video_id", "nunique"))
    )
    base_counts = (
        base.groupby(["category_name", "trending_region"], as_index=False)
        .agg(source_videos=("video_id", "nunique"))
        .rename(columns={"trending_region": "trending_region_x"})
    )

    result = pair_counts.merge(base_counts, on=["category_name", "trending_region_x"], how="left")
    result["expansion_probability"] = result["shared_videos"] / result["source_videos"].replace(0, np.nan)
    result["expansion_probability"] = result["expansion_probability"].fillna(0)
    result = result.rename(
        columns={
            "trending_region_x": "source_region",
            "trending_region_y": "target_region",
        }
    )
    return result.sort_values("expansion_probability", ascending=False)
