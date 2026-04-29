"""Time-based holdout evaluation for the predictive models.

Each function mirrors the corresponding builder in business_analysis.py but
replays the same training logic on a temporal split:
  * Train on rows older than a cutoff
  * Predict the holdout (newer rows where the ground truth is now known)
  * Compute the metric appropriate to the prediction type

Each function returns a dict with `n_train`, `n_test`, and the metric(s).
If there isn't enough data to do a real evaluation it returns an empty dict;
the dashboard should treat that as "metric not yet available."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — bounded in [0, 200], handles zeros gracefully."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[order] == 1))


# ---------------------------------------------------------------------------
# 1. Entry probability — binary classifier
# ---------------------------------------------------------------------------
def evaluate_trending_entry_probability(df: pd.DataFrame, top_k: int = 30) -> dict:
    if df.empty:
        return {}

    work = df.sort_values(["trending_region", "collected_at", "video_id"]).copy()
    work["batch_order"] = (
        work.groupby("trending_region")["collected_at"]
        .rank(method="dense")
        .astype(int)
    )
    work["title_has_question_int"] = work["title_has_question"].astype(int)
    work["title_has_number_int"] = work["title_has_number"].astype(int)

    presence = (
        work[["video_id", "trending_region", "batch_order"]]
        .drop_duplicates()
        .assign(in_next_batch=1)
    )
    presence["batch_order"] = presence["batch_order"] - 1

    full = work.merge(
        presence,
        on=["video_id", "trending_region", "batch_order"],
        how="left",
    )
    full["in_next_batch"] = full["in_next_batch"].fillna(0).astype(int)

    feature_cols = [
        "view_count", "velocity", "video_age_hours",
        "like_rate", "comment_rate", "tag_count",
        "title_word_count", "title_has_question_int",
        "title_has_number_int", "channel_subscriber_count",
    ]

    full = full.dropna(subset=feature_cols)
    if len(full) < 30 or full["in_next_batch"].nunique() < 2:
        return {}

    # Use the most-recent complete batch per region as the holdout, so we
    # have a real label (it appeared / didn't appear in the next batch).
    max_batch = int(full["batch_order"].max())
    test_batch = max_batch - 1  # latest batch with a known forward label
    if test_batch < 1:
        return {}
    train = full[full["batch_order"] < test_batch]
    test = full[full["batch_order"] == test_batch]
    if len(train) < 20 or len(test) < 5 or train["in_next_batch"].nunique() < 2:
        return {}

    model = LogisticRegression(max_iter=1000)
    model.fit(train[feature_cols], train["in_next_batch"])
    proba = model.predict_proba(test[feature_cols])[:, 1]
    y_test = test["in_next_batch"].to_numpy()

    out = {"n_train": int(len(train)), "n_test": int(len(test))}
    if len(np.unique(y_test)) >= 2:
        try:
            out["auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            pass
    out["precision_at_k"] = _precision_at_k(y_test, proba, top_k)
    out["k"] = int(min(top_k, len(test)))
    return out


# ---------------------------------------------------------------------------
# 2. View count forecast — per-video linear regression on step
# ---------------------------------------------------------------------------
def evaluate_view_count_forecast(df: pd.DataFrame, top_n_videos: int = 8) -> dict:
    if df.empty:
        return {}

    work = (
        df.dropna(subset=["collected_at"])
        .sort_values(["video_id", "trending_region", "collected_at"])
    )
    counts = (
        work.groupby(["video_id", "trending_region"])
        .size()
        .reset_index(name="points")
    )
    # Need >=5 so we can hold out the last point and still fit on >=4.
    counts = counts[counts["points"] >= 5].head(top_n_videos)
    if counts.empty:
        return {}

    selected = work.merge(
        counts[["video_id", "trending_region"]],
        on=["video_id", "trending_region"], how="inner",
    )

    actuals, predictions = [], []
    for _, group in selected.groupby(["video_id", "trending_region"]):
        group = group.reset_index(drop=True)
        group["step"] = np.arange(len(group))
        train = group.iloc[:-1]
        test = group.iloc[[-1]]
        if len(train) < 4:
            continue
        model = LinearRegression()
        model.fit(train[["step"]], train["view_count"])
        pred = float(model.predict(test[["step"]])[0])
        actuals.append(float(test["view_count"].iloc[0]))
        predictions.append(max(0.0, pred))

    if not actuals:
        return {}

    actuals = np.array(actuals)
    predictions = np.array(predictions)
    return {
        "n_videos": int(len(actuals)),
        "smape_pct": _smape(actuals, predictions),
        "mae_views": float(mean_absolute_error(actuals, predictions)),
    }


# ---------------------------------------------------------------------------
# 3. Trending duration prediction — total trending hours per video
# ---------------------------------------------------------------------------
def evaluate_trending_duration_prediction(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    work = df.sort_values(["video_id", "trending_region", "collected_at"]).copy()
    latest_batch_id = (
        work.sort_values("collected_at").iloc[-1]["collection_batch_id"]
        if "collection_batch_id" in work.columns and not work.empty
        else None
    )

    episodes = (
        work.groupby(["video_id", "trending_region"], as_index=False)
        .agg(
            age_at_entry=("video_age_hours", "min"),
            initial_velocity=("velocity", "first"),
            initial_er=("engagement_rate", "first"),
            subscriber_count=("channel_subscriber_count", "first"),
            total_hours_trending=("collection_batch_id", lambda x: x.nunique() * 0.0125),
            last_seen=("collected_at", "max"),
            last_batch=("collection_batch_id", "last"),
        )
    )

    # Only use COMPLETED runs as ground truth — i.e. videos that didn't appear
    # in the very latest batch. Otherwise the "total hours" target is censored.
    if latest_batch_id is not None:
        episodes = episodes[episodes["last_batch"] != latest_batch_id]

    feature_cols = ["age_at_entry", "initial_velocity", "initial_er", "subscriber_count"]
    episodes = episodes.dropna(subset=feature_cols + ["total_hours_trending"])
    if len(episodes) < 20:
        return {}

    # Time-based split on when the run ended.
    episodes = episodes.sort_values("last_seen")
    cutoff_idx = int(len(episodes) * 0.8)
    train = episodes.iloc[:cutoff_idx]
    test = episodes.iloc[cutoff_idx:]
    if len(train) < 10 or len(test) < 5:
        return {}

    model = LinearRegression()
    model.fit(train[feature_cols], train["total_hours_trending"])
    pred = np.clip(model.predict(test[feature_cols]), 0, None)
    actual = test["total_hours_trending"].to_numpy()

    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae_hours": float(mean_absolute_error(actual, pred)),
        "mean_actual_hours": float(np.mean(actual)),
    }


# ---------------------------------------------------------------------------
# 4. Peak rank forecast — predict best (lowest) rank achieved per video
# ---------------------------------------------------------------------------
def evaluate_peak_rank_forecast(df: pd.DataFrame, within_n_ranks: int = 3) -> dict:
    if df.empty:
        return {}

    work = df.sort_values(["video_id", "trending_region", "collected_at"]).copy()
    work["prev_rank"] = work.groupby(["video_id", "trending_region"])["trending_rank"].shift(1)
    work["rank_delta"] = (work["prev_rank"] - work["trending_rank"]).fillna(0)

    episodes = (
        work.groupby(["video_id", "trending_region"], as_index=False)
        .agg(
            current_rank=("trending_rank", "last"),
            avg_rank_delta=("rank_delta", "mean"),
            current_velocity=("velocity", "last"),
            current_er=("engagement_rate", "last"),
            best_rank_achieved=("trending_rank", "min"),
            observations=("collection_batch_id", "nunique"),
            last_seen=("collected_at", "max"),
        )
    )

    feature_cols = ["current_rank", "avg_rank_delta", "current_velocity", "current_er", "observations"]
    episodes = episodes.dropna(subset=feature_cols + ["best_rank_achieved"])
    # Need at least a few observations for the "best so far" to be meaningful.
    episodes = episodes[episodes["observations"] >= 3]
    if len(episodes) < 20:
        return {}

    episodes = episodes.sort_values("last_seen")
    cutoff_idx = int(len(episodes) * 0.8)
    train = episodes.iloc[:cutoff_idx]
    test = episodes.iloc[cutoff_idx:]
    if len(train) < 10 or len(test) < 5:
        return {}

    model = LinearRegression()
    model.fit(train[feature_cols], train["best_rank_achieved"])
    pred = np.clip(model.predict(test[feature_cols]), 1, None)
    actual = test["best_rank_achieved"].to_numpy()

    abs_err = np.abs(pred - actual)
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae_ranks": float(mean_absolute_error(actual, pred)),
        f"pct_within_{within_n_ranks}": float(np.mean(abs_err <= within_n_ranks) * 100.0),
    }


# ---------------------------------------------------------------------------
# 5. Category share forecast — projects view share over 5-min buckets
# ---------------------------------------------------------------------------
def evaluate_category_share_forecast(df: pd.DataFrame, holdout_buckets: int = 4, top_n_categories: int = 6) -> dict:
    if df.empty:
        return {}

    work = df.dropna(subset=["collected_at"]).copy()
    work["time_bucket"] = work["collected_at"].dt.floor("5min")

    grouped = (
        work.groupby(["category_name", "time_bucket"], as_index=False)
        .agg(total_views=("view_count", "sum"))
    )
    if grouped.empty:
        return {}

    totals = grouped.groupby("time_bucket", as_index=False).agg(bucket_total=("total_views", "sum"))
    grouped = grouped.merge(totals, on="time_bucket", how="left")
    grouped["view_share"] = (grouped["total_views"] / grouped["bucket_total"].replace(0, pd.NA)).fillna(0)

    top_categories = (
        grouped.groupby("category_name")["total_views"].sum()
        .sort_values(ascending=False).head(top_n_categories).index.tolist()
    )
    grouped = grouped[grouped["category_name"].isin(top_categories)].copy()

    actuals, predictions = [], []
    for _, group in grouped.groupby("category_name"):
        group = group.sort_values("time_bucket").reset_index(drop=True)
        # Need enough buckets to train AND hold out.
        if len(group) < holdout_buckets + 4:
            continue
        group["step"] = np.arange(len(group))
        train = group.iloc[:-holdout_buckets]
        test = group.iloc[-holdout_buckets:]
        model = LinearRegression()
        model.fit(train[["step"]], train["view_share"])
        pred = np.clip(model.predict(test[["step"]]), 0, 1)
        actuals.extend(test["view_share"].tolist())
        predictions.extend(pred.tolist())

    if not actuals:
        return {}

    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)
    return {
        "n_test": int(len(actuals)),
        "mae_share_pct": float(mean_absolute_error(actuals, predictions) * 100.0),
        "smape_pct": _smape(actuals, predictions),
    }
