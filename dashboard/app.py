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
    # prepare_dashboard_df normalises raw Silver rows for the predictive /
    # prescriptive ML pipelines below. Descriptive aggregations no longer
    # need it because they aggregate in Spark.
    prepare_dashboard_df,
    # The following builders are only used by tab 2 (Predictive) and tab 3
    # (Prescriptive). They wrap sklearn / regression code that operates on
    # numpy/pandas arrays, so they remain pandas-backed.
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
from analytics.predictive_metrics import (
    evaluate_category_share_forecast,
    evaluate_peak_rank_forecast,
    evaluate_trending_duration_prediction,
    evaluate_trending_entry_probability,
    evaluate_view_count_forecast,
)
from analytics.spark_descriptive import (
    build_spark_category_summary,
    build_spark_duration_distribution,
    build_spark_subscriber_tier_distribution,
    build_spark_tag_usage_frequency,
    build_spark_trending_rank_distribution,
    build_spark_views_timeseries,
)
from analytics.spark_diagnostics import (
    build_channel_mix_shift_diagnostic,
    build_duration_category_overindex_diagnostic,
    build_duration_engagement_diagnostic,
    build_duration_region_diagnostic,
    build_duration_velocity_diagnostic,
    build_engagement_shift_diagnostic,
    build_category_share_drivers_diagnostic,
    build_high_engagement_tags_diagnostic,
    build_persistence_distribution_diagnostic,
    build_rank_new_vs_persisting_diagnostic,
    build_subscriber_engagement_diagnostic,
    build_subscriber_persistence_diagnostic,
    build_subscriber_region_diagnostic,
    build_subscriber_views_diagnostic,
    build_tag_density_performance_diagnostic,
    build_tag_effectiveness_diagnostic,
    build_tag_region_concentration_diagnostic,
    build_top_rank_channel_concentration_diagnostic,
    build_velocity_shift_diagnostic,
    build_velocity_vs_rank_diagnostic,
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


def _extract_chart_selection(event_state, selection_name: str, expected_fields: list[str]) -> dict[str, object]:
    if not event_state or "selection" not in event_state:
        return {}

    selection_state = event_state.selection.get(selection_name, {})
    if not selection_state:
        return {}

    def _walk(node):
        found: dict[str, object] = {}
        if isinstance(node, dict):
            for field_name in expected_fields:
                if field_name in node and node[field_name] not in (None, [], ""):
                    value = node[field_name]
                    found[field_name] = value[0] if isinstance(value, list) and len(value) == 1 else value
            for value in node.values():
                if isinstance(value, (dict, list)):
                    nested = _walk(value)
                    for nested_key, nested_value in nested.items():
                        found.setdefault(nested_key, nested_value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    nested = _walk(item)
                    for nested_key, nested_value in nested.items():
                        found.setdefault(nested_key, nested_value)
        return found

    return _walk(selection_state)


def _normalize_context_value(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


# ---------------------------------------------------------------------------
# Diagnostic explanation helpers
# ---------------------------------------------------------------------------
# Each diagnostic chart calls a small explainer that inspects the dataframe
# and produces a sentence describing the actual shape of the data. The
# explanation must stay valid as the trend changes (rising / falling /
# stable / dominated by X / concentrated / diversified / ...), so we always
# look at the data rather than hard-coding an interpretation.

def _explain_trend_direction(values: list[float]) -> str:
    """Classify a numeric series as rising / falling / stable / volatile."""
    clean = [v for v in values if v is not None and not pd.isna(v)]
    if len(clean) < 2:
        return "flat"
    first = clean[0]
    last = clean[-1]
    if first == 0:
        # Avoid divide-by-zero; fall back to absolute delta.
        delta_pct = 0 if last == 0 else (1.0 if last > 0 else -1.0)
    else:
        delta_pct = (last - first) / abs(first)
    peak = max(clean)
    trough = min(clean)
    swing = (peak - trough) / abs(peak) if peak else 0
    if abs(delta_pct) < 0.05 and swing < 0.25:
        return "stable"
    if abs(delta_pct) < 0.05 and swing >= 0.25:
        return "volatile"
    return "rising" if delta_pct > 0 else "falling"


def _pct_delta(first: float, last: float) -> str:
    if first is None or last is None or pd.isna(first) or pd.isna(last):
        return "n/a"
    if first == 0:
        return "n/a"
    pct = (last - first) / abs(first) * 100
    return f"{pct:+.1f}%"


def _fmt_int(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(round(float(value))):,}"


def _fmt_pct(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.{digits}f}%"


def _fmt_num(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _explanation(text: str) -> None:
    """Render an inline explanation under a diagnostic chart."""
    st.markdown(
        f"<div style='font-size:0.85rem; color:#475569; "
        f"background:#f1f5f9; padding:0.5rem 0.75rem; "
        f"border-left:3px solid #6366f1; border-radius:4px; "
        f"margin-top:-0.4rem; margin-bottom:0.6rem;'>"
        f"<b>What this chart says:</b> {text}</div>",
        unsafe_allow_html=True,
    )


def _model_metrics_strip(label: str, metrics: dict) -> None:
    """Render a small accuracy strip under a predictive chart.

    `metrics` is the dict returned by an evaluate_* function. If it's empty
    or missing keys, we render a 'not enough data yet' note so users know
    why no number is shown.
    """
    if not metrics:
        st.markdown(
            f"<div style='font-size:0.8rem; color:#94a3b8; "
            f"background:#f8fafc; padding:0.4rem 0.7rem; "
            f"border-left:3px solid #cbd5e1; border-radius:4px; "
            f"margin-top:-0.4rem; margin-bottom:0.6rem;'>"
            f"<b>{label} accuracy:</b> not enough holdout data yet — "
            f"keep streaming and the metric will populate.</div>",
            unsafe_allow_html=True,
        )
        return

    parts = []
    if "auc" in metrics:
        parts.append(f"ROC-AUC <b>{metrics['auc']:.3f}</b>")
    if "precision_at_k" in metrics:
        parts.append(
            f"Precision@{metrics.get('k', '?')} "
            f"<b>{metrics['precision_at_k'] * 100:.1f}%</b>"
        )
    if "smape_pct" in metrics:
        parts.append(f"sMAPE <b>{metrics['smape_pct']:.1f}%</b>")
    if "mae_views" in metrics:
        parts.append(f"MAE <b>{int(metrics['mae_views']):,}</b> views")
    if "mae_hours" in metrics:
        parts.append(
            f"MAE <b>{metrics['mae_hours']:.2f} h</b> "
            f"(mean actual {metrics['mean_actual_hours']:.2f} h)"
        )
    if "mae_ranks" in metrics:
        parts.append(f"MAE <b>{metrics['mae_ranks']:.2f} ranks</b>")
    for k, v in metrics.items():
        if k.startswith("pct_within_"):
            n = k.split("_")[-1]
            parts.append(f"{v:.1f}% within ±{n} ranks")
    if "mae_share_pct" in metrics:
        parts.append(f"MAE <b>{metrics['mae_share_pct']:.2f} pp</b> share")

    sample = []
    if "n_train" in metrics:
        sample.append(f"train n={metrics['n_train']:,}")
    if "n_test" in metrics:
        sample.append(f"test n={metrics['n_test']:,}")
    if "n_videos" in metrics:
        sample.append(f"videos held-out={metrics['n_videos']:,}")
    sample_str = " • ".join(sample) if sample else ""

    body = " • ".join(parts) if parts else "no metric available"
    if sample_str:
        body = f"{body}<span style='color:#94a3b8'> &nbsp;|&nbsp; {sample_str}</span>"

    st.markdown(
        f"<div style='font-size:0.85rem; color:#0f172a; "
        f"background:#ecfeff; padding:0.5rem 0.75rem; "
        f"border-left:3px solid #06b6d4; border-radius:4px; "
        f"margin-top:-0.4rem; margin-bottom:0.6rem;'>"
        f"<b>{label} accuracy (time-based holdout):</b> {body}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=120, show_spinner=False)
def load_entry_probability_metrics(_df: pd.DataFrame) -> dict:
    return evaluate_trending_entry_probability(_df)


@st.cache_data(ttl=120, show_spinner=False)
def load_view_forecast_metrics(_df: pd.DataFrame) -> dict:
    return evaluate_view_count_forecast(_df)


@st.cache_data(ttl=120, show_spinner=False)
def load_duration_prediction_metrics(_df: pd.DataFrame) -> dict:
    return evaluate_trending_duration_prediction(_df)


@st.cache_data(ttl=120, show_spinner=False)
def load_peak_rank_metrics(_df: pd.DataFrame) -> dict:
    return evaluate_peak_rank_forecast(_df)


@st.cache_data(ttl=120, show_spinner=False)
def load_category_share_metrics(_df: pd.DataFrame) -> dict:
    return evaluate_category_share_forecast(_df)


def _explain_velocity_shift(df) -> str:
    if df.empty or "avg_velocity" not in df.columns:
        return "Not enough data to interpret the trend."
    series = df.sort_values("time_bucket")["avg_velocity"].tolist()
    direction = _explain_trend_direction(series)
    delta = _pct_delta(series[0] if series else None, series[-1] if series else None)
    if direction == "rising":
        return f"Velocity is **rising** ({delta} from start to end), so videos in this category are gaining views faster over time — the category is heating up."
    if direction == "falling":
        return f"Velocity is **falling** ({delta}), so videos are accumulating views more slowly — momentum is cooling off."
    if direction == "volatile":
        return "Velocity is **swinging** sharply between buckets — the category has bursty traction rather than a steady trend."
    return f"Velocity is **stable** ({delta}) — the category's traction is holding steady, neither accelerating nor decelerating."


def _explain_persistence_distribution(df) -> str:
    if df.empty or "video_count" not in df.columns:
        return "Not enough data to interpret persistence."
    total = df["video_count"].sum()
    if total == 0:
        return "No videos in scope."
    one_shot = df.loc[df["persistence_bucket"] == "1 batch", "video_count"].sum()
    long_lived = df.loc[df["persistence_bucket"] == "7+ batches", "video_count"].sum()
    one_shot_pct = one_shot / total
    long_lived_pct = long_lived / total
    if one_shot_pct > 0.5:
        return (
            f"**{_fmt_pct(one_shot_pct)}** of videos appear in trending only once — "
            "this category is mostly **one-shot wonders**, with little staying power."
        )
    if long_lived_pct > 0.25:
        return (
            f"**{_fmt_pct(long_lived_pct)}** of videos persist 7+ batches — "
            "this category has a **long-lived** trending core that holds attention."
        )
    return (
        f"Persistence is **mixed**: {_fmt_pct(one_shot_pct)} one-shot, "
        f"{_fmt_pct(long_lived_pct)} long-lived — the category has both transient and durable hits."
    )


def _explain_engagement_shift(df) -> str:
    if df.empty or "avg_engagement_rate" not in df.columns:
        return "Not enough data to interpret engagement."
    overall = (
        df.groupby("time_bucket", as_index=False)["avg_engagement_rate"]
        .mean()
        .sort_values("time_bucket")
    )
    series = overall["avg_engagement_rate"].tolist()
    direction = _explain_trend_direction(series)
    delta = _pct_delta(series[0] if series else None, series[-1] if series else None)
    region_note = ""
    if "trending_region" in df.columns and df["trending_region"].nunique() > 1:
        per_region = (
            df.groupby("trending_region")["avg_engagement_rate"]
            .mean()
            .sort_values(ascending=False)
        )
        if not per_region.empty:
            region_note = (
                f" Strongest region right now is **{per_region.index[0]}** "
                f"at {_fmt_pct(per_region.iloc[0], 2)}."
            )
    if direction == "rising":
        return f"Engagement rate is **rising** ({delta}) — viewers are interacting more per view.{region_note}"
    if direction == "falling":
        return f"Engagement rate is **falling** ({delta}) — viewers are watching but interacting less.{region_note}"
    if direction == "volatile":
        return f"Engagement rate is **volatile** across buckets — no clean trend yet.{region_note}"
    return f"Engagement rate is **stable** ({delta}) around {_fmt_pct(series[-1] if series else None, 2)}.{region_note}"


def _explain_channel_mix_shift(df) -> str:
    if df.empty or "share" not in df.columns:
        return "Not enough data to interpret channel mix."
    pivot = (
        df.pivot_table(
            index="time_bucket",
            columns="subscriber_tier",
            values="share",
            aggfunc="mean",
        )
        .fillna(0)
        .sort_index()
    )
    if pivot.empty:
        return "No channel-mix data."
    first = pivot.iloc[0]
    last = pivot.iloc[-1]
    deltas = (last - first).sort_values(ascending=False)
    if deltas.empty:
        return "No tier changes detected."
    rising_tier = deltas.index[0]
    rising_delta = deltas.iloc[0]
    falling_tier = deltas.index[-1]
    falling_delta = deltas.iloc[-1]
    dominant_tier = last.idxmax() if not last.empty else "n/a"
    dominant_share = last.max() if not last.empty else 0
    if abs(rising_delta) < 0.02 and abs(falling_delta) < 0.02:
        return (
            f"Channel-size mix is **steady** — **{dominant_tier}** carries about "
            f"{_fmt_pct(dominant_share)} of trending and isn't budging."
        )
    return (
        f"**{rising_tier}** creators are gaining share (**{_fmt_pct(rising_delta)}** swing), "
        f"while **{falling_tier}** are losing ground ({_fmt_pct(falling_delta)}). "
        f"Right now **{dominant_tier}** dominates at {_fmt_pct(dominant_share)} of trending."
    )


def _explain_duration_engagement(df) -> str:
    if df.empty or "avg_engagement_rate" not in df.columns:
        return "Not enough data to interpret duration vs engagement."
    df_sorted = df.sort_values("avg_engagement_rate", ascending=False)
    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]
    spread = float(best["avg_engagement_rate"]) - float(worst["avg_engagement_rate"])
    if spread < 0.005:
        return (
            f"Engagement is **flat across duration buckets** (spread "
            f"{_fmt_pct(spread, 2)}) — duration isn't really shaping engagement here."
        )
    return (
        f"**{best['duration_bucket']}** wins on engagement at "
        f"{_fmt_pct(best['avg_engagement_rate'], 2)}, vs {worst['duration_bucket']} at "
        f"{_fmt_pct(worst['avg_engagement_rate'], 2)} — a {_fmt_pct(spread, 2)} gap."
    )


def _explain_duration_velocity(df) -> str:
    if df.empty or "avg_velocity" not in df.columns:
        return "Not enough data to interpret velocity by duration."
    df_sorted = df.sort_values("avg_velocity", ascending=False)
    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]
    if float(worst["avg_velocity"]) <= 0:
        ratio_text = "much higher than"
    else:
        ratio = float(best["avg_velocity"]) / float(worst["avg_velocity"])
        ratio_text = f"**{ratio:.1f}x** the velocity of"
    return (
        f"**{best['duration_bucket']}** videos rack up views fastest "
        f"({_fmt_num(best['avg_velocity'])} avg velocity) — {ratio_text} "
        f"the slowest bucket ({worst['duration_bucket']})."
    )


def _explain_duration_overindex(df) -> str:
    if df.empty or "lift" not in df.columns:
        return "Not enough data to interpret over-indexing."
    over = df[df["lift"] > 1.25].sort_values("lift", ascending=False)
    under = df[df["lift"] < 0.75].sort_values("lift")
    if over.empty and under.empty:
        return (
            "Categories are spread roughly evenly across duration buckets — "
            "no strong over- or under-indexing."
        )
    parts = []
    if not over.empty:
        top = over.iloc[0]
        parts.append(
            f"**{top['category_name']}** over-indexes in **{top['duration_bucket']}** "
            f"(lift {top['lift']:.2f}x)"
        )
    if not under.empty:
        bot = under.iloc[0]
        parts.append(
            f"**{bot['category_name']}** under-indexes in **{bot['duration_bucket']}** "
            f"(lift {bot['lift']:.2f}x)"
        )
    return "; ".join(parts) + "."


def _explain_duration_region(df) -> str:
    if df.empty or "avg_engagement_rate" not in df.columns:
        return "Not enough data."
    best = df.sort_values("avg_engagement_rate", ascending=False).iloc[0]
    worst = df.sort_values("avg_engagement_rate", ascending=True).iloc[0]
    return (
        f"Best engagement combo: **{best['duration_bucket']}** in **{best['trending_region']}** "
        f"({_fmt_pct(best['avg_engagement_rate'], 2)}). Weakest: "
        f"**{worst['duration_bucket']}** in **{worst['trending_region']}** "
        f"({_fmt_pct(worst['avg_engagement_rate'], 2)})."
    )


def _explain_tag_frequency_engagement(df) -> str:
    if df.empty or len(df) < 3:
        return "Not enough tags to detect a pattern."
    try:
        corr = df["videos_using_tag"].corr(df["avg_engagement_rate"])
    except Exception:
        corr = None
    if corr is None or pd.isna(corr):
        return "Frequency vs engagement relationship is unclear."
    if corr > 0.3:
        return (
            f"Frequency and engagement move **together** (corr {corr:+.2f}) — "
            "the most-used tags are also among the higher-engagement ones."
        )
    if corr < -0.3:
        return (
            f"Frequency and engagement move **inversely** (corr {corr:+.2f}) — "
            "high-frequency tags are actually under-performing on engagement."
        )
    return (
        f"Frequency and engagement are **largely uncorrelated** (corr {corr:+.2f}) — "
        "popular tags aren't a reliable signal of engagement on their own."
    )


def _explain_high_engagement_tags(df) -> str:
    if df.empty:
        return "No qualifying tags."
    top = df.sort_values("avg_engagement_rate", ascending=False).head(3)
    parts = [
        f"**{row['tag']}** ({_fmt_pct(row['avg_engagement_rate'], 2)})"
        for _, row in top.iterrows()
    ]
    return f"Top tags by engagement: {', '.join(parts)}."


def _explain_tag_region_concentration(df) -> str:
    if df.empty or "region_concentration" not in df.columns:
        return "Not enough data."
    by_tag = df.drop_duplicates("tag")[["tag", "region_concentration"]]
    region_specific = by_tag[by_tag["region_concentration"] >= 0.7]
    global_tags = by_tag[by_tag["region_concentration"] <= 0.4]
    most_specific = by_tag.sort_values("region_concentration", ascending=False).iloc[0]
    if not region_specific.empty and not global_tags.empty:
        return (
            f"**{len(region_specific)}** tags are highly region-specific (>=70% in one region) "
            f"and **{len(global_tags)}** behave globally (<=40% concentration). "
            f"Most concentrated: **{most_specific['tag']}** "
            f"({_fmt_pct(most_specific['region_concentration'])})."
        )
    if not region_specific.empty:
        return (
            f"Most tags lean **region-specific** — "
            f"**{len(region_specific)}** of {len(by_tag)} have >=70% concentration in one region."
        )
    return (
        "Most tags here behave **globally** — usage is spread fairly evenly across regions."
    )


def _explain_tag_density(df) -> str:
    if df.empty:
        return "Not enough data."
    order = ["0 tags", "1-5 tags", "6-10 tags", "11-20 tags", "21+ tags"]
    df = df.copy()
    df["__order"] = df["tag_density_bucket"].map({k: i for i, k in enumerate(order)})
    df = df.sort_values("__order")
    best = df.sort_values("avg_engagement_rate", ascending=False).iloc[0]
    series = df["avg_engagement_rate"].tolist()
    direction = _explain_trend_direction(series)
    if direction == "rising":
        shape = "increases as videos add more tags"
    elif direction == "falling":
        shape = "decreases as videos add more tags"
    elif direction == "volatile":
        shape = "moves unevenly across tag-density buckets"
    else:
        shape = "is roughly flat across tag-density buckets"
    return (
        f"Engagement {shape}. Sweet spot is **{best['tag_density_bucket']}** at "
        f"{_fmt_pct(best['avg_engagement_rate'], 2)}."
    )


def _explain_subscriber_views(df) -> str:
    if df.empty:
        return "Not enough data."
    best = df.sort_values("avg_views", ascending=False).iloc[0]
    worst = df.sort_values("avg_views", ascending=True).iloc[0]
    if float(worst["avg_views"]) <= 0:
        ratio = "many times"
    else:
        ratio = f"{float(best['avg_views']) / float(worst['avg_views']):.1f}x"
    return (
        f"**{best['subscriber_tier']}** channels pull the most views per video "
        f"({_fmt_int(best['avg_views'])}) — about **{ratio}** the **{worst['subscriber_tier']}** average."
    )


def _explain_subscriber_engagement(df) -> str:
    if df.empty:
        return "Not enough data."
    best = df.sort_values("avg_engagement_rate", ascending=False).iloc[0]
    worst = df.sort_values("avg_engagement_rate", ascending=True).iloc[0]
    spread = float(best["avg_engagement_rate"]) - float(worst["avg_engagement_rate"])
    if spread < 0.005:
        return "Engagement is **roughly equal across tiers** — channel size doesn't drive engagement here."
    return (
        f"**{best['subscriber_tier']}** earns the highest engagement "
        f"({_fmt_pct(best['avg_engagement_rate'], 2)}), versus **{worst['subscriber_tier']}** at "
        f"{_fmt_pct(worst['avg_engagement_rate'], 2)} — a {_fmt_pct(spread, 2)} gap."
    )


def _explain_subscriber_persistence(df) -> str:
    if df.empty:
        return "Not enough data."
    best = df.sort_values("avg_batches", ascending=False).iloc[0]
    worst = df.sort_values("avg_batches", ascending=True).iloc[0]
    return (
        f"**{best['subscriber_tier']}** videos stay trending longest "
        f"(avg **{_fmt_num(best['avg_batches'])}** batches), while "
        f"**{worst['subscriber_tier']}** churn fastest at "
        f"{_fmt_num(worst['avg_batches'])} batches."
    )


def _explain_subscriber_region(df) -> str:
    if df.empty or "tier_share" not in df.columns:
        return "Not enough data."
    top = df.sort_values("tier_share", ascending=False).iloc[0]
    return (
        f"**{top['trending_region']}** leans most heavily on **{top['subscriber_tier']}** creators "
        f"({_fmt_pct(top['tier_share'])} of that region's trending). Use this to spot regional "
        f"creator-size biases."
    )


def _explain_new_vs_persisting(df) -> str:
    if df.empty or "videos" not in df.columns:
        return "Not enough data."
    pivot = (
        df.pivot_table(
            index="time_bucket",
            columns="entry_status",
            values="videos",
            aggfunc="sum",
        )
        .fillna(0)
        .sort_index()
    )
    if pivot.empty:
        return "No data."
    new_series = pivot.get("New entry", pd.Series(dtype=float)).tolist()
    pers_series = pivot.get("Persisting", pd.Series(dtype=float)).tolist()
    new_total = sum(new_series)
    pers_total = sum(pers_series)
    grand = new_total + pers_total
    if grand == 0:
        return "No videos in scope."
    new_share = new_total / grand
    new_dir = _explain_trend_direction(new_series) if new_series else "flat"
    if new_share > 0.6:
        flavor = "fed by **fresh entries** — most slots are new arrivals."
    elif new_share < 0.3:
        flavor = "carried by **persisting videos** — the same titles keep showing up."
    else:
        flavor = "a **balanced mix** of fresh entries and persisting videos."
    return f"This category is {flavor} New-entry volume is **{new_dir}** over the window."


def _explain_top_rank_concentration(df) -> str:
    if df.empty or "top_rank_slots" not in df.columns:
        return "Not enough data."
    total = float(df["top_rank_slots"].sum())
    if total == 0:
        return "No top-rank slots in scope."
    top = df.sort_values("top_rank_slots", ascending=False).iloc[0]
    top_share = float(top["top_rank_slots"]) / total
    top3_share = float(df.sort_values("top_rank_slots", ascending=False).head(3)["top_rank_slots"].sum()) / total
    if top3_share > 0.6:
        return (
            f"Top-rank slots are **highly concentrated** — top 3 channels hold "
            f"{_fmt_pct(top3_share)}, led by **{top['channel_title']}** at {_fmt_pct(top_share)}."
        )
    if top3_share < 0.35:
        return (
            f"Top-rank slots are **broadly distributed** — top 3 channels hold only "
            f"{_fmt_pct(top3_share)}; no single channel dominates."
        )
    return (
        f"Top-rank distribution is **moderately concentrated** — top 3 channels hold "
        f"{_fmt_pct(top3_share)}, with **{top['channel_title']}** leading at {_fmt_pct(top_share)}."
    )


def _explain_velocity_vs_rank(df) -> str:
    if df.empty or len(df) < 3:
        return "Not enough videos to detect a pattern."
    try:
        # Negative correlation = high velocity → low (better) rank.
        corr = df["avg_velocity"].corr(df["best_rank"])
    except Exception:
        corr = None
    if corr is None or pd.isna(corr):
        return "Velocity vs rank relationship is unclear."
    if corr < -0.3:
        return (
            f"Velocity strongly drives rank (corr {corr:+.2f}) — "
            "higher-velocity videos consistently land **better** trending ranks."
        )
    if corr > 0.3:
        return (
            f"Velocity and rank move **against expectation** (corr {corr:+.2f}) — "
            "high-velocity videos aren't translating into top ranks here."
        )
    return (
        f"Velocity has a **weak link** to rank (corr {corr:+.2f}) — other factors "
        "(timing, channel size, topic) are doing more of the work."
    )


def _explain_share_drivers(df) -> str:
    if df.empty:
        return "Not enough data."
    df = df.sort_values("time_bucket")
    videos_dir = _explain_trend_direction(df["unique_videos"].tolist())
    avg_views_dir = _explain_trend_direction(df["avg_views_per_video"].tolist())
    if videos_dir == "rising" and avg_views_dir == "rising":
        return "Share is rising on **both fronts** — more videos AND each video is getting bigger."
    if videos_dir == "rising" and avg_views_dir != "rising":
        return "Growth is **volume-led** — more videos are entering trending, not stronger individual performances."
    if videos_dir != "rising" and avg_views_dir == "rising":
        return "Growth is **strength-led** — same number of videos, but each one is pulling more views."
    if videos_dir == "falling" and avg_views_dir == "falling":
        return "Share is **shrinking on both fronts** — fewer videos and weaker individual reach."
    return (
        f"Volume is {videos_dir}, per-video reach is {avg_views_dir} — "
        "no single driver is dominating right now."
    )


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


# ---------------------------------------------------------------------------
# Spark-backed descriptive loaders
# ---------------------------------------------------------------------------
# These run their aggregations in Spark against the Silver Delta layer and
# return a small pandas DataFrame only at the end (so Streamlit/Altair can
# render). No row-by-row pandas processing happens for descriptive charts.

@st.cache_data(ttl=60, show_spinner=False)
def load_spark_category_summary(category_name, trending_region, history_window):
    sdf = build_spark_category_summary(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_spark_views_timeseries(category_name, trending_region, history_window):
    sdf = build_spark_views_timeseries(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_spark_duration_distribution(category_name, trending_region, history_window):
    sdf = build_spark_duration_distribution(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_spark_subscriber_tier_distribution(category_name, trending_region, history_window):
    sdf = build_spark_subscriber_tier_distribution(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_spark_tag_usage_frequency(category_name, trending_region, history_window):
    sdf = build_spark_tag_usage_frequency(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_spark_trending_rank_distribution(category_name, trending_region, history_window):
    sdf = build_spark_trending_rank_distribution(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_spark_kpis(category_name, trending_region, history_window):
    """Headline KPIs computed entirely in Spark (no pandas processing)."""
    spark = get_spark()
    sdf = spark.read.format("delta").load(SILVER_DELTA_PATH)
    if category_name:
        sdf = sdf.filter(col("category_name") == category_name)
    if trending_region:
        sdf = sdf.filter(col("trending_region") == trending_region)
    window_days = {"Last 24 Hours": 1, "Last 7 Days": 7, "Last 30 Days": 30, "All Available": None}
    days = window_days.get(history_window)
    if days is not None and "collected_at_ts" in sdf.columns:
        sdf = sdf.filter(col("collected_at_ts") >= expr(f"current_timestamp() - INTERVAL {days} DAYS"))

    # Latest row per (video_id, trending_region) within the window — same
    # semantics as the previous pandas drop_duplicates(keep='last').
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, desc, sum as F_sum, avg as F_avg, countDistinct
    window = Window.partitionBy("video_id", "trending_region").orderBy(desc("collected_at_ts"))
    latest = sdf.withColumn("__rn", row_number().over(window)).filter(col("__rn") == 1)

    row = (
        latest.agg(
            countDistinct("video_id").alias("unique_videos"),
            F_sum("view_count").alias("total_views"),
            F_avg("engagement_rate").alias("avg_engagement_rate"),
            countDistinct("category_name").alias("tracked_categories"),
        )
        .collect()
    )
    if not row:
        return {"unique_videos": 0, "total_views": 0, "avg_engagement_rate": 0.0, "tracked_categories": 0}
    r = row[0].asDict()
    # Replace None with safe defaults so the metric formatter never blows up.
    return {
        "unique_videos": int(r["unique_videos"] or 0),
        "total_views": int(r["total_views"] or 0),
        "avg_engagement_rate": float(r["avg_engagement_rate"] or 0.0),
        "tracked_categories": int(r["tracked_categories"] or 0),
    }


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


@st.cache_data(ttl=60, show_spinner=False)
def load_velocity_shift_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_velocity_shift_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_persistence_distribution_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_persistence_distribution_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_engagement_shift_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_engagement_shift_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_channel_mix_shift_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_channel_mix_shift_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_duration_engagement_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_duration_engagement_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_duration_velocity_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_duration_velocity_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_duration_category_overindex_diagnostic(trending_region: str | None, history_window: str):
    sdf = build_duration_category_overindex_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_duration_region_diagnostic(category_name: str | None, history_window: str):
    sdf = build_duration_region_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_subscriber_views_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_subscriber_views_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_subscriber_engagement_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_subscriber_engagement_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_subscriber_persistence_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_subscriber_persistence_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_subscriber_region_diagnostic(category_name: str | None, history_window: str):
    sdf = build_subscriber_region_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_rank_new_vs_persisting_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_rank_new_vs_persisting_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_top_rank_channel_concentration_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_top_rank_channel_concentration_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_velocity_vs_rank_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_velocity_vs_rank_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_category_share_drivers_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_category_share_drivers_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_tag_effectiveness_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_tag_effectiveness_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_high_engagement_tags_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_high_engagement_tags_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_tag_region_concentration_diagnostic(category_name: str | None, history_window: str):
    sdf = build_tag_region_concentration_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=60, show_spinner=False)
def load_tag_density_performance_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_tag_density_performance_diagnostic(
        get_spark(),
        SILVER_DELTA_PATH,
        history_window,
        category_name=category_name,
        trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


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
# Spark-side filter values: None means "no filter", anything else is a literal match.
diagnostic_category = None if selected_category == "All" else selected_category
diagnostic_region = None if selected_region == "All" else selected_region
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
# --- Spark-driven descriptive aggregations -------------------------------
# All tab-1 (descriptive) tables are computed by reading the Silver Delta
# layer directly with Spark and aggregating there. Pandas is used ONLY at
# the very last step (via _safe_pandas_from_spark) because Streamlit /
# Altair render via pandas/Arrow.
summary_df = (
    gold_category_summary_df
    if use_gold_category_summary
    else load_spark_category_summary(diagnostic_category, diagnostic_region, history_window)
)

use_gold_views_ts = (
    not gold_views_timeseries_df.empty and selected_region == "All" and selected_category == "All"
)
views_ts_df = (
    gold_views_timeseries_df
    if use_gold_views_ts
    else load_spark_views_timeseries(diagnostic_category, diagnostic_region, history_window)
)

duration_dist_df = load_spark_duration_distribution(diagnostic_category, diagnostic_region, history_window)
subscriber_tier_df = load_spark_subscriber_tier_distribution(diagnostic_category, diagnostic_region, history_window)
tag_usage_df = load_spark_tag_usage_frequency(diagnostic_category, diagnostic_region, history_window)
trending_rank_dist_df = load_spark_trending_rank_distribution(diagnostic_category, diagnostic_region, history_window)

# --- Predictive / prescriptive pipeline (still pandas-backed) ------------
# These wrap sklearn / regression code that genuinely needs numpy/pandas
# arrays, so the legacy pandas-based builders remain for tab 2 and tab 3.
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


# KPI metrics — computed entirely in Spark.
kpis = load_spark_kpis(diagnostic_category, diagnostic_region, history_window)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Unique Videos", f"{kpis['unique_videos']:,}")
k2.metric("Total Views", f"{kpis['total_views']:,}")
k3.metric("Avg Engagement Rate", f"{kpis['avg_engagement_rate'] * 100:.2f}%")
k4.metric("Tracked Categories", f"{kpis['tracked_categories']:,}")

tab1, tab2, tab3 = st.tabs(
    ["Descriptive", "Predictive", "Prescriptive"]
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
    # st.altair_chart(category_chart, width="stretch")

    st.subheader("Views Trend Over Time by Category")
    st.markdown("Analytical question: Which categories are sustaining momentum over time?")

    if views_ts_df.empty:
        st.info("Not enough timestamped data yet for time-series trend analysis.")
    else:
        views_select = alt.selection_point(
            name="views_trend_select",
            fields=["category_name"],
            on="click",
            toggle=False,
            empty=False,
        )
        views_legend_select = alt.selection_point(
            name="views_trend_legend_select",
            fields=["category_name"],
            bind="legend",
            on="click",
            toggle=False,
            empty=False,
        )
        views_base = alt.Chart(views_ts_df).encode(
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
        views_lines_layer = views_base.mark_line(point=True).encode(
            opacity=alt.condition(
                views_select | views_legend_select,
                alt.value(1),
                alt.value(0.45),
            ),
            strokeWidth=alt.condition(
                views_select | views_legend_select,
                alt.value(3.5),
                alt.value(1.8),
            ),
        )
        # Wide, transparent line layer so clicks anywhere along a category's
        # trace register a selection (the visible line/points are thin and
        # easy to miss). This is the actual selectable target.
        views_click_layer = views_base.mark_line(
            opacity=0.001,
            strokeWidth=18,
            interpolate="linear",
        ).add_params(views_select, views_legend_select)
        views_line = (views_lines_layer + views_click_layer).properties(height=350)
        st.caption("Click a category line (or its legend entry) to open Spark diagnostics from the Silver layer.")
        views_event = st.altair_chart(
            views_line,
            width="stretch",
            on_select="rerun",
            selection_mode=["views_trend_select", "views_trend_legend_select"],
            key="views_trend_chart",
        )

        views_context = _extract_chart_selection(
            views_event,
            "views_trend_select",
            ["category_name"],
        )
        selected_views_category = _normalize_context_value(views_context.get("category_name"))
        if not selected_views_category:
            views_legend_context = _extract_chart_selection(
                views_event,
                "views_trend_legend_select",
                ["category_name"],
            )
            selected_views_category = _normalize_context_value(
                views_legend_context.get("category_name")
            )

        if selected_views_category:
            diagnostic_region = None if selected_region == "All" else selected_region
            velocity_diag_df = load_velocity_shift_diagnostic(
                str(selected_views_category),
                diagnostic_region,
                history_window,
            )
            persistence_diag_df = load_persistence_distribution_diagnostic(
                str(selected_views_category),
                diagnostic_region,
                history_window,
            )
            engagement_shift_df = load_engagement_shift_diagnostic(
                str(selected_views_category),
                diagnostic_region,
                history_window,
            )
            channel_mix_df = load_channel_mix_shift_diagnostic(
                str(selected_views_category),
                diagnostic_region,
                history_window,
            )

            st.markdown(
                f"**Diagnostics for `{selected_views_category}`**"
                + (f" in `{diagnostic_region}`" if diagnostic_region else "")
            )
            diag_col1, diag_col2 = st.columns(2)

            with diag_col1:
                if velocity_diag_df.empty:
                    st.info("No velocity diagnostic data available for this selection.")
                else:
                    velocity_chart = (
                        alt.Chart(velocity_diag_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("time_bucket:T", title="Time"),
                            y=alt.Y("avg_velocity:Q", title="Average Velocity"),
                            tooltip=[
                                alt.Tooltip("time_bucket:T", title="Time"),
                                alt.Tooltip("avg_velocity:Q", title="Avg Velocity", format=",.2f"),
                                alt.Tooltip(
                                    "avg_engagement_rate:Q",
                                    title="Avg Engagement Rate",
                                    format=".2%",
                                ),
                                alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                            ],
                        )
                        .properties(height=300, title="Diagnostic 1: Velocity Shift")
                    )
                    st.altair_chart(velocity_chart, width="stretch")
                    _explanation(_explain_velocity_shift(velocity_diag_df))

            with diag_col2:
                if persistence_diag_df.empty:
                    st.info("No persistence diagnostic data available for this selection.")
                else:
                    persistence_order = ["1 batch", "2-3 batches", "4-6 batches", "7+ batches"]
                    persistence_chart = (
                        alt.Chart(persistence_diag_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "persistence_bucket:N",
                                sort=persistence_order,
                                title="Persistence Bucket",
                            ),
                            y=alt.Y("video_count:Q", title="Video Count"),
                            tooltip=[
                                "persistence_bucket",
                                alt.Tooltip("video_count:Q", title="Videos", format=","),
                                alt.Tooltip("avg_peak_views:Q", title="Avg Peak Views", format=","),
                            ],
                        )
                        .properties(height=300, title="Diagnostic 2: Trending Persistence")
                    )
                    st.altair_chart(persistence_chart, width="stretch")
                    _explanation(_explain_persistence_distribution(persistence_diag_df))

            diag_col3, diag_col4 = st.columns(2)

            with diag_col3:
                if engagement_shift_df.empty:
                    st.info("No engagement-rate diagnostic data available for this selection.")
                else:
                    engagement_shift_chart = (
                        alt.Chart(engagement_shift_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("time_bucket:T", title="Time"),
                            y=alt.Y(
                                "avg_engagement_rate:Q",
                                title="Avg Engagement Rate",
                                axis=alt.Axis(format=".1%"),
                            ),
                            color=alt.Color("trending_region:N", title="Region"),
                            tooltip=[
                                alt.Tooltip("time_bucket:T", title="Time"),
                                alt.Tooltip("trending_region:N", title="Region"),
                                alt.Tooltip(
                                    "avg_engagement_rate:Q",
                                    title="Avg Engagement Rate",
                                    format=".2%",
                                ),
                                alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                                alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                            ],
                        )
                        .properties(
                            height=300,
                            title="Diagnostic 3: Engagement Rate Shift Over Time",
                        )
                    )
                    st.altair_chart(engagement_shift_chart, width="stretch")
                    _explanation(_explain_engagement_shift(engagement_shift_df))

            with diag_col4:
                if channel_mix_df.empty:
                    st.info("No channel-mix diagnostic data available for this selection.")
                else:
                    tier_order = [
                        "Small (<100K)",
                        "Mid (100K-1M)",
                        "Large (1M-10M)",
                        "Mega (10M+)",
                    ]
                    channel_mix_chart = (
                        alt.Chart(channel_mix_df)
                        .mark_area()
                        .encode(
                            x=alt.X("time_bucket:T", title="Time"),
                            y=alt.Y(
                                "share:Q",
                                stack="normalize",
                                title="Share of Trending Videos",
                                axis=alt.Axis(format=".0%"),
                            ),
                            color=alt.Color(
                                "subscriber_tier:N",
                                title="Channel Size",
                                sort=tier_order,
                            ),
                            order=alt.Order(
                                "subscriber_tier:N",
                                sort="ascending",
                            ),
                            tooltip=[
                                alt.Tooltip("time_bucket:T", title="Time"),
                                alt.Tooltip("subscriber_tier:N", title="Channel Size"),
                                alt.Tooltip("videos:Q", title="Videos", format=","),
                                alt.Tooltip("share:Q", title="Share", format=".1%"),
                            ],
                        )
                        .properties(
                            height=300,
                            title="Diagnostic 4: Channel Size Mix Shift",
                        )
                    )
                    st.altair_chart(channel_mix_chart, width="stretch")
                    _explanation(_explain_channel_mix_shift(channel_mix_df))

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
    #     st.altair_chart(area_chart, width="stretch")
    
    st.subheader("Video Duration Distribution")
    st.markdown("Analytical question: Which duration buckets are most common and which perform better?")

    if duration_dist_df.empty:
        st.info("No duration distribution data available.")
    else:
        duration_select = alt.selection_point(
            name="duration_distribution_select",
            fields=["duration_bucket", "category_name"],
        )
        duration_chart = (
            alt.Chart(duration_dist_df)
            .mark_bar()
            .encode(
                x=alt.X("duration_bucket:N", title="Duration Bucket"),
                y=alt.Y("video_count:Q", title="Video Count"),
                color=alt.Color("category_name:N", title="Category"),
                opacity=alt.condition(duration_select, alt.value(1), alt.value(0.5)),
                tooltip=[
                    "trending_region",
                    "category_name",
                    "duration_bucket",
                    "video_count",
                    alt.Tooltip("avg_views_in_bucket:Q", format=","),
                    alt.Tooltip("avg_er_in_bucket:Q", format=".2%"),
                ],
            )
            .add_params(duration_select)
            .properties(height=400)
        )
        st.caption("Click a duration/category bar to diagnose how engagement behaves across duration buckets.")
        duration_event = st.altair_chart(
            duration_chart,
            width="stretch",
            on_select="rerun",
            selection_mode="duration_distribution_select",
            key="duration_distribution_chart",
        )

        duration_context = _extract_chart_selection(
            duration_event,
            "duration_distribution_select",
            ["duration_bucket", "category_name"],
        )
        selected_duration_category = _normalize_context_value(duration_context.get("category_name"))
        selected_duration_bucket = _normalize_context_value(duration_context.get("duration_bucket"))

        if selected_duration_category or selected_duration_bucket:
            diagnostic_region = None if selected_region == "All" else selected_region
            duration_diag_df = load_duration_engagement_diagnostic(
                str(selected_duration_category) if selected_duration_category else None,
                diagnostic_region,
                history_window,
            )

            st.markdown(
                "**Diagnostic 3: Duration Bucket vs Engagement**"
                + (
                    f" for `{selected_duration_category}`"
                    if selected_duration_category
                    else ""
                )
            )

            if duration_diag_df.empty:
                st.info("No duration diagnostic data available for this selection.")
            else:
                if selected_duration_bucket:
                    duration_diag_df["is_selected"] = (
                        duration_diag_df["duration_bucket"] == selected_duration_bucket
                    )
                else:
                    duration_diag_df["is_selected"] = False

                duration_diag_chart = (
                    alt.Chart(duration_diag_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("duration_bucket:N", title="Duration Bucket"),
                        y=alt.Y("avg_engagement_rate:Q", title="Average Engagement Rate"),
                        color=alt.condition(
                            alt.datum.is_selected,
                            alt.value("#f59e0b"),
                            alt.value("#38bdf8"),
                        ),
                        tooltip=[
                            "duration_bucket",
                            alt.Tooltip(
                                "avg_engagement_rate:Q",
                                title="Avg Engagement Rate",
                                format=".2%",
                            ),
                            alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                            alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(duration_diag_chart, width="stretch")
                _explanation(_explain_duration_engagement(duration_diag_df))

            duration_velocity_df = load_duration_velocity_diagnostic(
                str(selected_duration_category) if selected_duration_category else None,
                diagnostic_region,
                history_window,
            )
            duration_overindex_df = load_duration_category_overindex_diagnostic(
                diagnostic_region,
                history_window,
            )
            duration_region_df = load_duration_region_diagnostic(
                str(selected_duration_category) if selected_duration_category else None,
                history_window,
            )

            dur_diag_col1, dur_diag_col2 = st.columns(2)

            with dur_diag_col1:
                st.markdown("**Diagnostic 4: Velocity by Duration Bucket**")
                if duration_velocity_df.empty:
                    st.info("No velocity-by-duration data available for this selection.")
                else:
                    if selected_duration_bucket:
                        duration_velocity_df["is_selected"] = (
                            duration_velocity_df["duration_bucket"] == selected_duration_bucket
                        )
                    else:
                        duration_velocity_df["is_selected"] = False

                    velocity_by_duration_chart = (
                        alt.Chart(duration_velocity_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("duration_bucket:N", title="Duration Bucket"),
                            y=alt.Y("avg_velocity:Q", title="Average Velocity"),
                            color=alt.condition(
                                alt.datum.is_selected,
                                alt.value("#f59e0b"),
                                alt.value("#22c55e"),
                            ),
                            tooltip=[
                                "duration_bucket",
                                alt.Tooltip("avg_velocity:Q", title="Avg Velocity", format=",.2f"),
                                alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                                alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(velocity_by_duration_chart, width="stretch")
                    _explanation(_explain_duration_velocity(duration_velocity_df))

            with dur_diag_col2:
                st.markdown("**Diagnostic 6: Duration x Region Engagement**")
                if duration_region_df.empty:
                    st.info("No duration-by-region data available for this selection.")
                else:
                    duration_region_heatmap = (
                        alt.Chart(duration_region_df)
                        .mark_rect()
                        .encode(
                            x=alt.X("trending_region:N", title="Region"),
                            y=alt.Y("duration_bucket:N", title="Duration Bucket"),
                            color=alt.Color(
                                "avg_engagement_rate:Q",
                                title="Avg Engagement Rate",
                                scale=alt.Scale(scheme="blues"),
                            ),
                            tooltip=[
                                "trending_region",
                                "duration_bucket",
                                alt.Tooltip(
                                    "avg_engagement_rate:Q",
                                    title="Avg Engagement Rate",
                                    format=".2%",
                                ),
                                alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                                alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(duration_region_heatmap, width="stretch")
                    _explanation(_explain_duration_region(duration_region_df))

            st.markdown("**Diagnostic 5: Category Over-Indexing per Duration Bucket**")
            st.caption(
                "Lift > 1 (warmer color) means a category is over-represented in that "
                "duration bucket relative to its overall share of trending videos."
            )
            if duration_overindex_df.empty:
                st.info("No over-indexing data available for this selection.")
            else:
                # Replace nulls in lift so Altair can render them.
                duration_overindex_df = duration_overindex_df.copy()
                duration_overindex_df["lift"] = duration_overindex_df["lift"].fillna(0)

                overindex_heatmap = (
                    alt.Chart(duration_overindex_df)
                    .mark_rect()
                    .encode(
                        x=alt.X("duration_bucket:N", title="Duration Bucket"),
                        y=alt.Y(
                            "category_name:N",
                            title="Category",
                            sort="-x",
                        ),
                        color=alt.Color(
                            "lift:Q",
                            title="Lift (bucket / overall)",
                            scale=alt.Scale(scheme="redyellowblue", domainMid=1, reverse=True),
                        ),
                        tooltip=[
                            "duration_bucket",
                            "category_name",
                            alt.Tooltip("videos_in_bucket_cat:Q", title="Videos", format=","),
                            alt.Tooltip("bucket_share:Q", title="Bucket Share", format=".1%"),
                            alt.Tooltip("overall_share:Q", title="Overall Share", format=".1%"),
                            alt.Tooltip("lift:Q", title="Lift", format=".2f"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(overindex_heatmap, width="stretch")
                _explanation(_explain_duration_overindex(duration_overindex_df))

    st.subheader("Channel Subscriber Size Distribution")
    st.markdown("Analytical question: Are trending videos dominated by mega channels or smaller creators?")

    if subscriber_tier_df.empty:
        st.info("No subscriber tier data available.")
    else:
        subscriber_select = alt.selection_point(
            name="subscriber_tier_select",
            fields=["subscriber_tier", "trending_region"],
            on="click",
            toggle=False,
            empty=False,
        )
        subscriber_chart = (
            alt.Chart(subscriber_tier_df)
            .mark_bar()
            .encode(
                x=alt.X("trending_region:N", title="Region"),
                y=alt.Y("video_count:Q", stack="normalize", title="Share of Trending Videos"),
                color=alt.Color("subscriber_tier:N", title="Subscriber Tier"),
                opacity=alt.condition(subscriber_select, alt.value(1), alt.value(0.5)),
                tooltip=[
                    "trending_region",
                    "subscriber_tier",
                    "video_count",
                    alt.Tooltip("pct:Q", format=".1f"),
                ],
            )
            .add_params(subscriber_select)
            .properties(height=400)
        )
        st.caption("Click a stacked segment (a tier in a region) to open Spark diagnostics from the Silver layer.")
        subscriber_event = st.altair_chart(
            subscriber_chart,
            width="stretch",
            on_select="rerun",
            selection_mode="subscriber_tier_select",
            key="subscriber_tier_chart",
        )

        subscriber_context = _extract_chart_selection(
            subscriber_event,
            "subscriber_tier_select",
            ["subscriber_tier", "trending_region"],
        )
        selected_subscriber_tier = _normalize_context_value(subscriber_context.get("subscriber_tier"))
        selected_subscriber_region = _normalize_context_value(subscriber_context.get("trending_region"))

        if selected_subscriber_tier or selected_subscriber_region:
            # Region scope: clicked region wins; otherwise fall back to the
            # global region filter from the sidebar.
            if selected_subscriber_region:
                diagnostic_region = str(selected_subscriber_region)
            else:
                diagnostic_region = None if selected_region == "All" else selected_region
            diagnostic_category = None  # subscriber-tier section is category-agnostic

            subscriber_views_df = load_subscriber_views_diagnostic(
                diagnostic_category, diagnostic_region, history_window
            )
            subscriber_engagement_df = load_subscriber_engagement_diagnostic(
                diagnostic_category, diagnostic_region, history_window
            )
            subscriber_persistence_df = load_subscriber_persistence_diagnostic(
                diagnostic_category, diagnostic_region, history_window
            )
            subscriber_region_df = load_subscriber_region_diagnostic(
                diagnostic_category, history_window
            )

            scope_bits = []
            if selected_subscriber_tier:
                scope_bits.append(f"tier `{selected_subscriber_tier}`")
            if selected_subscriber_region:
                scope_bits.append(f"region `{selected_subscriber_region}`")
            st.markdown(
                "**Subscriber-tier diagnostics" + (
                    f" for {' in '.join(scope_bits)}**"
                    if scope_bits
                    else "**"
                )
            )

            tier_order = [
                "Small (<100K)",
                "Mid (100K-1M)",
                "Large (1M-10M)",
                "Mega (10M+)",
            ]

            sub_col1, sub_col2 = st.columns(2)

            with sub_col1:
                st.markdown("**Diagnostic 1: Avg Views by Subscriber Tier**")
                if subscriber_views_df.empty:
                    st.info("No avg-views data available for this selection.")
                else:
                    df_views = subscriber_views_df.copy()
                    df_views["is_selected"] = (
                        df_views["subscriber_tier"] == selected_subscriber_tier
                        if selected_subscriber_tier
                        else False
                    )
                    views_chart = (
                        alt.Chart(df_views)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "subscriber_tier:N",
                                sort=tier_order,
                                title="Subscriber Tier",
                            ),
                            y=alt.Y(
                                "avg_views:Q",
                                title="Avg Views per Video",
                                axis=alt.Axis(format="~s"),
                            ),
                            color=alt.condition(
                                alt.datum.is_selected,
                                alt.value("#f97316"),
                                alt.value("#0ea5e9"),
                            ),
                            tooltip=[
                                "subscriber_tier",
                                alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                                alt.Tooltip("avg_likes:Q", title="Avg Likes", format=","),
                                alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(views_chart, width="stretch")
                    _explanation(_explain_subscriber_views(df_views))

            with sub_col2:
                st.markdown("**Diagnostic 2: Avg Engagement by Subscriber Tier**")
                if subscriber_engagement_df.empty:
                    st.info("No engagement data available for this selection.")
                else:
                    df_eng = subscriber_engagement_df.copy()
                    df_eng["is_selected"] = (
                        df_eng["subscriber_tier"] == selected_subscriber_tier
                        if selected_subscriber_tier
                        else False
                    )
                    eng_chart = (
                        alt.Chart(df_eng)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "subscriber_tier:N",
                                sort=tier_order,
                                title="Subscriber Tier",
                            ),
                            y=alt.Y(
                                "avg_engagement_rate:Q",
                                title="Avg Engagement Rate",
                                axis=alt.Axis(format=".1%"),
                            ),
                            color=alt.condition(
                                alt.datum.is_selected,
                                alt.value("#f97316"),
                                alt.value("#a855f7"),
                            ),
                            tooltip=[
                                "subscriber_tier",
                                alt.Tooltip(
                                    "avg_engagement_rate:Q",
                                    title="Avg Engagement Rate",
                                    format=".2%",
                                ),
                                alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                                alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(eng_chart, width="stretch")
                    _explanation(_explain_subscriber_engagement(df_eng))

            sub_col3, sub_col4 = st.columns(2)

            with sub_col3:
                st.markdown("**Diagnostic 3: Persistence by Subscriber Tier**")
                st.caption("Avg distinct collection batches each tier's videos appear in trending.")
                if subscriber_persistence_df.empty:
                    st.info("No persistence data available for this selection.")
                else:
                    df_pers = subscriber_persistence_df.copy()
                    df_pers["is_selected"] = (
                        df_pers["subscriber_tier"] == selected_subscriber_tier
                        if selected_subscriber_tier
                        else False
                    )
                    persistence_chart = (
                        alt.Chart(df_pers)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "subscriber_tier:N",
                                sort=tier_order,
                                title="Subscriber Tier",
                            ),
                            y=alt.Y("avg_batches:Q", title="Avg Batches Persisted"),
                            color=alt.condition(
                                alt.datum.is_selected,
                                alt.value("#f97316"),
                                alt.value("#22c55e"),
                            ),
                            tooltip=[
                                "subscriber_tier",
                                alt.Tooltip("avg_batches:Q", title="Avg Batches", format=".2f"),
                                alt.Tooltip(
                                    "median_batches:Q",
                                    title="Median Batches",
                                    format=".0f",
                                ),
                                alt.Tooltip("avg_peak_views:Q", title="Avg Peak Views", format=","),
                                alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(persistence_chart, width="stretch")
                    _explanation(_explain_subscriber_persistence(df_pers))

            with sub_col4:
                st.markdown("**Diagnostic 4: Region-Specific Creator-Size Dependence**")
                st.caption("Share of trending videos by tier within each region.")
                if subscriber_region_df.empty:
                    st.info("No region/tier data available for this selection.")
                else:
                    region_tier_chart = (
                        alt.Chart(subscriber_region_df)
                        .mark_rect()
                        .encode(
                            x=alt.X("trending_region:N", title="Region"),
                            y=alt.Y(
                                "subscriber_tier:N",
                                sort=tier_order,
                                title="Subscriber Tier",
                            ),
                            color=alt.Color(
                                "tier_share:Q",
                                title="Share of Region",
                                scale=alt.Scale(scheme="purples"),
                            ),
                            tooltip=[
                                "trending_region",
                                "subscriber_tier",
                                alt.Tooltip("videos:Q", title="Videos", format=","),
                                alt.Tooltip("region_total:Q", title="Region Total", format=","),
                                alt.Tooltip("tier_share:Q", title="Share", format=".1%"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(region_tier_chart, width="stretch")
                    _explanation(_explain_subscriber_region(subscriber_region_df))

    st.subheader("Top Tag Usage Frequency")
    st.markdown("Analytical question: Which tags appear most often in trending videos?")

    if tag_usage_df.empty:
        st.info("No tag usage data available.")
    else:
        tag_select = alt.selection_point(
            name="tag_usage_select",
            fields=["tag", "category_name"],
        )
        tag_chart = (
            alt.Chart(tag_usage_df)
            .mark_bar()
            .encode(
                x=alt.X("videos_using_tag:Q", title="Videos Using Tag"),
                y=alt.Y("tag:N", sort="-x", title="Tag"),
                color=alt.Color("category_name:N", title="Category"),
                opacity=alt.condition(tag_select, alt.value(1), alt.value(0.5)),
                tooltip=[
                    "tag",
                    "category_name",
                    "trending_region",
                    "videos_using_tag",
                ],
            )
            .add_params(tag_select)
            .properties(height=500)
        )
        st.caption("Click a tag bar to see whether common tags are also high-performing tags.")
        tag_event = st.altair_chart(
            tag_chart,
            width="stretch",
            on_select="rerun",
            selection_mode="tag_usage_select",
            key="tag_usage_chart",
        )

        tag_context = _extract_chart_selection(
            tag_event,
            "tag_usage_select",
            ["tag", "category_name"],
        )
        selected_tag = _normalize_context_value(tag_context.get("tag"))
        selected_tag_category = _normalize_context_value(tag_context.get("category_name"))

        if selected_tag or selected_tag_category:
            diagnostic_region = None if selected_region == "All" else selected_region
            tag_diag_df = load_tag_effectiveness_diagnostic(
                str(selected_tag_category) if selected_tag_category else None,
                diagnostic_region,
                history_window,
            )

            st.markdown(
                "**Diagnostic 4: Tag Frequency vs Engagement**"
                + (f" for `{selected_tag_category}`" if selected_tag_category else "")
            )

            if tag_diag_df.empty:
                st.info("No tag diagnostic data available for this selection.")
            else:
                if selected_tag:
                    tag_diag_df["is_selected"] = tag_diag_df["tag"] == selected_tag
                else:
                    tag_diag_df["is_selected"] = False

                tag_diag_chart = (
                    alt.Chart(tag_diag_df)
                    .mark_circle(opacity=0.8)
                    .encode(
                        x=alt.X("videos_using_tag:Q", title="Videos Using Tag"),
                        y=alt.Y("avg_engagement_rate:Q", title="Average Engagement Rate"),
                        size=alt.Size("avg_views:Q", title="Average Views"),
                        color=alt.condition(
                            alt.datum.is_selected,
                            alt.value("#f97316"),
                            alt.value("#6366f1"),
                        ),
                        tooltip=[
                            "tag",
                            alt.Tooltip("videos_using_tag:Q", title="Videos Using Tag", format=","),
                            alt.Tooltip(
                                "avg_engagement_rate:Q",
                                title="Avg Engagement Rate",
                                format=".2%",
                            ),
                            alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                        ],
                    )
                    .properties(height=360)
                )
                st.altair_chart(tag_diag_chart, width="stretch")
                _explanation(_explain_tag_frequency_engagement(tag_diag_df))

            high_er_tags_df = load_high_engagement_tags_diagnostic(
                str(selected_tag_category) if selected_tag_category else None,
                diagnostic_region,
                history_window,
            )
            tag_region_df = load_tag_region_concentration_diagnostic(
                str(selected_tag_category) if selected_tag_category else None,
                history_window,
            )
            tag_density_df = load_tag_density_performance_diagnostic(
                str(selected_tag_category) if selected_tag_category else None,
                diagnostic_region,
                history_window,
            )

            tag_diag_col1, tag_diag_col2 = st.columns(2)

            with tag_diag_col1:
                st.markdown("**Diagnostic 5: Highest-Engagement Tags**")
                st.caption("Tags ranked by avg engagement rate (min 5 videos so single-use noise is filtered).")
                if high_er_tags_df.empty:
                    st.info("No high-engagement tag data available for this selection.")
                else:
                    if selected_tag:
                        high_er_tags_df = high_er_tags_df.copy()
                        high_er_tags_df["is_selected"] = high_er_tags_df["tag"] == selected_tag
                    else:
                        high_er_tags_df = high_er_tags_df.copy()
                        high_er_tags_df["is_selected"] = False

                    high_er_chart = (
                        alt.Chart(high_er_tags_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "avg_engagement_rate:Q",
                                title="Avg Engagement Rate",
                                axis=alt.Axis(format=".1%"),
                            ),
                            y=alt.Y("tag:N", sort="-x", title="Tag"),
                            color=alt.condition(
                                alt.datum.is_selected,
                                alt.value("#f97316"),
                                alt.value("#10b981"),
                            ),
                            tooltip=[
                                "tag",
                                alt.Tooltip(
                                    "avg_engagement_rate:Q",
                                    title="Avg Engagement Rate",
                                    format=".2%",
                                ),
                                alt.Tooltip(
                                    "videos_using_tag:Q",
                                    title="Videos Using Tag",
                                    format=",",
                                ),
                                alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                            ],
                        )
                        .properties(height=360)
                    )
                    st.altair_chart(high_er_chart, width="stretch")
                    _explanation(_explain_high_engagement_tags(high_er_tags_df))

            with tag_diag_col2:
                st.markdown("**Diagnostic 7: Tag-Heavy vs Tag-Light Performance**")
                st.caption("Bucketed by tag count per video — does adding more tags actually help?")
                if tag_density_df.empty:
                    st.info("No tag density data available for this selection.")
                else:
                    density_order = [
                        "0 tags",
                        "1-5 tags",
                        "6-10 tags",
                        "11-20 tags",
                        "21+ tags",
                    ]
                    density_base = alt.Chart(tag_density_df).encode(
                        x=alt.X(
                            "tag_density_bucket:N",
                            sort=density_order,
                            title="Tag Density Bucket",
                        ),
                    )
                    density_bars = density_base.mark_bar(color="#6366f1").encode(
                        y=alt.Y(
                            "avg_engagement_rate:Q",
                            title="Avg Engagement Rate",
                            axis=alt.Axis(format=".1%"),
                        ),
                        tooltip=[
                            "tag_density_bucket",
                            alt.Tooltip(
                                "avg_engagement_rate:Q",
                                title="Avg Engagement Rate",
                                format=".2%",
                            ),
                            alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                            alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                        ],
                    )
                    density_views_line = density_base.mark_line(
                        color="#f59e0b",
                        point=True,
                    ).encode(
                        y=alt.Y(
                            "avg_views:Q",
                            title="Avg Views",
                            axis=alt.Axis(format="~s", titleColor="#f59e0b"),
                        ),
                    )
                    density_chart = alt.layer(
                        density_bars, density_views_line
                    ).resolve_scale(y="independent").properties(height=360)
                    st.altair_chart(density_chart, width="stretch")
                    _explanation(_explain_tag_density(tag_density_df))

            st.markdown("**Diagnostic 6: Tag Region Concentration (Region-Specific vs Global)**")
            st.caption(
                "Each row is a top tag; bars show how its usage splits across regions. "
                "Tags dominated by one region are region-specific; flat splits are global."
            )
            if tag_region_df.empty:
                st.info("No region-concentration data available for this selection.")
            else:
                tag_region_chart = (
                    alt.Chart(tag_region_df)
                    .mark_bar()
                    .encode(
                        y=alt.Y(
                            "tag:N",
                            sort=alt.EncodingSortField(
                                field="region_concentration",
                                order="descending",
                            ),
                            title="Tag (sorted: most region-specific on top)",
                        ),
                        x=alt.X(
                            "region_share:Q",
                            stack="normalize",
                            title="Share of Tag Usage by Region",
                            axis=alt.Axis(format=".0%"),
                        ),
                        color=alt.Color("trending_region:N", title="Region"),
                        tooltip=[
                            "tag",
                            "trending_region",
                            alt.Tooltip("videos_in_region:Q", title="Videos in Region", format=","),
                            alt.Tooltip("videos_total:Q", title="Total Videos", format=","),
                            alt.Tooltip("region_share:Q", title="Region Share", format=".1%"),
                            alt.Tooltip(
                                "region_concentration:Q",
                                title="Concentration (max share)",
                                format=".1%",
                            ),
                        ],
                    )
                    .properties(height=420)
                )
                st.altair_chart(tag_region_chart, width="stretch")
                _explanation(_explain_tag_region_concentration(tag_region_df))


    st.subheader("Trending Rank Distribution by Category")
    st.markdown("Mode 1: Latest snapshot only. Out of the current trending slots, how many belong to each category?")

    if trending_rank_dist_df.empty:
        st.info("No latest snapshot distribution data available.")
    else:
        trending_rank_select = alt.selection_point(
            name="trending_rank_select",
            fields=["category_name", "trending_region"],
            on="click",
            toggle=False,
            empty=False,
        )
        trending_dist_chart = (
            alt.Chart(trending_rank_dist_df)
            .mark_bar()
            .encode(
                x=alt.X("video_count:Q", title="Current Trending Video Count"),
                y=alt.Y("category_name:N", sort="-x", title="Category"),
                color=alt.Color("trending_region:N", title="Region"),
                opacity=alt.condition(trending_rank_select, alt.value(1), alt.value(0.5)),
                tooltip=[
                    "trending_region",
                    "category_name",
                    "video_count",
                    alt.Tooltip("pct_of_trending:Q", title="% of Trending", format=".2f"),
                ],
            )
            .add_params(trending_rank_select)
            .properties(height=450)
        )
        st.caption("Click a category bar to open Spark diagnostics from the Silver layer.")
        trending_rank_event = st.altair_chart(
            trending_dist_chart,
            width="stretch",
            on_select="rerun",
            selection_mode="trending_rank_select",
            key="trending_rank_chart",
        )

        trending_rank_context = _extract_chart_selection(
            trending_rank_event,
            "trending_rank_select",
            ["category_name", "trending_region"],
        )
        selected_rank_category = _normalize_context_value(trending_rank_context.get("category_name"))
        selected_rank_region = _normalize_context_value(trending_rank_context.get("trending_region"))

        if selected_rank_category:
            # Region scope: clicked region wins; otherwise fall back to the
            # global region filter from the sidebar.
            if selected_rank_region:
                rank_diag_region = str(selected_rank_region)
            else:
                rank_diag_region = None if selected_region == "All" else selected_region

            new_vs_persisting_df = load_rank_new_vs_persisting_diagnostic(
                str(selected_rank_category), rank_diag_region, history_window
            )
            channel_concentration_df = load_top_rank_channel_concentration_diagnostic(
                str(selected_rank_category), rank_diag_region, history_window
            )
            velocity_rank_df = load_velocity_vs_rank_diagnostic(
                str(selected_rank_category), rank_diag_region, history_window
            )
            share_drivers_df = load_category_share_drivers_diagnostic(
                str(selected_rank_category), rank_diag_region, history_window
            )

            scope_bits = [f"`{selected_rank_category}`"]
            if selected_rank_region:
                scope_bits.append(f"region `{selected_rank_region}`")
            st.markdown("**Trending-rank diagnostics for " + " in ".join(scope_bits) + "**")

            rank_col1, rank_col2 = st.columns(2)

            with rank_col1:
                st.markdown("**Diagnostic 1: New Entries vs Persisting Videos**")
                st.caption("Are trending slots filled by fresh videos, or the same ones holding on?")
                if new_vs_persisting_df.empty:
                    st.info("No new-vs-persisting data available for this selection.")
                else:
                    new_vs_persisting_chart = (
                        alt.Chart(new_vs_persisting_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("time_bucket:T", title="Time"),
                            y=alt.Y(
                                "videos:Q",
                                title="Distinct Videos",
                                stack="zero",
                            ),
                            color=alt.Color(
                                "entry_status:N",
                                title="Entry Status",
                                scale=alt.Scale(
                                    domain=["New entry", "Persisting"],
                                    range=["#22c55e", "#6366f1"],
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip("time_bucket:T", title="Time"),
                                "entry_status",
                                alt.Tooltip("videos:Q", title="Videos", format=","),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(new_vs_persisting_chart, width="stretch")
                    _explanation(_explain_new_vs_persisting(new_vs_persisting_df))

            with rank_col2:
                st.markdown("**Diagnostic 2: Top-Rank Channel Concentration**")
                st.caption("Top-10 trending slots held by each channel — flat = diverse, spiky = concentrated.")
                if channel_concentration_df.empty:
                    st.info("No top-rank channel data available for this selection.")
                else:
                    channel_concentration_chart = (
                        alt.Chart(channel_concentration_df)
                        .mark_bar(color="#f97316")
                        .encode(
                            x=alt.X("top_rank_slots:Q", title="Top-Rank Slots Held"),
                            y=alt.Y("channel_title:N", sort="-x", title="Channel"),
                            tooltip=[
                                "channel_title",
                                alt.Tooltip("top_rank_slots:Q", title="Top-Rank Slots", format=","),
                                alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                                alt.Tooltip("avg_rank:Q", title="Avg Rank", format=".1f"),
                                alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(channel_concentration_chart, width="stretch")
                    _explanation(_explain_top_rank_concentration(channel_concentration_df))

            rank_col3, rank_col4 = st.columns(2)

            with rank_col3:
                st.markdown("**Diagnostic 3: Velocity vs Best Trending Rank**")
                st.caption("Lower (better) rank toward the top. Downward-sloping cloud = velocity drives rank.")
                if velocity_rank_df.empty:
                    st.info("No velocity-vs-rank data available for this selection.")
                else:
                    velocity_rank_chart = (
                        alt.Chart(velocity_rank_df)
                        .mark_circle(opacity=0.7)
                        .encode(
                            x=alt.X("avg_velocity:Q", title="Avg Velocity"),
                            y=alt.Y(
                                "best_rank:Q",
                                title="Best Trending Rank (lower = better)",
                                scale=alt.Scale(reverse=True),
                            ),
                            size=alt.Size("avg_views:Q", title="Avg Views"),
                            color=alt.Color(
                                "avg_engagement_rate:Q",
                                title="Avg Engagement Rate",
                                scale=alt.Scale(scheme="viridis"),
                            ),
                            tooltip=[
                                "video_id",
                                "channel_title",
                                alt.Tooltip("avg_velocity:Q", title="Avg Velocity", format=",.2f"),
                                alt.Tooltip("best_rank:Q", title="Best Rank", format=".0f"),
                                alt.Tooltip("avg_views:Q", title="Avg Views", format=","),
                                alt.Tooltip(
                                    "avg_engagement_rate:Q",
                                    title="Avg Engagement Rate",
                                    format=".2%",
                                ),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(velocity_rank_chart, width="stretch")
                    _explanation(_explain_velocity_vs_rank(velocity_rank_df))

            with rank_col4:
                st.markdown("**Diagnostic 4: Category Share Drivers — Volume vs Per-Video Strength**")
                st.caption("Are more videos showing up, or are the same number of videos getting bigger?")
                if share_drivers_df.empty:
                    st.info("No share-drivers data available for this selection.")
                else:
                    share_base = alt.Chart(share_drivers_df).encode(
                        x=alt.X("time_bucket:T", title="Time"),
                    )
                    share_videos_bars = share_base.mark_bar(color="#0ea5e9", opacity=0.7).encode(
                        y=alt.Y(
                            "unique_videos:Q",
                            title="Unique Videos",
                            axis=alt.Axis(titleColor="#0ea5e9"),
                        ),
                        tooltip=[
                            alt.Tooltip("time_bucket:T", title="Time"),
                            alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                            alt.Tooltip("avg_views_per_video:Q", title="Avg Views/Video", format=","),
                            alt.Tooltip("total_views:Q", title="Total Views", format=","),
                        ],
                    )
                    share_avg_line = share_base.mark_line(
                        color="#f59e0b", point=True
                    ).encode(
                        y=alt.Y(
                            "avg_views_per_video:Q",
                            title="Avg Views per Video",
                            axis=alt.Axis(format="~s", titleColor="#f59e0b"),
                        ),
                    )
                    share_drivers_chart = alt.layer(
                        share_videos_bars, share_avg_line
                    ).resolve_scale(y="independent").properties(height=320)
                    st.altair_chart(share_drivers_chart, width="stretch")
                    _explanation(_explain_share_drivers(share_drivers_df))




with tab2:
    st.subheader("2. What will happen?")
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
        st.altair_chart(entry_chart, width="stretch")
        _model_metrics_strip(
            "Trending Entry Probability",
            load_entry_probability_metrics(df),
        )

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
        st.altair_chart((confidence_band + actual_line + forecast_line).properties(height=450), width="stretch")
        _model_metrics_strip(
            "View Count Forecast",
            load_view_forecast_metrics(df),
        )

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
        st.altair_chart(duration_prediction_chart, width="stretch")
        _model_metrics_strip(
            "Trending Duration Prediction",
            load_duration_prediction_metrics(df),
        )

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
        st.altair_chart(peak_rank_chart, width="stretch")
        _model_metrics_strip(
            "Peak Rank Forecast",
            load_peak_rank_metrics(df),
        )

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
        st.altair_chart(category_share_chart, width="stretch")
        _model_metrics_strip(
            "Category Share Forecast",
            load_category_share_metrics(df),
        )

with tab3:
    st.subheader("3. What should be done?")
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
        st.altair_chart(posting_heatmap, width="stretch")

        best_slots = optimal_posting_df[optimal_posting_df["slot_rank"] == 1].copy()
        if not best_slots.empty:
            st.dataframe(best_slots, width="stretch")

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
        st.altair_chart(gap_chart, width="stretch")

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
        st.altair_chart(partner_chart, width="stretch")

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
        st.altair_chart(format_chart, width="stretch")

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
        st.altair_chart(alert_chart, width="stretch")

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
        st.altair_chart(expansion_chart, width="stretch")

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
    st.dataframe(df, width="stretch")

import time

time.sleep(10)
st.rerun()
