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
    build_view_count_forecast_v2,
    build_trending_duration_prediction,
    build_peak_rank_forecast,
    build_category_share_forecast,
)
from analytics.predictive_metrics import (
    evaluate_category_share_forecast,
    evaluate_peak_rank_forecast,
    evaluate_trending_duration_prediction,
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
    # v2 mechanical-decomposition diagnostics (one pair per descriptive chart)
    build_views_volume_strength_diagnostic,
    build_views_new_vs_carryover_diagnostic,
    build_duration_slot_footprint_diagnostic,
    build_duration_audience_response_diagnostic,
    build_tier_effort_reward_diagnostic,
    build_tier_persistence_engagement_diagnostic,
    build_tag_adoption_intensity_diagnostic,
    build_tag_cooccurrence_diagnostic,
    build_rank_slot_turnover_diagnostic,
    build_rank_channel_concentration_v2_diagnostic,
)

MEDALLION_PATHS = get_medallion_paths()
SILVER_DELTA_PATH = MEDALLION_PATHS["silver"]
GOLD_LATEST_SNAPSHOT_PATH = MEDALLION_PATHS["gold"]["latest_snapshot"]
GOLD_CATEGORY_SUMMARY_PATH = MEDALLION_PATHS["gold"]["category_summary"]
GOLD_VIEWS_TIMESERIES_PATH = MEDALLION_PATHS["gold"]["views_timeseries"]
GOLD_REGION_TIMESERIES_PATH = MEDALLION_PATHS["gold"]["region_timeseries"]
GOLD_CHANNEL_LEADERBOARD_PATH = MEDALLION_PATHS["gold"]["channel_leaderboard"]
GOLD_DURATION_DISTRIBUTION_PATH = MEDALLION_PATHS["gold"]["duration_distribution"]
GOLD_SUBSCRIBER_TIER_PATH = MEDALLION_PATHS["gold"]["subscriber_tier_distribution"]
GOLD_TAG_USAGE_PATH = MEDALLION_PATHS["gold"]["tag_usage_frequency"]
GOLD_TRENDING_RANK_DIST_PATH = MEDALLION_PATHS["gold"]["trending_rank_distribution"]
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


def _render_suggestion_box(title: str, bullets: list[str]) -> None:
    body = "".join(f"<li>{bullet}</li>" for bullet in bullets)
    st.markdown(
        f"<div style='font-size:0.88rem; color:#0f172a; "
        f"background:#eff6ff; padding:0.65rem 0.85rem; "
        f"border-left:3px solid #2563eb; border-radius:4px; "
        f"margin-top:-0.2rem; margin-bottom:0.9rem;'>"
        f"<b>{title}</b><ul style='margin:0.35rem 0 0 1rem; padding:0;'>{body}</ul></div>",
        unsafe_allow_html=True,
    )


def _suggest_from_view_forecast(selected_rows: pd.DataFrame) -> list[str]:
    if selected_rows.empty:
        return []
    actual = selected_rows[selected_rows["series"] == "Actual"].sort_values("time_bucket")
    forecast = selected_rows[selected_rows["series"] == "Forecast"].sort_values("time_bucket")
    if actual.empty or forecast.empty:
        return []

    current_views = float(actual["forecast_views"].iloc[-1])
    next_forecast = float(forecast["forecast_views"].iloc[0])
    upper = float(forecast["upper_bound"].iloc[0])
    lower = float(forecast["lower_bound"].iloc[0])
    growth_pct = 0.0 if current_views == 0 else ((next_forecast - current_views) / current_views) * 100.0
    spread_pct = 0.0 if next_forecast == 0 else ((upper - lower) / max(next_forecast, 1.0)) * 100.0

    bullets = []
    if growth_pct >= 10:
        bullets.append(f"Momentum is still strong: projected views are up about {growth_pct:.1f}% next step, so this is a good candidate to keep amplifying.")
    elif growth_pct <= -10:
        bullets.append(f"Projected views are softening by about {abs(growth_pct):.1f}%, so avoid overcommitting budget and watch for decay.")
    else:
        bullets.append("Projected view growth is fairly flat, so treat this as a maintain-and-monitor title rather than a breakout.")

    if spread_pct >= 35:
        bullets.append("The forecast band is wide, which means uncertainty is high; use lighter-touch actions and recheck after the next batch.")
    else:
        bullets.append("The forecast band is reasonably tight, so you can treat the short-horizon projection as directionally dependable.")

    bullets.append("Use the next observed batch as the decision checkpoint: continue boosting only if actual views track near or above the forecast line.")
    return bullets


def _suggest_from_duration_prediction(row: pd.Series) -> list[str]:
    remaining = float(row.get("predicted_remaining_hours", 0) or 0)
    current_rank = float(row.get("current_rank", 0) or 0)
    bullets = []
    if remaining >= 6:
        bullets.append(f"This video still has roughly {remaining:.1f} hours of modeled runway, so it is worth supporting with sustained promotion.")
    elif remaining >= 2:
        bullets.append(f"There is still about {remaining:.1f} hours of expected trending life left, so treat it as a near-term opportunity window.")
    else:
        bullets.append(f"Modeled remaining life is short at about {remaining:.1f} hours, so any intervention should be immediate or not at all.")

    if current_rank <= 10:
        bullets.append("It is already sitting near the top of trending, so the main action is defense: keep momentum from slipping.")
    else:
        bullets.append("It still has room to climb from its current rank, so concentrate support now rather than late in the cycle.")

    bullets.append("Prioritize this title relative to others only while its remaining-life estimate stays above your operational threshold.")
    return bullets


def _suggest_from_peak_rank_forecast(row: pd.Series) -> list[str]:
    expected_gain = float(row.get("expected_rank_gain", 0) or 0)
    predicted_peak = float(row.get("predicted_peak_rank", 0) or 0)
    velocity = float(row.get("current_velocity", 0) or 0)
    bullets = []
    if expected_gain >= 10:
        bullets.append(f"The model sees major upside with about {expected_gain:.1f} ranks of headroom, so this is a strong push candidate.")
    elif expected_gain >= 3:
        bullets.append(f"There is moderate upside of about {expected_gain:.1f} ranks, so a selective boost makes sense if inventory is limited.")
    else:
        bullets.append("Expected rank improvement is limited, so treat this as steady-state content rather than a breakout bet.")

    bullets.append(f"Best-case modeled peak is around rank {predicted_peak:.1f}; use that as the ceiling when setting expectations.")

    if velocity > 0:
        bullets.append("Current velocity is positive, so the right action is to reinforce what is already working instead of changing packaging mid-flight.")
    else:
        bullets.append("Velocity is not especially strong, so don’t assume rank headroom will realize without extra support.")
    return bullets


def _suggest_from_category_share_forecast(selected_rows: pd.DataFrame) -> list[str]:
    if selected_rows.empty:
        return []
    actual = selected_rows[selected_rows["series"] == "Actual"].sort_values("time_bucket")
    forecast = selected_rows[selected_rows["series"] == "Forecast"].sort_values("time_bucket")
    if actual.empty or forecast.empty:
        return []

    latest_actual = float(actual["forecast_share"].iloc[-1])
    final_forecast = float(forecast["forecast_share"].iloc[-1])
    delta_pp = (final_forecast - latest_actual) * 100.0
    bullets = []
    if delta_pp >= 1.0:
        bullets.append(f"Share is projected to expand by about {delta_pp:.2f} percentage points, so this category deserves more inventory and attention.")
    elif delta_pp <= -1.0:
        bullets.append(f"Share is projected to contract by about {abs(delta_pp):.2f} points, so avoid over-weighting this category in the next cycle.")
    else:
        bullets.append("Projected share is fairly stable, so category investment can stay close to current levels.")

    bullets.append("Use this share forecast as an allocation signal across categories rather than as a title-level action.")
    bullets.append("Revisit the category mix after each new bucket, because share forecasts can move quickly when a few large videos enter or exit.")
    return bullets


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


# --- v2 conclusion explainers (one per new diagnostic) -------------------
# Each takes the small aggregated DataFrame returned by the matching loader
# and produces a sentence describing what the chart is saying about the
# trend. The wording flips with the data shape (volume-led vs strength-led,
# concentrated vs broad, etc.).


def _explain_views_volume_strength(df) -> str:
    if df.empty or "unique_videos" not in df.columns:
        return "Not enough data to interpret the trend."
    df = df.sort_values("time_bucket")
    vids_dir = _explain_trend_direction(df["unique_videos"].tolist())
    avg_dir = _explain_trend_direction(df["avg_views_per_video"].tolist())
    vids_delta = _pct_delta(df["unique_videos"].iloc[0], df["unique_videos"].iloc[-1])
    avg_delta = _pct_delta(df["avg_views_per_video"].iloc[0], df["avg_views_per_video"].iloc[-1])
    head = f"Unique videos {vids_dir} ({vids_delta}); avg views/video {avg_dir} ({avg_delta})."
    if vids_dir == "rising" and avg_dir == "rising":
        return f"{head} **Genuine heat** — both more videos AND each video pulling more views."
    if vids_dir == "rising" and avg_dir != "rising":
        return f"{head} **Volume-led** — the rise comes from a flood of new videos, not stronger individual performance."
    if vids_dir != "rising" and avg_dir == "rising":
        return f"{head} **Strength-led** — same number of videos, but each one is pulling more views (a few hits compounding)."
    if vids_dir == "falling" and avg_dir == "falling":
        return f"{head} **Cooling on both fronts** — fewer videos and weaker individual reach."
    return f"{head} No single lever is dominating right now."


def _explain_views_new_vs_carryover(df) -> str:
    if df.empty or "videos" not in df.columns:
        return "Not enough data."
    pivot = (
        df.pivot_table(index="time_bucket", columns="entry_status", values="videos", aggfunc="sum")
        .fillna(0).sort_index()
    )
    if pivot.empty:
        return "No videos in scope."
    new_total = pivot.get("New entry", pd.Series(dtype=float)).sum()
    carry_total = pivot.get("Carry-over", pd.Series(dtype=float)).sum()
    grand = new_total + carry_total
    if grand == 0:
        return "No videos in scope."
    new_share = new_total / grand
    new_dir = _explain_trend_direction(pivot.get("New entry", pd.Series(dtype=float)).tolist())
    if new_share > 0.6:
        story = "fed by **fresh entries** — most slots are new arrivals."
    elif new_share < 0.3:
        story = "carried by **persisting videos** — the same titles keep showing up."
    else:
        story = "a **balanced mix** of fresh entries and persisting videos."
    return f"This category is {story} New-entry volume is **{new_dir}** over the window — that tells you whether the trend is being fed by churn or by compounding hits."


def _explain_duration_slot_footprint(df) -> str:
    if df.empty or "distinct_videos" not in df.columns:
        return "Not enough data."
    df = df.copy()
    df["slot_footprint"] = df["distinct_videos"] * df["avg_batches_per_video"]
    top = df.sort_values("slot_footprint", ascending=False).iloc[0]
    bucket = top["duration_bucket"]
    vids = int(top["distinct_videos"])
    persist = float(top["avg_batches_per_video"])
    # Decide whether dominance is volume-driven or persistence-driven by
    # comparing this bucket's two levers against the others.
    others = df[df["duration_bucket"] != bucket]
    median_vids = float(others["distinct_videos"].median()) if not others.empty else 0
    median_persist = float(others["avg_batches_per_video"].median()) if not others.empty else 0
    vol_lead = vids > median_vids * 1.5 if median_vids else True
    persist_lead = persist > median_persist * 1.5 if median_persist else False
    if vol_lead and persist_lead:
        story = (
            f"**{bucket}** dominates because BOTH levers are large — "
            f"{vids} distinct videos × {persist:.1f} avg batches each."
        )
    elif vol_lead:
        story = (
            f"**{bucket}** dominates as a **volume game** — "
            f"{vids} distinct videos churn through trending, each lasting only {persist:.1f} batches."
        )
    elif persist_lead:
        story = (
            f"**{bucket}** dominates as a **persistence game** — only {vids} distinct videos, "
            f"but each holds its slot for {persist:.1f} batches on average."
        )
    else:
        story = (
            f"**{bucket}** has the largest slot footprint ({vids} videos × {persist:.1f} batches) — "
            "no single lever is doing all the work."
        )
    return story


def _explain_duration_audience_response(df) -> str:
    if df.empty or "avg_engagement_rate" not in df.columns:
        return "Not enough data."
    er_best = df.sort_values("avg_engagement_rate", ascending=False).iloc[0]
    views_best = df.sort_values("avg_views_per_video", ascending=False).iloc[0]
    if er_best["duration_bucket"] == views_best["duration_bucket"]:
        return (
            f"**{er_best['duration_bucket']}** wins on BOTH avg engagement "
            f"({_fmt_pct(er_best['avg_engagement_rate'], 2)}) and avg views "
            f"({_fmt_int(er_best['avg_views_per_video'])}) — viewers genuinely prefer this length here."
        )
    return (
        f"Engagement leads with **{er_best['duration_bucket']}** "
        f"({_fmt_pct(er_best['avg_engagement_rate'], 2)}), while views per video lead with "
        f"**{views_best['duration_bucket']}** ({_fmt_int(views_best['avg_views_per_video'])}). "
        "The two don't agree — slot dominance probably isn't an audience-preference story."
    )


def _explain_tier_effort_reward(df) -> str:
    if df.empty or "distinct_channels" not in df.columns:
        return "Not enough data."
    df = df.copy()
    df["total_slots"] = df["distinct_channels"] * df["avg_slots_per_channel"]
    top = df.sort_values("total_slots", ascending=False).iloc[0]
    others = df[df["subscriber_tier"] != top["subscriber_tier"]]
    median_channels = float(others["distinct_channels"].median()) if not others.empty else 0
    median_avg = float(others["avg_slots_per_channel"].median()) if not others.empty else 0
    broad = top["distinct_channels"] > median_channels * 1.5 if median_channels else True
    intense = top["avg_slots_per_channel"] > median_avg * 1.5 if median_avg else False
    if broad and intense:
        story = (
            f"**{top['subscriber_tier']}** dominates with BOTH wide participation "
            f"({int(top['distinct_channels'])} channels) AND high per-channel intensity "
            f"({top['avg_slots_per_channel']:.1f} slots each)."
        )
    elif broad:
        story = (
            f"**{top['subscriber_tier']}** dominates through **broad participation** — "
            f"{int(top['distinct_channels'])} channels each contributing only "
            f"{top['avg_slots_per_channel']:.1f} slots on average."
        )
    elif intense:
        story = (
            f"**{top['subscriber_tier']}** dominates through **concentrated intensity** — "
            f"only {int(top['distinct_channels'])} channels, but each holds "
            f"{top['avg_slots_per_channel']:.1f} slots."
        )
    else:
        story = (
            f"**{top['subscriber_tier']}** leads with {int(top['distinct_channels'])} channels × "
            f"{top['avg_slots_per_channel']:.1f} slots each — neither lever is overwhelming."
        )
    return story


def _explain_tier_persistence_engagement(df) -> str:
    if df.empty or "avg_batches_per_video" not in df.columns:
        return "Not enough data."
    persist_top = df.sort_values("avg_batches_per_video", ascending=False).iloc[0]
    er_top = df.sort_values("avg_engagement_rate", ascending=False).iloc[0]
    if persist_top["subscriber_tier"] == er_top["subscriber_tier"]:
        return (
            f"**{persist_top['subscriber_tier']}** holds slots through BOTH stickiness "
            f"({persist_top['avg_batches_per_video']:.1f} avg batches) and audience response "
            f"({_fmt_pct(persist_top['avg_engagement_rate'], 2)} ER) — the strongest tier all around."
        )
    return (
        f"**{persist_top['subscriber_tier']}** sticks longest "
        f"({persist_top['avg_batches_per_video']:.1f} avg batches), but **{er_top['subscriber_tier']}** "
        f"earns better engagement ({_fmt_pct(er_top['avg_engagement_rate'], 2)}). "
        "Different tiers win on different mechanics here."
    )


def _explain_tag_adoption_intensity(df) -> str:
    if df.empty or "distinct_channels" not in df.columns:
        return "Not enough tag data in scope."
    df = df.copy()
    median_channels = df["distinct_channels"].median()
    median_per_channel = df["videos_per_channel"].median()
    broadly_adopted = df[
        (df["distinct_channels"] >= median_channels) & (df["videos_per_channel"] < median_per_channel)
    ]
    flooders = df[
        (df["distinct_channels"] < median_channels) & (df["videos_per_channel"] >= median_per_channel)
    ]
    parts = []
    if not broadly_adopted.empty:
        b = broadly_adopted.iloc[0]
        parts.append(
            f"**{b['tag']}** is broadly adopted ({int(b['distinct_channels'])} channels, "
            f"{b['videos_per_channel']:.2f} videos/channel) — many creators, light intensity."
        )
    if not flooders.empty:
        f0 = flooders.iloc[0]
        parts.append(
            f"**{f0['tag']}** is flooded by a few channels "
            f"({int(f0['distinct_channels'])} channels × {f0['videos_per_channel']:.2f} videos each)."
        )
    if not parts:
        top = df.sort_values("videos_using_tag", ascending=False).iloc[0]
        return (
            f"**{top['tag']}** leads with {int(top['distinct_channels'])} channels averaging "
            f"{top['videos_per_channel']:.2f} videos each — balanced adoption."
        )
    return " ".join(parts)


def _explain_tag_cooccurrence(df) -> str:
    if df.empty or "co_videos" not in df.columns:
        return "Not enough data to compute co-occurrence."
    primary = df["primary_tag"].iloc[0] if "primary_tag" in df.columns and len(df) else "?"
    df_sorted = df.sort_values("co_videos", ascending=False)
    top = df_sorted.iloc[0]
    top_share = float(top.get("co_share", 0.0))
    if top_share >= 0.6:
        story = (
            f"**{primary}** travels with **{top['tag']}** {_fmt_pct(top_share)} of the time — "
            "strong template effect; the dominance is at least partly carried by a co-tag pattern."
        )
    elif top_share >= 0.3:
        story = (
            f"**{primary}** most often co-occurs with **{top['tag']}** "
            f"({_fmt_pct(top_share)}) — moderate template effect."
        )
    else:
        story = (
            f"**{primary}**'s co-occurrences are spread out (top partner **{top['tag']}** at only "
            f"{_fmt_pct(top_share)}) — the tag is genuinely versatile, not riding a template."
        )
    return story


def _explain_rank_slot_turnover(df) -> str:
    if df.empty or "turnover_rate" not in df.columns:
        return "Not enough data."
    avg_rate = float(df["turnover_rate"].mean())
    distinct = int(df["distinct_videos"].sum())
    if avg_rate >= 0.5:
        story = (
            f"Turnover is **high** (avg {avg_rate:.2f}) — {distinct} distinct videos rotated through "
            "this category's slots. Vibrant pipeline, lots of new entrants competing."
        )
    elif avg_rate <= 0.2:
        story = (
            f"Turnover is **low** (avg {avg_rate:.2f}) — only {distinct} distinct videos held the slots. "
            "Stale pipeline; a few persistent videos are hogging the category."
        )
    else:
        story = (
            f"Turnover is **moderate** (avg {avg_rate:.2f}) — {distinct} distinct videos through the slots. "
            "Some rotation, some persistence."
        )
    return story


def _explain_rank_channel_concentration_v2(df) -> str:
    if df.empty or "share_pct" not in df.columns:
        return "Not enough data."
    df_sorted = df.sort_values("share_pct", ascending=False)
    top3_share = float(df_sorted.head(3)["share_pct"].sum())
    top1 = df_sorted.iloc[0]
    if top3_share >= 60:
        return (
            f"Top 3 channels hold **{top3_share:.1f}%** of this category's slot observations, led by "
            f"**{top1['channel_title']}** at {top1['share_pct']:.1f}%. **Highly concentrated** — "
            "the category dominance is really a few-channel story."
        )
    if top3_share <= 30:
        return (
            f"Top 3 channels hold only **{top3_share:.1f}%** — **broadly distributed**. "
            "The category dominance comes from many creators contributing, not a small cartel."
        )
    return (
        f"Top 3 channels hold **{top3_share:.1f}%**, with **{top1['channel_title']}** "
        f"leading at {top1['share_pct']:.1f}%. **Moderately concentrated**."
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


# --- v2 diagnostic loaders (one pair per descriptive chart) ---------------
# Each pair mechanically decomposes the trend the descriptive chart shows.

@st.cache_data(ttl=120, show_spinner=False)
def load_views_volume_strength_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_views_volume_strength_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=120, show_spinner=False)
def load_views_new_vs_carryover_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_views_new_vs_carryover_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=120, show_spinner=False)
def load_duration_slot_footprint_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_duration_slot_footprint_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=120, show_spinner=False)
def load_duration_audience_response_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_duration_audience_response_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=120, show_spinner=False)
def load_tier_effort_reward_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_tier_effort_reward_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=120, show_spinner=False)
def load_tier_persistence_engagement_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_tier_persistence_engagement_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=120, show_spinner=False)
def load_tag_adoption_intensity_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_tag_adoption_intensity_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=120, show_spinner=False)
def load_tag_cooccurrence_diagnostic(category_name: str | None, trending_region: str | None, history_window: str):
    sdf = build_tag_cooccurrence_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=120, show_spinner=False)
def load_rank_slot_turnover_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_rank_slot_turnover_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


@st.cache_data(ttl=120, show_spinner=False)
def load_rank_channel_concentration_v2_diagnostic(category_name: str, trending_region: str | None, history_window: str):
    sdf = build_rank_channel_concentration_v2_diagnostic(
        get_spark(), SILVER_DELTA_PATH, history_window,
        category_name=category_name, trending_region=trending_region,
    )
    return _safe_pandas_from_spark(sdf)


gold_latest_snapshot_df = load_optional_delta(GOLD_LATEST_SNAPSHOT_PATH)
gold_category_summary_df = load_optional_delta(GOLD_CATEGORY_SUMMARY_PATH)
gold_views_timeseries_df = load_optional_delta(GOLD_VIEWS_TIMESERIES_PATH)
gold_region_timeseries_df = load_optional_delta(GOLD_REGION_TIMESERIES_PATH)
gold_channel_leaderboard_df = load_optional_delta(GOLD_CHANNEL_LEADERBOARD_PATH)
gold_duration_distribution_df = load_optional_delta(GOLD_DURATION_DISTRIBUTION_PATH)
gold_subscriber_tier_df = load_optional_delta(GOLD_SUBSCRIBER_TIER_PATH)
gold_tag_usage_df = load_optional_delta(GOLD_TAG_USAGE_PATH)
gold_trending_rank_dist_df = load_optional_delta(GOLD_TRENDING_RANK_DIST_PATH)

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

# --- Gold-first descriptive aggregations ---------------------------------
# All tab-1 charts read pre-aggregated Gold parquet whenever it is available
# (a tiny in-memory pandas filter against a small table — milliseconds).
# Spark on Silver is only the fallback path when the Gold table is missing
# (e.g. on a brand-new install before refresh_gold_tables has run).

def _filter_gold(df, category, region):
    if df is None or df.empty:
        return df
    out = df
    if category and "category_name" in out.columns:
        out = out[out["category_name"] == category]
    if region and "trending_region" in out.columns:
        out = out[out["trending_region"] == region]
    return out


# --- category summary --------------------------------------------------
# The new schema is per (category_name, trending_region) with extra sum_*
# columns. Older Gold files on disk only have category_name, so we detect
# the schema and pick the right path. This keeps the dashboard alive on
# old Gold until refresh_gold_tables runs again.
_cs_has_region = (
    not gold_category_summary_df.empty
    and "trending_region" in gold_category_summary_df.columns
)
_cs_has_sums = (
    _cs_has_region
    and {"sum_engagement_rate", "sum_like_rate", "sample_size"}.issubset(
        gold_category_summary_df.columns
    )
)

if _cs_has_region:
    cs_filtered = _filter_gold(gold_category_summary_df, diagnostic_category, diagnostic_region)
    if cs_filtered.empty:
        summary_df = cs_filtered
    elif diagnostic_region is not None:
        summary_df = cs_filtered.copy()
    elif _cs_has_sums:
        agg = cs_filtered.groupby("category_name", as_index=False).agg(
            videos=("videos", "sum"),
            total_views=("total_views", "sum"),
            total_likes=("total_likes", "sum"),
            total_comments=("total_comments", "sum"),
            sum_engagement_rate=("sum_engagement_rate", "sum"),
            sum_like_rate=("sum_like_rate", "sum"),
            sample_size=("sample_size", "sum"),
        )
        agg["avg_engagement_rate"] = (
            agg["sum_engagement_rate"] / agg["sample_size"].replace(0, pd.NA)
        ).fillna(0)
        agg["avg_like_rate"] = (
            agg["sum_like_rate"] / agg["sample_size"].replace(0, pd.NA)
        ).fillna(0)
        summary_df = agg.sort_values("total_views", ascending=False)
    else:
        # Region in schema but no sum_* columns — older transitional Gold.
        # Roll up sums correctly, weight averages by `videos`.
        agg = cs_filtered.groupby("category_name", as_index=False).agg(
            videos=("videos", "sum"),
            total_views=("total_views", "sum"),
            total_likes=("total_likes", "sum"),
            total_comments=("total_comments", "sum"),
            avg_engagement_rate_x_videos=(
                "avg_engagement_rate",
                lambda s: float((s * cs_filtered.loc[s.index, "videos"]).sum()),
            ),
            avg_like_rate_x_videos=(
                "avg_like_rate",
                lambda s: float((s * cs_filtered.loc[s.index, "videos"]).sum()),
            ),
        )
        agg["avg_engagement_rate"] = (
            agg["avg_engagement_rate_x_videos"] / agg["videos"].replace(0, pd.NA)
        ).fillna(0)
        agg["avg_like_rate"] = (
            agg["avg_like_rate_x_videos"] / agg["videos"].replace(0, pd.NA)
        ).fillna(0)
        summary_df = agg.drop(
            columns=["avg_engagement_rate_x_videos", "avg_like_rate_x_videos"]
        ).sort_values("total_views", ascending=False)
elif not gold_category_summary_df.empty and selected_region == "All" and selected_category == "All":
    # Old Gold (no region column) only safe to use when no filters are set.
    summary_df = gold_category_summary_df
else:
    summary_df = load_spark_category_summary(diagnostic_category, diagnostic_region, history_window)

# --- views timeseries --------------------------------------------------
_vt_has_region = (
    not gold_views_timeseries_df.empty
    and "trending_region" in gold_views_timeseries_df.columns
)
if _vt_has_region:
    vt_filtered = _filter_gold(gold_views_timeseries_df, diagnostic_category, diagnostic_region)
    if vt_filtered.empty:
        views_ts_df = vt_filtered
    elif diagnostic_region is not None:
        views_ts_df = vt_filtered.copy()
    else:
        views_ts_df = (
            vt_filtered.groupby(["category_name", "time_bucket"], as_index=False)
            .agg(
                total_views=("total_views", "sum"),
                total_engagements=("total_engagements", "sum"),
            )
        )
elif not gold_views_timeseries_df.empty and selected_region == "All" and selected_category == "All":
    views_ts_df = gold_views_timeseries_df
else:
    views_ts_df = load_spark_views_timeseries(diagnostic_category, diagnostic_region, history_window)

if not gold_duration_distribution_df.empty:
    duration_dist_df = _filter_gold(gold_duration_distribution_df, diagnostic_category, diagnostic_region)
else:
    duration_dist_df = load_spark_duration_distribution(diagnostic_category, diagnostic_region, history_window)

if not gold_subscriber_tier_df.empty:
    # Gold stores per (region, category, tier). Filter on user selection,
    # then re-aggregate to (region, tier) and recompute share within region
    # so the chart still shows correct percentages under any filter combo.
    sub_filtered = _filter_gold(gold_subscriber_tier_df, diagnostic_category, diagnostic_region)
    if sub_filtered.empty:
        subscriber_tier_df = sub_filtered
    else:
        sub_agg = (
            sub_filtered.groupby(["trending_region", "subscriber_tier"], as_index=False)
            ["video_count"].sum()
        )
        region_totals = sub_agg.groupby("trending_region")["video_count"].sum()
        sub_agg["pct"] = (
            sub_agg["video_count"] / sub_agg["trending_region"].map(region_totals) * 100.0
        ).fillna(0)
        subscriber_tier_df = sub_agg
else:
    subscriber_tier_df = load_spark_subscriber_tier_distribution(diagnostic_category, diagnostic_region, history_window)

if not gold_tag_usage_df.empty:
    # Gold stores per (tag, region, category). Filter on user selection, sum
    # videos_using_tag per tag, take top 30, then attach the dominant
    # (category, region) for each top tag so the chart still has color/tooltip.
    tag_filtered = _filter_gold(gold_tag_usage_df, diagnostic_category, diagnostic_region)
    if tag_filtered.empty:
        tag_usage_df = tag_filtered
    else:
        top_tags = (
            tag_filtered.groupby("tag", as_index=False)["videos_using_tag"].sum()
            .sort_values("videos_using_tag", ascending=False)
            .head(30)
        )
        # Dominant (category, region) per top tag.
        dominant = (
            tag_filtered.merge(top_tags[["tag"]], on="tag", how="inner")
            .sort_values("videos_using_tag", ascending=False)
            .drop_duplicates("tag")
            [["tag", "category_name", "trending_region"]]
        )
        tag_usage_df = top_tags.merge(dominant, on="tag", how="left")
else:
    tag_usage_df = load_spark_tag_usage_frequency(diagnostic_category, diagnostic_region, history_window)

if not gold_trending_rank_dist_df.empty:
    # Gold already stores video_count and pct_of_trending per (region, category)
    # against the full region total — that share remains valid under category
    # filtering (it's still "% of THAT region's trending slots").
    trending_rank_dist_df = _filter_gold(gold_trending_rank_dist_df, diagnostic_category, diagnostic_region)
else:
    trending_rank_dist_df = load_spark_trending_rank_distribution(diagnostic_category, diagnostic_region, history_window)

# --- Predictive pipeline (still pandas-backed) ---------------------------
# These wrap sklearn / regression code that genuinely needs numpy/pandas
# arrays, so the legacy pandas-based builders remain for tab 2.
video_forecast_df = build_view_count_forecast_v2(df)
duration_prediction_df = build_trending_duration_prediction(df)
peak_rank_forecast_df = build_peak_rank_forecast(df)
category_share_forecast_df = build_category_share_forecast(df)


# KPI metrics — computed entirely in Spark.
kpis = load_spark_kpis(diagnostic_category, diagnostic_region, history_window)


def _human_readable_views(n: int) -> str:
    """Compact view count: 1.2B / 345M / 12K so the metric tile stays readable."""
    n = int(n or 0)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


k1, k2, k3, k4 = st.columns(4)
k1.metric("Unique Videos", f"{kpis['unique_videos']:,}")
k2.metric("Total Views", _human_readable_views(kpis["total_views"]))
k3.metric("Avg Engagement Rate", f"{kpis['avg_engagement_rate'] * 100:.2f}%")
k4.metric("Tracked Categories", f"{kpis['tracked_categories']:,}")

tab1, tab2 = st.tabs(
    ["Descriptive", "Predictive"]
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
            diagnostic_region_local = None if selected_region == "All" else selected_region
            volume_strength_df = load_views_volume_strength_diagnostic(
                str(selected_views_category), diagnostic_region_local, history_window,
            )
            new_carryover_df = load_views_new_vs_carryover_diagnostic(
                str(selected_views_category), diagnostic_region_local, history_window,
            )

            st.markdown(
                f"**Why is `{selected_views_category}` moving?**"
                + (f" in `{diagnostic_region_local}`" if diagnostic_region_local else "")
            )
            diag_col1, diag_col2 = st.columns(2)

            with diag_col1:
                st.markdown("**Diagnostic 1: Volume vs Strength Decomposition**")
                st.caption("Total views = unique videos × avg views per video. Tells you which lever moved.")
                if volume_strength_df.empty:
                    st.info("Not enough data for this category yet.")
                else:
                    volume_strength_chart = (
                        alt.Chart(volume_strength_df).encode(x=alt.X("time_bucket:T", title="Time"))
                    )
                    videos_line = volume_strength_chart.mark_line(point=True, color="#0ea5e9").encode(
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
                    avg_line = volume_strength_chart.mark_line(point=True, color="#f59e0b").encode(
                        y=alt.Y(
                            "avg_views_per_video:Q",
                            title="Avg Views per Video",
                            axis=alt.Axis(format="~s", titleColor="#f59e0b"),
                        ),
                    )
                    st.altair_chart(
                        alt.layer(videos_line, avg_line)
                        .resolve_scale(y="independent")
                        .properties(height=320),
                        width="stretch",
                    )
                    _explanation(_explain_views_volume_strength(volume_strength_df))

            with diag_col2:
                st.markdown("**Diagnostic 2: New Entrants vs Carry-overs**")
                st.caption("Where are the videos in this category coming from each bucket — fresh entries, or returning?")
                if new_carryover_df.empty:
                    st.info("Not enough data for this category yet.")
                else:
                    new_carry_chart = (
                        alt.Chart(new_carryover_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("time_bucket:T", title="Time"),
                            y=alt.Y("videos:Q", title="Distinct Videos", stack="zero"),
                            color=alt.Color(
                                "entry_status:N",
                                title="Entry Status",
                                scale=alt.Scale(
                                    domain=["New entry", "Carry-over"],
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
                    st.altair_chart(new_carry_chart, width="stretch")
                    _explanation(_explain_views_new_vs_carryover(new_carryover_df))

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
            diagnostic_region_local = None if selected_region == "All" else selected_region
            slot_footprint_df = load_duration_slot_footprint_diagnostic(
                str(selected_duration_category) if selected_duration_category else None,
                diagnostic_region_local,
                history_window,
            )
            audience_response_df = load_duration_audience_response_diagnostic(
                str(selected_duration_category) if selected_duration_category else None,
                diagnostic_region_local,
                history_window,
            )

            st.markdown(
                "**Why does this duration distribution look the way it does?**"
                + (f" for `{selected_duration_category}`" if selected_duration_category else "")
            )

            dur_col1, dur_col2 = st.columns(2)

            with dur_col1:
                st.markdown("**Diagnostic A: Slot Footprint Decomposition**")
                st.caption("Slot footprint = distinct videos × avg batches each persists. Reveals whether bucket dominance is volume-led or persistence-led.")
                if slot_footprint_df.empty:
                    st.info("Not enough data for this selection.")
                else:
                    # Two-panel grouped chart: one bar per (duration_bucket, lever).
                    melted = slot_footprint_df.melt(
                        id_vars=["duration_bucket"],
                        value_vars=["distinct_videos", "avg_batches_per_video"],
                        var_name="lever",
                        value_name="value",
                    )
                    melted["lever"] = melted["lever"].map({
                        "distinct_videos": "Distinct videos (volume)",
                        "avg_batches_per_video": "Avg batches each (persistence)",
                    })
                    if selected_duration_bucket:
                        melted["is_selected"] = melted["duration_bucket"] == selected_duration_bucket
                    else:
                        melted["is_selected"] = False
                    slot_chart = (
                        alt.Chart(melted)
                        .mark_bar()
                        .encode(
                            x=alt.X("duration_bucket:N", title="Duration Bucket"),
                            y=alt.Y("value:Q", title="Value"),
                            color=alt.Color("lever:N", title="Lever"),
                            column=alt.Column(
                                "lever:N",
                                title=None,
                                header=alt.Header(labelFontWeight="bold"),
                            ),
                            opacity=alt.condition(
                                alt.datum.is_selected,
                                alt.value(1.0),
                                alt.value(0.55),
                            ),
                            tooltip=[
                                "duration_bucket",
                                "lever",
                                alt.Tooltip("value:Q", title="Value", format=",.2f"),
                            ],
                        )
                        .properties(width=180, height=280)
                        .resolve_scale(y="independent")
                    )
                    st.altair_chart(slot_chart, width="stretch")
                    _explanation(_explain_duration_slot_footprint(slot_footprint_df))

            with dur_col2:
                st.markdown("**Diagnostic B: Audience Response per Bucket**")
                st.caption("Avg engagement rate × avg views per video. Reveals whether bucket dominance reflects audience preference or just publishing volume.")
                if audience_response_df.empty:
                    st.info("Not enough data for this selection.")
                else:
                    ar_base = alt.Chart(audience_response_df).encode(
                        x=alt.X("duration_bucket:N", title="Duration Bucket"),
                    )
                    er_bars = ar_base.mark_bar(color="#a855f7").encode(
                        y=alt.Y(
                            "avg_engagement_rate:Q",
                            title="Avg Engagement Rate",
                            axis=alt.Axis(format=".1%", titleColor="#a855f7"),
                        ),
                        tooltip=[
                            "duration_bucket",
                            alt.Tooltip("avg_engagement_rate:Q", title="Avg Engagement Rate", format=".2%"),
                            alt.Tooltip("avg_views_per_video:Q", title="Avg Views/Video", format=","),
                            alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                        ],
                    )
                    views_line = ar_base.mark_line(color="#f59e0b", point=True).encode(
                        y=alt.Y(
                            "avg_views_per_video:Q",
                            title="Avg Views per Video",
                            axis=alt.Axis(format="~s", titleColor="#f59e0b"),
                        ),
                    )
                    st.altair_chart(
                        alt.layer(er_bars, views_line)
                        .resolve_scale(y="independent")
                        .properties(height=320),
                        width="stretch",
                    )
                    _explanation(_explain_duration_audience_response(audience_response_df))

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
            if selected_subscriber_region:
                diag_region_local = str(selected_subscriber_region)
            else:
                diag_region_local = None if selected_region == "All" else selected_region
            diag_category_local = None  # subscriber-tier section is category-agnostic

            tier_effort_df = load_tier_effort_reward_diagnostic(
                diag_category_local, diag_region_local, history_window,
            )
            tier_persistence_df = load_tier_persistence_engagement_diagnostic(
                diag_category_local, diag_region_local, history_window,
            )

            scope_bits = []
            if selected_subscriber_tier:
                scope_bits.append(f"tier `{selected_subscriber_tier}`")
            if selected_subscriber_region:
                scope_bits.append(f"region `{selected_subscriber_region}`")
            st.markdown(
                "**Why does this tier mix look the way it does?**" + (
                    f" — {' in '.join(scope_bits)}" if scope_bits else ""
                )
            )

            tier_order = ["Small (<100K)", "Mid (100K-1M)", "Large (1M-10M)", "Mega (10M+)"]

            sub_col1, sub_col2 = st.columns(2)

            with sub_col1:
                st.markdown("**Diagnostic A: Effort vs Reward per Tier**")
                st.caption("Distinct channels × avg trending slots per channel. Decomposes broad participation vs concentrated dominance.")
                if tier_effort_df.empty:
                    st.info("Not enough data for this selection.")
                else:
                    melted = tier_effort_df.melt(
                        id_vars=["subscriber_tier"],
                        value_vars=["distinct_channels", "avg_slots_per_channel"],
                        var_name="lever",
                        value_name="value",
                    )
                    melted["lever"] = melted["lever"].map({
                        "distinct_channels": "Distinct channels (participation)",
                        "avg_slots_per_channel": "Avg slots/channel (intensity)",
                    })
                    if selected_subscriber_tier:
                        melted["is_selected"] = melted["subscriber_tier"] == selected_subscriber_tier
                    else:
                        melted["is_selected"] = False
                    effort_chart = (
                        alt.Chart(melted)
                        .mark_bar()
                        .encode(
                            x=alt.X("subscriber_tier:N", sort=tier_order, title="Subscriber Tier"),
                            y=alt.Y("value:Q", title="Value"),
                            color=alt.Color("lever:N", title="Lever"),
                            column=alt.Column("lever:N", title=None),
                            opacity=alt.condition(
                                alt.datum.is_selected, alt.value(1.0), alt.value(0.55)
                            ),
                            tooltip=[
                                "subscriber_tier",
                                "lever",
                                alt.Tooltip("value:Q", title="Value", format=",.2f"),
                            ],
                        )
                        .properties(width=160, height=280)
                        .resolve_scale(y="independent")
                    )
                    st.altair_chart(effort_chart, width="stretch")
                    _explanation(_explain_tier_effort_reward(tier_effort_df))

            with sub_col2:
                st.markdown("**Diagnostic B: Persistence × Engagement per Tier**")
                st.caption("Avg batches each video persists × avg engagement rate. Decomposes whether a tier holds slots through stickiness or audience response.")
                if tier_persistence_df.empty:
                    st.info("Not enough data for this selection.")
                else:
                    pe_base = alt.Chart(tier_persistence_df).encode(
                        x=alt.X("subscriber_tier:N", sort=tier_order, title="Subscriber Tier"),
                    )
                    persistence_bars = pe_base.mark_bar(color="#22c55e").encode(
                        y=alt.Y(
                            "avg_batches_per_video:Q",
                            title="Avg Batches per Video",
                            axis=alt.Axis(titleColor="#22c55e"),
                        ),
                        tooltip=[
                            "subscriber_tier",
                            alt.Tooltip("avg_batches_per_video:Q", title="Avg Batches", format=".2f"),
                            alt.Tooltip("avg_engagement_rate:Q", title="Avg ER", format=".2%"),
                            alt.Tooltip("unique_videos:Q", title="Unique Videos", format=","),
                        ],
                    )
                    er_line = pe_base.mark_line(color="#a855f7", point=True).encode(
                        y=alt.Y(
                            "avg_engagement_rate:Q",
                            title="Avg Engagement Rate",
                            axis=alt.Axis(format=".1%", titleColor="#a855f7"),
                        ),
                    )
                    st.altair_chart(
                        alt.layer(persistence_bars, er_line)
                        .resolve_scale(y="independent")
                        .properties(height=320),
                        width="stretch",
                    )
                    _explanation(_explain_tier_persistence_engagement(tier_persistence_df))

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
            diag_region_local = None if selected_region == "All" else selected_region
            tag_adoption_df = load_tag_adoption_intensity_diagnostic(
                str(selected_tag_category) if selected_tag_category else None,
                diag_region_local, history_window,
            )
            tag_cooccur_df = load_tag_cooccurrence_diagnostic(
                str(selected_tag_category) if selected_tag_category else None,
                diag_region_local, history_window,
            )

            st.markdown(
                "**Why are these tags dominating?**"
                + (f" for `{selected_tag_category}`" if selected_tag_category else "")
            )

            tag_col1, tag_col2 = st.columns(2)

            with tag_col1:
                st.markdown("**Diagnostic A: Channel Adoption vs Per-Channel Intensity**")
                st.caption("For top tags: distinct channels using × avg videos per channel. Decomposes whether a tag is broadly adopted or flooded by a few creators.")
                if tag_adoption_df.empty:
                    st.info("Not enough tag data in scope.")
                else:
                    if selected_tag:
                        tag_adoption_df = tag_adoption_df.copy()
                        tag_adoption_df["is_selected"] = tag_adoption_df["tag"] == selected_tag
                    else:
                        tag_adoption_df = tag_adoption_df.copy()
                        tag_adoption_df["is_selected"] = False
                    adoption_chart = (
                        alt.Chart(tag_adoption_df)
                        .mark_circle(opacity=0.8)
                        .encode(
                            x=alt.X(
                                "distinct_channels:Q",
                                title="Distinct Channels Using Tag (adoption)",
                            ),
                            y=alt.Y(
                                "videos_per_channel:Q",
                                title="Videos per Channel (intensity)",
                            ),
                            size=alt.Size(
                                "videos_using_tag:Q",
                                title="Videos Using Tag",
                            ),
                            color=alt.condition(
                                alt.datum.is_selected,
                                alt.value("#f97316"),
                                alt.value("#0ea5e9"),
                            ),
                            tooltip=[
                                "tag",
                                alt.Tooltip("videos_using_tag:Q", title="Total Videos", format=","),
                                alt.Tooltip("distinct_channels:Q", title="Channels", format=","),
                                alt.Tooltip("videos_per_channel:Q", title="Videos/Channel", format=".2f"),
                            ],
                        )
                        .properties(height=360)
                    )
                    st.altair_chart(adoption_chart, width="stretch")
                    _explanation(_explain_tag_adoption_intensity(tag_adoption_df))

            with tag_col2:
                st.markdown("**Diagnostic B: Co-occurrence with the Top Tag**")
                st.caption("Other tags that appear alongside the most-used tag. Reveals whether the dominance is part of a multi-tag template.")
                if tag_cooccur_df.empty:
                    st.info("Not enough data to compute co-occurrence.")
                else:
                    cooccur_chart = (
                        alt.Chart(tag_cooccur_df)
                        .mark_bar(color="#a855f7")
                        .encode(
                            x=alt.X("co_share:Q", title="Co-occurrence Share", axis=alt.Axis(format=".0%")),
                            y=alt.Y("tag:N", sort="-x", title="Co-occurring Tag"),
                            tooltip=[
                                "tag",
                                "primary_tag",
                                alt.Tooltip("co_videos:Q", title="Co-occurring Videos", format=","),
                                alt.Tooltip("co_share:Q", title="Share", format=".1%"),
                            ],
                        )
                        .properties(height=360)
                    )
                    st.altair_chart(cooccur_chart, width="stretch")
                    _explanation(_explain_tag_cooccurrence(tag_cooccur_df))


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
            if selected_rank_region:
                rank_diag_region = str(selected_rank_region)
            else:
                rank_diag_region = None if selected_region == "All" else selected_region

            slot_turnover_df = load_rank_slot_turnover_diagnostic(
                str(selected_rank_category), rank_diag_region, history_window
            )
            channel_concentration_df = load_rank_channel_concentration_v2_diagnostic(
                str(selected_rank_category), rank_diag_region, history_window
            )

            scope_bits = [f"`{selected_rank_category}`"]
            if selected_rank_region:
                scope_bits.append(f"region `{selected_rank_region}`")
            st.markdown("**Why does this category hold this many slots?** — " + " in ".join(scope_bits))

            rank_col1, rank_col2 = st.columns(2)

            with rank_col1:
                st.markdown("**Diagnostic A: Slot Turnover Rate**")
                st.caption("Distinct videos / slot observations per region. High = vibrant pipeline; low = a few persistent videos hogging the category.")
                if slot_turnover_df.empty:
                    st.info("Not enough data for this selection.")
                else:
                    turnover_chart = (
                        alt.Chart(slot_turnover_df)
                        .mark_bar(color="#06b6d4")
                        .encode(
                            x=alt.X("trending_region:N", title="Region"),
                            y=alt.Y("turnover_rate:Q", title="Turnover Rate (distinct / slot obs.)"),
                            tooltip=[
                                "trending_region",
                                alt.Tooltip("distinct_videos:Q", title="Distinct Videos", format=","),
                                alt.Tooltip("slot_observations:Q", title="Slot Observations", format=","),
                                alt.Tooltip("turnover_rate:Q", title="Turnover", format=".2f"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(turnover_chart, width="stretch")
                    _explanation(_explain_rank_slot_turnover(slot_turnover_df))

            with rank_col2:
                st.markdown("**Diagnostic B: Channel Concentration (Herfindahl-style)**")
                st.caption("Each channel's share of all category slot observations. Top-3 share signals concentrated dominance vs broad distribution.")
                if channel_concentration_df.empty:
                    st.info("Not enough channel data for this selection.")
                else:
                    concentration_chart = (
                        alt.Chart(channel_concentration_df)
                        .mark_bar(color="#f97316")
                        .encode(
                            x=alt.X("share_pct:Q", title="Share of Category Slots (%)"),
                            y=alt.Y("channel_title:N", sort="-x", title="Channel"),
                            tooltip=[
                                "channel_title",
                                alt.Tooltip("slot_observations:Q", title="Slot Observations", format=","),
                                alt.Tooltip("share_pct:Q", title="Share (%)", format=".2f"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(concentration_chart, width="stretch")
                    _explanation(_explain_rank_channel_concentration_v2(channel_concentration_df))




with tab2:
    st.subheader("2. What will happen?")
    st.markdown("Predictive analytics estimates how current videos and categories may evolve next. Click a prediction chart to see action suggestions directly underneath it.")

    st.subheader("View Count Forecasting")
    st.markdown("Mode 3 per-video forecasting. The graph projects future view counts for currently trending videos with a simple confidence band.")

    if video_forecast_df.empty:
        st.info("View forecasting needs at least 4 observations for a currently trending video.")
    else:
        forecast_select = alt.selection_point(
            name="view_forecast_select",
            fields=["title"],
            on="click",
            toggle=False,
            empty=False,
        )
        actual_line = (
            alt.Chart(video_forecast_df[video_forecast_df["series"] == "Actual"])
            .mark_line(point=True)
            .encode(
                x=alt.X("time_bucket:T", title="Time"),
                y=alt.Y("forecast_views:Q", title="View Count"),
                color=alt.Color("title:N", title="Video"),
                opacity=alt.condition(forecast_select, alt.value(1), alt.value(0.35)),
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
                opacity=alt.condition(forecast_select, alt.value(1), alt.value(0.35)),
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
        forecast_chart = (
            (confidence_band + actual_line + forecast_line)
            .add_params(forecast_select)
            .properties(height=450)
        )
        forecast_event = st.altair_chart(
            forecast_chart,
            width="stretch",
            on_select="rerun",
            selection_mode="view_forecast_select",
            key="view_forecast_chart",
        )
        _model_metrics_strip(
            "View Count Forecast",
            load_view_forecast_metrics(df),
        )
        forecast_context = _extract_chart_selection(
            forecast_event,
            "view_forecast_select",
            ["title"],
        )
        selected_forecast_title = _normalize_context_value(forecast_context.get("title"))
        if selected_forecast_title:
            selected_forecast_rows = video_forecast_df[
                video_forecast_df["title"] == selected_forecast_title
            ].copy()
            _render_suggestion_box(
                f"Suggestions for `{selected_forecast_title}`",
                _suggest_from_view_forecast(selected_forecast_rows),
            )

    st.subheader("Trending Duration Prediction")
    st.markdown("Historical episodes are used to estimate total trending lifespan and remaining time for the videos in the current batch.")

    if duration_prediction_df.empty:
        st.info("Trending duration prediction needs more accumulated history before it becomes stable.")
    else:
        duration_select = alt.selection_point(
            name="duration_prediction_select",
            fields=["title"],
            on="click",
            toggle=False,
            empty=False,
        )
        duration_prediction_chart = (
            alt.Chart(duration_prediction_df)
            .mark_bar()
            .encode(
                x=alt.X("predicted_remaining_hours:Q", title="Predicted Remaining Hours"),
                y=alt.Y("title:N", sort="-x", title="Video"),
                color=alt.Color("category_name:N", title="Category"),
                opacity=alt.condition(duration_select, alt.value(1), alt.value(0.4)),
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
            .add_params(duration_select)
            .properties(height=500)
        )
        duration_event = st.altair_chart(
            duration_prediction_chart,
            width="stretch",
            on_select="rerun",
            selection_mode="duration_prediction_select",
            key="duration_prediction_chart",
        )
        _model_metrics_strip(
            "Trending Duration Prediction",
            load_duration_prediction_metrics(df),
        )
        duration_context = _extract_chart_selection(
            duration_event,
            "duration_prediction_select",
            ["title"],
        )
        selected_duration_title = _normalize_context_value(duration_context.get("title"))
        if selected_duration_title:
            duration_row = duration_prediction_df[
                duration_prediction_df["title"] == selected_duration_title
            ].iloc[0]
            _render_suggestion_box(
                f"Suggestions for `{selected_duration_title}`",
                _suggest_from_duration_prediction(duration_row),
            )

    st.subheader("Peak Rank and Category Forecasting")
    st.markdown("These charts estimate the best future rank a current video may reach and the next category-share trajectory using the accumulated time-series history.")

    if peak_rank_forecast_df.empty:
        st.info("Peak-rank forecasting needs more historical trajectories first.")
    else:
        peak_rank_select = alt.selection_point(
            name="peak_rank_select",
            fields=["title"],
            on="click",
            toggle=False,
            empty=False,
        )
        peak_rank_chart = (
            alt.Chart(peak_rank_forecast_df)
            .mark_bar()
            .encode(
                x=alt.X("expected_rank_gain:Q", title="Expected Rank Improvement"),
                y=alt.Y("title:N", sort="-x", title="Video"),
                color=alt.Color("category_name:N", title="Category"),
                opacity=alt.condition(peak_rank_select, alt.value(1), alt.value(0.4)),
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
            .add_params(peak_rank_select)
            .properties(height=450)
        )
        peak_rank_event = st.altair_chart(
            peak_rank_chart,
            width="stretch",
            on_select="rerun",
            selection_mode="peak_rank_select",
            key="peak_rank_chart",
        )
        _model_metrics_strip(
            "Peak Rank Forecast",
            load_peak_rank_metrics(df),
        )
        peak_context = _extract_chart_selection(
            peak_rank_event,
            "peak_rank_select",
            ["title"],
        )
        selected_peak_title = _normalize_context_value(peak_context.get("title"))
        if selected_peak_title:
            peak_row = peak_rank_forecast_df[
                peak_rank_forecast_df["title"] == selected_peak_title
            ].iloc[0]
            _render_suggestion_box(
                f"Suggestions for `{selected_peak_title}`",
                _suggest_from_peak_rank_forecast(peak_row),
            )

    if category_share_forecast_df.empty:
        st.info("Category-share forecasting needs at least 2 time buckets per category.")
    else:
        share_select = alt.selection_point(
            name="category_share_select",
            fields=["category_name"],
            on="click",
            toggle=False,
            empty=False,
            nearest=True,
        )
        share_base = alt.Chart(category_share_forecast_df).encode(
            x=alt.X("time_bucket:T", title="Time"),
            y=alt.Y("forecast_share:Q", title="Forecast Category Share"),
            color=alt.Color("category_name:N", title="Category"),
            opacity=alt.condition(share_select, alt.value(1), alt.value(0.35)),
            strokeDash=alt.StrokeDash("series:N", title="Series"),
            tooltip=[
                "category_name",
                "series",
                "time_bucket",
                alt.Tooltip("forecast_share:Q", title="Category Share", format=".2%"),
            ],
        )
        share_lines = share_base.mark_line()
        share_points = (
            share_base.mark_point(size=90, filled=True)
            .add_params(share_select)
        )
        category_share_chart = (
            alt.layer(share_lines, share_points)
            .properties(height=400)
        )
        share_event = st.altair_chart(
            category_share_chart,
            width="stretch",
            on_select="rerun",
            selection_mode="category_share_select",
            key="category_share_chart",
        )
        _model_metrics_strip(
            "Category Share Forecast",
            load_category_share_metrics(df),
        )
        share_context = _extract_chart_selection(
            share_event,
            "category_share_select",
            ["category_name"],
        )
        selected_share_category = _normalize_context_value(share_context.get("category_name"))
        if selected_share_category:
            share_rows = category_share_forecast_df[
                category_share_forecast_df["category_name"] == selected_share_category
            ].copy()
            _render_suggestion_box(
                f"Suggestions for `{selected_share_category}`",
                _suggest_from_category_share_forecast(share_rows),
            )

with st.expander("Raw Data"):
    st.dataframe(df, width="stretch")

import time

time.sleep(10)
st.rerun()
