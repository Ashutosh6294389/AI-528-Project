import json
import math
import time
from datetime import datetime, timezone

import requests
from kafka import KafkaProducer

from config import (
    API_KEY,
    KAFKA_BROKER,
    TOPIC,
    REGIONS,
    CATEGORY_IDS,
    MAX_RESULTS,
    REQUEST_TIMEOUT,
    TARGET_RUNTIME_HOURS,
    DAILY_QUOTA_UNITS,
    INITIAL_POLL_SECONDS,
    TARGET_VIDEO_CALLS_PER_CYCLE,
)

BASE_URL = "https://www.googleapis.com/youtube/v3"

if not API_KEY or API_KEY == "YOUR_REAL_YOUTUBE_API_KEY":
    raise ValueError("Set your real YouTube Data API key in data_ingestion/config.py")

session = requests.Session()
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

unsupported_pairs = set()
cached_category_maps = {}

FALLBACK_CATEGORY_MAP = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "20": "Gaming",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
}

PAIR_CURSOR = 0
current_poll_seconds = INITIAL_POLL_SECONDS
RUN_START = time.time()
QUOTA_USED = 0


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def fetch_json(endpoint, params):
    response = session.get(f"{BASE_URL}/{endpoint}", params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_category_map(region):
    global QUOTA_USED

    if region in cached_category_maps:
        return cached_category_maps[region]

    payload = fetch_json(
        "videoCategories",
        {"part": "snippet", "regionCode": region, "key": API_KEY},
    )
    QUOTA_USED += 1

    category_map = {item["id"]: item["snippet"]["title"] for item in payload.get("items", [])}
    merged = {**FALLBACK_CATEGORY_MAP, **category_map}
    cached_category_maps[region] = merged
    return merged


def fetch_popular_videos(region, category_id=None, page_token=None):
    params = {
        "part": "snippet,statistics,contentDetails",
        "chart": "mostPopular",
        "regionCode": region,
        "maxResults": MAX_RESULTS,
        "key": API_KEY,
    }
    if category_id:
        params["videoCategoryId"] = category_id
    if page_token:
        params["pageToken"] = page_token
    return fetch_json("videos", params)


def build_record(item, region, category_map, fetched_at):
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    content = item.get("contentDetails", {})

    views = safe_int(stats.get("viewCount"))
    likes = safe_int(stats.get("likeCount"))
    comments = safe_int(stats.get("commentCount"))
    engagements = likes + comments

    category_id = snippet.get("categoryId", "")
    category_name = category_map.get(category_id) or FALLBACK_CATEGORY_MAP.get(category_id) or "Other"

    return {
        "video_id": item.get("id"),
        "title": snippet.get("title", ""),
        "channel_title": snippet.get("channelTitle", "Unknown"),
        "published_at": snippet.get("publishedAt"),
        "fetched_at": fetched_at,
        "region": region,
        "category_id": category_id,
        "category": category_name,
        "duration": content.get("duration"),
        "views": views,
        "likes": likes,
        "comments": comments,
        "engagements": engagements,
        "like_rate": round(likes / views, 6) if views else 0.0,
        "comment_rate": round(comments / views, 6) if views else 0.0,
        "engagement_rate": round(engagements / views, 6) if views else 0.0,
    }


def get_active_pairs():
    pairs = []
    for region in REGIONS:
        category_map = fetch_category_map(region)
        valid_categories = [c for c in CATEGORY_IDS if c in category_map]
        for category_id in valid_categories:
            if (region, category_id) not in unsupported_pairs:
                pairs.append((region, category_id))
    return pairs


def fetch_youtube_data():
    global PAIR_CURSOR, QUOTA_USED

    fetched_at = datetime.now(timezone.utc).isoformat()
    records = []
    seen = set()
    raw_items = 0
    video_calls = 0

    pairs = get_active_pairs()
    if not pairs:
        return records, raw_items, video_calls

    # Round-robin over region/category pairs so we spread coverage over time
    count = min(TARGET_VIDEO_CALLS_PER_CYCLE, len(pairs))
    selected_pairs = []
    for i in range(count):
        idx = (PAIR_CURSOR + i) % len(pairs)
        selected_pairs.append(pairs[idx])

    PAIR_CURSOR = (PAIR_CURSOR + count) % len(pairs)

    for region, category_id in selected_pairs:
        try:
            category_map = cached_category_maps.get(region, FALLBACK_CATEGORY_MAP)
            payload = fetch_popular_videos(region=region, category_id=category_id)
            QUOTA_USED += 1
            video_calls += 1

            items = payload.get("items", [])
            raw_items += len(items)

            for item in items:
                record = build_record(item, region, category_map, fetched_at)
                dedupe_key = (record["video_id"], record["region"])
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    records.append(record)

        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 404:
                unsupported_pairs.add((region, category_id))
            else:
                print(f"[WARN] Failed for region={region}, category={category_id}: {exc}")
        except Exception as exc:
            print(f"[WARN] Failed for region={region}, category={category_id}: {exc}")

    return records, raw_items, video_calls


def compute_next_sleep(cycle_seconds):
    elapsed = max(1, time.time() - RUN_START)
    quota_per_second_target = DAILY_QUOTA_UNITS / (TARGET_RUNTIME_HOURS * 3600)
    quota_per_second_actual = QUOTA_USED / elapsed

    # If we are behind target, shorten sleep; if ahead, lengthen it.
    if quota_per_second_actual < quota_per_second_target:
        return max(5, current_poll_seconds - 5)
    return min(120, current_poll_seconds + 5)



if __name__ == "__main__":
    try:
        while True:
            cycle_start = time.time()

            batch, raw_items, video_calls = fetch_youtube_data()

            for record in batch:
                producer.send(TOPIC, value=record)
            producer.flush()

            cycle_seconds = round(time.time() - cycle_start, 2)
            elapsed_hours = (time.time() - RUN_START) / 3600
            target_used = DAILY_QUOTA_UNITS * min(elapsed_hours / TARGET_RUNTIME_HOURS, 1.0)

            print(
                f"sent={len(batch)} unique | raw={raw_items} | "
                f"video_calls={video_calls} | quota_used~={QUOTA_USED} | "
                f"target_quota_now~={int(target_used)} | cycle_time={cycle_seconds}s | "
                f"sleep={current_poll_seconds}s | ts={datetime.now().strftime('%H:%M:%S')}"
            )

            current_poll_seconds = compute_next_sleep(cycle_seconds)
            time.sleep(current_poll_seconds)


    except KeyboardInterrupt:
        print("\nProducer stopped by user.")
        producer.flush()
        producer.close()
