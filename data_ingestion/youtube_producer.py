import json
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
    PAGES_PER_CATEGORY,
    POLL_SECONDS,
    REQUEST_TIMEOUT,
)

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


BASE_URL = "https://www.googleapis.com/youtube/v3"

if not API_KEY or API_KEY == "YOUR_REAL_YOUTUBE_API_KEY":
    raise ValueError("Set your real YouTube Data API key in data_ingestion/config.py")

session = requests.Session()

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

unsupported_pairs = set()


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def fetch_json(endpoint, params):
    response = session.get(
        f"{BASE_URL}/{endpoint}",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def fetch_category_map(region):
    payload = fetch_json(
        "videoCategories",
        {
            "part": "snippet",
            "regionCode": region,
            "key": API_KEY,
        },
    )
    return {
        item["id"]: item["snippet"]["title"]
        for item in payload.get("items", [])
    }


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


def fetch_youtube_data():
    fetched_at = datetime.now(timezone.utc).isoformat()
    records = []
    seen = set()

    for region in REGIONS:
        try:
            category_map = fetch_category_map(region)
        except Exception as exc:
            print(f"[WARN] Could not fetch category map for {region}: {exc}")
            category_map = {}

        region_categories = [cat for cat in CATEGORY_IDS if cat in category_map]
        if not region_categories:
            region_categories = CATEGORY_IDS[:]

        for category_id in region_categories:
            if (region, category_id) in unsupported_pairs:
                continue

            next_token = None

            for _ in range(PAGES_PER_CATEGORY):
                try:
                    payload = fetch_popular_videos(
                        region=region,
                        category_id=category_id,
                        page_token=next_token,
                    )

                    items = payload.get("items", [])
                    for item in items:
                        record = build_record(item, region, category_map, fetched_at)
                        dedupe_key = (record["video_id"], record["region"])

                        if dedupe_key not in seen:
                            seen.add(dedupe_key)
                            records.append(record)

                    next_token = payload.get("nextPageToken")
                    if not next_token:
                        break

                except requests.HTTPError as exc:
                    status_code = exc.response.status_code if exc.response is not None else None

                    if status_code == 404:
                        unsupported_pairs.add((region, category_id))
                        print(f"[INFO] Skipping unsupported category {category_id} for region {region}")
                    else:
                        print(f"[WARN] Failed for region={region}, category={category_id}: {exc}")
                    break

                except Exception as exc:
                    print(f"[WARN] Failed for region={region}, category={category_id}: {exc}")
                    break

    return records


if __name__ == "__main__":
    while True:
        try:
            batch = fetch_youtube_data()

            for record in batch:
                producer.send(TOPIC, value=record)

            producer.flush()
            print(f"Sent {len(batch)} enriched YouTube records to Kafka at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as exc:
            print(f"[ERROR] Producer failed: {exc}")

        time.sleep(POLL_SECONDS)
