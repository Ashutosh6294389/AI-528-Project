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
    MAX_RESULTS,
    PAGES_PER_REGION,
    POLL_SECONDS,
    REQUEST_TIMEOUT,
)

BASE_URL = "https://www.googleapis.com/youtube/v3"
session = requests.Session()

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

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

def fetch_popular_videos(region, page_token=None):
    params = {
        "part": "snippet,statistics,contentDetails",
        "chart": "mostPopular",
        "regionCode": region,
        "maxResults": MAX_RESULTS,
        "key": API_KEY,
    }
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

    return {
        "video_id": item.get("id"),
        "title": snippet.get("title", ""),
        "channel_title": snippet.get("channelTitle", "Unknown"),
        "published_at": snippet.get("publishedAt"),
        "fetched_at": fetched_at,
        "region": region,
        "category_id": snippet.get("categoryId"),
        "category": category_map.get(snippet.get("categoryId"), "Other"),
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
            next_token = None

            for _ in range(PAGES_PER_REGION):
                payload = fetch_popular_videos(region, page_token=next_token)

                for item in payload.get("items", []):
                    record = build_record(item, region, category_map, fetched_at)
                    dedupe_key = (record["video_id"], record["region"])

                    if dedupe_key not in seen:
                        seen.add(dedupe_key)
                        records.append(record)

                next_token = payload.get("nextPageToken")
                if not next_token:
                    break

        except Exception as exc:
            print(f"[WARN] Region {region} failed: {exc}")

    return records

if __name__ == "__main__":
    while True:
        try:
            batch = fetch_youtube_data()

            for record in batch:
                producer.send(TOPIC, value=record)

            producer.flush()
            print(f"Sent {len(batch)} enriched YouTube records to Kafka")
        except Exception as exc:
            print(f"[ERROR] Producer failed: {exc}")

        time.sleep(POLL_SECONDS)
