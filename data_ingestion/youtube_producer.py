import json
import time
import uuid
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
    CHANNEL_BATCH_SIZE,
    DESCRIPTION_LIMIT,
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
    category_map = {
        item["id"]: item["snippet"]["title"]
        for item in payload.get("items", [])
    }
    return {**FALLBACK_CATEGORY_MAP, **category_map}


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


def chunked(values, size):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def fetch_channel_details(channel_ids):
    details = {}
    if not channel_ids:
        return details

    for chunk in chunked(sorted(set(channel_ids)), CHANNEL_BATCH_SIZE):
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            try:
                payload = fetch_json(
                    "channels",
                    {
                        "part": "snippet,statistics",
                        "id": ",".join(chunk),
                        "key": API_KEY,
                    },
                )

                for item in payload.get("items", []):
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})

                    details[item.get("id")] = {
                        "channel_id": item.get("id"),
                        "channel_title": snippet.get("title", "Unknown"),
                        "channel_subscriber_count": safe_int(stats.get("subscriberCount")),
                        "channel_view_count": safe_int(stats.get("viewCount")),
                        "channel_video_count": safe_int(stats.get("videoCount")),
                        "channel_country": snippet.get("country"),
                    }

                break

            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                attempts += 1

                if status_code == 503 and attempts < max_attempts:
                    print(
                        f"[WARN] channels.list temporary 503, retrying "
                        f"({attempts}/{max_attempts})..."
                    )
                    time.sleep(2 * attempts)
                    continue

                print(f"[WARN] Failed to fetch channel details: {exc}")
                break

            except Exception as exc:
                attempts += 1
                if attempts < max_attempts:
                    print(
                        f"[WARN] Channel details fetch failed, retrying "
                        f"({attempts}/{max_attempts}): {exc}"
                    )
                    time.sleep(2 * attempts)
                    continue

                print(f"[WARN] Failed to fetch channel details: {exc}")
                break

    return details


def build_record(item, region, category_map, channel_map, collected_at, page_number, rank, batch_id):
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    content = item.get("contentDetails", {})

    category_id = snippet.get("categoryId", "")
    category_name = category_map.get(category_id) or FALLBACK_CATEGORY_MAP.get(category_id) or "Other"
    channel_id = snippet.get("channelId")

    channel_info = channel_map.get(channel_id, {})

    views = safe_int(stats.get("viewCount"))
    likes = safe_int(stats.get("likeCount"))
    comments = safe_int(stats.get("commentCount"))
    favorites = safe_int(stats.get("favoriteCount"))

    thumbnails = snippet.get("thumbnails", {})
    thumb = (
        thumbnails.get("high", {})
        or thumbnails.get("medium", {})
        or thumbnails.get("default", {})
    )

    return {
        # Trending context
        "collection_batch_id": batch_id,
        "collected_at": collected_at,
        "surface": "mostPopular",
        "trending_region": region,
        "trending_category_id": category_id,
        "trending_page": page_number,
        "trending_rank": rank,

        # Video identity
        "video_id": item.get("id"),
        "title": snippet.get("title", ""),
        "description": (snippet.get("description") or "")[:DESCRIPTION_LIMIT],
        "published_at": snippet.get("publishedAt"),
        "category_id": category_id,
        "category_name": category_name,
        "tags": json.dumps(snippet.get("tags", [])),
        "default_language": snippet.get("defaultLanguage"),
        "thumbnail_url": thumb.get("url"),

        # Engagement metrics
        "view_count": views,
        "like_count": likes,
        "comment_count": comments,
        "favorite_count": favorites,

        # Content details
        "duration_iso": content.get("duration"),
        "definition": content.get("definition"),
        "caption": str(content.get("caption", "")).lower() == "true",
        "licensed_content": bool(content.get("licensedContent", False)),
        "content_rating": json.dumps(content.get("contentRating", {})),
        "projection": content.get("projection"),

        # Channel info
        "channel_id": channel_id,
        "channel_title": channel_info.get("channel_title", snippet.get("channelTitle", "Unknown")),
        "channel_subscriber_count": channel_info.get("channel_subscriber_count", 0),
        "channel_view_count": channel_info.get("channel_view_count", 0),
        "channel_video_count": channel_info.get("channel_video_count", 0),
        "channel_country": channel_info.get("channel_country"),
    }


def fetch_youtube_data():
    collected_at = datetime.now(timezone.utc).isoformat()
    batch_id = str(uuid.uuid4())
    records = []
    seen = set()

    for region in REGIONS:
        try:
            category_map = fetch_category_map(region)
        except Exception as exc:
            print(f"[WARN] Could not fetch category map for {region}: {exc}")
            category_map = FALLBACK_CATEGORY_MAP.copy()

        for category_id in CATEGORY_IDS:
            if (region, category_id) in unsupported_pairs:
                continue

            next_token = None

            for page_index in range(PAGES_PER_CATEGORY):
                try:
                    payload = fetch_popular_videos(
                        region=region,
                        category_id=category_id,
                        page_token=next_token,
                    )

                    items = payload.get("items", [])
                    if not items:
                        break

                    channel_ids = [
                        item.get("snippet", {}).get("channelId")
                        for item in items
                        if item.get("snippet", {}).get("channelId")
                    ]
                    channel_map = fetch_channel_details(channel_ids)

                    for item_index, item in enumerate(items, start=1):
                        rank = page_index * MAX_RESULTS + item_index

                        record = build_record(
                            item=item,
                            region=region,
                            category_map=category_map,
                            channel_map=channel_map,
                            collected_at=collected_at,
                            page_number=page_index + 1,
                            rank=rank,
                            batch_id=batch_id,
                        )

                        dedupe_key = (
                            record["collection_batch_id"],
                            record["trending_region"],
                            record["video_id"],
                        )
                        if dedupe_key not in seen:
                            seen.add(dedupe_key)
                            records.append(record)

                    next_token = payload.get("nextPageToken")
                    if not next_token:
                        break

                except requests.HTTPError as exc:
                    status_code = exc.response.status_code if exc.response is not None else None
                    if status_code in (403, 404):
                        unsupported_pairs.add((region, category_id))
                        print(f"[INFO] Skipping unsupported/forbidden category {category_id} for region {region}")
                    else:
                        print(f"[WARN] Failed for region={region}, category={category_id}: {exc}")
                    break

                except Exception as exc:
                    print(f"[WARN] Failed for region={region}, category={category_id}: {exc}")
                    break

    return records


if __name__ == "__main__":
    try:
        while True:
            batch = fetch_youtube_data()

            for record in batch:
                producer.send(TOPIC, value=record)

            producer.flush()
            print(f"Sent {len(batch)} enriched YouTube records to Kafka at {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\nProducer stopped by user.")
        producer.flush()
        producer.close()
