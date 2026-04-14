from kafka import KafkaProducer
import requests
import json
import time
from config import API_KEY, KAFKA_BROKER, TOPIC

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER, 
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
) 

# ✅ Category Mapping (FIX)
CATEGORY_MAP = {
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
    "28": "Science & Tech"
}

def fetch_youtube_data():
    try:
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&chart=mostPopular&regionCode=IN&maxResults=10&key={API_KEY}"
        response = requests.get(url, timeout=5)
        return response.json()
    except Exception as e:
        print("Error fetching data:", e)
        return {"items": []}

while True:
    data = fetch_youtube_data()

    for item in data.get('items', []):
        category_id = item['snippet'].get('categoryId')

        record = {
            "title": item['snippet']['title'],
            "category": CATEGORY_MAP.get(category_id, "Other"),  # ✅ FIX
            "views": item['statistics'].get('viewCount', 0),
            "likes": item['statistics'].get('likeCount', 0),
            "comments": item['statistics'].get('commentCount', 0)
        }

        producer.send(TOPIC, value=record)

    print("Sent data to Kafka")
    time.sleep(10)