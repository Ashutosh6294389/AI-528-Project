API_KEY = "AIzaSyBQVw7oXrIa2d81IuGuAze56iHYVPW6ExA"
KAFKA_BROKER = "localhost:9092"
TOPIC = "youtube-data"

REGIONS = ["IN", "US", "GB", "CA", "AU", "DE", "FR", "JP"]
CATEGORY_IDS = ["1", "2", "10", "15", "17", "20", "22", "23", "24", "25", "26", "28"]

MAX_RESULTS = 50
PAGES_PER_CATEGORY = 1
REQUEST_TIMEOUT = 20

TARGET_RUNTIME_HOURS = 7
DAILY_QUOTA_UNITS = 10000

# Start conservative; the producer below self-tunes sleep.
INITIAL_POLL_SECONDS = 20

# About 30 videos.list calls/cycle is a strong starting point for a 7h burn target.
TARGET_VIDEO_CALLS_PER_CYCLE = 30

