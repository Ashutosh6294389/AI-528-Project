1. Activate the environment

```bash
cd ~/Downloads/AI528
source venv/bin/activate
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH=$JAVA_HOME/bin:$PATH
```

2. Start MongoDB (data store)

```bash
brew services start mongodb-community         # or: mongod --config /opt/homebrew/etc/mongod.conf
```

Optional connection overrides (defaults shown):

```bash
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB="youtube_analytics"
export BRONZE_TTL_DAYS=30                  # auto-expire Bronze docs
```

3. Start Kafka

```bash
/opt/homebrew/opt/kafka/bin/kafka-server-start /opt/homebrew/etc/kafka/server.properties
```

3. Create the topic once

```bash
/opt/homebrew/opt/kafka/bin/kafka-topics \
  --create \
  --topic youtube-data \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1
```

4. Run the producer

```bash
python3 data_ingestion/youtube_producer.py
```

5. Run the Spark medallion pipeline

```bash
python3 spark_processing/spark_streaming.py
```

6. Open the dashboard

```bash
streamlit run dashboard/app.py
```

Current medallion layout (MongoDB-backed)

All three layers now live in MongoDB. Streaming checkpoints stay on disk
(Spark requirement).

Collections:

- Bronze: `bronze_raw` (TTL on `ingestion_timestamp`, default 30 days)
- Silver: `silver_enriched`
- Gold: 9 collections — `gold_latest_snapshot`, `gold_category_summary`,
  `gold_views_timeseries`, `gold_region_timeseries`,
  `gold_channel_leaderboard`, `gold_duration_distribution`,
  `gold_subscriber_tier_distribution`, `gold_tag_usage_frequency`,
  `gold_trending_rank_distribution`.
- Streaming checkpoint: `storage/checkpoints_medallion` (on disk).

NoSQL features used:

- Compound indexes on every collection (created on first pipeline start
  by `analytics.mongo_io.setup_indexes`).
- TTL index on Bronze: docs older than `BRONZE_TTL_DAYS` auto-expire.
- Native BSON arrays — `tags_array` stored as an array, no flattening.
- Atomic Gold rebuilds via the MongoDB Spark connector's overwrite
  mode (writes to a staging collection then renames).

Persistence behavior

- Bronze is append-only inside its TTL window.
- Silver appends cleansed records and preserves ingestion metadata.
- Gold collections are rebuilt from Silver so dashboards always reflect
  the latest full history.
- Do not delete `storage/checkpoints_medallion` unless you intentionally
  want Spark to replay Kafka data.

Migrating from the previous Delta layout (one-shot)

```bash
python spark_processing/backfill_medallion.py
```

If `storage/delta_tables/youtube_enriched` exists, the script ingests
it into Bronze + Silver and rebuilds all 9 Gold collections. With no
Delta source present it just refreshes Gold from whatever is in Silver.




/opt/homebrew/opt/kafka/bin/kafka-server-start /opt/homebrew/etc/kafka/server.properties
