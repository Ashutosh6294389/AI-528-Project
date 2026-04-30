# AI528 YouTube Analytics Pipeline

This project ingests YouTube trending data, streams it through Kafka,
stores Bronze in MongoDB, stores Silver/Gold as local Delta tables, and
serves a Streamlit dashboard on top.

Full technical documentation is available in [docs/project_documentation.md](/Users/ashutoshsingh/Desktop/AI-528_GOATS/AI528/docs/project_documentation.md:1).

## Quick start (local)

```bash
cd /Users/ashutoshsingh/Desktop/AI-528_GOATS/AI528
source venv/bin/activate
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH=$JAVA_HOME/bin:$PATH
```

Optional: copy values from `.env.example` into your shell or `.env`.

### 1. Start MongoDB

```bash
brew services start mongodb-community
```

### 2. Start Kafka

```bash
/opt/homebrew/opt/kafka/bin/kafka-server-start /opt/homebrew/etc/kafka/server.properties
```

Create the topic once:

```bash
/opt/homebrew/opt/kafka/bin/kafka-topics \
  --create \
  --topic youtube-data \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1
```

### 3. Run the apps

```bash
python3 data_ingestion/youtube_producer.py
python3 spark_processing/spark_streaming.py
streamlit run dashboard/app.py
```

## Storage model

The medallion layers use a hybrid layout:

- Bronze: `bronze_raw`
- Silver: `storage/delta_tables/silver/youtube_enriched`
- Gold:
  - `storage/delta_tables/gold/latest_snapshot`
  - `storage/delta_tables/gold/category_summary`
  - `storage/delta_tables/gold/views_timeseries`
  - `storage/delta_tables/gold/region_timeseries`
  - `storage/delta_tables/gold/channel_leaderboard`
  - `storage/delta_tables/gold/duration_distribution`
  - `storage/delta_tables/gold/subscriber_tier_distribution`
  - `storage/delta_tables/gold/tag_usage_frequency`
  - `storage/delta_tables/gold/trending_rank_distribution`

Spark checkpoints remain on disk in:

- `storage/checkpoints_medallion`

## Managed cloud support

The project is now configurable for:

- MongoDB Atlas
- self-managed MongoDB replica sets
- Confluent Cloud Kafka
- local Spark or managed Spark / Databricks-style runtimes

Use environment variables from [.env.example](/Users/ashutoshsingh/Desktop/AI528/.env.example:1), then follow:

- [Managed cloud setup](/Users/ashutoshsingh/Desktop/AI528/docs/managed_cloud_setup.md:1)

## One-shot backfill

If you still have the old local Delta dump:

```bash
python3 spark_processing/backfill_medallion.py
```

If `storage/delta_tables/silver/youtube_enriched` exists, the script
migrates it into Mongo Bronze + Silver and rebuilds Gold.
