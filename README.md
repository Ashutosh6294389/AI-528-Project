1. Activate the environment

```bash
cd ~/Downloads/AI528
source venv/bin/activate
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH=$JAVA_HOME/bin:$PATH
```

2. Start Kafka

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

Current medallion layout

- Bronze: `storage/delta_tables/bronze/youtube_raw`
- Silver: `storage/delta_tables/silver/youtube_enriched`
- Gold: `storage/delta_tables/gold/*`
- Streaming checkpoint: `storage/checkpoints_medallion`

Persistence behavior

- Bronze is append-only and keeps raw historical records.
- Silver appends cleansed records and preserves ingestion metadata.
- Gold tables are rebuilt from Silver so dashboards always reflect the latest full history.
- The pipeline no longer creates a new run folder on each start.
- Do not delete `storage/checkpoints_medallion` unless you intentionally want Spark to replay Kafka data.




/opt/homebrew/opt/kafka/bin/kafka-server-start /opt/homebrew/etc/kafka/server.properties
