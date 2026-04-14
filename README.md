0. CLEAN START (optional but recommended)
pkill -f kafka
pkill -f java
⚙️ 1. ACTIVATE ENVIRONMENT
cd ~/Downloads/AI528
source venv/bin/activate
☕ 2. SET JAVA (VERY IMPORTANT)
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
export PATH=$JAVA_HOME/bin:$PATH

Verify:

java -version
🟡 3. START KAFKA (Terminal 1)
/opt/homebrew/opt/kafka/bin/kafka-server-start /opt/homebrew/etc/kafka/server.properties

👉 Keep this running

🟡 4. CREATE TOPIC (Terminal 2 — run once)
/opt/homebrew/opt/kafka/bin/kafka-topics \
--create \
--topic youtube-data \
--bootstrap-server localhost:9092 \
--partitions 1 \
--replication-factor 1

👉 If "already exists" → ignore

🟡 5. RUN PRODUCER (Terminal 2)
python3 data_ingestion/youtube_producer.py



rm -rf ~/.ivy2
rm -rf ~/.cache
rm -rf ~/spark-warehouse

rm -rf storage/checkpoints_csv 

rm -rf storage/output   

👉 This starts sending data to Kafka

🔵 6. RUN SPARK STREAMING (Terminal 3)
python3 spark_processing/spark_streaming.py
🎯 DONE — PIPELINE RUNNING

streamlit run dashboard/app.py

Flow:

Producer → Kafka → Spark Streaming → Output
⚠️ TROUBLESHOOT QUICK FIXES
❌ Kafka not starting
pkill -f kafka
pkill -f java
❌ Port busy (9092)
lsof -i :9092
kill -9 <PID>
❌ Spark Java error
export JAVA_HOME=$(/usr/libexec/java_home -v 17)
❌ Producer error (NoBrokersAvailable)

👉 Kafka is not running → start Step 3

🔥 OPTIONAL (RESET EVERYTHING)

If things break:

rm -rf ~/.ivy2 ~/.cache ~/spark-warehouse
🚀 ONE-LINE QUICK START (ADVANCED)

If everything is already configured:

source venv/bin/activate && export JAVA_HOME=$(/usr/libexec/java_home -v 17)

Then open 3 terminals and run steps 3–6.

🎉 YOU ARE NOW FULLY SET

Your system is now:
✔ Stable
✔ Repeatable
✔ Production-like