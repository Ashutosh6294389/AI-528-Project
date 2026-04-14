#!/bin/bash

# Start Zookeeper
zookeeper-server-start.sh config/zookeeper.properties &

sleep 5

# Start Kafka
kafka-server-start.sh config/server.properties &

sleep 5

# Create Topic
kafka-topics.sh --create --topic youtube-data --bootstrap-server localhost:9092