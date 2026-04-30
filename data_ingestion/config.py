"""Backwards-compatible ingestion config shim.

Historically the producer imported constants from this file directly. We keep
that interface stable, but the real values now come from environment-driven
`runtime_config` so the same producer can target local Kafka or Confluent
Cloud without source edits.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_config import (
    CATEGORY_IDS,
    CHANNEL_BATCH_SIZE,
    DESCRIPTION_LIMIT,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    MAX_RESULTS,
    PAGES_PER_CATEGORY,
    POLL_SECONDS,
    REGIONS,
    REQUEST_TIMEOUT,
    YOUTUBE_API_KEY,
    build_kafka_python_producer_kwargs,
)

API_KEY = "AIzaSyC95ktcV3NNyHUQi_1j_4-Pp7xTELnPU2c"
KAFKA_BROKER = KAFKA_BOOTSTRAP_SERVERS
TOPIC = KAFKA_TOPIC

__all__ = [
    "API_KEY",
    "KAFKA_BROKER",
    "TOPIC",
    "REGIONS",
    "CATEGORY_IDS",
    "MAX_RESULTS",
    "PAGES_PER_CATEGORY",
    "POLL_SECONDS",
    "REQUEST_TIMEOUT",
    "CHANNEL_BATCH_SIZE",
    "DESCRIPTION_LIMIT",
    "build_kafka_python_producer_kwargs",
]
