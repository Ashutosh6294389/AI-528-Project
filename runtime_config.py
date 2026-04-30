"""Shared runtime configuration for local and managed-cloud deployments.

This module keeps data-plane connection details out of the pipeline code so
we can switch between:

- local Kafka + local MongoDB
- Confluent Cloud Kafka + MongoDB Atlas
- local / remote MongoDB replica sets
- local Spark or cluster-managed Spark

The project should be runnable in all of those modes without source edits.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# YouTube ingestion
# ---------------------------------------------------------------------------
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
REGIONS = [part.strip() for part in os.environ.get("YOUTUBE_REGIONS", "IN,US,GB,CA,AU").split(",") if part.strip()]
CATEGORY_IDS = [
    part.strip()
    for part in os.environ.get(
        "YOUTUBE_CATEGORY_IDS",
        "1,2,10,15,17,20,22,23,24,25,26,28",
    ).split(",")
    if part.strip()
]
MAX_RESULTS = int(os.environ.get("YOUTUBE_MAX_RESULTS", "50"))
PAGES_PER_CATEGORY = int(os.environ.get("YOUTUBE_PAGES_PER_CATEGORY", "1"))
POLL_SECONDS = int(os.environ.get("YOUTUBE_POLL_SECONDS", "45"))
REQUEST_TIMEOUT = int(os.environ.get("YOUTUBE_REQUEST_TIMEOUT", "20"))
CHANNEL_BATCH_SIZE = int(os.environ.get("YOUTUBE_CHANNEL_BATCH_SIZE", "50"))
DESCRIPTION_LIMIT = int(os.environ.get("YOUTUBE_DESCRIPTION_LIMIT", "500"))


# ---------------------------------------------------------------------------
# Kafka
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS",
    os.environ.get("KAFKA_BROKER", "localhost:9092"),
)
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "youtube-data")
KAFKA_SECURITY_PROTOCOL = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SASL_MECHANISM = os.environ.get("KAFKA_SASL_MECHANISM", "PLAIN")
KAFKA_SASL_USERNAME = os.environ.get("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.environ.get("KAFKA_SASL_PASSWORD", "")
KAFKA_SASL_JAAS_CONFIG = os.environ.get("KAFKA_SASL_JAAS_CONFIG", "")
KAFKA_STARTING_OFFSETS = os.environ.get("KAFKA_STARTING_OFFSETS", "latest")

# Kafka Spark connector coordinates for Spark 3.4.x / Scala 2.12.
KAFKA_SPARK_PACKAGES = ",".join(
    [
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1",
        "org.apache.spark:spark-token-provider-kafka-0-10_2.12:3.4.1",
    ]
)


def _derived_sasl_jaas_config() -> str:
    if KAFKA_SASL_JAAS_CONFIG.strip():
        return KAFKA_SASL_JAAS_CONFIG.strip()
    if not KAFKA_SASL_USERNAME or not KAFKA_SASL_PASSWORD:
        return ""
    return (
        "org.apache.kafka.common.security.plain.PlainLoginModule required "
        f'username="{KAFKA_SASL_USERNAME}" password="{KAFKA_SASL_PASSWORD}";'
    )


def build_kafka_python_producer_kwargs() -> dict:
    """Return kafka-python producer kwargs for local or Confluent Cloud."""
    kwargs = {
        "bootstrap_servers": KAFKA_BOOTSTRAP_SERVERS,
    }
    if KAFKA_SECURITY_PROTOCOL.upper() != "PLAINTEXT":
        kwargs.update(
            {
                "security_protocol": KAFKA_SECURITY_PROTOCOL,
                "sasl_mechanism": KAFKA_SASL_MECHANISM,
                "sasl_plain_username": KAFKA_SASL_USERNAME,
                "sasl_plain_password": KAFKA_SASL_PASSWORD,
            }
        )
    return kwargs


def build_kafka_spark_options(topic: str | None = None) -> dict[str, str]:
    """Return Spark readStream Kafka options for local or managed brokers."""
    options = {
        "kafka.bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "subscribe": topic or KAFKA_TOPIC,
        "startingOffsets": KAFKA_STARTING_OFFSETS,
    }

    if KAFKA_SECURITY_PROTOCOL.upper() != "PLAINTEXT":
        options["kafka.security.protocol"] = KAFKA_SECURITY_PROTOCOL
        options["kafka.sasl.mechanism"] = KAFKA_SASL_MECHANISM
        jaas_config = _derived_sasl_jaas_config()
        if jaas_config:
            options["kafka.sasl.jaas.config"] = jaas_config

    return options


def use_local_kafka_jars() -> bool:
    """Use checked-in Kafka jars for local runs unless explicitly disabled."""
    default_local = KAFKA_SECURITY_PROTOCOL.upper() == "PLAINTEXT"
    return _env_flag("KAFKA_SPARK_USE_LOCAL_JARS", default=default_local)


def local_kafka_jars_csv(project_root: str | Path | None = None) -> str:
    root = Path(project_root) if project_root else PROJECT_ROOT
    jars_dir = root / "jars"
    jar_names = [
        "spark-sql-kafka-0-10_2.12-3.4.1.jar",
        "spark-token-provider-kafka-0-10_2.12-3.4.1.jar",
        "kafka-clients-3.4.1.jar",
        "commons-pool2-2.11.1.jar",
    ]
    jar_paths = [str(jars_dir / name) for name in jar_names if (jars_dir / name).exists()]
    return ",".join(jar_paths)


def local_delta_jars_csv(project_root: str | Path | None = None) -> str:
    root = Path(project_root) if project_root else PROJECT_ROOT
    jars_dir = root / "jars"
    jar_names = [
        "delta-core_2.12-2.4.0.jar",
        "delta-storage-2.4.0.jar",
    ]
    jar_paths = [str(jars_dir / name) for name in jar_names if (jars_dir / name).exists()]
    return ",".join(jar_paths)


# ---------------------------------------------------------------------------
# Spark runtime
# ---------------------------------------------------------------------------
SPARK_RUNTIME = os.environ.get("SPARK_RUNTIME", "local").strip().lower()
SPARK_MASTER = os.environ.get("SPARK_MASTER", "").strip()


def resolve_spark_master(default_local_master: str | None = None) -> str | None:
    """Resolve the master to use.

    - explicit `SPARK_MASTER` wins
    - `SPARK_RUNTIME=databricks` means "do not force a master"
    - otherwise fall back to the caller-provided local default
    """
    if SPARK_MASTER:
        return SPARK_MASTER
    if SPARK_RUNTIME == "databricks":
        return None
    return default_local_master
