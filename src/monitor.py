from __future__ import annotations

import csv
import logging
import time
from datetime import UTC, datetime

import psutil

from src.config import REPORTS_DIR, TABLES_DIR
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def collect_resource_metrics(
    seconds: int = 10, interval: float = 1.0
) -> list[dict[str, float | str]]:
    """Collect CPU and memory samples at a validated interval."""
    if seconds < 1:
        raise ValueError("seconds must be at least 1")
    if interval <= 0:
        raise ValueError("interval must be greater than 0")

    metrics: list[dict[str, float | str]] = []
    for _ in range(seconds):
        metrics.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
            }
        )
        time.sleep(interval)
    return metrics


def save_metrics(metrics: list[dict[str, float | str]]) -> None:
    """Persist collected resource samples as a CSV report."""
    if not metrics:
        raise ValueError("At least one resource metric is required")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / "resource_monitoring.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "cpu_percent", "memory_percent"])
        writer.writeheader()
        writer.writerows(metrics)
    logger.info(
        "Resource monitoring saved to %s",
        output_path.relative_to(REPORTS_DIR.parent),
    )


def main() -> None:
    """Collect and persist the default resource-monitoring window."""
    save_metrics(collect_resource_metrics())


if __name__ == "__main__":
    configure_logging()
    main()
