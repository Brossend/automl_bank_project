from __future__ import annotations

import csv
import time
from datetime import datetime

import psutil

from src.config import REPORTS_DIR, TABLES_DIR


def collect_resource_metrics(seconds: int = 10, interval: float = 1.0) -> list[dict[str, float | str]]:
    metrics = []
    for _ in range(seconds):
        metrics.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
            }
        )
        time.sleep(interval)
    return metrics


def save_metrics(metrics: list[dict[str, float | str]]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / "resource_monitoring.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "cpu_percent", "memory_percent"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Resource monitoring saved to {output_path.relative_to(REPORTS_DIR.parent)}")


def main():
    save_metrics(collect_resource_metrics())


if __name__ == "__main__":
    main()
