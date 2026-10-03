from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATASET_URL, RAW_DATA_DIR, RAW_DATA_PATH
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def _extract_nested_bank_csv(archive_path: Path, output_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        with zipfile.ZipFile(archive_path, "r") as outer_zip:
            outer_zip.extractall(tmp_dir)

        nested_bank_zip = next(tmp_dir.rglob("bank.zip"), None)

        if nested_bank_zip is not None:
            with zipfile.ZipFile(nested_bank_zip, "r") as inner_zip:
                inner_zip.extractall(tmp_dir / "bank")

        bank_csv = next(tmp_dir.rglob("bank.csv"), None)

        if bank_csv is None:
            raise FileNotFoundError("bank.csv was not found inside the UCI archive")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bank_csv, output_path)


def download_dataset(force: bool = False) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and not RAW_DATA_PATH.is_file():
        raise IsADirectoryError(f"Dataset path is not a file: '{RAW_DATA_PATH}'")

    if RAW_DATA_PATH.is_file() and not force:
        logger.info("Dataset already exists: %s", RAW_DATA_PATH)
        return RAW_DATA_PATH

    archive_path = RAW_DATA_DIR / "bank_marketing.zip"
    logger.info("Downloading dataset from %s", DATASET_URL)
    try:
        urllib.request.urlretrieve(DATASET_URL, archive_path)
        logger.info("Extracting %s", RAW_DATA_PATH.name)
        _extract_nested_bank_csv(archive_path, RAW_DATA_PATH)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Failed to download or extract dataset from {DATASET_URL}") from exc
    finally:
        archive_path.unlink(missing_ok=True)

    logger.info("Dataset saved to %s", RAW_DATA_PATH)

    return RAW_DATA_PATH


def ensure_dataset() -> Path:
    if not RAW_DATA_PATH.is_file():
        return download_dataset(force=False)
    return RAW_DATA_PATH


if __name__ == "__main__":
    configure_logging()
    download_dataset(force=True)
