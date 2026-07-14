import logging
import re
import shutil
from pathlib import Path
from typing import Optional

import gdown

from backend.config import settings

logger = logging.getLogger(__name__)


def _extract_file_id(url: str) -> Optional[str]:
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_from_gdrive(url: str, job_dir: Path, label: str = "file") -> Path:
    file_id = _extract_file_id(url)
    if not file_id:
        raise ValueError(f"Could not extract file ID from URL: {url}")

    dest = job_dir / f"{label}"
    logger.info("Downloading %s (ID: %s) to %s", label, file_id, dest)

    downloaded_path = gdown.download(
        url,
        output=str(dest),
        quiet=False,
        fuzzy=True,
        resume=True,
    )

    if downloaded_path is None:
        raise RuntimeError(f"Download failed for {label} (ID: {file_id})")

    path = Path(downloaded_path)
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Downloaded %s: %.1f MB", label, size_mb)

    return path


def cleanup_job_files(job_dir: Path):
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.info("Cleaned up job directory: %s", job_dir)
