import logging
import subprocess
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)


def extract_audio_from_video(video_path: Path, output_path: Path) -> Path:
    output_path = output_path.with_suffix(".wav")
    logger.info("Extracting audio from %s to %s", video_path.name, output_path.name)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{result.stderr}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Audio extracted: %.1f MB", size_mb)
    return output_path


def convert_audio_to_wav(audio_path: Path, output_path: Path) -> Path:
    output_path = output_path.with_suffix(".wav")
    logger.info("Converting %s to WAV", audio_path.name)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio conversion failed:\n{result.stderr}")

    return output_path
