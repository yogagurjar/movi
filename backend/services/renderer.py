import logging
import shutil
import subprocess
import uuid
from pathlib import Path

from backend.config import settings
from backend.models import MatchResult, TimelineSegment

logger = logging.getLogger(__name__)


def _clamp_speed(required: float) -> float:
    low = 1.0 - settings.SPEED_MAX_DELTA
    high = 1.0 + settings.SPEED_MAX_DELTA
    return max(low, min(high, required))


def build_timeline(match_results: list[MatchResult]) -> list[TimelineSegment]:
    timeline: list[TimelineSegment] = []
    for mr in match_results:
        if mr.timeline is None:
            continue
        seg = mr.timeline
        if seg.original_duration <= 0:
            continue

        required = seg.original_duration / seg.target_duration if seg.target_duration > 0 else 1.0
        speed = _clamp_speed(required)

        adjusted = TimelineSegment(
            scene_index=seg.scene_index,
            source_path=seg.source_path,
            trim_start=seg.trim_start,
            trim_end=seg.trim_end,
            original_duration=seg.original_duration,
            target_duration=seg.original_duration / speed,
            speed_factor=round(speed, 4),
            voice_segment_index=seg.voice_segment_index,
            voice_text=seg.voice_text,
        )
        timeline.append(adjusted)

    logger.info("Timeline built: %d segments", len(timeline))
    return timeline


def render_video(
    movie_path: Path,
    voiceover_path: Path,
    timeline: list[TimelineSegment],
    output_path: Path,
) -> Path:
    if not timeline:
        raise ValueError("Timeline is empty, nothing to render")

    output_path = output_path.with_suffix(".mp4")
    logger.info("Rendering %d segments to %s", len(timeline), output_path.name)

    temp_dir = output_path.parent / f"_render_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    segment_files: list[Path] = []
    try:
        for i, seg in enumerate(timeline):
            seg_path = temp_dir / f"seg_{i:05d}.ts"
            _render_segment(movie_path, seg, seg_path)
            segment_files.append(seg_path)

        concat_path = temp_dir / "concat.txt"
        with open(concat_path, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf.resolve()}'\n")

        codec = settings.OUTPUT_CODEC if settings.GPU_ENABLED else "libx264"
        pixel_fmt = "p010le" if settings.GPU_ENABLED else "yuv420p"

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_path),
            "-i", str(voiceover_path),
            "-c:v", codec,
            "-pix_fmt", pixel_fmt,
            "-r", str(settings.OUTPUT_FPS),
            "-s", f"{settings.OUTPUT_WIDTH}x{settings.OUTPUT_HEIGHT}",
        ]

        if settings.GPU_ENABLED:
            cmd.extend(["-preset", "p2", "-tune", "hq", "-rc", "vbr", "-cq", str(settings.OUTPUT_CRF)])
            cmd.extend(["-b:v", "20M"])
        else:
            cmd.extend(["-preset", "medium", "-crf", str(settings.OUTPUT_CRF)])

        cmd.extend([
            "-c:a", settings.AUDIO_CODEC,
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ])

        logger.info("FFmpeg command: %s", " ".join(str(c) for c in cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"Render failed:\n{result.stderr[:2000]}")

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("Render complete: %.1f MB", size_mb)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return output_path


def _render_segment(movie_path: Path, seg: TimelineSegment, output_path: Path):
    speed = seg.speed_factor
    trim_start = seg.trim_start
    trim_end = seg.trim_end

    setpts = f"PTS/{speed}" if speed != 1.0 else "PTS"

    codec = settings.OUTPUT_CODEC if settings.GPU_ENABLED else "libx264"
    pixel_fmt = "p010le" if settings.GPU_ENABLED else "yuv420p"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(movie_path),
        "-vf",
        f"trim=start={trim_start}:end={trim_end},setpts={setpts},"
        f"scale={settings.OUTPUT_WIDTH}:{settings.OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={settings.OUTPUT_WIDTH}:{settings.OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
        "-an",
        "-c:v", codec,
        "-pix_fmt", pixel_fmt,
        "-r", str(settings.OUTPUT_FPS),
    ]
    if settings.GPU_ENABLED:
        cmd.extend(["-preset", "p2", "-rc", "vbr", "-cq", str(settings.OUTPUT_CRF)])
    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Segment render failed for scene {seg.scene_index}:\n{result.stderr[:1000]}")
