import logging
import shutil
import subprocess
import uuid
from pathlib import Path

from backend.config import settings
from backend.models import MatchResult, TimelineSegment

logger = logging.getLogger(__name__)

_FFMPEG_HAS_CUDA: bool | None = None
_FFMPEG_HAS_NVENC: bool | None = None


def _probe_ffmpeg_cuda() -> bool:
    global _FFMPEG_HAS_CUDA
    if _FFMPEG_HAS_CUDA is None:
        try:
            result = subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True, timeout=15)
            _FFMPEG_HAS_CUDA = "cuda" in result.stdout.lower()
        except Exception:
            _FFMPEG_HAS_CUDA = False
    return _FFMPEG_HAS_CUDA


def _probe_nvenc() -> bool:
    global _FFMPEG_HAS_NVENC
    if _FFMPEG_HAS_NVENC is None:
        try:
            result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=15)
            _FFMPEG_HAS_NVENC = "h264_nvenc" in result.stdout
        except Exception:
            _FFMPEG_HAS_NVENC = False
    return _FFMPEG_HAS_NVENC


def _video_codec() -> str:
    if settings.GPU_ENABLED and _probe_nvenc():
        return "h264_nvenc"
    return "libx264"


def _pixel_fmt() -> str:
    return "yuv420p"


def _fwaccel_flags() -> list[str]:
    if settings.GPU_ENABLED and _probe_ffmpeg_cuda():
        return ["-hwaccel", "cuda"]
    return []


def build_timeline(match_results: list[MatchResult]) -> list[TimelineSegment]:
    timeline: list[TimelineSegment] = []
    for mr in match_results:
        if mr.timeline is None:
            continue
        seg = mr.timeline
        if seg.target_duration <= 0:
            continue

        source_paths = seg.source_paths if seg.source_paths else ([seg.source_path] if seg.source_path else [])

        if seg.target_duration > 8.0 and len(source_paths) >= 3:
            split_count = 3
            split_duration = seg.target_duration / split_count
            for i in range(split_count):
                img_path = source_paths[min(i, len(source_paths) - 1)]
                split_seg = TimelineSegment(
                    scene_index=seg.scene_index,
                    source_path=img_path,
                    source_paths=[img_path],
                    target_duration=split_duration,
                    voice_segment_index=seg.voice_segment_index,
                    voice_text=seg.voice_text,
                )
                timeline.append(split_seg)
        elif seg.target_duration > 5.0 and len(source_paths) >= 2:
            split_count = 2
            split_duration = seg.target_duration / split_count
            for i in range(split_count):
                img_path = source_paths[min(i, len(source_paths) - 1)]
                split_seg = TimelineSegment(
                    scene_index=seg.scene_index,
                    source_path=img_path,
                    source_paths=[img_path],
                    target_duration=split_duration,
                    voice_segment_index=seg.voice_segment_index,
                    voice_text=seg.voice_text,
                )
                timeline.append(split_seg)
        else:
            timeline.append(seg)

    logger.info("Timeline built: %d image segments", len(timeline))
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
    logger.info("Rendering %d image segments to %s", len(timeline), output_path.name)

    temp_dir = output_path.parent / f"_render_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    segment_files: list[Path] = []
    seg_total = len(timeline)
    log_interval = max(1, seg_total // 10)
    try:
        for i, seg in enumerate(timeline):
            if i > 0 and i % log_interval == 0:
                logger.info("Rendering segments: %d%% (%d/%d)", int(i * 100 / seg_total), i, seg_total)
            seg_path = temp_dir / f"seg_{i:05d}.ts"
            _render_segment(seg, seg_path)
            segment_files.append(seg_path)

        concat_path = temp_dir / "concat.txt"
        with open(concat_path, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf.resolve()}'\n")

        codec = _video_codec()
        pixel_fmt = _pixel_fmt()

        cmd = [
            "ffmpeg",
            "-y",
        ]
        cmd.extend(_fwaccel_flags())
        cmd.extend(["-f", "concat", "-safe", "0", "-i", str(concat_path), "-i", str(voiceover_path),
            "-c:v", codec,
            "-pix_fmt", pixel_fmt,
            "-r", str(settings.OUTPUT_FPS),
            "-s", f"{settings.OUTPUT_WIDTH}x{settings.OUTPUT_HEIGHT}",
        ])

        if codec == "h264_nvenc":
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


def _render_segment(seg: TimelineSegment, output_path: Path):
    image_path = seg.source_path
    duration = seg.target_duration

    if not image_path or not Path(image_path).exists():
        raise FileNotFoundError(f"Keyframe not found: {image_path}")

    codec = _video_codec()
    pixel_fmt = _pixel_fmt()

    cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-vf",
        f"scale={settings.OUTPUT_WIDTH}:{settings.OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={settings.OUTPUT_WIDTH}:{settings.OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
        "-an",
        "-c:v", codec,
        "-pix_fmt", pixel_fmt,
        "-r", str(settings.OUTPUT_FPS),
        "-t", str(duration),
    ]
    if codec == "h264_nvenc":
        cmd.extend(["-preset", "p2", "-tune", "hq", "-rc", "vbr", "-cq", str(settings.OUTPUT_CRF), "-b:v", "20M"])
    cmd.append(str(output_path))

    logger.debug("ffmpeg cmd: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Segment render failed for scene {seg.scene_index}:\n{' '.join(cmd)}\n{result.stderr[:3000]}")
