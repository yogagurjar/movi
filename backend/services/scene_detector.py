import logging
import re
import subprocess
from pathlib import Path

from backend.config import settings
from backend.models import SceneInfo

logger = logging.getLogger(__name__)


def _get_video_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    raw = result.stdout.strip()
    if "/" in raw:
        num, den = raw.split("/")
        return float(num) / float(den)
    return float(raw) if raw else 30.0


def _get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def _scene_to_seconds(scene_time) -> tuple[float, float]:
    start = scene_time[0].get_seconds()
    end = scene_time[1].get_seconds()
    return start, end


def detect_scenes(video_path: Path, scenes_dir: Path) -> list[SceneInfo]:
    scenes_dir.mkdir(parents=True, exist_ok=True)
    fps = _get_video_fps(video_path)
    duration = _get_video_duration(video_path)
    logger.info("Video: %.1f sec @ %.2f fps", duration, fps)

    logger.info("Detecting scenes with FFmpeg (GPU hwaccel)...")
    cmd = [
        "ffmpeg", "-hwaccel", "cuda",
        "-i", str(video_path),
        "-vf", "select='gt(scene,0.3)',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("FFmpeg scene detection failed, falling back to ContentDetector")
        try:
            from scenedetect import detect, ContentDetector
            scene_list = detect(str(video_path), ContentDetector(threshold=settings.SCENE_DETECTION_THRESHOLD))
        except Exception:
            scene_list = []
        if not scene_list:
            scene_list = [([0.0], [duration])]
        scenes = []
        for idx, scene in enumerate(scene_list):
            start, end = _scene_to_seconds(scene)
            scenes.append(SceneInfo(scene_index=idx, start_time=round(start, 3), end_time=round(end, 3), duration=round(end - start, 3), keyframe_path=None))
        logger.info("Detected %d scenes via ContentDetector fallback", len(scenes))
        return scenes

    scene_times: list[float] = [0.0]
    for line in result.stderr.split("\n"):
        if "pts_time:" in line:
            match = re.search(r"pts_time:(\d+\.\d+)", line)
            if match:
                t = float(match.group(1))
                if t > 0.0 and t < duration:
                    scene_times.append(t)
    scene_times.append(duration)

    scenes: list[SceneInfo] = []
    for i in range(len(scene_times) - 1):
        start, end = scene_times[i], scene_times[i + 1]
        if end - start >= 0.5:
            scenes.append(SceneInfo(scene_index=len(scenes), start_time=round(start, 3), end_time=round(end, 3), duration=round(end - start, 3), keyframe_path=None))

    logger.info("Detected %d scenes via FFmpeg GPU", len(scenes))
    return scenes


def extract_keyframes(video_path: Path, scenes: list[SceneInfo], keyframes_dir: Path) -> list[SceneInfo]:
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %d keyframes via FFmpeg GPU...", len(scenes))

    for scene in scenes:
        mid_time = (scene.start_time + scene.end_time) / 2.0
        kf_path = keyframes_dir / f"scene_{scene.scene_index:05d}.jpg"

        cmd = [
            "ffmpeg", "-hwaccel", "cuda",
            "-ss", str(mid_time),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            str(kf_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if kf_path.exists() and kf_path.stat().st_size > 0:
            scene.keyframe_path = str(kf_path)
        else:
            retry_time = scene.start_time + 0.1
            retry_cmd = [
                "ffmpeg", "-hwaccel", "cuda",
                "-ss", str(retry_time),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                "-y",
                str(kf_path),
            ]
            subprocess.run(retry_cmd, capture_output=True, text=True)
            if kf_path.exists() and kf_path.stat().st_size > 0:
                scene.keyframe_path = str(kf_path)

    extracted = sum(1 for s in scenes if s.keyframe_path is not None)
    logger.info("Extracted %d / %d keyframes", extracted, len(scenes))
    return scenes
