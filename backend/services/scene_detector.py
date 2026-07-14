import logging
import subprocess
from pathlib import Path

import cv2

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

    logger.info("Detecting scenes with PySceneDetect...")
    try:
        from scenedetect import detect
        from scenedetect.detectors import AdaptiveDetector

        scene_list = detect(
            str(video_path),
            AdaptiveDetector(adaptive_threshold=settings.SCENE_DETECTION_THRESHOLD),
        )
    except Exception as e:
        logger.warning("AdaptiveDetector failed (%s), falling back to ContentDetector", e)
        from scenedetect import detect, ContentDetector
        scene_list = detect(
            str(video_path),
            ContentDetector(threshold=settings.SCENE_DETECTION_THRESHOLD),
        )

    if not scene_list:
        scene_list = [([0.0], [duration])]

    scenes: list[SceneInfo] = []
    for idx, scene in enumerate(scene_list):
        start, end = _scene_to_seconds(scene)
        scenes.append(
            SceneInfo(
                scene_index=idx,
                start_time=round(start, 3),
                end_time=round(end, 3),
                duration=round(end - start, 3),
                keyframe_path=None,
            )
        )

    logger.info("Detected %d scenes", len(scenes))
    return scenes


def extract_keyframes(video_path: Path, scenes: list[SceneInfo], keyframes_dir: Path) -> list[SceneInfo]:
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %d keyframes...", len(scenes))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0

    for scene in scenes:
        mid_time = (scene.start_time + scene.end_time) / 2.0
        frame_idx = int(mid_time * original_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            mid_time = scene.start_time + 0.1
            frame_idx = int(mid_time * original_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

        if ret:
            kf_path = keyframes_dir / f"scene_{scene.scene_index:05d}.jpg"
            cv2.imwrite(str(kf_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            scene.keyframe_path = str(kf_path)

    cap.release()

    extracted = sum(1 for s in scenes if s.keyframe_path is not None)
    logger.info("Extracted %d / %d keyframes", extracted, len(scenes))
    return scenes
