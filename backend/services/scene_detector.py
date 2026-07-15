import logging
import re
import subprocess
from pathlib import Path

from backend.config import settings
from backend.models import SceneInfo

logger = logging.getLogger(__name__)

_FFMPEG_HAS_CUDA: bool | None = None


def _ffmpeg_has_cuda() -> bool:
    global _FFMPEG_HAS_CUDA
    if _FFMPEG_HAS_CUDA is None:
        try:
            result = subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True, timeout=15)
            _FFMPEG_HAS_CUDA = "cuda" in result.stdout.lower()
        except Exception:
            _FFMPEG_HAS_CUDA = False
    return _FFMPEG_HAS_CUDA


def _hwaccel_flags() -> list[str]:
    if settings.GPU_ENABLED and _ffmpeg_has_cuda():
        return ["-hwaccel", "cuda"]
    return []


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


def _pytorch_scenes(video_path: Path, threshold: float = 0.3, min_scene_len: float = 0.5) -> list[SceneInfo]:
    """
    GPU-accelerated scene detection matching FFmpeg's histogram-based approach.
    Uses per-channel histogram SAD on GPU (same metric as FFmpeg's scene filter).
    Much faster than FFmpeg CPU on Kaggle (uses T4/P100 GPU).
    """
    import torch
    import torchvision.io as io
    from torchvision.transforms import functional as F

    device = torch.device(settings.TORCH_DEVICE)
    if device.type != "cuda":
        return []

    duration = _get_video_duration(video_path)
    if duration <= 0:
        return []

    logger.info("Detecting scenes via PyTorch CUDA histogram on %s...", device)

    try:
        reader = io.VideoReader(str(video_path), "video")
        meta = reader.get_metadata()
        vs = meta.get("video", {})
        fps_val = float(vs.get("fps", [30.0])[0]) if vs.get("fps") else 30.0
        min_frames = max(1, int(min_scene_len * fps_val))
        sample_rate = max(1, int(fps_val / 12))

        torch.cuda.empty_cache()

        prev_hist = None
        times = [0.0]
        frame_idx = 0
        last_scene_frame = 0

        for frame_data in reader:
            img = frame_data['data']
            pts = float(frame_data['pts'])

            if frame_idx % sample_rate == 0:
                curr_gpu = img.to(device, non_blocking=True).float()
                _, h, w = curr_gpu.shape

                if max(h, w) > 240:
                    scale = 240.0 / max(h, w)
                    curr_small = F.resize(curr_gpu, [int(h * scale), int(w * scale)], antialias=True)
                else:
                    curr_small = curr_gpu

                n_ch = curr_small.shape[0]
                hists = []
                for c in range(n_ch):
                    flat = curr_small[c].reshape(-1)
                    hist_c = torch.histc(flat, bins=256, min=0, max=255)
                    hists.append(hist_c / hist_c.sum())
                curr_hist = torch.stack(hists)

                if prev_hist is not None:
                    sad = torch.abs(curr_hist - prev_hist).sum().item()
                    scene_score = sad / (n_ch * 2)

                    if scene_score > threshold and (frame_idx - last_scene_frame) >= min_frames:
                        if 0.0 < pts < duration:
                            times.append(pts)
                            last_scene_frame = frame_idx

                prev_hist = curr_hist

            frame_idx += 1

        times.append(duration)

        scenes = []
        for i in range(len(times) - 1):
            s, e = times[i], times[i + 1]
            if e - s >= min_scene_len:
                scenes.append(SceneInfo(scene_index=len(scenes), start_time=round(s, 3), end_time=round(e, 3), duration=round(e - s, 3), keyframe_path=None))

        logger.info("PyTorch CUDA histogram detected %d scenes from %d frames (threshold=%.2f)", len(scenes), frame_idx, threshold)
        return scenes

    except Exception as e:
        logger.warning("PyTorch scene detection failed: %s", e)
        return []


def detect_scenes(video_path: Path, scenes_dir: Path) -> list[SceneInfo]:
    scenes_dir.mkdir(parents=True, exist_ok=True)
    fps = _get_video_fps(video_path)
    duration = _get_video_duration(video_path)
    logger.info("Video: %.1f sec @ %.2f fps", duration, fps)

    def _ffmpeg_scenes(use_gpu: bool) -> list[SceneInfo]:
        cmd = ["ffmpeg"] + (_hwaccel_flags() if use_gpu else [])
        cmd += ["-i", str(video_path), "-vf", "select='gt(scene,0.3)',showinfo", "-vsync", "vfr", "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.warning("FFmpeg%s failed: %s", " GPU" if use_gpu else "", result.stderr[:300])
            return []
        times = [0.0]
        stderr_preview = result.stderr[:500]
        if "pts_time" not in stderr_preview:
            logger.warning("No pts_time in FFmpeg%s output, threshold may be too high", " GPU" if use_gpu else "")
        for line in result.stderr.split("\n"):
            m = re.search(r"pts_time:(\d+\.\d+)", line)
            if m:
                t = float(m.group(1))
                if 0.0 < t < duration:
                    times.append(t)
        times.append(duration)
        out = []
        for i in range(len(times) - 1):
            s, e = times[i], times[i + 1]
            if e - s >= 0.5:
                out.append(SceneInfo(scene_index=len(out), start_time=round(s, 3), end_time=round(e, 3), duration=round(e - s, 3), keyframe_path=None))
        return out

    scenes = _pytorch_scenes(video_path)
    if len(scenes) > 1:
        logger.info("Detected %d scenes via PyTorch CUDA", len(scenes))
        save_scenes(scenes, scenes_dir)
        return scenes
    logger.warning("PyTorch CUDA gave %d scenes, trying FFmpeg...", len(scenes))

    scenes = _ffmpeg_scenes(use_gpu=True)
    if len(scenes) > 1:
        logger.info("Detected %d scenes via FFmpeg GPU", len(scenes))
        save_scenes(scenes, scenes_dir)
        return scenes

    logger.warning("FFmpeg GPU gave %d scenes, retrying CPU FFmpeg...", len(scenes))
    scenes = _ffmpeg_scenes(use_gpu=False)
    if len(scenes) > 1:
        logger.info("Detected %d scenes via FFmpeg CPU", len(scenes))
        save_scenes(scenes, scenes_dir)
        return scenes

    logger.warning("FFmpeg gave %d scenes, falling back to ContentDetector...", len(scenes))
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

    logger.info("Detected %d scenes", len(scenes))
    save_scenes(scenes, scenes_dir)
    return scenes


def extract_keyframes(video_path: Path, scenes: list[SceneInfo], keyframes_dir: Path) -> list[SceneInfo]:
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %d keyframes via FFmpeg%s...", len(scenes), " GPU" if settings.GPU_ENABLED else "")

    for scene in scenes:
        mid_time = (scene.start_time + scene.end_time) / 2.0
        kf_path = keyframes_dir / f"scene_{scene.scene_index:05d}.jpg"

        cmd = ["ffmpeg"] + _hwaccel_flags()
        cmd += ["-ss", str(mid_time), "-i", str(video_path), "-vframes", "1", "-q:v", "2", "-y", str(kf_path)]
        subprocess.run(cmd, capture_output=True, text=True)

        if kf_path.exists() and kf_path.stat().st_size > 0:
            scene.keyframe_path = str(kf_path)
        else:
            retry_time = scene.start_time + 0.1
            retry_cmd = ["ffmpeg"] + _hwaccel_flags()
            retry_cmd += ["-ss", str(retry_time), "-i", str(video_path), "-vframes", "1", "-q:v", "2", "-y", str(kf_path)]
            subprocess.run(retry_cmd, capture_output=True, text=True)
            if kf_path.exists() and kf_path.stat().st_size > 0:
                scene.keyframe_path = str(kf_path)

    extracted = sum(1 for s in scenes if s.keyframe_path is not None)
    logger.info("Extracted %d / %d keyframes", extracted, len(scenes))
    save_scenes(scenes, keyframes_dir.parent)
    return scenes


def save_scenes(scenes: list[SceneInfo], scenes_dir: Path):
    import json
    data = [s.model_dump() for s in scenes]
    (scenes_dir / "scenes.json").write_text(json.dumps(data, indent=2, default=str))


def load_scenes(scenes_dir: Path) -> list[SceneInfo]:
    import json
    path = scenes_dir / "scenes.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [SceneInfo(**s) for s in data]
