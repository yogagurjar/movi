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


def _get_video_resolution(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    parts = result.stdout.strip().split(",")
    return int(parts[0]), int(parts[1])


def _decord_scenes(video_path: Path, threshold: float = 0.3, min_scene_len: float = 0.5) -> list[SceneInfo] | None:
    """
    GPU-accelerated scene detection via decord library with CUDA context.
    Uses GPU NVDEC decoder + PyTorch CUDA histogram. Requires `decord` with CUDA.
    """
    import torch

    try:
        import decord
    except ImportError:
        return None

    device = torch.device(settings.TORCH_DEVICE)
    if device.type != "cuda":
        return None

    fps = _get_video_fps(video_path)
    duration = _get_video_duration(video_path)
    if duration <= 0 or fps <= 0:
        return None

    sample_rate = max(1, int(fps / 12))
    min_frames = max(1, int(min_scene_len * fps))
    min_frames_sampled = max(1, int(min_frames / sample_rate))
    pts_step = sample_rate / fps

    try:
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(video_path), ctx=decord.gpu(0))
        num_frames = len(vr)
        logger.info("Detecting scenes via decord GPU (%d frames, sample every %d)...", num_frames, sample_rate)

        prev_hist = None
        times = [0.0]
        frame_idx = 0
        last_scene_frame = 0
        log_interval = max(1, num_frames // (sample_rate * 10))

        for i in range(0, num_frames, sample_rate):
            frame = vr[i].float().to(device, non_blocking=True)
            frame = frame.permute(2, 0, 1)

            n_ch = 3
            hists = []
            for c in range(n_ch):
                flat = frame[c].reshape(-1)
                hist_c = torch.histc(flat, bins=256, min=0, max=255)
                hists.append(hist_c / max(hist_c.sum(), 1))
            curr_hist = torch.stack(hists)

            if prev_hist is not None:
                sad = torch.abs(curr_hist - prev_hist).sum().item()
                scene_score = sad / (n_ch * 2)
                pts = i / fps

                if scene_score > threshold and (frame_idx - last_scene_frame) >= min_frames_sampled:
                    if 0.0 < pts < duration:
                        times.append(pts)
                        last_scene_frame = frame_idx

            if frame_idx > 0 and frame_idx % log_interval == 0:
                pct = min(100, int(frame_idx * sample_rate * 100 / num_frames))
                logger.info("Scene detection [decord GPU]: %d%% (%d/%d frames, %d scenes found)", pct, i, num_frames, len(times) - 1)

            prev_hist = curr_hist
            frame_idx += 1

        times.append(duration)

        scenes = []
        for i in range(len(times) - 1):
            s, e = times[i], times[i + 1]
            if e - s >= min_scene_len:
                scenes.append(SceneInfo(scene_index=len(scenes), start_time=round(s, 3), end_time=round(e, 3), duration=round(e - s, 3), keyframe_path=None))

        logger.info("decord GPU detected %d scenes from %d sampled frames (threshold=%.2f)", len(scenes), frame_idx, threshold)
        return scenes

    except Exception as e:
        logger.info("decord GPU not available (CUDA not compiled), using FFmpeg pipe fallback...")
        return None


def _pytorch_scenes(video_path: Path, threshold: float = 0.3, min_scene_len: float = 0.5) -> list[SceneInfo]:
    """
    GPU-accelerated scene detection.
    Tier 1: decord with GPU NVDEC (fastest, full GPU pipeline).
    Tier 2: FFmpeg pipe + PyTorch CUDA histogram (CPU decode, GPU compute).
    """
    scenes = _decord_scenes(video_path, threshold, min_scene_len)
    if scenes is not None:
        return scenes

    import numpy as np
    import torch

    device = torch.device(settings.TORCH_DEVICE)
    if device.type != "cuda":
        return []

    fps = _get_video_fps(video_path)
    duration = _get_video_duration(video_path)
    if duration <= 0 or fps <= 0:
        return []

    width, height = _get_video_resolution(video_path)
    sample_rate = max(1, int(fps / 12))
    min_frames = max(1, int(min_scene_len * fps))
    min_frames_sampled = max(1, int(min_frames / sample_rate))
    pts_step = sample_rate / fps

    scale = 360.0 / max(width, height)
    out_w = int(width * scale / 2) * 2
    out_h = int(height * scale / 2) * 2
    if out_w < 2:
        out_w, out_h = width, height

    logger.info("Detecting scenes via FFmpeg pipe + PyTorch CUDA (%dx%d, sample every %d frame)...", out_w, out_h, sample_rate)

    select_expr = f"not(mod(n,{sample_rate}))"
    cmd = ["ffmpeg"] + _hwaccel_flags() + [
        "-i", str(video_path),
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB,scale={out_w}:{out_h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vsync", "0", "-an",
        "-"
    ]

    frame_bytes = out_w * out_h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    try:
        prev_hist = None
        times = [0.0]
        frame_idx = 0
        last_scene_frame = 0

        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break

            frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w, 3).copy()
            frame = torch.from_numpy(frame_np).to(device, non_blocking=True).float().permute(2, 0, 1)

            n_ch = 3
            hists = []
            for c in range(n_ch):
                flat = frame[c].reshape(-1)
                hist_c = torch.histc(flat, bins=256, min=0, max=255)
                hists.append(hist_c / max(hist_c.sum(), 1))
            curr_hist = torch.stack(hists)

            if prev_hist is not None:
                sad = torch.abs(curr_hist - prev_hist).sum().item()
                scene_score = sad / (n_ch * 2)

                pts = frame_idx * pts_step
                if scene_score > threshold and (frame_idx - last_scene_frame) >= min_frames_sampled:
                    if 0.0 < pts < duration:
                        times.append(pts)
                        last_scene_frame = frame_idx

            if frame_idx > 0 and frame_idx % 50 == 0:
                pct = min(100, int(frame_idx * sample_rate * 100 / (duration * fps)))
                logger.info("Scene detection [FFmpeg pipe]: %d%% (%d frames, %d scenes found)", pct, frame_idx, len(times) - 1)

            prev_hist = curr_hist
            frame_idx += 1

        times.append(duration)

        scenes = []
        for i in range(len(times) - 1):
            s, e = times[i], times[i + 1]
            if e - s >= min_scene_len:
                scenes.append(SceneInfo(scene_index=len(scenes), start_time=round(s, 3), end_time=round(e, 3), duration=round(e - s, 3), keyframe_path=None))

        logger.info("FFmpeg pipe detected %d scenes from %d sampled frames (threshold=%.2f)", len(scenes), frame_idx, threshold)
        return scenes

    except Exception as e:
        logger.warning("FFmpeg pipe scene detection failed: %s", e)
        return []
    finally:
        proc.stdout.close()
        proc.wait()


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
    total = len(scenes)
    logger.info("Extracting %d keyframes via FFmpeg single-pass%s...", total, " GPU" if settings.GPU_ENABLED and _ffmpeg_has_cuda() else "")

    fps = _get_video_fps(video_path)
    select_parts = []
    for scene in scenes:
        mid = (scene.start_time + scene.end_time) / 2.0
        frame_num = max(0, int(mid * fps))
        select_parts.append(f"eq(n,{frame_num})")
    select_expr = "+".join(select_parts)

    # Single FFmpeg pass: sequential decode, pick specified frames, output as numbered JPEGs
    cmd = ["ffmpeg"] + _hwaccel_flags() + [
        "-i", str(video_path),
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-vsync", "0",
        "-q:v", "2",
        "-y",
        str(keyframes_dir / "kf_%05d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.warning("Single-pass keyframe extraction failed (%s), falling back to per-scene...", result.stderr[:200])
        for idx, scene in enumerate(scenes):
            if idx > 0 and idx % max(1, total // 10) == 0:
                logger.info("Keyframe extraction: %d%% (%d/%d)", int(idx * 100 / total), idx, total)
            mid_time = (scene.start_time + scene.end_time) / 2.0
            kf_path = keyframes_dir / f"scene_{scene.scene_index:05d}.jpg"
            cmd = ["ffmpeg"] + _hwaccel_flags() + ["-ss", str(mid_time), "-i", str(video_path), "-vframes", "1", "-q:v", "2", "-y", str(kf_path)]
            subprocess.run(cmd, capture_output=True, text=True)
            if kf_path.exists() and kf_path.stat().st_size > 0:
                scene.keyframe_path = str(kf_path)
            else:
                retry_cmd = ["ffmpeg"] + _hwaccel_flags() + ["-ss", str(scene.start_time + 0.1), "-i", str(video_path), "-vframes", "1", "-q:v", "2", "-y", str(kf_path)]
                subprocess.run(retry_cmd, capture_output=True, text=True)
                if kf_path.exists() and kf_path.stat().st_size > 0:
                    scene.keyframe_path = str(kf_path)
    else:
        logger.info("Single-pass extraction complete, mapping %d keyframes...", total)
        for i, scene in enumerate(scenes):
            kf_path = keyframes_dir / f"kf_{i:05d}.jpg"
            if kf_path.exists() and kf_path.stat().st_size > 0:
                scene.keyframe_path = str(kf_path)
            else:
                # Single frame fallback for missed keyframes
                mid_time = (scene.start_time + scene.end_time) / 2.0
                fallback_path = keyframes_dir / f"scene_{scene.scene_index:05d}.jpg"
                fallback_cmd = ["ffmpeg"] + _hwaccel_flags() + ["-ss", str(mid_time), "-i", str(video_path), "-vframes", "1", "-q:v", "2", "-y", str(fallback_path)]
                subprocess.run(fallback_cmd, capture_output=True, text=True)
                if fallback_path.exists() and fallback_path.stat().st_size > 0:
                    scene.keyframe_path = str(fallback_path)

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
