import json
import logging
import time
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from backend.config import settings
from backend.models import (
    DiskInfo,
    GpuInfo,
    JobStatus,
    JobStatusResponse,
    ProcessResponse,
    ProcessingStats,
)
from backend.services.audio_extractor import convert_audio_to_wav, extract_audio_from_video
from backend.services.downloader import download_from_gdrive, cleanup_job_files
from backend.services.event_extractor import extract_events
from backend.services.scene_indexer import index_scenes, load_scene_index
from backend.services.matcher import match_events_to_scenes
from backend.services.renderer import build_timeline, render_video
from backend.services.scene_detector import detect_scenes, extract_keyframes, load_scenes
from backend.services.transcriber import transcribe_audio, load_transcript

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
_jobs: dict[str, dict] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persist_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return
    path = settings.JOBS_DIR / f"{job_id}.json"
    serializable = {
        k: v for k, v in job.items()
        if k != "_lock"
    }
    if isinstance(serializable.get("created_at"), datetime):
        serializable["created_at"] = serializable["created_at"].isoformat()
    if isinstance(serializable.get("updated_at"), datetime):
        serializable["updated_at"] = serializable["updated_at"].isoformat()
    path.write_text(json.dumps(serializable, indent=2, default=str))


def _load_jobs():
    if not settings.JOBS_DIR.exists():
        return
    for path in settings.JOBS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            jid = data.get("job_id")
            if jid:
                if "status" in data:
                    data["status"] = JobStatus(data["status"])
                _jobs[jid] = data
        except Exception as e:
            logger.warning("Failed to load job %s: %s", path.name, e)


def _get_gpu_info() -> GpuInfo:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        free = round(
            (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3),
            2,
        )
        return GpuInfo(available=True, name=name, memory_total_gb=total, memory_free_gb=free)
    return GpuInfo(available=False, name="", memory_total_gb=0.0, memory_free_gb=0.0)


def _get_disk_info() -> DiskInfo:
    import shutil
    total, used, free = shutil.disk_usage(settings.BASE_DIR)
    total_gb = round(total / (1024**3), 2)
    used_gb = round(used / (1024**3), 2)
    free_gb = round(free / (1024**3), 2)
    free_pct = round((free / total) * 100, 1)
    return DiskInfo(total_gb=total_gb, used_gb=used_gb, free_gb=free_gb, free_percent=free_pct)


def _cleanup_old_jobs():
    jobs_to_delete: list[tuple[str, datetime]] = []
    now = datetime.now(timezone.utc)
    for jid, job in _jobs.items():
        if job.get("status") in (JobStatus.COMPLETED, JobStatus.FAILED):
            created = job.get("created_at")
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            if created and (now - created).total_seconds() > settings.MAX_JOB_AGE_HOURS * 3600:
                jobs_to_delete.append((jid, created))

    jobs_to_delete.sort(key=lambda x: x[1])
    disk = _get_disk_info()
    for jid, _ in jobs_to_delete:
        if disk.free_percent >= settings.TARGET_FREE_SPACE_PERCENT:
            break
        job_dir = settings.DOWNLOAD_DIR / jid
        if job_dir.exists():
            cleanup_job_files(job_dir)
        _jobs.pop(jid, None)
        jpath = settings.JOBS_DIR / f"{jid}.json"
        if jpath.exists():
            jpath.unlink()
        disk = _get_disk_info()
        logger.info("Cleaned up job %s, free space: %.1f%%", jid, disk.free_percent)


def _run_pipeline(job_id: str, movie_url: str, voiceover_url: str, use_local_paths: bool = False):
    job = _jobs.get(job_id)
    if not job:
        return

    job_dir = settings.DOWNLOAD_DIR / job_id
    scenes_dir = settings.SCENES_DIR / job_id
    kf_dir = settings.KEYFRAMES_DIR / job_id
    output_dir = settings.OUTPUT_DIR / job_id
    transcript_dir = settings.TRANSCRIPT_DIR / job_id
    index_dir = settings.SCENE_INDEX_DIR / job_id

    for d in [job_dir, scenes_dir, kf_dir, output_dir, transcript_dir, index_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats = ProcessingStats()
    start_time = time.time()

    try:
        checkpoint = _did_checkpoint(job_id, "downloaded")
        if not checkpoint:
            if use_local_paths:
                _update_job(job_id, JobStatus.DOWNLOADING, "Using local files...", 5.0)
                movie_path = Path(movie_url)
                voiceover_path = Path(voiceover_url)
                if not movie_path.exists():
                    raise FileNotFoundError(f"Movie not found: {movie_path}")
                if not voiceover_path.exists():
                    raise FileNotFoundError(f"Voiceover not found: {voiceover_path}")
            else:
                _update_job(job_id, JobStatus.DOWNLOADING, "Downloading movie and voiceover...", 5.0)
                movie_path = download_from_gdrive(movie_url, job_dir, "movie")
                voiceover_path = download_from_gdrive(voiceover_url, job_dir, "voiceover")
            _checkpoint(job_id, "downloaded")
        else:
            logger.info("Checkpoint: download already complete, skipping")
            movie_path = next(job_dir.glob("movie*"), None)
            voiceover_path = next(job_dir.glob("voiceover*"), None)
            if not movie_path or not voiceover_path:
                raise FileNotFoundError("Checkpoint files missing, restart required")
        disk = _get_disk_info()
        logger.info("Disk after download: %.1f%% free", disk.free_percent)

        checkpoint = _did_checkpoint(job_id, "audio_extracted")
        if not checkpoint:
            movie_ext = movie_path.suffix.lower()
            if movie_ext in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
                _update_job(job_id, JobStatus.EXTRACTING_AUDIO, "Extracting movie audio...", 8.0)
                movie_audio = extract_audio_from_video(movie_path, transcript_dir / "movie_audio")
            else:
                movie_audio = convert_audio_to_wav(movie_path, transcript_dir / "movie_audio")

            _update_job(job_id, JobStatus.EXTRACTING_AUDIO, "Converting voiceover...", 10.0)
            voice_wav = convert_audio_to_wav(voiceover_path, transcript_dir / "voiceover")
            _checkpoint(job_id, "audio_extracted")
        else:
            logger.info("Checkpoint: audio already extracted, skipping")
            voice_wav = transcript_dir / "voiceover.wav"

        checkpoint = _did_checkpoint(job_id, "transcribed")
        if not checkpoint:
            _update_job(job_id, JobStatus.TRANSCRIBING, "Transcribing voiceover...", 13.0)
            voice_transcript = transcribe_audio(voice_wav)
            stats.total_voice_segments = len(voice_transcript.segments)
            logger.info("Voiceover: %d segments, %.1f sec",
                        stats.total_voice_segments, voice_transcript.duration_sec)

            _update_job(job_id, JobStatus.TRANSCRIBING, "Transcribing movie audio for context...", 17.0)
            try:
                movie_transcript = transcribe_audio(movie_audio)
                stats.total_scenes = len(movie_transcript.segments)
                logger.info("Movie transcript: %d segments, %.1f sec",
                            len(movie_transcript.segments), movie_transcript.duration_sec)
            except Exception as e:
                logger.warning("Movie transcription failed (non-fatal): %s", e)
                movie_transcript = None
            _checkpoint(job_id, "transcribed")
        else:
            logger.info("Checkpoint: transcription already complete, skipping")
            voice_transcript = load_transcript(transcript_dir)
            movie_transcript = None

        checkpoint = _did_checkpoint(job_id, "scenes_detected")
        if not checkpoint:
            _update_job(job_id, JobStatus.DETECTING_SCENES, "Detecting scenes...", 25.0)
            scenes = detect_scenes(movie_path, scenes_dir)
            stats.total_scenes = len(scenes)
            _checkpoint(job_id, "scenes_detected")
        else:
            logger.info("Checkpoint: scenes already detected, skipping")
            scenes = load_scenes(scenes_dir)

        checkpoint = _did_checkpoint(job_id, "keyframes_extracted")
        if not checkpoint:
            _update_job(job_id, JobStatus.EXTRACTING_KEYFRAMES, "Extracting keyframes...", 30.0)
            scenes = extract_keyframes(movie_path, scenes, kf_dir)
            _checkpoint(job_id, "keyframes_extracted")
        else:
            logger.info("Checkpoint: keyframes already extracted, skipping")

        checkpoint = _did_checkpoint(job_id, "scenes_indexed")
        if not checkpoint:
            _update_job(job_id, JobStatus.INDEXING_SCENES, "Indexing scenes with Qwen + FAISS...", 40.0)
            movie_dialogue = movie_transcript.full_text if movie_transcript else ""
            scene_indices = index_scenes(scenes, index_dir, movie_dialogue)
            _checkpoint(job_id, "scenes_indexed")
        else:
            logger.info("Checkpoint: scenes already indexed, skipping")
            scene_indices, _ = load_scene_index(index_dir)
            if not scene_indices:
                scene_indices = []

        checkpoint = _did_checkpoint(job_id, "events_extracted")
        if not checkpoint:
            _update_job(job_id, JobStatus.EXTRACTING_EVENTS, "Extracting events from voiceover...", 55.0)
            events = extract_events(voice_transcript.segments, use_llm=False)
            stats.matched_segments = len(events)
            _checkpoint(job_id, "events_extracted")
        else:
            logger.info("Checkpoint: events already extracted, skipping")
            events = []

        checkpoint = _did_checkpoint(job_id, "matched")
        if not checkpoint:
            _update_job(job_id, JobStatus.MATCHING, "Matching events to scenes via FAISS + Qwen...", 60.0)
            _, faiss_index = load_scene_index(index_dir)
            match_results = match_events_to_scenes(events, scene_indices, faiss_index)
            stats.matched_segments = sum(1 for m in match_results if m.timeline is not None)
            stats.skipped_segments = len(events) - stats.matched_segments
            _checkpoint(job_id, "matched")
        else:
            logger.info("Checkpoint: matching already complete, skipping")
            match_results = []

        checkpoint = _did_checkpoint(job_id, "rendered")
        if not checkpoint:
            _update_job(job_id, JobStatus.RENDERING, "Building timeline and rendering...", 80.0)
            timeline = build_timeline(match_results)
            if not timeline:
                raise RuntimeError("No segments matched, cannot render")

            output_path = render_video(movie_path, voice_wav, timeline, output_dir / "recap")
            stats.rendering_duration_sec = round(sum(t.target_duration for t in timeline), 2)
            stats.total_processing_time_sec = round(time.time() - start_time, 2)
            _checkpoint(job_id, "rendered")
        else:
            logger.info("Checkpoint: rendering already complete, skipping")

        job["output_url"] = f"/output/{job_id}/recap.mp4"
        job["output_filename"] = "recap.mp4"
        job["stats"] = stats.model_dump()

        _cleanup_old_jobs()
        _update_job(job_id, JobStatus.COMPLETED, "Processing complete!", 100.0)

    except Exception as e:
        logger.exception("Pipeline failed for job %s", job_id)
        _update_job(job_id, JobStatus.FAILED, f"Pipeline error: {e}", 0.0)
        job["error"] = str(e)


def _checkpoint(job_id: str, stage_name: str):
    job = _jobs.get(job_id)
    if not job:
        return
    job["checkpoint"] = stage_name
    _persist_job(job_id)


def _did_checkpoint(job_id: str, stage_name: str) -> bool:
    job = _jobs.get(job_id)
    return job is not None and job.get("checkpoint") == stage_name


def _update_job(job_id: str, status: JobStatus, stage: str, progress: float):
    job = _jobs.get(job_id)
    if not job:
        return
    job["status"] = status
    job["current_stage"] = stage
    job["progress"] = progress
    job["updated_at"] = datetime.now(timezone.utc)
    _persist_job(job_id)


def create_and_start_job(movie_url: str, voiceover_url: str, webhook_url: Optional[str] = None, use_local_paths: bool = False) -> ProcessResponse:
    _load_jobs()
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    job: dict = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "progress": 0.0,
        "current_stage": "Queued",
        "error": None,
        "output_url": None,
        "output_filename": None,
        "created_at": now,
        "updated_at": now,
        "estimated_remaining_sec": None,
        "webhook_url": webhook_url,
        "gpu": _get_gpu_info().model_dump(),
    }
    _jobs[job_id] = job
    _persist_job(job_id)

    _executor.submit(_run_pipeline, job_id, movie_url, voiceover_url, use_local_paths)

    return ProcessResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job queued successfully",
    )


def get_job_status(job_id: str) -> JobStatusResponse | None:
    job = _jobs.get(job_id)
    if not job:
        _load_jobs()
        job = _jobs.get(job_id)
    if not job:
        return None

    remaining = job.get("estimated_remaining_sec")
    if job["status"] == JobStatus.PENDING:
        remaining = 600
    elif job["status"] == JobStatus.DOWNLOADING:
        remaining = 480
    elif job["status"] == JobStatus.EXTRACTING_AUDIO:
        remaining = 300
    elif job["status"] == JobStatus.TRANSCRIBING:
        remaining = 420
    elif job["status"] == JobStatus.DETECTING_SCENES:
        remaining = 360
    elif job["status"] == JobStatus.EXTRACTING_KEYFRAMES:
        remaining = 300
    elif job["status"] == JobStatus.INDEXING_SCENES:
        remaining = 600
    elif job["status"] == JobStatus.EXTRACTING_EVENTS:
        remaining = 120
    elif job["status"] == JobStatus.MATCHING:
        remaining = 480
    elif job["status"] == JobStatus.RENDERING:
        remaining = 180
    elif job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED):
        remaining = 0

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        current_stage=job["current_stage"],
        error=job.get("error"),
        output_url=job.get("output_url"),
        output_filename=job.get("output_filename"),
        created_at=job["created_at"] if isinstance(job["created_at"], datetime) else datetime.fromisoformat(job["created_at"]),
        updated_at=job["updated_at"] if isinstance(job["updated_at"], datetime) else datetime.fromisoformat(job["updated_at"]),
        estimated_remaining_sec=remaining,
    )
