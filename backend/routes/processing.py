import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import settings
from backend.models import JobStatusResponse, ProcessRequest, ProcessResponse
from backend.services.job_manager import create_and_start_job, get_job_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["Processing"])


@router.post("/process-gdrive", response_model=ProcessResponse)
async def process_gdrive(request: ProcessRequest):
    if not request.movie_url or not request.voiceover_url:
        raise HTTPException(status_code=400, detail="Both movie_url and voiceover_url are required")
    response = create_and_start_job(
        movie_url=request.movie_url,
        voiceover_url=request.voiceover_url,
        webhook_url=request.webhook_url,
        use_local_paths=request.use_local_paths,
    )
    return response


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/download/{job_id}")
async def download_video(job_id: str):
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if status.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed (status: {status.status})")
    if not status.output_url:
        raise HTTPException(status_code=404, detail="Output file not found")

    file_path = settings.OUTPUT_DIR / job_id / "recap.mp4"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found on disk")

    filename = status.output_filename or "recap.mp4"
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4",
    )
