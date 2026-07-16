from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime


class JobStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    DETECTING_SCENES = "detecting_scenes"
    EXTRACTING_KEYFRAMES = "extracting_keyframes"
    INDEXING_SCENES = "indexing_scenes"
    EXTRACTING_EVENTS = "extracting_events"
    MATCHING = "matching"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessRequest(BaseModel):
    movie_url: str = Field(..., description="Google Drive URL OR local file path for the movie")
    voiceover_url: str = Field(..., description="Google Drive URL OR local file path for the voiceover")
    webhook_url: Optional[str] = Field(None, description="Optional callback URL on completion")
    use_local_paths: bool = Field(False, description="If True, movie_url/voiceover_url are treated as local file paths")


class ProcessResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=100.0)
    current_stage: str
    error: Optional[str] = None
    output_url: Optional[str] = None
    output_filename: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    estimated_remaining_sec: Optional[int] = None


class SceneInfo(BaseModel):
    scene_index: int
    start_time: float
    end_time: float
    duration: float
    keyframe_path: Optional[str] = None
    keyframe_paths: list[str] = []


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float
    probability: float


class TranscriptionSegment(BaseModel):
    segment_index: int
    text: str
    start: float
    end: float
    words: list[TranscriptionWord] = []


class TranscriptionResult(BaseModel):
    segments: list[TranscriptionSegment]
    full_text: str
    duration_sec: float


class ClipMatchCandidate(BaseModel):
    scene_index: int
    similarity: float


class VerificationResult(BaseModel):
    scene_index: int
    clip_similarity: float
    nvidia_confidence: Optional[float] = None
    final_confidence: float
    nvidia_reasoning: Optional[str] = None
    accepted: bool


class TimelineSegment(BaseModel):
    scene_index: int
    source_path: str = ""
    source_paths: list[str] = []
    trim_start: float = 0.0
    trim_end: float = 0.0
    original_duration: float = 0.0
    target_duration: float = 0.0
    speed_factor: float = 1.0
    voice_segment_index: int = 0
    voice_text: str = ""


class MatchResult(BaseModel):
    voice_segment_index: int
    voice_text: str
    voice_start: float
    voice_end: float
    candidates: list[ClipMatchCandidate] = []
    verification: Optional[VerificationResult] = None
    timeline: Optional[TimelineSegment] = None


class SceneIndex(BaseModel):
    scene_index: int
    start_time: float
    end_time: float
    duration: float
    keyframe_paths: list[str]
    summary: str = ""
    characters: list[str] = []
    location: str = ""
    emotion: str = ""
    objects: list[str] = []
    actions: list[str] = []
    dialogue: str = ""


class EventSegment(BaseModel):
    event_index: int
    text: str
    start: float
    end: float
    duration: float
    segment_indices: list[int]


class GpuInfo(BaseModel):
    available: bool
    name: str
    memory_total_gb: float
    memory_free_gb: float


class DiskInfo(BaseModel):
    total_gb: float
    used_gb: float
    free_gb: float
    free_percent: float


class ProcessingStats(BaseModel):
    total_scenes: int = 0
    total_voice_segments: int = 0
    matched_segments: int = 0
    skipped_segments: int = 0
    total_clip_candidates: int = 0
    nvidia_api_calls: int = 0
    rendering_duration_sec: Optional[float] = None
    total_processing_time_sec: Optional[float] = None
