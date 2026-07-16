from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    NVIDIA_API_KEY: str = ""
    NVIDIA_API_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "qwen/qwen3.5-397b-a17b"

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DOWNLOAD_DIR: Path = BASE_DIR / "data" / "downloads"
    SCENES_DIR: Path = BASE_DIR / "data" / "scenes"
    KEYFRAMES_DIR: Path = BASE_DIR / "data" / "keyframes"
    OUTPUT_DIR: Path = BASE_DIR / "data" / "output"
    TRANSCRIPT_DIR: Path = BASE_DIR / "data" / "transcripts"
    JOBS_DIR: Path = BASE_DIR / "data" / "jobs"
    LOG_DIR: Path = BASE_DIR / "data" / "logs"
    SCENE_INDEX_DIR: Path = BASE_DIR / "data" / "scene_index"

    GPU_ENABLED: bool = True
    TORCH_DEVICE: str = "cuda"

    SCENE_DETECTION_THRESHOLD: float = 27.0
    KEYFRAME_INTERVAL_SEC: float = 0.5

    WHISPER_MODEL_SIZE: str = "large-v3-turbo"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"

    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
    CLIP_BATCH_SIZE: int = 64

    QWEN_MODEL_NAME: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384
    FAISS_TOP_K: int = 10

    CONFIDENCE_ACCEPT: float = 0.85
    CONFIDENCE_FALLBACK: float = 0.50
    CONFIDENCE_MIN: float = 0.30
    TOP_K_CANDIDATES: int = 5
    SCENE_REUSE_CONFIDENCE: float = 0.90

    OUTPUT_FPS: int = 30
    OUTPUT_WIDTH: int = 1920
    OUTPUT_HEIGHT: int = 1080
    OUTPUT_CRF: int = 18
    OUTPUT_CODEC: str = "h264_nvenc"
    AUDIO_CODEC: str = "aac"
    SPEED_MAX_DELTA: float = 0.15

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_FILE_SIZE_GB: int = 10

    NGROK_AUTH_TOKEN: Optional[str] = None
    NGROK_ENABLED: bool = True

    MAX_JOB_AGE_HOURS: int = 24
    MIN_FREE_SPACE_PERCENT: float = 15.0
    TARGET_FREE_SPACE_PERCENT: float = 25.0

    MAX_WORKERS: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def model_post_init(self, __context):
        dirs = [
            self.DOWNLOAD_DIR, self.SCENES_DIR, self.KEYFRAMES_DIR,
            self.OUTPUT_DIR, self.TRANSCRIPT_DIR, self.JOBS_DIR,
            self.LOG_DIR, self.SCENE_INDEX_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        import torch
        if torch.cuda.is_available():
            self.TORCH_DEVICE = "cuda"
            self.WHISPER_DEVICE = "cuda"
        else:
            self.TORCH_DEVICE = "cpu"
            self.WHISPER_DEVICE = "cpu"
            self.WHISPER_COMPUTE_TYPE = "int8"
            self.GPU_ENABLED = False


settings = Settings()
