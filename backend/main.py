import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
import nest_asyncio
nest_asyncio.apply()

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.routes.processing import router as processing_router
from backend.services.ngrok_manager import start_ngrok, stop_ngrok

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Movie Recap Generator API")
    if settings.NGROK_ENABLED:
        start_ngrok()
    yield
    if settings.NGROK_ENABLED:
        stop_ngrok()
    logger.info("Shutting down Movie Recap Generator API")


app = FastAPI(
    title="Movie Recap Generator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="output")

app.include_router(processing_router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "gpu": settings.GPU_ENABLED, "device": settings.TORCH_DEVICE}


def main():
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
