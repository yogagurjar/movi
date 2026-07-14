import logging
from pathlib import Path

from faster_whisper import WhisperModel

from backend.config import settings
from backend.models import TranscriptionResult, TranscriptionSegment, TranscriptionWord

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info(
            "Loading Whisper model %s on %s (%s)",
            settings.WHISPER_MODEL_SIZE,
            settings.WHISPER_DEVICE,
            settings.WHISPER_COMPUTE_TYPE,
        )
        _model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            num_workers=settings.MAX_WORKERS,
        )
    return _model


def transcribe_audio(audio_path: Path) -> TranscriptionResult:
    model = _get_model()
    logger.info("Transcribing %s", audio_path.name)

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    duration = round(info.duration, 2) if info and info.duration else 0.0
    seg_list: list[TranscriptionSegment] = []
    full_text_parts: list[str] = []

    for idx, seg in enumerate(segments):
        words = []
        if seg.words:
            words = [
                TranscriptionWord(
                    word=w.word.strip(),
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    probability=round(w.probability, 4),
                )
                for w in seg.words
            ]

        seg_list.append(
            TranscriptionSegment(
                segment_index=idx,
                text=seg.text.strip(),
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                words=words,
            )
        )
        full_text_parts.append(seg.text.strip())

    result = TranscriptionResult(
        segments=seg_list,
        full_text=" ".join(full_text_parts),
        duration_sec=duration,
    )

    logger.info(
        "Transcription complete: %d segments, %.1f sec",
        len(seg_list),
        duration,
    )
    return result
