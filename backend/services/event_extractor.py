import json
import logging
from pathlib import Path

import torch

from backend.config import settings
from backend.models import EventSegment, TranscriptionSegment

logger = logging.getLogger(__name__)

_qwen_model = None
_qwen_tokenizer = None
_device = None

_EVENT_GAP_THRESHOLD = 1.5


def _load_qwen_text():
    global _qwen_model, _qwen_tokenizer, _device
    if _qwen_model is not None:
        return
    _device = torch.device(settings.TORCH_DEVICE)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        _qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as e:
        logger.warning("Failed to load Qwen text model: %s, using heuristic grouping", e)
        _qwen_model = None


def _group_by_heuristic(segments: list[TranscriptionSegment]) -> list[EventSegment]:
    events: list[EventSegment] = []
    if not segments:
        return events

    current = [segments[0]]
    for seg in segments[1:]:
        gap = seg.start - current[-1].end
        if gap < _EVENT_GAP_THRESHOLD and (len(current) < 3 or (seg.end - current[0].start) < 12):
            current.append(seg)
        else:
            text = " ".join(s.text for s in current)
            events.append(EventSegment(
                event_index=len(events),
                text=text,
                start=current[0].start,
                end=current[-1].end,
                duration=round(current[-1].end - current[0].start, 2),
                segment_indices=[s.segment_index for s in current],
            ))
            current = [seg]

    if current:
        text = " ".join(s.text for s in current)
        events.append(EventSegment(
            event_index=len(events),
            text=text,
            start=current[0].start,
            end=current[-1].end,
            duration=round(current[-1].end - current[0].start, 2),
            segment_indices=[s.segment_index for s in current],
        ))

    return events


def _group_by_llm(segments: list[TranscriptionSegment]) -> list[EventSegment]:
    _load_qwen_text()
    if _qwen_model is None:
        logger.info("Qwen text model not available, using heuristic grouping")
        return _group_by_heuristic(segments)

    seg_texts = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments]
    full_text = "\n".join(seg_texts)

    prompt = (
        "Group these transcribed voiceover segments into logical events. "
        "Each event should be one complete action or idea. "
        "Merge consecutive segments that belong to the same event. "
        "Respond ONLY with a JSON array like:\n"
        '[{"start": 0.0, "end": 5.2, "text": "Merged text", "segments": [0, 1, 2]}]\n\n'
        f"{full_text}"
    )

    messages = [{"role": "user", "content": prompt}]
    text = _qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _qwen_tokenizer(text=[text], return_tensors="pt", padding=True).to(_device)

    with torch.no_grad():
        generated_ids = _qwen_model.generate(
            **inputs, max_new_tokens=1024, temperature=0.1, do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = _qwen_tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    import re
    json_match = re.search(r'\[.*\]', output_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            events = []
            for item in parsed:
                events.append(EventSegment(
                    event_index=len(events),
                    text=item.get("text", ""),
                    start=float(item.get("start", 0)),
                    end=float(item.get("end", 0)),
                    duration=round(float(item.get("end", 0)) - float(item.get("start", 0)), 2),
                    segment_indices=[int(i) for i in item.get("segments", [])],
                ))
            if events:
                return events
        except Exception:
            pass

    logger.info("LLM grouping failed or returned invalid JSON, using heuristic fallback")
    return _group_by_heuristic(segments)


def extract_events(segments: list[TranscriptionSegment], use_llm: bool = False) -> list[EventSegment]:
    if not segments:
        return []

    logger.info("Extracting events from %d voice segments...", len(segments))
    if use_llm:
        events = _group_by_llm(segments)
    else:
        events = _group_by_heuristic(segments)

    for ev in events[:5]:
        logger.info("Event[%d]: %.1f-%.1f (%.1fs) %d segs | text='%s'",
                    ev.event_index, ev.start, ev.end, ev.duration,
                    len(ev.segment_indices), ev.text[:80])

    skipped = len(segments) - sum(len(e.segment_indices) for e in events)
    logger.info("Event extraction complete: %d events from %d segments (%d skipped gaps)",
                len(events), len(segments), skipped)
    return events
