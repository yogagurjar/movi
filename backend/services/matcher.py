import json
import logging
from pathlib import Path

import torch

from backend.config import settings
from backend.models import (
    ClipMatchCandidate,
    EventSegment,
    MatchResult,
    SceneIndex,
    TimelineSegment,
    VerificationResult,
)
from backend.services.qwen_utils import get_device, load_qwen, resize_for_qwen_batch, get_model

logger = logging.getLogger(__name__)


def _qwen_verify_event(
    event_text: str,
    candidates: list[tuple[int, SceneIndex]],
) -> list[tuple[int, float, str]]:
    if not candidates:
        return []
    qwen_model, qwen_processor, device = get_model()
    if qwen_model is None:
        return []

    results: list[tuple[int, float, str]] = []
    for scene_idx, scene_index in candidates:
        try:
            kfp_list = [str(kfp) for kfp in scene_index.keyframe_paths if Path(kfp).exists()]
            images = resize_for_qwen_batch(kfp_list)

            if not images:
                results.append((scene_idx, 0.0, "no_images"))
                continue

            summary_text = scene_index.summary or ""
            chars = ", ".join(scene_index.characters) if scene_index.characters else "unknown"
            loc = scene_index.location or "unknown"
            prompt = (
                f"On a scale of 0.0 to 1.0, how well do these images match the event description? "
                f"Scene summary: {summary_text} | Characters: {chars} | Location: {loc}\n"
                f"Event description: {event_text}\n\n"
                f"Respond ONLY with a JSON object: "
                f'{{"confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}'
            )
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}],
                }
            ]
            text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            processed = [qwen_processor.image_processor(images=[img], return_tensors="pt") for img in images]
            vis = {"pixel_values": torch.cat([p["pixel_values"] for p in processed], dim=0)}
            if "image_grid_thw" in processed[0]:
                vis["image_grid_thw"] = torch.cat([p["image_grid_thw"] for p in processed], dim=0)

            tok = qwen_processor.tokenizer(text=[text], padding=True, return_tensors="pt")
            inputs = {"pixel_values": vis["pixel_values"].to(device)}
            if "image_grid_thw" in vis:
                inputs["image_grid_thw"] = vis["image_grid_thw"].to(device)
            inputs["input_ids"] = tok["input_ids"].to(device)
            inputs["attention_mask"] = tok["attention_mask"].to(device)

            with torch.no_grad():
                generated_ids = qwen_model.generate(
                    **inputs, max_new_tokens=150, temperature=0.1, do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            import re
            json_match = re.search(r'\{[^{}]*\}', output_text)
            if json_match:
                parsed = json.loads(json_match.group())
                confidence = float(parsed.get("confidence", 0.0))
                reasoning = str(parsed.get("reasoning", ""))
                results.append((scene_idx, min(max(confidence, 0.0), 1.0), reasoning))
            else:
                results.append((scene_idx, 0.0, f"no_json: {output_text[:100]}"))
        except Exception as e:
            logger.warning("Qwen VL error for scene %d: %s", scene_idx, e)
            results.append((scene_idx, 0.0, f"qwen_error: {e}"))

    return results


def match_events_to_scenes(
    events: list[EventSegment],
    scene_indices: list[SceneIndex],
    faiss_index,
) -> list[MatchResult]:
    if not scene_indices or not events or faiss_index is None:
        logger.warning("Missing scene_indices, events, or faiss_index")
        return []

    from backend.services.scene_indexer import search_scenes, _build_search_text

    logger.info("Matching %d events to %d indexed scenes via FAISS + Qwen...",
                len(events), len(scene_indices))

    used_scenes: set[int] = set()
    match_results: list[MatchResult] = []
    total = len(events)
    log_interval = max(1, total // 10)

    for vi, event in enumerate(events):
        if vi > 0 and vi % log_interval == 0:
            matched = sum(1 for m in match_results if m.timeline is not None)
            logger.info("Matching: %d%% (%d/%d events, %d matched)",
                        int(vi * 100 / total), vi, total, matched)

        top_candidates = search_scenes(event.text, scene_indices, faiss_index, settings.FAISS_TOP_K)

        raw_candidates = [
            ClipMatchCandidate(scene_index=si, similarity=round(score, 4))
            for si, score in top_candidates
        ]
        candidates = [c for c in raw_candidates if c.similarity >= settings.CONFIDENCE_MIN * 0.5]
        if not candidates and raw_candidates:
            candidates = [raw_candidates[0]]

        scene_candidates = [
            (si, scene_indices[si])
            for si, _ in top_candidates
            if 0 <= si < len(scene_indices)
        ]

        qwen_results = _qwen_verify_event(event.text, scene_candidates)

        qwen_map: dict[int, tuple[float, str]] = {}
        for si, conf, reason in qwen_results:
            qwen_map[si] = (conf, reason)

        best_candidate: int | None = None
        best_confidence: float = 0.0
        best_reasoning: str | None = None
        best_sim: float = 0.0
        qwen_all_failed = all(qwen_map.get(si, (0.0, ""))[0] == 0.0 for si, _ in top_candidates)

        for c in candidates:
            qv_conf, qv_reason = qwen_map.get(c.scene_index, (0.0, ""))
            if qwen_all_failed:
                final_conf = c.similarity
            else:
                final_conf = (c.similarity * 0.3) + (qv_conf * 0.7)

            if c.scene_index in used_scenes and final_conf < settings.SCENE_REUSE_CONFIDENCE:
                continue

            if final_conf > settings.CONFIDENCE_ACCEPT:
                if best_candidate is None or final_conf > best_confidence:
                    best_candidate = c.scene_index
                    best_confidence = final_conf
                    best_reasoning = qv_reason
                    best_sim = c.similarity

        if best_candidate is None:
            for c in candidates:
                qv_conf, qv_reason = qwen_map.get(c.scene_index, (0.0, ""))
                if qwen_all_failed:
                    final_conf = c.similarity
                else:
                    final_conf = (c.similarity * 0.3) + (qv_conf * 0.7)

                if c.scene_index in used_scenes:
                    continue

                if settings.CONFIDENCE_FALLBACK <= final_conf < settings.CONFIDENCE_ACCEPT:
                    if best_candidate is None or final_conf > best_confidence:
                        best_candidate = c.scene_index
                        best_confidence = final_conf
                        best_reasoning = qv_reason
                        best_sim = c.similarity

        if best_candidate is None and candidates:
            best_candidate = candidates[0].scene_index
            best_confidence = candidates[0].similarity
            best_sim = candidates[0].similarity
            best_reasoning = "fallback_top1"

        if vi < 3:
            logger.info("Match[%d]: faiss_top=%.4f, candidates=%d, qwen_ok=%s, accepted=%s, conf=%.4f, text='%s'",
                        vi, top_candidates[0][1] if top_candidates else 0,
                        len(candidates), not qwen_all_failed,
                        best_candidate is not None, best_confidence, event.text[:60])

        accepted = best_candidate is not None
        if accepted:
            used_scenes.add(best_candidate)
            si = scene_indices[best_candidate]
            image_paths = si.keyframe_paths
            timeline = TimelineSegment(
                scene_index=si.scene_index,
                source_path=image_paths[len(image_paths) // 2] if image_paths else "",
                source_paths=image_paths,
                trim_start=si.start_time,
                trim_end=si.end_time,
                original_duration=si.duration,
                target_duration=event.duration,
                voice_segment_index=vi,
                voice_text=event.text,
            )
        else:
            timeline = None

        match_results.append(
            MatchResult(
                voice_segment_index=vi,
                voice_text=event.text,
                voice_start=event.start,
                voice_end=event.end,
                candidates=candidates,
                verification=VerificationResult(
                    scene_index=best_candidate or -1,
                    clip_similarity=best_sim,
                    final_confidence=best_confidence,
                    nvidia_reasoning=best_reasoning,
                    accepted=accepted,
                ) if accepted else None,
                timeline=timeline,
            )
        )

    matched = sum(1 for m in match_results if m.timeline is not None)
    logger.info("Matching complete: %d / %d events matched (%.1f%%)",
                matched, len(events), matched * 100 / max(len(events), 1))
    return match_results
