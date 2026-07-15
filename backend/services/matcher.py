import base64
import json
import logging
from pathlib import Path

import httpx
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import settings
from backend.models import (
    ClipMatchCandidate,
    MatchResult,
    TranscriptionSegment,
    VerificationResult,
    TimelineSegment,
    SceneInfo,
)

logger = logging.getLogger(__name__)

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_device = None


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer, _device
    if _clip_model is not None:
        return
    import open_clip
    _device = torch.device(settings.TORCH_DEVICE)
    logger.info(
        "Loading CLIP model %s / %s on %s",
        settings.CLIP_MODEL_NAME,
        settings.CLIP_PRETRAINED,
        _device,
    )
    _clip_model, _clip_preprocess = open_clip.create_model_and_transforms(
        settings.CLIP_MODEL_NAME,
        pretrained=settings.CLIP_PRETRAINED,
        device=_device,
    )[:2]
    _clip_tokenizer = open_clip.get_tokenizer(settings.CLIP_MODEL_NAME)
    _clip_model.eval()


def _encode_images(image_paths: list[str]) -> np.ndarray:
    _load_clip()
    import PIL.Image
    images = []
    for path in image_paths:
        try:
            img = PIL.Image.open(path).convert("RGB")
            images.append(_clip_preprocess(img).to(_device))
        except Exception as e:
            logger.warning("Failed to load image %s: %s", path, e)
            images.append(torch.zeros((3, 224, 224), device=_device))

    if not images:
        return np.zeros((0, 512))

    batch = torch.stack(images)
    with torch.no_grad():
        embeddings = _clip_model.encode_image(batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy()


def _encode_texts(texts: list[str]) -> np.ndarray:
    _load_clip()
    tokens = _clip_tokenizer(texts).to(_device)
    with torch.no_grad():
        embeddings = _clip_model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy()


def _nvidia_verify(
    voice_text: str,
    candidate_scenes: list[tuple[int, str]],
) -> list[tuple[int, float, str]]:
    if not candidate_scenes:
        return []
    results: list[tuple[int, float, str]] = []
    for scene_idx, kf_path in candidate_scenes:
        try:
            with open(kf_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            logger.warning("Failed to encode keyframe %s: %s", kf_path, e)
            results.append((scene_idx, 0.0, "image_load_error"))
            continue

        prompt = (
            f"On a scale of 0.0 to 1.0, how well does this image match "
            f"the following description? Only respond with a JSON object: "
            f'{{"confidence": <float>, "reasoning": "<brief explanation>"}}\n\n'
            f"Description: {voice_text}"
        )

        try:
            resp = httpx.post(
                f"{settings.NVIDIA_API_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.NVIDIA_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_url},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.60,
                    "top_p": 0.95,
                    "top_k": 20,
                    "presence_penalty": 0,
                    "repetition_penalty": 1,
                },
                timeout=60,
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            import re
            json_match = re.search(r'\{[^{}]*\}', content)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = {"confidence": 0.0, "reasoning": content.strip()}
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = str(parsed.get("reasoning", ""))
            results.append((scene_idx, min(max(confidence, 0.0), 1.0), reasoning))
        except Exception as e:
            logger.warning("NVIDIA API error for scene %d: %s", scene_idx, e)
            results.append((scene_idx, 0.0, f"api_error: {e}"))

    return results


def match_voice_to_scenes(
    voice_segments: list[TranscriptionSegment],
    scenes: list[SceneInfo],
) -> list[MatchResult]:
    if not scenes or not voice_segments:
        return []

    valid_scenes = [(s.scene_index, s.keyframe_path) for s in scenes if s.keyframe_path]
    if not valid_scenes:
        logger.warning("No keyframes available for matching")
        return []

    scene_indices = [s[0] for s in valid_scenes]
    kf_paths = [s[1] for s in valid_scenes]
    voice_texts = [seg.text for seg in voice_segments]

    logger.info(
        "Encoding %d keyframes and %d voice segments...",
        len(kf_paths), len(voice_texts),
    )
    image_embs = _encode_images(kf_paths)
    text_embs = _encode_texts(voice_texts)

    sim_matrix = cosine_similarity(text_embs, image_embs)

    all_sims = sim_matrix.flatten()
    logger.info(
        "CLIP sim matrix: shape=%s, range=[%.4f, %.4f], mean=%.4f, median=%.4f",
        sim_matrix.shape, all_sims.min(), all_sims.max(), all_sims.mean(), np.median(all_sims),
    )

    used_scenes: set[int] = set()
    match_results: list[MatchResult] = []

    for vi, seg in enumerate(voice_segments):
        sims = sim_matrix[vi]
        top5_idx = np.argsort(sims)[::-1][:settings.TOP_K_CANDIDATES]

        raw_candidates = [
            (scene_indices[i], float(sims[i]))
            for i in top5_idx
        ]
        candidates = [
            ClipMatchCandidate(scene_index=idx, similarity=round(sim, 4))
            for idx, sim in raw_candidates
            if sim >= settings.CONFIDENCE_MIN * 0.5
        ]
        if not candidates and raw_candidates:
            best = max(raw_candidates, key=lambda x: x[1])
            candidates = [ClipMatchCandidate(scene_index=best[0], similarity=round(best[1], 4))]

        nvidia_input = [
            (c.scene_index, kf_paths[scene_indices.index(c.scene_index)])
            for c in candidates
        ]
        nvidia_results = _nvidia_verify(seg.text, nvidia_input)

        nvidia_map: dict[int, tuple[float, str]] = {}
        for si, conf, reason in nvidia_results:
            nvidia_map[si] = (conf, reason)

        best_candidate: int | None = None
        best_confidence: float = 0.0
        best_reasoning: str | None = None
        best_clip_sim: float = 0.0
        nvidia_all_failed = all(nvidia_map.get(c.scene_index, (0.0, ""))[0] == 0.0 for c in candidates)

        for c in candidates:
            nv_conf, nv_reason = nvidia_map.get(c.scene_index, (0.0, ""))
            if nvidia_all_failed:
                final_conf = c.similarity
            else:
                final_conf = (c.similarity * 0.3) + (nv_conf * 0.7)

            if c.scene_index in used_scenes and final_conf < settings.SCENE_REUSE_CONFIDENCE:
                continue

            if final_conf > settings.CONFIDENCE_ACCEPT:
                if best_candidate is None or final_conf > best_confidence:
                    best_candidate = c.scene_index
                    best_confidence = final_conf
                    best_reasoning = nv_reason
                    best_clip_sim = c.similarity

        if best_candidate is None:
            for c in candidates:
                nv_conf, nv_reason = nvidia_map.get(c.scene_index, (0.0, ""))
                if nvidia_all_failed:
                    final_conf = c.similarity
                else:
                    final_conf = (c.similarity * 0.3) + (nv_conf * 0.7)

                if c.scene_index in used_scenes:
                    continue

                if settings.CONFIDENCE_FALLBACK <= final_conf < settings.CONFIDENCE_ACCEPT:
                    if best_candidate is None or final_conf > best_confidence:
                        best_candidate = c.scene_index
                        best_confidence = final_conf
                        best_reasoning = nv_reason
                        best_clip_sim = c.similarity

        if best_candidate is None and candidates:
            best_candidate = candidates[0].scene_index
            best_confidence = candidates[0].similarity
            best_clip_sim = candidates[0].similarity
            best_reasoning = "fallback_top1"

        if vi < 5 or (vi % 50 == 0):
            logger.info(
                "Match[%d]: top_sim=%.4f, candidates=%d, nvidia_ok=%s, accepted=%s, conf=%.4f, text='%s'",
                vi, candidates[0].similarity if candidates else 0,
                len(candidates), not nvidia_all_failed,
                best_candidate is not None, best_confidence,
                seg.text[:60],
            )

        accepted = best_candidate is not None
        if accepted:
            used_scenes.add(best_candidate)
            scene = scenes[best_candidate]
            timeline = TimelineSegment(
                scene_index=scene.scene_index,
                source_path=scene.keyframe_path or "",
                trim_start=scene.start_time,
                trim_end=scene.end_time,
                original_duration=scene.duration,
                target_duration=seg.end - seg.start,
                speed_factor=1.0,
                voice_segment_index=seg.segment_index,
                voice_text=seg.text,
            )
        else:
            timeline = None

        match_results.append(
            MatchResult(
                voice_segment_index=seg.segment_index,
                voice_text=seg.text,
                voice_start=seg.start,
                voice_end=seg.end,
                candidates=candidates,
                verification=VerificationResult(
                    scene_index=best_candidate or -1,
                    clip_similarity=best_clip_sim,
                    nvidia_confidence=best_confidence,
                    final_confidence=best_confidence,
                    nvidia_reasoning=best_reasoning,
                    accepted=accepted,
                ) if accepted else None,
                timeline=timeline,
            )
        )

    matched = sum(1 for m in match_results if m.timeline is not None)
    not_matched = [m for m in match_results if m.timeline is None]
    for nm in not_matched[:5]:
        logger.info(
            "NON-MATCH: seg[%d] sim=%.4f conf=%.4f accepted=%s text='%s'",
            nm.voice_segment_index, nm.verification.clip_similarity if nm.verification else 0,
            nm.verification.final_confidence if nm.verification else 0,
            nm.verification.accepted if nm.verification else False,
            nm.voice_text[:60],
        )
    logger.info(
        "Matching complete: %d / %d segments matched",
        matched, len(voice_segments),
    )
    return match_results
