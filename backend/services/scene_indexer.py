import json
import logging
from pathlib import Path

import numpy as np
import torch

from backend.config import settings
from backend.models import SceneIndex
from backend.services.qwen_utils import get_device, load_qwen, resize_for_qwen, get_model

logger = logging.getLogger(__name__)

_embed_model = None


def _load_embed_model():
    global _embed_model
    if _embed_model is not None:
        return
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model %s...", settings.EMBEDDING_MODEL_NAME)
    _embed_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=settings.TORCH_DEVICE)


def _qwen_scene_summary(keyframe_paths: list[str]) -> SceneIndex | None:
    qwen_model, qwen_processor, device = get_model()
    images = []
    for kfp in keyframe_paths:
        p = Path(kfp)
        if p.exists():
            images.append(resize_for_qwen(kfp))

    if not images:
        return None

    prompt = (
        "You are analyzing a movie scene. Look at these 3 images (start, middle, end) "
        "and describe what happens in this scene. "
        "Respond ONLY with a JSON object. Do NOT include markdown formatting.\n\n"
        '{\n'
        '  "summary": "One sentence describing the main action",\n'
        '  "characters": ["character1", "character2"],\n'
        '  "location": "where this takes place",\n'
        '  "emotion": "overall emotion (e.g. tense, joyful, sad, neutral)",\n'
        '  "objects": ["key_object1", "key_object2"],\n'
        '  "actions": ["action1", "action2"]\n'
        '}'
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}],
        }
    ]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs_np = qwen_processor(text=[text], images=images)
    inputs = {}
    for key in ["input_ids", "attention_mask"]:
        if key in inputs_np:
            inputs[key] = torch.tensor(inputs_np[key].tolist(), dtype=torch.long).to(device)
    if "pixel_values" in inputs_np:
        inputs["pixel_values"] = torch.tensor(inputs_np["pixel_values"].tolist(), dtype=torch.float16).to(device)
    if "image_grid_thw" in inputs_np:
        inputs["image_grid_thw"] = torch.tensor(inputs_np["image_grid_thw"].tolist(), dtype=torch.long).to(device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=256, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    import re

    text = output_text.strip()

    # Try markdown code block first: ```json ... ``` or ``` ... ```
    code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    # Find outermost { ... } with nesting support
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                        return SceneIndex(
                            scene_index=0,
                            start_time=0,
                            end_time=0,
                            duration=0,
                            keyframe_paths=keyframe_paths,
                            summary=parsed.get("summary", ""),
                            characters=parsed.get("characters", []),
                            location=parsed.get("location", ""),
                            emotion=parsed.get("emotion", ""),
                            objects=parsed.get("objects", []),
                            actions=parsed.get("actions", []),
                        )
                    except Exception:
                        break
    return None


def index_scenes(scenes: list, scene_index_dir: Path, movie_transcript: str = "") -> list[SceneIndex]:
    scene_index_dir.mkdir(parents=True, exist_ok=True)
    _load_embed_model()

    scene_indices: list[SceneIndex] = []
    total = len(scenes)
    log_interval = max(1, total // 10)

    for i, scene in enumerate(scenes):
        if i > 0 and i % log_interval == 0:
            logger.info("Indexing scenes: %d%% (%d/%d)", int(i * 100 / total), i, total)

        keyframe_paths = getattr(scene, 'keyframe_paths', [])
        if not keyframe_paths:
            continue

        result = _qwen_scene_summary(keyframe_paths)
        if result is None:
            result = SceneIndex(
                scene_index=scene.scene_index,
                start_time=scene.start_time,
                end_time=scene.end_time,
                duration=scene.duration,
                keyframe_paths=keyframe_paths,
                summary="",
                characters=[],
                location="",
                emotion="",
                objects=[],
                actions=[],
            )

        result.scene_index = scene.scene_index
        result.start_time = scene.start_time
        result.end_time = scene.end_time
        result.duration = scene.duration
        scene_indices.append(result)

        if i < 3:
            logger.info("Scene[%d]: %s | chars=%s | loc=%s | emotion=%s",
                        scene.scene_index, result.summary[:60], result.characters,
                        result.location, result.emotion)

    logger.info("Scene indexing complete: %d scenes indexed", len(scene_indices))

    search_texts = [_build_search_text(s) for s in scene_indices]
    embeddings = _embed_model.encode(search_texts, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss_path = scene_index_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    logger.info("FAISS index saved: %d vectors, dim=%d", len(scene_indices), dim)

    data_path = scene_index_dir / "scene_indices.json"
    data_path.write_text(json.dumps([s.model_dump() for s in scene_indices], indent=2, default=str))

    return scene_indices


def _build_search_text(si: SceneIndex) -> str:
    parts = []
    if si.summary:
        parts.append(si.summary)
    if si.characters:
        parts.append("Characters: " + ", ".join(si.characters))
    if si.location:
        parts.append("Location: " + si.location)
    if si.emotion:
        parts.append("Emotion: " + si.emotion)
    if si.objects:
        parts.append("Objects: " + ", ".join(si.objects))
    if si.actions:
        parts.append("Actions: " + ", ".join(si.actions))
    return ". ".join(parts)


def load_scene_index(scene_index_dir: Path) -> tuple[list[SceneIndex], list[str] | None]:
    data_path = scene_index_dir / "scene_indices.json"
    faiss_path = scene_index_dir / "faiss.index"
    if not data_path.exists() or not faiss_path.exists():
        return [], None
    data = json.loads(data_path.read_text())
    scene_indices = [SceneIndex(**s) for s in data]
    import faiss
    index = faiss.read_index(str(faiss_path))
    return scene_indices, index


def search_scenes(event_text: str, scene_indices: list[SceneIndex], index, top_k: int = 10) -> list[tuple[int, float]]:
    _load_embed_model()
    emb = _embed_model.encode([event_text])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    scores, indices = index.search(emb.astype(np.float32), top_k)
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
