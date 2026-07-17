import importlib
import logging
import subprocess
import sys
from pathlib import Path

import torch

from backend.config import settings

logger = logging.getLogger(__name__)

_qwen_model = None
_qwen_processor = None
_device = None


def get_device():
    global _device
    if _device is None:
        _device = torch.device(settings.TORCH_DEVICE)
    return _device


def load_qwen():
    global _qwen_model, _qwen_processor, _device
    if _qwen_model is not None:
        return

    _device = torch.device(settings.TORCH_DEVICE)

    try:
        import transformers as _tf
        if tuple(int(x) for x in _tf.__version__.split(".")[:2]) < (4, 49):
            raise ImportError(f"transformers {_tf.__version__} too old")
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    except (ImportError, AttributeError):
        target_dir = str(Path(__file__).resolve().parent.parent / ".transformers_deps")
        logger.info("Installing Qwen deps to %s...", target_dir)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir",
             "--target", target_dir,
             "numpy==1.26.4", "transformers>=4.49.0,<4.50", "accelerate==1.3.0",
             "qwen-vl-utils", "bitsandbytes"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error("Auto-install failed:\n%s", result.stderr[:500])
            raise RuntimeError("Qwen dependencies required; run: pip install numpy==1.26.4 'transformers>=4.49.0,<4.50' accelerate==1.3.0 bitsandbytes qwen-vl-utils")
        logger.info("Installed, adding %s to sys.path...", target_dir)
        sys.path.insert(0, target_dir)
        for mod in list(sys.modules):
            if mod.startswith(('numpy', 'transformers', 'accelerate', 'bitsandbytes', 'qwen', 'tokenizers', 'huggingface', 'safetensors', 'sentencepiece')):
                del sys.modules[mod]
        importlib.invalidate_caches()
        import numpy as np
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
        logger.info("numpy: %s (%s)", np.__version__, np.__file__)

    try:
        logger.info("Loading %s with 4-bit quantization on %s...", settings.QWEN_MODEL_NAME, _device)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        _qwen_model = AutoModelForVision2Seq.from_pretrained(
            settings.QWEN_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        _qwen_processor = AutoProcessor.from_pretrained(settings.QWEN_MODEL_NAME)
        try:
            _qwen_model = torch.compile(_qwen_model, mode="reduce-overhead")
            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning("torch.compile failed (continuing without): %s", e)
    except Exception as e:
        logger.warning("Failed to load Qwen model: %s", e)
        raise


def resize_for_qwen(image_path: str, max_size: int = 1024):
    import PIL.Image
    img = PIL.Image.open(image_path).convert("RGB")
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        img = img.resize((int(w * ratio), int(h * ratio)), PIL.Image.LANCZOS)
    return img


def resize_for_qwen_batch(image_paths: list[str], max_size: int = 1024):
    import PIL.Image
    imgs = []
    for path in image_paths:
        p = Path(path)
        if p.exists():
            img = PIL.Image.open(path).convert("RGB")
            w, h = img.size
            if w > max_size or h > max_size:
                ratio = min(max_size / w, max_size / h)
                img = img.resize((int(w * ratio), int(h * ratio)), PIL.Image.LANCZOS)
            imgs.append(img)

    if not imgs:
        return []

    max_w = max(img.width for img in imgs)
    max_h = max(img.height for img in imgs)

    padded = []
    for img in imgs:
        if img.width != max_w or img.height != max_h:
            canvas = PIL.Image.new("RGB", (max_w, max_h), (0, 0, 0))
            canvas.paste(img, ((max_w - img.width) // 2, (max_h - img.height) // 2))
            padded.append(canvas)
        else:
            padded.append(img)

    return padded


def get_model():
    load_qwen()
    return _qwen_model, _qwen_processor, get_device()
