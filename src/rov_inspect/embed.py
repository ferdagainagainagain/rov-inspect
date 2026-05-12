"""
Local image-embedding backend: DINOv3 (Meta, 2025) via HuggingFace
transformers, running on PyTorch MPS for Apple Silicon.

Used for content-aware dedup in segment.py: visually-identical frames
can be merged even when the VLM disagrees slightly on categorical
fields. Runs entirely on Apple Silicon, no API calls, no costs.

Model choice (M4, 24 GB):
- DINOv3 ViT-S/16 (~22M params, ~90 MB) -> default, fast pooled features.

Weights are auto-downloaded on first load to the Hugging Face cache.
"""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

EMBEDDER_DEFAULT = "facebook/dinov3-vits16-pretrain-lvd1689m"

@dataclass
class LocalEmbedder:
    """Loaded transformers DINOv3 model + image processor bundle.

    Loaded once and reused for the whole pipeline run. Mirrors the
    LocalVLM pattern in vlm_local.py.
    """
    model: object
    processor: object
    device: str
    name: str


def load_local_embedder(name: str = EMBEDDER_DEFAULT) -> LocalEmbedder:
    """Lazy-import transformers/torch and load DINOv3. Done once per run."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as e:
        raise RuntimeError(
            "transformers/torch not installed. They ship transitively with "
            "mlx-vlm; reinstall the 'local' extra: uv pip install -e '.[local]'"
        ) from e

    print(f"Loading embedder {name} (first run downloads weights)…")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(name)
    model.to(device)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(name)
    return LocalEmbedder(model=model, processor=processor, device=device, name=name)


def embed_frame(img_bgr: np.ndarray, embedder: LocalEmbedder) -> np.ndarray:
    """Embed one BGR frame and return an L2-normalized float32 vector."""
    import torch
    from PIL import Image

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    inputs = embedder.processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(embedder.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = embedder.model(**inputs)
    cls = outputs.last_hidden_state[:, 0, :]

    feat = cls.cpu().numpy().astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(feat))
    if norm > 0.0:
        feat = feat / norm
    return feat


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
