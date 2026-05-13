"""
Local text-prompted segmentation backend: SAM 3 (Meta, 2025) via
HuggingFace transformers, running on PyTorch MPS for Apple Silicon.

Used as an OPTIONAL post-processing step on top of a finished pipeline
run — not part of the main pipeline. Mirrors the LocalEmbedder pattern
in embed.py: lazy load, dataclass holding model+processor+device, a
single function that turns one BGR frame into a boolean mask.

Inference is heavy (~20 s/frame on MPS). Use the script in
scripts/run_sam3_postprocess.py rather than calling this in a tight loop.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

import cv2
import numpy as np


SAM3_DEFAULT = "facebook/sam3"


# SAM 3 is used as a *spatial verifier* on top of the VLM's categorical
# claims: for each thing the VLM names in the Descrizione, we run the
# corresponding SAM 3 text prompt and draw outlines of what SAM 3
# locates. Substrate categorization remains the VLM's job; SAM 3 just
# answers "where does this thing actually sit in the frame".
#
# Fauna was deliberately removed: the Messina data has very few real
# fauna occurrences, and the VLM hallucinated 'riccio di mare' on
# frames where SAM 3 found 0% — i.e. the VLM was wrong, not SAM 3.
# Substrate-stone prompts ('rocks', 'cobbles') were re-enabled now
# that overlays are outline-only — over-firing is visually obvious
# instead of hidden under a colour fill.
FINDING_PROMPTS: dict[str, str] = {
    "alghe":             "vegetation",
    "rocce":             "rocks",
    "ciottoli":          "cobbles",
    "rifiuto":           "waste",
    "rete_da_pesca":     "net",
    "tubo_condotta":     "pipe",
}
SUBSTRATE_PROMPTS = FINDING_PROMPTS  # alias for backward compatibility


# Mapping from VLM-output keywords (in Italian) to (finding_class, sam3_prompt).
# Used by run_sam3_postprocess.py to select per-figure prompts based on the
# VLM's report rather than running every prompt on every frame.
DESCRIZIONE_KEYWORDS_TO_PROMPT: dict[str, tuple[str, str]] = {
    "vegetazione":  ("alghe",          "vegetation"),
    "alga":         ("alghe",          "vegetation"),
    "alghe":        ("alghe",          "vegetation"),
    "asparagopsis": ("alghe",          "vegetation"),
    "corallinales": ("alghe",          "vegetation"),
    "jania":        ("alghe",          "vegetation"),
    "rocce":        ("rocce",          "rocks"),
    "roccia":       ("rocce",          "rocks"),
    "ciottoli":     ("ciottoli",       "cobbles"),
    "ciottolo":     ("ciottoli",       "cobbles"),
    "rifiuto":      ("rifiuto",        "waste"),
    "relitto":      ("rifiuto",        "waste"),
    "rete":         ("rete_da_pesca",  "net"),
    "tubo":         ("tubo_condotta",  "pipe"),
    "condotta":     ("tubo_condotta",  "pipe"),
}

@dataclass
class LocalSAM3:
    """Loaded transformers SAM 3 model + processor bundle.

    Loaded once and reused for a whole post-processing run. Mirrors the
    LocalEmbedder pattern in embed.py.
    """
    model: object
    processor: object
    device: str
    name: str


def load_local_sam3(name: str = SAM3_DEFAULT) -> LocalSAM3:
    """Lazy-import transformers/torch and load SAM 3."""
    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
    except ImportError as e:
        raise RuntimeError(
            "transformers/torch not installed. They ship transitively with "
            "mlx-vlm; reinstall the 'local' extra: uv pip install -e '.[local]'"
        ) from e

    print(f"loading {name}…")
    t0 = time.perf_counter()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = Sam3Processor.from_pretrained(name)
    model = Sam3Model.from_pretrained(name)
    model.to(device)
    model.eval()
    print(f"load time: {time.perf_counter() - t0:.2f} s")
    return LocalSAM3(model=model, processor=processor, device=device, name=name)


def segment_frame(img_bgr: np.ndarray, sam3: LocalSAM3, prompt: str) -> np.ndarray:
    """Run SAM 3 on one BGR frame with a text prompt.

    Returns a (H, W) bool array; True where any detected instance of the
    prompt overlaps. Multiple instances are merged via union — for the
    coverage / overlay use case we don't care about per-instance identity.
    """
    import torch
    from PIL import Image

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    H, W = img_rgb.shape[:2]

    inputs = sam3.processor(images=pil, text=[prompt], return_tensors="pt")
    inputs = {k: v.to(sam3.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = sam3.model(**inputs)

    results = sam3.processor.post_process_instance_segmentation(
        outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[(H, W)]
    )
    res = results[0]
    masks = res["masks"].detach().cpu().numpy()
    if len(masks) == 0:
        return np.zeros((H, W), dtype=bool)
    return np.any(masks > 0.5, axis=0)
