"""
Local VLM backend using mlx-vlm with Qwen2.5-VL.

Runs entirely on Apple Silicon — no API calls, no costs, no data
leaving the machine. This is the **default** backend for the project.

Model choice (M4, 24 GB):
- Qwen2.5-VL-7B-Instruct-4bit  -> default, ~6 GB, 3-8 s/frame
- Qwen2.5-VL-3B-Instruct-4bit  -> --fast, ~3 GB, 1-3 s/frame, weaker

Model weights are auto-downloaded on first load (~5 GB for 7B-4bit)
to your Hugging Face cache (~/.cache/huggingface).
"""
from __future__ import annotations
from pathlib import Path
import tempfile
from dataclasses import dataclass

import numpy as np
import cv2

from .schema import FrameAnalysis
from .vlm_common import SYSTEM_PROMPT, user_prompt, schema_json_str, parse_response


MODEL_DEFAULT = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
MODEL_FAST = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


@dataclass
class LocalVLM:
    """Loaded mlx-vlm model + processor + config bundle.

    Loaded once and reused for the whole pipeline run. The first
    load is slow (~30-60 s); subsequent inferences are fast.
    """
    model: object
    processor: object
    config: dict
    model_name: str


def load_local_vlm(model_name: str = MODEL_DEFAULT) -> LocalVLM:
    """Lazy-import mlx-vlm and load the model. Done once per run."""
    try:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
    except ImportError as e:
        raise RuntimeError(
            "mlx-vlm not installed. Run: uv pip install 'mlx-vlm>=0.1.10'\n"
            "(Requires macOS on Apple Silicon.)"
        ) from e

    print(f"Loading {model_name} (first run downloads ~5 GB)…")
    model, processor = load(model_name)
    config = load_config(model_name)
    return LocalVLM(model=model, processor=processor, config=config, model_name=model_name)


def _save_temp_jpeg(img_bgr: np.ndarray) -> Path:
    """mlx-vlm accepts image paths or PIL images; a temp JPEG is the
    simplest reliable interface across mlx-vlm versions."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    import os
    os.close(fd)
    cv2.imwrite(path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return Path(path)


def analyze_frame_local(
    img_bgr: np.ndarray,
    vlm: LocalVLM,
    max_tokens: int = 1024,
) -> FrameAnalysis:
    """Run one VLM call on one frame and return a validated FrameAnalysis."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    tmp_path = _save_temp_jpeg(img_bgr)
    try:
        prompt = (
            f"<system>{SYSTEM_PROMPT}</system>\n\n"
            f"{user_prompt(schema_json_str())}"
        )
        formatted = apply_chat_template(
            vlm.processor, vlm.config, prompt, num_images=1
        )
        output = generate(
            vlm.model,
            vlm.processor,
            formatted,
            image=[str(tmp_path)],
            max_tokens=max_tokens,
            temperature=0.1,
            verbose=False,
        )
        # mlx-vlm versions return either a string or a GenerationResult-like
        # object; normalize.
        if hasattr(output, "text"):
            text = output.text
        else:
            text = str(output)
        return parse_response(text)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
