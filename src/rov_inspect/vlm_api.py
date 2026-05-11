"""
API backend using the Anthropic API.

NOT USED BY DEFAULT in this project — kept for reference and for
ablation comparisons in the writeup. The university constraint is
that the working pipeline must be free; `vlm_local.py` is the
production path.

To enable, set ANTHROPIC_API_KEY and pass --backend api to the CLI.
"""
from __future__ import annotations
import base64

import cv2
import numpy as np

from .schema import FrameAnalysis
from .vlm_common import SYSTEM_PROMPT, user_prompt, schema_json_str, parse_response


def _encode_image(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image")
    return base64.standard_b64encode(buf.tobytes()).decode('ascii')


def analyze_frame_api(
    img_bgr: np.ndarray,
    client=None,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
) -> FrameAnalysis:
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. Run: uv pip install anthropic"
        ) from e
    client = client or Anthropic()

    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _encode_image(img_bgr),
                    },
                },
                {"type": "text", "text": user_prompt(schema_json_str())},
            ],
        }],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text"))
    return parse_response(text)
