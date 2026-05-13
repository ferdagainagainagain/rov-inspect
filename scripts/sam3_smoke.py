#!/usr/bin/env python3
"""Smoke test for Meta SAM 3 via HuggingFace transformers.

Loads SAM 3 once and runs one or more text prompts against a single
image, saving a green-outline overlay PNG per prompt. NOT integrated
with the pipeline — this script exists for quick visual sanity checks
of prompt phrasings on individual frames.

Usage:
    python scripts/sam3_smoke.py
    python scripts/sam3_smoke.py --input path/to/frame.jpg \\
        --prompts "sand" "gravel" "rocks" "algae" \\
        --out-dir out/some_run/sam3_smoke
"""
from __future__ import annotations
import argparse
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from rov_inspect.segment_sam3 import load_local_sam3, segment_frame


DEFAULT_INPUT = Path("out/video2_phase1_embed_thr050/frames/frame_001_t16s.jpg")
DEFAULT_PROMPTS = ["sand", "gravel", "rocks", "algae"]
DEFAULT_OUT_DIR = Path("/tmp")


def slugify(prompt: str) -> str:
    """Lowercase, collapse non-alphanumerics to a single underscore."""
    s = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return s or "prompt"


def render_overlay(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Green translucent fill + green outline (matches the previous
    smoke-test style so outputs across prompts look comparable)."""
    overlay = img_bgr.copy()
    if not mask.any():
        return overlay
    color = np.zeros_like(img_bgr)
    color[..., 1] = 255  # green
    blended = cv2.addWeighted(img_bgr, 0.5, color, 0.5, 0)
    overlay = np.where(mask[..., None], blended, overlay)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Image to segment.")
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                   help="One or more text prompts.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Directory to write overlay PNGs.")
    args = p.parse_args()

    input_path: Path = args.input.resolve()
    out_dir: Path = args.out_dir
    prompts: list[str] = list(args.prompts)

    print(f"input:    {input_path}")
    print(f"out-dir:  {out_dir}")
    print(f"prompts:  {prompts}")

    if not input_path.exists():
        print(f"input image not found: {input_path}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    sam3 = load_local_sam3()
    print(f"device:   {sam3.device}")

    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        print(f"could not decode image: {input_path}", file=sys.stderr)
        return 1

    saved: list[Path] = []
    for prompt in prompts:
        slug = slugify(prompt)
        out_path = out_dir / f"sam3_{slug}.png"
        try:
            t0 = time.perf_counter()
            mask = segment_frame(img_bgr, sam3, prompt)
            inference_s = time.perf_counter() - t0
            overlay = render_overlay(img_bgr, mask)
            cv2.imwrite(str(out_path), overlay)
            saved.append(out_path)
            covered = float(mask.mean()) * 100.0
            print(
                f"  prompt={prompt!r:<14} time={inference_s:5.2f} s  "
                f"coverage={covered:5.1f}%  -> {out_path}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  prompt={prompt!r}: failed ({type(e).__name__}: {e})")
            traceback.print_exc()

    print()
    print(f"saved {len(saved)}/{len(prompts)} overlays:")
    for p_ in saved:
        print(f"  {p_}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
