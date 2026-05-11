#!/usr/bin/env python3
"""Smoke-test the local VLM on a single image.

Useful for verifying the model loads and produces valid JSON before
investing time in a full pipeline run.

Usage:
    python scripts/test_local_vlm.py path/to/frame.jpg
    python scripts/test_local_vlm.py path/to/frame.jpg --fast
"""
import argparse
import sys
from pathlib import Path

import cv2

from rov_inspect.vlm_local import load_local_vlm, analyze_frame_local, MODEL_DEFAULT, MODEL_FAST
from rov_inspect.render import render_caption


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Cannot read image: {args.image}", file=sys.stderr)
        sys.exit(1)

    model_name = MODEL_FAST if args.fast else MODEL_DEFAULT
    vlm = load_local_vlm(model_name)
    print(f"Loaded {model_name}. Running inference…")
    analysis = analyze_frame_local(img, vlm)

    print("\n── Structured output ──")
    print(analysis.model_dump_json(indent=2))
    print("\n── Rendered caption ──")
    print(render_caption(analysis))


if __name__ == "__main__":
    main()
