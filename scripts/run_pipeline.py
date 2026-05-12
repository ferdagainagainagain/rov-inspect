#!/usr/bin/env python3
"""CLI entry: end-to-end pipeline on one video.

Default backend is local (mlx-vlm + Qwen2.5-VL on Apple Silicon).
Zero cost, runs offline, no data leaves the machine.

Example:
    python scripts/run_pipeline.py \
      --video "/path/to/VIDEO 1/videos/..._SD.mp4" \
      --log   "/path/to/VIDEO 1/data/combined_log.csv" \
      --out   out/video1_dev \
      --start 0 --end 120

For a fast first run (3B model):
    ... --fast
"""
import argparse
from pathlib import Path

from rov_inspect.pipeline import run_pipeline
from rov_inspect.vlm_local import MODEL_DEFAULT, MODEL_FAST


def main() -> None:
    p = argparse.ArgumentParser(description="Run the ROV inspection pipeline on one video.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--gpx", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("out"))
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--sample-fps", type=float, default=1.0)
    p.add_argument("--top-per-bucket", type=int, default=3)
    p.add_argument("--bucket-seconds", type=int, default=20)

    p.add_argument(
        "--backend",
        choices=["local", "api"],
        default="local",
        help="VLM backend (default: local). 'api' requires ANTHROPIC_API_KEY and is NOT free.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Use the smaller 3B model for quicker iteration (lower quality).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name. Local default: Qwen2.5-VL-7B-Instruct-4bit. "
             "API default: claude-sonnet-4-6.",
    )
    p.add_argument(
        "--use-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use DINOv3 embeddings for content-aware segment dedup "
             "(default: on). Pass --no-use-embeddings for categorical-only "
             "merging — required for ablation comparisons.",
    )
    p.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.92,
        help="Cosine-similarity threshold above which adjacent frames are "
             "considered the same content (default: 0.92).",
    )
    p.add_argument(
        "--enhance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply underwater image enhancement (gray-world WB, CLAHE on "
             "L*, mild unsharp mask) before VLM + embedding (default: on). "
             "Pass --no-enhance to use raw frames.",
    )
    args = p.parse_args()

    model = args.model
    if model is None and args.backend == "local":
        model = MODEL_FAST if args.fast else MODEL_DEFAULT

    run_pipeline(
        video_path=args.video,
        log_path=args.log,
        out_dir=args.out,
        start_sec=args.start,
        end_sec=args.end,
        sample_fps=args.sample_fps,
        top_per_bucket=args.top_per_bucket,
        bucket_seconds=args.bucket_seconds,
        backend=args.backend,
        model=model,
        gpx_path=args.gpx,
        use_embeddings=args.use_embeddings,
        embedding_threshold=args.embedding_threshold,
        enhance=args.enhance,
    )


if __name__ == "__main__":
    main()
