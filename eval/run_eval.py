#!/usr/bin/env python3
"""Run the local VLM on the GT subset and dump structured predictions.

The output file is consumed by eval/compare.py to produce F1 tables
across configurations (baseline vs --enhance vs different models).

Usage:
    python eval/run_eval.py --config-name baseline
    python eval/run_eval.py --config-name enhanced --enhance
    python eval/run_eval.py --config-name gemma --model mlx-community/gemma-4-e4b-it-4bit
"""
from __future__ import annotations
import argparse
import json
import traceback
from pathlib import Path

import cv2
from tqdm import tqdm

from rov_inspect.vlm_local import load_local_vlm, analyze_frame_local, MODEL_DEFAULT
from rov_inspect.enhance import enhance_frame


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config-name", type=str, required=True)
    p.add_argument(
        "--annotations",
        type=Path,
        default=Path("eval/gt_annotations_subset.json"),
    )
    p.add_argument(
        "--enhance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply underwater enhancement before VLM (default: on).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=MODEL_DEFAULT,
        help=f"VLM model name (default: {MODEL_DEFAULT}).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval/results"),
    )
    args = p.parse_args()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    print(f"Loaded {len(annotations)} GT entries from {args.annotations}")

    vlm = load_local_vlm(args.model)

    config = {
        "name": args.config_name,
        "enhance": bool(args.enhance),
        "model": args.model,
        "annotations": str(args.annotations),
    }

    entries: list[dict] = []
    for entry in tqdm(annotations, desc=f"VLM [{args.config_name}]"):
        image_path = Path(entry["image_path"])
        result = {
            "figure_id": entry["figure_id"],
            "image_path": entry["image_path"],
            "expected": entry["expected"],
            "predicted": None,
            "config": config,
        }
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise FileNotFoundError(f"cannot read {image_path}")
            if args.enhance:
                img = enhance_frame(img)
            analysis = analyze_frame_local(img, vlm)
            result["predicted"] = json.loads(analysis.model_dump_json())
        except Exception as e:  # noqa: BLE001
            print(f"\n[{entry['figure_id']}] failed: {type(e).__name__}: {e}")
            traceback.print_exc()
        entries.append(result)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.config_name}.json"
    payload = {"config": config, "entries": entries}
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    n_ok = sum(1 for e in entries if e["predicted"] is not None)
    print(f"\n✓ Wrote {out_path} — {n_ok}/{len(entries)} successful predictions")


if __name__ == "__main__":
    main()
