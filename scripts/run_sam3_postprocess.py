#!/usr/bin/env python3
"""Post-processing: report-driven SAM 3 verification on Video 3.

For each figure in out/video3_enhanced/report.md, we inspect the VLM's
Descrizione line, look up which discrete findings (rete / tubo /
pneumatico / rifiuto / relitto / riccio / stella / mollusco / trave)
the VLM flagged, and run only the corresponding SAM 3 prompts. SAM 3
acts as a spatial verifier on top of the VLM's categorical claims —
not as a substrate classifier.

Outputs land in out/video3_enhanced/sam3_verification/:
  - figure_NN_overlay.png      mask overlay
  - figure_NN_comparison.png   side-by-side (if --make-comparison)
  - figure_NN_coverage.json    per-figure structured record
  - summary.json               run-wide aggregate
  - sam_verification.md        consolidated demo report

Hardcoded to Video 3 for this iteration. Any --run-dir argument is
accepted but ignored — generalization to other videos comes later.

Usage:
    python scripts/run_sam3_postprocess.py --make-comparison
"""
from __future__ import annotations
import argparse
import json
import re
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from rov_inspect.segment_sam3 import (
    DESCRIZIONE_KEYWORDS_TO_PROMPT,
    SAM3_DEFAULT,
    load_local_sam3,
    segment_frame,
)


# Hardcoded target for this iteration (per current spec).
TARGET_RUN_DIR = Path("out/video3_enhanced")
OUT_SUBDIR = "sam3_verification"
REPORT_TITLE = "SAM 3 Verification Report — Video 3"


# Priority order, highest first. When a pixel is covered by masks of
# multiple finding classes the highest-priority class wins. Pixels not
# covered by any finding mask land in "unclassified" — the implicit
# substrate handled by the VLM.
# Classes mirror segment_sam3.FINDING_PROMPTS / DESCRIZIONE_KEYWORDS_TO_PROMPT
# exactly — that module is authoritative. Priority orders the discrete
# anthropic objects above the substrate-stone classes, and substrate-stone
# above the most diffuse class (alghe), so when masks overlap the more
# specific finding wins.
PRIORITY_HIGH_TO_LOW = [
    "rete_da_pesca",
    "tubo_condotta",
    "rifiuto",
    "rocce",
    "ciottoli",
    "alghe",
]


# RGB tuples (color-blind aware). cv2 uses BGR — convert at use time.
CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "rete_da_pesca": (230, 25,  75),   # crimson
    "tubo_condotta": (200, 80,  60),   # brick
    "rifiuto":       (180, 30,  30),   # dark red
    "rocce":         (128, 128, 128),  # gray
    "ciottoli":      (200, 175, 150),  # tan/light stone
    "alghe":         (60,  180, 75),   # green
    "unclassified":  (0,   0,   0),    # only used as a JSON label
}


# ── report parsing ────────────────────────────────────────────────────

FIG_HEADER_RE = re.compile(r"^##\s*Figura\s+(\d+)\s*—\s*(.+)$", re.IGNORECASE)
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
DESC_RE = re.compile(r"^\*\*Descrizione:\*\*\s*(.+)$")
NOTE_RE = re.compile(r"^_Note:_\s*(.+)$")
HEADER_META_RE = re.compile(
    r"t\s*=\s*([\d:]+).*?profondit[àa]\s+([\d.]+)\s*m",
    re.IGNORECASE,
)


def parse_report(report_path: Path) -> list[dict]:
    """Walk report.md and pull out per-figure entries.

    Returns a list of dicts:
      {figure_id, image_rel, header_meta, t_str, depth_str, descrizione, note}.
    """
    if not report_path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")
    lines = report_path.read_text(encoding="utf-8").splitlines()

    entries: list[dict] = []
    cur: dict | None = None
    for raw in lines:
        line = raw.rstrip()
        m = FIG_HEADER_RE.match(line)
        if m:
            if cur is not None:
                entries.append(cur)
            header_meta = m.group(2).strip()
            mm = HEADER_META_RE.search(header_meta)
            t_str = mm.group(1) if mm else ""
            depth_str = mm.group(2) if mm else ""
            cur = {
                "figure_id": int(m.group(1)),
                "image_rel": None,
                "header_meta": header_meta,
                "t_str": t_str,
                "depth_str": depth_str,
                "descrizione": "",
                "note": "",
            }
            continue
        if cur is None:
            continue
        m = IMG_RE.search(line)
        if m and cur["image_rel"] is None:
            cur["image_rel"] = m.group(1)
            continue
        m = DESC_RE.match(line)
        if m:
            cur["descrizione"] = m.group(1).strip()
            continue
        m = NOTE_RE.match(line)
        if m:
            cur["note"] = m.group(1).strip()
    if cur is not None:
        entries.append(cur)
    return entries


def select_prompts_for(entry: dict) -> tuple[list[str], list[tuple[str, str]]]:
    """Inspect the figure's Descrizione, return (matched_keywords,
    prompts_to_run). prompts_to_run is a list of (finding_class,
    sam3_text_prompt) tuples, deduplicated."""
    desc = entry["descrizione"].lower()
    matched: list[str] = []
    prompts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kw, (cls, sam3_prompt) in DESCRIZIONE_KEYWORDS_TO_PROMPT.items():
        if kw in desc:
            matched.append(kw)
            tup = (cls, sam3_prompt)
            if tup not in seen:
                seen.add(tup)
                prompts.append(tup)
    return matched, prompts


# ── overlay rendering ─────────────────────────────────────────────────

def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (b, g, r)


def render_overlay(
    image_bgr: np.ndarray,
    label_map: np.ndarray,
    classes_in_order: list[str],
    alpha: float = 0.4,  # kept for signature stability; no longer used.
) -> np.ndarray:
    """Draw class outlines onto the frame, no fill.

    The underlying image is fully visible inside each contour so the
    user can see what SAM 3 actually detected without colour bleed.
    """
    del alpha  # explicitly unused — overlays are outline-only now.
    overlay = image_bgr.copy()
    H, W = image_bgr.shape[:2]
    thickness = 3 if max(H, W) > 1280 else 2
    for i, cls in enumerate(classes_in_order, start=1):
        sel = (label_map == i).astype(np.uint8)
        if not sel.any():
            continue
        contours, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            overlay, contours, -1,
            _rgb_to_bgr(CLASS_COLORS_RGB[cls]), thickness,
        )
    _draw_legend(overlay, classes_in_order)
    return overlay


def _draw_legend(img: np.ndarray, classes: list[str]) -> None:
    if not classes:
        return
    pad = 8
    box = 14
    line_h = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    panel_h = pad * 2 + line_h * len(classes)
    panel_w = 230
    x0, y0 = 10, 10
    sub = img[y0:y0 + panel_h, x0:x0 + panel_w]
    if sub.shape[0] == panel_h and sub.shape[1] == panel_w:
        sub[:] = cv2.addWeighted(sub, 0.4, np.zeros_like(sub), 0.0, 0)
    for i, cls in enumerate(classes):
        y = y0 + pad + i * line_h
        cv2.rectangle(
            img, (x0 + pad, y), (x0 + pad + box, y + box),
            _rgb_to_bgr(CLASS_COLORS_RGB[cls]), -1,
        )
        cv2.putText(
            img, cls, (x0 + pad + box + 6, y + box - 2),
            font, fs, (255, 255, 255), 1, cv2.LINE_AA,
        )


# ── per-frame work ────────────────────────────────────────────────────

def process_frame_selective(
    image_path: Path,
    sam3,
    prompts_to_run: list[tuple[str, str]],
) -> tuple[
    np.ndarray, np.ndarray, list[str],
    dict[str, float], dict[str, float], float,
]:
    """Run only the supplied (class, prompt) pairs and priority-combine.

    Returns:
      img_bgr, label_map, classes_in_order, class_coverage,
      prompt_coverage, total_finding_pct
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"cannot read {image_path}")
    H, W = img_bgr.shape[:2]
    total = H * W

    # Run each unique prompt once and remember its raw mask.
    raw_masks: dict[str, np.ndarray] = {}
    prompt_coverage: dict[str, float] = {}
    for _cls, prompt in prompts_to_run:
        if prompt in raw_masks:
            continue
        mask = segment_frame(img_bgr, sam3, prompt)
        raw_masks[prompt] = mask
        prompt_coverage[prompt] = round(100.0 * float(mask.mean()), 2)

    # Union per finding class (multiple prompts may map to the same class).
    classes_present = list({cls for cls, _ in prompts_to_run})
    class_masks: dict[str, np.ndarray] = {
        cls: np.zeros((H, W), dtype=bool) for cls in classes_present
    }
    for cls, prompt in prompts_to_run:
        class_masks[cls] |= raw_masks[prompt]

    # Build label_map: paint low-priority first so high overwrites.
    classes_in_order = sorted(
        classes_present,
        key=lambda c: PRIORITY_HIGH_TO_LOW.index(c) if c in PRIORITY_HIGH_TO_LOW else 9999,
    )
    cls_to_idx = {c: i + 1 for i, c in enumerate(classes_in_order)}
    label_map = np.zeros((H, W), dtype=np.int32)
    for cls in reversed(PRIORITY_HIGH_TO_LOW):
        if cls not in class_masks:
            continue
        label_map[class_masks[cls]] = cls_to_idx[cls]

    class_coverage: dict[str, float] = {}
    for cls in classes_in_order:
        n = int((label_map == cls_to_idx[cls]).sum())
        pct = round(100.0 * n / total, 2)
        if pct > 0.0:
            class_coverage[cls] = pct
    unclassified = round(100.0 * int((label_map == 0).sum()) / total, 2)
    class_coverage["unclassified"] = unclassified
    total_finding = round(100.0 - unclassified, 2)

    return img_bgr, label_map, classes_in_order, class_coverage, prompt_coverage, total_finding


# ── consolidated markdown ─────────────────────────────────────────────

def render_verification_md(
    title: str,
    n_processed: int,
    n_skipped: int,
    figure_blocks: list[str],
) -> str:
    intro = (
        "Spatial verification layer on top of the VLM categorical output. "
        "For each figure where the VLM flagged a discrete finding "
        "(fishing net, pipe, tire, debris, fauna, etc.), SAM 3 was run "
        "with the corresponding text prompt to produce pixel-level masks."
    )
    stats = (
        f"_Processed: {n_processed} figures with discrete findings. "
        f"Skipped: {n_skipped} figures with substrate-only content._"
    )
    body = "\n\n".join(figure_blocks) if figure_blocks else "_No figures processed._"
    return f"# {title}\n\n{intro}\n\n{stats}\n\n{body}\n"


def figure_md_block(
    entry: dict,
    matched_keywords: list[str],
    prompt_coverage: dict[str, float],
    comparison_filename: str,
) -> str:
    t_str = entry["t_str"] or "?"
    depth_str = entry["depth_str"] or "?"
    header = f"## Figura {entry['figure_id']} — t={t_str}, depth {depth_str} m"
    desc = entry["descrizione"]
    kw_line = ", ".join(matched_keywords) if matched_keywords else "—"
    if prompt_coverage:
        ran_items = [f"{prompt} ({pct:.2f}% coverage)"
                     for prompt, pct in prompt_coverage.items()]
        ran_line = ", ".join(ran_items)
    else:
        ran_line = "—"
    img_ref = f"![Figure {entry['figure_id']} comparison]({comparison_filename})"
    return "\n".join([
        header,
        "",
        f"**VLM said:** {desc}",
        "",
        f"**Keywords matched:** {kw_line}",
        "",
        f"**SAM 3 ran:** {ran_line}",
        "",
        img_ref,
        "",
        "---",
    ])


# ── main ──────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Accepted but ignored — target is hardcoded to "
                        "out/video3_enhanced for this iteration.")
    p.add_argument("--make-comparison", action="store_true",
                   help="Also write side-by-side comparison PNGs.")
    p.add_argument("--model", default=SAM3_DEFAULT,
                   help=f"SAM 3 model id (default: {SAM3_DEFAULT}).")
    args = p.parse_args()

    if args.run_dir is not None and args.run_dir != TARGET_RUN_DIR:
        print(f"note: --run-dir={args.run_dir} ignored; using {TARGET_RUN_DIR}")

    run_dir = TARGET_RUN_DIR
    report_path = run_dir / "report.md"
    frames_dir = run_dir / "frames"
    out_dir = run_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = parse_report(report_path)
    print(f"parsed {len(entries)} figures from {report_path}")

    # Per-figure prompt selection.
    plans: list[tuple[dict, list[str], list[tuple[str, str]]]] = []
    skipped: list[dict] = []
    for e in entries:
        matched, prompts = select_prompts_for(e)
        if prompts:
            plans.append((e, matched, prompts))
            print(f"  keep fig {e['figure_id']:>2}: keywords={matched} "
                  f"prompts={[p for _, p in prompts]}")
        else:
            skipped.append(e)
            print(f"  skip fig {e['figure_id']:>2}: no discrete finding")
    print(f"processing {len(plans)} figures, skipping {len(skipped)}")

    summary = {
        "run_dir": str(run_dir),
        "sam3_model": args.model,
        "skipped_figure_ids": [e["figure_id"] for e in skipped],
        "processed": [],
        "errors": [],
        "total_inference_s": 0.0,
    }

    if not plans:
        print("nothing to do.")
        # Still write a minimal markdown so the consolidated artifact exists.
        md = render_verification_md(REPORT_TITLE, 0, len(skipped), [])
        (out_dir / "sam_verification.md").write_text(md, encoding="utf-8")
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    sam3 = load_local_sam3(args.model)

    t_start = time.perf_counter()
    figure_blocks: list[str] = []

    for entry, matched_keywords, prompts_to_run in tqdm(plans, desc="SAM 3"):
        fig = entry["figure_id"]
        image_rel = entry["image_rel"]
        if image_rel is None:
            print(f"  fig {fig}: no image path in report — skipping")
            summary["errors"].append({"figure_id": fig, "error": "no image path"})
            continue
        image_path = run_dir / image_rel
        if not image_path.exists():
            image_path = frames_dir / Path(image_rel).name

        try:
            t0 = time.perf_counter()
            (img_bgr, label_map, classes_in_order,
             class_coverage, prompt_coverage, total_finding) = process_frame_selective(
                image_path, sam3, prompts_to_run,
            )
            inference_s = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            print(f"  fig {fig}: failed ({type(e).__name__}: {e})")
            traceback.print_exc()
            summary["errors"].append({
                "figure_id": fig, "error": f"{type(e).__name__}: {e}"
            })
            continue

        overlay = render_overlay(img_bgr, label_map, classes_in_order)
        cv2.imwrite(str(out_dir / f"figure_{fig:02d}_overlay.png"), overlay)

        comparison_filename = f"figure_{fig:02d}_comparison.png"
        if args.make_comparison:
            cmp_img = np.hstack([img_bgr, overlay])
            cv2.imwrite(str(out_dir / comparison_filename), cmp_img)
            md_image_ref = comparison_filename
        else:
            md_image_ref = f"figure_{fig:02d}_overlay.png"

        cov_payload = {
            "figure_id": fig,
            "image_path": str(image_path),
            "vlm_flagged_keywords": matched_keywords,
            "prompts_run": [
                {"finding_class": cls, "sam3_prompt": prm}
                for cls, prm in prompts_to_run
            ],
            "prompt_coverage_pct": prompt_coverage,
            "coverage_pct": class_coverage,
            "total_finding_coverage_pct": total_finding,
            "sam3_model": args.model,
            "inference_s": round(inference_s, 2),
        }
        (out_dir / f"figure_{fig:02d}_coverage.json").write_text(
            json.dumps(cov_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        prompt_log = ", ".join(
            f"{prm}={pct:.2f}%" for prm, pct in prompt_coverage.items()
        )
        print(f"  fig {fig:>2}: keywords={matched_keywords} "
              f"prompts_run={[p for _, p in prompts_to_run]}  {prompt_log}")

        figure_blocks.append(figure_md_block(
            entry, matched_keywords, prompt_coverage, md_image_ref,
        ))
        summary["processed"].append({
            "figure_id": fig,
            "vlm_flagged_keywords": matched_keywords,
            "prompts_run": [list(t) for t in prompts_to_run],
            "prompt_coverage_pct": prompt_coverage,
            "coverage_pct": class_coverage,
            "total_finding_coverage_pct": total_finding,
            "inference_s": round(inference_s, 2),
        })

    elapsed = time.perf_counter() - t_start
    summary["total_inference_s"] = round(elapsed, 2)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md = render_verification_md(
        REPORT_TITLE,
        len(summary["processed"]),
        len(skipped),
        figure_blocks,
    )
    md_path = out_dir / "sam_verification.md"
    md_path.write_text(md, encoding="utf-8")

    print()
    print(f"processed {len(summary['processed'])} figures, skipped {len(skipped)}")
    print(f"total elapsed: {elapsed:.1f} s")
    print(f"errors: {len(summary['errors'])}")
    print(f"sam_verification.md: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
