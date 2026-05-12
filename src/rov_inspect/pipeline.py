"""End-to-end pipeline with local-VLM default and API fallback."""
from __future__ import annotations
from pathlib import Path
from typing import Literal

import cv2

from .sync import synchronize
from .candidates import extract_candidates
from .segment import merge_segments
from .render import render_markdown_report

Backend = Literal["local", "api"]


def run_pipeline(
    video_path: Path,
    log_path: Path,
    out_dir: Path,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    sample_fps: float = 1.0,
    top_per_bucket: int = 3,
    bucket_seconds: int = 20,
    backend: Backend = "local",
    model: str | None = None,
    gpx_path: Path | None = None,
    use_embeddings: bool = True,
    embedding_threshold: float = 0.92,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # ── 1. Inspect video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    cap.release()
    print(f"[1/5] Video: {fps:.2f} fps, {n_frames} frames ({duration:.1f} s)")

    # ── 2. Sync telemetry
    telem = synchronize(log_path, fps=fps, n_frames=n_frames, gpx_path=gpx_path)
    print(f"[2/5] Telemetry synced to {len(telem.df)} per-frame rows")

    # ── 3. Extract candidate frames
    cands = extract_candidates(
        video_path, telem,
        sample_fps=sample_fps,
        start_sec=start_sec,
        end_sec=end_sec,
        top_per_bucket=top_per_bucket,
        bucket_seconds=bucket_seconds,
    )
    if not cands:
        print("No candidates extracted — check the time range.")
        return
    print(f"[3/5] {len(cands)} candidate frames in [{start_sec}, {end_sec or duration}] s")

    # ── 4. VLM analysis (backend-dispatched)
    if backend == "local":
        from .vlm_local import load_local_vlm, analyze_frame_local, MODEL_DEFAULT
        vlm = load_local_vlm(model or MODEL_DEFAULT)

        def call(img):
            return analyze_frame_local(img, vlm)
    elif backend == "api":
        from .vlm_api import analyze_frame_api
        from anthropic import Anthropic
        client = Anthropic()

        def call(img):
            return analyze_frame_api(img, client=client, model=model or "claude-sonnet-4-6")
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    embedder = None
    if use_embeddings:
        from .embed import load_local_embedder, embed_frame
        embedder = load_local_embedder()

    analyses = []
    embeddings: list = []
    for i, c in enumerate(cands, 1):
        print(f"[4/5] VLM {i}/{len(cands)} (t={c.t_sec:.1f}s, score={c.score:.2f})…", flush=True)
        try:
            analyses.append(call(c.image))
        except Exception as e:  # noqa: BLE001
            print(f"      skipped: {type(e).__name__}: {e}")
            analyses.append(None)
        if embedder is not None:
            try:
                embeddings.append(embed_frame(c.image, embedder))
            except Exception as e:  # noqa: BLE001
                print(f"      embed skipped: {type(e).__name__}: {e}")
                embeddings.append(None)

    if embedder is not None:
        paired = [
            (c, a, e)
            for c, a, e in zip(cands, analyses, embeddings)
            if a is not None and e is not None
        ]
        if not paired:
            print("All VLM calls failed. Aborting.")
            return
        cands_ok = [t[0] for t in paired]
        analyses_ok = [t[1] for t in paired]
        embeddings_ok = [t[2] for t in paired]
    else:
        paired = [(c, a) for c, a in zip(cands, analyses) if a is not None]
        if not paired:
            print("All VLM calls failed. Aborting.")
            return
        cands_ok = [t[0] for t in paired]
        analyses_ok = [t[1] for t in paired]
        embeddings_ok = None

    # ── 5. Segment + render
    segments = merge_segments(
        cands_ok,
        analyses_ok,
        embeddings=embeddings_ok,
        embedding_threshold=embedding_threshold,
    )
    print(f"[5/5] {len(segments)} segments after dedup → rendering")

    items = []
    for i, seg in enumerate(segments, 1):
        rep_c, rep_a = seg.representative
        img_name = f"frame_{i:03d}_t{int(rep_c.t_sec)}s.jpg"
        cv2.imwrite(str(frames_dir / img_name), rep_c.image)
        meta = {"t_sec": rep_c.t_sec, "image_rel": f"frames/{img_name}"}
        if rep_c.frame_idx < len(telem.df):
            row = telem.df.iloc[rep_c.frame_idx]
            for col in ("depth_m", "lat", "lon", "heading_deg"):
                if col in row.index:
                    meta[col] = float(row[col])
        items.append((rep_a, meta))

    render_markdown_report(items, out_dir / "report.md")
    print(f"\n✓ Done. Open {out_dir / 'report.md'}")
