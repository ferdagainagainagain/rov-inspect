"""
Frame candidate extraction and quality scoring.

A "candidate" is a frame likely worth sending to the VLM. We prefer:
  - slow ROV (operator inspecting something carefully)
  - sharp image (Laplacian variance high)
  - reasonable brightness (not black, not blown out)

Motivation: VLM calls are the slow step in the local pipeline too
(3-8 s per frame on M4 with Qwen2.5-VL-7B). Cutting candidates 10x
via cheap CV signals is the biggest throughput lever.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

from .sync import SyncedTelemetry


@dataclass
class FrameCandidate:
    frame_idx: int
    t_sec: float
    image: np.ndarray
    sharpness: float
    brightness: float
    speed_mps: float
    score: float = 0.0


def laplacian_sharpness(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def extract_candidates(
    video_path: Path,
    telemetry: SyncedTelemetry,
    sample_fps: float = 1.0,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    top_per_bucket: int = 3,
    bucket_seconds: int = 20,
) -> list[FrameCandidate]:
    """Sample frames from the video and score them for VLM-worthiness.

    Candidates are grouped into fixed-width time buckets of
    ``bucket_seconds`` and the top ``top_per_bucket`` frames by composite
    score are kept from each bucket. Bucketing by a fixed interval
    (rather than per-minute) forces uniform temporal coverage across the
    video — long stretches of "good" footage cannot drown out shorter
    stretches.

    Shallow-depth frames are dropped only when the camera is not pointed
    downward (likely water-surface or descent/ascent artifacts). Shallow
    frames captured with a downward-tilted camera — e.g. breakwater
    inspections in shallow port water — are kept as valid survey content.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_sec is None:
        end_sec = n_total / src_fps

    step = max(1, int(round(src_fps / sample_fps)))
    start_frame = int(start_sec * src_fps)
    end_frame = min(n_total, int(end_sec * src_fps))

    raw: list[FrameCandidate] = []
    for fi in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = laplacian_sharpness(gray)
        bright = float(gray.mean()) / 255.0

        try:
            telem_row = telemetry.at_frame(fi)
            speed = float(telem_row.get('speed_mps', 0.0))
            depth = float(telem_row.get('depth_m', 999.0))
            camera_pitch = float(telem_row.get('camera_pitch_deg', -30.0))
        except (IndexError, KeyError):
            speed = 0.0
            depth = 999.0
            camera_pitch = -30.0

        # Skip likely water-surface artifacts: shallow AND camera not pointed down.
        # Allows shallow survey content (e.g. breakwater inspection) to pass through.
        if depth < 0.8 and camera_pitch > -10.0:
            continue

        raw.append(FrameCandidate(
            frame_idx=fi,
            t_sec=fi / src_fps,
            image=frame,
            sharpness=sharp,
            brightness=bright,
            speed_mps=speed,
        ))
    cap.release()

    if not raw:
        return []

    def norm(arr: np.ndarray) -> np.ndarray:
        rng = float(arr.max() - arr.min())
        return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

    s = norm(np.array([c.sharpness for c in raw]))
    slow = 1.0 - norm(np.array([c.speed_mps for c in raw]))
    bright_pen = np.clip(
        1.0 - 2.0 * np.abs(np.array([c.brightness for c in raw]) - 0.45),
        0.0, 1.0,
    )
    scores = 0.5 * s + 0.3 * slow + 0.2 * bright_pen
    for c, sc in zip(raw, scores):
        c.score = float(sc)

    buckets: dict[int, list[FrameCandidate]] = {}
    for c in raw:
        buckets.setdefault(int(c.t_sec // bucket_seconds), []).append(c)
    selected: list[FrameCandidate] = []
    for bucket in buckets.values():
        bucket.sort(key=lambda x: -x.score)
        selected.extend(bucket[:top_per_bucket])
    selected.sort(key=lambda x: x.t_sec)
    return selected
