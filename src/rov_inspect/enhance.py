"""
Underwater image enhancement for ROV footage.

ROV frames are typically blue/green-cast, low-contrast, and slightly
out of focus due to water absorption and turbidity. A light enhancement
pass — gray-world white balance, CLAHE on L*, and a mild unsharp mask —
restores enough chrominance and edge detail to help both the VLM
(category discrimination) and the embedder (visual-similarity dedup).

The pipeline applies the same enhanced frame to both downstream models
so they see identical inputs.
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    overall = float(means.mean())
    scale = np.where(means > 0, overall / means, 1.0)
    img *= scale
    return np.clip(img, 0, 255).astype(np.uint8)


def _clahe_on_l(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _unsharp_mask(img_bgr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(img_bgr, 1.5, blurred, -0.5, 0)


def enhance_frame(img_bgr: np.ndarray) -> np.ndarray:
    """Apply gray-world WB → CLAHE on L* → mild unsharp mask."""
    out = _gray_world_white_balance(img_bgr)
    out = _clahe_on_l(out)
    out = _unsharp_mask(out)
    return out
