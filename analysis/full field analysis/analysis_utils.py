"""Small shared helpers used by the analysis pipeline.

These functions deliberately stay free of image-analysis policy. They provide
filesystem setup, display normalization for debug images, and coordinate
conversion used by several modules.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from analysis_model import MM_PER_PIXEL, TARGET_PIXELS


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""

    path.mkdir(parents=True, exist_ok=True)


def sanitize_id(text: str) -> str:
    """Make a filesystem-safe identifier from a path or analysis name."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def normalize_for_display(image: np.ndarray, low_q: float = 0.2, high_q: float = 99.8) -> np.ndarray:
    """Map an arbitrary numeric image to uint8 for debug previews."""

    values = image.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    low = float(np.percentile(finite, low_q))
    high = float(np.percentile(finite, high_q))
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    values = np.where(np.isfinite(values), values, low)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def px_to_centered_mm(x_px: float, y_px: float) -> tuple[float, float]:
    """Convert warped target pixel coordinates to millimeters from target center."""

    x_mm = (float(x_px) + 0.5 - (TARGET_PIXELS / 2.0)) * MM_PER_PIXEL
    y_mm = (float(y_px) + 0.5 - (TARGET_PIXELS / 2.0)) * MM_PER_PIXEL
    return x_mm, y_mm
