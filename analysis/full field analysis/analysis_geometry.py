"""Target geometry construction.

The physical five-dot target is represented as masks in the normalized warped
image. Other modules consume this single geometry object instead of rebuilding
marker masks independently.
"""

from __future__ import annotations

import numpy as np

from analysis_model import *


def build_geometry(target_pixels: int = TARGET_PIXELS) -> TargetGeometry:
    """Build masks from the printed target dimensions."""

    scale = float(target_pixels) / float(TARGET_PIXELS)
    marker_centers = EXPECTED_MARKER_CENTERS * scale
    dot_radius_px = DOT_RADIUS_PX * scale
    yy, xx = np.indices((target_pixels, target_pixels), dtype=np.float32)
    marker_masks: list[np.ndarray] = []
    marker_ring_masks: list[np.ndarray] = []
    marker_exclusion_mask = np.zeros((target_pixels, target_pixels), dtype=bool)
    template_model = np.ones((target_pixels, target_pixels), dtype=np.float32)
    full_mask = np.ones((target_pixels, target_pixels), dtype=bool)
    border_px = max(4, int(round(35.0 * scale)))
    border_mask = np.zeros((target_pixels, target_pixels), dtype=bool)
    border_mask[border_px:-border_px, border_px:-border_px] = True

    for center_x, center_y in marker_centers:
        distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
        marker_mask = distance_sq <= (dot_radius_px**2)
        ring_mask = (distance_sq >= ((95.0 * scale) ** 2)) & (distance_sq <= ((165.0 * scale) ** 2))
        exclusion_mask = distance_sq <= ((115.0 * scale) ** 2)
        marker_masks.append(marker_mask)
        marker_ring_masks.append(ring_mask)
        marker_exclusion_mask |= exclusion_mask
        template_model[marker_mask] = 0.0

    calibration_mask = border_mask & (~marker_exclusion_mask)
    analysis_mask = border_mask & (~marker_exclusion_mask)
    plot_mask = full_mask
    white_background_mask = border_mask & (~marker_exclusion_mask)
    return TargetGeometry(
        marker_centers=marker_centers.astype(np.float32),
        marker_masks=marker_masks,
        marker_exclusion_mask=marker_exclusion_mask,
        calibration_mask=calibration_mask,
        analysis_mask=analysis_mask,
        plot_mask=plot_mask,
        white_background_mask=white_background_mask,
        marker_ring_masks=marker_ring_masks,
        template_model=template_model,
    )
