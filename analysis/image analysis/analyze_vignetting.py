#!/usr/bin/env python3
"""Visualize vignetting for the A-D focal-length/aperture configurations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import cv2
if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rawpy

from variable_exposure_analysis import detect_inner_corners, compute_destination, warp_image

ROW_LABELS = (
    ("1", "f/4 / widest"),
    ("2", "f/5.6"),
    ("3", "f/8"),
    ("4", "f/10"),
    ("5", "f/20"),
)
COL_LABELS = (
    ("A", "55 mm"),
    ("B", "70 mm"),
    ("C", "~90 mm"),
    ("D", "~190 mm with spacers"),
)
BORDER_ARTIFACT_CROP_FRACTION = 0.03
NORMALIZED_BASE_LEVEL = 0.86
COLOR_SCALE_LOW_PERCENTILE = 0.5
COLOR_SCALE_HIGH_PERCENTILE = 95.0


def _display_image(image: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    current_max = max(image.shape)
    if current_max <= max_dim:
        return image
    scale = max_dim / float(current_max)
    width = max(int(round(image.shape[1] * scale)), 1)
    height = max(int(round(image.shape[0] * scale)), 1)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _normalize_for_heatmap(image: np.ndarray) -> Tuple[np.ndarray, float]:
    field = np.asarray(image, dtype=np.float32)
    valid = field[np.isfinite(field)]
    scale = float(np.median(valid) / NORMALIZED_BASE_LEVEL) if valid.size else 0.0
    if scale <= 1e-6:
        return np.zeros_like(image), scale
    return np.clip(field / scale, 0.0, 1.0), scale


def _crop_border(image: np.ndarray, fraction: float) -> np.ndarray:
    crop_y = int(round(image.shape[0] * fraction))
    crop_x = int(round(image.shape[1] * fraction))
    if crop_y <= 0 and crop_x <= 0:
        return image
    return image[crop_y : image.shape[0] - crop_y, crop_x : image.shape[1] - crop_x]


def _has_border_artifacts(cfg: str) -> bool:
    return cfg.startswith("1") or cfg.endswith("D")


def _color_limits(heatmaps: list[np.ndarray]) -> Tuple[float, float]:
    values = np.concatenate([heatmap[np.isfinite(heatmap)].ravel() for heatmap in heatmaps])
    vmin = float(np.percentile(values, COLOR_SCALE_LOW_PERCENTILE))
    vmax = float(np.percentile(values, COLOR_SCALE_HIGH_PERCENTILE))
    if vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vignetting-dir",
        type=Path,
        default=Path("Vignetting"),
        help="Directory containing vignetting CR2 files and their OFF counterparts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Vignetting/vignetting_grid.png"),
        help="Where to save the vignetting grid plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the vignetting grid instead of only saving it",
    )
    return parser.parse_args()


def _load_frame(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with rawpy.imread(str(path)) as raw:
        measurement = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            gamma=(1, 1),
            output_bps=16,
        )
        preview = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            bright=12.0,
            output_bps=8,
            gamma=(2.2, 4.5),
        )
    measurement_gray = cv2.cvtColor(measurement, cv2.COLOR_RGB2GRAY).astype(np.float32) / 65535.0
    preview_gray = cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.clip(measurement_gray, 0.0, 1.0), np.clip(preview_gray, 0.0, 1.0)


def _warp_roi(
    image: np.ndarray,
    preview: np.ndarray,
    label: str,
    matrix: np.ndarray | None = None,
    size: Tuple[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Warp `image` to the ROI, optionally reusing an existing homography."""
    if matrix is None or size is None:
        corners, _ = detect_inner_corners(preview, label=label, extra_images=[image])
        matrix, size = compute_destination(corners)
    warped = warp_image(image, matrix, size)
    return warped, matrix, size


def _off_path_for_config(vignetting_dir: Path, cfg: str) -> Path | None:
    """Return matching OFF frame path, supporting both old and new naming."""
    for name in (f"OFF_{cfg}.CR2", f"OFF-{cfg}.CR2"):
        candidate = vignetting_dir / name
        if candidate.exists():
            return candidate
    return None


def main():
    args = parse_args()
    vignetting_dir = args.vignetting_dir

    configs = [f"{row_key}{col_key}" for row_key, _ in ROW_LABELS for col_key, _ in COL_LABELS]
    fig, axes = plt.subplots(len(ROW_LABELS), len(COL_LABELS), figsize=(15, 18))
    axes_by_cfg = {}
    titles_by_cfg = {}
    for r, (row_key, row_title) in enumerate(ROW_LABELS):
        for c, (col_key, col_suffix) in enumerate(COL_LABELS):
            cfg = f"{row_key}{col_key}"
            ax = axes[r, c]
            axes_by_cfg[cfg] = ax
            titles_by_cfg[cfg] = f"{row_title} {col_suffix}"
            ax.axis("off")
            ax.set_title(titles_by_cfg[cfg], fontsize=11, pad=8)
            if col_key == "D":
                ax.set_facecolor("#F4F1EA")
    image_artists = []
    heatmaps = []
    colorbar_limits = (0.0, 1.0)

    for cfg in configs:
        frame_path = vignetting_dir / f"{cfg}.CR2"
        off_path = _off_path_for_config(vignetting_dir, cfg)
        ax = axes_by_cfg[cfg]
        if not frame_path.exists() or off_path is None:
            missing_note = "f/4 not available" if cfg == "1D" else "Not captured"
            ax.text(
                0.5,
                0.5,
                missing_note,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#777777",
                fontsize=10,
            )
            print(f"Skipping {cfg}: missing frame or OFF")
            continue
        frame, preview = _load_frame(frame_path)
        off_frame, off_preview = _load_frame(off_path)
        warped, matrix, size = _warp_roi(frame, preview, label=cfg)
        warped_off, _, _ = _warp_roi(
            off_frame,
            off_preview,
            label=f"OFF_{cfg}",
            matrix=matrix,
            size=size,
        )
        diff = np.clip(warped - warped_off, 0.0, None)
        if _has_border_artifacts(cfg):
            diff = _crop_border(diff, BORDER_ARTIFACT_CROP_FRACTION)
        normalized, norm_scale = _normalize_for_heatmap(diff)
        displayed = _display_image(normalized)
        heatmaps.append(normalized)
        image_artists.append(ax.imshow(displayed, cmap="turbo"))
        print(f"Processed {cfg}: min={diff.min():.3f}, max={diff.max():.3f}, norm={norm_scale:.3f}")

    if heatmaps:
        vmin, vmax = _color_limits(heatmaps)
        for artist in image_artists:
            artist.set_clim(0.0, 1.0)
        colorbar_limits = (vmin, vmax)
        print(f"Color scale: {vmin:.3f} to {vmax:.3f} normalized intensity")
    fig.subplots_adjust(right=0.88, top=0.92, wspace=0.05, hspace=0.35)
    fig.suptitle("Vignetting patterns", fontsize=20)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    if image_artists:
        cbar = fig.colorbar(image_artists[0], cax=cbar_ax)
        cbar.ax.set_ylim(*colorbar_limits)
        ticks = np.linspace(*colorbar_limits, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
        cbar.set_label("Normalized intensity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
