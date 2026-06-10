#!/usr/bin/env python3
"""
Analyze testrig captures by aligning each mirror exposure, subtracting the OFF frame,
and reporting beam intensities plus σ-based diameters for the single- and three-mirror tests.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy

from analyze_manual import detect_inner_corners, compute_destination, warp_image

# Cache tuple describing a previously detected alignment:
# (homography matrix, warped size, inner corners, marker polygons, label).
AlignmentCache = Tuple[np.ndarray, Tuple[int, int], np.ndarray, List[np.ndarray], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testrig-dir", type=Path, default=Path("Testrig"))
    parser.add_argument("--output-dir", type=Path, default=Path("Testrig/plots"))
    parser.add_argument(
        "--configurations",
        nargs="+",
        default=["1", "3"],
        help="Mirror configurations to inspect (subset of 1 and 3).",
    )
    parser.add_argument("--show", action="store_true", help="Display each plot interactively")
    return parser.parse_args()


def _to_gray(image: np.ndarray, denom: float) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / denom
    return np.clip(gray, 0.0, 1.0)


def load_frame(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with rawpy.imread(str(path)) as raw:
        measurement_rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
        )
        preview_rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            bright=12.0,
            output_bps=8,
            gamma=(2.2, 4.5),
        )
    return _to_gray(measurement_rgb, 65535.0), _to_gray(preview_rgb, 255.0)


def _extra_detection_images(preview: np.ndarray, measurement: np.ndarray | None = None) -> List[np.ndarray]:
    """Return a robust list of boosted views to help ArUco detection."""

    def normalize(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        arr -= arr.min()
        span = arr.max()
        if span <= 1e-6:
            span = 1.0
        return np.clip(arr / span, 0.0, 1.0)

    def boost_views(src: np.ndarray) -> List[np.ndarray]:
        norm = normalize(src)
        base_u8 = (norm * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(base_u8)
        eq = cv2.equalizeHist(base_u8)
        log_variant = np.log1p(norm * 12.0)
        log_variant = (log_variant - log_variant.min()) / max(log_variant.max() - log_variant.min(), 1e-6)
        stretched = cv2.normalize(norm, None, 0.0, 1.0, cv2.NORM_MINMAX)
        boosted = [
            norm,
            np.clip(norm * 4.0, 0.0, 1.0),
            np.clip(norm * 8.0, 0.0, 1.0),
            np.clip(norm * 12.0, 0.0, 1.0),
            np.power(norm + 1e-6, 0.4),
            np.power(norm + 1e-6, 0.3),
            stretched,
            log_variant,
            clahe.astype(np.float32) / 255.0,
            eq.astype(np.float32) / 255.0,
        ]
        blur = cv2.GaussianBlur(base_u8, (5, 5), 0)
        boosted.append(blur.astype(np.float32) / 255.0)
        boosted.append(1.0 - boosted[-1])  # inverted blur
        boosted.append(1.0 - stretched)
        return boosted

    images: List[np.ndarray] = []
    for source in (measurement, preview):
        if source is not None:
            images.extend(boost_views(source))
    return images


def compute_bounding_matrix(
    preview: np.ndarray,
    fallback: np.ndarray | None = None,
    label: str = "testrig",
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, List[np.ndarray]]:
    """
    Return the board homography plus marker geometry for a noisy preview.

    The `fallback` measurement supplies extra low-noise views so ArUco detection
    succeeds even when the preview is nearly black. Once markers are found we
    envelope every marker polygon to build the rectangle we later unwarp.
    """
    extra_images = _extra_detection_images(preview, fallback)
    inner_corners, marker_polys = detect_inner_corners(
        preview,
        label=label,
        extra_images=extra_images,
        use_center_roi=False,
    )
    all_pts = np.concatenate(marker_polys, axis=0)
    min_x = np.min(all_pts[:, 0])
    max_x = np.max(all_pts[:, 0])
    min_y = np.min(all_pts[:, 1])
    max_y = np.max(all_pts[:, 1])
    bounding = np.array(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ],
        dtype=np.float32,
    )
    matrix, size = compute_destination(bounding)
    return matrix, size, inner_corners, marker_polys


def compute_beam_sigmas_mm(image: np.ndarray, scale_x: float, scale_y: float) -> Tuple[float, float]:
    """Return σ_x/σ_y beam widths (mm) using the second-moment definition."""
    peak = image.max()
    if peak <= 0:
        return 0.0, 0.0
    # Mask pixels within the standard D4σ support (>= peak / e^2) so dark noise is ignored.
    mask = image >= peak / np.e**2
    if not np.any(mask):
        mask = image > 0
    weighted = image * mask
    total = weighted.sum()
    if total <= 0:
        return 0.0, 0.0
    y, x = np.indices(image.shape)
    x_mean = (x * weighted).sum() / total
    y_mean = (y * weighted).sum() / total
    x_var = (((x - x_mean) ** 2) * weighted).sum() / total
    y_var = (((y - y_mean) ** 2) * weighted).sum() / total
    sigma_x = np.sqrt(max(x_var, 0.0)) * scale_x
    sigma_y = np.sqrt(max(y_var, 0.0)) * scale_y
    return float(sigma_x), float(sigma_y)


def process_configuration(
    mirror_id: str,
    cfg: str,
    cfg_path: Path,
    off_path: Path,
    output_dir: Path,
    show: bool,
    last_alignment: AlignmentCache | None,
) -> Tuple[Dict[str, float], AlignmentCache | None]:
    """
    Warp and analyze a mirror configuration. If marker detection fails we reuse
    the most recent successful homography so dark captures still produce stats.
    """
    measurement, preview = load_frame(cfg_path)
    off_measurement, _ = load_frame(off_path)
    alignment_label = f"mirror {mirror_id} cfg {cfg}"

    try:
        matrix, size, inner_corners, marker_polys = compute_bounding_matrix(
            preview,
            fallback=measurement,
            label=alignment_label,
        )
        alignment_cache: AlignmentCache = (
            matrix.copy(),
            size,
            inner_corners.copy(),
            [poly.copy() for poly in marker_polys],
            alignment_label,
        )
    except RuntimeError as exc:
        if last_alignment is None:
            raise
        matrix, size, inner_corners, marker_polys, cache_label = last_alignment
        alignment_cache = last_alignment
        print(f"{alignment_label}: {exc}; reusing alignment from {cache_label}")
    warped_frame = warp_image(measurement, matrix, size)
    warped_off = warp_image(off_measurement, matrix, size)
    diff = np.clip(warped_frame - warped_off, 0.0, None)

    # Warp the marker corners so we crop the diff image to the calibrated ROI.
    warped_inner = cv2.perspectiveTransform(inner_corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    x_min = max(int(np.floor(warped_inner[:, 0].min())), 0)
    x_max = min(int(np.ceil(warped_inner[:, 0].max())), diff.shape[1])
    y_min = max(int(np.floor(warped_inner[:, 1].min())), 0)
    y_max = min(int(np.ceil(warped_inner[:, 1].max())), diff.shape[0])
    roi = diff[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        roi = diff

    widths = []
    heights = []
    for poly in marker_polys:
        widths.append(np.linalg.norm(poly[1] - poly[0]))
        widths.append(np.linalg.norm(poly[2] - poly[3]))
        heights.append(np.linalg.norm(poly[2] - poly[1]))
        heights.append(np.linalg.norm(poly[3] - poly[0]))
    avg_width = max(float(np.mean(widths)), 1e-6)
    avg_height = max(float(np.mean(heights)), 1e-6)
    # Each ArUco marker is 5 mm per side, giving our pixel-to-mm scale factors.
    scale_x_mm = 5.0 / avg_width
    scale_y_mm = 5.0 / avg_height

    sigma_x, sigma_y = compute_beam_sigmas_mm(roi, scale_x_mm, scale_y_mm)
    mean_intensity = float(roi.mean())
    total_intensity = float(roi.sum())
    peak_intensity = float(roi.max())

    output_dir.mkdir(parents=True, exist_ok=True)
    width_mm = size[0] * scale_x_mm
    height_mm = size[1] * scale_y_mm
    extent = [0, width_mm, height_mm, 0]
    normalized = roi / peak_intensity if peak_intensity > 0 else roi
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(normalized, cmap="turbo", extent=extent, aspect="equal")
    ax.set_title(f"Mirror {mirror_id} ({cfg}-mirror test)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized intensity")
    ax.text(
        0.02,
        0.02,
        f"σ_x={sigma_x:.2f} mm\nσ_y={sigma_y:.2f} mm\nPeak={peak_intensity:.3e}",
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.4, boxstyle="round"),
    )
    plot_path = output_dir / f"mirror_{mirror_id}_cfg_{cfg}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)

    return {
        "mirror": mirror_id,
        "configuration": cfg,
        "mean_intensity": mean_intensity,
        "peak_intensity": peak_intensity,
        "total_intensity": total_intensity,
        "sigma_x_mm": sigma_x,
        "sigma_y_mm": sigma_y,
        "plot_path": str(plot_path),
    }, alignment_cache


def main():
    args = parse_args()
    testrig_dir = args.testrig_dir
    cfgs = set(cfg.upper() for cfg in args.configurations)

    file_map: Dict[str, Dict[str, Path]] = defaultdict(dict)
    pattern = re.compile(r"(?P<mirror>\d+)-(?P<suffix>.+)", re.IGNORECASE)
    for path in testrig_dir.glob("*.CR2"):
        match = pattern.match(path.stem)
        if not match:
            continue
        mirror = match.group("mirror")
        suffix = match.group("suffix").upper()
        file_map[mirror][suffix] = path

    last_alignment: AlignmentCache | None = None
    for mirror_id in sorted(file_map.keys(), key=lambda m: int(m)):
        entries = file_map[mirror_id]
        off_path = entries.get("OFF")
        if not off_path:
            print(f"Skipping mirror {mirror_id}: missing OFF.")
            continue
        for cfg in cfgs:
            cfg_path = entries.get(cfg)
            if not cfg_path:
                continue
            try:
                stats, last_alignment = process_configuration(
                    mirror_id,
                    cfg,
                    cfg_path,
                    off_path,
                    args.output_dir,
                    args.show,
                    last_alignment,
                )
                print(
                    f"Mirror {mirror_id} cfg {cfg}: "
                    f"σ_x={stats['sigma_x_mm']:.2f} mm, "
                    f"σ_y={stats['sigma_y_mm']:.2f} mm"
                )
            except RuntimeError as exc:
                print(f"Mirror {mirror_id} cfg {cfg}: {exc}")
    print("Finished processing requested mirror configurations.")


if __name__ == "__main__":
    main()
