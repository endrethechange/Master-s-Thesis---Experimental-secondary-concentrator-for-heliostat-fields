#!/usr/bin/env python3
"""Analyze Lambertian captures at 0°, 10°, 20°, 30°, 40°, 60°, and 70°."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import csv
import cv2
import exifread
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import rawpy

from analyze_manual import detect_inner_corners, compute_destination, warp_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambertian-dir",
        type=Path,
        default=Path("Lambertian"),
        help="Directory containing angle.CR2 files and OFF_* references",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        type=str,
        default=["0", "10", "20", "30", "40", "60", "70"],
        help="Angles (in degrees) to analyze. Provide as strings matching filenames",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("Lambertian/debug"),
        help="Directory for warped preview overlays",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("Lambertian/angles_vs_intensity.png"),
        help="Where to save the Lambertian plot",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("Lambertian/angles_vs_intensity.csv"),
        help="Where to save the per-angle statistics",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )
    return parser.parse_args()


def _read_iso(path: Path) -> float:
    with path.open("rb") as fh:
        tags = exifread.process_file(fh, stop_tag="EXIF ISOSpeedRatings", details=False)
    iso_tag = tags.get("EXIF ISOSpeedRatings")
    return float(getattr(iso_tag, "values", [0])[0])


def _load_frame(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
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
    iso = _read_iso(path)
    return np.clip(measurement_gray, 0.0, 1.0), np.clip(preview_gray, 0.0, 1.0), iso


def _warp_roi(
    image: np.ndarray,
    preview: np.ndarray,
    label: str,
    matrix: np.ndarray | None = None,
    size: Tuple[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Warp `image` to the ROI, using `matrix` if supplied."""
    if matrix is None or size is None:
        corners, _ = detect_inner_corners(preview, label=label, extra_images=[image])
        matrix, size = compute_destination(corners)
    warped = warp_image(image, matrix, size)
    return warped, matrix, size


def main():
    args = parse_args()
    lambert_dir = args.lambertian_dir
    debug_dir = args.debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)

    mean_values: Dict[str, float] = {}
    per_angle_stats: List[Tuple[str, float, float, float, float]] = []

    for angle in args.angles:
        frame_path = lambert_dir / f"{angle}.CR2"
        off_path = lambert_dir / f"OFF_{angle}.CR2"
        if not frame_path.exists() or not off_path.exists():
            print(f"Skipping {angle}: missing frame or OFF.")
            continue

        frame, preview, iso = _load_frame(frame_path)
        off_frame, off_preview, off_iso = _load_frame(off_path)

        warped_frame, matrix, size = _warp_roi(frame, preview, label=angle)
        warped_off, _, _ = _warp_roi(off_frame, off_preview, label=f"OFF_{angle}", matrix=matrix, size=size)

        diff = np.clip(
            (warped_frame / max(iso, 1.0)) - (warped_off / max(off_iso, 1.0)),
            0.0,
            None,
        )
        window = diff[diff.mean() - 2 * diff.std() <= diff]
        roi_mean = float(diff.mean())
        roi_std = float(diff.std())
        window_mean = float(window.mean()) if window.size else roi_mean

        mean_values[angle] = roi_mean
        per_angle_stats.append((angle, roi_mean, window_mean, iso, off_iso))

        diff_image = np.clip(diff / (diff.max() if diff.max() > 0 else 1.0), 0.0, 1.0)
        cv2.imwrite(
            str(debug_dir / f"{angle}_diff.png"),
            np.clip(diff_image * 255.0, 0, 255).astype(np.uint8),
        )
        print(
            f"Angle {angle}: mean={roi_mean:.6f}, "
            f"mean+-std={roi_mean - roi_std:.6f}/{roi_mean + roi_std:.6f}, "
            f"window_mean={window_mean:.6f}, ISO={iso:.0f}"
        )

    if not mean_values:
        print("No frames processed.")
        return

    sorted_angles = sorted(mean_values.keys(), key=lambda a: float(a))
    intensities = [mean_values[a] for a in sorted_angles]
    # Use the 0° capture as the Lambertian reference if available, otherwise the max intensity.
    if "0" in mean_values:
        reference_angle = "0"
    else:
        reference_angle = max(mean_values, key=mean_values.get)
    reference_intensity = mean_values[reference_angle]
    normalized_intensities = [
        (value / reference_intensity) if reference_intensity > 0 else 0.0 for value in intensities
    ]
    drops = [100.0 * (1 - val) for val in normalized_intensities]
    max_drop = max(drops)
    min_drop = min(drops)
    print(
        f"Lambertian reference angle {reference_angle}° intensity = {reference_intensity:.6f}. "
        f"Max deviation = {max_drop:.2f}%, min deviation = {min_drop:.2f}%."
    )

    text_color = "#2D2D2D"
    point_color = "#4A90E2"
    grey_line = "#6AB187"
    angles_deg = np.array([float(a) for a in sorted_angles], dtype=np.float32)

    fig, (ax_main, ax_dev) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax_main.scatter(angles_deg, intensities, color=point_color)
    ax_main.set_ylabel("Mean intensity", color=text_color)
    ax_main.set_title("Lambertian test", color=text_color)
    ax_main.yaxis.grid(True, color="#CCCCCC", alpha=0.5)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["left"].set_visible(False)
    ax_main.spines["bottom"].set_color(text_color)
    ax_main.tick_params(colors=text_color)
    for spine in ax_main.spines.values():
        spine.set_color(text_color)

    deviation_percent = [(norm - 1.0) * 100.0 for norm in normalized_intensities]
    ax_dev.axhline(0.0, color=grey_line, linestyle="--", linewidth=1)
    ax_dev.plot(angles_deg, deviation_percent, marker="o", color=point_color, linewidth=1.5)
    ax_dev.set_xlabel("Angle (degrees)", color=text_color)
    ax_dev.set_ylabel("Deviation (%)", color=text_color)
    ax_dev.yaxis.grid(True, color="#CCCCCC", alpha=0.5)
    ax_dev.spines["right"].set_visible(False)
    ax_dev.spines["top"].set_visible(False)
    ax_dev.spines["left"].set_visible(False)
    ax_dev.spines["bottom"].set_color(text_color)
    ax_dev.tick_params(colors=text_color)
    ax_dev.yaxis.set_major_formatter(ticker.PercentFormatter())
    for spine in ax_dev.spines.values():
        spine.set_color(text_color)

    fig.tight_layout()
    args.plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_path, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["angle", "mean_intensity", "window_mean", "ISO", "OFF_ISO", "normalized_intensity"])
        normalization_lookup = {
            angle: (mean / reference_intensity) if reference_intensity > 0 else 0.0
            for angle, mean in mean_values.items()
        }
        for angle, mean, window_mean, iso, off_iso in per_angle_stats:
            writer.writerow(
                [
                    angle,
                    f"{mean:.6f}",
                    f"{window_mean:.6f}",
                    f"{iso:.0f}",
                    f"{off_iso:.0f}",
                    f"{normalization_lookup[angle]:.4f}",
                ]
            )


if __name__ == "__main__":
    main()
