#!/usr/bin/env python3
"""
Shared helper for analyzing partially automated exposure series.

Each run:
- loads every CR2 and its metadata (aperture, shutter time, ISO),
- detects ArUco markers so the region of interest (ROI) can be rectified,
- subtracts the OFF frame inside that ROI,
- normalizes the measured signal with I = P * F^2 / (E * ISO),
- and plots the resulting brightness plus residuals.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import exifread
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import rawpy

from analyze_manual import detect_inner_corners, compute_destination, warp_image


def _to_gray(image: np.ndarray, denom: float) -> np.ndarray:
    """Shared RGB->grayscale helper that also normalizes from [0, 255] to [0,1]."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / denom
    return np.clip(gray, 0.0, 1.0)


def _read_rational(value, default: float) -> float:
    if hasattr(value, "num") and hasattr(value, "den") and value.den:
        return float(value.num) / float(value.den)
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        return float(value[0]) / float(value[1])
    return default


def _read_exif_metadata(path: Path) -> Tuple[float, float, float]:
    """Return (aperture, shutter, iso) using exifread so we support older rawpy builds."""
    with path.open("rb") as fh:
        tags = exifread.process_file(fh, stop_tag="ISOSpeedRatings", details=False)
    aperture = _read_rational(
        getattr(tags.get("EXIF FNumber"), "values", [(0, 1)])[0],
        default=0.0,
    )
    shutter = _read_rational(
        getattr(tags.get("EXIF ExposureTime"), "values", [(0, 1)])[0],
        default=0.0,
    )
    iso_value = tags.get("EXIF ISOSpeedRatings")
    iso = float(getattr(iso_value, "values", [0])[0])
    return aperture, shutter, iso


def _load_frame(path: Path) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Return (measurement_gray, preview_gray, aperture, shutter, ISO) for a CR2 frame.
    """
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
    aperture, shutter, iso = _read_exif_metadata(path)
    measurement = _to_gray(measurement_rgb, 65535.0)
    preview = _to_gray(preview_rgb, 255.0)
    return measurement, preview, aperture, shutter, iso


def _positive(value: float, eps: float = 1e-9) -> float:
    return value if value > eps else eps


def plot_diff_means_auto(
    stats: Sequence[Tuple[int, float]],
    output_path: Path,
    error_output_path: Path,
    show_plot: bool,
    frame_count: int,
) -> None:
    if not stats:
        print("No diff_mean values to plot.")
        return
    indices = np.array([idx for idx, _ in stats], dtype=np.float32)
    values = np.array([val for _, val in stats], dtype=np.float32)
    percent_step = 100.0 / max(frame_count, 1)
    percentages = indices * percent_step
    slope, intercept = np.polyfit(indices, values, 1)
    fit = slope * indices + intercept
    residuals = values - fit
    denom = np.maximum(np.abs(fit), 1e-9)
    residual_percent = (residuals / denom) * 100.0  # Normalize by fit so low inputs stay realistic.

    text_color = "#2D2D2D"
    point_color = "#4A90E2"
    fit_color = "#6AB187"
    # Set global font size for all elements
    plt.rcParams.update({'font.size': 14}) 

    fig, (ax_lin, ax_res) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    ax_lin.scatter(percentages, values, color=point_color)
    ax_lin.plot(percentages, fit, color=fit_color, linewidth=2)
    ax_lin.set_ylabel("Mean intensity", color=text_color,  fontsize=16)
    ax_lin.set_title("Sensor Linearity, fixed f-number", color=text_color, fontsize=20)
    ax_lin.yaxis.grid(True, color="#CCCCCC", alpha=0.5)
    ax_lin.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax_lin.spines["right"].set_visible(False)
    ax_lin.spines["top"].set_visible(False)
    ax_lin.spines["left"].set_visible(False)
    ax_lin.spines["bottom"].set_color(text_color)

    ax_res.axhline(0.0, color=fit_color, linestyle="--", linewidth=1)
    ax_res.plot(percentages, residual_percent, marker="o", color=point_color, linewidth=1.5)
    ax_res.set_xlabel("Input level", color=text_color,  fontsize=16)
    ax_res.set_ylabel("Residual", color=text_color,  fontsize=16)
    ax_res.set_title("Linearity residuals", color=text_color, fontsize=20)
    ax_res.yaxis.grid(True, color="#CCCCCC", alpha=0.5)
    ax_res.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax_res.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax_res.spines["right"].set_visible(False)
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["left"].set_visible(False)
    ax_res.spines["bottom"].set_color(text_color)
    for ax in (ax_lin, ax_res):
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close(fig)

    res_fig, res_ax = plt.subplots(figsize=(6, 4))
    res_ax.axhline(0.0, color=fit_color, linestyle="--", linewidth=1)
    res_ax.plot(percentages, residual_percent, marker="o", color=point_color, linewidth=1.5)
    res_ax.set_xlabel("Input level (%)", color=text_color)
    res_ax.set_ylabel("Residual (%)", color=text_color)
    res_ax.yaxis.grid(True, color="#CCCCCC", alpha=0.5)
    res_ax.tick_params(colors=text_color)
    for spine in res_ax.spines.values():
        spine.set_color(text_color)
    error_output_path.parent.mkdir(parents=True, exist_ok=True)
    res_fig.tight_layout()
    res_fig.savefig(error_output_path, dpi=200)
    plt.close(res_fig)


def run_variable_analysis(
    default_dir: Path,
    default_plot: Path,
    default_error_plot: Path,
    label: str,
    skip_frames: Sequence[int] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description=f"Normalize and analyze the {label} capture series."
    )
    parser.add_argument(
        "--manual-dir",
        type=Path,
        default=default_dir,
        help="Directory with OFF.CR2 and numbered captures",
    )
    parser.add_argument(
        "--off-name",
        type=str,
        default="OFF.CR2",
        help="Filename of the OFF/background frame",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=15,
        help="Number of numbered frames (1..N) to inspect",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=default_plot,
        help="Where to save the diff_mean vs fit plot",
    )
    parser.add_argument(
        "--error-plot-path",
        type=Path,
        default=default_error_plot,
        help="Where to save the fit residual plot",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plots after saving",
    )
    args = parser.parse_args()

    manual_dir = args.manual_dir
    off_path = manual_dir / args.off_name
    if not off_path.exists():
        raise FileNotFoundError(f"Missing OFF frame: {off_path}")

    off_measurement, _, off_aperture, off_shutter, off_iso = _load_frame(off_path)
    off_correction = (_positive(off_aperture) ** 2) / (
        _positive(off_shutter) * _positive(off_iso)
    )

    diff_stats_map: Dict[int, float] = {}
    last_alignment: Tuple[np.ndarray, Tuple[int, int]] | None = None
    last_alignment_source: int | None = None
    frame_indices = list(range(1, args.frame_count + 1))
    # Traverse from brightest to darkest so fallback homographies stay valid.
    process_order = list(reversed(frame_indices))
    for idx in process_order:
        frame_path = manual_dir / f"{idx}.CR2"
        if not frame_path.exists():
            print(f"Skipping missing frame {frame_path}")
            continue
        measurement, preview, aperture, shutter, iso = _load_frame(frame_path)
        matrix: np.ndarray | None = None
        size: Tuple[int, int] | None = None
        try:
            inner_corners, _ = detect_inner_corners(
                preview,
                label=f"{idx:02d}",
                extra_images=[measurement],
            )
            matrix, size = compute_destination(inner_corners)
            last_alignment = (matrix.copy(), size)
            last_alignment_source = idx
        except RuntimeError as exc:
            if last_alignment is None or last_alignment_source is None:
                print(f"Frame {idx:02d}: {exc}")
                continue
            matrix, size = last_alignment
            # Keep processing by falling back to the most recent successful alignment.
            print(
                f"Frame {idx:02d}: {exc}; reusing alignment from frame {last_alignment_source:02d}"
            )
        assert matrix is not None and size is not None
        warped_frame = warp_image(measurement, matrix, size)
        warped_off = warp_image(off_measurement, matrix, size)

        frame_correction = (_positive(aperture) ** 2) / (
            _positive(shutter) * _positive(iso)
        )
        corrected_frame = warped_frame * frame_correction
        corrected_off = warped_off * off_correction
        diff = np.clip(corrected_frame - corrected_off, 0.0, None)

        # Normalize according to I = P * F^2 / (E * ISO).
        diff_mean = float(diff.mean())
        diff_stats_map[idx] = diff_mean

        print(
            f"Frame {idx:02d}: diff_mean={diff_mean:.6f} "
            f"(input {idx * (100.0 / args.frame_count):.2f}%, "
            f"F={aperture:.2f}, E={shutter:.5f}s, ISO={iso:.0f})"
        )

    diff_stats = sorted(diff_stats_map.items(), key=lambda entry: entry[0])
    skip_set = set(skip_frames or [])
    if skip_set:
        # Drop troublesome frames (e.g., first capture) from plotting so axes stay readable.
        diff_stats = [item for item in diff_stats if item[0] not in skip_set]

    plot_diff_means_auto(
        diff_stats,
        args.plot_path,
        args.error_plot_path,
        args.show_plot,
        args.frame_count,
    )
    print(f"Saved diff_mean plot to {args.plot_path}")
    print(f"Saved residual plot to {args.error_plot_path}")
