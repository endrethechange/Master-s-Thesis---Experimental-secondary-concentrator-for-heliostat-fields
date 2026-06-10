#!/usr/bin/env python3
"""Analyze ISO vs brightness inside ArUco marker region for CR2 files."""

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rawpy
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "rawpy is required. Install with: pip install rawpy"
    ) from exc

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "opencv-contrib-python is required. Install with: pip install opencv-contrib-python"
    ) from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


ISO_RE = re.compile(r"^(\d+)(?:\.CR2)?$", re.IGNORECASE)

COMMON_DICTS = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_7X7_50",
    "DICT_7X7_100",
]


def parse_iso_from_name(path: Path) -> Optional[int]:
    match = ISO_RE.match(path.stem)
    if not match:
        return None
    return int(match.group(1))


def load_linear_raw(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    with rawpy.imread(str(path)) as raw:
        raw_image = raw.raw_image_visible.astype(np.float32)
        try:
            black = float(np.mean(raw.black_level_per_channel))
        except Exception:
            black = float(raw.black_level) if hasattr(raw, "black_level") else 0.0
        white = float(raw.white_level) if hasattr(raw, "white_level") else np.max(raw_image)
        denom = max(1.0, white - black)
        linear = (raw_image - black) / denom
        linear = np.clip(linear, 0.0, 1.0)

        detection = None
        try:
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                gamma=(1.0, 1.0),
                output_bps=16,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            )
            gray16 = cv2.cvtColor(rgb16, cv2.COLOR_RGB2GRAY)
            detection = gray16.astype(np.float32) / 65535.0
        except Exception:
            detection = None

        return linear, detection


def prepare_detection_images(bases: list[np.ndarray], max_variants: int) -> list[np.ndarray]:
    # Build a set of variants that improve visibility in dark frames.
    # These are ONLY for detection, not for measurement.
    def to_uint8(img: np.ndarray) -> np.ndarray:
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

    variants = []

    for base in bases:
        base = np.clip(base, 0.0, 1.0)
        base8 = to_uint8(base)
        variants.append(base8)
        variants.append(clahe.apply(base8))
        variants.append(cv2.equalizeHist(base8))
        variants.append(cv2.normalize(base8, None, 0, 255, cv2.NORM_MINMAX))

        # Percentile normalization to stretch dark images
        p1, p99 = np.percentile(base, [1, 99.5])
        denom = max(1e-6, p99 - p1)
        norm = np.clip((base - p1) / denom, 0.0, 1.0)
        norm8 = to_uint8(norm)
        variants.append(norm8)
        variants.append(clahe.apply(norm8))

        # Gain-based variants
        for gain in [2.0, 4.0, 8.0, 16.0, 32.0]:
            g = np.clip(base * gain, 0.0, 1.0)
            g8 = to_uint8(g)
            variants.append(g8)
            variants.append(clahe.apply(g8))

        # Gamma variants (brighten shadows)
        for gamma in [0.8, 0.6, 0.5, 0.4, 0.3]:
            g = np.clip(np.power(base, gamma), 0.0, 1.0)
            g8 = to_uint8(g)
            variants.append(g8)
            variants.append(clahe.apply(g8))

        # Adaptive threshold variants for extremely dark frames
        for k in [11, 15, 21, 31]:
            if k % 2 == 0:
                continue
            th = cv2.adaptiveThreshold(base8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, k, 5)
            variants.append(th)

    # Remove exact duplicate variants and cap total count to keep runtime bounded.
    deduped = []
    seen = set()
    for img in variants:
        key = hash(img.tobytes())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(img)
        if len(deduped) >= max_variants:
            break

    return deduped


def get_aruco_dict(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_markers(gray: np.ndarray, dictionary) -> tuple[list[np.ndarray], Optional[np.ndarray]]:
    parameters = cv2.aruco.DetectorParameters()
    # More permissive settings to help detect dark/low-contrast markers.
    parameters.adaptiveThreshConstant = 7
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.minOtsuStdDev = 1.0
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def find_markers_with_fallback(gray_variants: list[np.ndarray], dict_names: list[str]):
    best = None
    for dict_name in dict_names:
        dictionary = get_aruco_dict(dict_name)
        for gray in gray_variants:
            corners, ids = detect_markers(gray, dictionary)
            if ids is not None and len(ids) >= 4:
                return corners, ids, dict_name
            if ids is not None and best is None:
                best = (corners, ids, dict_name)
    if best is not None:
        return best
    return [], None, None


def inner_corners_from_markers(corners: list[np.ndarray]) -> np.ndarray:
    all_corners = np.concatenate(corners, axis=0)
    center = np.mean(all_corners.reshape(-1, 2), axis=0)
    inner = []
    for marker in corners:
        marker = marker.reshape(-1, 2)
        dists = np.linalg.norm(marker - center, axis=1)
        inner.append(marker[np.argmin(dists)])
    return np.array(inner, dtype=np.float32)


def extract_inner_corner_by_id(
    corners: list[np.ndarray],
    ids: np.ndarray,
    target_id: int,
    board_center: np.ndarray,
) -> Optional[np.ndarray]:
    idx = np.where(ids.flatten() == target_id)[0]
    if idx.size == 0:
        return None
    marker = corners[int(idx[0])].reshape(-1, 2)
    # Inner corner is the one closest to the board center.
    dists = np.linalg.norm(marker - board_center, axis=1)
    return marker[np.argmin(dists)]


def quad_from_marker_ids(corners: list[np.ndarray], ids: np.ndarray, mapping: dict[str, int]) -> Optional[np.ndarray]:
    marker_centers = np.array([np.mean(c.reshape(-1, 2), axis=0) for c in corners], dtype=np.float32)
    board_center = np.mean(marker_centers, axis=0)

    tl = extract_inner_corner_by_id(corners, ids, mapping["tl"], board_center)
    tr = extract_inner_corner_by_id(corners, ids, mapping["tr"], board_center)
    bl = extract_inner_corner_by_id(corners, ids, mapping["bl"], board_center)
    br = extract_inner_corner_by_id(corners, ids, mapping["br"], board_center)
    if any(v is None for v in (tl, tr, bl, br)):
        return None
    return np.array([tl, tr, br, bl], dtype=np.float32)


def order_quad(points: np.ndarray) -> np.ndarray:
    if len(points) < 4:
        raise ValueError("Need at least 4 points to order a quad")
    # Use convex hull to reduce to outer 4 points if more are provided.
    hull = cv2.convexHull(points.astype(np.float32))
    hull = hull.reshape(-1, 2)
    if len(hull) > 4:
        # Approximate hull to 4 points
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
        if len(approx) >= 4:
            hull = approx
    if len(hull) != 4:
        # Fallback: select 4 extreme points by angle around centroid
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        order = np.argsort(angles)
        hull = points[order][:4]
    # Order: top-left, top-right, bottom-right, bottom-left
    s = hull.sum(axis=1)
    diff = np.diff(hull, axis=1).ravel()
    tl = hull[np.argmin(s)]
    br = hull[np.argmax(s)]
    tr = hull[np.argmin(diff)]
    bl = hull[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def create_mask(shape: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    return mask


def analyze_image(
    path: Path,
    dict_names: list[str],
    marker_id_map: dict[str, int],
    debug_dir: Optional[Path],
    max_detection_variants: int,
):
    linear, detection = load_linear_raw(path)
    bases = [linear]
    if detection is not None:
        bases.insert(0, detection)
    gray_variants = prepare_detection_images(bases, max_detection_variants)

    corners, ids, dict_name = find_markers_with_fallback(gray_variants, dict_names)
    if ids is None or len(ids) < 4:
        return None

    quad = quad_from_marker_ids(corners, ids, marker_id_map)
    if quad is None:
        inner = inner_corners_from_markers(corners)
        quad = order_quad(inner)
    mask = create_mask(linear.shape, quad)

    roi_vals = linear[mask == 255]
    mean = float(np.mean(roi_vals))
    std = float(np.std(roi_vals))

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_img = cv2.cvtColor(gray_variants[0], cv2.COLOR_GRAY2BGR)
        cv2.polylines(debug_img, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.aruco.drawDetectedMarkers(debug_img, corners, ids)
        out_path = debug_dir / f"{path.stem}_debug.png"
        cv2.imwrite(str(out_path), debug_img)

    return {
        "iso": parse_iso_from_name(path),
        "mean": mean,
        "std": std,
        "n_pixels": int(roi_vals.size),
        "dict": dict_name,
    }


def plot_results(rows: list[dict], output_path: Path, show_plot: bool = False):
    iso = np.array([r["iso"] for r in rows], dtype=float)
    mean = np.array([r["mean"] for r in rows], dtype=float)
    input_level = iso / np.max(iso) * 100.0

    coeffs = np.polyfit(iso, mean, 1)
    fit = np.polyval(coeffs, iso)
    residuals = mean - fit
    denom = np.maximum(np.abs(fit), 1e-9)
    residuals_pct = residuals / denom * 100.0

    text_color = "#2D2D2D"
    point_color = "#4A90E2"
    fit_color = "#6AB187"

    with plt.rc_context({"font.size": 14}):
        fig, (ax_lin, ax_res) = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

        ax_lin.scatter(input_level, mean, color=point_color)
        ax_lin.plot(input_level, fit, color=fit_color, linewidth=2)
        ax_lin.set_ylabel("Mean intensity", color=text_color, fontsize=16)
        ax_lin.set_title("Sensor Linearity, variable ISO", color=text_color, fontsize=20)
        ax_lin.yaxis.grid(True, color="#CCCCCC", alpha=0.5)
        ax_lin.xaxis.set_major_formatter(PercentFormatter())

        ax_res.axhline(0.0, color=fit_color, linestyle="--", linewidth=1)
        ax_res.plot(input_level, residuals_pct, marker="o", color=point_color, linewidth=1.5)
        ax_res.set_xlabel("Input level", color=text_color, fontsize=16)
        ax_res.set_ylabel("Residual", color=text_color, fontsize=16)
        ax_res.set_title("Linearity residuals", color=text_color, fontsize=20)
        ax_res.yaxis.grid(True, color="#CCCCCC", alpha=0.5)
        ax_res.xaxis.set_major_formatter(PercentFormatter())
        ax_res.yaxis.set_major_formatter(PercentFormatter())

        for ax in (ax_lin, ax_res):
            ax.tick_params(colors=text_color)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color(text_color)
            for spine in ax.spines.values():
                spine.set_color(text_color)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        if show_plot:
            plt.show()
        plt.close(fig)


def write_csv(rows: list[dict], output_path: Path):
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iso", "mean", "std", "n_pixels", "dict"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze ISO vs brightness inside ArUco region.")
    parser.add_argument(
        "--input-dir",
        default="ISO Camera Test",
        help="Folder containing CR2 files (default: ISO Camera Test)",
    )
    parser.add_argument(
        "--output-dir",
        default="iso_analysis_output",
        help="Output folder for CSV and plots",
    )
    parser.add_argument(
        "--aruco-dict",
        default="DICT_4X4_50",
        help="ArUco dictionary name (OpenCV)",
    )
    parser.add_argument(
        "--try-common-dicts",
        action="store_true",
        help="Try common dictionaries if detection fails",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write debug images with detected markers and ROI",
    )
    parser.add_argument(
        "--max-detection-variants",
        type=int,
        default=24,
        help="Max number of grayscale variants tested for marker detection per image (default: 24)",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot interactively after saving it",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dict_names = [args.aruco_dict]
    if args.try_common_dicts:
        dict_names = [args.aruco_dict] + [d for d in COMMON_DICTS if d != args.aruco_dict]

    marker_id_map = {"tl": 1, "tr": 2, "bl": 3, "br": 4}

    rows = []
    candidates = []
    for path in input_dir.glob("*.CR2"):
        iso = parse_iso_from_name(path)
        if iso is None:
            continue
        candidates.append((iso, path))

    if not candidates:
        raise SystemExit(f"No ISO-named .CR2 files found in: {input_dir}")

    start = time.time()
    print(
        f"Analyzing {len(candidates)} files with {len(dict_names)} dict(s) and "
        f"up to {args.max_detection_variants} detection variants per image..."
    )

    for idx, (iso, path) in enumerate(sorted(candidates, key=lambda x: x[0]), start=1):
        print(f"[{idx}/{len(candidates)}] ISO {iso}: {path.name}")
        result = analyze_image(
            path,
            dict_names,
            marker_id_map,
            output_dir / "debug" if args.debug else None,
            args.max_detection_variants,
        )
        if result is None:
            print(f"Warning: could not detect markers in {path.name}")
            continue
        rows.append(result)
        print(f"  mean={result['mean']:.6f}, std={result['std']:.6f}, dict={result['dict']}")

    rows.sort(key=lambda r: r["iso"])
    if not rows:
        raise SystemExit("No images analyzed. Check ArUco detection or input folder.")

    csv_path = output_dir / "iso_linearity.csv"
    plot_path = output_dir / "iso_linearity.png"

    write_csv(rows, csv_path)
    plot_results(rows, plot_path, args.show_plot)

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Done in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
