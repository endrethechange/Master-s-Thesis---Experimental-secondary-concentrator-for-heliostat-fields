"""
Align the manual linearity captures using five steps:
1) load each CR2 image,
2) detect the four ArUco markers,
3) perspective-transform the region bounded by those markers,
4) crop to the inner rectangle, and
5) plot brightness vs. a linear fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import rawpy

ARUCO_DICT = cv2.aruco.DICT_4X4_1000
MARKER_IDS = [1, 2, 4, 3]  # TL, TR, BR, BL
DETECT_MAX_DIM = 1000  # pixels
CENTER_ROI_MARGIN = 0.02
CENTER_ROI_MAX_DIM = 3200  # pixels
LOW_LIGHT_MEAN_THRESHOLD = 0.14
LOW_LIGHT_RANGE_THRESHOLD = 0.1
DETECTION_TIME_LIMIT = 110.0  # seconds


def _get_detector_parameters() -> "cv2.aruco_DetectorParameters":
    """Return ArUco detector parameters tuned for noisy, low-light images."""
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 85
    parameters.adaptiveThreshWinSizeStep = 8
    parameters.adaptiveThreshConstant = 5
    if hasattr(parameters, "cornerRefinementMethod"):
        parameters.cornerRefinementMethod = getattr(
            cv2.aruco,
            "CORNER_REFINE_SUBPIX",
            getattr(cv2.aruco, "CORNER_REFINE_NONE", 0),
        )
        parameters.cornerRefinementMaxIterations = getattr(parameters, "cornerRefinementMaxIterations", 30)
        parameters.cornerRefinementWinSize = getattr(parameters, "cornerRefinementWinSize", 5)
    parameters.minMarkerPerimeterRate = 0.002
    parameters.maxMarkerPerimeterRate = 6.0
    parameters.perspectiveRemoveIgnoredMarginPerCell = 0.3
    parameters.errorCorrectionRate = 0.8
    parameters.minCornerDistanceRate = 0.02
    parameters.cornerRefinementMinAccuracy = 0.01
    if hasattr(parameters, "detectInvertedMarker"):
        parameters.detectInvertedMarker = True
    if hasattr(parameters, "useAruco3Detection"):
        parameters.useAruco3Detection = True
    return parameters

# Define Command Line Interface (CLI) arguments once
ARG_DEFS = (
    (("--manual-dir",), dict(type=Path, default=Path("Linearity/Manual"), help="Directory containing OFF.CR2 and numbered frames")),
    (("--off-name",), dict(type=str, default="OFF.CR2", help="Filename of the OFF/background frame")),
    (("--frame-count",), dict(type=int, default=15, help="How many numbered frames (1..N) to analyze")),
    (("--plot-path",), dict(type=Path, default=Path("Linearity/diff_mean_fit.png"), help="Output path for the diff_mean linear-fit plot")),
    (("--error-plot-path",), dict(type=Path, default=Path("Linearity/diff_mean_residuals.png"), help="Output path for the diff_mean residual plot")),
    (("--show-plot",), dict(action="store_true", help="Display the diff_mean plot interactively after saving")),
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    for flags, kwargs in ARG_DEFS:
        parser.add_argument(*flags, **kwargs)
    return parser.parse_args()


def _to_gray(image: np.ndarray, denom: float) -> np.ndarray:
    """Shared RGB->grayscale helper that also normalizes from [0, 255] to [0,1]."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / denom
    return np.clip(gray, 0.0, 1.0)


def load_measurement_and_preview(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CR2 frame twice: linear for analysis and bright for ArUco detection."""
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


def _is_low_light_scene(image: np.ndarray) -> bool:
    """Flags exceptionally dark/flat previews."""
    mean_val = float(np.mean(image))
    low, high = np.percentile(image, (1.0, 99.0))
    return mean_val < LOW_LIGHT_MEAN_THRESHOLD or (high - low) < LOW_LIGHT_RANGE_THRESHOLD


def _boost_low_light_variants(image: np.ndarray) -> List[np.ndarray]:
    """Generate aggressively brightened/denoised views for very dark frames."""
    if image.size == 0:
        return []
    img = np.asarray(image, dtype=np.float32)
    img -= img.min()
    max_val = img.max()
    if max_val <= 1e-6:
        return []
    img = (img / max_val * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    if hasattr(cv2, "fastNlMeansDenoising"):
        denoised = cv2.fastNlMeansDenoising(clahe, None, h=7, templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = cv2.medianBlur(clahe, 5)
    boosted = cv2.convertScaleAbs(denoised, alpha=1.9, beta=-25)
    sharpened = cv2.addWeighted(
        boosted,
        1.4,
        cv2.GaussianBlur(boosted, (5, 5), 0),
        -0.4,
        15,
    )
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        33,
        3,
    )
    morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    inverted = cv2.bitwise_not(morph)
    return [boosted, sharpened, morph, inverted]


def _detection_variants(image: np.ndarray, extended: bool = False) -> List[np.ndarray]:
    """Generate contrast views for ArUco detection (fast subset + optional extended set)."""
    image = np.asarray(image, dtype=np.float32)
    image -= image.min()
    peak = image.max()
    if peak <= 1e-6:
        peak = 1.0
    image /= peak

    p_low, p_high = np.percentile(image, (0.5, 99.0))
    scale = max(p_high - p_low, 1e-3)
    stretched = np.clip((image - p_low) / scale, 0.0, 1.0)

    def to_u8(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    variants: List[np.ndarray] = []

    base = to_u8(stretched)
    variants.append(base)
    variants.append(cv2.convertScaleAbs(base, alpha=1.3, beta=-10))

    gamma_img = to_u8(np.power(stretched + 1e-6, 0.6))
    variants.append(gamma_img)

    eq = cv2.equalizeHist(base)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(base)
    gaussian = cv2.GaussianBlur(base, (5, 5), 0)
    median = cv2.medianBlur(base, 5)
    variants.extend([eq, clahe, cv2.bitwise_not(eq), cv2.bitwise_not(clahe), gaussian])

    lap = cv2.convertScaleAbs(cv2.Laplacian(base, cv2.CV_8U, ksize=3), alpha=3.0, beta=128)
    variants.append(lap)

    adaptive = cv2.adaptiveThreshold(
        median,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        41,
        4,
    )
    kernel = np.ones((3, 3), np.uint8)
    variants.append(adaptive)
    variants.append(cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1))

    variants.append(cv2.bitwise_not(base))

    if extended:
        variants.append(cv2.convertScaleAbs(base, alpha=1.6, beta=-25))
        variants.append(to_u8(np.power(stretched + 1e-6, 0.45)))
        sharpened = cv2.addWeighted(base, 1.6, gaussian, -0.6, 0)
        variants.append(sharpened)
        highpass = cv2.normalize(cv2.subtract(base, cv2.GaussianBlur(base, (21, 21), 0)), None, 0, 255, cv2.NORM_MINMAX)
        variants.append(highpass.astype(np.uint8))
        median_strong = cv2.medianBlur(base, 7)
        adaptive_strong = cv2.adaptiveThreshold(
            median_strong,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            4,
        )
        variants.append(adaptive_strong)
        variants.append(cv2.morphologyEx(adaptive_strong, cv2.MORPH_CLOSE, kernel, iterations=2))
        variants.append(cv2.bitwise_not(clahe))
    return variants


def _marker_quad_center(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0)


def _marker_quad_size(points: np.ndarray) -> float:
    edges = [
        np.linalg.norm(points[1] - points[0]),
        np.linalg.norm(points[2] - points[1]),
        np.linalg.norm(points[3] - points[2]),
        np.linalg.norm(points[0] - points[3]),
    ]
    return float(np.mean(edges))


def _assemble_marker_geometry(id_map: Dict[int, np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]] | None:
    if len(id_map) != 4:
        return None

    centers = {marker_id: _marker_quad_center(points) for marker_id, points in id_map.items()}
    tl = centers[MARKER_IDS[0]]
    tr = centers[MARKER_IDS[1]]
    br = centers[MARKER_IDS[2]]
    bl = centers[MARKER_IDS[3]]
    if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
        return None

    roi_center = np.mean(np.stack([tl, tr, br, bl], axis=0), axis=0)
    ordered_polys = [id_map[marker_id].astype(np.float32) for marker_id in MARKER_IDS]
    ordered_points: List[np.ndarray] = []
    for pts in ordered_polys:
        inner_idx = np.argmin(np.linalg.norm(pts - roi_center, axis=1))
        ordered_points.append(pts[inner_idx])
    return np.array(ordered_points, dtype=np.float32), ordered_polys


def _marker_maps_compatible(existing: Dict[int, np.ndarray], candidate: Dict[int, np.ndarray]) -> bool:
    overlap = set(existing).intersection(candidate)
    if not overlap:
        return True
    for marker_id in overlap:
        existing_pts = existing[marker_id]
        candidate_pts = candidate[marker_id]
        mean_size = max((_marker_quad_size(existing_pts) + _marker_quad_size(candidate_pts)) * 0.5, 1.0)
        center_delta = np.linalg.norm(_marker_quad_center(existing_pts) - _marker_quad_center(candidate_pts))
        if center_delta > max(mean_size * 0.8, 20.0):
            return False
    return True


def _merge_marker_maps(marker_maps: Sequence[Dict[int, np.ndarray]]) -> Tuple[np.ndarray, List[np.ndarray]] | None:
    partials = [marker_map for marker_map in marker_maps if marker_map]
    if not partials:
        return None

    for seed in sorted(partials, key=lambda item: len(item), reverse=True):
        merged = {marker_id: points.copy() for marker_id, points in seed.items()}
        direct = _assemble_marker_geometry(merged)
        if direct is not None:
            return direct
        for candidate in partials:
            if candidate is seed or not _marker_maps_compatible(merged, candidate):
                continue
            updated = False
            for marker_id, points in candidate.items():
                if marker_id not in merged:
                    merged[marker_id] = points.copy()
                    updated = True
            if not updated:
                continue
            combined = _assemble_marker_geometry(merged)
            if combined is not None:
                return combined
    return None


def _detect_from_center_roi(
    preview_image: np.ndarray,
    extra_images: Sequence[np.ndarray],
    dictionary: "cv2.aruco_Dictionary",
    parameters: "cv2.aruco_DetectorParameters",
    start_time: float,
) -> Tuple[np.ndarray | None, List[np.ndarray] | None, str | None]:
    """
    Detect the marker board from a full-detail center crop before global downscaling.

    The vignetting/linearity markers sit around the central target, so a center crop keeps
    their detail without the cost of running ArUco over the entire RAW-sized frame.
    """
    h, w = preview_image.shape
    y0 = int(round(h * CENTER_ROI_MARGIN))
    y1 = max(int(round(h * (1.0 - CENTER_ROI_MARGIN))), y0 + 1)
    x0 = int(round(w * CENTER_ROI_MARGIN))
    x1 = max(int(round(w * (1.0 - CENTER_ROI_MARGIN))), x0 + 1)
    sources = list(extra_images) + [preview_image]
    last_error = "center ROI attempts found no markers"
    partial_maps: List[Dict[int, np.ndarray]] = []

    for source in sources:
        crop = source[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        resize_factor = 1.0
        max_dim = max(crop.shape)
        if max_dim > CENTER_ROI_MAX_DIM:
            resize_factor = CENTER_ROI_MAX_DIM / float(max_dim)
            crop = cv2.resize(crop, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        variants = _detection_variants(crop, extended=False)
        if _is_low_light_scene(source):
            variants = _boost_low_light_variants(crop) + variants
        for idx, variant in enumerate(variants[:10]):
            if (time.perf_counter() - start_time) > DETECTION_TIME_LIMIT:
                return None, None, f"timeout after {DETECTION_TIME_LIMIT:.0f}s"
            corners, ids, _ = cv2.aruco.detectMarkers(variant, dictionary, parameters=parameters)
            if ids is None:
                continue
            id_map: Dict[int, np.ndarray] = {}
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                marker_id = int(marker_id)
                if marker_id not in MARKER_IDS:
                    continue
                pts = marker_corners[0] / resize_factor
                pts[:, 0] += x0
                pts[:, 1] += y0
                id_map[marker_id] = pts.astype(np.float32)
            if id_map:
                partial_maps.append({marker_id: pts.copy() for marker_id, pts in id_map.items()})
                merged = _merge_marker_maps(partial_maps)
                if merged is not None:
                    return merged[0], merged[1], None
            if len(id_map) != 4:
                missing = [mid for mid in MARKER_IDS if mid not in id_map]
                last_error = f"center ROI variant={idx}, missing {missing}"
                continue
            assembled = _assemble_marker_geometry(id_map)
            if assembled is not None:
                return assembled[0], assembled[1], None

    merged = _merge_marker_maps(partial_maps)
    if merged is not None:
        return merged[0], merged[1], None
    return None, None, last_error


def detect_inner_corners(
    preview_image: np.ndarray,
    label: str,
    extra_images: Sequence[np.ndarray] | None = None,
    use_center_roi: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Detect the four markers and return their inward corners plus full quads."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = _get_detector_parameters()
    start_time = time.perf_counter()
    dark_scene = _is_low_light_scene(preview_image)

    # Downscale large frames to keep detection time reasonable.
    base_factor = 1.0
    max_dim = max(preview_image.shape)
    if max_dim > DETECT_MAX_DIM:
        base_factor = DETECT_MAX_DIM / float(max_dim)

    def _resize(img: np.ndarray) -> np.ndarray:
        if base_factor == 1.0:
            return img
        return cv2.resize(img, None, fx=base_factor, fy=base_factor, interpolation=cv2.INTER_AREA)

    scaled_preview = _resize(preview_image)
    scaled_extras = [_resize(img) for img in (extra_images or [])]

    scales = [1.0, 0.75]
    if max_dim > 1500:
        scales.append(0.6)
    if max_dim > 2000:
        scales.append(0.45)
    last_error = "no attempts"

    if use_center_roi:
        # First try a full-detail center crop, because the marker board sits around the
        # central target and global downscaling can shrink those markers too aggressively.
        roi_result = _detect_from_center_roi(
            preview_image,
            list(extra_images or []),
            dictionary,
            parameters,
            start_time,
        )
        if roi_result[0] is not None:
            return roi_result[0], roi_result[1]  # type: ignore[misc]
        if roi_result[2] is not None:
            last_error = roi_result[2]

    def _run_detection(
        sources: Sequence[np.ndarray],
        extended: bool,
        predefined_variants: Sequence[np.ndarray] | None = None,
    ) -> Tuple[np.ndarray | None, List[np.ndarray] | None]:
        """Iterate through scale/variant combos until all four IDs are seen."""
        nonlocal last_error
        max_variants = 10 if not extended else 16
        for scale in scales:
            partial_maps: List[Dict[int, np.ndarray]] = []
            if predefined_variants is not None:
                source_variants = list(predefined_variants[:max_variants])
            else:
                source_variants = []
                for source in sources:
                    for variant in _detection_variants(source, extended=extended):
                        source_variants.append(variant)
                        if len(source_variants) >= max_variants:
                            break
                    if len(source_variants) >= max_variants:
                        break
            # Resize copies for this specific scale so we never mutate the originals.
            scaled_variants = [
                variant if scale == 1.0 else cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                for variant in source_variants
            ]
            for idx, variant in enumerate(scaled_variants):
                if (time.perf_counter() - start_time) > DETECTION_TIME_LIMIT:
                    last_error = f"timeout after {DETECTION_TIME_LIMIT:.0f}s"
                    return None, None
                corners, ids, _ = cv2.aruco.detectMarkers(variant, dictionary, parameters=parameters)
                if ids is None:
                    last_error = f"scale={scale:.2f}, variant={idx}, no markers"
                    continue
                id_map: Dict[int, np.ndarray] = {}
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    marker_id = int(marker_id)
                    if marker_id not in MARKER_IDS:
                        continue
                    pts = marker_corners[0] / (scale * base_factor)
                    id_map[marker_id] = pts.astype(np.float32)
                if id_map:
                    partial_maps.append({marker_id: pts.copy() for marker_id, pts in id_map.items()})
                    merged = _merge_marker_maps(partial_maps)
                    if merged is not None:
                        return merged[0], merged[1]
                if len(id_map) != 4:
                    missing = [mid for mid in MARKER_IDS if mid not in id_map]
                    last_error = f"scale={scale:.2f}, variant={idx}, missing {missing}"
                    continue
                assembled = _assemble_marker_geometry(id_map)
                if assembled is not None:
                    return assembled[0], assembled[1]
            merged = _merge_marker_maps(partial_maps)
            if merged is not None:
                return merged[0], merged[1]
        return None, None

    # 1) For very dark previews try the aggressive boosted variants first.
    if dark_scene:
        boosted_variants = _boost_low_light_variants(scaled_preview)
        for extra in scaled_extras:
            boosted_variants.extend(_boost_low_light_variants(extra))
        if boosted_variants:
            result = _run_detection([], extended=False, predefined_variants=boosted_variants)
            if result[0] is not None:
                return result  # type: ignore[misc]

    # 2) Try the plain preview.
    result = _run_detection([scaled_preview], extended=False)
    if result[0] is not None:
        return result  # type: ignore[misc]

    # 3) Fall back to extended processing that also draws from the linear measurement.
    if scaled_extras:
        result = _run_detection([scaled_preview] + scaled_extras, extended=True)
        if result[0] is not None:
            return result  # type: ignore[misc]

    label_hint = f" for {label}" if label else ""
    raise RuntimeError(f"Marker detection failed{label_hint} ({last_error})")




def compute_destination(points: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Compute the destination rectangle and perspective matrix."""
    width_top = np.linalg.norm(points[1] - points[0])
    width_bottom = np.linalg.norm(points[2] - points[3])
    height_left = np.linalg.norm(points[3] - points[0])
    height_right = np.linalg.norm(points[2] - points[1])
    width = max(int(round(max(width_top, width_bottom))), 1)
    height = max(int(round(max(height_left, height_right))), 1)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, dst)
    return matrix, (width, height)


def warp_image(image: np.ndarray, matrix: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Apply a perspective transform to `image`."""
    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR)


def plot_diff_means(
    stats: Sequence[Tuple[int, float]],
    output_path: Path,
    error_output_path: Path,
    show_plot: bool,
    frame_count: int,
) -> None:
    """Plot diff_mean vs. input percentage with a linear fit plus residuals."""
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
    residual_percent = (residuals / denom) * 100.0  # Normalize by fit to avoid huge % near zero.

    text_color = "#2D2D2D"
    point_color = "#4A90E2"
    fit_color = "#6AB187"
    # Set global font size for all elements
    plt.rcParams.update({'font.size': 14}) 
    fig, (ax_lin, ax_res) = plt.subplots(2, 1, figsize=(7, 8), sharex=False)
    ax_lin.scatter(percentages, values, color=point_color)
    ax_lin.plot(percentages, slope * indices + intercept, color=fit_color, linewidth=2)
    ax_lin.set_ylabel("Mean intensity", color=text_color, fontsize=16)
    ax_lin.set_title("Sensor Linearity", color=text_color, fontsize=20)
    ax_lin.yaxis.grid(True, color="#CCCCCC", alpha=0.5)
    ax_lin.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax_lin.spines["right"].set_visible(False)
    ax_lin.spines["top"].set_visible(False)
    ax_lin.spines["left"].set_visible(False)
    ax_lin.spines["bottom"].set_color(text_color)

    ax_res.axhline(0.0, color=fit_color, linestyle="--", linewidth=1)
    ax_res.plot(percentages, residual_percent, marker="o", color=point_color, linewidth=1.5)
    ax_res.set_xlabel("Input level", color=text_color, fontsize=16)
    ax_res.set_ylabel("Residual", color=text_color, fontsize=16)
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


def main() -> None:
    args = parse_args()
    manual_dir = args.manual_dir
    off_path = manual_dir / args.off_name
    if not off_path.exists():
        raise FileNotFoundError(f"Missing OFF frame: {off_path}")
    off_measurement, _ = load_measurement_and_preview(off_path)

    diff_stats: List[Tuple[int, float]] = []
    alignment_cache: Tuple[np.ndarray, Tuple[int, int]] | None = None
    cache_source: int | None = None
    frame_indices = list(range(1, args.frame_count + 1))
    # Work from brightest → darkest so the cached homography (when needed) is high quality.
    for idx in reversed(frame_indices):
        frame_path = manual_dir / f"{idx}.CR2"
        if not frame_path.exists():
            print(f"Skipping missing frame {frame_path}")
            continue
        measurement, preview = load_measurement_and_preview(frame_path)
        try:
            inner_corners, _ = detect_inner_corners(
                preview,
                label=f"{idx:02d}",
                extra_images=[measurement],
            )
            matrix, size = compute_destination(inner_corners)
            alignment_cache = (matrix.copy(), size)
            cache_source = idx
        except RuntimeError as exc:
            if alignment_cache is None or cache_source is None:
                print(f"Frame {idx:02d}: {exc}")
                continue
            matrix, size = alignment_cache
            print(
                f"Frame {idx:02d}: {exc}; reusing alignment from frame {cache_source:02d}"
            )
        warped_frame = warp_image(measurement, matrix, size)
        warped_off = warp_image(off_measurement, matrix, size)
        diff = np.clip(warped_frame - warped_off, 0.0, None)
        diff_mean = float(diff.mean())
        diff_stats.append((idx, diff_mean))
        print(
            f"Frame {idx:02d}: diff_mean={diff_mean:.6f} "
            f"(input {idx * (100.0 / args.frame_count):.2f}%)"
        )

    diff_stats.sort(key=lambda entry: entry[0])

    plot_diff_means(
        diff_stats,
        args.plot_path,
        args.error_plot_path,
        args.show_plot,
        args.frame_count,
    )
    print(f"Saved diff_mean plot to {args.plot_path}")
    print(f"Saved residual plot to {args.error_plot_path}")


if __name__ == "__main__":
    main()
