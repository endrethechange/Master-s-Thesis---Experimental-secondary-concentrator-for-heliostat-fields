"""Target detection, SEC-specific marker refinement, and perspective warping.

The detector first proposes candidate quadrilaterals from bright target contours
and from the printed five-dot marker pattern. Each candidate is warped into the
known target frame, scored over the four possible rotations, and optionally
refined with SEC-specific marker searches when the secondary concentrator hides
or washes out marker contrast.

Only geometry is estimated here. Photometric measurements use the raw-derived
luminance array that is warped with the selected transform; the detector preview
never feeds reported irradiance or power values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from analysis_geometry import build_geometry
from analysis_io import load_raw_luminance, photometric_normalize
from analysis_model import *
from analysis_utils import normalize_for_display


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four source points as top-left, top-right, bottom-right, bottom-left."""

    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def is_convex_quad(points: np.ndarray) -> bool:
    """Return True when four points form a finite convex quadrilateral."""

    quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(quad)):
        return False
    area = float(abs(cv2.contourArea(quad)))
    if area <= 1.0:
        return False
    return bool(cv2.isContourConvex(quad.astype(np.float32)))


def source_quad_is_plausible(quad: np.ndarray, image_shape: tuple[int, int]) -> bool:
    """Reject impossible or wildly oversized source quadrilaterals."""

    if not is_convex_quad(quad):
        return False
    height, width = image_shape[:2]
    area = float(abs(cv2.contourArea(quad.astype(np.float32))))
    image_area = float(height * width)
    if area < image_area * 0.006 or area > image_area * 0.40:
        return False
    x_values = quad[:, 0]
    y_values = quad[:, 1]
    margin_x = width * 0.15
    margin_y = height * 0.15
    if np.min(x_values) < -margin_x or np.max(x_values) > width + margin_x:
        return False
    if np.min(y_values) < -margin_y or np.max(y_values) > height + margin_y:
        return False
    side_lengths = [float(np.linalg.norm(quad[(index + 1) % 4] - quad[index])) for index in range(4)]
    if min(side_lengths) <= 1.0:
        return False
    if max(side_lengths) / min(side_lengths) > 12.0:
        return False
    return True


def generate_contour_quads(display_gray: np.ndarray, thresholds: list[int], close_kernel: int) -> Iterable[np.ndarray]:
    """Generate quadrilateral candidates from bright target contours."""

    blurred = cv2.GaussianBlur(display_gray, (5, 5), 0)
    max_width = display_gray.shape[1] * 0.55
    max_height = display_gray.shape[0] * 0.55
    for threshold in thresholds:
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        if close_kernel > 1:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 900 or area > display_gray.size * 0.35:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width < 25 or height < 25:
                continue
            if width > max_width or height > max_height:
                continue
            aspect = max(width, height) / max(min(width, height), 1)
            if aspect > 3.2:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                yield order_quad(approx.reshape(4, 2).astype(np.float32))
            else:
                yield order_quad(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32))


def iou_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union for two bounding boxes."""

    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = float((ix1 - ix0) * (iy1 - iy0))
    area_a = float((ax1 - ax0) * (ay1 - ay0))
    area_b = float((bx1 - bx0) * (by1 - by0))
    return intersection / max(area_a + area_b - intersection, 1.0)


def candidate_regions(display_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find likely target regions before searching for dot patterns."""

    regions: list[tuple[int, int, int, int]] = []

    def add_region(x: int, y: int, width: int, height: int, pad_fraction: float = 0.45) -> None:
        pad_x = max(20, int(width * pad_fraction))
        pad_y = max(20, int(height * pad_fraction))
        bbox = (
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(display_gray.shape[1], x + width + pad_x),
            min(display_gray.shape[0], y + height + pad_y),
        )
        if any(iou_bbox(bbox, existing) > 0.78 for existing in regions):
            return
        regions.append(bbox)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(display_gray)
    for threshold in [115, 140, 165, 190]:
        _, binary = cv2.threshold(clahe, threshold, 255, cv2.THRESH_BINARY)
        for close_size in [7, 13]:
            mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, num_labels):
                x, y, width, height, area = stats[label]
                if area < 900 or area > display_gray.size * 0.35:
                    continue
                aspect = max(width, height) / max(min(width, height), 1)
                if aspect > 3.2:
                    continue
                add_region(int(x), int(y), int(width), int(height))

    center_x = display_gray.shape[1] // 2
    center_y = display_gray.shape[0] // 2
    for side_fraction in [0.18, 0.25, 0.34, 0.45]:
        side = int(min(display_gray.shape[:2]) * side_fraction)
        if side >= 80:
            add_region(center_x - side // 2, center_y - side // 2, side, side, pad_fraction=0.15)

    ranked = sorted(
        regions,
        key=lambda bbox: (
            -abs(((bbox[2] + bbox[0]) / 2.0) - center_x) - abs(((bbox[3] + bbox[1]) / 2.0) - center_y),
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
        ),
        reverse=True,
    )
    return ranked[:8]


def generate_region_quads(display_gray: np.ndarray, region: tuple[int, int, int, int]) -> Iterable[np.ndarray]:
    """Generate quadrilateral candidates inside one region."""

    x0, y0, x1, y1 = region
    roi = display_gray[y0:y1, x0:x1]
    if roi.size == 0:
        return
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
    min_area = max(600, int(roi.shape[0] * roi.shape[1] * 0.015))
    max_area = roi.shape[0] * roi.shape[1] * 0.92
    seen: set[tuple[int, ...]] = set()
    for threshold in [115, 145, 175, 205]:
        _, binary = cv2.threshold(clahe, threshold, 255, cv2.THRESH_BINARY)
        for close_size in [5, 11]:
            mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if min(width, height) < 25:
                    continue
                aspect = max(width, height) / max(min(width, height), 1.0)
                if aspect > 3.0:
                    continue
                box = cv2.boxPoints(rect).astype(np.float32)
                box[:, 0] += x0
                box[:, 1] += y0
                ordered = order_quad(box)
                key = tuple(np.round(ordered.reshape(-1) / 4.0).astype(int).tolist())
                if key in seen:
                    continue
                seen.add(key)
                yield ordered


@dataclass
class BlobCandidate:
    """Candidate dark marker dot in the source image."""

    x: float
    y: float
    area: float
    darkness: float


@dataclass(frozen=True)
class MarkerCandidate:
    """Candidate printed marker location in warped target coordinates."""

    point: np.ndarray
    score: float
    contrast: float
    fallback: bool = False


def detect_dark_blobs(
    display_gray: np.ndarray,
    region: tuple[int, int, int, int],
    allowed_mask: np.ndarray | None = None,
) -> list[BlobCandidate]:
    """Detect dark blobs that could be the five printed target markers."""

    x0, y0, x1, y1 = region
    roi = display_gray[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    kernel_size = max(25, int(min(roi.shape[:2]) * 0.22))
    kernel_size += 1 if kernel_size % 2 == 0 else 0
    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    candidates: list[BlobCandidate] = []
    for percentile in [93.0, 95.0, 97.0, 98.5]:
        threshold = max(12, int(np.percentile(blackhat, percentile)))
        _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, num_labels):
            x, y, width, height, area = stats[label]
            if area < 25 or area > max(4500, roi.size * 0.06):
                continue
            aspect = max(width, height) / max(min(width, height), 1)
            if aspect > 2.8:
                continue
            centroid_x = float(centroids[label][0] + x0)
            centroid_y = float(centroids[label][1] + y0)
            if allowed_mask is not None:
                mask_x = int(round(centroid_x))
                mask_y = int(round(centroid_y))
                if mask_x < 0 or mask_x >= allowed_mask.shape[1] or mask_y < 0 or mask_y >= allowed_mask.shape[0]:
                    continue
                if not bool(allowed_mask[mask_y, mask_x]):
                    continue
            component = labels[y : y + height, x : x + width] == label
            darkness = float(blackhat[y : y + height, x : x + width][component].mean())
            candidates.append(BlobCandidate(centroid_x, centroid_y, float(area), darkness))

    deduplicated: list[BlobCandidate] = []
    for blob in sorted(candidates, key=lambda item: item.darkness * math.sqrt(max(item.area, 1.0)), reverse=True):
        if any(math.dist((blob.x, blob.y), (other.x, other.y)) < 10.0 for other in deduplicated):
            continue
        deduplicated.append(blob)
        if len(deduplicated) >= 14:
            break
    return deduplicated


def sec_marker_allowed_mask(display_gray: np.ndarray) -> np.ndarray | None:
    """Return a SEC-only marker mask that suppresses the secondary concentrator."""

    blurred = cv2.GaussianBlur(display_gray, (0, 0), 3)
    height, width = display_gray.shape[:2]
    best_component: tuple[float, float] | None = None
    for percentile in [99.8, 99.5, 99.2, 98.8, 98.0]:
        threshold = float(np.percentile(blurred, percentile))
        binary = (blurred >= threshold).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label in range(1, num_labels):
            x, y, component_width, component_height, area = stats[label]
            if area < 10 or area > display_gray.size * 0.08:
                continue
            center_x, center_y = centroids[label]
            if center_y < height * 0.32:
                continue
            component = labels == label
            mean_intensity = float(blurred[component].mean())
            score = mean_intensity + (float(center_y) * 0.12) - (abs(float(center_x) - (width / 2.0)) * 0.05)
            score += min(float(area), 2000.0) * 0.01
            if best_component is None or score > best_component[0]:
                best_component = (score, float(center_y))
    if best_component is None:
        return None
    marker_floor = int(max(0, round(best_component[1] - max(35.0, min(height, width) * 0.070))))
    y_grid = np.arange(height, dtype=np.int32)[:, None]
    return np.broadcast_to(y_grid >= marker_floor, (height, width)).copy()


def collinear_triplet(points: np.ndarray, max_error: float) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Check whether three points are nearly collinear."""

    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    direction = vh[0].astype(np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    distances = np.abs((points - centroid) @ normal)
    projections = (points - centroid) @ direction
    return bool(float(np.max(distances)) <= max_error), direction, normal, projections.astype(np.float32)


def quad_from_marker_pattern(blobs: list[BlobCandidate], expected_centers: np.ndarray) -> Iterable[np.ndarray]:
    """Infer the target square from the unique five-dot marker pattern."""

    if len(blobs) < 5:
        return
    points = np.array([(blob.x, blob.y) for blob in blobs], dtype=np.float32)
    point_indices = list(range(len(blobs)))
    expected_corner_dots = np.array(
        [
            expected_centers[0],
            expected_centers[1],
            expected_centers[4],
            expected_centers[3],
        ],
        dtype=np.float32,
    )
    target_corners = np.array(
        [
            [[0.0, 0.0]],
            [[TARGET_PIXELS - 1.0, 0.0]],
            [[TARGET_PIXELS - 1.0, TARGET_PIXELS - 1.0]],
            [[0.0, TARGET_PIXELS - 1.0]],
        ],
        dtype=np.float32,
    )
    scored_quads: list[tuple[float, float, np.ndarray]] = []
    for right_indices in combinations(point_indices, 3):
        right_points = points[list(right_indices)]
        span = max(np.ptp(right_points[:, 0]), np.ptp(right_points[:, 1]))
        is_collinear, direction, normal, projections = collinear_triplet(right_points, max_error=max(5.0, span * 0.14))
        if not is_collinear:
            continue
        ordered_right_options = [
            right_points[np.argsort(projections)],
            right_points[np.argsort(projections)][::-1],
        ]
        remaining = [index for index in point_indices if index not in right_indices]
        for left_indices in combinations(remaining, 2):
            left_points = points[list(left_indices)]
            side_distances = (left_points - right_points.mean(axis=0)) @ normal
            if side_distances[0] * side_distances[1] <= 0:
                continue
            if min(abs(float(side_distances[0])), abs(float(side_distances[1]))) < 8.0:
                continue
            for ordered_right in ordered_right_options:
                top_right, middle_right, bottom_right = ordered_right
                segment_1 = float(np.linalg.norm(middle_right - top_right))
                segment_2 = float(np.linalg.norm(bottom_right - middle_right))
                if min(segment_1, segment_2) < 6.0:
                    continue
                if min(segment_1, segment_2) / max(segment_1, segment_2) < 0.15:
                    continue
                for top_left, bottom_left in [(left_points[0], left_points[1]), (left_points[1], left_points[0])]:
                    if min(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)) < 12.0:
                        continue
                    observed_corner_dots = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
                    if not is_convex_quad(observed_corner_dots):
                        continue
                    observed_area = float(abs(cv2.contourArea(observed_corner_dots.astype(np.float32))))
                    if observed_area < 200.0:
                        continue
                    homography = cv2.getPerspectiveTransform(expected_corner_dots, observed_corner_dots)
                    predicted_middle = cv2.perspectiveTransform(expected_centers[2].reshape(1, 1, 2), homography).reshape(2)
                    middle_error = float(np.linalg.norm(predicted_middle - middle_right))
                    if middle_error > max(10.0, span * 0.12):
                        continue
                    projected = cv2.perspectiveTransform(target_corners, homography).reshape(4, 2)
                    if not is_convex_quad(projected):
                        continue
                    area = float(abs(cv2.contourArea(projected.astype(np.float32))))
                    if area < 900.0:
                        continue
                    # The marker centers sit close to the frame corners; a valid target frame
                    # should therefore be only moderately larger than the four-dot quadrilateral.
                    area_ratio = area / observed_area
                    if area_ratio < 0.60 or area_ratio > 3.00:
                        continue
                    scored_quads.append((middle_error, -area, projected.astype(np.float32)))

    for _, _, quad in sorted(scored_quads, key=lambda item: (item[0], item[1]))[:240]:
        yield quad


def candidate_quads(
    display_gray: np.ndarray,
    expected_centers: np.ndarray,
    marker_allowed_mask: np.ndarray | None = None,
) -> Iterable[np.ndarray]:
    """Yield target quadrilateral candidates from contours and marker dots."""

    seen: set[tuple[int, ...]] = set()

    def maybe_yield(quad: np.ndarray) -> Iterable[np.ndarray]:
        key = tuple(np.round(quad.reshape(-1) / 4.0).astype(int).tolist())
        if key in seen:
            return
        seen.add(key)
        yield quad

    full_region = (0, 0, display_gray.shape[1], display_gray.shape[0])
    for quad in quad_from_marker_pattern(detect_dark_blobs(display_gray, full_region, marker_allowed_mask), expected_centers):
        yield from maybe_yield(quad)

    regions = candidate_regions(display_gray)
    for region in regions:
        blobs = detect_dark_blobs(display_gray, region, marker_allowed_mask)
        for quad in quad_from_marker_pattern(blobs, expected_centers):
            yield from maybe_yield(quad)

    for quad in generate_contour_quads(display_gray, [90, 115, 140, 165, 190, 215], close_kernel=5):
        yield from maybe_yield(quad)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(display_gray)
    for quad in generate_contour_quads(clahe, [115, 140, 165, 190, 215], close_kernel=9):
        yield from maybe_yield(quad)
    for region in regions:
        for quad in generate_region_quads(display_gray, region):
            yield from maybe_yield(quad)


def match_centers(expected: np.ndarray, observed: np.ndarray, max_distance: float) -> tuple[int, float]:
    """Greedily match expected marker centers to observed dark components."""

    if observed.size == 0:
        return 0, max_distance
    remaining = observed.tolist()
    distances: list[float] = []
    for expected_point in expected:
        if not remaining:
            break
        current = [math.dist(expected_point.tolist(), point) for point in remaining]
        index = int(np.argmin(current))
        if current[index] <= max_distance:
            distances.append(float(current[index]))
            remaining.pop(index)
    if not distances:
        return 0, max_distance
    return len(distances), float(np.mean(distances))


def score_warp(warped_gray: np.ndarray, geometry: TargetGeometry) -> tuple[float, int]:
    """Score each possible 90-degree orientation of a warped target candidate."""

    normalized = normalize_for_display(warped_gray).astype(np.float32)
    size_scale = float(warped_gray.shape[0]) / float(TARGET_PIXELS)
    min_marker_area = max(60, int(round(1500.0 * size_scale * size_scale)))
    max_marker_area = max(min_marker_area + 1, int(round(60000.0 * size_scale * size_scale)))
    max_marker_distance = 135.0 * size_scale
    best_score = -1e12
    best_turns = 0
    for turns in range(4):
        rotated = np.rot90(normalized, turns)
        blurred = cv2.GaussianBlur(rotated, (0, 0), 4)
        white_values = blurred[geometry.white_background_mask]
        white_mean = float(np.mean(white_values)) if white_values.size else 0.0
        white_std = float(np.std(white_values)) if white_values.size else 0.0
        marker_means = np.array([float(np.mean(blurred[mask])) for mask in geometry.marker_masks], dtype=np.float32)
        ring_means = np.array([float(np.mean(blurred[mask])) for mask in geometry.marker_ring_masks], dtype=np.float32)
        marker_contrast = ring_means - marker_means
        contrast_mean = float(np.mean(marker_contrast))
        template = geometry.template_model
        roi = geometry.white_background_mask | np.logical_or.reduce(geometry.marker_masks)
        roi_values = blurred[roi]
        roi_std = float(np.std(roi_values))
        normalized_roi = (blurred - float(np.mean(roi_values))) / max(roi_std, 1.0)
        normalized_template = (template - float(np.mean(template[roi]))) / max(float(np.std(template[roi])), 1e-6)
        template_score = float(np.mean(normalized_roi[roi] * normalized_template[roi]))

        dark_cutoff = min(float(np.percentile(white_values, 20.0)), white_mean - max(contrast_mean * 0.30, 8.0))
        dark_mask = (blurred < dark_cutoff).astype(np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        centers: list[tuple[float, float]] = []
        for label in range(1, num_labels):
            x, y, width, height, area = stats[label]
            if area < min_marker_area or area > max_marker_area:
                continue
            aspect = max(width, height) / max(min(width, height), 1)
            if aspect > 2.5:
                continue
            centers.append((float(centroids[label][0]), float(centroids[label][1])))
        observed = np.array(centers, dtype=np.float32) if centers else np.empty((0, 2), dtype=np.float32)
        marker_matches, marker_distance = match_centers(geometry.marker_centers, observed, max_distance=max_marker_distance)
        marker_dark_fraction = float(np.mean([np.mean(blurred[mask] < dark_cutoff) for mask in geometry.marker_masks]))
        stray_dark_fraction = float(np.mean(blurred[geometry.white_background_mask] < dark_cutoff))
        score = (
            (contrast_mean * 2.0)
            + (marker_matches * 35.0)
            + (marker_dark_fraction * 85.0)
            + (template_score * 80.0)
            + (white_mean * 0.04)
            - (white_std * 0.04)
            - (abs(len(centers) - 5) * 12.0)
            - (marker_distance * 0.15)
            - (stray_dark_fraction * 35.0)
        )
        if score > best_score:
            best_score = score
            best_turns = turns
    return best_score, best_turns


def marker_layout_quality(warped_gray: np.ndarray, geometry: TargetGeometry) -> tuple[int, float]:
    """Count how many expected marker dots have dark-dot contrast in a warped frame."""

    normalized = normalize_for_display(warped_gray).astype(np.float32)
    blurred = cv2.GaussianBlur(normalized, (0, 0), 3)
    board_values = blurred[geometry.white_background_mask]
    board_std = float(np.std(board_values)) if board_values.size else 0.0
    min_contrast = max(5.0, 0.08 * board_std)
    valid_count = 0
    contrasts: list[float] = []
    for marker_mask, ring_mask in zip(geometry.marker_masks, geometry.marker_ring_masks):
        marker_values = blurred[marker_mask]
        ring_values = blurred[ring_mask]
        if marker_values.size == 0 or ring_values.size == 0:
            continue
        marker_median = float(np.median(marker_values))
        ring_median = float(np.median(ring_values))
        ring_std = float(np.std(ring_values))
        contrast = ring_median - marker_median
        dark_cutoff = ring_median - max(min_contrast, 0.35 * ring_std)
        dark_fraction = float(np.mean(marker_values < dark_cutoff))
        contrasts.append(contrast)
        if contrast > min_contrast and dark_fraction > 0.20:
            valid_count += 1
    mean_contrast = float(np.mean(contrasts)) if contrasts else 0.0
    return valid_count, mean_contrast


def target_surface_quality(warped_gray: np.ndarray, geometry: TargetGeometry) -> tuple[float, float]:
    """Measure brightness and variation of the target's white interior."""

    normalized = normalize_for_display(warped_gray).astype(np.float32)
    white_values = normalized[geometry.white_background_mask]
    if white_values.size == 0:
        return 0.0, 255.0
    return float(np.mean(white_values)), float(np.std(white_values))


def resize_for_detection(display_gray: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Resize an image for faster target detection and return the scale factor."""

    height, width = display_gray.shape[:2]
    largest = max(height, width)
    if largest <= max_dim:
        return display_gray, 1.0
    scale = float(max_dim) / float(largest)
    resized = cv2.resize(display_gray, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def inverse_rotate_points_ccw(points: np.ndarray, turns: int, size: int = TARGET_PIXELS) -> np.ndarray:
    """Map final rotated target coordinates back to the unrotated warp frame."""

    unrotated = np.asarray(points, dtype=np.float32).copy()
    for _ in range(turns % 4):
        x = unrotated[:, 0].copy()
        y = unrotated[:, 1].copy()
        unrotated[:, 0] = (size - 1) - y
        unrotated[:, 1] = x
    return unrotated


def projected_marker_points(
    quad: np.ndarray,
    turns: int,
    geometry: TargetGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return expected unrotated markers and their source-image projection."""

    target_to_source = cv2.getPerspectiveTransform(TARGET_CORNERS, quad.astype(np.float32))
    unrotated_markers = inverse_rotate_points_ccw(geometry.marker_centers, turns)
    marker_points = cv2.perspectiveTransform(unrotated_markers.reshape(-1, 1, 2), target_to_source).reshape(-1, 2)
    return unrotated_markers, marker_points


def marker_span_px(marker_points: np.ndarray) -> float:
    """Return the largest current target edge span in source pixels."""

    return max(
        float(np.linalg.norm(marker_points[1] - marker_points[0])),
        float(np.linalg.norm(marker_points[4] - marker_points[3])),
        float(np.linalg.norm(marker_points[3] - marker_points[0])),
        float(np.linalg.norm(marker_points[4] - marker_points[1])),
    )


def quad_from_marker_refit(
    marker_points: np.ndarray,
    unrotated_markers: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    """Fit target corners from source marker points and validate the result."""

    source_to_unrotated, _ = cv2.findHomography(
        marker_points.astype(np.float32),
        unrotated_markers.astype(np.float32),
        method=0,
    )
    if source_to_unrotated is None:
        return None
    try:
        target_to_source = np.linalg.inv(source_to_unrotated)
    except np.linalg.LinAlgError:
        return None
    refined_quad = cv2.perspectiveTransform(TARGET_CORNERS.reshape(-1, 1, 2), target_to_source).reshape(-1, 2)
    if not source_quad_is_plausible(refined_quad.astype(np.float32), image_shape):
        return None
    return refined_quad.astype(np.float32)


def candidate_marker_points_source(
    quad_small: np.ndarray,
    turns: int,
    score_geometry: TargetGeometry,
    score_destination: np.ndarray,
    score_pixels: int,
) -> np.ndarray:
    """Project expected marker centers from a scored target candidate back to source pixels."""

    target_to_source = cv2.getPerspectiveTransform(score_destination.astype(np.float32), quad_small.astype(np.float32))
    unrotated_points = inverse_rotate_points_ccw(score_geometry.marker_centers, turns, size=score_pixels)
    return cv2.perspectiveTransform(unrotated_points.reshape(-1, 1, 2), target_to_source).reshape(-1, 2)


def parallel_angle_error_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Return the unsigned angle error from parallel between two vectors."""

    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-6 or norm_b <= 1e-6:
        return 90.0
    cosine = abs(float(np.dot(vector_a, vector_b) / (norm_a * norm_b)))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def marker_pattern_geometry_penalty(
    marker_points_source: np.ndarray,
    image_shape: tuple[int, int],
    marker_allowed_mask: np.ndarray | None = None,
) -> float | None:
    """Return a geometry penalty for the five-dot target pattern, or None if impossible."""

    points = np.asarray(marker_points_source, dtype=np.float32).reshape(5, 2)
    if not np.all(np.isfinite(points)):
        return None
    if marker_allowed_mask is not None:
        height, width = marker_allowed_mask.shape[:2]
        allowed_count = 0
        for point in points:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if 0 <= x < width and 0 <= y < height and bool(marker_allowed_mask[y, x]):
                allowed_count += 1
        if allowed_count < 4:
            return None

    top_left, top_right, middle_right, bottom_left, bottom_right = points
    marker_quad = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    if not is_convex_quad(marker_quad):
        return None
    marker_area = float(abs(cv2.contourArea(marker_quad.astype(np.float32))))
    image_area = float(image_shape[0] * image_shape[1])
    if marker_area < image_area * 0.0008 or marker_area > image_area * 0.28:
        return None

    right_points = np.array([top_right, middle_right, bottom_right], dtype=np.float32)
    right_length = float(np.linalg.norm(bottom_right - top_right))
    if right_length < 8.0:
        return None
    is_collinear, right_direction, right_normal, right_projection = collinear_triplet(
        right_points,
        max_error=max(5.0, right_length * 0.16),
    )
    if not is_collinear:
        return None
    if not (
        (right_projection[0] < right_projection[1] < right_projection[2])
        or (right_projection[0] > right_projection[1] > right_projection[2])
    ):
        return None
    middle_fraction = float(np.linalg.norm(middle_right - top_right) / max(right_length, 1e-6))
    if middle_fraction < 0.22 or middle_fraction > 0.78:
        return None

    right_centroid = right_points.mean(axis=0)
    left_side_distances = np.array(
        [
            float((top_left - right_centroid) @ right_normal),
            float((bottom_left - right_centroid) @ right_normal),
        ],
        dtype=np.float32,
    )
    if left_side_distances[0] * left_side_distances[1] <= 0:
        return None
    if min(abs(float(left_side_distances[0])), abs(float(left_side_distances[1]))) < max(8.0, right_length * 0.35):
        return None

    top_edge = top_right - top_left
    bottom_edge = bottom_right - bottom_left
    left_edge = bottom_left - top_left
    right_edge = bottom_right - top_right
    top_length = float(np.linalg.norm(top_edge))
    bottom_length = float(np.linalg.norm(bottom_edge))
    left_length = float(np.linalg.norm(left_edge))
    if min(top_length, bottom_length, left_length, right_length) < 8.0:
        return None
    if max(top_length, bottom_length) / max(min(top_length, bottom_length), 1.0) > 2.7:
        return None
    if max(left_length, right_length) / max(min(left_length, right_length), 1.0) > 2.7:
        return None

    left_parallel_error = parallel_angle_error_degrees(left_edge, right_edge)
    row_parallel_error = parallel_angle_error_degrees(top_edge, bottom_edge)
    if left_parallel_error > 38.0 or row_parallel_error > 30.0:
        return None

    collinearity_error = float(np.max(np.abs((right_points - right_points.mean(axis=0)) @ right_normal)))
    penalty = (
        (left_parallel_error * 3.0)
        + (row_parallel_error * 2.0)
        + (abs(math.log(max(top_length, 1.0) / max(bottom_length, 1.0))) * 35.0)
        + (abs(math.log(max(left_length, 1.0) / max(right_length, 1.0))) * 35.0)
        + (abs(middle_fraction - 0.5) * 70.0)
        + (collinearity_error * 1.6)
    )
    return float(penalty)


def circular_mean_kernel(radius: float) -> np.ndarray:
    """Build a normalized circular averaging kernel."""

    kernel_radius = max(1, int(math.ceil(radius)))
    y_grid, x_grid = np.ogrid[-kernel_radius : kernel_radius + 1, -kernel_radius : kernel_radius + 1]
    mask = (x_grid * x_grid + y_grid * y_grid) <= (radius * radius)
    kernel = mask.astype(np.float32)
    return kernel / max(float(kernel.sum()), 1.0)


def ring_mean_kernel(inner_radius: float, outer_radius: float) -> np.ndarray:
    """Build a normalized ring averaging kernel around a marker dot."""

    kernel_radius = max(1, int(math.ceil(outer_radius)))
    y_grid, x_grid = np.ogrid[-kernel_radius : kernel_radius + 1, -kernel_radius : kernel_radius + 1]
    distance_squared = x_grid * x_grid + y_grid * y_grid
    mask = (distance_squared >= inner_radius * inner_radius) & (distance_squared <= outer_radius * outer_radius)
    kernel = mask.astype(np.float32)
    return kernel / max(float(kernel.sum()), 1.0)


def sec_local_marker_candidates(
    display_gray: np.ndarray,
    predicted: np.ndarray,
    dot_radius: float,
    search_radius: float,
    marker_allowed_mask: np.ndarray | None,
) -> list[tuple[np.ndarray, float]]:
    """Find compact dark-dot candidates near one predicted SEC marker position."""

    predicted = np.asarray(predicted, dtype=np.float32).reshape(2)
    height, width = display_gray.shape[:2]
    padding = max(8, int(math.ceil(dot_radius * 3.0)))
    x0 = max(0, int(math.floor(float(predicted[0]) - search_radius - padding)))
    y0 = max(0, int(math.floor(float(predicted[1]) - search_radius - padding)))
    x1 = min(width, int(math.ceil(float(predicted[0]) + search_radius + padding + 1)))
    y1 = min(height, int(math.ceil(float(predicted[1]) + search_radius + padding + 1)))
    roi = display_gray[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return [(predicted, 12.0)]

    blurred = cv2.GaussianBlur(roi, (0, 0), max(0.8, dot_radius * 0.15))
    disk_kernel = circular_mean_kernel(max(2.0, dot_radius * 0.75))
    marker_ring_kernel = ring_mean_kernel(dot_radius * 1.05, dot_radius * 2.25)
    disk_mean = cv2.filter2D(blurred, -1, disk_kernel, borderType=cv2.BORDER_REPLICATE)
    ring_mean = cv2.filter2D(blurred, -1, marker_ring_kernel, borderType=cv2.BORDER_REPLICATE)
    contrast = ring_mean - disk_mean

    y_indices, x_indices = np.indices(contrast.shape)
    source_x = x_indices + x0
    source_y = y_indices + y0
    distance = np.sqrt((source_x - float(predicted[0])) ** 2 + (source_y - float(predicted[1])) ** 2)
    valid = distance <= search_radius
    valid[:padding, :] = False
    valid[-padding:, :] = False
    valid[:, :padding] = False
    valid[:, -padding:] = False
    if marker_allowed_mask is not None:
        valid &= marker_allowed_mask[np.clip(source_y, 0, height - 1), np.clip(source_x, 0, width - 1)]

    allowed_values = display_gray[marker_allowed_mask] if marker_allowed_mask is not None and bool(marker_allowed_mask.any()) else display_gray
    local_white_floor = float(np.percentile(allowed_values, 48.0))
    valid &= ring_mean > (local_white_floor * 0.38)
    ranked_score = contrast - (distance * 0.38) - (disk_mean * 0.10) + (ring_mean * 0.02)
    ranked_score[~valid] = -1e9

    candidates: list[tuple[np.ndarray, float]] = [(predicted, 12.0)]
    for flat_index in np.argsort(ranked_score.ravel())[::-1][:1600]:
        y_index, x_index = np.unravel_index(int(flat_index), ranked_score.shape)
        if ranked_score[y_index, x_index] <= -1e8:
            continue
        candidate = np.array([float(x_index + x0), float(y_index + y0)], dtype=np.float32)
        if any(float(np.linalg.norm(candidate - existing[0])) < max(5.0, dot_radius * 1.45) for existing in candidates):
            continue
        candidates.append((candidate, float(ranked_score[y_index, x_index])))
        if len(candidates) >= 9:
            break

    region = (x0, y0, x1, y1)
    expected_dot_area = math.pi * max(dot_radius, 1.0) ** 2
    for blob in detect_dark_blobs(display_gray, region, marker_allowed_mask):
        if blob.area > max(900.0, expected_dot_area * 12.0):
            continue
        candidate = np.array([blob.x, blob.y], dtype=np.float32)
        distance_to_predicted = float(np.linalg.norm(candidate - predicted))
        if distance_to_predicted > search_radius:
            continue
        score = (blob.darkness * 0.22) - (distance_to_predicted * 0.20)
        if any(float(np.linalg.norm(candidate - existing[0])) < max(5.0, dot_radius * 1.35) for existing in candidates):
            continue
        candidates.append((candidate, float(score)))

    return sorted(candidates, key=lambda item: item[1], reverse=True)[:9]


def sec_candidate_markers_are_plausible(marker_points_source: np.ndarray, marker_allowed_mask: np.ndarray) -> bool:
    """Reject SEC candidates that put target markers on the secondary concentrator."""

    height, width = marker_allowed_mask.shape[:2]
    allowed_count = 0
    for point in marker_points_source:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height and bool(marker_allowed_mask[y, x]):
            allowed_count += 1
    if allowed_count < 4:
        return False
    top_right, middle_right, bottom_right = marker_points_source[[1, 2, 4]]
    y_tolerance = max(4.0, height * 0.006)
    if top_right[1] > middle_right[1] + y_tolerance:
        return False
    if middle_right[1] > bottom_right[1] + y_tolerance:
        return False
    right_edge_span_y = float(bottom_right[1] - top_right[1])
    right_edge_span_x = abs(float(bottom_right[0] - top_right[0]))
    right_edge_length = float(np.linalg.norm(bottom_right - top_right))
    return right_edge_span_y >= max(8.0, right_edge_length * 0.35, right_edge_span_x * 0.85)


def sec_refined_right_column_quad(
    display_gray: np.ndarray,
    quad_small: np.ndarray,
    turns: int,
    geometry: TargetGeometry,
    marker_allowed_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Correct SEC cases where the right marker column was assigned one dot too high."""

    if marker_allowed_mask is None:
        return None
    unrotated_markers, marker_points = projected_marker_points(quad_small, turns, geometry)

    middle_to_bottom = marker_points[4] - marker_points[2]
    step_length = float(np.linalg.norm(middle_to_bottom))
    if step_length < 8.0:
        return None
    extrapolated_bottom = marker_points[4] + middle_to_bottom
    dot_diameter = max(6.0, marker_span_px(marker_points) * (DOT_DIAMETER_PX / TARGET_PIXELS))
    expected_dot_area = math.pi * (dot_diameter * 0.5) ** 2
    max_dot_area = max(220.0, expected_dot_area * 14.0)
    search_radius = max(18.0, step_length * 1.25)
    full_region = (0, 0, display_gray.shape[1], display_gray.shape[0])
    candidates = detect_dark_blobs(display_gray, full_region, marker_allowed_mask)

    best_candidate: BlobCandidate | None = None
    best_score = 1e12
    for candidate in candidates:
        if candidate.area > max_dot_area:
            continue
        candidate_point = np.array([candidate.x, candidate.y], dtype=np.float32)
        distance_to_extrapolated = float(np.linalg.norm(candidate_point - extrapolated_bottom))
        if distance_to_extrapolated > search_radius:
            continue
        displacement = candidate_point - marker_points[4]
        forward_fraction = float(np.dot(displacement, middle_to_bottom) / max(step_length * step_length, 1e-6))
        if forward_fraction < 0.45 or forward_fraction > 1.55:
            continue
        perpendicular = abs(float((middle_to_bottom[0] * displacement[1]) - (middle_to_bottom[1] * displacement[0]))) / max(step_length, 1e-6)
        score = perpendicular + (abs(forward_fraction - 1.0) * step_length) + (distance_to_extrapolated * 0.15) - (candidate.darkness * 0.02)
        if score < best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None:
        return None

    shifted_markers = marker_points.copy()
    shifted_markers[1] = marker_points[2]
    shifted_markers[2] = marker_points[4]
    shifted_markers[4] = np.array([best_candidate.x, best_candidate.y], dtype=np.float32)
    return quad_from_marker_refit(shifted_markers, unrotated_markers, display_gray.shape)


def sec_refined_left_column_quad(
    display_gray: np.ndarray,
    quad_small: np.ndarray,
    turns: int,
    geometry: TargetGeometry,
    marker_allowed_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Refit only the SEC left markers while preserving the already validated right column."""

    if marker_allowed_mask is None:
        return None
    unrotated_markers, marker_points = projected_marker_points(quad_small, turns, geometry)

    marker_span = marker_span_px(marker_points)
    if marker_span < 18.0:
        return None
    dot_radius = max(4.0, marker_span * (DOT_RADIUS_PX / TARGET_PIXELS))
    search_radius = max(52.0, min(105.0, marker_span * 0.55))
    left_candidate_lists = {
        0: sec_local_marker_candidates(display_gray, marker_points[0], dot_radius, search_radius, marker_allowed_mask),
        3: sec_local_marker_candidates(display_gray, marker_points[3], dot_radius, search_radius, marker_allowed_mask),
    }
    current_penalty = marker_pattern_geometry_penalty(marker_points, display_gray.shape, marker_allowed_mask)
    best_score = -1e12
    best_markers: np.ndarray | None = None
    for top_left_candidate, bottom_left_candidate in product(left_candidate_lists[0], left_candidate_lists[3]):
        refined_markers = marker_points.copy()
        refined_markers[0] = top_left_candidate[0]
        refined_markers[3] = bottom_left_candidate[0]
        if float(np.linalg.norm(refined_markers[0] - refined_markers[3])) < max(10.0, dot_radius * 2.0):
            continue
        geometry_penalty = marker_pattern_geometry_penalty(refined_markers, display_gray.shape, marker_allowed_mask)
        if geometry_penalty is None:
            continue
        if current_penalty is not None and geometry_penalty > current_penalty + 35.0:
            continue
        left_movement = float(np.mean(np.linalg.norm(refined_markers[[0, 3]] - marker_points[[0, 3]], axis=1)))
        score = float(top_left_candidate[1] + bottom_left_candidate[1]) - (geometry_penalty * 2.2) - (left_movement * 0.45)
        if current_penalty is not None and geometry_penalty < current_penalty:
            score += min(100.0, (current_penalty - geometry_penalty) * 2.5)
        if score > best_score:
            best_score = score
            best_markers = refined_markers

    if best_markers is None:
        return None
    if current_penalty is not None:
        best_penalty = marker_pattern_geometry_penalty(best_markers, display_gray.shape, marker_allowed_mask)
        if best_penalty is None or best_penalty >= current_penalty - 3.0:
            return None
    return quad_from_marker_refit(best_markers, unrotated_markers, display_gray.shape)


def warped_marker_contrast_candidates(
    warped_display: np.ndarray,
    expected_point: np.ndarray,
    search_radius: float,
    max_candidates: int = 6,
) -> list[MarkerCandidate]:
    """Find dark printed-dot candidates near one expected marker in the warped frame."""

    expected_point = np.asarray(expected_point, dtype=np.float32).reshape(2)
    height, width = warped_display.shape[:2]
    disk_radius = max(10.0, DOT_RADIUS_PX * 0.48)
    ring_inner = DOT_RADIUS_PX * 0.82
    ring_outer = DOT_RADIUS_PX * 1.72
    padding = int(math.ceil(ring_outer)) + 4
    x0 = max(0, int(math.floor(float(expected_point[0]) - search_radius - padding)))
    y0 = max(0, int(math.floor(float(expected_point[1]) - search_radius - padding)))
    x1 = min(width, int(math.ceil(float(expected_point[0]) + search_radius + padding + 1)))
    y1 = min(height, int(math.ceil(float(expected_point[1]) + search_radius + padding + 1)))
    roi = warped_display[y0:y1, x0:x1]
    fallback = MarkerCandidate(expected_point.copy(), -18.0, 0.0, True)
    if roi.size == 0:
        return [fallback]

    disk_kernel = circular_mean_kernel(disk_radius)
    ring_kernel = ring_mean_kernel(ring_inner, ring_outer)
    disk_mean = cv2.filter2D(roi, -1, disk_kernel, borderType=cv2.BORDER_REPLICATE)
    ring_mean = cv2.filter2D(roi, -1, ring_kernel, borderType=cv2.BORDER_REPLICATE)
    contrast = ring_mean - disk_mean

    y_indices, x_indices = np.indices(roi.shape)
    source_x = x_indices + x0
    source_y = y_indices + y0
    distance = np.sqrt((source_x - float(expected_point[0])) ** 2 + (source_y - float(expected_point[1])) ** 2)
    valid = distance <= search_radius
    if not bool(np.any(valid)):
        return [fallback]

    local_mid = float(np.percentile(roi[valid], 55.0))
    ranked = contrast - (distance * 0.055) + (np.clip(local_mid - disk_mean, -40.0, 70.0) * 0.045)
    ranked[~valid] = -1e9

    candidates: list[MarkerCandidate] = []
    minimum_separation = max(28.0, DOT_RADIUS_PX * 0.48)
    for flat_index in np.argsort(ranked.ravel())[::-1][:2500]:
        y_index, x_index = np.unravel_index(int(flat_index), ranked.shape)
        if ranked[y_index, x_index] <= -1e8:
            continue
        point = np.array([float(x_index + x0), float(y_index + y0)], dtype=np.float32)
        candidate_contrast = float(contrast[y_index, x_index])
        candidate_score = float(ranked[y_index, x_index])
        if candidate_contrast < 1.2 and candidate_score < 0.0:
            continue
        if any(float(np.linalg.norm(point - existing.point)) < minimum_separation for existing in candidates):
            continue
        candidates.append(MarkerCandidate(point, candidate_score, candidate_contrast, False))
        if len(candidates) >= max_candidates:
            break

    candidates.append(fallback)
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def sec_refined_warped_marker_quad(
    display_gray: np.ndarray,
    quad_small: np.ndarray,
    turns: int,
    geometry: TargetGeometry,
    marker_allowed_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Snap a roughly detected SEC target to dark markers after warping to target space."""

    source_to_unrotated = cv2.getPerspectiveTransform(quad_small.astype(np.float32), TARGET_CORNERS)
    try:
        unrotated_to_source = np.linalg.inv(source_to_unrotated)
    except np.linalg.LinAlgError:
        return None

    warped_unrotated = cv2.warpPerspective(display_gray, source_to_unrotated, (TARGET_PIXELS, TARGET_PIXELS))
    warped_final = np.rot90(warped_unrotated, turns)
    warped_display = normalize_for_display(warped_final).astype(np.float32)
    warped_display = cv2.GaussianBlur(warped_display, (0, 0), 3.0)

    search_radius = 185.0
    candidate_lists = [
        warped_marker_contrast_candidates(warped_display, expected, search_radius)
        for expected in geometry.marker_centers
    ]
    if sum(any(not candidate.fallback for candidate in candidates) for candidates in candidate_lists) < 4:
        return None

    expected_final = geometry.marker_centers.astype(np.float32)
    expected_unrotated = inverse_rotate_points_ccw(expected_final, turns)
    current_marker_source = cv2.perspectiveTransform(
        expected_unrotated.reshape(-1, 1, 2),
        unrotated_to_source,
    ).reshape(-1, 2)
    current_penalty = marker_pattern_geometry_penalty(current_marker_source, display_gray.shape, marker_allowed_mask)
    current_layout_count, current_layout_contrast = marker_layout_quality(warped_final, geometry)

    best_score = -1e12
    best_quad: np.ndarray | None = None
    best_h: np.ndarray | None = None
    for selected in product(*candidate_lists):
        detected_count = sum(not candidate.fallback for candidate in selected)
        if detected_count < 4:
            continue
        points_final = np.array([candidate.point for candidate in selected], dtype=np.float32)
        if any(
            float(np.linalg.norm(points_final[i] - points_final[j])) < DOT_RADIUS_PX * 0.70
            for i in range(5)
            for j in range(i + 1, 5)
        ):
            continue
        points_unrotated = inverse_rotate_points_ccw(points_final, turns)
        points_source = cv2.perspectiveTransform(points_unrotated.reshape(-1, 1, 2), unrotated_to_source).reshape(-1, 2)
        geometry_penalty = marker_pattern_geometry_penalty(points_source, display_gray.shape, marker_allowed_mask)
        if geometry_penalty is None:
            continue
        refined_source_to_unrotated, _ = cv2.findHomography(
            points_source.astype(np.float32),
            expected_unrotated.astype(np.float32),
            method=0,
        )
        if refined_source_to_unrotated is None:
            continue
        try:
            refined_unrotated_to_source = np.linalg.inv(refined_source_to_unrotated)
        except np.linalg.LinAlgError:
            continue
        refined_quad = cv2.perspectiveTransform(TARGET_CORNERS.reshape(-1, 1, 2), refined_unrotated_to_source).reshape(-1, 2)
        if not source_quad_is_plausible(refined_quad.astype(np.float32), display_gray.shape):
            continue

        projected = cv2.perspectiveTransform(points_source.reshape(-1, 1, 2), refined_source_to_unrotated).reshape(-1, 2)
        residual = float(np.mean(np.linalg.norm(projected - expected_unrotated, axis=1)))
        movement = np.linalg.norm(points_final - expected_final, axis=1)
        evidence = float(sum(candidate.score for candidate in selected if not candidate.fallback))
        contrast = float(sum(max(candidate.contrast, 0.0) for candidate in selected if not candidate.fallback))
        fallback_count = 5 - detected_count
        score = (
            evidence
            + (detected_count * 10.0)
            + min(55.0, contrast * 0.25)
            - (geometry_penalty * 1.15)
            - (residual * 3.0)
            - (fallback_count * 24.0)
            - (float(np.mean(movement)) * 0.030)
        )
        if current_penalty is not None and geometry_penalty < current_penalty:
            score += min(120.0, (current_penalty - geometry_penalty) * 2.0)
        if score > best_score:
            best_score = score
            best_quad = refined_quad.astype(np.float32)
            best_h = refined_source_to_unrotated

    if best_quad is None or best_h is None:
        return None

    refined_warped = np.rot90(cv2.warpPerspective(display_gray, best_h, (TARGET_PIXELS, TARGET_PIXELS)), turns)
    refined_layout_count, refined_layout_contrast = marker_layout_quality(refined_warped, geometry)
    if refined_layout_count < current_layout_count and current_layout_count >= 3:
        return None
    if refined_layout_count == current_layout_count and refined_layout_contrast < current_layout_contrast - 1.5:
        return None
    return best_quad


def sec_refined_fallback_source_marker_quad(
    display_gray: np.ndarray,
    quad_small: np.ndarray,
    turns: int,
    geometry: TargetGeometry,
    marker_allowed_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Refit a fallback SEC target from source-space dark dots when the whole target is shifted down."""

    if marker_allowed_mask is None:
        return None
    unrotated_markers, marker_points = projected_marker_points(quad_small, turns, geometry)
    marker_span = marker_span_px(marker_points)
    if marker_span < 18.0:
        return None
    dot_radius = max(4.0, marker_span * (DOT_RADIUS_PX / TARGET_PIXELS))
    search_radius = max(52.0, min(110.0, marker_span * 0.62))
    top_up_limit = max(18.0, dot_radius * 3.5)
    column_up_limit = max(34.0, dot_radius * 6.0)
    left_right_limit = max(26.0, dot_radius * 4.5)
    right_left_limit = max(16.0, dot_radius * 3.0)
    bottom_left_up_limit = max(42.0, dot_radius * 7.0)

    candidate_lists: list[list[tuple[np.ndarray, float]]] = []
    for marker_index, predicted in enumerate(marker_points):
        candidates: list[tuple[np.ndarray, float]] = []
        for candidate, score in sec_local_marker_candidates(display_gray, predicted, dot_radius, search_radius, marker_allowed_mask):
            if marker_index in {0, 1} and float(candidate[1]) < float(predicted[1]) - top_up_limit:
                continue
            if marker_index in {0, 3} and float(candidate[0]) > float(predicted[0]) + left_right_limit:
                continue
            if marker_index in {1, 2, 4} and float(candidate[0]) < float(predicted[0]) - right_left_limit:
                continue
            if marker_index in {2, 4} and float(candidate[1]) < float(predicted[1]) - column_up_limit:
                continue
            candidates.append((candidate, score))
        if marker_index == 3:
            x0 = max(0, int(math.floor(float(predicted[0]) - search_radius)))
            y0 = max(0, int(math.floor(float(predicted[1]) - search_radius)))
            x1 = min(display_gray.shape[1], int(math.ceil(float(predicted[0]) + search_radius)))
            y1 = min(display_gray.shape[0], int(math.ceil(float(predicted[1]) + search_radius)))
            for blob in detect_dark_blobs(display_gray, (x0, y0, x1, y1), marker_allowed_mask):
                candidate = np.array([blob.x, blob.y], dtype=np.float32)
                offset = candidate - predicted
                upward = -float(offset[1])
                if upward < max(5.0, dot_radius * 0.8) or upward > bottom_left_up_limit:
                    continue
                if abs(float(offset[0])) > max(18.0, dot_radius * 3.6):
                    continue
                distance = float(np.linalg.norm(offset))
                score = (blob.darkness * 0.22) + (upward * 0.45) - (distance * 0.10)
                if any(float(np.linalg.norm(candidate - existing[0])) < max(5.0, dot_radius * 1.2) for existing in candidates):
                    continue
                candidates.append((candidate, float(score)))
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
        candidate_lists.append(candidates[:8])

    best_score = -1e12
    best_quad: np.ndarray | None = None
    best_marker_points: np.ndarray | None = None
    minimum_separation = max(6.0, dot_radius * 1.45)
    for selected in product(*candidate_lists):
        candidate_points = np.array([item[0] for item in selected], dtype=np.float32)
        if any(
            float(np.linalg.norm(candidate_points[i] - candidate_points[j])) < minimum_separation
            for i in range(5)
            for j in range(i + 1, 5)
        ):
            continue
        if not (candidate_points[1, 1] < candidate_points[2, 1] < candidate_points[4, 1]):
            continue
        right_up_shift = marker_points[[1, 2, 4], 1] - candidate_points[[1, 2, 4], 1]
        if float(np.mean(right_up_shift)) < max(6.0, dot_radius * 1.1):
            continue
        geometry_penalty = marker_pattern_geometry_penalty(candidate_points, display_gray.shape, marker_allowed_mask)
        if geometry_penalty is None:
            continue
        refined_quad = quad_from_marker_refit(candidate_points, unrotated_markers, display_gray.shape)
        if refined_quad is None:
            continue
        right_x_range = float(np.ptp(candidate_points[[1, 2, 4], 0]))
        left_x_range = abs(float(candidate_points[3, 0] - candidate_points[0, 0]))
        movement = float(np.mean(np.linalg.norm(candidate_points - marker_points, axis=1)))
        fallback_count = sum(float(np.linalg.norm(candidate_points[index] - marker_points[index])) < 1.0 for index in range(5))
        local_score = float(sum(item[1] for item in selected))
        score = (
            local_score
            - (geometry_penalty * 0.55)
            - (movement * 0.05)
            - (fallback_count * 22.0)
            + (float(np.mean(right_up_shift)) * 0.35)
            - (right_x_range * 0.75)
            - (left_x_range * 0.45)
        )
        if score > best_score:
            best_score = score
            best_quad = refined_quad.astype(np.float32)
            best_marker_points = candidate_points.copy()

    if best_marker_points is None:
        return best_quad

    left_edge = best_marker_points[0] - best_marker_points[3]
    left_edge_length = float(np.linalg.norm(left_edge))
    if left_edge_length < 8.0:
        return best_quad
    adjusted_markers = best_marker_points.copy()
    bottom_left_bias = max(5.0, min(12.0, dot_radius * 1.8))
    adjusted_markers[3] = adjusted_markers[3] + (left_edge / left_edge_length) * bottom_left_bias
    adjusted_quad = quad_from_marker_refit(adjusted_markers, unrotated_markers, display_gray.shape)
    return best_quad if adjusted_quad is None else adjusted_quad


def sec_refined_top_right_quad(
    display_gray: np.ndarray,
    quad_small: np.ndarray,
    turns: int,
    geometry: TargetGeometry,
    marker_allowed_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Snap the SEC top-right marker to a nearby compact dark dot without shifting the right column."""

    if marker_allowed_mask is None:
        return None
    unrotated_markers, marker_points = projected_marker_points(quad_small, turns, geometry)
    marker_span = marker_span_px(marker_points)
    if marker_span < 18.0:
        return None
    dot_radius = max(4.0, marker_span * (DOT_RADIUS_PX / TARGET_PIXELS))
    search_radius = max(34.0, min(72.0, marker_span * 0.42))
    candidates = sec_local_marker_candidates(display_gray, marker_points[1], dot_radius, search_radius, marker_allowed_mask)
    current_penalty = marker_pattern_geometry_penalty(marker_points, display_gray.shape, marker_allowed_mask)

    best_score = -1e12
    best_quad: np.ndarray | None = None
    right_column_axis = marker_points[4] - marker_points[1]
    right_column_length = float(np.linalg.norm(right_column_axis))
    if right_column_length < 8.0:
        return None
    right_column_axis = right_column_axis / right_column_length
    for candidate, candidate_score in candidates:
        movement_vector = candidate - marker_points[1]
        movement = float(np.linalg.norm(movement_vector))
        if movement < 1.0 or movement > max(14.0, dot_radius * 2.9):
            continue
        if candidate_score < 10.0:
            continue
        if float(candidate[1]) < float(marker_points[1, 1]) - max(6.0, dot_radius * 1.0):
            continue
        if float(candidate[1]) > float(marker_points[1, 1]) + max(12.0, dot_radius * 2.0):
            continue
        if float(candidate[0]) < float(marker_points[1, 0]) - max(10.0, dot_radius * 1.8):
            continue
        along_column = float(np.dot(movement_vector, right_column_axis))
        if along_column > max(12.0, dot_radius * 2.0):
            continue
        refined_markers = marker_points.copy()
        refined_markers[1] = candidate
        geometry_penalty = marker_pattern_geometry_penalty(refined_markers, display_gray.shape, marker_allowed_mask)
        if geometry_penalty is None:
            continue
        refined_quad = quad_from_marker_refit(refined_markers, unrotated_markers, display_gray.shape)
        if refined_quad is None:
            continue
        score = float(candidate_score) - (movement * 0.22) - (geometry_penalty * 0.35)
        if current_penalty is not None and geometry_penalty < current_penalty:
            score += min(24.0, (current_penalty - geometry_penalty) * 1.1)
        if score > best_score:
            best_score = score
            best_quad = refined_quad.astype(np.float32)

    return best_quad


def sec_refined_top_left_quad(
    display_gray: np.ndarray,
    quad_small: np.ndarray,
    turns: int,
    geometry: TargetGeometry,
    marker_allowed_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Snap only the SEC top-left marker when one strong nearby dark dot is visible."""

    if marker_allowed_mask is None:
        return None
    unrotated_markers, marker_points = projected_marker_points(quad_small, turns, geometry)
    marker_span = marker_span_px(marker_points)
    if marker_span < 18.0:
        return None
    dot_radius = max(4.0, marker_span * (DOT_RADIUS_PX / TARGET_PIXELS))
    search_radius = max(45.0, min(80.0, marker_span * 0.50))
    candidates = sec_local_marker_candidates(display_gray, marker_points[0], dot_radius, search_radius, marker_allowed_mask)

    best_score = -1e12
    best_quad: np.ndarray | None = None
    current_penalty = marker_pattern_geometry_penalty(marker_points, display_gray.shape, marker_allowed_mask)
    for candidate, candidate_score in candidates:
        movement = float(np.linalg.norm(candidate - marker_points[0]))
        if movement < 1.0:
            continue
        if candidate_score < 10.0:
            continue
        if movement > max(18.0, dot_radius * 3.6):
            continue
        if float(candidate[1]) < float(marker_points[0, 1]) - max(2.0, dot_radius * 0.25):
            continue
        if float(candidate[1]) > float(marker_points[0, 1]) + max(18.0, dot_radius * 3.2):
            continue
        refined_markers = marker_points.copy()
        refined_markers[0] = candidate
        geometry_penalty = marker_pattern_geometry_penalty(refined_markers, display_gray.shape, marker_allowed_mask)
        if geometry_penalty is None:
            continue
        refined_quad = quad_from_marker_refit(refined_markers, unrotated_markers, display_gray.shape)
        if refined_quad is None:
            continue
        downward = max(0.0, float(candidate[1] - marker_points[0, 1]))
        score = float(candidate_score) - (movement * 0.15) - (geometry_penalty * 0.40) + (downward * 0.80)
        if current_penalty is not None and geometry_penalty < current_penalty:
            score += min(30.0, (current_penalty - geometry_penalty) * 1.2)
        if score > best_score:
            best_score = score
            best_quad = refined_quad.astype(np.float32)

    return best_quad


def detect_target(display_gray: np.ndarray, geometry: TargetGeometry, max_dim: int, exposure_kind: str | None = None) -> DetectionResult:
    """Find the target square and its orientation in one source image."""

    small_gray, scale = resize_for_detection(display_gray, max_dim=max_dim)
    marker_allowed_mask = sec_marker_allowed_mask(small_gray) if exposure_kind == "SEC" else None
    score_pixels = 700
    score_geometry = build_geometry(score_pixels)
    score_destination = np.array(
        [[0.0, 0.0], [score_pixels - 1.0, 0.0], [score_pixels - 1.0, score_pixels - 1.0], [0.0, score_pixels - 1.0]],
        dtype=np.float32,
    )
    best_score = -1e12
    best_quad_small: np.ndarray | None = None
    best_turns = 0
    fallback_score = -1e12
    fallback_quad_small: np.ndarray | None = None
    fallback_turns = 0

    candidate_count = 0
    for quad_small in candidate_quads(small_gray, geometry.marker_centers, marker_allowed_mask):
        if not source_quad_is_plausible(quad_small, small_gray.shape):
            continue
        if candidate_count >= 1200:
            break
        candidate_count += 1
        h_small = cv2.getPerspectiveTransform(quad_small.astype(np.float32), score_destination)
        warped_small = cv2.warpPerspective(small_gray, h_small, (score_pixels, score_pixels))
        score, turns = score_warp(warped_small, score_geometry)
        if marker_allowed_mask is not None:
            marker_points_small = candidate_marker_points_source(quad_small, turns, score_geometry, score_destination, score_pixels)
            if not sec_candidate_markers_are_plausible(marker_points_small, marker_allowed_mask):
                continue
        rotated_warp = np.rot90(warped_small, turns)
        layout_count, layout_contrast = marker_layout_quality(rotated_warp, score_geometry)
        surface_mean, surface_std = target_surface_quality(rotated_warp, score_geometry)
        area_fraction = float(abs(cv2.contourArea(quad_small.astype(np.float32)))) / float(small_gray.shape[0] * small_gray.shape[1])
        # SEC light spots can wash out dot contrast, so avoid preferring a small
        # bright patch over the full convex target when falling back to surface score.
        sec_area_bonus = area_fraction * (5000.0 if exposure_kind == "SEC" else 0.0)
        sec_fallback_area_bonus = area_fraction * (12000.0 if exposure_kind == "SEC" else 0.0)
        if layout_count >= 4:
            score += (layout_count * 110.0) + (layout_contrast * 1.5) + (surface_mean * 0.25) + sec_area_bonus
            if score > best_score:
                best_score = score
                best_quad_small = quad_small
                best_turns = turns
            continue
        if surface_mean >= 70.0 and area_fraction >= 0.010:
            score += (
                (layout_count * 60.0)
                + (layout_contrast * 1.0)
                + (surface_mean * 1.2)
                - (surface_std * 0.20)
                + sec_fallback_area_bonus
            )
            if score > fallback_score:
                fallback_score = score
                fallback_quad_small = quad_small
                fallback_turns = turns

    used_fallback_target = False
    if best_quad_small is None:
        best_quad_small = fallback_quad_small
        best_turns = fallback_turns
        best_score = fallback_score
        used_fallback_target = True
    if best_quad_small is None:
        raise RuntimeError("Failed to locate a convex target with a valid marker pattern or bright target surface.")

    if exposure_kind == "SEC":
        source_marker_refined = False
        if used_fallback_target:
            refined_quad_small = sec_refined_fallback_source_marker_quad(
                small_gray,
                best_quad_small,
                best_turns,
                geometry,
                marker_allowed_mask,
            )
            if refined_quad_small is not None:
                best_quad_small = refined_quad_small
                best_score += 90.0
                source_marker_refined = True
        if not source_marker_refined:
            refined_quad_small = sec_refined_right_column_quad(small_gray, best_quad_small, best_turns, geometry, marker_allowed_mask)
            if refined_quad_small is not None:
                best_quad_small = refined_quad_small
                best_score += 75.0
            refined_quad_small = sec_refined_left_column_quad(small_gray, best_quad_small, best_turns, geometry, marker_allowed_mask)
            if refined_quad_small is not None:
                best_quad_small = refined_quad_small
                best_score += 45.0
            refined_quad_small = sec_refined_top_right_quad(small_gray, best_quad_small, best_turns, geometry, marker_allowed_mask)
            if refined_quad_small is not None:
                best_quad_small = refined_quad_small
                best_score += 24.0
            refined_quad_small = sec_refined_top_left_quad(small_gray, best_quad_small, best_turns, geometry, marker_allowed_mask)
            if refined_quad_small is not None:
                best_quad_small = refined_quad_small
                best_score += 30.0
                source_marker_refined = True
        if not source_marker_refined:
            refined_quad_small = sec_refined_warped_marker_quad(small_gray, best_quad_small, best_turns, geometry, marker_allowed_mask)
            if refined_quad_small is not None:
                best_quad_small = refined_quad_small
                best_score += 65.0

    source_quad = best_quad_small / scale
    source_to_unrotated_target = cv2.getPerspectiveTransform(source_quad.astype(np.float32), TARGET_CORNERS)
    unrotated_marker_points = inverse_rotate_points_ccw(geometry.marker_centers, best_turns)
    target_to_source = np.linalg.inv(source_to_unrotated_target)
    marker_points_source = cv2.perspectiveTransform(unrotated_marker_points.reshape(-1, 1, 2), target_to_source).reshape(-1, 2)
    source_corners_from_final = cv2.perspectiveTransform(
        inverse_rotate_points_ccw(TARGET_CORNERS, best_turns).reshape(-1, 1, 2),
        target_to_source,
    ).reshape(-1, 2)
    return DetectionResult(
        source_quad=source_corners_from_final.astype(np.float32),
        marker_points_source=marker_points_source.astype(np.float32),
        marker_centers_warped=geometry.marker_centers.copy(),
        score=float(best_score),
        turns_ccw=int(best_turns),
        source_to_unrotated_target=source_to_unrotated_target,
    )


def warp_with_detection(image: np.ndarray, detection: DetectionResult) -> np.ndarray:
    """Warp a full-resolution image to the 1500 x 1500 target frame."""

    warped = cv2.warpPerspective(image, detection.source_to_unrotated_target, (TARGET_PIXELS, TARGET_PIXELS))
    return np.rot90(warped, detection.turns_ccw).copy()


def save_detection_debug(source_luminance: np.ndarray, detection: DetectionResult, out_path: Path) -> None:
    """Save a source-image debug overlay with frame and five marker locations."""

    overlay = cv2.cvtColor(normalize_for_display(source_luminance), cv2.COLOR_GRAY2RGB)
    cv2.polylines(overlay, [detection.source_quad.astype(np.int32)], True, (255, 60, 60), thickness=5)
    for label, point in zip(MARKER_NAMES, detection.marker_points_source):
        center = tuple(np.round(point).astype(int).tolist())
        cv2.circle(overlay, center, 16, (80, 255, 120), thickness=4)
        cv2.putText(overlay, label, (center[0] + 18, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 120), 2, cv2.LINE_AA)
    Image.fromarray(overlay).save(out_path)


def save_warp_debug(warped_luminance: np.ndarray, geometry: TargetGeometry, out_path: Path) -> None:
    """Save a warped-target debug preview with expected marker circles."""

    overlay = cv2.cvtColor(normalize_for_display(warped_luminance), cv2.COLOR_GRAY2RGB)
    cv2.rectangle(overlay, (0, 0), (TARGET_PIXELS - 1, TARGET_PIXELS - 1), (255, 60, 60), thickness=5)
    for label, point in zip(MARKER_NAMES, geometry.marker_centers):
        center = tuple(np.round(point).astype(int).tolist())
        cv2.circle(overlay, center, int(round(DOT_RADIUS_PX)), (80, 255, 120), thickness=5)
        cv2.putText(overlay, label, (center[0] + 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 120), 3, cv2.LINE_AA)
    Image.fromarray(overlay).save(out_path)


def warp_exposure(
    exposure: Exposure,
    geometry: TargetGeometry,
    debug_dir: Path,
    detection_max_dim: int,
    detection_override: DetectionResult | None = None,
) -> WarpedExposure:
    """Load, detect, warp, normalize, and write debug images for one exposure."""

    luminance, detection_luminance, metadata, saturated_fraction, raw_stats = load_raw_luminance(exposure.path)
    # Detection works from the robust preview image, but the unnormalized raw
    # luminance is what gets warped and analyzed for all reported values.
    display_gray = normalize_for_display(detection_luminance)
    detection = detection_override or detect_target(display_gray, geometry, max_dim=detection_max_dim, exposure_kind=exposure.kind)
    warped_luminance = warp_with_detection(luminance, detection).astype(np.float32)
    warped_detection_luminance = warp_with_detection(detection_luminance, detection).astype(np.float32)
    normalized_luminance = photometric_normalize(warped_luminance, metadata)
    save_detection_debug(detection_luminance, detection, debug_dir / f"{exposure.source_id}_detected.png")
    save_warp_debug(warped_detection_luminance, geometry, debug_dir / f"{exposure.source_id}_warped.png")
    return WarpedExposure(
        exposure=exposure,
        metadata=metadata,
        luminance=warped_luminance,
        normalized_luminance=normalized_luminance,
        detection=detection,
        saturated_fraction=saturated_fraction,
        measured_luminance_source=MEASURED_LUMINANCE_SOURCE,
        luminance_definition=MEASURED_LUMINANCE_DESCRIPTION,
        target_detection_source=TARGET_DETECTION_DESCRIPTION,
        raw_sensor_stats=raw_stats,
    )
