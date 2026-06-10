"""Input discovery, EXIF parsing, and CR2-to-linear-image loading.

Measured values come from the camera raw mosaic: each black-subtracted 2x2
Bayer cell is averaged as (R + G1 + G2 + B) / 4 and expanded back to full
resolution. There is no white balance, demosaic tone curve, auto brightness,
gamma, or human-eye weighting.
A separate camera-rendered preview is used only to find the target corners,
because that was the robust detector input before the module split. It does not
feed reported peak, heatmap, power, or calibration values.

Saturated raw samples are measured and reported as diagnostics. They remain
clipped at the sensor white level because a single clipped exposure cannot
recover the missing above-white signal.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import exifread
import numpy as np
import rawpy

from analysis_model import *
from analysis_utils import sanitize_id


def parse_fraction(value: object | None) -> float | None:
    """Parse EXIF rational values such as 1/250 into floats."""

    if value is None:
        return None
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            denominator_value = float(denominator)
            return None if denominator_value == 0.0 else float(numerator) / denominator_value
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def read_camera_metadata(path: Path) -> CameraMetadata:
    """Read ISO, exposure time, f-number, and orientation from a CR2 file."""

    with path.open("rb") as handle:
        tags = exifread.process_file(handle, details=False)
    iso = parse_fraction(tags.get("EXIF ISOSpeedRatings"))
    exposure_s = parse_fraction(tags.get("EXIF ExposureTime"))
    f_number = parse_fraction(tags.get("EXIF FNumber"))
    orientation = str(tags.get("Image Orientation", ""))
    missing = []
    if iso is None:
        missing.append("ISO")
    if exposure_s is None:
        missing.append("ExposureTime")
    if f_number is None:
        missing.append("FNumber")
    if missing:
        raise RuntimeError(f"{path.name}: missing required EXIF fields: {', '.join(missing)}")
    return CameraMetadata(iso=float(iso), exposure_s=float(exposure_s), f_number=float(f_number), orientation=orientation)


def apply_orientation(image: np.ndarray, orientation: str) -> np.ndarray:
    """Apply the EXIF orientation so all images share the same pixel orientation."""

    if "90 CCW" in orientation:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if "90 CW" in orientation:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if "180" in orientation:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def green_color_indices(raw: rawpy.RawPy) -> set[int]:
    """Return rawpy CFA color indices that describe green photosites."""

    description = raw.color_desc.decode(errors="ignore")
    return {index for index, label in enumerate(description) if label.upper() == "G"}


def black_corrected_raw_visible(raw: rawpy.RawPy) -> tuple[np.ndarray, np.ndarray]:
    """Return black-subtracted visible raw values and their CFA color-index map."""

    raw_values = raw.raw_image_visible.astype(np.float32)
    color_indices = raw.raw_colors_visible
    black_levels = np.asarray(raw.black_level_per_channel, dtype=np.float32)
    black_map = black_levels[np.clip(color_indices, 0, len(black_levels) - 1)]
    # Keep the sensor's linear raw scale. Only the camera black level is removed;
    # high values are not tone-mapped or rescaled, so saturation remains visible
    # as raw samples near the camera white level.
    corrected = np.clip(raw_values - black_map, 0.0, None)
    return corrected.astype(np.float32), color_indices


def bayer_cell_mean_luminance(corrected_raw: np.ndarray) -> np.ndarray:
    """Average each 2x2 Bayer cell and expand the cell means to full resolution."""

    height, width = corrected_raw.shape
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    if even_height < 2 or even_width < 2:
        return corrected_raw.astype(np.float32)

    cells = corrected_raw[:even_height, :even_width].reshape(even_height // 2, 2, even_width // 2, 2)
    cell_mean = cells.mean(axis=(1, 3), dtype=np.float32)
    expanded = np.repeat(np.repeat(cell_mean, 2, axis=0), 2, axis=1)

    # Raw visible dimensions are normally even, but preserve shape if a camera
    # crop ever leaves a final row/column outside complete Bayer cells.
    if expanded.shape != corrected_raw.shape:
        full = np.empty_like(corrected_raw, dtype=np.float32)
        full[:even_height, :even_width] = expanded
        if even_height < height:
            full[even_height:, :even_width] = expanded[-1:, :]
        if even_width < width:
            full[:even_height, even_width:] = expanded[:, -1:]
        if even_height < height and even_width < width:
            full[even_height:, even_width:] = expanded[-1, -1]
        return full
    return expanded.astype(np.float32)


def target_detection_preview(raw: rawpy.RawPy) -> tuple[np.ndarray, float]:
    """Return a robust linear preview used only for target geometry detection."""

    preview_rgb = raw.postprocess(
        use_camera_wb=True,
        no_auto_bright=True,
        output_bps=16,
        gamma=(1.0, 1.0),
        bright=1.0,
    )
    preview = np.mean(preview_rgb.astype(np.float32), axis=2)
    clip_fraction = float(np.mean(np.any(preview_rgb >= 65535, axis=2))) if preview_rgb.size else 0.0
    return preview.astype(np.float32), clip_fraction


def raw_sensor_stats(raw: rawpy.RawPy) -> dict[str, float | None]:
    """Collect raw sensor diagnostics before any scalar-image conversion."""

    raw_values = raw.raw_image_visible.astype(np.float32)
    color_indices = raw.raw_colors_visible
    green_indices = green_color_indices(raw)
    green_mask = np.isin(color_indices, list(green_indices)) if green_indices else np.zeros_like(color_indices, dtype=bool)
    rawpy_white_level = float(raw.white_level)
    camera_white_levels = getattr(raw, "camera_white_level_per_channel", None)
    if camera_white_levels is None or not camera_white_levels:
        saturation_levels = np.full_like(raw_values, rawpy_white_level, dtype=np.float32)
        saturation_level_for_summary = rawpy_white_level
    else:
        camera_white = np.asarray(camera_white_levels, dtype=np.float32)
        saturation_levels = camera_white[np.clip(color_indices, 0, len(camera_white) - 1)]
        saturation_level_for_summary = float(np.min(camera_white))
    # This is the over-exposure check: saturated_fraction reports how much of the
    # raw mosaic has reached the camera saturation level. The analysis reports this
    # but does not invent replacement values for clipped pixels.
    saturated = raw_values >= saturation_levels
    return {
        "raw_white_level": saturation_level_for_summary,
        "rawpy_white_level": rawpy_white_level,
        "camera_white_level_min": saturation_level_for_summary,
        "raw_min_value": float(np.min(raw_values)) if raw_values.size else None,
        "raw_max_value": float(np.max(raw_values)) if raw_values.size else None,
        "raw_green_max_value": float(np.max(raw_values[green_mask])) if np.any(green_mask) else None,
        "raw_saturated_fraction": float(np.mean(saturated)) if raw_values.size and saturation_level_for_summary > 0.0 else None,
        "raw_green_saturated_fraction": float(np.mean(saturated[green_mask])) if np.any(green_mask) else None,
        "target_detection_preview_clip_fraction": None,
    }


def load_raw_luminance(path: Path) -> tuple[np.ndarray, np.ndarray, CameraMetadata, float, dict[str, float | None]]:
    """Load one CR2 as a raw-bayer-mean scalar measured-intensity image."""

    metadata = read_camera_metadata(path)
    with rawpy.imread(str(path)) as raw:
        stats = raw_sensor_stats(raw)
        detection_luminance, detection_clip_fraction = target_detection_preview(raw)
        stats["target_detection_preview_clip_fraction"] = detection_clip_fraction
        corrected_raw, _ = black_corrected_raw_visible(raw)
        luminance = bayer_cell_mean_luminance(corrected_raw)
    luminance = apply_orientation(luminance, metadata.orientation)
    detection_luminance = apply_orientation(detection_luminance, metadata.orientation)
    saturated_fraction = float(stats["raw_saturated_fraction"] or 0.0)
    return (
        luminance.astype(np.float32),
        detection_luminance.astype(np.float32),
        metadata,
        saturated_fraction,
        stats,
    )


def photometric_normalize(luminance: np.ndarray, metadata: CameraMetadata) -> np.ndarray:
    """Normalize linear luminance for f-number, exposure time, and ISO."""

    return luminance.astype(np.float32) * ((metadata.f_number**2) / (metadata.exposure_s * metadata.iso))


def parse_exposure_name(path: Path, input_dir: Path) -> Exposure | None:
    """Parse one filename, returning None when it is not a valid requested name."""

    stem = path.stem
    kind: str
    season: str
    is_off: bool
    variant: str | None
    case_match = CASE_NAME_RE.match(stem)
    if case_match:
        kind, season, off_variant, on_variant = case_match.groups()
        kind = kind.upper()
        season = season.upper()
        is_off = "-OFF" in stem.upper() or "_OFF" in stem.upper()
        variant = off_variant if is_off else on_variant
    else:
        nrm_match = NRM_NAME_RE.match(stem) or PREFIXED_NRM_NAME_RE.match(stem) or DASH_NRM_NAME_RE.match(stem)
        if not nrm_match:
            return None
        season, variant = nrm_match.groups()
        kind = "NRM"
        season = season.upper()
        is_off = False
    if season not in VALID_SEASONS:
        return None
    try:
        relative_path = path.relative_to(input_dir).with_suffix("")
        relative_stem = relative_path.as_posix()
        relative_parent = relative_path.parent
    except ValueError:
        relative_stem = path.stem
        relative_parent = Path(".")
    if kind == "NRM":
        nrm_stem = f"NRM_{season}" + (f"-{variant}" if variant is not None else "")
        relative_stem = nrm_stem if relative_parent == Path(".") else (relative_parent / nrm_stem).as_posix()
    return Exposure(
        path=path,
        kind=kind,
        season=season,
        is_off=is_off,
        variant=variant,
        source_id=sanitize_id(relative_stem),
    )


def discover_inputs(input_dir: Path, recursive: bool) -> list[Exposure]:
    """Find and validate CR2 inputs."""

    cr2_paths = sorted(input_dir.rglob("*.CR2") if recursive else input_dir.glob("*.CR2"))
    cr2_paths += sorted(input_dir.rglob("*.cr2") if recursive else input_dir.glob("*.cr2"))
    exposures: list[Exposure] = []
    invalid: list[Path] = []
    for path in cr2_paths:
        parsed = parse_exposure_name(path, input_dir)
        if parsed is None:
            invalid.append(path)
        else:
            exposures.append(parsed)
    if invalid:
        examples = ", ".join(str(path.relative_to(input_dir)) for path in invalid[:8])
        suffix = "" if len(invalid) <= 8 else f", and {len(invalid) - 8} more"
        raise SystemExit(
            "Invalid CR2 filename(s): "
            f"{examples}{suffix}. Expected NRM_01.CR2, REF_06.CR2, REF_06-1.CR2, "
            "REF_06_OFF.CR2, REF_06-OFF.CR2, REF_06-OFF-1.CR2, or REF_06-NRM.CR2."
        )
    return exposures
