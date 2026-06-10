"""Calibration, spot analysis, per-image plots, and text/CSV reporting.

This module assumes images have already been warped into target coordinates. It
subtracts matching OFF frames, fills printed marker regions for field metrics,
calibrates normalized camera intensity to W/m2, and writes the per-case outputs
that describe each measurement.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image

from analysis_comparison import build_field_plot_data, encirclement_dict
from analysis_geometry import build_geometry
from analysis_model import *
from analysis_utils import ensure_dir, normalize_for_display, px_to_centered_mm


def trimmed_mean(values: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> float:
    """Compute a mean after dropping extreme tails."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    low = float(np.percentile(finite, low_q))
    high = float(np.percentile(finite, high_q))
    trimmed = finite[(finite >= low) & (finite <= high)]
    if trimmed.size == 0:
        return float(np.mean(finite))
    return float(np.mean(trimmed))


def build_calibration(nrm: WarpedExposure, geometry: TargetGeometry, solar_irradiance: float) -> Calibration:
    """Build W/m^2 calibration from one NRM image."""

    reference_intensity = trimmed_mean(nrm.normalized_luminance[geometry.calibration_mask], low_q=2.0, high_q=98.0)
    if reference_intensity <= 0.0:
        raise RuntimeError(f"{nrm.exposure.path.name}: NRM reference intensity is non-positive after normalization.")
    scale = float(solar_irradiance) / reference_intensity
    return Calibration(
        nrm=nrm,
        reference_intensity=reference_intensity,
        scale_w_m2_per_intensity=scale,
        source="NRM",
        reference_case=nrm.exposure.stem,
        reference_irradiance_w_m2=float(solar_irradiance),
        reference_stat="target-mean",
        reference_units="W/m2",
    )


def select_off_exposure(on: Exposure, off_by_exact_key: dict[tuple[str, str, str | None], Exposure], off_by_base_key: dict[tuple[str, str], Exposure]) -> Exposure | None:
    """Select the correct OFF exposure for an ON image."""

    exact = off_by_exact_key.get(on.on_key)
    if exact is not None:
        return exact
    return off_by_base_key.get(on.base_key)


def applicable_calibrations(on: Exposure, calibrations: list[Calibration]) -> tuple[list[Calibration], bool]:
    """Return calibrations for an ON image and whether a single-NRM fallback was used."""

    nrm_calibrations = [calibration for calibration in calibrations if calibration.nrm is not None]
    if not nrm_calibrations:
        return [], False
    if len(nrm_calibrations) == 1:
        nrm = nrm_calibrations[0].nrm
        return nrm_calibrations, nrm is not None and nrm.exposure.season != on.season
    same_season = [
        calibration
        for calibration in nrm_calibrations
        if calibration.nrm is not None and calibration.nrm.exposure.season == on.season
    ]
    return same_season, False


def calibration_label(calibration: Calibration | None) -> str:
    """Return a short user-facing calibration description."""

    if calibration is None:
        return "Calibration: none"
    if calibration.nrm is not None:
        return f"NRM: {calibration.nrm.exposure.stem}"
    return f"Scale: {calibration.reference_case} {calibration.source}"


def robust_mad(values: np.ndarray) -> float:
    """Return robust standard-deviation estimate from median absolute deviation."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return 1.4826 * mad


def find_peak_and_sigma(image: np.ndarray, geometry: TargetGeometry, smooth_sigma_px: float) -> tuple[SpotResult, np.ndarray, np.ndarray]:
    """Find peak intensity and Gaussian-equivalent sigma values."""

    valid_values = image[geometry.analysis_mask]
    background_floor = float(np.percentile(valid_values, 5.0)) if valid_values.size else 0.0
    background_corrected = np.clip(image.astype(np.float32) - background_floor, 0.0, None)
    smoothed = cv2.GaussianBlur(background_corrected, (0, 0), smooth_sigma_px) if smooth_sigma_px > 0 else background_corrected
    masked_for_peak = np.where(geometry.analysis_mask, smoothed, -np.inf)
    peak_y, peak_x = np.unravel_index(int(np.argmax(masked_for_peak)), masked_for_peak.shape)
    peak_value = float(masked_for_peak[peak_y, peak_x])
    noise = robust_mad(smoothed[geometry.analysis_mask])
    threshold = max(3.0 * noise, 0.05 * peak_value)
    candidate_mask = (smoothed >= threshold) & geometry.analysis_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask.astype(np.uint8), connectivity=8)
    peak_label = int(labels[peak_y, peak_x]) if num_labels > 1 else 0

    centroid_x_px: float | None = None
    centroid_y_px: float | None = None
    sigma_x_px: float | None = None
    sigma_y_px: float | None = None
    component_area = 0
    spot_detected = False
    component_mask = np.zeros_like(candidate_mask, dtype=bool)

    if peak_value > 0.0 and peak_label > 0:
        component_mask = labels == peak_label
        component_area = int(stats[peak_label, cv2.CC_STAT_AREA])
        weights = np.where(component_mask, smoothed, 0.0).astype(np.float64)
        total_weight = float(weights.sum())
        if total_weight > 0.0 and component_area >= 20:
            yy, xx = np.indices(weights.shape, dtype=np.float64)
            centroid_x_px = float((xx * weights).sum() / total_weight)
            centroid_y_px = float((yy * weights).sum() / total_weight)
            variance_x = float((((xx - centroid_x_px) ** 2) * weights).sum() / total_weight)
            variance_y = float((((yy - centroid_y_px) ** 2) * weights).sum() / total_weight)
            sigma_x_px = math.sqrt(max(variance_x, 0.0))
            sigma_y_px = math.sqrt(max(variance_y, 0.0))
            spot_detected = sigma_x_px > 0.0 and sigma_y_px > 0.0

    peak_x_mm, peak_y_mm = px_to_centered_mm(float(peak_x), float(peak_y))
    sigma_x_mm = None if sigma_x_px is None else sigma_x_px * MM_PER_PIXEL
    sigma_y_mm = None if sigma_y_px is None else sigma_y_px * MM_PER_PIXEL
    result = SpotResult(
        peak_value=peak_value,
        peak_x_px=int(peak_x),
        peak_y_px=int(peak_y),
        peak_x_mm_centered=peak_x_mm,
        peak_y_mm_centered=peak_y_mm,
        centroid_x_px=centroid_x_px,
        centroid_y_px=centroid_y_px,
        sigma_x_px=sigma_x_px,
        sigma_y_px=sigma_y_px,
        sigma_x_mm=sigma_x_mm,
        sigma_y_mm=sigma_y_mm,
        spot_detected=spot_detected,
        background_floor=background_floor,
        threshold=float(threshold),
        component_area_px=component_area,
    )
    return result, smoothed.astype(np.float32), component_mask


def corrected_measurement_luminance(on: WarpedExposure, off: WarpedExposure | None) -> np.ndarray:
    """Return exposure-normalized ON minus matching OFF target luminance."""

    corrected = on.normalized_luminance.copy()
    if off is not None:
        corrected = np.clip(corrected - off.normalized_luminance, 0.0, None)
    return fill_marker_regions(corrected, build_geometry(corrected.shape[0]))


def fill_marker_regions(image: np.ndarray, geometry: TargetGeometry) -> np.ndarray:
    """Interpolate over printed target markers after OFF subtraction."""

    filled = image.astype(np.float32).copy()
    finite_values = filled[np.isfinite(filled)]
    if finite_values.size == 0:
        return filled
    fallback = float(np.percentile(finite_values, 5.0))
    filled = np.where(np.isfinite(filled), filled, fallback).astype(np.float32)
    yy, xx = np.indices(filled.shape, dtype=np.float32)
    scale = float(filled.shape[0]) / float(TARGET_PIXELS)
    inner_fill_radius = 120.0 * scale
    outer_fill_radius = 175.0 * scale
    sample_inner = 215.0 * scale
    sample_outer = 340.0 * scale
    original = filled.copy()
    for center_x, center_y in geometry.marker_centers:
        distance = np.sqrt((xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2)
        sample_mask = (
            (distance >= sample_inner)
            & (distance <= sample_outer)
            & np.isfinite(original)
            & (~geometry.marker_exclusion_mask)
        )
        values = original[sample_mask]
        if values.size < 60:
            continue
        sample_x = xx[sample_mask] - float(center_x)
        sample_y = yy[sample_mask] - float(center_y)
        design = np.column_stack([sample_x.astype(np.float64), sample_y.astype(np.float64), np.ones(values.size)])
        coefficients, *_ = np.linalg.lstsq(design, values.astype(np.float64), rcond=None)
        blend_mask = distance < outer_fill_radius
        fill_x = xx[blend_mask] - float(center_x)
        fill_y = yy[blend_mask] - float(center_y)
        estimated = coefficients[0] * fill_x + coefficients[1] * fill_y + coefficients[2]
        feather = np.clip((outer_fill_radius - distance[blend_mask]) / max(outer_fill_radius - inner_fill_radius, 1e-6), 0.0, 1.0)
        filled[blend_mask] = (
            (filled[blend_mask].astype(np.float64) * (1.0 - feather))
            + (np.clip(estimated, 0.0, None) * feather)
        ).astype(np.float32)
    return filled


def build_unscaled_measurement_field(
    on: WarpedExposure,
    off: WarpedExposure | None,
    geometry: TargetGeometry,
    smooth_sigma_px: float,
) -> FieldPlotData:
    """Build a measurement field in camera-intensity units for calibration fitting."""

    corrected = corrected_measurement_luminance(on, off)
    _, smoothed, _ = find_peak_and_sigma(corrected, geometry, smooth_sigma_px=smooth_sigma_px)
    return build_field_plot_data(
        source="measurement_unscaled",
        kind=on.exposure.kind,
        season=on.exposure.season,
        title=f"Unscaled measured {on.exposure.stem}",
        irradiance_w_m2=smoothed,
        valid_mask=np.ones_like(smoothed, dtype=bool),
        x_min_mm=-TARGET_SIZE_MM / 2.0,
        x_max_mm=TARGET_SIZE_MM / 2.0,
        y_min_mm=-TARGET_SIZE_MM / 2.0,
        y_max_mm=TARGET_SIZE_MM / 2.0,
        cell_area_m2=(MM_PER_PIXEL / 1000.0) ** 2,
    )


def build_ref_simulation_calibration(
    on_exposures: list[Exposure],
    selected_off: dict[Path, Exposure | None],
    warped_cache: dict[Path, WarpedExposure],
    simulation_fields: dict[tuple[str, str], FieldPlotData],
    geometry: TargetGeometry,
    smooth_sigma_px: float,
    requested_season: str,
    reference_stat: str,
) -> tuple[Calibration | None, list[dict[str, object]]]:
    """Create a single measurement scale by matching real REF data to simulation REF data."""

    source_names = {
        "total-power": "REF_SIMULATION_TOTAL_POWER",
        "peak": "REF_SIMULATION_PEAK",
        "encircled-mean": "REF_SIMULATION_ENCIRCLED_MEAN",
    }
    source_name = source_names[reference_stat]
    ref_exposures = sorted(
        [item for item in on_exposures if item.kind == "REF"],
        key=lambda item: (item.season, item.variant or "", item.path.name),
    )
    diagnostics: list[dict[str, object]] = []
    for ref in ref_exposures:
        simulation_ref = simulation_fields.get(("REF", ref.season))
        if simulation_ref is None:
            continue
        off = selected_off.get(ref.path)
        unscaled_field = build_unscaled_measurement_field(
            warped_cache[ref.path],
            None if off is None else warped_cache[off.path],
            geometry,
            smooth_sigma_px,
        )
        measured_mean = unscaled_field.encirclement.mean_w_m2
        simulated_mean = simulation_ref.encirclement.mean_w_m2
        measured_total = unscaled_field.total_power_kw
        simulated_total = simulation_ref.total_power_kw
        measured_peak = float(np.nanmax(unscaled_field.irradiance_w_m2)) if np.isfinite(unscaled_field.irradiance_w_m2).any() else math.nan
        simulated_peak = float(np.nanmax(simulation_ref.irradiance_w_m2)) if np.isfinite(simulation_ref.irradiance_w_m2).any() else math.nan
        if reference_stat == "total-power":
            measured_reference = measured_total
            simulated_reference = simulated_total
            reference_units = "measured integrated a.u.*m2/1000, simulation kW"
        elif reference_stat == "peak":
            measured_reference = measured_peak
            simulated_reference = simulated_peak
            reference_units = "measured a.u., simulation W/m2"
        else:
            measured_reference = measured_mean
            simulated_reference = simulated_mean
            reference_units = "measured a.u., simulation W/m2"
        scale = simulated_reference / measured_reference if measured_reference > 0.0 else math.nan
        diagnostics.append(
            {
                "case": f"REF_{ref.season}",
                "real_file": ref.path.name,
                "off_file": "" if off is None else off.path.name,
                "reference_stat": reference_stat,
                "reference_units": reference_units,
                "measured_reference_value": measured_reference,
                "simulation_reference_value": simulated_reference,
                "measured_encircled_mean_au": measured_mean,
                "simulation_encircled_mean_w_m2": simulated_mean,
                "measured_total_power_au_area": measured_total,
                "simulation_total_power_kw": simulated_total,
                "measured_peak_au": measured_peak,
                "simulation_peak_w_m2": simulated_peak,
                "scale_w_m2_per_intensity": scale,
                "selected_for_scale": False,
            }
        )

    valid_rows = [row for row in diagnostics if np.isfinite(float(row["scale_w_m2_per_intensity"])) and float(row["scale_w_m2_per_intensity"]) > 0.0]
    if requested_season != "mean":
        selected_rows = [row for row in valid_rows if row["case"] == f"REF_{requested_season}"]
        if not selected_rows:
            selected_rows = valid_rows
    else:
        selected_rows = valid_rows
    if not selected_rows:
        return None, diagnostics

    for row in selected_rows:
        row["selected_for_scale"] = True
    scales = np.array([float(row["scale_w_m2_per_intensity"]) for row in selected_rows], dtype=np.float64)
    measured_references = np.array([float(row["measured_reference_value"]) for row in selected_rows], dtype=np.float64)
    simulated_references = np.array([float(row["simulation_reference_value"]) for row in selected_rows], dtype=np.float64)
    reference_case = "+".join(str(row["case"]) for row in selected_rows)
    reference_units = str(selected_rows[0]["reference_units"]) if selected_rows else ""
    calibration = Calibration(
        nrm=None,
        reference_intensity=float(np.mean(measured_references)),
        scale_w_m2_per_intensity=float(np.mean(scales)),
        source=source_name,
        reference_case=reference_case,
        reference_irradiance_w_m2=float(np.mean(simulated_references)),
        reference_stat=reference_stat,
        reference_units=reference_units,
    )
    return calibration, diagnostics


def scale_spot_result(result: SpotResult, scale: float) -> SpotResult:
    """Scale intensity fields while leaving geometry fields unchanged."""

    return SpotResult(
        peak_value=result.peak_value * scale,
        peak_x_px=result.peak_x_px,
        peak_y_px=result.peak_y_px,
        peak_x_mm_centered=result.peak_x_mm_centered,
        peak_y_mm_centered=result.peak_y_mm_centered,
        centroid_x_px=result.centroid_x_px,
        centroid_y_px=result.centroid_y_px,
        sigma_x_px=result.sigma_x_px,
        sigma_y_px=result.sigma_y_px,
        sigma_x_mm=result.sigma_x_mm,
        sigma_y_mm=result.sigma_y_mm,
        spot_detected=result.spot_detected,
        background_floor=result.background_floor * scale,
        threshold=result.threshold * scale,
        component_area_px=result.component_area_px,
    )


def save_corrected_preview(image: np.ndarray, geometry: TargetGeometry, out_path: Path) -> None:
    """Save the corrected warped image used for analysis."""

    preview = cv2.cvtColor(normalize_for_display(np.where(geometry.plot_mask, image, np.nan)), cv2.COLOR_GRAY2RGB)
    for point in geometry.marker_centers:
        center = tuple(np.round(point).astype(int).tolist())
        cv2.circle(preview, center, int(round(115.0)), (180, 180, 180), thickness=3)
    Image.fromarray(preview).save(out_path)


def save_heatmap(
    analysis_id: str,
    heatmap_data: np.ndarray,
    geometry: TargetGeometry,
    result: SpotResult,
    unit: str,
    calibration: Calibration | None,
    off_exposure: Exposure | None,
    encirclement: EncirclementResult | None,
    total_power_kw_value: float | None,
    out_path: Path,
    show_plot: bool,
) -> None:
    """Save a 6 mm x 6 mm turbo heatmap with peak and sigma text."""

    plotted = np.where(geometry.plot_mask, heatmap_data, np.nan)
    finite = plotted[np.isfinite(plotted)]
    vmax = float(np.percentile(finite, 99.8)) if finite.size else 1.0
    vmax = max(vmax, float(np.nanmax(plotted)) if finite.size else 1.0, 1.0)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#d8d2c4")
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=180)
    image = ax.imshow(
        plotted,
        cmap=cmap,
        origin="upper",
        extent=(-TARGET_SIZE_MM / 2.0, TARGET_SIZE_MM / 2.0, TARGET_SIZE_MM / 2.0, -TARGET_SIZE_MM / 2.0),
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_xlim(-TARGET_SIZE_MM / 2.0, TARGET_SIZE_MM / 2.0)
    ax.set_ylim(TARGET_SIZE_MM / 2.0, -TARGET_SIZE_MM / 2.0)
    ax.margins(0.0)
    ax.scatter(result.peak_x_mm_centered, result.peak_y_mm_centered, s=35, c="white", marker="x", linewidths=1.6)
    if result.spot_detected and result.centroid_x_px is not None and result.centroid_y_px is not None and result.sigma_x_mm is not None and result.sigma_y_mm is not None:
        centroid_x_mm, centroid_y_mm = px_to_centered_mm(result.centroid_x_px, result.centroid_y_px)
        ellipse = Ellipse(
            (centroid_x_mm, centroid_y_mm),
            width=2.0 * result.sigma_x_mm,
            height=2.0 * result.sigma_y_mm,
            angle=0.0,
            fill=False,
            edgecolor="white",
            linewidth=1.3,
            linestyle="--",
        )
        ax.add_patch(ellipse)
    if encirclement is not None:
        aperture = Ellipse(
            (encirclement.center_x_mm, encirclement.center_y_mm),
            width=encirclement.diameter_x_mm,
            height=encirclement.diameter_y_mm,
            angle=0.0,
            fill=False,
            edgecolor="#66f2ff",
            linewidth=1.5,
            linestyle=":",
        )
        ax.add_patch(aperture)
        ax.scatter(encirclement.center_x_mm, encirclement.center_y_mm, s=20, c="#66f2ff", marker="+", linewidths=1.2)
    unit_label = "W/m^2" if unit == "W/m^2" else "normalized intensity [a.u.]"
    calibration_text = calibration_label(calibration)
    off_text = f"OFF subtracted: {off_exposure.stem}" if off_exposure is not None else "OFF subtracted: no"
    sigma_text = "spot not detected"
    if result.spot_detected and result.sigma_x_mm is not None and result.sigma_y_mm is not None:
        sigma_text = f"sigma_x={result.sigma_x_mm:.3f} mm, sigma_y={result.sigma_y_mm:.3f} mm"
    encircled_text = ""
    if encirclement is not None and unit == "W/m^2":
        reference_text = ""
        if encirclement.reference_mean_w_m2 is not None and encirclement.mean_reference_ratio is not None:
            reference_text = f"; mean/REF={encirclement.mean_reference_ratio:.3g}"
        encircled_text = (
            f"\nEncircled peak={encirclement.peak_w_m2 / 1000.0:.3g} kW/m^2; "
            f"mean={encirclement.mean_w_m2 / 1000.0:.3g} kW/m^2; "
            f"aperture={encirclement.diameter_x_mm:.3g}x{encirclement.diameter_y_mm:.3g} mm"
            f"{reference_text}; Ptotal={total_power_kw_value:.3g} kW"
        )
    ax.set_title(
        f"{analysis_id}\nPeak: {result.peak_value:.3g} {unit_label}; {sigma_text}{encircled_text}\n{calibration_text}; {off_text}",
        fontsize=14,
    )
    ax.set_xlabel("x [mm from target center]", fontsize=14)
    ax.set_ylabel("y [mm from target center]", fontsize=14)
    ax.tick_params(labelsize=13)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
    colorbar.set_label(f"Irradiance [{unit_label}]" if unit == "W/m^2" else f"Light intensity [{unit_label}]")
    colorbar.ax.tick_params(labelsize=13)
    colorbar.ax.yaxis.label.set_size(14)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    if show_plot:
        fig.show()
    else:
        plt.close(fig)


def exposure_metadata_dict(warped: WarpedExposure) -> dict[str, object]:
    """Serialize exposure metadata for JSON output."""

    return {
        "file": warped.exposure.path.name,
        "kind": warped.exposure.kind,
        "season": warped.exposure.season,
        "variant": warped.exposure.variant,
        "is_off": warped.exposure.is_off,
        "iso": warped.metadata.iso,
        "exposure_s": warped.metadata.exposure_s,
        "f_number": warped.metadata.f_number,
        "orientation": warped.metadata.orientation,
        "target_detection_score": warped.detection.score,
        "target_rotation_turns_ccw": warped.detection.turns_ccw,
        "saturated_fraction_estimate": warped.saturated_fraction,
        "measured_luminance_source": warped.measured_luminance_source,
        "luminance_definition": warped.luminance_definition,
        "target_detection_source": warped.target_detection_source,
        "raw_sensor": warped.raw_sensor_stats,
    }


def spot_result_dict(result: SpotResult, unit: str) -> dict[str, object]:
    """Serialize spot measurements for JSON and CSV output."""

    return {
        "peak_value": result.peak_value,
        "peak_unit": unit,
        "peak_x_px": result.peak_x_px,
        "peak_y_px": result.peak_y_px,
        "peak_x_mm_from_center": result.peak_x_mm_centered,
        "peak_y_mm_from_center": result.peak_y_mm_centered,
        "centroid_x_px": result.centroid_x_px,
        "centroid_y_px": result.centroid_y_px,
        "sigma_x_px": result.sigma_x_px,
        "sigma_y_px": result.sigma_y_px,
        "sigma_x_mm": result.sigma_x_mm,
        "sigma_y_mm": result.sigma_y_mm,
        "spot_detected": result.spot_detected,
        "background_floor": result.background_floor,
        "spot_threshold": result.threshold,
        "component_area_px": result.component_area_px,
    }


def analysis_id_for(on: Exposure, calibration: Calibration | None, calibration_count: int, used_single_nrm_fallback: bool) -> str:
    """Build an output identifier that avoids collisions and clarifies NRM use."""

    if calibration is None:
        return on.stem
    if calibration.nrm is not None and (calibration_count > 1 or used_single_nrm_fallback):
        return f"{on.stem}__using_{calibration.nrm.exposure.stem}"
    return on.stem


def analyze_one(
    on: WarpedExposure,
    off: WarpedExposure | None,
    calibration: Calibration | None,
    calibration_count: int,
    used_single_nrm_fallback: bool,
    geometry: TargetGeometry,
    output_dir: Path,
    smooth_sigma_px: float,
    show_plot: bool = False,
) -> AnalysisOutput:
    """Analyze one ON image with one optional OFF and one optional calibration."""

    corrected = corrected_measurement_luminance(on, off)
    raw_result, smoothed, component_mask = find_peak_and_sigma(corrected, geometry, smooth_sigma_px=smooth_sigma_px)
    if calibration is None:
        scaled_result = raw_result
        heatmap_data = smoothed
        unit = "a.u."
    else:
        scaled_result = scale_spot_result(raw_result, calibration.scale_w_m2_per_intensity)
        heatmap_data = smoothed * calibration.scale_w_m2_per_intensity
        unit = "W/m^2"

    analysis_id = analysis_id_for(on.exposure, calibration, calibration_count, used_single_nrm_fallback)
    field: FieldPlotData | None = None
    encirclement: EncirclementResult | None = None
    total_power_kw_value: float | None = None
    if calibration is not None:
        field = build_field_plot_data(
            source="measurement",
            kind=on.exposure.kind,
            season=on.exposure.season,
            title=f"Measured {analysis_id}",
            irradiance_w_m2=heatmap_data,
            valid_mask=np.ones_like(heatmap_data, dtype=bool),
            x_min_mm=-TARGET_SIZE_MM / 2.0,
            x_max_mm=TARGET_SIZE_MM / 2.0,
            y_min_mm=-TARGET_SIZE_MM / 2.0,
            y_max_mm=TARGET_SIZE_MM / 2.0,
            cell_area_m2=(MM_PER_PIXEL / 1000.0) ** 2,
        )
        encirclement = field.encirclement
        total_power_kw_value = field.total_power_kw
    analysis_dir = output_dir / analysis_id
    ensure_dir(analysis_dir)
    save_corrected_preview(corrected, geometry, analysis_dir / f"{analysis_id}_corrected_preview.png")
    save_heatmap(
        analysis_id=analysis_id,
        heatmap_data=heatmap_data,
        geometry=geometry,
        result=scaled_result,
        unit=unit,
        calibration=calibration,
        off_exposure=None if off is None else off.exposure,
        encirclement=encirclement,
        total_power_kw_value=total_power_kw_value,
        out_path=analysis_dir / f"{analysis_id}_heatmap.png",
        show_plot=show_plot,
    )
    result_json = {
        "analysis_id": analysis_id,
        "method": {
            "target_size_mm": TARGET_SIZE_MM,
            "target_pixels": TARGET_PIXELS,
            "mm_per_pixel": MM_PER_PIXEL,
            "measured_luminance_source": on.measured_luminance_source,
            "luminance_definition": on.luminance_definition,
            "target_detection_source": on.target_detection_source,
            "photometric_normalization": "I = P * F^2 / (E * ISO)",
            "sigma_definition": "intensity-weighted second central moments of the connected spot component after background subtraction; equals Gaussian sigma for an ideal Gaussian spot",
            "encirclement_definition": "aperture center is chosen to maximize integrated irradiance inside the scaled receiver aperture",
            "marker_handling": "printed marker dots are excluded from NRM calibration, peak search, and sigma calculation",
            "smoothing_sigma_px": smooth_sigma_px,
        },
        "on_image": exposure_metadata_dict(on),
        "off_image": None if off is None else exposure_metadata_dict(off),
        "off_subtraction_done": off is not None,
        "calibrated_to_irradiance": calibration is not None,
        "calibration": None
        if calibration is None
        else {
            "source": calibration.source,
            "reference_case": calibration.reference_case,
            "nrm_image": None if calibration.nrm is None else exposure_metadata_dict(calibration.nrm),
            "reference_intensity": calibration.reference_intensity,
            "reference_irradiance_w_m2": calibration.reference_irradiance_w_m2,
            "reference_stat": calibration.reference_stat,
            "reference_units": calibration.reference_units,
            "scale_w_m2_per_intensity": calibration.scale_w_m2_per_intensity,
        },
        "spot": spot_result_dict(scaled_result, unit),
        "unscaled_spot": spot_result_dict(raw_result, "a.u."),
        "total_power_kw": total_power_kw_value,
        "encirclement": None if encirclement is None else encirclement_dict(encirclement),
        "spot_component_area_px": int(np.count_nonzero(component_mask)),
    }
    with (analysis_dir / f"{analysis_id}_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result_json, handle, indent=2)
    summary = {
        "analysis_id": analysis_id,
        "on_file": on.exposure.path.name,
        "off_file": "" if off is None else off.exposure.path.name,
        "nrm_file": "" if calibration is None or calibration.nrm is None else calibration.nrm.exposure.path.name,
        "calibration_source": "" if calibration is None else calibration.source,
        "calibration_reference_case": "" if calibration is None else calibration.reference_case,
        "measured_luminance_source": on.measured_luminance_source,
        "raw_white_level": on.raw_sensor_stats.get("raw_white_level"),
        "rawpy_white_level": on.raw_sensor_stats.get("rawpy_white_level"),
        "camera_white_level_min": on.raw_sensor_stats.get("camera_white_level_min"),
        "raw_green_max_value": on.raw_sensor_stats.get("raw_green_max_value"),
        "raw_saturated_fraction": on.raw_sensor_stats.get("raw_saturated_fraction"),
        "raw_green_saturated_fraction": on.raw_sensor_stats.get("raw_green_saturated_fraction"),
        "season": on.exposure.season,
        "kind": on.exposure.kind,
        "variant": "" if on.exposure.variant is None else on.exposure.variant,
        "off_subtracted": off is not None,
        "calibrated_to_irradiance": calibration is not None,
        "peak_value": scaled_result.peak_value,
        "peak_unit": unit,
        "peak_x_mm_from_center": scaled_result.peak_x_mm_centered,
        "peak_y_mm_from_center": scaled_result.peak_y_mm_centered,
        "sigma_x_mm": scaled_result.sigma_x_mm,
        "sigma_y_mm": scaled_result.sigma_y_mm,
        "spot_detected": scaled_result.spot_detected,
        "total_power_kw": "" if total_power_kw_value is None else total_power_kw_value,
        "encircled_power_kw": "" if encirclement is None else encirclement.power_kw,
        "encircled_peak_kw_m2": "" if encirclement is None else encirclement.peak_w_m2 / 1000.0,
        "encircled_mean_kw_m2": "" if encirclement is None else encirclement.mean_w_m2 / 1000.0,
        "encircled_center_x_mm": "" if encirclement is None else encirclement.center_x_mm,
        "encircled_center_y_mm": "" if encirclement is None else encirclement.center_y_mm,
        "aperture_diameter_x_mm": "" if encirclement is None else encirclement.diameter_x_mm,
        "aperture_diameter_y_mm": "" if encirclement is None else encirclement.diameter_y_mm,
        "encircled_reference_mean_kw_m2": ""
        if encirclement is None or encirclement.reference_mean_w_m2 is None
        else encirclement.reference_mean_w_m2 / 1000.0,
        "encircled_mean_reference_ratio": "" if encirclement is None or encirclement.mean_reference_ratio is None else encirclement.mean_reference_ratio,
        "aperture_note": "" if encirclement is None else encirclement.aperture_note,
        "reference_intensity": "" if calibration is None else calibration.reference_intensity,
        "reference_irradiance_w_m2": "" if calibration is None else calibration.reference_irradiance_w_m2,
        "reference_stat": "" if calibration is None else calibration.reference_stat,
        "reference_units": "" if calibration is None else calibration.reference_units,
        "scale_w_m2_per_intensity": "" if calibration is None else calibration.scale_w_m2_per_intensity,
    }
    return AnalysisOutput(summary=summary, field=field)


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write the CSV summary."""

    fieldnames = [
        "analysis_id",
        "on_file",
        "off_file",
        "nrm_file",
        "calibration_source",
        "calibration_reference_case",
        "measured_luminance_source",
        "raw_white_level",
        "rawpy_white_level",
        "camera_white_level_min",
        "raw_green_max_value",
        "raw_saturated_fraction",
        "raw_green_saturated_fraction",
        "season",
        "kind",
        "variant",
        "off_subtracted",
        "calibrated_to_irradiance",
        "peak_value",
        "peak_unit",
        "peak_x_mm_from_center",
        "peak_y_mm_from_center",
        "sigma_x_mm",
        "sigma_y_mm",
        "spot_detected",
        "total_power_kw",
        "encircled_power_kw",
        "encircled_peak_kw_m2",
        "encircled_mean_kw_m2",
        "encircled_center_x_mm",
        "encircled_center_y_mm",
        "aperture_diameter_x_mm",
        "aperture_diameter_y_mm",
        "encircled_reference_mean_kw_m2",
        "encircled_mean_reference_ratio",
        "aperture_note",
        "reference_intensity",
        "reference_irradiance_w_m2",
        "reference_stat",
        "reference_units",
        "scale_w_m2_per_intensity",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_calibration_diagnostics(calibrations: list[Calibration], output_dir: Path) -> None:
    """Write NRM calibration values used for W/m^2 scaling."""

    fieldnames = [
        "nrm_file",
        "season",
        "iso",
        "exposure_s",
        "f_number",
        "reference_intensity",
        "scale_w_m2_per_intensity",
        "assigned_solar_irradiance_w_m2",
        "saturated_fraction_estimate",
    ]
    with (output_dir / "calibration_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for calibration in calibrations:
            nrm = calibration.nrm
            writer.writerow(
                {
                    "nrm_file": nrm.exposure.path.name,
                    "season": nrm.exposure.season,
                    "iso": nrm.metadata.iso,
                    "exposure_s": nrm.metadata.exposure_s,
                    "f_number": nrm.metadata.f_number,
                    "reference_intensity": calibration.reference_intensity,
                    "scale_w_m2_per_intensity": calibration.scale_w_m2_per_intensity,
                    "assigned_solar_irradiance_w_m2": calibration.reference_intensity * calibration.scale_w_m2_per_intensity,
                    "saturated_fraction_estimate": nrm.saturated_fraction,
                }
            )


def write_ref_match_diagnostics(rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write REF simulation-to-measurement scaling diagnostics."""

    fieldnames = [
        "case",
        "real_file",
        "off_file",
        "reference_stat",
        "reference_units",
        "measured_reference_value",
        "simulation_reference_value",
        "measured_encircled_mean_au",
        "simulation_encircled_mean_w_m2",
        "measured_total_power_au_area",
        "simulation_total_power_kw",
        "measured_peak_au",
        "simulation_peak_w_m2",
        "scale_w_m2_per_intensity",
        "selected_for_scale",
    ]
    with (output_dir / "ref_simulation_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_methodology(
    output_dir: Path,
    solar_irradiance: float,
    measurement_calibration: str,
    ref_calibration_season: str,
    ref_calibration_stat: str,
) -> None:
    """Write a short plain-text methodology note beside the outputs."""

    text = f"""Full-field target analysis methodology

Target geometry:
- Physical target size: {TARGET_SIZE_MM} mm x {TARGET_SIZE_MM} mm.
- Warped analysis frame: {TARGET_PIXELS} px x {TARGET_PIXELS} px.
- Pixel scale: {MM_PER_PIXEL:.6f} mm/px.
- Marker dot diameter: {DOT_DIAMETER_PX} px; marker regions are excluded from NRM calibration and spot statistics.
- After ON-OFF subtraction, printed marker regions are filled with feathered local plane interpolation for heatmaps and total/encircled receiver-field estimates because OFF images do not perfectly cancel the target's black-dot contrast under concentrated light.

Light intensity:
- Measured CR2 scalar source: {MEASURED_LUMINANCE_SOURCE}.
- Source definition: {MEASURED_LUMINANCE_DESCRIPTION}.
- Target detection image: {TARGET_DETECTION_DESCRIPTION}.
- Target detection uses a display-normalized copy only to compute the perspective transform. All reported peaks, heatmaps, total powers, encircled powers, and calibration values use the unnormalized measured CR2 scalar source above.
- Exposure correction uses I = P * F^2 / (E * ISO).
- Raw saturation diagnostics are reported as raw_saturated_fraction, raw_green_saturated_fraction, raw_green_max_value, and camera_white_level_min. Saturated pixels remain clipped at the camera white level; the script does not recover missing above-white signal from a single exposure.

Irradiance calibration:
- Requested measured-image calibration mode: {measurement_calibration}.
- Requested REF simulation calibration statistic: {ref_calibration_stat}.
- In ref-simulation mode with total-power selected, the real REF total target power over the 6 mm x 6 mm target is matched to the matching REF simulation total target power, then that one W/m^2-per-intensity scale is applied to all measured REF/SEC plots.
- In ref-simulation mode with peak selected, the real REF peak is matched to the matching REF simulation peak, then that one W/m^2-per-intensity scale is applied to all measured REF/SEC plots.
- In ref-simulation mode with encircled-mean selected, the previous real REF aperture mean to simulation REF aperture mean scaling is used instead.
- Requested REF simulation calibration case: {ref_calibration_season}. If that case is unavailable, the script falls back to any available REF simulation/measurement match.
- NRM diagnostics are still written when NRM files exist, but NRM is only used for measured W/m^2 scaling when --measurement-calibration nrm is requested or REF-simulation matching is unavailable.
- If no matching calibration is available, heatmaps are written in normalized intensity units.

Spot size:
- The heatmap is background-subtracted and lightly Gaussian-smoothed before peak detection.
- The spot component is the connected region around the peak above max(3*MAD noise, 5% of peak).
- sigma_x and sigma_y are intensity-weighted second central moments in the target x/y axes.
- For an ideal Gaussian light spot, these moments are the Gaussian sigma values.

Encircled aperture metrics:
- REF uses a circular aperture of {REF_APERTURE_DIAMETER_MM:.4f} mm diameter, corresponding to 0.16 m at 1:200 scale.
- SEC starts from the physical elliptical aspect ratio {SEC_APERTURE_DIAMETER_X_MM:.5f} mm x {SEC_APERTURE_DIAMETER_Y_MM:.5f} mm, corresponding to 0.47315 m x 0.31665 m at 1:200 scale.
- All SEC comparison plots use one shared same-aspect ellipse. The chosen ellipse is the largest shared aperture whose best-positioned SEC mean irradiance is at least the same-source same-season REF aperture mean for every available SEC plot; if that is impossible, the aperture with the best worst-case ratio is used.
- The aperture center is searched to maximize integrated irradiance/mean inside the aperture, not to match the intensity-weighted centroid.
- Total power is the sum of W/m^2 over valid comparison cells multiplied by the cell area and reported in kW.
- The solar irradiance setting ({solar_irradiance} W/m^2 by default) is only used when NRM calibration is requested or needed as fallback; suns are not reported in the current outputs.
- Simulation CSV fields are rotated into the measured target frame, then cropped to the physical {TARGET_SIZE_MM} mm x {TARGET_SIZE_MM} mm target window before comparison metrics are computed.
- The comparison plot uses one shared colorbar range. The default comparison color scale is linear; if --comparison-gamma is set below 1.0, the nonlinear display gamma only changes color rendering and does not change the reported irradiance, power, mean, or peak values.
- Per-case *_heatmap.png files are diagnostic plots with their own color range; use simulation_comparison.png for direct same-scale visual comparison.
- The active comparison plot layout is written to comparison_layout_used.json. Edit that JSON and pass it back with --comparison-layout to tune figure size, panel positions, font sizes, title positions, aperture info box placement, and colorbar size/position.

Power-vs-irradiance curves:
- Additional measured and simulated power-vs-irradiance plots are written to power_vs_irradiance/.
- These curves use the same aperture center found by the encircled-power analysis for each field. The aperture is scaled from tiny to the full field while preserving the field's aperture shape: circular for REF and elliptical for SEC.
- The y-value is integrated power in W; the x-value is mean irradiance inside the aperture in kW/m2.
- Open-circle markers show the actual same-source encircled REF and SEC mean-irradiance/power points. A dotted vertical guide extends from the REF marker to the matching SEC curve at the same mean irradiance.
"""
    (output_dir / "methodology.txt").write_text(text, encoding="utf-8")
