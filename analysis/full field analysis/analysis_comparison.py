"""Simulation loading, shared-aperture comparison plots, and power curves.

The measurement module produces fields in target coordinates; this module loads
matching simulation fields, applies comparable REF/SEC receiver apertures, and
writes the aggregate plots and CSVs used to compare measured and simulated
irradiance.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter

from analysis_model import *


def aperture_for_kind(kind: str) -> ApertureSpec:
    """Return the scaled receiver aperture for REF or SEC fields."""

    if kind == "REF":
        return ApertureSpec("0.8 mm circular REF aperture", REF_APERTURE_DIAMETER_MM, REF_APERTURE_DIAMETER_MM)
    if kind == "SEC":
        return ApertureSpec("scaled elliptical SEC aperture", SEC_APERTURE_DIAMETER_X_MM, SEC_APERTURE_DIAMETER_Y_MM)
    raise ValueError(f"Unsupported aperture kind: {kind}")


def pixel_center_mm(
    row: int,
    col: int,
    shape: tuple[int, int],
    x_min_mm: float,
    x_max_mm: float,
    y_min_mm: float,
    y_max_mm: float,
) -> tuple[float, float]:
    """Convert array row/column to plot coordinates in millimeters."""

    rows, cols = shape
    dx_mm = (x_max_mm - x_min_mm) / float(cols)
    dy_mm = (y_max_mm - y_min_mm) / float(rows)
    x_mm = x_min_mm + (float(col) + 0.5) * dx_mm
    y_mm = y_min_mm + (float(row) + 0.5) * dy_mm
    return x_mm, y_mm


def aperture_kernel(dx_mm: float, dy_mm: float, aperture: ApertureSpec) -> np.ndarray:
    """Build a binary aperture kernel in grid-cell units."""

    radius_x_mm = aperture.diameter_x_mm * 0.5
    radius_y_mm = aperture.diameter_y_mm * 0.5
    radius_x_px = max(1, int(math.ceil(radius_x_mm / max(dx_mm, 1e-12))))
    radius_y_px = max(1, int(math.ceil(radius_y_mm / max(dy_mm, 1e-12))))
    x_mm = np.arange(-radius_x_px, radius_x_px + 1, dtype=np.float32) * dx_mm
    y_mm = np.arange(-radius_y_px, radius_y_px + 1, dtype=np.float32) * dy_mm
    xx, yy = np.meshgrid(x_mm, y_mm)
    mask = ((xx / max(radius_x_mm, 1e-12)) ** 2) + ((yy / max(radius_y_mm, 1e-12)) ** 2) <= 1.0
    return mask.astype(np.float32)


def aperture_mask_for_center(
    shape: tuple[int, int],
    x_min_mm: float,
    x_max_mm: float,
    y_min_mm: float,
    y_max_mm: float,
    center_x_mm: float,
    center_y_mm: float,
    aperture: ApertureSpec,
) -> np.ndarray:
    """Return grid cells inside an aperture centered at the requested coordinates."""

    rows, cols = shape
    dx_mm = (x_max_mm - x_min_mm) / float(cols)
    dy_mm = (y_max_mm - y_min_mm) / float(rows)
    x_mm = x_min_mm + (np.arange(cols, dtype=np.float32) + 0.5) * dx_mm
    y_mm = y_min_mm + (np.arange(rows, dtype=np.float32) + 0.5) * dy_mm
    radius_x_mm = aperture.diameter_x_mm * 0.5
    radius_y_mm = aperture.diameter_y_mm * 0.5
    return (
        ((x_mm[None, :] - center_x_mm) / max(radius_x_mm, 1e-12)) ** 2
        + ((y_mm[:, None] - center_y_mm) / max(radius_y_mm, 1e-12)) ** 2
    ) <= 1.0


def total_power_kw(irradiance_w_m2: np.ndarray, valid_mask: np.ndarray, cell_area_m2: float) -> float:
    """Integrate irradiance over all valid plot cells and return kW."""

    valid = valid_mask & np.isfinite(irradiance_w_m2)
    if not np.any(valid):
        return 0.0
    return float(np.sum(np.clip(irradiance_w_m2[valid], 0.0, None), dtype=np.float64) * cell_area_m2 / 1000.0)


def find_max_encirclement(
    irradiance_w_m2: np.ndarray,
    valid_mask: np.ndarray,
    x_min_mm: float,
    x_max_mm: float,
    y_min_mm: float,
    y_max_mm: float,
    cell_area_m2: float,
    aperture: ApertureSpec,
    max_search_dim: int = 450,
) -> EncirclementResult:
    """Find the aperture placement with maximum integrated irradiance."""

    rows, cols = irradiance_w_m2.shape
    finite_valid = valid_mask & np.isfinite(irradiance_w_m2)
    data = np.where(finite_valid, np.clip(irradiance_w_m2, 0.0, None), 0.0).astype(np.float32)
    valid_float = finite_valid.astype(np.float32)
    scale = max(1, int(math.ceil(max(rows, cols) / float(max_search_dim))))
    if scale > 1:
        search_cols = max(1, int(round(cols / scale)))
        search_rows = max(1, int(round(rows / scale)))
        search_data = cv2.resize(data, (search_cols, search_rows), interpolation=cv2.INTER_AREA)
        search_valid = cv2.resize(valid_float, (search_cols, search_rows), interpolation=cv2.INTER_AREA)
    else:
        search_data = data
        search_valid = valid_float
        search_rows, search_cols = rows, cols

    dx_search_mm = (x_max_mm - x_min_mm) / float(search_cols)
    dy_search_mm = (y_max_mm - y_min_mm) / float(search_rows)
    kernel = aperture_kernel(dx_search_mm, dy_search_mm, aperture)
    summed = cv2.filter2D(search_data.astype(np.float64), cv2.CV_64F, kernel.astype(np.float64), borderType=cv2.BORDER_CONSTANT)
    valid_counts = cv2.filter2D(search_valid.astype(np.float64), cv2.CV_64F, kernel.astype(np.float64), borderType=cv2.BORDER_CONSTANT)
    minimum_valid = max(1.0, float(kernel.sum()) * 0.80)
    score = np.where(valid_counts >= minimum_valid, summed, -np.inf)
    if not np.isfinite(score).any():
        score = summed
    search_row, search_col = np.unravel_index(int(np.nanargmax(score)), score.shape)
    center_x_mm, center_y_mm = pixel_center_mm(search_row, search_col, score.shape, x_min_mm, x_max_mm, y_min_mm, y_max_mm)

    aperture_mask = aperture_mask_for_center(
        irradiance_w_m2.shape,
        x_min_mm,
        x_max_mm,
        y_min_mm,
        y_max_mm,
        center_x_mm,
        center_y_mm,
        aperture,
    )
    exact_mask = aperture_mask & finite_valid
    cell_count = int(np.count_nonzero(exact_mask))
    if cell_count == 0:
        return EncirclementResult(center_x_mm, center_y_mm, aperture.diameter_x_mm, aperture.diameter_y_mm, 0.0, 0.0, 0.0, 0.0, 0)
    values = np.clip(irradiance_w_m2[exact_mask], 0.0, None).astype(np.float64)
    area_m2 = float(cell_count * cell_area_m2)
    power_kw = float(values.sum() * cell_area_m2 / 1000.0)
    mean_w_m2 = float((power_kw * 1000.0) / max(area_m2, 1e-18))
    peak_w_m2 = float(values.max())
    return EncirclementResult(
        center_x_mm=center_x_mm,
        center_y_mm=center_y_mm,
        diameter_x_mm=aperture.diameter_x_mm,
        diameter_y_mm=aperture.diameter_y_mm,
        area_m2=area_m2,
        power_kw=power_kw,
        mean_w_m2=mean_w_m2,
        peak_w_m2=peak_w_m2,
        cell_count=cell_count,
    )


def scaled_sec_aperture(scale: float) -> ApertureSpec:
    """Return the SEC ellipse scaled while preserving its physical aspect ratio."""

    return ApertureSpec(
        f"adaptive SEC ellipse ({scale:.3g}x physical SEC aspect)",
        SEC_APERTURE_DIAMETER_X_MM * scale,
        SEC_APERTURE_DIAMETER_Y_MM * scale,
    )


def build_field_plot_data(
    source: str,
    kind: str,
    season: str,
    title: str,
    irradiance_w_m2: np.ndarray,
    valid_mask: np.ndarray,
    x_min_mm: float,
    x_max_mm: float,
    y_min_mm: float,
    y_max_mm: float,
    cell_area_m2: float,
) -> FieldPlotData:
    """Build a comparison-plot field with total and max-encircled metrics."""

    aperture = aperture_for_kind(kind)
    finite_valid = valid_mask & np.isfinite(irradiance_w_m2)
    clean_irradiance = np.where(finite_valid, np.clip(irradiance_w_m2, 0.0, None), np.nan).astype(np.float32)
    encirclement = find_max_encirclement(
        clean_irradiance,
        finite_valid,
        x_min_mm,
        x_max_mm,
        y_min_mm,
        y_max_mm,
        cell_area_m2,
        aperture,
    )
    return FieldPlotData(
        source=source,
        kind=kind,
        season=season,
        title=title,
        irradiance_w_m2=clean_irradiance,
        valid_mask=finite_valid,
        x_min_mm=x_min_mm,
        x_max_mm=x_max_mm,
        y_min_mm=y_min_mm,
        y_max_mm=y_max_mm,
        cell_area_m2=cell_area_m2,
        total_power_kw=total_power_kw(clean_irradiance, finite_valid, cell_area_m2),
        aperture=aperture,
        encirclement=encirclement,
    )


def parse_simulation_case(path: Path) -> tuple[str, str] | None:
    """Parse REF/SEC and _01/_06 identity from a simulation CSV filename."""

    match = SIMULATION_NAME_RE.search(path.stem)
    if not match:
        return None
    season = match.group(1) or match.group(4)
    kind = match.group(2) or match.group(3)
    if season is None or kind is None:
        return None
    return kind.upper(), season.upper()


def median_step(values: np.ndarray) -> float:
    """Return the median positive step in a monotonic coordinate vector."""

    diffs = np.diff(values.astype(np.float64))
    finite = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 0.0)])
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def orient_simulation_grid(
    irradiance: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    rotation_turns_ccw: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate simulation data into the measured target frame."""

    turns = rotation_turns_ccw % 4
    oriented = irradiance
    x_values = x_centers.astype(np.float32)
    y_values = y_centers.astype(np.float32)
    for _ in range(turns):
        old_x = x_values.copy()
        old_y = y_values.copy()
        oriented = np.rot90(oriented, k=1)
        x_values = -old_y
        y_values = old_x[::-1]
    if x_values[-1] < x_values[0]:
        x_values = x_values[::-1]
        oriented = np.fliplr(oriented)
    if y_values[-1] < y_values[0]:
        y_values = y_values[::-1]
        oriented = np.flipud(oriented)
    return oriented, x_values, y_values


def crop_to_target_window(
    irradiance: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    target_size_mm: float = TARGET_SIZE_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop simulation fields to the physical 6 mm x 6 mm target window."""

    half_size_mm = target_size_mm * 0.5
    tolerance_mm = 1e-6
    x_keep = (x_centers >= -half_size_mm - tolerance_mm) & (x_centers <= half_size_mm + tolerance_mm)
    y_keep = (y_centers >= -half_size_mm - tolerance_mm) & (y_centers <= half_size_mm + tolerance_mm)
    if not np.any(x_keep) or not np.any(y_keep):
        return irradiance, x_centers, y_centers
    cropped = irradiance[np.ix_(y_keep, x_keep)]
    return cropped, x_centers[x_keep], y_centers[y_keep]


def load_simulation_field(path: Path, rotation_turns_ccw: int) -> FieldPlotData:
    """Load one simulation receiver CSV as a comparison field."""

    parsed = parse_simulation_case(path)
    if parsed is None:
        raise ValueError(f"{path.name}: could not parse simulation case from filename.")
    kind, season = parsed
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: simulation CSV has no data rows.")
    max_row = max(int(row["row"]) for row in rows)
    max_col = max(int(row["col"]) for row in rows)
    row_count = max_row + 1
    col_count = max_col + 1
    irradiance = np.full((row_count, col_count), np.nan, dtype=np.float32)
    u_grid = np.full((row_count, col_count), np.nan, dtype=np.float32)
    v_grid = np.full((row_count, col_count), np.nan, dtype=np.float32)
    for row in rows:
        r = int(row["row"])
        c = int(row["col"])
        irradiance[r, c] = float(row["irradiance_w_m2"])
        u_grid[r, c] = float(row["u_m"]) * 1000.0
        v_grid[r, c] = float(row["v_m"]) * 1000.0

    x_centers = np.nanmedian(u_grid, axis=0)
    y_centers = np.nanmedian(v_grid, axis=1)
    if x_centers[-1] < x_centers[0]:
        x_centers = x_centers[::-1]
        irradiance = np.fliplr(irradiance)
    if y_centers[-1] < y_centers[0]:
        y_centers = y_centers[::-1]
        irradiance = np.flipud(irradiance)
    irradiance, x_centers, y_centers = orient_simulation_grid(irradiance, x_centers, y_centers, rotation_turns_ccw)
    irradiance, x_centers, y_centers = crop_to_target_window(irradiance, x_centers, y_centers)
    dx_mm = median_step(x_centers)
    dy_mm = median_step(y_centers)
    x_min_mm = float(np.nanmin(x_centers) - (dx_mm * 0.5))
    x_max_mm = float(np.nanmax(x_centers) + (dx_mm * 0.5))
    y_min_mm = float(np.nanmin(y_centers) - (dy_mm * 0.5))
    y_max_mm = float(np.nanmax(y_centers) + (dy_mm * 0.5))
    valid_mask = np.isfinite(irradiance)
    return build_field_plot_data(
        source="simulation",
        kind=kind,
        season=season,
        title=f"Simulation {kind}_{season}",
        irradiance_w_m2=irradiance,
        valid_mask=valid_mask,
        x_min_mm=x_min_mm,
        x_max_mm=x_max_mm,
        y_min_mm=y_min_mm,
        y_max_mm=y_max_mm,
        cell_area_m2=(dx_mm / 1000.0) * (dy_mm / 1000.0),
    )


def load_simulation_fields(
    simulation_dir: Path,
    rotation_turns_ccw: int,
) -> dict[tuple[str, str], FieldPlotData]:
    """Load all available simulation CSVs keyed by (kind, season)."""

    fields: dict[tuple[str, str], FieldPlotData] = {}
    if not simulation_dir.exists():
        return fields
    for path in sorted(simulation_dir.glob("*.csv")):
        parsed = parse_simulation_case(path)
        if parsed is None:
            continue
        field = load_simulation_field(path, rotation_turns_ccw)
        fields[(field.kind, field.season)] = field
    return fields


def collect_sec_reference_cases(
    field_groups: list[dict[tuple[str, str], FieldPlotData]],
) -> list[tuple[FieldPlotData, float]]:
    """Collect SEC fields with their same-source, same-season REF aperture mean."""

    cases: list[tuple[FieldPlotData, float]] = []
    for fields in field_groups:
        for season in sorted(VALID_SEASONS):
            ref = fields.get(("REF", season))
            sec = fields.get(("SEC", season))
            if ref is None or sec is None:
                continue
            reference_mean = ref.encirclement.mean_w_m2
            if reference_mean > 0.0 and np.isfinite(reference_mean):
                cases.append((sec, reference_mean))
    return cases


def common_sec_scale_limits(cases: list[tuple[FieldPlotData, float]]) -> tuple[float, float]:
    """Return SEC aperture scale limits that are valid for every SEC comparison field."""

    min_scales: list[float] = []
    max_scales: list[float] = []
    for field, _ in cases:
        rows, cols = field.irradiance_w_m2.shape
        dx_mm = abs((field.x_max_mm - field.x_min_mm) / max(float(cols), 1.0))
        dy_mm = abs((field.y_max_mm - field.y_min_mm) / max(float(rows), 1.0))
        min_diameter_mm = max(0.04, min(dx_mm, dy_mm) * 3.0)
        min_scales.append(max(min_diameter_mm / SEC_APERTURE_DIAMETER_X_MM, min_diameter_mm / SEC_APERTURE_DIAMETER_Y_MM))
        max_scales.append(
            min(
                abs(field.x_max_mm - field.x_min_mm) * 0.98 / SEC_APERTURE_DIAMETER_X_MM,
                abs(field.y_max_mm - field.y_min_mm) * 0.98 / SEC_APERTURE_DIAMETER_Y_MM,
            )
        )
    min_scale = max(min_scales, default=1.0)
    max_scale = max(min(max_scales, default=1.0), min_scale)
    return min_scale, max_scale


def evaluate_common_sec_aperture(
    cases: list[tuple[FieldPlotData, float]],
    scale: float,
) -> tuple[float, list[tuple[FieldPlotData, EncirclementResult, float]]]:
    """Evaluate one shared SEC aperture scale against every SEC/REF pair."""

    aperture = scaled_sec_aperture(scale)
    evaluated: list[tuple[FieldPlotData, EncirclementResult, float]] = []
    ratios: list[float] = []
    for sec, reference_mean in cases:
        encirclement = find_max_encirclement(
            sec.irradiance_w_m2,
            sec.valid_mask,
            sec.x_min_mm,
            sec.x_max_mm,
            sec.y_min_mm,
            sec.y_max_mm,
            sec.cell_area_m2,
            aperture,
        )
        encirclement.reference_mean_w_m2 = reference_mean
        encirclement.mean_reference_ratio = encirclement.mean_w_m2 / max(reference_mean, 1e-12)
        evaluated.append((sec, encirclement, reference_mean))
        ratios.append(encirclement.mean_reference_ratio)
    return min(ratios, default=0.0), evaluated


def apply_common_sec_aperture(field_groups: list[dict[tuple[str, str], FieldPlotData]]) -> ApertureSpec | None:
    """Apply one shared SEC ellipse to all SEC comparison plots."""

    cases = collect_sec_reference_cases(field_groups)
    if not cases:
        return None
    min_scale, max_scale = common_sec_scale_limits(cases)
    best_any: tuple[float, float, list[tuple[FieldPlotData, EncirclementResult, float]]] | None = None
    best_feasible: tuple[float, float, list[tuple[FieldPlotData, EncirclementResult, float]]] | None = None
    first_infeasible_above: float | None = None
    for scale in np.geomspace(min_scale, max_scale, 72):
        min_ratio, evaluated = evaluate_common_sec_aperture(cases, float(scale))
        if best_any is None or min_ratio > best_any[1]:
            best_any = (float(scale), min_ratio, evaluated)
        if min_ratio >= 1.0:
            best_feasible = (float(scale), min_ratio, evaluated)
            first_infeasible_above = None
        elif best_feasible is not None and first_infeasible_above is None:
            first_infeasible_above = float(scale)

    if best_feasible is not None and first_infeasible_above is not None:
        low = best_feasible[0]
        high = first_infeasible_above
        for _ in range(10):
            mid = math.sqrt(low * high)
            min_ratio, evaluated = evaluate_common_sec_aperture(cases, mid)
            if min_ratio >= 1.0:
                best_feasible = (mid, min_ratio, evaluated)
                low = mid
            else:
                high = mid

    selected = best_feasible if best_feasible is not None else best_any
    if selected is None:
        return None
    selected_scale, selected_min_ratio, selected_evaluated = selected
    aperture = scaled_sec_aperture(selected_scale)
    if selected_min_ratio >= 1.0:
        note = (
            "common SEC ellipse: largest shared same-aspect ellipse whose best-positioned mean "
            "is at least the same-source same-season REF aperture mean for every available SEC plot"
        )
    else:
        note = (
            "common SEC ellipse: no shared same-aspect ellipse reached every same-source same-season REF aperture mean; "
            "selected the scale with the best worst-case mean ratio"
        )
    for sec, encirclement, _ in selected_evaluated:
        encirclement.aperture_note = note
        sec.encirclement = encirclement
        sec.aperture = ApertureSpec("common SEC ellipse", encirclement.diameter_x_mm, encirclement.diameter_y_mm)
    return aperture


def safe_percent(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator/denominator as percent, preserving missing values."""

    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return 100.0 * numerator / denominator


def efficiency_metrics(fields: dict[tuple[str, str], FieldPlotData], season: str) -> dict[str, float | None]:
    """Compute REF-normalized efficiencies for one season and one source."""

    ref = fields.get(("REF", season))
    sec = fields.get(("SEC", season))
    total_ref = None if ref is None else ref.total_power_kw
    enc_ref = None if ref is None else ref.encirclement.power_kw
    total_sec = None if sec is None else sec.total_power_kw
    enc_sec = None if sec is None else sec.encirclement.power_kw
    return {
        "encircled_sec_vs_total_ref_pct": safe_percent(enc_sec, total_ref),
        "total_sec_vs_total_ref_pct": safe_percent(total_sec, total_ref),
        "encircled_ref_vs_total_ref_pct": safe_percent(enc_ref, total_ref),
    }


def aperture_annotation(field: FieldPlotData) -> str:
    """Build the text label placed beside each comparison subplot."""

    enc = field.encirclement
    return (
        f"Aperture:\n{enc.diameter_x_mm:.3g} x {enc.diameter_y_mm:.3g} mm\n"
        f"Mean: {enc.mean_w_m2 / 1000.0:.3g} kW/m2\n"
        f"Peak: {enc.peak_w_m2 / 1000.0:.3g} kW/m2\n"
        f"Enc P: {enc.power_kw * 1000.0:.3g} W\n"
        f"Tot P: {field.total_power_kw * 1000.0:.3g} W"
    )


def comparison_panel_title(source_label: str, kind: str) -> str:
    """Return the short title used above one comparison panel."""

    if kind == "REF":
        return f"{source_label}\nreference field"
    return f"{source_label} field\nw/ secondary concentrator"


def colorbar_upper_limit_kw_m2(vmax_kw_m2: float) -> float:
    """Round the colorbar upper limit to a clean number."""

    if vmax_kw_m2 <= 0.0:
        return 1.0
    if vmax_kw_m2 <= 10.0:
        return float(math.ceil(vmax_kw_m2))
    if vmax_kw_m2 <= 100.0:
        return float(math.ceil(vmax_kw_m2 / 10.0) * 10.0)
    if vmax_kw_m2 <= 1000.0:
        return float(math.ceil(vmax_kw_m2 / 100.0) * 100.0)
    return float(math.ceil(vmax_kw_m2 / 1000.0) * 1000.0)


def colorbar_tick_values(upper_kw_m2: float, tick_step_kw_m2: float | None) -> np.ndarray:
    """Return evenly spaced numeric colorbar ticks."""

    if tick_step_kw_m2 is None or tick_step_kw_m2 <= 0.0:
        if upper_kw_m2 <= 10.0:
            tick_step_kw_m2 = max(1.0, upper_kw_m2 / 5.0)
        elif upper_kw_m2 <= 100.0:
            tick_step_kw_m2 = 20.0
        elif upper_kw_m2 <= 1000.0:
            tick_step_kw_m2 = 200.0
        else:
            tick_step_kw_m2 = 1000.0
    tick_count = int(math.floor(upper_kw_m2 / tick_step_kw_m2)) + 1
    ticks = np.arange(tick_count, dtype=np.float64) * tick_step_kw_m2
    if ticks.size == 0 or ticks[-1] < upper_kw_m2 - (tick_step_kw_m2 * 0.25):
        ticks = np.append(ticks, upper_kw_m2)
    return ticks


def write_comparison_metrics(
    output_dir: Path,
    simulation_fields: dict[tuple[str, str], FieldPlotData],
    measurement_fields: dict[tuple[str, str], FieldPlotData],
) -> None:
    """Write all comparison plot metrics to CSV."""

    fieldnames = [
        "source",
        "case",
        "total_power_kw",
        "encircled_power_kw",
        "encircled_center_x_mm",
        "encircled_center_y_mm",
        "aperture_diameter_x_mm",
        "aperture_diameter_y_mm",
        "encircled_peak_kw_m2",
        "encircled_mean_kw_m2",
        "encircled_reference_mean_kw_m2",
        "encircled_mean_reference_ratio",
        "aperture_note",
        "encircled_sec_vs_total_ref_pct",
        "total_sec_vs_total_ref_pct",
        "encircled_ref_vs_total_ref_pct",
    ]
    rows: list[dict[str, object]] = []
    for source, fields in [("simulation", simulation_fields), ("measurement", measurement_fields)]:
        for season in sorted(VALID_SEASONS):
            metrics = efficiency_metrics(fields, season)
            for kind in ["REF", "SEC"]:
                field = fields.get((kind, season))
                if field is None:
                    continue
                enc = field.encirclement
                rows.append(
                    {
                        "source": source,
                        "case": f"{kind}_{season}",
                        "total_power_kw": field.total_power_kw,
                        "encircled_power_kw": enc.power_kw,
                        "encircled_center_x_mm": enc.center_x_mm,
                        "encircled_center_y_mm": enc.center_y_mm,
                        "aperture_diameter_x_mm": enc.diameter_x_mm,
                        "aperture_diameter_y_mm": enc.diameter_y_mm,
                        "encircled_peak_kw_m2": enc.peak_w_m2 / 1000.0,
                        "encircled_mean_kw_m2": enc.mean_w_m2 / 1000.0,
                        "encircled_reference_mean_kw_m2": ""
                        if enc.reference_mean_w_m2 is None
                        else enc.reference_mean_w_m2 / 1000.0,
                        "encircled_mean_reference_ratio": "" if enc.mean_reference_ratio is None else enc.mean_reference_ratio,
                        "aperture_note": enc.aperture_note,
                        **metrics,
                    }
                )
    with (output_dir / "simulation_comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def field_cell_size_mm(field: FieldPlotData) -> tuple[float, float]:
    """Return absolute field cell dimensions in millimeters."""

    rows, cols = field.irradiance_w_m2.shape
    return (
        abs((field.x_max_mm - field.x_min_mm) / max(float(cols), 1.0)),
        abs((field.y_max_mm - field.y_min_mm) / max(float(rows), 1.0)),
    )


def aperture_power_curve(source: str, field: FieldPlotData, points: int = 220) -> list[PowerCurvePoint]:
    """Build a same-shape aperture sweep around the field's encirclement center."""

    valid = field.valid_mask & np.isfinite(field.irradiance_w_m2)
    if not np.any(valid):
        return []
    enc = field.encirclement
    center_x_mm = enc.center_x_mm
    center_y_mm = enc.center_y_mm
    center_source = "encircled aperture center"
    rows, cols = field.irradiance_w_m2.shape
    dx_mm = (field.x_max_mm - field.x_min_mm) / float(cols)
    dy_mm = (field.y_max_mm - field.y_min_mm) / float(rows)
    x_mm = field.x_min_mm + (np.arange(cols, dtype=np.float32) + 0.5) * dx_mm
    y_mm = field.y_min_mm + (np.arange(rows, dtype=np.float32) + 0.5) * dy_mm
    radius_x_mm = max(enc.diameter_x_mm * 0.5, 1e-12)
    radius_y_mm = max(enc.diameter_y_mm * 0.5, 1e-12)
    distance_sq = (
        ((x_mm[None, :] - center_x_mm) / radius_x_mm) ** 2
        + ((y_mm[:, None] - center_y_mm) / radius_y_mm) ** 2
    )[valid]
    values = np.clip(field.irradiance_w_m2[valid], 0.0, None).astype(np.float64)
    if distance_sq.size == 0:
        return []

    order = np.argsort(distance_sq)
    distance_sq = distance_sq[order]
    values = values[order]
    cumulative_irradiance = np.cumsum(values, dtype=np.float64)
    cell_dx_mm, cell_dy_mm = field_cell_size_mm(field)
    min_scale = max(
        min(
            cell_dx_mm / max(enc.diameter_x_mm, 1e-12),
            cell_dy_mm / max(enc.diameter_y_mm, 1e-12),
        ),
        1e-9,
    )
    max_scale = max(math.sqrt(float(distance_sq[-1])) * 1.001, min_scale * 1.01, 1.0)
    scales = np.unique(np.concatenate([np.geomspace(min_scale, max_scale, max(points, 2)), np.array([1.0])]))

    curve: list[PowerCurvePoint] = []
    seen_cell_counts: set[int] = set()
    for scale in scales:
        radius_sq = float(scale) ** 2
        cell_count = int(np.searchsorted(distance_sq, radius_sq, side="right"))
        if cell_count == 0 or cell_count in seen_cell_counts:
            continue
        seen_cell_counts.add(cell_count)
        power_w = float(cumulative_irradiance[cell_count - 1] * field.cell_area_m2)
        aperture_area_m2 = float(cell_count * field.cell_area_m2)
        mean_irradiance_kw_m2 = power_w / max(aperture_area_m2, 1e-18) / 1000.0
        curve.append(
            PowerCurvePoint(
                source=source,
                kind=field.kind,
                season=field.season,
                aperture_diameter_mm=float(enc.diameter_x_mm * scale),
                aperture_area_m2=aperture_area_m2,
                mean_irradiance_kw_m2=mean_irradiance_kw_m2,
                power_w=power_w,
                cell_count=cell_count,
                center_x_mm=center_x_mm,
                center_y_mm=center_y_mm,
                center_source=center_source,
            )
        )
    return sorted(curve, key=lambda item: item.mean_irradiance_kw_m2)


def write_power_curve_csv(output_dir: Path, curves: dict[tuple[str, str, str], list[PowerCurvePoint]]) -> Path:
    """Write all power-vs-irradiance curve points to CSV."""

    fieldnames = [
        "source",
        "source_label",
        "season",
        "season_label",
        "case",
        "case_label",
        "aperture_diameter_mm",
        "aperture_area_m2",
        "mean_irradiance_kw_m2",
        "power_w",
        "cell_count",
        "center_x_mm",
        "center_y_mm",
        "center_source",
    ]
    out_path = output_dir / "power_vs_irradiance_curves.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (_, _, _), curve in sorted(curves.items()):
            for point in curve:
                writer.writerow(
                    {
                        "source": point.source,
                        "source_label": POWER_CURVE_SOURCE_LABELS.get(point.source, point.source.title()),
                        "season": point.season,
                        "season_label": SEASON_LABELS.get(point.season, point.season),
                        "case": point.kind,
                        "case_label": POWER_CURVE_CASE_LABELS.get(point.kind, point.kind),
                        "aperture_diameter_mm": point.aperture_diameter_mm,
                        "aperture_area_m2": point.aperture_area_m2,
                        "mean_irradiance_kw_m2": point.mean_irradiance_kw_m2,
                        "power_w": point.power_w,
                        "cell_count": point.cell_count,
                        "center_x_mm": point.center_x_mm,
                        "center_y_mm": point.center_y_mm,
                        "center_source": point.center_source,
                    }
                )
    return out_path


def style_power_curve_axis(ax: plt.Axes) -> None:
    """Apply the requested minimal horizontal-grid chart styling."""

    text_color = "#555555"
    grid_color = "#d8d8d8"
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=grid_color, linewidth=1.2)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=4, width=1.0, color="#9a9a9a", labelcolor=text_color, labelsize=12)
    ax.tick_params(axis="y", length=0, labelcolor=text_color, labelsize=12)
    ax.title.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)


def encircled_power_curve_marker(field: FieldPlotData | None) -> tuple[float, float] | None:
    """Return the field's actual encircled mean irradiance and power."""

    if field is None:
        return None
    enc = field.encirclement
    x_kw_m2 = enc.mean_w_m2 / 1000.0
    y_w = enc.power_kw * 1000.0
    if not (np.isfinite(x_kw_m2) and np.isfinite(y_w)):
        return None
    return float(x_kw_m2), float(y_w)


def interpolate_power_curve_at_mean(curve: list[PowerCurvePoint], mean_irradiance_kw_m2: float) -> float | None:
    """Return the plotted curve power at one mean-irradiance value."""

    if len(curve) < 2 or not np.isfinite(mean_irradiance_kw_m2):
        return None
    x_values = np.array([point.mean_irradiance_kw_m2 for point in curve], dtype=np.float64)
    y_values = np.array([point.power_w for point in curve], dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(finite) < 2:
        return None
    x_values = x_values[finite]
    y_values = y_values[finite]
    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    unique_x, unique_indices = np.unique(x_values, return_index=True)
    if unique_x.size < 2 or mean_irradiance_kw_m2 < unique_x[0] or mean_irradiance_kw_m2 > unique_x[-1]:
        return None
    unique_y = y_values[unique_indices]
    return float(np.interp(mean_irradiance_kw_m2, unique_x, unique_y))


def add_encircled_power_curve_markers(
    ax: plt.Axes,
    season: str,
    curves: dict[tuple[str, str, str], list[PowerCurvePoint]],
    fields: dict[tuple[str, str], FieldPlotData],
    source: str,
) -> None:
    """Mark REF/SEC encircled mean points and the REF-to-SEC vertical guide."""

    markers: dict[str, tuple[float, float]] = {}
    for kind in ("REF", "SEC"):
        marker = encircled_power_curve_marker(fields.get((kind, season)))
        if marker is None:
            continue
        markers[kind] = marker
        ax.scatter(
            marker[0],
            marker[1],
            s=58,
            marker="o",
            facecolor="white",
            edgecolor=POWER_CURVE_COLORS[kind],
            linewidth=2.0,
            zorder=5,
        )

    ref_marker = markers.get("REF")
    sec_curve = curves.get((source, "SEC", season), [])
    if ref_marker is None or not sec_curve:
        return
    sec_y_at_ref_mean = interpolate_power_curve_at_mean(sec_curve, ref_marker[0])
    if sec_y_at_ref_mean is None:
        return
    ax.plot(
        [ref_marker[0], ref_marker[0]],
        [ref_marker[1], sec_y_at_ref_mean],
        color="#777777",
        linewidth=1.6,
        linestyle=(0, (1.2, 2.4)),
        zorder=4,
    )


def save_power_curve_plot(
    output_dir: Path,
    source: str,
    season: str,
    curves: dict[tuple[str, str, str], list[PowerCurvePoint]],
    fields: dict[tuple[str, str], FieldPlotData],
    show_plot: bool,
) -> Path:
    """Save one measured or simulated season power-vs-irradiance plot."""

    source_label = POWER_CURVE_SOURCE_LABELS.get(source, source.title())
    season_label = SEASON_LABELS.get(season, season)
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)
    for kind in ("REF", "SEC"):
        curve = curves.get((source, kind, season), [])
        if not curve:
            continue
        ax.plot(
            [point.mean_irradiance_kw_m2 for point in curve],
            [point.power_w for point in curve],
            color=POWER_CURVE_COLORS[kind],
            linewidth=3.0,
            solid_capstyle="round",
            label=POWER_CURVE_CASE_LABELS[kind],
        )
    add_encircled_power_curve_markers(ax, season, curves, fields, source)
    ax.set_title(f"{source_label} {season_label.lower()} case: power vs irradiance", fontsize=17, pad=18)
    ax.set_xlabel("Mean irradiance [kW/m2]", fontsize=13, labelpad=10)
    ax.set_ylabel("Power [W]", fontsize=13, labelpad=10)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    style_power_curve_axis(ax)
    legend = ax.legend(
        frameon=False,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        handlelength=2.8,
        handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_color("#555555")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    filename_source = "simulated" if source == "simulation" else "measured"
    out_path = output_dir / f"{filename_source}_{season_label.lower()}_power_vs_irradiance.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show_plot:
        fig.show()
    else:
        plt.close(fig)
    return out_path


def save_power_vs_irradiance_outputs(
    output_dir: Path,
    simulation_fields: dict[tuple[str, str], FieldPlotData],
    measurement_fields: dict[tuple[str, str], FieldPlotData],
    show_plot: bool,
) -> None:
    """Write measured and simulated power-vs-irradiance plots and curve data."""

    curve_dir = output_dir / "power_vs_irradiance"
    curve_dir.mkdir(parents=True, exist_ok=True)
    curves: dict[tuple[str, str, str], list[PowerCurvePoint]] = {}
    for source, fields in [("simulation", simulation_fields), ("measurement", measurement_fields)]:
        for (kind, season), field in sorted(fields.items()):
            curve = aperture_power_curve(source, field)
            if curve:
                curves[(source, kind, season)] = curve
    if not curves:
        return
    fields_by_source = {"measurement": measurement_fields, "simulation": simulation_fields}
    for source in ("measurement", "simulation"):
        for season in ("06", "01"):
            if curves.get((source, "REF", season)) or curves.get((source, "SEC", season)):
                save_power_curve_plot(curve_dir, source, season, curves, fields_by_source[source], show_plot)
    write_power_curve_csv(curve_dir, curves)


def default_comparison_layout() -> dict[str, object]:
    """Return editable defaults for the comparison plot layout."""

    return {
        "figure": {"width": 8.1, "height": 10.8, "dpi": 180, "save_pad_inches": 0.02},
        "fonts": {
            "family": "Helvetica",
            "main_title": 18,
            "main_weight": "normal",
            "season_header": 12.5,
            "season_weight": "normal",
            "axis_title": 8.8,
            "axis_label": 7.4,
            "tick": 7.2,
            "aperture": 7.5,
            "missing": 8.0,
            "colorbar_label": 10.5,
            "colorbar_tick": 8.5,
        },
        "axis": {
            "xlabel_labelpad": 1,
            "ylabel_labelpad": 1,
            "tick_pad": 1,
            "aperture_linewidth": 1.0,
            "aperture_center_size": 16,
            "aperture_center_linewidth": 1.0,
            "ticks_mm": [-2.0, 0.0, 2.0],
        },
        "positions": {
            "main_title_xy": [0.50, 0.975],
            "summer_header_xy": [0.50, 0.918],
            "winter_header_xy": [0.50, 0.456],
            "divider": {"x0": 0.10, "x1": 0.79, "y": 0.505, "linewidth": 0.5, "color": "#b5b5b5"},
            "plot_width": 0.145,
            "plot_height": 0.109,
            "title_offset_y": 0.012,
            "box_offset_x": 0.016,
            "box_anchor_y_fraction": 0.52,
            "columns": {"simulation": 0.080, "measurement": 0.520},
            "rows": {"REF_06": 0.745, "SEC_06": 0.545, "REF_01": 0.270, "SEC_01": 0.050},
        },
        "info_box": {
            "facecolor": "#eeeeee",
            "edgecolor": "#707070",
            "text_color": "#303030",
            "bbox_alpha": 1.0,
            "bbox_pad": 3.0,
            "line_color": "#8a8a34",
            "line_width": 0.8,
        },
        "colorbar": {
            "rect": [0.875, 0.300, 0.026, 0.410],
            "tick_step_kw_m2": 1000.0,
            "labelpad": 7.0,
        },
    }


def update_nested_dict(base: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    """Recursively apply JSON layout overrides to default layout values."""

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            update_nested_dict(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value
    return base


def load_comparison_layout(layout_path: Path | None) -> dict[str, object]:
    """Load an optional comparison layout override file."""

    layout = default_comparison_layout()
    if layout_path is None:
        return layout
    with layout_path.expanduser().open(encoding="utf-8") as handle:
        overrides = json.load(handle)
    if not isinstance(overrides, dict):
        raise ValueError(f"{layout_path}: comparison layout must be a JSON object.")
    return update_nested_dict(layout, overrides)


def write_comparison_layout(layout: dict[str, object], output_dir: Path) -> None:
    """Write the active comparison layout so it can be edited and reused."""

    with (output_dir / "comparison_layout_used.json").open("w", encoding="utf-8") as handle:
        json.dump(layout, handle, indent=2)


def save_simulation_comparison(
    output_dir: Path,
    simulation_fields: dict[tuple[str, str], FieldPlotData],
    measurement_fields: dict[tuple[str, str], FieldPlotData],
    comparison_gamma: float,
    layout: dict[str, object],
    preview_before_save: bool,
    show_plot: bool,
) -> None:
    """Save the 4x2 shared-scale simulation-vs-measurement comparison plot."""

    all_fields = list(simulation_fields.values()) + list(measurement_fields.values())
    if not all_fields:
        return
    max_axis_mm = TARGET_SIZE_MM / 2.0
    finite_maxima = [
        float(np.nanmax(field.irradiance_w_m2)) / 1000.0 for field in all_fields if np.isfinite(field.irradiance_w_m2).any()
    ]
    vmax_kw_m2 = max(max(finite_maxima, default=1.0), 1.0)
    display_vmax_kw_m2 = colorbar_upper_limit_kw_m2(vmax_kw_m2)
    gamma = max(0.05, float(comparison_gamma))
    if abs(gamma - 1.0) <= 1e-9:
        color_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=display_vmax_kw_m2)
    else:
        color_norm = matplotlib.colors.PowerNorm(gamma=gamma, vmin=0.0, vmax=display_vmax_kw_m2)
    cases = [("REF", "06"), ("SEC", "06"), ("REF", "01"), ("SEC", "01")]
    figure_layout = layout["figure"]
    fonts = layout["fonts"]
    axis_layout = layout["axis"]
    positions = layout["positions"]
    info_layout = layout["info_box"]
    colorbar_layout = layout["colorbar"]
    font_family = str(fonts.get("family", "Helvetica"))
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [font_family, "Arial", "DejaVu Sans"]})
    fig = plt.figure(
        figsize=(float(figure_layout["width"]), float(figure_layout["height"])),
        dpi=int(figure_layout["dpi"]),
    )
    fig.patch.set_facecolor("white")

    main_title_xy = positions["main_title_xy"]
    fig.text(
        float(main_title_xy[0]),
        float(main_title_xy[1]),
        "Simulation vs measured irradiance",
        ha="center",
        va="center",
        fontsize=float(fonts["main_title"]),
        fontweight=str(fonts.get("main_weight", "normal")),
        fontfamily=font_family,
        color="#303030",
    )
    summer_header_xy = positions["summer_header_xy"]
    fig.text(
        float(summer_header_xy[0]),
        float(summer_header_xy[1]),
        "Summer solstice, 21.06.2023 14:17",
        ha="center",
        va="center",
        fontsize=float(fonts["season_header"]),
        fontweight=str(fonts.get("season_weight", "normal")),
        fontfamily=font_family,
        color="#303030",
    )
    winter_header_xy = positions["winter_header_xy"]
    fig.text(
        float(winter_header_xy[0]),
        float(winter_header_xy[1]),
        "Winter case, 15.01.2023 15:00",
        ha="center",
        va="center",
        fontsize=float(fonts["season_header"]),
        fontweight=str(fonts.get("season_weight", "normal")),
        fontfamily=font_family,
        color="#303030",
    )
    divider = positions["divider"]
    fig.add_artist(
        Line2D(
            [float(divider["x0"]), float(divider["x1"])],
            [float(divider["y"]), float(divider["y"])],
            transform=fig.transFigure,
            color=str(divider["color"]),
            linewidth=float(divider["linewidth"]),
        )
    )

    plot_width = float(positions["plot_width"])
    plot_height = float(positions["plot_height"])
    title_offset_y = float(positions["title_offset_y"])
    box_offset_x = float(positions["box_offset_x"])
    box_anchor_y_fraction = float(positions["box_anchor_y_fraction"])
    column_lefts = positions["columns"]
    row_bottoms = positions["rows"]
    axis_ticks = [float(value) for value in axis_layout["ticks_mm"]]
    sources = [("Simulation", "simulation", simulation_fields), ("Measured", "measurement", measurement_fields)]
    axes: list[plt.Axes] = []
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("#ffffff")
    last_image = None
    for kind, season in cases:
        row_key = f"{kind}_{season}"
        plot_bottom = float(row_bottoms[row_key])
        for source_label, source_key, fields in sources:
            plot_left = float(column_lefts[source_key])
            ax = fig.add_axes([plot_left, plot_bottom, plot_width, plot_height])
            axes.append(ax)
            field = fields.get((kind, season))
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor("white")
            ax.set_xlim(-max_axis_mm, max_axis_mm)
            ax.set_ylim(max_axis_mm, -max_axis_mm)
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            ax.margins(0.0)
            fig.text(
                plot_left - 0.006,
                plot_bottom + plot_height + title_offset_y,
                comparison_panel_title(source_label, kind),
                ha="left",
                va="bottom",
                fontsize=float(fonts["axis_title"]),
                fontfamily=font_family,
                color="#303030",
                linespacing=0.95,
            )
            ax.set_xlabel(
                "x [mm]",
                fontsize=float(fonts["axis_label"]),
                labelpad=float(axis_layout["xlabel_labelpad"]),
                fontfamily=font_family,
            )
            ax.set_ylabel(
                "y [mm]",
                fontsize=float(fonts["axis_label"]),
                labelpad=float(axis_layout["ylabel_labelpad"]),
                fontfamily=font_family,
            )
            ax.tick_params(
                labelsize=float(fonts["tick"]),
                pad=float(axis_layout["tick_pad"]),
                labelbottom=True,
                labelleft=True,
            )
            if field is None:
                ax.text(
                    0.5,
                    0.5,
                    "not available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=float(fonts["missing"]),
                    fontfamily=font_family,
                    color="#303030",
                )
                continue
            plotted = np.where(field.valid_mask, field.irradiance_w_m2 / 1000.0, np.nan)
            last_image = ax.imshow(
                plotted,
                cmap=cmap,
                origin="upper",
                extent=(field.x_min_mm, field.x_max_mm, field.y_max_mm, field.y_min_mm),
                norm=color_norm,
                interpolation="nearest",
            )
            enc = field.encirclement
            ax.add_patch(
                Ellipse(
                    (enc.center_x_mm, enc.center_y_mm),
                    width=enc.diameter_x_mm,
                    height=enc.diameter_y_mm,
                    angle=0.0,
                    fill=False,
                    edgecolor="#66f2ff",
                    linewidth=float(axis_layout["aperture_linewidth"]),
                    linestyle="--",
                )
            )
            ax.scatter(
                enc.center_x_mm,
                enc.center_y_mm,
                s=float(axis_layout["aperture_center_size"]),
                c="#66f2ff",
                marker="+",
                linewidths=float(axis_layout["aperture_center_linewidth"]),
            )
            box_x = plot_left + plot_width + box_offset_x
            box_y = plot_bottom + (plot_height * box_anchor_y_fraction)
            ax.annotate(
                aperture_annotation(field),
                xy=(enc.center_x_mm, enc.center_y_mm),
                xycoords="data",
                xytext=(box_x, box_y),
                textcoords=fig.transFigure,
                annotation_clip=False,
                clip_on=False,
                va="center",
                ha="left",
                fontsize=float(fonts["aperture"]),
                fontfamily=font_family,
                color=str(info_layout["text_color"]),
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": str(info_layout["facecolor"]),
                    "alpha": float(info_layout["bbox_alpha"]),
                    "edgecolor": str(info_layout["edgecolor"]),
                    "pad": float(info_layout["bbox_pad"]),
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": str(info_layout["line_color"]),
                    "linewidth": float(info_layout["line_width"]),
                    "shrinkA": 0.0,
                    "shrinkB": 3.0,
                },
            )
    if last_image is not None:
        colorbar_axis = fig.add_axes([float(value) for value in colorbar_layout["rect"]])
        colorbar = fig.colorbar(last_image, cax=colorbar_axis)
        colorbar.set_ticks(colorbar_tick_values(display_vmax_kw_m2, float(colorbar_layout["tick_step_kw_m2"])))
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        colorbar.set_label("Irradiance [kW/m2]", labelpad=float(colorbar_layout["labelpad"]))
        colorbar.ax.tick_params(labelsize=float(fonts["colorbar_tick"]))
        colorbar.ax.yaxis.label.set_size(float(fonts["colorbar_label"]))
        colorbar.ax.yaxis.label.set_family(font_family)
    if preview_before_save:
        print(
            "Previewing comparison plot before save. Close the window to save simulation_comparison.png.",
            flush=True,
        )
        plt.show(block=True)
    fig.savefig(output_dir / "simulation_comparison.png", pad_inches=float(figure_layout["save_pad_inches"]))
    if show_plot:
        fig.show()
    else:
        plt.close(fig)
    write_comparison_metrics(output_dir, simulation_fields, measurement_fields)


def encirclement_dict(enc: EncirclementResult) -> dict[str, object]:
    """Serialize encirclement metrics without display-only derived units."""

    return {
        "center_x_mm": enc.center_x_mm,
        "center_y_mm": enc.center_y_mm,
        "diameter_x_mm": enc.diameter_x_mm,
        "diameter_y_mm": enc.diameter_y_mm,
        "area_m2": enc.area_m2,
        "power_kw": enc.power_kw,
        "mean_w_m2": enc.mean_w_m2,
        "mean_kw_m2": enc.mean_w_m2 / 1000.0,
        "peak_w_m2": enc.peak_w_m2,
        "peak_kw_m2": enc.peak_w_m2 / 1000.0,
        "cell_count": enc.cell_count,
        "reference_mean_w_m2": enc.reference_mean_w_m2,
        "reference_mean_kw_m2": None if enc.reference_mean_w_m2 is None else enc.reference_mean_w_m2 / 1000.0,
        "mean_reference_ratio": enc.mean_reference_ratio,
        "aperture_note": enc.aperture_note,
    }


def sync_summary_encirclements(
    rows: list[dict[str, object]],
    measurement_fields: dict[tuple[str, str], FieldPlotData],
    output_dir: Path,
) -> None:
    """Update measurement summary/JSON rows after shared comparison apertures are applied."""

    for row in rows:
        field = measurement_fields.get((str(row["kind"]), str(row["season"])))
        if field is None or not row.get("calibrated_to_irradiance"):
            continue
        enc = field.encirclement
        row["total_power_kw"] = field.total_power_kw
        row["encircled_power_kw"] = enc.power_kw
        row["encircled_peak_kw_m2"] = enc.peak_w_m2 / 1000.0
        row["encircled_mean_kw_m2"] = enc.mean_w_m2 / 1000.0
        row["encircled_center_x_mm"] = enc.center_x_mm
        row["encircled_center_y_mm"] = enc.center_y_mm
        row["aperture_diameter_x_mm"] = enc.diameter_x_mm
        row["aperture_diameter_y_mm"] = enc.diameter_y_mm
        row["encircled_reference_mean_kw_m2"] = "" if enc.reference_mean_w_m2 is None else enc.reference_mean_w_m2 / 1000.0
        row["encircled_mean_reference_ratio"] = "" if enc.mean_reference_ratio is None else enc.mean_reference_ratio
        row["aperture_note"] = enc.aperture_note
        analysis_id = str(row["analysis_id"])
        result_path = output_dir / analysis_id / f"{analysis_id}_result.json"
        if not result_path.exists():
            continue
        with result_path.open(encoding="utf-8") as handle:
            result_json = json.load(handle)
        result_json["total_power_kw"] = field.total_power_kw
        result_json["encirclement"] = encirclement_dict(enc)
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result_json, handle, indent=2)
