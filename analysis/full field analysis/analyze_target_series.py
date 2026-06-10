#!/usr/bin/env python3
"""
Analyze Canon CR2 images of the 6 mm x 6 mm five-dot target.

The script does five things:
1. Validates input file names against the NRM/REF/SEC convention, NRM = Target normal to the sun, REF = Reference target, SEC = Secondary target.
2. Loads measured CR2 images as a single linear intensity image. The default is
   black-subtracted raw green photosites with no white balance, gamma, auto
   brightness, or per-image scaling; the legacy rawpy RGB path is optional.
3. Corrects exposure differences with I = P * F^2 / (E * ISO), where P is the
   selected linear scalar pixel value, F is f-number, E is exposure time in
   seconds, and ISO is the camera ISO.
4. Detects the five-dot target, warps it to the known 1500 x 1500 pixel square of the target,
   subtracts the matching OFF image when available, and calibrates measured
   fields either from REF simulation matching or from NRM images.
5. Writes heatmaps, debug images, JSON metadata, and a CSV summary.

Sigma values are estimated from the intensity-weighted second central moments
of the detected light spot after background subtraction. For an ideal Gaussian
spot, these second moments equal sigma_x and sigma_y. The values are reported in
the target x/y axes, not as rotated principal axes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

# Matplotlib must choose a non-GUI backend before pyplot is imported when the
# script is running unattended. The two preview flags deliberately keep GUI
# support available.
SHOW_PLOT_REQUESTED = "--show-plot" in sys.argv[1:] or "--preview-comparison-before-save" in sys.argv[1:]
if not SHOW_PLOT_REQUESTED:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt

from analysis_comparison import (
    apply_common_sec_aperture,
    load_comparison_layout,
    load_simulation_fields,
    save_power_vs_irradiance_outputs,
    save_simulation_comparison,
    sync_summary_encirclements,
    write_comparison_layout,
)
from analysis_detection import warp_exposure
from analysis_geometry import build_geometry
from analysis_io import discover_inputs
from analysis_measurement import (
    analysis_id_for,
    analyze_one,
    applicable_calibrations,
    build_calibration,
    build_ref_simulation_calibration,
    select_off_exposure,
    write_calibration_diagnostics,
    write_methodology,
    write_ref_match_diagnostics,
    write_summary,
)
from analysis_model import (
    Calibration,
    FieldPlotData,
    MEASURED_LUMINANCE_DESCRIPTIONS,
    SUNLIGHT_IRRADIANCE_W_M2,
    WarpedExposure,
)
from analysis_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate 6 mm x 6 mm heatmaps from Canon CR2 target images named "
            "NRM_01, REF_01, SEC_01, NRM_06, REF_06, SEC_06, with optional "
            "_OFF and -number suffixes."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("."),
        help="Directory containing the CR2 files. Default: current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_output"),
        help="Directory where heatmaps, debug images, and reports are written.",
    )
    parser.add_argument(
        "--simulation-dir",
        type=Path,
        default=Path("simulation_results"),
        help="Directory containing simulation receiver CSV exports. Default: simulation_results.",
    )
    parser.add_argument(
        "--simulation-rotation-ccw",
        type=int,
        default=1,
        help="Quarter turns applied to simulation CSV fields before comparison. Default: 1.",
    )
    parser.add_argument(
        "--comparison-gamma",
        type=float,
        default=1.0,
        help="Power-law color normalization for the shared comparison plot. Use 1.0 for a linear colorbar. Default: 1.0.",
    )
    parser.add_argument(
        "--comparison-layout",
        type=Path,
        default=None,
        help="Optional JSON file overriding comparison plot layout, font, annotation, and colorbar settings.",
    )
    parser.add_argument(
        "--measurement-calibration",
        choices=("ref-simulation", "nrm", "none"),
        default="ref-simulation",
        help=(
            "How measured REF/SEC images are scaled to W/m^2. "
            "ref-simulation matches a real REF statistic to the matching REF simulation. Default: ref-simulation."
        ),
    )
    parser.add_argument(
        "--measured-luminance-source",
        choices=tuple(MEASURED_LUMINANCE_DESCRIPTIONS),
        default="raw-green",
        help=(
            "Scalar image source for measured CR2 files. raw-green uses black-subtracted raw green photosites with fixed "
            "local interpolation at red/blue positions and no white balance/demosaic tone scaling. raw-photosite uses the "
            "black-subtracted Bayer mosaic directly. postprocess-rgb keeps the legacy rawpy RGB path. Default: raw-green."
        ),
    )
    parser.add_argument(
        "--ref-calibration-season",
        choices=("06", "01", "mean"),
        default="06",
        help=(
            "REF case used for ref-simulation measurement scaling. "
            "Use mean to average all available REF_01/REF_06 ratios. Default: 06."
        ),
    )
    parser.add_argument(
        "--ref-calibration-stat",
        choices=("total-power", "peak", "encircled-mean"),
        default="total-power",
        help=(
            "Statistic used for ref-simulation measurement scaling. "
            "total-power matches measured REF total target power to simulated REF total target power. "
            "peak matches the measured REF peak to the simulated REF peak. "
            "encircled-mean uses the previous aperture mean method. Default: total-power."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively. Invalid CR2 names in subfolders will also raise errors.",
    )
    parser.add_argument(
        "--solar-irradiance",
        type=float,
        default=SUNLIGHT_IRRADIANCE_W_M2,
        help="Irradiance assigned to the NRM target mean. Default: 800 W/m^2.",
    )
    parser.add_argument(
        "--detection-max-dim",
        type=int,
        default=1000,
        help="Maximum image dimension used during target detection. Default: 1000.",
    )
    parser.add_argument(
        "--smooth-sigma-px",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma in warped pixels before peak/sigma analysis. Default: 3 px.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show generated matplotlib figures in popup windows in addition to writing PNG files.",
    )
    parser.add_argument(
        "--preview-comparison-before-save",
        action="store_true",
        help=(
            "Open the final comparison figure before saving simulation_comparison.png. "
            "Use the Matplotlib toolbar to adjust subplot spacing, then close the window to save."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the full analysis pipeline."""

    args = parse_args()

    # Resolve paths and write the layout file first so every run records the
    # exact comparison-plot settings used for later adjustment/reproduction.
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    simulation_dir = args.simulation_dir
    if not simulation_dir.is_absolute():
        simulation_dir = (input_dir / simulation_dir).resolve()
    ensure_dir(output_dir)
    comparison_layout = load_comparison_layout(args.comparison_layout)
    write_comparison_layout(comparison_layout, output_dir)
    debug_dir = output_dir / "debug"
    ensure_dir(debug_dir)

    # Build the target masks once and classify every CR2 filename into its role:
    # ON measurement, optional OFF background, or NRM calibration image.
    geometry = build_geometry()
    exposures = discover_inputs(input_dir, recursive=args.recursive)
    on_exposures = [item for item in exposures if item.kind in {"REF", "SEC"} and not item.is_off]
    off_exposures = [item for item in exposures if item.kind in {"REF", "SEC"} and item.is_off]
    nrm_exposures = [item for item in exposures if item.kind == "NRM"]

    if not on_exposures:
        raise SystemExit("No REF_ or SEC_ ON images were found. At least one valid REF_/SEC_ CR2 file is required.")

    duplicate_off_keys = {}
    for off in off_exposures:
        key = off.on_key
        duplicate_off_keys.setdefault(key, []).append(off)
    duplicated = [items for items in duplicate_off_keys.values() if len(items) > 1]
    if duplicated:
        names = ", ".join(", ".join(item.path.name for item in group) for group in duplicated)
        raise SystemExit(f"Multiple OFF images map to the same ON variant: {names}")

    off_by_exact_key = {off.on_key: off for off in off_exposures if off.variant is not None}
    off_by_base_key = {off.base_key: off for off in off_exposures if off.variant is None}
    selected_off = {on.path: select_off_exposure(on, off_by_exact_key, off_by_base_key) for on in on_exposures}

    # Warp every ON and NRM image independently. OFF images reuse the matching
    # ON target transform so subtraction happens in exactly the same target frame.
    warped_cache: dict[Path, WarpedExposure] = {}
    primary_paths = {item.path: item for item in on_exposures + nrm_exposures}
    for exposure in sorted(primary_paths.values(), key=lambda item: item.path.name):
        print(f"Warping {exposure.path.name} ...", flush=True)
        warped_cache[exposure.path] = warp_exposure(
            exposure,
            geometry,
            debug_dir,
            detection_max_dim=args.detection_max_dim,
            measured_luminance_source=args.measured_luminance_source,
        )
    for on in sorted(on_exposures, key=lambda item: (item.kind, item.season, item.variant or "", item.path.name)):
        off = selected_off[on.path]
        if off is None or off.path in warped_cache:
            continue
        print(f"Warping {off.path.name} with {on.path.name} target transform ...", flush=True)
        warped_cache[off.path] = warp_exposure(
            off,
            geometry,
            debug_dir,
            detection_max_dim=args.detection_max_dim,
            measured_luminance_source=args.measured_luminance_source,
            detection_override=warped_cache[on.path].detection,
        )

    # Calibration can come from NRM images or from matching measured REF fields
    # to simulation. NRM diagnostics are written even when REF-simulation scaling
    # is selected, because they are useful for exposure/saturation checks.
    nrm_calibrations = [
        build_calibration(warped_cache[nrm.path], geometry, args.solar_irradiance)
        for nrm in sorted(nrm_exposures, key=lambda item: item.path.name)
    ]
    write_calibration_diagnostics(nrm_calibrations, output_dir)
    simulation_fields = load_simulation_fields(simulation_dir, args.simulation_rotation_ccw)

    measurement_calibration: Calibration | None = None
    if args.measurement_calibration == "ref-simulation":
        measurement_calibration, ref_match_rows = build_ref_simulation_calibration(
            on_exposures,
            selected_off,
            warped_cache,
            simulation_fields,
            geometry,
            args.smooth_sigma_px,
            args.ref_calibration_season,
            args.ref_calibration_stat,
        )
        write_ref_match_diagnostics(ref_match_rows, output_dir)
        if measurement_calibration is None:
            print("REF-simulation calibration unavailable; falling back to NRM calibration if possible.", flush=True)
    rows: list[dict[str, object]] = []
    measurement_fields: dict[tuple[str, str], FieldPlotData] = {}

    # Analyze each ON image with the requested calibration choice. The returned
    # summary row feeds summary.csv, and calibrated fields feed comparison plots.
    for on in sorted(on_exposures, key=lambda item: (item.kind, item.season, item.variant or "", item.path.name)):
        if measurement_calibration is not None:
            possible_calibrations = [measurement_calibration]
            used_single_nrm_fallback = False
        elif args.measurement_calibration == "none":
            possible_calibrations = [None]
            used_single_nrm_fallback = False
        else:
            possible_calibrations, used_single_nrm_fallback = applicable_calibrations(on, nrm_calibrations)
            if not possible_calibrations:
                possible_calibrations = [None]
        off = selected_off[on.path]
        warped_on = warped_cache[on.path]
        warped_off = None if off is None else warped_cache[off.path]
        for calibration in possible_calibrations:
            analysis_name = analysis_id_for(on, calibration, len(possible_calibrations), used_single_nrm_fallback)
            print(f"Analyzing {analysis_name} ...", flush=True)
            analysis_output = analyze_one(
                on=warped_on,
                off=warped_off,
                calibration=calibration,
                calibration_count=len(possible_calibrations),
                used_single_nrm_fallback=used_single_nrm_fallback,
                geometry=geometry,
                output_dir=output_dir,
                smooth_sigma_px=args.smooth_sigma_px,
                show_plot=args.show_plot,
            )
            rows.append(analysis_output.summary)
            if analysis_output.field is not None:
                measurement_fields.setdefault((analysis_output.field.kind, analysis_output.field.season), analysis_output.field)

    # Apply the common SEC aperture after all fields exist, then rewrite summary
    # rows and per-case JSON so the final reported aperture matches the plots.
    common_sec_aperture = apply_common_sec_aperture([simulation_fields, measurement_fields])
    if common_sec_aperture is not None:
        print(
            "Common SEC aperture: "
            f"{common_sec_aperture.diameter_x_mm:.4g} mm x {common_sec_aperture.diameter_y_mm:.4g} mm",
            flush=True,
        )
    sync_summary_encirclements(rows, measurement_fields, output_dir)
    write_summary(rows, output_dir)

    # Comparison and power-curve outputs are skipped only when there is neither
    # simulation data nor calibrated measurement data to compare.
    if simulation_fields or measurement_fields:
        save_simulation_comparison(
            output_dir,
            simulation_fields,
            measurement_fields,
            args.comparison_gamma,
            comparison_layout,
            args.preview_comparison_before_save,
            args.show_plot,
        )
        save_power_vs_irradiance_outputs(output_dir, simulation_fields, measurement_fields, args.show_plot)

    # The text methodology travels with the generated data so later readers can
    # see the calibration mode, raw source, and aperture assumptions in one file.
    write_methodology(
        output_dir,
        solar_irradiance=args.solar_irradiance,
        measured_luminance_source=args.measured_luminance_source,
        measurement_calibration=args.measurement_calibration,
        ref_calibration_season=args.ref_calibration_season,
        ref_calibration_stat=args.ref_calibration_stat,
    )
    if args.show_plot:
        plt.show()
    print(f"Wrote {len(rows)} heatmap analysis output(s) to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
