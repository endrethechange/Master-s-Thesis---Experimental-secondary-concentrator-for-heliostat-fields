#!/usr/bin/env python3
"""Plot captured power versus mean irradiance for REF and SEC simulations.

The curve is built by placing a circular aperture on the brightest receiver
cell, then growing that aperture from tiny to large. For each aperture size the
script integrates receiver irradiance inside the aperture and plots:

    x = mean irradiance inside aperture [kW/m2]
    y = captured power inside aperture [W]

The default inputs are the four CSV files in simulation_results. Measured
power-vs-irradiance plots are written by analyze_target_series.py during the
full camera-analysis pipeline.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib
import numpy as np

SHOW_REQUESTED = "--show" in sys.argv[1:]
if not SHOW_REQUESTED:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt


CASE_RE = re.compile(r"(?:(01|06)[-_](REF|SEC)|(REF|SEC)[-_](01|06))", re.IGNORECASE)
SEASON_LABELS = {"01": "Winter", "06": "Summer"}
CASE_LABELS = {"REF": "Reference", "SEC": "With secondary"}
CASE_COLORS = {"REF": "#E95565", "SEC": "#4B90E3"}
TEXT_COLOR = "#555555"
GRID_COLOR = "#d8d8d8"


@dataclass(frozen=True)
class SimulationField:
    """One receiver irradiance field loaded from a simulation CSV."""

    kind: str
    season: str
    path: Path
    irradiance_w_m2: np.ndarray
    valid_mask: np.ndarray
    x_mm: np.ndarray
    y_mm: np.ndarray
    cell_area_m2: float


@dataclass(frozen=True)
class CurvePoint:
    """One aperture-sweep point for the power-vs-irradiance curve."""

    kind: str
    season: str
    aperture_diameter_mm: float
    aperture_area_m2: float
    mean_irradiance_kw_m2: float
    power_w: float
    cell_count: int
    center_x_mm: float
    center_y_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create summer and winter power-vs-irradiance plots from simulation receiver CSVs."
    )
    parser.add_argument(
        "--simulation-dir",
        type=Path,
        default=Path("simulation_results"),
        help="Directory containing simulation receiver CSV exports. Default: simulation_results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("power_irradiance_output"),
        help="Directory where plots and curve CSV are written. Default: power_irradiance_output.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=180,
        help="Number of aperture diameters to sample before duplicate cell counts are removed. Default: 180.",
    )
    parser.add_argument(
        "--min-diameter-mm",
        type=float,
        default=None,
        help="Smallest aperture diameter in mm. Default: about one grid cell.",
    )
    parser.add_argument(
        "--max-diameter-mm",
        type=float,
        default=None,
        help="Largest aperture diameter in mm. Default: large enough to cover the full receiver field.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output plot DPI. Default: 300.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def parse_case(path: Path) -> tuple[str, str] | None:
    """Return (kind, season) from a simulation CSV filename."""

    match = CASE_RE.search(path.stem)
    if not match:
        return None
    season = match.group(1) or match.group(4)
    kind = match.group(2) or match.group(3)
    if season is None or kind is None:
        return None
    return kind.upper(), season.upper()


def median_step(values: np.ndarray) -> float:
    """Return the median positive step in a coordinate vector."""

    diffs = np.diff(values.astype(np.float64))
    finite = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 0.0)])
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def load_simulation_field(path: Path) -> SimulationField:
    """Load one simulation receiver CSV."""

    parsed = parse_case(path)
    if parsed is None:
        raise ValueError(f"{path.name}: could not parse REF/SEC and season from filename.")
    kind, season = parsed
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    if not rows:
        raise ValueError(f"{path.name}: no data rows found.")

    row_count = max(int(row["row"]) for row in rows) + 1
    col_count = max(int(row["col"]) for row in rows) + 1
    irradiance = np.full((row_count, col_count), np.nan, dtype=np.float64)
    x_grid = np.full((row_count, col_count), np.nan, dtype=np.float64)
    y_grid = np.full((row_count, col_count), np.nan, dtype=np.float64)

    for row in rows:
        r = int(row["row"])
        c = int(row["col"])
        irradiance[r, c] = float(row["irradiance_w_m2"])
        x_grid[r, c] = float(row["u_m"]) * 1000.0
        y_grid[r, c] = float(row["v_m"]) * 1000.0

    x_mm = np.nanmedian(x_grid, axis=0)
    y_mm = np.nanmedian(y_grid, axis=1)
    if x_mm[-1] < x_mm[0]:
        x_mm = x_mm[::-1]
        irradiance = np.fliplr(irradiance)
    if y_mm[-1] < y_mm[0]:
        y_mm = y_mm[::-1]
        irradiance = np.flipud(irradiance)

    dx_mm = median_step(x_mm)
    dy_mm = median_step(y_mm)
    valid_mask = np.isfinite(irradiance)
    return SimulationField(
        kind=kind,
        season=season,
        path=path,
        irradiance_w_m2=irradiance,
        valid_mask=valid_mask,
        x_mm=x_mm,
        y_mm=y_mm,
        cell_area_m2=(dx_mm / 1000.0) * (dy_mm / 1000.0),
    )


def load_fields(simulation_dir: Path) -> dict[tuple[str, str], SimulationField]:
    """Load all REF/SEC simulation CSVs from a directory."""

    if not simulation_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {simulation_dir}")

    fields: dict[tuple[str, str], SimulationField] = {}
    for path in sorted(simulation_dir.glob("*.csv")):
        parsed = parse_case(path)
        if parsed is None:
            continue
        field = load_simulation_field(path)
        fields[(field.kind, field.season)] = field
    return fields


def default_max_diameter_mm(field: SimulationField, center_x_mm: float, center_y_mm: float) -> float:
    """Return a diameter large enough to cover every valid cell from the aperture center."""

    x_grid, y_grid = np.meshgrid(field.x_mm, field.y_mm)
    valid = field.valid_mask & np.isfinite(field.irradiance_w_m2)
    distances = np.hypot(x_grid[valid] - center_x_mm, y_grid[valid] - center_y_mm)
    if distances.size == 0:
        return 1.0
    return float(2.002 * np.max(distances))


def aperture_curve(
    field: SimulationField,
    points: int,
    min_diameter_mm: float | None,
    max_diameter_mm: float | None,
) -> list[CurvePoint]:
    """Sweep a circular aperture from tiny to large around the brightest cell."""

    valid = field.valid_mask & np.isfinite(field.irradiance_w_m2)
    if not np.any(valid):
        return []

    clean = np.where(valid, np.clip(field.irradiance_w_m2, 0.0, None), np.nan)
    peak_row, peak_col = np.unravel_index(int(np.nanargmax(clean)), clean.shape)
    center_x_mm = float(field.x_mm[peak_col])
    center_y_mm = float(field.y_mm[peak_row])
    dx_mm = median_step(field.x_mm)
    dy_mm = median_step(field.y_mm)

    smallest = max(min(dx_mm, dy_mm), 1e-6) if min_diameter_mm is None else float(min_diameter_mm)
    largest = (
        default_max_diameter_mm(field, center_x_mm, center_y_mm)
        if max_diameter_mm is None
        else float(max_diameter_mm)
    )
    if largest <= smallest:
        raise ValueError(f"{field.path.name}: max aperture diameter must be larger than min diameter.")

    x_grid, y_grid = np.meshgrid(field.x_mm, field.y_mm)
    diameters = np.geomspace(smallest, largest, max(points, 2))
    curve: list[CurvePoint] = []
    seen_cell_counts: set[int] = set()

    for diameter_mm in diameters:
        radius_mm = float(diameter_mm) * 0.5
        aperture_mask = valid & ((x_grid - center_x_mm) ** 2 + (y_grid - center_y_mm) ** 2 <= radius_mm**2)
        cell_count = int(np.count_nonzero(aperture_mask))
        if cell_count == 0 or cell_count in seen_cell_counts:
            continue
        seen_cell_counts.add(cell_count)

        power_w = float(np.nansum(clean[aperture_mask]) * field.cell_area_m2)
        aperture_area_m2 = float(cell_count * field.cell_area_m2)
        mean_irradiance_kw_m2 = power_w / max(aperture_area_m2, 1e-18) / 1000.0
        curve.append(
            CurvePoint(
                kind=field.kind,
                season=field.season,
                aperture_diameter_mm=float(diameter_mm),
                aperture_area_m2=aperture_area_m2,
                mean_irradiance_kw_m2=mean_irradiance_kw_m2,
                power_w=power_w,
                cell_count=cell_count,
                center_x_mm=center_x_mm,
                center_y_mm=center_y_mm,
            )
        )

    return sorted(curve, key=lambda point: point.mean_irradiance_kw_m2)


def write_curve_csv(output_dir: Path, curves: dict[tuple[str, str], list[CurvePoint]]) -> Path:
    """Write all generated curve points to one CSV file."""

    out_path = output_dir / "power_vs_irradiance_curves.csv"
    fieldnames = [
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
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (kind, season), curve in sorted(curves.items()):
            for point in curve:
                writer.writerow(
                    {
                        "season": season,
                        "season_label": SEASON_LABELS.get(season, season),
                        "case": kind,
                        "case_label": CASE_LABELS.get(kind, kind),
                        "aperture_diameter_mm": point.aperture_diameter_mm,
                        "aperture_area_m2": point.aperture_area_m2,
                        "mean_irradiance_kw_m2": point.mean_irradiance_kw_m2,
                        "power_w": point.power_w,
                        "cell_count": point.cell_count,
                        "center_x_mm": point.center_x_mm,
                        "center_y_mm": point.center_y_mm,
                    }
                )
    return out_path


def style_axis(ax: plt.Axes) -> None:
    """Apply minimal horizontal-grid chart styling."""

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1.2)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=4, width=1.0, color="#9a9a9a", labelcolor=TEXT_COLOR, labelsize=12)
    ax.tick_params(axis="y", length=0, labelcolor=TEXT_COLOR, labelsize=12)
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def plot_season(output_dir: Path, season: str, curves: dict[tuple[str, str], list[CurvePoint]], dpi: int, show: bool) -> Path:
    """Save one season plot."""

    season_label = SEASON_LABELS.get(season, season)
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=dpi)

    for kind in ("REF", "SEC"):
        curve = curves.get((kind, season), [])
        if not curve:
            continue
        x_values = [point.mean_irradiance_kw_m2 for point in curve]
        y_values = [point.power_w for point in curve]
        ax.plot(
            x_values,
            y_values,
            color=CASE_COLORS[kind],
            linewidth=3.0,
            solid_capstyle="round",
            label=CASE_LABELS[kind],
        )

    ax.set_title(f"Simulated {season_label.lower()} case: power vs irradiance", fontsize=17, pad=18)
    ax.set_xlabel("Mean irradiance [kW/m2]", fontsize=13, labelpad=10)
    ax.set_ylabel("Power [W]", fontsize=13, labelpad=10)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    style_axis(ax)
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
        text.set_color(TEXT_COLOR)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    out_path = output_dir / f"simulated_{season_label.lower()}_power_vs_irradiance.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = load_fields(args.simulation_dir)
    curves: dict[tuple[str, str], list[CurvePoint]] = {}
    for key, field in fields.items():
        curves[key] = aperture_curve(field, args.points, args.min_diameter_mm, args.max_diameter_mm)

    written_plots: list[Path] = []
    for season in ("06", "01"):
        if curves.get(("REF", season)) or curves.get(("SEC", season)):
            written_plots.append(plot_season(output_dir, season, curves, args.dpi, args.show))

    csv_path = write_curve_csv(output_dir, curves)
    print("Wrote:")
    for path in written_plots:
        print(f"  {path}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
