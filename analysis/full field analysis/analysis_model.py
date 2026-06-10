"""Shared constants and data containers for the full-field target analysis.

The processing modules exchange these dataclasses instead of loose dictionaries
so it is clear which objects represent camera metadata, warped images,
calibrations, measured spots, and comparison fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_PIXELS = 1500
TARGET_SIZE_MM = 6.0
MM_PER_PIXEL = TARGET_SIZE_MM / TARGET_PIXELS
DOT_DIAMETER_PX = 150.0
DOT_RADIUS_PX = DOT_DIAMETER_PX / 2.0
DOT_TANGENT_SQUARE_PX = 1294.5
DOT_CENTER_MARGIN_PX = ((TARGET_PIXELS - DOT_TANGENT_SQUARE_PX) / 2.0) + DOT_RADIUS_PX
RIGHT_DOT_X_PX = TARGET_PIXELS - DOT_CENTER_MARGIN_PX
LEFT_DOT_X_PX = DOT_CENTER_MARGIN_PX
TOP_DOT_Y_PX = DOT_CENTER_MARGIN_PX
BOTTOM_DOT_Y_PX = TARGET_PIXELS - DOT_CENTER_MARGIN_PX
MIDDLE_DOT_Y_PX = TARGET_PIXELS / 2.0
SUNLIGHT_IRRADIANCE_W_M2 = 800.0
VALID_SEASONS = {"01", "06"}
SEASON_LABELS = {"01": "Winter", "06": "Summer"}
POWER_CURVE_SOURCE_LABELS = {"simulation": "Simulated", "measurement": "Measured"}
POWER_CURVE_CASE_LABELS = {"REF": "Reference", "SEC": "With secondary"}
POWER_CURVE_COLORS = {"REF": "#E95565", "SEC": "#4B90E3"}
MEASURED_LUMINANCE_SOURCE = "raw-bayer-mean"
MEASURED_LUMINANCE_DESCRIPTION = (
    "black-subtracted raw Bayer values averaged over each non-overlapping 2x2 Bayer cell as (R + G1 + G2 + B) / 4, "
    "then expanded back to full resolution; no white balance, demosaic tone scaling, gamma, or human-eye weighting"
)
TARGET_DETECTION_DESCRIPTION = (
    "camera-rendered linear preview used only to locate the target geometry; reported values use raw-bayer-mean linear luminance"
)
CASE_NAME_RE = re.compile(r"^(REF|SEC)_(01|06)(?:(?:[-_]OFF)(?:-(\d+))?|-(\d+))?$", re.IGNORECASE)
NRM_NAME_RE = re.compile(r"^NRM_(01|06)(?:-(\d+))?$", re.IGNORECASE)
PREFIXED_NRM_NAME_RE = re.compile(r"^(?:REF|SEC)_(01|06)[-_]NRM(?:-(\d+))?$", re.IGNORECASE)
DASH_NRM_NAME_RE = re.compile(r"^(01|06)-NRM(?:-(\d+))?$", re.IGNORECASE)
SIMULATION_NAME_RE = re.compile(r"(?:(01|06)[-_](REF|SEC)|(REF|SEC)[-_](01|06))", re.IGNORECASE)
MODEL_SCALE = 200.0
REF_APERTURE_DIAMETER_MM = (0.16 / MODEL_SCALE) * 1000.0
SEC_APERTURE_DIAMETER_X_MM = (0.47315 / MODEL_SCALE) * 1000.0
SEC_APERTURE_DIAMETER_Y_MM = (0.31665 / MODEL_SCALE) * 1000.0

MARKER_NAMES = ("top_left", "top_right", "middle_right", "bottom_left", "bottom_right")
EXPECTED_MARKER_CENTERS = np.array(
    [
        [LEFT_DOT_X_PX, TOP_DOT_Y_PX],
        [RIGHT_DOT_X_PX, TOP_DOT_Y_PX],
        [RIGHT_DOT_X_PX, MIDDLE_DOT_Y_PX],
        [LEFT_DOT_X_PX, BOTTOM_DOT_Y_PX],
        [RIGHT_DOT_X_PX, BOTTOM_DOT_Y_PX],
    ],
    dtype=np.float32,
)
TARGET_CORNERS = np.array(
    [[0.0, 0.0], [TARGET_PIXELS - 1.0, 0.0], [TARGET_PIXELS - 1.0, TARGET_PIXELS - 1.0], [0.0, TARGET_PIXELS - 1.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class CameraMetadata:
    """Exposure data needed for photometric normalization."""

    iso: float
    exposure_s: float
    f_number: float
    orientation: str


@dataclass(frozen=True)
class Exposure:
    """Parsed identity for one CR2 input file."""

    path: Path
    kind: str
    season: str
    is_off: bool
    variant: str | None
    source_id: str

    @property
    def stem(self) -> str:
        """Return the original filename stem."""

        return self.path.stem

    @property
    def base_key(self) -> tuple[str, str]:
        """Return the REF/SEC/NRM and season key."""

        return (self.kind, self.season)

    @property
    def on_key(self) -> tuple[str, str, str | None]:
        """Return the key used to match ON images and numbered OFF images."""

        return (self.kind, self.season, self.variant)


@dataclass
class TargetGeometry:
    """Known target geometry and masks in the 1500 x 1500 warped frame."""

    marker_centers: np.ndarray
    marker_masks: list[np.ndarray]
    marker_exclusion_mask: np.ndarray
    calibration_mask: np.ndarray
    analysis_mask: np.ndarray
    plot_mask: np.ndarray
    white_background_mask: np.ndarray
    marker_ring_masks: list[np.ndarray]
    template_model: np.ndarray


@dataclass
class DetectionResult:
    """Target-detection outputs used for debug rendering and final warping."""

    source_quad: np.ndarray
    marker_points_source: np.ndarray
    marker_centers_warped: np.ndarray
    score: float
    turns_ccw: int
    source_to_unrotated_target: np.ndarray


@dataclass
class WarpedExposure:
    """Linear, photometrically normalized target image plus debug metadata."""

    exposure: Exposure
    metadata: CameraMetadata
    luminance: np.ndarray
    normalized_luminance: np.ndarray
    detection: DetectionResult
    saturated_fraction: float
    measured_luminance_source: str
    luminance_definition: str
    target_detection_source: str
    raw_sensor_stats: dict[str, float | None]


@dataclass
class Calibration:
    """Conversion from normalized camera intensity to W/m^2."""

    nrm: WarpedExposure | None
    reference_intensity: float
    scale_w_m2_per_intensity: float
    source: str = "NRM"
    reference_case: str = ""
    reference_irradiance_w_m2: float = SUNLIGHT_IRRADIANCE_W_M2
    reference_stat: str = "target-mean"
    reference_units: str = "W/m2"


@dataclass
class SpotResult:
    """Measured spot properties before optional irradiance scaling."""

    peak_value: float
    peak_x_px: int
    peak_y_px: int
    peak_x_mm_centered: float
    peak_y_mm_centered: float
    centroid_x_px: float | None
    centroid_y_px: float | None
    sigma_x_px: float | None
    sigma_y_px: float | None
    sigma_x_mm: float | None
    sigma_y_mm: float | None
    spot_detected: bool
    background_floor: float
    threshold: float
    component_area_px: int


@dataclass(frozen=True)
class ApertureSpec:
    """Physical receiver aperture used for max-encircled-power measurements."""

    name: str
    diameter_x_mm: float
    diameter_y_mm: float


@dataclass
class EncirclementResult:
    """Maximum-power aperture placement and irradiance statistics."""

    center_x_mm: float
    center_y_mm: float
    diameter_x_mm: float
    diameter_y_mm: float
    area_m2: float
    power_kw: float
    mean_w_m2: float
    peak_w_m2: float
    cell_count: int
    reference_mean_w_m2: float | None = None
    mean_reference_ratio: float | None = None
    aperture_note: str = ""


@dataclass
class FieldPlotData:
    """One real or simulated irradiance field for comparison plotting."""

    source: str
    kind: str
    season: str
    title: str
    irradiance_w_m2: np.ndarray
    valid_mask: np.ndarray
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    cell_area_m2: float
    total_power_kw: float
    aperture: ApertureSpec
    encirclement: EncirclementResult


@dataclass(frozen=True)
class PowerCurvePoint:
    """One aperture-sweep point for power-vs-irradiance plotting."""

    source: str
    kind: str
    season: str
    aperture_diameter_mm: float
    aperture_area_m2: float
    mean_irradiance_kw_m2: float
    power_w: float
    cell_count: int
    center_x_mm: float
    center_y_mm: float
    center_source: str


@dataclass
class AnalysisOutput:
    """Outputs from one real-image analysis."""

    summary: dict[str, object]
    field: FieldPlotData | None


__all__ = [name for name in globals() if name.isupper()] + [
    "CameraMetadata",
    "Exposure",
    "TargetGeometry",
    "DetectionResult",
    "WarpedExposure",
    "Calibration",
    "SpotResult",
    "ApertureSpec",
    "EncirclementResult",
    "FieldPlotData",
    "PowerCurvePoint",
    "AnalysisOutput",
]
