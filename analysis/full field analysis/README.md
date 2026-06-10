# Full Field Analysis Code

This folder contains the Python code used for full field image analysis, calibration, plotting, and comparison of measured and simulated target data.

## Main script

* `analyze_target_series.py` — main script for processing Canon CR2 images of the target. It loads the image series, detects and warps the target, applies calibration, subtracts matching OFF images where available, and writes heatmaps, debug images, metadata, and summary files.

## Supporting modules

* `analysis_detection.py` — detects the five-dot target, estimates the target geometry, and warps images into the normalized target frame.
* `analysis_geometry.py` — builds the target geometry, marker masks, calibration masks, and analysis masks.
* `analysis_io.py` — handles input discovery, EXIF metadata, CR2 loading, raw Bayer luminance extraction, and exposure normalization.
* `analysis_measurement.py` — performs calibration, spot analysis, OFF-image correction, marker filling, and per-image reporting.
* `analysis_comparison.py` — loads simulation results and creates comparison plots, aperture calculations, and power curves.
* `analysis_model.py` — contains shared constants, filename patterns, physical target dimensions, and dataclasses used across the analysis pipeline.
* `analysis_utils.py` — contains small shared helper functions for directories, safe filenames, image normalization, and coordinate conversion.

## Notes

The code is project-specific and assumes the file naming, target geometry, camera image format, calibration approach, and simulation data structure used in the thesis work.

It is included for transparency, inspection, and documentation of the analysis workflow.
