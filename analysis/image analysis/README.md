# Image Analysis

This folder contains Python scripts used for camera and image-analysis tests in the thesis.

Most of the code in this folder was adapted from the specialization project preceding the master's thesis. Only smaller adjustments were made for the thesis work, except for `analyze_iso_linearity.py`, which was written specifically for this project.

## Scripts

* `analyze_manual.py` — analyzes manual exposure linearity captures using ArUco marker detection and perspective correction.
* `variable_exposure_analysis.py` — shared helper code for partially automated exposure-series analysis.
* `analyze_full_auto.py` — analyzes the full-auto exposure series.
* `analyze_av_fixed_f.py` — analyzes the aperture-priority/fixed-f-number exposure series.
* `analyze_auto_iso.py` — analyzes the auto-ISO exposure series.
* `analyze_iso_linearity.py` — analyzes ISO versus image brightness. This script was made for the master's thesis project.
* `analyze_vignetting.py` — visualizes vignetting for different focal-length and aperture configurations.
* `analyze_lambertian.py` — analyzes Lambertian response at different target angles.
* `analyze_testrig.py` — analyzes test-rig captures by aligning mirror exposures, subtracting OFF frames, and reporting beam intensity and width metrics.

## Target files

The SVG files in this folder are ArUco marker targets used for image alignment and region-of-interest detection.

## Raw image files

The raw CR2 image files used by these scripts are not included in this repository due to file size constraints.

They can be requested if needed.

## Notes

The scripts are project-specific and depend on the folder names, CR2 image files, ArUco marker layout, and camera-test setup used during the thesis work.
