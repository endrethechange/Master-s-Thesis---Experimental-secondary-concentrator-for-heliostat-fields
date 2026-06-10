#!/usr/bin/env python3
"""Analyze the Full auto capture series with intensity normalization."""

from pathlib import Path

from variable_exposure_analysis import run_variable_analysis


def main() -> None:
    default_dir = Path("Linearity/Full auto")
    run_variable_analysis(
        default_dir=default_dir,
        default_plot=default_dir / "diff_mean_fit.png",
        default_error_plot=default_dir / "diff_mean_residuals.png",
        label="Full auto",
        skip_frames=(1,),
    )


if __name__ == "__main__":
    main()
