#!/usr/bin/env python3
"""Analyze the Av (fixed f-number) capture series with intensity normalization."""

from pathlib import Path

from variable_exposure_analysis import run_variable_analysis


def main() -> None:
    default_dir = Path("Linearity/Av Fixed f-number")
    run_variable_analysis(
        default_dir=default_dir,
        default_plot=default_dir / "diff_mean_fit.png",
        default_error_plot=default_dir / "diff_mean_residuals.png",
        label="Av Fixed f-number",
        skip_frames=(1,),
    )


if __name__ == "__main__":
    main()
