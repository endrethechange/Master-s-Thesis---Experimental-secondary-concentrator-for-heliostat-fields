from pathlib import Path
import argparse
import csv
import math


def fmt_kw(v: float) -> str:
    """Format power in kW with 5 decimal places."""
    return f"{v:.5f}"


def label_box(x, y, title, value_scaled, value_unscaled=None, align="left", title_size=14, value_size=16):
    """Generate label box with scaled (model) and unscaled (full-scale) power values."""
    value_txt_scaled = f"{value_scaled:.5f} kW"
    if value_unscaled is not None:
        value_txt_scaled = f"{value_scaled:.5f} kW (1:200)"
        value_txt_unscaled = f"{value_unscaled:.2f} kW"
        max_len = max(len(value_txt_scaled), len(value_txt_unscaled))
        w = max(len(title) * title_size * 0.56, max_len * value_size * 0.54) + 16
        h = title_size + value_size * 2 + 18
    else:
        w = max(len(title) * title_size * 0.56, len(value_txt_scaled) * value_size * 0.54) + 16
        h = title_size + value_size + 16
    
    if align == "right":
        bx = x - w
        tx = x - 8
        anchor = "end"
    else:
        bx = x
        tx = x + 8
        anchor = "start"
    by = y - h / 2
    
    if value_unscaled is not None:
        return f"""
  <g class="label">
    <rect x="{bx:.2f}" y="{by:.2f}" width="{w:.2f}" height="{h:.2f}" rx="2" fill="#ffffff" opacity="0.96"/>
    <text x="{tx:.2f}" y="{by + 15:.2f}" font-family="Helvetica, Arial, sans-serif" fill="#404040" text-anchor="{anchor}">
      <tspan x="{tx:.2f}" font-size="{title_size}" font-weight="400">{title}</tspan>
      <tspan x="{tx:.2f}" dy="{value_size}" font-size="{value_size}" font-weight="400">{value_txt_scaled}</tspan>
      <tspan x="{tx:.2f}" dy="{value_size}" font-size="{value_size - 2}" font-weight="400" fill="#888888">{value_txt_unscaled}</tspan>
    </text>
  </g>"""
    else:
        return f"""
  <g class="label">
    <rect x="{bx:.2f}" y="{by:.2f}" width="{w:.2f}" height="{h:.2f}" rx="2" fill="#ffffff" opacity="0.96"/>
    <text x="{tx:.2f}" y="{by + 18:.2f}" font-family="Helvetica, Arial, sans-serif" fill="#404040" text-anchor="{anchor}">
      <tspan x="{tx:.2f}" font-size="{title_size}" font-weight="400">{title}</tspan>
      <tspan x="{tx:.2f}" dy="{value_size + 2}" font-size="{value_size}" font-weight="400">{value_txt_scaled}</tspan>
    </text>
  </g>"""


def rounded_polygon_path(points, radii):
    """Return an SVG path for a closed polygon with rounded corners."""
    if len(points) < 3:
        raise ValueError("rounded_polygon_path requires at least three points")

    def corner(prev_pt, pt, next_pt, radius):
        prev_len = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
        next_len = math.hypot(next_pt[0] - pt[0], next_pt[1] - pt[1])
        if radius <= 0 or prev_len == 0 or next_len == 0:
            return pt, pt

        cut = min(radius, prev_len * 0.45, next_len * 0.45)
        in_pt = (
            pt[0] + (prev_pt[0] - pt[0]) * cut / prev_len,
            pt[1] + (prev_pt[1] - pt[1]) * cut / prev_len,
        )
        out_pt = (
            pt[0] + (next_pt[0] - pt[0]) * cut / next_len,
            pt[1] + (next_pt[1] - pt[1]) * cut / next_len,
        )
        return in_pt, out_pt

    rounded = [
        corner(points[i - 1], points[i], points[(i + 1) % len(points)], radii[i])
        for i in range(len(points))
    ]
    start = rounded[0][1]
    parts = [f"M {start[0]:.2f} {start[1]:.2f}"]
    for i in range(1, len(points)):
        in_pt, out_pt = rounded[i]
        pt = points[i]
        parts.append(f"L {in_pt[0]:.2f} {in_pt[1]:.2f}")
        parts.append(f"Q {pt[0]:.2f} {pt[1]:.2f} {out_pt[0]:.2f} {out_pt[1]:.2f}")

    in_pt, out_pt = rounded[0]
    pt = points[0]
    parts.append(f"L {in_pt[0]:.2f} {in_pt[1]:.2f}")
    parts.append(f"Q {pt[0]:.2f} {pt[1]:.2f} {out_pt[0]:.2f} {out_pt[1]:.2f} Z")
    return " ".join(parts)


def right_offpage_arrow_path(x, y, w, h, rect_frac=0.36, r=4.0):
    """Return the rounded right-pointing arrowhead used for the encircled-power output."""
    rect_w = w * rect_frac
    x0, shoulder_x, tip = x, x + rect_w, x + w
    y0, y1, ym = y, y + h, y + h / 2
    base_r = min(r, h * 0.20, rect_w * 0.35)
    shoulder_r = min((w - rect_w) * 0.18, h * 0.12, 5.5)
    tip_r = min((w - rect_w) * 0.20, h * 0.10, 5.0)
    return rounded_polygon_path(
        [
            (x0, y0),
            (shoulder_x, y0),
            (tip, ym),
            (shoulder_x, y1),
            (x0, y1),
        ],
        [base_r, shoulder_r, tip_r, shoulder_r, base_r],
    )


def down_offpage_arrow_path(cx, y, w, h, rect_frac=0.34, r=3.0):
    """Return the rounded downward arrowhead used for the spillage output."""
    rect_h = h * rect_frac
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1, tip = y, y + rect_h, y + h
    top_r = min(r, w * 0.16, rect_h * 0.35)
    shoulder_r = min(w * 0.09, (h - rect_h) * 0.12, 5.0)
    tip_r = min(w * 0.10, (h - rect_h) * 0.14, 4.5)
    return rounded_polygon_path(
        [
            (x0, y0),
            (x1, y0),
            (x1, y1),
            (cx, tip),
            (x0, y1),
        ],
        [top_r, top_r, shoulder_r, tip_r, shoulder_r],
    )


def make_sankey(total_power_kw=381, encircled_power_kw=300, total_irradiance_kw_m2=None, encircled_irradiance_kw_m2=None, out_svg=Path("sankey.svg")):
    """Create one SVG Sankey diagram for total heliostat power and aperture losses."""
    if total_power_kw <= 0:
        raise ValueError("total_power_kw must be > 0")
    if encircled_power_kw < 0 or encircled_power_kw > total_power_kw:
        raise ValueError("encircled_power_kw must be between 0 and total_power_kw")

    # Split the incoming power into the aperture contribution and the lost
    # spillage contribution. These fractions control the visible branch heights.
    spillage_kw = total_power_kw - encircled_power_kw
    enc_frac = encircled_power_kw / total_power_kw
    sp_frac = spillage_kw / total_power_kw
    
    # Calculate full-scale power values for the 1:200 model. A 1:200 linear
    # scale corresponds to a 1:40,000 area scale, so power is multiplied by
    # 40,000 when irradiance is unchanged.
    full_scale_factor = 40000
    total_power_full_scale = total_power_kw * full_scale_factor
    encircled_power_full_scale = encircled_power_kw * full_scale_factor
    spillage_power_full_scale = spillage_kw * full_scale_factor

    # Canvas and base geometry. total_h is the height of the input stream; the
    # two output branches are scaled to partition this same height.
    W, H = 820, 280
    total_h = 115.0
    min_visible = 1.5

    # Convert power fractions to SVG heights. The minimum visible height keeps
    # very small non-zero branches legible, then the values are renormalized so
    # the two branch heights still sum to total_h.
    if encircled_power_kw == 0:
        enc_h = 0.0
        sp_h = total_h if spillage_kw > 0 else 0.0
    elif spillage_kw == 0:
        enc_h = total_h
        sp_h = 0.0
    else:
        enc_h = max(min_visible, total_h * enc_frac)
        sp_h = max(min_visible, total_h * sp_frac)
        height_scale = total_h / (enc_h + sp_h)
        enc_h *= height_scale
        sp_h *= height_scale

    # Input stream and split location. A small overlap between shapes hides
    # anti-aliasing gaps where the input block meets the two outgoing branches.
    x_input, y_top = 260.0, 44.0
    input_w = 14.0
    split_x = x_input + input_w - 2.5

    # Encircled-power branch geometry: a constant-height rectangular body with
    # a right-pointing off-page arrow to show that the useful power continues.
    body_len = 160.0
    head_w = 38.0
    head_overlap = 1.5
    body_x = split_x - 1.0
    body_y = y_top
    head_x = body_x + body_len - head_overlap
    tip_x = head_x + head_w
    enc_mid = body_y + enc_h / 2 if enc_h > 0 else body_y

    # Spillage branch geometry. The branch leaves the split horizontally, bends
    # downward, and ends in a down-pointing off-page arrow.
    sp_center_y = y_top + enc_h + sp_h / 2 if sp_h > 0 else y_top + enc_h
    spill_x = split_x + 60.0
    spill_head_h = max(18.0, min(64.0, sp_h * 0.92)) if sp_h > 0 else 0.0
    spill_min_bend_r = max(20.0, sp_h * 0.80) if sp_h > 0 else 20.0
    spill_head_y = max(178.0, sp_center_y + spill_min_bend_r) if sp_h > 0 else 178.0
    spill_tail_end_y = spill_head_y

    # Build the spillage branch centerline. The stroke width is sp_h, so the
    # visible branch thickness remains proportional to spillage_kw along the
    # whole curved path.
    spill_start_x = split_x - 1.0
    available_curve_y = max(2.0, spill_tail_end_y - sp_center_y)
    horizontal_curve_space = max(2.0, spill_x - spill_start_x - 8.0)
    quarter_r = min(spill_min_bend_r, horizontal_curve_space, available_curve_y)
    spill_centerline = (
        f"M {spill_start_x:.2f} {sp_center_y:.2f} "
        f"H {spill_x - quarter_r:.2f} "
        f"Q {spill_x:.2f} {sp_center_y:.2f} {spill_x:.2f} {sp_center_y + quarter_r:.2f} "
        f"V {spill_tail_end_y:.2f}"
    )

    encircled_flow_svg = ""
    if enc_h > 0:
        encircled_flow_svg = f"""
  <!-- Encircled-power output: rectangular flow plus a rounded off-page arrow. -->
  <rect x="{body_x:.2f}" y="{body_y:.2f}" width="{body_len:.2f}" height="{enc_h:.4f}" fill="url(#encircledGradient)"/>
  <path d="{right_offpage_arrow_path(head_x, body_y, head_w, enc_h, rect_frac=0.36, r=4.0)}" fill="#156082"/>"""

    spillage_flow_svg = ""
    if sp_h > 0:
        spillage_flow_svg = f"""
  <!-- Spillage output: curved constant-thickness flow scaled by spillage_kw. -->
  <path d="{spill_centerline}" fill="none" stroke="url(#spillageGradient)" stroke-width="{sp_h:.4f}" stroke-linecap="butt" stroke-linejoin="round"/>
  <path d="{down_offpage_arrow_path(spill_x, spill_head_y, sp_h, spill_head_h, rect_frac=0.34, r=3.0)}" fill="#EA6B66"/>"""

    input_path = f'<rect x="{x_input:.2f}" y="{y_top:.2f}" width="{input_w:.2f}" height="{total_h:.2f}" rx="4" ry="4" fill="#F7C75C"/>'

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}px" height="{H}px" viewBox="0 0 {W} {H}" role="img" aria-label="Total Power from Heliostats {fmt_kw(total_power_kw)} kW; Power Inside Aperture {fmt_kw(encircled_power_kw)} kW; Spillage {fmt_kw(spillage_kw)} kW">
  <defs>
    <linearGradient id="encircledGradient" x1="{body_x}" y1="0" x2="{head_x}" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#FFE599"/>
      <stop offset="100%" stop-color="#D4E1F5"/>
    </linearGradient>
    <linearGradient id="spillageGradient" x1="{split_x}" y1="{sp_center_y}" x2="{spill_x}" y2="{spill_head_y}" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#FFE599"/>
      <stop offset="100%" stop-color="#F8CECC"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff"/>

  <!-- Source values used to scale the diagram. -->
  <!-- total_power_kw={fmt_kw(total_power_kw)}, encircled_power_kw={fmt_kw(encircled_power_kw)}, spillage_kw={fmt_kw(spillage_kw)} -->
  <!-- encircled_h={enc_h:.4f}, spillage_h={sp_h:.4f}; both are scaled against total_h={total_h:.4f}. -->

  {encircled_flow_svg}

  {spillage_flow_svg}

  <!-- Total-power input. Drawn last with a slight overlap so the rounded edge covers junction gaps. -->
  {input_path}

  <!-- Labels: Helvetica, dark gray text, white backing, positioned next to each flow component. -->
  {label_box(16, y_top + total_h / 2, "Total Power from Heliostats", total_power_kw, total_power_full_scale, align="left")}
  {label_box(tip_x + 18, enc_mid, "Power Inside Aperture", encircled_power_kw, encircled_power_full_scale, align="left")}
  {label_box(spill_x + sp_h / 2 + 16, spill_head_y + spill_head_h / 2, "Spillage", spillage_kw, spillage_power_full_scale, align="left")}
</svg>
'''
    out_svg = Path(out_svg)
    out_svg.write_text(svg, encoding="utf-8")
    return out_svg


def sanitize_name(value: str) -> str:
    """Return a filesystem-safe name fragment for generated SVG files."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value).strip("_")


def make_sankey_batch(csv_path: Path, output_dir: Path):
    """Generate one Sankey SVG per CSV row."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if "total_power_kw" not in reader.fieldnames or "encircled_power_kw" not in reader.fieldnames:
            raise ValueError("CSV must contain the fields total_power_kw and encircled_power_kw")
        for index, row in enumerate(reader, start=1):
            try:
                total_kw = float(row["total_power_kw"])
                enc_kw = float(row["encircled_power_kw"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid number in row {index}: {exc}") from exc

            source = sanitize_name(row.get("source", "row") or "row")
            case = sanitize_name(row.get("case", f"{index:03d}"))
            out_name = f"{source}_{case}.svg" if source else f"row_{index:03d}.svg"
            out_path = output_dir / out_name
            make_sankey(total_kw, enc_kw, out_svg=out_path)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Sankey diagrams from a CSV file and write SVG outputs.")
    parser.add_argument("--csv", type=Path, default=Path("simulation_comparison_metrics.csv"), help="Input CSV file")
    parser.add_argument("--outdir", type=Path, default=Path("Output visual"), help="Directory for SVG output files")
    args = parser.parse_args()
    make_sankey_batch(args.csv, args.outdir)
