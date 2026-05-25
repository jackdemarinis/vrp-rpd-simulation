"""Exploratory visualization of the new CAD layout.

Prints the layout summary and writes a top-down SVG to
`Physical Set Up/cad_layout.svg`. Open the SVG in any browser.

Renders with Y flipped so the depot appears at the bottom — matching the
CAD screenshot orientation. Math coords (math frame) match
`vrp_rpd_sim.cad_layout`: +Y points north, depot is at high Y.
"""

from __future__ import annotations

import pathlib

from vrp_rpd_sim import cad_layout as L


SVG_PX_PER_INCH = 10
SVG_MARGIN_PX = 40

ROAD_STROKE = "#d6c4d4"     # pink-ish, matching the CAD pink
ROAD_WIDTH_PX = 16          # visual width of a road strip
INTERSECTION_FILL = "#202024"
INTERSECTION_RADIUS_PX = 5
DEPOT_SLOT_FILL = "#3b86c2"
DEPOT_SLOT_RADIUS_PX = 7
GRID_CORNER_DEPOT_FILL = "#c25e3b"  # highlight the slot-1 grid corner
CENTERLINE = "#7a6678"      # darker pink for the road centerline
CENTERLINE_WIDTH_PX = 1.5
LABEL_FILL = "#202024"
LABEL_FONT_SIZE_PX = 10


def to_svg(x_in: float, y_in: float, world_height_in: float) -> tuple[float, float]:
    """World inches → SVG pixels, with Y flipped so depot is at the bottom."""
    x_px = SVG_MARGIN_PX + x_in * SVG_PX_PER_INCH
    y_px = SVG_MARGIN_PX + (world_height_in - y_in) * SVG_PX_PER_INCH
    return x_px, y_px


def build_svg() -> str:
    world_w = L.WORLD_WIDTH_IN
    world_h = L.WORLD_HEIGHT_IN
    svg_w = int(world_w * SVG_PX_PER_INCH + 2 * SVG_MARGIN_PX)
    svg_h = int(world_h * SVG_PX_PER_INCH + 2 * SVG_MARGIN_PX)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">'
    )
    parts.append('<rect width="100%" height="100%" fill="#f4f1ea"/>')

    # Title and frame caption.
    parts.append(
        f'<text x="{SVG_MARGIN_PX}" y="{SVG_MARGIN_PX - 16}" '
        f'font-family="Helvetica,Arial,sans-serif" font-size="14" '
        f'fill="{LABEL_FILL}">CAD layout (top-down, depot at bottom)</text>'
    )

    # Road strips: every grid row and column is a straight road, plus the
    # L-shape connecting the depot.
    parts.append(f'<g stroke="{ROAD_STROKE}" stroke-width="{ROAD_WIDTH_PX}" stroke-linecap="round">')
    for x in L.GRID_XS:
        x1_px, y1_px = to_svg(x, L.GRID_YS[0], world_h)
        x2_px, y2_px = to_svg(x, L.GRID_YS[-1], world_h)
        parts.append(f'<line x1="{x1_px}" y1="{y1_px}" x2="{x2_px}" y2="{y2_px}"/>')
    for y in L.GRID_YS:
        x1_px, y1_px = to_svg(L.GRID_XS[0], y, world_h)
        x2_px, y2_px = to_svg(L.GRID_XS[-1], y, world_h)
        parts.append(f'<line x1="{x1_px}" y1="{y1_px}" x2="{x2_px}" y2="{y2_px}"/>')
    # Depot vertical arm: from NE grid corner up to the horizontal arm.
    x1_px, y1_px = to_svg(L.GRID_CORNER_DEPOT_SLOT[0], L.GRID_CORNER_DEPOT_SLOT[1], world_h)
    x2_px, y2_px = to_svg(L.DEPOT_VERTICAL_X_IN, L.DEPOT_ARM_Y_IN, world_h)
    parts.append(f'<line x1="{x1_px}" y1="{y1_px}" x2="{x2_px}" y2="{y2_px}"/>')
    # Depot horizontal arm.
    x1_px, y1_px = to_svg(L.DEPOT_ARM_XS[0], L.DEPOT_ARM_Y_IN, world_h)
    x2_px, y2_px = to_svg(L.DEPOT_ARM_XS[-1], L.DEPOT_ARM_Y_IN, world_h)
    parts.append(f'<line x1="{x1_px}" y1="{y1_px}" x2="{x2_px}" y2="{y2_px}"/>')
    parts.append('</g>')

    # Centerlines (thin) — visualize the single-lane two-way.
    parts.append(
        f'<g stroke="{CENTERLINE}" stroke-width="{CENTERLINE_WIDTH_PX}" '
        f'stroke-dasharray="3 2">'
    )
    for x in L.GRID_XS:
        x1_px, y1_px = to_svg(x, L.GRID_YS[0], world_h)
        x2_px, y2_px = to_svg(x, L.GRID_YS[-1], world_h)
        parts.append(f'<line x1="{x1_px}" y1="{y1_px}" x2="{x2_px}" y2="{y2_px}"/>')
    for y in L.GRID_YS:
        x1_px, y1_px = to_svg(L.GRID_XS[0], y, world_h)
        x2_px, y2_px = to_svg(L.GRID_XS[-1], y, world_h)
        parts.append(f'<line x1="{x1_px}" y1="{y1_px}" x2="{x2_px}" y2="{y2_px}"/>')
    parts.append('</g>')

    # Grid intersection stations (64 dots).
    parts.append(f'<g fill="{INTERSECTION_FILL}">')
    for index, (x, y) in enumerate(L.GRID_INTERSECTIONS):
        if (x, y) == L.GRID_CORNER_DEPOT_SLOT:
            continue  # rendered specially below
        x_px, y_px = to_svg(x, y, world_h)
        parts.append(
            f'<circle cx="{x_px}" cy="{y_px}" r="{INTERSECTION_RADIUS_PX}"/>'
            f'<title>station {index} at ({x:.2f}, {y:.2f})</title>'
        )
    parts.append('</g>')

    # Depot slots (10).
    for index, slot in enumerate(L.ALL_DEPOT_SLOTS):
        x_px, y_px = to_svg(slot[0], slot[1], world_h)
        fill = GRID_CORNER_DEPOT_FILL if slot == L.GRID_CORNER_DEPOT_SLOT else DEPOT_SLOT_FILL
        parts.append(
            f'<circle cx="{x_px}" cy="{y_px}" r="{DEPOT_SLOT_RADIUS_PX}" fill="{fill}" '
            f'stroke="#202024" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{x_px + DEPOT_SLOT_RADIUS_PX + 3}" y="{y_px + 4}" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="{LABEL_FONT_SIZE_PX}" '
            f'fill="{LABEL_FILL}">A{index + 1}</text>'
        )

    # Axis ticks at the world corners for reference.
    for (label_x, label_y, anchor_x, anchor_y, anchor_text) in [
        (L.WORLD_MIN[0], L.WORLD_MIN[1], -16, 12, f"({L.WORLD_MIN[0]:.0f},{L.WORLD_MIN[1]:.0f})"),
        (L.WORLD_MAX[0], L.WORLD_MIN[1], 4, 12, f"({L.WORLD_MAX[0]:.2f},{L.WORLD_MIN[1]:.0f})"),
        (L.WORLD_MIN[0], L.WORLD_MAX[1], -16, -6, f"({L.WORLD_MIN[0]:.0f},{L.WORLD_MAX[1]:.2f})"),
        (L.WORLD_MAX[0], L.WORLD_MAX[1], 4, -6, f"({L.WORLD_MAX[0]:.2f},{L.WORLD_MAX[1]:.2f})"),
    ]:
        x_px, y_px = to_svg(label_x, label_y, world_h)
        parts.append(
            f'<text x="{x_px + anchor_x}" y="{y_px + anchor_y}" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="9" '
            f'fill="#605954">{anchor_text}</text>'
        )

    parts.append('</svg>')
    return "".join(parts)


def main() -> None:
    print(L.describe())
    print()
    print(f"Total parking slots in approach order:")
    for index, slot in enumerate(L.ALL_DEPOT_SLOTS):
        print(f"  A{index + 1}: ({slot[0]:6.2f}, {slot[1]:6.2f})")

    repo_root = pathlib.Path(__file__).resolve().parent
    out_path = repo_root / "Physical Set Up" / "cad_layout.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_svg(), encoding="utf-8")
    print()
    print(f"SVG written to {out_path}")


if __name__ == "__main__":
    main()
