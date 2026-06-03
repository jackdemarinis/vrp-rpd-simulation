"""CAD-derived layout constants for the new 8x8 + L-depot world.

Source: `Physical Set Up/Configuration 1.obj`, parsed 2026-05-21.
Raw CAD units are centimeters; all values here are inches and translated
so the south-west grid intersection sits at (0, 0). +Y points "north"
(toward the depot) in this frame; rendering may flip Y for display.

This module is intentionally self-contained — it does not import anything
from the rest of the simulation. Nothing else imports it yet either; it's
a staging ground for the geometry migration.
"""

from __future__ import annotations

from typing import List, Tuple

Coord = Tuple[float, float]


# Raw CAD origin of the south-west grid intersection, in inches.
# Subtracted from every parsed point so the grid starts at (0, 0).
RAW_ORIGIN_OFFSET_IN: Coord = (13.30, 9.96)


# Grid geometry --------------------------------------------------------------

GRID_PITCH_IN: float = 9.88
GRID_ROWS: int = 8
GRID_COLS: int = 8

GRID_XS: List[float] = [round(i * GRID_PITCH_IN, 4) for i in range(GRID_COLS)]
GRID_YS: List[float] = [round(i * GRID_PITCH_IN, 4) for i in range(GRID_ROWS)]

# Convenience: all 64 grid intersection coordinates, row-major from SW.
GRID_INTERSECTIONS: List[Coord] = [
    (x, y) for y in GRID_YS for x in GRID_XS
]


# White squares (work cells) -------------------------------------------------
#
# The gray corridors run *along* the grid lines, so the cream squares are the
# gaps between four adjacent intersections. An 8x8 intersection grid therefore
# encloses a 7x7 = 49 array of white squares. Each square is indexed (cix, ciy)
# by its SW-corner column/row, cix in 0..COLS-2, ciy in 0..ROWS-2.

WHITE_SQUARE_COLS: int = GRID_COLS - 1
WHITE_SQUARE_ROWS: int = GRID_ROWS - 1


def white_square_center(cix: int, ciy: int) -> Coord:
    """Center of white square (cix, ciy), where a station is serviced."""
    return (
        GRID_XS[cix] + GRID_PITCH_IN / 2.0,
        GRID_YS[ciy] + GRID_PITCH_IN / 2.0,
    )


def white_square_north_entry(cix: int, ciy: int) -> Coord:
    """North-edge entry point of square (cix, ciy).

    Sits on the top corridor (y = GRID_YS[ciy + 1]) directly above the
    center. A robot turns south here to dip into the square — the only way
    in or out (dead-end pocket).
    """
    return (
        GRID_XS[cix] + GRID_PITCH_IN / 2.0,
        GRID_YS[ciy + 1],
    )


# Depot geometry -------------------------------------------------------------

# The depot is an L-shape that hooks into the NE grid corner.
# All slot pitches are exactly 4 inches in the CAD model.
DEPOT_SLOT_PITCH_IN: float = 4.0

# Horizontal arm: 7 slots running along y = 81.16, from x = 45.16 to x = 69.16.
DEPOT_ARM_Y_IN: float = 81.16
DEPOT_ARM_XS: List[float] = [45.16, 49.16, 53.16, 57.16, 61.16, 65.16, 69.16]
DEPOT_HORIZONTAL_SLOTS: List[Coord] = [(x, DEPOT_ARM_Y_IN) for x in DEPOT_ARM_XS]

# Vertical arm: 2 slots between the horizontal arm and the grid corner,
# at x = 69.16, y = 73.16 and 77.16.
DEPOT_VERTICAL_X_IN: float = 69.16
DEPOT_VERTICAL_YS: List[float] = [73.16, 77.16]
DEPOT_VERTICAL_SLOTS: List[Coord] = [(DEPOT_VERTICAL_X_IN, y) for y in DEPOT_VERTICAL_YS]

# The 10th parking slot is the NE grid corner intersection itself.
GRID_CORNER_DEPOT_SLOT: Coord = (GRID_XS[-1], GRID_YS[-1])  # (69.16, 69.16)


def depot_slots_in_approach_order() -> List[Coord]:
    """All 10 parking slots in the order an Alvik traverses on the way home.

    Order: enter at the NE grid corner, drive north along the vertical arm,
    turn west onto the horizontal arm, and park rightmost-first.
    Slot 0 is the grid-corner station; slot 9 is the far-west arm slot.
    """
    return [
        GRID_CORNER_DEPOT_SLOT,
        *DEPOT_VERTICAL_SLOTS,                # (69.16, 73.16), (69.16, 77.16)
        *list(reversed(DEPOT_HORIZONTAL_SLOTS)),  # (69.16, 81.16) ... (45.16, 81.16)
    ]


ALL_DEPOT_SLOTS: List[Coord] = depot_slots_in_approach_order()

ALVIK_COUNT: int = 10  # one per depot slot


# World bounds (for rendering / sanity checks) -------------------------------

WORLD_MIN: Coord = (
    min(x for x, _ in GRID_INTERSECTIONS + ALL_DEPOT_SLOTS),
    min(y for _, y in GRID_INTERSECTIONS + ALL_DEPOT_SLOTS),
)
WORLD_MAX: Coord = (
    max(x for x, _ in GRID_INTERSECTIONS + ALL_DEPOT_SLOTS),
    max(y for _, y in GRID_INTERSECTIONS + ALL_DEPOT_SLOTS),
)
WORLD_WIDTH_IN: float = WORLD_MAX[0] - WORLD_MIN[0]
WORLD_HEIGHT_IN: float = WORLD_MAX[1] - WORLD_MIN[1]


def describe() -> str:
    """Human-readable summary of the layout. Useful for quick eyeballing."""
    lines = [
        f"CAD layout (inches, SW grid corner at origin):",
        f"  Grid: {GRID_COLS}x{GRID_ROWS} intersections, pitch {GRID_PITCH_IN} in",
        f"    x = {GRID_XS}",
        f"    y = {GRID_YS}",
        f"  Stations: {WHITE_SQUARE_COLS * WHITE_SQUARE_ROWS} white-square centers "
        f"({WHITE_SQUARE_COLS}x{WHITE_SQUARE_ROWS}), entered from the north",
        f"  Depot: L-shape, {len(ALL_DEPOT_SLOTS)} parking slots, "
        f"pitch {DEPOT_SLOT_PITCH_IN} in",
        f"    horizontal arm at y = {DEPOT_ARM_Y_IN}, x in {DEPOT_ARM_XS}",
        f"    vertical arm at x = {DEPOT_VERTICAL_X_IN}, y in {DEPOT_VERTICAL_YS}",
        f"    NE grid corner doubles as slot 0 at {GRID_CORNER_DEPOT_SLOT}",
        f"  Fleet: ALVIK_COUNT = {ALVIK_COUNT}",
        f"  World bounds: {WORLD_MIN} to {WORLD_MAX} "
        f"({WORLD_WIDTH_IN:.2f} x {WORLD_HEIGHT_IN:.2f} in)",
    ]
    return "\n".join(lines)
