"""CAD-derived layout constants, extended to a 3D cube world.

Source: `Physical Set Up/Configuration 1.obj`, parsed 2026-05-21. The
original 8x8 grid is replicated up a third (Z / depth) axis to form an
8x8x7 intersection lattice — "as if the agents were under water and the
stations were in a cube instead of a square." Z uses the same pitch as
X/Y. The south-west-bottom grid intersection sits at (0, 0, 0); +Y points
"north" (toward the depot) and +Z points "up". Stations fill every Z-plane
(7x7x7 = 343 cells); the depot shares the top plane (its NE corner) rather
than owning a separate layer. Rendering may flip axes for display.

This module is intentionally self-contained — it does not import anything
from the rest of the simulation.
"""

from __future__ import annotations

from typing import List, Tuple

Coord = Tuple[float, float, float]


# Raw CAD origin of the south-west grid intersection, in inches.
# Subtracted from every parsed point so the grid starts at (0, 0).
RAW_ORIGIN_OFFSET_IN: Tuple[float, float] = (13.30, 9.96)


# Grid geometry --------------------------------------------------------------

GRID_PITCH_IN: float = 9.88
GRID_ROWS: int = 8
GRID_COLS: int = 8
# Number of Z (depth) planes. Stations sit on every plane; the depot shares
# the top plane (its NE corner), so there is no dedicated empty depot layer.
# 7 planes x 7x7 cells = 343 stations.
GRID_LAYERS: int = 7

GRID_XS: List[float] = [round(i * GRID_PITCH_IN, 4) for i in range(GRID_COLS)]
GRID_YS: List[float] = [round(i * GRID_PITCH_IN, 4) for i in range(GRID_ROWS)]
GRID_ZS: List[float] = [round(i * GRID_PITCH_IN, 4) for i in range(GRID_LAYERS)]

# Convenience: all grid intersection coordinates, layer-major from the
# SW-bottom corner (z outermost, then y, then x).
GRID_INTERSECTIONS: List[Coord] = [
    (x, y, z) for z in GRID_ZS for y in GRID_YS for x in GRID_XS
]


# White squares (work cells) -------------------------------------------------
#
# The gray corridors run *along* the grid lines, so the cream squares are the
# gaps between four adjacent intersections. An 8x8 intersection grid therefore
# encloses a 7x7 = 49 array of white squares. Each square is indexed (cix, ciy)
# by its SW-corner column/row, cix in 0..COLS-2, ciy in 0..ROWS-2.

WHITE_SQUARE_COLS: int = GRID_COLS - 1
WHITE_SQUARE_ROWS: int = GRID_ROWS - 1
# Stations occupy every Z-plane (including the top one, which also hosts the
# depot at its NE corner) — so there are GRID_LAYERS station planes.
WHITE_SQUARE_LAYERS: int = GRID_LAYERS


def white_square_center(cix: int, ciy: int, ciz: int) -> Coord:
    """Center of station cell (cix, ciy) on Z-plane ciz, where it's serviced.

    Stations live on every Z-plane (ciz in 0..WHITE_SQUARE_LAYERS-1). The
    center sits exactly on its Z-plane and is offset to the cell midpoint in
    X/Y, so each plane reproduces the original 2D pocket layout. The depot
    shares the top plane (its NE corner) rather than owning a separate layer.
    """
    return (
        GRID_XS[cix] + GRID_PITCH_IN / 2.0,
        GRID_YS[ciy] + GRID_PITCH_IN / 2.0,
        GRID_ZS[ciz],
    )


def white_square_north_entry(cix: int, ciy: int, ciz: int) -> Coord:
    """North-edge entry point of station cell (cix, ciy) on Z-plane ciz.

    Sits on the top corridor of the cell's own Z-plane (y = GRID_YS[ciy + 1],
    z = GRID_ZS[ciz]) directly above the center. A robot turns south here to
    dip into the center — the only way in or out (dead-end pocket). Vertical
    (Z) travel happens on the intersection lattice, not through the pocket.
    """
    return (
        GRID_XS[cix] + GRID_PITCH_IN / 2.0,
        GRID_YS[ciy + 1],
        GRID_ZS[ciz],
    )


# Depot geometry -------------------------------------------------------------

# The depot is an L-shape that hooks into the NE grid corner. It stays flat
# on the top Z-layer; outbound vehicles egress here and descend into the
# cube. All slot pitches are exactly 4 inches in the CAD model.
DEPOT_SLOT_PITCH_IN: float = 4.0

# Z of the depot plane: the top grid layer.
DEPOT_Z_IN: float = GRID_ZS[-1]

# Horizontal arm: 7 slots running along y = 81.16, from x = 45.16 to x = 69.16.
DEPOT_ARM_Y_IN: float = 81.16
DEPOT_ARM_XS: List[float] = [45.16, 49.16, 53.16, 57.16, 61.16, 65.16, 69.16]
DEPOT_HORIZONTAL_SLOTS: List[Coord] = [
    (x, DEPOT_ARM_Y_IN, DEPOT_Z_IN) for x in DEPOT_ARM_XS
]

# Vertical arm: 2 slots between the horizontal arm and the grid corner,
# at x = 69.16, y = 73.16 and 77.16.
DEPOT_VERTICAL_X_IN: float = 69.16
DEPOT_VERTICAL_YS: List[float] = [73.16, 77.16]
DEPOT_VERTICAL_SLOTS: List[Coord] = [
    (DEPOT_VERTICAL_X_IN, y, DEPOT_Z_IN) for y in DEPOT_VERTICAL_YS
]

# The 10th parking slot is the NE grid corner intersection (top layer).
GRID_CORNER_DEPOT_SLOT: Coord = (GRID_XS[-1], GRID_YS[-1], GRID_ZS[-1])


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

_ALL_POINTS: List[Coord] = GRID_INTERSECTIONS + ALL_DEPOT_SLOTS
WORLD_MIN: Coord = (
    min(p[0] for p in _ALL_POINTS),
    min(p[1] for p in _ALL_POINTS),
    min(p[2] for p in _ALL_POINTS),
)
WORLD_MAX: Coord = (
    max(p[0] for p in _ALL_POINTS),
    max(p[1] for p in _ALL_POINTS),
    max(p[2] for p in _ALL_POINTS),
)
WORLD_WIDTH_IN: float = WORLD_MAX[0] - WORLD_MIN[0]
WORLD_HEIGHT_IN: float = WORLD_MAX[1] - WORLD_MIN[1]
WORLD_DEPTH_IN: float = WORLD_MAX[2] - WORLD_MIN[2]


def describe() -> str:
    """Human-readable summary of the layout. Useful for quick eyeballing."""
    station_count = WHITE_SQUARE_COLS * WHITE_SQUARE_ROWS * WHITE_SQUARE_LAYERS
    lines = [
        f"CAD layout (inches, SW-bottom grid corner at origin):",
        f"  Lattice: {GRID_COLS}x{GRID_ROWS}x{GRID_LAYERS} intersections, "
        f"pitch {GRID_PITCH_IN} in",
        f"    x = {GRID_XS}",
        f"    y = {GRID_YS}",
        f"    z = {GRID_ZS}",
        f"  Stations: {station_count} cell centers "
        f"({WHITE_SQUARE_COLS}x{WHITE_SQUARE_ROWS}x{WHITE_SQUARE_LAYERS}), "
        f"entered from the north of each Z-layer",
        f"  Depot: L-shape on top layer (z={DEPOT_Z_IN}), "
        f"{len(ALL_DEPOT_SLOTS)} parking slots, pitch {DEPOT_SLOT_PITCH_IN} in",
        f"    horizontal arm at y = {DEPOT_ARM_Y_IN}, x in {DEPOT_ARM_XS}",
        f"    vertical arm at x = {DEPOT_VERTICAL_X_IN}, y in {DEPOT_VERTICAL_YS}",
        f"    NE grid corner doubles as slot 0 at {GRID_CORNER_DEPOT_SLOT}",
        f"  Fleet: ALVIK_COUNT = {ALVIK_COUNT}",
        f"  World bounds: {WORLD_MIN} to {WORLD_MAX} "
        f"({WORLD_WIDTH_IN:.2f} x {WORLD_HEIGHT_IN:.2f} x {WORLD_DEPTH_IN:.2f} in)",
    ]
    return "\n".join(lines)
