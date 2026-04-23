"""Editable simulation constants for the VRP-RPD Pygame demo."""

from __future__ import annotations

# World geometry -------------------------------------------------------------

WORLD_SIZE_IN = 96.0  # 8 ft x 8 ft

VERTICAL_ROAD_COUNT = 7
HORIZONTAL_ROAD_COUNT = 7

ROAD_STRIP_WIDTH_IN = 2.0
ROAD_MEDIAN_GAP_IN = 3.0
ROAD_ENVELOPE_WIDTH_IN = (2 * ROAD_STRIP_WIDTH_IN) + ROAD_MEDIAN_GAP_IN
LANE_DIVIDER_COLOR = (230, 190, 60)
LANE_DIVIDER_WIDTH_IN = 0.25
LANE_DIVIDER_DASH_IN = 2.0
LANE_DIVIDER_GAP_IN = 2.0

# Inferred from the user's requested depot footprint plus the required
# 7x7 Manhattan network. The paper itself uses a distance matrix rather
# than physical geometry, so this spacing is intentionally editable.
ROAD_EDGE_MARGIN_IN = 14.0

STATION_ROAD_INDEX_LAYOUT = {
    "bottom": [1, 2, 4, 5],
    "right": [1, 2, 4, 5],
    "top": [1, 2, 4, 5],
    "left": [1, 2, 4, 5],
}

# Alvik and depot ------------------------------------------------------------

ALVIK_SIZE_CM = 9.6
ALVIK_SIZE_IN = ALVIK_SIZE_CM / 2.54
MIN_ALVIK_CLEARANCE_IN = 0.5

ALVIK_COUNT = 9

# The paper separates transport vehicles from identical resources. This demo
# uses the 9 visible Alviks as the transport agents and assigns each one
# capacity 1 so the total number of in-system resources equals the number of
# Alviks. Keep editable.
VEHICLE_CAPACITY = 1

DEPOT_STACK_COLUMNS = 3
DEPOT_STACK_ROWS = 3
# Slightly larger than the minimum clearance so robots can peel out of the
# waffle stack without immediate deadlock while still respecting the
# 0.5-inch minimum robot-to-robot spacing.
DEPOT_STACK_GAP_IN = 0.75
DEPOT_ANCHOR_IN = (6.5, 6.5)

# Stations and processing ----------------------------------------------------

ACTIVE_JOB_COUNT = 16
PROCESSING_TIME_VARIANT = "base"  # "base", "2x", "5x", "1R10", "1R20"
PROCESSING_TIME_SEED = 7
PROCESSING_TIME_SCALE = 1.0
FIXED_PROCESSING_TIME_SEC = None

# Optional explicit overrides in simulation seconds.
PROCESSING_TIME_OVERRIDES = {}

# Motion and animation -------------------------------------------------------

# Arduino's official Alvik datasheet lists motion "up to 13 cm/s".
ALVIK_SPEED_IN_PER_SEC = 13.0 / 2.54
SIM_SPEED_MULTIPLIER = 6.0
FPS = 60

# Rendering -----------------------------------------------------------------

WINDOW_WIDTH_PX = 1000
WINDOW_HEIGHT_PX = 800
WORLD_PADDING_PX = 60
HUD_WIDTH_PX = 280
MIN_WINDOW_WIDTH_PX = 980
MIN_WINDOW_HEIGHT_PX = 760
START_FULLSCREEN = False
DEPOT_APPROACH_RESERVATION_IN = ROAD_ENVELOPE_WIDTH_IN + ALVIK_SIZE_IN + MIN_ALVIK_CLEARANCE_IN

# Display lane center on each duct-tape strip rather than the road midpoint.
LANE_CENTER_OFFSET_IN = (ROAD_MEDIAN_GAP_IN / 2.0) + (ROAD_STRIP_WIDTH_IN / 2.0)

# Solver --------------------------------------------------------------------

SOLVER_RANDOM_SEED = 7

# Paper-faithful structure, CPU-scaled budgets for a Python/Pygame demo.
ALNS_MAX_ITERATIONS = 8
ALNS_INITIAL_TEMPERATURE_COEFF = 0.30
ALNS_COOLING_RATE = 0.9998
ALNS_REHEAT_FACTOR = 0.50
ALNS_STAGNATION_THRESHOLD = 2000
ALNS_WEIGHT_UPDATE_INTERVAL = 100
ALNS_REACTION_FACTOR = 0.1
ALNS_PICKUP_REPOSITION_INTERVAL = 200
ALNS_CROSS_AGENT_RELOCATE_INTERVAL = 500
ALNS_SCORE_SIGMA_1 = 33
ALNS_SCORE_SIGMA_2 = 9
ALNS_SCORE_SIGMA_3 = 13
ALNS_MIN_OPERATOR_WEIGHT = 0.1
ALNS_WORST_REMOVAL_POWER = 6
ALNS_SHAW_PHI = 9
ALNS_SHAW_CHI = 3
ALNS_SHAW_OMEGA = 5

BRKGA_POPULATION_SIZE = 36
BRKGA_ELITE_PROPORTION = 0.15
BRKGA_MUTANT_PROPORTION = 0.15
BRKGA_ELITE_BIAS = 0.70
BRKGA_GENERATIONS = 24
BRKGA_WARM_START_PROPORTION = 0.15
BRKGA_WARM_SEED_COUNT = 8
BRKGA_GENE_PERTURBATION = 0.03
BRKGA_INFEASIBILITY_PENALTY = 10**6

# Misc ----------------------------------------------------------------------

APP_TITLE = "VRP-RPD Manhattan Grid Simulation"
