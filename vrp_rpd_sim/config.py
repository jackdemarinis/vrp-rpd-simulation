"""Editable simulation constants and runtime configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from . import cad_layout

# World geometry -------------------------------------------------------------
#
# Sourced from `Physical Set Up/Configuration 1.obj`. The 8x8 grid sits with
# its SW intersection at (0,0). The depot is an L-shape that hooks into the
# NE grid corner. See `vrp_rpd_sim.cad_layout` for the raw coordinates.

WORLD_WIDTH_IN = cad_layout.WORLD_WIDTH_IN
WORLD_HEIGHT_IN = cad_layout.WORLD_HEIGHT_IN
WORLD_MIN = cad_layout.WORLD_MIN
WORLD_MAX = cad_layout.WORLD_MAX

# Road visual width — single lane, two-way. Wide enough for one Alvik plus
# a little visual padding so traffic reads as on-road.
ROAD_WIDTH_IN = 4.0

# Lane center offset: 0 in the new world. Roads are single-lane, so the
# lane center is the road center. Kept for backward compat with the
# solver's `_lane_point` helper, which still consults this value.
LANE_CENTER_OFFSET_IN = 0.0

# Right-hand traffic flag is moot in single-lane world but the solver
# multiplies by LANE_DIRECTION_SIGN, so keep it at 1 (no effect when
# the offset is 0).
DRIVE_SIDE_RIGHT = True
LANE_DIRECTION_SIGN = 1 if DRIVE_SIDE_RIGHT else -1

# Alvik and depot ------------------------------------------------------------

ALVIK_SIZE_CM = 9.6
ALVIK_SIZE_IN = ALVIK_SIZE_CM / 2.54
MIN_ALVIK_CLEARANCE_IN = 0.5

# Bumped from 9 → 10 to match the CAD depot (7-arm + 2-vertical + grid-corner).
ALVIK_COUNT = cad_layout.ALVIK_COUNT

# The paper separates transport vehicles from identical resources. This demo
# uses the visible Alviks as the transport agents and keeps capacity editable.
VEHICLE_CAPACITY = 4

# Reference anchor for the depot. Sweep clustering in the solver uses this
# point for the atan2 polar sort. We pick the centroid of the 10 parking
# slots so the angular ordering wraps the whole queue.
DEPOT_ANCHOR_IN = (
    sum(x for x, _ in cad_layout.ALL_DEPOT_SLOTS) / len(cad_layout.ALL_DEPOT_SLOTS),
    sum(y for _, y in cad_layout.ALL_DEPOT_SLOTS) / len(cad_layout.ALL_DEPOT_SLOTS),
)

# Stations and processing ----------------------------------------------------

# The 49 white squares (7x7 cells between the corridors) are the stations now
# (vs. one-per-intersection, which had no physical work cell to pull into).
ACTIVE_JOB_COUNT = 49
PROCESSING_TIME_VARIANT = "base"  # "base", "2x", "5x", "1R10", "1R20"
PROCESSING_TIME_SEED = 7
PROCESSING_TIME_SCALE = 1.0
FIXED_PROCESSING_TIME_SEC = None

# Per-action service dwell, in seconds: how long an Alvik pauses at a cell to
# perform a drop (D) or a pickup (P). Added on top of the station's processing
# time (the drop→ready wait). 0.0 keeps the original instantaneous behavior.
DROP_TIME_SEC = 0.0
PICKUP_TIME_SEC = 0.0

PROCESSING_TIME_OVERRIDES: dict[int, float] = {}

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

# Runtime config -------------------------------------------------------------

DEFAULT_RUNTIME_CONFIG_PATH = Path("simulation_config.json")
DEFAULT_SOLUTION_CACHE_DIR = ".solution_cache"
SOLUTION_CACHE_SCHEMA_VERSION = 2
SOLUTION_CACHE_ALGORITHM_VERSION = "2026-06-03-squares"


@dataclass(frozen=True)
class InstanceConfig:
    active_job_count: int = ACTIVE_JOB_COUNT
    vehicle_count: int = ALVIK_COUNT
    vehicle_capacity: int = VEHICLE_CAPACITY
    alvik_speed_in_per_sec: float = ALVIK_SPEED_IN_PER_SEC
    processing_time_variant: str = PROCESSING_TIME_VARIANT
    processing_time_seed: int = PROCESSING_TIME_SEED
    processing_time_scale: float = PROCESSING_TIME_SCALE
    fixed_processing_time_sec: float | None = FIXED_PROCESSING_TIME_SEC
    drop_time_sec: float = DROP_TIME_SEC
    pickup_time_sec: float = PICKUP_TIME_SEC
    processing_time_overrides: dict[int, float] = field(
        default_factory=lambda: dict(PROCESSING_TIME_OVERRIDES)
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "processing_time_overrides",
            {
                int(station_id): float(value)
                for station_id, value in self.processing_time_overrides.items()
            },
        )
        if self.active_job_count < 1:
            raise ValueError("active_job_count must be at least 1.")
        if self.vehicle_count < 1:
            raise ValueError("vehicle_count must be at least 1.")
        if self.vehicle_capacity < 1:
            raise ValueError("vehicle_capacity must be at least 1.")
        if self.alvik_speed_in_per_sec <= 0:
            raise ValueError("alvik_speed_in_per_sec must be positive.")
        if self.processing_time_scale <= 0:
            raise ValueError("processing_time_scale must be positive.")
        if self.drop_time_sec < 0:
            raise ValueError("drop_time_sec must be non-negative.")
        if self.pickup_time_sec < 0:
            raise ValueError("pickup_time_sec must be non-negative.")


@dataclass(frozen=True)
class SolverConfig:
    random_seed: int = SOLVER_RANDOM_SEED
    alns_max_iterations: int = ALNS_MAX_ITERATIONS
    alns_initial_temperature_coeff: float = ALNS_INITIAL_TEMPERATURE_COEFF
    alns_cooling_rate: float = ALNS_COOLING_RATE
    alns_reheat_factor: float = ALNS_REHEAT_FACTOR
    alns_stagnation_threshold: int = ALNS_STAGNATION_THRESHOLD
    alns_weight_update_interval: int = ALNS_WEIGHT_UPDATE_INTERVAL
    alns_reaction_factor: float = ALNS_REACTION_FACTOR
    alns_pickup_reposition_interval: int = ALNS_PICKUP_REPOSITION_INTERVAL
    alns_cross_agent_relocate_interval: int = ALNS_CROSS_AGENT_RELOCATE_INTERVAL
    alns_score_sigma_1: int = ALNS_SCORE_SIGMA_1
    alns_score_sigma_2: int = ALNS_SCORE_SIGMA_2
    alns_score_sigma_3: int = ALNS_SCORE_SIGMA_3
    alns_min_operator_weight: float = ALNS_MIN_OPERATOR_WEIGHT
    alns_worst_removal_power: int = ALNS_WORST_REMOVAL_POWER
    alns_shaw_phi: int = ALNS_SHAW_PHI
    alns_shaw_chi: int = ALNS_SHAW_CHI
    alns_shaw_omega: int = ALNS_SHAW_OMEGA
    brkga_population_size: int = BRKGA_POPULATION_SIZE
    brkga_elite_proportion: float = BRKGA_ELITE_PROPORTION
    brkga_mutant_proportion: float = BRKGA_MUTANT_PROPORTION
    brkga_elite_bias: float = BRKGA_ELITE_BIAS
    brkga_generations: int = BRKGA_GENERATIONS
    brkga_warm_start_proportion: float = BRKGA_WARM_START_PROPORTION
    brkga_warm_seed_count: int = BRKGA_WARM_SEED_COUNT
    brkga_gene_perturbation: float = BRKGA_GENE_PERTURBATION
    brkga_infeasibility_penalty: float = float(BRKGA_INFEASIBILITY_PENALTY)

    def __post_init__(self) -> None:
        positive_ints = {
            "alns_max_iterations": self.alns_max_iterations,
            "alns_stagnation_threshold": self.alns_stagnation_threshold,
            "alns_weight_update_interval": self.alns_weight_update_interval,
            "alns_pickup_reposition_interval": self.alns_pickup_reposition_interval,
            "alns_cross_agent_relocate_interval": self.alns_cross_agent_relocate_interval,
            "brkga_population_size": self.brkga_population_size,
            "brkga_generations": self.brkga_generations,
            "brkga_warm_seed_count": self.brkga_warm_seed_count,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1.")
        proportions = {
            "alns_reaction_factor": self.alns_reaction_factor,
            "brkga_elite_proportion": self.brkga_elite_proportion,
            "brkga_mutant_proportion": self.brkga_mutant_proportion,
            "brkga_elite_bias": self.brkga_elite_bias,
            "brkga_warm_start_proportion": self.brkga_warm_start_proportion,
        }
        for name, value in proportions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.alns_initial_temperature_coeff <= 0:
            raise ValueError("alns_initial_temperature_coeff must be positive.")
        if not 0.0 < self.alns_cooling_rate < 1.0:
            raise ValueError("alns_cooling_rate must be between 0 and 1.")
        if self.alns_reheat_factor <= 0:
            raise ValueError("alns_reheat_factor must be positive.")
        if self.alns_min_operator_weight <= 0:
            raise ValueError("alns_min_operator_weight must be positive.")
        if self.alns_worst_removal_power < 1:
            raise ValueError("alns_worst_removal_power must be at least 1.")
        if self.brkga_gene_perturbation < 0:
            raise ValueError("brkga_gene_perturbation must be non-negative.")
        if self.brkga_infeasibility_penalty <= 0:
            raise ValueError("brkga_infeasibility_penalty must be positive.")


@dataclass(frozen=True)
class AppConfig:
    sim_speed_multiplier: float = SIM_SPEED_MULTIPLIER
    start_fullscreen: bool = START_FULLSCREEN

    def __post_init__(self) -> None:
        if self.sim_speed_multiplier <= 0:
            raise ValueError("sim_speed_multiplier must be positive.")


@dataclass(frozen=True)
class MapfConfig:
    """Settings for the offline MAPF (Multi-Agent Path Finding) stage that
    rewrites the solver's routes into a collision-free space-time schedule.

    Priority order: slowest agent (largest VRP `return_time`) plans first
    on its solver-derived Dijkstra path; subsequent agents space-time A*
    against a reservation table, waiting or detouring as needed.
    """

    enabled: bool = True
    # Time gap between two vehicles' uses of a shared node/edge. Must
    # be at least body_diameter * sqrt(2) / speed so two vehicles
    # approaching the same intersection from perpendicular edges don't
    # have their 3.78" bodies overlap at the intersection corner.
    # 3.78 * sqrt(2) / 5.12 ≈ 1.04s; bump slightly for safety margin.
    clearance_sec: float = 1.1
    # Cap A* search horizon as a multiple of the solver's makespan.
    max_horizon_multiplier: float = 8.0
    # Time the L-shaped depot corridor blocks the depot mouth on entry/exit.
    # The MAPF planner reserves the depot vertex for this interval before a
    # vehicle's grid trip starts and after it ends so two vehicles never
    # share the corridor. Must be >= the longest actual corridor walk so
    # the physical traversal fits inside the reservation. Deepest slot
    # (slot 9, distance 36in) needs 36/5.118 ≈ 7.03s; round up for safety.
    corridor_traverse_sec: float = 7.5
    # Fallback chain when a lower-priority agent has no space-time path.
    fallback_delay_initial_sec: float = 5.0
    fallback_delay_growth: float = 1.8
    fallback_max_delay_attempts: int = 5
    fallback_max_swap_attempts: int = 8
    # Stage-3 fallback: if both delay and pairwise swap exhaust, retry
    # the entire plan with ascending-route-length priority order.
    # Disabled by default because depot slot assignment is coupled to
    # makespan-descending order — any alternate priority leaves the slot-0
    # vehicle (parked at the depot mouth) NOT planned first, which causes
    # corridor collisions.
    fallback_allow_reverse_priority: bool = False

    def __post_init__(self) -> None:
        if self.clearance_sec < 0:
            raise ValueError("clearance_sec must be non-negative.")
        if self.max_horizon_multiplier <= 0:
            raise ValueError("max_horizon_multiplier must be positive.")
        if self.corridor_traverse_sec < 0:
            raise ValueError("corridor_traverse_sec must be non-negative.")
        if self.fallback_delay_initial_sec < 0:
            raise ValueError("fallback_delay_initial_sec must be non-negative.")
        if self.fallback_delay_growth <= 0:
            raise ValueError("fallback_delay_growth must be positive.")
        if self.fallback_max_delay_attempts < 0:
            raise ValueError("fallback_max_delay_attempts must be non-negative.")
        if self.fallback_max_swap_attempts < 0:
            raise ValueError("fallback_max_swap_attempts must be non-negative.")


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    directory: str = DEFAULT_SOLUTION_CACHE_DIR
    schema_version: int = SOLUTION_CACHE_SCHEMA_VERSION
    algorithm_version: str = SOLUTION_CACHE_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1.")
        if not self.directory:
            raise ValueError("directory must not be empty.")


@dataclass(frozen=True)
class RuntimeConfig:
    instance: InstanceConfig = field(default_factory=InstanceConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    app: AppConfig = field(default_factory=AppConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    mapf: MapfConfig = field(default_factory=MapfConfig)


def default_runtime_config() -> RuntimeConfig:
    return RuntimeConfig()


def load_runtime_config(path: str | Path | None = None) -> RuntimeConfig:
    config_path = Path(path) if path is not None else DEFAULT_RUNTIME_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Runtime config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        raw = json.load(config_file) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Runtime config root must be a JSON object.")

    defaults = default_runtime_config()
    return RuntimeConfig(
        instance=_load_dataclass(InstanceConfig, defaults.instance, raw.get("instance")),
        solver=_load_dataclass(SolverConfig, defaults.solver, raw.get("solver")),
        app=_load_dataclass(AppConfig, defaults.app, raw.get("app")),
        cache=_load_dataclass(CacheConfig, defaults.cache, raw.get("cache")),
        mapf=_load_dataclass(MapfConfig, defaults.mapf, raw.get("mapf")),
    )


def apply_cli_overrides(
    runtime_config: RuntimeConfig,
    *,
    job_count: int | None = None,
    processing_scale: float | None = None,
    fixed_processing_time: float | None = None,
    sim_speed: float | None = None,
    fullscreen: bool | None = None,
    seed: int | None = None,
) -> RuntimeConfig:
    instance_config = runtime_config.instance
    app_config = runtime_config.app
    solver_config = runtime_config.solver

    if job_count is not None:
        instance_config = replace(instance_config, active_job_count=job_count)
    if processing_scale is not None:
        instance_config = replace(instance_config, processing_time_scale=processing_scale)
    if fixed_processing_time is not None:
        instance_config = replace(instance_config, fixed_processing_time_sec=fixed_processing_time)
    if sim_speed is not None:
        app_config = replace(app_config, sim_speed_multiplier=sim_speed)
    if fullscreen is not None:
        app_config = replace(app_config, start_fullscreen=fullscreen)
    if seed is not None:
        solver_config = replace(solver_config, random_seed=seed)

    return replace(
        runtime_config,
        instance=instance_config,
        app=app_config,
        solver=solver_config,
    )


def _load_dataclass[T](
    dataclass_type: type[T],
    defaults: T,
    raw_values: Any,
) -> T:
    if raw_values is None:
        return defaults
    if not isinstance(raw_values, Mapping):
        raise ValueError(f"{dataclass_type.__name__} overrides must be a JSON object.")
    return replace(defaults, **dict(raw_values))


# Misc ----------------------------------------------------------------------

APP_TITLE = "VRP-RPD CAD-Layout Simulation"
