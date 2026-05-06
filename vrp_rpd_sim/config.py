"""Editable simulation constants and runtime configuration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

# World geometry -------------------------------------------------------------

WORLD_SIZE_IN = 120.0  # 10 ft x 10 ft

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

# Alvik and depot ------------------------------------------------------------

ALVIK_SIZE_CM = 9.6
ALVIK_SIZE_IN = ALVIK_SIZE_CM / 2.54
MIN_ALVIK_CLEARANCE_IN = 0.5

ALVIK_COUNT = 9

# The paper separates transport vehicles from identical resources. This demo
# uses the 9 visible Alviks as the transport agents and keeps capacity editable.
VEHICLE_CAPACITY = 4

DEPOT_STACK_COLUMNS = 3
DEPOT_STACK_ROWS = 3
# Slightly larger than the minimum clearance so robots can peel out of the
# waffle stack without immediate deadlock while still respecting the
# 0.5-inch minimum robot-to-robot spacing.
DEPOT_STACK_GAP_IN = 0.75
DEPOT_ANCHOR_IN = (6.5, 6.5)

# Stations and processing ----------------------------------------------------

ACTIVE_JOB_COUNT = 36
PROCESSING_TIME_VARIANT = "base"  # "base", "2x", "5x", "1R10", "1R20"
PROCESSING_TIME_SEED = 7
PROCESSING_TIME_SCALE = 1.0
FIXED_PROCESSING_TIME_SEC = None

# Optional explicit overrides in simulation seconds.
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
DEPOT_APPROACH_RESERVATION_IN = ROAD_ENVELOPE_WIDTH_IN + ALVIK_SIZE_IN + MIN_ALVIK_CLEARANCE_IN

# Display lane center on each duct-tape strip rather than the road midpoint.
LANE_CENTER_OFFSET_IN = (ROAD_MEDIAN_GAP_IN / 2.0) + (ROAD_STRIP_WIDTH_IN / 2.0)
DEPOT_ENTRY_LANE_OFFSET_IN = LANE_CENTER_OFFSET_IN
DEPOT_ENTRY_TOP_COUNT = 3
DEPOT_ENTRY_RIGHT_COUNT = 3

# Right-hand traffic: northbound on east half (+X), eastbound on south half (-Y).
# Flip to False for left-hand traffic.
DRIVE_SIDE_RIGHT = True
LANE_DIRECTION_SIGN = 1 if DRIVE_SIDE_RIGHT else -1

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
SOLUTION_CACHE_SCHEMA_VERSION = 1
SOLUTION_CACHE_ALGORITHM_VERSION = "2026-04-30"


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

APP_TITLE = "VRP-RPD Manhattan Grid Simulation"
