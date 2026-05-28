"""Shared data structures for the VRP-RPD simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .config import InstanceConfig

Coord = Tuple[float, float]
Route = List["Operation"]
Routes = List[Route]


@dataclass(frozen=True)
class Operation:
    customer_id: int
    kind: str  # "D" or "P"

    @property
    def key(self) -> str:
        return f"{self.kind}{self.customer_id}"


@dataclass(frozen=True)
class Station:
    station_id: int
    name: str
    side: str
    node_id: str
    coord: Coord


@dataclass
class WorldGraph:
    coords: Dict[str, Coord]
    edges: Dict[str, List[Tuple[str, float]]]
    distances: Dict[str, Dict[str, float]]
    shortest_paths: Dict[Tuple[str, str], List[str]]
    road_xs: List[float]
    road_ys: List[float]
    depot_anchor: Coord
    depot_slots: List[Coord]
    depot_entries: Dict[str, List[Coord]]
    road_gap_in: float
    lane_center_offset_in: float


@dataclass
class Instance:
    world: WorldGraph
    stations: List[Station]
    active_job_ids: List[int]
    vehicle_count: int
    capacity: int
    processing_times: Dict[int, float]
    instance_config: "InstanceConfig"
    depot_node: str = "depot"
    station_by_id: Dict[int, Station] = field(init=False)

    def __post_init__(self) -> None:
        self.station_by_id = {station.station_id: station for station in self.stations}


@dataclass
class OperationEvent:
    vehicle_id: int
    op_index: int
    operation: Operation
    node_id: str
    arrival_time: float
    completion_time: float
    wait_time: float
    load_before: int
    load_after: int


@dataclass
class EvaluatedSolution:
    routes: Routes
    feasible: bool
    makespan: float
    events_by_vehicle: Dict[int, List[OperationEvent]]
    drop_times: Dict[int, float]
    pickup_times: Dict[int, float]
    return_times: Dict[int, float]
    drop_vehicle: Dict[int, int]
    pickup_vehicle: Dict[int, int]
    critical_vehicle: int
    completion_times: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def completed_jobs(self) -> int:
        return len(self.pickup_times)

    @property
    def active_jobs(self) -> int:
        active = 0
        for customer_id, drop_time in self.drop_times.items():
            if customer_id not in self.pickup_times and drop_time >= 0:
                active += 1
        return active


@dataclass(frozen=True)
class NodeVisit:
    """One time-stamped occupancy on the grid by a vehicle.

    A vehicle is at `node_id` over the closed interval `[t_enter, t_exit]`.
    Between consecutive visits the vehicle moves along the connecting edge
    at constant speed. `is_service=True` marks a planned dwell at a
    customer node (drop or pickup) or the depot return.
    """

    node_id: str
    t_enter: float
    t_exit: float
    is_service: bool = False
    customer_id: int | None = None
    op_kind: str | None = None  # "D", "P", or None


@dataclass
class ScheduledSolution:
    """Per-vehicle collision-free space-time schedule.

    Produced by the MAPF post-processing stage from an EvaluatedSolution.
    `base` is the original solver output (kept untouched). `paths[vid]` is
    the sequence of grid-level NodeVisits for vehicle `vid` from depot
    departure through depot return. `return_times` is the MAPF-adjusted
    per-vehicle finish time (>= base.return_times[vid] for every v).
    """

    base: EvaluatedSolution
    paths: Dict[int, List[NodeVisit]]
    return_times: Dict[int, float]
    makespan: float
    priority_order: List[int]
    reason: str = "ok"
