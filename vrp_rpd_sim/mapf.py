"""Offline MAPF (Multi-Agent Path Finding) post-processor.

Inserts a Prioritized Planning + Space-Time A* stage between the VRP solver
and the simulator so that runtime collisions and deadlocks are eliminated.

Algorithm summary
-----------------
1. Sort vehicles by `EvaluatedSolution.return_times` descending. The
   slowest-makespan vehicle goes first and plans on its solver-derived
   shortest path.
2. For each subsequent vehicle, run Space-Time A* over the grid graph
   while treating already-planned vehicles' reservations as moving
   obstacles. Wait actions and detours are added as needed.
3. Each agent's MAPF return time accumulates any waiting/detour delay.

Conflict model
--------------
- Vertex: two agents on the same node with overlapping time intervals.
- Edge / swap: two agents on (u, v) and (v, u) in overlapping time windows.
- Following: collapsed into vertex conflicts via a `clearance_sec` margin.

The depot vertex itself has unlimited capacity (parking is off-grid in
slots). To model the single-vehicle L-corridor between depot and its
parking slots, each vehicle exclusively reserves the depot vertex for
`corridor_traverse_sec` immediately before its grid trip starts and
immediately after it ends.

Dwells
------
Service waits at customer nodes (drops and pickups) come from the solver's
`OperationEvent.wait_time + 0` for pickups (the solver bakes in the
drop-then-process precedence) and 0 for drops. These dwells are upper
bounds because MAPF only delays things further; the simulator's
station-readiness check is a no-op for valid MAPF schedules.
"""

from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from .config import MapfConfig
from .model import (
    EvaluatedSolution,
    Instance,
    NodeVisit,
    Operation,
    OperationEvent,
    ScheduledSolution,
)
from .solver import VRPRPDSolver


EPS = 1e-6


class MapfInfeasible(RuntimeError):
    """Raised when the planner exhausts its fallback chain."""

    def __init__(self, vehicle_id: int, reason: str) -> None:
        super().__init__(f"MAPF infeasible for vehicle {vehicle_id}: {reason}")
        self.vehicle_id = vehicle_id
        self.reason = reason


# ---------------------------------------------------------------------------
# Reservation table
# ---------------------------------------------------------------------------


@dataclass
class _Interval:
    start: float
    end: float
    vehicle_id: int


class ReservationTable:
    """Continuous-time vertex/edge reservation store.

    `node_reservations[node]` is a list of `_Interval` sorted by `start`.
    `edge_reservations[(u, v)]` is keyed on the directed edge.
    Membership tests use binary search.

    Depot/mouth equivalence: the abstract `depot` node and the grid node
    that physically coincides with it (the depot mouth, typically
    `i_7_7`) share a single reservation bucket. Any reservation or query
    against `depot` is rerouted to the mouth bucket, so corridor windows
    block grid pass-throughs and vice versa.
    """

    def __init__(
        self,
        depot_node: str,
        clearance_sec: float,
        *,
        mouth_node: str | None = None,
    ) -> None:
        self._depot_node = depot_node
        self._mouth_node = mouth_node
        self._clearance_sec = clearance_sec
        self._node_reservations: Dict[str, List[_Interval]] = {}
        self._edge_reservations: Dict[Tuple[str, str], List[_Interval]] = {}

    def _canonical_node(self, node: str) -> str:
        # Depot and mouth share one physical spot, so collapse to one
        # bucket. This makes every existing depot/mouth check naturally
        # block the other.
        if self._mouth_node is not None and node == self._depot_node:
            return self._mouth_node
        return node

    def reserve_vertex(self, node: str, t_enter: float, t_exit: float, vehicle_id: int) -> None:
        if t_exit < t_enter:
            t_exit = t_enter
        canonical = self._canonical_node(node)
        interval = _Interval(t_enter, t_exit, vehicle_id)
        bucket = self._node_reservations.setdefault(canonical, [])
        bisect.insort(bucket, interval, key=lambda iv: iv.start)

    def reserve_edge(self, u: str, v: str, t_enter: float, t_exit: float, vehicle_id: int) -> None:
        if t_exit < t_enter:
            t_exit = t_enter
        interval = _Interval(t_enter, t_exit, vehicle_id)
        bucket = self._edge_reservations.setdefault((u, v), [])
        bisect.insort(bucket, interval, key=lambda iv: iv.start)

    def is_vertex_free(
        self,
        node: str,
        t_enter: float,
        t_exit: float,
        *,
        exclude_vehicle: int | None = None,
    ) -> bool:
        canonical = self._canonical_node(node)
        return not self._has_overlap(
            self._node_reservations.get(canonical, ()),
            t_enter - self._clearance_sec,
            t_exit + self._clearance_sec,
            exclude_vehicle,
        )

    def is_edge_free(
        self,
        u: str,
        v: str,
        t_enter: float,
        t_exit: float,
        *,
        exclude_vehicle: int | None = None,
    ) -> bool:
        # Same-direction traversal conflict.
        if self._has_overlap(
            self._edge_reservations.get((u, v), ()),
            t_enter - self._clearance_sec,
            t_exit + self._clearance_sec,
            exclude_vehicle,
        ):
            return False
        # Opposite-direction swap conflict.
        if self._has_overlap(
            self._edge_reservations.get((v, u), ()),
            t_enter - self._clearance_sec,
            t_exit + self._clearance_sec,
            exclude_vehicle,
        ):
            return False
        return True

    def earliest_vertex_free(
        self,
        node: str,
        not_before: float,
        duration: float,
        *,
        exclude_vehicle: int | None = None,
    ) -> float:
        """Return the earliest t >= not_before such that [t, t+duration] is
        clearance-free of conflicts at the node."""
        canonical = self._canonical_node(node)
        intervals = self._node_reservations.get(canonical, ())
        return self._scan_earliest(
            intervals, not_before, duration, exclude_vehicle, self._clearance_sec
        )

    def earliest_edge_free(
        self,
        u: str,
        v: str,
        not_before: float,
        duration: float,
        *,
        exclude_vehicle: int | None = None,
    ) -> float:
        """Earliest t >= not_before such that traversing (u,v) during
        [t, t+duration] has no same- or opposite-direction conflict."""
        t = not_before
        for _ in range(128):
            same = self._scan_earliest(
                self._edge_reservations.get((u, v), ()),
                t,
                duration,
                exclude_vehicle,
                self._clearance_sec,
            )
            opp = self._scan_earliest(
                self._edge_reservations.get((v, u), ()),
                t,
                duration,
                exclude_vehicle,
                self._clearance_sec,
            )
            advanced = max(same, opp)
            if advanced <= t + EPS:
                return t
            t = advanced
        return t

    def vehicle_reservations(self, vehicle_id: int) -> List[Tuple[str, _Interval]]:
        """Used by the swap-fallback heuristic to find a blocker."""
        out: List[Tuple[str, _Interval]] = []
        for node, intervals in self._node_reservations.items():
            for interval in intervals:
                if interval.vehicle_id == vehicle_id:
                    out.append((node, interval))
        return out

    def clear_vehicle(self, vehicle_id: int) -> None:
        for bucket in self._node_reservations.values():
            bucket[:] = [iv for iv in bucket if iv.vehicle_id != vehicle_id]
        for bucket in self._edge_reservations.values():
            bucket[:] = [iv for iv in bucket if iv.vehicle_id != vehicle_id]

    @staticmethod
    def _has_overlap(
        intervals: Sequence[_Interval],
        t_enter: float,
        t_exit: float,
        exclude_vehicle: int | None,
    ) -> bool:
        for interval in intervals:
            if interval.vehicle_id == exclude_vehicle:
                continue
            if interval.start >= t_exit:
                break
            if interval.end > t_enter:
                return True
        return False

    @staticmethod
    def _scan_earliest(
        intervals: Sequence[_Interval],
        not_before: float,
        duration: float,
        exclude_vehicle: int | None,
        clearance: float = 0.0,
    ) -> float:
        t = not_before
        for interval in intervals:
            if interval.vehicle_id == exclude_vehicle:
                continue
            eff_end = interval.end + clearance
            eff_start = interval.start - clearance
            if eff_end <= t:
                continue
            if eff_start >= t + duration:
                break
            t = eff_end
        return t


# ---------------------------------------------------------------------------
# Space-Time A*
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SearchNode:
    node_id: str
    t_arrive: float

    def __lt__(self, other: "_SearchNode") -> bool:
        return (self.t_arrive, self.node_id) < (other.t_arrive, other.node_id)


def _space_time_astar(
    *,
    start_node: str,
    start_time: float,
    goal_node: str,
    instance: Instance,
    reservations: ReservationTable,
    speed_in_per_sec: float,
    vehicle_id: int,
    horizon: float,
    config: MapfConfig,
    goal_hold_duration: float = 0.0,
) -> List[NodeVisit] | None:
    """Plan a single leg from start_node to goal_node starting at start_time.

    `goal_hold_duration` is the time the vehicle will dwell at the goal
    after arrival (service time). A* only accepts an arrival that leaves
    the goal node free for the full hold window — without this, a pickup
    dwell could overlap another vehicle's passage through the goal node.

    Returns a list of NodeVisits including the start node (with t_enter,
    t_exit reflecting any waiting before departure) and the goal node
    (with t_enter = arrival, t_exit = arrival; the caller appends any
    service dwell). Returns None if no feasible path within `horizon`.
    """
    if start_node == goal_node:
        if goal_hold_duration > 0.0 and not reservations.is_vertex_free(
            start_node, start_time, start_time + goal_hold_duration, exclude_vehicle=vehicle_id
        ):
            return None
        return [NodeVisit(start_node, start_time, start_time, is_service=False)]

    world = instance.world
    # heuristic: straight-line time-to-goal on the precomputed graph.
    distances_from_goal = world.distances.get(goal_node)
    if distances_from_goal is None:
        return None

    def heuristic(node: str) -> float:
        dist = distances_from_goal.get(node)
        if dist is None:
            return math.inf
        return dist / speed_in_per_sec

    horizon_t = start_time + horizon

    # priority queue: (f, g, counter, search_node)
    counter = 0
    open_heap: List[Tuple[float, float, int, _SearchNode]] = []
    initial = _SearchNode(start_node, start_time)
    h0 = heuristic(start_node)
    heapq.heappush(open_heap, (start_time + h0, start_time, counter, initial))
    # came_from maps state -> (prev_state, edge_depart_time, edge_arrive_time)
    came_from: Dict[_SearchNode, Tuple[_SearchNode, float, float]] = {}
    # closed: state -> best g (time arriving at node). Allow re-expansion if
    # better time discovered later (not expected with consistent heuristic).
    best_g: Dict[_SearchNode, float] = {initial: start_time}

    edges = world.edges
    depot_node = instance.depot_node

    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        if g > best_g.get(current, math.inf) + EPS:
            continue
        if current.node_id == goal_node:
            return _reconstruct(current, came_from, start_node, start_time)

        if g > horizon_t:
            return None

        for neighbor, edge_weight in edges.get(current.node_id, ()):
            edge_time = edge_weight / speed_in_per_sec if speed_in_per_sec > 0 else 0.0
            # For the goal node, require the arrival window to cover the
            # full service dwell, so a pickup can't overlap another
            # vehicle's pass-through.
            arrival_hold = goal_hold_duration if neighbor == goal_node else 0.0

            t_depart = _find_valid_departure(
                current_node=current.node_id,
                neighbor=neighbor,
                edge_time=edge_time,
                t_ready=g,
                horizon_t=horizon_t,
                reservations=reservations,
                vehicle_id=vehicle_id,
                depot_node=depot_node,
                arrival_hold=arrival_hold,
            )
            if t_depart is None or t_depart > horizon_t:
                continue
            t_arrive = t_depart + edge_time
            if t_arrive > horizon_t:
                continue

            successor = _SearchNode(neighbor, t_arrive)
            if t_arrive >= best_g.get(successor, math.inf) - EPS:
                continue
            best_g[successor] = t_arrive
            came_from[successor] = (current, t_depart, t_arrive)
            counter += 1
            f = t_arrive + heuristic(neighbor)
            heapq.heappush(open_heap, (f, t_arrive, counter, successor))

    return None


def _find_valid_departure(
    *,
    current_node: str,
    neighbor: str,
    edge_time: float,
    t_ready: float,
    horizon_t: float,
    reservations: ReservationTable,
    vehicle_id: int,
    depot_node: str,
    arrival_hold: float = 0.0,
) -> float | None:
    """Smallest t_depart >= t_ready such that:
    - Vehicle can wait at current_node during [t_ready, t_depart].
    - Edge (current_node -> neighbor) is free during [t_depart, t_depart + edge_time].
    - Neighbor is free for [t_depart + edge_time, t_depart + edge_time + max(EPS, arrival_hold)].

    `arrival_hold` covers the service-dwell window at a goal node. For
    pass-through visits use 0 (only the arrival instant must be clear).
    All checks honor the reservation table's clearance margin.
    """
    arrival_window = max(EPS, arrival_hold)
    t = t_ready
    for _ in range(128):
        if t > horizon_t:
            return None

        # 1. Wait at current_node during [t_ready, t].
        if not reservations.is_vertex_free(
            current_node, t_ready, t, exclude_vehicle=vehicle_id
        ):
            next_t = reservations.earliest_vertex_free(
                current_node,
                t_ready,
                max(EPS, t - t_ready),
                exclude_vehicle=vehicle_id,
            )
            if next_t <= t + EPS:
                return None
            t = next_t
            continue

        # 2. Edge (current -> neighbor) must be free during [t, t + edge_time].
        if not reservations.is_edge_free(
            current_node, neighbor, t, t + edge_time, exclude_vehicle=vehicle_id
        ):
            next_t = reservations.earliest_edge_free(
                current_node, neighbor, t, edge_time, exclude_vehicle=vehicle_id
            )
            if next_t <= t + EPS:
                return None
            t = next_t
            continue

        # 3. Arrival at neighbor plus service hold must not conflict.
        arrival = t + edge_time
        if not reservations.is_vertex_free(
            neighbor, arrival, arrival + arrival_window, exclude_vehicle=vehicle_id
        ):
            next_t = reservations.earliest_vertex_free(
                neighbor,
                arrival,
                arrival_window,
                exclude_vehicle=vehicle_id,
            )
            if next_t <= arrival + EPS:
                return None
            t = next_t - edge_time
            continue

        return t

    return None


def _reconstruct(
    goal: _SearchNode,
    came_from: Dict[_SearchNode, Tuple[_SearchNode, float, float]],
    start_node: str,
    start_time: float,
) -> List[NodeVisit]:
    """Walk back from goal building NodeVisits.

    For each interior node, t_enter = arrival_time, t_exit = departure_time
    (so the wait-before-move shows up as a non-zero dwell). The start node's
    t_enter = start_time, t_exit = departure_for_first_edge. The goal
    node's t_exit = t_enter (caller appends service dwell separately).
    """
    # First, build a sequence of (node_id, arrival_at_node, depart_from_node)
    chain: List[Tuple[str, float, float]] = []
    node = goal
    # Goal: arrival is node.t_arrive; depart is also t_arrive (caller extends).
    chain.append((node.node_id, node.t_arrive, node.t_arrive))
    while node in came_from:
        prev, depart_from_prev, arrive_at_node = came_from[node]
        # Update the depart time on the just-pushed entry (corresponds to current node).
        # That entry's depart = depart_from_prev? No: depart_from_prev is when we LEFT prev.
        # The "current" node's arrive is arrive_at_node (matches node.t_arrive).
        # The "current" node's depart-time will be set when we look at THIS node's outgoing
        # edge — but we're going backwards, so depart of current = depart_from_prev for the
        # OUTGOING edge of `prev`. That depart is the depart time of `prev`.
        chain.append((prev.node_id, prev.t_arrive, depart_from_prev))
        node = prev

    chain.reverse()

    # The start node's actual arrival is start_time; it might differ from
    # prev.t_arrive if the search expanded a delayed start (shouldn't because
    # initial state had t_arrive = start_time, but assert just in case).
    visits: List[NodeVisit] = []
    for idx, (node_id, t_enter, t_exit) in enumerate(chain):
        if idx == 0:
            t_enter = start_time
        visits.append(NodeVisit(node_id, t_enter, t_exit, is_service=False))
    return visits


# ---------------------------------------------------------------------------
# Prioritized planner
# ---------------------------------------------------------------------------


def plan_space_time(
    instance: Instance,
    solver: VRPRPDSolver,
    solution: EvaluatedSolution,
    mapf_config: MapfConfig,
    *,
    departure_stagger_sec: float = 0.2,
) -> ScheduledSolution:
    """Build a collision-free space-time schedule from a solver solution.

    The slowest-makespan vehicle plans first. Lower-priority vehicles
    wait or detour to avoid spatial conflicts. Their MAPF return times
    increase to absorb the delay.
    """
    if not mapf_config.enabled:
        return _build_passthrough(instance, solver, solution, departure_stagger_sec)

    if not solution.feasible:
        raise MapfInfeasible(-1, "solver produced an infeasible solution")

    primary_order = _priority_order(solution)
    horizon = max(
        1.0,
        solution.makespan * mapf_config.max_horizon_multiplier,
    )

    last_error: str | None = None
    for attempt, priority_order in enumerate(_priority_attempts(solution, mapf_config)):
        try:
            return _plan_with_order(
                instance=instance,
                solver=solver,
                solution=solution,
                mapf_config=mapf_config,
                priority_order=priority_order,
                horizon=horizon,
                departure_stagger_sec=departure_stagger_sec,
                attempt_label=f"attempt{attempt}",
            )
        except MapfInfeasible as exc:
            last_error = str(exc)
            continue

    raise MapfInfeasible(
        -1,
        f"all priority orderings exhausted: {last_error}",
    )


def _priority_attempts(
    solution: EvaluatedSolution, mapf_config: MapfConfig
) -> List[List[int]]:
    """Yield priority orderings to try, in order of preference."""
    orders = [_priority_order(solution)]
    if mapf_config.fallback_allow_reverse_priority:
        # Ascending route-length: short routes first, often clears bottlenecks
        # for longer routes by getting them out of the way early.
        by_length = sorted(
            solution.return_times.keys(),
            key=lambda v: (len(solution.routes[v]), v),
        )
        if by_length not in orders:
            orders.append(by_length)
        # Pure ascending makespan as a last resort (opposite of the default).
        by_makespan_asc = sorted(
            solution.return_times.keys(),
            key=lambda v: (solution.return_times[v], v),
        )
        if by_makespan_asc not in orders:
            orders.append(by_makespan_asc)
    return orders


def _plan_with_order(
    *,
    instance: Instance,
    solver: VRPRPDSolver,
    solution: EvaluatedSolution,
    mapf_config: MapfConfig,
    priority_order: List[int],
    horizon: float,
    departure_stagger_sec: float,
    attempt_label: str,
) -> ScheduledSolution:
    reservations = ReservationTable(
        instance.depot_node,
        mapf_config.clearance_sec,
        mouth_node=_depot_mouth_node(instance),
    )

    paths: Dict[int, List[NodeVisit]] = {vid: [] for vid in range(instance.vehicle_count)}
    return_times: Dict[int, float] = {vid: 0.0 for vid in range(instance.vehicle_count)}

    active_vehicle_ids = [vid for vid in range(instance.vehicle_count) if solution.routes[vid]]
    launch_index = {vid: idx for idx, vid in enumerate(active_vehicle_ids)}

    egress_walks, ingress_walks = _per_vehicle_corridor_times(
        instance, solver.speed_in_per_sec, priority_order, active_vehicle_ids
    )

    reason = "ok"
    swap_attempts = 0
    pending = list(priority_order)
    delay_attempts: Dict[int, int] = {vid: 0 for vid in pending}
    start_delay: Dict[int, float] = {vid: 0.0 for vid in pending}

    while pending:
        vehicle_id = pending.pop(0)
        if not solution.routes[vehicle_id]:
            paths[vehicle_id] = []
            return_times[vehicle_id] = 0.0
            continue

        base_start = launch_index.get(vehicle_id, 0) * departure_stagger_sec
        start_time = base_start + start_delay[vehicle_id]
        result = _plan_one_agent(
            vehicle_id=vehicle_id,
            solver=solver,
            instance=instance,
            solution=solution,
            reservations=reservations,
            mapf_config=mapf_config,
            start_time=start_time,
            horizon=horizon,
            egress_walk_sec=egress_walks.get(vehicle_id, 0.0),
            ingress_walk_sec=ingress_walks.get(vehicle_id, 0.0),
        )
        if result is not None:
            visits, return_time = result
            paths[vehicle_id] = visits
            return_times[vehicle_id] = return_time
            _commit_reservations(reservations, vehicle_id, visits, mapf_config, instance)
            continue

        if delay_attempts[vehicle_id] < mapf_config.fallback_max_delay_attempts:
            delay_attempts[vehicle_id] += 1
            growth = mapf_config.fallback_delay_growth ** (delay_attempts[vehicle_id] - 1)
            start_delay[vehicle_id] = (
                start_delay[vehicle_id]
                + mapf_config.fallback_delay_initial_sec * growth
            )
            reason = f"fallback:delay v{vehicle_id} ({attempt_label})"
            pending.insert(0, vehicle_id)
            continue

        if swap_attempts < mapf_config.fallback_max_swap_attempts:
            swap_attempts += 1
            blocker = _find_blocker(reservations, instance, solver, vehicle_id)
            if blocker is not None and blocker in paths and paths[blocker]:
                reservations.clear_vehicle(blocker)
                paths[blocker] = []
                return_times[blocker] = 0.0
                start_delay[vehicle_id] = 0.0
                delay_attempts[vehicle_id] = 0
                pending.insert(0, blocker)
                pending.insert(0, vehicle_id)
                reason = f"fallback:swap v{vehicle_id}<->v{blocker} ({attempt_label})"
                continue

        raise MapfInfeasible(
            vehicle_id,
            f"no space-time path within horizon={horizon:.1f}s "
            f"after {delay_attempts[vehicle_id]} delay retries "
            f"and {swap_attempts} swap retries ({attempt_label})",
        )

    makespan = max(return_times.values(), default=0.0)
    return ScheduledSolution(
        base=solution,
        paths=paths,
        return_times=return_times,
        makespan=makespan,
        priority_order=priority_order,
        reason=reason,
    )


def _priority_order(solution: "EvaluatedSolution") -> List[int]:
    return sorted(
        solution.return_times.keys(),
        key=lambda v: (-solution.return_times[v], v),
    )


def _plan_one_agent(
    *,
    vehicle_id: int,
    solver: VRPRPDSolver,
    instance: Instance,
    solution: EvaluatedSolution,
    reservations: ReservationTable,
    mapf_config: MapfConfig,
    start_time: float,
    horizon: float,
    egress_walk_sec: float = 0.0,
    ingress_walk_sec: float = 0.0,
) -> Tuple[List[NodeVisit], float] | None:
    """Plan one full agent trip: depot -> stops -> depot.

    `egress_walk_sec` / `ingress_walk_sec`: per-vehicle L-corridor walk
    times based on slot assignment. The MAPF reservation only blocks the
    depot mouth at the INSTANT of crossing (zero-duration visit). The
    physical walk before/after the crossing is implicit; render computes
    when to start walking using each vehicle's home_slot distance. This
    lets consecutive vehicles follow each other through the corridor at
    body-clearance distance rather than each blocking the corridor for
    the full traversal.

    On success returns (NodeVisit list, return_time including the
    post-arrival corridor walk). On failure returns None.
    """
    depot = instance.depot_node
    events = solution.events_by_vehicle.get(vehicle_id, [])
    speed = solver.speed_in_per_sec

    # The corridor is reserved for this vehicle's ACTUAL walk time (not a
    # fixed worst-case 7.5s). Slot 0 vehicle has zero walk; slot 9 has
    # 7.03s. Slide forward until a free window of length egress_walk_sec
    # is available at the depot mouth, so the in-corridor lock prevents
    # any other vehicle from walking past this one's still-parked slots.
    visits: List[NodeVisit] = []
    egress_window = max(egress_walk_sec, EPS)
    earliest_egress_start = max(0.0, start_time - egress_window)
    egress_start = reservations.earliest_vertex_free(
        depot, earliest_egress_start, egress_window, exclude_vehicle=vehicle_id,
    )
    adjusted_start_time = egress_start + egress_window
    if adjusted_start_time > start_time + EPS:
        start_time = adjusted_start_time
    visits.append(NodeVisit(depot, egress_start, start_time, is_service=False))

    current_node = depot
    current_time = start_time

    # Group consecutive events at the same node into a single stop. The
    # combined dwell must be passed to A* as `goal_hold_duration` so the
    # planner finds an arrival time at which the node stays free for the
    # entire dwell (otherwise a same-node drop+pickup pair would skip
    # the reservation check and overlap a pass-through visit).
    i = 0
    while i < len(events):
        target_node = events[i].node_id
        j = i
        combined_dwell = 0.0
        last_completion = events[i].arrival_time
        while j < len(events) and events[j].node_id == target_node:
            ev = events[j]
            ev_dwell = max(0.0, ev.completion_time - ev.arrival_time)
            if j > i:
                # Gap between consecutive same-node ops (e.g., processing
                # time between drop and pickup) counts toward the dwell.
                gap = max(0.0, ev.arrival_time - last_completion)
                combined_dwell += gap
            combined_dwell += ev_dwell
            last_completion = ev.completion_time
            j += 1

        if target_node != current_node:
            leg = _space_time_astar(
                start_node=current_node,
                start_time=current_time,
                goal_node=target_node,
                instance=instance,
                reservations=reservations,
                speed_in_per_sec=speed,
                vehicle_id=vehicle_id,
                horizon=horizon,
                config=mapf_config,
                goal_hold_duration=combined_dwell,
            )
            if leg is None:
                return None
            first = leg[0]
            if visits and visits[-1].node_id == first.node_id:
                prev = visits[-1]
                merged = NodeVisit(
                    node_id=prev.node_id,
                    t_enter=prev.t_enter,
                    t_exit=max(prev.t_exit, first.t_exit),
                    is_service=prev.is_service,
                    customer_id=prev.customer_id,
                    op_kind=prev.op_kind,
                )
                visits[-1] = merged
                visits.extend(leg[1:])
            else:
                visits.extend(leg)
            arrival_visit = visits[-1]
            current_time = arrival_visit.t_enter
            current_node = target_node

        # Apply each event's service visit in order. The first marks the
        # drop (instantaneous), the second the pickup (with the full
        # processing wait). Their combined extent is at most the
        # `combined_dwell` already cleared by A*.
        for k in range(i, j):
            ev = events[k]
            ev_dwell = max(0.0, ev.completion_time - ev.arrival_time)
            if k > i:
                gap = max(0.0, ev.arrival_time - events[k - 1].completion_time)
                current_time += gap
            service_t_enter = current_time
            service_t_exit = current_time + ev_dwell
            prev = visits[-1]
            merged = NodeVisit(
                node_id=prev.node_id,
                t_enter=prev.t_enter,
                t_exit=max(prev.t_exit, service_t_exit),
                is_service=True,
                customer_id=ev.operation.customer_id,
                op_kind=ev.operation.kind,
            )
            visits[-1] = merged
            current_time = service_t_exit

        i = j

    # Return leg to depot. The full corridor walk (`ingress_window`) still
    # counts toward this vehicle's return time, but the depot mouth is only
    # reserved for a body-clearance follow gap (`mouth_hold`) at the crossing
    # instant — not the whole walk. Parking is deepest-first by arrival
    # order, so returning vehicles file in nose-to-tail at clearance spacing
    # and a follower always stops short of the vehicle ahead, never passing a
    # parked Alvik. Holding the mouth for the entire walk (as before) forced
    # them to enter and park strictly one at a time.
    ingress_window = max(ingress_walk_sec, EPS)
    mouth_hold = max(EPS, mapf_config.clearance_sec)
    if current_node != depot:
        leg = _space_time_astar(
            start_node=current_node,
            start_time=current_time,
            goal_node=depot,
            instance=instance,
            reservations=reservations,
            speed_in_per_sec=speed,
            vehicle_id=vehicle_id,
            horizon=horizon,
            config=mapf_config,
            goal_hold_duration=mouth_hold,
        )
        if leg is None:
            return None
        first = leg[0]
        if visits and visits[-1].node_id == first.node_id:
            prev = visits[-1]
            merged = NodeVisit(
                node_id=prev.node_id,
                t_enter=prev.t_enter,
                t_exit=max(prev.t_exit, first.t_exit),
                is_service=prev.is_service,
                customer_id=prev.customer_id,
                op_kind=prev.op_kind,
            )
            visits[-1] = merged
            visits.extend(leg[1:])
        else:
            visits.extend(leg)
        current_time = visits[-1].t_enter

    final_visit = visits[-1]
    # return_time = fully parked (full corridor walk); the committed mouth
    # reservation only spans the clearance follow gap so the next returning
    # vehicle can cross right behind this one instead of waiting it out.
    ingress_exit = final_visit.t_enter + ingress_window
    visits[-1] = NodeVisit(
        node_id=final_visit.node_id,
        t_enter=final_visit.t_enter,
        t_exit=final_visit.t_enter + mouth_hold,
        is_service=False,
        customer_id=None,
        op_kind=None,
    )
    return visits, ingress_exit


def _corridor_distance_for_slot(
    slot: Tuple[float, ...], depot_coord: Tuple[float, ...]
) -> float:
    """L-corridor Manhattan walk distance from depot mouth to a parking slot.

    The depot is L-shaped and flat on the top Z-layer: a vertical arm from
    the mouth up to a corner, then a horizontal arm. Slots lie on one of the
    two arms, so the path distance equals |dx| + |dy| (+ |dz|, which is 0
    while the depot stays planar but kept general).
    """
    return sum(abs(s - d) for s, d in zip(slot, depot_coord))


def _per_vehicle_corridor_times(
    instance: Instance,
    speed: float,
    priority_order: List[int],
    active_vehicle_ids: List[int],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute each active vehicle's egress and ingress corridor walk times.

    Egress: vehicle priority_index = i is parked at ordered_slots[i].
    Ingress: dynamic deepest-first assignment matches priority order, so
    vehicle priority_index = i parks at ordered_slots[N-1-i].
    Returns (egress_walks, ingress_walks) keyed by vehicle_id, in seconds.
    """
    depot_coord = instance.world.coords[instance.depot_node]
    ordered_slots = sorted(
        instance.world.depot_slots,
        key=lambda s: (
            math.dist(s, depot_coord),
            -s[1],
            -s[0],
            -s[2],
        ),
    )
    active_set = set(active_vehicle_ids)
    priority_active = [vid for vid in priority_order if vid in active_set]
    n_active = len(priority_active)
    egress_walks: Dict[int, float] = {}
    ingress_walks: Dict[int, float] = {}
    for idx, vid in enumerate(priority_active):
        if idx < len(ordered_slots):
            egress_walks[vid] = (
                _corridor_distance_for_slot(ordered_slots[idx], depot_coord) / speed
            )
        else:
            egress_walks[vid] = 0.0
        ingress_idx = n_active - 1 - idx
        if 0 <= ingress_idx < len(ordered_slots):
            ingress_walks[vid] = (
                _corridor_distance_for_slot(ordered_slots[ingress_idx], depot_coord)
                / speed
            )
        else:
            ingress_walks[vid] = 0.0
    return egress_walks, ingress_walks


def _depot_mouth_node(instance: Instance) -> str | None:
    """Return the grid node coincident with the depot's coordinate.

    The depot is a virtual node that shares an (x, y, z) with the top-layer
    NE grid corner (typically `i_7_7_7`). The MAPF graph treats them as
    distinct IDs but physically they're the same spot. Callers use this to
    bind depot-vertex reservations onto the grid node so other vehicles
    can't pass through the corner while the corridor is in use.
    """
    depot = instance.depot_node
    depot_coord = instance.world.coords.get(depot)
    if depot_coord is None:
        return None
    for node, coord in instance.world.coords.items():
        if node == depot:
            continue
        if all(abs(c - d) < 1e-6 for c, d in zip(coord, depot_coord)):
            return node
    return None


def _commit_reservations(
    reservations: ReservationTable,
    vehicle_id: int,
    visits: List[NodeVisit],
    mapf_config: MapfConfig,
    instance: Instance,
) -> None:
    depot = instance.depot_node
    for visit in visits:
        # Skip vertex reservation for the depot itself except for the
        # corridor-lock windows (egress and ingress visits have non-zero
        # duration). Depot reservations go through the mouth bucket via
        # canonical mapping, blocking grid pass-throughs at the corner.
        if visit.node_id == depot:
            if visit.t_exit > visit.t_enter + EPS:
                reservations.reserve_vertex(
                    visit.node_id, visit.t_enter, visit.t_exit, vehicle_id
                )
        else:
            reservations.reserve_vertex(
                visit.node_id, visit.t_enter, visit.t_exit, vehicle_id
            )
    for prev, nxt in zip(visits, visits[1:]):
        if prev.node_id == nxt.node_id:
            continue
        reservations.reserve_edge(
            prev.node_id, nxt.node_id, prev.t_exit, nxt.t_enter, vehicle_id
        )


def _find_blocker(
    reservations: ReservationTable,
    instance: Instance,
    solver: VRPRPDSolver,
    vehicle_id: int,
) -> int | None:
    """Heuristic: pick the agent whose reservations sit on this vehicle's
    most-needed corridor (the depot mouth)."""
    depot = instance.depot_node
    grid_ne = None
    for neighbor, _ in instance.world.edges.get(depot, ()):
        grid_ne = neighbor
        break
    candidates = (
        reservations._node_reservations.get(grid_ne, ())  # noqa: SLF001
        if grid_ne
        else ()
    )
    for interval in candidates:
        if interval.vehicle_id != vehicle_id:
            return interval.vehicle_id
    return None


def _build_passthrough(
    instance: Instance,
    solver: VRPRPDSolver,
    solution: EvaluatedSolution,
    departure_stagger_sec: float,
) -> ScheduledSolution:
    """No-MAPF baseline: build NodeVisits straight from solver timings.

    Used when MapfConfig.enabled=False. Provides the same data shape so
    the simulator can run on a uniform path. Spatial conflicts are NOT
    resolved — relies on the (now-removed) reactive layer or accepts
    visible collisions.
    """
    depot = instance.depot_node
    speed = solver.speed_in_per_sec
    paths: Dict[int, List[NodeVisit]] = {}
    return_times: Dict[int, float] = {}
    active_vehicle_ids = [vid for vid in range(instance.vehicle_count) if solution.routes[vid]]
    launch_index = {vid: idx for idx, vid in enumerate(active_vehicle_ids)}

    for vehicle_id in range(instance.vehicle_count):
        events = solution.events_by_vehicle.get(vehicle_id, [])
        if not events:
            paths[vehicle_id] = []
            return_times[vehicle_id] = 0.0
            continue
        start_time = launch_index.get(vehicle_id, 0) * departure_stagger_sec
        visits: List[NodeVisit] = [NodeVisit(depot, start_time, start_time, False)]
        current_node = depot
        current_time = start_time
        for event in events:
            target_node = event.node_id
            if target_node != current_node:
                node_path = instance.world.shortest_paths.get(
                    (current_node, target_node), [current_node, target_node]
                )
                for idx in range(1, len(node_path)):
                    prev = node_path[idx - 1]
                    nxt = node_path[idx]
                    edge_dist = 0.0
                    for neighbor, weight in instance.world.edges.get(prev, ()):
                        if neighbor == nxt:
                            edge_dist = weight
                            break
                    edge_time = edge_dist / speed if speed > 0 else 0.0
                    current_time += edge_time
                    visits.append(NodeVisit(nxt, current_time, current_time, False))
                current_node = target_node
            dwell = max(0.0, event.completion_time - event.arrival_time)
            current_time += dwell
            prev = visits[-1]
            visits[-1] = NodeVisit(
                prev.node_id,
                prev.t_enter,
                current_time,
                True,
                event.operation.customer_id,
                event.operation.kind,
            )
        if current_node != depot:
            node_path = instance.world.shortest_paths.get(
                (current_node, depot), [current_node, depot]
            )
            for idx in range(1, len(node_path)):
                prev = node_path[idx - 1]
                nxt = node_path[idx]
                edge_dist = 0.0
                for neighbor, weight in instance.world.edges.get(prev, ()):
                    if neighbor == nxt:
                        edge_dist = weight
                        break
                edge_time = edge_dist / speed if speed > 0 else 0.0
                current_time += edge_time
                visits.append(NodeVisit(nxt, current_time, current_time, False))
        paths[vehicle_id] = visits
        return_times[vehicle_id] = current_time

    makespan = max(return_times.values(), default=0.0)
    return ScheduledSolution(
        base=solution,
        paths=paths,
        return_times=return_times,
        makespan=makespan,
        priority_order=list(range(instance.vehicle_count)),
        reason="passthrough",
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@dataclass
class _ScheduledInterval:
    vehicle_id: int
    t_enter: float
    t_exit: float


def validate(scheduled: ScheduledSolution, *, clearance_sec: float = 0.25) -> None:
    """Sanity-check the schedule has no vertex/edge conflicts.

    Raises ValueError with a descriptive message on the first violation.
    Depot vertex conflicts are tolerated (unlimited parking capacity).
    """
    depot = "depot"  # well-known instance.depot_node default; caller passes scheduled only
    # We can't access instance here easily — but the planner uses
    # instance.depot_node which equals "depot" by default. The frozen
    # schedule structure doesn't carry it. Accept the convention.
    vertex_intervals: Dict[str, List[_ScheduledInterval]] = {}
    edge_intervals: Dict[Tuple[str, str], List[_ScheduledInterval]] = {}

    for vid, visits in scheduled.paths.items():
        for visit in visits:
            if visit.node_id == depot and not (visit.t_exit > visit.t_enter + EPS):
                continue
            bucket = vertex_intervals.setdefault(visit.node_id, [])
            bucket.append(_ScheduledInterval(vid, visit.t_enter, visit.t_exit))
        for prev, nxt in zip(visits, visits[1:]):
            if prev.node_id == nxt.node_id:
                continue
            bucket = edge_intervals.setdefault(
                (prev.node_id, nxt.node_id), []
            )
            bucket.append(_ScheduledInterval(vid, prev.t_exit, nxt.t_enter))

    for node, intervals in vertex_intervals.items():
        if node == depot:
            continue
        intervals.sort(key=lambda iv: iv.t_enter)
        for i in range(1, len(intervals)):
            a = intervals[i - 1]
            b = intervals[i]
            if a.vehicle_id == b.vehicle_id:
                continue
            if a.t_exit + clearance_sec > b.t_enter:
                raise ValueError(
                    f"Vertex conflict at {node}: v{a.vehicle_id} "
                    f"[{a.t_enter:.2f},{a.t_exit:.2f}] overlaps "
                    f"v{b.vehicle_id} [{b.t_enter:.2f},{b.t_exit:.2f}]"
                )

    for (u, v), intervals in edge_intervals.items():
        reverse = edge_intervals.get((v, u), ())
        for a in intervals:
            for b in reverse:
                if a.vehicle_id == b.vehicle_id:
                    continue
                if a.t_enter < b.t_exit + clearance_sec and b.t_enter < a.t_exit + clearance_sec:
                    raise ValueError(
                        f"Swap conflict on edge ({u},{v}): v{a.vehicle_id} "
                        f"[{a.t_enter:.2f},{a.t_exit:.2f}] vs "
                        f"v{b.vehicle_id} reverse [{b.t_enter:.2f},{b.t_exit:.2f}]"
                    )
