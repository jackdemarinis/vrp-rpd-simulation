"""World geometry, graph generation, and processing-time setup.

The world matches `Physical Set Up/Configuration 1.obj`: an 8×8 grid of
intersections with single-lane two-way roads, plus an L-shaped depot that
hooks into the NE grid corner. The 49 white squares (the 7×7 cream cells
between the corridors) are the stations: each one has a center node where
the robot is serviced and a north-entry node on the corridor above it, so a
station is reached only by dipping south off the top corridor (a dead-end
pocket). Depot parking has 10 slots (the NE corner is the first parking
slot).
"""

from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

from . import cad_layout, config
from .model import Coord, Instance, Station, WorldGraph


def build_instance(
    job_count: int | None = None,
    processing_scale: float | None = None,
    fixed_processing_time: float | None = None,
    instance_config: config.InstanceConfig | None = None,
) -> Instance:
    instance_config = instance_config or config.default_runtime_config().instance
    resolved_job_count = (
        instance_config.active_job_count if job_count is None else job_count
    )
    resolved_processing_scale = (
        instance_config.processing_time_scale
        if processing_scale is None
        else processing_scale
    )
    resolved_fixed_processing_time = (
        instance_config.fixed_processing_time_sec
        if fixed_processing_time is None
        else fixed_processing_time
    )
    resolved_instance_config = replace(
        instance_config,
        active_job_count=resolved_job_count,
        processing_time_scale=resolved_processing_scale,
        fixed_processing_time_sec=resolved_fixed_processing_time,
    )
    world, stations = build_world()
    processing_times = build_processing_times(
        world,
        stations,
        processing_scale=resolved_processing_scale,
        fixed_processing_time=resolved_fixed_processing_time,
        instance_config=resolved_instance_config,
    )
    active_job_ids = select_active_job_ids(stations, resolved_job_count)
    return Instance(
        world=world,
        stations=stations,
        active_job_ids=active_job_ids,
        vehicle_count=resolved_instance_config.vehicle_count,
        capacity=resolved_instance_config.vehicle_capacity,
        processing_times=processing_times,
        instance_config=resolved_instance_config,
    )


def build_world() -> Tuple[WorldGraph, List[Station]]:
    road_xs = list(cad_layout.GRID_XS)
    road_ys = list(cad_layout.GRID_YS)

    coords: Dict[str, Coord] = {}
    edges: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    # 64 grid intersections, indexed (ix, iy) with ix=col, iy=row, SW origin.
    for ix, x in enumerate(road_xs):
        for iy, y in enumerate(road_ys):
            coords[_grid_node_id(ix, iy)] = (x, y)

    # Vertical corridor edges between adjacent intersections (unchanged).
    for ix in range(len(road_xs)):
        for iy in range(len(road_ys) - 1):
            _add_edge(
                edges,
                _grid_node_id(ix, iy),
                _grid_node_id(ix, iy + 1),
                abs(road_ys[iy + 1] - road_ys[iy]),
            )

    # Horizontal corridor edges. Every horizontal segment on rows 1..ROWS-1
    # is the *north edge* of the white square directly below it, so we split
    # it through a north-entry node `n_{cix}_{ciy}` and hang the square's
    # center node `c_{cix}_{ciy}` off that entry by a southward dip. The
    # center has degree 1 (only its entry), making the square a dead-end
    # pocket reachable only from the north. Row-0 segments enclose nothing
    # below them, so they stay plain edges.
    for iy in range(len(road_ys)):
        for ix in range(len(road_xs) - 1):
            left = _grid_node_id(ix, iy)
            right = _grid_node_id(ix + 1, iy)
            segment = abs(road_xs[ix + 1] - road_xs[ix])
            if iy >= 1:
                cix, ciy = ix, iy - 1
                entry_node = _entry_node_id(cix, ciy)
                center_node = _center_node_id(cix, ciy)
                coords[entry_node] = cad_layout.white_square_north_entry(cix, ciy)
                coords[center_node] = cad_layout.white_square_center(cix, ciy)
                _add_edge(edges, left, entry_node, segment / 2.0)
                _add_edge(edges, entry_node, right, segment / 2.0)
                _add_edge(
                    edges,
                    entry_node,
                    center_node,
                    abs(coords[entry_node][1] - coords[center_node][1]),
                )
            else:
                _add_edge(edges, left, right, segment)

    # 49 stations, one per white square, at the square centers. station_id
    # 1..49 row-major from SW so id 1 = square (col 0, row 0) and id 49 =
    # square (col 6, row 6).
    stations: List[Station] = []
    station_id = 1
    for ciy in range(cad_layout.WHITE_SQUARE_ROWS):
        for cix in range(cad_layout.WHITE_SQUARE_COLS):
            node_id = _center_node_id(cix, ciy)
            stations.append(
                Station(
                    station_id=station_id,
                    name=str(station_id),
                    side="square",
                    node_id=node_id,
                    coord=coords[node_id],
                )
            )
            station_id += 1

    # Depot abstract node sits coincident with the NE grid corner. Outgoing
    # vehicles transition depot → i_7_7 with zero travel cost; inbound
    # vehicles do the reverse. The depot's physical L-shape (9 extra dots)
    # is handled outside the routing graph by the render-layer corridor
    # logic, similar to how the prior 3×3 depot was handled.
    depot_node = "depot"
    grid_ne_corner = _grid_node_id(len(road_xs) - 1, len(road_ys) - 1)
    coords[depot_node] = coords[grid_ne_corner]
    _add_edge(edges, depot_node, grid_ne_corner, 0.0)

    distances, shortest_paths = _all_pairs_shortest_paths(coords, edges)

    depot_slots = list(cad_layout.ALL_DEPOT_SLOTS)
    depot_entries = {"main": [coords[depot_node]]}

    world = WorldGraph(
        coords=coords,
        edges=dict(edges),
        distances=distances,
        shortest_paths=shortest_paths,
        road_xs=road_xs,
        road_ys=road_ys,
        depot_anchor=config.DEPOT_ANCHOR_IN,
        depot_slots=depot_slots,
        depot_entries=depot_entries,
        road_gap_in=cad_layout.GRID_PITCH_IN,
        lane_center_offset_in=config.LANE_CENTER_OFFSET_IN,
    )
    return world, stations


def build_processing_times(
    world: WorldGraph,
    stations: Iterable[Station],
    processing_scale: float | None = None,
    fixed_processing_time: float | None = None,
    instance_config: config.InstanceConfig | None = None,
) -> Dict[int, float]:
    instance_config = instance_config or config.default_runtime_config().instance
    station_list = list(stations)
    travel_values = []
    for station_a in station_list:
        for station_b in station_list:
            if station_a.station_id >= station_b.station_id:
                continue
            distance = world.distances[station_a.node_id][station_b.node_id]
            travel_values.append(distance / instance_config.alvik_speed_in_per_sec)
    if not travel_values:
        d_min = d_max = 0.0
    else:
        d_min = min(travel_values)
        d_max = max(travel_values)

    rng = random.Random(instance_config.processing_time_seed)
    base_times = {
        station.station_id: rng.uniform(d_min, d_max) for station in station_list
    }

    override_fixed = (
        instance_config.fixed_processing_time_sec
        if fixed_processing_time is None
        else fixed_processing_time
    )
    if override_fixed is not None:
        processing_times = {
            station.station_id: float(override_fixed) for station in station_list
        }
    else:
        variant = instance_config.processing_time_variant.lower()
        if variant == "base":
            processing_times = dict(base_times)
        elif variant == "2x":
            processing_times = {station_id: value * 2.0 for station_id, value in base_times.items()}
        elif variant == "5x":
            processing_times = {station_id: value * 5.0 for station_id, value in base_times.items()}
        elif variant == "1r10":
            processing_times = {
                station_id: value * rng.randint(1, 10) for station_id, value in base_times.items()
            }
        elif variant == "1r20":
            processing_times = {
                station_id: value * rng.randint(1, 20) for station_id, value in base_times.items()
            }
        else:
            raise ValueError(
                f"Unsupported processing-time variant: {instance_config.processing_time_variant}"
            )

    scale = instance_config.processing_time_scale if processing_scale is None else processing_scale
    processing_times = {
        station_id: value * scale for station_id, value in processing_times.items()
    }

    processing_times.update(instance_config.processing_time_overrides)
    return processing_times


def select_active_job_ids(stations: Iterable[Station], count: int) -> List[int]:
    station_list = list(stations)
    if count < 1:
        raise ValueError("ACTIVE_JOB_COUNT must be at least 1.")
    if count > len(station_list):
        raise ValueError("ACTIVE_JOB_COUNT cannot exceed the number of stations.")

    ordered_ids = [station.station_id for station in station_list]
    if count == len(ordered_ids):
        return ordered_ids

    selected: List[int] = []
    used_indexes = set()
    total = len(ordered_ids)
    for pick in range(count):
        index = min(total - 1, math.floor(((pick + 0.5) * total) / count))
        while index in used_indexes and index + 1 < total:
            index += 1
        while index in used_indexes and index > 0:
            index -= 1
        used_indexes.add(index)
        selected.append(ordered_ids[index])
    return selected


def _grid_node_id(ix: int, iy: int) -> str:
    return f"i_{ix}_{iy}"


def _center_node_id(cix: int, ciy: int) -> str:
    """Service node at the center of white square (cix, ciy)."""
    return f"c_{cix}_{ciy}"


def _entry_node_id(cix: int, ciy: int) -> str:
    """North-edge entry node for white square (cix, ciy)."""
    return f"n_{cix}_{ciy}"


def _add_edge(
    edges: Dict[str, List[Tuple[str, float]]],
    source: str,
    target: str,
    weight: float,
) -> None:
    edges[source].append((target, weight))
    edges[target].append((source, weight))


def _all_pairs_shortest_paths(
    coords: Dict[str, Coord],
    edges: Dict[str, List[Tuple[str, float]]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[Tuple[str, str], List[str]]]:
    distances: Dict[str, Dict[str, float]] = {}
    shortest_paths: Dict[Tuple[str, str], List[str]] = {}
    for source in coords:
        source_distances, parents = _dijkstra(source, edges)
        distances[source] = source_distances
        for target in coords:
            shortest_paths[(source, target)] = _reconstruct_path(source, target, parents)
    return distances, shortest_paths


def _dijkstra(
    source: str,
    edges: Dict[str, List[Tuple[str, float]]],
) -> Tuple[Dict[str, float], Dict[str, str | None]]:
    distances = {source: 0.0}
    parents: Dict[str, str | None] = {source: None}
    heap: List[Tuple[float, str]] = [(0.0, source)]
    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > distances[node]:
            continue
        for neighbor, weight in edges.get(node, []):
            next_dist = current_dist + weight
            if next_dist < distances.get(neighbor, math.inf):
                distances[neighbor] = next_dist
                parents[neighbor] = node
                heapq.heappush(heap, (next_dist, neighbor))
    return distances, parents


def _reconstruct_path(
    source: str,
    target: str,
    parents: Dict[str, str | None],
) -> List[str]:
    if target not in parents:
        return [source]
    path = [target]
    node = target
    while node != source:
        parent = parents.get(node)
        if parent is None:
            break
        path.append(parent)
        node = parent
    path.reverse()
    return path


def euclidean(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
