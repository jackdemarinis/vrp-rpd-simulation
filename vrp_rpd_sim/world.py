"""World geometry, graph generation, and processing-time setup."""

from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from . import config
from .model import Coord, Instance, Station, WorldGraph


def build_instance(
    job_count: int | None = None,
    processing_scale: float | None = None,
    fixed_processing_time: float | None = None,
) -> Instance:
    world, stations = build_world()
    processing_times = build_processing_times(
        world,
        stations,
        processing_scale=processing_scale,
        fixed_processing_time=fixed_processing_time,
    )
    active_job_ids = select_active_job_ids(stations, job_count or config.ACTIVE_JOB_COUNT)
    return Instance(
        world=world,
        stations=stations,
        active_job_ids=active_job_ids,
        vehicle_count=config.ALVIK_COUNT,
        capacity=config.VEHICLE_CAPACITY,
        processing_times=processing_times,
    )


def build_world() -> Tuple[WorldGraph, List[Station]]:
    road_gap = _compute_road_gap(
        config.WORLD_SIZE_IN,
        config.ROAD_EDGE_MARGIN_IN,
        config.ROAD_ENVELOPE_WIDTH_IN,
        config.VERTICAL_ROAD_COUNT,
    )
    road_xs = _road_centers(config.VERTICAL_ROAD_COUNT, road_gap)
    road_ys = _road_centers(config.HORIZONTAL_ROAD_COUNT, road_gap)

    coords: Dict[str, Coord] = {}
    edges: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    bottom_nodes = []
    top_nodes = []
    left_nodes = []
    right_nodes = []

    for ix, x in enumerate(road_xs):
        bottom_id = f"vbot_{ix}"
        top_id = f"vtop_{ix}"
        coords[bottom_id] = (x, 0.0)
        coords[top_id] = (x, config.WORLD_SIZE_IN)
        bottom_nodes.append(bottom_id)
        top_nodes.append(top_id)

    for iy, y in enumerate(road_ys):
        left_id = f"hleft_{iy}"
        right_id = f"hright_{iy}"
        coords[left_id] = (0.0, y)
        coords[right_id] = (config.WORLD_SIZE_IN, y)
        left_nodes.append(left_id)
        right_nodes.append(right_id)

    for ix, x in enumerate(road_xs):
        column_nodes = [bottom_nodes[ix]]
        for iy, y in enumerate(road_ys):
            node_id = f"i_{ix}_{iy}"
            coords[node_id] = (x, y)
            column_nodes.append(node_id)
        column_nodes.append(top_nodes[ix])
        _connect_consecutive(coords, edges, column_nodes, axis="y")

    for iy, y in enumerate(road_ys):
        row_nodes = [left_nodes[iy]]
        for ix, x in enumerate(road_xs):
            row_nodes.append(f"i_{ix}_{iy}")
        row_nodes.append(right_nodes[iy])
        _connect_consecutive(coords, edges, row_nodes, axis="x")

    stations: List[Station] = []
    station_counter = 1
    for side in ("bottom", "right", "top", "left"):
        for road_index in config.STATION_ROAD_INDEX_LAYOUT[side]:
            station_id = station_counter
            station_name = f"S{station_id:02d}"
            node_id = f"station_{station_id:02d}"
            if side == "bottom":
                anchor = bottom_nodes[road_index]
            elif side == "top":
                anchor = top_nodes[road_index]
            elif side == "left":
                anchor = left_nodes[road_index]
            else:
                anchor = right_nodes[road_index]
            coords[node_id] = coords[anchor]
            _add_edge(edges, node_id, anchor, 0.0)
            stations.append(
                Station(
                    station_id=station_id,
                    name=station_name,
                    side=side,
                    node_id=node_id,
                    coord=coords[node_id],
                )
            )
            station_counter += 1

    depot_node = "depot"
    depot_access = _depot_access_coord(road_xs, road_ys)
    coords[depot_node] = depot_access
    _add_edge(
        edges,
        depot_node,
        bottom_nodes[0],
        abs(coords[depot_node][1] - 0.0),
    )
    _add_edge(
        edges,
        depot_node,
        left_nodes[0],
        abs(coords[depot_node][0] - 0.0),
    )

    distances, shortest_paths = _all_pairs_shortest_paths(coords, edges)
    depot_slots = _build_depot_slots(config.DEPOT_ANCHOR_IN)
    world = WorldGraph(
        coords=coords,
        edges=dict(edges),
        distances=distances,
        shortest_paths=shortest_paths,
        road_xs=road_xs,
        road_ys=road_ys,
        depot_anchor=config.DEPOT_ANCHOR_IN,
        depot_slots=depot_slots,
        depot_entries=_depot_entry_points(road_xs, road_ys, depot_slots),
        road_gap_in=road_gap,
        lane_center_offset_in=config.LANE_CENTER_OFFSET_IN,
    )
    return world, stations


def build_processing_times(
    world: WorldGraph,
    stations: Iterable[Station],
    processing_scale: float | None = None,
    fixed_processing_time: float | None = None,
) -> Dict[int, float]:
    station_list = list(stations)
    travel_values = []
    for station_a in station_list:
        for station_b in station_list:
            if station_a.station_id >= station_b.station_id:
                continue
            distance = world.distances[station_a.node_id][station_b.node_id]
            travel_values.append(distance / config.ALVIK_SPEED_IN_PER_SEC)
    d_min = min(travel_values)
    d_max = max(travel_values)

    rng = random.Random(config.PROCESSING_TIME_SEED)
    base_times = {
        station.station_id: rng.uniform(d_min, d_max) for station in station_list
    }

    override_fixed = config.FIXED_PROCESSING_TIME_SEC if fixed_processing_time is None else fixed_processing_time
    if override_fixed is not None:
        processing_times = {
            station.station_id: float(override_fixed) for station in station_list
        }
    else:
        variant = config.PROCESSING_TIME_VARIANT.lower()
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
            raise ValueError(f"Unsupported processing-time variant: {config.PROCESSING_TIME_VARIANT}")

    scale = config.PROCESSING_TIME_SCALE if processing_scale is None else processing_scale
    processing_times = {
        station_id: value * scale for station_id, value in processing_times.items()
    }

    processing_times.update(config.PROCESSING_TIME_OVERRIDES)
    return processing_times


def select_active_job_ids(stations: Iterable[Station], count: int) -> List[int]:
    station_list = list(stations)
    if count < 1:
        raise ValueError("ACTIVE_JOB_COUNT must be at least 1.")
    if count > len(station_list):
        raise ValueError("ACTIVE_JOB_COUNT cannot exceed the number of perimeter stations.")

    side_queues = {
        side: [station.station_id for station in station_list if station.side == side]
        for side in ("bottom", "right", "top", "left")
    }
    selected: List[int] = []
    while len(selected) < count:
        for side in ("bottom", "right", "top", "left"):
            if len(selected) >= count:
                break
            if side_queues[side]:
                selected.append(side_queues[side].pop(0))
    return selected


def _compute_road_gap(
    world_size: float,
    edge_margin: float,
    road_width: float,
    road_count: int,
) -> float:
    usable = world_size - (2.0 * edge_margin) - (road_count * road_width)
    if usable <= 0:
        raise ValueError("World is too small for the requested road layout.")
    return usable / (road_count - 1)


def _road_centers(count: int, road_gap: float) -> List[float]:
    centers = []
    center = config.ROAD_EDGE_MARGIN_IN + (config.ROAD_ENVELOPE_WIDTH_IN / 2.0)
    pitch = config.ROAD_ENVELOPE_WIDTH_IN + road_gap
    for index in range(count):
        centers.append(center + (index * pitch))
    return centers


def _connect_consecutive(
    coords: Dict[str, Coord],
    edges: Dict[str, List[Tuple[str, float]]],
    node_ids: List[str],
    axis: str,
) -> None:
    for left, right in zip(node_ids, node_ids[1:]):
        if axis == "x":
            weight = abs(coords[right][0] - coords[left][0])
        else:
            weight = abs(coords[right][1] - coords[left][1])
        _add_edge(edges, left, right, weight)


def _add_edge(
    edges: Dict[str, List[Tuple[str, float]]],
    source: str,
    target: str,
    weight: float,
) -> None:
    edges[source].append((target, weight))
    edges[target].append((source, weight))


def _build_depot_slots(anchor: Coord) -> List[Coord]:
    slots = []
    step = config.ALVIK_SIZE_IN + config.DEPOT_STACK_GAP_IN
    x_offset = ((config.DEPOT_STACK_COLUMNS - 1) * step) / 2.0
    y_offset = ((config.DEPOT_STACK_ROWS - 1) * step) / 2.0
    for row in range(config.DEPOT_STACK_ROWS):
        for col in range(config.DEPOT_STACK_COLUMNS):
            x = anchor[0] - x_offset + (col * step)
            y = anchor[1] - y_offset + (row * step)
            slots.append((x, y))
    return slots[: config.ALVIK_COUNT]


def _depot_access_coord(road_xs: List[float], road_ys: List[float]) -> Coord:
    return (
        road_xs[0] - config.LANE_CENTER_OFFSET_IN,
        road_ys[0] + config.LANE_CENTER_OFFSET_IN,
    )


def _depot_entry_points(
    road_xs: List[float],
    road_ys: List[float],
    depot_slots: List[Coord],
) -> Dict[str, List[Coord]]:
    col_xs = sorted({round(x, 6) for x, _ in depot_slots})
    row_ys = sorted({round(y, 6) for _, y in depot_slots})
    top_y = road_ys[0] - config.DEPOT_ENTRY_LANE_OFFSET_IN
    right_x = road_xs[0] - config.DEPOT_ENTRY_LANE_OFFSET_IN
    return {
        "top": [(x, top_y) for x in col_xs[: config.DEPOT_ENTRY_TOP_COUNT]],
        "right": [(right_x, y) for y in row_ys[: config.DEPOT_ENTRY_RIGHT_COUNT]],
    }


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
