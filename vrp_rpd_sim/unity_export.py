"""Export authoritative VRP-RPD simulation snapshots for Unity playback."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from . import config
from .render import PALETTE, SimulationApp
from .solution_cache import solve_with_solution_cache
from .solver import SolverRunResult, VRPRPDSolver
from .world import build_instance


SCHEMA_VERSION = 1


def build_unity_export(
    runtime_config: config.RuntimeConfig,
    *,
    capture_interval_sec: float = 0.1,
    debug_depot: bool = False,
) -> Dict[str, Any]:
    if capture_interval_sec <= 0:
        raise ValueError("capture_interval_sec must be positive.")

    instance = build_instance(instance_config=runtime_config.instance)
    solver = VRPRPDSolver(instance, solver_config=runtime_config.solver)
    result, _, _ = solve_with_solution_cache(solver, runtime_config.cache)

    app = SimulationApp(
        instance,
        solver,
        result,
        sim_speed=runtime_config.app.sim_speed_multiplier,
        job_count=runtime_config.instance.active_job_count,
        processing_scale=runtime_config.instance.processing_time_scale,
        fixed_processing_time=runtime_config.instance.fixed_processing_time_sec,
        fullscreen=False,
        debug_depot=debug_depot,
        cache_config=runtime_config.cache,
        headless=True,
    )

    initial_vehicle_specs = _build_vehicle_specs(app)
    frames = [_capture_frame(app)]
    next_capture_time = capture_interval_sec
    driver_step_sec = min(0.02, capture_interval_sec)
    max_duration_sec = max(result.best.makespan * 10.0, result.best.makespan + 900.0, 1200.0)
    simulation_completed = False
    termination_reason = ""

    while not all(vehicle.completed for vehicle in app.vehicles):
        if app.sim_time >= max_duration_sec:
            termination_reason = (
                "Export stopped at the safety duration limit before all vehicles completed."
            )
            break
        step = min(driver_step_sec, max_duration_sec - app.sim_time)
        app._step_simulation(step)
        if app.sim_time + 1e-9 >= next_capture_time:
            frames.append(_capture_frame(app))
            next_capture_time += capture_interval_sec
    else:
        simulation_completed = True

    if not math.isclose(frames[-1]["timeSec"], app.sim_time, rel_tol=0.0, abs_tol=1e-9):
        frames.append(_capture_frame(app))

    completion_time_sec = (
        max((vehicle.completion_time or 0.0) for vehicle in app.vehicles)
        if simulation_completed
        else -1.0
    )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "coordinateMapping": {
            "pygameXToUnity": "x",
            "pygameYToUnity": "z",
            "unityHeightAxis": "y",
            "sourceUnits": "inches",
            "recommendedUnityScale": 0.0254,
        },
        "worldWidthIn": config.WORLD_WIDTH_IN,
        "worldHeightIn": config.WORLD_HEIGHT_IN,
        "roadWidthIn": config.ROAD_WIDTH_IN,
        "laneCenterOffsetIn": config.LANE_CENTER_OFFSET_IN,
        "roadXs": [float(value) for value in instance.world.road_xs],
        "roadYs": [float(value) for value in instance.world.road_ys],
        "activeJobIds": [int(customer_id) for customer_id in instance.active_job_ids],
        "vehicleCapacity": instance.capacity,
        "vehicleCount": instance.vehicle_count,
        "alvikSizeIn": config.ALVIK_SIZE_IN,
        "alvikSpeedInPerSec": instance.instance_config.alvik_speed_in_per_sec,
        "plannedMakespanSec": result.best.makespan,
        "actualCompletionTimeSec": completion_time_sec,
        "captureIntervalSec": capture_interval_sec,
        "depot": _build_depot_data(instance),
        "stations": _build_station_specs(instance),
        "vehicles": initial_vehicle_specs,
        "frames": frames,
        "summary": _build_summary(
            result,
            frames,
            completion_time_sec,
            simulation_completed=simulation_completed,
            terminated_at_sec=app.sim_time,
            termination_reason=termination_reason,
        ),
    }


def export_unity_json(
    output_path: str | Path,
    runtime_config: config.RuntimeConfig,
    *,
    capture_interval_sec: float = 0.1,
    debug_depot: bool = False,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_payload = build_unity_export(
        runtime_config,
        capture_interval_sec=capture_interval_sec,
        debug_depot=debug_depot,
    )
    destination.write_text(
        json.dumps(export_payload, indent=2),
        encoding="utf-8",
    )
    return destination


def capture_frame(app: SimulationApp) -> Dict[str, Any]:
    """Public entry to capture a single frame from a running SimulationApp."""
    return _capture_frame(app)


def build_payload_from_recording(
    app: SimulationApp,
    frames: List[Dict[str, Any]],
    *,
    capture_interval_sec: float,
    simulation_completed: bool,
    termination_reason: str = "",
) -> Dict[str, Any]:
    """Build the JSON payload from an already-running app + captured frames.

    Use this when you've recorded frames live (e.g., from pygame's run loop)
    rather than running the headless export driver.
    """
    instance = app.instance
    result = app.result
    completion_time_sec = (
        max((vehicle.completion_time or 0.0) for vehicle in app.vehicles)
        if simulation_completed
        else -1.0
    )
    return {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "coordinateMapping": {
            "pygameXToUnity": "x",
            "pygameYToUnity": "z",
            "unityHeightAxis": "y",
            "sourceUnits": "inches",
            "recommendedUnityScale": 0.0254,
        },
        "worldWidthIn": config.WORLD_WIDTH_IN,
        "worldHeightIn": config.WORLD_HEIGHT_IN,
        "roadWidthIn": config.ROAD_WIDTH_IN,
        "laneCenterOffsetIn": config.LANE_CENTER_OFFSET_IN,
        "roadXs": [float(value) for value in instance.world.road_xs],
        "roadYs": [float(value) for value in instance.world.road_ys],
        "activeJobIds": [int(customer_id) for customer_id in instance.active_job_ids],
        "vehicleCapacity": instance.capacity,
        "vehicleCount": instance.vehicle_count,
        "alvikSizeIn": config.ALVIK_SIZE_IN,
        "alvikSpeedInPerSec": instance.instance_config.alvik_speed_in_per_sec,
        "plannedMakespanSec": result.best.makespan,
        "actualCompletionTimeSec": completion_time_sec,
        "captureIntervalSec": capture_interval_sec,
        "depot": _build_depot_data(instance),
        "stations": _build_station_specs(instance),
        "vehicles": _build_vehicle_specs(app),
        "frames": frames,
        "summary": _build_summary(
            result,
            frames,
            completion_time_sec,
            simulation_completed=simulation_completed,
            terminated_at_sec=app.sim_time,
            termination_reason=termination_reason,
        ),
    }


def write_payload(output_path: str | Path, payload: Dict[str, Any]) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def _build_summary(
    result: SolverRunResult,
    frames: List[Dict[str, Any]],
    completion_time_sec: float,
    *,
    simulation_completed: bool,
    terminated_at_sec: float,
    termination_reason: str,
) -> Dict[str, Any]:
    return {
        "selectedLabel": result.selected_label,
        "bestLabel": result.best_label,
        "frameCount": len(frames),
        "simulationCompleted": simulation_completed,
        "completionTimeSec": completion_time_sec,
        "terminatedAtSec": terminated_at_sec,
        "terminationReason": termination_reason,
        "solverMakespansSec": {
            "initial": result.initial.makespan,
            "alns": result.alns.makespan,
            "brkga": result.brkga.makespan,
            "best": result.best.makespan,
        },
    }


def _build_depot_data(instance) -> Dict[str, Any]:
    depot_access = instance.world.coords[instance.depot_node]
    return {
        "nodeId": instance.depot_node,
        "accessPoint": _point_dict(depot_access),
        "anchorPoint": _point_dict(instance.world.depot_anchor),
        "slots": [
            {
                "slotIndex": slot_index,
                **_point_dict(slot),
            }
            for slot_index, slot in enumerate(instance.world.depot_slots)
        ],
        "entries": [
            {
                "entryId": f"{group}_{index}",
                "group": group,
                **_point_dict(point),
            }
            for group, points in instance.world.depot_entries.items()
            for index, point in enumerate(points)
        ],
    }


def _build_station_specs(instance) -> List[Dict[str, Any]]:
    return [
        {
            "stationId": station.station_id,
            "name": station.name,
            "nodeId": station.node_id,
            "x": station.coord[0],
            "y": station.coord[1],
            "processingTimeSec": instance.processing_times[station.station_id],
            "active": station.station_id in instance.active_job_ids,
        }
        for station in instance.stations
    ]


def _build_vehicle_specs(app: SimulationApp) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for vehicle in app.vehicles:
        color = vehicle.color or PALETTE[vehicle.vehicle_id % len(PALETTE)]
        specs.append(
            {
                "vehicleId": vehicle.vehicle_id,
                "displayName": f"Alvik {vehicle.vehicle_id + 1}",
                "colorRgb": list(color),
                "startX": vehicle.position[0],
                "startY": vehicle.position[1],
                "startLoad": vehicle.load,
                "route": [
                    {
                        "customerId": operation.customer_id,
                        "kind": operation.kind,
                    }
                    for operation in vehicle.route
                ],
            }
        )
    return specs


def _capture_frame(app: SimulationApp) -> Dict[str, Any]:
    station_snapshots: List[Dict[str, Any]] = []
    for station in app.instance.stations:
        state_name, state_color = app._station_state(station.station_id)
        progress = app.station_progress[station.station_id]
        station_snapshots.append(
            {
                "stationId": station.station_id,
                "state": state_name,
                "colorRgb": list(state_color),
                "droppedAtSec": _nullable_time(progress["dropped_at"]),
                "readyAtSec": _nullable_time(progress["ready_at"]),
                "pickedAtSec": _nullable_time(progress["picked_at"]),
            }
        )

    vehicle_snapshots: List[Dict[str, Any]] = []
    for vehicle in app.vehicles:
        vehicle_snapshots.append(
            {
                "vehicleId": vehicle.vehicle_id,
                "x": vehicle.position[0],
                "y": vehicle.position[1],
                "load": vehicle.load,
                "routeIndex": vehicle.route_index,
                "currentNode": vehicle.current_node,
                "targetNode": vehicle.target_node or "",
                "waitingCustomerId": vehicle.waiting_customer_id if vehicle.waiting_customer_id is not None else -1,
                "completed": vehicle.completed,
                "completionTimeSec": _nullable_time(vehicle.completion_time),
                "homeX": vehicle.home_slot[0],
                "homeY": vehicle.home_slot[1],
                "hasActivePath": vehicle.active_path is not None,
            }
        )

    return {
        "timeSec": app.sim_time,
        "completedJobs": app._completed_jobs(),
        "activeJobs": app._active_jobs(),
        "vehiclesCompleted": sum(1 for vehicle in app.vehicles if vehicle.completed),
        "vehicles": vehicle_snapshots,
        "stations": station_snapshots,
    }


def _point_dict(point) -> Dict[str, float]:
    return {
        "x": float(point[0]),
        "y": float(point[1]),
    }


def _nullable_time(value: float | None) -> float:
    if value is None:
        return -1.0
    return float(value)
