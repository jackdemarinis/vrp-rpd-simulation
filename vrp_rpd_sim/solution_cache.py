"""Disk-backed solution caching keyed by the effective solve configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .config import CacheConfig, SolverConfig
from .model import Instance, Operation, Routes
from .solver import SolverRunResult, VRPRPDSolver


def solve_with_solution_cache(
    solver: VRPRPDSolver,
    cache_config: CacheConfig,
) -> tuple[SolverRunResult, bool, Path | None]:
    if not cache_config.enabled:
        return solver.solve(), False, None

    cache_path = solution_cache_path(
        instance=solver.instance,
        solver_config=solver.solver_config,
        cache_config=cache_config,
    )
    cached = load_cached_solution(
        cache_path=cache_path,
        solver=solver,
        cache_config=cache_config,
    )
    if cached is not None:
        return cached, True, cache_path

    result = solver.solve()
    save_cached_solution(
        cache_path=cache_path,
        solver=solver,
        result=result,
        cache_config=cache_config,
    )
    return result, False, cache_path


def solution_cache_path(
    *,
    instance: Instance,
    solver_config: SolverConfig,
    cache_config: CacheConfig,
) -> Path:
    key = _solution_cache_key(
        instance=instance,
        solver_config=solver_config,
        cache_config=cache_config,
    )
    return Path(cache_config.directory) / f"{key}.json"


def load_cached_solution(
    *,
    cache_path: Path,
    solver: VRPRPDSolver,
    cache_config: CacheConfig,
) -> SolverRunResult | None:
    if not cache_path.exists():
        return None

    expected_key = _solution_cache_key(
        instance=solver.instance,
        solver_config=solver.solver_config,
        cache_config=cache_config,
    )
    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            raw = json.load(cache_file)
        if raw.get("cache_key") != expected_key:
            return None

        initial = solver.evaluate(_deserialize_routes(raw["stages"]["initial"]), require_complete=True)
        alns = solver.evaluate(_deserialize_routes(raw["stages"]["alns"]), require_complete=True)
        brkga = solver.evaluate(_deserialize_routes(raw["stages"]["brkga"]), require_complete=True)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    labeled = {
        "initial": initial,
        "alns": alns,
        "brkga": brkga,
    }
    best_label = raw.get("best_label")
    if best_label not in labeled:
        best_label = min(labeled, key=lambda label: labeled[label].makespan)

    selected_label = raw.get("selected_label", "paper")
    return SolverRunResult(
        initial=initial,
        alns=alns,
        brkga=brkga,
        selected_label=selected_label,
        best_label=best_label,
        best=labeled[best_label],
    )


def save_cached_solution(
    *,
    cache_path: Path,
    solver: VRPRPDSolver,
    result: SolverRunResult,
    cache_config: CacheConfig,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": _solution_cache_key(
            instance=solver.instance,
            solver_config=solver.solver_config,
            cache_config=cache_config,
        ),
        "selected_label": result.selected_label,
        "best_label": result.best_label,
        "stages": {
            "initial": _serialize_routes(result.initial.routes),
            "alns": _serialize_routes(result.alns.routes),
            "brkga": _serialize_routes(result.brkga.routes),
        },
    }
    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, indent=2)


def _solution_cache_key(
    *,
    instance: Instance,
    solver_config: SolverConfig,
    cache_config: CacheConfig,
) -> str:
    payload = {
        "schema_version": cache_config.schema_version,
        "algorithm_version": cache_config.algorithm_version,
        "instance": _instance_signature(instance),
        "solver": asdict(solver_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _instance_signature(instance: Instance) -> Dict[str, Any]:
    return {
        "vehicle_count": instance.vehicle_count,
        "capacity": instance.capacity,
        "depot_node": instance.depot_node,
        "active_job_ids": list(instance.active_job_ids),
        "processing_times": {
            str(customer_id): round(value, 10)
            for customer_id, value in sorted(instance.processing_times.items())
        },
        "instance_config": asdict(instance.instance_config),
        "stations": [
            {
                "station_id": station.station_id,
                "node_id": station.node_id,
                "coord": _rounded_coord(station.coord),
            }
            for station in instance.stations
        ],
        "world": {
            "coords": {
                node_id: _rounded_coord(coord)
                for node_id, coord in sorted(instance.world.coords.items())
            },
            "edges": {
                node_id: [
                    [neighbor, round(weight, 10)]
                    for neighbor, weight in sorted(neighbors, key=lambda item: (item[0], item[1]))
                ]
                for node_id, neighbors in sorted(instance.world.edges.items())
            },
        },
    }


def _serialize_routes(routes: Routes) -> List[List[Dict[str, Any]]]:
    return [
        [
            {
                "customer_id": operation.customer_id,
                "kind": operation.kind,
            }
            for operation in route
        ]
        for route in routes
    ]


def _deserialize_routes(raw_routes: List[List[Dict[str, Any]]]) -> Routes:
    return [
        [
            Operation(
                customer_id=int(operation["customer_id"]),
                kind=str(operation["kind"]),
            )
            for operation in route
        ]
        for route in raw_routes
    ]


def _rounded_coord(coord: tuple[float, float]) -> List[float]:
    return [round(coord[0], 10), round(coord[1], 10)]
