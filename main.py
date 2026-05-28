"""Entry point for the VRP-RPD Manhattan-grid simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from vrp_rpd_sim import config
from vrp_rpd_sim.mapf import MapfInfeasible, plan_space_time, validate as validate_schedule
from vrp_rpd_sim.unity_export import export_unity_json
from vrp_rpd_sim.solution_cache import solve_with_solution_cache
from vrp_rpd_sim.world import build_instance
from vrp_rpd_sim.solver import VRPRPDSolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VRP-RPD Pygame simulation")
    parser.add_argument(
        "--config",
        default=str(config.DEFAULT_RUNTIME_CONFIG_PATH),
        help="Path to the runtime configuration JSON file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Solve the instance and print a short summary without starting Pygame.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of active jobs to schedule across the 16 perimeter stations.",
    )
    parser.add_argument(
        "--sim-speed",
        type=float,
        default=None,
        help="Initial playback speed multiplier.",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=None,
        help="Multiplier applied to generated processing times.",
    )
    parser.add_argument(
        "--fixed-process-seconds",
        type=float,
        default=None,
        help="Optional fixed processing time for every job.",
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Start the simulation in fullscreen mode.",
    )
    parser.add_argument(
        "--debug-depot",
        action="store_true",
        help="Print depot return debug logs while the simulation is running.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override solver/collision RNG seed. With fixed-timestep playback, same seed = same outcome every run.",
    )
    parser.add_argument(
        "--export-unity-json",
        default=None,
        help="Write a Unity playback snapshot JSON file and exit.",
    )
    parser.add_argument(
        "--record-unity-json",
        default=None,
        help="Run pygame normally and record frames to this JSON file. Auto-quits when all vehicles complete.",
    )
    parser.add_argument(
        "--unity-capture-interval",
        type=float,
        default=0.1,
        help="Snapshot cadence in seconds for Unity export playback data.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime_config = config.load_runtime_config(Path(args.config))
    runtime_config = config.apply_cli_overrides(
        runtime_config,
        job_count=args.jobs,
        processing_scale=args.process_scale,
        fixed_processing_time=args.fixed_process_seconds,
        sim_speed=args.sim_speed,
        fullscreen=args.fullscreen,
        seed=args.seed,
    )
    instance = build_instance(
        instance_config=runtime_config.instance,
    )
    solver = VRPRPDSolver(instance, solver_config=runtime_config.solver)
    result, cache_hit, cache_path = solve_with_solution_cache(solver, runtime_config.cache)

    print("VRP-RPD solve summary")
    print(f"  Initial makespan: {result.initial.makespan:.2f}s")
    print(f"  ALNS makespan:    {result.alns.makespan:.2f}s")
    print(f"  BRKGA makespan:   {result.brkga.makespan:.2f}s")
    print(f"  Pipeline mode:    {result.selected_label} (ALNS -> BRKGA)")
    print(f"  Best stage seen:  {result.best_label}")
    print(f"  Final used:       {result.best_label}")
    if cache_path is not None:
        print(f"  Solution cache:   {'hit' if cache_hit else 'miss'} ({cache_path})")

    try:
        scheduled = plan_space_time(instance, solver, result.best, runtime_config.mapf)
        validate_schedule(scheduled, clearance_sec=runtime_config.mapf.clearance_sec)
    except MapfInfeasible as exc:
        raise SystemExit(f"MAPF planner could not produce a collision-free schedule: {exc}") from exc

    delta = scheduled.makespan - result.best.makespan
    print(
        f"  MAPF makespan:    {scheduled.makespan:.2f}s "
        f"(+{delta:.2f}s vs solver, reason={scheduled.reason})"
    )

    if args.export_unity_json:
        export_path = export_unity_json(
            args.export_unity_json,
            runtime_config,
            capture_interval_sec=args.unity_capture_interval,
            debug_depot=args.debug_depot,
        )
        print(f"  Unity export:     {export_path}")
        return

    if args.headless:
        return

    try:
        from vrp_rpd_sim.render import SimulationApp
    except ModuleNotFoundError as exc:
        if exc.name == "pygame":
            raise SystemExit(
                "A compatible pygame package is not installed. Install the dependency from "
                "requirements.txt and rerun."
            ) from exc
        raise

    app = SimulationApp(
        instance,
        solver,
        result,
        scheduled=scheduled,
        sim_speed=runtime_config.app.sim_speed_multiplier,
        job_count=runtime_config.instance.active_job_count,
        processing_scale=runtime_config.instance.processing_time_scale,
        fixed_processing_time=runtime_config.instance.fixed_processing_time_sec,
        fullscreen=runtime_config.app.start_fullscreen,
        debug_depot=args.debug_depot,
        cache_config=runtime_config.cache,
        mapf_config=runtime_config.mapf,
        record_unity_path=args.record_unity_json,
        record_capture_interval_sec=args.unity_capture_interval,
        auto_quit_on_complete=args.record_unity_json is not None,
    )
    app.run()


if __name__ == "__main__":
    main()
