"""Entry point for the VRP-RPD Manhattan-grid simulation."""

from __future__ import annotations

import argparse

from vrp_rpd_sim import config
from vrp_rpd_sim.world import build_instance
from vrp_rpd_sim.solver import VRPRPDSolver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VRP-RPD Pygame simulation")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Solve the instance and print a short summary without starting Pygame.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=config.ACTIVE_JOB_COUNT,
        help="Number of active jobs to schedule across the 16 perimeter stations.",
    )
    parser.add_argument(
        "--sim-speed",
        type=float,
        default=config.SIM_SPEED_MULTIPLIER,
        help="Initial playback speed multiplier.",
    )
    parser.add_argument(
        "--process-scale",
        type=float,
        default=config.PROCESSING_TIME_SCALE,
        help="Multiplier applied to generated processing times.",
    )
    parser.add_argument(
        "--fixed-process-seconds",
        type=float,
        default=config.FIXED_PROCESSING_TIME_SEC,
        help="Optional fixed processing time for every job.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=config.START_FULLSCREEN,
        help="Start the simulation in fullscreen mode.",
    )
    parser.add_argument(
        "--debug-depot",
        action="store_true",
        help="Print depot return debug logs while the simulation is running.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    instance = build_instance(
        job_count=args.jobs,
        processing_scale=args.process_scale,
        fixed_processing_time=args.fixed_process_seconds,
    )
    solver = VRPRPDSolver(instance)
    result = solver.solve()

    print("VRP-RPD solve summary")
    print(f"  Initial makespan: {result.initial.makespan:.2f}s")
    print(f"  ALNS makespan:    {result.alns.makespan:.2f}s")
    print(f"  BRKGA makespan:   {result.brkga.makespan:.2f}s")
    print(f"  Using:            {result.best_label}")

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
        sim_speed=args.sim_speed,
        job_count=args.jobs,
        processing_scale=args.process_scale,
        fixed_processing_time=args.fixed_process_seconds,
        fullscreen=args.fullscreen,
        debug_depot=args.debug_depot,
    )
    app.run()


if __name__ == "__main__":
    main()
