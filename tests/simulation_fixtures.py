import json
from pathlib import Path

from vrp_rpd_sim.model import Operation
from vrp_rpd_sim.render import SimulationApp
from vrp_rpd_sim.solver import SolverRunResult, VRPRPDSolver
from vrp_rpd_sim.world import build_instance


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture_routes(name: str = "default_36_job_routes.json"):
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return [
        [
            Operation(customer_id=int(raw_operation[1:]), kind=raw_operation[0])
            for raw_operation in route
        ]
        for route in payload["routes"]
    ]


def build_fixture_app(name: str = "default_36_job_routes.json") -> SimulationApp:
    routes = load_fixture_routes(name)
    return build_app_from_routes(routes, job_count=36)


def build_app_from_routes(routes, job_count: int) -> SimulationApp:
    instance = build_instance(job_count=job_count)
    solver = VRPRPDSolver(instance)
    evaluated = solver.evaluate(routes, require_complete=True)
    result = SolverRunResult(
        initial=evaluated,
        alns=evaluated,
        brkga=evaluated,
        selected_label="fixture",
        best_label="fixture",
        best=evaluated,
    )
    return SimulationApp(instance, solver, result)
