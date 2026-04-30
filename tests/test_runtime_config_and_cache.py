import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from vrp_rpd_sim.config import default_runtime_config, load_runtime_config
from vrp_rpd_sim.solution_cache import solve_with_solution_cache
from vrp_rpd_sim.solver import VRPRPDSolver
from vrp_rpd_sim.world import build_instance


class RuntimeConfigTests(unittest.TestCase):
    def test_load_runtime_config_reads_capacity_and_solver_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "simulation_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "instance": {
                            "vehicle_capacity": 7,
                            "active_job_count": 12,
                            "processing_time_overrides": {"3": 42.5},
                        },
                        "solver": {
                            "alns_max_iterations": 11,
                            "brkga_generations": 5,
                        },
                        "cache": {
                            "directory": "custom-cache",
                        },
                    }
                ),
                encoding="utf-8",
            )

            runtime_config = load_runtime_config(config_path)

            self.assertEqual(7, runtime_config.instance.vehicle_capacity)
            self.assertEqual(12, runtime_config.instance.active_job_count)
            self.assertEqual({3: 42.5}, runtime_config.instance.processing_time_overrides)
            self.assertEqual(11, runtime_config.solver.alns_max_iterations)
            self.assertEqual(5, runtime_config.solver.brkga_generations)
            self.assertEqual("custom-cache", runtime_config.cache.directory)


class SolutionCacheTests(unittest.TestCase):
    def test_second_solve_uses_cached_routes(self) -> None:
        defaults = default_runtime_config()
        instance_config = replace(
            defaults.instance,
            active_job_count=4,
            vehicle_count=2,
            vehicle_capacity=2,
            fixed_processing_time_sec=1.0,
        )
        solver_config = replace(
            defaults.solver,
            alns_max_iterations=1,
            brkga_population_size=8,
            brkga_generations=1,
            brkga_warm_seed_count=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_config = replace(defaults.cache, directory=tmpdir)
            instance = build_instance(instance_config=instance_config)
            first_solver = VRPRPDSolver(instance, solver_config=solver_config)

            first_result, first_hit, cache_path = solve_with_solution_cache(
                first_solver,
                cache_config,
            )

            self.assertFalse(first_hit)
            self.assertIsNotNone(cache_path)
            self.assertTrue(cache_path.exists())

            class FailIfSolvedSolver(VRPRPDSolver):
                def solve(self):  # type: ignore[override]
                    raise AssertionError("cache miss unexpectedly triggered solve()")

            second_solver = FailIfSolvedSolver(instance, solver_config=solver_config)
            second_result, second_hit, second_cache_path = solve_with_solution_cache(
                second_solver,
                cache_config,
            )

            self.assertTrue(second_hit)
            self.assertEqual(cache_path, second_cache_path)
            self.assertEqual(first_result.best_label, second_result.best_label)
            self.assertAlmostEqual(first_result.best.makespan, second_result.best.makespan)
            self.assertEqual(
                [[op.key for op in route] for route in first_result.best.routes],
                [[op.key for op in route] for route in second_result.best.routes],
            )


if __name__ == "__main__":
    unittest.main()
