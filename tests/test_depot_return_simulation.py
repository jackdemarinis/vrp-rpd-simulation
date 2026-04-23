import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from vrp_rpd_sim.render import SimulationApp, euclidean
from vrp_rpd_sim.solver import VRPRPDSolver
from vrp_rpd_sim.world import build_instance


class DepotReturnSimulationTests(unittest.TestCase):
    TEST_JOB_COUNT = 16

    def tearDown(self) -> None:
        pygame.quit()

    def test_concurrent_depot_return_overlaps(self) -> None:
        instance = build_instance(job_count=self.TEST_JOB_COUNT)
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        saw_overlap = False
        for _ in range(1200):
            before_positions = {vehicle.vehicle_id: vehicle.position for vehicle in app.vehicles}
            app._step_simulation(0.25)
            moved_inside_depot = [
                vehicle
                for vehicle in app.vehicles
                if app._is_homebound(vehicle)
                and not vehicle.completed
                and euclidean(before_positions[vehicle.vehicle_id], vehicle.position) > 1e-4
                and vehicle.position[0] <= 15.0 + 1e-6
                and vehicle.position[1] <= 15.0 + 1e-6
            ]
            if len(moved_inside_depot) >= 2:
                saw_overlap = True
                break

        self.assertTrue(saw_overlap, msg="expected multiple homebound vehicles to move inside the depot together")

    def test_full_simulation_returns_every_vehicle_to_the_grid(self) -> None:
        instance = build_instance(job_count=self.TEST_JOB_COUNT)
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        for _ in range(320):
            app._step_simulation(1.0)
            if all(vehicle.completed for vehicle in app.vehicles):
                break

        incomplete = [
            (
                vehicle.vehicle_id,
                tuple(round(value, 2) for value in vehicle.position),
                tuple(round(value, 2) for value in vehicle.home_slot),
            )
            for vehicle in app.vehicles
            if not vehicle.completed
        ]
        self.assertFalse(incomplete, msg=f"vehicles stuck before depot return completes: {incomplete}")

        for vehicle in app.vehicles:
            self.assertLessEqual(
                euclidean(vehicle.position, vehicle.home_slot),
                1e-6,
                msg=f"vehicle {vehicle.vehicle_id} did not finish on its assigned depot slot",
            )

        xs = {round(vehicle.position[0], 3) for vehicle in app.vehicles}
        ys = {round(vehicle.position[1], 3) for vehicle in app.vehicles}
        self.assertEqual(3, len(xs))
        self.assertEqual(3, len(ys))


if __name__ == "__main__":
    unittest.main()
