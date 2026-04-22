import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from vrp_rpd_sim.render import SimulationApp, euclidean
from vrp_rpd_sim.solver import VRPRPDSolver
from vrp_rpd_sim.world import build_instance


class DepotReturnSimulationTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_full_simulation_returns_every_vehicle_to_the_grid(self) -> None:
        instance = build_instance()
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
