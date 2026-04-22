import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from vrp_rpd_sim.render import SimulationApp
from vrp_rpd_sim.solver import VRPRPDSolver
from vrp_rpd_sim.world import build_instance, build_world


class DepotLayoutTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_depot_slots_form_three_by_three_grid(self) -> None:
        world, _ = build_world()

        xs = {round(slot[0], 3) for slot in world.depot_slots}
        ys = {round(slot[1], 3) for slot in world.depot_slots}

        self.assertEqual(9, len(world.depot_slots))
        self.assertEqual(3, len(xs))
        self.assertEqual(3, len(ys))

    def test_homebound_paths_end_at_assigned_slots(self) -> None:
        instance = build_instance()
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        depot_access = instance.world.coords[instance.depot_node]
        for vehicle in app.vehicles:
            if not vehicle.route:
                continue
            last_stop = solver.customer_node(vehicle.route[-1].customer_id)
            path = app._travel_points(
                instance.world.coords[last_stop],
                last_stop,
                instance.depot_node,
                True,
                home_slot=vehicle.home_slot,
            )
            self.assertEqual(vehicle.home_slot, path[-1])
            self.assertIn(depot_access, path)

    def test_return_slots_fill_from_furthest_corner_first(self) -> None:
        instance = build_instance()
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        active_vehicles = [vehicle for vehicle in app.vehicles if vehicle.route]
        for vehicle in active_vehicles[:3]:
            app._assign_return_slot(vehicle)

        assigned_slots = [vehicle.home_slot for vehicle in active_vehicles[:3]]
        self.assertEqual(app.depot_return_slots[:3], assigned_slots)


if __name__ == "__main__":
    unittest.main()
