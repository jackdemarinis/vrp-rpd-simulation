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

    def test_depot_entry_points_geometry(self) -> None:
        world, _ = build_world()

        top_entries = [(round(x, 3), round(y, 3)) for x, y in world.depot_entries["top"]]
        right_entries = [(round(x, 3), round(y, 3)) for x, y in world.depot_entries["right"]]

        self.assertEqual(
            [(1.97, 15.0), (6.5, 15.0), (11.03, 15.0)],
            top_entries,
        )
        self.assertEqual(
            [(15.0, 1.97), (15.0, 6.5), (15.0, 11.03)],
            right_entries,
        )

    def test_homebound_paths_end_at_assigned_slots(self) -> None:
        instance = build_instance()
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        active_vehicles = [vehicle for vehicle in app.vehicles if vehicle.route]
        for vehicle in active_vehicles:
            last_stop = solver.customer_node(vehicle.route[-1].customer_id)
            vehicle.current_node = last_stop
            vehicle.position = instance.world.coords[last_stop]
            vehicle.route_index = len(vehicle.route)
            vehicle.active_path = None

        app._assign_return_targets()

        for vehicle in active_vehicles[:3]:
            path = app._travel_points(
                vehicle.position,
                vehicle.current_node,
                instance.depot_node,
                True,
                home_slot=vehicle.home_slot,
                depot_entry=vehicle.depot_entry,
                depot_corridor=vehicle.depot_corridor,
            )
            self.assertEqual(vehicle.home_slot, path[-1])
            self.assertIn(vehicle.depot_entry, path)

    def test_corridor_stack_invariant_no_crossover(self) -> None:
        instance = build_instance()
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        top_corridor_slots = app.depot_corridor_slots["top_1"]
        self.assertEqual(
            sorted(top_corridor_slots, key=lambda slot: (round(slot[1], 6), round(slot[0], 6))),
            top_corridor_slots,
        )

        app.reserved_return_slots = {
            slot for slot in app.depot_return_slots if slot not in set(top_corridor_slots)
        }
        app._refresh_corridor_frontier()

        observed = []
        for slot in top_corridor_slots:
            observed.append(app.corridor_next_slot["top_1"])
            app.reserved_return_slots.add(slot)
            app._refresh_corridor_frontier()

        self.assertEqual(top_corridor_slots, observed)


if __name__ == "__main__":
    unittest.main()
