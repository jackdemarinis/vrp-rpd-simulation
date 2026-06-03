import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from vrp_rpd_sim import cad_layout, config
from vrp_rpd_sim.model import Operation
from vrp_rpd_sim.solver import VRPRPDSolver
from vrp_rpd_sim.world import build_instance, build_world


class CadLayoutWorldTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_grid_has_eight_by_eight_intersections(self) -> None:
        world, stations = build_world()

        self.assertEqual(8, len(world.road_xs))
        self.assertEqual(8, len(world.road_ys))
        self.assertEqual(49, len(stations))
        self.assertTrue(all(station.side == "square" for station in stations))

    def test_grid_pitch_matches_cad_layout(self) -> None:
        world, _ = build_world()

        for index, x in enumerate(cad_layout.GRID_XS):
            self.assertAlmostEqual(world.road_xs[index], x, places=4)
        for index, y in enumerate(cad_layout.GRID_YS):
            self.assertAlmostEqual(world.road_ys[index], y, places=4)

    def test_depot_has_ten_slots_in_l_shape(self) -> None:
        world, _ = build_world()

        self.assertEqual(10, len(world.depot_slots))
        # First slot is the NE grid corner — doubles as a routing node.
        self.assertEqual(cad_layout.GRID_CORNER_DEPOT_SLOT, world.depot_slots[0])
        # Vertical arm slots share x = entry x.
        for slot in cad_layout.DEPOT_VERTICAL_SLOTS:
            self.assertIn(slot, world.depot_slots)
        # Horizontal arm slots share y = arm y.
        for slot in cad_layout.DEPOT_HORIZONTAL_SLOTS:
            self.assertIn(slot, world.depot_slots)

    def test_single_main_depot_entry(self) -> None:
        world, _ = build_world()

        self.assertIn("main", world.depot_entries)
        self.assertEqual(1, len(world.depot_entries["main"]))
        # Entry coincides with the NE grid intersection.
        ne_corner_coord = world.coords[f"i_{len(world.road_xs) - 1}_{len(world.road_ys) - 1}"]
        self.assertEqual(ne_corner_coord, world.depot_entries["main"][0])

    def test_alvik_fleet_size_is_ten(self) -> None:
        instance = build_instance()

        self.assertEqual(10, config.ALVIK_COUNT)
        self.assertEqual(10, instance.vehicle_count)

    def test_default_instance_activates_all_49_stations(self) -> None:
        instance = build_instance()

        self.assertEqual(49, len(instance.stations))
        self.assertEqual(49, len(instance.active_job_ids))

    def test_vehicle_capacity_is_four_and_five_drops_are_infeasible(self) -> None:
        instance = build_instance()
        solver = VRPRPDSolver(instance)

        self.assertEqual(4, config.VEHICLE_CAPACITY)
        self.assertEqual(4, instance.capacity)

        feasible_customer_ids = instance.active_job_ids[:4]
        infeasible_customer_ids = instance.active_job_ids[:5]

        feasible_route = [[Operation(customer_id, "D") for customer_id in feasible_customer_ids] + [Operation(customer_id, "P") for customer_id in feasible_customer_ids]]
        feasible_route.extend([[] for _ in range(instance.vehicle_count - 1)])
        feasible = solver.evaluate(feasible_route, require_complete=False)
        self.assertTrue(feasible.feasible, msg=feasible.reason)

        infeasible_route = [[Operation(customer_id, "D") for customer_id in infeasible_customer_ids] + [Operation(customer_id, "P") for customer_id in infeasible_customer_ids]]
        infeasible_route.extend([[] for _ in range(instance.vehicle_count - 1)])
        infeasible = solver.evaluate(infeasible_route, require_complete=False)
        self.assertFalse(infeasible.feasible)
        self.assertIn("Capacity violation", infeasible.reason)


if __name__ == "__main__":
    unittest.main()
