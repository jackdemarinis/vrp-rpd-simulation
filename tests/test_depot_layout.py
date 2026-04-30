import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from vrp_rpd_sim import config
from vrp_rpd_sim.model import Operation
from vrp_rpd_sim.render import SimulationApp
from vrp_rpd_sim.solver import VRPRPDSolver
from vrp_rpd_sim.world import build_instance, build_world


class DepotLayoutTests(unittest.TestCase):
    TEST_JOB_COUNT = 16

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

    def test_stations_fill_all_road_enclosed_blocks(self) -> None:
        world, stations = build_world()

        self.assertEqual(36, len(stations))
        self.assertEqual(
            ((world.road_xs[0] + world.road_xs[1]) / 2.0, (world.road_ys[0] + world.road_ys[1]) / 2.0),
            stations[0].coord,
        )
        self.assertEqual(
            ((world.road_xs[5] + world.road_xs[6]) / 2.0, (world.road_ys[5] + world.road_ys[6]) / 2.0),
            stations[-1].coord,
        )
        self.assertTrue(all(station.side == "interior" for station in stations))
        for station in stations:
            cx, cy = station.coord
            self.assertGreater(cx, world.road_xs[0])
            self.assertLess(cx, world.road_xs[-1])
            self.assertGreater(cy, world.road_ys[0])
            self.assertLess(cy, world.road_ys[-1])

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

    def test_default_instance_activates_all_36_stations(self) -> None:
        instance = build_instance()

        self.assertEqual(36, len(instance.stations))
        self.assertEqual(36, len(instance.active_job_ids))

    def test_homebound_paths_end_at_assigned_slots(self) -> None:
        instance = build_instance(job_count=self.TEST_JOB_COUNT)
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

        assigned_slots = []
        for _ in range(3):
            app._assign_return_targets()
            vehicle = next(
                vehicle
                for vehicle in active_vehicles
                if vehicle.return_slot_index is not None and not vehicle.completed
            )
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
            assigned_slots.append(vehicle.home_slot)
            vehicle.completed = True

        self.assertEqual(app.depot_return_slots[:3], assigned_slots)

    def test_corridor_stack_invariant_no_crossover(self) -> None:
        instance = build_instance(job_count=self.TEST_JOB_COUNT)
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
