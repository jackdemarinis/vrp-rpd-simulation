import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from vrp_rpd_sim import config
from vrp_rpd_sim.render import SimulationApp, euclidean
from vrp_rpd_sim.solver import VRPRPDSolver
from vrp_rpd_sim.world import build_instance


class DepotReturnSimulationTests(unittest.TestCase):
    TEST_JOB_COUNT = 16

    def tearDown(self) -> None:
        pygame.quit()

    def test_simulation_keeps_vehicle_centers_separated(self) -> None:
        instance = build_instance(job_count=self.TEST_JOB_COUNT)
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        min_allowed = config.ALVIK_SIZE_IN / 2.0
        closest = None
        for _ in range(3600):
            app._step_simulation(0.1)
            for index, first in enumerate(app.vehicles):
                for second in app.vehicles[index + 1 :]:
                    distance = euclidean(first.position, second.position)
                    if distance < min_allowed - 1e-6:
                        closest = (
                            first.vehicle_id,
                            second.vehicle_id,
                            tuple(round(value, 2) for value in first.position),
                            tuple(round(value, 2) for value in second.position),
                            distance,
                        )
                        break
                if closest is not None:
                    break
            if closest is not None or all(vehicle.completed for vehicle in app.vehicles):
                break

        self.assertIsNone(closest, msg=f"vehicles overlapped reservation guard: {closest}")

    def test_full_simulation_returns_every_vehicle_to_the_grid(self) -> None:
        instance = build_instance(job_count=self.TEST_JOB_COUNT)
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        for _ in range(420):
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

    def test_paths_stay_axis_aligned_for_outbound_and_return_travel(self) -> None:
        instance = build_instance(job_count=self.TEST_JOB_COUNT)
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        for vehicle in app.vehicles:
            if not vehicle.route:
                continue

            first_target = solver.customer_node(vehicle.route[0].customer_id)
            outbound = app._travel_points(vehicle.position, vehicle.current_node, first_target, False)
            for start, end in zip(outbound, outbound[1:]):
                self.assertTrue(
                    abs(start[0] - end[0]) <= 1e-6 or abs(start[1] - end[1]) <= 1e-6,
                    msg=f"vehicle {vehicle.vehicle_id} outbound path contains diagonal segment {start} -> {end}",
                )

            vehicle.current_node = solver.customer_node(vehicle.route[-1].customer_id)
            vehicle.position = instance.world.coords[vehicle.current_node]
            vehicle.route_index = len(vehicle.route)
            slot = app.depot_return_slots[0]
            corridor = app._best_corridor_for_return_slot(vehicle, slot)
            vehicle.home_slot = slot
            vehicle.return_slot_index = app.depot_return_slot_ranks[slot]
            vehicle.depot_entry = app.depot_corridor_entries[corridor]
            vehicle.depot_corridor = corridor
            homebound = app._travel_points(
                vehicle.position,
                vehicle.current_node,
                instance.depot_node,
                True,
                home_slot=vehicle.home_slot,
                depot_entry=vehicle.depot_entry,
                depot_corridor=vehicle.depot_corridor,
            )
            for start, end in zip(homebound, homebound[1:]):
                self.assertTrue(
                    abs(start[0] - end[0]) <= 1e-6 or abs(start[1] - end[1]) <= 1e-6,
                    msg=f"vehicle {vehicle.vehicle_id} return path contains diagonal segment {start} -> {end}",
                )

    def test_outbound_departures_clear_depot_control_zone(self) -> None:
        instance = build_instance(job_count=self.TEST_JOB_COUNT)
        solver = VRPRPDSolver(instance)
        result = solver.solve()
        app = SimulationApp(instance, solver, result)

        depot_entries = {
            (round(x, 6), round(y, 6))
            for entries in instance.world.depot_entries.values()
            for x, y in entries
        }
        for vehicle in app.vehicles:
            if not vehicle.route:
                continue
            first_target = solver.customer_node(vehicle.route[0].customer_id)
            outbound = app._travel_points(vehicle.position, vehicle.current_node, first_target, False)
            self.assertTrue(
                any((round(x, 6), round(y, 6)) in depot_entries for x, y in outbound),
                msg=f"vehicle {vehicle.vehicle_id} outbound path does not use a depot egress corridor",
            )

        saw_departure_clear = False
        for _ in range(300):
            app._step_simulation(0.1)
            active_departures = [
                vehicle
                for vehicle in app.vehicles
                if vehicle.route
                and vehicle.current_node == instance.depot_node
                and vehicle.active_path is not None
                and app._is_in_depot_control_zone(vehicle.position)
            ]
            self.assertLessEqual(
                len(active_departures),
                1,
                msg=f"multiple vehicles departing inside depot control zone: {active_departures}",
            )
            if any(vehicle.route and not app._is_in_depot_control_zone(vehicle.position) for vehicle in app.vehicles):
                saw_departure_clear = True

        self.assertTrue(saw_departure_clear, msg="no departing vehicle cleared the depot control zone")

if __name__ == "__main__":
    unittest.main()
