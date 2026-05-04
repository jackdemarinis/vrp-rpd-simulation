import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from vrp_rpd_sim import config
from vrp_rpd_sim.model import Operation
from vrp_rpd_sim.render import MovementCandidate, build_path_state, euclidean
from vrp_rpd_sim.solver import VRPRPDSolver
from vrp_rpd_sim.world import build_instance
from simulation_fixtures import build_app_from_routes, build_fixture_app


class DepotReturnSimulationTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_full_fixture_simulation_keeps_hard_clearance_without_freezing(self) -> None:
        app = build_fixture_app()
        min_allowed = app._minimum_vehicle_center_distance()
        closest = None
        last_positions = [vehicle.position for vehicle in app.vehicles]
        still_ticks = 0
        for _ in range(6000):
            app._step_simulation(0.1)
            moved = any(
                euclidean(vehicle.position, previous) > 1e-7
                for vehicle, previous in zip(app.vehicles, last_positions)
            )
            last_positions = [vehicle.position for vehicle in app.vehicles]
            still_ticks = 0 if moved else still_ticks + 1
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

        self.assertIsNone(closest, msg=f"vehicles overlapped visible bodies: {closest}")
        self.assertLess(still_ticks, 200, msg="simulation froze with active vehicles")
        self.assertGreaterEqual(app._completed_jobs(), 20)

    def test_full_simulation_returns_every_vehicle_to_the_grid(self) -> None:
        active_ids = build_instance(job_count=4).active_job_ids
        routes = [
            [Operation(active_ids[0], "D"), Operation(active_ids[0], "P")],
            [Operation(active_ids[1], "D"), Operation(active_ids[1], "P")],
            [Operation(active_ids[2], "D"), Operation(active_ids[2], "P")],
            [Operation(active_ids[3], "D"), Operation(active_ids[3], "P")],
            [],
            [],
            [],
            [],
            [],
        ]
        app = build_app_from_routes(routes, job_count=4)

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
        app = build_fixture_app()
        instance = app.instance
        solver = app.solver

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
        app = build_fixture_app()
        instance = app.instance
        solver = app.solver

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

    def test_conflict_arbitration_prefers_longer_blocked_vehicle(self) -> None:
        app = build_fixture_app()
        first = app.vehicles[0]
        second = app.vehicles[1]
        shared_station = {"station:demo"}
        candidates = {
            first.vehicle_id: MovementCandidate(
                vehicle=first,
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, 0.0),
                end_position=(1.0, 0.0),
                resources=shared_station,
            ),
            second.vehicle_id: MovementCandidate(
                vehicle=second,
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, 5.0),
                end_position=(1.0, 5.0),
                resources=shared_station,
            ),
        }
        app.blocked_counts[first.vehicle_id] = 0
        app.blocked_counts[second.vehicle_id] = 3

        current_positions = {
            first.vehicle_id: first.position,
            second.vehicle_id: second.position,
        }
        winners = app._select_movement_winners(candidates, 1.0, {}, current_positions)

        self.assertEqual({second.vehicle_id}, winners)

    def test_intersection_right_of_way_prefers_more_prior_pauses(self) -> None:
        app = build_fixture_app()
        first = app.vehicles[0]
        second = app.vehicles[1]
        for vehicle in app.vehicles:
            vehicle.completed = True
            vehicle.active_path = None

        first.completed = False
        second.completed = False
        center = (app.instance.world.road_xs[1], app.instance.world.road_ys[1])
        radius = app._intersection_control_radius()
        first.position = (center[0] - radius - 1.0, center[1])
        second.position = (center[0], center[1] - radius - 1.0)
        first.active_path = build_path_state([first.position, center])
        second.active_path = build_path_state([second.position, center])
        app.pause_counts[first.vehicle_id] = 1
        app.pause_counts[second.vehicle_id] = 5
        resource = app._intersection_resource_for_point(center)

        winners = app._intersection_winners([first, second], radius + 2.0, {})

        self.assertIsNotNone(resource)
        self.assertEqual(second.vehicle_id, winners[resource])
        allowed = app._max_resource_safe_travel(
            first,
            first.active_path.total_length,
            {},
            {resource: second.vehicle_id},
        )
        self.assertLess(allowed, first.active_path.total_length)

    def test_far_apart_vehicles_move_in_same_substep(self) -> None:
        app = build_fixture_app()
        first = app.vehicles[0]
        second = app.vehicles[1]
        for vehicle in app.vehicles:
            vehicle.completed = True
            vehicle.active_path = None

        first.completed = False
        second.completed = False
        first.position = (30.0, 30.0)
        second.position = (30.0, 50.0)
        first.active_path = build_path_state([first.position, (35.0, 30.0)])
        second.active_path = build_path_state([second.position, (35.0, 50.0)])

        app._advance_vehicles(0.1)

        self.assertGreater(first.position[0], 30.0)
        self.assertGreater(second.position[0], 30.0)

    def test_independent_conflict_groups_choose_independent_winners(self) -> None:
        app = build_fixture_app()
        vehicles = app.vehicles[:4]
        first_station = {"station:first"}
        second_station = {"station:second"}
        candidates = {
            vehicles[0].vehicle_id: MovementCandidate(
                vehicle=vehicles[0],
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, 0.0),
                end_position=(1.0, 0.0),
                resources=first_station,
            ),
            vehicles[1].vehicle_id: MovementCandidate(
                vehicle=vehicles[1],
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, 5.0),
                end_position=(1.0, 5.0),
                resources=first_station,
            ),
            vehicles[2].vehicle_id: MovementCandidate(
                vehicle=vehicles[2],
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, 20.0),
                end_position=(1.0, 20.0),
                resources=second_station,
            ),
            vehicles[3].vehicle_id: MovementCandidate(
                vehicle=vehicles[3],
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, 25.0),
                end_position=(1.0, 25.0),
                resources=second_station,
            ),
        }
        app.blocked_counts[vehicles[0].vehicle_id] = 0
        app.blocked_counts[vehicles[1].vehicle_id] = 3
        app.blocked_counts[vehicles[2].vehicle_id] = 4
        app.blocked_counts[vehicles[3].vehicle_id] = 0

        current_positions = {
            vehicle.vehicle_id: vehicle.position for vehicle in vehicles
        }
        winners = app._select_movement_winners(candidates, 1.0, {}, current_positions)

        self.assertEqual({vehicles[1].vehicle_id, vehicles[2].vehicle_id}, winners)

    def test_shared_abstract_road_segment_does_not_serialize_far_apart_candidates(self) -> None:
        app = build_fixture_app()
        first = app.vehicles[0]
        second = app.vehicles[1]
        shared_road_segment = {"segment:h:demo"}
        candidates = {
            first.vehicle_id: MovementCandidate(
                vehicle=first,
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, 0.0),
                end_position=(1.0, 0.0),
                resources=shared_road_segment,
            ),
            second.vehicle_id: MovementCandidate(
                vehicle=second,
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, 20.0),
                end_position=(1.0, 20.0),
                resources=shared_road_segment,
            ),
        }

        current_positions = {
            first.vehicle_id: first.position,
            second.vehicle_id: second.position,
        }
        winners = app._select_movement_winners(candidates, 1.0, {}, current_positions)

        self.assertEqual({first.vehicle_id, second.vehicle_id}, winners)

    def test_shared_intersection_resource_does_not_serialize_clear_lanes(self) -> None:
        app = build_fixture_app()
        first = app.vehicles[0]
        second = app.vehicles[1]
        lane_gap = app._minimum_vehicle_center_distance() + 0.1
        candidates = {
            first.vehicle_id: MovementCandidate(
                vehicle=first,
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, 0.0),
                end_position=(1.0, 0.0),
                resources={"intersection:demo"},
            ),
            second.vehicle_id: MovementCandidate(
                vehicle=second,
                start_distance=0.0,
                end_distance=1.0,
                start_position=(0.0, lane_gap),
                end_position=(1.0, lane_gap),
                resources={"intersection:demo"},
            ),
        }

        current_positions = {
            first.vehicle_id: first.position,
            second.vehicle_id: second.position,
        }
        winners = app._select_movement_winners(candidates, 1.0, {}, current_positions)

        self.assertEqual({first.vehicle_id, second.vehicle_id}, winners)

    def test_homebound_vehicles_receive_return_paths_concurrently(self) -> None:
        active_ids = build_instance(job_count=4).active_job_ids
        routes = [
            [Operation(active_ids[0], "D"), Operation(active_ids[0], "P")],
            [Operation(active_ids[1], "D"), Operation(active_ids[1], "P")],
            [Operation(active_ids[2], "D"), Operation(active_ids[2], "P")],
            [Operation(active_ids[3], "D"), Operation(active_ids[3], "P")],
            [],
            [],
            [],
            [],
            [],
        ]
        app = build_app_from_routes(routes, job_count=4)
        active_vehicles = [vehicle for vehicle in app.vehicles if vehicle.route]
        for vehicle in active_vehicles:
            last_stop = app.solver.customer_node(vehicle.route[-1].customer_id)
            vehicle.current_node = last_stop
            vehicle.position = app.instance.world.coords[last_stop]
            vehicle.route_index = len(vehicle.route)
            vehicle.active_path = None

        app._prime_vehicle_targets()

        returning = [
            vehicle
            for vehicle in active_vehicles
            if vehicle.return_slot_index is not None and vehicle.active_path is not None
        ]
        self.assertGreaterEqual(len(returning), 2)

    def test_opposite_direction_lanes_use_separate_offsets(self) -> None:
        instance = build_instance(job_count=4)
        solver = VRPRPDSolver(instance)
        eastbound = solver.lane_polyline(["i_1_1", "i_2_1"])
        westbound = solver.lane_polyline(["i_2_1", "i_1_1"])

        self.assertGreater(
            abs(eastbound[0][1] - westbound[0][1]),
            config.ALVIK_SIZE_IN + config.MIN_ALVIK_CLEARANCE_IN,
        )

if __name__ == "__main__":
    unittest.main()
