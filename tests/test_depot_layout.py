import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from dataclasses import replace

from vrp_rpd_sim import cad_layout, config
from vrp_rpd_sim.mapf import plan_space_time, validate
from vrp_rpd_sim.model import Operation
from vrp_rpd_sim.solver import VRPRPDSolver
from vrp_rpd_sim.world import build_instance, build_world


class CadLayoutWorldTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_lattice_is_8x8x7_with_343_stations(self) -> None:
        world, stations = build_world()

        self.assertEqual(8, len(world.road_xs))
        self.assertEqual(8, len(world.road_ys))
        self.assertEqual(7, len(world.road_zs))
        # 7x7x7 cube cells, each a station.
        self.assertEqual(343, len(stations))
        self.assertTrue(all(station.side == "cube" for station in stations))
        # Stations occupy every Z-plane, including the top one (which also
        # hosts the depot — the depot no longer owns a dedicated empty layer).
        station_planes = sorted({round(s.coord[2], 4) for s in stations})
        self.assertEqual(cad_layout.GRID_ZS, station_planes)
        depot_z = round(world.coords["depot"][2], 4)
        self.assertEqual(cad_layout.GRID_ZS[-1], depot_z)
        self.assertIn(depot_z, station_planes)

    def test_lattice_has_vertical_edges(self) -> None:
        world, _ = build_world()

        # Every interior intersection should connect up and down a layer.
        up_neighbor = "i_3_3_4"
        self.assertIn(
            up_neighbor,
            {n for n, _ in world.edges["i_3_3_3"]},
            msg="missing vertical (Z) corridor edge",
        )
        # A grid node should reach 6 neighbours (N/S/E/W + up/down) in the
        # interior of the cube.
        interior_degree = len(world.edges["i_3_3_3"])
        self.assertEqual(6, interior_degree)

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
        # Entry coincides with the top-layer NE grid intersection.
        ne_corner = (
            f"i_{len(world.road_xs) - 1}_{len(world.road_ys) - 1}_{len(world.road_zs) - 1}"
        )
        self.assertEqual(world.coords[ne_corner], world.depot_entries["main"][0])

    def test_alvik_fleet_size_is_ten(self) -> None:
        instance = build_instance()

        self.assertEqual(10, config.ALVIK_COUNT)
        self.assertEqual(10, instance.vehicle_count)

    def test_default_instance_has_343_stations_and_active_subset(self) -> None:
        instance = build_instance()

        self.assertEqual(343, len(instance.stations))
        self.assertEqual(config.ACTIVE_JOB_COUNT, len(instance.active_job_ids))
        self.assertLessEqual(len(instance.active_job_ids), len(instance.stations))

    def test_mapf_plans_through_z_without_conflicts(self) -> None:
        runtime = config.default_runtime_config()
        runtime = replace(
            runtime,
            instance=replace(runtime.instance, active_job_count=6, vehicle_count=2),
        )
        instance = build_instance(instance_config=runtime.instance)
        solver = VRPRPDSolver(instance, solver_config=runtime.solver)
        result = solver.solve()
        self.assertTrue(result.best.feasible, msg=result.best.reason)

        scheduled = plan_space_time(instance, solver, result.best, runtime.mapf)
        # Raises on any vertex/edge conflict — vertical edges must not break it.
        validate(scheduled, clearance_sec=runtime.mapf.clearance_sec)

        # At least one vehicle's route must change depth (descend into cube).
        changes_z = any(
            len({round(instance.world.coords[v.node_id][2], 4) for v in visits}) > 1
            for visits in scheduled.paths.values()
            if visits
        )
        self.assertTrue(changes_z, msg="no planned path traverses the Z axis")

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
