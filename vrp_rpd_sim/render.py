"""Pygame renderer for the VRP-RPD simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Set, Tuple

import pygame

from . import config
from .model import Instance, Operation
from .solution_cache import solve_with_solution_cache
from .solver import SolverRunResult, VRPRPDSolver, orthogonalize_points
from .world import build_instance


Coord = Tuple[float, float]


PALETTE = [
    (214, 73, 51),
    (74, 122, 186),
    (68, 157, 121),
    (204, 136, 64),
    (140, 95, 194),
    (61, 162, 168),
    (190, 84, 128),
    (113, 142, 62),
    (76, 92, 117),
]

BG = (244, 241, 234)
HUD_BG = (35, 42, 52)
ROAD_STRIP = (92, 95, 99)
ROAD_GAP = (225, 216, 197)
ROAD_EDGE = (54, 60, 67)
DEPOT_BG = (214, 231, 223)
DEPOT_EDGE = (88, 128, 111)
STATION_IDLE = (190, 191, 194)
STATION_PROCESSING = (231, 166, 59)
STATION_READY = (83, 167, 121)
STATION_DONE = (69, 123, 185)
STATION_INACTIVE = (205, 199, 183)
TEXT_DARK = (42, 48, 55)
TEXT_LIGHT = (240, 243, 246)
SLIDER_TRACK = (87, 97, 109)
SLIDER_FILL = (88, 184, 125)
SLIDER_KNOB = (245, 247, 249)
DEPARTURE_STAGGER_SECONDS = 0.2
COLLISION_CONTACT_BUFFER_IN = 0.05
INTERSECTION_PRIORITY = {"N": 0, "E": 1, "S": 2, "W": 3}


@dataclass
class PathState:
    points: List[Coord]
    total_length: float
    distance: float = 0.0

    def position(self) -> Coord:
        return self.position_at(self.distance)

    def position_at(self, distance: float) -> Coord:
        if self.total_length <= 0:
            return self.points[-1]
        fraction = distance / self.total_length
        return interpolate_polyline(self.points, fraction)


@dataclass
class VehicleState:
    vehicle_id: int
    color: Tuple[int, int, int]
    home_slot: Coord
    route: List[Operation]
    current_node: str
    position: Coord
    load: int
    route_index: int = 0
    active_path: PathState | None = None
    target_node: str | None = None
    waiting_customer_id: int | None = None
    completed: bool = False
    completion_time: float | None = None
    return_slot_index: int | None = None
    depot_entry: Coord | None = None
    depot_corridor: str | None = None
    departure_release_time: float = 0.0

    def load_marker(self) -> int:
        return self.load


@dataclass
class MovementCandidate:
    vehicle: VehicleState
    start_distance: float
    end_distance: float
    start_position: Coord
    end_position: Coord
    resources: Set[str]


class SimulationApp:
    def __init__(
        self,
        instance: Instance,
        solver: VRPRPDSolver,
        result: SolverRunResult,
        *,
        sim_speed: float | None = None,
        job_count: int | None = None,
        processing_scale: float | None = None,
        fixed_processing_time: float | None = None,
        fullscreen: bool | None = None,
        debug_depot: bool = False,
        cache_config: config.CacheConfig | None = None,
        headless: bool = False,
    ) -> None:
        self.headless = headless
        if not self.headless:
            pygame.init()
            pygame.display.set_caption(config.APP_TITLE)

        self.instance = instance
        self.solver = solver
        self.instance_config = instance.instance_config
        self.solver_config = solver.solver_config
        self.cache_config = cache_config or config.default_runtime_config().cache
        self.result = result
        self.solution = result.best

        self.fixed_processing_time = (
            fixed_processing_time
            if fixed_processing_time is not None
            else self.instance_config.fixed_processing_time_sec
        )
        self.active_job_count = job_count if job_count is not None else len(instance.active_job_ids)
        self.processing_scale = (
            processing_scale
            if processing_scale is not None
            else self.instance_config.processing_time_scale
        )
        self.fullscreen = fullscreen if fullscreen is not None else config.START_FULLSCREEN

        self.sim_speed_min = 0.25
        self.sim_speed_max = 12.0
        self.sim_speed_multiplier = sim_speed if sim_speed is not None else config.SIM_SPEED_MULTIPLIER
        self.sim_speed_multiplier = max(self.sim_speed_min, min(self.sim_speed_max, self.sim_speed_multiplier))

        self.sim_time = 0.0
        self.paused = False
        self.status_message = ""
        self.dragging_speed_slider = False
        self.speed_slider_rect = pygame.Rect(0, 0, 0, 0)
        self.speed_knob_rect = pygame.Rect(0, 0, 0, 0)
        self.debug_depot = debug_depot
        self.collision_rng = random.Random(self.solver_config.random_seed)
        self.blocked_counts: Dict[int, int] = {}
        self.pause_counts: Dict[int, int] = {}
        self.blocked_substeps = 0
        self.depot_return_slots = sorted(
            self.instance.world.depot_slots,
            key=lambda slot: (round(slot[1], 6), round(slot[0], 6)),
        )
        self.depot_return_slot_ranks = {
            slot: index for index, slot in enumerate(self.depot_return_slots)
        }
        self.reserved_return_slots: Set[Coord] = set()
        self.depot_corridor_entries: Dict[str, Coord] = {}
        self.depot_corridor_slots: Dict[str, List[Coord]] = {}
        self.slot_corridors: Dict[Coord, List[str]] = {}
        self.corridor_next_slot: Dict[str, Coord | None] = {}
        self._build_depot_return_topology()

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        if not self.headless:
            self.screen = self._set_display_mode(self.fullscreen)
            self.clock = pygame.time.Clock()
            self._rebuild_fonts()
            self._recompute_layout()

        self.station_progress = self._build_station_progress()
        self.vehicles = self._build_vehicle_states()
        self._refresh_corridor_frontier()

    def run(self) -> None:
        if self.headless:
            raise RuntimeError("SimulationApp.run() is unavailable in headless mode.")
        running = True
        while running:
            if self.clock is None:
                raise RuntimeError("Simulation clock was not initialized.")
            dt = min(self.clock.tick_busy_loop(config.FPS) / 1000.0, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event)
                elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    width = max(config.MIN_WINDOW_WIDTH_PX, event.w)
                    height = max(config.MIN_WINDOW_HEIGHT_PX, event.h)
                    self._resize_window(width, height)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_speed_slider = False
                elif event.type == pygame.MOUSEMOTION and self.dragging_speed_slider:
                    self._update_speed_from_mouse(event.pos)

            if not self.paused:
                self._step_simulation(dt * self.sim_speed_multiplier)
            self._draw()

        pygame.quit()

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
            return
        if event.key == pygame.K_r:
            self._reset_simulation()
            return
        if event.key in {pygame.K_f, pygame.K_F11}:
            self.fullscreen = not self.fullscreen
            self.screen = self._set_display_mode(self.fullscreen)
            self._rebuild_fonts()
            self._recompute_layout()
            return
        if event.key == pygame.K_MINUS:
            if self.active_job_count > 1:
                self.active_job_count -= 1
                self._refresh_solution("Reduced active jobs")
            return
        if event.key == pygame.K_EQUALS:
            if self.active_job_count < len(self.instance.stations):
                self.active_job_count += 1
                self._refresh_solution("Increased active jobs")
            return
        if event.key == pygame.K_COMMA:
            self.processing_scale = max(0.25, self.processing_scale - 0.25)
            self._refresh_solution("Reduced processing time scale")
            return
        if event.key == pygame.K_PERIOD:
            self.processing_scale = min(12.0, self.processing_scale + 0.25)
            self._refresh_solution("Increased processing time scale")

    def _handle_mouse_down(self, pos: Tuple[int, int]) -> None:
        if self.speed_slider_rect.collidepoint(pos) or self.speed_knob_rect.collidepoint(pos):
            self.dragging_speed_slider = True
            self._update_speed_from_mouse(pos)

    def _update_speed_from_mouse(self, pos: Tuple[int, int]) -> None:
        if self.speed_slider_rect.width <= 0:
            return
        x = max(self.speed_slider_rect.left, min(self.speed_slider_rect.right, pos[0]))
        fraction = (x - self.speed_slider_rect.left) / self.speed_slider_rect.width
        self.sim_speed_multiplier = self.sim_speed_min + (
            fraction * (self.sim_speed_max - self.sim_speed_min)
        )

    def _refresh_solution(self, message: str) -> None:
        self.status_message = f"{message}. Solving..."
        self.instance_config = replace(
            self.instance_config,
            active_job_count=self.active_job_count,
            processing_time_scale=self.processing_scale,
            fixed_processing_time_sec=self.fixed_processing_time,
        )
        self.instance = build_instance(
            job_count=self.active_job_count,
            processing_scale=self.processing_scale,
            fixed_processing_time=self.fixed_processing_time,
            instance_config=self.instance_config,
        )
        self.instance_config = self.instance.instance_config
        self.solver = VRPRPDSolver(self.instance, solver_config=self.solver_config)
        self.result, _, _ = solve_with_solution_cache(self.solver, self.cache_config)
        self.solution = self.result.best
        self._reset_simulation()
        self.status_message = message

    def _reset_simulation(self) -> None:
        self.sim_time = 0.0
        self.paused = False
        self.collision_rng = random.Random(self.solver_config.random_seed)
        self.blocked_counts.clear()
        self.pause_counts.clear()
        self.blocked_substeps = 0
        self.reserved_return_slots.clear()
        self._build_depot_return_topology()
        self.station_progress = self._build_station_progress()
        self.vehicles = self._build_vehicle_states()
        self._refresh_corridor_frontier()

    def _set_display_mode(self, fullscreen: bool) -> pygame.Surface:
        if fullscreen:
            info = pygame.display.Info()
            return pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
        return pygame.display.set_mode(
            (config.WINDOW_WIDTH_PX, config.WINDOW_HEIGHT_PX),
            pygame.RESIZABLE,
        )

    def _resize_window(self, width: int, height: int) -> None:
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self._rebuild_fonts()
        self._recompute_layout()

    def _rebuild_fonts(self) -> None:
        font_scale = max(1.0, self.screen.get_height() / 900.0)
        self.font = pygame.font.SysFont("Avenir Next, Helvetica, Arial", int(18 * font_scale))
        self.font_small = pygame.font.SysFont("Avenir Next, Helvetica, Arial", int(13 * font_scale))
        self.font_large = pygame.font.SysFont(
            "Avenir Next, Helvetica, Arial",
            int(26 * font_scale),
            bold=True,
        )

    def _recompute_layout(self) -> None:
        screen_w, screen_h = self.screen.get_size()
        padding = max(24, int(min(screen_w, screen_h) * 0.04))
        hud_width = min(max(250, int(screen_w * 0.23)), 340)
        world_w = max(300, screen_w - hud_width - (3 * padding))
        world_h = max(300, screen_h - (2 * padding))
        self.world_scale = min(world_w / config.WORLD_SIZE_IN, world_h / config.WORLD_SIZE_IN)
        self.world_width_px = config.WORLD_SIZE_IN * self.world_scale
        self.world_height_px = config.WORLD_SIZE_IN * self.world_scale
        self.world_left_px = padding
        self.world_top_px = (screen_h - self.world_height_px) / 2.0
        self.hud_left_px = self.world_left_px + self.world_width_px + padding
        self.hud_top_px = padding
        self.hud_width_px = screen_w - self.hud_left_px - padding
        self.hud_height_px = screen_h - (2 * padding)

    def _build_station_progress(self) -> Dict[int, Dict[str, float | None]]:
        return {
            station.station_id: {
                "dropped_at": None,
                "ready_at": None,
                "picked_at": None,
            }
            for station in self.instance.stations
        }

    def _build_vehicle_states(self) -> List[VehicleState]:
        depot_access = self.instance.world.coords[self.instance.depot_node]
        ordered_slots = sorted(
            self.instance.world.depot_slots,
            key=lambda slot: (euclidean(slot, depot_access), -slot[1], -slot[0]),
        )
        active_vehicle_ids = [
            vehicle_id for vehicle_id, route in enumerate(self.solution.routes) if route
        ]
        idle_vehicle_ids = [
            vehicle_id for vehicle_id, route in enumerate(self.solution.routes) if not route
        ]
        slot_by_vehicle = {
            vehicle_id: ordered_slots[index]
            for index, vehicle_id in enumerate(active_vehicle_ids + idle_vehicle_ids)
        }

        vehicles = []
        launch_order = {
            vehicle_id: index for index, vehicle_id in enumerate(active_vehicle_ids)
        }
        for vehicle_id in range(self.instance.vehicle_count):
            home_slot = slot_by_vehicle[vehicle_id]
            route = list(self.solution.routes[vehicle_id])
            return_slot_index = None
            if not route:
                return_slot_index = self.depot_return_slot_ranks[home_slot]
                self.reserved_return_slots.add(home_slot)
            vehicles.append(
                VehicleState(
                    vehicle_id=vehicle_id,
                    color=PALETTE[vehicle_id % len(PALETTE)],
                    home_slot=home_slot,
                    route=route,
                    current_node=self.instance.depot_node,
                    position=home_slot,
                    load=self.instance.capacity,
                    return_slot_index=return_slot_index,
                    departure_release_time=(
                        launch_order.get(vehicle_id, 0) * DEPARTURE_STAGGER_SECONDS
                    ),
                )
            )
        return vehicles

    def _build_depot_return_topology(self) -> None:
        self.depot_corridor_entries = {}
        self.depot_corridor_slots = {}
        self.slot_corridors = {}

        slots_by_x: Dict[float, List[Coord]] = {}
        slots_by_y: Dict[float, List[Coord]] = {}
        for slot in self.instance.world.depot_slots:
            slots_by_x.setdefault(round(slot[0], 6), []).append(slot)
            slots_by_y.setdefault(round(slot[1], 6), []).append(slot)

        for index, entry in enumerate(self.instance.world.depot_entries["top"]):
            corridor_id = f"top_{index}"
            ordered_slots = sorted(
                slots_by_x.get(round(entry[0], 6), []),
                key=lambda slot: (round(slot[1], 6), round(slot[0], 6)),
            )
            self.depot_corridor_entries[corridor_id] = entry
            self.depot_corridor_slots[corridor_id] = ordered_slots
            for slot in ordered_slots:
                self.slot_corridors.setdefault(slot, []).append(corridor_id)

        for index, entry in enumerate(self.instance.world.depot_entries["right"]):
            corridor_id = f"right_{index}"
            ordered_slots = sorted(
                slots_by_y.get(round(entry[1], 6), []),
                key=lambda slot: (round(slot[0], 6), round(slot[1], 6)),
            )
            self.depot_corridor_entries[corridor_id] = entry
            self.depot_corridor_slots[corridor_id] = ordered_slots
            for slot in ordered_slots:
                self.slot_corridors.setdefault(slot, []).append(corridor_id)

        self._refresh_corridor_frontier()

    def _debug_depot_message(self, message: str) -> None:
        if self.debug_depot:
            print(f"[depot {self.sim_time:6.2f}s] {message}")

    def _is_homebound(self, vehicle: VehicleState) -> bool:
        return vehicle.route_index >= len(vehicle.route) and not vehicle.completed

    def _refresh_corridor_frontier(self) -> None:
        self.corridor_next_slot = {}
        for corridor_id, slots in self.depot_corridor_slots.items():
            next_slot = None
            for slot in slots:
                if slot not in self.reserved_return_slots:
                    next_slot = slot
                    break
            self.corridor_next_slot[corridor_id] = next_slot

    def _assign_return_targets(self) -> None:
        active_corridors = {
            vehicle.depot_corridor
            for vehicle in self.vehicles
            if self._is_homebound(vehicle)
            and vehicle.return_slot_index is not None
            and vehicle.depot_corridor is not None
        }
        while True:
            assignment = self._next_return_assignment(active_corridors)
            if assignment is None:
                return
            vehicle, slot, corridor_id = assignment
            vehicle.home_slot = slot
            vehicle.return_slot_index = self.depot_return_slot_ranks[slot]
            vehicle.depot_entry = self.depot_corridor_entries[corridor_id]
            vehicle.depot_corridor = corridor_id
            self.reserved_return_slots.add(slot)
            active_corridors.add(corridor_id)
            self._debug_depot_message(
                f"assign V{vehicle.vehicle_id + 1} to {corridor_id} slot {vehicle.return_slot_index + 1} "
                f"at {tuple(round(value, 2) for value in slot)}"
            )
            self._refresh_corridor_frontier()
            continue

    def _next_return_assignment(
        self,
        active_corridors: Set[str | None],
    ) -> Tuple[VehicleState, Coord, str] | None:
        unassigned = [
            vehicle
            for vehicle in self.vehicles
            if self._is_homebound(vehicle) and vehicle.return_slot_index is None
        ]
        if not unassigned:
            return None

        for slot in self.depot_return_slots:
            if slot in self.reserved_return_slots:
                continue
            candidate_corridors = [
                corridor_id
                for corridor_id in self.slot_corridors.get(slot, [])
                if corridor_id not in active_corridors
                and self.corridor_next_slot.get(corridor_id) == slot
            ]
            if not candidate_corridors:
                # Transient unavailability (corridor active or filling out of order):
                # wait, preserving the left-first slot iteration.
                return None
            clear_corridors = [
                corridor_id
                for corridor_id in candidate_corridors
                if self._corridor_path_clear(corridor_id, slot)
            ]
            if not clear_corridors:
                # Persistent geometric block: skip this slot so other vehicles
                # aren't stuck behind it.
                continue
            vehicle = min(unassigned, key=self._return_assignment_order)
            corridor_id = min(
                clear_corridors,
                key=lambda candidate_corridor: self._return_candidate_cost(
                    vehicle,
                    candidate_corridor,
                    slot,
                ),
            )
            return vehicle, slot, corridor_id

        if self._next_return_slot() is None:
            raise RuntimeError("No available depot return slot.")
        return None

    def _corridor_path_clear(self, corridor_id: str, slot: Coord) -> bool:
        # Corridor slots are ordered deepest-first; the entry sits at the shallow
        # end. To reach `slot`, the vehicle drives past every shallower slot in
        # the corridor (those after `slot` in the ordering). If any of those are
        # already reserved by a different corridor's vehicle, the path is
        # geometrically blocked.
        slots_in_corridor = self.depot_corridor_slots.get(corridor_id, [])
        try:
            target_index = slots_in_corridor.index(slot)
        except ValueError:
            return False
        for blocking_slot in slots_in_corridor[target_index + 1:]:
            if blocking_slot in self.reserved_return_slots:
                return False
        return True

    def _next_return_slot(self) -> Coord | None:
        for slot in self.depot_return_slots:
            if slot not in self.reserved_return_slots:
                return slot
        return None

    def _return_assignment_order(self, vehicle: VehicleState) -> Tuple[float, int]:
        if vehicle.current_node == self.instance.depot_node:
            return 0.0, vehicle.vehicle_id
        return (
            polyline_length(self.solver.path_coords(vehicle.current_node, self.instance.depot_node)),
            vehicle.vehicle_id,
        )

    def _best_corridor_for_return_slot(self, vehicle: VehicleState, slot: Coord) -> str:
        ordered_candidates = [
            corridor_id
            for corridor_id in self.slot_corridors.get(slot, [])
            if self.corridor_next_slot.get(corridor_id) == slot
        ]
        clear_candidates = [
            corridor_id
            for corridor_id in ordered_candidates
            if self._corridor_path_clear(corridor_id, slot)
        ]
        corridors = clear_candidates or ordered_candidates or list(self.slot_corridors.get(slot, []))
        if not corridors:
            raise RuntimeError(f"No depot corridor reaches return slot {slot}.")
        return min(
            corridors,
            key=lambda corridor_id: self._return_candidate_cost(vehicle, corridor_id, slot),
        )

    def _return_candidate_cost(
        self,
        vehicle: VehicleState,
        corridor_id: str,
        slot: Coord,
    ) -> float:
        points = self._candidate_return_path(vehicle, self.depot_corridor_entries[corridor_id], corridor_id, slot)
        return polyline_length(points) + (self.depot_return_slot_ranks[slot] * 1e-6)

    def _candidate_return_path(
        self,
        vehicle: VehicleState,
        depot_entry: Coord,
        depot_corridor: str,
        slot: Coord,
    ) -> List[Coord]:
        if vehicle.current_node == self.instance.depot_node:
            return dedupe_points([vehicle.position, *self._depot_parking_points(vehicle.position, slot)])
        return dedupe_points(
            [
                vehicle.position,
                *self._return_to_depot_entry_coords(vehicle.current_node, depot_entry, depot_corridor),
                *self._depot_entry_to_slot(depot_entry, slot, depot_corridor),
            ]
        )

    def _step_simulation(self, delta_sim_time: float) -> None:
        remaining = delta_sim_time
        while remaining > 1e-9:
            step = min(0.02, remaining)
            self.sim_time += step
            self._prime_vehicle_targets()
            self._advance_vehicles(step)
            self._resolve_arrivals()
            remaining -= step

    def _prime_vehicle_targets(self) -> None:
        self._assign_return_targets()
        for vehicle in self.vehicles:
            if vehicle.completed:
                continue

            if vehicle.waiting_customer_id is not None:
                station_state = self.station_progress[vehicle.waiting_customer_id]
                ready_at = station_state["ready_at"]
                if ready_at is not None and self.sim_time >= ready_at:
                    station_state["picked_at"] = self.sim_time
                    vehicle.load = min(self.instance.capacity, vehicle.load + 1)
                    vehicle.waiting_customer_id = None
                    vehicle.route_index += 1
                else:
                    continue

            if vehicle.active_path is not None:
                continue

            if vehicle.route_index < len(vehicle.route):
                if (
                    vehicle.route_index == 0
                    and vehicle.current_node == self.instance.depot_node
                    and self.sim_time < vehicle.departure_release_time
                ):
                    continue
                if (
                    vehicle.route_index == 0
                    and vehicle.current_node == self.instance.depot_node
                    and self._depot_departure_control_zone_occupied(exclude_vehicle_id=vehicle.vehicle_id)
                ):
                    continue
                op = vehicle.route[vehicle.route_index]
                target_node = self.solver.customer_node(op.customer_id)
                vehicle.target_node = target_node
                points = self._travel_points(vehicle.position, vehicle.current_node, target_node, False)
                vehicle.active_path = build_path_state(points)
                if vehicle.active_path is None:
                    self._handle_arrival(vehicle)
                continue

            if vehicle.return_slot_index is None:
                self._assign_return_targets()
            if vehicle.return_slot_index is None:
                continue
            if (
                vehicle.current_node != self.instance.depot_node
                or euclidean(vehicle.position, vehicle.home_slot) > 1e-6
            ):
                vehicle.target_node = self.instance.depot_node
                if vehicle.current_node != self.instance.depot_node:
                    points = self._travel_points(
                        vehicle.position,
                        vehicle.current_node,
                        self.instance.depot_node,
                        True,
                        home_slot=vehicle.home_slot,
                        depot_entry=vehicle.depot_entry,
                        depot_corridor=vehicle.depot_corridor,
                    )
                else:
                    points = self._depot_parking_points(vehicle.position, vehicle.home_slot)
                vehicle.active_path = build_path_state(points)
                if vehicle.active_path is None:
                    if euclidean(vehicle.position, vehicle.home_slot) <= 1e-6:
                        vehicle.completed = True
                        vehicle.completion_time = self.sim_time
            else:
                vehicle.completed = True
                vehicle.completion_time = self.sim_time

    def _advance_vehicles(self, step: float) -> None:
        max_distance = self.instance.instance_config.alvik_speed_in_per_sec * step
        current_owners = self._current_resource_owners()
        current_positions = {
            vehicle.vehicle_id: vehicle.position
            for vehicle in self.vehicles
        }
        movable = [
            vehicle
            for vehicle in self.vehicles
            if not vehicle.completed and vehicle.waiting_customer_id is None and vehicle.active_path is not None
        ]
        intersection_winners = self._intersection_winners(movable, max_distance, current_owners)

        candidates: Dict[int, MovementCandidate] = {}
        for vehicle in movable:
            path = vehicle.active_path
            if path is None:
                continue
            if path.total_length - path.distance <= 1e-5:
                path.distance = path.total_length
                vehicle.position = path.position()
                self.blocked_counts[vehicle.vehicle_id] = 0
                continue
            travel = min(max_distance, path.total_length - path.distance)
            if travel <= 0:
                continue

            start_distance = path.distance
            allowed = self._max_resource_safe_travel(
                vehicle,
                travel,
                current_owners,
                intersection_winners,
            )
            if allowed <= 1e-6:
                self._record_pause(vehicle.vehicle_id)
                continue

            end_distance = start_distance + allowed
            candidates[vehicle.vehicle_id] = MovementCandidate(
                vehicle=vehicle,
                start_distance=start_distance,
                end_distance=end_distance,
                start_position=path.position_at(start_distance),
                end_position=path.position_at(end_distance),
                resources=self._resources_for_path_range(path, start_distance, end_distance),
            )

        safe_candidates = self._clearance_safe_candidates(candidates, current_positions)
        accepted_ids = self._select_movement_winners(
            safe_candidates,
            max_distance,
            current_owners,
            current_positions,
        )
        accepted_ids = self._filter_clearance_safe_acceptances(
            safe_candidates,
            current_positions,
            accepted_ids,
        )
        backoff_candidate = None
        rejected_ids = set(candidates) - accepted_ids
        if rejected_ids:
            backoff_candidate = self._select_backoff_candidate(
                [
                    vehicle
                    for vehicle in movable
                    if vehicle.vehicle_id in rejected_ids
                ],
                max_distance,
                current_owners,
                current_positions,
            )
            if backoff_candidate is not None and not self._candidate_clear_against_accepted_movements(
                backoff_candidate,
                safe_candidates,
                accepted_ids,
            ):
                backoff_candidate = None
        if not accepted_ids:
            if backoff_candidate is not None:
                path = backoff_candidate.vehicle.active_path
                if path is not None:
                    path.distance = backoff_candidate.end_distance
                    backoff_candidate.vehicle.position = path.position()
                    self.blocked_counts[backoff_candidate.vehicle.vehicle_id] = 0
                    self.blocked_substeps = 0
                return

        for candidate in safe_candidates.values():
            vehicle_id = candidate.vehicle.vehicle_id
            if vehicle_id not in accepted_ids:
                self._record_pause(vehicle_id)
                continue
            path = candidate.vehicle.active_path
            if path is None:
                continue
            path.distance = candidate.end_distance
            candidate.vehicle.position = path.position()
            self.blocked_counts[vehicle_id] = 0

        backoff_vehicle_id = None
        if backoff_candidate is not None and accepted_ids:
            path = backoff_candidate.vehicle.active_path
            if path is not None:
                backoff_vehicle_id = backoff_candidate.vehicle.vehicle_id
                path.distance = backoff_candidate.end_distance
                backoff_candidate.vehicle.position = path.position()
                self.blocked_counts[backoff_vehicle_id] = 0

        blocked_candidate_ids = set(candidates) - set(safe_candidates)
        for vehicle_id in blocked_candidate_ids:
            if vehicle_id == backoff_vehicle_id:
                continue
            self._record_pause(vehicle_id)

        moved_count = len(accepted_ids) + (1 if backoff_vehicle_id is not None else 0)
        if movable and moved_count == 0:
            self.blocked_substeps += 1
            if self.blocked_substeps >= 50:
                self.status_message = "Traffic blocked: no clearance-safe move available"
        else:
            self.blocked_substeps = 0
            if self.status_message.startswith("Traffic blocked"):
                self.status_message = ""

    def _max_resource_safe_travel(
        self,
        vehicle: VehicleState,
        requested_travel: float,
        current_owners: Dict[str, int],
        intersection_winners: Dict[str, int],
    ) -> float:
        path = vehicle.active_path
        if path is None:
            return 0.0

        start_distance = path.distance
        requested_end = min(path.total_length, start_distance + requested_travel)
        owned_now = self._vehicle_current_resources(vehicle)
        if self._resource_motion_allowed(
            vehicle.vehicle_id,
            path,
            start_distance,
            requested_end,
            owned_now,
            current_owners,
            intersection_winners,
        ):
            return requested_end - start_distance

        low = start_distance
        high = requested_end
        for _ in range(14):
            mid = (low + high) / 2.0
            if self._resource_motion_allowed(
                vehicle.vehicle_id,
                path,
                start_distance,
                mid,
                owned_now,
                current_owners,
                intersection_winners,
            ):
                low = mid
            else:
                high = mid
        return max(0.0, low - start_distance)

    def _resource_motion_allowed(
        self,
        vehicle_id: int,
        path: PathState,
        start_distance: float,
        end_distance: float,
        owned_now: Set[str],
        current_owners: Dict[str, int],
        intersection_winners: Dict[str, int],
    ) -> bool:
        resources = self._resources_for_path_range(path, start_distance, end_distance)
        for resource in resources:
            is_intersection = resource.startswith("intersection:")
            if is_intersection:
                owner = current_owners.get(resource)
                if owner is not None and owner != vehicle_id and resource not in owned_now:
                    return False
                winner = intersection_winners.get(resource)
                if winner is not None and winner != vehicle_id and resource not in owned_now:
                    return False
                continue
            if not self._is_exclusive_resource(resource):
                continue
            owner = current_owners.get(resource)
            if owner is not None and owner != vehicle_id and resource not in owned_now:
                return False
            winner = intersection_winners.get(resource)
            if winner is not None and winner != vehicle_id and resource not in owned_now:
                return False
        return True

    def _clearance_safe_candidates(
        self,
        candidates: Dict[int, MovementCandidate],
        current_positions: Dict[int, Coord],
    ) -> Dict[int, MovementCandidate]:
        safe: Dict[int, MovementCandidate] = {}
        for vehicle_id, candidate in candidates.items():
            clear = True
            for other_id, other_position in current_positions.items():
                if other_id == vehicle_id or other_id in candidates:
                    continue
                if self._movement_conflicts_with_stationary(candidate, other_position):
                    clear = False
                    break
            if clear:
                safe[vehicle_id] = candidate
        return safe

    def _select_movement_winners(
        self,
        candidates: Dict[int, MovementCandidate],
        max_distance: float,
        current_owners: Dict[str, int],
        current_positions: Dict[int, Coord],
    ) -> Set[int]:
        if not candidates:
            return set()

        all_candidates = list(candidates.values())
        conflict_graph: Dict[int, Set[int]] = {
            candidate.vehicle.vehicle_id: set()
            for candidate in all_candidates
        }
        for index, first in enumerate(all_candidates):
            first_id = first.vehicle.vehicle_id
            for second in all_candidates[index + 1 :]:
                second_id = second.vehicle.vehicle_id
                if self._movement_candidates_conflict(first, second):
                    conflict_graph[first_id].add(second_id)
                    conflict_graph[second_id].add(first_id)

        accepted: Set[int] = set()
        visited: Set[int] = set()
        for candidate in all_candidates:
            vehicle_id = candidate.vehicle.vehicle_id
            if vehicle_id in visited:
                continue

            component_ids = self._movement_conflict_component(vehicle_id, conflict_graph)
            visited.update(component_ids)
            component = [candidates[component_id] for component_id in sorted(component_ids)]
            if len(component) == 1:
                accepted.add(vehicle_id)
                continue

            if len(component) == 2:
                head_on_ids = self._try_resolve_head_on(
                    component,
                    candidates,
                    max_distance,
                    current_owners,
                    current_positions,
                )
                if head_on_ids is not None:
                    accepted.update(head_on_ids)
                    continue

            feasible_candidates = [
                component_candidate
                for component_candidate in component
                if self._candidate_clear_against_stationary_component(component_candidate, component)
            ]
            if feasible_candidates:
                winner = self._choose_candidate_winner(feasible_candidates)
                accepted.add(winner.vehicle.vehicle_id)

        return accepted

    def _try_resolve_head_on(
        self,
        component: List[MovementCandidate],
        candidates: Dict[int, MovementCandidate],
        max_distance: float,
        current_owners: Dict[str, int],
        current_positions: Dict[int, Coord],
    ) -> Set[int] | None:
        first, second = component
        first_vec = (
            first.end_position[0] - first.start_position[0],
            first.end_position[1] - first.start_position[1],
        )
        second_vec = (
            second.end_position[0] - second.start_position[0],
            second.end_position[1] - second.start_position[1],
        )
        dot = first_vec[0] * second_vec[0] + first_vec[1] * second_vec[1]
        # Resolve head-on (dot < 0) and perpendicular merges (dot ≈ 0) in one
        # substep. Same-direction conflicts (dot > 0) are following-distance
        # issues, not deadlocks — leave those to the regular backoff.
        if dot > 1e-9:
            return None

        clearance = self._minimum_vehicle_center_distance()
        if euclidean(first.start_position, second.start_position) > 2.0 * clearance:
            return None

        if self._movement_priority(first.vehicle) >= self._movement_priority(second.vehicle):
            winner, loser = first, second
        else:
            winner, loser = second, first

        reverse_candidate = self._build_reverse_candidate(
            loser.vehicle,
            max_distance,
            current_owners,
        )
        if reverse_candidate is None:
            return None

        if self._movement_candidates_conflict(reverse_candidate, winner):
            return None

        if not self._candidate_clear_against_nonaccepted_vehicles(
            reverse_candidate,
            current_positions,
            {winner.vehicle.vehicle_id, loser.vehicle.vehicle_id},
        ):
            return None

        candidates[loser.vehicle.vehicle_id] = reverse_candidate
        return {winner.vehicle.vehicle_id, loser.vehicle.vehicle_id}

    def _build_reverse_candidate(
        self,
        vehicle: VehicleState,
        max_distance: float,
        current_owners: Dict[str, int],
    ) -> MovementCandidate | None:
        path = vehicle.active_path
        if path is None or path.distance <= 1e-6:
            return None
        start_distance = path.distance
        end_distance = max(0.0, start_distance - max_distance)
        resources = self._resources_for_path_range(path, end_distance, start_distance)
        owned_now = self._vehicle_current_resources(vehicle)
        for resource in resources:
            if not self._is_exclusive_resource(resource):
                continue
            owner = current_owners.get(resource)
            if owner is not None and owner != vehicle.vehicle_id and resource not in owned_now:
                return None
        return MovementCandidate(
            vehicle=vehicle,
            start_distance=start_distance,
            end_distance=end_distance,
            start_position=path.position_at(start_distance),
            end_position=path.position_at(end_distance),
            resources=resources,
        )

    def _movement_conflict_component(
        self,
        start_vehicle_id: int,
        conflict_graph: Dict[int, Set[int]],
    ) -> Set[int]:
        component = set()
        stack = [start_vehicle_id]
        while stack:
            vehicle_id = stack.pop()
            if vehicle_id in component:
                continue
            component.add(vehicle_id)
            stack.extend(conflict_graph[vehicle_id] - component)
        return component

    def _movement_candidates_conflict(
        self,
        first: MovementCandidate,
        second: MovementCandidate,
    ) -> bool:
        if self._candidate_resources_conflict(first, second):
            return True
        return self._movements_violate_clearance(
            first.start_position,
            first.end_position,
            second.start_position,
            second.end_position,
            planning=False,
        )

    def _candidate_resources_conflict(
        self,
        first: MovementCandidate,
        second: MovementCandidate,
    ) -> bool:
        return any(
            self._is_exclusive_resource(resource)
            for resource in first.resources & second.resources
        )

    def _is_exclusive_resource(self, resource: str) -> bool:
        return resource.startswith(("station:", "depot-slot:"))

    def _candidate_clear_against_stationary_component(
        self,
        candidate: MovementCandidate,
        component: List[MovementCandidate],
    ) -> bool:
        for other in component:
            if other.vehicle.vehicle_id == candidate.vehicle.vehicle_id:
                continue
            if self._movement_conflicts_with_stationary(candidate, other.start_position, planning=False):
                return False
        return True

    def _candidate_clear_against_nonaccepted_vehicles(
        self,
        candidate: MovementCandidate,
        current_positions: Dict[int, Coord],
        accepted_ids: Set[int],
    ) -> bool:
        for other_id, other_position in current_positions.items():
            if other_id == candidate.vehicle.vehicle_id or other_id in accepted_ids:
                continue
            if self._movement_conflicts_with_stationary(candidate, other_position, planning=False):
                return False
        return True

    def _filter_clearance_safe_acceptances(
        self,
        candidates: Dict[int, MovementCandidate],
        current_positions: Dict[int, Coord],
        accepted_ids: Set[int],
    ) -> Set[int]:
        accepted = set(accepted_ids)
        changed = True
        while changed:
            changed = False
            for vehicle_id in list(accepted):
                if self._candidate_clear_against_nonaccepted_vehicles(
                    candidates[vehicle_id],
                    current_positions,
                    accepted,
                ):
                    continue
                accepted.remove(vehicle_id)
                changed = True
        return accepted

    def _candidate_clear_against_accepted_movements(
        self,
        candidate: MovementCandidate,
        candidates: Dict[int, MovementCandidate],
        accepted_ids: Set[int],
    ) -> bool:
        for accepted_id in accepted_ids:
            accepted = candidates.get(accepted_id)
            if accepted is None:
                continue
            if self._movement_candidates_conflict(candidate, accepted):
                return False
        return True

    def _select_backoff_candidate(
        self,
        movable: List[VehicleState],
        max_distance: float,
        current_owners: Dict[str, int],
        current_positions: Dict[int, Coord],
    ) -> MovementCandidate | None:
        candidates: List[MovementCandidate] = []
        for vehicle in movable:
            path = vehicle.active_path
            if path is None or path.distance <= 1e-6:
                continue
            start_distance = path.distance
            end_distance = max(0.0, start_distance - max_distance)
            resources = self._resources_for_path_range(path, end_distance, start_distance)
            owned_now = self._vehicle_current_resources(vehicle)
            if any(
                self._is_exclusive_resource(resource)
                and owner is not None
                and owner != vehicle.vehicle_id
                and resource not in owned_now
                for resource, owner in (
                    (resource, current_owners.get(resource))
                    for resource in resources
                )
            ):
                continue
            candidate = MovementCandidate(
                vehicle=vehicle,
                start_distance=start_distance,
                end_distance=end_distance,
                start_position=path.position_at(start_distance),
                end_position=path.position_at(end_distance),
                resources=resources,
            )
            if self._candidate_clear_against_nonaccepted_vehicles(
                candidate,
                current_positions,
                {vehicle.vehicle_id},
            ):
                candidates.append(candidate)
        if not candidates:
            return None
        return self._choose_candidate_winner(candidates)

    def _choose_candidate_winner(
        self,
        candidates: List[MovementCandidate],
    ) -> MovementCandidate:
        max_blocked = max(
            self.blocked_counts.get(candidate.vehicle.vehicle_id, 0)
            for candidate in candidates
        )
        tied = [
            candidate
            for candidate in candidates
            if self.blocked_counts.get(candidate.vehicle.vehicle_id, 0) == max_blocked
        ]
        return self.collision_rng.choice(tied)

    def _movement_conflicts_with_stationary(
        self,
        candidate: MovementCandidate,
        other_position: Coord,
        *,
        planning: bool = True,
    ) -> bool:
        return self._movements_violate_clearance(
            candidate.start_position,
            candidate.end_position,
            other_position,
            other_position,
            planning=planning,
        )

    def _movements_violate_clearance(
        self,
        first_start: Coord,
        first_end: Coord,
        second_start: Coord,
        second_end: Coord,
        *,
        planning: bool = True,
    ) -> bool:
        min_distance = self._minimum_vehicle_center_distance()
        start_distance = euclidean(first_start, second_start)
        closest_distance = moving_points_min_distance(
            first_start,
            first_end,
            second_start,
            second_end,
        )
        if start_distance < min_distance - 1e-6:
            end_distance = euclidean(first_end, second_end)
            return end_distance <= start_distance + 1e-6
        return closest_distance < min_distance - 1e-6

    def _minimum_vehicle_center_distance(self) -> float:
        # Runtime collision is based on the visible robot body. Keep only a tiny
        # numerical buffer so opposite-lane traffic is not blocked by invisible clearance.
        return config.ALVIK_SIZE_IN + COLLISION_CONTACT_BUFFER_IN

    def _record_pause(self, vehicle_id: int) -> None:
        self.blocked_counts[vehicle_id] = self.blocked_counts.get(vehicle_id, 0) + 1
        self.pause_counts[vehicle_id] = self.pause_counts.get(vehicle_id, 0) + 1

    def _current_resource_owners(self) -> Dict[str, int]:
        claims: Dict[str, List[VehicleState]] = {}
        for vehicle in self.vehicles:
            resources = self._vehicle_current_resources(vehicle)
            for resource in resources:
                claims.setdefault(resource, []).append(vehicle)

        owners: Dict[str, int] = {}
        for resource, vehicles in claims.items():
            owner = max(vehicles, key=self._movement_priority)
            owners[resource] = owner.vehicle_id
        return owners

    def _vehicle_current_resources(self, vehicle: VehicleState) -> Set[str]:
        resources = self._resources_for_position(vehicle.position)
        if vehicle.active_path is not None:
            distance = vehicle.active_path.distance
            start = max(0.0, distance - 1e-4)
            end = min(vehicle.active_path.total_length, distance + 1e-4)
            resources.update(self._resources_for_path_range(vehicle.active_path, start, end))
        return resources

    def _resources_for_path_range(
        self,
        path: PathState,
        start_distance: float,
        end_distance: float,
    ) -> Set[str]:
        start_distance = max(0.0, min(path.total_length, start_distance))
        end_distance = max(start_distance, min(path.total_length, end_distance))

        resources = set()
        resources.update(self._resources_for_position(path.position_at(start_distance)))
        resources.update(self._resources_for_position(path.position_at(end_distance)))

        cumulative = path_cumulative_lengths(path.points)
        for index, (start, end) in enumerate(zip(path.points, path.points[1:])):
            segment_start = cumulative[index]
            segment_end = cumulative[index + 1]
            if segment_end < start_distance - 1e-9 or segment_start > end_distance + 1e-9:
                continue
            if start_distance - 1e-9 <= segment_start <= end_distance + 1e-9:
                resources.update(self._resources_for_position(start))
            if start_distance - 1e-9 <= segment_end <= end_distance + 1e-9:
                resources.update(self._resources_for_position(end))
        return resources

    def _resources_for_position(self, position: Coord) -> Set[str]:
        resources = set()
        intersection = self._intersection_resource_for_point(position)
        if intersection is not None:
            resources.add(intersection)

        depot_slot = self._depot_slot_resource(position)
        if depot_slot is not None:
            resources.add(depot_slot)

        station = self._station_resource(position)
        if station is not None:
            resources.add(station)
        return resources

    def _intersection_resource_for_point(self, point: Coord) -> str | None:
        nearest_x = nearest_index(point[0], self.instance.world.road_xs)
        nearest_y = nearest_index(point[1], self.instance.world.road_ys)
        center = (
            self.instance.world.road_xs[nearest_x],
            self.instance.world.road_ys[nearest_y],
        )
        if euclidean(point, center) > self._intersection_control_radius():
            return None
        return f"intersection:{nearest_x}:{nearest_y}"

    def _intersection_control_radius(self) -> float:
        return config.ALVIK_SIZE_IN + config.MIN_ALVIK_CLEARANCE_IN

    def _depot_slot_resource(self, point: Coord) -> str | None:
        for index, slot in enumerate(self.depot_return_slots):
            if euclidean(point, slot) <= 1e-6:
                return f"depot-slot:{index}"
        return None

    def _station_resource(self, point: Coord) -> str | None:
        for station in self.instance.stations:
            if euclidean(point, station.coord) <= 1e-6:
                return f"station:{station.station_id}"
        return None

    def _intersection_winners(
        self,
        vehicles: List[VehicleState],
        max_distance: float,
        current_owners: Dict[str, int],
    ) -> Dict[str, int]:
        requests: Dict[str, List[Tuple[Tuple[int, int, int], int]]] = {}
        for vehicle in vehicles:
            path = vehicle.active_path
            if path is None:
                continue
            requested_travel = min(max_distance, path.total_length - path.distance)
            if requested_travel <= 0:
                continue
            start_distance = path.distance
            end_distance = start_distance + requested_travel
            owned_now = self._vehicle_current_resources(vehicle)
            resources = self._resources_for_path_range(path, start_distance, end_distance)
            for resource in resources:
                if not resource.startswith("intersection:") or resource in owned_now:
                    continue
                owner = current_owners.get(resource)
                if owner is not None and owner != vehicle.vehicle_id:
                    continue
                requests.setdefault(resource, []).append(
                    (
                        self._intersection_request_rank(vehicle, path, resource),
                        vehicle.vehicle_id,
                    )
                )

        winners = {}
        for resource, ranked_requests in requests.items():
            winners[resource] = min(ranked_requests)[1]
        return winners

    def _intersection_request_rank(
        self,
        vehicle: VehicleState,
        path: PathState,
        resource: str,
    ) -> Tuple[int, int, int]:
        center = self._intersection_center(resource)
        start_position = path.position_at(path.distance)
        dx = start_position[0] - center[0]
        dy = start_position[1] - center[1]
        if abs(dy) >= abs(dx):
            approach = "N" if dy >= 0 else "S"
        else:
            approach = "E" if dx >= 0 else "W"
        pause_count = max(
            self.pause_counts.get(vehicle.vehicle_id, 0),
            self.blocked_counts.get(vehicle.vehicle_id, 0),
        )
        return (-pause_count, INTERSECTION_PRIORITY[approach], vehicle.vehicle_id)

    def _intersection_center(self, resource: str) -> Coord:
        _, x_index, y_index = resource.split(":")
        return (
            self.instance.world.road_xs[int(x_index)],
            self.instance.world.road_ys[int(y_index)],
        )

    def _movement_priority(self, vehicle: VehicleState) -> Tuple[int, int, int, float, int, int]:
        return (
            self._depot_departure_phase(vehicle),
            self._station_exit_phase(vehicle),
            0 if self._is_homebound(vehicle) else 1,
            0.0 if vehicle.active_path is None else vehicle.active_path.distance,
            vehicle.route_index,
            -vehicle.vehicle_id,
        )

    def _station_exit_phase(self, vehicle: VehicleState) -> int:
        # A vehicle whose current_node is still a station/access point is mid-merge
        # onto the road. Give it priority over road traffic until it reaches the
        # next graph node (i.e., the lane-offset intersection).
        return 1 if vehicle.current_node.startswith("station") else 0

    def _depot_departure_phase(self, vehicle: VehicleState) -> int:
        if vehicle.current_node != self.instance.depot_node or vehicle.route_index >= len(vehicle.route):
            return 0
        depot_edge_x = self.instance.world.depot_entries["right"][0][0]
        road_lane_y = self.instance.world.road_ys[0]
        if vehicle.position[1] < road_lane_y - 1e-6:
            return 3
        if vehicle.position[0] < depot_edge_x - 1e-6:
            return 2
        return 1

    def _is_in_depot_control_zone(self, position: Coord) -> bool:
        depot_edge_x = self.instance.world.depot_entries["right"][0][0]
        road_lane_y = self.instance.world.road_ys[0]
        return position[0] <= depot_edge_x + 1e-6 and position[1] <= road_lane_y + 1e-6

    def _resolve_arrivals(self) -> None:
        for vehicle in self.vehicles:
            if vehicle.active_path is None or vehicle.completed:
                continue
            if vehicle.active_path.distance + 1e-5 < vehicle.active_path.total_length:
                continue
            self._handle_arrival(vehicle)

    def _handle_arrival(self, vehicle: VehicleState) -> None:
        target_node = vehicle.target_node
        vehicle.active_path = None
        if target_node is None:
            return

        vehicle.current_node = target_node
        if vehicle.route_index >= len(vehicle.route):
            vehicle.target_node = None
            if euclidean(vehicle.position, vehicle.home_slot) <= 1e-6:
                vehicle.completed = True
                vehicle.completion_time = self.sim_time
            return

        op = vehicle.route[vehicle.route_index]
        if target_node != self.solver.customer_node(op.customer_id):
            vehicle.target_node = None
            return

        if op.kind == "D":
            station_state = self.station_progress[op.customer_id]
            station_state["dropped_at"] = self.sim_time
            station_state["ready_at"] = self.sim_time + self.instance.processing_times[op.customer_id]
            vehicle.load = max(0, vehicle.load - 1)
            vehicle.route_index += 1
            vehicle.target_node = None
            return

        station_state = self.station_progress[op.customer_id]
        ready_at = station_state["ready_at"]
        if ready_at is not None and self.sim_time >= ready_at:
            station_state["picked_at"] = self.sim_time
            vehicle.load = min(self.instance.capacity, vehicle.load + 1)
            vehicle.route_index += 1
            vehicle.target_node = None
            return

        vehicle.waiting_customer_id = op.customer_id
        vehicle.target_node = None

    def _travel_points(
        self,
        current_position: Coord,
        current_node: str,
        target_node: str,
        returning_home: bool,
        home_slot: Coord | None = None,
        depot_entry: Coord | None = None,
        depot_corridor: str | None = None,
    ) -> List[Coord]:
        points = [current_position]
        depot_access = self.instance.world.coords[self.instance.depot_node]

        if (
            current_node == self.instance.depot_node
            and not returning_home
            and euclidean(current_position, depot_access) > 1e-6
        ):
            points.extend(self._depot_departure_points(current_position, depot_access))

        if returning_home and target_node == self.instance.depot_node:
            if depot_entry is None or depot_corridor is None:
                raise RuntimeError("Missing depot entry assignment for homebound travel.")
            if current_node != self.instance.depot_node:
                points.extend(self._return_to_depot_entry_coords(current_node, depot_entry, depot_corridor))
                if home_slot is not None:
                    points.extend(self._depot_entry_to_slot(depot_entry, home_slot, depot_corridor))
            elif home_slot is not None:
                points.extend(self._depot_parking_points(current_position, home_slot))
            return dedupe_points(points)

        network_points = self.solver.path_coords(current_node, target_node)
        if current_node == self.instance.depot_node:
            network_points = self._trim_depot_boundary_hop(network_points, from_start=True)
        if target_node == self.instance.depot_node:
            network_points = self._trim_depot_boundary_hop(network_points, from_start=False)
        points.extend(network_points)
        if returning_home and home_slot is not None:
            points.extend(self._depot_access_to_slot(depot_access, home_slot))
        return dedupe_points(points)

    def _depot_departure_points(self, slot: Coord, depot_access: Coord) -> List[Coord]:
        slot_x, slot_y = slot
        top_y = self.instance.world.depot_entries["top"][0][1]
        right_x = self.instance.world.depot_entries["right"][0][0]

        if self._uses_right_departure_corridor(slot):
            return dedupe_points([slot, (right_x, slot_y), depot_access])

        corner = self._depot_entry_corner()
        return dedupe_points([slot, (slot_x, top_y), corner, depot_access])

    def _depot_departure_control_zone_occupied(self, *, exclude_vehicle_id: int | None = None) -> bool:
        for other in self.vehicles:
            if other.vehicle_id == exclude_vehicle_id:
                continue
            if other.completed or other.active_path is None:
                continue
            if other.route_index != 0 or other.current_node != self.instance.depot_node:
                continue
            if self._is_in_depot_control_zone(other.position):
                return True
        return False

    def _uses_right_departure_corridor(self, slot: Coord) -> bool:
        slot_x, slot_y = slot
        top_row_y = max(y for _, y in self.instance.world.depot_slots)
        right_col_x = max(x for x, _ in self.instance.world.depot_slots)
        return (
            math.isclose(slot_x, right_col_x)
            or (slot_x > slot_y and not math.isclose(slot_y, top_row_y))
        )

    def _depot_parking_points(
        self,
        current_position: Coord,
        home_slot: Coord,
    ) -> List[Coord]:
        points = [current_position]
        if not math.isclose(current_position[0], home_slot[0]):
            points.append((home_slot[0], current_position[1]))
        if not (
            math.isclose(current_position[0], home_slot[0])
            and math.isclose(current_position[1], home_slot[1])
        ):
            points.append(home_slot)
        return dedupe_points(points)

    def _depot_access_to_slot(self, depot_access: Coord, slot: Coord) -> List[Coord]:
        slot_x, slot_y = slot
        _, access_y = depot_access
        points = []
        if not math.isclose(slot_x, depot_access[0]):
            points.append((slot_x, access_y))
        if not (math.isclose(slot_x, depot_access[0]) and math.isclose(slot_y, access_y)):
            points.append(slot)
        return points

    def _return_to_depot_entry_coords(
        self,
        current_node: str,
        depot_entry: Coord,
        depot_corridor: str,
    ) -> List[Coord]:
        points = self.solver.path_coords(current_node, self.instance.depot_node)
        points = self._trim_depot_boundary_hop(points, from_start=False)
        depot_access = self.instance.world.coords[self.instance.depot_node]
        if points and euclidean(points[-1], depot_access) <= 1e-6:
            points = points[:-1]
        if points and self._is_boundary_point(points[-1]):
            points = points[:-1]

        if depot_corridor.startswith("right"):
            corner = self._depot_entry_corner()
            if not points or euclidean(points[-1], corner) > 1e-6:
                points.append(corner)

        if not points or euclidean(points[-1], depot_entry) > 1e-6:
            points.append(depot_entry)
        return orthogonalize_points(dedupe_points(points))

    def _depot_entry_to_slot(
        self,
        depot_entry: Coord,
        slot: Coord,
        depot_corridor: str,
    ) -> List[Coord]:
        points = []
        if depot_corridor.startswith("top"):
            if not math.isclose(depot_entry[0], slot[0]):
                points.append((slot[0], depot_entry[1]))
        else:
            if not math.isclose(depot_entry[1], slot[1]):
                points.append((depot_entry[0], slot[1]))
        if euclidean(depot_entry, slot) > 1e-6:
            points.append(slot)
        return points

    def _depot_entry_corner(self) -> Coord:
        top_y = self.instance.world.depot_entries["top"][0][1]
        right_x = self.instance.world.depot_entries["right"][0][0]
        return (right_x, top_y)

    def _trim_depot_boundary_hop(
        self,
        points: List[Coord],
        *,
        from_start: bool,
    ) -> List[Coord]:
        if len(points) < 3:
            return points
        boundary_index = 1 if from_start else -2
        depot_index = 0 if from_start else -1
        boundary = points[boundary_index]
        depot_access = points[depot_index]
        if not self._is_boundary_point(boundary) or self._is_boundary_point(depot_access):
            return points
        trimmed = list(points)
        trimmed.pop(boundary_index)
        return trimmed

    def _is_boundary_point(self, point: Coord) -> bool:
        x, y = point
        return (
            math.isclose(x, 0.0)
            or math.isclose(y, 0.0)
            or math.isclose(x, config.WORLD_SIZE_IN)
            or math.isclose(y, config.WORLD_SIZE_IN)
        )

    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_world()
        self._draw_hud()
        pygame.display.flip()

    def _draw_world(self) -> None:
        world_rect = pygame.Rect(
            self.world_left_px,
            self.world_top_px,
            self.world_width_px,
            self.world_height_px,
        )
        pygame.draw.rect(self.screen, (239, 235, 225), world_rect)
        pygame.draw.rect(self.screen, TEXT_DARK, world_rect, width=2)

        for x in self.instance.world.road_xs:
            self._draw_vertical_road(x)
        for y in self.instance.world.road_ys:
            self._draw_horizontal_road(y)

        self._draw_stations()
        self._draw_vehicles()

    def _draw_vertical_road(self, x_in: float) -> None:
        center_x, _ = self._to_screen((x_in, 0.0))
        total_w = config.ROAD_ENVELOPE_WIDTH_IN * self.world_scale
        road_y = self.world_top_px
        road_h = self.world_height_px
        left_x = center_x - (total_w / 2.0)

        pygame.draw.rect(self.screen, ROAD_STRIP, (left_x, road_y, total_w, road_h), border_radius=4)

        divider_spans = [(0.0, config.WORLD_SIZE_IN)]
        half_envelope = config.ROAD_ENVELOPE_WIDTH_IN / 2.0
        for y_in in self.instance.world.road_ys:
            divider_spans = subtract_span_list(
                divider_spans,
                max(0.0, y_in - half_envelope),
                min(config.WORLD_SIZE_IN, y_in + half_envelope),
            )

        dash_len_px = config.LANE_DIVIDER_DASH_IN * self.world_scale
        gap_len_px = config.LANE_DIVIDER_GAP_IN * self.world_scale
        divider_width_px = max(1, int(round(config.LANE_DIVIDER_WIDTH_IN * self.world_scale)))
        for start_y, end_y in divider_spans:
            start_px = self._to_screen((x_in, start_y))
            end_px = self._to_screen((x_in, end_y))
            self._draw_dashed_line(
                start_px,
                end_px,
                config.LANE_DIVIDER_COLOR,
                dash_len_px,
                gap_len_px,
                divider_width_px,
            )

    def _draw_horizontal_road(self, y_in: float) -> None:
        _, center_y = self._to_screen((0.0, y_in))
        total_w = config.ROAD_ENVELOPE_WIDTH_IN * self.world_scale
        road_x = self.world_left_px
        road_w = self.world_width_px
        top_y = center_y - (total_w / 2.0)

        pygame.draw.rect(self.screen, ROAD_STRIP, (road_x, top_y, road_w, total_w), border_radius=4)

        divider_spans = [(0.0, config.WORLD_SIZE_IN)]
        half_envelope = config.ROAD_ENVELOPE_WIDTH_IN / 2.0
        for x_in in self.instance.world.road_xs:
            divider_spans = subtract_span_list(
                divider_spans,
                max(0.0, x_in - half_envelope),
                min(config.WORLD_SIZE_IN, x_in + half_envelope),
            )

        dash_len_px = config.LANE_DIVIDER_DASH_IN * self.world_scale
        gap_len_px = config.LANE_DIVIDER_GAP_IN * self.world_scale
        divider_width_px = max(1, int(round(config.LANE_DIVIDER_WIDTH_IN * self.world_scale)))
        for start_x, end_x in divider_spans:
            start_px = self._to_screen((start_x, y_in))
            end_px = self._to_screen((end_x, y_in))
            self._draw_dashed_line(
                start_px,
                end_px,
                config.LANE_DIVIDER_COLOR,
                dash_len_px,
                gap_len_px,
                divider_width_px,
            )

    def _draw_dashed_line(
        self,
        start_px: Coord,
        end_px: Coord,
        color: Tuple[int, int, int],
        dash_len: float,
        gap_len: float,
        width: int,
    ) -> None:
        if dash_len <= 0 or width <= 0:
            return

        dx = end_px[0] - start_px[0]
        dy = end_px[1] - start_px[1]
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return

        progress = 0.0
        cycle_len = dash_len + max(0.0, gap_len)
        while progress < length:
            segment_end = min(progress + dash_len, length)
            start_ratio = progress / length
            end_ratio = segment_end / length
            dash_start = (
                start_px[0] + (dx * start_ratio),
                start_px[1] + (dy * start_ratio),
            )
            dash_end = (
                start_px[0] + (dx * end_ratio),
                start_px[1] + (dy * end_ratio),
            )
            pygame.draw.line(
                self.screen,
                color,
                (round(dash_start[0]), round(dash_start[1])),
                (round(dash_end[0]), round(dash_end[1])),
                width=width,
            )
            progress += cycle_len

    def _draw_stations(self) -> None:
        gap_px = self.instance.world.road_gap_in * self.world_scale
        station_size = max(12, int(round(gap_px)) - 1)
        active_ids = set(self.instance.active_job_ids)

        for station in self.instance.stations:
            state, color = self._station_state(station.station_id)
            if station.station_id not in active_ids:
                state = "off"
                color = STATION_INACTIVE

            x_px, y_px = self._to_screen(station.coord)
            rect = pygame.Rect(0, 0, station_size, station_size)
            rect.center = (x_px, y_px)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, ROAD_EDGE, rect, width=1, border_radius=4)

            label_font = self._fit_font(station.name, station_size - 4, station_size - 4)
            label = label_font.render(station.name, True, TEXT_DARK)
            self.screen.blit(
                label,
                (
                    rect.centerx - (label.get_width() / 2),
                    rect.centery - (label.get_height() / 2),
                ),
            )

    def _draw_vehicles(self) -> None:
        alvik_px = config.ALVIK_SIZE_IN * self.world_scale
        shadow_offset = 3
        for vehicle in self.vehicles:
            x_px, y_px = self._to_screen(vehicle.position)
            rect = pygame.Rect(0, 0, alvik_px, alvik_px)
            rect.center = (x_px, y_px)

            pygame.draw.rect(
                self.screen,
                (120, 126, 132),
                rect.move(shadow_offset, shadow_offset),
                border_radius=8,
            )
            pygame.draw.rect(self.screen, vehicle.color, rect, border_radius=8)
            pygame.draw.rect(self.screen, ROAD_EDGE, rect, width=2, border_radius=8)

            marker_color = STATION_READY if vehicle.load_marker() > 0 else TEXT_LIGHT
            marker_radius = max(4, int(alvik_px * 0.12))
            pygame.draw.circle(
                self.screen,
                marker_color,
                (rect.right - marker_radius - 5, rect.top + marker_radius + 5),
                marker_radius,
            )
            vehicle_label = self.font_small.render(str(vehicle.vehicle_id + 1), True, TEXT_LIGHT)
            self.screen.blit(vehicle_label, (rect.x + 6, rect.y + 4))

    def _draw_hud(self) -> None:
        hud_rect = pygame.Rect(
            self.hud_left_px,
            self.hud_top_px,
            self.hud_width_px,
            self.hud_height_px,
        )
        pygame.draw.rect(self.screen, HUD_BG, hud_rect, border_radius=16)

        y = hud_rect.y + 24
        lines = [
            ("VRP-RPD", self.font_large),
            (
                f"Alvik speed: {self.instance.instance_config.alvik_speed_in_per_sec * 2.54:.1f} cm/s",
                self.font,
            ),
            (f"Active jobs: {len(self.instance.active_job_ids)} / {len(self.instance.stations)}", self.font),
            (f"Process scale: {self.processing_scale:.2f}x", self.font),
            ("", self.font),
            (f"Time: {self.sim_time:6.1f}s", self.font),
            (f"Completed jobs: {self._completed_jobs()}", self.font),
            (f"Active jobs now: {self._active_jobs()}", self.font),
            (f"Planned makespan: {self.solution.makespan:6.1f}s", self.font),
            (f"Sim completion: {self._sim_completion_text()}", self.font),
            (f"Cross-agent jobs: {self._cross_agent_jobs()}", self.font),
            ("", self.font),
            ("Controls", self.font),
            ("Drag slider for speed", self.font_small),
            ("Space pause, R reset", self.font_small),
            ("- = jobs down/up", self.font_small),
            (", . process time down/up", self.font_small),
            ("F or F11 fullscreen", self.font_small),
        ]
        for text, font in lines:
            if text:
                surface = font.render(text, True, TEXT_LIGHT)
                self.screen.blit(surface, (hud_rect.x + 20, y))
            y += 28 if font == self.font_large else 22

        slider_label = self.font.render(
            f"Playback Speed {self.sim_speed_multiplier:.2f}x",
            True,
            TEXT_LIGHT,
        )
        self.screen.blit(slider_label, (hud_rect.x + 20, y + 6))
        y += 40
        self._draw_speed_slider(hud_rect.x + 20, y, hud_rect.width - 40)
        y += 36

        if self.status_message:
            status = self.font_small.render(self.status_message, True, TEXT_LIGHT)
            self.screen.blit(status, (hud_rect.x + 20, y + 8))

    def _draw_speed_slider(self, x: float, y: float, width: float) -> None:
        track_rect = pygame.Rect(int(x), int(y), int(width), 8)
        self.speed_slider_rect = track_rect
        pygame.draw.rect(self.screen, SLIDER_TRACK, track_rect, border_radius=4)

        fraction = (self.sim_speed_multiplier - self.sim_speed_min) / (
            self.sim_speed_max - self.sim_speed_min
        )
        filled_w = max(0, min(track_rect.width, int(track_rect.width * fraction)))
        if filled_w > 0:
            pygame.draw.rect(
                self.screen,
                SLIDER_FILL,
                pygame.Rect(track_rect.x, track_rect.y, filled_w, track_rect.height),
                border_radius=4,
            )

        knob_x = track_rect.x + filled_w
        knob_y = track_rect.centery
        self.speed_knob_rect = pygame.Rect(0, 0, 16, 16)
        self.speed_knob_rect.center = (knob_x, knob_y)
        pygame.draw.circle(self.screen, SLIDER_KNOB, self.speed_knob_rect.center, 8)
        pygame.draw.circle(self.screen, ROAD_EDGE, self.speed_knob_rect.center, 8, width=2)

    def _station_state(self, customer_id: int) -> Tuple[str, Tuple[int, int, int]]:
        if customer_id not in self.instance.active_job_ids:
            return "off", STATION_INACTIVE
        progress = self.station_progress[customer_id]
        dropped_at = progress["dropped_at"]
        ready_at = progress["ready_at"]
        picked_at = progress["picked_at"]
        if dropped_at is None or self.sim_time < dropped_at:
            return "idle", STATION_IDLE
        if ready_at is not None and self.sim_time < ready_at:
            return "proc", STATION_PROCESSING
        if picked_at is None or self.sim_time < picked_at:
            return "ready", STATION_READY
        return "done", STATION_DONE

    def _completed_jobs(self) -> int:
        return sum(
            1
            for customer_id in self.instance.active_job_ids
            if self.station_progress[customer_id]["picked_at"] is not None
        )

    def _active_jobs(self) -> int:
        active = 0
        for customer_id in self.instance.active_job_ids:
            progress = self.station_progress[customer_id]
            if progress["dropped_at"] is not None and progress["picked_at"] is None:
                active += 1
        return active

    def _cross_agent_jobs(self) -> int:
        return sum(
            1
            for customer_id in self.solution.drop_vehicle
            if self.solution.drop_vehicle.get(customer_id) != self.solution.pickup_vehicle.get(customer_id)
        )

    def _sim_completion_text(self) -> str:
        if all(vehicle.completed for vehicle in self.vehicles):
            completion = max(
                (vehicle.completion_time or 0.0) for vehicle in self.vehicles
            )
            return f"{completion:6.1f}s"
        return "running"

    def _to_screen(self, coord: Coord) -> Tuple[float, float]:
        x_px = self.world_left_px + (coord[0] * self.world_scale)
        y_px = self.world_top_px + ((config.WORLD_SIZE_IN - coord[1]) * self.world_scale)
        return x_px, y_px

    def _fit_font(self, text: str, max_width: int, max_height: int) -> pygame.font.Font:
        size = min(self.font_small.get_height(), max_height)
        size = max(8, size)
        while size > 8:
            candidate = pygame.font.SysFont("Avenir Next, Helvetica, Arial", size)
            surface = candidate.render(text, True, TEXT_DARK)
            if surface.get_width() <= max_width and surface.get_height() <= max_height:
                return candidate
            size -= 1
        return pygame.font.SysFont("Avenir Next, Helvetica, Arial", 8)


def build_path_state(points: List[Coord]) -> PathState | None:
    if len(points) < 2:
        return None
    total_length = sum(euclidean(a, b) for a, b in zip(points, points[1:]))
    if total_length <= 1e-9:
        return None
    return PathState(points=points, total_length=total_length)


def interpolate_polyline(points: List[Coord], fraction: float) -> Coord:
    if not points:
        return (0.0, 0.0)
    if len(points) == 1:
        return points[0]
    fraction = max(0.0, min(1.0, fraction))
    lengths = [euclidean(a, b) for a, b in zip(points, points[1:])]
    total = sum(lengths)
    if total <= 0:
        return points[-1]
    target = total * fraction
    traversed = 0.0
    for index, seg_len in enumerate(lengths):
        if traversed + seg_len >= target:
            local = 0.0 if seg_len == 0 else (target - traversed) / seg_len
            ax, ay = points[index]
            bx, by = points[index + 1]
            return (ax + ((bx - ax) * local), ay + ((by - ay) * local))
        traversed += seg_len
    return points[-1]


def path_cumulative_lengths(points: List[Coord]) -> List[float]:
    cumulative = [0.0]
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += euclidean(start, end)
        cumulative.append(total)
    return cumulative


def dedupe_points(points: List[Coord]) -> List[Coord]:
    deduped: List[Coord] = []
    for point in points:
        if not deduped or not (
            math.isclose(point[0], deduped[-1][0]) and math.isclose(point[1], deduped[-1][1])
        ):
            deduped.append(point)
    return deduped


def subtract_span_list(
    spans: List[Tuple[float, float]],
    block_start: float,
    block_end: float,
) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for start, end in spans:
        if block_end <= start or block_start >= end:
            result.append((start, end))
            continue
        if block_start > start:
            result.append((start, block_start))
        if block_end < end:
            result.append((block_end, end))
    return result


def euclidean(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def moving_points_min_distance(
    first_start: Coord,
    first_end: Coord,
    second_start: Coord,
    second_end: Coord,
) -> float:
    rel_start = (
        first_start[0] - second_start[0],
        first_start[1] - second_start[1],
    )
    rel_velocity = (
        (first_end[0] - first_start[0]) - (second_end[0] - second_start[0]),
        (first_end[1] - first_start[1]) - (second_end[1] - second_start[1]),
    )
    velocity_len_sq = (rel_velocity[0] * rel_velocity[0]) + (rel_velocity[1] * rel_velocity[1])
    if velocity_len_sq <= 1e-12:
        return math.hypot(rel_start[0], rel_start[1])

    t = -(
        (rel_start[0] * rel_velocity[0]) + (rel_start[1] * rel_velocity[1])
    ) / velocity_len_sq
    t = max(0.0, min(1.0, t))
    closest = (
        rel_start[0] + (rel_velocity[0] * t),
        rel_start[1] + (rel_velocity[1] * t),
    )
    return math.hypot(closest[0], closest[1])


def polyline_length(points: List[Coord]) -> float:
    return sum(euclidean(a, b) for a, b in zip(points, points[1:]))


def nearest_index(value: float, candidates: List[float]) -> int:
    return min(range(len(candidates)), key=lambda index: abs(candidates[index] - value))
