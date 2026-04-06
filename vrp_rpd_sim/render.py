"""Pygame renderer for the VRP-RPD simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pygame

from . import config
from .model import Instance, Operation
from .solver import SolverRunResult, VRPRPDSolver
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

    def load_marker(self) -> int:
        return self.load


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
    ) -> None:
        pygame.init()
        pygame.display.set_caption(config.APP_TITLE)

        self.instance = instance
        self.solver = solver
        self.result = result
        self.solution = result.best

        self.fixed_processing_time = fixed_processing_time
        self.active_job_count = job_count if job_count is not None else len(instance.active_job_ids)
        self.processing_scale = (
            processing_scale if processing_scale is not None else config.PROCESSING_TIME_SCALE
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

        self.screen = self._set_display_mode(self.fullscreen)
        self.clock = pygame.time.Clock()
        self._rebuild_fonts()
        self._recompute_layout()

        self.station_progress = self._build_station_progress()
        self.vehicles = self._build_vehicle_states()

    def run(self) -> None:
        running = True
        while running:
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
        self.instance = build_instance(
            job_count=self.active_job_count,
            processing_scale=self.processing_scale,
            fixed_processing_time=self.fixed_processing_time,
        )
        self.solver = VRPRPDSolver(self.instance)
        self.result = self.solver.solve()
        self.solution = self.result.best
        self._reset_simulation()
        self.status_message = message

    def _reset_simulation(self) -> None:
        self.sim_time = 0.0
        self.paused = False
        self.station_progress = self._build_station_progress()
        self.vehicles = self._build_vehicle_states()

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
        for vehicle_id in range(self.instance.vehicle_count):
            home_slot = slot_by_vehicle[vehicle_id]
            vehicles.append(
                VehicleState(
                    vehicle_id=vehicle_id,
                    color=PALETTE[vehicle_id % len(PALETTE)],
                    home_slot=home_slot,
                    route=list(self.solution.routes[vehicle_id]),
                    current_node=self.instance.depot_node,
                    position=home_slot,
                    load=self.instance.capacity,
                )
            )
        return vehicles

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
                op = vehicle.route[vehicle.route_index]
                target_node = self.solver.customer_node(op.customer_id)
                vehicle.target_node = target_node
                points = self._travel_points(vehicle.position, vehicle.current_node, target_node, False)
                vehicle.active_path = build_path_state(points)
                if vehicle.active_path is None:
                    self._handle_arrival(vehicle)
                continue

            if (
                vehicle.current_node != self.instance.depot_node
                or euclidean(vehicle.position, vehicle.home_slot) > 1e-6
            ):
                if self._depot_return_in_progress(vehicle.vehicle_id):
                    continue
                vehicle.target_node = self.instance.depot_node
                points = self._travel_points(
                    vehicle.position,
                    vehicle.current_node,
                    self.instance.depot_node,
                    True,
                    home_slot=vehicle.home_slot,
                )
                vehicle.active_path = build_path_state(points)
                if vehicle.active_path is None:
                    vehicle.completed = True
                    vehicle.completion_time = self.sim_time
            else:
                vehicle.completed = True
                vehicle.completion_time = self.sim_time

    def _depot_return_in_progress(self, vehicle_id: int) -> bool:
        for other in self.vehicles:
            if other.vehicle_id == vehicle_id or other.completed:
                continue
            if other.target_node == self.instance.depot_node:
                return True
        return False

    def _advance_vehicles(self, step: float) -> None:
        max_distance = config.ALVIK_SPEED_IN_PER_SEC * step
        proposed_positions = {
            vehicle.vehicle_id: vehicle.position
            for vehicle in self.vehicles
            if not vehicle.completed
        }
        movable = [
            vehicle
            for vehicle in self.vehicles
            if not vehicle.completed and vehicle.waiting_customer_id is None and vehicle.active_path is not None
        ]
        movable.sort(
            key=lambda vehicle: (
                vehicle.route_index,
                0.0 if vehicle.active_path is None else vehicle.active_path.distance,
                -vehicle.vehicle_id,
            ),
            reverse=True,
        )

        for vehicle in movable:
            path = vehicle.active_path
            if path is None:
                continue
            travel = min(max_distance, path.total_length - path.distance)
            if travel <= 0:
                continue

            allowed = self._max_safe_travel(vehicle, travel, proposed_positions)
            if allowed <= 1e-5:
                continue
            path.distance += allowed
            vehicle.position = path.position()
            proposed_positions[vehicle.vehicle_id] = vehicle.position

    def _max_safe_travel(
        self,
        vehicle: VehicleState,
        travel: float,
        proposed_positions: Dict[int, Coord],
    ) -> float:
        path = vehicle.active_path
        if path is None:
            return 0.0

        low = 0.0
        high = travel
        if self._is_safe_position(
            vehicle.vehicle_id,
            path.position_at(path.distance + high),
            proposed_positions,
        ):
            return high

        for _ in range(14):
            mid = (low + high) / 2.0
            candidate = path.position_at(path.distance + mid)
            if self._is_safe_position(vehicle.vehicle_id, candidate, proposed_positions):
                low = mid
            else:
                high = mid
        return low

    def _is_safe_position(
        self,
        vehicle_id: int,
        candidate: Coord,
        proposed_positions: Dict[int, Coord],
    ) -> bool:
        min_box_spacing = config.ALVIK_SIZE_IN + config.MIN_ALVIK_CLEARANCE_IN
        for other_id, other_position in proposed_positions.items():
            if other_id == vehicle_id:
                continue
            if (
                abs(candidate[0] - other_position[0]) < min_box_spacing
                and abs(candidate[1] - other_position[1]) < min_box_spacing
            ):
                return False
        return True

    def _resolve_arrivals(self) -> None:
        for vehicle in self.vehicles:
            if vehicle.active_path is None or vehicle.completed:
                continue
            if vehicle.active_path.distance + 1e-9 < vehicle.active_path.total_length:
                continue
            self._handle_arrival(vehicle)

    def _handle_arrival(self, vehicle: VehicleState) -> None:
        target_node = vehicle.target_node
        vehicle.active_path = None
        if target_node is None:
            return

        vehicle.current_node = target_node
        if vehicle.route_index >= len(vehicle.route):
            vehicle.completed = True
            vehicle.completion_time = self.sim_time
            vehicle.target_node = None
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
    ) -> List[Coord]:
        points = [current_position]
        depot_access = self.instance.world.coords[self.instance.depot_node]

        if current_node == self.instance.depot_node and euclidean(current_position, depot_access) > 1e-6:
            points.extend(self._slot_to_depot_access(current_position, depot_access))

        if returning_home and target_node == self.instance.depot_node:
            network_points = self._natural_exit_coords(current_node)
        else:
            network_points = self.solver.path_coords(current_node, target_node)
        if current_node == self.instance.depot_node:
            network_points = self._trim_depot_boundary_hop(network_points, from_start=True)
        if target_node == self.instance.depot_node:
            network_points = self._trim_depot_boundary_hop(network_points, from_start=False)
        points.extend(network_points)

        if returning_home and home_slot is not None:
            points.extend(self._depot_access_to_slot(depot_access, home_slot))
        return dedupe_points(points)

    def _slot_to_depot_access(self, slot: Coord, depot_access: Coord) -> List[Coord]:
        slot_x, slot_y = slot
        access_x, access_y = depot_access
        points = []
        if not math.isclose(slot_y, access_y):
            points.append((slot_x, access_y))
        if not math.isclose(slot_x, access_x):
            points.append((access_x, access_y))
        return points

    def _depot_access_to_slot(self, depot_access: Coord, slot: Coord) -> List[Coord]:
        slot_x, slot_y = slot
        _, access_y = depot_access
        points = []
        if not math.isclose(slot_x, depot_access[0]):
            points.append((slot_x, access_y))
        if not (math.isclose(slot_x, depot_access[0]) and math.isclose(slot_y, access_y)):
            points.append(slot)
        return points

    def _station_for_node(self, node_id: str):
        for station in self.instance.stations:
            if station.node_id == node_id:
                return station
        return None

    def _exit_waypoint(self, node_id: str) -> str | None:
        station = self._station_for_node(node_id)
        if station is None:
            return None
        if station.side == "top":
            for ix, x in enumerate(self.instance.world.road_xs):
                if math.isclose(station.coord[0], x):
                    return f"i_{ix}_0"
        elif station.side == "right":
            for iy, y in enumerate(self.instance.world.road_ys):
                if math.isclose(station.coord[1], y):
                    return f"i_0_{iy}"
        return None

    def _natural_exit_coords(self, current_node: str) -> List[Coord]:
        waypoint = self._exit_waypoint(current_node)
        if waypoint is None:
            return self.solver.path_coords(current_node, self.instance.depot_node)
        path1 = self.instance.world.shortest_paths[(current_node, waypoint)]
        path2 = self.instance.world.shortest_paths[(waypoint, self.instance.depot_node)]
        combined = list(path1) + list(path2[1:])
        return self.solver.lane_polyline(combined)

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

        self._draw_depot()
        self._draw_stations()
        self._draw_vehicles()

    def _draw_vertical_road(self, x_in: float) -> None:
        center_x, _ = self._to_screen((x_in, 0.0))
        strip_w = config.ROAD_STRIP_WIDTH_IN * self.world_scale
        gap_w = config.ROAD_MEDIAN_GAP_IN * self.world_scale
        total_w = config.ROAD_ENVELOPE_WIDTH_IN * self.world_scale
        road_y = self.world_top_px
        road_h = self.world_height_px
        left_x = center_x - (total_w / 2.0)

        pygame.draw.rect(self.screen, ROAD_STRIP, (left_x, road_y, strip_w, road_h), border_radius=4)
        pygame.draw.rect(
            self.screen,
            ROAD_STRIP,
            (left_x + strip_w + gap_w, road_y, strip_w, road_h),
            border_radius=4,
        )

    def _draw_horizontal_road(self, y_in: float) -> None:
        _, center_y = self._to_screen((0.0, y_in))
        strip_w = config.ROAD_STRIP_WIDTH_IN * self.world_scale
        gap_w = config.ROAD_MEDIAN_GAP_IN * self.world_scale
        total_w = config.ROAD_ENVELOPE_WIDTH_IN * self.world_scale
        road_x = self.world_left_px
        road_w = self.world_width_px
        top_y = center_y - (total_w / 2.0)

        pygame.draw.rect(self.screen, ROAD_STRIP, (road_x, top_y, road_w, strip_w), border_radius=4)
        pygame.draw.rect(
            self.screen,
            ROAD_STRIP,
            (road_x, top_y + strip_w + gap_w, road_w, strip_w),
            border_radius=4,
        )

    def _draw_depot(self) -> None:
        slots = self.instance.world.depot_slots
        xs = [coord[0] for coord in slots]
        ys = [coord[1] for coord in slots]
        margin = 0.9
        left, top = self._to_screen((min(xs) - margin, max(ys) + margin))
        right, bottom = self._to_screen((max(xs) + margin, min(ys) - margin))
        rect = pygame.Rect(left, top, right - left, bottom - top)
        pygame.draw.rect(self.screen, DEPOT_BG, rect, border_radius=12)
        pygame.draw.rect(self.screen, DEPOT_EDGE, rect, width=2, border_radius=12)
        label = self.font.render("Depot", True, DEPOT_EDGE)
        self.screen.blit(label, (rect.x + 10, rect.y + 8))

    def _draw_stations(self) -> None:
        station_w = max(54, int(config.ALVIK_SIZE_IN * self.world_scale * 1.35))
        station_h = max(34, int(config.ALVIK_SIZE_IN * self.world_scale * 0.85))
        active_ids = set(self.instance.active_job_ids)

        for station in self.instance.stations:
            state, color = self._station_state(station.station_id)
            if station.station_id not in active_ids:
                state = "off"
                color = STATION_INACTIVE

            x_px, y_px = self._to_screen(station.coord)
            rect = pygame.Rect(0, 0, station_w, station_h)
            rect.center = (x_px, y_px)
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, ROAD_EDGE, rect, width=2, border_radius=8)

            label_font = self._fit_font(station.name, station_w - 8, max(8, (station_h // 2) - 6))
            status_font = self._fit_font(state, station_w - 8, max(8, (station_h // 2) - 6))
            label = label_font.render(station.name, True, TEXT_DARK)
            status = status_font.render(state, True, TEXT_DARK)
            self.screen.blit(
                label,
                (rect.centerx - (label.get_width() / 2), rect.y + 4),
            )
            self.screen.blit(
                status,
                (rect.centerx - (status.get_width() / 2), rect.bottom - status.get_height() - 4),
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
            (f"Alvik speed: {config.ALVIK_SPEED_IN_PER_SEC * 2.54:.1f} cm/s", self.font),
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
