"""Pygame renderer for the VRP-RPD simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Set, Tuple

import pygame

from . import cad_layout, config
from .mapf import MapfInfeasible, plan_space_time, validate as validate_schedule
from .model import Instance, NodeVisit, Operation, ScheduledSolution
from .solution_cache import solve_with_solution_cache
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
# Underwater background gradient (top = near surface, deep = sea floor) and
# the color distant Z-planes fade toward, so the cube recedes into the water.
WATER_TOP = (46, 124, 176)
WATER_DEEP = (5, 28, 62)
WATER_FADE = (16, 58, 104)
# Per-plane translucency so you can see through the stacked "glass" layers.
LAYER_BOARD_ALPHA = 92
LAYER_CELL_ALPHA = 52
HUD_BG = (35, 42, 52)
ROAD_STRIP = (92, 95, 99)
ROAD_GAP = (225, 216, 197)
ROAD_EDGE = (54, 60, 67)
# Fill for the white-square work cells. Matches the world background so the
# cells read as openings punched into the gray grid board.
CELL_FILL = (239, 235, 225)
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
    # MAPF-driven space-time schedule. `schedule_index` points at the current
    # NodeVisit; the vehicle dwells at `schedule[schedule_index].node_id`
    # until `t_exit` then transitions to `schedule[schedule_index+1]`.
    schedule: List[NodeVisit] = field(default_factory=list)
    schedule_index: int = 0
    # Track which special transitions have been built. The first
    # transition is home_slot -> depot (L-corridor egress); the last is
    # depot -> home_slot (L-corridor ingress).
    egress_built: bool = False
    ingress_built: bool = False
    # When an active_path is set, record sim_time at start and the planned
    # arrival sim_time at end. _advance_vehicles uses these to interpolate
    # position directly from sim_time rather than integrating step-by-step,
    # so MAPF schedule times are honored exactly (no quantization drift).
    path_start_time: float = 0.0
    path_arrival_time: float = 0.0
    # Unused after MAPF integration; retained so downstream visualization
    # (currently dead code) compiles.
    yield_reverse_substeps_remaining: int = 0

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
        scheduled: ScheduledSolution | None = None,
        sim_speed: float | None = None,
        job_count: int | None = None,
        processing_scale: float | None = None,
        fixed_processing_time: float | None = None,
        fullscreen: bool | None = None,
        debug_depot: bool = False,
        cache_config: config.CacheConfig | None = None,
        mapf_config: config.MapfConfig | None = None,
        headless: bool = False,
        record_unity_path: str | None = None,
        record_capture_interval_sec: float = 0.1,
        auto_quit_on_complete: bool = False,
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
        self.mapf_config = mapf_config or config.default_runtime_config().mapf
        self.result = result
        self.solution = result.best
        if scheduled is None:
            scheduled = plan_space_time(instance, solver, result.best, self.mapf_config)
            validate_schedule(scheduled, clearance_sec=self.mapf_config.clearance_sec)
        self.scheduled = scheduled

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

        # Isometric 3D view state. `iso_yaw` rotates the cube around its
        # vertical axis and `iso_pitch` tilts it (drag to orbit); `iso_zoom`
        # scales (mouse wheel); `active_layer` (None = show all) isolates one
        # Z-plane. Geometry filled in by _compute_iso_params.
        self.iso_yaw = 0.6
        self.iso_pitch = 0.5
        self.iso_zoom = 1.0
        self.active_layer: int | None = None
        self.iso_scale = 1.0
        self.iso_cx = 0.0
        self.iso_cy = 0.0
        # Drag-to-rotate + cached render surfaces (built in _recompute_layout).
        self.dragging_rotate = False
        self.last_drag_pos: Tuple[int, int] | None = None
        self._water_bg: "pygame.Surface | None" = None
        self._layer_overlay: "pygame.Surface | None" = None
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

        # Stuck detection while returning to depot. Tracks per-vehicle: time the
        # vehicle entered the homebound state, last sim_time it made forward
        # progress, and whether a stuck warning has already been printed (so we
        # warn once per stall, not every frame).
        self.homebound_since: Dict[int, float] = {}
        self.last_progress_time: Dict[int, float] = {}
        self.last_progress_distance: Dict[int, float] = {}
        self.last_progress_signal: Dict[int, Tuple] = {}
        self.stuck_warned: Dict[int, bool] = {}
        self.no_slot_warned: bool = False
        self.stuck_warn_threshold_sec: float = 4.0
        # Forward-progress watermarks: max path.distance reached + the time
        # it was reached, used by the deadlock recovery trigger.
        self._max_path_distance: Dict[int, float] = {}
        self._max_path_time: Dict[int, float] = {}

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

        # Recording: capture playback frames from a live pygame run, then write
        # the same JSON shape the headless exporter produces.
        self._record_unity_path = record_unity_path
        self._record_capture_interval = record_capture_interval_sec
        self._record_frames: list = []
        self._record_next_capture_time: float = record_capture_interval_sec
        self._auto_quit_on_complete = auto_quit_on_complete

    def run(self) -> None:
        if self.headless:
            raise RuntimeError("SimulationApp.run() is unavailable in headless mode.")
        if self._record_unity_path is not None:
            from .unity_export import capture_frame
            self._record_frames.append(capture_frame(self))

        # Fixed-timestep accumulator: the simulator advances in deterministic
        # 0.02 s chunks regardless of frame rate, so a given seed yields the
        # exact same outcome every run. Real-time playback speed is preserved
        # because we still scale wall-clock dt by sim_speed_multiplier.
        SIM_STEP = 0.02
        ACCUMULATOR_CAP = 0.5  # avoid spiral-of-death after a long pause/stall
        sim_accumulator = 0.0

        running = True
        recorded_complete = False
        while running:
            if self.clock is None:
                raise RuntimeError("Simulation clock was not initialized.")
            real_dt = self.clock.tick_busy_loop(config.FPS) / 1000.0
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
                    self.dragging_rotate = False
                    self.last_drag_pos = None
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_speed_slider:
                        self._update_speed_from_mouse(event.pos)
                    elif self.dragging_rotate:
                        self._rotate_from_drag(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    self._zoom_view(event.y)

            if not self.paused:
                sim_accumulator = min(
                    sim_accumulator + real_dt * self.sim_speed_multiplier,
                    ACCUMULATOR_CAP,
                )
                while sim_accumulator >= SIM_STEP:
                    self._step_simulation(SIM_STEP)
                    sim_accumulator -= SIM_STEP
                if self._record_unity_path is not None:
                    from .unity_export import capture_frame
                    while self.sim_time + 1e-9 >= self._record_next_capture_time:
                        self._record_frames.append(capture_frame(self))
                        self._record_next_capture_time += self._record_capture_interval
                    if (
                        self._auto_quit_on_complete
                        and all(v.completed for v in self.vehicles)
                    ):
                        recorded_complete = True
                        running = False
            self._draw()

        pygame.quit()

        if self._record_unity_path is not None:
            from .unity_export import capture_frame, build_payload_from_recording, write_payload
            # Final frame so the very last state is in the file.
            if not self._record_frames or self._record_frames[-1]["timeSec"] != self.sim_time:
                self._record_frames.append(capture_frame(self))
            simulation_completed = recorded_complete or all(v.completed for v in self.vehicles)
            payload = build_payload_from_recording(
                self,
                self._record_frames,
                capture_interval_sec=self._record_capture_interval,
                simulation_completed=simulation_completed,
                termination_reason=(
                    "" if simulation_completed
                    else "Recording stopped before all vehicles completed (window closed)."
                ),
            )
            path = write_payload(self._record_unity_path, payload)
            print(
                f"[recording] wrote {len(self._record_frames)} frames to {path} "
                f"(simulationCompleted={simulation_completed}, sim_time={self.sim_time:.2f}s)"
            )

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
            return
        # --- Isometric 3D view controls ---
        if event.key == pygame.K_LEFTBRACKET:
            self.iso_yaw -= math.radians(15)
            self._compute_iso_params()
            return
        if event.key == pygame.K_RIGHTBRACKET:
            self.iso_yaw += math.radians(15)
            self._compute_iso_params()
            return
        if event.key == pygame.K_UP:
            base = -1 if self.active_layer is None else self.active_layer
            self.active_layer = min(config.GRID_LAYERS - 1, base + 1)
            self.status_message = f"Showing Z-layer {self.active_layer}"
            return
        if event.key == pygame.K_DOWN:
            base = config.GRID_LAYERS if self.active_layer is None else self.active_layer
            self.active_layer = max(0, base - 1)
            self.status_message = f"Showing Z-layer {self.active_layer}"
            return
        if event.key == pygame.K_a:
            self.active_layer = None
            self.status_message = "Showing all Z-layers"
            return

    def _handle_mouse_down(self, pos: Tuple[int, int]) -> None:
        if self.speed_slider_rect.collidepoint(pos) or self.speed_knob_rect.collidepoint(pos):
            self.dragging_speed_slider = True
            self._update_speed_from_mouse(pos)
            return
        # Click anywhere in the world (left of the HUD) to grab and orbit the cube.
        if pos[0] < self.hud_left_px:
            self.dragging_rotate = True
            self.last_drag_pos = pos

    def _rotate_from_drag(self, pos: Tuple[int, int]) -> None:
        if self.last_drag_pos is None:
            self.last_drag_pos = pos
            return
        dx = pos[0] - self.last_drag_pos[0]
        dy = pos[1] - self.last_drag_pos[1]
        self.last_drag_pos = pos
        # Horizontal drag spins around the vertical axis; vertical drag tilts.
        self.iso_yaw += dx * 0.01
        self.iso_pitch = max(0.08, min(1.3, self.iso_pitch - dy * 0.004))
        self._compute_iso_params()

    def _zoom_view(self, wheel_y: float) -> None:
        self.iso_zoom = max(0.25, min(6.0, self.iso_zoom * (1.12 ** wheel_y)))

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
        try:
            self.scheduled = plan_space_time(
                self.instance, self.solver, self.solution, self.mapf_config
            )
            validate_schedule(self.scheduled, clearance_sec=self.mapf_config.clearance_sec)
        except MapfInfeasible as exc:
            self.status_message = f"{message}. MAPF infeasible: {exc}"
            return
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
        self.homebound_since.clear()
        self.last_progress_time.clear()
        self.last_progress_distance.clear()
        self.last_progress_signal.clear()
        self.stuck_warned.clear()
        self.no_slot_warned = False
        self._max_path_distance.clear()
        self._max_path_time.clear()
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
        self.world_scale = min(
            world_w / config.WORLD_WIDTH_IN,
            world_h / config.WORLD_HEIGHT_IN,
        )
        self.world_width_px = config.WORLD_WIDTH_IN * self.world_scale
        self.world_height_px = config.WORLD_HEIGHT_IN * self.world_scale
        self.world_left_px = padding
        self.world_top_px = (screen_h - self.world_height_px) / 2.0
        self.hud_left_px = self.world_left_px + self.world_width_px + padding
        self.hud_top_px = padding
        self.hud_width_px = screen_w - self.hud_left_px - padding
        self.hud_height_px = screen_h - (2 * padding)
        self._compute_iso_params()
        self._water_bg = self._build_water_background(screen_w, screen_h)
        self._layer_overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)

    def _build_water_background(self, width: int, height: int) -> "pygame.Surface":
        """Vertical gradient from WATER_TOP (surface) to WATER_DEEP (sea floor)."""
        surface = pygame.Surface((width, height))
        height = max(1, height)
        for y in range(height):
            t = y / (height - 1) if height > 1 else 0.0
            color = tuple(
                int(top + (deep - top) * t)
                for top, deep in zip(WATER_TOP, WATER_DEEP)
            )
            pygame.draw.line(surface, color, (0, y), (width, y))
        return surface

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
        # Slot 0 is the depot mouth (== i_7_7), so the vehicle parked there
        # physically blocks the corridor entrance. To make MAPF's
        # slowest-first priority order line up with the FIFO egress that
        # the L-corridor topology requires, assign slots in the same order:
        # slot 0 = slowest route (planned first by MAPF, leaves the mouth
        # first), slot 9 = fastest route (planned last, returns first).
        return_times = self.solution.return_times
        active_vehicle_ids = sorted(
            [vid for vid, route in enumerate(self.solution.routes) if route],
            key=lambda v: (-return_times[v], v),
        )
        idle_vehicle_ids = [
            vehicle_id for vehicle_id, route in enumerate(self.solution.routes) if not route
        ]
        slot_by_vehicle = {
            vehicle_id: ordered_slots[index]
            for index, vehicle_id in enumerate(active_vehicle_ids + idle_vehicle_ids)
        }
        # Dynamic ingress slot assignment: first vehicle to return parks at
        # the deepest active slot, second at the next-deepest, etc. This
        # guarantees parked vehicles never sit in a later-arriving vehicle's
        # corridor walk path. The original home_slot only determines the
        # EGRESS starting point; ingress is reassigned chronologically.
        active_slots = ordered_slots[:len(active_vehicle_ids)]
        self.ingress_slot_queue = list(reversed(active_slots))

        vehicles = []
        launch_order = {
            vehicle_id: index for index, vehicle_id in enumerate(active_vehicle_ids)
        }
        for vehicle_id in range(self.instance.vehicle_count):
            home_slot = slot_by_vehicle[vehicle_id]
            route = list(self.solution.routes[vehicle_id])
            schedule = list(self.scheduled.paths.get(vehicle_id, []))
            return_slot_index = self.depot_return_slot_ranks.get(home_slot)
            if return_slot_index is not None:
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
                    schedule=schedule,
                    schedule_index=0,
                    egress_built=False,
                    ingress_built=False,
                    completed=not schedule,
                )
            )
        return vehicles

    def _build_depot_return_topology(self) -> None:
        # The new CAD depot has a single L-shaped corridor with one entry at
        # the NE grid corner. Slots are listed deepest-first so the corridor
        # frontier picks the back of the queue first; that's the only safe
        # parking order when every robot enters and exits through the same
        # mouth.
        self.depot_corridor_entries = {}
        self.depot_corridor_slots = {}
        self.slot_corridors = {}

        main_entry = self.instance.world.depot_entries["main"][0]
        approach_order = list(self.instance.world.depot_slots)
        deepest_first = list(reversed(approach_order))

        corridor_id = "main"
        self.depot_corridor_entries[corridor_id] = main_entry
        self.depot_corridor_slots[corridor_id] = deepest_first
        for slot in deepest_first:
            self.slot_corridors.setdefault(slot, []).append(corridor_id)

        self._refresh_corridor_frontier()

    def _debug_depot_message(self, message: str) -> None:
        if self.debug_depot:
            print(f"[depot {self.sim_time:6.2f}s] {message}")

    def _warn_depot_message(self, message: str) -> None:
        # Stuck/anomaly diagnostics: print regardless of --debug-depot so the
        # user can see why a vehicle isn't completing its return.
        print(f"[depot WARN {self.sim_time:6.2f}s] {message}")

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
                self._log_unassigned_homebound(active_corridors)
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

    def _log_unassigned_homebound(self, active_corridors: Set[str | None]) -> None:
        unassigned = [
            vehicle
            for vehicle in self.vehicles
            if self._is_homebound(vehicle) and vehicle.return_slot_index is None
        ]
        if not unassigned:
            self.no_slot_warned = False
            return
        if self.no_slot_warned:
            return
        free_slots = [
            slot for slot in self.depot_return_slots
            if slot not in self.reserved_return_slots
        ]
        ids = ", ".join(f"V{v.vehicle_id + 1}" for v in unassigned)
        self._warn_depot_message(
            f"no return assignment available for {ids}: "
            f"{len(free_slots)}/{len(self.depot_return_slots)} slots free, "
            f"reserved={len(self.reserved_return_slots)}, active_corridors={sorted(c for c in active_corridors if c)}, "
            f"corridor_next_slot={ {cid: (None if s is None else tuple(round(v, 2) for v in s)) for cid, s in self.corridor_next_slot.items()} }"
        )
        self.no_slot_warned = True

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
                # No reachable corridor for this slot right now (every corridor
                # that lands here is already in use, or this isn't its frontier).
                # Skip and try a deeper slot — bailing out here would deadlock
                # any vehicle whose only reachable slot is past this one.
                continue
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

        # Cross-corridor path crossing: top corridors run down a column and
        # right corridors run along a row, so they share the corner cell where
        # column x equals row y. Reject this candidate if the slot or any cell
        # on the way to it is already on another active vehicle's swept path.
        candidate_cells = set(slots_in_corridor[target_index:])
        for other in self.vehicles:
            if other.completed:
                continue
            other_corridor = other.depot_corridor
            if other_corridor is None or other_corridor == corridor_id:
                continue
            if other.return_slot_index is None:
                continue
            other_cells = self._corridor_swept_cells(other_corridor, other.home_slot)
            if candidate_cells & other_cells:
                return False
        return True

    def _corridor_swept_cells(self, corridor_id: str, target_slot: Coord) -> Set[Coord]:
        """Depot-grid cells that a vehicle parking via (corridor, target_slot) occupies.

        Includes the target plus every shallower slot in the corridor (those
        the vehicle must drive past to reach `target_slot`).
        """
        slots_in_corridor = self.depot_corridor_slots.get(corridor_id, [])
        try:
            target_index = slots_in_corridor.index(target_slot)
        except ValueError:
            return {target_slot}
        return set(slots_in_corridor[target_index:])

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
            self._assert_no_physical_collisions()
            self._check_vehicle_stuck()
            remaining -= step

    def _check_vehicle_stuck(self) -> None:
        """Watchdog covering ALL vehicle states (mid-route, depot startup, return).

        Fires once per stall when a vehicle's progress signal hasn't changed
        for `stuck_warn_threshold_sec`. Dumps full diagnostic context: path
        progress, what op is next, who's nearby, who shares planned resources.
        Resets on progress, completion, or fresh path.
        """
        for vehicle in self.vehicles:
            vid = vehicle.vehicle_id
            if vehicle.completed:
                self._clear_stuck_state(vid)
                continue

            # Vehicle is legitimately waiting on a station's processing time —
            # not a deadlock, just timed wait. Skip.
            if vehicle.waiting_customer_id is not None:
                station_state = self.station_progress.get(vehicle.waiting_customer_id, {})
                ready_at = station_state.get("ready_at")
                if ready_at is None or self.sim_time < ready_at:
                    self._clear_stuck_state(vid)
                    continue

            # MAPF-planned wait: vehicle is sitting at home_slot before its
            # corridor egress window, or holding at a scheduled NodeVisit
            # dwell. Not a deadlock.
            if vehicle.schedule:
                idx = vehicle.schedule_index
                if 0 <= idx < len(vehicle.schedule):
                    current_visit = vehicle.schedule[idx]
                    if self.sim_time + 1e-9 < current_visit.t_exit:
                        self._clear_stuck_state(vid)
                        continue
                if idx == 0 and not vehicle.egress_built:
                    self._clear_stuck_state(vid)
                    continue

            signal = self._vehicle_progress_signal(vehicle)
            previous = self.last_progress_signal.get(vid) if hasattr(self, "last_progress_signal") else None
            if previous is None or signal != previous:
                self.last_progress_signal[vid] = signal
                self.last_progress_time[vid] = self.sim_time
                self.stuck_warned[vid] = False
                continue

            stalled_for = self.sim_time - self.last_progress_time.get(vid, self.sim_time)
            if stalled_for >= self.stuck_warn_threshold_sec and not self.stuck_warned.get(vid, False):
                self._dump_vehicle_stuck(vehicle, stalled_for)
                self.stuck_warned[vid] = True

    def _clear_stuck_state(self, vehicle_id: int) -> None:
        if hasattr(self, "last_progress_signal"):
            self.last_progress_signal.pop(vehicle_id, None)
        self.last_progress_time.pop(vehicle_id, None)
        self.last_progress_distance.pop(vehicle_id, None)
        self.stuck_warned.pop(vehicle_id, None)
        self.homebound_since.pop(vehicle_id, None)

    def _vehicle_progress_signal(self, vehicle: VehicleState) -> Tuple:
        # Tuple of state that, if unchanged across the threshold window, means
        # the vehicle has made no meaningful progress. Includes route_index so
        # arrivals/op-completions count as progress.
        path_distance = (
            round(vehicle.active_path.distance, 4)
            if vehicle.active_path is not None
            else None
        )
        return (
            vehicle.route_index,
            round(vehicle.position[0], 3),
            round(vehicle.position[1], 3),
            round(vehicle.position[2], 3),
            path_distance,
            vehicle.waiting_customer_id,
            vehicle.target_node,
            vehicle.depot_corridor,
            vehicle.return_slot_index,
        )

    def _dump_vehicle_stuck(self, vehicle: VehicleState, stalled_for: float) -> None:
        vid = vehicle.vehicle_id
        op_str = "RETURN" if vehicle.route_index >= len(vehicle.route) else (
            f"{vehicle.route[vehicle.route_index].kind}"
            f"{vehicle.route[vehicle.route_index].customer_id}"
        )
        path = vehicle.active_path
        if path is not None:
            path_info = (
                f"path={path.distance:.2f}/{path.total_length:.2f} "
                f"(remaining {path.total_length - path.distance:.2f})"
            )
        else:
            path_info = "no active_path"

        # Identify near neighbors and what they're doing.
        my_pos = vehicle.position
        nearby = []
        for other in self.vehicles:
            if other.vehicle_id == vid or other.completed:
                continue
            dist = euclidean(my_pos, other.position)
            if dist <= 12.0:  # ~3.4x alvik footprint
                nearby.append((dist, other))
        nearby.sort(key=lambda pair: pair[0])

        depot_extras = ""
        if self._is_homebound(vehicle):
            depot_extras = (
                f", home_slot={tuple(round(v, 2) for v in vehicle.home_slot)}"
                f", corridor={vehicle.depot_corridor}"
                f", slot_idx={vehicle.return_slot_index}"
            )

        self._warn_depot_message(
            f"V{vid + 1} STUCK {stalled_for:.1f}s "
            f"(blocked={self.blocked_counts.get(vid, 0)}, pause={self.pause_counts.get(vid, 0)}): "
            f"pos={tuple(round(v, 2) for v in my_pos)}, "
            f"node={vehicle.current_node}, target={vehicle.target_node}, "
            f"route_idx={vehicle.route_index}/{len(vehicle.route)}, next_op={op_str}, "
            f"waiting_cust={vehicle.waiting_customer_id}, "
            f"{path_info}"
            f"{depot_extras}"
        )
        for dist, other in nearby[:4]:
            other_op = "RETURN" if other.route_index >= len(other.route) else (
                f"{other.route[other.route_index].kind}"
                f"{other.route[other.route_index].customer_id}"
            )
            other_remaining = (
                f"{other.active_path.total_length - other.active_path.distance:.2f}"
                if other.active_path is not None
                else "no_path"
            )
            self._warn_depot_message(
                f"  near V{other.vehicle_id + 1} @ "
                f"{tuple(round(v, 2) for v in other.position)} "
                f"(dist {dist:.2f}, next_op={other_op}, "
                f"target={other.target_node}, path_remaining={other_remaining})"
            )

    def _prime_vehicle_targets(self) -> None:
        """Build the next polyline for each vehicle from its MAPF schedule.

        Each vehicle follows a list of NodeVisits produced by the offline
        MAPF planner. The first visit is the depot-egress window (vehicle
        walks the L-corridor from its home slot to the depot mouth). The
        last visit is the depot-ingress window (depot mouth to home
        slot). Intermediate visits are grid intersections; transitions
        between them are single-edge polylines traversed at constant
        speed.
        """
        speed = self.instance.instance_config.alvik_speed_in_per_sec
        depot_node = self.instance.depot_node
        for vehicle in self.vehicles:
            if vehicle.completed or vehicle.active_path is not None:
                continue
            if not vehicle.schedule:
                vehicle.completed = True
                vehicle.completion_time = self.sim_time
                continue

            idx = vehicle.schedule_index
            if idx >= len(vehicle.schedule):
                if euclidean(vehicle.position, vehicle.home_slot) <= 1e-6:
                    vehicle.completed = True
                    vehicle.completion_time = self.sim_time
                continue

            current = vehicle.schedule[idx]

            # Phase 1: depot egress (first visit). Vehicle starts at home_slot.
            if idx == 0 and not vehicle.egress_built:
                corridor_distance = self._corridor_distance(vehicle.home_slot)
                actual_corridor_time = corridor_distance / speed if speed > 0 else 0.0
                t_start_move = current.t_exit - actual_corridor_time
                if self.sim_time + 1e-9 < t_start_move:
                    continue  # still parked, waiting for egress window
                points = self._egress_polyline(vehicle.home_slot)
                vehicle.active_path = build_path_state(points)
                vehicle.target_node = depot_node
                vehicle.egress_built = True
                vehicle.path_start_time = t_start_move
                vehicle.path_arrival_time = current.t_exit
                if vehicle.active_path is None:
                    vehicle.position = self.instance.world.coords[depot_node]
                    vehicle.current_node = depot_node
                continue

            # Phase 3: depot ingress (last visit). Vehicle starts at depot.
            if idx == len(vehicle.schedule) - 1 and not vehicle.ingress_built:
                # Dynamic slot assignment: take the deepest still-available
                # slot. Order of arrival at the mouth determines parking
                # depth, so the corridor never has to be walked past a
                # parked Alvik.
                if self.ingress_slot_queue:
                    vehicle.home_slot = self.ingress_slot_queue.pop(0)
                corridor_distance = self._corridor_distance(vehicle.home_slot)
                actual_corridor_time = corridor_distance / speed if speed > 0 else 0.0
                points = self._ingress_polyline(vehicle.home_slot)
                vehicle.active_path = build_path_state(points)
                vehicle.target_node = depot_node
                vehicle.ingress_built = True
                vehicle.path_start_time = self.sim_time
                vehicle.path_arrival_time = self.sim_time + actual_corridor_time
                if vehicle.active_path is None:
                    vehicle.position = vehicle.home_slot
                    vehicle.completed = True
                    vehicle.completion_time = self.sim_time
                continue

            # Phase 2: mid-grid. Dwell until t_exit then traverse to next visit.
            if self.sim_time + 1e-9 < current.t_exit:
                # Pickup safety net: extra wait if station isn't ready yet
                # (MAPF assumed solver wait_time; reality may be later).
                if current.is_service and current.op_kind == "P" and current.customer_id is not None:
                    ready_at = self.station_progress.get(current.customer_id, {}).get("ready_at")
                    if ready_at is not None and self.sim_time < ready_at:
                        continue
                continue

            if idx + 1 >= len(vehicle.schedule):
                # End of schedule reached without ingress (idle/degenerate).
                vehicle.completed = True
                vehicle.completion_time = self.sim_time
                continue

            next_visit = vehicle.schedule[idx + 1]
            # Exit-effect for pickups completed at this visit.
            if current.is_service and current.op_kind == "P" and current.customer_id is not None:
                station_state = self.station_progress.get(current.customer_id)
                if station_state is not None and station_state.get("picked_at") is None:
                    station_state["picked_at"] = self.sim_time
                    vehicle.load = min(self.instance.capacity, vehicle.load + 1)
                    vehicle.route_index += 1
                vehicle.waiting_customer_id = None

            cur_coord = self.instance.world.coords[current.node_id]
            nxt_coord = self.instance.world.coords[next_visit.node_id]
            if euclidean(cur_coord, nxt_coord) <= 1e-6:
                # Zero-distance transition (e.g., depot ↔ i_7_7).
                vehicle.schedule_index += 1
                vehicle.current_node = next_visit.node_id
                vehicle.position = nxt_coord
                self._handle_visit_entry(vehicle, vehicle.schedule_index)
                continue

            points = dedupe_points([cur_coord, nxt_coord])
            vehicle.active_path = build_path_state(points)
            vehicle.target_node = next_visit.node_id
            # Pin traversal timing to MAPF schedule: vehicle leaves at
            # current.t_exit and arrives at next_visit.t_enter exactly,
            # so _advance_vehicles interpolates without integration drift.
            vehicle.path_start_time = current.t_exit
            vehicle.path_arrival_time = next_visit.t_enter

    def _corridor_distance(self, home_slot: Coord) -> float:
        depot_access = self.instance.world.coords[self.instance.depot_node]
        if euclidean(home_slot, depot_access) <= 1e-6:
            return 0.0
        # _l_path omits the leading start point, so prepend depot_access
        # to capture the depot→corner leg as well as corner→slot.
        return polyline_length([depot_access, *self._l_path(depot_access, home_slot)])

    def _egress_polyline(self, home_slot: Coord) -> List[Coord]:
        depot_access = self.instance.world.coords[self.instance.depot_node]
        if euclidean(home_slot, depot_access) <= 1e-6:
            return [depot_access]
        return dedupe_points([home_slot, *self._l_path(home_slot, depot_access)])

    def _ingress_polyline(self, home_slot: Coord) -> List[Coord]:
        depot_access = self.instance.world.coords[self.instance.depot_node]
        if euclidean(home_slot, depot_access) <= 1e-6:
            return [home_slot]
        return dedupe_points([depot_access, *self._l_path(depot_access, home_slot)])

    def _handle_visit_entry(self, vehicle: VehicleState, idx: int) -> None:
        """Apply station/load side-effects when entering a service visit."""
        if idx >= len(vehicle.schedule):
            return
        visit = vehicle.schedule[idx]
        if not visit.is_service or visit.customer_id is None:
            return
        station_state = self.station_progress.get(visit.customer_id)
        if station_state is None:
            return
        if visit.op_kind == "D":
            station_state["dropped_at"] = self.sim_time
            station_state["ready_at"] = (
                self.sim_time + self.instance.processing_times[visit.customer_id]
            )
            vehicle.load = max(0, vehicle.load - 1)
            vehicle.route_index += 1
        elif visit.op_kind == "P":
            # Mark waiting; the actual pickup completes when the vehicle
            # leaves this visit (see exit-effect in _prime_vehicle_targets).
            vehicle.waiting_customer_id = visit.customer_id

    def _advance_vehicles(self, step: float) -> None:
        """Schedule-pinned position update along the MAPF polyline.

        Position is computed from sim_time and the path's planned start/
        arrival times rather than by integrating velocity*dt. This
        eliminates sim_step quantization drift — at any sim_time t the
        vehicle is at exactly the position MAPF expects it at t, so
        zero-collision guarantees from the planner translate directly
        to zero physical collisions at runtime.
        """
        for vehicle in self.vehicles:
            if vehicle.completed or vehicle.active_path is None:
                continue
            path = vehicle.active_path
            t_start = vehicle.path_start_time
            t_arrival = vehicle.path_arrival_time
            duration = t_arrival - t_start
            if duration <= 1e-9:
                fraction = 1.0
            else:
                fraction = (self.sim_time - t_start) / duration
                fraction = max(0.0, min(1.0, fraction))
            path.distance = fraction * path.total_length
            vehicle.position = path.position()

    def _assert_no_physical_collisions(self) -> None:
        """Hard tripwire: any two active Alviks within a body-diameter
        of each other is a planner bug (model gap between the MAPF graph
        and physical layout). Halt the sim with diagnostics rather than
        silently rendering vehicles driving through each other.

        Center-to-center distance >= ALVIK_SIZE_IN guarantees the square
        footprints don't overlap, which is the zero-collision contract.
        """
        min_sep = config.ALVIK_SIZE_IN
        active = [v for v in self.vehicles if not v.completed]
        for i, va in enumerate(active):
            for vb in active[i + 1:]:
                # Full 3D separation — two Alviks stacked at the same (x, y)
                # but on different Z-planes are not in contact.
                dist = euclidean(va.position, vb.position)
                if dist + 1e-6 < min_sep:
                    raise RuntimeError(
                        f"COLLISION at sim_time={self.sim_time:.3f}s: "
                        f"V{va.vehicle_id} and V{vb.vehicle_id} are "
                        f"{dist:.2f}in apart (min {min_sep:.2f}in). "
                        f"V{va.vehicle_id}@{tuple(round(c, 2) for c in va.position)}, "
                        f"V{vb.vehicle_id}@{tuple(round(c, 2) for c in vb.position)}. "
                        f"MAPF schedule has a model gap — investigate."
                    )

    def _lookahead_yield_travel(
        self,
        vehicle: VehicleState,
        requested_travel: float,
        max_distance_per_substep: float,
    ) -> float:
        """Predictive yield: project this vehicle and every higher-priority
        vehicle along their paths over the next ~75 substeps (~1.5 sim-sec at
        full speed). If a clearance violation is predicted at any future step,
        return a smaller travel value that keeps the vehicle clear throughout
        the lookahead window. Higher-priority vehicles are unaffected because
        they don't yield — only the lower-priority side reduces speed.
        """
        path = vehicle.active_path
        if path is None or requested_travel <= 1e-6:
            return requested_travel
        if max_distance_per_substep <= 0:
            return requested_travel

        lookahead_substeps = 40  # ~0.8 sim-seconds at full speed
        my_priority = self._movement_priority(vehicle)
        # Use the actual vehicle clearance, no extra margin — extra margin
        # makes lower-priority vehicles yield too early, which causes new
        # cluster pile-ups behind them.
        min_clearance = self._minimum_vehicle_center_distance() * 1.05
        # Skip lookahead entirely when no nearby vehicle could possibly
        # conflict within the window — saves the expensive trajectory loop.
        max_relative_speed = max_distance_per_substep * lookahead_substeps * 2.0
        proximity_radius = max_relative_speed + min_clearance

        higher_priority_others: List[Tuple[VehicleState, List[Coord]]] = []
        for other in self.vehicles:
            if other.vehicle_id == vehicle.vehicle_id:
                continue
            if other.completed or other.active_path is None:
                continue
            if other.waiting_customer_id is not None:
                continue
            if euclidean(vehicle.position, other.position) > proximity_radius:
                continue
            if self._movement_priority(other) <= my_priority:
                continue
            other_path = other.active_path
            other_positions = []
            for step in range(1, lookahead_substeps + 1):
                d = min(
                    other_path.distance + max_distance_per_substep * step,
                    other_path.total_length,
                )
                other_positions.append(other_path.position_at(d))
            higher_priority_others.append((other, other_positions))

        if not higher_priority_others:
            return requested_travel

        start_d = path.distance
        path_total = path.total_length

        def keeps_clearance(this_step_travel: float) -> bool:
            base = start_d + this_step_travel
            for step in range(1, lookahead_substeps + 1):
                my_d = base if step == 1 else base + max_distance_per_substep * (step - 1)
                if my_d > path_total:
                    my_d = path_total
                my_pos = path.position_at(my_d)
                for _other, other_positions in higher_priority_others:
                    op = other_positions[step - 1]
                    dx = my_pos[0] - op[0]
                    dy = my_pos[1] - op[1]
                    if dx * dx + dy * dy < min_clearance * min_clearance:
                        return False
            return True

        if keeps_clearance(requested_travel):
            return requested_travel

        lo, hi = 0.0, requested_travel
        for _ in range(10):
            mid = (lo + hi) / 2.0
            if keeps_clearance(mid):
                lo = mid
            else:
                hi = mid
        return max(0.0, lo)

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
                continue

            # No feasible forward winner. For 3+ vehicle clearance deadlocks
            # (the kind that wedge near the depot exit or in tight station
            # clusters), try a coordinated multi-reverse: highest-priority
            # member moves forward, the rest reverse simultaneously, ignoring
            # resource conflicts between cluster members.
            if len(component) >= 3:
                cluster_ids = self._try_resolve_cluster_deadlock(
                    component,
                    candidates,
                    max_distance,
                    current_owners,
                    current_positions,
                )
                if cluster_ids is not None:
                    accepted.update(cluster_ids)

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

    def _try_resolve_cluster_deadlock(
        self,
        component: List[MovementCandidate],
        candidates: Dict[int, MovementCandidate],
        max_distance: float,
        current_owners: Dict[str, int],
        current_positions: Dict[int, Coord],
    ) -> Set[int] | None:
        """N-way clearance-deadlock breaker.

        Picks one cluster member as forward winner; everyone else reverses
        simultaneously. Treat *intra-cluster* resource ownership as relaxed
        (since the other owners are also stepping out of the way), but still
        require all movements to be mutually conflict-free and clear against
        stationary vehicles outside the cluster.
        """
        cluster_ids = {c.vehicle.vehicle_id for c in component}
        ordered = sorted(
            component,
            key=lambda c: self._movement_priority(c.vehicle),
            reverse=True,
        )

        for proposed_winner in ordered:
            winner_id = proposed_winner.vehicle.vehicle_id
            losers = [c for c in component if c.vehicle.vehicle_id != winner_id]

            reverses: Dict[int, MovementCandidate] = {}
            ok = True
            for loser in losers:
                reverse = self._build_relaxed_reverse_candidate(
                    loser.vehicle,
                    max_distance,
                    current_owners,
                    relaxed_against=cluster_ids,
                )
                if reverse is None:
                    ok = False
                    break
                reverses[loser.vehicle.vehicle_id] = reverse
            if not ok:
                continue

            # Winner forward must not collide with any reverse end-position.
            if any(
                self._movement_candidates_conflict(proposed_winner, rev)
                for rev in reverses.values()
            ):
                continue

            # Reverses must not collide with each other.
            reverse_list = list(reverses.values())
            mutually_clear = True
            for i, a in enumerate(reverse_list):
                for b in reverse_list[i + 1:]:
                    if self._movement_candidates_conflict(a, b):
                        mutually_clear = False
                        break
                if not mutually_clear:
                    break
            if not mutually_clear:
                continue

            # Reverses must clear stationary vehicles outside the cluster.
            outside_clear = True
            for rev in reverses.values():
                if not self._candidate_clear_against_nonaccepted_vehicles(
                    rev, current_positions, cluster_ids
                ):
                    outside_clear = False
                    break
            if not outside_clear:
                continue

            # All checks pass — apply the reverses and accept.
            for vid, rev in reverses.items():
                candidates[vid] = rev
            return cluster_ids
        return None

    def _force_cluster_reverse(
        self,
        movable: List[VehicleState],
        max_distance: float,
        current_owners: Dict[str, int],
        current_positions: Dict[int, Coord],
        *,
        anchor: VehicleState | None = None,
    ) -> bool:
        """Last-resort deadlock breaker.

        Many of our deadlocks happen *before* vehicles ever become movement
        candidates: each vehicle's `_max_resource_safe_travel` returns 0
        because some resource on its next path segment is owned by another
        stuck vehicle. The conflict graph never sees them, so the regular
        backoff/cluster resolvers can't help.

        Groups the stuck movable vehicles by spatial proximity, picks the
        cluster containing `anchor` if given (else the most-blocked cluster),
        and forces every member to reverse simultaneously, ignoring
        intra-cluster resource ownership. After applying, resource ownership
        cycles are broken and forward motion resumes next substep.
        """
        cluster_radius = 12.0
        clusters = self._build_proximity_clusters(movable, cluster_radius)
        if anchor is not None:
            clusters = [
                cluster for cluster in clusters
                if any(v.vehicle_id == anchor.vehicle_id for v in cluster)
            ]
        else:
            clusters.sort(
                key=lambda cluster: -sum(self.blocked_counts.get(v.vehicle_id, 0) for v in cluster),
            )

        # Reverse distance must exceed the inter-vehicle clearance — anything
        # smaller leaves the cluster geometry essentially unchanged. Reverse
        # by ~2x clearance (roughly 7 inches) so the released resources truly
        # become free for the stayer to advance into next substep.
        reverse_distance = max_distance + 2.0 * self._minimum_vehicle_center_distance()
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            cluster_ids = {v.vehicle_id for v in cluster}

            # Pick highest-priority member as the "stayer" — they keep their
            # position so they have a chance to advance next substep once the
            # others have cleared. Reversing every member makes the cluster
            # oscillate forever (everyone retreats together, same conflict).
            ordered = sorted(
                cluster,
                key=lambda v: self._movement_priority(v),
                reverse=True,
            )
            stayer = ordered[0]
            losers = ordered[1:]

            reverses: Dict[int, MovementCandidate] = {}
            for vehicle in losers:
                rev = self._build_relaxed_reverse_candidate(
                    vehicle,
                    reverse_distance,
                    current_owners,
                    relaxed_against=cluster_ids,
                )
                if rev is not None:
                    reverses[vehicle.vehicle_id] = rev
            if not reverses:
                self._warn_depot_message(
                    f"force_reverse skipped cluster "
                    f"{sorted(v.vehicle_id + 1 for v in cluster)} "
                    f"(stayer V{stayer.vehicle_id + 1}): "
                    f"no losers could build a reverse candidate"
                )
                continue

            rev_list = list(reverses.values())
            mutually_clear = True
            conflict_pair = None
            for i, a in enumerate(rev_list):
                for b in rev_list[i + 1:]:
                    if self._movement_candidates_conflict(a, b):
                        mutually_clear = False
                        conflict_pair = (a.vehicle.vehicle_id + 1, b.vehicle.vehicle_id + 1)
                        break
                if not mutually_clear:
                    break
            if not mutually_clear:
                self._warn_depot_message(
                    f"force_reverse rejected cluster "
                    f"{sorted(v.vehicle_id + 1 for v in cluster)}: "
                    f"reverse paths conflict between V{conflict_pair[0]} and V{conflict_pair[1]}"
                )
                continue

            outside_failure = None
            for rev in reverses.values():
                if not self._candidate_clear_against_nonaccepted_vehicles(
                    rev, current_positions, cluster_ids
                ):
                    outside_failure = rev.vehicle.vehicle_id + 1
                    break
            if outside_failure is not None:
                self._warn_depot_message(
                    f"force_reverse rejected cluster "
                    f"{sorted(v.vehicle_id + 1 for v in cluster)}: "
                    f"V{outside_failure}'s reverse conflicts with a vehicle outside the cluster"
                )
                continue

            # Convert the planned reverse into a per-vehicle counter so the
            # retreat plays out gradually at normal speed (~max_distance per
            # substep) instead of teleporting in one frame. Other vehicles
            # see the cluster member moving backward over many frames; the
            # released forward resources clear progressively, breaking the
            # deadlock without any visual jump.
            substeps_per_reverse = max(
                1,
                int(round(reverse_distance / max(max_distance, 1e-6))),
            )
            self._warn_depot_message(
                f"force_reverse SCHEDULED on cluster "
                f"{sorted(v.vehicle_id + 1 for v in cluster)} "
                f"(stayer V{stayer.vehicle_id + 1}, "
                f"losers V{[v.vehicle_id + 1 for v in losers]} "
                f"reverse over {substeps_per_reverse} substeps)"
            )
            for vid in reverses.keys():
                vehicle_obj = next(v for v in losers if v.vehicle_id == vid)
                vehicle_obj.yield_reverse_substeps_remaining = substeps_per_reverse
                self.blocked_counts[vid] = 0
            return True
        return False

    def _build_proximity_clusters(
        self,
        vehicles: List[VehicleState],
        radius: float,
    ) -> List[List[VehicleState]]:
        visited: Set[int] = set()
        clusters: List[List[VehicleState]] = []
        for vehicle in vehicles:
            if vehicle.vehicle_id in visited:
                continue
            cluster: List[VehicleState] = []
            stack = [vehicle]
            while stack:
                current = stack.pop()
                if current.vehicle_id in visited:
                    continue
                visited.add(current.vehicle_id)
                cluster.append(current)
                for other in vehicles:
                    if other.vehicle_id in visited:
                        continue
                    if euclidean(current.position, other.position) <= radius:
                        stack.append(other)
            clusters.append(cluster)
        return clusters

    def _build_relaxed_reverse_candidate(
        self,
        vehicle: VehicleState,
        max_distance: float,
        current_owners: Dict[str, int],
        *,
        relaxed_against: Set[int],
    ) -> MovementCandidate | None:
        """Like _build_reverse_candidate, but ignores resource ownership held
        by vehicles in `relaxed_against` (the rest of the deadlocked cluster)."""
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
            if (
                owner is not None
                and owner != vehicle.vehicle_id
                and owner not in relaxed_against
                and resource not in owned_now
            ):
                return None
        return MovementCandidate(
            vehicle=vehicle,
            start_distance=start_distance,
            end_distance=end_distance,
            start_position=path.position_at(start_distance),
            end_position=path.position_at(end_distance),
            resources=resources,
        )

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
        entry = self.instance.world.depot_entries["main"][0]
        arm_y = cad_layout.DEPOT_ARM_Y_IN
        if vehicle.position[1] <= entry[1] + 1e-6:
            # Vehicle has reached (or passed) the entry — it's on the grid.
            return 3
        on_arm = (
            math.isclose(vehicle.position[1], arm_y, abs_tol=1e-6)
            and abs(vehicle.position[0] - entry[0]) > 1e-6
        )
        if on_arm:
            return 1  # still on the horizontal arm
        return 2  # on the vertical column heading toward the entry

    def _is_in_depot_control_zone(self, position: Coord) -> bool:
        # The depot's exclusive zone is the L corridor itself — the vertical
        # column above the entry, plus the horizontal arm. The entry point
        # at the NE grid corner is the boundary, not part of the zone, so a
        # vehicle that has reached (69.16, 69.16) is considered to have
        # exited the depot. This is critical: a too-broad zone (e.g., the
        # entire top row of the grid) deadlocks all queued vehicles while
        # the leading vehicle traverses along y = entry.y.
        entry = self.instance.world.depot_entries["main"][0]
        arm_y = cad_layout.DEPOT_ARM_Y_IN
        epsilon = 1e-3
        on_column = (
            abs(position[0] - entry[0]) < epsilon
            and position[1] > entry[1] + epsilon
        )
        on_arm = (
            abs(position[1] - arm_y) < epsilon
            and position[0] < entry[0] + epsilon
        )
        return on_column or on_arm

    def _resolve_arrivals(self) -> None:
        """Advance schedule_index for vehicles that finished their polyline.

        Vehicles fall into three completion cases:
        - Just finished the depot egress (idx == 0): now at depot mouth;
          advance schedule_index to land on the first grid visit.
        - Just finished the depot ingress (idx == len-1): now at home
          slot; mark vehicle completed.
        - Just finished a normal grid-edge polyline: advance to the
          next NodeVisit and apply any entry-side service effects.
        """
        depot_node = self.instance.depot_node
        for vehicle in self.vehicles:
            if vehicle.completed or vehicle.active_path is None:
                continue
            path = vehicle.active_path
            if path.distance + 1e-5 < path.total_length:
                continue

            vehicle.active_path = None
            idx = vehicle.schedule_index

            if not vehicle.schedule:
                vehicle.completed = True
                vehicle.completion_time = self.sim_time
                continue

            if idx == 0 and vehicle.egress_built:
                # Arrived at depot mouth after L-corridor walk.
                vehicle.position = self.instance.world.coords[depot_node]
                vehicle.current_node = depot_node
                vehicle.schedule_index = min(idx + 1, len(vehicle.schedule) - 1)
                next_visit = vehicle.schedule[vehicle.schedule_index]
                vehicle.position = self.instance.world.coords[next_visit.node_id]
                vehicle.current_node = next_visit.node_id
                self._handle_visit_entry(vehicle, vehicle.schedule_index)
                continue

            if idx == len(vehicle.schedule) - 1 and vehicle.ingress_built:
                # Arrived at home slot after L-corridor walk.
                vehicle.position = vehicle.home_slot
                vehicle.completed = True
                vehicle.completion_time = self.sim_time
                continue

            # Normal grid-edge arrival: advance index, snap position to
            # the next visit's node coords, apply service effects.
            if idx + 1 < len(vehicle.schedule):
                vehicle.schedule_index = idx + 1
                next_visit = vehicle.schedule[vehicle.schedule_index]
                vehicle.position = self.instance.world.coords[next_visit.node_id]
                vehicle.current_node = next_visit.node_id
                self._handle_visit_entry(vehicle, vehicle.schedule_index)
            else:
                # Schedule exhausted; finalize.
                vehicle.completed = True
                vehicle.completion_time = self.sim_time

    def _handle_arrival(self, vehicle: VehicleState) -> None:
        """Deprecated: replaced by `_handle_visit_entry` and
        `_resolve_arrivals` after the MAPF integration. Kept as a no-op
        for any external callers that may still reference it."""
        return

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
        points.extend(network_points)
        if returning_home and home_slot is not None:
            points.extend(self._depot_access_to_slot(depot_access, home_slot))
        return dedupe_points(points)

    def _depot_departure_points(self, slot: Coord, depot_access: Coord) -> List[Coord]:
        # Drive out of the L corridor: slot → (entry corner if on horizontal arm) → entry.
        # `depot_access` is the L's entry coord (the NE grid corner).
        return dedupe_points([slot, *self._l_path(slot, depot_access)])

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

    def _depot_parking_points(
        self,
        current_position: Coord,
        home_slot: Coord,
    ) -> List[Coord]:
        if euclidean(current_position, home_slot) <= 1e-6:
            return dedupe_points([current_position])
        depot_access = self.instance.world.coords[self.instance.depot_node]
        path: List[Coord] = [current_position]
        if euclidean(current_position, depot_access) > 1e-6:
            path.extend(self._l_path(current_position, depot_access))
        path.extend(self._l_path(depot_access, home_slot))
        return dedupe_points(path)

    def _depot_access_to_slot(self, depot_access: Coord, slot: Coord) -> List[Coord]:
        return self._l_path(depot_access, slot)

    def _return_to_depot_entry_coords(
        self,
        current_node: str,
        depot_entry: Coord,
        depot_corridor: str,
    ) -> List[Coord]:
        points = self.solver.path_coords(current_node, self.instance.depot_node)
        depot_access = self.instance.world.coords[self.instance.depot_node]
        if points and euclidean(points[-1], depot_access) <= 1e-6:
            points = points[:-1]
        if not points or euclidean(points[-1], depot_entry) > 1e-6:
            points.append(depot_entry)
        # path_coords already returns a clean axis-aligned 3D polyline, so
        # the old 2D orthogonalization step is unnecessary here.
        return dedupe_points(points)

    def _depot_entry_to_slot(
        self,
        depot_entry: Coord,
        slot: Coord,
        depot_corridor: str,
    ) -> List[Coord]:
        return self._l_path(depot_entry, slot)

    def _l_path(self, start: Coord, end: Coord) -> List[Coord]:
        """Polyline from `start` to `end` along the L-shaped depot corridor.

        `start` and `end` must lie on the L (entry, vertical arm, or
        horizontal arm). Returns the intermediate + end points (omits the
        leading `start`). Empty list if start == end.
        """
        if euclidean(start, end) <= 1e-6:
            return []
        entry = self.instance.world.depot_entries["main"][0]
        arm_y = cad_layout.DEPOT_ARM_Y_IN

        def on_horizontal_arm(point: Coord) -> bool:
            return (
                math.isclose(point[1], arm_y, abs_tol=1e-6)
                and abs(point[0] - entry[0]) > 1e-6
            )

        start_on_arm = on_horizontal_arm(start)
        end_on_arm = on_horizontal_arm(end)
        # Depot is planar on the top Z-plane; the corner shares the entry's Z.
        corner = (entry[0], arm_y, entry[2])

        points: List[Coord] = []
        if start_on_arm and end_on_arm:
            points.append(end)
        elif start_on_arm and not end_on_arm:
            if euclidean(start, corner) > 1e-6:
                points.append(corner)
            if euclidean(points[-1] if points else start, end) > 1e-6:
                points.append(end)
        elif not start_on_arm and end_on_arm:
            if euclidean(start, corner) > 1e-6:
                points.append(corner)
            if euclidean(points[-1] if points else start, end) > 1e-6:
                points.append(end)
        else:
            points.append(end)
        return points

    def _draw(self) -> None:
        if self._water_bg is not None:
            self.screen.blit(self._water_bg, (0, 0))
        else:
            self.screen.fill(WATER_DEEP)
        self._draw_world()
        self._draw_hud()
        pygame.display.flip()

    def _draw_world(self) -> None:
        # Isometric cube: draw Z-planes back (lowest) to front (highest) so
        # nearer planes overlap farther ones, then the depot (top plane),
        # the cube's vertical struts, and finally the vehicles on top.
        self._draw_vertical_struts()
        for layer_idx in range(config.GRID_LAYERS):
            self._draw_layer_board(layer_idx)
            self._draw_layer_stations(layer_idx)
        self._draw_depot_dock()
        self._draw_depot_slots()
        self._draw_vehicles()

    def _draw_layer_board(self, layer_idx: int) -> None:
        """Draw one Z-plane as a translucent 'glass' sheet: a tinted corridor
        board with cream work cells, painted onto an alpha overlay and blended
        over the water so you can see through the stacked layers. Only the
        lower WHITE_SQUARE_LAYERS planes carry cells; the top is the depot."""
        overlay = self._layer_overlay
        if overlay is None:
            return
        z = cad_layout.GRID_ZS[layer_idx]
        road_xs = self.instance.world.road_xs
        road_ys = self.instance.world.road_ys
        half = config.ROAD_WIDTH_IN / 2.0

        overlay.fill((0, 0, 0, 0))
        board = self._iso_quad(
            road_xs[0] - half, road_ys[0] - half,
            road_xs[-1] + half, road_ys[-1] + half, z,
        )
        pygame.draw.polygon(
            overlay, self._layer_tint(ROAD_STRIP, layer_idx) + (LAYER_BOARD_ALPHA,), board
        )
        if layer_idx < cad_layout.WHITE_SQUARE_LAYERS:
            cell_rgba = self._layer_tint(CELL_FILL, layer_idx) + (LAYER_CELL_ALPHA,)
            for ciy in range(cad_layout.WHITE_SQUARE_ROWS):
                for cix in range(cad_layout.WHITE_SQUARE_COLS):
                    cell = self._iso_quad(
                        road_xs[cix] + half, road_ys[ciy] + half,
                        road_xs[cix + 1] - half, road_ys[ciy + 1] - half, z,
                    )
                    pygame.draw.polygon(overlay, cell_rgba, cell)
        self.screen.blit(overlay, (0, 0))
        # Crisp opaque outline so the plane edges stay legible through the glass.
        pygame.draw.polygon(self.screen, self._layer_tint(ROAD_EDGE, layer_idx), board, width=1)

    def _draw_layer_stations(self, layer_idx: int) -> None:
        z = cad_layout.GRID_ZS[layer_idx]
        radius = max(2, int(round(config.ALVIK_SIZE_IN * self._marker_scale() * 0.28)))
        active_ids = set(self.instance.active_job_ids)
        edge = self._layer_tint(ROAD_EDGE, layer_idx)
        for station in self.instance.stations:
            if abs(station.coord[2] - z) > 1e-6:
                continue
            _, color = self._station_state(station.station_id)
            if station.station_id not in active_ids:
                color = STATION_INACTIVE
            color = self._layer_tint(color, layer_idx)
            sx, sy = self._iso(station.coord)
            pygame.draw.circle(self.screen, color, (int(sx), int(sy)), radius)
            pygame.draw.circle(self.screen, edge, (int(sx), int(sy)), radius, width=1)

    def _draw_vertical_struts(self) -> None:
        """Four vertical edges of the lattice box, conveying the cube's depth."""
        gx = self.instance.world.road_xs[-1]
        gy = self.instance.world.road_ys[-1]
        z_lo = cad_layout.GRID_ZS[0]
        z_hi = cad_layout.GRID_ZS[-1]
        strut_color = (150, 196, 224)
        for x, y in ((0.0, 0.0), (gx, 0.0), (0.0, gy), (gx, gy)):
            a = self._iso((x, y, z_lo))
            b = self._iso((x, y, z_hi))
            pygame.draw.line(
                self.screen, strut_color,
                (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1,
            )

    def _draw_depot_dock(self) -> None:
        """The L-shaped depot dock, drawn as two iso quads on the top Z-plane."""
        z = cad_layout.DEPOT_Z_IN
        layer_idx = config.GRID_LAYERS - 1
        hw = config.ALVIK_SIZE_IN * 0.5 + 1.5
        grid_corner_y = self.instance.world.road_ys[-1]
        arm_y = cad_layout.DEPOT_ARM_Y_IN
        vert_x = cad_layout.DEPOT_VERTICAL_X_IN
        arm_west = cad_layout.DEPOT_ARM_XS[0]
        arm_east = cad_layout.DEPOT_ARM_XS[-1]
        color = self._layer_tint(ROAD_STRIP, layer_idx)

        pygame.draw.polygon(
            self.screen, color,
            self._iso_quad(vert_x - hw, grid_corner_y - hw, vert_x + hw, arm_y + hw, z),
        )
        pygame.draw.polygon(
            self.screen, color,
            self._iso_quad(arm_west - hw, arm_y - hw, arm_east + hw, arm_y + hw, z),
        )

    def _draw_depot_slots(self) -> None:
        radius = max(2, int(round(config.ALVIK_SIZE_IN * self._marker_scale() * 0.35)))
        for slot in self.instance.world.depot_slots:
            x_px, y_px = self._iso(slot)
            pygame.draw.circle(self.screen, DEPOT_BG, (int(x_px), int(y_px)), radius)
            pygame.draw.circle(self.screen, DEPOT_EDGE, (int(x_px), int(y_px)), radius, width=1)

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

    def _draw_vehicles(self) -> None:
        alvik_px = max(4, int(round(config.ALVIK_SIZE_IN * self._marker_scale())))
        shadow_offset = 3
        # Back-to-front so nearer Alviks overlap farther ones.
        ordered = sorted(self.vehicles, key=lambda v: self._iso_depth(v.position))
        for vehicle in ordered:
            x_px, y_px = self._iso(vehicle.position)
            rect = pygame.Rect(0, 0, alvik_px, alvik_px)
            rect.center = (int(x_px), int(y_px))

            pygame.draw.rect(
                self.screen,
                (120, 126, 132),
                rect.move(shadow_offset, shadow_offset),
                border_radius=6,
            )
            pygame.draw.rect(self.screen, vehicle.color, rect, border_radius=6)
            pygame.draw.rect(self.screen, ROAD_EDGE, rect, width=2, border_radius=6)

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
            (
                "Z-layer: "
                + ("all" if self.active_layer is None else str(self.active_layer)),
                self.font,
            ),
            ("", self.font),
            ("Controls", self.font),
            ("Drag slider for speed", self.font_small),
            ("Space pause, R reset", self.font_small),
            ("- = jobs down/up", self.font_small),
            (", . process time down/up", self.font_small),
            ("Drag mouse to rotate cube", self.font_small),
            ("Scroll wheel to zoom", self.font_small),
            ("[ ] rotate, Up/Down layer, A all", self.font_small),
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

    def _iso_raw(self, coord: Coord) -> Tuple[float, float]:
        """Project world (x, y, z) inches to unscaled isometric plane coords.

        The X/Y plane is rotated by `iso_yaw` about the vertical axis and
        tilted into a 2:1 diamond; +Z (depth/height) lifts the point up the
        screen. Scaling/centering is applied by `_iso`.
        """
        c = math.cos(self.iso_yaw)
        s = math.sin(self.iso_yaw)
        xr = coord[0] * c - coord[1] * s
        yr = coord[0] * s + coord[1] * c
        return (xr - yr, (xr + yr) * self.iso_pitch - coord[2])

    def _iso(self, coord: Coord) -> Tuple[float, float]:
        px, py = self._iso_raw(coord)
        s = self.iso_scale * self.iso_zoom
        return (self.iso_cx + px * s, self.iso_cy + py * s)

    def _marker_scale(self) -> float:
        """Pixels-per-inch for sizing dots/tiles, so they grow when zooming."""
        return self.iso_scale * self.iso_zoom

    # Back-compat alias: any remaining caller gets the isometric projection.
    def _to_screen(self, coord: Coord) -> Tuple[float, float]:
        return self._iso(coord)

    def _iso_depth(self, coord: Coord) -> float:
        """Painter's-algorithm key: larger = nearer the viewer (draw later)."""
        c = math.cos(self.iso_yaw)
        s = math.sin(self.iso_yaw)
        xr = coord[0] * c - coord[1] * s
        yr = coord[0] * s + coord[1] * c
        return (xr + yr) + coord[2] * 1e-3

    def _iso_quad(
        self, x0: float, y0: float, x1: float, y1: float, z: float
    ) -> List[Tuple[float, float]]:
        """Four screen-space corners of the planar world rect [x0,x1]x[y0,y1]
        at height z — a parallelogram under the isometric projection."""
        return [
            self._iso((x0, y0, z)),
            self._iso((x1, y0, z)),
            self._iso((x1, y1, z)),
            self._iso((x0, y1, z)),
        ]

    def _layer_tint(
        self, color: Tuple[int, int, int], layer_idx: int
    ) -> Tuple[int, int, int]:
        """Fade `color` toward the background by depth/active-layer state.

        With no active layer, lower planes (farther back) read dimmer. When a
        layer is isolated, only it shows at full strength.
        """
        n = max(1, config.GRID_LAYERS - 1)
        if self.active_layer is not None:
            f = 1.0 if layer_idx == self.active_layer else 0.16
        else:
            f = 0.45 + 0.55 * (layer_idx / n)
        # Deeper planes fade toward the dark water so the cube recedes.
        return tuple(int(b + (c - b) * f) for c, b in zip(color, WATER_FADE))

    def _compute_iso_params(self) -> None:
        """Fit the projected world cube into the world drawing rectangle."""
        if not hasattr(self, "world_width_px"):
            return
        corners = [
            (x, y, z)
            for x in (0.0, config.WORLD_WIDTH_IN)
            for y in (0.0, config.WORLD_HEIGHT_IN)
            for z in (0.0, config.WORLD_DEPTH_IN)
        ]
        projected = [self._iso_raw(pt) for pt in corners]
        xs = [p[0] for p in projected]
        ys = [p[1] for p in projected]
        span_x = (max(xs) - min(xs)) or 1.0
        span_y = (max(ys) - min(ys)) or 1.0
        self.iso_scale = 0.92 * min(
            self.world_width_px / span_x, self.world_height_px / span_y
        )
        mid_x = (min(xs) + max(xs)) / 2.0
        mid_y = (min(ys) + max(ys)) / 2.0
        self.iso_cx = self.world_left_px + self.world_width_px / 2.0 - self.iso_scale * mid_x
        self.iso_cy = self.world_top_px + self.world_height_px / 2.0 - self.iso_scale * mid_y

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
        return (0.0, 0.0, 0.0)
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
            a = points[index]
            b = points[index + 1]
            return tuple(ac + ((bc - ac) * local) for ac, bc in zip(a, b))
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
            len(point) == len(deduped[-1])
            and all(math.isclose(a, b) for a, b in zip(point, deduped[-1]))
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
    return math.dist(a, b)


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
