Part 1 — Current Architecture Analysis
How the board layout is generated
world.py build_world() does the following:

Computes road spacing via _compute_road_gap():

road_gap = (96" - 2×14"_margin - 7×7"_envelope) / (7-1) ≈ 3.17" between adjacent road edges
pitch = 7" + 3.17" ≈ 10.17" between road centers
First road center at 14 + 3.5 = 17.5", last at 78.5"
Emits 49 intersection nodes i_{ix}_{iy} (ix=0..6, iy=0..6) plus 28 perimeter boundary nodes (vbot_*, vtop_*, hleft_*, hright_*). These form the routable graph.

Connects nodes via _connect_consecutive() in columns then rows, building a full bidirectional weighted adjacency list.

Attaches station nodes on the perimeter: bottom/top boundary nodes at road indices [1,2,4,5], left/right boundary nodes at road indices [1,2,4,5]. 16 stations total.

Places the depot: single logical node "depot" at (road_xs[0] - 2.5", road_ys[0] + 2.5") ≈ (15", 20") — physically in the large blank corner margin outside the road grid. Connected to vbot_0 and hleft_0 with zero-ish weight edges.

Computes all-pairs shortest paths via Dijkstra.

Builds depot slots: 3×3 grid anchored at DEPOT_ANCHOR_IN = (6.5", 6.5"). With Alvik size + gap, slots land at x ≈ {2, 6.5, 11} and y ≈ {2, 6.5, 11} — deep in the bottom-left corner margin (well under road_xs[0] ≈ 17.5").

Computes 6 depot entry points via _depot_entry_points(): 3 "top" entries at y = road_ys[0] - 2.5" and 3 "right" entries at x = road_xs[0] - 2.5", aligned to depot slot columns/rows.

How depot positions are defined
Logical graph position: coords["depot"] at (road_xs[0] - LANE_CENTER_OFFSET_IN, road_ys[0] + LANE_CENTER_OFFSET_IN) (world.py:284)
Physical parking positions: depot_slots list — 3×3 grid in bottom-left corner (world.py:271)
Entry points: depot_entries["top"] and depot_entries["right"] — 6 coords on the innermost road lanes (world.py:291)
Visual: a single large green rectangle rendered by _draw_depot() (render.py:1113) covering all 9 slots, labeled "Depot"
Config anchor: DEPOT_ANCHOR_IN = (6.5, 6.5) drives slot generation
How roads/intersections/internal cells are represented
Roads are drawn as full-width/height grey rectangles across the world surface (_draw_vertical_road / _draw_horizontal_road)
Road intersections = intersection nodes i_{ix}_{iy} in the graph, no special visual marker
Internal cells (the 6×6 = 36 white blocks between adjacent road pairs): completely absent from the code — no nodes, no data structures, no rendering, nothing
The interior block between vertical roads ix, ix+1 and horizontal roads iy, iy+1 has its center at:


cx = (road_xs[ix] + road_xs[ix+1]) / 2
cy = (road_ys[iy] + road_ys[iy+1]) / 2
This is currently open white space with no representation in the model.

How package placement currently works
_handle_arrival() (render.py:764) fires when a vehicle reaches its target node
If op.kind == "D": vehicle.load -= 1, station timer starts — no direction check
If op.kind == "P": waits until ready_at, then vehicle.load += 1 — no direction check
There is zero approach-direction enforcement anywhere in the codebase
How robot capacity is defined
VEHICLE_CAPACITY = 1 in config.py:44
Flows through: Instance.capacity → initial load = self.instance.capacity in _build_vehicle_states() → enforced in solver.py evaluate() with load > self.instance.capacity → infeasible
The rendering updates load via min(capacity, load+1) and max(0, load-1), both relative to instance.capacity
How interaction direction is currently handled
It is not handled at all. No side or direction concept exists for station or depot interactions. The Station dataclass has a side field (for which perimeter side the station is on), but this is never used to gate a "D" or "P" operation.

Part 2 — Detailed Implementation Plan
Files and classes that need to change
File	What changes
config.py	VEHICLE_CAPACITY: 1 → 4; remove/comment DEPOT_ANCHOR_IN dependency note
model.py	Add interior_depots: List[Tuple[int, int, Coord]] to WorldGraph
world.py	Add _interior_block_centers(); populate WorldGraph.interior_depots; leave depot_node/depot_slots/depot_entries intact for robot routing
render.py	Replace _draw_depot() with _draw_interior_depots(); add north-approach enforcement in _handle_arrival(); add last_approach_dy to VehicleState
Layout-generation logic that needs to change
In world.py, add a new pure helper immediately after _depot_entry_points:


def _interior_block_centers(road_xs, road_ys):
    """Return (ix, iy, center_coord) for every road-enclosed block."""
    blocks = []
    for ix in range(len(road_xs) - 1):   # 0..5 → 6 columns
        for iy in range(len(road_ys) - 1): # 0..5 → 6 rows
            cx = (road_xs[ix] + road_xs[ix + 1]) / 2.0
            cy = (road_ys[iy] + road_ys[iy + 1]) / 2.0
            blocks.append((ix, iy, (cx, cy)))
    return blocks   # 36 entries
Add one line in build_world() before constructing WorldGraph:


interior_depots = _interior_block_centers(road_xs, road_ys)
And pass it to WorldGraph(... interior_depots=interior_depots).

Do not touch depot_node, depot_slots, depot_entries, or any of the path/entry topology. Those drive robot routing and return behavior — they stay exactly as-is.

How to detect valid internal white cells
The rule is already encoded in _interior_block_centers above: every (ix, iy) where 0 ≤ ix < VERTICAL_ROAD_COUNT - 1 and 0 ≤ iy < HORIZONTAL_ROAD_COUNT - 1 is a valid interior block. With 7×7 roads, that is exactly 6×6 = 36 depots.

No filtering is needed — all 36 blocks are bounded by four road corridors by construction.

How to remove perimeter depots
The current "depot" in the corner is not a perimeter depot in the usual sense — it is the robot staging area. Do not delete depot_node or depot_slots (robots still need to park and the solver still routes from the depot).

What to remove is the visual in render.py:1113–1124 (_draw_depot()). Delete that method body and replace with a call to _draw_interior_depots().

How to resize depot visuals and simplify labels
In render.py, replace _draw_depot():


def _draw_interior_depots(self) -> None:
    gap_px = self.instance.world.road_gap_in * self.world_scale
    size_px = max(10, int(gap_px * 0.55))   # ~55% of the gap width
    for idx, (ix, iy, coord) in enumerate(self.instance.world.interior_depots):
        x_px, y_px = self._to_screen(coord)
        rect = pygame.Rect(0, 0, size_px, size_px)
        rect.center = (round(x_px), round(y_px))
        pygame.draw.rect(self.screen, DEPOT_BG, rect, border_radius=3)
        pygame.draw.rect(self.screen, DEPOT_EDGE, rect, width=1, border_radius=3)
        label = self._fit_font(str(idx + 1), size_px - 4, size_px - 4)
        surf = label.render(str(idx + 1), True, DEPOT_EDGE)
        self.screen.blit(surf, (rect.centerx - surf.get_width() // 2,
                                rect.centery - surf.get_height() // 2))
Replace the self._draw_depot() call in _draw_world() with self._draw_interior_depots().

Labels are just integers 1–36. No "Depot" prefix. _fit_font already exists and will scale the text down to fit.

How to enforce north-side-only package placement
Step A — Track last approach direction on VehicleState.

Add a field to VehicleState:


last_approach_dy: float = 0.0   # positive = arrived from south (traveling north), negative = from north
Step B — Update last_approach_dy when a vehicle finishes a path.

In _handle_arrival(), before clearing vehicle.active_path, capture the final movement direction:


if vehicle.active_path and len(vehicle.active_path.points) >= 2:
    p1 = vehicle.active_path.points[-2]
    p2 = vehicle.active_path.points[-1]
    vehicle.last_approach_dy = p2[1] - p1[1]   # positive = northward, negative = southward
Step C — Gate "D" and "P" operations.

In _handle_arrival(), add a check before executing a drop or pickup:


if op.kind in {"D", "P"}:
    # Enforce north-side-only: robot must have been traveling southward (dy < 0)
    if vehicle.last_approach_dy >= 0:
        # Not approaching from north — stall and wait for a valid re-approach
        vehicle.target_node = None
        return
Note: "Traveling southward" in world coordinates means y is decreasing as the robot moves, i.e., p2.y < p1.y, so last_approach_dy < 0.

This is a minimal, clean enforcement — no graph changes required. If a robot arrives from east, west, or south, the operation simply doesn't fire and the robot must re-approach (the solver path will generally not trigger this since paths along the grid already approach from natural directions, but it acts as a hard constraint if a future path is incorrect).

How to update each robot to hold exactly 4 packages
One line change in config.py:


VEHICLE_CAPACITY = 4
Everything else is data-driven through self.instance.capacity. The solver's capacity check in evaluate() already uses self.instance.capacity. Initial load in _build_vehicle_states() uses self.instance.capacity. The load increment/decrement in _handle_arrival() and _prime_vehicle_targets() are already clamped correctly.

Part 3 — Step-by-Step Implementation Order
config.py — Change VEHICLE_CAPACITY = 4. Verify simulation still launches headless (python main.py --headless).

model.py — Add interior_depots: List[Tuple[int, int, Coord]] = field(default_factory=list) to WorldGraph.

world.py — Add _interior_block_centers() function; call it in build_world() and pass result to WorldGraph.

render.py — visual — Add _draw_interior_depots() method; replace _draw_depot() body (or just rename the call in _draw_world()). Run the sim, verify 36 small labeled blocks appear inside the road grid.

render.py — direction enforcement — Add last_approach_dy: float = 0.0 to VehicleState; update it at the top of _handle_arrival(); add the north-side gate on D/P operations.

Regression check — Run the full simulation (python main.py), watch all 9 robots complete their routes, confirm no robots stall permanently due to the direction check.

Testing Plan
Test	What to check
test_interior_block_centers	Assert len(world.interior_depots) == 36; assert block (0,0) center ≈ midpoint of road_xs[0] and road_xs[1]; assert all centers are strictly inside the road bounds
test_no_depot_outside_grid	Assert no entry in interior_depots has cx < road_xs[0] or cx > road_xs[-1] (none are in the margin)
test_vehicle_capacity_is_4	Assert instance.capacity == 4; assert a route with 4 deliveries before any pickup evaluates feasible; assert 5 deliveries evaluate infeasible
test_north_approach_gate	Construct a mock VehicleState with last_approach_dy = +1 (arrived from south); run _handle_arrival on a D operation; assert station_progress["dropped_at"] remains None
test_north_approach_passes	Same mock but last_approach_dy = -1 (arrived from north); assert drop fires correctly
Existing tests	Confirm pytest tests/ still passes — depot return slot topology tests should be unaffected since depot_slots/depot_entries are unchanged
Assumptions
The 7×7 road grid with road_gap ≈ 3.17" produces a visible but tight gap. Depot icons at ~55% of the gap width (~12–14px at 1000px window) will be small but legible.
The robot staging area (slots in the corner) remains functionally unchanged. The visual "depot markers" inside the grid are independent of where robots physically park.
"North side only" applies at the moment of D/P interaction — direction is computed from the final path segment. Robots already naturally approach grid intersections along road axes, so this will rarely block valid deliveries. It will block any future path that arrives via an east/west/south segment.
The solver routes are not explicitly constrained by direction — the direction check is a runtime enforcement only, not a solver constraint. If needed, solver-side direction constraints would require a much larger change.
Changing VEHICLE_CAPACITY to 4 will change the solver's output routes (more packages per robot = different packing). The ALNS/BRKGA parameters may produce suboptimal results at capacity=4 without re-tuning, but the logic is correct.
Interior depots are numbered 1–36 in (ix, iy) order, left-to-right, bottom-to-top.
Things to Verify Before Implementing
Confirm actual road gap size visually — run python main.py and look at how wide the white spaces between roads appear. If they look too narrow for even a small icon, ROAD_EDGE_MARGIN_IN should be reduced (from 14" to ~6–8") to allow more gap. This is a config-only change but affects spacing everywhere.

Confirm "north side" intent — does "from the north side" mean: (a) robot's final path segment is moving southward, or (b) robot must physically be above the depot (have a higher y than the block center at time of interaction? These produce slightly different results at corners.

Confirm where robots should spawn — should robot home slots also move inside the road grid, or stay in the corner? The current plan leaves slots in the corner (less disruptive), but if robots should visually start at an interior depot, that requires also updating DEPOT_ANCHOR_IN and _depot_entry_points.

Confirm numbering scheme — 1–36 in what order? The plan assumes bottom-left first (ix=0, iy=0 = depot 1), incrementing ix first. If a different order is preferred, adjust the enumeration in _draw_interior_depots.

Check if ACTIVE_JOB_COUNT = 16 is still valid — it currently caps at len(stations) = 16. With 36 depots being a new concept, verify whether the job count concept needs to extend to depots or stays with perimeter stations only.