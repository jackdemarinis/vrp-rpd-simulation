# Plan: Replace single depot entry with 6 parallel entries

## Context

When robots (Alviks) return to the depot, every one of them funnels through a single access point at `(15", 20")` on the bottom-left corner of the road grid. The existing code in [vrp_rpd_sim/render.py](vrp_rpd_sim/render.py) enforces strict one-at-a-time serialization via `_sync_depot_return_owner` / `_should_hold_for_depot_return` / `DEPOT_APPROACH_RESERVATION_IN = 11.28"` — while one robot parks, every other homebound robot stops outside an 11.28" bubble and waits. With 9 robots returning in sequence, this dominates the end of each run and makes the simulation feel stalled.

The fix: expose **6 entry points** arranged around the depot (3 on the top edge, 3 on the right edge), let robots use them concurrently, and assign `(entry, slot)` pairs dynamically so robots end up "planned" into the tightest collision-free parking. Rely on the existing `_max_safe_travel` / sweep collision framework for physical safety instead of the owner bubble.

## Design overview

- Keep a **single logical `depot_node`** in the solver graph — the 6 entries are a rendering/pathing concern. Solver distances change by <5% using any of the six points, which is far below the routing cost signal, so there's no reason to pay the complexity of multiple graph nodes.
- Add **6 entry-point coordinates** derived from depot column/row positions, placed on the innermost road lanes.
- **Delete the owner serialization**: there is no "one owner at a time." Robots flow into depot unidirectionally (all traffic converges inward, none exits), so the sweep-based collision detector handles natural following without deadlock.
- Dynamic **global mincost assignment** of `(robot → corridor → slot)` whenever the homebound set changes, with a **corridor-stack invariant** (deepest slot in a corridor is assigned first so a later robot never has to pass a parked one).

## Entry-point coordinates

Depot slot positions (from `_build_depot_slots`, anchor `(6.5, 6.5)`, step `4.53"`):
- column x-values: `{1.97, 6.50, 11.03}`
- row y-values: `{1.97, 6.50, 11.03}`

6 entry points (on the nearest road lanes, `LANE_CENTER_OFFSET_IN = 2.5"` inside the first road envelope):
- **TOP entries** on the south lane of horizontal road 0 (`y = road_ys[0] - LANE_CENTER_OFFSET_IN = 15`):
  - `TOP_0 = (1.97, 15)`
  - `TOP_1 = (6.50, 15)`
  - `TOP_2 = (11.03, 15)`
- **RIGHT entries** on the west lane of vertical road 0 (`x = road_xs[0] - LANE_CENTER_OFFSET_IN = 15`):
  - `RIGHT_0 = (15, 1.97)`
  - `RIGHT_1 = (15, 6.50)`
  - `RIGHT_2 = (15, 11.03)`

Each top entry serves its column's 3 slots; each right entry serves its row's 3 slots. Every slot is reachable through 2 corridors (its column's top entry and its row's right entry), which gives the assigner flexibility.

## Corridor-stack invariant

A "corridor" is the straight-line path inside the depot from an entry point along one axis. If robot A parks at a shallow slot first, robot B arriving later through the same corridor would have to pass **through** A's slot to reach a deeper one — a collision waiting to happen.

**Fill each corridor deepest-first**:
- Top corridors (entering from north, moving south): fill row 0 (`y ≈ 1.97`) before row 1 before row 2.
- Right corridors (entering from east, moving west): fill col 0 (`x ≈ 1.97`) before col 1 before col 2.

At any moment, a corridor has one "next available" slot — its deepest unreserved slot. A robot can only be assigned a corridor whose next-available slot is not yet claimed.

## Files to modify

### `vrp_rpd_sim/config.py`
Add constants:
```python
DEPOT_ENTRY_LANE_OFFSET_IN = LANE_CENTER_OFFSET_IN  # 2.5"; semantic alias
DEPOT_ENTRY_TOP_COUNT = 3
DEPOT_ENTRY_RIGHT_COUNT = 3
```
Leave `DEPOT_APPROACH_RESERVATION_IN` in place (unused after this change) or remove once confirmed no other module imports it.

### `vrp_rpd_sim/world.py`
In [world.py:116-130](vrp_rpd_sim/world.py#L116-L130), extend the depot section (keep the existing single `depot_node` and its two edges untouched — those drive solver distances). Add a helper next to `_depot_access_coord`:

```python
def _depot_entry_points(road_xs, road_ys, depot_slots) -> dict:
    col_xs = sorted({round(x, 6) for x, _ in depot_slots})
    row_ys = sorted({round(y, 6) for _, y in depot_slots})
    top_y = road_ys[0] - config.LANE_CENTER_OFFSET_IN   # south lane of h-road 0
    right_x = road_xs[0] - config.LANE_CENTER_OFFSET_IN # west lane of v-road 0
    return {
        "top":   [(x, top_y) for x in col_xs],
        "right": [(right_x, y) for y in row_ys],
    }
```

Store on `WorldGraph` as `depot_entries: dict[str, list[Coord]]`.

### `vrp_rpd_sim/render.py`
**Remove** (owner serialization is gone):
- `self.depot_return_owner_id` field and init (render.py:129)
- `self._debug_depot_zero_move`, `self._debug_depot_holds` fields
- `_release_depot_return_owner_if_needed` (render.py:359-370)
- `_sync_depot_return_owner` (render.py:372-393)
- `_should_hold_for_depot_return` (render.py:395-417)
- The owner-prioritization sort + hold check inside `_advance_vehicles` (render.py:515-566). Keep the safe-motion logic (`_max_safe_travel`, `_is_safe_motion`, sweeps).

**Add / replace**:
- New field: `self.corridor_next_slot: dict[str, Coord | None]` — one entry per corridor ID (`"top_0"..."top_2"`, `"right_0"..."right_2"`), pointing to the current deepest unreserved slot in that corridor.
- Extend `VehicleState` with `depot_entry: Coord | None` and `depot_corridor: str | None`.
- Replace `_assign_return_slot(vehicle)` (render.py:339-353) with `_assign_return_targets()`:
  - Gather all homebound vehicles without an assignment.
  - Candidate set per robot = `{(corridor, next_slot_of_corridor) for each corridor whose next_slot is defined}`.
  - Cost of a candidate = road-graph distance from `vehicle.position` to the corridor's entry + Manhattan L-distance from the entry to the slot.
  - Solve a mincost matching (robot × candidate) — a small Hungarian over ≤9 rows × ≤6 cols. If adding `scipy` is undesirable, a straightforward Kuhn–Munkres implementation fits in ~40 lines; a good-enough alternative for this size is repeated greedy with improvement swaps.
  - When a slot is claimed, advance the "next available" pointer on every corridor that contains it (corner slots live in two corridors).
- Modify `_travel_points` returning-home branch (render.py:730-740) to route through `vehicle.depot_entry` instead of the single `depot_access`.
- Replace `_depot_access_to_slot` (render.py:768-776) with `_depot_entry_to_slot(entry, slot, corridor_kind)`:
  - `corridor_kind == "top"`: path is `entry → (entry.x, slot.y) → slot` (move south first, then east/west to slot's column — but by construction `entry.x == slot.x`, so it collapses to one segment).
  - `corridor_kind == "right"`: path is `entry → (slot.x, entry.y) → slot` (move west, then north/south — again collapses since `entry.y == slot.y`).
  - Because entries are axis-aligned with their slots by design, each in-depot approach is a single straight line — zero risk of clipping another parked robot.
- `_depot_parking_points` (render.py:743-756) keeps its current L-logic for the already-inside-depot case (used only when a robot is mid-park and its path is re-primed); wire it to use the assigned entry when applicable.
- `_slot_to_depot_access` (render.py:758-766) — used when a robot leaves a depot slot at simulation start. Parameterize on `vehicle.depot_entry` so the exit path mirrors the entry corridor. (Robots start parked and leave once; this keeps leaving traffic symmetric with entering.)

### `tests/test_depot_layout.py` and `tests/test_depot_return_simulation.py`
- Replace `test_return_slots_fill_from_furthest_corner_first` with `test_corridor_stack_invariant_no_crossover`: build a small scenario where 3 robots are assigned to the same top column and assert their slots fill deepest-first within that column.
- Add `test_depot_entry_points_geometry`: assert `world.depot_entries["top"]` returns 3 coords at `y = 15` matching the column xs, and similarly for `"right"` at `x = 15`.
- Add `test_concurrent_depot_return_overlaps`: run the full sim and assert that at some tick ≥2 homebound vehicles make non-zero forward progress simultaneously — the direct inverse of the old "one at a time" behavior.
- Keep `test_full_simulation_returns_every_vehicle_to_the_grid` — the end state (every robot in a unique depot slot) must still hold. Expect total sim time to drop noticeably.

## Reused functions / utilities

- `_max_safe_travel`, `_is_safe_motion`, `_positions_conflict`, `_sweeps_conflict` ([render.py:568-620](vrp_rpd_sim/render.py#L568-L620)) — the existing collision framework is the entire safety net after owner removal. No changes.
- `reserved_return_slots: Set[Coord]` ([render.py:139](vrp_rpd_sim/render.py#L139)) — still used; just populated by `_assign_return_targets` instead of `_assign_return_slot`.
- `depot_return_slots` / `depot_return_slot_ranks` ([render.py:132-138](vrp_rpd_sim/render.py#L132-L138)) — keep as stable tiebreaker when costs are equal.
- `dedupe_points`, `build_path_state` — already handle the new path shapes.

## Verification

1. **Unit tests** (above): `pytest tests/` passes, including the new concurrency and corridor-stack tests.
2. **Full sim wall-clock**: run `python main.py` with `ACTIVE_JOB_COUNT = 16` and a long processing time (e.g., `PROCESSING_TIME_VARIANT = "5x"`). Visually confirm that when robots arrive home, multiple enter the depot at the same time through distinct entries on the top and right edges — no queue forms outside the bottom-left corner. Time to "all 9 parked" should drop by roughly the old serial-parking duration.
3. **Visual sanity**: watch for any robot momentarily stopping *inside* the depot. If two robots pick the same corridor in the same tick, the sweep detector should make the second pause briefly behind the first — acceptable and looks natural. Any stall *longer* than ~1s indicates a bug.
4. **Debug flag**: if `debug_depot=True`, log the `(robot, corridor, slot)` assignment on each re-plan. Verify the corridor-stack invariant holds (no shallow slot assigned while a deeper one in the same corridor is unclaimed).
5. **No regressions**: robots still path correctly to customer nodes (non-depot targets unchanged); the solver output is identical (single depot node untouched in the graph).
