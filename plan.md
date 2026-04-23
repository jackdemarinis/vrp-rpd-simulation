Plan: Make the grid render as real 2-lane roads with a dotted yellow center line
Context
The user is confused by the current visual: the logical grid is 7×7, but each "street" is physically modeled as two 2-inch lanes separated by a 3-inch median gap (for bidirectional robot travel without collisions). Because the median gap is drawn in the same cream color as the city blocks, the screen reads as if there are 14×14 skinny streets rather than 7×7 two-lane roads.

Goal: make each street clearly look like one two-lane road, by filling the whole road envelope with road-surface color and painting a dashed yellow center line down the middle — the same visual language a human recognizes as "a two-way road."

This is purely a rendering change. No changes to the simulation model, graph, solver, or robot paths.

What's happening today
Relevant code lives in vrp_rpd_sim/render.py and vrp_rpd_sim/config.py.

config.py:9-10 — VERTICAL_ROAD_COUNT = 7, HORIZONTAL_ROAD_COUNT = 7
ROAD_STRIP_WIDTH_IN = 2.0, ROAD_MEDIAN_GAP_IN = 3.0, ROAD_ENVELOPE_WIDTH_IN = 7.0
Colors: ROAD_STRIP = (92, 95, 99) (dark lane), ROAD_GAP = (225, 216, 197) (cream median — same as city blocks), ROAD_EDGE = (54, 60, 67)
render.py:840 _draw_world() — fills cream background, then iterates road_xs / road_ys and calls the two road drawers
render.py:859 _draw_vertical_road(x_in) — draws two dark rectangles (left lane + right lane), leaving the cream background to show through as the median
render.py:876 _draw_horizontal_road(y_in) — same for horizontal roads
render.py:1083 _to_screen(coord) — world-inches → screen-pixels, world_scale handles resize
Because the two lanes are drawn separately and the median is the same color as the blocks, the eye sees two thin streets per road. The robots already travel bidirectionally — no lane direction is encoded per cell; this is purely visual.

Recommended approach
Unify each road into a single continuous dark surface, then paint a dashed yellow center line.

In both _draw_vertical_road() and _draw_horizontal_road(), replace the two separate lane rectangles with one rectangle spanning the full 7-inch envelope in ROAD_STRIP color. (The lanes + median visually merge into one road.)
Add a helper _draw_dashed_line(start_px, end_px, color, dash_len, gap_len, width) that draws a dashed segment between two screen-space points. It will be used for both orientations.
After drawing each full-width road rectangle, draw a dashed yellow line along the road's centerline:
Vertical road at world-x x_in: line runs from (x_in, 0) to (x_in, WORLD_SIZE_IN) in world space.
Horizontal road at world-y y_in: line runs from (0, y_in) to (WORLD_SIZE_IN, y_in).
Skip dashes through intersections. Where a vertical road crosses a horizontal road, don't draw center-line dashes across the intersection square — that's how real roads look and it prevents a yellow "+" from appearing at every crossing. Implement by splitting each line into segments between intersection spans (reuse the existing subtract_span_list() helper referenced by the explore agent if it fits; otherwise inline the span math using road_xs / road_ys and ROAD_ENVELOPE_WIDTH_IN / 2).
Station & depot stubs. Each station is a small rectangle attached to the edge of its road. These currently sit on the dark lane strip. After the road-fill change they'll still sit on road surface — no change needed. The depot is drawn last and will continue to cover its corner.
New constants (add to config.py alongside the other road constants)
LANE_DIVIDER_COLOR = (230, 190, 60)   # warm yellow
LANE_DIVIDER_WIDTH_IN = 0.25          # thin line, scales with world
LANE_DIVIDER_DASH_IN = 2.0            # dash length in world inches
LANE_DIVIDER_GAP_IN = 2.0             # gap length in world inches
Working in world-inches (not pixels) keeps the dash pattern stable across window resizes — _to_screen and world_scale handle the scaling, same as every other dimension in the renderer.

Files to touch
vrp_rpd_sim/config.py — add 4 constants above
vrp_rpd_sim/render.py — modify _draw_vertical_road (line 859) and _draw_horizontal_road (line 876); add _draw_dashed_line helper nearby
Why this over alternatives
"Leave the two lanes, just add a yellow line in the gap" — doesn't fix the 14×14 illusion, because the cream median still reads as a city block.
"Add solid white outer edge lines too" — more realistic but more code + risk of visual clutter at this zoom level. Can be added later if the dashed-yellow alone isn't clear enough.
Verification
Pure visual change; no tests needed beyond eyeballing.

Run python main.py from the repo root.
Confirm the grid now reads as 7 horizontal + 7 vertical two-lane roads (one unified dark surface per road) rather than 14×14 thin streets.
Confirm dashed yellow line runs down the middle of every road and does not continue through intersections.
Resize the window — dash pattern should scale smoothly with the world, not get pixelated or weirdly spaced.
Toggle job count with - / = and playback with the slider — rendering should be unaffected beyond what robots/stations are active.
Watch robots travel — they should still appear to be in "lanes" (one on each side of the yellow line), which is already how their pathing works; now it'll look intentional.