# VRP-RPD Simulation

A Manhattan-grid simulation of the Vehicle Routing Problem with Roaming Pickup and Delivery (VRP-RPD), with an ALNS → BRKGA solver pipeline and a Pygame visualizer.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

## Running

```
python main.py                  # solve + render
python main.py --headless       # solve only, print summary
python main.py --jobs 24        # override active job count
python main.py --sim-speed 4    # initial playback speed multiplier
```

Other flags: `--process-scale`, `--fixed-process-seconds`, `--fullscreen / --no-fullscreen`, `--config <path>`, `--export-unity-json <path>`, `--unity-capture-interval <sec>`.

Configuration defaults live in [simulation_config.json](simulation_config.json) (instance / solver / app / cache sections).

## Layout

- [main.py](main.py) — CLI entry point.
- [vrp_rpd_sim/world.py](vrp_rpd_sim/world.py) — grid + station + depot geometry.
- [vrp_rpd_sim/solver.py](vrp_rpd_sim/solver.py) — ALNS + BRKGA pipeline.
- [vrp_rpd_sim/render.py](vrp_rpd_sim/render.py) — Pygame visualizer and depot return logic.
- [vrp_rpd_sim/solution_cache.py](vrp_rpd_sim/solution_cache.py) — solution cache.
- [vrp_rpd_sim/unity_export.py](vrp_rpd_sim/unity_export.py) — Unity playback export.

## Debugging depot returns

Vehicles park at one of several depot slots reached through corridors; if a return gets stuck, the simulator emits diagnostics that explain why.

### Always-on warnings

These print without any flag, prefixed `[depot WARN <sim_time>s]`:

- **Stuck watchdog** — fires once per stall when a homebound vehicle's path has not advanced for ≥ 4 sim-seconds. Reports position, `current_node`, assigned `home_slot`, corridor, target, path progress, and pause/blocked counts.
- **No assignment available** — fires when one or more homebound vehicles cannot be given a slot/corridor. Reports free-slot count, reservations, active corridors, and the per-corridor frontier so you can see whether all slots are full, a corridor is geometrically blocked, or the frontier mismatches.
- **Empty return path** — fires if path construction returns no points while the vehicle is not yet at its home slot (the silent-stuck failure shape).

### Verbose lifecycle logs

Add `--debug-depot` for full per-vehicle lifecycle output, prefixed `[depot <sim_time>s]`:

```
python main.py --debug-depot
```

This adds:

- Slot/corridor assignment events.
- Homebound transition (where the vehicle was when its route ran out).
- Return path build details (point count, length, corridor, start/end coords).
- Completion events.

The watchdog threshold is `stuck_warn_threshold_sec` on `SimulationApp` (default 4.0). Adjust in [vrp_rpd_sim/render.py](vrp_rpd_sim/render.py) if you want it more or less sensitive.

### Reading a stall

Typical workflow:

1. Run `python main.py --debug-depot` and reproduce the stuck behavior.
2. The first `[depot WARN ...]` line tells you the failure mode:
   - "no return assignment available" → slot/corridor allocation issue.
   - "got empty return path" → path-building issue (geometry / mismatched corridor).
   - "STUCK while homebound" → motion blocked despite a valid path (look at `blocked_count` and the position vs. `home_slot`).
3. Cross-reference with the assignment / path-built lines printed earlier for that vehicle.

## Tests

```
pytest
```
