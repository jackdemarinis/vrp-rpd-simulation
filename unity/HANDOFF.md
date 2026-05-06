# Unity Integration Handoff

This document captures everything a fresh Claude session needs to know to continue the Unity integration work without re-deriving context.

## Quick facts

- **User**: jack (jackfastime@gmail.com), windows 11, PowerShell.
- **Python repo**: `C:\Users\jackf\Documents\Github\vrp-rpd-simulation` (this repo).
- **Unity project**: separate directory the user will tell you about.
- **Goal**: render the existing pygame VRP-RPD simulation inside Unity using 3D models, with **identical** behavior. No new movement logic in Unity — playback only.

## Architecture (do not change without asking)

Python is the **authoritative simulator**. Unity is a **dumb playback renderer**.

```
Python sim (vrp_rpd_sim/)
    │
    │  python main.py --export-unity-json playback.json
    ▼
playback.json  ──>  Unity StreamingAssets/  ──>  PlaybackDriver.cs lerps frames
```

### Why this and not other approaches

- **Don't port the sim to C#.** ALNS/BRKGA solvers are large; behavioral drift would be the entire point gone.
- **Don't embed Python in Unity.** Unity is C#/Mono; there's no clean way. User already tried zipping the Python package into Unity and got compile collisions — confirmed dead end.
- **Don't run Python as a live server.** User wants minimal work and "exact same as pygame"; pre-baked JSON playback meets both.

The user's existing `vrp_rpd_sim/unity_export.py` was purpose-built for this approach.

## What exists in the Python repo (already done)

### Sim itself

Standard ALNS+BRKGA pipeline → Pygame visualizer in `vrp_rpd_sim/render.py`. Entry: `main.py`. Config: `simulation_config.json`.

### Export pipeline

`vrp_rpd_sim/unity_export.py`. Run with:
```
python main.py --export-unity-json playback.json --unity-capture-interval 0.05
```

Produces a JSON with `schemaVersion: 1` containing:
- World metadata (`worldSizeIn`, `roadXs`, `roadYs`, `coordinateMapping`)
- Static specs (`stations`, `vehicles`, `depot.slots`, `depot.entries`)
- Per-frame snapshots (`frames[]` with vehicle positions + station states)
- Solver summary

Coordinate mapping baked into the JSON: **pygame X → Unity X, pygame Y → Unity Z**, units inches, recommended Unity scale **0.0254** (= meters).

### Recently-fixed bug (depot returns)

`vrp_rpd_sim/render.py` had two bugs causing vehicles to get stuck returning to the depot:
1. `_next_return_assignment` did `return None` instead of `continue` when a slot's only corridors were active — caused deadlock when deeper free slots existed.
2. `_corridor_path_clear` didn't check cross-corridor path crossings — top corridor and right corridor share corner cells.

Both fixed. Always-on stuck warning + `--debug-depot` verbose lifecycle logging added.

User has not yet verified the fixes with a fresh run; if they re-export JSON and Unity playback shows vehicles correctly returning to depot, the fixes are confirmed.

### Unity-side scripts (in `unity/Assets/Scripts/`)

Three C# files, ready to drop into a Unity project:

- **`PlaybackData.cs`** — `[Serializable]` classes mirroring the JSON schema for `JsonUtility.FromJson<PlaybackData>`. Field names case-sensitive, must match the JSON exactly.
- **`PlaybackDriver.cs`** — MonoBehaviour. Loads `playback.json` from `StreamingAssets/`, instantiates prefabs for vehicles/stations/depot slots, lerps positions between bracketing frames each `Update()`. Auto-sizes prefabs from `alvikSizeIn` so the user doesn't fiddle with prefab scales. Has `worldScale` (default 0.0254), tints vehicles by palette color, tints stations by per-frame state.
- **`CameraRig.cs`** — Top-down camera controller. WASD pan (speed scales with zoom), scroll/Q-E zoom, Shift = sprint. Works for both orthographic and perspective.

User has tested the Python → JSON → Unity loop and it works. The pygame and Unity views show the same vehicles moving along the same paths.

## Current state of the Unity scene (per user's last screenshot)

- Ground Plane sized for the 60" × 60" world at `worldScale=0.0254`.
- Top-down orthographic camera at `(0.762, 2, 0.762)`, rotation `(90, 0, 0)`, ortho size ~1.0.
- Placeholder cubes for vehicles, stations, depot slots — auto-sized by the driver.
- Vehicles tinted by their pygame palette colors. Stations turn green/yellow/etc. on state changes.

User saw a scale issue (vehicle cubes looked chunky) → fixed by adding auto-sizing to `PlaybackDriver`. They have not yet confirmed the fix in-editor.

## What's next: editor setup script

The user is moving the Claude session into their Unity project directory so files can be written directly into `Assets/`. Once you confirm the new working directory, the planned next step is:

### `Assets/Editor/VRPSceneSetup.cs`

A Unity Editor script with a `[MenuItem("Tools/VRP/Set Up Scene")]` that, on click, builds the entire scene:

1. Create `Assets/StreamingAssets/` if missing.
2. Create primitive prefabs in `Assets/Prefabs/`:
   - `VehiclePrefab` — Cube, scale 1, default material
   - `StationPrefab` — Cube or Sphere, scale 1
   - `DepotSlotPrefab` — Cube, scale 1, faint white material
3. Create the ground Plane: position `(0.762, 0, 0.762)`, scale `(0.1524, 1, 0.1524)`. Tint dark grey.
4. Configure Main Camera: position `(0.762, 2, 0.762)`, rotation `(90, 0, 0)`, projection Orthographic, size 1.0. Add `CameraRig` component.
5. Create `PlaybackController` empty GameObject. Add `PlaybackDriver`. Wire the three prefabs into its inspector slots. Set `jsonFileName = "playback.json"`, `worldScale = 0.0254`.
6. Add a Directional Light if missing.
7. (Optional) Mark scene dirty and prompt user to save.

Editor scripts go under `Assets/Editor/`. Use `UnityEditor` namespace, `EditorApplication`, `AssetDatabase`, `PrefabUtility`. Need `using UnityEngine; using UnityEditor;`.

After the user clicks the menu item once, their workflow becomes: regenerate JSON → copy to `StreamingAssets/` → press Play.

## How to operate from inside the Unity project

When the user launches you from their Unity project root:

- **Working directory** will be `<UnityProject>/`. Verify with `pwd` (PowerShell) or by listing the root — should see `Assets/`, `ProjectSettings/`, `Packages/`.
- **Don't put `.cs` files at the project root.** Unity only compiles scripts under `Assets/`. Place runtime scripts in `Assets/Scripts/`, editor scripts in `Assets/Editor/`.
- **Don't write `.meta` files yourself.** Unity generates them on next focus. If you write a `.cs`, Unity will create the `.meta` automatically.
- **Don't edit `.unity` or `.prefab` YAML by hand** unless absolutely necessary — easy to corrupt them. Prefer editor scripts.
- **The Python repo is at** `C:\Users\jackf\Documents\Github\vrp-rpd-simulation`. If the user wants you to also read/regenerate the JSON, that's where the simulator lives. The scripts in `unity/Assets/Scripts/` there are the **source of truth** for the Unity scripts — when you change one in the Unity project, also update the copy in the Python repo, or the user will diverge.

### What the user can do; what they can't

- **They press Play.** You can't see the result; ask for a screenshot or console paste.
- **They click menu items.** You write the editor scripts that produce those menu items.
- **They drag prefabs in the Inspector.** Avoid this — make the editor script do the wiring.
- **Compile errors come from the Console.** Ask them to paste any red lines after they save a script.

## Constraints / preferences from the user

- Wants minimal work; "rely on you."
- Doesn't want to introduce 3D movement logic — keep playback only.
- Was frustrated when zipping Python into Unity created collisions. Don't suggest re-trying that.
- Communicates terse and informal; respond in kind, not with corporate-doc tone.
- Uses Windows 11 + PowerShell. Pipeline chain (`&&`, `||`) is **not** available in Windows PowerShell 5.1 — use `;` and `if ($?)`.

## Memory system

Claude Code project memory for this user lives at:
```
C:\Users\jackf\.claude\projects\c--Users-jackf-Documents-Github-vrp-rpd-simulation\memory\
```

If you save user/feedback/project memories during the session, write them there and add an entry to that directory's `MEMORY.md`.

Note: when running you in the **Unity project** directory, Claude Code will use a *different* memory directory keyed off the new path. Memories saved here won't carry over automatically. Read this HANDOFF.md as your bridge.

## First actions for the next session

1. Confirm working directory (`pwd`) — should be the Unity project, not the Python repo.
2. Confirm the three scripts (`PlaybackData.cs`, `PlaybackDriver.cs`, `CameraRig.cs`) are present under `Assets/Scripts/`. If not, copy them from the Python repo's `unity/Assets/Scripts/`.
3. Confirm `playback.json` is in `Assets/StreamingAssets/`. If not, generate it from the Python repo and copy.
4. Ask the user whether they want the editor setup script (`VRPSceneSetup.cs`) written now, or whether they want to manually verify the current scene first.
5. If writing the editor script: place it at `Assets/Editor/VRPSceneSetup.cs`, then tell the user to click `Tools → VRP → Set Up Scene` once.

Don't start writing code until step 1 is confirmed.
