using System;
using System.Collections;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class PlaybackDriver : MonoBehaviour
{
    [Header("Playback")]
    [Tooltip("File name inside Assets/StreamingAssets, e.g. playback.json")]
    public string jsonFileName = "playback.json";
    [Tooltip("1 = realtime, 2 = double speed, 0.5 = half speed")]
    public float playbackSpeed = 1f;
    public bool loop = true;
    public bool autoPlay = true;

    [Header("Prefabs (cubes work fine until you have 3D models)")]
    public GameObject vehiclePrefab;
    public GameObject stationPrefab;
    public GameObject depotSlotPrefab;

    [Header("Table & World Scale")]
    [Tooltip("If true, worldScale is auto-derived so the simulation square fits exactly on the tabletop (recommended).")]
    public bool autoFitToTable = true;
    [Tooltip("Tabletop edge length in meters. 8 ft = 2.4384 m. Sim is fit to this when autoFitToTable is on.")]
    public float tableSizeMeters = 2.4384f;
    [Tooltip("Tabletop top-surface Y in meters above the floor. 3 ft = 0.9144 m.")]
    public float tableHeightMeters = 0.9144f;
    [Tooltip("Build a wooden table mesh at runtime. Leave OFF if you've authored a Table in the scene via Tools → VRP → Build Table.")]
    public bool generateTable = false;
    [Tooltip("Tabletop slab thickness in meters.")]
    public float tableThicknessMeters = 0.05f;
    [Tooltip("Square cross-section of each table leg, in meters.")]
    public float tableLegThicknessMeters = 0.06f;
    [Tooltip("Inset of legs from the tabletop edge, in meters.")]
    public float tableLegInsetMeters = 0.08f;
    public Color tableTopColor = new Color(0.42f, 0.27f, 0.16f);
    public Color tableLegColor = new Color(0.32f, 0.20f, 0.12f);
    [Tooltip("Manual override (used only when autoFitToTable is off). Multiplier from inches to Unity meters.")]
    public float worldScale = 0.0254f;
    [Tooltip("Local Y for vehicle/station markers above the tabletop surface (meters).")]
    public float modelHeight = 0f;

    [Header("Sizing (overrides prefab transform scale)")]
    [Tooltip("Force uniform scale on all spawned objects from alvikSizeIn * worldScale. Turn off if your prefabs already have correct authoring scale.")]
    public bool autoSizePrefabs = true;
    [Tooltip("Multiplier on alvikSizeIn for vehicle uniform scale")]
    public float vehicleSizeMultiplier = 1f;
    [Tooltip("Multiplier on alvikSizeIn for station marker uniform scale")]
    public float stationSizeMultiplier = 0.6f;
    [Tooltip("Multiplier on alvikSizeIn for depot slot marker uniform scale")]
    public float depotSlotSizeMultiplier = 1f;

    [Header("Vehicle Orientation")]
    [Tooltip("Euler offset applied AFTER LookRotation. If your Alvik mesh's nose isn't along +Z, fix it here once (e.g. (0,-90,0) if it points along +X).")]
    public Vector3 vehicleOrientationOffset = Vector3.zero;

    [Header("Visual Tinting")]
    public bool tintVehiclesByPaletteColor = true;
    public bool tintStationsByState = true;

    [Header("Environment (Roads)")]
    [Tooltip("Build dark-grey road strips and yellow dashed centerlines from the JSON road grid.")]
    public bool generateRoads = true;
    public Color roadColor = new Color(0.15f, 0.15f, 0.18f);
    public Color laneMarkingColor = new Color(0.95f, 0.78f, 0.18f);
    [Tooltip("If true, road height + vehicle/station height are auto-stacked: roads sit flush on the tabletop, vehicles sit on top of the roads. Overrides roadHeightOffset, laneMarkingHeightOffset, and modelHeight at runtime.")]
    public bool autoFitVerticalLayers = true;
    [Tooltip("Thickness of road slabs in world units (Y).")]
    public float roadThickness = 0.01f;
    [Tooltip("Center-Y position of road slabs above _SimRoot (tabletop surface). Ignored when autoFitVerticalLayers is on.")]
    public float roadHeightOffset = 0.015f;
    [Tooltip("Y position of lane markings above the road. Ignored when autoFitVerticalLayers is on.")]
    public float laneMarkingHeightOffset = 0.008f;
    [Tooltip("Thickness of lane marking slabs in world units (Y).")]
    public float laneMarkingThickness = 0.006f;
    public bool generateLaneMarkings = true;
    [Tooltip("Length of each yellow dash, in source units (inches).")]
    public float laneDashLengthIn = 5f;
    [Tooltip("Gap between dashes, in source units (inches).")]
    public float laneDashGapIn = 4f;
    [Tooltip("Width of each yellow dash strip, in source units (inches).")]
    public float laneDashWidthIn = 0.4f;

    [Header("Debug Logging")]
    [Tooltip("Log each spawned object's actual world-bounds size vs the target.")]
    public bool logSpawnSizes = true;
    [Tooltip("Log when two vehicles get closer than (overlapFraction * alvikSize) apart.")]
    public bool logVehicleOverlaps = true;
    [Range(0.1f, 1.5f)] public float overlapFraction = 0.9f;
    [Tooltip("Log when a vehicle moves more than (jumpMultiplier * alvikSpeed * dt) in one frame — indicates a teleport.")]
    public bool logPositionJumps = true;
    [Range(1.5f, 10f)] public float jumpMultiplier = 2f;
    [Tooltip("Throttle: minimum seconds between repeated overlap warnings for the same vehicle pair.")]
    public float overlapLogCooldown = 1f;

    [Header("File Logging")]
    [Tooltip("Write logs to a file at <project>/playback_session.log (in addition to Console).")]
    public bool writeLogFile = true;
    public string logFileName = "playback_session.log";
    [Tooltip("Write a full state snapshot every N seconds of playback time.")]
    public float snapshotIntervalSec = 0.5f;
    [Tooltip("In each snapshot, also write the pairwise distance matrix between all vehicles.")]
    public bool snapshotIncludeDistanceMatrix = true;

    PlaybackData data;
    float playbackTime;
    Transform _simRoot;       // parent for all sim-space children (centered on this transform, on top of the table)
    Transform _tableRoot;     // parent for the table mesh
    Transform[] vehicleTransforms;
    Renderer[] vehicleRenderers;
    Renderer[] stationRenderers;
    Vector3[] lastVehiclePositions;
    float[,] lastOverlapLogTime; // [i,j] = last time we logged the i<->j pair
    int frameCursor;
    float totalDuration;

    StreamWriter logFile;
    float lastSnapshotPlaybackTime = -999f;
    string logFilePath;

    void Start()
    {
        OpenLogFile();
        StartCoroutine(LoadAndInit());
    }

    IEnumerator LoadAndInit()
    {
        var path = Path.Combine(Application.streamingAssetsPath, jsonFileName);
        string text = null;

        // On Android/WebGL, streamingAssetsPath is a URL inside the APK/jar — File.IO won't work.
        // UnityWebRequest handles both URL and plain file paths, so use it everywhere.
        bool needsWebRequest = path.Contains("://") || path.StartsWith("jar:");
        if (needsWebRequest)
        {
            using (var req = UnityWebRequest.Get(path))
            {
                yield return req.SendWebRequest();
#if UNITY_2020_2_OR_NEWER
                if (req.result != UnityWebRequest.Result.Success)
#else
                if (req.isHttpError || req.isNetworkError)
#endif
                {
                    Debug.LogError($"PlaybackDriver: failed to load JSON from {path}: {req.error}");
                    enabled = false;
                    yield break;
                }
                text = req.downloadHandler.text;
            }
        }
        else
        {
            if (!File.Exists(path))
            {
                Debug.LogError($"PlaybackDriver: cannot find playback JSON at {path}");
                enabled = false;
                yield break;
            }
            text = File.ReadAllText(path);
        }

        data = JsonUtility.FromJson<PlaybackData>(text);
        if (data == null || data.frames == null || data.frames.Length == 0)
        {
            Debug.LogError("PlaybackDriver: playback JSON parsed empty");
            enabled = false;
            yield break;
        }

        if (autoFitToTable && data.worldSizeIn > 0f)
        {
            float derived = tableSizeMeters / data.worldSizeIn;
            Debug.Log($"PlaybackDriver: autoFitToTable on. Overriding worldScale {worldScale:F4} -> {derived:F4} " +
                      $"(table={tableSizeMeters:F3}m, sim={data.worldSizeIn:F1}in)");
            worldScale = derived;
        }

        if (autoFitVerticalLayers)
        {
            // Stack from the tabletop (y=0) upward:
            //   road slab          [0,                          roadThickness]
            //   lane markings      [roadThickness,              roadThickness + laneMarkingThickness]
            //   vehicle/station    base = roadThickness + laneMarkingThickness + tiny clearance
            float prevRoadH = roadHeightOffset;
            float prevLaneH = laneMarkingHeightOffset;
            float prevModelH = modelHeight;
            roadHeightOffset = roadThickness * 0.5f;
            // laneMarkingHeightOffset is added to roadHeightOffset to get the lane center Y.
            laneMarkingHeightOffset = (roadThickness * 0.5f) + (laneMarkingThickness * 0.5f);
            modelHeight = roadThickness + laneMarkingThickness + 0.0005f;
            Debug.Log($"PlaybackDriver: autoFitVerticalLayers on. roadHeightOffset {prevRoadH:F4}->{roadHeightOffset:F4}, " +
                      $"laneMarkingHeightOffset {prevLaneH:F4}->{laneMarkingHeightOffset:F4}, " +
                      $"modelHeight {prevModelH:F4}->{modelHeight:F4}");
        }

        totalDuration = data.frames[data.frames.Length - 1].timeSec;
        SpawnWorld();
        ApplyFrame(0f);
        Debug.Log($"PlaybackDriver: loaded {data.frames.Length} frames spanning {totalDuration:F2}s, " +
                  $"{data.vehicles.Length} vehicles, {data.stations.Length} stations, " +
                  $"alvikSizeIn={data.alvikSizeIn:F3} (target world size = {data.alvikSizeIn * worldScale:F4}m), " +
                  $"alvikSpeedInPerSec={data.alvikSpeedInPerSec:F3} (world speed = {data.alvikSpeedInPerSec * worldScale:F4}m/s)");

        if (vehicleTransforms != null && vehicleTransforms.Length > 0)
        {
            lastVehiclePositions = new Vector3[vehicleTransforms.Length];
            for (int i = 0; i < vehicleTransforms.Length; i++)
                if (vehicleTransforms[i] != null) lastVehiclePositions[i] = vehicleTransforms[i].position;
            lastOverlapLogTime = new float[vehicleTransforms.Length, vehicleTransforms.Length];
            for (int i = 0; i < vehicleTransforms.Length; i++)
                for (int j = 0; j < vehicleTransforms.Length; j++)
                    lastOverlapLogTime[i, j] = -1000f;
        }
    }

    void Update()
    {
        if (!autoPlay || data == null) return;

        playbackTime += Time.deltaTime * playbackSpeed;
        if (loop && playbackTime > totalDuration)
        {
            playbackTime = 0f;
            frameCursor = 0;
        }
        playbackTime = Mathf.Clamp(playbackTime, 0f, totalDuration);
        ApplyFrame(playbackTime);
    }

    void SpawnWorld()
    {
        if (generateTable) BuildTable();

        // Re-use _SimRoot if one was authored in the scene (e.g. by the editor menu);
        // otherwise create a fresh one on top of the tabletop. Sim content positions are local
        // to _SimRoot, centered on its origin.
        var existing = transform.Find("_SimRoot");
        if (existing != null)
        {
            _simRoot = existing;
            // Clear any leftover authored stubs so we always reflect the current JSON.
            for (int i = _simRoot.childCount - 1; i >= 0; i--) Destroy(_simRoot.GetChild(i).gameObject);
        }
        else
        {
            var simRootGo = new GameObject("_SimRoot");
            _simRoot = simRootGo.transform;
            _simRoot.SetParent(transform, false);
            _simRoot.localPosition = new Vector3(0f, tableHeightMeters, 0f);
            _simRoot.localRotation = Quaternion.identity;
            _simRoot.localScale = Vector3.one;
        }

        if (generateRoads) SpawnEnvironment();

        float baseSize = data.alvikSizeIn * worldScale;

        // Stations
        stationRenderers = new Renderer[data.stations != null ? data.stations.Length : 0];
        if (stationPrefab != null && data.stations != null)
        {
            for (int i = 0; i < data.stations.Length; i++)
            {
                var s = data.stations[i];
                var go = Instantiate(stationPrefab, _simRoot);
                go.transform.localPosition = SimToLocal(s.x, s.y);
                go.transform.localRotation = Quaternion.identity;
                go.name = string.IsNullOrEmpty(s.name) ? $"station_{s.stationId}" : s.name;
                if (autoSizePrefabs) NormalizeToTargetSize(go, baseSize * stationSizeMultiplier);
                if (logSpawnSizes) LogActualBounds(go, $"station {go.name}", baseSize * stationSizeMultiplier);
                stationRenderers[i] = go.GetComponentInChildren<Renderer>();
            }
        }

        // Depot slots (visual only — purely for spatial reference)
        if (depotSlotPrefab != null && data.depot != null && data.depot.slots != null)
        {
            foreach (var slot in data.depot.slots)
            {
                var go = Instantiate(depotSlotPrefab, _simRoot);
                go.transform.localPosition = SimToLocal(slot.x, slot.y);
                go.transform.localRotation = Quaternion.identity;
                go.name = $"DepotSlot_{slot.slotIndex}";
                if (autoSizePrefabs) NormalizeToTargetSize(go, baseSize * depotSlotSizeMultiplier);
                if (logSpawnSizes) LogActualBounds(go, $"depotSlot {go.name}", baseSize * depotSlotSizeMultiplier);
            }
        }

        // Vehicles
        vehicleTransforms = new Transform[data.vehicles != null ? data.vehicles.Length : 0];
        vehicleRenderers = new Renderer[data.vehicles != null ? data.vehicles.Length : 0];
        if (vehiclePrefab != null && data.vehicles != null)
        {
            for (int i = 0; i < data.vehicles.Length; i++)
            {
                var v = data.vehicles[i];
                var go = Instantiate(vehiclePrefab, _simRoot);
                go.transform.localPosition = SimToLocal(v.startX, v.startY);
                go.transform.localRotation = Quaternion.Euler(vehicleOrientationOffset);
                go.name = string.IsNullOrEmpty(v.displayName) ? $"vehicle_{v.vehicleId}" : v.displayName;
                if (autoSizePrefabs) NormalizeToTargetSize(go, baseSize * vehicleSizeMultiplier);
                if (logSpawnSizes) LogActualBounds(go, $"vehicle {go.name}", baseSize * vehicleSizeMultiplier);
                vehicleTransforms[i] = go.transform;
                vehicleRenderers[i] = go.GetComponentInChildren<Renderer>();
                if (tintVehiclesByPaletteColor && vehicleRenderers[i] != null
                    && v.colorRgb != null && v.colorRgb.Length >= 3)
                {
                    vehicleRenderers[i].material.color = new Color(
                        v.colorRgb[0] / 255f,
                        v.colorRgb[1] / 255f,
                        v.colorRgb[2] / 255f);
                }
            }
        }
    }

    void BuildTable()
    {
        var go = new GameObject("Table");
        _tableRoot = go.transform;
        _tableRoot.SetParent(transform, false);
        _tableRoot.localPosition = Vector3.zero;
        _tableRoot.localRotation = Quaternion.identity;

        var topMat = MakeMaterial(tableTopColor);
        var legMat = MakeMaterial(tableLegColor);

        // Tabletop: top surface lands at y = tableHeightMeters; slab extends downward by tableThicknessMeters.
        float topCenterY = tableHeightMeters - (tableThicknessMeters * 0.5f);
        var top = GameObject.CreatePrimitive(PrimitiveType.Cube);
        top.name = "Tabletop";
        Destroy(top.GetComponent<Collider>());
        top.transform.SetParent(_tableRoot, false);
        top.transform.localPosition = new Vector3(0f, topCenterY, 0f);
        top.transform.localScale = new Vector3(tableSizeMeters, tableThicknessMeters, tableSizeMeters);
        top.GetComponent<Renderer>().sharedMaterial = topMat;

        // Four legs: from floor (y=0) to underside of slab (y = tableHeightMeters - tableThicknessMeters).
        float legHeight = Mathf.Max(0.01f, tableHeightMeters - tableThicknessMeters);
        float legY = legHeight * 0.5f;
        float legOffset = (tableSizeMeters * 0.5f) - tableLegInsetMeters - (tableLegThicknessMeters * 0.5f);
        Vector2[] legXZ = {
            new Vector2(+legOffset, +legOffset),
            new Vector2(+legOffset, -legOffset),
            new Vector2(-legOffset, +legOffset),
            new Vector2(-legOffset, -legOffset),
        };
        for (int i = 0; i < legXZ.Length; i++)
        {
            var leg = GameObject.CreatePrimitive(PrimitiveType.Cube);
            leg.name = $"Leg_{i}";
            Destroy(leg.GetComponent<Collider>());
            leg.transform.SetParent(_tableRoot, false);
            leg.transform.localPosition = new Vector3(legXZ[i].x, legY, legXZ[i].y);
            leg.transform.localScale = new Vector3(tableLegThicknessMeters, legHeight, tableLegThicknessMeters);
            leg.GetComponent<Renderer>().sharedMaterial = legMat;
        }

        Debug.Log($"[Table] Built {tableSizeMeters:F2}m x {tableSizeMeters:F2}m tabletop, top at y={tableHeightMeters:F2}m, legs {legHeight:F2}m tall.");
    }

    // Scales a spawned prefab so its largest world-space bounds dimension equals targetSize,
    // preserving the prefab's authored aspect ratio (so flat slabs stay flat, tall things stay tall).
    static void NormalizeToTargetSize(GameObject go, float targetSize)
    {
        var renderers = go.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return;
        Bounds b = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++) b.Encapsulate(renderers[i].bounds);
        float maxExtent = Mathf.Max(b.size.x, Mathf.Max(b.size.y, b.size.z));
        if (maxExtent < 1e-6f) return;
        float factor = targetSize / maxExtent;
        go.transform.localScale *= factor;
    }

    static void LogActualBounds(GameObject go, string label, float targetSize)
    {
        var renderers = go.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            Debug.LogWarning($"[Spawn] {label}: no renderers found, cannot measure bounds");
            return;
        }
        Bounds b = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++) b.Encapsulate(renderers[i].bounds);
        Debug.Log($"[Spawn] {label}: bounds size = ({b.size.x:F4}, {b.size.y:F4}, {b.size.z:F4})m, " +
                  $"target = {targetSize:F4}m, localScale = {go.transform.localScale}");
    }

    void SpawnEnvironment()
    {
        if (data == null || data.roadXs == null || data.roadYs == null)
        {
            Debug.LogWarning($"[Env] Skipped: data null? {data == null}, roadXs null? {data?.roadXs == null}, roadYs null? {data?.roadYs == null}");
            return;
        }
        Debug.Log($"[Env] Building roads: {data.roadXs.Length} vertical, {data.roadYs.Length} horizontal, " +
                  $"width={data.roadEnvelopeWidthIn:F2}in, worldSize={data.worldSizeIn:F1}in, worldScale={worldScale:F4}");

        // Environment is parented to _SimRoot so it travels with the simulation (table moves -> roads move).
        var envRoot = new GameObject("Environment");
        envRoot.transform.SetParent(_simRoot, false);
        envRoot.transform.localPosition = Vector3.zero;
        envRoot.transform.localRotation = Quaternion.identity;

        var roadMat = MakeMaterial(roadColor);
        var laneMat = MakeMaterial(laneMarkingColor);

        float halfIn = data.worldSizeIn * 0.5f;
        float roadWidth = data.roadEnvelopeWidthIn * worldScale;
        float worldLen = data.worldSizeIn * worldScale;

        // Vertical roads (constant X), span full Z range, centered on simRoot origin.
        foreach (float xIn in data.roadXs)
        {
            float x = (xIn - halfIn) * worldScale;
            CreateFlatRect(envRoot.transform, $"Road_X_{xIn:F1}",
                new Vector3(x, roadHeightOffset, 0f),
                new Vector3(roadWidth, roadThickness, worldLen),
                roadMat);
        }

        // Horizontal roads (constant Y in sim -> constant Z in Unity), span full X range.
        foreach (float yIn in data.roadYs)
        {
            float z = (yIn - halfIn) * worldScale;
            CreateFlatRect(envRoot.transform, $"Road_Y_{yIn:F1}",
                new Vector3(0f, roadHeightOffset, z),
                new Vector3(worldLen, roadThickness, roadWidth),
                roadMat);
        }

        if (generateLaneMarkings)
            SpawnLaneMarkings(envRoot.transform, laneMat);
    }

    void SpawnLaneMarkings(Transform parent, Material mat)
    {
        float dashLen = laneDashLengthIn * worldScale;
        float dashGap = laneDashGapIn * worldScale;
        float dashWidth = laneDashWidthIn * worldScale;
        float halfIn = data.worldSizeIn * 0.5f;
        float worldHalf = halfIn * worldScale;
        float pitch = dashLen + dashGap;
        if (pitch <= 0f) return;

        float y = roadHeightOffset + laneMarkingHeightOffset;

        // Vertical roads → dashes running along Z, centered on simRoot.
        foreach (float xIn in data.roadXs)
        {
            float x = (xIn - halfIn) * worldScale;
            for (float zCenter = -worldHalf + dashGap * 0.5f + dashLen * 0.5f;
                 zCenter + dashLen * 0.5f <= worldHalf;
                 zCenter += pitch)
            {
                CreateFlatRect(parent, $"Lane_X_{xIn:F1}_{zCenter:F2}",
                    new Vector3(x, y, zCenter),
                    new Vector3(dashWidth, laneMarkingThickness, dashLen),
                    mat);
            }
        }

        // Horizontal roads → dashes running along X.
        foreach (float yIn in data.roadYs)
        {
            float z = (yIn - halfIn) * worldScale;
            for (float xCenter = -worldHalf + dashGap * 0.5f + dashLen * 0.5f;
                 xCenter + dashLen * 0.5f <= worldHalf;
                 xCenter += pitch)
            {
                CreateFlatRect(parent, $"Lane_Y_{yIn:F1}_{xCenter:F2}",
                    new Vector3(xCenter, y, z),
                    new Vector3(dashLen, laneMarkingThickness, dashWidth),
                    mat);
            }
        }
    }

    static int debugRectsLogged = 0;
    static void CreateFlatRect(Transform parent, string name, Vector3 center, Vector3 size, Material mat)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = center;
        go.transform.localScale = size;
        var r = go.GetComponent<Renderer>();
        if (r != null) r.sharedMaterial = mat;

        if (debugRectsLogged < 4)
        {
            debugRectsLogged++;
            Debug.Log($"[Env] Created '{name}' at world={go.transform.position} localPos={center} scale={size} " +
                      $"renderer.bounds.center={r?.bounds.center} bounds.size={r?.bounds.size} " +
                      $"layer={go.layer} active={go.activeInHierarchy}");
        }
    }

    static Material MakeMaterial(Color color)
    {
        Shader s = Shader.Find("Universal Render Pipeline/Lit");
        if (s == null) s = Shader.Find("Standard");
        var m = new Material(s);
        if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", color);
        m.color = color;
        return m;
    }

    void OpenLogFile()
    {
        if (!writeLogFile) return;
        try
        {
            // Editor / desktop builds: <project>/playback_session.log  (dataPath = <project>/Assets in editor, <build>/<exe>_Data in standalone)
            // Android (Quest): dataPath is the read-only APK — use persistentDataPath instead.
            string baseDir = Application.platform == RuntimePlatform.Android
                ? Application.persistentDataPath
                : Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            logFilePath = Path.Combine(baseDir, logFileName);
            logFile = new StreamWriter(logFilePath, false, Encoding.UTF8) { AutoFlush = true };
            logFile.WriteLine($"=== PlaybackDriver session started {DateTime.Now:yyyy-MM-dd HH:mm:ss} ===");
            logFile.WriteLine($"logFilePath: {logFilePath}");
            Application.logMessageReceived += OnUnityLogMirror;
            Debug.Log($"PlaybackDriver: writing log file to {logFilePath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"PlaybackDriver: failed to open log file: {e.Message}");
            logFile = null;
        }
    }

    void CloseLogFile()
    {
        if (logFile == null) return;
        try
        {
            Application.logMessageReceived -= OnUnityLogMirror;
            logFile.WriteLine($"=== session closed {DateTime.Now:yyyy-MM-dd HH:mm:ss} ===");
            logFile.Flush();
            logFile.Close();
        }
        catch { }
        logFile = null;
    }

    void OnUnityLogMirror(string condition, string stackTrace, LogType type)
    {
        if (logFile == null) return;
        try { logFile.WriteLine($"[{Time.time:F3}] [{type}] {condition}"); }
        catch { }
    }

    void OnDisable() { CloseLogFile(); }
    void OnDestroy() { CloseLogFile(); }
    void OnApplicationQuit() { CloseLogFile(); }

    void WriteSnapshot(float time, Frame fa, Frame fb, float t)
    {
        if (logFile == null || data == null) return;
        var sb = new StringBuilder();
        sb.AppendLine();
        sb.AppendLine($"--- snapshot t={time:F3}s (frameCursor={frameCursor}, fa.t={fa.timeSec:F3}, fb.t={fb.timeSec:F3}, lerp={t:F2}) ---");
        sb.AppendLine($"  jobs: completed={fa.completedJobs} active={fa.activeJobs} vehiclesCompleted={fa.vehiclesCompleted}");

        if (fa.vehicles != null && vehicleTransforms != null)
        {
            int count = Mathf.Min(fa.vehicles.Length, vehicleTransforms.Length);
            for (int i = 0; i < count; i++)
            {
                var va = fa.vehicles[i];
                Vector3 wp = vehicleTransforms[i] != null ? vehicleTransforms[i].position : Vector3.zero;
                sb.AppendLine($"  V{va.vehicleId,-2} simPos=({va.x,7:F2},{va.y,7:F2}) " +
                              $"worldPos=({wp.x:F3},{wp.z:F3}) " +
                              $"node={va.currentNode,-12} target={va.targetNode,-12} " +
                              $"routeIdx={va.routeIndex,-2} load={va.load} " +
                              $"hasPath={va.hasActivePath} done={va.completed} " +
                              $"waitingFor={va.waitingCustomerId}");
            }

            if (snapshotIncludeDistanceMatrix && count > 1)
            {
                sb.AppendLine($"  pairwise world distances (m):");
                sb.Append("       ");
                for (int j = 0; j < count; j++) sb.Append($"   V{fa.vehicles[j].vehicleId,-2}  ");
                sb.AppendLine();
                for (int i = 0; i < count; i++)
                {
                    sb.Append($"   V{fa.vehicles[i].vehicleId,-2}");
                    Vector3 pi = vehicleTransforms[i] != null ? vehicleTransforms[i].position : Vector3.zero;
                    for (int j = 0; j < count; j++)
                    {
                        if (j <= i) { sb.Append("        "); continue; }
                        Vector3 pj = vehicleTransforms[j] != null ? vehicleTransforms[j].position : Vector3.zero;
                        Vector3 d = pj - pi; d.y = 0f;
                        sb.Append($" {d.magnitude,7:F3}");
                    }
                    sb.AppendLine();
                }
            }
        }

        try { logFile.Write(sb.ToString()); }
        catch { }
    }

    void ApplyFrame(float time)
    {
        // Advance/rewind cursor so frames[frameCursor] is the latest frame at-or-before `time`.
        while (frameCursor + 1 < data.frames.Length && data.frames[frameCursor + 1].timeSec <= time)
            frameCursor++;
        while (frameCursor > 0 && data.frames[frameCursor].timeSec > time)
            frameCursor--;

        int lo = frameCursor;
        int hi = Mathf.Min(lo + 1, data.frames.Length - 1);
        Frame fa = data.frames[lo];
        Frame fb = data.frames[hi];
        float dt = fb.timeSec - fa.timeSec;
        float t = dt > 1e-6f ? Mathf.Clamp01((time - fa.timeSec) / dt) : 0f;

        // Vehicles: lerp position, smoothly rotate to face travel direction.
        if (vehicleTransforms != null && fa.vehicles != null && fb.vehicles != null)
        {
            int count = Mathf.Min(Mathf.Min(fa.vehicles.Length, fb.vehicles.Length), vehicleTransforms.Length);
            for (int i = 0; i < count; i++)
            {
                var va = fa.vehicles[i];
                var vb = fb.vehicles[i];
                Vector3 pa = SimToLocal(va.x, va.y);
                Vector3 pb = SimToLocal(vb.x, vb.y);
                Transform tr = vehicleTransforms[i];
                if (tr == null) continue;
                tr.localPosition = Vector3.Lerp(pa, pb, t);

                Vector3 dir = pb - pa;
                dir.y = 0f;
                if (dir.sqrMagnitude > 1e-8f)
                {
                    Quaternion target = Quaternion.LookRotation(dir) * Quaternion.Euler(vehicleOrientationOffset);
                    tr.localRotation = Quaternion.Slerp(tr.localRotation, target, 0.4f);
                }
            }
        }

        // Periodic state snapshot to log file.
        if (writeLogFile && logFile != null && time - lastSnapshotPlaybackTime >= snapshotIntervalSec)
        {
            lastSnapshotPlaybackTime = time;
            WriteSnapshot(time, fa, fb, t);
        }

        // Diagnostics: detect vehicle-vehicle overlaps and large position jumps.
        if ((logVehicleOverlaps || logPositionJumps) && vehicleTransforms != null)
        {
            float worldAlvikSize = data.alvikSizeIn * worldScale;
            float overlapThreshold = worldAlvikSize * overlapFraction;
            float dtFrame = Time.deltaTime;
            float maxStep = data.alvikSpeedInPerSec * worldScale * dtFrame * jumpMultiplier + worldAlvikSize * 0.1f;

            for (int i = 0; i < vehicleTransforms.Length; i++)
            {
                if (vehicleTransforms[i] == null) continue;
                Vector3 pos = vehicleTransforms[i].position;

                // Position-jump detection
                if (logPositionJumps && lastVehiclePositions != null)
                {
                    Vector3 step = pos - lastVehiclePositions[i];
                    step.y = 0f;
                    float dist = step.magnitude;
                    if (dist > maxStep && dtFrame > 0f)
                    {
                        Debug.LogWarning($"[Jump] {vehicleTransforms[i].name} moved {dist:F4}m in {dtFrame * 1000f:F1}ms " +
                                         $"(max expected {maxStep:F4}m at {jumpMultiplier:F1}x speed) at t={time:F2}s. " +
                                         $"Likely a frame-cursor reset or large gap between JSON frames.");
                    }
                    lastVehiclePositions[i] = pos;
                }

                // Vehicle-vehicle overlap detection
                if (logVehicleOverlaps)
                {
                    for (int j = i + 1; j < vehicleTransforms.Length; j++)
                    {
                        if (vehicleTransforms[j] == null) continue;
                        Vector3 d = vehicleTransforms[j].position - pos;
                        d.y = 0f;
                        float dist = d.magnitude;
                        if (dist < overlapThreshold)
                        {
                            float now = Time.time;
                            if (now - lastOverlapLogTime[i, j] >= overlapLogCooldown)
                            {
                                lastOverlapLogTime[i, j] = now;
                                Debug.LogWarning($"[Overlap] {vehicleTransforms[i].name} & {vehicleTransforms[j].name} " +
                                                 $"are {dist:F4}m apart (threshold {overlapThreshold:F4}m, alvikSize {worldAlvikSize:F4}m) at t={time:F2}s. " +
                                                 $"sim positions: " +
                                                 $"a=({fa.vehicles[i].x:F2},{fa.vehicles[i].y:F2})→({fb.vehicles[i].x:F2},{fb.vehicles[i].y:F2}), " +
                                                 $"b=({fa.vehicles[j].x:F2},{fa.vehicles[j].y:F2})→({fb.vehicles[j].x:F2},{fb.vehicles[j].y:F2})");
                            }
                        }
                    }
                }
            }
        }

        // Stations: snap to nearer frame's color so the marker doesn't blend through wrong colors.
        if (tintStationsByState && stationRenderers != null)
        {
            Frame current = t < 0.5f ? fa : fb;
            if (current.stations != null)
            {
                int count = Mathf.Min(current.stations.Length, stationRenderers.Length);
                for (int i = 0; i < count; i++)
                {
                    var s = current.stations[i];
                    if (stationRenderers[i] == null || s.colorRgb == null || s.colorRgb.Length < 3) continue;
                    stationRenderers[i].material.color = new Color(
                        s.colorRgb[0] / 255f,
                        s.colorRgb[1] / 255f,
                        s.colorRgb[2] / 255f);
                }
            }
        }
    }

    Vector3 SimToLocal(float simX, float simY)
    {
        // Pygame X -> Unity X; Pygame Y -> Unity Z (top-down view).
        // Centered on _SimRoot's local origin so the sim sits centered on the tabletop.
        float halfIn = data != null ? data.worldSizeIn * 0.5f : 0f;
        return new Vector3((simX - halfIn) * worldScale, modelHeight, (simY - halfIn) * worldScale);
    }

    public float TotalDuration => totalDuration;
    public float CurrentTime => playbackTime;

    public void SetTime(float t)
    {
        playbackTime = Mathf.Clamp(t, 0f, totalDuration);
        frameCursor = 0;
        if (data != null) ApplyFrame(playbackTime);
    }
}
