using System.IO;
using UnityEngine;

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

    [Header("World Scale")]
    [Tooltip("Multiplier from pygame inches to Unity world units. 0.0254 = meters; 1.0 = inches as-is")]
    public float worldScale = 0.0254f;
    [Tooltip("Y height for vehicle and station markers")]
    public float modelHeight = 0f;

    [Header("Auto-sizing (sets prefab scale at spawn time, in inches)")]
    [Tooltip("If true, override the prefab's localScale on spawn so models match sim dimensions.")]
    public bool overridePrefabScale = true;
    [Tooltip("Vehicle footprint in inches. 0 = use alvikSizeIn from the JSON.")]
    public float vehicleSizeIn = 0f;
    public float vehicleHeightIn = 2f;
    public float stationSizeIn = 2f;
    public float stationHeightIn = 2f;
    public float depotSlotSizeIn = 3.5f;
    public float depotSlotHeightIn = 0.25f;

    [Header("Visual Tinting")]
    public bool tintVehiclesByPaletteColor = true;
    public bool tintStationsByState = true;

    PlaybackData data;
    float playbackTime;
    Transform[] vehicleTransforms;
    Renderer[] vehicleRenderers;
    Renderer[] stationRenderers;
    int frameCursor;
    float totalDuration;

    void Start()
    {
        var path = Path.Combine(Application.streamingAssetsPath, jsonFileName);
        if (!File.Exists(path))
        {
            Debug.LogError($"PlaybackDriver: cannot find playback JSON at {path}");
            enabled = false;
            return;
        }

        var text = File.ReadAllText(path);
        data = JsonUtility.FromJson<PlaybackData>(text);
        if (data == null || data.frames == null || data.frames.Length == 0)
        {
            Debug.LogError("PlaybackDriver: playback JSON parsed empty");
            enabled = false;
            return;
        }

        totalDuration = data.frames[data.frames.Length - 1].timeSec;
        SpawnWorld();
        ApplyFrame(0f);
        Debug.Log($"PlaybackDriver: loaded {data.frames.Length} frames spanning {totalDuration:F2}s, " +
                  $"{data.vehicles.Length} vehicles, {data.stations.Length} stations");
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
        float vSize = vehicleSizeIn > 0f ? vehicleSizeIn : data.alvikSizeIn;
        Vector3 vehicleScale = SizeIn(vSize, vehicleHeightIn, vSize);
        Vector3 stationScale = SizeIn(stationSizeIn, stationHeightIn, stationSizeIn);
        Vector3 slotScale    = SizeIn(depotSlotSizeIn, depotSlotHeightIn, depotSlotSizeIn);

        // Half-height offsets so cube/sphere prefabs (centered at origin) sit on the plane.
        vehicleY = vehicleHeightIn * worldScale * 0.5f;
        stationY = stationHeightIn * worldScale * 0.5f;
        slotY    = depotSlotHeightIn * worldScale * 0.5f;

        // Stations
        stationRenderers = new Renderer[data.stations != null ? data.stations.Length : 0];
        if (stationPrefab != null && data.stations != null)
        {
            for (int i = 0; i < data.stations.Length; i++)
            {
                var s = data.stations[i];
                var go = Instantiate(stationPrefab, SimToWorldStation(s.x, s.y), Quaternion.identity, transform);
                go.name = string.IsNullOrEmpty(s.name) ? $"station_{s.stationId}" : s.name;
                if (overridePrefabScale) go.transform.localScale = stationScale;
                stationRenderers[i] = go.GetComponentInChildren<Renderer>();
            }
        }

        // Depot slots (visual only — purely for spatial reference)
        if (depotSlotPrefab != null && data.depot != null && data.depot.slots != null)
        {
            foreach (var slot in data.depot.slots)
            {
                var go = Instantiate(depotSlotPrefab, SimToWorldSlot(slot.x, slot.y), Quaternion.identity, transform);
                go.name = $"DepotSlot_{slot.slotIndex}";
                if (overridePrefabScale) go.transform.localScale = slotScale;
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
                var go = Instantiate(vehiclePrefab, SimToWorld(v.startX, v.startY), Quaternion.identity, transform);
                go.name = string.IsNullOrEmpty(v.displayName) ? $"vehicle_{v.vehicleId}" : v.displayName;
                if (overridePrefabScale) go.transform.localScale = vehicleScale;
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

    Vector3 SizeIn(float xIn, float yIn, float zIn)
    {
        return new Vector3(xIn * worldScale, yIn * worldScale, zIn * worldScale);
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
                Vector3 pa = SimToWorld(va.x, va.y);
                Vector3 pb = SimToWorld(vb.x, vb.y);
                Transform tr = vehicleTransforms[i];
                if (tr == null) continue;
                tr.position = Vector3.Lerp(pa, pb, t);

                Vector3 dir = pb - pa;
                dir.y = 0f;
                if (dir.sqrMagnitude > 1e-8f)
                {
                    Quaternion target = Quaternion.LookRotation(dir);
                    tr.rotation = Quaternion.Slerp(tr.rotation, target, 0.4f);
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

    float vehicleY;   // computed at spawn time so vehicles ride on the ground plane
    float stationY;
    float slotY;

    Vector3 SimToWorld(float simX, float simY)
    {
        return new Vector3(simX * worldScale, modelHeight + vehicleY, simY * worldScale);
    }

    Vector3 SimToWorldStation(float simX, float simY)
    {
        return new Vector3(simX * worldScale, modelHeight + stationY, simY * worldScale);
    }

    Vector3 SimToWorldSlot(float simX, float simY)
    {
        return new Vector3(simX * worldScale, modelHeight + slotY, simY * worldScale);
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
