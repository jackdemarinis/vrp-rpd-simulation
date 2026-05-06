using System;
using UnityEngine;

// Mirrors the JSON written by vrp_rpd_sim/unity_export.py.
// Field names are case-sensitive and must match the JSON exactly.
[Serializable]
public class PlaybackData
{
    public int schemaVersion;
    public string generatedAtUtc;
    public CoordinateMapping coordinateMapping;
    public float worldSizeIn;
    public float roadEnvelopeWidthIn;
    public float laneCenterOffsetIn;
    public float[] roadXs;
    public float[] roadYs;
    public int[] activeJobIds;
    public int vehicleCapacity;
    public int vehicleCount;
    public float alvikSizeIn;
    public float alvikSpeedInPerSec;
    public float plannedMakespanSec;
    public float actualCompletionTimeSec;
    public float captureIntervalSec;
    public DepotData depot;
    public StationSpec[] stations;
    public VehicleSpec[] vehicles;
    public Frame[] frames;
    public Summary summary;
}

[Serializable]
public class CoordinateMapping
{
    public string pygameXToUnity;
    public string pygameYToUnity;
    public string unityHeightAxis;
    public string sourceUnits;
    public float recommendedUnityScale;
}

[Serializable]
public class DepotData
{
    public string nodeId;
    public Point accessPoint;
    public Point anchorPoint;
    public DepotSlot[] slots;
    public DepotEntry[] entries;
}

[Serializable]
public class Point
{
    public float x;
    public float y;
}

[Serializable]
public class DepotSlot
{
    public int slotIndex;
    public float x;
    public float y;
}

[Serializable]
public class DepotEntry
{
    public string entryId;
    public string group;
    public float x;
    public float y;
}

[Serializable]
public class StationSpec
{
    public int stationId;
    public string name;
    public string nodeId;
    public float x;
    public float y;
    public float processingTimeSec;
    public bool active;
}

[Serializable]
public class VehicleSpec
{
    public int vehicleId;
    public string displayName;
    public int[] colorRgb;
    public float startX;
    public float startY;
    public int startLoad;
    public RouteOp[] route;
}

[Serializable]
public class RouteOp
{
    public int customerId;
    public string kind;
}

[Serializable]
public class Frame
{
    public float timeSec;
    public int completedJobs;
    public int activeJobs;
    public int vehiclesCompleted;
    public VehicleSnapshot[] vehicles;
    public StationSnapshot[] stations;
}

[Serializable]
public class VehicleSnapshot
{
    public int vehicleId;
    public float x;
    public float y;
    public int load;
    public int routeIndex;
    public string currentNode;
    public string targetNode;
    public int waitingCustomerId;
    public bool completed;
    public float completionTimeSec;
    public float homeX;
    public float homeY;
    public bool hasActivePath;
}

[Serializable]
public class StationSnapshot
{
    public int stationId;
    public string state;
    public int[] colorRgb;
    public float droppedAtSec;
    public float readyAtSec;
    public float pickedAtSec;
}

[Serializable]
public class Summary
{
    public string selectedLabel;
    public string bestLabel;
    public int frameCount;
    public bool simulationCompleted;
    public float completionTimeSec;
    public float terminatedAtSec;
    public string terminationReason;
    public SolverMakespans solverMakespansSec;
}

[Serializable]
public class SolverMakespans
{
    public float initial;
    public float alns;
    public float brkga;
    public float best;
}
