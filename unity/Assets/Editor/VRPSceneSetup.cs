using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

// Editor-time scene builders. These create real GameObjects in the scene so you can see them
// in the hierarchy and tweak them before pressing Play. Run from Tools → VRP → ...
public static class VRPSceneSetup
{
    const string TableName = "Table";
    const string FloorName = "Floor";
    const string GeneratedFolder = "Assets/_GeneratedMaterials";

    [MenuItem("Tools/VRP/Build Table")]
    public static void BuildTable()
    {
        var driver = Object.FindFirstObjectByType<PlaybackDriver>();
        if (driver == null)
        {
            EditorUtility.DisplayDialog("Build Table",
                "No PlaybackDriver found in the scene. Open the VRPSim scene first.", "OK");
            return;
        }

        // Re-build: remove any existing Table child and recreate.
        var existing = driver.transform.Find(TableName);
        if (existing != null) Object.DestroyImmediate(existing.gameObject);

        var tableGo = new GameObject(TableName);
        Undo.RegisterCreatedObjectUndo(tableGo, "Build Table");
        tableGo.transform.SetParent(driver.transform, false);
        tableGo.transform.localPosition = Vector3.zero;
        tableGo.transform.localRotation = Quaternion.identity;
        tableGo.transform.localScale = Vector3.one;

        var topMat = MakeMaterial("VRP/TableTop", driver.tableTopColor);
        var legMat = MakeMaterial("VRP/TableLeg", driver.tableLegColor);

        // Tabletop slab: top surface at y = tableHeightMeters, extending downward.
        float topCenterY = driver.tableHeightMeters - (driver.tableThicknessMeters * 0.5f);
        var top = GameObject.CreatePrimitive(PrimitiveType.Cube);
        top.name = "Tabletop";
        Object.DestroyImmediate(top.GetComponent<Collider>());
        top.transform.SetParent(tableGo.transform, false);
        top.transform.localPosition = new Vector3(0f, topCenterY, 0f);
        top.transform.localScale = new Vector3(driver.tableSizeMeters, driver.tableThicknessMeters, driver.tableSizeMeters);
        top.GetComponent<Renderer>().sharedMaterial = topMat;

        // Legs from floor up to underside of slab.
        float legHeight = Mathf.Max(0.01f, driver.tableHeightMeters - driver.tableThicknessMeters);
        float legY = legHeight * 0.5f;
        float legOffset = (driver.tableSizeMeters * 0.5f) - driver.tableLegInsetMeters - (driver.tableLegThicknessMeters * 0.5f);
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
            Object.DestroyImmediate(leg.GetComponent<Collider>());
            leg.transform.SetParent(tableGo.transform, false);
            leg.transform.localPosition = new Vector3(legXZ[i].x, legY, legXZ[i].y);
            leg.transform.localScale = new Vector3(driver.tableLegThicknessMeters, legHeight, driver.tableLegThicknessMeters);
            leg.GetComponent<Renderer>().sharedMaterial = legMat;
        }

        // Also pre-create an empty _SimRoot at the right height so it's visible in the hierarchy.
        var simRootName = "_SimRoot";
        var simRootExisting = driver.transform.Find(simRootName);
        if (simRootExisting == null)
        {
            var simRoot = new GameObject(simRootName);
            Undo.RegisterCreatedObjectUndo(simRoot, "Create _SimRoot");
            simRoot.transform.SetParent(driver.transform, false);
            simRoot.transform.localPosition = new Vector3(0f, driver.tableHeightMeters, 0f);
            simRoot.transform.localRotation = Quaternion.identity;
            simRoot.transform.localScale = Vector3.one;
        }
        else
        {
            simRootExisting.localPosition = new Vector3(0f, driver.tableHeightMeters, 0f);
        }

        // Make sure runtime doesn't try to also build a table (we just authored one).
        if (driver.generateTable)
        {
            Undo.RecordObject(driver, "Disable runtime table");
            driver.generateTable = false;
        }

        EditorSceneManager.MarkSceneDirty(driver.gameObject.scene);
        Selection.activeGameObject = tableGo;
        Debug.Log($"[VRP] Built Table: {driver.tableSizeMeters:F2}m square, top at y={driver.tableHeightMeters:F2}m");
    }

    [MenuItem("Tools/VRP/Remove Table")]
    public static void RemoveTable()
    {
        var driver = Object.FindFirstObjectByType<PlaybackDriver>();
        if (driver == null) return;
        var existing = driver.transform.Find(TableName);
        if (existing != null)
        {
            Undo.DestroyObjectImmediate(existing.gameObject);
            EditorSceneManager.MarkSceneDirty(driver.gameObject.scene);
        }
    }

    // Saves materials under Assets/_GeneratedMaterials/<name>.mat so Unity doesn't recreate
    // a fresh instance each time. Color updates propagate by overwriting the asset.
    static Material MakeMaterial(string name, Color color)
    {
        EnsureGeneratedFolder();
        string path = $"{GeneratedFolder}/{name.Replace('/', '_')}.mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        var shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        if (mat == null)
        {
            mat = new Material(shader);
            AssetDatabase.CreateAsset(mat, path);
        }
        if (mat.shader != shader) mat.shader = shader;
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", color);
        mat.color = color;
        EditorUtility.SetDirty(mat);
        return mat;
    }

    static void EnsureGeneratedFolder()
    {
        if (!AssetDatabase.IsValidFolder(GeneratedFolder))
            AssetDatabase.CreateFolder("Assets", "_GeneratedMaterials");
    }

    // ---------- Stage 2: Floor + Skybox ----------

    [MenuItem("Tools/VRP/Build Floor")]
    public static void BuildFloor()
    {
        var scene = EditorSceneManager.GetActiveScene();

        // Remove any existing root-level Floor in this scene.
        foreach (var go in scene.GetRootGameObjects())
        {
            if (go.name == FloorName)
            {
                Undo.DestroyObjectImmediate(go);
                break;
            }
        }

        // Unity's built-in Plane is 10m x 10m at scale (1,1,1). Scale 2x for 20m x 20m.
        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        Undo.RegisterCreatedObjectUndo(floor, "Build Floor");
        floor.name = FloorName;
        floor.transform.position = Vector3.zero;
        floor.transform.localScale = new Vector3(2f, 1f, 2f);

        var gridTex = MakeOrLoadGridTexture("FloorGrid", 512, 16);
        var floorMat = MakeOrLoadFloorMaterial("Floor", gridTex);
        floor.GetComponent<Renderer>().sharedMaterial = floorMat;

        EditorSceneManager.MarkSceneDirty(scene);
        Selection.activeGameObject = floor;
        Debug.Log("[VRP] Built Floor (20m x 20m).");
    }

    [MenuItem("Tools/VRP/Remove Floor")]
    public static void RemoveFloor()
    {
        var scene = EditorSceneManager.GetActiveScene();
        foreach (var go in scene.GetRootGameObjects())
        {
            if (go.name == FloorName)
            {
                Undo.DestroyObjectImmediate(go);
                EditorSceneManager.MarkSceneDirty(scene);
                return;
            }
        }
    }

    [MenuItem("Tools/VRP/Set Procedural Skybox")]
    public static void SetProceduralSkybox()
    {
        EnsureGeneratedFolder();
        const string path = GeneratedFolder + "/Skybox.mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        var shader = Shader.Find("Skybox/Procedural");
        if (shader == null)
        {
            Debug.LogError("[VRP] Skybox/Procedural shader not found. Built-in Skybox shaders may have been stripped from this project.");
            return;
        }
        if (mat == null)
        {
            mat = new Material(shader);
            AssetDatabase.CreateAsset(mat, path);
        }
        mat.shader = shader;
        if (mat.HasProperty("_SkyTint")) mat.SetColor("_SkyTint", new Color(0.55f, 0.70f, 0.95f));
        if (mat.HasProperty("_GroundColor")) mat.SetColor("_GroundColor", new Color(0.30f, 0.32f, 0.35f));
        if (mat.HasProperty("_AtmosphereThickness")) mat.SetFloat("_AtmosphereThickness", 1.0f);
        if (mat.HasProperty("_Exposure")) mat.SetFloat("_Exposure", 1.1f);
        if (mat.HasProperty("_SunDisk")) mat.SetInt("_SunDisk", 2);
        if (mat.HasProperty("_SunSize")) mat.SetFloat("_SunSize", 0.04f);
        EditorUtility.SetDirty(mat);

        RenderSettings.skybox = mat;
        RenderSettings.ambientMode = AmbientMode.Skybox;
        RenderSettings.ambientIntensity = 1.0f;
        DynamicGI.UpdateEnvironment();

        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log("[VRP] Procedural skybox assigned. Edit colors on " + path + ".");
    }

    [MenuItem("Tools/VRP/Build Floor + Skybox")]
    public static void BuildFloorAndSkybox()
    {
        BuildFloor();
        SetProceduralSkybox();
    }

    // ---------- Stage 3: XR Locomotion ----------

    [MenuItem("Tools/VRP/Add XR Locomotion")]
    public static void AddXRLocomotion()
    {
        var rig = FindCameraRig();
        if (rig == null)
        {
            EditorUtility.DisplayDialog("Add XR Locomotion",
                "Could not find an OVR Camera Rig in the scene. Open VRPSim and confirm the [BuildingBlock] Camera Rig is present.",
                "OK");
            return;
        }
        var loco = rig.GetComponent<XRLocomotion>();
        if (loco == null)
        {
            loco = Undo.AddComponent<XRLocomotion>(rig);
            Debug.Log("[VRP] Added XRLocomotion to " + rig.name);
        }
        else
        {
            Debug.Log("[VRP] XRLocomotion already present on " + rig.name);
        }
        Selection.activeGameObject = rig;
        EditorSceneManager.MarkSceneDirty(rig.scene);
    }

    [MenuItem("Tools/VRP/Disable Legacy Main Camera")]
    public static void DisableLegacyMainCamera()
    {
        var scene = EditorSceneManager.GetActiveScene();
        int disabled = 0;
        foreach (var go in scene.GetRootGameObjects())
        {
            if (go.name == "Main Camera")
            {
                Undo.RecordObject(go, "Disable Main Camera");
                go.SetActive(false);
                disabled++;
            }
        }
        if (disabled == 0)
            Debug.Log("[VRP] No top-level 'Main Camera' object to disable.");
        else
        {
            Debug.Log($"[VRP] Disabled {disabled} legacy Main Camera object(s). The OVR rig's CenterEyeAnchor camera will be used.");
            EditorSceneManager.MarkSceneDirty(scene);
        }
    }

    static GameObject FindCameraRig()
    {
        var scene = EditorSceneManager.GetActiveScene();
        foreach (var root in scene.GetRootGameObjects())
        {
            var found = FindCameraRigRecursive(root.transform);
            if (found != null) return found.gameObject;
        }
        return null;
    }

    static Transform FindCameraRigRecursive(Transform t)
    {
        // Match the Meta Building Blocks Camera Rig either by name or by an OVRCameraRig component.
        if (t.name.IndexOf("Camera Rig", System.StringComparison.OrdinalIgnoreCase) >= 0)
            return t;
        var ovr = t.GetComponent("OVRCameraRig");
        if (ovr != null) return t;
        for (int i = 0; i < t.childCount; i++)
        {
            var found = FindCameraRigRecursive(t.GetChild(i));
            if (found != null) return found;
        }
        return null;
    }

    // Generates a soft grid texture (light gray base, slightly darker grid lines) once and caches it as a PNG asset.
    static Texture2D MakeOrLoadGridTexture(string name, int size, int cells)
    {
        EnsureGeneratedFolder();
        string path = $"{GeneratedFolder}/{name}.png";
        var existing = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        if (existing != null) return existing;

        var tex = new Texture2D(size, size, TextureFormat.RGBA32, true);
        var baseColor = new Color(0.55f, 0.55f, 0.58f);
        var lineColor = new Color(0.40f, 0.40f, 0.43f);
        int cellPx = Mathf.Max(2, size / cells);
        int lineHalf = Mathf.Max(1, cellPx / 64);
        var pixels = new Color32[size * size];
        var b32 = (Color32)baseColor;
        var l32 = (Color32)lineColor;
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int dx = x % cellPx;
                int dy = y % cellPx;
                bool onLine = dx <= lineHalf || dx >= cellPx - lineHalf
                           || dy <= lineHalf || dy >= cellPx - lineHalf;
                pixels[y * size + x] = onLine ? l32 : b32;
            }
        }
        tex.SetPixels32(pixels);
        tex.Apply();

        File.WriteAllBytes(path, tex.EncodeToPNG());
        AssetDatabase.ImportAsset(path);
        var importer = (TextureImporter)AssetImporter.GetAtPath(path);
        if (importer != null)
        {
            importer.wrapMode = TextureWrapMode.Repeat;
            importer.filterMode = FilterMode.Bilinear;
            importer.mipmapEnabled = true;
            importer.SaveAndReimport();
        }
        return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
    }

    static Material MakeOrLoadFloorMaterial(string name, Texture2D albedo)
    {
        EnsureGeneratedFolder();
        string path = $"{GeneratedFolder}/{name}.mat";
        var mat = AssetDatabase.LoadAssetAtPath<Material>(path);
        var shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null) shader = Shader.Find("Standard");
        if (mat == null)
        {
            mat = new Material(shader);
            AssetDatabase.CreateAsset(mat, path);
        }
        mat.shader = shader;
        if (mat.HasProperty("_BaseMap")) mat.SetTexture("_BaseMap", albedo);
        if (mat.HasProperty("_MainTex")) mat.SetTexture("_MainTex", albedo);
        // Tile the texture so on a 20m floor you get a sensible cell density (one tile per ~2m).
        var tile = new Vector2(10f, 10f);
        if (mat.HasProperty("_BaseMap")) mat.SetTextureScale("_BaseMap", tile);
        if (mat.HasProperty("_MainTex")) mat.SetTextureScale("_MainTex", tile);
        if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", 0.1f);
        if (mat.HasProperty("_Metallic")) mat.SetFloat("_Metallic", 0.0f);
        if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", Color.white);
        EditorUtility.SetDirty(mat);
        return mat;
    }
}
