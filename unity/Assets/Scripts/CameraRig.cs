using UnityEngine;

[RequireComponent(typeof(Camera))]
public class CameraRig : MonoBehaviour
{
    [Header("WASD Pan")]
    [Tooltip("Pan speed scales with zoom. 1 = pan ~one view-height per second.")]
    public float panSpeed = 1f;
    public float shiftMultiplier = 3f;

    [Header("Zoom (mouse wheel + Q/E)")]
    [Tooltip("Fraction of zoom per scroll tick.")]
    public float scrollZoomSpeed = 0.15f;
    [Tooltip("Fraction of zoom per second when holding Q or E.")]
    public float keyZoomSpeed = 0.5f;

    [Header("Limits (orthographic)")]
    public float minOrthoSize = 0.15f;
    public float maxOrthoSize = 4f;

    [Header("Limits (perspective)")]
    public float minHeight = 0.3f;
    public float maxHeight = 10f;

    Camera cam;

    void Awake()
    {
        cam = GetComponent<Camera>();
    }

    void Update()
    {
        if (cam == null) return;
        HandlePan();
        HandleZoom();
    }

    void HandlePan()
    {
        float forward = (Input.GetKey(KeyCode.W) ? 1f : 0f) - (Input.GetKey(KeyCode.S) ? 1f : 0f);
        float right   = (Input.GetKey(KeyCode.D) ? 1f : 0f) - (Input.GetKey(KeyCode.A) ? 1f : 0f);
        if (forward == 0f && right == 0f) return;

        float speed = panSpeed * (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)
            ? shiftMultiplier
            : 1f);

        // Make pan speed scale with current zoom so panning feels the same at any zoom.
        float scale = cam.orthographic ? cam.orthographicSize : Mathf.Max(0.5f, transform.position.y);

        // Pan in world XZ regardless of camera tilt — feels natural for a top-down view.
        Vector3 dir = new Vector3(right, 0f, forward).normalized;
        transform.position += dir * speed * scale * Time.unscaledDeltaTime;
    }

    void HandleZoom()
    {
        float scroll = Input.mouseScrollDelta.y;
        float keyZoom = (Input.GetKey(KeyCode.E) ? 1f : 0f) - (Input.GetKey(KeyCode.Q) ? 1f : 0f);
        float zoomDelta = -scroll * scrollZoomSpeed - keyZoom * keyZoomSpeed * Time.unscaledDeltaTime;
        if (Mathf.Abs(zoomDelta) < 1e-6f) return;

        if (cam.orthographic)
        {
            cam.orthographicSize = Mathf.Clamp(
                cam.orthographicSize * (1f + zoomDelta),
                minOrthoSize, maxOrthoSize);
        }
        else
        {
            Vector3 p = transform.position;
            p.y = Mathf.Clamp(p.y * (1f + zoomDelta), minHeight, maxHeight);
            transform.position = p;
        }
    }
}
