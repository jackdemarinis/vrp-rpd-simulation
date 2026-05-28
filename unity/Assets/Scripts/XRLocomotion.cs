using UnityEngine;

// Stick-based locomotion for the Meta Building Blocks Camera Rig.
// Attach to the [BuildingBlock] Camera Rig GameObject (the root of the OVR rig hierarchy).
//   Left thumbstick  -> walk in the head-facing direction
//   Right thumbstick -> smooth-turn around the head's vertical axis
// Reads OVRInput from com.meta.xr.sdk.core; no XR Interaction Toolkit dependency.
[DisallowMultipleComponent]
public class XRLocomotion : MonoBehaviour
{
    [Header("Movement (left stick)")]
    public bool enableMovement = true;
    [Tooltip("Walking speed in meters per second when stick is fully pushed.")]
    public float moveSpeed = 1.5f;
    [Tooltip("Hold the right grip to multiply move speed by this factor (run).")]
    public float runMultiplier = 2.0f;

    [Header("Turning (right stick)")]
    public bool enableTurn = true;
    [Tooltip("Smooth turn rate in degrees per second when stick is fully pushed sideways.")]
    public float turnSpeed = 90f;

    [Header("Input")]
    [Range(0f, 0.5f)]
    [Tooltip("Stick values below this are treated as zero (handles drift).")]
    public float stickDeadzone = 0.15f;

    [Header("References (auto-found if blank)")]
    [Tooltip("CenterEyeAnchor under TrackingSpace. Used as movement-direction reference and turn pivot.")]
    public Transform centerEyeAnchor;

    void Awake()
    {
        if (centerEyeAnchor == null)
        {
            var t = transform.Find("TrackingSpace/CenterEyeAnchor");
            if (t != null) centerEyeAnchor = t;
            else Debug.LogWarning("[XRLocomotion] CenterEyeAnchor not found under TrackingSpace. " +
                                  "Movement direction will fall back to rig forward.");
        }
    }

    void Update()
    {
        Vector2 leftStick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);
        Vector2 rightStick = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick);

        if (enableMovement)
        {
            float mag = leftStick.magnitude;
            if (mag > stickDeadzone)
            {
                float scaled = Mathf.Clamp01((mag - stickDeadzone) / (1f - stickDeadzone));
                Vector2 stick = leftStick.normalized * scaled;

                Transform headT = centerEyeAnchor != null ? centerEyeAnchor : transform;
                Vector3 forward = headT.forward; forward.y = 0f; forward.Normalize();
                Vector3 right = headT.right; right.y = 0f; right.Normalize();
                Vector3 dir = forward * stick.y + right * stick.x;

                float speed = moveSpeed;
                if (runMultiplier > 1f && OVRInput.Get(OVRInput.Button.SecondaryHandTrigger))
                    speed *= runMultiplier;

                transform.position += dir * (speed * Time.deltaTime);
            }
        }

        if (enableTurn)
        {
            float ax = rightStick.x;
            if (Mathf.Abs(ax) > stickDeadzone)
            {
                float scaled = Mathf.Sign(ax) * Mathf.Clamp01((Mathf.Abs(ax) - stickDeadzone) / (1f - stickDeadzone));
                float deltaAngle = scaled * turnSpeed * Time.deltaTime;
                Vector3 pivot = centerEyeAnchor != null ? centerEyeAnchor.position : transform.position;
                transform.RotateAround(pivot, Vector3.up, deltaAngle);
            }
        }
    }
}
