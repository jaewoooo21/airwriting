using UnityEngine;

namespace AirWriting
{
    [RequireComponent(typeof(AirWritingReceiver))]
    public class ArmController : MonoBehaviour
    {
        [Header("Arm Transforms")]
        public Transform upperArm; // S1
        public Transform forearm;  // S2
        public Transform hand;     // S3

        [Header("Visualization")]
        [Tooltip("The point representing the tip of the pen/finger.")]
        public Transform penTip;
        
        [Tooltip("A TrailRenderer or similar component attached to the pen tip.")]
        public TrailRenderer trailRenderer;
        
        public Material writingMaterial;
        public Material idleMaterial;
        public MeshRenderer penRenderer;

        private AirWritingReceiver _receiver;

        [Header("Offsets / Calibration")]
        [Tooltip("Apply an initial rotation offset if the base model doesn't align with the world correctly.")]
        public Vector3 baseRotationOffset = Vector3.zero;
        
        private Quaternion _baseOffsetQuat;

        void Awake()
        {
            _receiver = GetComponent<AirWritingReceiver>();
            _baseOffsetQuat = Quaternion.Euler(baseRotationOffset);
        }

        void OnEnable()
        {
            if (_receiver != null)
                _receiver.OnDataReceived += OnDataReceived;
        }

        void OnDisable()
        {
            if (_receiver != null)
                _receiver.OnDataReceived -= OnDataReceived;
        }

        private void OnDataReceived(AirWritingParsedData data)
        {
            // Apply Rotations
            // If the model hierarchy is parented (UpperArm -> Forearm -> Hand),
            // and the python script returns *absolute world rotations* per sensor,
            // you should assign them to transform.rotation (world rotation), not localRotation.
            
            if (upperArm != null)
                upperArm.rotation = _baseOffsetQuat * data.UpperArmRot;

            if (forearm != null)
                forearm.rotation = _baseOffsetQuat * data.ForearmRot;

            if (hand != null)
                hand.rotation = _baseOffsetQuat * data.HandRot;

            // Optional: If you want to use the Forward Kinematics absolute position 
            // calculated by the Python script instead of Unity's built in hierarchy,
            // you could uncomment this:
            // if (hand != null) hand.position = data.HandPosition;

            // Handle Pen Interaction
            HandlePenState(data);
        }

        private void HandlePenState(AirWritingParsedData data)
        {
            bool isWriting = data.IsPenDown;

            if (trailRenderer != null)
            {
                // Emit trailing particle line only when the pen switch is down
                trailRenderer.emitting = isWriting;
            }

            if (penRenderer != null)
            {
                // Change color based on whether we are writing or floating
                penRenderer.material = isWriting ? writingMaterial : idleMaterial;
            }
        }
    }
}
