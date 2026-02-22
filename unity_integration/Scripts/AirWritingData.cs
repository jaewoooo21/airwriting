using System;
using System.Collections.Generic;
using UnityEngine;

namespace AirWriting
{
    /// <summary>
    /// Represents the JSON payload sent from Python (controller.py)
    /// </summary>
    [Serializable]
    public class AirWritingPacket
    {
        public string t; // e.g. "f" for fusion
        public long ms;  // Timestamp
        
        // S1: Upper Arm
        public float[] S1q; // Quaternion [w, x, y, z]
        public float[] S1e; // Euler [x, y, z]

        // S2: Forearm
        public float[] S2q;
        public float[] S2e;

        // S3: Hand/Pen
        public float[] S3q;
        public float[] S3e;
        
        public float[] S3p; // Position [x, y, z]
        public float[] S3v; // Velocity [x, y, z]
        
        public bool S3z;    // ZUPT active
        public bool S3zaru; // ZARU active
        
        public bool pen;    // Pen switch down state
    }

    /// <summary>
    /// Parsed data structured for easy access by Unity controllers.
    /// Handles Coordinate System conversion from Python (Right-handed, Z-up/Y-up) 
    /// to Unity (Left-handed, Y-up).
    /// </summary>
    public class AirWritingParsedData
    {
        public long Timestamp;
        
        // Unity Rotations (w, x, y, z)
        public Quaternion UpperArmRot;
        public Quaternion ForearmRot;
        public Quaternion HandRot;

        // Unity Position (x, y, z)
        public Vector3 HandPosition;
        public Vector3 HandVelocity;

        public bool IsZuptActive;
        public bool IsPenDown;

        public AirWritingParsedData(AirWritingPacket packet)
        {
            if (packet == null) return;
            
            Timestamp = packet.ms;

            // --- Coordinate System Conversion ---
            // The Madgwick/ESKF in Python often outputs:
            // [w, x, y, z]
            // We need to map this to Unity's coordinate system.
            // Often, this requires swapping Y/Z or negating axes depending on physical mounting.
            // Adjust the mapping below based on the actual physical orientation of the ESP32.
            // Standard Right-to-Left Hand conversion for Quaternions:
            // Unity Q = new Quaternion(-x, -z, -y, w) OR similar depending on base frame.
            
            UpperArmRot = ParseQuaternion(packet.S1q);
            ForearmRot = ParseQuaternion(packet.S2q);
            HandRot = ParseQuaternion(packet.S3q);

            HandPosition = ParseVector3(packet.S3p);
            HandVelocity = ParseVector3(packet.S3v);

            IsZuptActive = packet.S3z;
            IsPenDown = packet.pen;
        }

        private Quaternion ParseQuaternion(float[] pyQ)
        {
            if (pyQ == null || pyQ.Length < 4) return Quaternion.identity;
            
            // Assuming Python delivers [w, x, y, z]
            float w = pyQ[0];
            float x = pyQ[1];
            float y = pyQ[2];
            float z = pyQ[3];

            // Unity format and coordinate conversion 
            // NOTE: This mapping might need tweaking depending on the Python 'world' frame!
            // commonly mapping right-handed Z-up to Unity Y-up: 
            // x_unity = y_py
            // y_unity = z_py
            // z_unity = x_py
            return new Quaternion(-x, -z, -y, w);  // Example conversion mapping, might need adjustment
        }

        private Vector3 ParseVector3(float[] pyV)
        {
            if (pyV == null || pyV.Length < 3) return Vector3.zero;
            
            // X, Y, Z from python to unity.
            float x = pyV[0];
            float y = pyV[1];
            float z = pyV[2];

            // Map python right-handed space to Unity left-handed space
            return new Vector3(x, z, y); // Example conversion mapping
        }
    }
}
