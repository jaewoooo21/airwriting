from airwriting_imu.core.config_loader import ConfigLoader
import sys

try:
    c = ConfigLoader(None).load_all()
    print("IMU Sensors:", c.imu_sensors)
    print("Skeleton:", [s['sensor'] for s in c.skeleton_chain])
    print("FK enabled:", c.fusion.get("forward_kinematics", {}).get("enabled", True))
except Exception as e:
    print("Error:", e)
