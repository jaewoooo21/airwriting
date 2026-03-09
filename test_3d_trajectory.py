import asyncio
import websockets
import json
import math
import time

async def send_mock_trajectory():
    uri = "ws://localhost:18800"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Simulate a 3D spiral
            for t in range(0, 200):
                angle = t * 0.1
                r = 0.05 + 0.001 * t
                
                # Create a spiral moving forward in Z
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                z = t * 0.002
                
                # Mock IMU stream data
                data = {
                    "type": "imu_stream",
                    "pos": {"x": x, "y": y, "z": z},
                    "pen": True,
                    # Add mock S1, S2, S3 quaternions just to keep hand-widget happy
                    "S1q": [1, 0, 0, 0],
                    "S2q": [1, 0, 0, 0],
                    "S3q": [1, 0, 0, 0]
                }
                
                await websocket.send(json.dumps(data))
                print(f"Sent pos: {x:.3f}, {y:.3f}, {z:.3f}")
                await asyncio.sleep(0.02)  # 50Hz
                
            # Pen up
            data = {"type": "imu_stream", "pos": {"x": x, "y": y, "z": z}, "pen": False}
            await websocket.send(json.dumps(data))
            print("Pen up sent. Test finished.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(send_mock_trajectory())
