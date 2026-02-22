import asyncio
import websockets
import json

async def test_ws():
    try:
        async with websockets.connect("ws://localhost:8765") as ws:
            print("Connected to WS.")
            for _ in range(5):
                msg = await ws.recv()
                d = json.loads(msg)
                print(f"S1e: {d.get('S1e')}")
                print(f"S3e: {d.get('S3e')}")
                print(f"S3fk: {d.get('S3fk')}")
                print("---")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(test_ws())
