import asyncio
import websockets
import json

async def monitor_actions():
    uri = "ws://localhost:18800"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to ActionDispatcher!")
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                print(f"\n[RECEIVED] {data.get('type', 'UNKNOWN')}")
                print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(monitor_actions())
    except KeyboardInterrupt:
        print("\nStopping monitor...")
