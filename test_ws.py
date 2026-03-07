import asyncio
import websockets

async def test():
    try:
        async with websockets.connect('ws://127.0.0.1:18800') as ws:
            print("Connected!")
            res = await ws.recv()
            print("Received:", res)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    asyncio.run(test())
