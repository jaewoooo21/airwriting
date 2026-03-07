import asyncio
import websockets

async def test():
    try:
        print("Connecting to ws://10.191.239.131:18800")
        async with websockets.connect('ws://10.191.239.131:18800') as ws:
            print("Connected! Waiting for 5 seconds...")
            
            # Wait for config
            res = await ws.recv()
            print("Received:", res[:50] + "...")
            
            # Stay alive
            await asyncio.sleep(5)
            print("Still alive after 5 seconds!")
            
    except Exception as e:
        print("Failed with exception:", repr(e))

if __name__ == "__main__":
    asyncio.run(test())
