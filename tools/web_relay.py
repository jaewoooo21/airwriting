"""
WebSocket Relay Server for AirWriting Web UI
============================================
Receives JSON data on UDP port 12346 (from main.py) and broadcasts it
to all connected WebSocket clients on port 8765. This allows a browser
to receive the exact same 3D tracking data as the PyQt digital twin.
"""

import asyncio
import json
import socket
import logging
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

UDP_IP = "0.0.0.0"
UDP_PORT = 12346
WS_HOST = "localhost"
WS_PORT = 18765

connected_clients = set()

async def ws_handler(websocket):
    """Handle new WebSocket connections."""
    try:
        # WebSockets v11+ doesn't use remote_address directly if it's behind a proxy, but we can try
        client_id = f"Client"
        if hasattr(websocket, 'remote_address') and websocket.remote_address:
            client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logging.info(f"🔗 New Web client connected: {client_id}")
    except Exception:
        logging.info(f"🔗 New Web client connected")
    connected_clients.add(websocket)
    try:
        # Keep connection alive
        await websocket.wait_closed()
    finally:
        logging.info(f"❌ Web client disconnected: {client_id}")
        connected_clients.remove(websocket)

async def udp_receiver():
    """Listen for UDP packets from main.py and stream via WebSocket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    
    logging.info(f"👂 Listening for engine UDP packets on {UDP_IP}:{UDP_PORT}...")
    
    loop = asyncio.get_event_loop()
    debug_prints = 5
    
    while True:
        try:
            data, addr = await loop.sock_recvfrom(sock, 4096)
            
            if debug_prints > 0:
                logging.info(f"📦 RAW UDP (sample {debug_prints}): {data.decode('utf-8')[:150]}")
                debug_prints -= 1

            # Only broadcast to connected clients
            if connected_clients:
                # Need to run in an async context
                # websockets.broadcast is safe for concurrent sets
                websockets.broadcast(connected_clients, data.decode('utf-8'))
        except ConnectionResetError:
            # Windows UDP port unreachable error
            pass
        except Exception as e:
            logging.error(f"UDP Error: {e}")
            await asyncio.sleep(0.1)

async def main():
    logging.info(f"🚀 Starting AirWriting WebSocket Relay...")
    
    # Start WS server
    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    logging.info(f"🌐 WebSocket Server running at ws://{WS_HOST}:{WS_PORT}")
    
    # Start UDP listener
    await udp_receiver()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Shutting down.")
