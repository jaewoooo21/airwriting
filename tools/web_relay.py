"""
WebSocket Relay Server for AirWriting Web UI (Unified v2.5)
==========================================================
Receives JSON data on:
- UDP 12346: IMU Stream (from main.py)
- UDP 12348: Recognition Results (from main.py)
Broadcasts all received data to connected WebSocket clients on port 18765.
"""

import asyncio
import socket
import logging
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

UDP_IP = "0.0.0.0"
IMU_PORT = 12346
REC_PORT = 12348
WS_HOST = "0.0.0.0"
WS_PORT = 18765

connected_clients = set()

async def ws_handler(websocket):
    """Handle new WebSocket connections."""
    client_id = "Unknown"
    try:
        if hasattr(websocket, 'remote_address') and websocket.remote_address:
            client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logging.info(f"🔗 New Web client connected: {client_id}")
    except Exception:
        logging.info(f"🔗 New Web client connected")
    
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        logging.info(f"❌ Web client disconnected: {client_id}")
        connected_clients.remove(websocket)

async def udp_receiver(port, label):
    """Listen for UDP packets and stream via WebSocket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Enable SO_REUSEADDR for Windows to avoid conflict with other potential listeners
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, port))
    sock.setblocking(False)
    
    logging.info(f"👂 Listening for {label} on {UDP_IP}:{port}...")
    
    loop = asyncio.get_event_loop()
    while True:
        try:
            data, addr = await loop.sock_recvfrom(sock, 4096)
            if connected_clients:
                # Direct broadcast of received byte-string
                msg = data.decode('utf-8')
                websockets.broadcast(connected_clients, msg)
        except Exception as e:
            logging.error(f"UDP Error ({label}): {e}")
            await asyncio.sleep(0.1)

async def main():
    logging.info(f"🚀 Starting Unified AirWriting WebSocket Relay...")
    
    # Start WS server
    await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    logging.info(f"🌐 WebSocket Server running at ws://{WS_HOST}:{WS_PORT}")
    
    # Start UDP listeners for both IMU data and Recognition results
    await asyncio.gather(
        udp_receiver(IMU_PORT, "IMU Data"),
        udp_receiver(REC_PORT, "Recognition Results")
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Shutting down.")
