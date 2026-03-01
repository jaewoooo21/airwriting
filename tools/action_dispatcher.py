"""
Action Dispatcher — AirWriting Keyword → Phone Action Server
=============================================================
Receives recognized characters from the AirWriting engine and dispatches
them as keyword-mapped actions to connected mobile devices via WebSocket.

Architecture:
  Controller (pen-up) → ML predict → Action Dispatcher → Phone App (WS)

Usage:
  python tools/action_dispatcher.py

  # Or via start_all.bat (auto-launched)
"""
import asyncio
import json
import logging
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import websockets
except ImportError:
    print("❌ websockets not installed. Run: pip install websockets")
    sys.exit(1)

from tools.ml_engine import MLEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("ActionDispatcher")

import yaml

# ════════════════════════════════════════════════════════════════
# Keyword-Action Mapping
# ════════════════════════════════════════════════════════════════
ACTION_MAP = {}
CONFIDENCE_THRESHOLD = 0.7

try:
    _config_path = Path(__file__).parent.parent / "config" / "system.yaml"
    with open(_config_path, encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
        _ad = _cfg.get("action_dispatch", {})
        if "action_map" in _ad:
            ACTION_MAP = _ad["action_map"]
            # Ensure description fields exist to prevent UI errors
            for k, v in ACTION_MAP.items():
                if "description" not in v:
                    v["description"] = f"Action: {v.get('keyword', k)}"
        if "confidence_threshold" in _ad:
            CONFIDENCE_THRESHOLD = float(_ad["confidence_threshold"])
except Exception as e:
    log.error(f"Failed to load system.yaml config: {e}")

# Minimum confidence to trigger an action (fallback if unconfigured)
CONFIDENCE_THRESHOLD = 0.7

# WebSocket server for phone connections
WS_HOST = "0.0.0.0"
WS_PORT = 18800

# UDP listener for receiving recognized characters from engine
UDP_PORT = 12348


# ════════════════════════════════════════════════════════════════
# Connected Phone Clients
# ════════════════════════════════════════════════════════════════
connected_phones = set()
action_history = []


async def phone_handler(websocket):
    """Handle phone app WebSocket connections."""
    client_id = "Phone"
    try:
        if hasattr(websocket, 'remote_address') and websocket.remote_address:
            client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    except Exception:
        pass

    log.info(f"📱 Phone connected: {client_id}")
    connected_phones.add(websocket)

    # Send current action map on connect
    try:
        await websocket.send(json.dumps({
            "type": "config",
            "action_map": ACTION_MAP,
            "threshold": CONFIDENCE_THRESHOLD,
        }))
    except Exception:
        pass

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_phone_receive(websocket, client_id))
            tg.create_task(_phone_keepalive(websocket))
    except* Exception as eg:
        log.error(f"📱 Exception in phone_handler: {eg.exceptions}")
    finally:
        log.info(f"📱 Phone disconnected: {client_id}")
        connected_phones.discard(websocket)


async def _phone_receive(websocket, client_id):
    """Receive messages from phone (e.g. status updates, custom mappings)."""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type", "")
                if msg_type == "ack":
                    log.info(f"📱 {client_id} ACK: {data.get('keyword', '?')}")
                elif msg_type == "custom_map":
                    # Allow phone to add custom mappings
                    letter = data.get("letter", "").upper()
                    keyword = data.get("keyword", "")
                    intent = data.get("intent", "")
                    if letter and keyword:
                        ACTION_MAP[letter] = {
                            "keyword": keyword,
                            "intent": intent,
                            "description": f"Custom: {keyword}",
                        }
                        log.info(f"📱 Custom mapping added: {letter} → {keyword}")
            except json.JSONDecodeError:
                pass
    except websockets.ConnectionClosed:
        pass


async def _phone_keepalive(websocket):
    """Send periodic pings to keep the connection alive."""
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.ping()
    except (websockets.ConnectionClosed, asyncio.CancelledError):
        pass


# ════════════════════════════════════════════════════════════════
# Action Dispatch Logic
# ════════════════════════════════════════════════════════════════
async def dispatch_action(label: str, confidence: float):
    """Look up the keyword for a recognized label and broadcast to phones.

    Args:
        label: recognized character (e.g. "C")
        confidence: ML confidence score (0-1)
    """
    if confidence < CONFIDENCE_THRESHOLD:
        log.info(f"⚠️ Confidence too low: '{label}' ({confidence:.1%}) < {CONFIDENCE_THRESHOLD:.0%}")
        return

    action = ACTION_MAP.get(label.upper())
    if action is None:
        log.info(f"❓ No action mapped for '{label}'")
        # Still notify phones about the recognition (no action)
        payload = {
            "type": "recognition",
            "label": label,
            "confidence": confidence,
            "action": None,
            "message": f"'{label}' recognized but no action mapped",
        }
    else:
        log.info(
            f"🚀 ACTION: '{label}' → {action['keyword']} "
            f"(confidence={confidence:.1%}, intent={action['intent']})"
        )
        payload = {
            "type": "action",
            "label": label,
            "confidence": confidence,
            "keyword": action["keyword"],
            "intent": action["intent"],
            "description": action["description"],
        }

    # Record history
    action_history.append(payload)
    if len(action_history) > 100:
        action_history.pop(0)

    # Broadcast to all connected phones
    if connected_phones:
        message = json.dumps(payload)
        websockets.broadcast(connected_phones, message)
        log.info(f"📡 Broadcast to {len(connected_phones)} phone(s)")
    else:
        log.warning("📱 No phones connected — action not delivered")


# ════════════════════════════════════════════════════════════════
# UDP Listener (receives recognition results from engine)
# ════════════════════════════════════════════════════════════════
async def udp_listener():
    """Listen for recognition results on UDP port.

    Expected JSON format:
        {"type": "recognition", "label": "C", "confidence": 0.92}
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.setblocking(False)

    log.info(f"👂 Listening for recognition results on UDP :{UDP_PORT}")

    loop = asyncio.get_event_loop()
    while True:
        try:
            data, addr = await loop.sock_recvfrom(sock, 4096)
            msg = json.loads(data.decode("utf-8"))

            if msg.get("type") == "recognition":
                label = msg.get("label", "")
                confidence = msg.get("confidence", 0.0)
                await dispatch_action(label, confidence)
        except ConnectionResetError:
            pass
        except json.JSONDecodeError:
            pass
        except Exception as e:
            log.error(f"UDP error: {e}")
            await asyncio.sleep(0.1)


# ════════════════════════════════════════════════════════════════
# HTTP Status Endpoint (simple)
# ════════════════════════════════════════════════════════════════
def get_status():
    """Return current status as dict."""
    return {
        "connected_phones": len(connected_phones),
        "action_map": ACTION_MAP,
        "threshold": CONFIDENCE_THRESHOLD,
        "history_count": len(action_history),
        "recent_actions": action_history[-5:] if action_history else [],
    }


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
async def main():
    log.info("=" * 55)
    log.info("  📱 AirWriting Action Dispatcher")
    log.info(f"  WebSocket: ws://{WS_HOST}:{WS_PORT}")
    log.info(f"  UDP Input: :{UDP_PORT}")
    log.info(f"  Actions: {list(ACTION_MAP.keys())}")
    log.info("=" * 55)

    # Print action map
    log.info("📋 Action Map:")
    for letter, action in ACTION_MAP.items():
        log.info(f"  {letter} → {action['keyword']:10s} ({action['description']})")

    # Start WebSocket server for phone connections
    ws_server = await websockets.serve(phone_handler, WS_HOST, WS_PORT)
    log.info(f"🌐 WebSocket server ready on ws://{WS_HOST}:{WS_PORT}")

    # Start UDP listener for recognition results
    await udp_listener()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("🛑 Action Dispatcher stopped.")
