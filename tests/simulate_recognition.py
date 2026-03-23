import socket
import json

def simulate_recognition(label, confidence):
    port = 12349
    payload = {
        "type": "recognition",
        "label": label,
        "confidence": confidence,
        "timestamp": 123456789.0
    }
    raw = json.dumps(payload).encode('utf-8')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(raw, ("127.0.0.1", port))
    print(f"Sent simulation: {label} ({confidence*100}%) to port {port}")

if __name__ == "__main__":
    simulate_recognition("C", 0.95)
