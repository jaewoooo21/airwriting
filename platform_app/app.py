import os
import sqlite3
import numpy as np
import yaml
import socket
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools.ml_engine import MLEngine

# Get the directory of this script (web_app)
WEB_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(WEB_APP_DIR, 'comments.db')

# Initialize Flask
app = Flask(__name__, static_folder=WEB_APP_DIR, static_url_path='')

# Logging to file for debugging
import logging
log_file = os.path.join(WEB_APP_DIR, 'app.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def print_and_log(msg):
    print(msg)
    logging.info(msg)

# Initialize ML Engine
ml_engine = MLEngine()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS comments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  author TEXT,
                  text TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

local_ip = get_local_ip()

init_db()

@app.route('/')
def index():
    return send_from_directory(WEB_APP_DIR, 'index.html')

@app.route('/api/comments', methods=['GET'])
def get_comments():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM comments ORDER BY timestamp DESC")
    comments = [{"id": row[0], "author": row[1], "text": row[2], "timestamp": row[3]} for row in c.fetchall()]
    conn.close()
    return jsonify(comments)

@app.route('/api/comments', methods=['POST'])
def add_comment():
    data = request.get_json()
    author = data.get('author', 'Anonymous')
    text = data.get('text', '')
    timestamp = datetime.now().isoformat()
    
    if not text:
        return jsonify({"error": "Comment text is required"}), 400
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO comments (author, text, timestamp) VALUES (?, ?, ?)",
              (author, text, timestamp))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/api/ml/record', methods=['POST'])
def ml_record():
    data = request.get_json()
    label = data.get('label')
    stroke_data = data.get('stroke_full') or data.get('stroke_pos')
    
    if not label or stroke_data is None:
        return jsonify({"error": "Label and stroke_data required"}), 400
        
    # Save to CSV via ML Engine
    success = ml_engine.save_stroke(label, np.array(stroke_data))
    
    # Check if we should trigger training
    training_started = ml_engine.train_background()
    
    stats = ml_engine.get_stats()
    if training_started:
        message = f'Saved {label}. Training started!'
    elif stats.get("status") == "TRAINING":
        message = f'Saved {label}. Training is already running.'
    else:
        message = f'Saved {label}. {stats.get("trainability_reason", "Waiting for more data.")}'
    return jsonify({
        "status": "success",
        "message": message,
        "training_started": training_started,
        "stats": stats,
    })

@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    data = request.get_json()
    stroke_pos = data.get('stroke_pos') # list of [x,y,z]
    
    if not stroke_pos or len(stroke_pos) < 5:
        return jsonify({"predictions": []}), 400
        
    predictions = ml_engine.predict(np.array(stroke_pos), top_n=3)
    
    # Send prediction result to ESP32 OLED and ActionDispatcher via UDP
    if predictions:
        top_pred = predictions[0]
        letter = top_pred['label']
        acc = top_pred['confidence'] * 100
        
        try:
            msg = f"{letter},{acc:.1f}".encode('utf-8')
            udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            # 1. Update OLED (Subnet Broadcast)
            # Try 255.255.255.255 and also a targeted broadcast if possible
            try:
                udp_sock.sendto(msg, ('255.255.255.255', 5555))
                # If we have the local IP, try the subnet broadcast (assuming /24)
                if local_ip:
                    parts = local_ip.split('.')
                    if len(parts) == 4:
                        subnet = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
                        udp_sock.sendto(msg, (subnet, 5555))
            except:
                pass
            
            # 2. Notify Web Relay (12348) and Action Dispatcher (12349)
            action_pkt = json.dumps({
                "type": "recognition",
                "label": letter,
                "confidence": top_pred['confidence']
            }).encode('utf-8')
            udp_sock.sendto(action_pkt, ('127.0.0.1', 12348)) # Web UI
            udp_sock.sendto(action_pkt, ('127.0.0.1', 12349)) # Phone
            
            udp_sock.close()
            print_and_log(f"[Dispatch] Sent to WebRelay(12348), ActionDispatcher(12349) and OLED: {letter} ({acc:.1f}%)")
        except Exception as e:
            import traceback
            print_and_log(f"[Dispatch Failed] {e}\n{traceback.format_exc()}")
            
    return jsonify({
        "predictions": predictions
    })

@app.route('/api/ml/train', methods=['POST'])
def ml_train():
    # Trigger background training manually
    started = ml_engine.train_background()
    stats = ml_engine.get_stats()
    if started:
        message = "Manual training started."
    elif stats.get("status") == "TRAINING":
        message = "Training already in progress."
    else:
        message = f'Training not ready. {stats.get("trainability_reason", "")}'.strip()
    return jsonify({
        "status": "success",
        "message": message,
        "training_started": started,
        "stats": stats,
    })

@app.route('/api/ml/stats', methods=['GET'])
def ml_stats():
    return jsonify(ml_engine.get_stats())

# ════════════════════════════════════════════════════════════════
# Configuration API
# ════════════════════════════════════════════════════════════════
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'system.yaml')

@app.route('/api/config/system', methods=['GET'])
def get_system_config():
    """Retrieve full system.yaml configuration."""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config/system', methods=['PUT'])
def update_system_config():
    """Update system.yaml configuration."""
    data = request.get_json()
    try:
        # Load existing first to merge
        existing_config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                existing_config = yaml.safe_load(f) or {}
        
        # Simple deep-ish merge for top-level keys
        for k, v in data.items():
            if isinstance(v, dict) and k in existing_config and isinstance(existing_config[k], dict):
                existing_config[k].update(v)
            else:
                existing_config[k] = v

        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ip', methods=['GET'])
def get_ip():
    """Returns the local IP of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return jsonify({"ip": IP})

if __name__ == '__main__':
    # Use waitress for a better production-like server
    try:
        from waitress import serve
        print("Starting Flask server on port 5000...")
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        app.run(host='0.0.0.0', port=5000, debug=True)
