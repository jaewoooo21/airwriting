import os
import sqlite3
import threading
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools.ml_engine import MLEngine

# Get the directory of this script (web_app)
WEB_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(WEB_APP_DIR, 'comments.db')

# Initialize Flask, treating the web_app directory as the static folder
app = Flask(__name__, static_folder=WEB_APP_DIR, static_url_path='')

# Initialize ML Engine
ml_engine = MLEngine()

def init_db():
    """Initialize the SQLite database and create the comments table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

@app.route('/')
def index():
    """Serve the main index.html file."""
    return send_from_directory(WEB_APP_DIR, 'index.html')

@app.route('/api/comments', methods=['GET'])
def get_comments():
    """API endpoint to retrieve all comments."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT id, author, content, timestamp FROM comments ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    
    comments = []
    for row in rows:
        comments.append({
            'id': row['id'],
            'author': row['author'],
            'content': row['content'],
            'timestamp': row['timestamp']
        })
    return jsonify(comments)

@app.route('/api/comments', methods=['POST'])
def add_comment():
    """API endpoint to add a new comment."""
    data = request.get_json()
    if not data or 'author' not in data or 'content' not in data:
        return jsonify({'error': 'Missing author or content'}), 400
        
    author = data['author'].strip()
    content = data['content'].strip()
    
    if not author or not content:
        return jsonify({'error': 'Author and content cannot be empty'}), 400
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO comments (author, content) VALUES (?, ?)', (author, content))
    conn.commit()
    
    # Retrieve the inserted comment
    comment_id = cursor.lastrowid
    cursor.execute('SELECT id, author, content, timestamp FROM comments WHERE id = ?', (comment_id,))
    row = cursor.fetchone()
    conn.close()
    
    new_comment = {
        'id': row[0],
        'author': row[1],
        'content': row[2],
        'timestamp': row[3]
    }
    
    return jsonify(new_comment), 201

# ════════════════════════════════════════════════════════════════
# ML API Endpoints
# ════════════════════════════════════════════════════════════════
@app.route('/api/ml/record', methods=['POST'])
def ml_record():
    data = request.get_json()
    label = data.get('label')
    stroke_full = data.get('stroke_full') # list of [x,y,z, qw,qx,qy,qz]
    
    if not label or not stroke_full or len(stroke_full) < 5:
        return jsonify({"error": "Invalid data"}), 400
        
    ml_engine.save_stroke(label.strip().upper(), stroke_full)
    
    # Trigger background training
    threading.Thread(target=ml_engine.train_background, daemon=True).start()
    return jsonify({"status": "success", "message": f"Saved {label} & Training Started"})

@app.route('/api/ml/predict', methods=['POST'])
def ml_predict():
    data = request.get_json()
    stroke_pos = data.get('stroke_pos') # list of [x,y,z]
    
    if not stroke_pos or len(stroke_pos) < 5:
        return jsonify({"label": None, "confidence": 0.0}), 400
        
    label, conf = ml_engine.predict(np.array(stroke_pos))
    return jsonify({
        "label": label,
        "confidence": float(conf)
    })

if __name__ == '__main__':
    # Run the Flask app on port 5000 (accessible locally by default)
    # Host='0.0.0.0' allows external connections if deployed
    app.run(host='0.0.0.0', port=5000, debug=True)
