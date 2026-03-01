import socket
import json
import time
import os
import csv
import threading
import numpy as np

# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════
UDP_PORT = 12346
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_FILE = os.path.join(OUTPUT_DIR, 'airwriting_dataset.csv')

# FK Configuration
SEGMENTS = [
    ("S1", 0.25),
    ("S2", 0.18),
    ("S3", 0.08)
]
BONE_DIR = np.array([0.0, 1.0, 0.0])
ORIGIN = np.array([0.0, 0.0, 0.0])

# ════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════
def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])

def compute_fk_tip(data: dict):
    """Computes the 3D position of the pen tip."""
    pos = ORIGIN.copy()
    for sid, length in SEGMENTS:
        q = data.get(f"{sid}q", [1, 0, 0, 0])
        R = quat_to_rot(q)
        bone_vec = R @ (BONE_DIR * length)
        pos = pos + bone_vec
    return pos  # Return only the tip position

# ════════════════════════════════════════════════════════════════
# Main Data Collector
# ════════════════════════════════════════════════════════════════
class DataCollector:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', UDP_PORT))
        self.sock.settimeout(0.1)
        self.running = True
        
        self.current_label = None
        self.is_recording = False
        self.current_stroke = []
        self.session_id = int(time.time())
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        write_header = not os.path.exists(CSV_FILE)
        self.file = open(CSV_FILE, 'a', newline='')
        self.writer = csv.writer(self.file)
        
        if write_header:
            self.writer.writerow([
                'session_id', 'label', 'stroke_idx', 'frame_idx', 'timestamp',
                'fk_x', 'fk_y', 'fk_z',
                'q_w', 'q_x', 'q_y', 'q_z'
            ])

    def start_listening(self):
        print(f"[Collector] Listening on UDP {UDP_PORT}...")
        stroke_idx = 0
        
        last_pen_state = False
        
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                obj = json.loads(data.decode())
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Collector] Receive error: {e}")
                continue
                
            if obj.get('t') != 'f':
                continue
                
            pen_down = obj.get('pen', False)
            
            # Start of a stroke
            if pen_down and not last_pen_state and self.current_label is not None:
                print("🔴 Recording started...")
                self.is_recording = True
                self.current_stroke = []
            
            # During a stroke
            if self.is_recording and pen_down:
                tip_pos = compute_fk_tip(obj)
                s3q = obj.get('S3q', [1, 0, 0, 0])
                
                frame_data = [
                    self.session_id, self.current_label, stroke_idx, len(self.current_stroke), time.time(),
                    tip_pos[0], tip_pos[1], tip_pos[2],
                    s3q[0], s3q[1], s3q[2], s3q[3]
                ]
                self.current_stroke.append(frame_data)
                
            # End of a stroke
            if not pen_down and last_pen_state and self.is_recording:
                print(f"⚪ Recording stopped. Saved {len(self.current_stroke)} frames.")
                for row in self.current_stroke:
                    self.writer.writerow(row)
                self.file.flush()
                self.is_recording = False
                stroke_idx += 1
                
                # Signal the main thread that a stroke is completed
                if hasattr(self, 'stroke_callback') and self.stroke_callback:
                    self.stroke_callback()

            last_pen_state = pen_down

    def stop(self):
        self.running = False
        self.file.close()
        self.sock.close()


def main():
    collector = DataCollector()
    listener_thread = threading.Thread(target=collector.start_listening, daemon=True)
    listener_thread.start()
    
    print("\n" + "="*50)
    print("✨ AirWriting Data Collector")
    print("="*50)
    
    try:
        while True:
            target_char = input("\n👉 Enter the letter you want to write (e.g., A, B, C) or 'q' to quit: ").strip().upper()
            if target_char == 'Q':
                break
            
            if not target_char.isalpha() or len(target_char) != 1:
                print("Please enter a single alphabet character.")
                continue
                
            num_samples_str = input(f"How many samples of '{target_char}' do you want to record? (Default 5): ").strip()
            num_samples = int(num_samples_str) if num_samples_str.isdigit() else 5
            
            collector.current_label = target_char
            
            print(f"\n[Ready] Get ready to draw '{target_char}'.")
            print("Press and hold the PEN BUTTON, write the letter, then release the button.")
            
            samples_collected = 0
            
            # Define a callback to increment count precisely when a stroke finishes
            def on_stroke_done():
                nonlocal samples_collected
                samples_collected += 1
                if samples_collected < num_samples:
                    print(f"✅ Sample {samples_collected}/{num_samples} done. Ready for next...")
                else:
                    print(f"🎉 Completed {num_samples} samples for '{target_char}'!")
            
            collector.stroke_callback = on_stroke_done
            
            # Wait until the required number of samples are collected
            while samples_collected < num_samples:
                time.sleep(0.1)
                
            collector.current_label = None # Paused
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        collector.stop()
        print("Data Collection Stopped. Dataset saved to 'data/airwriting_dataset.csv'.")

if __name__ == "__main__":
    main()
