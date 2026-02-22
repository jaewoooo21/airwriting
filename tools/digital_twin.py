"""
AirWriting – Python 3D Digital Twin
====================================
Real-time 3D visualiser that receives the SAME JSON packets that
Unity would receive (UDP port 12346) and renders a skeletal arm
with pen trail using PyQtGraph's OpenGL backend.

Usage
-----
  Terminal 1:  python main.py                  # fusion engine
  Terminal 2:  python tools/digital_twin.py    # this visualiser

Controls
--------
  Mouse drag   : rotate camera
  Scroll       : zoom
"""

from __future__ import annotations

import json
import socket
import sys
import threading

import numpy as np

# ── PyQtGraph / Qt imports ──────────────────────────────────────
try:
    from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
except ImportError:
    print("=" * 60)
    print("  pyqtgraph 또는 PyOpenGL이 설치되어 있지 않습니다.")
    print("  pip install pyqtgraph PyOpenGL PyOpenGL_accelerate")
    print("=" * 60)
    sys.exit(1)

# ════════════════════════════════════════════════════════════════
# Config — matches imu.yaml skeleton_chain exactly
# ════════════════════════════════════════════════════════════════
LISTEN_PORT  = 12346
ORIGIN       = np.array([0.0, 0.0, 0.0])   # FK origin (shoulder)

# Skeleton chain from imu.yaml:
#   S1 (forearm) 0.25m → S2 (hand) 0.18m → S3 (finger) 0.08m
SEGMENTS = [
    ("S1", 0.25),   # forearm
    ("S2", 0.18),   # hand
    ("S3", 0.08),   # finger / pen tip
]

# Bone direction: Y-axis [0, 1, 0] (Forward into the screen)
BONE_DIR = np.array([0.0, 1.0, 0.0])

# Colours (RGBA 0‑1)
COL_IDLE    = (0.0, 0.5, 1.0, 1.0)     # Cyber Blue
COL_WRITING = (1.0, 0.1, 0.4, 1.0)     # Laser Pink
COL_GRID    = (0.1, 0.1, 0.2, 0.5)     # Dark Cyber Grid
COL_TRAIL   = (0.22, 0.89, 0.5, 1.0)   # Vivid Neon Green
COL_JOINT   = (0.5, 0.5, 0.5, 1.0)     # Gray Spheres

MAX_TRAIL   = 10000


# ════════════════════════════════════════════════════════════════
# Quaternion → Rotation Matrix  [w, x, y, z] Hamilton convention
# (identical to forward_kinematics.py)
# ════════════════════════════════════════════════════════════════
def quat_to_rot(q):
    w, x, y, z = q
    xx = x*x; yy = y*y; zz = z*z
    xy = x*y; xz = x*z; yz = y*z
    wx = w*x; wy = w*y; wz = w*z
    return np.array([
        [1 - 2*(yy + zz),  2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),      1 - 2*(xx + zz),  2*(yz - wx)],
        [2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy)],
    ])


# ════════════════════════════════════════════════════════════════
# UDP Receiver Thread
# ════════════════════════════════════════════════════════════════
class DataReceiver(threading.Thread):
    def __init__(self, port: int = LISTEN_PORT):
        super().__init__(daemon=True)
        self.port = port
        self.latest: dict | None = None
        self._lock = threading.Lock()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(1.0)
        print(f"[DigitalTwin] 👂 Listening on UDP :{self.port} …")
        while True:
            try:
                data, _ = sock.recvfrom(4096)
                obj = json.loads(data.decode())
                with self._lock:
                    self.latest = obj
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[DigitalTwin] recv error: {e}")

    def pop(self) -> dict | None:
        with self._lock:
            d = self.latest
            self.latest = None
            return d


# ════════════════════════════════════════════════════════════════
# Forward Kinematics
#   Mirrors forward_kinematics.py exactly
# ════════════════════════════════════════════════════════════════
def compute_fk(data: dict):
    """
    Returns list of joint positions [origin, after_S1, after_S2, after_S3].
    Uses the quaternion data (S1q, S2q, S3q) from the JSON packet.
    Applies Y↔Z swap: IMU world Y-up → PyQtGraph Z-up.
    """
    pos = ORIGIN.copy()
    positions = [pos.copy()]

    for sid, length in SEGMENTS:
        q = data.get(f"{sid}q", [1, 0, 0, 0])
        R = quat_to_rot(q)
        
        # All bones point forward in the zero-rotation state
        bone_vec = R @ (BONE_DIR * length)
            
        pos = pos + bone_vec
        positions.append(pos.copy())

    # Removed Y↔Z swap. Python ESKF and PyQtGraph both use Z=up.
    return positions  # 4 points: origin, elbow, wrist, pen-tip


# ════════════════════════════════════════════════════════════════
# Main Window
# ════════════════════════════════════════════════════════════════
class DigitalTwinWindow(QtWidgets.QMainWindow):
    def __init__(self):
        self.app = pg.mkQApp("AirWriting Pro Studio")
        super().__init__()
        self.setWindowTitle("✨ AirWriting Pro Studio (Exhibition Edition)")
        self.resize(1500, 950)
        self.setStyleSheet("background-color: #0B0F19; font-family: 'Segoe UI', sans-serif;")

        # Central Widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 3D View (Center) ──
        # We define this first so we can reference it, but add it to layout later
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((11, 15, 25, 255)) # Dark slate

        # Camera presets
        self._fpv = True
        self._cam_3rd = {"distance": 1.5, "elevation": 20, "azimuth": -40,
                         "center": QtGui.QVector3D(0.0, 0.25, 0.0)}
        
        # 1st person view: Looking from shoulder/head directly forward
        self._cam_1st = {"distance": 0.45, "elevation": 5, "azimuth": 90,
                         "center": QtGui.QVector3D(0.0, 0.5, 0.1)}
        self._apply_cam(self._cam_1st)

        # ── Left Sidebar: Word Learning ──
        left_sidebar = QtWidgets.QFrame()
        left_sidebar.setFixedWidth(300)
        left_sidebar.setStyleSheet("background-color: #111827; border-right: 1px solid #1E293B;")
        left_layout = QtWidgets.QVBoxLayout(left_sidebar)
        left_layout.setContentsMargins(25, 30, 25, 30)
        left_layout.setSpacing(15)

        vocab_title = QtWidgets.QLabel("📚 TARGET WORDS")
        vocab_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #38BDF8; letter-spacing: 1px;")
        left_layout.addWidget(vocab_title)

        words = ["1. AirWriting", "2. Digital Twin", "3. Graduation", "4. Project"]
        for w in words:
            lbl = QtWidgets.QLabel(w)
            lbl.setStyleSheet("font-size: 16px; color: #F8FAFC; padding: 10px; background: #1E293B; border-radius: 6px; border: 1px solid #334155;")
            left_layout.addWidget(lbl)
        
        left_layout.addStretch()

        # Recognition Placeholder
        recog_title = QtWidgets.QLabel("🤖 AI RECOGNITION")
        recog_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #94A3B8; letter-spacing: 1px;")
        left_layout.addWidget(recog_title)

        self.lbl_score = QtWidgets.QLabel("Graduation\n(98.5%)")
        self.lbl_score.setStyleSheet("font-size: 28px; font-weight: bold; color: #4ADE80; background: #064E3B; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #059669;")
        self.lbl_score.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.lbl_score)

        layout.addWidget(left_sidebar)
        
        # Add the 3D view to the middle
        layout.addWidget(self.view, stretch=5)

        # ── Right Sidebar: System Dashboard ──
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet("""
            QFrame { background-color: #111827; border-left: 1px solid #1E293B; }
            QLabel { color: #F8FAFC; font-size: 14px; }
        """)
        panel_layout = QtWidgets.QVBoxLayout(sidebar)
        panel_layout.setContentsMargins(25, 30, 25, 30)
        panel_layout.setSpacing(15)

        # Title
        title = QtWidgets.QLabel("ENGINEERING DASHBOARD")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #38BDF8; margin-bottom: 5px; letter-spacing: 1px;")
        panel_layout.addWidget(title)
        
        # Connection Status
        self.lbl_conn = QtWidgets.QLabel("🟢 UDP STREAM LINKED")
        self.lbl_conn.setStyleSheet("font-weight: bold; color: #4ADE80; font-family: 'Consolas';")
        panel_layout.addWidget(self.lbl_conn)

        line1 = QtWidgets.QFrame(); line1.setFrameShape(QtWidgets.QFrame.Shape.HLine); line1.setStyleSheet("color: #334155;")
        panel_layout.addWidget(line1)

        # Sensor Status
        self.lbl_pen = QtWidgets.QLabel("PEN EVENT: ⚪ IDLE")
        self.lbl_pen.setStyleSheet("font-weight: bold; font-size: 14px; color: #94A3B8; font-family: 'Consolas';")
        panel_layout.addWidget(self.lbl_pen)

        self.lbl_zupt = QtWidgets.QLabel("VELOCITY LOCK: INACTIVE")
        self.lbl_zupt.setStyleSheet("font-size: 13px; color: #94A3B8; font-family: 'Consolas';")
        panel_layout.addWidget(self.lbl_zupt)

        line2 = QtWidgets.QFrame(); line2.setFrameShape(QtWidgets.QFrame.Shape.HLine); line2.setStyleSheet("color: #334155;")
        panel_layout.addWidget(line2)

        # Coordinates
        coord_title = QtWidgets.QLabel("END-EFFECTOR POSITION [m]")
        coord_title.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: bold; text-transform: uppercase;")
        panel_layout.addWidget(coord_title)

        self.lbl_pos = QtWidgets.QLabel("X:  0.000\nY:  0.000\nZ:  0.000")
        self.lbl_pos.setStyleSheet("font-family: 'Consolas', monospace; font-size: 18px; background: #0B0F19; border: 1px solid #334155; padding: 15px; border-radius: 6px; color: #38BDF8;")
        panel_layout.addWidget(self.lbl_pos)

        # ── Live Velocity Graph ──
        plot_title = QtWidgets.QLabel("LIVE VELOCITY PROFILE")
        plot_title.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: bold; margin-top: 15px;")
        panel_layout.addWidget(plot_title)

        self.plot_vel = pg.PlotWidget()
        self.plot_vel.setBackground('#0B0F19')
        self.plot_vel.showAxis('bottom', False)
        self.plot_vel.showGrid(x=False, y=True, alpha=0.3)
        self.plot_vel.setYRange(0, 1.5)
        self.plot_curve = self.plot_vel.plot(pen=pg.mkPen(color='#38BDF8', width=2))
        self.plot_data = np.zeros(100)
        
        self.plot_vel.setFixedHeight(120)
        panel_layout.addWidget(self.plot_vel)

        panel_layout.addStretch()

        # Keyboard Hints
        hints = QtWidgets.QLabel("CAM: [V] Toggle | CANVAS: [R] Reset")
        hints.setStyleSheet("color: #64748B; font-size: 12px; font-family: 'Consolas';")
        panel_layout.addWidget(hints)

        layout.addWidget(sidebar)

        # ── 3D Environment Setup ──
        # Virtual Canvas Frame mapping the X-Z plane at Y=0.51m (Arm length)
        # This helps user see where the center is.
        frame_pts = np.array([
            [-0.3, 0.51,  0.3],
            [ 0.3, 0.51,  0.3],
            [ 0.3, 0.51, -0.1],
            [-0.3, 0.51, -0.1],
            [-0.3, 0.51,  0.3],
        ])
        canvas_frame = gl.GLLinePlotItem(pos=frame_pts, color=(0.4, 0.4, 0.5, 0.8), width=2.0)
        self.view.addItem(canvas_frame)

        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        grid.setSpacing(0.1, 0.1)
        grid.setColor(COL_GRID)
        grid.translate(0, 0, -0.15)
        self.view.addItem(grid)

        ax_len = 0.3
        for direction, color in [
            ([ax_len, 0, 0], (1, 0, 0, 0.7)),
            ([0, ax_len, 0], (0, 0.8, 0, 0.7)),
            ([0, 0, ax_len], (0, 0, 1, 0.7)),
        ]:
            ax = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], direction]), color=color, width=2.0, antialias=True)
            self.view.addItem(ax)

        self.arm_line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=COL_IDLE, width=6.0, antialias=True)
        self.view.addItem(self.arm_line)

        self.joint_meshes = []
        
        # Pen Handle (Small gray circle)
        md_handle = gl.MeshData.sphere(rows=10, cols=10, radius=0.006)
        m_handle = gl.GLMeshItem(meshdata=md_handle, smooth=True, color=COL_JOINT, shader="shaded")
        self.view.addItem(m_handle)
        self.joint_meshes.append(m_handle)

        # Pen Tip (Very tiny, sharp laser pink point)
        md_tip = gl.MeshData.sphere(rows=8, cols=8, radius=0.002)
        m_tip = gl.GLMeshItem(meshdata=md_tip, smooth=True, color=COL_WRITING, shader="shaded")
        self.view.addItem(m_tip)
        self.joint_meshes.append(m_tip)

        self.trail_pts: list[np.ndarray] = []
        self.trail_line = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=COL_TRAIL, width=4.0, antialias=True)
        self.view.addItem(self.trail_line)

        # ── Data receiver ──
        self.receiver = DataReceiver()
        self.receiver.start()

        # ── Timer ──
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

        self._pen_was_down = False
        self._frame_count = 0

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key.Key_V:
            self._toggle_view()
        elif ev.key() == QtCore.Qt.Key.Key_R:
            self.trail_pts.clear()
            self.trail_line.setData(pos=np.zeros((1, 3)))
        else:
            super().keyPressEvent(ev)

    def update_frame(self):
        data = self.receiver.pop()
        if data is None:
            return
        if data.get("t") != "f":
            return

        self._frame_count += 1
        pen = data.get("pen", False)
        zupt = data.get("S3z", False)

        positions = compute_fk(data)
        pts = np.array(positions)
        color = COL_WRITING if pen else COL_IDLE

        # Only draw the last segment (pen)
        self.arm_line.setData(pos=pts[2:4], color=color)

        # Only update the last two joints (pen handle and tip)
        for i, p in enumerate(positions[2:4]):
            tr = QtGui.QMatrix4x4()
            tr.translate(p[0], p[1], p[2])
            self.joint_meshes[i].setTransform(tr)

        # ── Pen tip = Forward Kinematics End Effector ──
        # By using FK, vertical movement directly maps to the user's arm pitch
        pen_tip_pos = positions[-1]

        if pen:
            self.trail_pts.append(pen_tip_pos.copy())
            if len(self.trail_pts) > MAX_TRAIL:
                self.trail_pts = self.trail_pts[-MAX_TRAIL:]
        elif self._pen_was_down and not pen:
            self.trail_pts.append(np.array([np.nan, np.nan, np.nan]))

        self._pen_was_down = pen

        if len(self.trail_pts) > 1:
            self.trail_line.setData(pos=np.array(self.trail_pts), color=COL_TRAIL)

        # ── Update UI Status ──
        if pen:
            self.lbl_pen.setText("PEN EVENT: 🔴 WRITING")
            self.lbl_pen.setStyleSheet("font-weight: bold; font-size: 14px; color: #F87171; font-family: 'Consolas';")
        else:
            self.lbl_pen.setText("PEN EVENT: ⚪ IDLE")
            self.lbl_pen.setStyleSheet("font-weight: bold; font-size: 14px; color: #94A3B8; font-family: 'Consolas';")

        if zupt:
            self.lbl_zupt.setText("VELOCITY LOCK: 🟢 ACTIVE")
            self.lbl_zupt.setStyleSheet("color: #4ADE80; font-weight: bold; font-family: 'Consolas';")
        else:
            self.lbl_zupt.setText("VELOCITY LOCK: ⚪ MOVING")
            self.lbl_zupt.setStyleSheet("color: #94A3B8; font-family: 'Consolas';")

        p = pen_tip_pos
        self.lbl_pos.setText(f"X: {p[0]:8.3f}\nY: {p[1]:8.3f}\nZ: {p[2]:8.3f}")

        # Update Live Velocity Graph
        s3v = data.get("S3v")
        vel_mag = np.linalg.norm(s3v) if s3v else 0.0
        self.plot_data = np.roll(self.plot_data, -1)
        self.plot_data[-1] = vel_mag
        self.plot_curve.setData(self.plot_data)

    def _apply_cam(self, preset: dict):
        self.view.opts["distance"] = preset["distance"]
        self.view.opts["elevation"] = preset["elevation"]
        self.view.opts["azimuth"] = preset["azimuth"]
        self.view.opts["center"] = preset["center"]

    def _toggle_view(self):
        self._fpv = not self._fpv
        if self._fpv:
            self._apply_cam(self._cam_1st)
            print("[DigitalTwin] 🥽 VR 1인칭 시점")
        else:
            self._apply_cam(self._cam_3rd)
            print("[DigitalTwin] 🎥 3인칭 시점")

    def run(self):
        self.show()
        print("[DigitalTwin] 🖥️  Window opened.")
        print("[DigitalTwin] Mouse drag=rotate, scroll=zoom")
        print("[DigitalTwin] V=시점전환(1인칭/3인칭), R=궤적 초기화")
        sys.exit(self.app.exec())


# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  ✨ AirWriting Python Digital Twin")
    print("  Listens on UDP :12346 for JSON from fusion engine")
    print("=" * 60)
    twin = DigitalTwinWindow()
    twin.run()
