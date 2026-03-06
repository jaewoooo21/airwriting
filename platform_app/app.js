const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('recordingOverlay');

// DOM Elements
const valConn = document.getElementById('valConn');
const valPos = document.getElementById('valPos');
const valZupt = document.getElementById('valZupt');

// Configuration
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsPort = window.location.hostname === 'localhost' ? ':18765' : '';
const WS_URL = `${protocol}//${window.location.hostname}${wsPort}`;
let ws = null;

// ══════════════════════════════════════════
// FK Config — EXACT copy from digital_twin.py
// ══════════════════════════════════════════
// Skeleton chain from imu.yaml:
//   S1 (forearm) 0.25m → S2 (hand) 0.18m → S3 (finger) 0.08m
const SEGMENTS = [
    { sid: "S1", length: 0.25 },
    { sid: "S2", length: 0.18 },
    { sid: "S3", length: 0.08 },
];
const BONE_DIR = [0.0, 1.0, 0.0]; // Y-forward (same as digital_twin.py)
const ORIGIN = [0.0, 0.0, 0.0];

// Quaternion to Rotation Matrix [w, x, y, z] — Hamilton convention
// EXACT copy from digital_twin.py quat_to_rot()
function quatToRot(q) {
    let w = q[0], x = q[1], y = q[2], z = q[3];
    let xx = x * x, yy = y * y, zz = z * z;
    let xy = x * y, xz = x * z, yz = y * z;
    let wx = w * x, wy = w * y, wz = w * z;
    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ];
}

// Matrix × Vector (3x3 × 3)
function matMul(R, v) {
    return [
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    ];
}

// EXACT copy of digital_twin.py compute_fk()
function computeFK(data) {
    let pos = [...ORIGIN];
    let positions = [[...pos]];

    for (let seg of SEGMENTS) {
        let q = data[seg.sid + "q"] || [1, 0, 0, 0];
        let R = quatToRot(q);
        // bone_vec = R @ (BONE_DIR * length)
        let scaledDir = [BONE_DIR[0] * seg.length, BONE_DIR[1] * seg.length, BONE_DIR[2] * seg.length];
        let boneVec = matMul(R, scaledDir);
        pos = [pos[0] + boneVec[0], pos[1] + boneVec[1], pos[2] + boneVec[2]];
        positions.push([...pos]);
    }
    return positions; // [origin, elbow, wrist, pen-tip]
}

// ══════════════════════════════════════════
// Camera System (matching digital_twin.py presets)
// ══════════════════════════════════════════
// digital_twin.py 1st person: distance=0.45, elevation=5, azimuth=90, center=(0, 0.5, 0.1)
// digital_twin.py 3rd person: distance=1.5, elevation=20, azimuth=-40, center=(0, 0.25, 0)
// We replicate this as orbital camera parameters
let camDistance = 0.4;
let camElevation = 10;     // degrees (level horizon)
let camAzimuth = 0;        // 0 degrees looks down the +Y axis (arm direction)
let camCenterX = 0.0;
let camCenterY = 0.25;     // Shifted slightly forward
let camCenterZ = 0.15;     // Shifted up to center the view vertically
let isFPV = true;

const CAM_1ST = { distance: 0.4, elevation: 10, azimuth: 0, cx: 0.0, cy: 0.25, cz: 0.15 };
const CAM_3RD = { distance: 1.2, elevation: 30, azimuth: 45, cx: 0.0, cy: 0.4, cz: 0.0 };
const FOCAL = 900;

let isDragging = false;
let lastMouseX = 0, lastMouseY = 0;

// Stroke history (3D world coordinates)
let strokeHistory = [];
let currentStroke = null;
let lastPenState = false;
let currentCursorPos = null;
let armPositions = null; // FK joint positions

// ML State Machine
let mlLearningLabel = null;
let mlAutoPredict = false;
let mlCurrentFull = [];
let mlCurrentPos = [];

// Cursor Element
const canvasContainer = document.querySelector('.canvas-container');
const liveCursor = document.createElement('div');
liveCursor.className = 'live-cursor';
if (canvasContainer) canvasContainer.appendChild(liveCursor);

// Canvas Setup
function resizeCanvas() {
    const parent = canvas.parentElement;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;
    requestAnimationFrame(renderScene);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ══════════════════════════════════════════
// 3D → 2D Projection (Orbital Camera, like PyQtGraph)
// ══════════════════════════════════════════
function projectWorld(wx, wy, wz) {
    // Translate to camera center
    let dx = wx - camCenterX;
    let dy = wy - camCenterY;
    let dz = wz - camCenterZ;

    // Azimuth rotation around Z axis
    let azRad = camAzimuth * Math.PI / 180;
    let cosA = Math.cos(azRad), sinA = Math.sin(azRad);
    let x1 = dx * cosA + dy * sinA;
    let y1 = -dx * sinA + dy * cosA;
    let z1 = dz;

    // Elevation rotation around X axis
    let elRad = camElevation * Math.PI / 180;
    let cosE = Math.cos(elRad), sinE = Math.sin(elRad);
    let y2 = y1 * cosE + z1 * sinE;
    let z2 = -y1 * sinE + z1 * cosE;

    // y2 is now the depth (into the screen)
    let depth = y2 + camDistance;
    if (depth < 0.01) depth = 0.01;

    let f = FOCAL;
    let sx = f * (x1 / depth);
    let sy = -f * (z2 / depth); // Canvas Y is down, World Z is up

    return { sx, sy, depth, visible: (y2 + camDistance) > 0.01 };
}

// ══════════════════════════════════════════
// Rendering
// ══════════════════════════════════════════
function renderScene() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    drawGrid(cx, cy);
    drawCanvasFrame(cx, cy);
    drawWorkspaceBox(cx, cy); // New visually anchored box
    drawAxisArrows(cx, cy);
    drawArm(cx, cy);
    drawStrokes(cx, cy);
    drawCursor(cx, cy);
}

// Minimal floor reference (just 2 lines instead of dense grid)
function drawGrid(cx, cy) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(40, 40, 70, 0.3)';
    ctx.shadowBlur = 0;
    const size = 0.6, gz = -0.15;
    ctx.beginPath();
    // X-axis line
    let a = projectWorld(-size, 0, gz), b = projectWorld(size, 0, gz);
    if (a.visible && b.visible) { ctx.moveTo(cx + a.sx, cy + a.sy); ctx.lineTo(cx + b.sx, cy + b.sy); }
    // Y-axis line
    a = projectWorld(0, -size, gz); b = projectWorld(0, size, gz);
    if (a.visible && b.visible) { ctx.moveTo(cx + a.sx, cy + a.sy); ctx.lineTo(cx + b.sx, cy + b.sy); }
    ctx.stroke();
}

// Virtual Canvas Frame (same as digital_twin.py: XZ plane at Y=0.51)
function drawCanvasFrame(cx, cy) {
    const framePts = [
        [-0.3, 0.51, 0.3], [0.3, 0.51, 0.3],
        [0.3, 0.51, -0.1], [-0.3, 0.51, -0.1], [-0.3, 0.51, 0.3]
    ];
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(102, 102, 128, 0.6)';
    ctx.beginPath();
    let first = true;
    for (let p of framePts) {
        let pt = projectWorld(p[0], p[1], p[2]);
        if (!pt.visible) continue;
        if (first) { ctx.moveTo(cx + pt.sx, cy + pt.sy); first = false; }
        else ctx.lineTo(cx + pt.sx, cy + pt.sy);
    }
    ctx.stroke();

    // Semi-transparent fill for the board
    ctx.fillStyle = 'rgba(20, 25, 40, 0.3)';
    ctx.beginPath();
    first = true;
    for (let p of framePts) {
        let pt = projectWorld(p[0], p[1], p[2]);
        if (!pt.visible) continue;
        if (first) { ctx.moveTo(cx + pt.sx, cy + pt.sy); first = false; }
        else ctx.lineTo(cx + pt.sx, cy + pt.sy);
    }
    ctx.closePath();
    ctx.fill();
}

// Draw the 3D Bounding Box representing the Anchored local workspace area
function drawWorkspaceBox(cx, cy) {
    if (!window.strokeAnchorPos || (!lastPenState && !isAutoRecording)) return;

    // Define a 0.4m x 0.4m semi-transparent board centered at the anchor coordinate
    // The ML engine learns coordinates relative to this center point.
    const aw = 0.2; // half-width (X)
    const ah = 0.2; // half-height (Z)
    const ax = window.strokeAnchorPos[0];
    const ay = window.strokeAnchorPos[1]; // Depth is locked (XZ plane)
    const az = window.strokeAnchorPos[2];

    const corners = [
        [ax - aw, ay, az - ah], [ax + aw, ay, az - ah],
        [ax + aw, ay, az + ah], [ax - aw, ay, az + ah],
        [ax - aw, ay, az - ah]
    ];

    // Border (Dashed)
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(74, 222, 128, 0.8)'; // Bright green bounding box
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    let first = true;
    for (const p of corners) {
        let pt = projectWorld(p[0], p[1], p[2]);
        if (!pt.visible) continue;
        if (first) { ctx.moveTo(cx + pt.sx, cy + pt.sy); first = false; }
        else ctx.lineTo(cx + pt.sx, cy + pt.sy);
    }
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash

    // Fill (Semi-transparent board)
    ctx.fillStyle = 'rgba(74, 222, 128, 0.15)';
    ctx.beginPath();
    first = true;
    for (const p of corners) {
        let pt = projectWorld(p[0], p[1], p[2]);
        if (!pt.visible) continue;
        if (first) { ctx.moveTo(cx + pt.sx, cy + pt.sy); first = false; }
        else ctx.lineTo(cx + pt.sx, cy + pt.sy);
    }
    ctx.closePath();
    ctx.fill();

    // Anchor Center Dot (0,0,0 coordinate for ML Mode)
    let centerPt = projectWorld(ax, ay, az);
    if (centerPt.visible) {
        ctx.fillStyle = 'rgba(74, 222, 128, 0.9)';
        ctx.beginPath();
        ctx.arc(cx + centerPt.sx, cy + centerPt.sy, 5, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Axis Arrows (RGB = XYZ)
function drawAxisArrows(cx, cy) {
    const L = 0.3;
    let o = projectWorld(0, 0, 0);
    if (!o.visible) return;
    const axes = [
        { v: [L, 0, 0], c: 'rgba(255, 0, 0, 0.7)' },
        { v: [0, L, 0], c: 'rgba(0, 200, 0, 0.7)' },
        { v: [0, 0, L], c: 'rgba(0, 0, 255, 0.7)' },
    ];
    ctx.lineWidth = 2;
    ctx.shadowBlur = 0;
    for (let a of axes) {
        let e = projectWorld(a.v[0], a.v[1], a.v[2]);
        if (!e.visible) continue;
        ctx.strokeStyle = a.c;
        ctx.beginPath();
        ctx.moveTo(cx + o.sx, cy + o.sy);
        ctx.lineTo(cx + e.sx, cy + e.sy);
        ctx.stroke();
    }
}

// Arm skeleton (last 2 segments: wrist→pen handle→pen tip)
function drawArm(cx, cy) {
    if (!armPositions || armPositions.length < 4) return;

    // Draw pen segment (wrist to tip) like digital_twin.py
    let col = lastPenState ? '#ff1a66' : '#0080ff';

    ctx.lineWidth = 5;
    ctx.strokeStyle = col;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    let pts = [armPositions[2], armPositions[3]]; // wrist, tip
    let first = true;
    for (let p of pts) {
        let pt = projectWorld(p[0], p[1], p[2]);
        if (!pt.visible) continue;
        if (first) { ctx.moveTo(cx + pt.sx, cy + pt.sy); first = false; }
        else ctx.lineTo(cx + pt.sx, cy + pt.sy);
    }
    ctx.stroke();

    // Joint dots
    for (let i = 2; i < 4; i++) {
        let p = armPositions[i];
        let pt = projectWorld(p[0], p[1], p[2]);
        if (!pt.visible) continue;
        let r = i === 3 ? 3 : 5; // tip is smaller
        ctx.fillStyle = i === 3 ? '#ff1a66' : '#888888';
        ctx.beginPath();
        ctx.arc(cx + pt.sx, cy + pt.sy, r, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Pen trail strokes
function drawStrokes(cx, cy) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#38e67a'; // Neon green trail (COL_TRAIL from digital_twin.py)
    ctx.shadowBlur = 8;
    ctx.shadowColor = 'rgba(56, 230, 122, 0.6)';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (const stroke of strokeHistory) {
        if (stroke.length < 2) continue;
        ctx.beginPath();
        let started = false;
        for (const p of stroke) {
            let pt = projectWorld(p[0], p[1], p[2]);
            if (!pt.visible) continue;
            if (!started) { ctx.moveTo(cx + pt.sx, cy + pt.sy); started = true; }
            else ctx.lineTo(cx + pt.sx, cy + pt.sy);
        }
        ctx.stroke();
    }
    ctx.shadowBlur = 0;
}

// Live cursor
function drawCursor(cx, cy) {
    if (!currentCursorPos) return;
    let pt = projectWorld(currentCursorPos[0], currentCursorPos[1], currentCursorPos[2]);
    if (!pt.visible) return;
    liveCursor.style.left = `${cx + pt.sx}px`;
    liveCursor.style.top = `${cy + pt.sy}px`;
    if (lastPenState) {
        liveCursor.style.background = '#ffffff';
        liveCursor.style.transform = 'translate(-50%, -50%) scale(1.5)';
    } else {
        liveCursor.style.background = 'transparent';
        liveCursor.style.transform = 'translate(-50%, -50%) scale(1.0)';
    }
}

// ══════════════════════════════════════════
// WebSocket
// ══════════════════════════════════════════
function connectWebSocket() {
    valConn.textContent = "🟡 CONNECTING...";
    valConn.className = "data-value warning";
    ws = new WebSocket(WS_URL);
    ws.onopen = () => { valConn.textContent = "🟢 CONNECTED (WS)"; valConn.className = "data-value success"; };
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.t === "f") updateFrame(data);
        } catch (e) { console.error("JSON parse error:", e); }
    };
    ws.onclose = () => { valConn.textContent = "🔴 DISCONNECTED"; valConn.className = "data-value warning"; setTimeout(connectWebSocket, 3000); };
    ws.onerror = (err) => console.error("WS Error:", err);
}
// ── Auto-connect on page load ──
connectWebSocket();

function updateFrame(data) {
    const pen = data.pen || false;
    const zupt = data.S3z || false;

    if (pen) overlay.classList.add('active');
    else overlay.classList.remove('active');

    if (zupt) { valZupt.textContent = "🟢 ACTIVE"; valZupt.className = "data-value success"; }
    else { valZupt.textContent = "⚪ INACTIVE"; valZupt.className = "data-value warning"; }

    if (data.S3e) updateLiveChart(data.S3e);

    // ── EXACT SAME FK as digital_twin.py ──
    let positions = computeFK(data);
    armPositions = positions;
    let penTip = positions[positions.length - 1]; // pen-tip = last FK joint
    currentCursorPos = penTip;

    valPos.innerHTML = `X: ${penTip[0].toFixed(3)}<br>Y: ${penTip[1].toFixed(3)}<br>Z: ${penTip[2].toFixed(3)}`;

    // Store latest data globally (for quaternion access in auto-record)
    window.latestData = data;

    // ── ML Auto-Predict Logic (Manual Pen Control) ──
    let localizedTip = [...penTip];

    // Calculate localized position based on anchor
    if (window.strokeAnchorPos) {
        localizedTip = [
            penTip[0] - window.strokeAnchorPos[0],
            penTip[1] - window.strokeAnchorPos[1],
            penTip[2] - window.strokeAnchorPos[2]
        ];
    }

    if (pen && !lastPenState) {
        // PEN DOWN EDGE -> Lock new anchor workspace!
        window.strokeAnchorPos = [...penTip];
        localizedTip = [0, 0, 0];

        currentStroke = [];
        strokeHistory.push(currentStroke);

        if (mlAutoPredict) {
            mlCurrentPos = [];
            updateMlStatus(`Analyzing...`);
            updateScoreBoard([]);
        }

        // ── Pen-Button Manual Collection Mode ──
        if (isAutoRecording && mlLearningLabel) {
            mlCurrentFull = [];
            window.manualRecAnchor = [...penTip];
            recordingOverlay.innerText = `🔴 REC (쓰는 중...)`;
            recordingOverlay.style.color = "#EF4444";
            recordingOverlay.style.fontSize = '20px';
            updateMlStatus(`🔴 '${mlLearningLabel}' 녹화 중...`);
        }
    }

    if (pen) {
        // Draw the trace in the REAL 3D World (Global coordinates)
        currentStroke.push([...penTip]);

        // Feed the ML Engine the Anchored Local Space coordinates
        if (mlAutoPredict) {
            mlCurrentPos.push([...localizedTip]);
        }

        // ── Manual collection: accumulate data while pen is held ──
        if (isAutoRecording && mlLearningLabel && window.manualRecAnchor) {
            let lx = penTip[0] - window.manualRecAnchor[0];
            let ly = penTip[1] - window.manualRecAnchor[1];
            let lz = penTip[2] - window.manualRecAnchor[2];
            let s3q = data.S3q || [1, 0, 0, 0];
            mlCurrentFull.push([lx, ly, lz, s3q[0], s3q[1], s3q[2], s3q[3]]);
        }
    }

    if (!pen && lastPenState) {
        // PEN UP EDGE
        if (mlAutoPredict && mlCurrentPos.length > 5) {
            sendMlPredict(mlCurrentPos);
        }

        // ── Manual collection: save on pen-up ──
        if (isAutoRecording && mlLearningLabel && mlCurrentFull.length > 5) {
            autoRecordSampleCount++;
            sendMlRecord(mlLearningLabel, mlCurrentFull);
            recordingOverlay.innerText = `✅ SAVED #${autoRecordSampleCount} — 다시 펜을 누르세요`;
            recordingOverlay.style.color = "#10B981";
            recordingOverlay.style.fontSize = '18px';
            updateMlStatus(`✅ '${mlLearningLabel}' #${autoRecordSampleCount} 저장!`);
            mlCurrentFull = [];
            window.manualRecAnchor = null;
        }
    }

    lastPenState = pen;
    requestAnimationFrame(renderScene);
}

// ══════════════════════════════════════════
// ML API Calls & UI
// ══════════════════════════════════════════
const btnMlRec = document.getElementById('btnMlRec');
const btnMlPredict = document.getElementById('btnMlPredict');
const mlStatusText = document.getElementById('mlStatusText');
const aiResultWord = document.getElementById('aiResultWord');
const aiResultScore = document.getElementById('aiResultScore');

// Modal Elements
const labelModal = document.getElementById('labelModal');
const labelInput = document.getElementById('labelInput');
const btnModalCancel = document.getElementById('btnModalCancel');
const btnModalStart = document.getElementById('btnModalStart');
const btnMlTrain = document.getElementById('btnMlTrain');

let isAutoRecording = false;
let autoRecordInterval = null;
let autoRecordPhase = 'idle';
let autoRecordTimer = 0;

let autoRecordSampleCount = 0;

function startAutoRecordingLoop() {
    autoRecordPhase = 'idle';
    autoRecordSampleCount = 0;

    // Clear any existing interval
    if (autoRecordInterval) clearInterval(autoRecordInterval);

    autoRecordInterval = setInterval(() => {
        if (!isAutoRecording || !mlLearningLabel) return;

        let now = performance.now();

        if (autoRecordPhase === 'idle') {
            autoRecordPhase = 'countdown';
            autoRecordTimer = now;
            recordingOverlay.innerText = "⏳ READY...";
            recordingOverlay.style.color = "#F59E0B";
            recordingOverlay.style.display = 'block';
            recordingOverlay.style.fontSize = '18px';
            updateMlStatus(`'${mlLearningLabel}' 수집 준비 중...`);
        }
        else if (autoRecordPhase === 'countdown') {
            let elapsed = (now - autoRecordTimer) / 1000;
            let remaining = Math.max(0, 1.5 - elapsed).toFixed(1);
            recordingOverlay.innerText = `⏳ ${remaining}s 후 시작...`;
            recordingOverlay.style.color = "#F59E0B";

            if (elapsed >= 1.5) {
                // Start actual recording
                autoRecordPhase = 'record';
                autoRecordTimer = now;
                mlCurrentFull = [];
                window.autoRecAnchor = null;
                recordingOverlay.innerText = "🔴 REC 3.0s (지금 쓰세요!)";
                recordingOverlay.style.color = "#EF4444";
                recordingOverlay.style.fontSize = '22px';
                updateMlStatus(`🔴 녹화 중... '${mlLearningLabel}'`);
            }
        }
        else if (autoRecordPhase === 'record') {
            let elapsed = (now - autoRecordTimer) / 1000;
            let remaining = Math.max(0, 3.0 - elapsed).toFixed(1);

            // Live countdown on overlay
            recordingOverlay.innerText = `🔴 REC ${remaining}s (지금 쓰세요!)`;

            // Collect data every tick (pen state irrelevant during auto-record)
            if (currentCursorPos) {
                if (!window.autoRecAnchor) {
                    window.autoRecAnchor = [...currentCursorPos];
                }

                let localX = currentCursorPos[0] - window.autoRecAnchor[0];
                let localY = currentCursorPos[1] - window.autoRecAnchor[1];
                let localZ = currentCursorPos[2] - window.autoRecAnchor[2];

                let s3q = [1, 0, 0, 0];
                if (window.latestData && window.latestData.S3q) s3q = window.latestData.S3q;

                mlCurrentFull.push([
                    localX, localY, localZ,
                    s3q[0], s3q[1], s3q[2], s3q[3]
                ]);
            }

            // 3 seconds elapsed → save and loop
            if (elapsed >= 3.0) {
                if (mlCurrentFull.length > 5) {
                    autoRecordSampleCount++;
                    sendMlRecord(mlLearningLabel, mlCurrentFull);

                    // Flash "SAVED" briefly
                    recordingOverlay.innerText = `✅ SAVED #${autoRecordSampleCount}`;
                    recordingOverlay.style.color = "#10B981";
                    recordingOverlay.style.fontSize = '20px';
                    updateMlStatus(`✅ '${mlLearningLabel}' #${autoRecordSampleCount} 저장 완료!`);
                } else {
                    recordingOverlay.innerText = "⚠️ 데이터 부족 (다시)";
                    recordingOverlay.style.color = "#F59E0B";
                    updateMlStatus("⚠️ 데이터 부족, 다시 시도...");
                }

                // Reset for next cycle
                mlCurrentFull = [];
                autoRecordPhase = 'saved_flash';
                autoRecordTimer = now;
            }
        }
        else if (autoRecordPhase === 'saved_flash') {
            // Brief 0.8s pause to show the SAVED message before restarting
            let elapsed = (now - autoRecordTimer) / 1000;
            if (elapsed >= 0.8) {
                autoRecordPhase = 'countdown';
                autoRecordTimer = now;
            }
        }
    }, 50); // 50ms tick (~20Hz)
}


if (btnMlRec) {
    btnMlRec.addEventListener('click', () => {
        if (isAutoRecording) {
            // Stop recording
            isAutoRecording = false;
            clearInterval(autoRecordInterval);
            autoRecordInterval = null;
            autoRecordPhase = 'idle';
            autoRecordSampleCount = 0;
            mlLearningLabel = null;
            btnMlRec.innerHTML = `🎯 [가이드] 단어 수집`;
            btnMlRec.style.background = '';
            recordingOverlay.style.display = 'none';
            recordingOverlay.style.fontSize = '';
            updateMlStatus("Idle");
            return;
        }

        // Show Modal
        if (labelModal) {
            labelModal.classList.add('active');
            labelInput.value = '';
            labelInput.focus();
        }
    });

    if (btnModalCancel) {
        btnModalCancel.addEventListener('click', () => {
            labelModal.classList.remove('active');
        });
    }

    const gridBtns = document.querySelectorAll('.grid-key-btn');
    gridBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (labelInput) {
                labelInput.value = btn.getAttribute('data-key');
                labelInput.focus();
            }
        });
    });

    if (btnModalStart) {
        btnModalStart.addEventListener('click', () => {
            let label = labelInput.value.trim().toUpperCase();
            if (label !== '') {
                mlLearningLabel = label;
                isAutoRecording = true;
                autoRecordSampleCount = 0;

                labelModal.classList.remove('active');

                btnMlRec.innerHTML = `⏹️ [중지] '${mlLearningLabel}' 수집 중...`;
                btnMlRec.style.background = '#F87171';

                // Show pen instruction
                recordingOverlay.innerText = `🎯 '${mlLearningLabel}' — 펜을 눌러 글씨를 쓰세요!`;
                recordingOverlay.style.color = '#38BDF8';
                recordingOverlay.style.fontSize = '18px';
                recordingOverlay.style.display = 'block';
                updateMlStatus(`대기: 펜을 누르면 녹화 시작`);
            }
        });
    }

    btnMlPredict.addEventListener('click', () => {
        mlAutoPredict = !mlAutoPredict;
        if (mlAutoPredict) {
            btnMlPredict.innerHTML = `⚡ [자동보정] 끄기`;
            btnMlPredict.classList.add('active');
            updateMlStatus("Waiting for gesture...");
        } else {
            btnMlPredict.innerHTML = `⚡ [자동보정] 켜기`;
            btnMlPredict.classList.remove('active');
            updateMlStatus("Idle");
            updateScoreBoard([]);
        }
    });

    if (btnMlTrain) {
        btnMlTrain.addEventListener('click', async () => {
            let originalText = btnMlTrain.innerText;
            btnMlTrain.innerText = "⏳ 학습 중... (Training)";
            btnMlTrain.disabled = true;
            try {
                const res = await fetch('/api/ml/train', { method: 'POST' });
                if (res.ok) {
                    setTimeout(() => {
                        btnMlTrain.innerText = "✅ 백그라운드 학습 시작 (10초 소요)";
                        setTimeout(() => {
                            btnMlTrain.innerText = originalText;
                            btnMlTrain.disabled = false;
                        }, 3000);
                    }, 500);
                }
            } catch (e) {
                console.error(e);
                btnMlTrain.innerText = "❌ 확인 필요";
                setTimeout(() => {
                    btnMlTrain.innerText = originalText;
                    btnMlTrain.disabled = false;
                }, 2000);
            }
        });
    }
}

let recognizedSequence = "";
let lastTop1 = null;
let autocorrectSuggestion = null;
const recognizedTextOverlay = document.getElementById('recognizedTextOverlay');

function updateTextOverlay() {
    if (!recognizedTextOverlay) return;
    if (recognizedSequence === "" && !autocorrectSuggestion) {
        recognizedTextOverlay.classList.remove('active');
        return;
    }
    recognizedTextOverlay.classList.add('active');

    if (autocorrectSuggestion && autocorrectSuggestion.distance > 0) {
        recognizedTextOverlay.innerHTML = `
            <span style="color:#F87171; text-decoration:line-through; opacity:0.6">${recognizedSequence}</span>
            <span style="color:#4ADE80; margin-left:8px">→ ${autocorrectSuggestion.word}?</span>
            <span style="font-size:12px; color:#888; margin-left:8px">[Enter=확정]</span>
        `;
    } else {
        recognizedTextOverlay.innerText = recognizedSequence;
    }
}

document.addEventListener('keydown', (e) => {
    if (!mlAutoPredict) return;

    if (e.code === 'Space') {
        e.preventDefault();
        if (lastTop1) {
            recognizedSequence += lastTop1.label;
            lastTop1 = null;

            // Autocorrect check
            if (typeof findClosestWords === 'function' && recognizedSequence.length >= 2) {
                let matches = findClosestWords(recognizedSequence);
                if (matches.length > 0 && matches[0].distance > 0) {
                    autocorrectSuggestion = matches[0];
                } else {
                    autocorrectSuggestion = null;
                }
            }
        }
        updateTextOverlay();
        strokeHistory = [];
        updateScoreBoard([]);
    } else if (e.code === 'Enter') {
        // Accept autocorrect suggestion
        if (autocorrectSuggestion && autocorrectSuggestion.distance > 0) {
            recognizedSequence = autocorrectSuggestion.word;
            autocorrectSuggestion = null;
            updateTextOverlay();
        }
    } else if (e.code === 'Backspace') {
        if (recognizedSequence.length > 0) {
            recognizedSequence = recognizedSequence.slice(0, -1);
            autocorrectSuggestion = null;
            if (recognizedSequence === "") {
                recognizedTextOverlay?.classList.remove('active');
            }
            updateTextOverlay();
        }
    }
});

function updateMlStatus(msg) {
    if (mlStatusText) mlStatusText.innerText = `Status: ${msg}`;
}

function updateScoreBoard(predictions) {
    if (!predictions || predictions.length === 0) {
        if (aiResultWord) aiResultWord.innerText = "??";
        if (aiResultScore) aiResultScore.innerText = "(0.0%)";
        const aiCandidates = document.getElementById('aiCandidates');
        if (aiCandidates) aiCandidates.innerHTML = '';
        return;
    }

    // Top 1
    const top1 = predictions[0];
    lastTop1 = top1;
    if (aiResultWord) aiResultWord.innerText = top1.label;
    if (aiResultScore) aiResultScore.innerText = `(${(top1.confidence * 100).toFixed(1)}%)`;

    // Top N loop
    const aiCandidates = document.getElementById('aiCandidates');
    if (aiCandidates) {
        aiCandidates.innerHTML = ''; // clear
        // We show up to top 3
        for (let i = 0; i < Math.min(3, predictions.length); i++) {
            let p = predictions[i];
            let percent = (p.confidence * 100).toFixed(1);
            let bar = document.createElement('div');
            bar.className = 'candidate-bar';

            // Highlight the first one with a different border or color
            if (i === 0) bar.style.borderLeftColor = '#4ADE80';

            bar.innerHTML = `
                <div class="c-name">${i + 1}. ${p.label}</div>
                <div class="c-val">${percent}%</div>
            `;
            aiCandidates.appendChild(bar);
        }
    }

    // Pulse animation
    const box = document.getElementById('aiScoreBox');
    if (box) {
        box.style.transform = 'scale(1.05)';
        box.style.borderColor = '#4ADE80';
        setTimeout(() => {
            box.style.transform = 'scale(1)';
            box.style.borderColor = 'var(--border-color)';
        }, 200);
    }
}

// ML Training Lab Charts
let sampleDistChart = null;
let accuracyChart = null;

function initLabCharts() {
    const ctxDist = document.getElementById('sampleDistChart')?.getContext('2d');
    if (ctxDist) {
        sampleDistChart = new Chart(ctxDist, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Samples Collected',
                    data: [],
                    backgroundColor: 'rgba(56, 189, 248, 0.5)',
                    borderColor: 'rgba(56, 189, 248, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { grid: { display: false }, ticks: { color: '#ccc' } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    const ctxAcc = document.getElementById('accuracyChart')?.getContext('2d');
    if (ctxAcc) {
        accuracyChart = new Chart(ctxAcc, {
            type: 'line',
            data: {
                labels: ['V1 (Base)', 'V2 (MARG)', 'V3 (FK)', 'V4 (Directional/Live)'],
                datasets: [{
                    label: 'Accuracy %',
                    data: [85, 93, 95, 0], // Live accuracy will be updated
                    borderColor: '#10B981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 70, max: 100, grid: { color: '#222' }, ticks: { color: '#aaa' } },
                    x: { grid: { display: false }, ticks: { color: '#aaa' } }
                }
            }
        });
    }
}
initLabCharts();

async function refreshMlStats() {
    try {
        const res = await fetch('/api/ml/stats');
        if (!res.ok) return;
        const text = await res.text();
        if (text.trim().startsWith('<')) {
            console.warn("Stats API returned HTML. Assuming static hosting.");
            return;
        }
        const stats = JSON.parse(text);

        // Update DOM
        const labAccuracy = document.getElementById('lab-accuracy');
        const labTotal = document.getElementById('lab-total-samples');
        const labTime = document.getElementById('lab-last-trained');
        const labMonitor = document.getElementById('lab-monitor');

        if (labAccuracy) labAccuracy.innerText = (stats.accuracy * 100).toFixed(1) + '%';

        let total = 0;
        const labels = Object.keys(stats.sample_counts).sort();
        const data = labels.map(l => {
            total += stats.sample_counts[l];
            return stats.sample_counts[l];
        });

        if (labTotal) labTotal.innerText = total;
        if (labTime) labTime.innerText = stats.last_trained;

        // Update Chart
        if (sampleDistChart) {
            sampleDistChart.data.labels = labels;
            sampleDistChart.data.datasets[0].data = data;
            sampleDistChart.update();
        }

        if (accuracyChart) {
            // Update the 'Live' point in the accuracy chart
            accuracyChart.data.datasets[0].data[3] = stats.accuracy * 100;
            accuracyChart.update();
        }

        if (labMonitor && stats.last_trained !== 'Never') {
            const time = new Date().toLocaleTimeString();
            const logLine = `<div style="color:#10B981">[${time}] Model updated. Accuracy: ${(stats.accuracy * 100).toFixed(1)}%</div>`;
            labMonitor.innerHTML = logLine + labMonitor.innerHTML;
        }

        // Update Health Checklist
        const healthDiverse = document.getElementById('health-diverse');
        if (healthDiverse) {
            if (labels.length >= 2) {
                healthDiverse.querySelector('.check-icon').innerText = '✓';
                healthDiverse.querySelector('.check-icon').style.color = '#10B981';
            } else {
                healthDiverse.querySelector('.check-icon').innerText = '!';
                healthDiverse.querySelector('.check-icon').style.color = '#f59e0b';
            }
        }

    } catch (e) { console.error("Stats fetch error:", e); }
}

// Stats polling (every 10s)
setInterval(refreshMlStats, 10000);
refreshMlStats();

async function sendMlRecord(label, strokeData) {
    updateMlStatus("Saving...");
    try {
        const res = await fetch('/api/ml/record', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: label, stroke_full: strokeData })
        });
        const json = await res.json();
        if (res.ok) {
            updateMlStatus(`Saved '${label}'!`);
            // Trigger stats refresh after a short delay for training to kick in
            setTimeout(refreshMlStats, 2000);

            // Aesthetic glow animation on the lab accuracy card if visible
            const accCard = document.querySelector('.stat-card.gold-glow');
            if (accCard) {
                accCard.style.boxShadow = '0 0 30px rgba(245, 158, 11, 0.4)';
                setTimeout(() => accCard.style.boxShadow = '', 1000);
            }
        } else {
            updateMlStatus(`Error: ${json.error}`);
        }
    } catch (e) {
        console.error(e);
        updateMlStatus("Network Error");
    }
}

async function sendMlPredict(strokePosData) {
    try {
        const res = await fetch('/api/ml/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stroke_pos: strokePosData })
        });
        const json = await res.json();

        if (res.ok && json.predictions && json.predictions.length > 0) {
            updateScoreBoard(json.predictions);
            updateMlStatus(`Predicted: ${json.predictions[0].label}`);
        } else {
            updateScoreBoard([]);
            updateMlStatus("Unrecognized");
        }
    } catch (e) {
        console.error(e);
    }
}

// ══════════════════════════════════════════
// Camera Controls (matching digital_twin.py)
// ══════════════════════════════════════════
function applyPreset(preset) {
    camDistance = preset.distance;
    camElevation = preset.elevation;
    camAzimuth = preset.azimuth;
    camCenterX = preset.cx;
    camCenterY = preset.cy;
    camCenterZ = preset.cz;
    requestAnimationFrame(renderScene);
}

window.addEventListener('keydown', (e) => {
    if (e.code === 'KeyV') {
        // V = Toggle view (same as digital_twin.py)
        isFPV = !isFPV;
        applyPreset(isFPV ? CAM_1ST : CAM_3RD);
    }
    if (e.code === 'KeyR' || e.code === 'Space') {
        e.preventDefault();
        // R = Reset trail (same as digital_twin.py)
        strokeHistory = [];
        requestAnimationFrame(renderScene);
    }
});

if (canvasContainer) {
    // Mouse drag = rotate camera (azimuth + elevation)
    canvasContainer.addEventListener('mousedown', (e) => {
        isDragging = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        canvasContainer.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        let dx = e.clientX - lastMouseX;
        let dy = e.clientY - lastMouseY;
        camAzimuth += dx * 0.3; // degrees
        camElevation += dy * 0.3;
        camElevation = Math.max(-89, Math.min(89, camElevation));
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        requestAnimationFrame(renderScene);
    });
    window.addEventListener('mouseup', () => {
        isDragging = false;
        canvasContainer.style.cursor = 'crosshair';
    });

    // Scroll = zoom (distance)
    canvasContainer.addEventListener('wheel', (e) => {
        e.preventDefault();
        camDistance *= (1 + e.deltaY * 0.001);
        camDistance = Math.max(0.1, Math.min(5.0, camDistance));
        requestAnimationFrame(renderScene);
    });
}

// ══════════════════════════════════════════
// Chart.js
// ══════════════════════════════════════════
Chart.defaults.color = '#666';
Chart.defaults.font.family = "'JetBrains Mono', monospace";
const ctxLive = document.getElementById('liveChart')?.getContext('2d');
const MAX_DATAPOINTS = 100;
let pitchData = [], rollData = [], yawData = [], labels = [];
let liveChart = null;

if (ctxLive) {
    liveChart = new Chart(ctxLive, {
        type: 'line',
        data: {
            labels, datasets: [
                { label: 'Pitch', borderColor: '#ffffff', data: pitchData, borderWidth: 1, pointRadius: 0, tension: 0.1 },
                { label: 'Roll', borderColor: '#888888', data: rollData, borderWidth: 1, pointRadius: 0, tension: 0.1 },
                { label: 'Yaw', borderColor: '#444444', data: yawData, borderWidth: 1, pointRadius: 0, tension: 0.1 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 10, color: '#aaa', font: { size: 10 } } } },
            scales: { x: { display: false }, y: { min: -180, max: 180, grid: { color: '#222' }, ticks: { stepSize: 90, color: '#666' } } }
        }
    });
}

function updateLiveChart(euler) {
    if (!liveChart) return;
    if (labels.length > MAX_DATAPOINTS) { labels.shift(); pitchData.shift(); rollData.shift(); yawData.shift(); }
    labels.push(''); pitchData.push(euler[0]); rollData.push(euler[1]); yawData.push(euler[2]);
    liveChart.update();
}


connectWebSocket();

// SPA Tabs
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const target = document.getElementById(btn.getAttribute('data-target'));
        if (target) target.classList.add('active');

        // Load comments logic if tab is contact
        if (target && target.id === 'tab-contact') {
            loadComments();
        }
    });
});

// Technology Sidebar Sub-navigation
const techNavBtns = document.querySelectorAll('.tech-nav-btn');
const techPanels = document.querySelectorAll('.tech-panel');
techNavBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        techNavBtns.forEach(b => b.classList.remove('active'));
        techPanels.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const target = document.getElementById(btn.getAttribute('data-panel'));
        if (target) target.classList.add('active');
    });
});

// ══════════════════════════════════════════
// Team Comments System
// ══════════════════════════════════════════
const commentForm = document.getElementById('commentForm');
const commentsList = document.getElementById('commentsList');
const submitBtn = document.getElementById('submitCommentBtn');

async function loadComments() {
    if (!commentsList) return;
    try {
        const res = await fetch('/api/comments');
        if (!res.ok) throw new Error('API Error');
        const comments = await res.json();
        renderComments(comments);
    } catch (e) {
        console.error('Failed to load comments:', e);
        commentsList.innerHTML = '<div class="loading-text" style="color:#ff3333">❌ 백엔드 서버가 로컬에서 실행중이 아닙니다 (정적 파일 모드). Flask앱을 실행해 주세요.</div>';
    }
}

function renderComments(comments) {
    if (comments.length === 0) {
        commentsList.innerHTML = '<div class="loading-text">아직 등록된 의견이 없습니다. 첫 의견을 남겨보세요!</div>';
        return;
    }

    commentsList.innerHTML = '';
    comments.forEach(c => {
        const dateObj = new Date(c.timestamp + 'Z'); // Convert UTC to local
        const dateStr = isNaN(dateObj) ? c.timestamp : dateObj.toLocaleString();

        const card = document.createElement('div');
        card.className = 'comment-card';
        card.innerHTML = `
            <div class="comment-meta">
                <div class="comment-author">${escapeHtml(c.author)}</div>
                <div class="comment-date">${escapeHtml(dateStr)}</div>
            </div>
            <div class="comment-body">${escapeHtml(c.content)}</div>
        `;
        commentsList.appendChild(card);
    });
}

if (commentForm) {
    commentForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const authorInput = document.getElementById('commentAuthor');
        const contentInput = document.getElementById('commentContent');

        const author = authorInput.value.trim();
        const content = contentInput.value.trim();
        if (!author || !content) return;

        // Form styling during submit
        submitBtn.disabled = true;
        submitBtn.textContent = '등록 중...';

        try {
            const res = await fetch('/api/comments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ author, content })
            });
            if (!res.ok) throw new Error('Failed to submit comment');

            // Clear form and reload
            authorInput.value = '';
            contentInput.value = '';
            await loadComments();
        } catch (e) {
            console.error('Submit error:', e);
            alert('댓글 등록에 실패했습니다. (Flask 서버가 켜져있는지 확인하세요)');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = '포스트 등록';
        }
    });
}

function escapeHtml(unsafe) {
    return (unsafe || '').toString()
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ══════════════════════════════════════════
// ACTION DISPATCHER & CONFIG
// ══════════════════════════════════════════
const actionMappingList = document.getElementById('actionMappingList');
const actionConnBadge = document.getElementById('actionConnBadge');
const actionConnDot = document.getElementById('actionConnDot');
const actionConnText = document.getElementById('actionConnText');
const actionHistory = document.getElementById('actionHistory');

let actionWs = null;

async function initActionControl() {
    try {
        const res = await fetch('/api/config/actions');
        if (res.ok) {
            const config = await res.json();
            if (actionMappingList && config.keywords) {
                actionMappingList.innerHTML = '';
                for (const [key, mapping] of Object.entries(config.keywords)) {
                    const li = document.createElement('li');
                    li.innerHTML = `<span class="check-icon" style="color:#38BDF8">⚡</span> <strong>${key}</strong> <span style="color:#64748b">→</span> ${mapping.name} <span style="font-size:11px;color:#64748b">(${mapping.intent})</span>`;
                    actionMappingList.appendChild(li);
                }
            }

            // Connect WS (Use configured port or default 18800)
            const wsPort = config.ports ? config.ports.websocket : 18800;
            const actionWsUrl = `ws://${window.location.hostname}:${wsPort}`;
            connectActionWs(actionWsUrl, wsPort);
        }
    } catch (e) { console.error("Action config error", e); }
}

function connectActionWs(url, port) {
    actionWs = new WebSocket(url);
    actionWs.onopen = () => {
        if (actionConnText) actionConnText.innerText = `CONNECTED (Port ${port})`;
        if (actionConnBadge) { actionConnBadge.style.borderColor = '#10B981'; actionConnBadge.style.color = '#10B981'; }
        if (actionConnDot) actionConnDot.style.background = '#10B981';
    };
    actionWs.onclose = () => {
        if (actionConnText) actionConnText.innerText = `DISCONNECTED (Port ${port})`;
        if (actionConnBadge) { actionConnBadge.style.borderColor = '#f59e0b'; actionConnBadge.style.color = '#f59e0b'; }
        if (actionConnDot) actionConnDot.style.background = '#f59e0b';
        setTimeout(() => connectActionWs(url, port), 3000);
    };
    actionWs.onerror = (e) => console.error("Action WS error", e);

    actionWs.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'action' && actionHistory) {
                const time = new Date().toLocaleTimeString();

                // Remove empty placeholder
                if (actionHistory.innerHTML.includes('No actions triggered')) {
                    actionHistory.innerHTML = '';
                }

                const div = document.createElement('div');
                div.style.marginBottom = '8px';
                div.style.borderBottom = '1px solid rgba(255, 255, 255, 0.05)';
                div.style.paddingBottom = '8px';

                div.innerHTML = `<div style="color:#4ADE80; font-weight:bold;">[${time}] 🚀 ACTION DISPATCH: ${data.label}</div>
                                 <div style="color:#94a3b8; font-size:13px; margin-top:4px;">Keyword: <span style="color:#38bdf8; font-weight:bold;">${data.keyword}</span></div>
                                 <div style="color:#64748b; font-size:12px; margin-top:2px;">Intent: ${data.intent} | Confidence: ${(data.confidence * 100).toFixed(1)}%</div>`;

                actionHistory.prepend(div);

                // Play a brief sound or visual pulse here if desired
            }
        } catch (e) { console.error(e); }
    };
}

initActionControl();
