const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('recordingOverlay');

// DOM Elements
const valConn = document.getElementById('valConn');
const valPos = document.getElementById('valPos');
const valZupt = document.getElementById('valZupt');

// Configuration
const WS_URL = "ws://localhost:8766";
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
    drawAxisArrows(cx, cy);
    drawArm(cx, cy);
    drawStrokes(cx, cy);
    drawCursor(cx, cy);
}

// Floor Grid
function drawGrid(cx, cy) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(26, 26, 51, 0.5)';
    ctx.shadowBlur = 0;
    const size = 1.0, step = 0.1, gz = -0.15;
    ctx.beginPath();
    for (let x = -size; x <= size + 0.001; x += step) {
        let a = projectWorld(x, -size, gz), b = projectWorld(x, size, gz);
        if (a.visible && b.visible) { ctx.moveTo(cx + a.sx, cy + a.sy); ctx.lineTo(cx + b.sx, cy + b.sy); }
    }
    for (let y = -size; y <= size + 0.001; y += step) {
        let a = projectWorld(-size, y, gz), b = projectWorld(size, y, gz);
        if (a.visible && b.visible) { ctx.moveTo(cx + a.sx, cy + a.sy); ctx.lineTo(cx + b.sx, cy + b.sy); }
    }
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
    // Prevent Mixed Content errors on deployed HTTPS sites
    if (window.location.protocol === 'https:' && window.location.hostname !== 'localhost') {
        valConn.innerHTML = "🔴 HTTPS BLOCKED<br><span style='font-size:0.7em; color:#999;'>(Local connection only)</span>";
        valConn.className = "data-value warning";
        console.warn("WebSocket strictly requires HTTP or localhost. Blocked by HTTPS Mixed Content policy.");
        return;
    }

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

    // ── ML Recording / Prediction Logic ──
    if (pen && !lastPenState) {
        // Pen Down -> Start recording
        currentStroke = [];
        strokeHistory.push(currentStroke);
        mlCurrentFull = [];
        mlCurrentPos = [];
        if (mlLearningLabel) {
            updateMlStatus(`Recording '${mlLearningLabel}'...`);
        } else if (mlAutoPredict) {
            updateMlStatus(`Analyzing...`);
            updateScoreBoard([]);
        }
    }

    if (pen) {
        // Pen Drag -> Collect data
        currentStroke.push([...penTip]);
        let s3q = data.S3q || [1, 0, 0, 0];
        mlCurrentPos.push([...penTip]);
        mlCurrentFull.push([
            penTip[0], penTip[1], penTip[2],
            s3q[0], s3q[1], s3q[2], s3q[3]
        ]);
    }

    if (!pen && lastPenState) {
        // Pen Up -> Process Data
        if (mlLearningLabel && mlCurrentFull.length > 5) {
            sendMlRecord(mlLearningLabel, mlCurrentFull);
        } else if (mlAutoPredict && mlCurrentPos.length > 5) {
            sendMlPredict(mlCurrentPos);
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

if (btnMlRec) {
    btnMlRec.addEventListener('click', () => {
        let label = prompt("학습시킬 단어나 글자를 입력하세요 (예: APPLE):");
        if (label && label.trim() !== '') {
            mlLearningLabel = label.trim().toUpperCase();
            btnMlRec.innerHTML = `🔴 '${mlLearningLabel}' 쓰는중... (떼면 완료)`;
            btnMlRec.style.background = '#F87171';
            updateMlStatus("Waiting for pen...");
        }
    });

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
}

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
            // Reset button
            mlLearningLabel = null;
            btnMlRec.innerHTML = `🎯 [가이드] 단어 학습`;
            btnMlRec.style.background = '';
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
const ctxLive = document.getElementById('liveChart').getContext('2d');
const MAX_DATAPOINTS = 100;
let pitchData = [], rollData = [], yawData = [], labels = [];
const liveChart = new Chart(ctxLive, {
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
function updateLiveChart(euler) {
    if (labels.length > MAX_DATAPOINTS) { labels.shift(); pitchData.shift(); rollData.shift(); yawData.shift(); }
    labels.push(''); pitchData.push(euler[0]); rollData.push(euler[1]); yawData.push(euler[2]);
    liveChart.update();
}

const ctxAcc = document.getElementById('accuracyChart');
if (ctxAcc) {
    new Chart(ctxAcc, {
        type: 'bar',
        data: {
            labels: ['Base Accel', 'Accel+Gyro', 'ZUPT+MARG', 'FK 2.5D'],
            datasets: [{
                label: 'Accuracy %', data: [85.2, 88.5, 93.8, 98.1],
                backgroundColor: ['#222', '#444', '#888', '#fff'], borderWidth: 1, borderColor: '#000'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { min: 80, max: 100, grid: { color: '#222' } }, x: { grid: { display: false } } }
        }
    });
}
const ctxDrift = document.getElementById('driftChart');
if (ctxDrift) {
    new Chart(ctxDrift, {
        type: 'line',
        data: {
            labels: ['0s', '5s', '10s', '15s', '20s', '25s', '30s'],
            datasets: [
                { label: 'No ZUPT', data: [0, 0.1, 0.4, 1.2, 2.5, 4.2, 6.5], borderColor: '#444', borderDash: [5, 5] },
                { label: 'Neural ZUPT', data: [0, 0.01, 0.02, 0.01, 0.03, 0.02, 0.01], borderColor: '#fff' }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: { y: { title: { display: true, text: 'Drift (m)', color: '#666' }, grid: { color: '#222' } }, x: { grid: { color: '#222' } } }
        }
    });
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

        // Load comments logic if tab is comments
        if (target && target.id === 'tab-comments') {
            loadComments();
        }
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
