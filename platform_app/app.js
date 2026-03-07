const tabs = Array.from(document.querySelectorAll('.nav-link'));
const pages = Array.from(document.querySelectorAll('.page'));
const drawingCanvas = document.getElementById('drawingCanvas');
const stageBannerTitle = document.getElementById('stageBannerTitle');
const stageBannerText = document.getElementById('stageBannerText');
const stageBannerAction = document.getElementById('stageBannerAction');
const recordingOverlay = document.getElementById('recordingOverlay');
const recognizedTextOverlay = document.getElementById('recognizedTextOverlay');
const aiResultWord = document.getElementById('aiResultWord');
const aiResultScore = document.getElementById('aiResultScore');
const aiCandidates = document.getElementById('aiCandidates');
const valConn = document.getElementById('valConn');
const valPos = document.getElementById('valPos');
const valZupt = document.getElementById('valZupt');
const studioWordLabel = document.getElementById('studioWordLabel');
const studioModeBadge = document.getElementById('studioModeBadge');
const modeSummary = document.getElementById('modeSummary');
const modeDemoBtn = document.getElementById('modeDemoBtn');
const modeLiveBtn = document.getElementById('modeLiveBtn');
const btnRunDemo = document.getElementById('btnRunDemo');
const btnTryLive = document.getElementById('btnTryLive');
const btnFallbackDemo = document.getElementById('btnFallbackDemo');
const waveformCanvas = document.getElementById('waveformCanvas');
const commentsList = document.getElementById('commentsList');
const commentsStatus = document.getElementById('commentsStatus');
const commentForm = document.getElementById('commentForm');
const commentFormStatus = document.getElementById('commentFormStatus');
const submitCommentBtn = document.getElementById('submitCommentBtn');
const commentAuthor = document.getElementById('commentAuthor');
const commentContent = document.getElementById('commentContent');
const pairingModeLabel = document.getElementById('pairingModeLabel');
const pairingUrl = document.getElementById('pairingUrl');
const pairingHostState = document.getElementById('pairingHostState');
const pairingQr = document.getElementById('pairingQr');
const pairingHelpText = document.getElementById('pairingHelpText');
const refreshPairingBtn = document.getElementById('refreshPairingBtn');

const drawCtx = drawingCanvas.getContext('2d');
const waveformCtx = waveformCanvas.getContext('2d');

const STUDIO = {
    mode: 'demo',
    ws: null,
    animationHandle: null,
    strokeHistory: [],
    currentStroke: [],
    pointer: [0.0, 0.42, 0.05],
    arm: [],
    isWriting: false,
    euler: [12, -4, 18],
    recognizedTextTimer: null,
    recognition: {
        label: 'IMU',
        score: 98.1,
        candidates: [
            { label: 'IMU', score: 98.1 },
            { label: 'AIR', score: 90.4 },
            { label: 'DRIFT', score: 84.7 },
        ],
    },
    waveHistory: {
        pitch: [],
        roll: [],
        yaw: [],
    },
};

const DEMO_WORDS = [
    {
        label: 'IMU',
        score: 98.1,
        candidates: [
            { label: 'IMU', score: 98.1 },
            { label: 'AIR', score: 91.3 },
            { label: 'WRITE', score: 87.8 },
        ],
        path: buildPath([
            [-0.18, 0.16], [-0.18, -0.12], [-0.12, -0.12], [-0.12, 0.16],
            [-0.02, 0.16], [-0.02, -0.12], [0.04, 0.02], [0.10, -0.12], [0.10, 0.16],
            [0.20, 0.16], [0.20, -0.12], [0.32, -0.12], [0.32, 0.16],
        ]),
    },
    {
        label: 'AIR',
        score: 96.4,
        candidates: [
            { label: 'AIR', score: 96.4 },
            { label: 'IMU', score: 89.8 },
            { label: 'INK', score: 85.1 },
        ],
        path: buildPath([
            [-0.24, -0.12], [-0.16, 0.16], [-0.08, -0.12], [-0.20, 0.00], [-0.12, 0.00],
            [0.02, 0.16], [0.02, -0.12], [0.12, -0.12], [0.12, 0.00], [0.02, 0.00],
            [0.22, 0.16], [0.22, -0.12], [0.22, 0.16], [0.34, -0.12], [0.26, 0.02], [0.36, 0.16],
        ]),
    },
    {
        label: 'DRIFT',
        score: 94.9,
        candidates: [
            { label: 'DRIFT', score: 94.9 },
            { label: 'RIGHT', score: 88.0 },
            { label: 'IMU', score: 84.1 },
        ],
        path: buildPath([
            [-0.30, 0.16], [-0.30, -0.12], [-0.18, -0.12], [-0.12, -0.06], [-0.12, 0.10], [-0.18, 0.16], [-0.30, 0.16],
            [-0.04, 0.16], [-0.04, -0.12], [0.10, -0.12], [0.00, 0.02], [0.10, 0.16],
            [0.18, 0.16], [0.30, 0.16], [0.24, 0.16], [0.24, -0.12],
        ]),
    },
];

let demoLoop = {
    wordIndex: 0,
    pointIndex: 0,
    phase: 'writing',
    pauseUntil: 0,
};

function buildPath(points) {
    const result = [];
    for (let i = 0; i < points.length - 1; i += 1) {
        const [x0, z0] = points[i];
        const [x1, z1] = points[i + 1];
        for (let step = 0; step < 8; step += 1) {
            const t = step / 8;
            result.push([
                lerp(x0, x1, t),
                0.52 + Math.sin((i + t) * 0.8) * 0.01,
                lerp(z0, z1, t),
            ]);
        }
    }
    result.push([points[points.length - 1][0], 0.52, points[points.length - 1][1]]);
    return result;
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function setActiveTab(targetId) {
    tabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.target === targetId));
    pages.forEach((page) => page.classList.toggle('active', page.id === targetId));
    if (targetId === 'tab-contact') {
        loadComments();
    } else if (targetId === 'tab-android') {
        refreshPairingInfo();
    }
}

tabs.forEach((tab) => {
    tab.addEventListener('click', () => setActiveTab(tab.dataset.target));
});

document.querySelectorAll('[data-target-jump]').forEach((button) => {
    button.addEventListener('click', () => setActiveTab(button.dataset.targetJump));
});

document.querySelectorAll('[data-open-studio]').forEach((button) => {
    button.addEventListener('click', () => {
        setActiveTab('tab-studio');
        setStudioMode(button.dataset.openStudio === 'live' ? 'live' : 'demo');
    });
});

modeDemoBtn.addEventListener('click', () => setStudioMode('demo'));
modeLiveBtn.addEventListener('click', () => setStudioMode('live'));
btnRunDemo.addEventListener('click', () => setStudioMode('demo'));
btnTryLive.addEventListener('click', () => setStudioMode('live'));
btnFallbackDemo.addEventListener('click', () => setStudioMode('demo'));
stageBannerAction.addEventListener('click', () => {
    setStudioMode(STUDIO.mode === 'demo' ? 'live' : 'demo');
});

function setStudioMode(mode) {
    STUDIO.mode = mode;
    modeDemoBtn.classList.toggle('active', mode === 'demo');
    modeLiveBtn.classList.toggle('active', mode === 'live');
    studioModeBadge.textContent = mode === 'demo' ? 'Demo playback' : 'Live relay';
    studioModeBadge.style.color = mode === 'demo' ? '#10b981' : '#0ea5e9';

    if (mode === 'demo') {
        teardownLiveSocket();
        startDemoLoop();
        updateBanner(
            'Demo mode is active',
            'This public deployment is rendering bundled sample strokes so the studio remains useful without sensors.',
            'Try live mode'
        );
        modeSummary.textContent = 'Demo mode ships with bundled playback. It updates the board, waveform, and label cards without any hardware.';
        valConn.textContent = 'Demo running';
    } else {
        stopDemoLoop();
        connectLiveSocket();
        updateBanner(
            'Live mode is waiting for a relay',
            'Use this when your local websocket relay is online. If the connection fails, the page will suggest a one-click return to Demo mode.',
            'Return to demo'
        );
        modeSummary.textContent = 'Live mode performs a single websocket attempt. When the relay is unavailable, the UI reports that state instead of looping forever.';
        valConn.textContent = 'Connecting to relay';
    }
}

function updateBanner(title, text, actionText) {
    stageBannerTitle.textContent = title;
    stageBannerText.textContent = text;
    stageBannerAction.textContent = actionText;
}

function connectLiveSocket() {
    teardownLiveSocket();
    valConn.textContent = 'Connecting to relay';
    valZupt.textContent = 'WAITING';

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsPort = window.location.hostname === 'localhost' ? ':18765' : '';
    const wsUrl = `${protocol}//${window.location.hostname}${wsPort}`;

    try {
        STUDIO.ws = new WebSocket(wsUrl);
    } catch (error) {
        handleLiveUnavailable(`Live socket failed to initialize: ${error.message}`);
        return;
    }

    STUDIO.ws.onopen = () => {
        valConn.textContent = 'Live relay connected';
        valZupt.textContent = 'LIVE';
    };

    STUDIO.ws.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            if (payload.t === 'f') {
                updateFromLiveFrame(payload);
            }
        } catch (error) {
            console.error('Live frame parse error:', error);
        }
    };

    STUDIO.ws.onerror = () => {
        handleLiveUnavailable('Live relay could not be reached from this deployment.');
    };

    STUDIO.ws.onclose = () => {
        if (STUDIO.mode === 'live') {
            handleLiveUnavailable('Live relay disconnected. Switch back to Demo mode or re-run your local relay.');
        }
    };
}

function teardownLiveSocket() {
    if (STUDIO.ws) {
        STUDIO.ws.onopen = null;
        STUDIO.ws.onclose = null;
        STUDIO.ws.onmessage = null;
        STUDIO.ws.onerror = null;
        STUDIO.ws.close();
        STUDIO.ws = null;
    }
}

function handleLiveUnavailable(message) {
    valConn.textContent = 'Live unavailable';
    valZupt.textContent = 'OFFLINE';
    updateBanner('Live relay unavailable', message, 'Return to demo');
}

function updateFromLiveFrame(data) {
    const positions = computeFK(data);
    STUDIO.arm = positions;
    STUDIO.pointer = positions[positions.length - 1];
    STUDIO.isWriting = Boolean(data.pen);
    STUDIO.euler = data.S3e || STUDIO.euler;

    if (STUDIO.isWriting) {
        STUDIO.currentStroke.push([...STUDIO.pointer]);
    } else if (STUDIO.currentStroke.length > 0) {
        STUDIO.strokeHistory.push(STUDIO.currentStroke.slice());
        STUDIO.currentStroke = [];
    }

    valPos.textContent = `X ${STUDIO.pointer[0].toFixed(3)} · Y ${STUDIO.pointer[1].toFixed(3)} · Z ${STUDIO.pointer[2].toFixed(3)}`;
    valZupt.textContent = data.S3z ? 'ACTIVE' : 'INACTIVE';
    pushWavePoint(STUDIO.euler);
    renderStudio();
}

function computeFK(data) {
    const segments = [
        { sid: 'S1', length: 0.25 },
        { sid: 'S2', length: 0.18 },
        { sid: 'S3', length: 0.08 },
    ];
    const positions = [[0, 0, 0]];
    let current = [0, 0, 0];
    segments.forEach((segment) => {
        const quaternion = data[`${segment.sid}q`] || [1, 0, 0, 0];
        const rotation = quatToRot(quaternion);
        const bone = matMul(rotation, [0, segment.length, 0]);
        current = [current[0] + bone[0], current[1] + bone[1], current[2] + bone[2]];
        positions.push(current);
    });
    return positions;
}

function quatToRot([w, x, y, z]) {
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ];
}

function matMul(rotation, vector) {
    return [
        rotation[0][0] * vector[0] + rotation[0][1] * vector[1] + rotation[0][2] * vector[2],
        rotation[1][0] * vector[0] + rotation[1][1] * vector[1] + rotation[1][2] * vector[2],
        rotation[2][0] * vector[0] + rotation[2][1] * vector[1] + rotation[2][2] * vector[2],
    ];
}

function startDemoLoop() {
    stopDemoLoop();
    demoLoop = {
        wordIndex: 0,
        pointIndex: 0,
        phase: 'writing',
        pauseUntil: performance.now() + 400,
    };
    STUDIO.strokeHistory = [];
    STUDIO.currentStroke = [];
    STUDIO.arm = [];
    STUDIO.pointer = [0.0, 0.42, 0.05];
    STUDIO.isWriting = false;
    STUDIO.waveHistory = { pitch: [], roll: [], yaw: [] };

    const tick = (now) => {
        runDemoStep(now);
        renderStudio();
        STUDIO.animationHandle = requestAnimationFrame(tick);
    };
    STUDIO.animationHandle = requestAnimationFrame(tick);
}

function stopDemoLoop() {
    if (STUDIO.animationHandle) {
        cancelAnimationFrame(STUDIO.animationHandle);
        STUDIO.animationHandle = null;
    }
}

function runDemoStep(now) {
    const word = DEMO_WORDS[demoLoop.wordIndex];
    if (now < demoLoop.pauseUntil) {
        STUDIO.isWriting = false;
        return;
    }

    if (demoLoop.phase === 'writing') {
        STUDIO.isWriting = true;
        const point = word.path[demoLoop.pointIndex];
        STUDIO.pointer = [...point];
        STUDIO.currentStroke.push([...point]);
        STUDIO.arm = buildArmToPointer(point);
        STUDIO.euler = [
            18 * Math.sin(now / 600 + demoLoop.pointIndex * 0.04),
            10 * Math.cos(now / 720 + demoLoop.pointIndex * 0.03),
            24 * Math.sin(now / 550 + demoLoop.pointIndex * 0.02),
        ];
        valPos.textContent = `X ${point[0].toFixed(3)} · Y ${point[1].toFixed(3)} · Z ${point[2].toFixed(3)}`;
        valZupt.textContent = demoLoop.pointIndex % 14 < 4 ? 'ACTIVE' : 'TRACKING';
        pushWavePoint(STUDIO.euler);

        demoLoop.pointIndex += 1;
        if (demoLoop.pointIndex >= word.path.length) {
            STUDIO.strokeHistory.push(STUDIO.currentStroke.slice());
            STUDIO.currentStroke = [];
            STUDIO.recognition = {
                label: word.label,
                score: word.score,
                candidates: word.candidates,
            };
            updateRecognition(word.label, word.score, word.candidates);
            valConn.textContent = `Demo playback · ${word.label}`;
            demoLoop.phase = 'pause';
            demoLoop.pauseUntil = now + 1150;
        }
    } else {
        STUDIO.isWriting = false;
        showRecognizedOverlay(word.label);
        demoLoop.wordIndex = (demoLoop.wordIndex + 1) % DEMO_WORDS.length;
        demoLoop.pointIndex = 0;
        demoLoop.phase = 'writing';
        demoLoop.pauseUntil = now + 380;
    }
}

function buildArmToPointer(pointer) {
    const shoulder = [0.0, 0.0, -0.08];
    const elbow = [pointer[0] * 0.35, pointer[1] * 0.45, pointer[2] * 0.25];
    const wrist = [pointer[0] * 0.7, pointer[1] * 0.82, pointer[2] * 0.55];
    const tip = [pointer[0], pointer[1], pointer[2]];
    return [shoulder, elbow, wrist, tip];
}

function updateRecognition(label, score, candidates) {
    aiResultWord.textContent = label;
    aiResultScore.textContent = `${score.toFixed(1)}%`;
    studioWordLabel.textContent = label;
    aiCandidates.innerHTML = '';
    candidates.forEach((candidate) => {
        const item = document.createElement('li');
        item.innerHTML = `<strong>${escapeHtml(candidate.label)}</strong><span>${candidate.score.toFixed(1)}%</span>`;
        aiCandidates.appendChild(item);
    });
}

function showRecognizedOverlay(label) {
    recognizedTextOverlay.textContent = label;
    recognizedTextOverlay.classList.add('active');
    clearTimeout(STUDIO.recognizedTextTimer);
    STUDIO.recognizedTextTimer = window.setTimeout(() => {
        recognizedTextOverlay.classList.remove('active');
    }, 850);
}

function pushWavePoint(euler) {
    [['pitch', euler[0]], ['roll', euler[1]], ['yaw', euler[2]]].forEach(([key, value]) => {
        STUDIO.waveHistory[key].push(value);
        if (STUDIO.waveHistory[key].length > 80) {
            STUDIO.waveHistory[key].shift();
        }
    });
}

function resizeCanvas() {
    const bounds = drawingCanvas.parentElement.getBoundingClientRect();
    drawingCanvas.width = Math.max(320, Math.floor(bounds.width));
    drawingCanvas.height = Math.max(420, Math.floor(bounds.height));
    renderStudio();
    renderWaveform();
}

window.addEventListener('resize', resizeCanvas);

function project([x, y, z]) {
    const azimuth = 0.82;
    const elevation = 0.42;
    const dx = x;
    const dy = y - 0.28;
    const dz = z;

    const x1 = dx * Math.cos(azimuth) + dy * Math.sin(azimuth);
    const y1 = -dx * Math.sin(azimuth) + dy * Math.cos(azimuth);
    const z1 = dz;

    const y2 = y1 * Math.cos(elevation) + z1 * Math.sin(elevation);
    const z2 = -y1 * Math.sin(elevation) + z1 * Math.cos(elevation);

    const depth = y2 + 0.65;
    const focal = drawingCanvas.width * 0.72;
    return {
        x: drawingCanvas.width / 2 + focal * (x1 / depth),
        y: drawingCanvas.height * 0.55 - focal * (z2 / depth),
    };
}

function renderStudio() {
    drawCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    drawBackground();
    drawBoard();
    drawStrokeHistory();
    drawArm();
    drawCursor();
    renderWaveform();
    recordingOverlay.classList.toggle('active', STUDIO.isWriting);
}

function drawBackground() {
    const gradient = drawCtx.createLinearGradient(0, 0, 0, drawingCanvas.height);
    gradient.addColorStop(0, '#10201d');
    gradient.addColorStop(1, '#214544');
    drawCtx.fillStyle = gradient;
    drawCtx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);

    drawCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    drawCtx.lineWidth = 1;
    for (let x = 0; x < drawingCanvas.width; x += 56) {
        drawCtx.beginPath();
        drawCtx.moveTo(x, 0);
        drawCtx.lineTo(x, drawingCanvas.height);
        drawCtx.stroke();
    }
    for (let y = 0; y < drawingCanvas.height; y += 56) {
        drawCtx.beginPath();
        drawCtx.moveTo(0, y);
        drawCtx.lineTo(drawingCanvas.width, y);
        drawCtx.stroke();
    }
}

function drawBoard() {
    const corners = [
        [-0.36, 0.52, 0.24],
        [0.36, 0.52, 0.24],
        [0.36, 0.52, -0.18],
        [-0.36, 0.52, -0.18],
    ].map(project);

    drawCtx.fillStyle = 'rgba(255, 255, 255, 0.06)';
    drawCtx.strokeStyle = 'rgba(117, 219, 201, 0.45)';
    drawCtx.lineWidth = 2;
    drawCtx.beginPath();
    corners.forEach((point, index) => {
        if (index === 0) {
            drawCtx.moveTo(point.x, point.y);
        } else {
            drawCtx.lineTo(point.x, point.y);
        }
    });
    drawCtx.closePath();
    drawCtx.fill();
    drawCtx.stroke();
}

function drawStrokeHistory() {
    drawCtx.lineCap = 'round';
    drawCtx.lineJoin = 'round';

    STUDIO.strokeHistory.forEach((stroke) => {
        drawStroke(stroke, 'rgba(134, 239, 172, 0.95)', 4);
    });
    drawStroke(STUDIO.currentStroke, 'rgba(255, 255, 255, 0.95)', 4.5);
}

function drawStroke(stroke, color, width) {
    if (stroke.length < 2) {
        return;
    }
    drawCtx.strokeStyle = color;
    drawCtx.shadowColor = color;
    drawCtx.shadowBlur = 14;
    drawCtx.lineWidth = width;
    drawCtx.beginPath();
    stroke.map(project).forEach((point, index) => {
        if (index === 0) {
            drawCtx.moveTo(point.x, point.y);
        } else {
            drawCtx.lineTo(point.x, point.y);
        }
    });
    drawCtx.stroke();
    drawCtx.shadowBlur = 0;
}

function drawArm() {
    if (!STUDIO.arm.length) {
        return;
    }
    const colors = ['rgba(255,255,255,0.22)', 'rgba(14,165,233,0.8)', 'rgba(16,185,129,0.9)'];
    for (let index = 0; index < STUDIO.arm.length - 1; index += 1) {
        const start = project(STUDIO.arm[index]);
        const end = project(STUDIO.arm[index + 1]);
        drawCtx.strokeStyle = colors[index] || 'rgba(255,255,255,0.5)';
        drawCtx.lineWidth = index === STUDIO.arm.length - 2 ? 6 : 8;
        drawCtx.beginPath();
        drawCtx.moveTo(start.x, start.y);
        drawCtx.lineTo(end.x, end.y);
        drawCtx.stroke();
    }
}

function drawCursor() {
    const pointer = project(STUDIO.pointer);
    drawCtx.fillStyle = STUDIO.mode === 'demo' ? '#f5d089' : '#57c8ff';
    drawCtx.beginPath();
    drawCtx.arc(pointer.x, pointer.y, STUDIO.isWriting ? 7 : 5, 0, Math.PI * 2);
    drawCtx.fill();
}

function renderWaveform() {
    waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    waveformCtx.fillStyle = 'rgba(16, 32, 29, 0.08)';
    waveformCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

    waveformCtx.strokeStyle = 'rgba(16, 32, 29, 0.12)';
    waveformCtx.lineWidth = 1;
    for (let row = 1; row <= 3; row += 1) {
        const y = (waveformCanvas.height / 4) * row;
        waveformCtx.beginPath();
        waveformCtx.moveTo(0, y);
        waveformCtx.lineTo(waveformCanvas.width, y);
        waveformCtx.stroke();
    }

    drawWave('pitch', '#57c8ff');
    drawWave('roll', '#10b981');
    drawWave('yaw', '#f5d089');
}

function drawWave(key, color) {
    const values = STUDIO.waveHistory[key];
    if (!values.length) {
        return;
    }
    waveformCtx.strokeStyle = color;
    waveformCtx.lineWidth = 2;
    waveformCtx.beginPath();
    values.forEach((value, index) => {
        const x = (index / Math.max(values.length - 1, 1)) * waveformCanvas.width;
        const y = waveformCanvas.height / 2 - value * 1.4;
        if (index === 0) {
            waveformCtx.moveTo(x, y);
        } else {
            waveformCtx.lineTo(x, y);
        }
    });
    waveformCtx.stroke();
}

async function loadComments() {
    if (!commentsList) {
        return;
    }
    commentsStatus.textContent = 'Loading comments';
    commentsList.innerHTML = '<div class="comment-empty">Loading feedback from the deployed Flask API...</div>';

    try {
        const response = await fetch('/api/comments');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const comments = await response.json();
        commentsStatus.textContent = comments.length ? 'Live comment feed' : 'No comments yet';
        renderComments(comments);
    } catch (error) {
        commentsStatus.textContent = 'Comment API unavailable';
        commentsList.innerHTML = '<div class="comment-empty">The feedback API is currently unavailable. Run the Flask app locally or redeploy the service if comments are required.</div>';
    }
}

function renderComments(comments) {
    if (!comments.length) {
        commentsList.innerHTML = '<div class="comment-empty">No feedback is stored yet. Post the first review from this page.</div>';
        return;
    }

    commentsList.innerHTML = '';
    comments.forEach((comment) => {
        const date = new Date(`${comment.timestamp}Z`);
        const item = document.createElement('article');
        item.className = 'comment-card';
        item.innerHTML = `
            <div class="comment-meta">
                <strong>${escapeHtml(comment.author)}</strong>
                <span>${escapeHtml(isNaN(date) ? comment.timestamp : date.toLocaleString())}</span>
            </div>
            <p>${escapeHtml(comment.content)}</p>
        `;
        commentsList.appendChild(item);
    });
}

if (commentForm) {
    commentForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const author = commentAuthor.value.trim();
        const content = commentContent.value.trim();
        if (!author || !content) {
            commentFormStatus.textContent = 'Name and message are both required.';
            return;
        }

        submitCommentBtn.disabled = true;
        commentFormStatus.textContent = 'Posting feedback...';

        try {
            const response = await fetch('/api/comments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ author, content }),
            });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            commentAuthor.value = '';
            commentContent.value = '';
            commentFormStatus.textContent = 'Feedback posted.';
            await loadComments();
        } catch (error) {
            commentFormStatus.textContent = 'Feedback could not be posted. Check that the Flask API is online.';
        } finally {
            submitCommentBtn.disabled = false;
        }
    });
}

function escapeHtml(value) {
    return (value || '')
        .toString()
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

let pairingQrInstance = null;

async function refreshPairingInfo() {
    if (!pairingUrl) {
        return;
    }

    const isPublicDeploy = !['localhost', '127.0.0.1'].includes(window.location.hostname) && window.location.hostname.includes('onrender.com');

    pairingModeLabel.textContent = isPublicDeploy ? 'Public deployment mode' : 'Local pairing mode';

    try {
        const response = await fetch('/api/config/ip');
        const data = await response.json();
        const localIp = data && data.ip ? data.ip : '127.0.0.1';
        const wsTarget = `ws://${localIp}:18800`;

        if (isPublicDeploy) {
            pairingUrl.textContent = wsTarget;
            pairingHostState.textContent = 'Public host cannot expose your private PC directly';
            pairingHelpText.textContent = 'This public Render site can explain the pairing flow, but the Android app should connect to the PC running your local AirWriting services. Run platform_app and action_dispatcher.py on that PC, then use the same URL format shown here.';
            renderPairingQr(null);
        } else {
            pairingUrl.textContent = wsTarget;
            pairingHostState.textContent = `Local host ready at ${localIp}:18800`;
            pairingHelpText.textContent = 'Use this QR code from the Android app or type the IP manually. The phone and PC must share the same Wi-Fi network, and action_dispatcher.py must be running.';
            renderPairingQr(wsTarget);
        }
    } catch (error) {
        pairingUrl.textContent = 'ws://unavailable:18800';
        pairingHostState.textContent = 'Could not fetch pairing IP';
        pairingHelpText.textContent = 'Start the Flask service locally to expose /api/config/ip, then refresh this panel. Public Render deploys cannot infer your private LAN target.';
        renderPairingQr(null);
    }
}

function renderPairingQr(text) {
    if (!pairingQr) {
        return;
    }
    pairingQr.innerHTML = '';
    pairingQrInstance = null;

    if (!text) {
        const placeholder = document.createElement('div');
        placeholder.className = 'muted-copy';
        placeholder.style.textAlign = 'center';
        placeholder.textContent = 'QR is available when this page is served from a local AirWriting host.';
        pairingQr.appendChild(placeholder);
        return;
    }

    if (typeof QRCode === 'undefined') {
        const fallback = document.createElement('div');
        fallback.className = 'muted-copy';
        fallback.textContent = text;
        pairingQr.appendChild(fallback);
        return;
    }

    pairingQrInstance = new QRCode(pairingQr, {
        text,
        width: 190,
        height: 190,
        colorDark: '#10201d',
        colorLight: '#ffffff',
        correctLevel: QRCode.CorrectLevel.H,
    });
}

if (refreshPairingBtn) {
    refreshPairingBtn.addEventListener('click', refreshPairingInfo);
}

updateRecognition(STUDIO.recognition.label, STUDIO.recognition.score, STUDIO.recognition.candidates);
resizeCanvas();
refreshPairingInfo();
setStudioMode('demo');
