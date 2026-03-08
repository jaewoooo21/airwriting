/* ==============================================
   AirWriting Platform — Main Application Script
   Developer / User Mode System
   ============================================== */

/* ---- Global State ---- */
let currentMode = null;       // 'developer' | 'user'
let currentTab = 'tab-home';
let currentTechPanel = 'arch-panel';
let ws = null;
let demoSimRunning = false;
let demoSimInterval = null;

/* ---- Constants ---- */
const DEV_ID = 'wodn1100';
const DEV_PW = '1234';
const DEV_ONLY_TABS = ['tab-team', 'tab-settings'];  // tabs hidden for user mode
const USER_ONLY_TABS = ['tab-technology', 'tab-contact']; // tabs hidden for dev mode

/* =========================================
   1. INITIALIZATION
   ========================================= */
document.addEventListener('DOMContentLoaded', () => {
  initTabNavigation();
  initTechSidebar();
  initStudioButtons();
  initQrModal();
  initMlLabelModal();
  initComments();
  fetchNetworkIp();
  fetchMlStats();

  // Show mode selection after intro finishes
  setTimeout(() => {
    const modal = document.getElementById('modeSelectModal');
    if (modal) modal.style.display = 'flex';
  }, 4200);
});

/* =========================================
   2. MODE SELECTION
   ========================================= */
window.showDevLogin = function() {
  document.getElementById('devLoginForm').style.display = 'block';
  document.getElementById('devIdInput').focus();
};

window.devLogin = function() {
  const id = document.getElementById('devIdInput').value.trim();
  const pw = document.getElementById('devPwInput').value.trim();
  const err = document.getElementById('devLoginError');

  if (id === DEV_ID && pw === DEV_PW) {
    err.style.display = 'none';
    enterMode('developer');
  } else {
    err.style.display = 'block';
    document.getElementById('devPwInput').value = '';
  }
};

window.enterUserMode = function() {
  enterMode('user');
};

// Allow Enter key in password field
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && document.getElementById('modeSelectModal').style.display === 'flex') {
    const devForm = document.getElementById('devLoginForm');
    if (devForm.style.display === 'block') {
      devLogin();
    }
  }
});

function enterMode(mode) {
  currentMode = mode;
  document.getElementById('modeSelectModal').style.display = 'none';
  applyModeRestrictions();
  selectTab('tab-home');
}

function applyModeRestrictions() {
  const isUser = currentMode === 'user';

  // Hide/show developer-only tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    const target = btn.dataset.target;
    if (DEV_ONLY_TABS.includes(target)) {
      btn.style.display = isUser ? 'none' : 'inline-block';
    }
    if (USER_ONLY_TABS.includes(target)) {
      btn.style.display = isUser ? 'inline-block' : 'none';
    }
  });

  // Hide/show developer-only elements by class
  document.querySelectorAll('.dev-feature').forEach(el => {
    if (isUser) {
      el.style.display = 'none';
    } else {
      if (el.id === 'actionDispatchBoard' || el.style.flexDirection) {
        el.style.display = 'flex';
      } else {
        el.style.display = 'block';
      }
    }
  });

  // Engineering panel visibility (right sidebar in Studio)
  const rightPanel = document.querySelector('.right-panel');
  if (rightPanel) {
    if (isUser) {
      rightPanel.style.display = 'none';
    } else {
      rightPanel.style.display = '';
    }
  }

  // Add mode indicator to nav
  let modeTag = document.getElementById('navModeTag');
  if (!modeTag) {
    modeTag = document.createElement('span');
    modeTag.id = 'navModeTag';
    modeTag.style.cssText = 'font-size:10px; padding:3px 8px; border-radius:4px; margin-left:10px; font-family:"JetBrains Mono",monospace; letter-spacing:0.5px;';
    document.querySelector('.nav-brand').appendChild(modeTag);
  }
  if (currentMode === 'developer') {
    modeTag.textContent = 'DEV';
    modeTag.style.background = 'rgba(139,92,246,0.1)';
    modeTag.style.color = '#8B5CF6';
  } else {
    modeTag.textContent = 'USER';
    modeTag.style.background = 'rgba(16,185,129,0.1)';
    modeTag.style.color = '#10B981';
  }
}

/* =========================================
   3. TAB NAVIGATION
   ========================================= */
function initTabNavigation() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => selectTab(btn.dataset.target));
  });
}

function selectTab(tabId) {
  currentTab = tabId;

  // Update tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.target === tabId);
  });

  // Update tab content
  document.querySelectorAll('.tab-content').forEach(section => {
    section.classList.toggle('active', section.id === tabId);
  });
  
  // reset scroll on tab switch
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* =========================================
   4. TECHNOLOGY SIDEBAR NAVIGATION
   ========================================= */
function initTechSidebar() {
  document.querySelectorAll('.tech-nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const panelId = btn.dataset.panel;
      if (!panelId) return;

      // Update active button
      document.querySelectorAll('.tech-nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Update active panel
      document.querySelectorAll('.tech-panel').forEach(p => {
        p.classList.remove('active');
        p.style.display = 'none';
      });
      const target = document.getElementById(panelId);
      if (target) {
        target.classList.add('active');
        target.style.display = 'block';
        target.style.position = 'relative';
        target.style.opacity = '1';
        target.style.visibility = 'visible';
      }
      currentTechPanel = panelId;
    });
  });

  // Ensure default panel is visible
  const defaultPanel = document.querySelector('.tech-panel.active') || document.querySelector('.tech-panel');
  if (defaultPanel) {
    defaultPanel.style.display = 'block';
    defaultPanel.style.position = 'relative';
    defaultPanel.style.opacity = '1';
    defaultPanel.style.visibility = 'visible';
  }
}

/* =========================================
   5. STUDIO BUTTONS
   ========================================= */
function initStudioButtons() {
  // Connect Android button
  const btnConnect = document.getElementById('btnConnectPhone');
  if (btnConnect) {
    btnConnect.addEventListener('click', () => {
      const modal = document.getElementById('qrModal');
      if (modal) modal.classList.add('active');
    });
  }

  // Demo Simulator button
  const btnDemo = document.getElementById('btnDemoSimulator');
  if (btnDemo) {
    btnDemo.addEventListener('click', toggleDemoSimulator);
  }

  // ML Recording button
  const btnMlRec = document.getElementById('btnMlRec');
  if (btnMlRec) {
    btnMlRec.addEventListener('click', () => {
      const modal = document.getElementById('mlLabelModal');
      if (modal) modal.classList.add('active');
    });
  }

  // ML Predict toggle
  const btnPredict = document.getElementById('btnMlPredict');
  if (btnPredict) {
    btnPredict.addEventListener('click', () => {
      btnPredict.classList.toggle('active');
      const isActive = btnPredict.classList.contains('active');
      btnPredict.textContent = isActive ? '⚡ [자동보정] 끄기' : '⚡ [자동보정] 켜기';
      const status = document.getElementById('mlStatusText');
      if (status) status.textContent = isActive ? 'Status: Auto-correction ON' : 'Status: Idle';
    });
  }

  // HTML buttons now use window.selectTab directly.
}

/* =========================================
   6. DEMO SIMULATOR
   ========================================= */
function toggleDemoSimulator() {
  const btn = document.getElementById('btnDemoSimulator');
  const status = document.getElementById('mlStatusText');
  const canvas = document.getElementById('drawingCanvas');

  if (demoSimRunning) {
    // Stop
    demoSimRunning = false;
    clearInterval(demoSimInterval);
    if (btn) btn.textContent = '▶️ Run Demo Simulator';
    if (status) status.textContent = 'Status: Idle';
    return;
  }

  // Start demo
  demoSimRunning = true;
  if (btn) btn.textContent = '⏹ Stop Simulator';
  if (status) status.textContent = 'Status: Simulating...';

  const resultWord = document.getElementById('aiResultWord');
  const resultScore = document.getElementById('aiResultScore');

  const demoWords = [
    { word: 'ㄱ', score: 97.2 },
    { word: 'ㄴ', score: 95.8 },
    { word: 'ㅏ', score: 98.1 },
    { word: 'A', score: 94.5 },
    { word: 'B', score: 96.3 },
    { word: 'AIR', score: 97.8 }
  ];
  let demoIdx = 0;

  demoSimInterval = setInterval(() => {
    if (!demoSimRunning) return;
    const demo = demoWords[demoIdx % demoWords.length];
    if (resultWord) resultWord.textContent = demo.word;
    if (resultScore) resultScore.textContent = `(${demo.score}%)`;

    // Show recognized overlay
    const overlay = document.getElementById('recognizedTextOverlay');
    if (overlay) {
      overlay.textContent = demo.word;
      overlay.classList.add('active');
      setTimeout(() => overlay.classList.remove('active'), 1500);
    }

    // Update Performance Overlay
    if (currentMode === 'developer') {
      const fpsEl = document.getElementById('perfFps');
      const latEl = document.getElementById('perfLatency');
      const driftEl = document.getElementById('perfDrift');
      const zuptEl = document.getElementById('perfZupt');
      if (fpsEl) fpsEl.textContent = (58 + Math.random() * 4).toFixed(1);
      if (latEl) latEl.textContent = (11 + Math.random() * 3).toFixed(1) + 'ms';
      if (driftEl) driftEl.textContent = (Math.random() * 0.05).toFixed(3) + 'm';
      if (zuptEl) zuptEl.textContent = Math.random() > 0.3 ? 'TRUE' : 'FALSE';
      
      // Update Live Terminal
      const terminal = document.getElementById('terminalOutput');
      if (terminal && Math.random() > 0.7) {
        const msg = document.createElement('div');
        msg.textContent = `[SIM] Replaying frame ${demoIdx}: ZUPT ${Math.random()>0.3?'Stabilized':'Moving'}, Score ${demo.score.toFixed(1)}`;
        terminal.appendChild(msg);
        if (terminal.childElementCount > 30) terminal.removeChild(terminal.firstChild);
        terminal.parentNode.scrollTop = terminal.parentNode.scrollHeight;
      }
    }

    demoIdx++;
  }, 1000); // Changed to 1000ms for more active simulation
}

/* =========================================
   7. QR CONNECT MODAL
   ========================================= */
function initQrModal() {
  const modal = document.getElementById('qrModal');
  const cancelBtn = document.getElementById('btnQrCancel');
  const ipInput = document.getElementById('ipInput');

  if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
      if (modal) modal.classList.remove('active');
    });
  }

  // Auto-fill IP from server
  fetchNetworkIp().then(ip => {
    if (ip && ipInput) ipInput.value = ip;
  });

  // Generate QR on IP input
  if (ipInput) {
    ipInput.addEventListener('input', () => {
      const ip = ipInput.value.trim();
      if (!ip) return;
      generateQr(`ws://${ip}:18800`);
    });
  }

  // Close on backdrop click
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.remove('active');
    });
  }
}

function generateQr(text) {
  const container = document.getElementById('qrCodeContainer');
  if (!container) return;
  container.innerHTML = '';
  if (typeof QRCode !== 'undefined') {
    new QRCode(container, {
      text: text,
      width: 180,
      height: 180,
      colorDark: '#0F172A',
      colorLight: '#FFFFFF',
      correctLevel: QRCode.CorrectLevel.M
    });
  }
}

/* =========================================
   8. ML LABEL MODAL
   ========================================= */
function initMlLabelModal() {
  const modal = document.getElementById('mlLabelModal');
  const cancelBtn = document.getElementById('btnModalCancel');
  const startBtn = document.getElementById('btnModalStart');

  if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
      if (modal) modal.classList.remove('active');
    });
  }

  if (startBtn) {
    startBtn.addEventListener('click', () => {
      const labelInput = document.getElementById('labelInput');
      const label = labelInput ? labelInput.value.trim() : '';
      if (!label) {
        // Check if a grid button was selected
        const selectedKey = document.querySelector('.grid-key-btn.selected');
        if (!selectedKey) { alert('라벨을 입력하거나 키를 선택하세요.'); return; }
      }
      if (modal) modal.classList.remove('active');
      const status = document.getElementById('mlStatusText');
      if (status) status.textContent = `Status: Recording "${label || 'selected key'}" for 3s...`;
      // Auto-reset after 3s
      setTimeout(() => {
        if (status) status.textContent = 'Status: Recording complete!';
        setTimeout(() => { if (status) status.textContent = 'Status: Idle'; }, 2000);
      }, 3000);
    });
  }

  // Grid key buttons
  document.querySelectorAll('.grid-key-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.grid-key-btn').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
      const labelInput = document.getElementById('labelInput');
      if (labelInput) labelInput.value = btn.dataset.key || btn.textContent.trim().charAt(0);
    });
  });

  // Close on backdrop
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.remove('active');
    });
  }
}

/* =========================================
   9. COMMENTS SYSTEM
   ========================================= */
function initComments() {
  loadComments();
  const form = document.getElementById('commentForm');
  if (form) {
    form.addEventListener('submit', handleCommentSubmit);
  }
}

async function loadComments() {
  const list = document.getElementById('commentsList');
  if (!list) return;

  try {
    const res = await fetch('/api/comments');
    if (!res.ok) throw new Error('failed');
    const data = await res.json();
    const comments = Array.isArray(data.comments) ? data.comments : (Array.isArray(data) ? data : []);

    if (comments.length === 0) {
      list.innerHTML = '<div class="loading-text">아직 등록된 의견이 없습니다.</div>';
      return;
    }

    list.innerHTML = '';
    comments.slice().reverse().slice(0, 10).forEach(c => {
      const card = document.createElement('div');
      card.className = 'comment-card';
      const author = escapeHtml(c.author || c.name || 'Anonymous');
      const content = escapeHtml(c.content || c.message || '');
      const date = c.created_at || c.createdAt || c.timestamp || '';
      card.innerHTML = `
        <div class="comment-meta">
          <span class="comment-author">${author}</span>
          <span class="comment-date">${formatTimestamp(date)}</span>
        </div>
        <div class="comment-body">${content}</div>
      `;
      list.appendChild(card);
    });
  } catch (err) {
    list.innerHTML = '<div class="loading-text">댓글을 불러오지 못했습니다.</div>';
  }
}

async function handleCommentSubmit(e) {
  e.preventDefault();
  const author = document.getElementById('commentAuthor');
  const content = document.getElementById('commentContent');
  const btn = document.getElementById('submitCommentBtn');

  if (!author.value.trim() || !content.value.trim()) {
    alert('이름과 메시지를 모두 입력해주세요.');
    return;
  }

  if (btn) btn.disabled = true;
  try {
    const res = await fetch('/api/comments', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ author: author.value.trim(), content: content.value.trim() })
    });
    if (!res.ok) throw new Error('failed');
    author.value = '';
    content.value = '';
    await loadComments();
  } catch (err) {
    alert('메시지를 전송하지 못했습니다.');
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* =========================================
   10. NETWORK IP & ML STATS
   ========================================= */
async function fetchNetworkIp() {
  try {
    const res = await fetch('/api/config/ip');
    const data = await res.json();
    const ip = data.ip || data.local_ip || '';
    const display = document.getElementById('networkIpDisplay');
    if (display && ip) display.textContent = `Network IP: ${ip}`;
    return ip;
  } catch (err) {
    return '';
  }
}

let sampleDistChart = null;

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
                    y: { grid: { display: false }, ticks: { color: '#64748B' } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
}
// Init charts directly or inside fetchMlStats later. We'll init here.
// Note: DOMContentLoaded already fired by the time this is loaded if script is at bottom, but just in case:
document.addEventListener('DOMContentLoaded', initLabCharts);

async function fetchMlStats() {
  if (!sampleDistChart) initLabCharts();

  try {
    const res = await fetch('/api/ml/stats');
    if (!res.ok) return;
    const data = await res.json();

    // Update stat cards if they exist
    const totalEl = document.getElementById('lab-total-samples');
    const accuracyEl = document.getElementById('lab-accuracy');

    if (totalEl && data.total_samples != null) totalEl.textContent = data.total_samples;
    if (accuracyEl && data.accuracy != null) accuracyEl.textContent = (data.accuracy * 100).toFixed(1) + '%';
    
    // Update chart
    if (sampleDistChart && data.class_counts) {
        const labels = Object.keys(data.class_counts);
        const counts = Object.values(data.class_counts);
        sampleDistChart.data.labels = labels;
        sampleDistChart.data.datasets[0].data = counts;
        sampleDistChart.update();
    }
  } catch (err) {
    // ML stats not available
  }
}

/* =========================================
   11. UTILITIES
   ========================================= */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatTimestamp(val) {
  if (!val) return '방금 전';
  const d = new Date(val);
  if (isNaN(d.getTime())) return String(val);
  return new Intl.DateTimeFormat('ko-KR', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
  }).format(d);
}

/* =========================================
   12. SYSTEM SETTINGS (DEV ONLY)
   ========================================= */
async function loadSystemConfig() {
  try {
    const res = await fetch('/api/config/system');
    if (!res.ok) throw new Error('failed loading config');
    const config = await res.json();
    
    // Update YAML Preview
    const preview = document.getElementById('yamlPreviewCode');
    if (preview && typeof jsyaml !== 'undefined') {
      preview.textContent = jsyaml.dump(config, { indent: 2 });
    } else if (preview) {
      preview.textContent = "jsyaml not loaded yet...";
    }

    // Populate Form Fields
    if (config.action_dispatch) {
      if (document.getElementById('confWsPort')) document.getElementById('confWsPort').value = config.action_dispatch.ws_port || 18800;
      if (document.getElementById('confUdpPort')) document.getElementById('confUdpPort').value = config.action_dispatch.udp_port || 12348;
    }
    if (config.fusion) {
      if (document.getElementById('confAccelNoise')) document.getElementById('confAccelNoise').value = config.fusion.accel_noise_std || 0.5;
      if (config.fusion.zupt) {
        if (document.getElementById('confGyroThresh')) document.getElementById('confGyroThresh').value = config.fusion.zupt.gyro_threshold || 0.05;
      }
      if (config.fusion.drift_observer) {
        if (document.getElementById('confDriftObs')) document.getElementById('confDriftObs').checked = !!config.fusion.drift_observer.enabled;
      }
      if (config.fusion.writing_plane) {
        if (document.getElementById('confPlaneLock')) document.getElementById('confPlaneLock').checked = !!config.fusion.writing_plane.absolute_lock;
      }
    }
  } catch (err) {
    if (document.getElementById('yamlPreviewCode')) {
      document.getElementById('yamlPreviewCode').textContent = "Failed to load config. Server unreachable.";
    }
  }
}

async function saveSystemConfig() {
  const btn = document.getElementById('btnSaveConfig');
  const msg = document.getElementById('configSaveMsg');
  if (btn) btn.disabled = true;
  
  // Construct update payload
  const payload = {
    action_dispatch: {
      ws_port: parseInt(document.getElementById('confWsPort').value) || 18800,
      udp_port: parseInt(document.getElementById('confUdpPort').value) || 12348
    },
    fusion: {
      accel_noise_std: parseFloat(document.getElementById('confAccelNoise').value) || 0.5,
      zupt: {
        gyro_threshold: parseFloat(document.getElementById('confGyroThresh').value) || 0.05
      },
      drift_observer: {
        enabled: document.getElementById('confDriftObs').checked
      },
      writing_plane: {
        absolute_lock: document.getElementById('confPlaneLock').checked
      }
    }
  };

  try {
    const res = await fetch('/api/config/system', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Failed to save');
    
    // Reload preview
    await loadSystemConfig();
    
    // Show success message
    if (msg) {
      msg.style.display = 'block';
      setTimeout(() => msg.style.display = 'none', 3000);
    }
  } catch (err) {
    alert("설정 저장에 실패했습니다.");
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Call load on init
document.addEventListener('DOMContentLoaded', () => {
  // ensure jsyaml is loaded
  if (typeof jsyaml === 'undefined') {
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/js-yaml/4.1.0/js-yaml.min.js';
    script.onload = () => loadSystemConfig();
    document.head.appendChild(script);
  } else {
    loadSystemConfig();
  }
});

/* =========================================
   13. EXPLICIT WINDOW BINDINGS FOR HTML 
   ========================================= */
window.saveSystemConfig = saveSystemConfig;
window.selectTab = selectTab;
window.enterMode = enterMode;
window.enterUserMode = function() { enterMode('user'); };
window.devLogin = window.devLogin || function() {
  const id = document.getElementById('devIdInput').value.trim();
  const pw = document.getElementById('devPwInput').value.trim();
  const err = document.getElementById('devLoginError');
  if (id === DEV_ID && pw === DEV_PW) {
    err.style.display = 'none';
    enterMode('developer');
  } else {
    err.style.display = 'block';
    document.getElementById('devPwInput').value = '';
  }
};
window.showDevLogin = function() {
  const form = document.getElementById('devLoginForm');
  if (form) form.style.display = 'block';
};

/* =========================================
   14. WEBSOCKET & 2D CANVAS DRAWING
   ========================================= */
let wsConnection = null;
let isRecording = false;
let recordedFrames = [];

const drawingCanvas = document.getElementById('drawingCanvas');
let ctx;
if (drawingCanvas) {
    const resizeCanvas = () => {
        const rect = drawingCanvas.parentElement.getBoundingClientRect();
        drawingCanvas.width = rect.width;
        drawingCanvas.height = rect.height;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    ctx = drawingCanvas.getContext('2d');
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#38bdf8';
}

let lastX = null, lastY = null;

function initRealTimeSystem() {
    const wsUrl = `ws://${window.location.hostname}:18800`;
    wsConnection = new WebSocket(wsUrl);

    wsConnection.onopen = () => {
        logToTerminal('[NET] WebSocket connected: ' + wsUrl);
        const connEl = document.getElementById('valConn');
        if (connEl) { connEl.textContent = '🟢 CONNECTED'; connEl.classList.remove('warning'); connEl.style.color = '#10b981'; }
    };

    wsConnection.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleIncomingData(data);
        } catch (err) {
            console.error('WS parse error:', err);
        }
    };

    wsConnection.onclose = () => {
        logToTerminal('[NET] WebSocket disconnected. Reconneting in 3s...');
        const connEl = document.getElementById('valConn');
        if (connEl) { connEl.textContent = '🔴 DISCONNECTED'; connEl.classList.add('warning'); connEl.style.color = '#ef4444'; }
        setTimeout(initRealTimeSystem, 3000);
    };

    wsConnection.onerror = (err) => {
        // Ignored to avoid console spam
    };
}

function handleIncomingData(data) {
    if (demoSimRunning) return; // Ignore real data if demo is running

    if (data.type === 'imu_stream' || data.S3q) {
        // 1. Update 3D Hand Mesh
        if (window.handWidgetUpdate) {
            window.handWidgetUpdate(data);
        }

        // 2. Data Recording
        if (isRecording) {
            recordedFrames.push(data);
        }

        // 3. 2D Canvas Drawing (Simple orthogonal projection)
        if (ctx && data.pos) {
            const canvasW = drawingCanvas.width;
            const canvasH = drawingCanvas.height;
            const screenX = canvasW / 2 + (data.pos.x * 500);
            const screenY = canvasH / 2 - (data.pos.z * 500);

            if (data.pen) {
                if (lastX !== null && lastY !== null) {
                    ctx.beginPath();
                    ctx.moveTo(lastX, lastY);
                    ctx.lineTo(screenX, screenY);
                    ctx.stroke();
                }
                lastX = screenX;
                lastY = screenY;
            } else {
                lastX = null;
                lastY = null;
            }
        }
    } else if (data.type === 'ml_result' || data.word) {
        // Handle Recognition Result
        const resultWord = document.getElementById('aiResultWord');
        const resultScore = document.getElementById('aiResultScore');
        if (resultWord) resultWord.textContent = data.word || data.label;
        if (resultScore) {
            const conf = data.score || data.confidence || 0;
            resultScore.textContent = `(${(conf * 100).toFixed(1)}%)`;
        }

        // Add Action Dispatch Toast
        showActionToast(data.word || data.label, data.intent);
    }
}

function logToTerminal(msg) {
    const term = document.getElementById('terminalOutput');
    if (!term) return;
    const div = document.createElement('div');
    div.textContent = msg;
    term.appendChild(div);
    term.parentElement.scrollTop = term.parentElement.scrollHeight;
}

/* =========================================
   15. DATA PIPELINE & ACTION DISPATCH
   ========================================= */
function initDataPipeline() {
    const btnRec = document.getElementById('btnRecData');
    const btnPlay = document.getElementById('btnPlayData');
    const filePlay = document.getElementById('filePlayData');
    const status = document.getElementById('recordStatus');

    if (btnRec) {
        btnRec.addEventListener('click', () => {
            if (!isRecording) {
                // Start Recording
                isRecording = true;
                recordedFrames = [];
                btnRec.textContent = '⏹ Stop Rec';
                btnRec.style.background = '#fbbf24';
                btnRec.style.color = '#1e293b';
                if (status) status.textContent = 'Recording in progress...';
                logToTerminal('[SYS] Data recording started.');
            } else {
                // Stop & Download
                isRecording = false;
                btnRec.textContent = '🔴 Record';
                btnRec.style.background = '#1e293b';
                btnRec.style.color = '#fbbf24';
                if (status) status.textContent = `Recorded ${recordedFrames.length} frames.`;
                logToTerminal(`[SYS] Recording stopped. Saved ${recordedFrames.length} frames.`);
                
                if (recordedFrames.length > 0) {
                    const blob = new Blob([JSON.stringify(recordedFrames, null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `airwriting_record_${Date.now()}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                }
            }
        });
    }

    if (btnPlay) {
        btnPlay.addEventListener('click', () => {
            filePlay.click();
        });
    }

    if (filePlay) {
        filePlay.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (ev) => {
                try {
                    const frames = JSON.parse(ev.target.result);
                    if (status) status.textContent = `Playing ${frames.length} frames...`;
                    logToTerminal(`[SYS] Playing back JSON file: ${file.name}`);
                    playRecordedFrames(frames);
                } catch (err) {
                    console.error('Failed to parse JSON', err);
                    if (status) status.textContent = 'Error loading JSON.';
                }
            };
            reader.readAsText(file);
        });
    }
}

function playRecordedFrames(frames) {
    if (!frames || frames.length === 0) return;
    
    // Clear Canvas
    if (ctx) ctx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    lastX = null;
    lastY = null;
    
    // Stop Demo if running
    if (demoSimRunning) toggleDemoSimulator();

    let i = 0;
    const interval = setInterval(() => {
        if (i >= frames.length) {
            clearInterval(interval);
            const status = document.getElementById('recordStatus');
            if (status) status.textContent = 'Playback finished.';
            logToTerminal('[SYS] Playback finished.');
            return;
        }
        handleIncomingData(frames[i]);
        i++;
    }, 1000 / 85); // approx 85Hz
}

function showActionToast(word, intent) {
    const board = document.getElementById('actionDispatchBoard');
    if (!board || currentMode !== 'developer') return;

    if (!intent) {
        if (word === 'CALL' || word === 'ㄱ') intent = 'Dialer / Phone App';
        else if (word === 'MUSIC' || word === 'ㅁ') intent = 'Music Player Play/Pause';
        else if (word === 'MAP' || word === 'ㄴ') intent = 'Navigation App';
        else intent = `Generic Text Input: ${word}`;
    }

    const toast = document.createElement('div');
    toast.style.background = 'rgba(16, 185, 129, 0.2)';
    toast.style.border = '1px solid rgba(16, 185, 129, 0.5)';
    toast.style.borderRadius = '8px';
    toast.style.padding = '12px';
    toast.style.backdropFilter = 'blur(4px)';
    toast.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
    toast.style.fontFamily = "'Inter', sans-serif";
    toast.style.transform = 'translateX(-20px)';
    toast.style.opacity = '0';
    toast.style.transition = 'all 0.3s ease';

    toast.innerHTML = `
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <span style="font-size:16px;">🚀</span>
            <span style="font-weight:700; color:#10b981; font-size:12px;">ACTION DISPATCHED</span>
        </div>
        <div style="color:#e2e8f0; font-size:14px; margin-bottom:2px;">[${word}] recognized.</div>
        <div style="color:#94a3b8; font-size:11px; font-family:'JetBrains Mono',monospace;">Intent: ${intent}</div>
    `;

    board.prepend(toast);

    setTimeout(() => {
        toast.style.transform = 'translateX(0)';
        toast.style.opacity = '1';
    }, 50);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(-20px)';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

document.addEventListener('DOMContentLoaded', () => {
    initRealTimeSystem();
    initDataPipeline();
});
