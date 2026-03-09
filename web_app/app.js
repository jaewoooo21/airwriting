/* ==============================================
   AirWriting Platform — Main Application Script
   Developer / User Mode System
   ============================================== */

/* ---- Global State ---- */
let currentTab = 'tab-home';
let currentTechPanel = 'arch-panel';
let ws = null;

/* ---- Constants ---- */
const DEV_PW = '1234';

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

  // Demo mode modal removed. Enter developer mode immediately.
  enterMode('developer');
});

/* =========================================
   2. MODE SELECTION
   ========================================= */
window.toggleDevMode = function() {
    const body = document.body;
    const btn = document.getElementById('btnAdminToggle');
    
    if (body.classList.contains('mode-user')) {
        const pw = prompt("엔지니어 관리자 비밀번호를 입력하세요 (기본 '1234'):");
        if (pw === DEV_PW || pw === 'admin') {
            body.classList.remove('mode-user');
            body.classList.add('mode-dev');
            if(btn) {
                btn.textContent = '🔓 엔지니어 모드 종료';
                btn.style.background = '#ef4444';
                btn.style.color = '#fff';
                btn.style.borderColor = '#ef4444';
            }
            if(document.getElementById('navSettingsBtn')) document.getElementById('navSettingsBtn').style.display = 'inline-block';
            if(document.getElementById('navTeamBtn')) document.getElementById('navTeamBtn').style.display = 'inline-block';
            if(document.getElementById('navContactBtn')) document.getElementById('navContactBtn').style.display = 'none';
        } else if (pw !== null) {
            alert('비밀번호가 일치하지 않습니다.');
        }
    } else {
        body.classList.remove('mode-dev');
        body.classList.add('mode-user');
        if(btn) {
            btn.textContent = '🔒 Admin Login';
            btn.style.background = '';
            btn.style.color = '';
            btn.style.borderColor = '';
        }
        if(document.getElementById('navSettingsBtn')) document.getElementById('navSettingsBtn').style.display = 'none';
        if(document.getElementById('navTeamBtn')) document.getElementById('navTeamBtn').style.display = 'none';
        if(document.getElementById('navContactBtn')) document.getElementById('navContactBtn').style.display = 'inline-block';
    }
};

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
      const status = document.getElementById('mlStatusText');
      if (btnPredict.classList.contains('active')) {
        btnPredict.classList.remove('active');
        btnPredict.textContent = '⚡ [자동보정] 켜기';
        btnPredict.style.background = '#1e293b';
        btnPredict.style.color = '#38bdf8';
        if (status) status.textContent = 'Status: Predict OFF';
        isPredictMode = false;
      } else {
        btnPredict.classList.add('active');
        btnPredict.textContent = '⚡ [자동보정] 끄기';
        btnPredict.style.background = '#38bdf8';
        btnPredict.style.color = '#0f172a';
        if (status) status.textContent = 'Status: Predict ON (Draw & Release)';
        isPredictMode = true;
        recordedFrames = []; // Clear buffer for new stroke
      }
    });
  }

  // HTML buttons now use window.selectTab directly.
}

/* =========================================
   6. REMOVED DEMO SIMULATOR
   ========================================= */

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
   8. ACTUAL ML RECORDING LOGIC
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
      let label = labelInput ? labelInput.value.trim() : '';
      if (!label) {
        // Check if a grid button was selected
        const selectedKey = document.querySelector('.grid-key-btn.selected');
        if (!selectedKey) { alert('라벨을 입력하거나 키를 선택하세요.'); return; }
        label = selectedKey.dataset.key || selectedKey.textContent.trim().charAt(0);
      }
      
      if (modal) modal.classList.remove('active');
      const status = document.getElementById('mlStatusText');
      
      if (status) {
          status.textContent = `Status: Ready to record "${label}". Start writing!`;
      }
      
      pendingRecordLabel = label;
      isWaitingForStroke = true;
      recordedFrames = [];
      if (strokeTimer) { clearTimeout(strokeTimer); strokeTimer = null; }
    });
  }

async function submitRecording(label, frames) {
    const status = document.getElementById('mlStatusText');
    if (!frames || frames.length < 5) {
        if (status) status.textContent = 'Status: Data too short, try again.';
        return;
    }
    
    // Extract format needed by backend
    const stroke_full = frames.map(f => {
        if (f.pos && f.S3q) {
            return [f.pos.x, f.pos.y, f.pos.z, f.S3q.w, f.S3q.x, f.S3q.y, f.S3q.z];
        } else {
            return [0,0,0, 1,0,0,0];
        }
    });
    
    if (status) status.textContent = `Status: Sending "${label}" ...`;

    try {
        const res = await fetch('/api/ml/record', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: label, stroke_full: stroke_full })
        });
        
        if (!res.ok) throw new Error('Network error');
        const result = await res.json();
        
        if (status) status.textContent = `Status: ${result.message || 'Saved successfully'}`;
        setTimeout(() => { if (status) status.textContent = 'Status: Idle'; }, 3000);
        
    } catch (err) {
        console.error("Recording save failed:", err);
        if (status) status.textContent = 'Status: Error saving data';
    }
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

/* =========================================
   14. WEBSOCKET & 2D CANVAS DRAWING
   ========================================= */
let wsConnection = null;
let isRecording = false;
let isPredictMode = false;
let recordedFrames = [];
let wasPenDown = false;
let isWaitingForStroke = false;
let pendingRecordLabel = '';
let strokeTimer = null;

const drawingCanvas = document.getElementById('drawingCanvas');

let trajScene, trajCamera, trajRenderer, trajControls;
let trajLine, trajMaterial, trajGeometry;
let trajPenTip, trajFloor;
let trajPositions = [];
let trajColors = [];
let lastTrajPos = null;
let lastTrajTime = 0;

function init3DCanvas() {
    if (!drawingCanvas) return;
    
    const rect = drawingCanvas.parentElement.getBoundingClientRect();
    
    trajScene = new THREE.Scene();
    trajCamera = new THREE.PerspectiveCamera(45, rect.width / rect.height, 0.1, 100);
    trajCamera.position.set(0, 1.5, 3); // Positioned above and looking slightly down
    trajCamera.lookAt(0, 0, 0);
    
    trajRenderer = new THREE.WebGLRenderer({ canvas: drawingCanvas, antialias: true, alpha: true });
    trajRenderer.setSize(rect.width, rect.height);
    trajRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    trajRenderer.setClearColor(0x000000, 0);
    trajRenderer.shadowMap.enabled = true;
    
    if (typeof THREE.OrbitControls !== 'undefined') {
        trajControls = new THREE.OrbitControls(trajCamera, trajRenderer.domElement);
        trajControls.enableDamping = true;
        trajControls.dampingFactor = 0.05;
    }
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    trajScene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(0, 4, 2);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    trajScene.add(dirLight);
    
    // Shadow receiving visual floor
    const floorGeo = new THREE.PlaneGeometry(6, 6);
    const floorMat = new THREE.MeshStandardMaterial({ 
        color: 0x0f172a, transparent: true, opacity: 0.6, roughness: 0.9 
    });
    trajFloor = new THREE.Mesh(floorGeo, floorMat);
    trajFloor.rotation.x = -Math.PI / 2;
    trajFloor.position.y = -1.0;
    trajFloor.receiveShadow = true;
    trajScene.add(trajFloor);
    
    const grid = new THREE.GridHelper(6, 30, 0x334155, 0x1e293b);
    grid.position.y = -0.99;
    trajScene.add(grid);
    
    // Shadow-casting pen tip
    const tipGeo = new THREE.SphereGeometry(0.04, 16, 16);
    const tipMat = new THREE.MeshStandardMaterial({ 
        color: 0x38bdf8, emissive: 0x0ea5e9, emissiveIntensity: 0.8 
    });
    trajPenTip = new THREE.Mesh(tipGeo, tipMat);
    trajPenTip.castShadow = true;
    trajPenTip.visible = false;
    trajScene.add(trajPenTip);
    
    trajGeometry = new THREE.BufferGeometry();
    trajMaterial = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 3 });
    trajLine = new THREE.Line(trajGeometry, trajMaterial);
    trajScene.add(trajLine);
    
    function animate() {
        requestAnimationFrame(animate);
        if (trajControls) trajControls.update();
        trajRenderer.render(trajScene, trajCamera);
    }
    animate();
    
    window.addEventListener('resize', () => {
        const r = drawingCanvas.parentElement.getBoundingClientRect();
        trajCamera.aspect = r.width / r.height;
        trajCamera.updateProjectionMatrix();
        trajRenderer.setSize(r.width, r.height);
    });
}
init3DCanvas();

function clearTrajectory() {
    trajPositions = [];
    trajColors = [];
    if (trajGeometry) {
        trajGeometry.setAttribute('position', new THREE.Float32BufferAttribute(trajPositions, 3));
        trajGeometry.setAttribute('color', new THREE.Float32BufferAttribute(trajColors, 3));
    }
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

        // 2. 3-Second Multi-Stroke Timer Logic
        if (isWaitingForStroke || isPredictMode) {
            // Detect first pen down
            if (data.pen && !strokeTimer) {
                const status = document.getElementById('mlStatusText');
                if (status) {
                    status.textContent = isPredictMode ? 'Status: Auto-predict (Listening 3s...)' : `Status: Recording "${pendingRecordLabel}" (Listening 3s...)`;
                }
                
                // Clear canvas at start of new multi-stroke sequence
                clearTrajectory();
                
                strokeTimer = setTimeout(() => {
                    // Timer finished
                    strokeTimer = null;
                    if (isWaitingForStroke) {
                        isWaitingForStroke = false;
                        submitRecording(pendingRecordLabel, recordedFrames);
                    } else if (isPredictMode) {
                        if (recordedFrames.length >= 5) {
                            triggerPrediction(recordedFrames);
                        }
                    }
                    recordedFrames = []; // clear buffer after send
                    
                    // Auto-clear canvas afterwards
                    setTimeout(() => {
                        if (!strokeTimer) clearTrajectory();
                    }, 1000); // Wait 1s so user can see what they drew
                    
                }, 3000);
            }
            
            // Record frames if pen is down and we are within the 3s window
            // Or regular 'isRecording' (from pipeline logic)
            if (strokeTimer && data.pen) {
                recordedFrames.push(data);
            }
        }
        
        // Handle normal continuous recording from the pipeline buttons
        if (isRecording && data.pen) {
            recordedFrames.push(data);
        }

        // 3. 3D WebGL Trajectory Drawing with Heatmap
        if (data.pos) {
            const viewfinderBox = document.getElementById('viewfinderBox');

            if (data.pen) {
                if (viewfinderBox) viewfinderBox.classList.add('active');
                
                // Scale position logic
                const currentPos = new THREE.Vector3(data.pos.x * 3, data.pos.y * 3, -data.pos.z * 3);
                const currentTime = Date.now();
                
                if (lastTrajPos !== null) {
                    const dist = currentPos.distanceTo(lastTrajPos);
                    const dt = (currentTime - lastTrajTime) / 1000.0 || 0.01;
                    const speed = dist / dt; 
                    
                    // Heatmap: Slow=Cyan, Fast=Orange/Red
                    const t = Math.min(speed / 2.0, 1.0);
                    const color = new THREE.Color();
                    if (t < 0.5) {
                        color.lerpColors(new THREE.Color(0x0ea5e9), new THREE.Color(0xf59e0b), t * 2);
                    } else {
                        color.lerpColors(new THREE.Color(0xf59e0b), new THREE.Color(0xef4444), (t - 0.5) * 2);
                    }
                    
                    trajPositions.push(currentPos.x, currentPos.y, currentPos.z);
                    trajColors.push(color.r, color.g, color.b);
                    
                    if (trajGeometry) {
                        trajGeometry.setAttribute('position', new THREE.Float32BufferAttribute(trajPositions, 3));
                        trajGeometry.setAttribute('color', new THREE.Float32BufferAttribute(trajColors, 3));
                        trajGeometry.computeBoundingSphere();
                        // Slight zoom out over time
                        if (trajControls) trajControls.target.copy(currentPos).multiplyScalar(0.1);
                    }
                    
                    if (trajPenTip) {
                        trajPenTip.position.copy(currentPos);
                        trajPenTip.visible = true;
                        trajPenTip.material.color.copy(color);
                        trajPenTip.material.emissive.copy(color);
                    }
                } else {
                    trajPositions.push(currentPos.x, currentPos.y, currentPos.z);
                    trajColors.push(0.05, 0.64, 0.91); // Cyan
                    if (trajPenTip) {
                        trajPenTip.position.copy(currentPos);
                        trajPenTip.visible = true;
                    }
                }
                
                lastTrajPos = currentPos;
                lastTrajTime = currentTime;
            } else {
                if (viewfinderBox) viewfinderBox.classList.remove('active');
                if (trajPenTip) trajPenTip.visible = false;
                lastTrajPos = null;
            }
        }
        
        wasPenDown = !!data.pen;
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
