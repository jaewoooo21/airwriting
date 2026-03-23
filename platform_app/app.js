/* ==============================================
   AirWriting Platform — Main Application Script
   Developer / User Mode System
   ============================================== */

/* ---- Global State ---- */
let currentTab = 'tab-home';
let currentTechPanel = 'arch-panel';
let ws = null;
let currentMode = 'developer';
const demoSimRunning = false;

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
  initQuickSkewControls();
  initMacroOsControls();
  initComments();
  fetchNetworkIp();
  fetchMlStats();
  window.setInterval(fetchMlStats, 5000);

  // Demo mode modal removed. Enter developer mode immediately.
  enterMode('developer');
});

function enterMode(mode) {
  currentMode = mode === 'developer' ? 'developer' : 'user';
  const body = document.body;
  if (!body) return;
  body.classList.toggle('mode-dev', currentMode === 'developer');
  body.classList.toggle('mode-user', currentMode !== 'developer');
}

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
      if (guidedRecordingActive) {
        stopGuidedRecording();
        clearTrajectory();
        return;
      }
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

  const btnMlTrain = document.getElementById('btnMlTrain');
  if (btnMlTrain) {
    btnMlTrain.addEventListener('click', () => {
      manualTrainModel();
    });
  }

  // HTML buttons now use window.selectTab directly.
}

/* =========================================
   5.5 MACRO OS PROFILE CONTROLS (v3.0 Pivot)
   ========================================= */
function initMacroOsControls() {
    const selector = document.getElementById('selectProfile');
    const desc = document.getElementById('profileDesc');
    
    if (!selector || !desc) return;

    const profileDetails = {
        'RUNNING': '러닝 모드: 전화, 음악, 위치공유, 손전등',
        'LAB': '실험 모드: 타이머, 카메라, 노트, 손전등',
        'PRESENTATION': '발표 모드: 다음, 이전, 캡처, 확인',
        'BED': '침대 모드: 음악, 조명, 전화'
    };

    selector.addEventListener('change', async () => {
        const val = selector.value;
        desc.textContent = profileDetails[val] || '';
        
        try {
            const res = await fetch('/api/macro_os/profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ profile: val })
            });
            if (!res.ok) throw new Error('API Error');
            console.log(`[MacroOS] Profile changed to ${val}`);
        } catch (err) {
            console.error('Failed to change profile:', err);
            alert('프로필 변경에 실패했습니다.');
        }
    });
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
          status.textContent = `Status: Ready for "${label}". Press pen down to start 3s capture.`;
      }

      armGuidedRecording(label);
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

async function submitRecording(label, frames) {
    const status = document.getElementById('mlStatusText');
    if (!frames || frames.length < 5) {
        if (status) status.textContent = 'Status: Data too short, try again.';
        return;
    }
    
    // Extract format needed by backend
    const stroke_full = frames.map(f => {
        const pos = extractCorrectedStrokePos(f);
        const quat = toQuatObject(f.S3q ?? f.S3qObj);
        if (pos && quat) {
            return [pos.x, pos.y, pos.z, quat.w, quat.x, quat.y, quat.z];
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
        
        if (status) {
            status.textContent = guidedRecordingActive
                ? `Status: ${result.message || `Saved "${label}".`} Press pen down for next 3s capture.`
                : `Status: ${result.message || 'Saved successfully'}`;
        }
        clearTrajectory();
        fetchMlStats();
        setTimeout(fetchMlStats, 1500);
        setTimeout(fetchMlStats, 5000);
        
    } catch (err) {
        console.error("Recording save failed:", err);
        if (status) status.textContent = 'Status: Error saving data';
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
    const lastTrainedEl = document.getElementById('lab-last-trained');
    const diversityEl = document.getElementById('health-diverse');
    const monitorEl = document.getElementById('lab-monitor');
    const btnMlTrain = document.getElementById('btnMlTrain');
    const badge = document.querySelector('#panel-training .model-badge');

    if (totalEl && data.total_samples != null) totalEl.textContent = data.total_samples;
    if (accuracyEl && data.accuracy != null) accuracyEl.textContent = (data.accuracy * 100).toFixed(1) + '%';
    if (lastTrainedEl) lastTrainedEl.textContent = data.last_trained || 'Never';

    if (diversityEl) {
        const ok = data.class_count >= 2;
        diversityEl.innerHTML = `<span class="check-icon">${ok ? 'OK' : '!'}</span> Dataset Diversity (${data.class_count || 0} labels, min ${data.min_samples_per_class || 0}/label)`;
    }

    if (badge) {
        const predictReady = data.predict_ready ? 'Predict Ready' : 'No Model';
        badge.innerHTML = `<span class="status-dot ${data.predict_ready ? 'green' : 'warning'}"></span> Random Forest (117-dim FK Stroke) / ${predictReady}`;
    }

    if (btnMlTrain) {
        btnMlTrain.disabled = data.status === 'TRAINING';
        btnMlTrain.textContent = data.status === 'TRAINING' ? 'Training...' : 'Train Model';
        btnMlTrain.style.opacity = data.status === 'TRAINING' ? '0.7' : '1';
        btnMlTrain.title = data.is_trainable ? 'Dataset is ready for training.' : (data.trainability_reason || 'Need more data.');
    }

    if (monitorEl) {
        const classes = Array.isArray(data.model_classes) ? data.model_classes.join(', ') : '';
        monitorEl.textContent = [
            `[MODEL] RandomForest / feature_dim=${data.model_feature_dim || 0} / predict_ready=${data.predict_ready ? 'YES' : 'NO'}`,
            `[DATA] total=${data.total_samples || 0} / labels=${data.class_count || 0} / min_per_label=${data.min_samples_per_class || 0}`,
            `[TRAIN] status=${data.status || 'UNKNOWN'} / trainable=${data.is_trainable ? 'YES' : 'NO'}`,
            `[RULE] ${data.trainability_reason || 'No status available.'}`,
            `[LAST] trained=${data.last_trained || 'Never'} / accuracy=${((data.accuracy || 0) * 100).toFixed(1)}%`,
            `[CLASSES] ${classes || '-'}`,
            data.last_error ? `[ERROR] ${data.last_error}` : '[ERROR] -',
        ].join('\n');
    }
    
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

async function manualTrainModel() {
  const status = document.getElementById('mlStatusText');
  if (status) status.textContent = 'Status: Checking training readiness...';

  try {
    const res = await fetch('/api/ml/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    if (!res.ok) throw new Error('Training request failed');

    const payload = await res.json();
    if (status) status.textContent = `Status: ${payload.message || 'Training request sent.'}`;
    fetchMlStats();
    if (payload.training_started) {
      setTimeout(fetchMlStats, 1500);
      setTimeout(fetchMlStats, 5000);
    }
  } catch (err) {
    console.error('Manual training failed:', err);
    if (status) status.textContent = 'Status: Manual training failed';
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
    const visualSkew = config.visualization?.writing_board?.plane_skew_deg;
    setPlaneSkewCorrection(typeof visualSkew === 'number' ? visualSkew : 15);
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
    },
    visualization: {
      writing_board: {
        plane_skew_deg: parseFloat(document.getElementById('confPlaneSkew').value) || 0
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
    setPlaneSkewCorrection(payload.visualization.writing_board.plane_skew_deg);
    
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

function toVec3Object(value) {
    if (Array.isArray(value) && value.length >= 3) {
        return {
            x: Number(value[0]) || 0,
            y: Number(value[1]) || 0,
            z: Number(value[2]) || 0,
        };
    }

    if (value && typeof value === 'object') {
        return {
            x: Number(value.x) || 0,
            y: Number(value.y) || 0,
            z: Number(value.z) || 0,
        };
    }

    return null;
}

function toQuatArray(value) {
    if (Array.isArray(value) && value.length >= 4) {
        return [
            Number(value[0]) || 1,
            Number(value[1]) || 0,
            Number(value[2]) || 0,
            Number(value[3]) || 0,
        ];
    }

    if (value && typeof value === 'object') {
        return [
            Number(value.w) || 1,
            Number(value.x) || 0,
            Number(value.y) || 0,
            Number(value.z) || 0,
        ];
    }

    return null;
}

function toQuatObject(value) {
    const arr = toQuatArray(value);
    if (!arr) return null;
    return { w: arr[0], x: arr[1], y: arr[2], z: arr[3] };
}

function normalizeIncomingData(raw) {
    const data = { ...raw };

    const pos = toVec3Object(data.pos ?? data.S3p ?? data.p);
    if (pos) data.pos = pos;

    const fk = toVec3Object(data.fk ?? data.S3fk);
    if (fk) data.fk = fk;

    const renderPos = toVec3Object(data.renderPos ?? data.fk ?? data.S3fk ?? data.pos ?? data.S3p ?? data.p);
    if (renderPos) data.renderPos = renderPos;

    const vel = toVec3Object(data.vel ?? data.S3v ?? data.v);
    if (vel) data.vel = vel;

    for (const key of ['S1q', 'S2q', 'S3q']) {
        const arr = toQuatArray(data[key]);
        if (arr) {
            data[key] = arr;
            data[`${key}Obj`] = toQuatObject(arr);
        }
    }

    if (data.type == null && data.t === 'f') {
        data.type = 'imu_stream';
    }

    if (data.word == null && data.type === 'recognition' && data.label) {
        data.word = data.label;
    }

    if (data.score == null && typeof data.confidence === 'number') {
        data.score = data.confidence;
    }

    if (data.zupt == null && typeof data.S3z === 'boolean') {
        data.zupt = data.S3z;
    }

    return data;
}

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
const boardOrigin = new THREE.Vector3(0.0, 0.04, -0.86);
const boardNormal = new THREE.Vector3(0, 0, 1);
const sceneUpAxis = new THREE.Vector3(0, 1, 0);
const boardTilt = new THREE.Euler(
    THREE.MathUtils.degToRad(-6),
    THREE.MathUtils.degToRad(4),
    0,
    'XYZ'
);
const boardBounds = {
    width: 1.24,
    height: 0.86,
    depth: 0.024,
    surfaceZ: 0.013,
};
let planeSkewCorrectionDeg = 15;
let planeSkewCorrectionRad = THREE.MathUtils.degToRad(planeSkewCorrectionDeg);
const penSpace = {
    scaleX: 4.0,
    scaleY: 5.0,
    scaleZ: 4.0,
    offsetX: 0.0,
    offsetY: -0.02,
    offsetZ: -0.46,
    drawDistance: 0.42,
};

let trajScene, trajCamera, trajRenderer;
let trajStage, trajBoardRoot;
let trajControls;
let trajLine, trajMaterial, trajGeometry;
let trajPenGroup, trajPenTip, trajPenBody;
let trajBoard, trajBoardFrame, trajBoardGlow, trajGround;
let trajInkGroup;
let trajBoardCursor, trajBoardCursorHalo;
let trajPositions = [];
let trajColors = [];
let lastTrajPos = null;
let lastInkPos = null;
let lastTrajTime = 0;
let lastPenDirection = new THREE.Vector3(0, -0.1, -1).normalize();
let penCalibrationOrigin = new THREE.Vector3(0, 0, 0);
let hasPenCalibrationOrigin = false;
let lastRenderSensorPos = null;
let guidedRecordingActive = false;

function setPlaneSkewCorrection(deg) {
    const safeDeg = Number.isFinite(deg) ? THREE.MathUtils.clamp(deg, -45, 45) : 15;
    planeSkewCorrectionDeg = safeDeg;
    planeSkewCorrectionRad = THREE.MathUtils.degToRad(planeSkewCorrectionDeg);
    const textValue = planeSkewCorrectionDeg.toFixed(1);
    const input = document.getElementById('confPlaneSkew');
    const quickSlider = document.getElementById('quickPlaneSkew');
    const quickNumber = document.getElementById('quickPlaneSkewNumber');
    const quickLabel = document.getElementById('quickPlaneSkewVal');
    if (input && Number(input.value) !== planeSkewCorrectionDeg) input.value = textValue;
    if (quickSlider && Number(quickSlider.value) !== planeSkewCorrectionDeg) quickSlider.value = textValue;
    if (quickNumber && Number(quickNumber.value) !== planeSkewCorrectionDeg) quickNumber.value = textValue;
    if (quickLabel) quickLabel.textContent = `${textValue}°`;
}

function initQuickSkewControls() {
    const quickSlider = document.getElementById('quickPlaneSkew');
    const quickNumber = document.getElementById('quickPlaneSkewNumber');
    const btnReset = document.getElementById('btnQuickSkewReset');
    const btnSave = document.getElementById('btnQuickSkewSave');

    if (!quickSlider || !quickNumber) return;

    const apply = (value) => {
        const parsed = Number.parseFloat(value);
        setPlaneSkewCorrection(Number.isFinite(parsed) ? parsed : planeSkewCorrectionDeg);
    };

    quickSlider.addEventListener('input', () => {
        apply(quickSlider.value);
    });

    quickNumber.addEventListener('input', () => {
        apply(quickNumber.value);
    });

    quickNumber.addEventListener('change', () => {
        apply(quickNumber.value);
    });

    if (btnReset) {
        btnReset.addEventListener('click', () => {
            apply(0);
        });
    }

    if (btnSave) {
        btnSave.addEventListener('click', async () => {
            const settingsInput = document.getElementById('confPlaneSkew');
            if (settingsInput) settingsInput.value = planeSkewCorrectionDeg.toFixed(1);
            await saveSystemConfig();
            const status = document.getElementById('mlStatusText');
            if (status) status.textContent = `Status: Skew correction saved (${planeSkewCorrectionDeg.toFixed(1)}°).`;
        });
    }

    setPlaneSkewCorrection(planeSkewCorrectionDeg);
}

function applySensorPlaneCorrection(pos) {
    const cosA = Math.cos(planeSkewCorrectionRad);
    const sinA = Math.sin(planeSkewCorrectionRad);
    return new THREE.Vector3(
        pos.x * cosA + pos.z * sinA,
        pos.y,
        -pos.x * sinA + pos.z * cosA
    );
}

function extractCorrectedStrokePos(frame) {
    const pos = toVec3Object(frame.fk ?? frame.S3fk ?? frame.pos ?? frame.S3p ?? frame.p);
    if (!pos) return null;
    return applySensorPlaneCorrection(new THREE.Vector3(pos.x, pos.y, pos.z));
}

function mapSensorPositionToWorld(pos) {
    const origin = hasPenCalibrationOrigin ? penCalibrationOrigin : pos;
    const correctedPos = applySensorPlaneCorrection(pos);
    const correctedOrigin = applySensorPlaneCorrection(origin);
    const calibrated = new THREE.Vector3(
        correctedPos.x - correctedOrigin.x,
        correctedPos.y - correctedOrigin.y,
        correctedPos.z - correctedOrigin.z
    );
    // Sensor axes: X = left/right, Z = up/down, Y = depth
    return new THREE.Vector3(
        THREE.MathUtils.clamp(calibrated.x * penSpace.scaleX + penSpace.offsetX, -0.65, 0.65),
        THREE.MathUtils.clamp(calibrated.z * penSpace.scaleY + penSpace.offsetY, -0.48, 0.48),
        THREE.MathUtils.clamp(
            penSpace.offsetZ - calibrated.y * penSpace.scaleZ,
            -0.95,
            0.2
        )
    );
}

function computeHandVisualOffset(data) {
    const rawOffset = new THREE.Vector3();

    const specs = [
        { key: 'S1q', len: 0.22, weight: 0.28 },
        { key: 'S2q', len: 0.14, weight: 0.45 },
        { key: 'S3q', len: 0.08, weight: 0.62 },
    ];

    specs.forEach(spec => {
        const arr = toQuatArray(data[spec.key]);
        if (!arr) return;
        const q = new THREE.Quaternion(arr[1], arr[2], arr[3], arr[0]);
        const neutral = new THREE.Vector3(0, spec.len, 0);
        const rotated = neutral.clone().applyQuaternion(q);
        rawOffset.add(rotated.sub(neutral).multiplyScalar(spec.weight));
    });

    return new THREE.Vector3(
        rawOffset.x * 0.85,
        rawOffset.z * -2.1,
        rawOffset.y * 0.8
    );
}

function boardLocalToWorld(point) {
    if (!trajBoardRoot) return point.clone();
    return trajBoardRoot.localToWorld(point.clone());
}

function worldToBoardLocal(point) {
    if (!trajBoardRoot) return point.clone();
    return trajBoardRoot.worldToLocal(point.clone());
}

function projectTipToBoardIfWritable(tipWorld) {
    if (!trajBoardRoot) return null;

    const localTip = worldToBoardLocal(tipWorld);
    const halfW = boardBounds.width * 0.5;
    const halfH = boardBounds.height * 0.5;
    const depth = localTip.z - boardBounds.surfaceZ;

    if (depth < 0 || depth > penSpace.drawDistance) {
        return null;
    }

    if (Math.abs(localTip.x) > halfW || Math.abs(localTip.y) > halfH) {
        return null;
    }

    return boardLocalToWorld(new THREE.Vector3(localTip.x, localTip.y, boardBounds.surfaceZ));
}

function projectTipToBoardPreview(tipWorld) {
    if (!trajBoardRoot) return null;

    const localTip = worldToBoardLocal(tipWorld);
    const halfW = boardBounds.width * 0.5;
    const halfH = boardBounds.height * 0.5;
    const clampedX = THREE.MathUtils.clamp(localTip.x, -halfW, halfW);
    const clampedY = THREE.MathUtils.clamp(localTip.y, -halfH, halfH);
    const depth = localTip.z - boardBounds.surfaceZ;
    const point = boardLocalToWorld(new THREE.Vector3(clampedX, clampedY, boardBounds.surfaceZ));

    return {
        point,
        depth,
        inBounds: Math.abs(localTip.x) <= halfW && Math.abs(localTip.y) <= halfH,
    };
}

function orientPenGroup(group, direction) {
    const dir = direction.clone();
    if (dir.lengthSq() < 1e-6) {
        dir.copy(lastPenDirection);
    }
    dir.normalize();
    lastPenDirection.copy(dir);
    group.quaternion.setFromUnitVectors(sceneUpAxis, dir);
}

function updateTrajectoryGeometry() {
    if (!trajGeometry) return;
    trajGeometry.setAttribute('position', new THREE.Float32BufferAttribute(trajPositions, 3));
    trajGeometry.setAttribute('color', new THREE.Float32BufferAttribute(trajColors, 3));
    if (trajPositions.length >= 6) {
        trajGeometry.computeBoundingSphere();
    }
}

function addInkSegment(start, end, color) {
    if (!trajInkGroup) return;

    const segment = end.clone().sub(start);
    const length = segment.length();
    if (length < 1e-4) return;

    const mid = start.clone().add(end).multiplyScalar(0.5);
    const geom = new THREE.CylinderGeometry(0.0055, 0.0055, length, 10);
    const mat = new THREE.MeshStandardMaterial({
        color: color.getHex(),
        emissive: color.clone().multiplyScalar(0.18),
        roughness: 0.28,
        metalness: 0.02,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.copy(mid);
    mesh.quaternion.setFromUnitVectors(sceneUpAxis, segment.clone().normalize());
    trajInkGroup.add(mesh);
}

function updateMlRecordButton() {
    const btnMlRec = document.getElementById('btnMlRec');
    if (!btnMlRec) return;
    if (guidedRecordingActive) {
        btnMlRec.textContent = '■ 가이드 학습 종료';
        btnMlRec.style.background = '#ef4444';
        btnMlRec.style.color = '#ffffff';
    } else {
        btnMlRec.textContent = '🎯 [가이드] 단어 학습';
        btnMlRec.style.background = '';
        btnMlRec.style.color = '';
    }
}

function stopGuidedRecording(message = 'Status: Guided recording stopped.') {
    guidedRecordingActive = false;
    isWaitingForStroke = false;
    pendingRecordLabel = '';
    recordedFrames = [];
    if (strokeTimer) {
        clearTimeout(strokeTimer);
        strokeTimer = null;
    }
    const status = document.getElementById('mlStatusText');
    if (status) status.textContent = message;
    updateMlRecordButton();
}

function armGuidedRecording(label) {
    guidedRecordingActive = true;
    pendingRecordLabel = label;
    isWaitingForStroke = true;
    recordedFrames = [];
    if (strokeTimer) {
        clearTimeout(strokeTimer);
        strokeTimer = null;
    }
    clearTrajectory();
    const status = document.getElementById('mlStatusText');
    if (status) status.textContent = `Status: Ready for "${label}". Press pen down to start 3s capture.`;
    updateMlRecordButton();
}

function recalibratePenFromCurrentPose() {
    if (!lastRenderSensorPos) return false;
    penCalibrationOrigin.copy(lastRenderSensorPos);
    hasPenCalibrationOrigin = true;
    lastTrajPos = null;
    lastInkPos = null;
    clearTrajectory();
    const status = document.getElementById('mlStatusText');
    if (status) status.textContent = 'Status: Pen recalibrated to current pose.';
    logToTerminal('[SYS] Pen visual calibration updated.');
    return true;
}

function init3DCanvas() {
    if (!drawingCanvas) return;
    
    const rect = drawingCanvas.parentElement.getBoundingClientRect();
    
    trajScene = new THREE.Scene();
    trajCamera = new THREE.PerspectiveCamera(42, rect.width / rect.height, 0.01, 100);
    trajCamera.position.set(0.0, 0.08, 0.72);
    trajCamera.lookAt(boardOrigin.x, boardOrigin.y, boardOrigin.z);
    
    trajRenderer = new THREE.WebGLRenderer({ canvas: drawingCanvas, antialias: true, alpha: true });
    trajRenderer.setSize(rect.width, rect.height);
    trajRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    trajRenderer.setClearColor(0x000000, 0);
    trajRenderer.shadowMap.enabled = true;

    if (typeof THREE.OrbitControls !== 'undefined') {
        trajControls = new THREE.OrbitControls(trajCamera, trajRenderer.domElement);
        trajControls.target.copy(boardOrigin);
        trajControls.enableDamping = true;
        trajControls.dampingFactor = 0.08;
        trajControls.enablePan = false;
    trajControls.minDistance = 0.4;
    trajControls.maxDistance = 2.2;
        trajControls.minPolarAngle = 0.7;
        trajControls.maxPolarAngle = 2.2;
    }

    const ambientLight = new THREE.AmbientLight(0xe6f6ff, 0.72);
    trajScene.add(ambientLight);
    
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(0.3, 1.1, 1.8);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    trajScene.add(dirLight);

    const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.5);
    rimLight.position.set(-1.4, 0.3, 1.1);
    trajScene.add(rimLight);

    trajStage = new THREE.Group();
    trajScene.add(trajStage);

    const groundGeo = new THREE.PlaneGeometry(6, 6);
    const groundMat = new THREE.MeshStandardMaterial({
        color: 0x08111d,
        transparent: true,
        opacity: 0.72,
        roughness: 0.95,
        metalness: 0.02,
    });
    trajGround = new THREE.Mesh(groundGeo, groundMat);
    trajGround.rotation.x = -Math.PI / 2;
    trajGround.position.set(0, -0.78, -0.34);
    trajGround.receiveShadow = true;
    trajStage.add(trajGround);

    const groundGrid = new THREE.GridHelper(5.4, 24, 0x1d4f73, 0x0f2233);
    groundGrid.position.set(0, -0.779, -0.34);
    trajStage.add(groundGrid);

    trajBoardRoot = new THREE.Group();
    trajBoardRoot.position.copy(boardOrigin);
    trajBoardRoot.rotation.copy(boardTilt);
    trajStage.add(trajBoardRoot);

    trajBoardGlow = new THREE.Mesh(
        new THREE.PlaneGeometry(boardBounds.width + 0.12, boardBounds.height + 0.12),
        new THREE.MeshBasicMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.08,
            depthWrite: false,
        })
    );
    trajBoardGlow.position.z = -0.03;
    trajBoardRoot.add(trajBoardGlow);

    trajBoard = new THREE.Mesh(
        new THREE.BoxGeometry(boardBounds.width, boardBounds.height, boardBounds.depth),
        new THREE.MeshStandardMaterial({
            color: 0x102433,
            transparent: true,
            opacity: 0.24,
            roughness: 0.6,
            metalness: 0.04,
        })
    );
    trajBoard.receiveShadow = true;
    trajBoardRoot.add(trajBoard);

    trajBoardFrame = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(boardBounds.width, boardBounds.height, boardBounds.depth)),
        new THREE.LineBasicMaterial({
            color: 0x7dd3fc,
            transparent: true,
            opacity: 0.42,
        })
    );
    trajBoardRoot.add(trajBoardFrame);

    trajBoardCursorHalo = new THREE.Mesh(
        new THREE.RingGeometry(0.016, 0.032, 32),
        new THREE.MeshBasicMaterial({
            color: 0x67e8f9,
            transparent: true,
            opacity: 0.0,
            side: THREE.DoubleSide,
            depthWrite: false,
        })
    );
    trajBoardCursorHalo.visible = false;
    trajStage.add(trajBoardCursorHalo);

    trajBoardCursor = new THREE.Mesh(
        new THREE.CircleGeometry(0.008, 24),
        new THREE.MeshBasicMaterial({
            color: 0xf8fafc,
            transparent: true,
            opacity: 0.0,
            depthWrite: false,
        })
    );
    trajBoardCursor.visible = false;
    trajStage.add(trajBoardCursor);

    trajInkGroup = new THREE.Group();
    trajStage.add(trajInkGroup);

    trajPenGroup = new THREE.Group();

    const shaftLength = 0.082;
    const shaftGeometry = new THREE.CylinderGeometry(0.004, 0.0065, shaftLength, 12);
    shaftGeometry.translate(0, -(shaftLength * 0.5 + 0.018), 0);
    trajPenBody = new THREE.Mesh(
        shaftGeometry,
        new THREE.MeshStandardMaterial({
            color: 0x38bdf8,
            emissive: 0x0ea5e9,
            emissiveIntensity: 0.55,
            roughness: 0.16,
            metalness: 0.38,
        })
    );
    trajPenGroup.add(trajPenBody);

    const tipLength = 0.024;
    const tipGeometry = new THREE.ConeGeometry(0.007, tipLength, 12);
    tipGeometry.translate(0, -(tipLength * 0.5), 0);
    trajPenTip = new THREE.Mesh(
        tipGeometry,
        new THREE.MeshStandardMaterial({
            color: 0xf8fafc,
            emissive: 0xffffff,
            emissiveIntensity: 0.14,
            roughness: 0.14,
            metalness: 0.08,
        })
    );
    trajPenGroup.add(trajPenTip);
    trajPenGroup.visible = false;
    trajStage.add(trajPenGroup);
    
    trajGeometry = new THREE.BufferGeometry();
    trajMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 });
    trajLine = new THREE.LineSegments(trajGeometry, trajMaterial);
    trajStage.add(trajLine);
    
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
    lastTrajPos = null;
    lastInkPos = null;
    lastPenDirection.set(0, -0.1, -1).normalize();
    if (trajBoardCursor) trajBoardCursor.visible = false;
    if (trajBoardCursorHalo) trajBoardCursorHalo.visible = false;
    if (trajInkGroup) {
        while (trajInkGroup.children.length) {
            const child = trajInkGroup.children.pop();
            child.geometry?.dispose?.();
            child.material?.dispose?.();
        }
    }
    updateTrajectoryGeometry();
}

let lastX = null, lastY = null;

function initRealTimeSystem() {
    const wsUrl = `ws://${window.location.hostname}:18765`;
    wsConnection = new WebSocket(wsUrl);

    wsConnection.onopen = () => {
        logToTerminal('[NET] WebSocket connected: ' + wsUrl);
        const connEl = document.getElementById('valConn');
        if (connEl) { connEl.textContent = '🟢 CONNECTED'; connEl.classList.remove('warning'); connEl.style.color = '#10b981'; }
    };

    wsConnection.onmessage = (event) => {
        try {
            const data = normalizeIncomingData(JSON.parse(event.data));
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
        const connEl = document.getElementById('valConn');
        if (connEl) {
            connEl.textContent = 'LIVE STREAM';
            connEl.classList.remove('warning');
            connEl.style.color = '#10b981';
        }

        // 1. Update 3D Hand Mesh
        if (window.handWidgetUpdate) {
            window.handWidgetUpdate(data);
        }

        // 2. 3-Second Multi-Stroke Timer Logic
        if (isWaitingForStroke || isPredictMode) {
            // Detect first pen down
            if (data.pen && !wasPenDown && !strokeTimer) {
                const status = document.getElementById('mlStatusText');
                if (status) {
                    status.textContent = isPredictMode ? 'Status: Auto-predict (Listening 3s...)' : `Status: Recording "${pendingRecordLabel}" (Listening 3s...)`;
                }
                
                // Clear canvas at start of new multi-stroke sequence
                recordedFrames = [];
                clearTrajectory();
                
                strokeTimer = setTimeout(() => {
                    // Timer finished
                    strokeTimer = null;
                    if (isWaitingForStroke) {
                        submitRecording(pendingRecordLabel, recordedFrames);
                        if (guidedRecordingActive) {
                            isWaitingForStroke = true;
                        } else {
                            isWaitingForStroke = false;
                        }
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
            const posEl = document.getElementById('valPos');
            if (posEl) {
                posEl.innerHTML =
                    `X: ${data.pos.x.toFixed(3)}<br>` +
                    `Y: ${data.pos.y.toFixed(3)}<br>` +
                    `Z: ${data.pos.z.toFixed(3)}`;
            }

            const zuptEl = document.getElementById('valZupt');
            if (zuptEl) {
                const isActive = !!data.zupt;
                zuptEl.textContent = isActive ? 'ACTIVE' : 'INACTIVE';
                zuptEl.classList.toggle('warning', !isActive);
                zuptEl.style.color = isActive ? '#10b981' : '#f59e0b';
            }

            const renderPos = toVec3Object(data.renderPos ?? data.fk ?? data.S3fk ?? data.pos ?? data.S3p ?? data.p);
            if (!renderPos) return;
            lastRenderSensorPos = new THREE.Vector3(renderPos.x, renderPos.y, renderPos.z);
            if (!hasPenCalibrationOrigin) {
                penCalibrationOrigin.copy(lastRenderSensorPos);
                hasPenCalibrationOrigin = true;
            }

            const currentPos = mapSensorPositionToWorld(renderPos).add(computeHandVisualOffset(data));
            const currentTime = Date.now();

            let direction = lastPenDirection.clone();
            if (lastTrajPos !== null) {
                const delta = currentPos.clone().sub(lastTrajPos);
                if (delta.lengthSq() > 1e-5) {
                    direction.copy(delta.normalize());
                }
            } else if (data.vel) {
                const velDir = new THREE.Vector3(data.vel.x, data.vel.z, -data.vel.y);
                if (velDir.lengthSq() > 1e-6) direction.copy(velDir.normalize());
            } else if (data.S3q && data.S3q.length === 4) {
                const q = new THREE.Quaternion(data.S3q[1], data.S3q[2], data.S3q[3], data.S3q[0]);
                direction.copy(new THREE.Vector3(0, 0, -1).applyQuaternion(q));
            }

            const towardBoard = boardOrigin.clone().sub(currentPos).normalize();
            direction.lerp(towardBoard, 0.32).normalize();

            if (trajPenGroup) {
                trajPenGroup.position.copy(currentPos);
                orientPenGroup(trajPenGroup, direction);
                trajPenGroup.visible = true;

                const bodyColor = data.pen ? new THREE.Color(0xff6b8d) : new THREE.Color(0x38bdf8);
                trajPenBody.material.color.copy(bodyColor);
                trajPenBody.material.emissive.copy(bodyColor);
            }

            const preview = projectTipToBoardPreview(currentPos);
            if (preview && trajBoardCursor && trajBoardCursorHalo) {
                const cursorNormal = boardNormal.clone().applyQuaternion(trajBoardRoot.quaternion).normalize();
                const cursorQuat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), cursorNormal);
                const glowColor = data.pen ? new THREE.Color(0xff6b8d) : new THREE.Color(0x67e8f9);
                const nearFactor = THREE.MathUtils.clamp(1 - (Math.max(preview.depth, 0) / Math.max(penSpace.drawDistance, 0.001)), 0, 1);
                const visible = preview.inBounds;

                trajBoardCursor.visible = visible;
                trajBoardCursorHalo.visible = visible;
                if (visible) {
                    trajBoardCursor.position.copy(preview.point).add(cursorNormal.clone().multiplyScalar(0.002));
                    trajBoardCursorHalo.position.copy(preview.point).add(cursorNormal.clone().multiplyScalar(0.0015));
                    trajBoardCursor.quaternion.copy(cursorQuat);
                    trajBoardCursorHalo.quaternion.copy(cursorQuat);
                    trajBoardCursor.material.color.copy(glowColor);
                    trajBoardCursorHalo.material.color.copy(glowColor);
                    trajBoardCursor.material.opacity = 0.35 + nearFactor * 0.55;
                    trajBoardCursorHalo.material.opacity = 0.08 + nearFactor * 0.38;
                    trajBoardCursor.scale.setScalar(data.pen ? 1.15 : 1.0);
                    trajBoardCursorHalo.scale.setScalar(1.0 + nearFactor * 0.55);
                }
            }

            if (data.pen) {
                const inkPos = projectTipToBoardIfWritable(currentPos);
                if (inkPos && lastInkPos !== null) {
                    const dist = inkPos.distanceTo(lastInkPos);
                    const dt = (currentTime - lastTrajTime) / 1000.0 || 0.01;
                    const speed = dist / dt;

                    const t = Math.min(speed / 1.4, 1.0);
                    const color = new THREE.Color();
                    color.lerpColors(new THREE.Color(0x7dd3fc), new THREE.Color(0xf8fafc), t);

                    trajPositions.push(lastInkPos.x, lastInkPos.y, lastInkPos.z);
                    trajPositions.push(inkPos.x, inkPos.y, inkPos.z);
                    trajColors.push(color.r, color.g, color.b);
                    trajColors.push(color.r, color.g, color.b);
                    addInkSegment(lastInkPos, inkPos, color);
                    updateTrajectoryGeometry();
                }
                lastInkPos = inkPos ? inkPos.clone() : null;
            } else {
                lastInkPos = null;
            }

            lastTrajPos = currentPos;
            lastTrajTime = currentTime;
        }
        
        wasPenDown = !!data.pen;
    } else if (data.type === 'ml_result' || data.type === 'recognition' || data.word) {
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

async function triggerPrediction(frames) {
    const status = document.getElementById('mlStatusText');
    const strokePos = frames
        .map(frame => extractCorrectedStrokePos(frame))
        .filter(Boolean)
        .map(pos => [pos.x, pos.y, pos.z]);

    if (strokePos.length < 5) {
        if (status) status.textContent = 'Status: Not enough points for prediction.';
        return;
    }

    if (status) status.textContent = 'Status: Predicting...';

    try {
        const res = await fetch('/api/ml/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stroke_pos: strokePos })
        });

        if (!res.ok) throw new Error('Prediction request failed');
        const payload = await res.json();
        const predictions = Array.isArray(payload.predictions) ? payload.predictions : [];
        const best = predictions[0];

        const resultWord = document.getElementById('aiResultWord');
        const resultScore = document.getElementById('aiResultScore');
        const candidates = document.getElementById('aiCandidates');
        const overlay = document.getElementById('recognizedTextOverlay');

        if (best) {
            if (resultWord) resultWord.textContent = best.label;
            if (resultScore) resultScore.textContent = `(${(best.confidence * 100).toFixed(1)}%)`;
            if (overlay) {
                overlay.textContent = best.label;
                overlay.style.opacity = '1';
                setTimeout(() => { overlay.style.opacity = '0'; }, 1200);
            }
            if (status) status.textContent = `Status: Predicted ${best.label}`;
        } else {
            if (resultWord) resultWord.textContent = '--';
            if (resultScore) resultScore.textContent = '(0.0%)';
            if (status) status.textContent = 'Status: No prediction available';
        }

        if (candidates) {
            candidates.innerHTML = predictions.slice(0, 3).map(pred => `
                <div class="candidate-bar">
                    <div class="c-name">${pred.label}</div>
                    <div class="c-val">${(pred.confidence * 100).toFixed(1)}%</div>
                </div>
            `).join('') || `
                <div class="candidate-bar">
                    <div class="c-name">-</div>
                    <div class="c-val">-</div>
                </div>
            `;
        }
    } catch (err) {
        console.error('Prediction failed:', err);
        if (status) status.textContent = 'Status: Prediction failed';
    }
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

function initKeyboardShortcuts() {
    document.addEventListener('keydown', (event) => {
        const tag = event.target?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;

        const key = event.key.toLowerCase();
        if (key === 'r') {
            clearTrajectory();
            const status = document.getElementById('mlStatusText');
            if (status && !guidedRecordingActive) status.textContent = 'Status: Board cleared.';
            event.preventDefault();
            return;
        }

        if (key === 'c') {
            const ok = recalibratePenFromCurrentPose();
            const status = document.getElementById('mlStatusText');
            if (!ok && status) status.textContent = 'Status: No live pen pose available for calibration.';
            event.preventDefault();
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initRealTimeSystem();
    initDataPipeline();
    initKeyboardShortcuts();
    updateMlRecordButton();
});
