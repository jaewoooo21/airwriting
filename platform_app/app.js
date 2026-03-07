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
const DEV_ONLY_TABS = ['tab-team'];  // tabs hidden for user mode
const DEV_ONLY_ELEMENTS = [
  'btnMlRec', 'btnMlPredict'  // ML control buttons
];

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
      btn.style.display = isUser ? 'none' : '';
    }
  });

  // Hide/show developer-only elements
  DEV_ONLY_ELEMENTS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = isUser ? 'none' : '';
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

  // Hero CTA buttons
  document.querySelectorAll('[onclick*="tab-studio"]').forEach(btn => {
    btn.removeAttribute('onclick');
    btn.addEventListener('click', () => selectTab('tab-studio'));
  });
  document.querySelectorAll('[onclick*="tab-technology"]').forEach(btn => {
    btn.removeAttribute('onclick');
    btn.addEventListener('click', () => selectTab('tab-technology'));
  });
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

    demoIdx++;
  }, 3000);
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

async function fetchMlStats() {
  try {
    const res = await fetch('/ml/stats');
    if (!res.ok) return;
    const data = await res.json();

    // Update stat cards if they exist
    const totalEl = document.getElementById('statTotalSamples');
    const classesEl = document.getElementById('statClasses');
    const accuracyEl = document.getElementById('statAccuracy');

    if (totalEl && data.total_samples != null) totalEl.textContent = data.total_samples;
    if (classesEl && data.n_classes != null) classesEl.textContent = data.n_classes;
    if (accuracyEl && data.accuracy != null) accuracyEl.textContent = (data.accuracy * 100).toFixed(1) + '%';
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
